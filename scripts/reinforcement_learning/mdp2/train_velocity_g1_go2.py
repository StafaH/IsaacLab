# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a Unitree G1 and a Unitree Go2 to walk in the worlds of one simulation, with a policy per robot.

    uv run python scripts/reinforcement_learning/mdp2/train_velocity_g1_go2.py --budget 300

Half of the worlds hold a G1 and the other half a Go2, and each half runs the velocity task of its robot. The
policy of a robot acts in the worlds of that robot and learns from them. The script logs, per robot, the error of
the base velocity [m/s] and the share of the last episodes that ended in a fall.

``--robots g1`` or ``--robots go2`` trains one robot alone, in a scene without the other, with the same task,
physics, and learner: the reference for the robot's half of the worlds.

``--style`` gives each robot a second task in that share of its worlds: following the commands in the style of
motion clips, of a person walking for the G1 and of a dog trotting for the Go2. A discriminator per robot learns to
tell the robot's motion from the clips', and its score is the style reward of those worlds. The policy of a robot
learns from both tasks. The script then also logs the mean style reward, between 0 and 1.

``--policy shared`` trains one network for both robots: an encoder per robot maps its observations to a latent
vector, a trunk that the robots share maps it on, and a decoder per robot maps the result to its actions.
"""

import argparse
import json
import time

import gymnasium as gym
import torch
import warp as wp
from isaaclab_experimental import mdp2
from ppo import PPO, mlp
from video import add_camera, record

from isaaclab.utils import math as math_utils
from isaaclab.utils import replace

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.hil import mdp as hil
from isaaclab_tasks.mdp2.velocity_g1_go2.flat_env_cfg import G1_GO2_STYLE_TASKS, G1_GO2_TASKS
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--robots", default="g1,go2", choices=["g1,go2", "g1", "go2"])
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=float, default=300.0, help="Samples to train for, in millions")
parser.add_argument("--style", type=float, default=0.0, help="Share of each robot's worlds with the style task")
parser.add_argument("--policy", default="separate", choices=["separate", "shared"])
parser.add_argument("--latent", type=int, default=256, help="Size of the latent vector of the shared policy")
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--init_std", type=float, default=1.0)
parser.add_argument("--entropy", type=float, default=0.005)
parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
parser.add_argument("--disc_lr", type=float, default=1.0e-4)
parser.add_argument("--disc_steps", type=int, default=2, help="Updates of a discriminator per iteration")
parser.add_argument("--disc_batch", type=int, default=4096)
parser.add_argument("--penalty", type=float, default=5.0, help="Weight of the discriminators' gradient penalty")
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--out", default=None, help="Path of a JSON file with the log")
parser.add_argument("--video", default=None, help="Path of an MP4 with worlds of every robot")
args = parser.parse_args()

TASK, DEVICE = "Isaac-Velocity-Flat-G1-Go2-MDP2", "cuda:0"
ROBOTS = ("g1", "go2")
NAMES = args.robots.split(",")
EYE = (1.7, -2.0, 0.5)
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True


def make_env(num_envs, style, video=False):
    """Return an environment, and the robot of each of its world configs."""
    cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, num_envs, args.seed
    if len(NAMES) == 1:
        # One robot alone: a scene without the other, whose worlds are alike and need no sleeping robot.
        other = ROBOTS[1 - ROBOTS.index(NAMES[0])]
        setattr(cfg.scene, other, None)
        setattr(cfg.scene, f"{other}_contact_forces", None)
        cfg.scene.clone_cfg.clone_combinations = []
        cfg.sim.physics.solver_cfg.sleep_inactive_assets = False

    # Every robot gets the same number of worlds. With a style, its style task comes first and takes its share.
    tasks, counts, robots = [], [], []
    for name in NAMES:
        count = num_envs // len(NAMES)
        velocity_task = G1_GO2_TASKS[ROBOTS.index(name)]
        styled = round(style * count)
        if styled > 0:
            tasks.append(G1_GO2_STYLE_TASKS[2 * ROBOTS.index(name)])
            counts.append(styled)
            robots.append(name)
        if video:
            # The video shows the robots walking forward; the Go2 task also commands backward and sideways velocities.
            command = velocity_task.commands["base_velocity"]
            forward = {**command.params, "lin_vel_x": (0.6, 1.0), "lin_vel_y": (0.0, 0.0)}
            velocity_task = replace(velocity_task, commands={"base_velocity": replace(command, params=forward)})
        if count > styled:
            tasks.append(velocity_task)
            counts.append(count - styled)
            robots.append(name)
    cfg.task, cfg.task_worlds = (tasks, counts) if len(tasks) > 1 else (tasks[0], None)
    if video:
        add_camera(cfg, eye=EYE)
    return gym.make(TASK, cfg=cfg).unwrapped, robots


def robot_worlds(env, robots):
    """Return the worlds of every robot, from the robot of each world config."""
    worlds = {}
    for name, config in zip(robots, env.mdp.world_configs):
        first = worlds[name].start if name in worlds else config.worlds.start
        worlds[name] = slice(first, config.worlds.stop)
    return worlds


def measure(env, worlds):
    """Return, per robot, the error of the base velocity [m/s] and the share of the last episodes that fell."""
    command = env.state.arrays[mdp2.commands.base_velocity].torch
    row = {}
    for name, robot in worlds.items():
        data = env.scene[name].data
        velocity = math_utils.quat_apply_inverse(
            data.root_link_pose_w.torch[robot, 3:7], data.root_com_vel_w.torch[robot, :3]
        )
        row[f"{name}/speed_error"] = (command[robot, :2] - velocity[:, :2]).norm(dim=-1).mean()
        falls = [
            value
            for term, value in env.extras["log"].items()
            if term.startswith(f"Episode_Termination/{name}_") and not term.endswith("time_out")
        ]
        row[f"{name}/falls"] = sum(falls)
    return row


env, robots = make_env(args.num_envs, args.style)
worlds = robot_worlds(env, robots)
if len(NAMES) > 1:
    # The tasks split the worlds in halves, and so must the scene, for each task to run where its robot is.
    layout = env.scene.clone_plan.topology.world_prototype_layout
    for index, name in enumerate(NAMES):
        assert (layout[worlds[name]] == index).all(), "Use an even number of worlds."
obs = env.reset()[0]["policy"]
num_envs, T = env.num_envs, args.steps
actions = torch.zeros(num_envs, env.mdp.action_dim, device=DEVICE)

# A learner per robot, on the worlds of its robot and the columns that its tasks use. The shared policy is an
# encoder and a decoder per robot around one trunk, which the learners of the robots train in turn.
trunk = mlp([args.latent, *args.hidden]) if args.policy == "shared" else None
dims = dict(zip(robots, env.mdp.world_configs))
learners = {}
for name, robot in worlds.items():
    obs_dim, action_dim = dims[name].observation_dims["policy"], dims[name].action_dim
    actor = None
    if trunk is not None:
        encoder, decoder = torch.nn.Linear(obs_dim, args.latent), torch.nn.Linear(args.hidden[-1], action_dim)
        actor = torch.nn.Sequential(encoder, torch.nn.ELU(), trunk, torch.nn.ELU(), decoder)
    learner = PPO(
        obs_dim,
        action_dim,
        robot.stop - robot.start,
        DEVICE,
        hidden=args.hidden,
        steps=T,
        entropy=args.entropy,
        init_std=args.init_std,
        actor=actor,
    )
    learners[name] = (learner, robot, obs_dim, action_dim)
    print(f"SETUP {name}: worlds {robot.start}-{robot.stop} obs {obs_dim} actions {action_dim}", flush=True)

# A discriminator per robot with a style task: the transitions of its clips, which it should score 1, the
# transitions of the worlds of the style task, which it should score -1, and the steps of a rollout to learn from.
styles = {}
for name, config in zip(robots, env.mdp.world_configs):
    op = next((op for op in env.mdp.term_ops if op.use == f"rewards.{name}_style"), None)
    if op is None or name in styles:
        continue
    clip = {
        field: env.state.arrays[mdp2.term_state(f"events.{name}_reset_to_motion", f"clip_{field}")].torch
        for field in ("joint_pos", "joint_vel", "root_pose", "root_vel")
    }
    features = hil.motion_features(clip["joint_pos"], clip["joint_vel"], clip["root_pose"], clip["root_vel"])
    reference = torch.cat([features[:-1], features[1:]], dim=-1)
    discriminator = op.params["discriminator"]
    discriminator.mean.copy_(reference.mean(0))
    discriminator.std.copy_(reference.std(0).clamp(min=0.05))
    styled = slice(config.worlds.start, config.worlds.stop)
    styles[name] = {
        "discriminator": discriminator,
        "optimizer": torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr),
        "reference": reference,
        "transition": env.state.arrays[mdp2.term_state(f"rewards.{name}_style", "transition")].torch,
        "worlds": styled,
        "steps": torch.zeros(T, len(config.worlds), reference.shape[1], device=DEVICE),
    }
    print(f"SETUP {name}: style worlds {styled.start}-{styled.stop}, {len(reference)} clip transitions", flush=True)


def update_discriminator(style):
    """Teach a discriminator to score its clips' transitions 1 and the robot's -1, and return the style reward."""
    discriminator, reference = style["discriminator"], style["reference"]
    robot_transitions = style["steps"].flatten(0, 1)
    for _ in range(args.disc_steps):
        fake = robot_transitions[torch.randint(len(robot_transitions), (args.disc_batch,), device=DEVICE)]
        real = reference[torch.randint(len(reference), (args.disc_batch,), device=DEVICE)].clone().requires_grad_()
        real_score, fake_score = discriminator(real), discriminator(fake)
        loss = (real_score - 1.0).square().mean() + (fake_score + 1.0).square().mean()
        (slope,) = torch.autograd.grad(real_score.sum(), real, create_graph=True)
        loss = loss + args.penalty * slope.square().sum(-1).mean()
        style["optimizer"].zero_grad(set_to_none=True)
        loss.backward()
        style["optimizer"].step()
    return (1.0 - 0.25 * (fake_score.detach() - 1.0).square()).clamp(min=0.0).mean()


def policy(obs):
    """Return the actions of every policy in its worlds, without exploration noise."""
    for learner, robot, obs_dim, action_dim in learners.values():
        actions[robot, :action_dim] = learner.policy(obs[robot, :obs_dim])
    return actions


iterations = round(args.budget * 1.0e6 / (T * num_envs))
history = []
measures, measured = {}, 0
wp.synchronize()
start = time.perf_counter()
for k in range(1, iterations + 1):
    for t in range(T):
        for learner, robot, obs_dim, action_dim in learners.values():
            actions[robot, :action_dim] = learner.act(obs[robot, :obs_dim])
        obs, reward, terminated, truncated, _ = env.step(actions)
        obs = obs["policy"]
        for learner, robot, _, _ in learners.values():
            learner.record(reward[robot], terminated[robot], truncated[robot])
        for style in styles.values():
            style["steps"][t] = style["transition"][style["worlds"]]
        # The measures are averaged over the steps between two log rows: the commands change every ten seconds.
        for name, value in measure(env, worlds).items():
            measures[name] = measures.get(name, 0.0) + value
        measured += 1
    for learner, robot, obs_dim, _ in learners.values():
        learner.update(obs[robot, :obs_dim])
    for name, style in styles.items():
        measures[f"{name}/style"] = measures.get(f"{name}/style", 0.0) + T * update_discriminator(style)

    if k % args.log_interval == 0 or k == iterations:
        torch.cuda.synchronize()
        row = {"iteration": k, "samples_m": k * T * num_envs / 1.0e6, "time": time.perf_counter() - start}
        row |= {name: value.item() / measured for name, value in measures.items()}
        measures, measured = {}, 0
        for name, (learner, _, _, _) in learners.items():
            row[f"{name}/reward"] = learner.buffer["rewards"].mean().item()
        print("LOG " + " ".join(f"{name} {value:.3f}" for name, value in row.items()), flush=True)
        # The file also holds the rewards of the last episodes, per term.
        row |= {term: value.item() for term, value in env.extras["log"].items() if term.startswith("Episode_Reward")}
        history.append(row)
train_time = time.perf_counter() - start
samples = iterations * T * num_envs
print(f"TRAIN {train_time:.1f} s, {samples / train_time / 1e6:.2f} M steps/s, graphs {env.num_graphs}")

# Without exploration noise, over twenty seconds from a reset; the measures are averaged over the last ten.
totals = {}
with torch.no_grad():
    obs = env.reset()[0]["policy"]
    for step in range(1000):
        obs = env.step(policy(obs))[0]["policy"]
        if step >= 500:
            for name, value in measure(env, worlds).items():
                totals[name] = totals.get(name, 0.0) + value / 500
totals = {name: value.item() for name, value in totals.items()}
print("EVAL " + " ".join(f"{name} {value:.3f}" for name, value in totals.items()))
if args.out:
    with open(args.out, "w") as file:
        json.dump({"args": vars(args), "train_time": train_time, "history": history, "evaluation": totals}, file)
env.close()

if args.video:
    # Four worlds per robot with its velocity task, in a row per robot; each camera follows the robot of its world.
    columns = 4
    env, robots = make_env(columns * len(NAMES), style=0.0, video=True)
    actions = torch.zeros(env.num_envs, env.mdp.action_dim, device=DEVICE)
    for name, robot in robot_worlds(env, robots).items():
        learners[name] = (learners[name][0], robot, *learners[name][2:])

    def roots():
        return torch.cat([env.scene[name].data.root_link_pose_w.torch[learners[name][1], :3] for name in NAMES])

    record(env, policy, args.video, 500, columns, eye=EYE, roots=roots)
    env.close()
