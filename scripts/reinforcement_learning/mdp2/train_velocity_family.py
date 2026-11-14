# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train one policy for the robots of a family, which walk in the worlds of one simulation.

    uv run python scripts/reinforcement_learning/mdp2/train_velocity_family.py --family quadrupeds --robots go2,go1

Each of ``--robots`` takes an equal share of the worlds and runs its velocity task. The policy reads and drives the
joints of the family: the joints of a robot take their places among them, and the places that a robot does not
have read zero and drive nothing. So one network, with no part of its own for a robot, acts in every world, and
can act in the worlds of a robot that it did not train on. The script logs, per robot, the error of the base
velocity [m/s] and the share of the last episodes that ended in a fall.

``--save`` writes the policy to a file and ``--load`` starts from one. With ``--budget 0`` the script trains
nothing and measures the loaded policy, so this measures a policy on a robot that it has not seen:

    ... --family quadrupeds --robots a1 --load go2_go1.pt --budget 0
"""

import argparse
import json
import time

import gymnasium as gym
import torch
import warp as wp
from isaaclab_experimental import mdp2
from ppo import PPO
from video import add_camera, record

from isaaclab.cloner import InclusionSet
from isaaclab.utils import math as math_utils
from isaaclab.utils import replace

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.velocity_family import humanoids_env_cfg, quadrupeds_env_cfg
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

FAMILIES = {
    "quadrupeds": ("Isaac-Velocity-Flat-Quadrupeds-MDP2", quadrupeds_env_cfg),
    "humanoids": ("Isaac-Velocity-Flat-Humanoids-MDP2", humanoids_env_cfg),
}

parser = argparse.ArgumentParser()
parser.add_argument("--family", default="quadrupeds", choices=list(FAMILIES))
parser.add_argument("--robots", default=None, help="Robots of the family to train, separated by commas: all")
parser.add_argument("--num_envs", type=int, default=4096, help="Worlds of each robot")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=float, default=100.0, help="Samples of each robot to train for, in millions")
parser.add_argument("--load", default=None, help="Path of a policy to start from")
parser.add_argument("--save", default=None, help="Path to write the policy to")
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--init_std", type=float, default=1.0)
parser.add_argument("--entropy", type=float, default=0.005)
parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--out", default=None, help="Path of a JSON file with the log")
parser.add_argument("--video", default=None, help="Path of an MP4 with worlds of every robot")
args = parser.parse_args()

TASK, family = FAMILIES[args.family]
NAMES = args.robots.split(",") if args.robots else list(family.TASKS)
DEVICE = "cuda:0"
EYE = (1.7, -2.0, 0.5)
# The policy reads the velocity of the base, gravity, and the command, then the positions, the velocities, and the
# last actions of the joints of the family.
BASE, JOINTS = 12, len(family.JOINTS)
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True


def make_env(worlds, video=False):
    """Return an environment with ``worlds`` worlds of each robot, in the order of the robots."""
    cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, worlds * len(NAMES), args.seed
    for name in family.TASKS:
        if name not in NAMES:
            setattr(cfg.scene, name, None)
            setattr(cfg.scene, f"{name}_contact_forces", None)
    # One robot alone: its worlds are alike and need no sleeping robot.
    cfg.scene.clone_cfg.clone_combinations = [InclusionSet(assets=[name]) for name in NAMES] if len(NAMES) > 1 else []
    cfg.sim.physics.solver_cfg.sleep_inactive_assets = len(NAMES) > 1
    tasks = [family.TASKS[name] for name in NAMES]
    if video:
        # The video shows the robots walking forward; the tasks also command sideways and backward velocities.
        for index, task in enumerate(tasks):
            command = task.commands["base_velocity"]
            forward = {**command.params, "lin_vel_x": (0.6, 1.0), "lin_vel_y": (0.0, 0.0)}
            tasks[index] = replace(task, commands={"base_velocity": replace(command, params=forward)})
        add_camera(cfg, eye=EYE)
    cfg.task, cfg.task_worlds = (tasks, [worlds] * len(NAMES)) if len(NAMES) > 1 else (tasks[0], None)
    return gym.make(TASK, cfg=cfg).unwrapped


def robot_layouts(env):
    """Return, per robot, its worlds, the places of its joints in the family, and the columns of its observations."""
    layouts = {}
    for name, config in zip(NAMES, env.mdp.world_configs):
        aliases = family.JOINT_ALIASES.get(name, {})
        joints = [family.JOINTS.index(aliases.get(joint, joint)) for joint in env.scene[name].joint_names]
        assert config.observation_dims["policy"] == BASE + 3 * len(joints)
        joints = torch.tensor(joints, device=DEVICE)
        columns = [torch.arange(BASE, device=DEVICE), *(BASE + k * JOINTS + joints for k in range(3))]
        layouts[name] = (slice(config.worlds.start, config.worlds.stop), joints, torch.cat(columns))
    return layouts


def measure(env, layouts):
    """Return, per robot, the error of the base velocity [m/s] and the share of the last episodes that fell."""
    command = env.state.arrays[mdp2.commands.base_velocity].torch
    row = {}
    for name, (robot, _, _) in layouts.items():
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


env = make_env(args.num_envs)
layouts = robot_layouts(env)
if len(NAMES) > 1:
    # The tasks split the worlds evenly, and so must the scene, for each task to run where its robot is.
    layout = env.scene.clone_plan.topology.world_prototype_layout
    for index, (robot, _, _) in enumerate(layouts.values()):
        assert (layout[robot] == index).all()
num_envs, T = env.num_envs, args.steps
actions = torch.zeros(num_envs, env.mdp.action_dim, device=DEVICE)
family_obs = torch.zeros(num_envs, BASE + 3 * JOINTS, device=DEVICE)
used = torch.zeros(num_envs, JOINTS, device=DEVICE)
for name, (robot, joints, _) in layouts.items():
    used[robot, joints] = 1.0
    print(f"SETUP {name}: worlds {robot.start}-{robot.stop} joints {len(joints)} of {JOINTS}", flush=True)

learner = PPO(
    family_obs.shape[1],
    JOINTS,
    num_envs,
    DEVICE,
    hidden=args.hidden,
    steps=T,
    entropy=args.entropy,
    init_std=args.init_std,
    action_mask=used if used.min() == 0.0 else None,
)
if args.load:
    learner.agent.load_state_dict(torch.load(args.load))


def to_family(obs, layouts):
    """Return the observations of every world with the joints of its robot in their places in the family."""
    for robot, _, columns in layouts.values():
        family_obs[robot, columns] = obs[robot, : len(columns)]
    return family_obs


def to_robots(family_actions, layouts):
    """Return the actions of every world for the joints of its robot."""
    for robot, joints, _ in layouts.values():
        actions[robot, : len(joints)] = family_actions[robot][:, joints]
    return actions


iterations = round(args.budget * 1.0e6 * len(NAMES) / (T * num_envs))
history = []
measures, measured = {}, 0
obs = to_family(env.reset()[0]["policy"], layouts)
wp.synchronize()
start = time.perf_counter()
for k in range(1, iterations + 1):
    for t in range(T):
        obs, reward, terminated, truncated, _ = env.step(to_robots(learner.act(obs), layouts))
        obs = to_family(obs["policy"], layouts)
        learner.record(reward, terminated, truncated)
        # The measures are averaged over the steps between two log rows: the commands change every ten seconds.
        for name, value in measure(env, layouts).items():
            measures[name] = measures.get(name, 0.0) + value
        measured += 1
    learner.update(obs)

    if k % args.log_interval == 0 or k == iterations:
        torch.cuda.synchronize()
        row = {"iteration": k, "samples_m": k * T * num_envs / 1.0e6, "time": time.perf_counter() - start}
        row |= {name: value.item() / measured for name, value in measures.items()}
        measures, measured = {}, 0
        for name, (robot, _, _) in layouts.items():
            row[f"{name}/reward"] = learner.buffer["rewards"][:, robot].mean().item()
        print("LOG " + " ".join(f"{name} {value:.3f}" for name, value in row.items()), flush=True)
        history.append(row)
train_time = time.perf_counter() - start
if iterations > 0:
    rate = iterations * T * num_envs / train_time / 1.0e6
    print(f"TRAIN {train_time:.1f} s, {rate:.2f} M steps/s, graphs {env.num_graphs}")
if args.save:
    torch.save(learner.agent.state_dict(), args.save)

# Without exploration noise, over twenty seconds from a reset; the measures are averaged over the last ten.
totals = {}
with torch.no_grad():
    obs = to_family(env.reset()[0]["policy"], layouts)
    for step in range(1000):
        obs = to_family(env.step(to_robots(learner.policy(obs), layouts))[0]["policy"], layouts)
        if step >= 500:
            for name, value in measure(env, layouts).items():
                totals[name] = totals.get(name, 0.0) + value / 500
totals = {name: value.item() for name, value in totals.items()}
print("EVAL " + " ".join(f"{name} {value:.3f}" for name, value in totals.items()))
if args.out:
    with open(args.out, "w") as file:
        json.dump({"args": vars(args), "train_time": train_time, "history": history, "evaluation": totals}, file)
env.close()

if args.video:
    # Four worlds per robot, in a row per robot; each camera follows the robot of its world.
    columns = 4
    env = make_env(columns, video=True)
    layouts = robot_layouts(env)
    actions = torch.zeros(env.num_envs, env.mdp.action_dim, device=DEVICE)
    family_obs = torch.zeros(env.num_envs, family_obs.shape[1], device=DEVICE)

    def policy(obs):
        return to_robots(learner.policy(to_family(obs, layouts)), layouts)

    def roots():
        return torch.cat([env.scene[name].data.root_link_pose_w.torch[layouts[name][0], :3] for name in NAMES])

    record(env, policy, args.video, 500, columns, eye=EYE, roots=roots)
    env.close()
