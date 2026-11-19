# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a Unitree G1 or Go2 to follow velocity commands, half of its worlds in the style of motion clips.

    uv run python scripts/reinforcement_learning/mdp2/train_velocity_style.py --robot g1 --budget 300

The environment is ``Isaac-Velocity-Flat-G1-Style-MDP2`` or ``Isaac-Velocity-Flat-UnitreeGo2-Style-MDP2``: the
first half of the worlds runs the style task, the second half the velocity task. One PPO policy acts in all of
them, and a discriminator learns to tell the transitions of the style worlds from those of the clips; its score is
the style reward. The learner is the one of ``train_velocity_g1_go2.py`` for one robot.

``--runtime mdp1`` trains the same task written with MDP1 managers, see ``velocity_style_mdp1_baseline.py``.
``--benchmark`` times ``env.step`` with random actions instead of training.

The script logs the error of the base velocity [m/s] in the style worlds and the velocity worlds, the share of the
last episodes that ended in a fall, and the style score, between 0 and 1. After training, it evaluates the policy
without exploration noise and reports the same measures.
"""

import argparse
import json
import statistics
import time

import gymnasium as gym
import torch
import warp as wp
from isaaclab_experimental import mdp2
from ppo import PPO

from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.hil import mdp as hil
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--robot", default="g1", choices=["g1", "go2"])
parser.add_argument("--runtime", default="mdp2", choices=["mdp2", "mdp1"])
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=float, default=300.0, help="Samples to train for, in millions")
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--init_std", type=float, default=1.0)
parser.add_argument("--entropy", type=float, default=0.005)
parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
parser.add_argument("--disc_lr", type=float, default=1.0e-4)
parser.add_argument("--disc_steps", type=int, default=2, help="Updates of the discriminator per iteration")
parser.add_argument("--disc_batch", type=int, default=4096)
parser.add_argument("--penalty", type=float, default=5.0, help="Weight of the discriminator's gradient penalty")
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--out", default=None, help="Path of a JSON file with the log")
parser.add_argument("--save", default=None, help="Path of a checkpoint of the policy and the discriminator")
parser.add_argument("--load", default=None, help="Checkpoint to load instead of training")
parser.add_argument("--video", default=None, help="Path of an MP4: worlds in the clip's style, then without it")
parser.add_argument("--benchmark", action="store_true", help="Time env.step with random actions, and exit")
args = parser.parse_args()

TASK = {"g1": "Isaac-Velocity-Flat-G1-Style-MDP2", "go2": "Isaac-Velocity-Flat-UnitreeGo2-Style-MDP2"}[args.robot]
DEVICE = "cuda:0"
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True


def make_env():
    """Return the environment, and its handles on the clips, the style transitions, and the velocity commands."""
    if args.runtime == "mdp1":
        from velocity_style_mdp1_baseline import style_cfg

        from isaaclab.envs import ManagerBasedRLEnv

        cfg = style_cfg(args.robot)
        cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, args.num_envs, args.seed
        env = ManagerBasedRLEnv(cfg)
        obs = env.reset()[0]["policy"]
        style = env.reward_manager.get_term_cfg("style").func
        clip = env.event_manager.get_term_cfg("reset_to_motion").func.clip
        handles = {
            "discriminator": style.discriminator,
            "transition": style.transition,
            "clip": {field: clip[f"clip_{field}"] for field in ("joint_pos", "joint_vel", "root_pose", "root_vel")},
            "command": lambda: env.command_manager.get_command("base_velocity"),
            "styled": slice(0, env.num_envs // 2),
            "action_dim": env.action_manager.total_action_dim,
        }
        return env, obs, handles

    cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, args.num_envs, args.seed
    if args.video:
        from video import add_camera

        add_camera(cfg)
    env = gym.make(TASK, cfg=cfg).unwrapped
    obs = env.reset()[0]["policy"]
    op = next(op for op in env.mdp.term_ops if op.use == "rewards.style")
    styled = env.mdp.world_configs[0].worlds
    handles = {
        "discriminator": op.params["discriminator"],
        "transition": env.state.arrays[mdp2.term_state("rewards.style", "transition")].torch,
        "clip": {
            field: env.state.arrays[mdp2.term_state("events.reset_to_motion", f"clip_{field}")].torch
            for field in ("joint_pos", "joint_vel", "root_pose", "root_vel")
        },
        "command": lambda: env.state.arrays[mdp2.commands.base_velocity].torch,
        "styled": slice(styled.start, styled.stop),
        "action_dim": env.mdp.action_dim,
    }
    return env, obs, handles


env, obs, handles = make_env()
num_envs, T, action_dim = env.num_envs, args.steps, handles["action_dim"]
styled = handles["styled"]
plain = slice(styled.stop, num_envs)
print(f"SETUP {args.runtime} {args.robot}: {num_envs} worlds, style worlds {styled.start}-{styled.stop}", flush=True)

if args.benchmark:
    # The median time of a step with random actions, after a warmup that also captures any graphs.
    actions = [torch.rand(num_envs, action_dim, device=DEVICE) * 0.2 - 0.1 for _ in range(16)]
    times = []
    with torch.no_grad():
        for i in range(360):
            wp.synchronize()
            torch.cuda.synchronize()
            start = time.perf_counter()
            env.step(actions[i % 16])
            wp.synchronize()
            torch.cuda.synchronize()
            if i >= 60:
                times.append(time.perf_counter() - start)
    step = statistics.median(times)
    print(
        f"BENCH {args.runtime} {args.robot}: {num_envs} worlds, step median {step * 1e6:.0f} us,"
        f" min {min(times) * 1e6:.0f} us, {num_envs / step / 1e6:.3f} M steps/s"
    )
    if args.out:
        with open(args.out, "w") as file:
            json.dump({"args": vars(args), "step_median_s": step, "step_min_s": min(times)}, file)
    env.close()
    raise SystemExit

learner = PPO(
    obs.shape[1],
    action_dim,
    num_envs,
    DEVICE,
    hidden=args.hidden,
    steps=T,
    entropy=args.entropy,
    init_std=args.init_std,
)

# The transitions of the clips, which the discriminator should score 1, normalize its inputs.
discriminator = handles["discriminator"]
clip = handles["clip"]
features = hil.motion_features(clip["joint_pos"], clip["joint_vel"], clip["root_pose"], clip["root_vel"])
reference = torch.cat([features[:-1], features[1:]], dim=-1)
discriminator.mean.copy_(reference.mean(0))
discriminator.std.copy_(reference.std(0).clamp(min=0.05))
optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr)
transitions = torch.zeros(T, styled.stop - styled.start, reference.shape[1], device=DEVICE)
print(f"SETUP obs {obs.shape[1]} actions {action_dim}, {len(reference)} clip transitions", flush=True)


def update_discriminator():
    """Teach the discriminator to score the clips' transitions 1 and the robot's -1, and return the style reward."""
    robot_transitions = transitions.flatten(0, 1)
    for _ in range(args.disc_steps):
        fake = robot_transitions[torch.randint(len(robot_transitions), (args.disc_batch,), device=DEVICE)]
        real = reference[torch.randint(len(reference), (args.disc_batch,), device=DEVICE)].clone().requires_grad_()
        real_score, fake_score = discriminator(real), discriminator(fake)
        loss = (real_score - 1.0).square().mean() + (fake_score + 1.0).square().mean()
        (slope,) = torch.autograd.grad(real_score.sum(), real, create_graph=True)
        loss = loss + args.penalty * slope.square().sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return (1.0 - 0.25 * (fake_score.detach() - 1.0).square()).clamp(min=0.0).mean()


def style_score():
    """The style reward of the last step in the style worlds, by the current discriminator."""
    with torch.no_grad():
        score = discriminator(handles["transition"][styled])
    return (1.0 - 0.25 * (score - 1.0).square()).clamp(min=0.0).mean()


def measure():
    """Return the error of the base velocity [m/s] in the style and velocity worlds, and the share that fell."""
    data = env.scene["robot"].data
    velocity = math_utils.quat_apply_inverse(data.root_link_pose_w.torch[:, 3:7], data.root_com_vel_w.torch[:, :3])
    error = (handles["command"]()[:, :2] - velocity[:, :2]).norm(dim=-1)
    falls = [
        value
        for term, value in env.extras["log"].items()
        if term.startswith("Episode_Termination/") and not term.endswith("time_out")
    ]
    return {"style/speed_error": error[styled].mean(), "velocity/speed_error": error[plain].mean(), "falls": sum(falls)}


if args.load:
    checkpoint = torch.load(args.load, map_location=DEVICE)
    learner.agent.load_state_dict(checkpoint["agent"])
    discriminator.load_state_dict(checkpoint["discriminator"])
iterations = 0 if args.load else round(args.budget * 1.0e6 / (T * num_envs))
history = []
measures, measured = {}, 0
wp.synchronize()
start = time.perf_counter()
for k in range(1, iterations + 1):
    for t in range(T):
        obs, reward, terminated, truncated, _ = env.step(learner.act(obs))
        obs = obs["policy"]
        learner.record(reward, terminated, truncated)
        transitions[t] = handles["transition"][styled]
        # The measures are averaged over the steps between two log rows: the commands change every ten seconds.
        for name, value in measure().items():
            measures[name] = measures.get(name, 0.0) + value
        measured += 1
    learner.update(obs)
    measures["style"] = measures.get("style", 0.0) + T * update_discriminator()

    if k % args.log_interval == 0 or k == iterations:
        torch.cuda.synchronize()
        row = {"iteration": k, "samples_m": k * T * num_envs / 1.0e6, "time": time.perf_counter() - start}
        row |= {name: float(value) / measured for name, value in measures.items()}
        row["reward"] = learner.buffer["rewards"].mean().item()
        measures, measured = {}, 0
        print("LOG " + " ".join(f"{name} {value:.3f}" for name, value in row.items()), flush=True)
        history.append(row)
train_time = time.perf_counter() - start
samples = iterations * T * num_envs
print(f"TRAIN {train_time:.1f} s, {samples / train_time / 1e6:.3f} M steps/s")

# Without exploration noise, over twenty seconds from a reset; the measures are averaged over the last ten.
totals = {}
with torch.no_grad():
    obs = env.reset()[0]["policy"]
    for step in range(1000):
        obs = env.step(learner.policy(obs))[0]["policy"]
        if step >= 500:
            for name, value in {**measure(), "style": style_score()}.items():
                totals[name] = totals.get(name, 0.0) + value / 500
totals = {name: float(value) for name, value in totals.items()}
print("EVAL " + " ".join(f"{name} {value:.3f}" for name, value in totals.items()))
if args.out:
    with open(args.out, "w") as file:
        log = {"args": vars(args), "train_time": train_time, "samples": samples, "steps_per_s": samples / train_time}
        json.dump({**log, "history": history, "evaluation": totals}, file)
if args.save:
    torch.save(
        {"args": vars(args), "agent": learner.agent.state_dict(), "discriminator": discriminator.state_dict()},
        args.save,
    )
if args.video:
    from video import record

    record(env, learner.policy, args.video, 400, min(3, num_envs // 2))
env.close()
