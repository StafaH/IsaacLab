# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train one policy on every skill of an MDP2 multi-skill task.

    uv run python scripts/reinforcement_learning/mdp2/train_skills.py --budget 300

The worlds are split between the skills, each with its own rewards, terminations, and resets, and the policy
observes which skill its world practices. The log reports, per skill, what the skill is about: the velocity error
of walking, for the postures how many worlds hold them, and for walking on the hind legs the mean speed [m/s]
of the worlds while upright.
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

from isaaclab.utils import math as math_utils
from isaaclab.utils import replace

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.skills.config.go2.go2_env_cfg import SKILLS
from isaaclab_tasks.mdp2.skills.skills_env_cfg import skill_tasks
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Skills-UnitreeGo2-MDP2")
parser.add_argument("--skills", default="walk,hind_legs,hind_walk,get_up", help="Skills to train")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=float, default=300.0, help="Samples to train for, in millions")
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--init_std", type=float, default=1.0)
parser.add_argument("--entropy", type=float, default=0.005)
parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--action_scale", type=float, default=None)
parser.add_argument("--out", default=None)
parser.add_argument("--video", default=None, help="Path of an MP4 with one world per skill")
args = parser.parse_args()

DEVICE = "cuda:0"
NAMES = args.skills.split(",")
EYE = (1.6, -1.8, 0.5)
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True


def make_env(num_envs, camera=False):
    cfg = resolve_presets(load_cfg_from_registry(args.task, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, num_envs, args.seed
    tasks = skill_tasks({name: SKILLS[name] for name in NAMES})
    if args.action_scale is not None:
        for task in tasks:
            task.actions["joint_pos"] = replace(task.actions["joint_pos"], scale=args.action_scale)
    cfg.task = tasks if len(tasks) > 1 else tasks[0]
    if camera:
        add_camera(cfg, eye=EYE)
    return gym.make(args.task, cfg=cfg).unwrapped


env = make_env(args.num_envs)
obs = env.reset()[0]["policy"]
num_envs, action_dim = env.num_envs, env.mdp.action_dim
T = args.steps
iterations = round(args.budget * 1.0e6 / (T * num_envs))
ppo = PPO(
    obs.shape[1],
    action_dim,
    num_envs,
    DEVICE,
    hidden=args.hidden,
    steps=T,
    entropy=args.entropy,
    init_std=args.init_std,
)
ranges = {name: config.worlds for name, config in zip(NAMES, env.mdp.world_configs)}


def measure(env):
    """Return, per skill, the measure of the skill: velocity error [m/s] of walking, share of worlds in the posture."""
    robot = env.scene["robot"].data
    pose, velocity = robot.root_link_pose_w.torch, robot.root_com_vel_w.torch
    quat, height = pose[:, 3:7], pose[:, 2]
    forward_up = math_utils.quat_apply(quat, torch.tensor([1.0, 0.0, 0.0], device=DEVICE).expand(len(quat), 3))[:, 2]
    top_up = math_utils.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=DEVICE).expand(len(quat), 3))[:, 2]
    belly = -math_utils.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=DEVICE).expand(len(quat), 3))[:, :2]
    belly_speed = (torch.nn.functional.normalize(belly, dim=-1) * velocity[:, :2]).sum(-1)
    command = env.state.arrays[mdp2.commands.base_velocity].torch
    speed_error = (command[:, :2] - math_utils.quat_apply_inverse(quat, velocity[:, :3])[:, :2]).norm(dim=-1)
    held = {
        "walk": speed_error,
        "hind_legs": ((forward_up > 0.8) & (height > 0.4)).float(),
        "handstand": ((forward_up < -0.8) & (height > 0.35)).float(),
        "hind_walk": torch.where((forward_up > 0.8) & (height > 0.4), belly_speed, torch.zeros_like(height)),
        "get_up": ((top_up > 0.9) & (height > 0.25)).float(),
    }
    names = NAMES if len(env.mdp.world_configs) > 1 else NAMES[:1]
    worlds = {name: config.worlds for name, config in zip(names, env.mdp.world_configs)}
    return {name: held[name][worlds[name].start : worlds[name].stop].mean().item() for name in names}


history = []
wp.synchronize()
start = time.perf_counter()
for k in range(1, iterations + 1):
    for _ in range(T):
        obs, reward, terminated, truncated, _ = env.step(ppo.act(obs))
        obs = obs["policy"]
        ppo.record(reward, terminated, truncated)
    ppo.update(obs)
    if k % args.log_interval == 0 or k == iterations:
        torch.cuda.synchronize()
        row = {
            "iteration": k,
            "samples_m": k * T * num_envs / 1.0e6,
            "time": time.perf_counter() - start,
            **measure(env),
        }
        row["reward"] = ppo.buffer["rewards"].mean().item()
        history.append(row)
        print("LOG " + " ".join(f"{name} {value:.3f}" for name, value in row.items()), flush=True)
train_time = time.perf_counter() - start
samples = iterations * T * num_envs
print(f"TRAIN {train_time:.1f} s, {samples / train_time / 1e6:.2f} M steps/s, graphs {env.num_graphs}")

# Without exploration noise, over ten seconds from a reset; the measures are averaged over the last five.
totals = {}
with torch.no_grad():
    obs = env.reset()[0]["policy"]
    for step in range(500):
        obs = env.step(ppo.policy(obs))[0]["policy"]
        if step >= 250:
            for name, value in measure(env).items():
                totals[name] = totals.get(name, 0.0) + value / 250
print("EVAL " + " ".join(f"{name} {value:.3f}" for name, value in totals.items()))
if args.out:
    with open(args.out, "w") as file:
        json.dump({"args": vars(args), "train_time": train_time, "history": history, "evaluation": totals}, file)
env.close()

if args.video:
    env = make_env(len(NAMES), camera=True)
    record(env, ppo.policy, args.video, 300, len(NAMES), eye=EYE)
    env.close()
