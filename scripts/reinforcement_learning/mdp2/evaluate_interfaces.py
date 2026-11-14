# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a policy of ``train_interfaces.py`` with every world as an evaluation world.

    uv run python scripts/reinforcement_learning/mdp2/evaluate_interfaces.py --model model.pt --out result.json

Every interface and split (training clips, held-out clips) gets the same number of worlds, on start frames spread
evenly over the clips of the split. Every world runs one episode without exploration noise. The script reports, per
interface and split, the share of the episodes that track for their whole length, their mean length [steps] and the
mean distance of the bodies to the clip [mm], and the share per clip. For the held-out clips it also reports the
share on the clips that no allocator reads (``unseen``), and with ``--clips all`` the share on the curated training
clips (``curated``) and on the hard ones (``hard``).
"""

import argparse
import json

import gymnasium as gym
import torch
from isaaclab_experimental import mdp2
from ppo import Agent

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.interfaces.config.g1.g1_env_cfg import (
    CLIP_SETS,
    TEST_CLIPS,
    TRAIN_CLIPS,
    VAL_CLIPS,
    arm_names,
    g1_interfaces_task,
)
from isaaclab_tasks.mdp2.interfaces.interfaces_env_cfg import INTERFACE
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Model or checkpoint of train_interfaces.py")
parser.add_argument("--task", default="Isaac-Interfaces-G1-MDP2")
parser.add_argument("--clips", default="curated", choices=list(CLIP_SETS), help="The training clips")
parser.add_argument("--local", action="store_true", help="Track without the position of the robot")
parser.add_argument("--push", type=float, default=1.0, help="Scale of the pushes of the pushed worlds")
parser.add_argument("--substeps", type=int, default=1, help="Physics steps per physics step of the task")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--hidden", type=int, nargs="+", default=[1024, 512, 256])
parser.add_argument("--out", default=None, help="Path of a JSON file with the result")
args = parser.parse_args()

TASK, DEVICE = args.task, "cuda:0"
train_clips = CLIP_SETS[args.clips]
CLIPS = (*train_clips, *TEST_CLIPS)
SPLITS = ("train", "test")

cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, args.num_envs, args.seed
cfg.sim.dt, cfg.decimation = cfg.sim.dt / args.substeps, cfg.decimation * args.substeps
cfg.sim.render_interval = cfg.decimation
if args.clips != "curated" or args.local:
    cfg.task = g1_interfaces_task(train=train_clips, pushes=cfg.pushes, local=args.local)
NAMES = arm_names(cfg.pushes)
env = gym.make(TASK, cfg=cfg).unwrapped
arrays = env.state.arrays
arrays[mdp2.term_state(INTERFACE, "pushes")].torch.mul_(args.push)
clip_range = arrays[mdp2.term_state("events.reset_to_motion", "clip_range")].torch.long()
fixed_start = arrays[mdp2.term_state("events.reset_to_motion", "fixed_start")].torch
body_pos_error = arrays[mdp2.quantities.body_pos_error].torch

N = args.num_envs
block = N // (len(NAMES) * len(SPLITS))
episode_frames = round(10.0 / env.step_dt)
group = torch.arange(N, device=DEVICE) // block
clip = torch.zeros(N, dtype=torch.long, device=DEVICE)
for index in range(len(NAMES) * len(SPLITS)):
    first = 0 if index % len(SPLITS) == 0 else len(train_clips)
    clips = clip_range[first : len(train_clips) if first == 0 else len(CLIPS)]
    frames = torch.cat([torch.arange(a, b - episode_frames, device=DEVICE) for a, b in clips.tolist()])
    owner = torch.cat(
        [torch.full((b - episode_frames - a,), first + k, device=DEVICE) for k, (a, b) in enumerate(clips.tolist())]
    )
    worlds = slice(index * block, (index + 1) * block)
    chosen = torch.linspace(0, len(frames) - 1, block, device=DEVICE).long()
    fixed_start[worlds], clip[worlds] = frames[chosen].float(), owner[chosen]
    arrays[mdp2.term_state(INTERFACE, "fixed")].torch[worlds] = index // len(SPLITS)

obs = env.reset()[0]
# The critic keeps the size it was trained with, which differs between the tasks; only the policy acts here.
state = torch.load(args.model)
state = state.get("agent", state)
critic_dim = state["critic.0.weight"].shape[1]
agent = Agent(obs["policy"].shape[1], env.mdp.action_dim, args.hidden, 1.0, critic_dim).to(DEVICE)
agent.load_state_dict(state)

# The first episode of every world: whether it is running, its steps, its outcome and its summed distance.
running = torch.ones(N, dtype=torch.bool, device=DEVICE)
length, success, distance = (torch.zeros(N, device=DEVICE) for _ in range(3))
with torch.no_grad():
    for _ in range(episode_frames + 1):
        obs, _, terminated, truncated, _ = env.step(agent.actor(agent.obs(obs["policy"])))
        length += running
        distance += running * body_pos_error.sqrt().nan_to_num(0.0)
        success += running & truncated
        running &= ~(terminated | truncated)

result = {"model": args.model, "episodes": block}
for index in range(len(NAMES) * len(SPLITS)):
    name = f"{NAMES[index // len(SPLITS)]}/{SPLITS[index % len(SPLITS)]}"
    worlds = group == index
    result[f"{name}/success"] = success[worlds].mean().item()
    result[f"{name}/length"] = length[worlds].mean().item()
    result[f"{name}/error_mm"] = 1.0e3 * (distance[worlds].sum() / length[worlds].sum()).item()
    for k in clip[worlds].unique().tolist():
        result[f"{name}/clips/{CLIPS[k]}"] = success[worlds & (clip == k)].mean().item()
    if index % len(SPLITS) == 1:
        unseen = worlds & ~sum(clip == CLIPS.index(k) for k in VAL_CLIPS).bool()
        result[f"{NAMES[index // len(SPLITS)]}/unseen/success"] = success[unseen].mean().item()
    elif len(train_clips) > len(TRAIN_CLIPS):
        hard = worlds & (clip >= len(TRAIN_CLIPS))
        result[f"{NAMES[index // len(SPLITS)]}/hard/success"] = success[hard].mean().item()
        result[f"{NAMES[index // len(SPLITS)]}/curated/success"] = success[worlds & ~hard].mean().item()
    print(
        f"EVAL {name} success {result[f'{name}/success']:.3f} length {result[f'{name}/length']:.0f}"
        f" error {result[f'{name}/error_mm']:.0f} mm"
    )
if args.out:
    with open(args.out, "w") as file:
        json.dump(result, file)
env.close()
