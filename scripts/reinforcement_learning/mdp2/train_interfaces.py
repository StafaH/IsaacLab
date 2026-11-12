# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train one policy to track motion clips through several command interfaces, with evaluation worlds.

    uv run python scripts/reinforcement_learning/mdp2/train_interfaces.py --shares full=1 --budget 1000

Every training world is told its targets through an interface drawn from ``--shares`` when its episode starts; the
shares change to ``--shares_after`` once ``--switch_at`` million samples are collected. The last ``--eval_envs``
worlds evaluate: each has a fixed interface and a fixed start frame, on a training clip or on a held-out clip, acts
without exploration noise, and gives no data to the learner. The script logs, per interface and split, the share
of evaluation episodes that track for their whole length and the mean distance of the bodies to the clip [mm].

``--teacher`` loads a policy that labels every step with its action from the targets of every group, and
``--self_distill`` uses the learning policy's own action from those targets as the label in the worlds of the
other interfaces. ``--imitation`` weights the distance to the labels, and ``--rl 0`` leaves only that loss.
"""

import argparse
import json
import time

import gymnasium as gym
import torch
import warp as wp
from isaaclab_experimental import mdp2
from ppo import PPO, Agent

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.interfaces.config.g1.g1_env_cfg import INTERFACES, TRAIN_CLIPS
from isaaclab_tasks.mdp2.interfaces.interfaces_env_cfg import INTERFACE
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--eval_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=float, default=1000.0, help="Samples to train for, in millions")
parser.add_argument("--shares", default="full=1", help="Shares of the interfaces, such as full=1,three_point=1")
parser.add_argument("--shares_after", default=None, help="Shares after --switch_at")
parser.add_argument("--switch_at", type=float, default=0.0, help="Samples after which the shares change, in millions")
parser.add_argument("--teacher", default=None, help="Model of a policy that labels the steps")
parser.add_argument("--self_distill", action="store_true")
parser.add_argument("--imitation", type=float, default=1.0)
parser.add_argument("--rl", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--init_std", type=float, default=1.0)
parser.add_argument("--entropy", type=float, default=0.005)
parser.add_argument("--hidden", type=int, nargs="+", default=[1024, 512, 256])
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--out", default=None, help="Path of a JSON file with the log")
parser.add_argument("--save", default=None, help="Path to save the model to")
args = parser.parse_args()

TASK, DEVICE = "Isaac-Interfaces-G1-MDP2", "cuda:0"
NAMES = list(INTERFACES)
SPLITS = ("train", "test")
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True


def parse_shares(text):
    """Return the share of every interface from text such as ``full=1,three_point=1``, normalized."""
    values = dict(item.split("=") for item in text.split(","))
    shares = torch.tensor([float(values.get(name, 0.0)) for name in NAMES], device=DEVICE)
    return shares / shares.sum()


cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, args.num_envs, args.seed
env = gym.make(TASK, cfg=cfg).unwrapped
arrays = env.state.arrays
shares = arrays[mdp2.term_state(INTERFACE, "shares")].torch
interface = arrays[mdp2.term_state(INTERFACE, "interface")].torch
clip_range = arrays[mdp2.term_state("events.reset_to_motion", "clip_range")].torch.long()
body_pos_error = arrays[mdp2.quantities.body_pos_error].torch

# The evaluation worlds: a block per interface and split, on start frames spread evenly over the split's clips.
N, E = args.num_envs, args.eval_envs
T, block = N - E, E // (len(NAMES) * len(SPLITS))
episode_frames = round(10.0 / env.step_dt)
group = torch.arange(E, device=DEVICE) // block
for index in range(len(NAMES) * len(SPLITS)):
    clips = clip_range[: len(TRAIN_CLIPS)] if index % len(SPLITS) == 0 else clip_range[len(TRAIN_CLIPS) :]
    frames = torch.cat([torch.arange(first, last - episode_frames, device=DEVICE) for first, last in clips.tolist()])
    worlds = slice(T + index * block, T + (index + 1) * block)
    starts = frames[torch.linspace(0, len(frames) - 1, block, device=DEVICE).long()]
    arrays[mdp2.term_state("events.reset_to_motion", "fixed_start")].torch[worlds] = starts.float()
    arrays[mdp2.term_state(INTERFACE, "fixed")].torch[worlds] = index // len(SPLITS)
shares.copy_(parse_shares(args.shares))

obs = env.reset()[0]
obs_dim, critic_dim, action_dim = obs["policy"].shape[1], obs["critic"].shape[1], env.mdp.action_dim
imitate = args.teacher is not None or args.self_distill
ppo = PPO(
    obs_dim,
    action_dim,
    T,
    DEVICE,
    hidden=args.hidden,
    steps=args.steps,
    entropy=args.entropy,
    init_std=args.init_std,
    critic_dim=critic_dim,
    imitation=imitate,
)
ppo.imitation_coef.fill_(args.imitation)
ppo.rl_coef.fill_(args.rl)
teacher = None
if args.teacher:
    teacher = Agent(obs_dim, action_dim, args.hidden, args.init_std, critic_dim).to(DEVICE)
    teacher.load_state_dict(torch.load(args.teacher))
print(f"SETUP obs {obs_dim} critic {critic_dim} actions {action_dim} train worlds {T} eval block {block}", flush=True)

iterations = round(args.budget * 1.0e6 / (args.steps * T))
episodes, completed = torch.zeros(E, device=DEVICE), torch.zeros(E, device=DEVICE)
distance, counted = torch.zeros(E, device=DEVICE), torch.zeros(E, device=DEVICE)
history = []
wp.synchronize()
start = time.perf_counter()
for k in range(1, iterations + 1):
    if args.shares_after and (k - 1) * args.steps * T >= args.switch_at * 1.0e6:
        shares.copy_(parse_shares(args.shares_after))
    for _ in range(args.steps):
        with torch.no_grad():
            targets, weights = None, None
            if teacher is not None:
                targets = teacher.actor(teacher.obs(obs["teacher"][:T]))
                weights = torch.ones(T, device=DEVICE)
            elif args.self_distill:
                targets = ppo.policy(obs["teacher"][:T])
                weights = (interface[:T] != NAMES.index("full")).float()
            action = ppo.act(obs["policy"][:T], obs["critic"][:T], targets, weights)
            action = torch.cat([action, ppo.policy(obs["policy"][T:])])
        obs, reward, terminated, truncated, _ = env.step(action)
        ppo.record(reward[:T], terminated[:T], truncated[:T])
        episodes += terminated[T:] | truncated[T:]
        completed += truncated[T:]
        distance += body_pos_error[T:].sqrt()
        counted += 1.0
    ppo.update(obs["policy"][:T], obs["critic"][:T])

    if k % args.log_interval == 0 or k == iterations:
        torch.cuda.synchronize()
        row = {"iteration": k, "samples_m": k * args.steps * T / 1.0e6, "time": time.perf_counter() - start}
        row["reward"] = ppo.buffer["rewards"].mean().item()
        row["shares"] = shares.tolist()
        for index in range(len(NAMES) * len(SPLITS)):
            name = f"{NAMES[index // len(SPLITS)]}/{SPLITS[index % len(SPLITS)]}"
            worlds = group == index
            row[f"{name}/success"] = (completed[worlds].sum() / episodes[worlds].sum().clamp(min=1.0)).item()
            row[f"{name}/error_mm"] = 1.0e3 * (distance[worlds].sum() / counted[worlds].sum()).item()
        for value in (episodes, completed, distance, counted):
            value.zero_()
        history.append(row)
        summary = " ".join(
            f"{name}:{row[f'{name}/train/success']:.2f}/{row[f'{name}/test/success']:.2f}"
            f"({row[f'{name}/train/error_mm']:.0f}/{row[f'{name}/test/error_mm']:.0f}mm)"
            for name in NAMES
        )
        print(f"LOG {row['samples_m']:.0f}M {row['time']:.0f}s reward {row['reward']:.3f} {summary}", flush=True)
        if args.out:
            with open(args.out, "w") as file:
                json.dump({"args": vars(args), "history": history}, file)
train_time = time.perf_counter() - start
samples = iterations * args.steps * T
print(f"TRAIN {train_time:.1f} s, {samples / train_time / 1e6:.2f} M steps/s, graphs {env.num_graphs}")
if args.save:
    torch.save(ppo.agent.state_dict(), args.save)
env.close()
