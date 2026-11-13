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
of evaluation episodes that track for their whole length, the same share and the length of an episode [steps]
averaged over the start frames, and the mean distance of the bodies to the clip [mm]. Per interface it logs the
mean reward of the training worlds, the mean length of their episodes and the share of them that failed.
``--save_every`` saves the state of the learner while training, and ``--resume`` continues from such a state in a
fresh simulation.

``--allocator`` moves the shares while training, at every ``--allocate_every`` log rows: ``progress``, ``failure``,
``headroom`` and ``uniform`` are the rules of :class:`~isaaclab_tasks.mdp2.interfaces.allocation.Allocator` on the
competence of every interface, the share of an episode that its evaluation worlds on training clips track, with
``--signal val`` that its evaluation worlds on the validation clips track, a part of the held-out clips, or with
``--signal rollout`` that its training worlds track. ``frontier`` gives the worlds to the interfaces whose
evaluation worlds on training clips have an open outcome: the priority of an interface is the mean of ``p (1 - p)``
over their start frames, with ``p`` the share of the recent episodes from a start frame that tracked to the end.
``tscl`` and ``alp`` follow the mean reward of the training worlds, and ``ranked`` their mean absolute advantage.
With ``--graph``, an interface gets worlds only once the policy is competent in the interfaces that it relaxes.

``--obs_clip`` bounds the normalized observations. Without it, an input that was constant so far, such as the flag
of an interface that had no worlds, takes a huge value when it changes.

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
from isaaclab_tasks.mdp2.interfaces.allocation import ALP, TSCL, Allocator, RankedScore
from isaaclab_tasks.mdp2.interfaces.config.g1.g1_env_cfg import PARENTS, TEST_CLIPS, TRAIN_CLIPS, VAL_CLIPS, arm_names
from isaaclab_tasks.mdp2.interfaces.interfaces_env_cfg import INTERFACE
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Interfaces-G1-MDP2")
parser.add_argument("--push", type=float, default=1.0, help="Scale of the pushes of the pushed worlds")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--eval_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=float, default=1000.0, help="Samples to train for, in millions")
parser.add_argument("--shares", default="full=1", help="Shares of the interfaces, such as full=1,three_point=1")
parser.add_argument("--shares_after", default=None, help="Shares after --switch_at")
parser.add_argument("--switch_at", type=float, default=0.0, help="Samples after which the shares change, in millions")
parser.add_argument(
    "--allocator",
    default=None,
    choices=["progress", "failure", "headroom", "frontier", "uniform", "tscl", "alp", "ranked"],
)
parser.add_argument("--signal", default="eval", choices=["eval", "val", "rollout"])
parser.add_argument("--allocate_every", type=int, default=1, help="Log rows between two updates of the shares")
parser.add_argument("--graph", action="store_true", help="Open an arm once the policy is competent in its parents")
parser.add_argument("--floor", type=float, default=0.2, help="Share that the open arms keep in equal parts")
parser.add_argument("--anchor_share", type=float, default=0.0, help="Least share of the arms without parents")
parser.add_argument("--fast", type=float, default=0.3, help="Rate of the fast average of the competence")
parser.add_argument("--slow", type=float, default=0.05, help="Rate of the slow average of the competence")
parser.add_argument("--rate", type=float, default=0.2, help="Rate at which the shares follow their target")
parser.add_argument("--teacher", default=None, help="Model of a policy that labels the steps")
parser.add_argument("--self_distill", action="store_true")
parser.add_argument("--imitation", type=float, default=1.0)
parser.add_argument("--rl", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--init_std", type=float, default=1.0)
parser.add_argument("--entropy", type=float, default=0.005)
parser.add_argument("--obs_clip", type=float, default=None, help="Range of the normalized observations")
parser.add_argument("--hidden", type=int, nargs="+", default=[1024, 512, 256])
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--out", default=None, help="Path of a JSON file with the log")
parser.add_argument("--save", default=None, help="Path to save the model to")
parser.add_argument("--save_every", type=float, default=0.0, help="Samples between checkpoints, in millions")
parser.add_argument("--resume", default=None, help="Checkpoint to continue from; samples count from zero again")
args = parser.parse_args()

TASK, DEVICE = args.task, "cuda:0"
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
NAMES = arm_names(cfg.pushes)
env = gym.make(TASK, cfg=cfg).unwrapped
arrays = env.state.arrays
arrays[mdp2.term_state(INTERFACE, "pushes")].torch.mul_(args.push)
shares = arrays[mdp2.term_state(INTERFACE, "shares")].torch
interface = arrays[mdp2.term_state(INTERFACE, "interface")].torch
clip_range = arrays[mdp2.term_state("events.reset_to_motion", "clip_range")].torch.long()
body_pos_error = arrays[mdp2.quantities.body_pos_error].torch

# The evaluation worlds: a block per interface and split, on start frames spread evenly over the split's clips.
N, E = args.num_envs, args.eval_envs
T, block = N - E, E // (len(NAMES) * len(SPLITS))
episode_frames = round(10.0 / env.step_dt)
group = torch.arange(E, device=DEVICE) // block
# The evaluation worlds on the validation clips.
val = torch.zeros(E, dtype=torch.bool, device=DEVICE)
val_ranges = [clip_range[len(TRAIN_CLIPS) + TEST_CLIPS.index(name)].tolist() for name in VAL_CLIPS]
for index in range(len(NAMES) * len(SPLITS)):
    clips = clip_range[: len(TRAIN_CLIPS)] if index % len(SPLITS) == 0 else clip_range[len(TRAIN_CLIPS) :]
    frames = torch.cat([torch.arange(first, last - episode_frames, device=DEVICE) for first, last in clips.tolist()])
    worlds = slice(T + index * block, T + (index + 1) * block)
    starts = frames[torch.linspace(0, len(frames) - 1, block, device=DEVICE).long()]
    arrays[mdp2.term_state("events.reset_to_motion", "fixed_start")].torch[worlds] = starts.float()
    if index % len(SPLITS) == 1:
        val[index * block : (index + 1) * block] = sum(
            (starts >= first) & (starts < last) for first, last in val_ranges
        )
    arrays[mdp2.term_state(INTERFACE, "fixed")].torch[worlds] = index // len(SPLITS)
shares.copy_(parse_shares(args.shares))

allocator = None
if args.allocator in ("progress", "failure", "headroom", "frontier", "uniform"):
    parents = [[NAMES.index(parent) for parent in PARENTS[name]] if args.graph else [] for name in NAMES]
    anchors = [index for index, name in enumerate(NAMES) if not PARENTS[name]]
    allocator = Allocator(
        parents,
        anchors,
        args.allocator,
        args.fast,
        args.slow,
        floor=args.floor,
        anchor_share=args.anchor_share,
        rate=args.rate,
    )
elif args.allocator:
    allocator = {"tscl": TSCL, "alp": ALP, "ranked": RankedScore}[args.allocator](len(NAMES))
if allocator is not None:
    shares.copy_(torch.tensor(allocator.shares, device=DEVICE))

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
    obs_clip=args.obs_clip,
)
ppo.imitation_coef.fill_(args.imitation)
ppo.rl_coef.fill_(args.rl)
if args.resume:
    state = torch.load(args.resume)
    ppo.agent.load_state_dict(state["agent"])
    for name in ("first", "second", "count", "lr"):
        getattr(ppo, name).copy_(state[name])
teacher = None
if args.teacher:
    teacher = Agent(obs_dim, action_dim, args.hidden, args.init_std, critic_dim, args.obs_clip).to(DEVICE)
    state = torch.load(args.teacher)
    teacher.load_state_dict(state.get("agent", state))
print(f"DEVICE {torch.cuda.get_device_name(0)}")
print(f"SETUP obs {obs_dim} critic {critic_dim} actions {action_dim} train worlds {T} eval block {block}", flush=True)

iterations = round(args.budget * 1.0e6 / (args.steps * T))
save_every = round(args.save_every * 1.0e6 / (args.steps * T))
episodes, completed = torch.zeros(E, device=DEVICE), torch.zeros(E, device=DEVICE)
distance, counted = torch.zeros(E, device=DEVICE), torch.zeros(E, device=DEVICE)
# Steps of the running episode of every world, and of the episodes that ended, for the evaluation worlds.
alive, lengths = torch.zeros(N, device=DEVICE), torch.zeros(E, device=DEVICE)
# Of the training worlds, per interface: steps, summed reward, episodes that ended, their steps, and failures.
rollout = torch.zeros(5, len(NAMES), device=DEVICE)
# The interface of every sample of a rollout, and per interface the samples and their summed absolute advantage.
sample_interface = torch.zeros(args.steps, T, dtype=torch.long, device=DEVICE)
advantage = torch.zeros(2, len(NAMES), device=DEVICE)
signal, frontier = [0.0] * len(NAMES), [0.0] * len(NAMES)
# The success of every evaluation world, averaged over the log rows.
start_success = None
history = []
wp.synchronize()
start = time.perf_counter()
for k in range(1, iterations + 1):
    if args.shares_after and (k - 1) * args.steps * T >= args.switch_at * 1.0e6:
        shares.copy_(parse_shares(args.shares_after))
    for step in range(args.steps):
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
            # The step draws the interface of the next episode of a world that ends: keep the one it had.
            had = interface[:T].long()
            sample_interface[step] = had
        obs, reward, terminated, truncated, _ = env.step(action)
        ppo.record(reward[:T], terminated[:T], truncated[:T])
        done = terminated | truncated
        alive += 1.0
        episodes += done[T:]
        completed += truncated[T:]
        lengths += alive[T:] * done[T:]
        distance += body_pos_error[T:].sqrt()
        counted += 1.0
        ended = done[:T].float()
        values = torch.stack([torch.ones_like(ended), reward[:T], ended, alive[:T] * ended, terminated[:T].float()])
        rollout.index_add_(1, had, values)
        alive *= ~done
    ppo.update(obs["policy"][:T], obs["critic"][:T])
    magnitude = ppo.flat["advantages"].abs()
    advantage.index_add_(1, sample_interface.reshape(-1), torch.stack([torch.ones_like(magnitude), magnitude]))

    if k % args.log_interval == 0:
        torch.cuda.synchronize()
        row = {"iteration": k, "samples_m": k * args.steps * T / 1.0e6, "time": time.perf_counter() - start}
        row["reward"] = ppo.buffer["rewards"].mean().item()
        row["shares"] = shares.tolist()
        succeeded = completed / episodes.clamp(min=1.0)
        start_success = succeeded if start_success is None else start_success + args.fast * (succeeded - start_success)
        for index in range(len(NAMES) * len(SPLITS)):
            name = f"{NAMES[index // len(SPLITS)]}/{SPLITS[index % len(SPLITS)]}"
            worlds = group == index
            # The share of the start frames whose outcome is open: 1 when all succeed half of the time.
            row[f"{name}/frontier"] = 4.0 * (start_success[worlds] * (1.0 - start_success[worlds])).mean().item()
            row[f"{name}/success"] = (completed[worlds].sum() / episodes[worlds].sum().clamp(min=1.0)).item()
            row[f"{name}/error_mm"] = 1.0e3 * (distance[worlds].sum() / counted[worlds].sum()).item()
            # Per start frame, so that a start that fails early does not count more often than one that succeeds.
            ended = episodes[worlds].clamp(min=1.0)
            row[f"{name}/start_success"] = succeeded[worlds].mean().item()
            row[f"{name}/length"] = (lengths[worlds] / ended).mean().item()
            if index % len(SPLITS) == 1:
                seen = val[worlds]
                row[f"{NAMES[index // len(SPLITS)]}/val/length"] = (lengths[worlds] / ended)[seen].mean().item()
        for index, name in enumerate(NAMES):
            steps, rewards, ended, ended_steps, failed = rollout[:, index].tolist()
            row[f"{name}/rollout/reward"] = rewards / max(steps, 1.0)
            row[f"{name}/rollout/length"] = ended_steps / max(ended, 1.0)
            row[f"{name}/rollout/failed"] = failed / max(ended, 1.0)
            row[f"{name}/rollout/advantage"] = advantage[1, index].item() / max(advantage[0, index].item(), 1.0)
            # The signal of the allocator; an interface without training worlds keeps the one it had.
            if args.allocator in ("tscl", "alp"):
                signal[index] = row[f"{name}/rollout/reward"] if steps else signal[index]
            elif args.allocator == "ranked":
                signal[index] = row[f"{name}/rollout/advantage"] if steps else signal[index]
            elif args.signal == "rollout":
                signal[index] = row[f"{name}/rollout/length"] / episode_frames if ended else signal[index]
            else:
                signal[index] = row[f"{name}/{'val' if args.signal == 'val' else 'train'}/length"] / episode_frames
            if args.signal == "rollout":
                # Training worlds have no fixed start frames: the share of their episodes that fail stands in.
                failed = row[f"{name}/rollout/failed"]
                frontier[index] = 4.0 * failed * (1.0 - failed) if ended else frontier[index]
            else:
                frontier[index] = row[f"{name}/train/frontier"]
        for value in (episodes, completed, distance, counted, lengths, rollout, advantage):
            value.zero_()
        if allocator is not None and (len(history) + 1) % args.allocate_every == 0:
            arguments = (signal, frontier) if args.allocator == "frontier" else (signal,)
            shares.copy_(torch.tensor(allocator.update(*arguments), device=DEVICE))
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
    if args.save and save_every and k % save_every == 0:
        state = {"agent": ppo.agent.state_dict(), "first": ppo.first, "second": ppo.second}
        state |= {"count": ppo.count, "lr": ppo.lr, "samples_m": k * args.steps * T / 1.0e6}
        torch.save(state, f"{args.save.removesuffix('.pt')}_{round(k * args.steps * T / 1.0e6)}M.pt")
train_time = time.perf_counter() - start
samples = iterations * args.steps * T
print(f"TRAIN {train_time:.1f} s, {samples / train_time / 1e6:.2f} M steps/s, graphs {env.num_graphs}")
if args.save:
    torch.save(ppo.agent.state_dict(), args.save)
env.close()
