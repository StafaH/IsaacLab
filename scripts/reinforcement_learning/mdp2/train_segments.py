# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train one policy to track motion clips, with a learner that chooses where on the clips the training worlds start.

    uv run python scripts/reinforcement_learning/mdp2/train_segments.py --clips all --allocator gain --budget 2000

The training clips are cut into segments of ``--segment_s`` seconds, and a segment is an arm: the share of a segment
is the share of the training worlds that start an episode in it. ``--clips all`` adds the clips that a curated
training set leaves out, and ``--library`` takes the clips from an index file instead: a list of clips, each with
the path of its array ``file`` and its ``split`` (``train``, ``val`` or ``unseen``).

The first ``--segment_envs`` evaluation worlds take the segments in turn: each starts on the first frame of a
segment, acts without exploration noise, and gives no data to the learner. The competence of a segment is the share
of its frames that these worlds track. A share ``--segment_pushed`` of these worlds is pushed, so that the
competence counts tracking under pushes too. The other ``--test_envs`` evaluation worlds start on fixed frames of
the held-out clips.

Without ``--allocator`` the task chooses the starts: a share ``--adaptive`` of them where the episodes of the
training worlds fail, the rest uniformly over the training clips. With ``--allocator``, a rule of
:class:`~isaaclab_tasks.mdp2.interfaces.allocation.Allocator` turns the competences into the shares at every log
row: ``gain``, ``failure``, ``progress``, ``headroom`` or ``uniform``. The starts of a segment are spread evenly
over it, or with ``--within`` a share of them goes where the episodes of the training worlds fail in it: the rule
says which segment, the failures say where in the segment. A row measures the segments whose evaluation
episodes ended in it, which are not all of them when there are more segments than evaluation worlds.

The script logs the success and the competence over the segments, of the curated clips and of the hard clips, the
success on the held-out clips, the share of the training worlds on the hard clips, and at every
``--segment_log`` rows per segment its competence and its training worlds.
"""

import argparse
import json
import time

import gymnasium as gym
import torch
import warp as wp
from isaaclab_experimental import mdp2
from ppo import PPO

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.interfaces.allocation import Allocator
from isaaclab_tasks.mdp2.interfaces.config.g1.g1_env_cfg import (
    CLIP_SETS,
    TEST_CLIPS,
    TRAIN_CLIPS,
    VAL_CLIPS,
    arm_names,
    g1_interfaces_task,
)
from isaaclab_tasks.mdp2.interfaces.interfaces_env_cfg import INTERFACE
from isaaclab_tasks.mdp2.tracking.config.g1.g1_env_cfg import ARMATURE
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Interfaces-Robust-G1-MDP2")
parser.add_argument("--clips", default="all", choices=list(CLIP_SETS), help="The training clips")
parser.add_argument("--library", default=None, help="Index file of a library of clips, in place of --clips")
parser.add_argument("--segment_log", type=int, default=1, help="Log rows between two logs of the segments")
parser.add_argument("--shares", default="full=1,full_pushed=1", help="Shares of the interfaces")
parser.add_argument("--segment_s", type=float, default=10.0, help="Length of a segment [s]")
parser.add_argument("--allocator", default=None, choices=["gain", "failure", "progress", "headroom", "uniform"])
parser.add_argument("--adaptive", type=float, default=0.5, help="Share of the starts where episodes fail")
parser.add_argument("--within", type=float, default=0.0, help="Share of a segment's starts where episodes fail in it")
parser.add_argument("--floor", type=float, default=0.2, help="Share that the segments keep in equal parts")
parser.add_argument("--fast", type=float, default=0.3, help="Rate of the fast average of the competence")
parser.add_argument("--slow", type=float, default=0.05, help="Rate of the slow average of the competence")
parser.add_argument("--rate", type=float, default=0.2, help="Rate at which the shares follow their target")
parser.add_argument("--horizon", type=float, default=100.0, help="Updates that the rule gain looks ahead")
parser.add_argument("--feet", type=float, default=6.0, help="Weight of the rewards for the feet")
parser.add_argument("--substeps", type=int, default=2, help="Physics steps per physics step of the task")
parser.add_argument("--local", action="store_true", help="Track without the position of the robot")
parser.add_argument("--randomize", action="store_true", help="Give every world its own robot and noisy measurements")
parser.add_argument("--armature", action="store_true", help="Give the joints the inertia of the real robot's motors")
parser.add_argument("--num_envs", type=int, default=9728)
parser.add_argument("--segment_envs", type=int, default=1536)
parser.add_argument("--segment_pushed", type=float, default=0.0, help="Share of these worlds that is pushed")
parser.add_argument("--test_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=float, default=1000.0, help="Samples to train for, in millions")
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--init_std", type=float, default=1.0)
parser.add_argument("--entropy", type=float, default=0.005)
parser.add_argument("--hidden", type=int, nargs="+", default=[1024, 512, 256])
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--out", default=None, help="Path of a JSON file with the log")
parser.add_argument("--save", default=None, help="Path to save the model to")
parser.add_argument("--save_every", type=float, default=0.0, help="Samples between checkpoints, in millions")
args = parser.parse_args()

TASK, DEVICE = args.task, "cuda:0"
RESET = "events.reset_to_motion"
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True

cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, args.num_envs, args.seed
cfg.sim.dt, cfg.decimation = cfg.sim.dt / args.substeps, cfg.decimation * args.substeps
cfg.sim.render_interval = cfg.decimation
if args.armature:
    cfg.scene.robot.actuators["body"].armature = ARMATURE
train_clips, test_clips, val_clips = CLIP_SETS[args.clips], TEST_CLIPS, VAL_CLIPS
if args.library:
    with open(args.library) as file:
        library = json.load(file)
    train_clips, val_clips, unseen_clips = (
        tuple(entry["file"] for entry in library if entry["split"] == split) for split in ("train", "val", "unseen")
    )
    test_clips = (*val_clips, *unseen_clips)
# With an allocator every start is drawn by the weights that this script writes, which hold the floor.
cfg.task = g1_interfaces_task(
    train=train_clips,
    test=test_clips,
    pushes=cfg.pushes,
    feet_weight=args.feet,
    adaptive=1.0 if args.allocator else args.adaptive,
    count_failures=args.allocator is None or args.within > 0.0,
    learner_weights=args.allocator is not None and args.within > 0.0,
    local=args.local,
    randomize=args.randomize,
)
NAMES = arm_names(cfg.pushes)
env = gym.make(TASK, cfg=cfg).unwrapped
arrays = env.state.arrays
clip_range = arrays[mdp2.term_state(RESET, "clip_range")].torch.long()
fixed_start = arrays[mdp2.term_state(RESET, "fixed_start")].torch
failures = arrays[mdp2.term_state(RESET, "failures")].torch
weights = arrays[mdp2.term_state(RESET, "weights")].torch if args.within > 0.0 else failures
world_start = arrays[mdp2.term_state(RESET, "start")].torch
body_pos_error = arrays[mdp2.quantities.body_pos_error].torch
values = dict(item.split("=") for item in args.shares.split(","))
interface_shares = torch.tensor([float(values.get(name, 0.0)) for name in NAMES], device=DEVICE)
arrays[mdp2.term_state(INTERFACE, "shares")].torch.copy_(interface_shares / interface_shares.sum())

# The segments of the training clips: the first frame, the clip, and the frames that an episode from it can track.
N, SE, TE = args.num_envs, args.segment_envs, args.test_envs
T = N - SE - TE
episode_frames = round(10.0 / env.step_dt)
segment_frames = round(args.segment_s / env.step_dt)
first, clip, target = [], [], []
for index, (begin, end) in enumerate(clip_range[: len(train_clips)].tolist()):
    for k in range(max(1, (end - begin) // segment_frames)):
        first.append(begin + k * segment_frames)
        clip.append(index)
        target.append(min(episode_frames, end - 2 - first[-1]))
first, clip = torch.tensor(first, device=DEVICE), torch.tensor(clip, device=DEVICE)
target = torch.tensor(target, dtype=torch.float32, device=DEVICE)
S = len(first)
hard = (clip >= len(TRAIN_CLIPS)) & (args.library is None)
train_frames = clip_range[len(train_clips) - 1, 1].item()


def segment_of(frames):
    """Return the segment of every frame of the training clips."""
    return (torch.searchsorted(first, frames.long().clamp(max=train_frames - 1), right=True) - 1).clamp(min=0)


# The bins of the task's start weights, each in the segment of its first frame; the bins of held-out clips have none.
bin_frames = cfg.task.events["reset_to_motion"].params["bin_frames"]
bin_first = torch.arange(len(weights), device=DEVICE) * bin_frames
bin_segment = segment_of(bin_first)
bin_train = bin_first < train_frames
bins_per_segment = torch.zeros(S, device=DEVICE).index_add_(0, bin_segment, bin_train.float()).clamp(min=1.0)


def write_shares(shares):
    """Write the share of every segment to the task, divided over the bins of the segment.

    A share ``--within`` of it follows the failures that the task counts in the bins of the segment.
    """
    spread = (1.0 / bins_per_segment)[bin_segment]
    if args.within > 0.0:
        counts = torch.where(bin_train, failures, 0.0)
        total = torch.zeros(S, device=DEVICE).index_add_(0, bin_segment, counts)[bin_segment]
        spread = torch.where(total > 0.0, (1.0 - args.within) * spread + args.within * counts / total, spread)
    weights.copy_(torch.where(bin_train, shares[bin_segment] * spread, 0.0))


# The evaluation worlds of the segments run one segment and have the next one queued as their start frame.
segment_worlds, test_worlds = slice(T, T + SE), slice(T + SE, N)
current = torch.arange(SE, device=DEVICE) % S
fixed_start[segment_worlds] = first[current].float()
# The evaluation worlds of the held-out clips, on start frames spread evenly over them.
test_ranges = clip_range[len(train_clips) :].tolist()
frames = torch.cat(
    [torch.arange(begin, max(begin + 1, end - episode_frames), device=DEVICE) for begin, end in test_ranges]
)
starts = frames[torch.linspace(0, len(frames) - 1, TE, device=DEVICE).long()]
fixed_start[test_worlds] = starts.float()
val_ranges = [test_ranges[test_clips.index(name)] for name in val_clips]
val = sum((starts >= begin) & (starts < end) for begin, end in val_ranges).bool()
arrays[mdp2.term_state(INTERFACE, "fixed")].torch[T:] = NAMES.index("full")
# The pushed evaluation worlds of the segments, which take the segments in turn as the others do.
pushed = round(args.segment_pushed * SE)
arrays[mdp2.term_state(INTERFACE, "fixed")].torch[T + SE - pushed : T + SE] = NAMES.index("full_pushed")

allocator = None
if args.allocator:
    allocator = Allocator(
        [[] for _ in range(S)],
        (),
        args.allocator,
        args.fast,
        args.slow,
        floor=args.floor,
        anchor_share=0.0,
        rate=args.rate,
        horizon=args.horizon,
    )
    segment_shares = torch.tensor(allocator.shares, dtype=torch.float32, device=DEVICE)
    write_shares(segment_shares)

obs = env.reset()[0]
queued = (SE + torch.arange(SE, device=DEVICE)) % S
cursor = torch.tensor(2 * SE % S, device=DEVICE)
fixed_start[segment_worlds] = first[queued].float()
obs_dim, critic_dim, action_dim = obs["policy"].shape[1], obs["critic"].shape[1], env.mdp.action_dim
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
)
print(f"DEVICE {torch.cuda.get_device_name(0)}")
print(f"SETUP obs {obs_dim} critic {critic_dim} train worlds {T} segments {S} hard {int(hard.sum())}", flush=True)

iterations = round(args.budget * 1.0e6 / (args.steps * T))
save_every = round(args.save_every * 1.0e6 / (args.steps * T))
alive = torch.zeros(N, device=DEVICE)
# Per segment, of the episodes of its evaluation worlds since the last log row: their number, the summed share of
# the frames that they tracked, and how many tracked to the end. Then the last known values of both.
measured = torch.zeros(3, S, device=DEVICE)
competence, success = torch.zeros(S, device=DEVICE), torch.zeros(S, device=DEVICE)
# Of the evaluation worlds on the held-out clips: episodes, episodes tracked to the end, summed distance and steps.
episodes, completed = torch.zeros(TE, device=DEVICE), torch.zeros(TE, device=DEVICE)
distance, counted = torch.zeros(TE, device=DEVICE), 0
groups = {"curated": ~hard, "hard": hard} if hard.any() else {"curated": ~hard}
history = []
wp.synchronize()
begin_time = time.perf_counter()
for k in range(1, iterations + 1):
    if allocator is not None and args.within > 0.0:
        # The failures move with every step, so the starts inside the segments follow them at every iteration.
        write_shares(segment_shares)
    for step in range(args.steps):
        with torch.no_grad():
            action = ppo.act(obs["policy"][:T], obs["critic"][:T])
            action = torch.cat([action, ppo.policy(obs["policy"][T:])])
        obs, reward, terminated, truncated, _ = env.step(action)
        # A world whose physics diverges ends its episode on numbers that are not finite: the learner gets zeros.
        obs = {name: value.nan_to_num(0.0, 0.0, 0.0) for name, value in obs.items()}
        reward = reward.nan_to_num(0.0, 0.0, 0.0)
        ppo.record(reward[:T], terminated[:T], truncated[:T])
        done = terminated | truncated
        alive += 1.0
        # The segment worlds that ended: credit their segment, and queue the next segments in turn.
        ended = done[segment_worlds]
        tracked = (alive[segment_worlds] / target[current]).clamp(max=1.0) * ended
        measured.index_add_(1, current, torch.stack([ended.float(), tracked, truncated[segment_worlds].float()]))
        current = torch.where(ended, queued, current)
        queued = torch.where(ended, (cursor + ended.cumsum(0) - 1) % S, queued)
        cursor = (cursor + ended.sum()) % S
        fixed_start[segment_worlds] = first[queued].float()
        episodes += done[test_worlds]
        completed += truncated[test_worlds]
        distance += body_pos_error[test_worlds].sqrt()
        counted += 1
        alive *= ~done
    ppo.update(obs["policy"][:T], obs["critic"][:T])

    if k % args.log_interval == 0:
        torch.cuda.synchronize()
        row = {"iteration": k, "samples_m": k * args.steps * T / 1.0e6, "time": time.perf_counter() - begin_time}
        row["reward"] = ppo.buffer["rewards"].mean().item()
        # A segment without an episode since the last row keeps what it had.
        seen = measured[0] > 0.0
        competence = torch.where(seen, measured[1] / measured[0].clamp(min=1.0), competence)
        success = torch.where(seen, measured[2] / measured[0].clamp(min=1.0), success)
        # Where the training worlds are now, by the frame on which their episode started.
        worlds = torch.zeros(S, device=DEVICE).index_add_(0, segment_of(world_start[:T]), torch.ones(T, device=DEVICE))
        for name, members in groups.items():
            row[f"{name}/success"] = success[members].mean().item()
            row[f"{name}/competence"] = competence[members].mean().item()
            row[f"{name}/worlds"] = (worlds[members].sum() / T).item()
        succeeded = completed / episodes.clamp(min=1.0)
        for name, members in (("test", torch.ones_like(val)), ("val", val), ("unseen", ~val)):
            row[f"{name}/success"] = succeeded[members].mean().item()
        row["test/error_mm"] = 1.0e3 * (distance.sum() / (counted * TE)).item()
        row["measured"] = seen.float().mean().item()
        if (len(history) + 1) % args.segment_log == 0:
            row["segments/competence"] = [round(value, 3) for value in competence.tolist()]
            row["segments/worlds"] = [round(value) for value in worlds.tolist()]
        measured.zero_()
        for value in (episodes, completed, distance):
            value.zero_()
        counted = 0
        if allocator is not None:
            shares = allocator.update(competence.cpu().numpy(), measured=seen.cpu().numpy())
            segment_shares = torch.tensor(shares, dtype=torch.float32, device=DEVICE)
            write_shares(segment_shares)
        history.append(row)
        summary = " ".join(
            f"{name} {row[f'{name}/success']:.2f} ({row[f'{name}/competence']:.2f}, {row[f'{name}/worlds']:.2f})"
            for name in groups
        )
        print(
            f"LOG {row['samples_m']:.0f}M {row['time']:.0f}s reward {row['reward']:.3f} {summary}"
            f" val {row['val/success']:.2f} unseen {row['unseen/success']:.2f} {row['test/error_mm']:.0f}mm",
            flush=True,
        )
        if args.out:
            segments = {"first": first.tolist(), "clip": clip.tolist()}
            segments["clips"] = [name.split("/")[-1] for name in train_clips]
            with open(args.out, "w") as file:
                json.dump({"args": vars(args), "segments": segments, "history": history}, file)
    if args.save and save_every and k % save_every == 0:
        state = {"agent": ppo.agent.state_dict(), "first": ppo.first, "second": ppo.second}
        state |= {"count": ppo.count, "lr": ppo.lr, "samples_m": k * args.steps * T / 1.0e6}
        torch.save(state, f"{args.save.removesuffix('.pt')}_{round(k * args.steps * T / 1.0e6)}M.pt")
train_time = time.perf_counter() - begin_time
samples = iterations * args.steps * T
print(f"TRAIN {train_time:.1f} s, {samples / train_time / 1e6:.2f} M steps/s, graphs {env.num_graphs}")
if args.save:
    torch.save(ppo.agent.state_dict(), args.save)
env.close()
