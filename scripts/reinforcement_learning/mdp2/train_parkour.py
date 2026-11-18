"""Train the G1 to climb boxes with worlds that track a clip and worlds that reach a goal, in one simulation.

A task is a clip on boxes of one height and a goal on it: ``up`` is to stand on the box from the first frame of the
clip, ``down`` to stand where the clip ends from its last frame on the box. A reference world starts on a frame of a
clip and tracks it. A goal world starts on a frame before the goal of a task and is rewarded once, at the end of its
episode, if the robot stands at the goal. ``--eval_envs`` worlds evaluate: each keeps one task and one start, acts
without exploration noise, and gives no data to the learner. Their starts are the rungs of a ladder: the first frame
of the task, and frames a quarter, a half, and three quarters of the way to the goal.

``--share`` is the share of the training worlds that track a clip. ``--schedule`` changes it while training.
"""

import argparse
import json
import os
import re
import time

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from hybrid_ppo import HybridPPO
from isaaclab_experimental import mdp2
from ppo import Agent

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.parkour.config.g1.g1_env_cfg import (
    CLIPS,
    EFFORT,
    MOTIONS_DIR,
    STIFFNESS,
    g1_task,
    g1_terrain,
)
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--clips", default="26", help="Numbers of the clips, as 0,6,26, or all")
parser.add_argument("--levels", default="0.8,0.9,1.0,1.1,1.2")
parser.add_argument("--goals", default="up", help="Goals of the tasks: up, down, or up,down")
parser.add_argument("--num_envs", type=int, default=4096, help="Training worlds")
parser.add_argument("--eval_envs", type=int, default=320)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--iterations", type=int, default=4000)
parser.add_argument("--share", type=float, default=0.5, help="Share of the training worlds that track a clip")
parser.add_argument("--schedule", default="fixed", choices=["fixed", "two_stage", "progress", "credit", "deficit"])
parser.add_argument("--switch", type=int, default=500, help="Iteration at which two_stage starts the goal worlds")
parser.add_argument("--interval", type=int, default=100, help="Iterations between changes of the share")
parser.add_argument("--delta", type=float, default=0.15, help="Distance of the shares of the two trial learners")
parser.add_argument("--move", type=float, default=0.1, help="Change of the share after a trial")
parser.add_argument("--margin", type=float, default=0.01, help="Lead in success that decides a trial")
parser.add_argument("--floor", type=float, default=0.05, help="Smallest share of the deficit schedule")
parser.add_argument("--allocation", default="uniform", choices=["uniform", "deficit"], help="Worlds of the tasks")
parser.add_argument("--smooth", type=float, default=0.5, help="Weight of the last trial in the success of a task")
parser.add_argument("--gate", action="store_true", help="Give a kind of world to a task only when it can teach it")
parser.add_argument("--starts", default="uniform", choices=["uniform", "loss"], help="Frames that worlds start on")
parser.add_argument("--teacher_log", default=None, help="Path of the log of the teacher, for its success per task")
parser.add_argument("--holdout_levels", default="", help="Levels that no training world gets, as 0.9,1.1")
parser.add_argument("--holdout_clips", default="", help="Numbers of the clips that no training world gets")
parser.add_argument("--ladder", default="0,1,2,3,4,4,4,4", help="Rungs of the evaluation worlds of a task")
parser.add_argument("--away", type=float, default=0.5, help="Share of the goal worlds that start away from the clip")
parser.add_argument("--offset", type=float, default=0.5, help="Largest distance of such a start along x and y [m]")
parser.add_argument("--turn", type=float, default=0.5, help="Largest turn of such a start [rad]")
parser.add_argument("--lost", type=float, default=0.3, help="RMS body distance that ends a tracking episode [m]")
parser.add_argument("--lookahead", type=float, default=1.0e6, help="Steps that a tracking goal runs ahead")
parser.add_argument("--extra", type=float, default=2.0, help="Time of a goal episode beyond that of the clip [s]")
parser.add_argument("--radius", type=float, default=0.3, help="Distance to the goal that counts as reaching it [m]")
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument("--kl", type=float, default=0.15)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--clip", type=float, default=0.3)
parser.add_argument("--value_coef", type=float, default=4.0)
parser.add_argument("--max_lr", type=float, default=1.0e-2)
parser.add_argument("--lam", type=float, default=0.97)
parser.add_argument("--action_scale", type=float, default=0.5, help="Scale of the policy's actions [rad]")
parser.add_argument("--failed", type=float, default=1.0, help="Factor on the penalty of a failed episode")
parser.add_argument("--ema", type=float, default=1.0, help="Weight of the new action in the low-pass of the actions")
parser.add_argument("--damping", type=float, default=0.04, help="Damping of the stiff gains over their stiffness")
parser.add_argument("--gains", default="soft", choices=["soft", "stiff"], help="Gains of the joints")
parser.add_argument(
    "--residual_rad",
    type=float,
    default=0.0,
    help="The same scale of a tracker's corrections for all joints [rad], in place of the factor",
)
parser.add_argument("--residual", type=float, default=0.5, help="Factor on the scale of a tracker's corrections")
parser.add_argument("--action_rate", type=float, default=1.0, help="Factor on the cost of the rate of the actions")
parser.add_argument("--power", type=float, default=1.0, help="Factor on the cost of the power of the joints")
parser.add_argument("--tracker", action="store_true", help="Train a tracker: the policy observes the clip")
parser.add_argument("--teacher", default=None, help="Path of a tracker whose actions the tracking worlds imitate")
parser.add_argument("--imitation", type=float, default=1.0, help="Weight of the distance to the teacher's actions")
parser.add_argument("--kind_norm", action="store_true", help="Normalize the advantages of each kind of world")
parser.add_argument("--upright", type=float, default=1.0, help="Weight of the upright posture in goal worlds")
parser.add_argument("--body_scale", type=float, default=100.0, help="Sharpness of the body position reward [1/m^2]")
parser.add_argument("--init_std", type=float, default=0.3)
parser.add_argument("--entropy", type=float, default=0.0)
parser.add_argument("--gamma_track", type=float, default=0.97)
parser.add_argument("--gamma_goal", type=float, default=0.997)
parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
parser.add_argument("--log_interval", type=int, default=200)
parser.add_argument("--out", default=None, help="Path of a JSON file with the log")
parser.add_argument("--save", default=None, help="Path of a file for the final policy")
parser.add_argument("--load", default=None, help="Path of a policy to start from")
parser.add_argument("--frozen", action="store_true", help="Evaluate the loaded policy without training it")
parser.add_argument("--video", default=None, help="Path of an MP4 of the final policy in the evaluation worlds")
parser.add_argument("--video_steps", type=int, default=750)
parser.add_argument("--drive", action="store_true", help="Trace the teacher's actions through the policy's action path")
parser.add_argument("--playback", action="store_true", help="Show the clips themselves in the video, without physics")
parser.add_argument(
    "--trace", default=None, help="Path of an NPZ of the joints of the evaluation worlds under the final policy"
)
args = parser.parse_args()

TASK, DEVICE = "Isaac-Parkour-G1-MDP2", "cuda:0"
START = "events.start_episode"
T, E = args.num_envs, args.eval_envs
N = T + E
RUNGS = 5
TAIL = 0.667  # [s]
# With the credit schedule, two learners with a share of the worlds each try shares on either side of the current one.
A = 2 if args.schedule == "credit" else 1
TA, EA = T // A, E // A
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True

clips = CLIPS if args.clips == "all" else tuple(f"climb_{int(clip):02d}" for clip in args.clips.split(","))
levels = tuple(args.levels.split(","))
cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, N, args.seed
cfg.scene.terrain.terrain_generator = g1_terrain(clips, levels)
cfg.task = g1_task(clips, levels, reference_input=args.tracker)
cfg.task.rewards["body_pos"].params["scale"] = args.body_scale
# The stiffer gains of the flat tracking task, to compare with those derived from the inertia of the motors.
STIFF = {".*_hip_.*": 100.0, ".*_knee_joint": 150.0, ".*_ankle_.*": 40.0, "waist_.*": 150.0, ".*_shoulder_.*": 40.0,
         ".*_elbow_joint": 40.0, ".*_wrist_.*": 20.0, ".*_hand_.*": 20.0}  # fmt: skip
if args.gains == "stiff":
    cfg.scene.robot.actuators["body"].stiffness = STIFF
    cfg.scene.robot.actuators["body"].damping = {joint: args.damping * gain for joint, gain in STIFF.items()}
if args.video:
    from video import add_camera, record

    import isaaclab.sim as sim_utils

    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.6, 0.68))
    add_camera(cfg, eye=(3.0, -3.0, 1.2))
env = gym.make(TASK, cfg=cfg).unwrapped
arrays = env.state.arrays
pending = arrays[mdp2.term_state(START, "pending")].torch
episode = arrays[mdp2.term_state(START, "episode")].torch
origin = arrays[mdp2.term_state("events.terrain_origin", "origin")].torch
weights = env.state.reward_weights.torch
clip_range = arrays[mdp2.term_state(START, "clip_range")].torch.long()
root_pose = arrays[mdp2.term_state(START, "clip_root_pose")].torch
rewards = [op.name.removeprefix("rewards.") for op in env.mdp.term_ops if op.name.startswith("rewards.")]
# The rewards of a world that tracks a clip, of one that reaches a goal, and of both.
TRACKING = ("joint_pos", "joint_vel", "body_pos", "root_vel")
tracking_rewards = torch.tensor([float(name in TRACKING) for name in rewards], device=DEVICE)
GOAL = ("goal_reached", "failed", "is_alive", "upright")
goal_rewards = torch.tensor([float(name in GOAL) for name in rewards], device=DEVICE)
shared_rewards = 1.0 - tracking_rewards - goal_rewards
default_weights = weights[0].clone()
default_weights[rewards.index("upright")] *= args.upright
default_weights[rewards.index("action_rate_l2")] *= args.action_rate
default_weights[rewards.index("failed")] *= args.failed
default_weights[rewards.index("joint_power")] *= args.power


def steps(seconds):
    """Return the steps of the controller in ``seconds``."""
    return round(seconds / env.step_dt)


def zeros(*shape):
    return torch.zeros(*shape, device=DEVICE)


# The clips of a column of the terrain follow each other in the clip table, a level after the other.
origins = env.scene.terrain.terrain_origins
tile_origin = torch.stack([origins[level, clip] for clip in range(len(clips)) for level in range(len(levels))])
first, last = clip_range[:, 0], clip_range[:, 1]

# A task is a clip, the frame it starts on, the frame whose root pose is its goal, and the frame after the last that
# a world tracks. The robot stands on a box in the frames where its root is upright, slow, and as high over its
# first frame as the box, less a margin: the goal is a third of a second into the longest such stretch, and a world
# tracks the clip to two thirds of a second after the goal, or to the end of the stretch. The levels of a clip have
# the same frames, so the middle level gives those of all of them.
tasks, names = [], []
for c, clip in enumerate(clips):
    k = c * len(levels) + len(levels) // 2
    with open(os.path.join(MOTIONS_DIR, f"{clip}_z{levels[len(levels) // 2]}.json")) as file:
        top = max(box["size"][2] for box in json.load(file)["boxes"] if box["size"][0] < 3.0)
    pose = root_pose[first[k] : last[k]]
    up = 1.0 - 2.0 * (pose[:, 3] ** 2 + pose[:, 4] ** 2)
    speed = torch.gradient(pose[:, :3], spacing=env.step_dt, dim=0)[0].norm(dim=-1)
    standing = (pose[:, 2] - pose[0, 2] > 0.8 * top - 0.12) & (up > 0.94) & (speed < 0.5)
    # The stretches of consecutive frames that stand on a box, as their first frame and the frame after their last.
    edges = torch.diff(standing.int(), prepend=standing.new_zeros(1).int(), append=standing.new_zeros(1).int())
    stretches = list(zip(edges.eq(1).nonzero().flatten().tolist(), edges.eq(-1).nonzero().flatten().tolist()))
    if not stretches or max(stop - begin for begin, stop in stretches) < steps(0.2):
        continue
    begin, stop = max(stretches, key=lambda stretch: stretch[1] - stretch[0])
    goal = begin + min(steps(0.333), (stop - begin) // 2)
    for row, level in enumerate(levels):
        k = c * len(levels) + row
        end = min(goal + steps(TAIL), stop - 1, last[k].item() - first[k].item() - 2) + 1
        if "up" in args.goals:
            tasks.append((k, first[k].item(), first[k].item() + goal, first[k].item() + end))
            names.append(f"{clip} z{level} up")
        if "down" in args.goals and root_pose[last[k] - 1, 2] - root_pose[first[k], 2] < 0.2:
            tasks.append((k, first[k].item() + stretches[-1][1] - 1, last[k].item() - 1, last[k].item()))
            names.append(f"{clip} z{level} down")
tasks = torch.tensor(tasks, device=DEVICE)
TASKS = len(tasks)
# The tasks that the training worlds get, and the share of them that each has: of the goal worlds, then of the
# tracking worlds.
held_levels = [f" z{level} " for level in args.holdout_levels.split(",") if level]
held_clips = [f"climb_{int(clip):02d} " for clip in args.holdout_clips.split(",") if clip]
seen = torch.tensor([not any(part in name for part in held_levels + held_clips) for name in names], device=DEVICE)
task_shares = (seen.float() / seen.sum()).repeat(2, 1)
# The share of the worlds of a task that start on each stretch of the way to its goal, between two rungs.
stretch_shares = torch.full((TASKS, RUNGS - 1), 1.0 / (RUNGS - 1), device=DEVICE)
# How often the teacher stands at the goal of every task from its first frame.
teachable = torch.ones(TASKS, device=DEVICE)
if args.teacher_log:
    with open(args.teacher_log) as file:
        teacher_log = json.load(file)
    by_task = {name: rungs[0] for name, rungs in zip(teacher_log["tasks"], teacher_log["history"][-1]["eval/tasks"])}
    teachable = torch.tensor([by_task[name] for name in names], device=DEVICE)

# Of every world: the kind of its running episode and of its next one. Kind 1 tracks a clip, kind 0 reaches a goal.
# The training worlds come first, a block per learner, then the evaluation worlds, a block per learner.
meta = {name: torch.zeros(N, dtype=torch.long, device=DEVICE) for name in ("kind", "next_kind")}
arm = torch.cat([torch.arange(T, device=DEVICE) // TA, torch.arange(E, device=DEVICE) // EA])
train = [slice(a * TA, (a + 1) * TA) for a in range(A)]
evals = [slice(T + a * EA, T + (a + 1) * EA) for a in range(A)]
# Every learner evaluates the same tasks from the same rungs.
ladder = torch.tensor([int(value) for value in args.ladder.split(",")], device=DEVICE)
slot = torch.arange(E, device=DEVICE) % EA
rung, eval_task = ladder[slot % len(ladder)], (slot // len(ladder)) % TASKS
base = {"credit": 1.0 - args.delta, "fixed": args.share, "deficit": 1.0 if args.gate else 0.5}.get(args.schedule, 1.0)
shares = torch.full((A,), base, device=DEVICE)
if args.schedule == "credit":
    shares[0], shares[1] = base + args.delta, base - args.delta
gamma = zeros(T)
offset_range = torch.tensor([args.offset, args.offset, args.turn], device=DEVICE)


def write_episodes(task, frame, tracks, offset):
    """Return the episodes of worlds that start on ``frame`` of ``task``, tracking the clip where ``tracks`` is one."""
    goal = task[:, 2]
    values = zeros(len(frame), 11)
    # The clip of a task ends a little after its goal: a world that tracks it then holds its last frame.
    values[:, 0], values[:, 1], values[:, 2] = frame, task[:, 3], goal
    values[:, 3] = tracks * args.lookahead + (1.0 - tracks) * 1.0e6
    values[:, 4] = tracks * args.lost**2 + (1.0 - tracks) * 1.0e9
    values[:, 5] = (1.0 - tracks) * args.radius
    # A goal world has the time that the clip takes to its goal and some, and a tracking world as much: it holds the
    # last frame of its clip for as long as a goal world may stand at the goal.
    values[:, 6] = tracks * (goal - frame + steps(args.extra)) + (1.0 - tracks) * 1.0e9
    values[:, 7] = tracks * 1.0e9 + (1.0 - tracks) * (goal - frame + steps(args.extra))
    values[:, 8:11] = offset
    return values


def draw():
    """Write the next episode of every world.

    A training world tracks a clip with the share of its learner, and starts on a frame of its task before the goal.
    A goal world may start on the first frame instead, away from where the clip starts. The evaluation worlds of a
    task start on the first frame of the task and on frames a quarter, a half, and three quarters of the way to the
    goal, and, on the last rung, on the first frame, away from where the clip starts.
    """
    kind = (torch.rand(T, device=DEVICE) < shares[arm[:T]]).long()
    drawn = torch.multinomial(task_shares, T, replacement=True)
    drawn = torch.where(kind == 1, drawn[1], drawn[0])
    task = tasks[drawn]
    begin, goal = task[:, 1], task[:, 2]
    way = torch.rand(T, device=DEVICE)
    if args.starts == "loss":
        # A stretch of the way to the goal, drawn by the episodes that it loses, and a frame in it.
        stretch = torch.multinomial(stretch_shares[drawn], 1).squeeze(1)
        way = (stretch + way) / (RUNGS - 1)
    frame = begin + (way * (goal - begin - steps(0.167))).long()
    away = (kind == 0) & (torch.rand(T, device=DEVICE) < args.away)
    frame = torch.where(away, begin, frame)
    offset = away[:, None] * (2.0 * torch.rand(T, 3, device=DEVICE) - 1.0) * offset_range
    pending[:T] = write_episodes(task, frame, kind.float(), offset)
    origin[:T] = tile_origin[task[:, 0]]
    meta["next_kind"][:T] = kind

    task = tasks[eval_task]
    begin, goal = task[:, 1], task[:, 2]
    frame = begin + (rung % (RUNGS - 1)) * (goal - begin - steps(0.167)) // (RUNGS - 1)
    frame = torch.where(rung == RUNGS - 1, begin, frame)
    offset = (rung == RUNGS - 1)[:, None] * (2.0 * torch.rand(E, 3, device=DEVICE) - 1.0) * offset_range
    pending[T:] = write_episodes(task, frame, zeros(E), offset)
    origin[T:] = tile_origin[task[:, 0]]


def start(worlds):
    """Give the worlds in the mask ``worlds``, which just reset, the rewards of the kind of their new episode."""
    meta["kind"].copy_(torch.where(worlds, meta["next_kind"], meta["kind"]))
    tracks = (meta["kind"] == 1).float()[:, None]
    kind_weights = default_weights * (tracks * tracking_rewards + (1.0 - tracks) * goal_rewards + shared_rewards)
    weights.copy_(torch.where(worlds[:, None], kind_weights, weights))
    gamma.copy_(tracks[:T, 0] * args.gamma_track + (1.0 - tracks[:T, 0]) * args.gamma_goal)


draw()
obs = env.reset()[0]
start(torch.ones(N, dtype=torch.bool, device=DEVICE))
learners = []
for a in range(A):
    ppo = HybridPPO(
        obs["policy"].shape[1],
        env.mdp.action_dim,
        TA,
        DEVICE,
        args.hidden,
        args.steps,
        epochs=args.epochs,
        minibatches=args.minibatches,
        lr=args.lr,
        max_lr=args.max_lr,
        kl=args.kl,
        lam=args.lam,
        clip=args.clip,
        value_coef=args.value_coef,
        entropy=args.entropy,
        init_std=args.init_std,
        critic_dim=obs["critic"].shape[1],
        kinds=meta["kind"][train[a]],
        kind_norm=args.kind_norm,
        imitation=args.teacher is not None,
    )
    ppo.gamma = gamma[train[a]]
    ppo.imitation_coef.fill_(args.imitation)
    learners.append(ppo)
if args.load:
    for ppo in learners:
        ppo.agent.load_state_dict(torch.load(args.load))
# The learners start from the same policy.
for ppo in learners[1:]:
    ppo.agent.load_state_dict(learners[0].agent.state_dict())

# A tracker observes the frame that its world tracks, which ends the critic's observations, in place of the last
# action, which ends the policy's, and corrects the joint positions of that frame, which start the frame.
ACTIONS = env.mdp.action_dim
robot = env.scene["robot"]
moved = [joint for joint, name in enumerate(robot.joint_names) if "_hand_" not in name]
default_joint_pos = robot.data.default_joint_pos.torch[0, moved]


def per_joint(table):
    """Return the value of ``table``, which maps expressions of joint names to values, for each moved joint."""
    names = [robot.joint_names[joint] for joint in moved]
    return torch.tensor([next(v for k, v in table.items() if re.fullmatch(k, name)) for name in names], device=DEVICE)


# A unit of action moves the target of a joint by a quarter of the angle at which its gain gives its largest
# torque. A tracker corrects the frame it tracks by a part of that.
UNIT = 0.25 * per_joint(EFFORT) / per_joint(STIFF if args.gains == "stiff" else STIFFNESS)
RESIDUAL = torch.full_like(UNIT, args.residual_rad) if args.residual_rad else args.residual * UNIT
SCALE = RESIDUAL if args.tracker else torch.full_like(UNIT, args.action_scale)
teacher = None
if args.teacher:
    tracker = torch.load(args.teacher)
    REFERENCE = tracker["obs.mean"].numel() - (obs["policy"].shape[1] - ACTIONS)
    hidden = [layer.shape[0] for name, layer in tracker.items() if name.startswith("actor.") and layer.dim() == 2][:-1]
    teacher = Agent(tracker["obs.mean"].numel(), ACTIONS, hidden, args.init_std, obs["critic"].shape[1])
    teacher.to(DEVICE).load_state_dict(tracker)


def teacher_actions(obs, rows=slice(0, T)):
    """Return the actions with which the policy sets the joint targets of the teacher, in the training worlds."""
    reference = obs["critic"][rows, -REFERENCE:]
    correction = teacher.actor(teacher.obs(torch.cat([obs["policy"][rows, :-ACTIONS], reference], dim=-1)))
    return (reference[:, moved] + RESIDUAL * correction - default_joint_pos) / SCALE


# Of the episodes that ended since the last log row: per learner and kind of training world, and per learner, task
# and rung of the evaluation worlds. And of the evaluation episodes that ended since the last trial, per learner.
stats = {name: zeros(A, 2) for name in ("episodes", "success", "steps", "lost")}
evaluation = {name: zeros(A, TASKS, RUNGS) for name in ("episodes", "success")}
trial = {name: zeros(A, TASKS, RUNGS) for name in ("episodes", "success")}
success_rate = zeros(TASKS, RUNGS)
length = zeros(N)
actions = zeros(N, ACTIONS)
smooth = zeros(N, ACTIONS)
fresh = torch.ones(N, 1, dtype=torch.bool, device=DEVICE)


def command(done=None):
    """Return the joint targets of the actions after a low-pass that starts anew with an episode."""
    if done is not None:
        fresh.copy_(done.unsqueeze(1))
        return None
    smooth.copy_(torch.where(fresh, actions, torch.lerp(smooth, actions, args.ema)))
    fresh.fill_(False)
    return smooth * SCALE


def rollout(obs):
    """Collect a rollout: the training worlds explore and feed their learner, the evaluation worlds do neither."""
    for _ in range(args.steps):
        with torch.no_grad():
            targets = teacher_actions(obs) if teacher else None
            tracks = (meta["kind"][:T] == 1).float()
            for a, ppo in enumerate(learners):
                imitation = (targets[train[a]], tracks[train[a]]) if teacher else (None, None)
                actions[train[a]] = ppo.act(obs["policy"][train[a]], obs["critic"][train[a]], *imitation)
                actions[evals[a]] = ppo.policy(obs["policy"][evals[a]])
        obs, reward, terminated, truncated, _ = env.step(command())
        # A world whose simulation diverged ends: keep what it held out of the learners.
        obs = {name: torch.nan_to_num(value) for name, value in obs.items()}
        reward = torch.nan_to_num(reward)
        for a, ppo in enumerate(learners):
            ppo.record(reward[train[a]], terminated[train[a]], truncated[train[a]])

        # An episode that reached its goal ends with the reward of the goal, which is 20.
        done = terminated | truncated
        length.add_(1.0)
        success = (terminated & (reward > 10.0)).float()
        index = arm[:T] * 2 + meta["kind"][:T]
        stats["episodes"].view(-1).index_add_(0, index, done[:T].float())
        stats["success"].view(-1).index_add_(0, index, success[:T])
        stats["steps"].view(-1).index_add_(0, index, done[:T] * length[:T])
        stats["lost"].view(-1).index_add_(0, index, terminated[:T].float())
        index = (arm[T:] * TASKS + eval_task) * RUNGS + rung
        evaluation["episodes"].view(-1).index_add_(0, index, done[T:].float())
        evaluation["success"].view(-1).index_add_(0, index, success[T:])
        trial["episodes"].view(-1).index_add_(0, index, done[T:].float())
        trial["success"].view(-1).index_add_(0, index, success[T:])
        length.mul_(1.0 - done.float())
        command(done)
        start(done)
        draw()
    return obs


def schedule(k):
    """Set the shares of the worlds that track a clip for the next iteration."""
    global base, progress
    if args.schedule == "two_stage":
        shares.fill_(1.0 if k < args.switch else args.share)
    if k % args.interval:
        return
    # The success of every task from every rung, smoothed over the trials.
    ended = trial["episodes"].sum(0)
    rate = trial["success"].sum(0) / ended.clamp(min=1.0)
    success_rate.copy_(torch.where(ended > 0.0, torch.lerp(success_rate, rate, args.smooth), success_rate))
    if args.starts == "loss":
        # An episode that fails from a rung where one from the next rung succeeds is lost between the two: the
        # worlds that start on the clip start there, and a few everywhere.
        on_clip = success_rate[:, : RUNGS - 1]
        ahead = torch.cat([on_clip[:, 1:], torch.ones(TASKS, 1, device=DEVICE)], dim=1)
        lost = (ahead - on_clip).clamp(min=0.0) + 0.1
        stretch_shares.copy_(lost / lost.sum(1, keepdim=True))
    if args.schedule == "progress":
        # The share falls from one to its final value as the tracking worlds stop losing their clips.
        tracked = 1.0 - (stats["lost"][:, 1].sum() / stats["episodes"][:, 1].sum().clamp(min=1.0)).item()
        progress = min(max(progress + (args.move if tracked > 0.7 else -args.move), 0.0), 1.0)
        shares.fill_(1.0 + progress * (args.share - 1.0))
    if args.schedule == "credit":
        # The learner with the larger share of tracking worlds is the first. The one whose evaluation worlds succeed
        # more often gives the other its policy, and the share moves towards its own.
        score = trial["success"].sum((1, 2)) / trial["episodes"].sum((1, 2)).clamp(min=1.0)
        lead = (score[0] - score[1]).item()
        if abs(lead) > args.margin:
            winner, loser = (0, 1) if lead > 0.0 else (1, 0)
            learners[loser].agent.load_state_dict(learners[winner].agent.state_dict())
            for name in ("first", "second", "count", "lr"):
                getattr(learners[loser], name).copy_(getattr(learners[winner], name))
            base = min(max(base + (args.move if lead > 0.0 else -args.move), args.delta), 1.0 - args.delta)
        shares[0], shares[1] = base + args.delta, base - args.delta
        trials.append({"iteration": k, "score": score.tolist(), "base": base})
    if args.schedule == "deficit" or args.allocation == "deficit":
        # Tracking worlds are for what the policy cannot do from the frames of the clip, goal worlds for what it
        # cannot do away from them: each kind gets worlds by the share of the evaluation worlds that still fail.
        on_clip, away = 1.0 - success_rate[:, : RUNGS - 1].mean(1), 1.0 - success_rate[:, RUNGS - 1]
        # What the goal worlds and the tracking worlds of a task are wanted for: what fails away from the clip, and
        # what fails on it. With the gate, a kind of world counts only where it can teach: tracking worlds where
        # the teacher succeeds, and goal worlds where the policy already succeeds from the frames of the clip. A
        # tracker has no teacher: its worlds count where it already succeeds in part.
        wanted = torch.stack([away, on_clip])
        if args.gate:
            wanted = torch.stack([(1.1 - on_clip) * away, (1.1 - on_clip if args.tracker else teachable) * on_clip])
        if args.allocation == "deficit":
            # A task gets the worlds of a kind by its share of what they are wanted for, and a few besides.
            floor = wanted + 0.05 * wanted.max(1, keepdim=True).values.clamp(min=1.0e-3)
            task_shares.copy_(seen * floor / (seen * floor).sum(1, keepdim=True))
    if args.schedule == "deficit":
        goal_want, track_want = (seen * wanted[0]).sum().item(), (seen * wanted[1]).sum().item()
        base = max(track_want / (track_want + goal_want + 1.0e-6), args.floor)
        shares.fill_(base)
        count = seen.sum().item()
        trials.append(
            {
                "iteration": k,
                "on_clip": (seen * on_clip).sum().item() / count,
                "away": (seen * away).sum().item() / count,
                "base": base,
            }
        )
    for value in trial.values():
        value.zero_()


print(
    f"SETUP obs {obs['policy'].shape[1]} critic {obs['critic'].shape[1]} training worlds {T} evaluation worlds {E}"
    f" clips {len(clips)}x{len(levels)} tasks {TASKS} learners {A}",
    flush=True,
)
history, trials, progress = [], [], 0.0
wp.synchronize()
begin = time.perf_counter()
for k in range(1, args.iterations + 1):
    obs = rollout(obs)
    for a, ppo in enumerate(learners):
        if args.frozen:
            ppo.t = 0
            continue
        ppo.update(obs["policy"][train[a]], obs["critic"][train[a]])

    if k % args.log_interval == 0 or k == args.iterations:
        wp.synchronize()
        episodes = stats["episodes"].sum(0).clamp(min=1.0)
        # The evaluation of a run is that of the learner whose evaluation worlds succeed most often.
        rate = evaluation["success"].sum((1, 2)) / evaluation["episodes"].sum((1, 2)).clamp(min=1.0)
        leader = int(rate.argmax())
        by_rung = evaluation["success"][leader].sum(0) / evaluation["episodes"][leader].sum(0).clamp(min=1.0)
        unseen = evaluation["success"][leader][~seen].sum(0) / evaluation["episodes"][leader][~seen].sum(0).clamp(
            min=1.0
        )
        row = {
            "iteration": k,
            "samples_m": k * args.steps * T / 1.0e6,
            "time": time.perf_counter() - begin,
            "shares": shares.tolist(),
            "share": shares.mean().item(),
            "goal/success": (stats["success"][:, 0].sum() / episodes[0]).item(),
            "goal/steps": (stats["steps"][:, 0].sum() / episodes[0]).item(),
            "track/steps": (stats["steps"][:, 1].sum() / episodes[1]).item(),
            "track/lost": (stats["lost"][:, 1].sum() / episodes[1]).item(),
            "eval/rungs": by_rung.tolist(),
            "eval/tasks": (evaluation["success"][leader] / evaluation["episodes"][leader].clamp(min=1.0)).tolist(),
            "eval/learners": rate.tolist(),
            "eval/unseen": unseen.tolist(),
            "task_shares": task_shares.tolist() if args.allocation == "deficit" else None,
            "std": learners[leader].agent.log_std.exp().mean().item(),
        }
        history.append(row)
        rungs = " ".join(f"{value:.2f}" for value in row["eval/rungs"])
        print(
            f"LOG {k} {row['samples_m']:.0f}M {row['time']:.0f}s share {row['share']:.2f} track steps"
            f" {row['track/steps']:.0f} lost {row['track/lost']:.2f} goal steps {row['goal/steps']:.0f} success"
            f" {row['goal/success']:.2f} eval by rung {rungs} std {row['std']:.2f}",
            flush=True,
        )
        if args.out:
            with open(args.out, "w") as file:
                json.dump(
                    {"args": vars(args), "tasks": names, "seen": seen.tolist(), "history": history, "trials": trials},
                    file,
                )
        if args.save:
            torch.save(learners[0].agent.state_dict(), args.save)
    # The schedule reads the statistics of the tracking worlds before a log row clears them.
    schedule(k)
    if k % args.log_interval == 0:
        for value in (*stats.values(), *evaluation.values()):
            value.zero_()
print(f"TRAIN {time.perf_counter() - begin:.1f} s, graphs {env.num_graphs}")
if args.save:
    torch.save(learners[0].agent.state_dict(), args.save)
if args.video:
    # The camera of a world looks at the middle of its tile from a fixed place: the evaluation worlds only.
    def act(policy_obs):
        actions[:T] = 0.0
        actions[T:] = learners[0].policy(policy_obs[T:])
        if args.playback:
            # Put the robot on the frames of its clip, one per step and then the last in place of the simulation.
            task = tasks[eval_task]
            frame = (task[:, 1] + shown[0]).clamp(max=task[:, 3] - 1)
            shown[0] += 1
            pose = root_pose[frame].clone()
            pose[:, :3] += origin[T:]
            worlds = torch.arange(T, N, device=DEVICE)
            robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=worlds)
            robot.write_root_velocity_to_sim_index(root_velocity=zeros(E, 6), env_ids=worlds)
            joints = clip_joints[frame]
            robot.write_joint_state_to_sim_index(position=joints, velocity=torch.zeros_like(joints), env_ids=worlds)
            actions[T:] = 0.0
        return command()

    # The training worlds are not shown: they wait far from the tiles.
    origin[:T, 0] += 1000.0
    shown = [0]
    clip_joints = arrays[mdp2.term_state(START, "clip_joint_pos")].torch

    look_at = lambda: origin + torch.tensor([0.0, 0.0, 0.9], device=DEVICE)  # noqa: E731
    record(env, act, args.video, args.video_steps, int(np.ceil(np.sqrt(N))), eye=(2.8, -3.4, 1.5), roots=look_at)
if args.trace:
    # The joints of the evaluation worlds at every step, with the clip of their task, to measure how they move.
    data = robot.data
    rows = {name: [] for name in ("joint_pos", "joint_vel", "target", "root", "done")}
    obs = env.reset()[0]
    with torch.no_grad():
        for _ in range(args.video_steps):
            actions[:T] = 0.0
            actions[T:] = learners[0].policy(obs["policy"][T:])
            if args.drive:
                actions[T:] = teacher_actions(obs, slice(T, N))
            obs, _, terminated, truncated, _ = env.step(command())
            command(terminated | truncated)
            rows["joint_pos"].append(data.joint_pos.torch[T:, moved].clone())
            rows["joint_vel"].append(data.joint_vel.torch[T:, moved].clone())
            rows["target"].append(data.joint_pos_target.torch[T:, moved].clone())
            rows["root"].append(data.root_link_pose_w.torch[T:, :7].clone())
            rows["done"].append((terminated | truncated)[T:].clone())
    table = arrays[mdp2.term_state(START, "clip_joint_pos")].torch
    np.savez(
        args.trace,
        **{name: torch.stack(values).cpu().numpy() for name, values in rows.items()},
        clip_joint_pos=table.cpu().numpy(),
        tasks=tasks.cpu().numpy(),
        rung=rung.cpu().numpy(),
        eval_task=eval_task.cpu().numpy(),
        moved=np.array(moved),
        step_dt=env.step_dt,
    )
env.close()
