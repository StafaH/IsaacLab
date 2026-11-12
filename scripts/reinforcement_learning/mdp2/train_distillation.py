# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a teacher and a student of an MDP2 distillation task, together in one simulation or one after the other.

    uv run python scripts/reinforcement_learning/mdp2/train_distillation.py --mode together --budget 60

``together``: the teacher learns with PPO in the teacher worlds, and in the same steps the student acts in the
student worlds and learns the actions that the teacher would take there, from the privileged observations of those
worlds. ``teacher`` and ``student`` are the two stages of the usual pipeline, each with every world: the teacher is
saved to ``--teacher``, which the student stage loads.
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

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Distillation-Flat-UnitreeGo2-MDP2")
parser.add_argument("--mode", default="together", choices=["together", "teacher", "student"])
parser.add_argument(
    "--teacher", default=None, help="File of the teacher: written by the teacher stage, read by the student's"
)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=float, default=60.0, help="Samples to train for, in millions, over all worlds")
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--student_lr", type=float, default=1.0e-3)
parser.add_argument("--student_epochs", type=int, default=2)
parser.add_argument("--student_hidden", type=int, nargs="+", default=[256, 256, 128])
parser.add_argument("--log_interval", type=int, default=20)
parser.add_argument("--out", default=None)
parser.add_argument("--video", default=None, help="Path of an MP4: two teacher worlds, then two student worlds")
args = parser.parse_args()

DEVICE = "cuda:0"
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True


def make_env(num_envs, mode, camera=False):
    cfg = resolve_presets(load_cfg_from_registry(args.task, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, num_envs, args.seed
    cfg.task = {"together": cfg.task, "teacher": cfg.task[0], "student": cfg.task[1]}[mode]
    if camera:
        add_camera(cfg)
    return gym.make(args.task, cfg=cfg).unwrapped


env = make_env(args.num_envs, args.mode)
obs = env.reset()[0]
num_envs, action_dim = env.num_envs, env.mdp.action_dim
dims = env.mdp.observation_dims
# The worlds that the teacher acts in, and those that the student acts in.
split = env.mdp.world_configs[0].worlds.stop if args.mode == "together" else (num_envs if args.mode == "teacher" else 0)
teaching, studying = slice(0, split), slice(split, num_envs)
T = args.steps
iterations = round(args.budget * 1.0e6 / (T * num_envs))

teacher = PPO(dims["privileged"], action_dim, max(split, 1), DEVICE, hidden=(128, 128, 128), steps=T, entropy=0.01)
if args.mode == "student":
    teacher.agent.load_state_dict(torch.load(args.teacher))
student, student_optimizer, student_obs = None, None, None
if args.mode != "teacher":
    student = torch.nn.Sequential(
        torch.nn.BatchNorm1d(dims["student"], affine=False), mlp([dims["student"], *args.student_hidden, action_dim])
    ).to(DEVICE)
    student_optimizer = torch.optim.Adam(student.parameters(), lr=args.student_lr)
    students = num_envs - split
    student_obs = torch.zeros(T, students, dims["student"], device=DEVICE)
    student_labels = torch.zeros(T, students, action_dim, device=DEVICE)

command = env.state.arrays[mdp2.commands.base_velocity].torch
robot = env.scene["robot"].data
actions = torch.zeros(num_envs, action_dim, device=DEVICE)
falls = torch.zeros(num_envs, device=DEVICE)


def measure():
    """Return the velocity tracking error [m/s] and the falls per second of the teacher's and the student's worlds."""
    # The root state aliases the solver's; derived data such as the body-frame velocity is only refreshed on demand.
    velocity = math_utils.quat_apply_inverse(robot.root_link_pose_w.torch[:, 3:7], robot.root_com_vel_w.torch[:, :3])
    error = (command[:, :2] - velocity[:, :2]).norm(dim=-1)
    result = {}
    for name, worlds in (("teacher", teaching), ("student", studying)):
        if error[worlds].numel():
            result[f"{name}_velocity_error"] = error[worlds].mean().item()
            result[f"{name}_falls_per_s"] = falls[worlds].mean().item() / (T * env.step_dt)
    return result


def rollout():
    global obs
    falls.zero_()
    for t in range(T):
        with torch.no_grad():
            if split:
                actions[teaching] = teacher.act(obs["privileged"][teaching]) if args.mode != "student" else 0.0
            if student is not None:
                # The student acts, and the teacher says what it would have done from what the student cannot see.
                student_obs[t] = obs["student"][studying]
                student_labels[t] = teacher.policy(obs["privileged"][studying])
                student.eval()
                actions[studying] = student(student_obs[t])
        obs, reward, terminated, truncated, _ = env.step(actions)
        falls.add_(terminated)
        if split and args.mode != "student":
            teacher.record(reward[teaching], terminated[teaching], truncated[teaching])


def update_student():
    """Fit the student's actions to the teacher's on the states that the student visited."""
    student.train()
    inputs, labels = student_obs.flatten(0, 1), student_labels.flatten(0, 1)
    for _ in range(args.student_epochs):
        for indices in torch.randperm(len(inputs), device=DEVICE).chunk(4):
            loss = (student(inputs[indices]) - labels[indices]).square().mean()
            student_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            student_optimizer.step()


history = []
wp.synchronize()
start = time.perf_counter()
for k in range(1, iterations + 1):
    rollout()
    if args.mode != "student":
        teacher.update(obs["privileged"][teaching])
    if student is not None:
        update_student()
    if k % args.log_interval == 0 or k == iterations:
        torch.cuda.synchronize()
        row = {"iteration": k, "samples_m": k * T * num_envs / 1.0e6, "time": time.perf_counter() - start, **measure()}
        row |= {"lr": float(teacher.lr), "std": teacher.agent.log_std.exp().mean().item()}
        row["reward"] = teacher.buffer["rewards"].mean().item()
        history.append(row)
        print("LOG " + " ".join(f"{name} {value:.3f}" for name, value in row.items()), flush=True)
train_time = time.perf_counter() - start
samples = iterations * T * num_envs
print(f"TRAIN {train_time:.1f} s, {samples / train_time / 1e6:.2f} M steps/s, graphs {env.num_graphs}")


def play(observations):
    """Act without exploration noise: the teacher in its worlds, the student in the others."""
    count = observations["privileged"].shape[0]
    half = count if student is None else (0 if args.mode == "student" else count // 2)
    result = torch.zeros(count, action_dim, device=DEVICE)
    result[:half] = teacher.policy(observations["privileged"][:half])
    if student is not None:
        student.eval()
        result[half:] = student(observations["student"][half:])
    return result


# Run both without exploration noise, and average the measures over ten seconds.
totals = {}
with torch.no_grad():
    for _ in range(12):
        falls.zero_()
        for _ in range(T):
            obs, _, terminated, _, _ = env.step(play(obs))
            falls.add_(terminated)
        for name, value in measure().items():
            totals[name] = totals.get(name, 0.0) + value / 12
print("EVAL " + " ".join(f"{name} {value:.3f}" for name, value in totals.items()))
if args.mode == "teacher" and args.teacher:
    torch.save(teacher.agent.state_dict(), args.teacher)
if args.out:
    with open(args.out, "w") as file:
        json.dump({"args": vars(args), "train_time": train_time, "history": history, "evaluation": totals}, file)
env.close()

if args.video and args.mode != "teacher":
    env = make_env(4, "together" if args.mode == "together" else "student", camera=True)

    class Obs(dict):
        """The observations of every group, which the recorder passes on as ``observations["policy"]``."""

        def __getitem__(self, key):
            return self if key == "policy" else dict.__getitem__(self, key)

    reset, step = env.reset, env.step
    env.reset = lambda: (Obs(reset()[0]), {})
    env.step = lambda action: (Obs(step(action)[0]),)
    record(env, play, args.video, 300, 4)
    env.close()
