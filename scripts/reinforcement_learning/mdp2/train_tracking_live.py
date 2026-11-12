# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train MDP2 motion tracking with evaluation worlds in the training simulation.

    uv run python scripts/reinforcement_learning/mdp2/train_tracking_live.py --motion walk --budget 8

The last ``--eval_envs`` worlds run the evaluation task, from the first frame of a clip without noise, and the policy
acts there without exploration. PPO trains on the other worlds only. The share of evaluation worlds whose last
episode reached the end of its clip is read every iteration, so the evaluation needs no second environment and no
pass over saved policies, and its curve has the resolution of an iteration.
"""

import argparse
import json
import time

import gymnasium as gym
import torch
import warp as wp
from ppo import PPO

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.tracking.config.humanoid.humanoid_env_cfg import evaluation_task, humanoid_task
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--motion", default="walk")
parser.add_argument("--num_envs", type=int, default=8192, help="Training worlds")
parser.add_argument("--eval_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=float, default=8.0, help="Training samples, in millions")
parser.add_argument("--lost_distance", type=float, default=None)
parser.add_argument("--out", default=None)
args = parser.parse_args()

DEVICE, TASK = "cuda:0", "Isaac-Tracking-Humanoid-Walk-MDP2"
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
begin = time.perf_counter()
cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, args.num_envs + args.eval_envs, args.seed
cfg.task = [humanoid_task(args.motion), evaluation_task(args.motion)]
cfg.task_worlds = [args.num_envs, args.eval_envs]
if args.lost_distance is not None:
    for task in cfg.task:
        task.terminations["tracking_lost"].params["threshold"] = args.lost_distance**2
env = gym.make(TASK, cfg=cfg).unwrapped
obs = env.reset()[0]["policy"]
training, evaluating = slice(0, args.num_envs), slice(args.num_envs, None)
T = 6
ppo = PPO(
    obs.shape[1],
    env.mdp.action_dim,
    args.num_envs,
    DEVICE,
    steps=T,
    epochs=4,
    kl=0.15,
    gamma=0.95,
    clip=0.3,
    value_coef=4.0,
    init_std=0.3,
)
iterations = round(args.budget * 1.0e6 / (T * args.num_envs))
actions = torch.zeros(env.num_envs, env.mdp.action_dim, device=DEVICE)
# Whether the last episode of each evaluation world reached the end of its clip; none has finished at first.
completed = torch.zeros(args.eval_envs, device=DEVICE)
curve = torch.zeros(iterations, device=DEVICE)
times = []


def iteration():
    global obs
    for _ in range(T):
        actions[training] = ppo.act(obs[training])
        with torch.no_grad():
            actions[evaluating] = ppo.policy(obs[evaluating])
        obs, reward, terminated, truncated, _ = env.step(actions)
        obs = obs["policy"]
        ppo.record(reward[training], terminated[training], truncated[training])
        done = terminated[evaluating] | truncated[evaluating]
        completed.copy_(torch.where(done, truncated[evaluating].float(), completed))
    ppo.update(obs[training])


# Warm up outside the clock: the first steps capture the step graph and the update graph.
for _ in range(3):
    iteration()
ppo.reset()
completed.zero_()
obs = env.reset()[0]["policy"]

wp.synchronize()
start = time.perf_counter()
for k in range(iterations):
    iteration()
    curve[k] = completed.mean()
    times.append(time.perf_counter() - start)
torch.cuda.synchronize()
train_time = time.perf_counter() - start
curve = curve.tolist()
reached = next((times[k] for k in range(iterations) if curve[k] >= 0.9), None)
samples = iterations * T * args.num_envs
print(
    f"LIVE {args.motion}: evaluation success {curve[-1]:.3f}, reached 90% at"
    f" {'never' if reached is None else f'{reached:.2f} s'}, training {train_time:.2f} s at"
    f" {samples / train_time / 1e6:.2f} M training steps/s, whole script {time.perf_counter() - begin:.1f} s,"
    f" graphs {env.num_graphs}"
)
if args.out:
    with open(args.out, "w") as file:
        json.dump({"args": vars(args), "times": times, "success": curve, "train_time": train_time}, file)
env.close()
