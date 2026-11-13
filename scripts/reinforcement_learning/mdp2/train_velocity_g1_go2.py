# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a Unitree G1 and a Unitree Go2 to walk in the worlds of one simulation, with a policy per robot.

    uv run python scripts/reinforcement_learning/mdp2/train_velocity_g1_go2.py --budget 800

Half of the worlds hold a G1 and the other half a Go2, and each half runs the velocity task of its robot. The
policy of a robot acts in the worlds of that robot and learns from them. The script logs, per robot, the error of
the base velocity [m/s] and the share of the last episodes that ended in a fall.

``--robots g1`` or ``--robots go2`` trains one robot alone, in a scene without the other, with the same task,
physics, and learner: the reference for the robot's half of the worlds.
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

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.velocity_g1_go2.flat_env_cfg import G1_GO2_TASKS
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--robots", default="g1,go2", choices=["g1,go2", "g1", "go2"])
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=float, default=800.0, help="Samples to train for, in millions")
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--init_std", type=float, default=1.0)
parser.add_argument("--entropy", type=float, default=0.005)
parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--out", default=None, help="Path of a JSON file with the log")
parser.add_argument("--video", default=None, help="Path of an MP4 with worlds of every robot")
args = parser.parse_args()

TASK, DEVICE = "Isaac-Velocity-Flat-G1-Go2-MDP2", "cuda:0"
ROBOTS = ("g1", "go2")
NAMES = args.robots.split(",")
EYE = (1.7, -2.0, 0.5)
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True


def make_env(num_envs, camera=False):
    cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, num_envs, args.seed
    if len(NAMES) == 1:
        # One robot alone: a scene without the other, whose worlds are alike and need no sleeping robot.
        other = ROBOTS[1 - ROBOTS.index(NAMES[0])]
        setattr(cfg.scene, other, None)
        setattr(cfg.scene, f"{other}_contact_forces", None)
        cfg.scene.clone_cfg.clone_combinations = []
        cfg.sim.physics.solver_cfg.sleep_inactive_assets = False
        cfg.task = G1_GO2_TASKS[ROBOTS.index(NAMES[0])]
    if camera:
        add_camera(cfg, eye=EYE)
    return gym.make(TASK, cfg=cfg).unwrapped


def measure(env):
    """Return, per robot, the error of the base velocity [m/s] and the share of the last episodes that fell."""
    command = env.state.arrays[mdp2.commands.base_velocity].torch
    row = {}
    for name, config in zip(NAMES, env.mdp.world_configs):
        worlds = slice(config.worlds.start, config.worlds.stop)
        data = env.scene[name].data
        velocity = math_utils.quat_apply_inverse(
            data.root_link_pose_w.torch[worlds, 3:7], data.root_com_vel_w.torch[worlds, :3]
        )
        row[f"{name}/speed_error"] = (command[worlds, :2] - velocity[:, :2]).norm(dim=-1).mean()
        falls = [
            value
            for term, value in env.extras["log"].items()
            if term.startswith(f"Episode_Termination/{name}_") and not term.endswith("time_out")
        ]
        row[f"{name}/falls"] = sum(falls)
    return row


env = make_env(args.num_envs)
if len(NAMES) > 1:
    # The tasks split the worlds in halves, and so must the scene, for each task to run where its robot is.
    layout = env.scene.clone_plan.topology.world_prototype_layout
    for index, config in enumerate(env.mdp.world_configs):
        assert (layout[config.worlds.start : config.worlds.stop] == index).all(), "Use an even number of worlds."
obs = env.reset()[0]["policy"]
num_envs, T = env.num_envs, args.steps
actions = torch.zeros(num_envs, env.mdp.action_dim, device=DEVICE)

# A learner per robot, on the worlds of its robot and the columns that its task uses.
learners = {}
for name, config in zip(NAMES, env.mdp.world_configs):
    worlds = slice(config.worlds.start, config.worlds.stop)
    obs_dim, action_dim = config.observation_dims["policy"], config.action_dim
    learner = PPO(
        obs_dim,
        action_dim,
        len(config.worlds),
        DEVICE,
        hidden=args.hidden,
        steps=T,
        entropy=args.entropy,
        init_std=args.init_std,
    )
    learners[name] = (learner, worlds, obs_dim, action_dim)
    print(f"SETUP {name}: worlds {len(config.worlds)} obs {obs_dim} actions {action_dim}", flush=True)


def policy(obs):
    """Return the actions of every policy in its worlds, without exploration noise."""
    for learner, worlds, obs_dim, action_dim in learners.values():
        actions[worlds, :action_dim] = learner.policy(obs[worlds, :obs_dim])
    return actions


iterations = round(args.budget * 1.0e6 / (T * num_envs))
history = []
measures, measured = {}, 0
wp.synchronize()
start = time.perf_counter()
for k in range(1, iterations + 1):
    for _ in range(T):
        for learner, worlds, obs_dim, action_dim in learners.values():
            actions[worlds, :action_dim] = learner.act(obs[worlds, :obs_dim])
        obs, reward, terminated, truncated, _ = env.step(actions)
        obs = obs["policy"]
        for learner, worlds, _, _ in learners.values():
            learner.record(reward[worlds], terminated[worlds], truncated[worlds])
        # The measures are averaged over the steps between two log rows: the commands change every ten seconds.
        for name, value in measure(env).items():
            measures[name] = measures.get(name, 0.0) + value
        measured += 1
    for learner, worlds, obs_dim, _ in learners.values():
        learner.update(obs[worlds, :obs_dim])

    if k % args.log_interval == 0 or k == iterations:
        torch.cuda.synchronize()
        row = {"iteration": k, "samples_m": k * T * num_envs / 1.0e6, "time": time.perf_counter() - start}
        row |= {name: value.item() / measured for name, value in measures.items()}
        measures, measured = {}, 0
        for name, (learner, _, _, _) in learners.items():
            row[f"{name}/reward"] = learner.buffer["rewards"].mean().item()
        print("LOG " + " ".join(f"{name} {value:.3f}" for name, value in row.items()), flush=True)
        # The file also holds the rewards of the last episodes, per term.
        row |= {term: value.item() for term, value in env.extras["log"].items() if term.startswith("Episode_Reward")}
        history.append(row)
train_time = time.perf_counter() - start
samples = iterations * T * num_envs
print(f"TRAIN {train_time:.1f} s, {samples / train_time / 1e6:.2f} M steps/s, graphs {env.num_graphs}")

# Without exploration noise, over twenty seconds from a reset; the measures are averaged over the last ten.
totals = {}
with torch.no_grad():
    obs = env.reset()[0]["policy"]
    for step in range(1000):
        obs = env.step(policy(obs))[0]["policy"]
        if step >= 500:
            for name, value in measure(env).items():
                totals[name] = totals.get(name, 0.0) + value / 500
totals = {name: value.item() for name, value in totals.items()}
print("EVAL " + " ".join(f"{name} {value:.3f}" for name, value in totals.items()))
if args.out:
    with open(args.out, "w") as file:
        json.dump({"args": vars(args), "train_time": train_time, "history": history, "evaluation": totals}, file)
env.close()

if args.video:
    # Four worlds per robot, in a row per robot; each camera follows the robot that its world holds.
    columns = 4
    env = make_env(columns * len(NAMES), camera=True)
    actions = torch.zeros(env.num_envs, env.mdp.action_dim, device=DEVICE)
    for index, name in enumerate(NAMES):
        learner, _, obs_dim, action_dim = learners[name]
        learners[name] = (learner, slice(index * columns, (index + 1) * columns), obs_dim, action_dim)

    def roots():
        robots = [env.scene[name].data.root_link_pose_w.torch[learners[name][1], :3] for name in NAMES]
        return torch.cat(robots)

    record(env, policy, args.video, 500, columns, eye=EYE, roots=roots)
    env.close()
