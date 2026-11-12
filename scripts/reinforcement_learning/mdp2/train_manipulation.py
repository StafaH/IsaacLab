# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train one policy for several Kuka Allegro manipulation tasks with RSL-RL, and report the success of each.

    uv run python scripts/reinforcement_learning/mdp2/train_manipulation.py --iterations 4000

The worlds are split between the tasks of ``isaaclab_tasks.mdp2.manipulation``: lifting and placing by default.
After training, the policy acts without exploration noise for ``--eval_steps`` steps, and the script reports per
task the share of finished episodes that reached their goal. ``--checkpoint`` loads a model instead of training,
and ``--video`` records it.
"""

import argparse
import json
import time

import gymnasium as gym
import torch
from isaaclab_experimental import mdp2
from rsl_rl.runners import OnPolicyRunner
from video import add_camera, record

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.manipulation.config.kuka_allegro.kuka_allegro_env_cfg import SKILLS
from isaaclab_tasks.mdp2.skills.skills_env_cfg import skill_tasks
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--skills", default="lift,place", help="Tasks to train, of lift, place, and reorient")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--iterations", type=int, default=4000)
parser.add_argument("--eval_steps", type=int, default=720)
parser.add_argument("--log_dir", default=None)
parser.add_argument("--out", default=None)
parser.add_argument("--checkpoint", default=None, help="Model to load instead of training")
parser.add_argument("--video", default=None, help="Path of an MP4 with one world per task")
args = parser.parse_args()

TASK = "Isaac-Manipulation-KukaAllegro-MDP2"
NAMES = args.skills.split(",")
PROGRESS = mdp2.term_state("events.adr", "progress")
EYE, CENTER = (0.9, -1.6, 0.6), (-0.5, 0.0, 0.5)
agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
agent_cfg.seed = args.seed


def make_env(num_envs, camera=False):
    cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = "cuda:0", num_envs, args.seed
    tasks = skill_tasks({name: SKILLS[name] for name in NAMES})
    cfg.task = tasks if len(tasks) > 1 else tasks[0]
    if camera:
        add_camera(cfg, eye=EYE)
    return RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=agent_cfg.clip_actions)


if args.video:
    # One world per task, at the end of the curriculum: full gravity and observation noise.
    env = make_env(len(NAMES), camera=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)
    env.unwrapped.state.arrays[PROGRESS].torch.fill_(1.0)
    actions = lambda _: policy(env.get_observations())  # noqa: E731
    record(env.unwrapped, actions, args.video, 720, len(NAMES), eye=EYE, center=CENTER)
    env.close()
    raise SystemExit

env = make_env(args.num_envs)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=args.log_dir, device=agent_cfg.device)
train_time = 0.0
if args.checkpoint:
    runner.load(args.checkpoint)
else:
    start = time.time()
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    train_time = time.time() - start
    print(f"TRAIN_TIME {train_time:.1f} s")

# Evaluate: an episode's success is the quantity's value on the step before the episode ended.
policy = runner.get_inference_policy(device=agent_cfg.device)
success = env.unwrapped.state.arrays[mdp2.quantities.success].torch
episodes, reached = torch.zeros_like(success), torch.zeros_like(success)
obs = env.get_observations()
with torch.inference_mode():
    for _ in range(args.eval_steps):
        before = success.clone()
        obs, _, dones, _ = env.step(policy(obs))
        episodes += dones
        reached += dones * before
results = {"train_time": train_time, "progress": env.unwrapped.state.arrays[PROGRESS].torch.mean().item()}
for name, config in zip(NAMES, env.unwrapped.mdp.world_configs):
    worlds = slice(config.worlds.start, config.worlds.stop)
    results[name] = (reached[worlds].sum() / episodes[worlds].sum()).item()
    print(f"EVAL {name}: {results[name]:.3f} of {int(episodes[worlds].sum())} episodes")
if args.out:
    with open(args.out, "w") as file:
        json.dump(results, file)
env.close()
