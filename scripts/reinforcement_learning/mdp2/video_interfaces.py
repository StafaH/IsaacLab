# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a policy of ``train_interfaces.py`` on one clip, with one tile per interface.

    uv run --extra ovrtx python scripts/reinforcement_learning/mdp2/video_interfaces.py --model model.pt \
        --clip dance1_subject3 --frame 600 --out video.mp4

Every world starts on the same frame of the clip and is told its targets through another interface, so that the
tiles show what the policy makes of the same motion with less and less information. A world that loses the clip
starts on that frame again. ``--renderer newton`` draws the tiles with the Newton Warp renderer instead of RTX.
"""

import argparse

import gymnasium as gym
import torch
from isaaclab_experimental import mdp2
from ppo import Agent
from video import add_camera, record

import isaaclab.sim as sim_utils
from isaaclab.app import launch_simulation
from isaaclab.assets import AssetBaseCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.interfaces.config.g1.g1_env_cfg import INTERFACES, PUSHED, TEST_CLIPS, TRAIN_CLIPS
from isaaclab_tasks.mdp2.interfaces.interfaces_env_cfg import INTERFACE
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Model or checkpoint of train_interfaces.py")
parser.add_argument("--task", default="Isaac-Interfaces-G1-MDP2")
parser.add_argument("--clip", default=TEST_CLIPS[0])
parser.add_argument("--frame", type=int, default=0, help="Frame of the clip on which the worlds start")
parser.add_argument("--interfaces", nargs="+", default=None, help="Interfaces to show; all of the task by default")
parser.add_argument("--push", type=float, default=1.0, help="Scale of the pushes of the pushed worlds")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--renderer", default="ovrtx", choices=["ovrtx", "newton"])
parser.add_argument("--light", type=float, nargs=2, default=[250.0, 900.0], help="Intensity of the sky and the sun")
parser.add_argument("--size", type=int, nargs=2, default=[960, 540], help="Width and height of a tile")
parser.add_argument("--columns", type=int, default=2)
parser.add_argument("--hidden", type=int, nargs="+", default=[1024, 512, 256])
parser.add_argument("--out", required=True, help="Path of the MP4")
args = parser.parse_args()

TASK, DEVICE = args.task, "cuda:0"
CLIPS = (*TRAIN_CLIPS, *TEST_CLIPS)

cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
names = [*INTERFACES, *PUSHED] if cfg.pushed else list(INTERFACES)
shown = args.interfaces or names
# The worlds are far apart, so that a tile shows one robot.
cfg.sim.device, cfg.scene.num_envs, cfg.scene.env_spacing = DEVICE, len(shown), 40.0
renderer_cfg = None
if args.renderer == "ovrtx":
    from isaaclab_ov.renderers import OVRTXRendererCfg

    renderer_cfg = OVRTXRendererCfg()
    cfg.scene.sky_light.spawn.intensity = args.light[0]
    cfg.scene.sun = AssetBaseCfg(
        prim_path="/World/sun",
        spawn=sim_utils.DistantLightCfg(intensity=args.light[1], angle=1.5, color=(1.0, 0.96, 0.9)),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.35, -0.2, 0.1, 0.9)),
    )
add_camera(cfg, *args.size, eye=(2.8, -3.4, 0.9), renderer_cfg=renderer_cfg)

with launch_simulation(cfg, {}):
    env = gym.make(TASK, cfg=cfg).unwrapped
    arrays = env.state.arrays
    clip_range = arrays[mdp2.term_state("events.reset_to_motion", "clip_range")].torch.long()
    first, last = clip_range[CLIPS.index(args.clip)].tolist()
    arrays[mdp2.term_state("events.reset_to_motion", "fixed_start")].torch.fill_(min(first + args.frame, last - 2))
    arrays[mdp2.term_state(INTERFACE, "fixed")].torch.copy_(
        torch.tensor([names.index(name) for name in shown], dtype=torch.float32, device=DEVICE)
    )
    arrays[mdp2.term_state(INTERFACE, "pushes")].torch.mul_(args.push)

    obs = env.reset()[0]
    state = torch.load(args.model)
    state = state.get("agent", state)
    critic_dim = state["critic.0.weight"].shape[1]
    agent = Agent(obs["policy"].shape[1], env.mdp.action_dim, args.hidden, 1.0, critic_dim).to(DEVICE)
    agent.load_state_dict(state)

    steps = round(args.seconds / env.step_dt)
    labels = [name.replace("_", " ") for name in shown]
    record(env, lambda obs: agent.actor(agent.obs(obs)), args.out, steps, args.columns, (2.8, -3.4, 0.9), labels=labels)
    env.close()
