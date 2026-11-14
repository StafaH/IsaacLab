# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a position-free policy of ``train_segments.py`` for a robot or another simulator.

    uv run python scripts/reinforcement_learning/mdp2/export_interfaces.py --model model.pt \
        --clip dance1_subject3 --out export/dance1

The policy must be trained with ``--local``: it then reads only what a robot measures. The folder ``--out`` gets

* ``policy.pt`` and ``policy.onnx``: the targets of the joints from the observations, normalization included,
* ``spec.json``: the joints with their default pose and gains, the scale of the actions, the control rate and the
  layout of the observations,
* ``clip.npz``: the frames of ``--clip`` at the control rate, as the joints, the bodies and the root of the robot,
  with the velocities of the joints and of the root,
* ``rollout.npz``: one episode of the policy on the clip in this simulator, as what the robot measured, what the
  policy saw and what it answered, to check an implementation of the observations against.

``sim2sim_interfaces.py`` runs such a folder in MuJoCo.
"""

import argparse
import json
import os
import re

import gymnasium as gym
import numpy as np
import torch
from isaaclab_experimental import mdp2
from ppo import Agent

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.interfaces.config.g1.g1_env_cfg import (
    BODY_GROUPS,
    CLIP_SETS,
    INTERFACES,
    JOINT_GROUPS,
    TEST_CLIPS,
    g1_interfaces_task,
)
from isaaclab_tasks.mdp2.interfaces.interfaces_env_cfg import INTERFACE
from isaaclab_tasks.mdp2.tracking.config.g1.g1_env_cfg import ARMATURE
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Model or checkpoint of train_segments.py --local")
parser.add_argument("--task", default="Isaac-Interfaces-G1-MDP2")
parser.add_argument("--clips", default="curated", choices=list(CLIP_SETS), help="The training clips")
parser.add_argument("--clip", default=TEST_CLIPS[0])
parser.add_argument("--frame", type=int, default=0, help="Frame of the clip on which the episode starts")
parser.add_argument("--interface", default="full", choices=list(INTERFACES))
parser.add_argument("--substeps", type=int, default=2, help="Physics steps per physics step of the task")
parser.add_argument("--armature", action="store_true", help="Give the joints the inertia of the real robot's motors")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--hidden", type=int, nargs="+", default=[1024, 512, 256])
parser.add_argument("--out", required=True, help="Folder of the exported files")
args = parser.parse_args()

TASK, DEVICE = args.task, "cuda:0"
CLIP = ("clip_joint_pos", "clip_joint_vel", "clip_body_pos", "clip_root_pose", "clip_root_vel")
CLIPS = (*CLIP_SETS[args.clips], *TEST_CLIPS)

cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
cfg.task = g1_interfaces_task(train=CLIP_SETS[args.clips], test=TEST_CLIPS, local=True)
cfg.sim.device, cfg.scene.num_envs = DEVICE, 1
cfg.sim.dt, cfg.decimation = cfg.sim.dt / args.substeps, cfg.decimation * args.substeps
cfg.sim.render_interval = cfg.decimation
if args.armature:
    cfg.scene.robot.actuators["body"].armature = ARMATURE
env = gym.make(TASK, cfg=cfg).unwrapped
arrays = env.state.arrays
reset = {
    name: arrays[mdp2.term_state("events.reset_to_motion", name)].torch for name in ("clip_range", "fixed_start", *CLIP)
}
first, last = reset["clip_range"][CLIPS.index(args.clip)].long().tolist()
reset["fixed_start"].fill_(min(first + args.frame, last - 2))
arrays[mdp2.term_state(INTERFACE, "fixed")].torch.fill_(list(INTERFACES).index(args.interface))

robot = env.scene["robot"]
data = robot.data


def ids(selector, names):
    """Return the indices of the names that a selector of the task matches, in the order of the robot."""
    return [k for k, name in enumerate(names) if any(re.fullmatch(pattern, name) for pattern in selector.patterns)]


# The observations of the policy in their order, each with the bodies or joints that it reads.
action = cfg.task.actions["joint_pos"]
terms = []
for name, term in cfg.task.observations["policy"].terms.items():
    entry = {"name": name, "scale": term.scale if term.scale is not None else 1.0}
    if "body_ids" in term.inputs:
        entry |= {"bodies": ids(term.inputs["body_ids"], robot.body_names), **term.params}
    elif "joint_ids" in term.inputs:
        entry |= {"joints": ids(term.inputs["joint_ids"], robot.joint_names), **term.params}
    terms.append(entry)
groups = [*BODY_GROUPS, *JOINT_GROUPS]
spec = {
    "step_dt": env.step_dt,
    "joints": robot.joint_names,
    "bodies": robot.body_names,
    "default_joint_pos": data.default_joint_pos.torch[0].tolist(),
    "stiffness": data.joint_stiffness.torch[0].tolist(),
    "damping": data.joint_damping.torch[0].tolist(),
    "armature": data.joint_armature.torch[0].tolist(),
    "effort_limits": data.joint_effort_limits.torch[0].tolist(),
    "action_joints": ids(action.subset, robot.joint_names),
    "action_scale": action.scale,
    "observations": terms,
    "interface": [float(group in INTERFACES[args.interface]) for group in groups],
    "clip": args.clip,
}

obs = env.reset()[0]
state = torch.load(args.model)
state = state.get("agent", state)
agent = Agent(obs["policy"].shape[1], env.mdp.action_dim, args.hidden, 1.0, state["critic.0.weight"].shape[1])
agent.load_state_dict(state)
policy = torch.nn.Sequential(agent.obs, agent.actor).to(DEVICE).eval()

# One episode: what the robot measured before every step, what the policy saw and what it answered.
names = ("joint_pos", "joint_vel", "root_link_pose_w", "root_com_vel_w", "body_link_pose_w")
rollout = {name: [] for name in (*names, "obs", "action", "frame")}
start = int(reset["fixed_start"][0].item())
with torch.no_grad():
    for step in range(round(args.seconds / env.step_dt)):
        actions = policy(obs["policy"])
        for name in names:
            rollout[name].append(getattr(data, name).torch[0].cpu().numpy().copy())
        rollout["obs"].append(obs["policy"][0].cpu().numpy())
        rollout["action"].append(actions[0].cpu().numpy())
        rollout["frame"].append(min(start + step, last - 1))
        obs, _, terminated, truncated, _ = env.step(actions)
        if (terminated | truncated).any():
            break
print(f"EXPORT clip {args.clip} frames {first}..{last} episode of {len(rollout['frame'])} steps")

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "spec.json"), "w") as file:
    json.dump(spec, file, indent=1)
clip = {name.removeprefix("clip_"): reset[name][first:last].cpu().numpy() for name in CLIP}
np.savez(os.path.join(args.out, "clip.npz"), **clip)
np.savez(os.path.join(args.out, "rollout.npz"), first=first, **{k: np.stack(v) for k, v in rollout.items()})
policy = policy.cpu()
example = torch.zeros(1, obs["policy"].shape[1])
torch.jit.trace(policy, example).save(os.path.join(args.out, "policy.pt"))
torch.onnx.export(
    policy,
    (example,),
    os.path.join(args.out, "policy.onnx"),
    input_names=["obs"],
    output_names=["action"],
    dynamo=False,
)
env.close()
