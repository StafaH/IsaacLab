# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MDP2 multi-interface tracking task of the Unitree G1 on Newton."""

import os

import pytest
import torch
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv

from isaaclab.utils import replace

from isaaclab_tasks.mdp2.interfaces.config.g1.g1_env_cfg import (
    INTERFACES,
    PUSHED,
    G1InterfacesEnvCfg,
    G1InterfacesPhysicsCfg,
    g1_interfaces_task,
)
from isaaclab_tasks.mdp2.interfaces.interfaces_env_cfg import INTERFACE
from isaaclab_tasks.mdp2.tracking.config.g1.g1_env_cfg import MOTIONS_DIR

CLIPS = ("walk1_subject1", "walk1_subject2")
NUM_ENVS = 64
RESET = "events.reset_to_motion"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not all(os.path.exists(os.path.join(MOTIONS_DIR, f"{c}.csv")) for c in CLIPS),
    reason="Needs a GPU and the downloaded G1 clips",
)


def test_worlds_see_the_targets_of_their_interface_and_evaluation_worlds_keep_their_start():
    cfg = G1InterfacesEnvCfg()
    cfg.sim.physics = G1InterfacesPhysicsCfg().newton_mjwarp
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = "cuda:0", NUM_ENVS, 0
    cfg.task = g1_interfaces_task(train=CLIPS[:1], test=CLIPS[1:])
    env = MDP2RLEnv(cfg)
    try:
        arrays = env.state.arrays
        names = list(INTERFACES)
        clip_range = arrays[mdp2.term_state(RESET, "clip_range")].torch
        held_out = int(clip_range[1, 0]) + 100

        # The last world evaluates the three-point interface on a frame of the held-out clip.
        arrays[mdp2.term_state(INTERFACE, "fixed")].torch[-1] = names.index("three_point")
        arrays[mdp2.term_state(RESET, "fixed_start")].torch[-1] = held_out
        arrays[mdp2.term_state(INTERFACE, "shares")].torch.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        observations = env.reset()[0]
        start = arrays[mdp2.term_state(RESET, "start")].torch
        assert start[-1] == held_out and (start[:-1] < clip_range[0, 1]).all()

        # Its policy sees fewer targets than the teacher's observations of the same world; a full world sees all.
        policy, teacher = observations["policy"], observations["teacher"]
        assert torch.equal(policy[0], teacher[0])
        hidden = (policy[-1] == 0.0) & (teacher[-1] != 0.0)
        shown = (policy[-1] != 0.0) & (policy[-1] == teacher[-1])
        assert hidden.sum() > 50 and shown.sum() > 20

        for _ in range(3):
            env.step(torch.zeros(NUM_ENVS, env.mdp.action_dim, device=env.device))
        assert env.num_graphs == 1
    finally:
        env.close()


def test_only_the_worlds_of_pushed_interfaces_are_pushed():
    cfg = G1InterfacesEnvCfg()
    cfg.sim.physics = G1InterfacesPhysicsCfg().newton_mjwarp
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = "cuda:0", 512, 0
    task = g1_interfaces_task(train=CLIPS[:1], test=CLIPS[1:], pushed=True)
    # Push within the first steps, before the robots fall.
    cfg.task = replace(
        task, events={**task.events, "push": replace(task.events["push"], interval_range_s=(0.04, 0.08))}
    )
    env = MDP2RLEnv(cfg)
    try:
        arrays = env.state.arrays
        names = [*INTERFACES, *PUSHED]
        fixed = arrays[mdp2.term_state(INTERFACE, "fixed")].torch
        fixed[:256] = names.index("full")
        fixed[256:] = names.index("full_pushed")
        # Pushes of ten times the default, to tell them from the motion of the robots.
        arrays[mdp2.term_state(INTERFACE, "pushes")].torch.mul_(10.0)
        observations = env.reset()[0]
        push = arrays[mdp2.term_state(INTERFACE, "push")].torch
        assert (push[:256] == 0.0).all() and (push[256:] == 10.0).all()
        # The critic sees the push of its world, after the targets of the interface.
        assert torch.equal(observations["critic"][:, -1], push)

        for _ in range(3):
            env.step(torch.zeros(512, env.mdp.action_dim, device=env.device))
        error = arrays[mdp2.quantities.root_vel_error].torch
        assert error[256:].mean() > 10.0 * error[:256].mean()
    finally:
        env.close()
