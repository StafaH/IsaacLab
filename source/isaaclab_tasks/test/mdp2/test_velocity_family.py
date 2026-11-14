# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MDP2 velocity tasks of the robots of a family in the worlds of one simulation on Newton."""

import pytest
import torch
from isaaclab_experimental.envs import MDP2RLEnv

from isaaclab_tasks.mdp2.velocity_family import humanoids_env_cfg, quadrupeds_env_cfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")

WORLDS = 2
"""Worlds of each robot."""


@pytest.mark.parametrize(
    "family, env_cfg",
    [
        (quadrupeds_env_cfg, quadrupeds_env_cfg.QuadrupedsFlatEnvCfg),
        (humanoids_env_cfg, humanoids_env_cfg.HumanoidsFlatEnvCfg),
    ],
)
def test_robots_of_a_family_move_in_their_worlds_and_have_the_joints_of_the_family(family, env_cfg):
    cfg = env_cfg()
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = WORLDS * len(family.TASKS)
    cfg.seed = 0
    cfg.task_worlds = [WORLDS] * len(family.TASKS)
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        start = {name: env.scene[name].data.joint_pos.torch.clone() for name in family.TASKS}
        for _ in range(5):
            env.step(torch.ones(env.num_envs, env.mdp.action_dim, device=env.device))
        assert env.num_graphs == 1
        for index, name in enumerate(family.TASKS):
            moved = (env.scene[name].data.joint_pos.torch - start[name]).abs().amax(dim=-1) > 1.0e-3
            mine = torch.arange(env.num_envs, device=env.device) // WORLDS == index
            assert torch.equal(moved, mine)
            # A policy of the family reads and drives every joint of the robot at a place of its own.
            aliases = family.JOINT_ALIASES.get(name, {})
            places = {family.JOINTS.index(aliases.get(joint, joint)) for joint in env.scene[name].joint_names}
            assert len(places) == env.scene[name].num_joints
    finally:
        env.close()
