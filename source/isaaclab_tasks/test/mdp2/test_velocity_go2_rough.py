# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Unitree Go2 rough velocity task of MDP2."""

import pytest
import torch
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv

from isaaclab.utils import replace

from isaaclab_tasks.mdp2.velocity_go2.rough_env_cfg import ROUGH_TASK, UnitreeGo2RoughEnvCfg, UnitreeGo2RoughPhysicsCfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")

NUM_ENVS = 8


def test_reset_moves_the_robot_and_its_height_scan_to_the_terrain_a_learner_assigns():
    cfg = UnitreeGo2RoughEnvCfg()
    cfg.sim.physics = UnitreeGo2RoughPhysicsCfg().newton_mjwarp
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = "cuda:0", NUM_ENVS, 0
    # Episodes of two steps, so that every world resets in every second step.
    time_out = mdp2.TimeLimit(episode_length_s=2 * cfg.sim.dt * cfg.decimation)
    cfg.task = replace(ROUGH_TASK, terminations={**ROUGH_TASK.terminations, "time_out": time_out})
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        # A learner assigns every world the hardest level of the last type of terrain.
        origin = env.scene.terrain.terrain_origins[-1, -1]
        env.state.arrays[mdp2.term_state("events.terrain_origin", "origin")].torch[:] = origin

        actions = torch.zeros(NUM_ENVS, env.mdp.action_dim, device=env.device)
        for _ in range(2):
            observations, _, _, truncated, _ = env.step(actions)
        assert truncated.all() and env.num_graphs == 1

        # The reset placed every robot within its pose range of the assigned origin.
        root = env.scene["robot"].data.root_link_pose_w.torch[:, :3]
        assert ((root[:, :2] - origin[:2]).abs() <= 0.5 + 1.0e-3).all()
        # The scanner cast its rays again from there, so the first observation of the episode shows the new terrain.
        scanner = env.scene_data.data("height_scanner", "pos_w").torch
        torch.testing.assert_close(scanner[:, :2], root[:, :2], atol=1.0e-4, rtol=0.0)
        assert observations["policy"].isfinite().all()
    finally:
        env.close()
