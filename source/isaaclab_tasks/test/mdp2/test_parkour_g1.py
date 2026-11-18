# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Unitree G1 parkour task of MDP2."""

import os

import pytest
import torch
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv

from isaaclab_tasks.mdp2.parkour.config.g1.g1_env_cfg import (
    MOTIONS_DIR,
    G1ParkourEnvCfg,
    G1ParkourPhysicsCfg,
    g1_task,
    g1_terrain,
)

CLIP, LEVEL = "climb_26", "1.0"
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU"),
    pytest.mark.skipif(
        not os.path.exists(os.path.join(MOTIONS_DIR, f"{CLIP}_z{LEVEL}.csv")), reason="The clips are downloaded"
    ),
]

START = "events.start_episode"


def test_a_goal_episode_starts_where_a_learner_puts_it_and_pays_once_for_standing_at_the_goal():
    cfg = G1ParkourEnvCfg()
    cfg.sim.physics = G1ParkourPhysicsCfg().newton_mjwarp
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = "cuda:0", 2, 0
    cfg.scene.terrain.terrain_generator = g1_terrain((CLIP,), (LEVEL,))
    cfg.task = g1_task((CLIP,), (LEVEL,))
    env = MDP2RLEnv(cfg)
    try:
        arrays = env.state.arrays
        pending = arrays[mdp2.term_state(START, "pending")].torch
        clip_root_pose = arrays[mdp2.term_state(START, "clip_root_pose")].torch
        last = int(arrays[mdp2.term_state(START, "clip_range")].torch[0, 1])
        # Both worlds start on a frame where the clip stands on the box, moved along x, and their episodes end after
        # two steps. The goal of the first is that frame, and the goal of the second the first frame of the clip.
        frame, moved = last - 10, 0.05
        for world, goal in enumerate((frame, 0)):
            pending[world] = torch.tensor(
                [frame, last, goal, 1.0e6, 1.0e9, 0.3, 1.0e9, 2.0, moved, 0.0, 0.0], device=env.device
            )
        env.reset()
        root = env.scene["robot"].data.root_link_pose_w.torch[:, :3] - env.scene.env_origins
        expected = clip_root_pose[frame, :3] + torch.tensor([moved, 0.0, 0.0], device=env.device)
        torch.testing.assert_close(root, expected.expand(2, 3), atol=1.0e-4, rtol=0.0)

        # The policy holds the joints where they are, by the action of the standing pose.
        actions = torch.zeros(2, env.mdp.action_dim, device=env.device)
        _, reward, terminated, _, _ = env.step(actions)
        assert not terminated.any() and reward[0] < 5.0
        _, reward, terminated, _, _ = env.step(actions)
        assert terminated.all()
        assert reward[0] > 19.0 and reward[1] < 5.0
    finally:
        env.close()
