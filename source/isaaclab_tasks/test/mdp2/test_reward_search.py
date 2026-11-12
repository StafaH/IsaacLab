# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MDP2 reward search task on Newton."""

import pytest
import torch
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv

from isaaclab_tasks.mdp2.reward_search.config.humanoid.humanoid_env_cfg import (
    CANDIDATES,
    HumanoidRewardSearchEnvCfg,
)
from isaaclab_tasks.mdp2.tracking.config.humanoid.humanoid_env_cfg import HumanoidTrackingPhysicsCfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")


def test_candidates_share_kernels_and_their_weights_change_in_the_captured_step():
    cfg = HumanoidRewardSearchEnvCfg()
    cfg.sim.physics = HumanoidTrackingPhysicsCfg().newton_mjwarp
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = 2 * len(CANDIDATES)
    cfg.seed = 0
    env = MDP2RLEnv(cfg)
    try:
        rewards = env.mdp.ops[mdp2.Phase.REWARDS]
        names = [op.name.split(".", 1)[1] for op in rewards]
        # Every candidate runs the same ops in all worlds, and is a row of the per-world weights.
        assert len(env.mdp.world_configs) == len(CANDIDATES) and all(op.worlds is None for op in rewards)
        weights = env.state.reward_weights.torch
        for index, candidate in enumerate(CANDIDATES):
            expected = torch.tensor([candidate.get(name, -0.03) for name in names], device=env.device)
            assert torch.allclose(weights[2 * index], expected)

        env.reset()
        actions = torch.zeros(env.num_envs, env.mdp.action_dim, device=env.device)
        for _ in range(3):
            env.step(actions)
        assert env.num_graphs == 1
        assert (env.transition.rewards.torch[:2] != 0.0).all()

        # A learner can rewrite a candidate's weights while the captured step keeps replaying.
        weights[:2] = 0.0
        env.step(actions)
        assert (env.transition.rewards.torch[:2] == 0.0).all() and (env.transition.rewards.torch[2:4] != 0.0).all()
    finally:
        env.close()
