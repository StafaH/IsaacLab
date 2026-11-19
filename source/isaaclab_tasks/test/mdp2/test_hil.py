# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MDP2 hybrid imitation task on Newton."""

import pytest
import torch
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv

from isaaclab_tasks.mdp2.hil import mdp as hil
from isaaclab_tasks.mdp2.hil.config.humanoid.humanoid_env_cfg import HumanoidHILEnvCfg, humanoid_tasks
from isaaclab_tasks.mdp2.tracking.config.humanoid.humanoid_env_cfg import HumanoidTrackingPhysicsCfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")

NUM_ENVS = 8


def test_worlds_split_between_tasks_and_the_graph_reads_the_live_discriminator():
    cfg = HumanoidHILEnvCfg()
    cfg.sim.physics = HumanoidTrackingPhysicsCfg().newton_mjwarp
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    cfg.task = humanoid_tasks("walk")
    env = MDP2RLEnv(cfg)
    try:
        ops = {op.name: op for op in env.mdp.term_ops}
        half = tuple(range(NUM_ENVS // 2))
        # The tracking worlds and the goal worlds have their own goals and rewards, and share the style reward.
        assert ops["rewards.joint_pos"].worlds == half
        assert ops["rewards.goal"].worlds == tuple(range(NUM_ENVS // 2, NUM_ENVS))
        assert ops["commands.goal[0]"].worlds == half and ops["rewards.style"].worlds is None
        discriminator = ops["rewards.style"].params["discriminator"]
        style = list(ops).index("rewards.style") - list(ops).index(env.mdp.ops[mdp2.Phase.REWARDS][0].name)

        env.reset()
        actions = torch.zeros(NUM_ENVS, env.mdp.action_dim, device=env.device)
        for _ in range(3):
            env.step(actions)
        assert env.num_graphs == 1

        # The transition holds the motion features before and after the step.
        robot = env.scene["robot"].data
        features = hil.motion_features(
            robot.joint_pos.torch, robot.joint_vel.torch, robot.root_link_pose_w.torch, robot.root_com_vel_w.torch
        )
        transition = env.state.arrays[mdp2.term_state("rewards.style", "transition")].torch
        assert torch.allclose(transition[:, features.shape[1] :], features, atol=1.0e-5)

        # The captured step scores transitions with the discriminator's current weights: a score of 1 gives every
        # world the full style reward, and a score of 3 gives none.
        weight = env.state.reward_weights.torch[:, style] * env.step_dt
        with torch.no_grad():
            discriminator.net[-1].weight.zero_()
            discriminator.net[-1].bias.fill_(1.0)
        env.step(actions)
        assert torch.allclose(env.transition.reward_terms.torch[:, style], weight)
        with torch.no_grad():
            discriminator.net[-1].bias.fill_(3.0)
        env.step(actions)
        assert (env.transition.reward_terms.torch[:, style] == 0.0).all()
    finally:
        env.close()
