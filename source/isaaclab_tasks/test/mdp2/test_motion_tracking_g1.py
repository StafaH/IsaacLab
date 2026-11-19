# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MDP2 port of the G1 motion tracking task ``Isaac-Motion-Tracking-G1``."""

import pytest
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.datasets import MotionLoader

pytestmark = [pytest.mark.kitless, pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")]

NUM_ENVS = 8


@pytest.fixture(scope="module")
def motion_file() -> str:
    """The clip of the task, downloaded once; the tests skip without it."""
    from isaaclab_tasks.core.motion_tracking.config.g1.env_cfg import G1CommandsCfg

    motion_file = G1CommandsCfg().motion.motion_file
    try:
        retrieve_file_path(motion_file)
    except Exception as error:
        pytest.skip(f"Cannot download the clip {motion_file}: {error}")
    return motion_file


def mdp1_layout() -> tuple[dict[str, tuple], list[str], list[str]]:
    """The observation shapes, reward terms, and termination terms of the MDP1 task, with few worlds."""
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.core.motion_tracking.config.g1.env_cfg import G1MotionTrackingEnvCfg

    cfg = G1MotionTrackingEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = "cuda:0"
    env = ManagerBasedRLEnv(cfg)
    try:
        observations, _ = env.reset()
        shapes = {group: tuple(values.shape[1:]) for group, values in observations.items()}
        return shapes, env.reward_manager.active_terms, env.termination_manager.active_terms
    finally:
        env.close()


def test_resets_onto_the_reference_and_steps_as_one_graph_like_mdp1(motion_file: str):
    from isaaclab_experimental import mdp2
    from isaaclab_experimental.envs import MDP2RLEnv

    from isaaclab_tasks.mdp2.motion_tracking_g1.env_cfg import G1MotionTrackingEnvCfg, G1MotionTrackingPhysicsCfg

    shapes, rewards, terminations = mdp1_layout()

    cfg = G1MotionTrackingEnvCfg()
    cfg.sim.physics = G1MotionTrackingPhysicsCfg().newton_mjwarp
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    # Without the perturbations, a reset puts the robot exactly on its start frame.
    reset = cfg.task.events["reset_to_motion"]
    reset.params.update(pose_range={}, velocity_range={}, joint_position_range=(0.0, 0.0))
    env = MDP2RLEnv(cfg)
    try:
        observations, _ = env.reset()
        assert {group: tuple(values.shape[1:]) for group, values in observations.items()} == shapes
        assert list(cfg.task.rewards) == rewards and list(cfg.task.terminations) == terminations

        # Root, joints, and forward kinematics of the anchor match the clip, in the robot's joint and body order.
        robot = env.scene["robot"]
        motion = MotionLoader(motion_file, robot.body_names, robot.joint_names, env.device)
        start = env.state.arrays[mdp2.term_state("events.reset_to_motion", "start")].torch.long()
        assert len(start.unique()) > 1
        origins = env.scene.env_origins
        limits = robot.data.soft_joint_pos_limits.torch
        joint_pos = motion.joint_pos[start].clamp(limits[..., 0], limits[..., 1])
        torch.testing.assert_close(robot.data.joint_pos.torch, joint_pos)
        torch.testing.assert_close(robot.data.joint_vel.torch, motion.joint_vel[start])
        torso = robot.body_names.index("torso_link")
        for body, atol in ((0, 1.0e-5), (torso, 1.0e-4)):
            pose = robot.data.body_link_pose_w.torch[:, body]
            torch.testing.assert_close(pose[:, :3], motion.body_pos_w[start, body] + origins, atol=atol, rtol=0.0)
            # Quaternions, (x, y, z, w), match up to their sign.
            alignment = (pose[:, 3:] * motion.body_quat_w[start, body]).sum(-1).abs()
            torch.testing.assert_close(alignment, torch.ones_like(alignment), atol=atol, rtol=0.0)

        # The training scripts start episodes at random lengths, which the reference frame does not follow.
        env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=400)
        ended = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.device)
        for _ in range(3):
            _, reward, terminated, truncated, _ = env.step(torch.zeros(NUM_ENVS, env.mdp.action_dim, device=env.device))
            ended |= terminated | truncated
        assert env.num_graphs == 1
        assert reward.shape == (NUM_ENVS,) and torch.isfinite(reward).all()
        # Observations show the frame after the one the step tracked.
        observed = env.state.arrays[mdp2.quantities.next_motion_frame].torch.long()
        assert not ended.all() and torch.equal(observed[~ended], (start + 4)[~ended])
    finally:
        env.close()
