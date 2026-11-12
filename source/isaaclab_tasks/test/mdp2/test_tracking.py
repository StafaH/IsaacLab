# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MDP2 humanoid motion tracking task on Newton."""

import os

import numpy as np
import pytest
import torch
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv

from isaaclab_tasks.mdp2.tracking.config.g1.g1_env_cfg import MOTIONS_DIR as G1_MOTIONS_DIR
from isaaclab_tasks.mdp2.tracking.config.g1.g1_env_cfg import G1TrackingEnvCfg, G1TrackingPhysicsCfg, g1_task
from isaaclab_tasks.mdp2.tracking.config.humanoid.humanoid_env_cfg import (
    MOTIONS_DIR,
    HumanoidTrackingEnvCfg,
    HumanoidTrackingPhysicsCfg,
    humanoid_task,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")

NUM_ENVS = 8
RESET = "events.reset_to_motion"


def test_reset_starts_on_the_clip_and_actions_correct_the_reference():
    cfg = HumanoidTrackingEnvCfg()
    cfg.sim.physics = HumanoidTrackingPhysicsCfg().newton_mjwarp
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    cfg.task = humanoid_task("walk")
    cfg.task.events["reset_to_motion"].params["noise"] = 0.0
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        robot = env.scene["robot"]
        start = env.state.arrays[mdp2.term_state(RESET, "start")].torch.long()
        assert len(start.unique()) > 1

        # The clip holds ball joints as rotation vectors, in its own joint and body order: the forward kinematics
        # of the reset state must put every body where the clip has it.
        clip = np.load(os.path.join(MOTIONS_DIR, "humanoid_walk.npz"))
        stride = round(float(clip["fps"]) * env.step_dt)
        body_ids = [list(clip["body_names"]).index(name) for name in robot.body_names]
        expected = torch.from_numpy(clip["body_positions"][:, body_ids]).to(env.device)[start * stride]
        expected += env.scene.env_origins[:, None]
        assert torch.allclose(robot.data.body_link_pose_w.torch[..., :3], expected, atol=1.0e-3)

        # With zero actions, the joint targets are the frame that the step tracked, and the next one is observed.
        for _ in range(3):
            env.step(torch.zeros(NUM_ENVS, env.mdp.action_dim, device=env.device))
        assert env.num_graphs == 1
        length = env.episode_length_buf
        assert (length == 3).all()
        clip_joint_pos = env.state.arrays[mdp2.term_state(RESET, "clip_joint_pos")].torch
        targets = env.scene_data.data("robot", "joint_pos_target").torch
        assert torch.equal(targets, clip_joint_pos[start + 3])
        reference = env.state.arrays[mdp2.quantities.reference_joint_pos].torch
        assert torch.equal(reference, clip_joint_pos[start + 4])
        assert torch.equal(env.observations["policy"][:, 111:139], reference)
    finally:
        env.close()


def test_failure_driven_resets_start_where_episodes_ended():
    cfg = HumanoidTrackingEnvCfg()
    cfg.sim.physics = HumanoidTrackingPhysicsCfg().newton_mjwarp
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = 64
    cfg.seed = 0
    cfg.task = humanoid_task("walk,run", adaptive=1.0)
    # Limp joints: every world falls off its clip within a few steps, so the failures pile up near the starts.
    cfg.task.actions["joint_pos"].scale = 0.0
    cfg.task.terminations["tracking_lost"].params["threshold"] = 0.01
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        state = {name: env.state.arrays[mdp2.term_state(RESET, name)].torch for name in ("failures", "start", "end")}
        assert state["failures"].sum() == 0.0
        actions = torch.zeros(env.num_envs, env.mdp.action_dim, device=env.device)
        for _ in range(60):
            env.step(actions)
        assert env.num_graphs == 1

        # Starts are drawn from the bins that hold failures, up to one bin before them, and stay on their clip.
        failures = state["failures"]
        assert failures.sum() > 0.0
        clip_range = env.state.arrays[mdp2.term_state(RESET, "clip_range")].torch
        start, end = state["start"], state["end"]
        assert ((end == clip_range[0, 1]) | (end == clip_range[1, 1])).all() and (start < end - 1).all()
        near_failures = torch.nn.functional.max_pool1d(failures[None, None], 3, 1, 1)[0, 0] > 0.0
        assert near_failures[(start / 15).long()].float().mean() > 0.9
    finally:
        env.close()


@pytest.mark.skipif(
    not os.path.exists(os.path.join(G1_MOTIONS_DIR, "walk1_subject1.csv")), reason="clip not downloaded"
)
def test_g1_clip_bodies_come_from_posing_the_robot():
    cfg = G1TrackingEnvCfg()
    cfg.sim.physics = G1TrackingPhysicsCfg().newton_mjwarp
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    cfg.task = g1_task("walk")
    cfg.task.events["reset_to_motion"].params["noise"] = 0.0
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        # The files hold joints only: the body positions of the clip are those of the robot posed on each frame,
        # so a reset puts every body on them, and the policy moves the 29 joints of the body.
        start = env.state.arrays[mdp2.term_state(RESET, "start")].torch.long()
        clip_body_pos = env.state.arrays[mdp2.term_state(RESET, "clip_body_pos")].torch
        expected = clip_body_pos[start] + env.scene.env_origins[:, None]
        actual = env.scene["robot"].data.body_link_pose_w.torch[..., :3]
        # A reset clamps the joints to their soft limits, which a few frames of the clip exceed.
        distance = (actual - expected).norm(dim=-1)
        assert distance.median() < 1.0e-3 and distance.max() < 0.05 and len(start.unique()) > 1
        assert env.mdp.action_dim == 29
    finally:
        env.close()
