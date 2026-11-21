# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MDP2 velocity tasks with the style of motion clips on Newton."""

import os

import pytest
import torch
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv

from isaaclab_tasks.mdp2.hil import mdp as hil
from isaaclab_tasks.mdp2.velocity_g1_go2.flat_env_cfg import G1_GO2_STYLE_TASKS, G1Go2FlatEnvCfg, G1Go2FlatPhysicsCfg
from isaaclab_tasks.mdp2.velocity_style.config.g1 import g1_env_cfg
from isaaclab_tasks.mdp2.velocity_style.config.go2 import go2_env_cfg

CLIPS = (*g1_env_cfg.CLIPS["motion_files"], *go2_env_cfg.CLIPS["motion_files"])
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU"),
    pytest.mark.skipif(not all(os.path.exists(clip) for clip in CLIPS), reason="clips not downloaded"),
]

WORLDS = 8
"""Worlds of the environment: the first half follow commands in the style of the clips, the rest without."""


@pytest.mark.parametrize(
    "env_cfg, features",
    [
        (g1_env_cfg.UnitreeG1FlatStyleEnvCfg, g1_env_cfg.NUM_FEATURES),
        (go2_env_cfg.UnitreeGo2FlatStyleEnvCfg, go2_env_cfg.NUM_FEATURES),
    ],
    ids=["g1", "go2"],
)
def test_style_worlds_start_on_the_clips_and_score_the_motion(env_cfg, features):
    cfg = env_cfg()
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = WORLDS
    cfg.seed = 0
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        styled = slice(0, WORLDS // 2)
        start = env.state.arrays[mdp2.term_state("events.reset_to_motion", "start")].torch[styled].long()
        clip_root_pose = env.state.arrays[mdp2.term_state("events.reset_to_motion", "clip_root_pose")].torch
        robot = env.scene["robot"].data
        assert torch.allclose(robot.root_link_pose_w.torch[styled, 2:7], clip_root_pose[start, 2:7], atol=1.0e-4)
        assert len(start.unique()) > 1

        env.step(torch.zeros(env.num_envs, env.mdp.action_dim, device=env.device))
        env.step(torch.zeros(env.num_envs, env.mdp.action_dim, device=env.device))
        assert env.num_graphs == 1
        # The discriminator scores transitions of the robot's motion features.
        current = hil.motion_features(
            robot.joint_pos.torch, robot.joint_vel.torch, robot.root_link_pose_w.torch, robot.root_com_vel_w.torch
        )
        transition = env.state.arrays[mdp2.term_state("rewards.style", "transition")].torch
        assert transition.shape[1] == 2 * features
        assert torch.allclose(transition[styled, features:], current[styled], atol=1.0e-5)
    finally:
        env.close()


def test_style_worlds_of_each_robot_start_on_its_clips_and_score_its_motion():
    """In one simulation of a G1 and a Go2, each robot's style worlds start on its clips and score its features."""
    per_task = 4
    cfg = G1Go2FlatEnvCfg()
    cfg.sim.physics = G1Go2FlatPhysicsCfg().newton_mjwarp
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = 4 * per_task
    cfg.seed = 0
    cfg.task = G1_GO2_STYLE_TASKS
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        sizes = {"g1": g1_env_cfg.NUM_FEATURES, "go2": go2_env_cfg.NUM_FEATURES}
        for index, name in enumerate(sizes):
            # The tasks of a robot take its worlds in order: the style task, then the velocity task.
            styled = slice(2 * per_task * index, 2 * per_task * index + per_task)
            reset = f"events.{name}_reset_to_motion"
            start = env.state.arrays[mdp2.term_state(reset, "start")].torch[styled].long()
            clip_root_pose = env.state.arrays[mdp2.term_state(reset, "clip_root_pose")].torch
            robot = env.scene[name].data
            assert torch.allclose(robot.root_link_pose_w.torch[styled, 2:7], clip_root_pose[start, 2:7], atol=1.0e-4)
            assert len(start.unique()) > 1

        env.step(torch.zeros(env.num_envs, env.mdp.action_dim, device=env.device))
        env.step(torch.zeros(env.num_envs, env.mdp.action_dim, device=env.device))
        assert env.num_graphs == 1
        for index, (name, size) in enumerate(sizes.items()):
            # The discriminator of a robot scores transitions of the features of that robot.
            styled = slice(2 * per_task * index, 2 * per_task * index + per_task)
            robot = env.scene[name].data
            features = hil.motion_features(
                robot.joint_pos.torch, robot.joint_vel.torch, robot.root_link_pose_w.torch, robot.root_com_vel_w.torch
            )
            transition = env.state.arrays[mdp2.term_state(f"rewards.{name}_style", "transition")].torch
            assert transition.shape[1] == 2 * size
            assert torch.allclose(transition[styled, size:], features[styled], atol=1.0e-5)
    finally:
        env.close()
