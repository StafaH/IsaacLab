# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MDP2 tasks that split their worlds between task configs: recovery, distillation, skills, and
manipulation."""

import pytest
import torch
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv

from isaaclab_tasks.mdp2.distillation.config.go2.go2_env_cfg import HISTORY_LENGTH, UnitreeGo2DistillationEnvCfg
from isaaclab_tasks.mdp2.lift_kuka_allegro.lift_env_cfg import KukaAllegroLiftPhysicsCfg
from isaaclab_tasks.mdp2.manipulation.config.kuka_allegro.kuka_allegro_env_cfg import (
    SKILLS,
    KukaAllegroManipulationEnvCfg,
)
from isaaclab_tasks.mdp2.recovery.config.humanoid.humanoid_env_cfg import HumanoidRecoveryEnvCfg
from isaaclab_tasks.mdp2.skills.config.go2.go2_env_cfg import UnitreeGo2SkillsEnvCfg
from isaaclab_tasks.mdp2.skills.skills_env_cfg import skill_tasks
from isaaclab_tasks.mdp2.tracking.config.humanoid.humanoid_env_cfg import HumanoidTrackingPhysicsCfg
from isaaclab_tasks.mdp2.velocity_go2.flat_env_cfg import UnitreeGo2FlatPhysicsCfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")

NUM_ENVS = 8


def make_env(cfg, physics) -> MDP2RLEnv:
    cfg.sim.physics = physics
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    return MDP2RLEnv(cfg)


def step(env: MDP2RLEnv, count: int) -> dict[str, torch.Tensor]:
    env.reset()
    for _ in range(count):
        observations = env.step(torch.zeros(NUM_ENVS, env.mdp.action_dim, device=env.device))[0]
    assert env.num_graphs == 1
    return observations


def test_recovery_pushes_only_the_pushed_worlds():
    env = make_env(HumanoidRecoveryEnvCfg(), HumanoidTrackingPhysicsCfg().newton_mjwarp)
    try:
        ops = {op.name: op for op in env.mdp.term_ops}
        assert ops["events.push"].worlds == tuple(range(NUM_ENVS // 2, NUM_ENVS))
        assert ops["rewards.body_pos"].worlds is None
        step(env, 3)
    finally:
        env.close()


def test_distillation_computes_student_observations_only_in_student_worlds():
    env = make_env(UnitreeGo2DistillationEnvCfg(), UnitreeGo2FlatPhysicsCfg().newton_mjwarp)
    try:
        dims = env.mdp.observation_dims
        # The student sees the privileged observations without the base velocity, over its history.
        assert dims["student"] == HISTORY_LENGTH * (dims["privileged"] - 3)
        observations = step(env, 3)
        half = NUM_ENVS // 2
        assert (observations["student"][:half] == 0.0).all() and (observations["student"][half:] != 0.0).any()
        assert (observations["privileged"] != 0.0).any(dim=1).all()
    finally:
        env.close()


def test_skills_tell_the_policy_which_skill_each_world_practices():
    env = make_env(UnitreeGo2SkillsEnvCfg(), UnitreeGo2FlatPhysicsCfg().newton_mjwarp)
    try:
        ops = {op.name: op for op in env.mdp.term_ops}
        quarter = NUM_ENVS // 4
        # Each skill's rewards run in its own worlds only; the observation of the skill runs everywhere.
        assert ops["rewards.track_lin_vel_xy_exp"].worlds == tuple(range(quarter))
        assert ops["rewards.walk"].worlds == tuple(range(2 * quarter, 3 * quarter))
        assert ops["observations.policy.skill"].worlds is None
        observations = step(env, 3)
        skill = env.state.arrays[mdp2.commands.skill].torch
        expected = torch.eye(4, device=env.device).repeat_interleave(quarter, dim=0)
        assert torch.equal(skill, expected) and torch.equal(observations["policy"][:, -4:], expected)
    finally:
        env.close()


def test_manipulation_tracks_the_orientation_only_in_the_reorienting_worlds():
    num_envs = 12
    cfg = KukaAllegroManipulationEnvCfg()
    cfg.task = skill_tasks(SKILLS)
    cfg.sim.physics = KukaAllegroLiftPhysicsCfg().newton_mjwarp
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = "cuda:0", num_envs, 0
    for task in cfg.task:
        task.events["reset"].params["bank_size"] = 2
    env = MDP2RLEnv(cfg)
    try:
        ops = {op.name: op for op in env.mdp.term_ops}
        third = num_envs // 3
        assert ops["rewards.orientation_tracking"].worlds == tuple(range(2 * third, num_envs))
        env.reset()
        for _ in range(3):
            env.step(torch.zeros(num_envs, env.mdp.action_dim, device=env.device))
        assert env.num_graphs == 1
        # Placing commands positions just above the table; lifting and reorienting command them higher.
        height = env.state.arrays[mdp2.commands.object_pose].torch[:, 2]
        assert (height[third : 2 * third] <= 0.42).all() and (height[:third] >= 0.55).all()
        error = env.state.arrays[mdp2.quantities.orientation_error].torch
        assert (error[2 * third :] > 0.0).any() and (error[: 2 * third] == 0.0).all()
    finally:
        env.close()
