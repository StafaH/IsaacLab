# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MDP2 Kuka Allegro lift task with a base camera on Newton."""

import pytest
import torch
from isaaclab_experimental.envs import MDP2RLEnv

from isaaclab_tasks.mdp2.lift_kuka_allegro.camera_env_cfg import KukaAllegroLiftCameraEnvCfg
from isaaclab_tasks.mdp2.lift_kuka_allegro.lift_env_cfg import KukaAllegroLiftPhysicsCfg

pytestmark = [pytest.mark.kitless, pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")]

NUM_ENVS = 4


def test_base_image_is_rendered_by_every_replayed_step():
    """The group ``base_image`` is MDP1's image, uint8 of shape (3, 64, 64), and every step of the one CUDA graph
    renders a new image of the moving arm."""
    cfg = KukaAllegroLiftCameraEnvCfg()
    cfg.sim.physics = KukaAllegroLiftPhysicsCfg().newton_mjwarp
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    cfg.task.events["reset"].params["bank_size"] = 2
    env = MDP2RLEnv(cfg)
    try:
        assert env.single_observation_space["base_image"].shape == (3, 64, 64)
        observations, _ = env.reset()
        assert observations["base_image"].shape == (NUM_ENVS, 3, 64, 64)
        assert observations["base_image"].dtype == torch.uint8
        generator = torch.Generator(device=env.device).manual_seed(0)
        images = []
        for step in range(8):
            actions = 2.0 * torch.rand(NUM_ENVS, env.mdp.action_dim, device=env.device, generator=generator) - 1.0
            observations, *_ = env.step(actions)
            if step >= 4:
                images.append(observations["base_image"].clone())
        assert env.is_captured and env.num_graphs == 1
        assert all(not torch.equal(before, after) for before, after in zip(images, images[1:]))
        # Colored: the table and the arm show, where a stale render graph draws black geometry on a white background.
        assert (images[-1].amax(dim=1) - images[-1].amin(dim=1)).amax() > 0
    finally:
        env.close()
