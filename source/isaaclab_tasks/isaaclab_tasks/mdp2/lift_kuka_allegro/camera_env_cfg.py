# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kuka Allegro lift with a base camera: the policy sees the object in a 64x64 RGB image, as in MDP1's
``Isaac-Lift-KukaAllegro-Camera``."""

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.lift.config.kuka_allegro.camera_cfg import BASE_CAMERA_CFG
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from .lift_env_cfg import TASK, KukaAllegroLiftEnvCfg, KukaAllegroLiftSceneCfg

base_camera = mdp2.Entity("base_camera")


@configclass
class KukaAllegroLiftCameraSceneCfg(KukaAllegroLiftSceneCfg):
    """The lift scene with MDP1's base camera. Its renderer is a preset as in MDP1: Newton's Warp renderer by
    default, ``newton_optix`` for OptiX."""

    base_camera: CameraCfg = replace(
        BASE_CAMERA_CFG, data_types=["rgb"], width=64, height=64, renderer_cfg=MultiBackendRendererCfg()
    )


CAMERA_TASK = replace(
    TASK,
    observations={
        **TASK.observations,
        # One term, so the group is the uint8 image of shape (3, 64, 64) that the CNN reads, as in MDP1.
        "base_image": {
            "object_observation_b": mdp2.Observation(mdp.image_rgb_channel_first, inputs={"image": base_camera.rgb})
        },
    },
)
"""The MDP of the lift task, with the base camera's image as the group ``base_image``."""


@configclass
class KukaAllegroLiftCameraEnvCfg(KukaAllegroLiftEnvCfg):
    """Configuration of the Kuka Allegro lift environment with a base camera."""

    scene: KukaAllegroLiftCameraSceneCfg = KukaAllegroLiftCameraSceneCfg(num_envs=4096, env_spacing=3)
    task: mdp2.TaskCfg = CAMERA_TASK
