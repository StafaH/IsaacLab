# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera (vision) variant of the Franka Lift task.

A single base-mounted RGB camera is added on top of the state env config in
:mod:`.franka_env_cfg`, together with the matching image observation group. The data type
and resolution are fixed (RGB, 64x64); only the renderer backend stays ``presets=``
selectable.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.config.franka.franka_env_cfg import (
    FrankaLiftEnvCfg,
    FrankaSceneCfg,
    StateObservationCfg,
)

# the table and object workspace are shared with the other lift tasks, so the camera looks at
# the same volume from the side of the table
BASE_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Camera",
    offset=CameraCfg.OffsetCfg(
        pos=(0.57, -0.8, 0.5),
        rot=(0.6124, 0.3536, 0.3536, 0.6124),
        convention="opengl",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 2.5)),
    width=64,
    height=64,
    renderer_cfg=MultiBackendRendererCfg(),
)


@configclass
class FrankaCameraSceneCfg(FrankaSceneCfg):
    """Franka scene with a single base-mounted camera."""

    base_camera: CameraCfg = BASE_CAMERA_CFG


@configclass
class CameraObservationsCfg(StateObservationCfg):
    """State observations plus the base camera image."""

    @configclass
    class BaseImageObsCfg(ObsGroup):
        """Camera observations for the policy group."""

        object_observation_b = ObsTerm(
            func=mdp.vision_camera,
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("base_camera")},
        )

    # image groups keep the group default of no history: a stack of frames per step costs more
    # memory than the state groups' history and the state groups already carry the temporal signal
    base_image: BaseImageObsCfg = BaseImageObsCfg()


@configclass
class FrankaLiftCameraEnvCfg(FrankaLiftEnvCfg):
    scene: FrankaCameraSceneCfg = FrankaCameraSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    observations: CameraObservationsCfg = CameraObservationsCfg()
