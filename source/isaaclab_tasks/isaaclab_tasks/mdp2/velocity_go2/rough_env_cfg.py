# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree Go2 velocity tracking on rough terrain: stairs, boxes, rough ground, and slopes, each at ten levels."""

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import PresetCfg

from .flat_env_cfg import (
    DECIMATION,
    GROUND_MATERIAL,
    SIM_DT,
    TASK,
    UnitreeGo2FlatPhysicsCfg,
    UnitreeGo2FlatSceneCfg,
    robot,
    scene,
)

FLAT_PHYSICS = UnitreeGo2FlatPhysicsCfg().newton_mjwarp
height_scanner = mdp2.Entity("height_scanner")

# The terrains are scaled down for the small robot. With a curriculum, a row of the grid is a level of difficulty.
SUB_TERRAINS = ROUGH_TERRAINS_CFG.sub_terrains
TERRAINS = replace(
    ROUGH_TERRAINS_CFG,
    curriculum=True,
    sub_terrains={
        **SUB_TERRAINS,
        "pyramid_stairs": replace(SUB_TERRAINS["pyramid_stairs"], step_height_range=(0.025, 0.12)),
        "pyramid_stairs_inv": replace(SUB_TERRAINS["pyramid_stairs_inv"], step_height_range=(0.025, 0.12)),
        "boxes": replace(SUB_TERRAINS["boxes"], grid_height_range=(0.025, 0.1)),
        "random_rough": replace(SUB_TERRAINS["random_rough"], noise_range=(0.01, 0.06), noise_step=0.01),
    },
)


@configclass
class UnitreeGo2RoughSceneCfg(UnitreeGo2FlatSceneCfg):
    """A Unitree Go2 on generated terrain, with a scanner of the terrain height around its base."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAINS,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=GROUND_MATERIAL,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        update_period=DECIMATION * SIM_DT,
    )


@configclass
class UnitreeGo2RoughPhysicsCfg(PresetCfg):
    """Newton physics presets of the Unitree Go2 rough velocity task, with room for the contacts of rough terrain."""

    newton_mjwarp: NewtonCfg = replace(
        FLAT_PHYSICS, solver_cfg=replace(FLAT_PHYSICS.solver_cfg, njmax=1000, nconmax=300)
    )
    default: NewtonCfg = newton_mjwarp


ROUGH_TASK = replace(
    TASK,
    observations={
        "policy": mdp2.ObservationGroup(
            **TASK.observations["policy"].terms,
            height_scan=mdp2.Observation(
                mdp.height_scan,
                inputs={"pos_w": height_scanner.pos_w, "ray_hits_w": height_scanner.ray_hits_w},
                params={"offset": 0.5},
                noise=mdp2.UniformNoise(-0.1, 0.1),
                clip=(-1.0, 1.0),
            ),
        )
    },
    events={
        # Before the reset of the base, which reads the origin.
        "terrain_origin": mdp2.Event(mdp.move_env_origin, outputs={"env_origins": scene.env_origins}),
        **TASK.events,
    },
    rewards={
        **TASK.rewards,
        "feet_air_time": replace(TASK.rewards["feet_air_time"], weight=0.01),
        # On slopes and stairs the base follows the terrain, not gravity.
        "flat_orientation_l2": None,
        "base_height_l2": mdp2.Reward(
            mdp.base_height_over_terrain_l2,
            inputs={"root_link_pose_w": robot.root_link_pose_w, "ray_hits_w": height_scanner.ray_hits_w},
            params={"target_height": 0.4},
            weight=-30.0,
        ),
    },
)
"""The flat task with an observation of the terrain height around the base, the base height above the terrain,
and an origin that a learner may move."""


@configclass
class UnitreeGo2RoughEnvCfg(MDP2RLEnvCfg):
    """Configuration of the Unitree Go2 rough velocity environment.

    The terrain importer starts every world on a type of terrain by its index and on a random level up to
    ``max_init_terrain_level``. A learner that chooses the terrain of a world writes the origin of a tile,
    ``scene.terrain.terrain_origins[level, type]``, into the ``origin`` state of the ``terrain_origin`` event, and
    the world moves there when it next resets.
    """

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics=UnitreeGo2RoughPhysicsCfg(),
        physics_material=GROUND_MATERIAL,
        use_newton_actuators=True,
    )
    scene: UnitreeGo2RoughSceneCfg = UnitreeGo2RoughSceneCfg(num_envs=4096, env_spacing=2.5)
    task: mdp2.TaskCfg = ROUGH_TASK
    decimation: int = DECIMATION
