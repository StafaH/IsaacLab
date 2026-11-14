# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree H1 velocity tracking: follow a commanded base velocity and heading on flat ground.

The task is that of the Unitree G1, see :mod:`..velocity_g1`, with the rewards bound to the joints and the feet of
the H1.
"""

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.unitree import H1_MINIMAL_CFG

from ..velocity_g1.flat_env_cfg import DECIMATION, GROUND_MATERIAL, SIM_DT, contact_forces, joint_deviation, robot
from ..velocity_g1.flat_env_cfg import TASK as G1_TASK

legs = robot.joints(".*_hip_.*", ".*_knee")
rewards = G1_TASK.rewards


@configclass
class UnitreeH1FlatSceneCfg(InteractiveSceneCfg):
    """A Unitree H1 on a ground plane, with contact sensors on all its bodies, lit by the sky."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=GROUND_MATERIAL,
    )
    robot: ArticulationCfg = replace(H1_MINIMAL_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True, update_period=SIM_DT
    )
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))


@configclass
class UnitreeH1FlatPhysicsCfg(PresetCfg):
    """Newton physics presets of the Unitree H1 flat velocity task."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=150, nconmax=60, cone="pyramidal", impratio=1.0, integrator="implicitfast"),
        num_substeps=2,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp


TASK = replace(
    G1_TASK,
    rewards={
        **rewards,
        "base_height_l2": replace(rewards["base_height_l2"], params={"target_height": 0.98}),
        "dof_torques_l2": replace(
            rewards["dof_torques_l2"], inputs={"applied_effort": robot.applied_effort, "joint_ids": legs}
        ),
        "dof_acc_l2": replace(rewards["dof_acc_l2"], inputs={"joint_acc": robot.joint_acc, "joint_ids": legs}),
        "feet_air_time": replace(
            rewards["feet_air_time"],
            inputs={**rewards["feet_air_time"].inputs, "body_ids": contact_forces.bodies(".*ankle_link")},
        ),
        "dof_pos_limits": replace(
            rewards["dof_pos_limits"],
            inputs={**rewards["dof_pos_limits"].inputs, "joint_ids": robot.joints(".*_ankle")},
        ),
        "joint_deviation_hip": joint_deviation(".*_hip_yaw", ".*_hip_roll", weight=-0.1),
        "joint_deviation_arms": joint_deviation(".*_shoulder_.*", ".*_elbow", weight=-0.1),
        "joint_deviation_fingers": None,
        "joint_deviation_torso": joint_deviation("torso", weight=-0.1),
    },
)
"""The MDP of the task."""


@configclass
class UnitreeH1FlatEnvCfg(MDP2RLEnvCfg):
    """Configuration of the Unitree H1 flat velocity environment."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics=UnitreeH1FlatPhysicsCfg(),
        physics_material=GROUND_MATERIAL,
        use_newton_actuators=True,
    )
    scene: UnitreeH1FlatSceneCfg = UnitreeH1FlatSceneCfg(num_envs=4096, env_spacing=2.5)
    task: mdp2.TaskCfg = TASK
    decimation: int = DECIMATION
