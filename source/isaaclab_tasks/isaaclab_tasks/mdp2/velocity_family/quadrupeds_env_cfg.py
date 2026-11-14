# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity tracking of three quadrupeds in the worlds of one simulation: a Unitree Go2, an A1, and a Go1.

A world holds one of the robots, as in :mod:`..velocity_g1_go2`. The robots have the same twelve joints, so one
policy can drive all of them, and a robot that it never trained on. Each runs the task of the Go2, see
:mod:`..velocity_go2`.
"""

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG

from ..velocity_g1_go2.flat_env_cfg import robot_task
from ..velocity_go2.flat_env_cfg import DECIMATION, GROUND_MATERIAL, SIM_DT, contact_forces, robot
from ..velocity_go2.flat_env_cfg import TASK as GO2_TASK

GO1_MOTOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    actuator_effort_limit={".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 35.55},
    saturation_effort={".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 35.55},
    actuator_velocity_limit={".*_hip_joint": 30.1, ".*_thigh_joint": 30.1, ".*_calf_joint": 20.06},
    stiffness=25.0,
    damping=0.5,
    friction=0.0,
)
"""The motors of the Go1 with the limits of its asset. MDP2 does not run its actuator network, which keeps state."""

JOINTS = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")]
"""The joints that a policy of the robots reads and drives, in its order."""

JOINT_ALIASES = {}
"""Per robot, the joints that go by another name in :attr:`JOINTS`. The quadrupeds name theirs alike."""


@configclass
class QuadrupedsFlatSceneCfg(InteractiveSceneCfg):
    """A ground plane with a Unitree Go2, A1, or Go1 in each world, and contact sensors on their bodies."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=GROUND_MATERIAL,
    )
    go2: ArticulationCfg = replace(UNITREE_GO2_CFG, prim_path="{ENV_REGEX_NS}/Go2")
    a1: ArticulationCfg = replace(UNITREE_A1_CFG, prim_path="{ENV_REGEX_NS}/A1")
    go1: ArticulationCfg = replace(
        UNITREE_GO1_CFG, prim_path="{ENV_REGEX_NS}/Go1", actuators={"base_legs": GO1_MOTOR_CFG}
    )
    go2_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Go2/[^/]*", history_length=3, track_air_time=True, update_period=SIM_DT
    )
    a1_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/A1/[^/]*", history_length=3, track_air_time=True, update_period=SIM_DT
    )
    go1_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Go1/[^/]*", history_length=3, track_air_time=True, update_period=SIM_DT
    )
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))
    # A world holds one of the robots: a third of the worlds each, in this order.
    clone_cfg: CloneCfg = CloneCfg(
        clone_combinations=[InclusionSet(assets=["go2"]), InclusionSet(assets=["a1"]), InclusionSet(assets=["go1"])]
    )


@configclass
class QuadrupedsFlatPhysicsCfg(PresetCfg):
    """Newton physics presets of the task. The robots that a world does not hold sleep in it."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=150,
            nconmax=60,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            sleep_inactive_assets=True,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp


# The base of the A1 and of the Go1 is their trunk.
trunk = {"body_ids": robot.bodies("trunk")}
base_contact = GO2_TASK.terminations["base_contact"]
TRUNK_TASK = replace(
    GO2_TASK,
    terminations={
        **GO2_TASK.terminations,
        "base_contact": replace(
            base_contact, inputs={**base_contact.inputs, "body_ids": contact_forces.bodies("trunk")}
        ),
    },
    events={
        **GO2_TASK.events,
        "add_base_mass": replace(GO2_TASK.events["add_base_mass"], inputs=trunk),
        "base_com": replace(GO2_TASK.events["base_com"], inputs=trunk),
    },
)
"""The MDP of the Go2 for a robot whose base is its trunk."""

TASKS = {
    "go2": robot_task(GO2_TASK, "go2"),
    "a1": robot_task(TRUNK_TASK, "a1"),
    "go1": robot_task(TRUNK_TASK, "go1"),
}
"""The MDP of the worlds of each robot."""


@configclass
class QuadrupedsFlatEnvCfg(MDP2RLEnvCfg):
    """Configuration of the environment where a third of the worlds each hold a Go2, an A1, and a Go1."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics=QuadrupedsFlatPhysicsCfg(),
        physics_material=GROUND_MATERIAL,
        use_newton_actuators=True,
    )
    scene: QuadrupedsFlatSceneCfg = QuadrupedsFlatSceneCfg(num_envs=12288, env_spacing=2.5)
    task: list[mdp2.TaskCfg] = list(TASKS.values())
    decimation: int = DECIMATION
