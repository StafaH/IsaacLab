# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity tracking of two humanoids in the worlds of one simulation: a Unitree G1 and a Unitree H1.

A world holds one of the robots, as in :mod:`..velocity_g1_go2`. The H1 has a subset of the joints of the G1, under
other names, so one policy can drive both through the joints of the G1. Each runs its own task, see
:mod:`..velocity_g1` and :mod:`..velocity_h1`.
"""

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG, H1_MINIMAL_CFG

from ..velocity_g1.flat_env_cfg import DECIMATION, GROUND_MATERIAL, SIM_DT
from ..velocity_g1.flat_env_cfg import TASK as G1_TASK
from ..velocity_g1_go2.flat_env_cfg import G1Go2FlatPhysicsCfg, robot_task
from ..velocity_h1.flat_env_cfg import TASK as H1_TASK

SIDES = ("left", "right")
LEG_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll")
HAND_JOINTS = ("zero", "one", "two", "three", "four", "five", "six")

JOINTS = [
    *(f"{side}_{joint}_joint" for joint in LEG_JOINTS for side in SIDES),
    "torso_joint",
    *(f"{side}_{joint}_joint" for joint in ARM_JOINTS for side in SIDES),
    *(f"{side}_{joint}_joint" for joint in HAND_JOINTS for side in SIDES),
]
"""The joints that a policy of the robots reads and drives, in its order: those of the G1."""

H1_JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "shoulder_pitch", "shoulder_roll", "shoulder_yaw")
JOINT_ALIASES = {
    "h1": {
        "torso": "torso_joint",
        **{f"{side}_{joint}": f"{side}_{joint}_joint" for joint in H1_JOINTS for side in SIDES},
        **{f"{side}_ankle": f"{side}_ankle_pitch_joint" for side in SIDES},
        **{f"{side}_elbow": f"{side}_elbow_pitch_joint" for side in SIDES},
    }
}
"""Per robot, the joints that go by another name in :attr:`JOINTS`."""


@configclass
class HumanoidsFlatSceneCfg(InteractiveSceneCfg):
    """A ground plane with a Unitree G1 or a Unitree H1 in each world, and contact sensors on their bodies."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=GROUND_MATERIAL,
    )
    g1: ArticulationCfg = replace(G1_MINIMAL_CFG, prim_path="{ENV_REGEX_NS}/G1")
    h1: ArticulationCfg = replace(H1_MINIMAL_CFG, prim_path="{ENV_REGEX_NS}/H1")
    g1_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/G1/[^/]*", history_length=3, track_air_time=True, update_period=SIM_DT
    )
    h1_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/H1/[^/]*", history_length=3, track_air_time=True, update_period=SIM_DT
    )
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))
    # A world holds one of the robots: the first half of the worlds the G1, the second half the H1.
    clone_cfg: CloneCfg = CloneCfg(clone_combinations=[InclusionSet(assets=["g1"]), InclusionSet(assets=["h1"])])


TASKS = {"g1": robot_task(G1_TASK, "g1"), "h1": robot_task(H1_TASK, "h1")}
"""The MDP of the worlds of each robot."""


@configclass
class HumanoidsFlatEnvCfg(MDP2RLEnvCfg):
    """Configuration of the environment where half of the worlds hold a G1 and the other half an H1."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics=G1Go2FlatPhysicsCfg(),
        physics_material=GROUND_MATERIAL,
        use_newton_actuators=True,
    )
    scene: HumanoidsFlatSceneCfg = HumanoidsFlatSceneCfg(num_envs=8192, env_spacing=2.5)
    task: list[mdp2.TaskCfg] = list(TASKS.values())
    decimation: int = DECIMATION
