# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 velocity tracking in the style of clips of a person walking.

The clips are those of the motion tracking task, see its ``motions`` folder.
"""

import os

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab_tasks.mdp2.hil import mdp as hil
from isaaclab_tasks.mdp2.tracking.config.g1.g1_env_cfg import MOTIONS_DIR
from isaaclab_tasks.mdp2.tracking.mdp.events import CSV_FPS, CSV_JOINTS
from isaaclab_tasks.mdp2.velocity_g1.flat_env_cfg import (
    DECIMATION,
    GROUND_MATERIAL,
    SIM_DT,
    TASK,
    UnitreeG1FlatPhysicsCfg,
    UnitreeG1FlatSceneCfg,
)

from ...style_env_cfg import style_task

WINDOWS = {"walk1_subject2": (200, 3001), "walk1_subject5": (200, 3001)}
"""File, first frame, and number of frames of each clip, at 30 fps."""

# The clips are for the G1 with 29 joints. This G1 has the same legs and shoulders, one waist joint, and elbows
# with two joints; the joints it lacks are left out, and its fingers stay in their default pose.
JOINTS = {name: name for name in CSV_JOINTS if "_hip_" in name or "_knee_" in name or "_ankle_" in name}
JOINTS |= {name: name for name in CSV_JOINTS if "_shoulder_" in name}
JOINTS |= {"waist_yaw_joint": "torso_joint"}
JOINTS |= {f"{side}_elbow_joint": f"{side}_elbow_pitch_joint" for side in ("left", "right")}
JOINTS |= {f"{side}_wrist_roll_joint": f"{side}_elbow_roll_joint" for side in ("left", "right")}

STEP_DT = SIM_DT * DECIMATION
CLIPS = {
    "motion_files": tuple(os.path.join(MOTIONS_DIR, f"{name}.csv") for name in WINDOWS),
    "windows": tuple(WINDOWS.values()),
    "frames": sum(round((count - 1) / (CSV_FPS * STEP_DT)) + 1 for _, count in WINDOWS.values()),
    "clips": len(WINDOWS),
    "bodies": 44,
    "joints": tuple(JOINTS.items()),
    # This G1 has shorter legs than the one of the clips: lower the root for its feet to reach the ground.
    "height_offset": -0.07,
    "ground_clearance": 0.02,
}
"""The clips, as params of the reset event that loads them."""

NUM_FEATURES = 84
"""Size of the motion features of the G1: 10 values of its root and the positions and velocities of 37 joints."""

G1_STYLE_TASKS = [style_task(TASK, CLIPS, hil.Discriminator(NUM_FEATURES), style_weight=4.0), TASK]
"""The MDP of the worlds that follow velocity commands in the style of the clips, and the velocity task."""


@configclass
class UnitreeG1FlatStyleEnvCfg(MDP2RLEnvCfg):
    """Configuration of the G1 environment where half of the worlds follow commands in the style of the clips."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics=UnitreeG1FlatPhysicsCfg(),
        physics_material=GROUND_MATERIAL,
        use_newton_actuators=True,
    )
    scene: UnitreeG1FlatSceneCfg = UnitreeG1FlatSceneCfg(num_envs=4096, env_spacing=2.5)
    task: list[mdp2.TaskCfg] = G1_STYLE_TASKS
    decimation: int = DECIMATION
