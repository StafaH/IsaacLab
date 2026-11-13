# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree Go2 velocity tracking in the style of clips of a dog trotting and turning.

The clips are not part of the repository, see the ``motions`` folder.
"""

import os

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab_tasks.mdp2.hil import mdp as hil
from isaaclab_tasks.mdp2.velocity_go2.flat_env_cfg import (
    DECIMATION,
    GROUND_MATERIAL,
    SIM_DT,
    TASK,
    UnitreeGo2FlatPhysicsCfg,
    UnitreeGo2FlatSceneCfg,
)

from ...style_env_cfg import style_task

MOTIONS_DIR = os.path.join(os.path.dirname(__file__), "motions")
"""Folder of the clips, which are downloaded and converted, see its README."""

FPS = 1.0 / 0.021
WINDOWS = {"trot0": (0, 950), "trot1": (0, 950), "leftturn0": (0, 950), "rightturn0": (0, 950)}
"""File, first frame, and number of frames of each clip, at the 47.6 fps of the clips."""

STEP_DT = SIM_DT * DECIMATION
CLIPS = {
    "motion_files": tuple(os.path.join(MOTIONS_DIR, f"{name}.csv") for name in WINDOWS),
    "windows": tuple(WINDOWS.values()),
    "frames": sum(round((count - 1) / (FPS * STEP_DT)) + 1 for _, count in WINDOWS.values()),
    "clips": len(WINDOWS),
    "bodies": 19,
    "fps": FPS,
    "columns": tuple(f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")),
    "ground_clearance": 0.02,
}
"""The clips, as params of the reset event that loads them."""

NUM_FEATURES = 34
"""Size of the motion features of the Go2: 10 values of its root and the positions and velocities of 12 joints."""

GO2_STYLE_TASKS = [style_task(TASK, CLIPS, hil.Discriminator(NUM_FEATURES), style_weight=2.25), TASK]
"""The MDP of the worlds that follow velocity commands in the style of the clips, and the velocity task."""


@configclass
class UnitreeGo2FlatStyleEnvCfg(MDP2RLEnvCfg):
    """Configuration of the Go2 environment where half of the worlds follow commands in the style of the clips."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics=UnitreeGo2FlatPhysicsCfg(),
        physics_material=GROUND_MATERIAL,
        use_newton_actuators=True,
    )
    scene: UnitreeGo2FlatSceneCfg = UnitreeGo2FlatSceneCfg(num_envs=4096, env_spacing=2.5)
    task: list[mdp2.TaskCfg] = GO2_STYLE_TASKS
    decimation: int = DECIMATION
