# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hybrid imitation learning with the 28-DoF humanoid: half of the worlds track a clip, the other half follow goals."""

import os

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab_tasks.mdp2.hil import mdp as hil
from isaaclab_tasks.mdp2.hil.hil_env_cfg import hil_tasks
from isaaclab_tasks.mdp2.tracking.config.humanoid.humanoid_env_cfg import (
    DECIMATION,
    MOTIONS_DIR,
    NUM_BODIES,
    SIM_DT,
    HumanoidTrackingPhysicsCfg,
    HumanoidTrackingSceneCfg,
)

NUM_FEATURES = 66
"""Size of the motion features of the humanoid: 10 values of its root and the positions and velocities of 28 joints."""


def humanoid_tasks(motion: str) -> list[mdp2.TaskCfg]:
    """Return the tracking and goal-following MDPs for the humanoid clip motion, with a new discriminator."""
    motion_file = os.path.join(MOTIONS_DIR, f"humanoid_{motion}.npz")
    return hil_tasks(motion_file, SIM_DT * DECIMATION, NUM_BODIES, hil.Discriminator(NUM_FEATURES))


@configclass
class HumanoidHILEnvCfg(MDP2RLEnvCfg):
    """Configuration of the humanoid hybrid imitation environment, for the walk clip.

    The style reward comes from the discriminator in the params of the style reward, which the learner trains.
    """

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT, render_interval=DECIMATION, physics=HumanoidTrackingPhysicsCfg(), use_newton_actuators=True
    )
    scene: HumanoidTrackingSceneCfg = HumanoidTrackingSceneCfg(num_envs=4096, env_spacing=4.0)
    task: list[mdp2.TaskCfg] = humanoid_tasks("walk")
    decimation: int = DECIMATION
