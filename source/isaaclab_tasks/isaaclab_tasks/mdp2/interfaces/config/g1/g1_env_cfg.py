# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-interface motion tracking for the Unitree G1 on the clips of the LAFAN1 retargeting dataset.

The clips are downloaded, see ``isaaclab_tasks/mdp2/tracking/config/g1/motions``. Of every kind of motion, the clips
of one subject are held out for evaluation. The clips that fall and get up are left out.
"""

import os

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass, replace

from isaaclab_tasks.mdp2.interfaces.interfaces_env_cfg import interface_task
from isaaclab_tasks.mdp2.tracking.config.g1.g1_env_cfg import (
    DECIMATION,
    MOTIONS_DIR,
    NUM_BODIES,
    SIM_DT,
    G1TrackingSceneCfg,
)
from isaaclab_tasks.utils import PresetCfg

robot = mdp2.Entity("robot")

TRAIN_CLIPS = (
    *(f"dance1_subject{k}" for k in (1, 2)),
    *(f"dance2_subject{k}" for k in (1, 2, 3, 4)),
    *(f"fight1_subject{k}" for k in (2, 3)),
    "fightAndSports1_subject1",
    *(f"jumps1_subject{k}" for k in (1, 2)),
    *(f"run1_subject{k}" for k in (2, 5)),
    "run2_subject1",
    "sprint1_subject2",
    *(f"walk1_subject{k}" for k in (1, 2, 5)),
    *(f"walk2_subject{k}" for k in (1, 3)),
    *(f"walk3_subject{k}" for k in (1, 2, 3, 4)),
    "walk4_subject1",
)
TEST_CLIPS = (
    "dance1_subject3",
    "dance2_subject5",
    "fight1_subject5",
    "fightAndSports1_subject4",
    "jumps1_subject5",
    "run2_subject4",
    "sprint1_subject4",
    "walk2_subject4",
    "walk3_subject5",
)

BODY_GROUPS = {
    "root": robot.bodies("pelvis"),
    "head": robot.bodies("torso_link"),
    "hands": robot.bodies(".*_wrist_yaw_link"),
    "feet": robot.bodies(".*_ankle_roll_link"),
    "limbs": robot.bodies(".*_knee_link", ".*_elbow_link"),
}
JOINT_GROUPS = {
    "upper_joints": robot.joints("waist_.*", ".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"),
    "lower_joints": robot.joints(".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"),
}
INTERFACES = {
    "full": (*BODY_GROUPS, *JOINT_GROUPS),
    "key_points": tuple(BODY_GROUPS),
    "three_point": ("head", "hands"),
    "upper_body": ("root", "upper_joints"),
}
"""The target groups of every interface. The robot has no head body, so its torso stands for the head."""


def frames(name: str) -> int:
    """Return the number of frames of a clip, one per row of its file."""
    with open(os.path.join(MOTIONS_DIR, f"{name}.csv")) as file:
        return sum(1 for _ in file)


def g1_interfaces_task(train: tuple[str, ...] = TRAIN_CLIPS, test: tuple[str, ...] = TEST_CLIPS) -> mdp2.TaskCfg:
    """Return the MDP that tracks the ``train`` clips through the interfaces, with the ``test`` clips for evaluation."""
    clips = (*train, *test)
    task = interface_task(
        [os.path.join(MOTIONS_DIR, f"{name}.csv") for name in clips],
        [frames(name) for name in clips],
        SIM_DT * DECIMATION,
        NUM_BODIES,
        BODY_GROUPS,
        JOINT_GROUPS,
        INTERFACES,
        train_clips=len(train),
    )
    # The policy moves the 29 joints of the body; the fingers keep their default pose.
    action = replace(task.actions["joint_pos"], subset=robot.joints("(?!.*_hand_).*"))
    return replace(task, actions={"joint_pos": action})


@configclass
class G1InterfacesPhysicsCfg(PresetCfg):
    """Newton physics presets of the task, with room for the contacts of hands and knees on the ground."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=300, nconmax=120, cone="pyramidal", impratio=1.0, integrator="implicitfast"),
        num_substeps=1,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class G1InterfacesEnvCfg(MDP2RLEnvCfg):
    """Configuration of the G1 multi-interface tracking environment."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT, render_interval=DECIMATION, physics=G1InterfacesPhysicsCfg(), use_newton_actuators=True
    )
    scene: G1TrackingSceneCfg = G1TrackingSceneCfg(num_envs=8192, env_spacing=4.0)
    task: mdp2.TaskCfg = None
    decimation: int = DECIMATION

    def __post_init__(self):
        # Built here, not at import: the task reads the lengths of the downloaded clips.
        if self.task is None:
            self.task = g1_interfaces_task()
