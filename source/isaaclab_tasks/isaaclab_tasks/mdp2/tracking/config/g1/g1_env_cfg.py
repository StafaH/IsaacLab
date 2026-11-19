# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 motion tracking: the 29-DoF G1 follows retargeted motion capture, frame by frame."""

import os

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.mdp2.tracking import mdp as tracking
from isaaclab_tasks.mdp2.tracking.tracking_env_cfg import tracking_task
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(__file__), "motions")
"""Folder of the retargeting files, which are downloaded, see its README."""

SIM_DT = 1 / 60
DECIMATION = 2
NUM_BODIES = 51
WINDOWS = {
    "walk": ("walk1_subject1", 100, 300),
    "dance": ("dance1_subject2", 100, 600),
    "jumps": ("jumps1_subject1", 300, 300),
    "fight": ("fight1_subject2", 300, 300),
}
"""File, first frame, and number of frames of each clip, at 30 fps."""

robot = mdp2.Entity("robot")


@configclass
class G1TrackingSceneCfg(InteractiveSceneCfg):
    """A Unitree G1 on a ground plane, lit by the sky."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    # Position gains for whole-body motion; the asset's own are for a fixed base.
    robot: ArticulationCfg = replace(
        G1_29DOF_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={
                    ".*_hip_.*": 100.0,
                    ".*_knee_joint": 150.0,
                    ".*_ankle_.*": 40.0,
                    "waist_.*": 150.0,
                    ".*_shoulder_.*": 40.0,
                    ".*_elbow_joint": 40.0,
                    ".*_wrist_.*": 20.0,
                    ".*_hand_.*": 20.0,
                },
                damping={
                    ".*_hip_.*": 4.0,
                    ".*_knee_joint": 6.0,
                    ".*_ankle_.*": 2.0,
                    "waist_.*": 6.0,
                    ".*_shoulder_.*": 2.0,
                    ".*_elbow_joint": 2.0,
                    ".*_wrist_.*": 1.0,
                    ".*_hand_.*": 1.0,
                },
            ),
        },
    )
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))


@configclass
class G1TrackingPhysicsCfg(PresetCfg):
    """Newton physics presets of the G1 motion tracking task."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=150, nconmax=60, cone="pyramidal", impratio=1.0, integrator="implicitfast"),
        num_substeps=1,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp


def g1_task(motions: str, start_range: tuple[float, float] = (0.0, 0.9), adaptive: float = 0.0) -> mdp2.TaskCfg:
    """Return the MDP that tracks the G1 clips ``motions``, such as ``"walk"`` or ``"walk,dance"``."""
    names = motions.split(",")
    files = [os.path.join(MOTIONS_DIR, f"{WINDOWS[name][0]}.csv") for name in names]
    lengths = [WINDOWS[name][2] for name in names]
    task = tracking_task(files, SIM_DT * DECIMATION, NUM_BODIES, start_range, adaptive, lengths)
    reset = task.events["reset_to_motion"]
    term = tracking.reset_to_csv_motion if adaptive == 0.0 else reset.term
    reset = replace(
        reset,
        term=term,
        inputs={**reset.inputs, "body_link_pose_w": robot.body_link_pose_w},
        params={**reset.params, "windows": tuple(WINDOWS[name][1:] for name in names)},
    )
    # The policy moves the 29 joints of the body; the fingers hold the pose of the clip, which is zero.
    action = replace(task.actions["joint_pos"], subset=robot.joints("(?!.*_hand_).*"))
    return replace(task, actions={"joint_pos": action}, events={**task.events, "reset_to_motion": reset})


@configclass
class G1TrackingEnvCfg(MDP2RLEnvCfg):
    """Configuration of the G1 motion tracking environment, for the walk clip."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT, render_interval=DECIMATION, physics=G1TrackingPhysicsCfg(), use_newton_actuators=True
    )
    scene: G1TrackingSceneCfg = G1TrackingSceneCfg(num_envs=2048, env_spacing=4.0)
    task: mdp2.TaskCfg = g1_task("walk")
    decimation: int = DECIMATION
