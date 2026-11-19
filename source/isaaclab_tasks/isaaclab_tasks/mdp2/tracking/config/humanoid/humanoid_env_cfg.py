# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Humanoid motion tracking: follow a motion clip, frame by frame, with a 28-DoF humanoid."""

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

from isaaclab_tasks.mdp2.tracking.tracking_env_cfg import tracking_task
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets import HUMANOID_28_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(__file__), *[".."] * 4, "contrib", "humanoid_amp", "motions")
"""The clips of the 28-DoF humanoid that ship with the AMP task. More are in ``motions`` next to this file, converted
from the same public set with ``scripts/reinforcement_learning/mdp2/convert_amp_motion.py``."""

SIM_DT = 1 / 60
DECIMATION = 2
NUM_BODIES = 15


@configclass
class HumanoidTrackingSceneCfg(InteractiveSceneCfg):
    """A 28-DoF humanoid on a ground plane, lit by the sky."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    # Position gains of the ball joints act on each of their three angles.
    robot: ArticulationCfg = replace(
        HUMANOID_28_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={
                    ".*_thigh:.*": 500.0,
                    ".*_knee": 500.0,
                    ".*_foot:.*": 400.0,
                    "torso:.*": 1000.0,
                    "head:.*": 100.0,
                    ".*_upper_arm:.*": 400.0,
                    ".*_elbow": 300.0,
                },
                damping={
                    ".*_thigh:.*": 50.0,
                    ".*_knee": 50.0,
                    ".*_foot:.*": 40.0,
                    "torso:.*": 100.0,
                    "head:.*": 10.0,
                    ".*_upper_arm:.*": 40.0,
                    ".*_elbow": 30.0,
                },
            ),
        },
    )
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))


@configclass
class HumanoidTrackingPhysicsCfg(PresetCfg):
    """Newton physics presets of the humanoid motion tracking task."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=100, nconmax=40, cone="pyramidal", impratio=1.0, integrator="implicitfast"),
        num_substeps=1,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp


def humanoid_task(motions: str, start_range: tuple[float, float] = (0.0, 0.9), adaptive: float = 0.0) -> mdp2.TaskCfg:
    """Return the MDP that tracks the humanoid clips ``motions``, such as ``"walk"`` or ``"walk,run,backflip"``."""
    local = os.path.join(os.path.dirname(__file__), "motions")
    files = [os.path.join(MOTIONS_DIR, f"humanoid_{motion}.npz") for motion in motions.split(",")]
    files = [file if os.path.exists(file) else os.path.join(local, os.path.basename(file)) for file in files]
    return tracking_task(files, SIM_DT * DECIMATION, NUM_BODIES, start_range, adaptive)


MOTIONS = ("walk", "run", "dance", "hop", "cartwheel", "backflip")
"""The clips of the humanoid."""


@configclass
class HumanoidTrackingEnvCfg(MDP2RLEnvCfg):
    """Configuration of the humanoid motion tracking environment, for the walk clip."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT, render_interval=DECIMATION, physics=HumanoidTrackingPhysicsCfg(), use_newton_actuators=True
    )
    scene: HumanoidTrackingSceneCfg = HumanoidTrackingSceneCfg(num_envs=2048, env_spacing=4.0)
    task: mdp2.TaskCfg = humanoid_task("walk")
    decimation: int = DECIMATION


@configclass
class HumanoidTrackingRunEnvCfg(HumanoidTrackingEnvCfg):
    """The environment for the run clip."""

    task: mdp2.TaskCfg = humanoid_task("run")


@configclass
class HumanoidTrackingDanceEnvCfg(HumanoidTrackingEnvCfg):
    """The environment for the dance clip."""

    task: mdp2.TaskCfg = humanoid_task("dance")


@configclass
class HumanoidTrackingBackflipEnvCfg(HumanoidTrackingEnvCfg):
    """The environment for the backflip clip."""

    task: mdp2.TaskCfg = humanoid_task("backflip")


@configclass
class HumanoidTrackingCartwheelEnvCfg(HumanoidTrackingEnvCfg):
    """The environment for the cartwheel clip."""

    task: mdp2.TaskCfg = humanoid_task("cartwheel")


@configclass
class HumanoidTrackingHopEnvCfg(HumanoidTrackingEnvCfg):
    """The environment for the hop clip."""

    task: mdp2.TaskCfg = humanoid_task("hop")


@configclass
class HumanoidTrackingAllEnvCfg(HumanoidTrackingEnvCfg):
    """The environment in which every world tracks one of all the clips, for one policy that tracks them all. Half of
    the episodes start where episodes fail."""

    task: mdp2.TaskCfg = humanoid_task(",".join(MOTIONS), adaptive=0.5)


def evaluation_task(motions: str) -> mdp2.TaskCfg:
    """Return the tracking task as it is evaluated: from the first frame of a clip, without noise."""
    task = humanoid_task(motions, start_range=(0.0, 0.0))
    task.events["reset_to_motion"].params["noise"] = 0.0
    return task


@configclass
class HumanoidTrackingLiveEvalEnvCfg(HumanoidTrackingEnvCfg):
    """The walk environment with evaluation worlds: the last 128 worlds start on the first frame without noise, so
    a learner that acts without exploration there reads its evaluation from the training simulation."""

    task: list[mdp2.TaskCfg] = [humanoid_task("walk"), evaluation_task("walk")]
    task_worlds: list[int] = [1920, 128]
