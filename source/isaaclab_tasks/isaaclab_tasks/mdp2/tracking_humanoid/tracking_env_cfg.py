# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Humanoid motion tracking: follow a motion clip, frame by frame, with a 28-DoF humanoid."""

import os

import numpy as np
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets import HUMANOID_28_CFG

from . import mdp as tracking

MOTIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "contrib", "humanoid_amp", "motions")
"""The clips of the 28-DoF humanoid that ship with the AMP task."""

SIM_DT = 1 / 60
DECIMATION = 2
NUM_BODIES = 15

robot = mdp2.Entity("robot")
scene = mdp2.Entity("scene")
joints = robot.joints(".*")

root_state = {"root_link_pose_w": robot.root_link_pose_w, "root_com_vel_w": robot.root_com_vel_w}
RESET = "events.reset_to_motion"
progress = {"start": mdp2.term_state(RESET, "start"), "episode_length": mdp2.episode_length}
clip_joint_pos = {"clip_joint_pos": mdp2.term_state(RESET, "clip_joint_pos"), **progress}
clip_body_pos = {
    "clip_body_pos": mdp2.term_state(RESET, "clip_body_pos"),
    "env_origins": scene.env_origins,
    **progress,
}


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


def tracking_task(motion: str, start_range: tuple[float, float] = (0.0, 0.9)) -> mdp2.TaskCfg:
    """Return the MDP that tracks the clip ``motion``, such as ``"walk"``, from starts in a fraction of the clip."""
    motion_file = os.path.join(MOTIONS_DIR, f"humanoid_{motion}.npz")
    clip = np.load(motion_file)
    frames = len(clip["dof_positions"][:: round(float(clip["fps"]) * SIM_DT * DECIMATION)])

    return mdp2.TaskCfg(
        # The policy corrects the pose of the frame it tracks next.
        actions={
            "joint_pos": mdp2.DirectAction(
                robot.joint_pos_target, scale=0.25, offset=mdp2.quantities.reference_joint_pos
            ),
        },
        quantities={
            "reference_joint_pos": mdp2.Quantity(
                tracking.reference_joint_pos, inputs={**clip_joint_pos, "joint_ids": joints}
            ),
            "joint_pos_error": mdp2.Quantity(
                tracking.joint_pos_error, inputs={"joint_pos": robot.joint_pos, **clip_joint_pos}, log=True
            ),
            "joint_vel_error": mdp2.Quantity(
                tracking.joint_vel_error,
                inputs={
                    "joint_vel": robot.joint_vel,
                    "clip_joint_vel": mdp2.term_state(RESET, "clip_joint_vel"),
                    **progress,
                },
            ),
            "body_pos_error": mdp2.Quantity(
                tracking.body_pos_error, inputs={"body_link_pose_w": robot.body_link_pose_w, **clip_body_pos}, log=True
            ),
            "root_vel_error": mdp2.Quantity(
                tracking.root_vel_error,
                inputs={
                    "root_com_vel_w": robot.root_com_vel_w,
                    "clip_root_vel": mdp2.term_state(RESET, "clip_root_vel"),
                    **progress,
                },
                params={"angular_scale": 0.1},
            ),
        },
        observations={
            "policy": mdp2.ObservationGroup(
                root_height=mdp2.Observation(tracking.root_height, inputs={"root_link_pose_w": robot.root_link_pose_w}),
                base_lin_vel=mdp2.Observation(mdp.base_lin_vel, inputs=root_state),
                base_ang_vel=mdp2.Observation(mdp.base_ang_vel, inputs=root_state),
                projected_gravity=mdp2.Observation(
                    mdp.projected_gravity,
                    inputs={"root_link_pose_w": robot.root_link_pose_w, "gravity_w": scene.gravity_w},
                ),
                joint_pos=mdp2.Observation(mdp.joint_pos, inputs={"joint_pos": robot.joint_pos, "joint_ids": joints}),
                joint_vel=mdp2.Observation(
                    mdp.joint_vel, inputs={"joint_vel": robot.joint_vel, "joint_ids": joints}, scale=0.1
                ),
                body_pos=mdp2.Observation(
                    tracking.body_pos_b,
                    inputs={
                        "body_link_pose_w": robot.body_link_pose_w,
                        "root_link_pose_w": robot.root_link_pose_w,
                    },
                ),
                reference_joint_pos=mdp2.Observation(
                    mdp.generated_commands, inputs={"command": mdp2.quantities.reference_joint_pos}
                ),
                reference_body_pos=mdp2.Observation(
                    tracking.reference_body_pos_b,
                    inputs={"root_link_pose_w": robot.root_link_pose_w, **clip_body_pos},
                    params={"bodies": NUM_BODIES},
                ),
                actions=mdp2.Observation(mdp.last_action, inputs={"actions": mdp2.last_action}),
            )
        },
        rewards={
            "joint_pos": mdp2.Reward(
                tracking.exp_kernel,
                inputs={"value": mdp2.quantities.joint_pos_error},
                params={"scale": 25.0},
                weight=15.0,
            ),
            "joint_vel": mdp2.Reward(
                tracking.exp_kernel,
                inputs={"value": mdp2.quantities.joint_vel_error},
                params={"scale": 0.05},
                weight=3.0,
            ),
            "body_pos": mdp2.Reward(
                tracking.exp_kernel,
                inputs={"value": mdp2.quantities.body_pos_error},
                params={"scale": 100.0},
                weight=9.0,
            ),
            "root_vel": mdp2.Reward(
                tracking.exp_kernel,
                inputs={"value": mdp2.quantities.root_vel_error},
                params={"scale": 1.0},
                weight=3.0,
            ),
            "action_rate_l2": mdp2.Reward(mdp.action_rate_l2, inputs={"actions": mdp2.last_action}, weight=-0.03),
        },
        terminations={
            "time_out": mdp2.TimeLimit(episode_length_s=frames * SIM_DT * DECIMATION),
            "motion_ended": mdp2.Termination(tracking.motion_ended, inputs=clip_joint_pos, time_out=True),
            # A mean distance of 0.25 m to the tracked bodies.
            "tracking_lost": mdp2.Termination(
                tracking.is_above, inputs={"value": mdp2.quantities.body_pos_error}, params={"threshold": 0.0625}
            ),
        },
        events={
            "reset_to_motion": mdp2.Event(
                tracking.reset_to_motion,
                inputs={"env_origins": scene.env_origins, "soft_joint_pos_limits": robot.soft_joint_pos_limits},
                outputs={"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel, **root_state},
                params={
                    "motion_file": motion_file,
                    "frames": frames,
                    "bodies": NUM_BODIES,
                    "start_range": start_range,
                    "noise": 0.02,
                },
            ),
        },
    )


@configclass
class HumanoidTrackingEnvCfg(MDP2RLEnvCfg):
    """Configuration of the humanoid motion tracking environment, for the walk clip."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT, render_interval=DECIMATION, physics=HumanoidTrackingPhysicsCfg(), use_newton_actuators=True
    )
    scene: HumanoidTrackingSceneCfg = HumanoidTrackingSceneCfg(num_envs=2048, env_spacing=4.0)
    task: mdp2.TaskCfg = tracking_task("walk")
    decimation: int = DECIMATION


@configclass
class HumanoidTrackingRunEnvCfg(HumanoidTrackingEnvCfg):
    """The environment for the run clip."""

    task: mdp2.TaskCfg = tracking_task("run")


@configclass
class HumanoidTrackingDanceEnvCfg(HumanoidTrackingEnvCfg):
    """The environment for the dance clip."""

    task: mdp2.TaskCfg = tracking_task("dance")
