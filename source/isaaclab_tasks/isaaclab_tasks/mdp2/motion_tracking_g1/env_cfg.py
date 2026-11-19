# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from HybridRobotics/whole_body_tracking (MIT).
# See docs/licenses/dependencies/motion-tracking-license.txt.

"""Unitree G1 whole-body motion tracking: the MDP of the MDP1 task ``Isaac-Motion-Tracking-G1``, in MDP2.

The scene, robot, physics, clip, and agent are the MDP1 task's own configs.
"""

import numpy as np
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_tasks.core.motion_tracking.config.g1.env_cfg import (
    G1_ACTION_SCALE,
    G1CommandsCfg,
    G1SceneCfg,
)
from isaaclab_tasks.core.motion_tracking.config.g1.env_cfg import (
    G1MotionTrackingEnvCfg as MDP1G1MotionTrackingEnvCfg,
)
from isaaclab_tasks.utils import PresetCfg

from . import mdp as tracking

MDP1_CFG = MDP1G1MotionTrackingEnvCfg()
MOTION = G1CommandsCfg().motion
EE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")

robot = mdp2.Entity("robot")
self_collision = mdp2.Entity("self_collision")
scene = mdp2.Entity("scene")
joints = robot.joints(".*")
anchor = robot.bodies(MOTION.anchor_body_name)
tracked_bodies = robot.bodies(*MOTION.body_names)

root_state = {"root_link_pose_w": robot.root_link_pose_w, "root_com_vel_w": robot.root_com_vel_w}
clip = tracking.clip_names
# Terminations and rewards track the frame the episode reached; observations the next one, as MDP1 observes after
# its command advances.
tracked = {"motion_frame": mdp2.quantities.motion_frame, "body_link_pose_w": robot.body_link_pose_w}
observed = {"motion_frame": mdp2.quantities.next_motion_frame, "body_link_pose_w": robot.body_link_pose_w}
anchor_pos = {"env_origins": scene.env_origins, "clip_body_pos": clip["clip_body_pos"], "anchor_ids": anchor}
anchor_ori = {"clip_body_quat": clip["clip_body_quat"], "anchor_ids": anchor}


def motion_tracking_task(motion_file: str = MOTION.motion_file) -> mdp2.TaskCfg:
    """Return the MDP that tracks the clip ``motion_file``, which it opens for its numbers of frames and bodies."""
    with np.load(retrieve_file_path(motion_file), allow_pickle=False) as data:
        frames, bodies = data["body_pos_w"].shape[:2]
    step_dt = MDP1_CFG.sim.dt * MDP1_CFG.decimation
    frame_progress = {"start": clip["start"], "steps": mdp2.term_state("quantities.episode_steps", "steps")}

    return mdp2.TaskCfg(
        actions={
            "joint_pos": mdp2.Action(
                tracking.joint_position,
                inputs={"default_joint_pos": robot.default_joint_pos, "joint_ids": joints},
                outputs={"joint_pos_target": robot.joint_pos_target},
                params={"scale": G1_ACTION_SCALE},
            ),
        },
        quantities={
            "episode_steps": mdp2.Quantity(tracking.episode_steps),
            "motion_frame": mdp2.Quantity(
                tracking.motion_frame, inputs=frame_progress, params={"ahead": 0, "frames": frames}
            ),
            "next_motion_frame": mdp2.Quantity(
                tracking.motion_frame, inputs=frame_progress, params={"ahead": 1, "frames": frames}
            ),
        },
        observations={
            "policy": {
                "command": mdp2.Observation(
                    tracking.motion_command,
                    inputs={
                        "motion_frame": mdp2.quantities.next_motion_frame,
                        "clip_joint_pos": clip["clip_joint_pos"],
                        "clip_joint_vel": clip["clip_joint_vel"],
                        "joint_ids": joints,
                    },
                ),
                "motion_anchor_ori_b": mdp2.Observation(
                    tracking.motion_anchor_ori_b,
                    inputs={**observed, **anchor_ori},
                    noise=mdp2.UniformNoise(-0.05, 0.05),
                ),
                "base_ang_vel": mdp2.Observation(
                    mdp.base_ang_vel, inputs=root_state, noise=mdp2.UniformNoise(-0.2, 0.2)
                ),
                "joint_pos": mdp2.Observation(
                    mdp.joint_pos_rel,
                    inputs={
                        "joint_pos": robot.joint_pos,
                        "default_joint_pos": robot.default_joint_pos,
                        "joint_ids": joints,
                    },
                    noise=mdp2.UniformNoise(-0.01, 0.01),
                ),
                "joint_vel": mdp2.Observation(
                    mdp.joint_vel_rel,
                    inputs={
                        "joint_vel": robot.joint_vel,
                        "default_joint_vel": robot.default_joint_vel,
                        "joint_ids": joints,
                    },
                    noise=mdp2.UniformNoise(-1.5, 1.5),
                ),
                "actions": mdp2.Observation(mdp.last_action, inputs={"actions": mdp2.last_action}),
            },
            "critic": {
                "command": mdp2.Observation(
                    tracking.motion_command,
                    inputs={
                        "motion_frame": mdp2.quantities.next_motion_frame,
                        "clip_joint_pos": clip["clip_joint_pos"],
                        "clip_joint_vel": clip["clip_joint_vel"],
                        "joint_ids": joints,
                    },
                ),
                "motion_anchor_pos_b": mdp2.Observation(
                    tracking.motion_anchor_pos_b, inputs={**observed, **anchor_pos}
                ),
                "motion_anchor_ori_b": mdp2.Observation(
                    tracking.motion_anchor_ori_b, inputs={**observed, **anchor_ori}
                ),
                "base_lin_vel": mdp2.Observation(mdp.base_lin_vel, inputs=root_state),
                "base_ang_vel": mdp2.Observation(mdp.base_ang_vel, inputs=root_state),
                "joint_pos": mdp2.Observation(
                    mdp.joint_pos_rel,
                    inputs={
                        "joint_pos": robot.joint_pos,
                        "default_joint_pos": robot.default_joint_pos,
                        "joint_ids": joints,
                    },
                ),
                "joint_vel": mdp2.Observation(
                    mdp.joint_vel_rel,
                    inputs={
                        "joint_vel": robot.joint_vel,
                        "default_joint_vel": robot.default_joint_vel,
                        "joint_ids": joints,
                    },
                ),
                "actions": mdp2.Observation(mdp.last_action, inputs={"actions": mdp2.last_action}),
                "body_pos": mdp2.Observation(
                    tracking.robot_body_pos_b,
                    inputs={
                        "body_link_pose_w": robot.body_link_pose_w,
                        "anchor_ids": anchor,
                        "body_ids": tracked_bodies,
                    },
                ),
                "body_ori": mdp2.Observation(
                    tracking.robot_body_ori_b,
                    inputs={
                        "body_link_pose_w": robot.body_link_pose_w,
                        "anchor_ids": anchor,
                        "body_ids": tracked_bodies,
                    },
                ),
            },
        },
        rewards={
            "motion_global_anchor_pos": mdp2.Reward(
                tracking.motion_global_anchor_position_error_exp,
                inputs={**tracked, **anchor_pos},
                params={"std": 0.3},
                weight=0.5,
            ),
            "motion_global_anchor_ori": mdp2.Reward(
                tracking.motion_global_anchor_orientation_error_exp,
                inputs={**tracked, **anchor_ori},
                params={"std": 0.4},
                weight=0.5,
            ),
            "motion_body_pos": mdp2.Reward(
                tracking.motion_relative_body_position_error_exp,
                inputs={**tracked, **anchor_pos, "clip_body_quat": clip["clip_body_quat"], "body_ids": tracked_bodies},
                params={"std": 0.3},
                weight=1.0,
            ),
            "motion_body_ori": mdp2.Reward(
                tracking.motion_relative_body_orientation_error_exp,
                inputs={**tracked, **anchor_ori, "body_ids": tracked_bodies},
                params={"std": 0.4},
                weight=1.0,
            ),
            "motion_body_lin_vel": mdp2.Reward(
                tracking.motion_global_body_velocity_error_exp,
                inputs={
                    "motion_frame": mdp2.quantities.motion_frame,
                    "body_com_vel_w": robot.body_com_vel_w,
                    "clip_body_vel": clip["clip_body_lin_vel"],
                    "body_ids": tracked_bodies,
                },
                params={"angular": False, "std": 1.0},
                weight=1.0,
            ),
            "motion_body_ang_vel": mdp2.Reward(
                tracking.motion_global_body_velocity_error_exp,
                inputs={
                    "motion_frame": mdp2.quantities.motion_frame,
                    "body_com_vel_w": robot.body_com_vel_w,
                    "clip_body_vel": clip["clip_body_ang_vel"],
                    "body_ids": tracked_bodies,
                },
                params={"angular": True, "std": 3.14},
                weight=1.0,
            ),
            "action_rate_l2": mdp2.Reward(mdp.action_rate_l2, inputs={"actions": mdp2.last_action}, weight=-0.01),
            "joint_limit": mdp2.Reward(
                mdp.joint_pos_limits,
                inputs={
                    "joint_pos": robot.joint_pos,
                    "soft_joint_pos_limits": robot.soft_joint_pos_limits,
                    "joint_ids": joints,
                },
                weight=-10.0,
            ),
            "self_collisions": mdp2.Reward(
                tracking.self_collision_cost,
                inputs={
                    "force_matrix_w_history": self_collision.force_matrix_w_history,
                    "body_ids": self_collision.bodies(".*"),
                },
                params={"threshold": 10.0},
                weight=-10.0,
            ),
        },
        terminations={
            "time_out": mdp2.TimeLimit(episode_length_s=MDP1_CFG.episode_length_s),
            "motion_end": mdp2.Termination(
                tracking.motion_end,
                inputs={"motion_frame": mdp2.quantities.motion_frame},
                params={"frames": frames},
                time_out=True,
            ),
            "anchor_pos": mdp2.Termination(
                tracking.bad_anchor_pos_z_only, inputs={**tracked, **anchor_pos}, params={"threshold": 0.25}
            ),
            "anchor_ori": mdp2.Termination(
                tracking.bad_anchor_ori,
                inputs={**tracked, **anchor_ori, "gravity_w": scene.gravity_w},
                params={"threshold": 0.8},
            ),
            "ee_body_pos": mdp2.Termination(
                tracking.bad_motion_body_pos_z_only,
                inputs={
                    **tracked,
                    "env_origins": scene.env_origins,
                    "clip_body_pos": clip["clip_body_pos"],
                    "body_ids": robot.bodies(*EE_BODIES),
                },
                params={"threshold": 0.25},
            ),
        },
        events={
            # Newton has one friction coefficient, so MDP1's separate dynamic friction and buckets do not apply.
            "physics_material": mdp2.Event(
                mdp.randomize_rigid_body_material,
                mode=mdp2.EventMode.STARTUP,
                inputs={"asset": robot, "body_ids": robot.bodies(".*")},
                params={"friction_range": (0.4, 1.2), "restitution_range": (0.0, 0.0)},
            ),
            "base_mass": mdp2.Event(
                mdp.randomize_rigid_body_mass,
                mode=mdp2.EventMode.STARTUP,
                inputs={"asset": robot, "body_ids": robot.bodies("torso_link")},
                params={
                    "mass_distribution_params": (0.8, 1.25),
                    "operation": "scale",
                    "distribution": "log_uniform",
                    "recompute_inertia": True,
                },
            ),
            # The countdown outlives the episodes, which are shorter than the interval.
            "push_robot": mdp2.Event(
                mdp.push_by_setting_velocity,
                outputs={"root_com_vel_w": robot.root_com_vel_w},
                mode=mdp2.EventMode.INTERVAL,
                interval_range_s=(10.0, 15.0),
                resample_interval_on_reset=False,
                params={"velocity_range": ((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))},
            ),
            # MDP1's motion command resets the robot onto the reference when it resamples.
            "reset_to_motion": mdp2.Event(
                tracking.reset_to_motion,
                inputs={
                    "env_origins": scene.env_origins,
                    "soft_joint_pos_limits": robot.soft_joint_pos_limits,
                    "terminated": mdp2.terminated,
                    "episode_steps": mdp2.quantities.episode_steps,
                },
                outputs={"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel, **root_state},
                params={
                    "motion_file": motion_file,
                    "frames": frames,
                    "bodies": bodies,
                    "bins": int(frames // (1.0 / step_dt)) + 1,
                    "pose_range": MOTION.pose_range,
                    "velocity_range": MOTION.velocity_range,
                    "joint_position_range": MOTION.joint_position_range,
                    "adaptive_kernel_size": MOTION.adaptive_kernel_size,
                    "adaptive_lambda": MOTION.adaptive_lambda,
                    "adaptive_uniform_ratio": MOTION.adaptive_uniform_ratio,
                    "adaptive_alpha": MOTION.adaptive_alpha,
                },
            ),
        },
    )


@configclass
class G1MotionTrackingPhysicsCfg(PresetCfg):
    """Newton physics presets of the task: MDP1's physics."""

    newton_mjwarp: NewtonCfg = MDP1_CFG.sim.physics
    default: NewtonCfg = newton_mjwarp


@configclass
class G1MotionTrackingEnvCfg(MDP2RLEnvCfg):
    """MDP1's G1 motion tracking task on its scene and physics, with the MDP in MDP2."""

    sim: sim_utils.SimulationCfg = replace(MDP1_CFG.sim, physics=G1MotionTrackingPhysicsCfg())
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    task: mdp2.TaskCfg | None = None  # Built on creation, from the clip
    decimation: int = MDP1_CFG.decimation

    def __post_init__(self):
        if self.task is None:
            self.task = motion_tracking_task()
