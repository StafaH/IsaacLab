# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree Go2 flat-terrain velocity tracking for the experimental MDP runtime.

The MDP mirrors the stable ``Isaac-Velocity-Flat-UnitreeGo2`` task (all active terms, weights, noise, and
ranges) and runs on its scene. Omitted: the startup randomization of friction, base mass, and base COM, and the
reset external force, which is zero in the stable task.
"""

from __future__ import annotations

import math

from isaaclab_experimental.mdp_runtime import (
    ActionTermCfg,
    CommandTermCfg,
    EventTermCfg,
    MdpCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)

from isaaclab.utils import configclass

STABLE_TASK = "Isaac-Velocity-Flat-UnitreeGo2"
"""Stable task whose simulation and scene this MDP runs on."""

CONTACT_SENSOR = "contact_forces"
"""Contact sensor of the stable scene read by the air-time reward and the base-contact termination."""


@configclass
class Go2FlatVelocityMdpCfg(MdpCfg):
    """Velocity tracking MDP on the ``robot`` articulation and ``contact_forces`` sensor of the stable scene."""

    episode_length_s: float = 20.0

    commands: dict[str, CommandTermCfg] = {
        "base_velocity": CommandTermCfg(
            term="uniform_velocity",
            resampling_time_range=(10.0, 10.0),
            params={
                "lin_vel_x": (-1.0, 1.0),
                "lin_vel_y": (-1.0, 1.0),
                "ang_vel_z": (-1.0, 1.0),
                "heading": (-math.pi, math.pi),
                "heading_command": True,
                "heading_control_stiffness": 0.5,
                "rel_standing_envs": 0.02,
                "rel_heading_envs": 1.0,
            },
        )
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": ActionTermCfg(term="joint_position", scale=0.25, params={"use_default_offset": True}),
    }

    observations: dict[str, ObservationGroupCfg] = {
        "policy": ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(term="base_lin_vel", noise=(-0.1, 0.1)),
                "base_ang_vel": ObservationTermCfg(term="base_ang_vel", noise=(-0.2, 0.2)),
                "projected_gravity": ObservationTermCfg(term="projected_gravity", noise=(-0.05, 0.05)),
                "velocity_commands": ObservationTermCfg(term="generated_commands", params={"command": "base_velocity"}),
                "joint_pos": ObservationTermCfg(term="joint_pos_rel", noise=(-0.01, 0.01)),
                "joint_vel": ObservationTermCfg(term="joint_vel_rel", noise=(-1.5, 1.5)),
                "actions": ObservationTermCfg(term="last_action"),
            }
        )
    }

    rewards: dict[str, RewardTermCfg] = {
        "track_lin_vel_xy_exp": RewardTermCfg(
            term="track_lin_vel_xy_exp", weight=1.5, params={"std": 0.5, "command": "base_velocity"}
        ),
        "track_ang_vel_z_exp": RewardTermCfg(
            term="track_ang_vel_z_exp", weight=0.75, params={"std": 0.5, "command": "base_velocity"}
        ),
        "lin_vel_z_l2": RewardTermCfg(term="lin_vel_z_l2", weight=-2.0),
        "ang_vel_xy_l2": RewardTermCfg(term="ang_vel_xy_l2", weight=-0.05),
        "dof_torques_l2": RewardTermCfg(term="joint_torques_l2", weight=-2.0e-4),
        "dof_acc_l2": RewardTermCfg(term="joint_acc_l2", weight=-2.5e-7),
        "action_rate_l2": RewardTermCfg(term="action_rate_l2", weight=-0.01),
        "feet_air_time": RewardTermCfg(
            term="feet_air_time",
            weight=0.25,
            params={"command": "base_velocity", "threshold": 0.5, "contact_bodies": ".*_foot"},
        ),
        "flat_orientation_l2": RewardTermCfg(term="flat_orientation_l2", weight=-2.5),
    }

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(term="time_out", time_out=True),
        "base_contact": TerminationTermCfg(term="illegal_contact", params={"threshold": 1.0, "contact_bodies": "base"}),
    }

    events: dict[str, EventTermCfg] = {
        "reset_base": EventTermCfg(
            term="reset_root_state_uniform",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {axis: (-0.5, 0.5) for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            term="reset_joints_by_scale", params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)}
        ),
        "push_robot": EventTermCfg(
            term="push_by_setting_velocity",
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        ),
    }
