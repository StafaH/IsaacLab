# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka end-effector pose reaching for the experimental MDP runtime.

The MDP mirrors the stable ``Isaac-Reach-Franka`` task (joint-position actions) and runs on its scene.
Differences: the reward-weight curriculum is omitted (change weights with
:meth:`~isaaclab_experimental.mdp_runtime.MdpProgram.set_reward_weight`), and the success termination uses the
current pose error instead of the error from the previous command update.
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

STABLE_TASK = "Isaac-Reach-Franka"
"""Stable task whose simulation and scene this MDP runs on."""

_HAND = {"command": "ee_pose", "bodies": ["panda_hand"]}


@configclass
class FrankaReachMdpCfg(MdpCfg):
    """Pose reaching MDP on the ``robot`` articulation of the stable scene."""

    episode_length_s: float = 12.0

    commands: dict[str, CommandTermCfg] = {
        "ee_pose": CommandTermCfg(
            term="uniform_pose",
            resampling_time_range=(4.0, 4.0),
            params={
                "pos_x": (0.35, 0.65),
                "pos_y": (-0.2, 0.2),
                "pos_z": (0.15, 0.5),
                "roll": (0.0, 0.0),
                "pitch": (math.pi, math.pi),
                "yaw": (-3.14, 3.14),
            },
        )
    }

    actions: dict[str, ActionTermCfg] = {
        "arm_action": ActionTermCfg(
            term="joint_position", scale=0.5, params={"joints": ["panda_joint.*"], "use_default_offset": True}
        ),
    }

    observations: dict[str, ObservationGroupCfg] = {
        "policy": ObservationGroupCfg(
            terms={
                "joint_pos": ObservationTermCfg(term="joint_pos_rel", noise=(-0.01, 0.01)),
                "joint_vel": ObservationTermCfg(term="joint_vel_rel", noise=(-0.01, 0.01)),
                "pose_command": ObservationTermCfg(term="generated_commands", params={"command": "ee_pose"}),
                "actions": ObservationTermCfg(term="last_action"),
            }
        )
    }

    rewards: dict[str, RewardTermCfg] = {
        "end_effector_position_tracking": RewardTermCfg(term="position_command_error", weight=-0.2, params=_HAND),
        "end_effector_orientation_tracking": RewardTermCfg(term="orientation_command_error", weight=-0.1, params=_HAND),
        "success": RewardTermCfg(term="termination_term", weight=10.0, params={"terms": ["success"]}),
        "action_rate": RewardTermCfg(term="action_rate_l2", weight=-1.0e-4),
        "action_magnitude": RewardTermCfg(term="action_l2", weight=-0.005),
        "joint_vel": RewardTermCfg(term="joint_vel_l2", weight=-1.0e-4, params={"joints": ["panda_joint.*"]}),
    }

    terminations: dict[str, TerminationTermCfg] = {
        "success": TerminationTermCfg(
            term="pose_command_success",
            params={**_HAND, "position_threshold": 0.05, "orientation_threshold": 0.2},
        ),
        "time_out": TerminationTermCfg(term="time_out", time_out=True),
    }

    events: dict[str, EventTermCfg] = {
        "reset_robot_joints": EventTermCfg(
            term="reset_joints_by_scale", params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)}
        ),
    }
