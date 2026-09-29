# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole balancing for the experimental MDP runtime.

The MDP mirrors the stable manager-based ``Isaac-Cartpole`` task
(:mod:`isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg`) term for term. The stable
``success_rate`` reward (weight 0, logging only) is omitted.
"""

from __future__ import annotations

import math

from isaaclab_experimental.mdp_runtime import (
    ActionTermCfg,
    EventTermCfg,
    MdpCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)

from isaaclab.utils import configclass

STABLE_TASK = "Isaac-Cartpole"
"""Stable task whose simulation and scene this MDP runs on."""


@configclass
class CartpoleMdpCfg(MdpCfg):
    """Cartpole MDP on the ``robot`` articulation of the stable task's scene."""

    episode_length_s: float = 5.0

    actions: dict[str, ActionTermCfg] = {
        "joint_effort": ActionTermCfg(term="joint_effort", params={"joints": ["slider_to_cart"]}, scale=100.0),
    }

    observations: dict[str, ObservationGroupCfg] = {
        "policy": ObservationGroupCfg(
            terms={
                "joint_pos_rel": ObservationTermCfg(term="joint_pos_rel"),
                "joint_vel_rel": ObservationTermCfg(term="joint_vel_rel"),
            }
        ),
    }

    rewards: dict[str, RewardTermCfg] = {
        "alive": RewardTermCfg(term="is_alive", weight=1.0),
        "terminating": RewardTermCfg(term="is_terminated", weight=-2.0),
        "pole_pos": RewardTermCfg(
            term="joint_pos_target_l2", weight=-1.0, params={"target": 0.0, "joints": ["cart_to_pole"]}
        ),
        "cart_vel": RewardTermCfg(term="joint_vel_l1", weight=-0.01, params={"joints": ["slider_to_cart"]}),
        "pole_vel": RewardTermCfg(term="joint_vel_l1", weight=-0.005, params={"joints": ["cart_to_pole"]}),
    }

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(term="time_out", time_out=True),
        "cart_out_of_bounds": TerminationTermCfg(
            term="joint_pos_out_of_manual_limit", params={"bounds": (-3.0, 3.0), "joints": ["slider_to_cart"]}
        ),
    }

    events: dict[str, EventTermCfg] = {
        "reset_cart_position": EventTermCfg(
            term="reset_joints_by_offset",
            params={"joints": ["slider_to_cart"], "position_range": (-1.0, 1.0), "velocity_range": (-0.5, 0.5)},
        ),
        "reset_pole_position": EventTermCfg(
            term="reset_joints_by_offset",
            params={
                "joints": ["cart_to_pole"],
                "position_range": (-0.25 * math.pi, 0.25 * math.pi),
                "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
            },
        ),
    }
