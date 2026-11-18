# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards of the parkour task."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2


@wp.func
def _goal_reached(
    world: int,
    at_goal: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    episode: wp.array2d(dtype=float),
) -> float:
    if float(episode_length[world]) >= episode[world, 7]:
        return at_goal[world]
    return 0.0


goal_reached = mdp2.Term(warp=_goal_reached)
"""One on the last step of an episode whose root is at the goal."""


@wp.func
def _failed(
    world: int,
    terminated: wp.array(dtype=wp.bool),
    episode_length: wp.array(dtype=wp.int32),
    episode: wp.array2d(dtype=float),
) -> float:
    if terminated[world] and float(episode_length[world]) < episode[world, 7]:
        return 1.0
    return 0.0


failed = mdp2.Term(warp=_failed)
"""One on the step that ends an episode before its time, bound to ``mdp2.terminated``."""


@wp.func
def _joint_power_l1(
    world: int,
    applied_effort: wp.array2d(dtype=float),
    joint_vel: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
) -> float:
    total = float(0.0)
    for k in range(joint_ids.shape[0]):
        total += wp.abs(applied_effort[world, joint_ids[k]] * joint_vel[world, joint_ids[k]])
    return total


joint_power_l1 = mdp2.Term(warp=_joint_power_l1)
"""Sum of the mechanical power of the joints, as magnitudes [W]."""
