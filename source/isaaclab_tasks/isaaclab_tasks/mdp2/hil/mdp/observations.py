# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms of the hybrid imitation learning tasks."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2


@wp.func
def _goal_b(
    world: int, element: int, command: wp.array2d(dtype=float), root_link_pose_w: wp.array(dtype=wp.transformf)
) -> float:
    forward = wp.quat_rotate(wp.transform_get_rotation(root_link_pose_w[world]), wp.vec3f(1.0, 0.0, 0.0))
    heading = wp.normalize(wp.vec2f(forward[0], forward[1]))
    if element == 0:
        return heading[0] * command[world, 0] + heading[1] * command[world, 1]
    if element == 1:
        return heading[0] * command[world, 1] - heading[1] * command[world, 0]
    return command[world, 2]


goal_b = mdp2.Term(warp=_goal_b, shape=("worlds", 3))
"""The goal's direction in the frame of the root's heading, and its speed."""
