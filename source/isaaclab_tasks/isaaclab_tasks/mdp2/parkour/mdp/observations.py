# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations of the parkour task."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2


@wp.func
def _goal_b(
    world: int, element: int, goal: wp.array2d(dtype=float), root_link_pose_w: wp.array(dtype=wp.transformf)
) -> float:
    pose = root_link_pose_w[world]
    forward = wp.quat_rotate(wp.transform_get_rotation(pose), wp.vec3f(1.0, 0.0, 0.0))
    heading = wp.atan2(forward[1], forward[0])
    offset = wp.vec3f(goal[world, 0], goal[world, 1], goal[world, 2]) - wp.transform_get_translation(pose)
    if element == 0:
        return wp.cos(heading) * offset[0] + wp.sin(heading) * offset[1]
    if element == 1:
        return wp.cos(heading) * offset[1] - wp.sin(heading) * offset[0]
    if element == 2:
        return offset[2]
    if element == 3:
        return wp.cos(goal[world, 3] - heading)
    return wp.sin(goal[world, 3] - heading)


goal_b = mdp2.Term(warp=_goal_b, shape=("worlds", 5))
"""The goal in the frame of the root's heading: its position [m], and the cosine and sine of its heading."""


@wp.func
def _episode_b(
    world: int,
    element: int,
    episode: wp.array2d(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    at_goal: wp.array(dtype=float),
) -> float:
    if element == 0:
        return wp.where(episode[world, 5] > 0.0, 1.0, 0.0)
    if element == 1:
        steps = wp.min(episode[world, 6], episode[world, 7]) - float(episode_length[world])
        return wp.clamp(steps / 300.0, 0.0, 3.0)
    return at_goal[world]


episode_b = mdp2.Term(warp=_episode_b, shape=("worlds", 3))
"""What a critic needs of the episode: whether it has a goal to reach, its remaining steps in hundreds of three, and
whether the root is at the goal."""
