# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quantities of the parkour task."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2


@wp.func
def _goal(
    world: int,
    element: int,
    clip_root_pose: wp.array2d(dtype=float),
    env_origins: wp.array(dtype=wp.vec3f),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    episode: wp.array2d(dtype=float),
) -> float:
    ahead = int(start[world]) + episode_length[world] + int(episode[world, 3])
    frame = wp.min(ahead, int(episode[world, 2]))
    if element < 3:
        return clip_root_pose[frame, element] + env_origins[world][element]
    rotation = wp.quatf(
        clip_root_pose[frame, 3], clip_root_pose[frame, 4], clip_root_pose[frame, 5], clip_root_pose[frame, 6]
    )
    forward = wp.quat_rotate(rotation, wp.vec3f(1.0, 0.0, 0.0))
    return wp.atan2(forward[1], forward[0])


goal = mdp2.Term(warp=_goal, shape=("worlds", 4))
"""The position [m] and heading [rad] of the clip's root at the goal frame of the episode, or at the frame that runs
ahead of the tracked one, whichever comes first."""


@wp.func
def _at_goal(
    world: int,
    goal: wp.array2d(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    episode: wp.array2d(dtype=float),
    height_tolerance: float,
    min_upright: float,
) -> float:
    pose = root_link_pose_w[world]
    offset = wp.vec3f(goal[world, 0], goal[world, 1], goal[world, 2]) - wp.transform_get_translation(pose)
    radius = episode[world, 5]
    up = wp.quat_rotate(wp.transform_get_rotation(pose), wp.vec3f(0.0, 0.0, 1.0))
    near = offset[0] * offset[0] + offset[1] * offset[1] < radius * radius
    if near and wp.abs(offset[2]) < height_tolerance and up[2] > min_upright:
        return 1.0
    return 0.0


at_goal = mdp2.Term(warp=_at_goal)
"""One if the root is upright within the episode's distance of the goal and ``height_tolerance`` [m] of its height."""
