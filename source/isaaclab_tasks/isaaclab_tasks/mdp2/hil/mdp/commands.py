# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal commands of the hybrid imitation learning tasks: a direction of travel in the world frame, and a speed."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2


@wp.func
def _reference_goal(
    world: int,
    command: wp.array2d(dtype=float),
    clip_root_vel: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    end: wp.array(dtype=float),
    lookahead: int,
):
    frame = wp.min(int(start[world]) + episode_length[world] + lookahead, int(end[world]) - 1)
    velocity = wp.vec2f(clip_root_vel[frame, 0], clip_root_vel[frame, 1])
    speed = wp.length(velocity)
    command[world, 0] = velocity[0] / wp.max(speed, 1.0e-6)
    command[world, 1] = velocity[1] / wp.max(speed, 1.0e-6)
    command[world, 2] = speed


reference_goal = mdp2.Term(
    warp=mdp2.Implementation(compute=_reference_goal, update=_reference_goal), shape=("worlds", 3)
)
"""The direction and speed of the clip's root ``lookahead`` frames after the frame that the world tracks."""


@wp.func
def _heading_goal(
    world: int,
    command: wp.array2d(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    max_turn: float,
    speed: wp.vec2f,
    rng: wp.int32,
):
    forward = wp.quat_rotate(wp.transform_get_rotation(root_link_pose_w[world]), wp.vec3f(1.0, 0.0, 0.0))
    angle = wp.atan2(forward[1], forward[0]) + wp.randf(wp.rand_init(rng, 0), -max_turn, max_turn)
    command[world, 0] = wp.cos(angle)
    command[world, 1] = wp.sin(angle)
    command[world, 2] = wp.randf(wp.rand_init(rng, 1), speed[0], speed[1])


heading_goal = mdp2.Term(warp=_heading_goal, shape=("worlds", 3), random=True)
"""A direction within ``max_turn`` [rad] of the root's heading, and a speed drawn from the range ``speed`` [m/s]."""
