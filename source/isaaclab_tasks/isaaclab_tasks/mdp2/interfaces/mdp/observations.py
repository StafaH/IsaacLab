# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms of the multi-interface tracking tasks: targets that a world sees only if its interface has them."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2

from isaaclab_tasks.mdp2.tracking.mdp import reference_frame


@wp.func
def _commanded_body_pos_b(
    world: int,
    body: int,
    axis: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_body_pos: wp.array3d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    end: wp.array(dtype=float),
    body_ids: wp.array(dtype=wp.int32),
    command: wp.array2d(dtype=float),
    group: int,
    ahead: int,
) -> float:
    frame = reference_frame(world, start, episode_length, end, ahead)
    b = body_ids[body]
    target = wp.vec3f(clip_body_pos[frame, b, 0], clip_body_pos[frame, b, 1], clip_body_pos[frame, b, 2])
    local = wp.transform_point(wp.transform_inverse(root_link_pose_w[world]), target + env_origins[world])
    return command[world, group] * local[axis]


commanded_body_pos_b = mdp2.Term(warp=_commanded_body_pos_b, shape=("worlds", "body_ids", 3))
"""Positions of the ``body_ids`` bodies in the frame ``ahead`` steps from now, in the robot's root frame [m].

Zero in the worlds whose ``command``, an interface, is zero at ``group``.
"""


@wp.func
def _commanded_joint_pos(
    world: int,
    joint: int,
    clip_joint_pos: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    end: wp.array(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
    command: wp.array2d(dtype=float),
    group: int,
    ahead: int,
) -> float:
    frame = reference_frame(world, start, episode_length, end, ahead)
    return command[world, group] * clip_joint_pos[frame, joint_ids[joint]]


commanded_joint_pos = mdp2.Term(warp=_commanded_joint_pos, shape=("worlds", "joint_ids"))
"""Positions of the ``joint_ids`` joints in the frame ``ahead`` steps from now [rad].

Zero in the worlds whose ``command``, an interface, is zero at ``group``.
"""
