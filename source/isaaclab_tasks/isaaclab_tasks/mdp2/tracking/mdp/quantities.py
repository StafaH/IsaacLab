# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quantities of the motion tracking tasks: the reference of the next step, and the tracking errors."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2

from .events import reference_frame


@wp.func
def _reference_joint_pos(
    world: int,
    joint: int,
    clip_joint_pos: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
) -> float:
    frame = reference_frame(world, start, episode_length, clip_joint_pos.shape[0] - 1) + 1
    return clip_joint_pos[frame, joint_ids[joint]]


reference_joint_pos = mdp2.Term(warp=_reference_joint_pos, shape=("worlds", "joint_ids"))
"""Joint positions of the frame that the next step tracks."""


@wp.func
def _joint_pos_error(
    world: int,
    joint_pos: wp.array2d(dtype=float),
    clip_joint_pos: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
) -> float:
    frame = reference_frame(world, start, episode_length, clip_joint_pos.shape[0])
    total = float(0.0)
    for j in range(joint_pos.shape[1]):
        error = joint_pos[world, j] - clip_joint_pos[frame, j]
        total += error * error
    return total / float(joint_pos.shape[1])


joint_pos_error = mdp2.Term(warp=_joint_pos_error)
"""Mean squared error of the joint positions to the tracked frame [rad^2]."""


@wp.func
def _joint_vel_error(
    world: int,
    joint_vel: wp.array2d(dtype=float),
    clip_joint_vel: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
) -> float:
    frame = reference_frame(world, start, episode_length, clip_joint_vel.shape[0])
    total = float(0.0)
    for j in range(joint_vel.shape[1]):
        error = joint_vel[world, j] - clip_joint_vel[frame, j]
        total += error * error
    return total / float(joint_vel.shape[1])


joint_vel_error = mdp2.Term(warp=_joint_vel_error)
"""Mean squared error of the joint velocities to the tracked frame [rad^2/s^2]."""


@wp.func
def _body_pos_error(
    world: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_body_pos: wp.array3d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
) -> float:
    frame = reference_frame(world, start, episode_length, clip_body_pos.shape[0])
    total = float(0.0)
    for b in range(body_link_pose_w.shape[1]):
        target = wp.vec3f(clip_body_pos[frame, b, 0], clip_body_pos[frame, b, 1], clip_body_pos[frame, b, 2])
        position = wp.transform_get_translation(body_link_pose_w[world, b]) - env_origins[world]
        total += wp.length_sq(position - target)
    return total / float(body_link_pose_w.shape[1])


body_pos_error = mdp2.Term(warp=_body_pos_error)
"""Mean squared distance of the bodies to their positions in the tracked frame [m^2]."""


@wp.func
def _root_vel_error(
    world: int,
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    clip_root_vel: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    angular_scale: float,
) -> float:
    frame = reference_frame(world, start, episode_length, clip_root_vel.shape[0])
    total = float(0.0)
    for k in range(6):
        error = root_com_vel_w[world][k] - clip_root_vel[frame, k]
        total += wp.where(k < 3, 1.0, angular_scale) * error * error
    return total


root_vel_error = mdp2.Term(warp=_root_vel_error)
"""Squared error of the root velocity to the tracked frame, the angular part scaled by ``angular_scale``."""
