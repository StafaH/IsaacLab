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
    end: wp.array(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
) -> float:
    frame = reference_frame(world, start, episode_length, end, 1)
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
    end: wp.array(dtype=float),
) -> float:
    frame = reference_frame(world, start, episode_length, end, 0)
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
    end: wp.array(dtype=float),
) -> float:
    frame = reference_frame(world, start, episode_length, end, 0)
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
    end: wp.array(dtype=float),
) -> float:
    frame = reference_frame(world, start, episode_length, end, 0)
    total = float(0.0)
    for b in range(body_link_pose_w.shape[1]):
        target = wp.vec3f(clip_body_pos[frame, b, 0], clip_body_pos[frame, b, 1], clip_body_pos[frame, b, 2])
        position = wp.transform_get_translation(body_link_pose_w[world, b]) - env_origins[world]
        total += wp.length_sq(position - target)
    return total / float(body_link_pose_w.shape[1])


body_pos_error = mdp2.Term(warp=_body_pos_error)
"""Mean squared distance of the bodies to their positions in the tracked frame [m^2]."""


@wp.func
def _subset_pos_error(
    world: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_body_pos: wp.array3d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    end: wp.array(dtype=float),
    body_ids: wp.array(dtype=wp.int32),
) -> float:
    frame = reference_frame(world, start, episode_length, end, 0)
    total = float(0.0)
    for k in range(body_ids.shape[0]):
        b = body_ids[k]
        target = wp.vec3f(clip_body_pos[frame, b, 0], clip_body_pos[frame, b, 1], clip_body_pos[frame, b, 2])
        position = wp.transform_get_translation(body_link_pose_w[world, b]) - env_origins[world]
        total += wp.length_sq(position - target)
    return total / float(body_ids.shape[0])


subset_pos_error = mdp2.Term(warp=_subset_pos_error)
"""Mean squared distance of the ``body_ids`` bodies to their positions in the tracked frame [m^2]."""


@wp.func
def _local_pos_error(
    world: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_body_pos: wp.array3d(dtype=float),
    clip_root_pose: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    end: wp.array(dtype=float),
    body_ids: wp.array(dtype=wp.int32),
) -> float:
    frame = reference_frame(world, start, episode_length, end, 0)
    root = wp.transform_get_translation(root_link_pose_w[world]) - env_origins[world]
    drift = wp.vec3f(root[0] - clip_root_pose[frame, 0], root[1] - clip_root_pose[frame, 1], 0.0)
    total = float(0.0)
    for k in range(body_ids.shape[0]):
        b = body_ids[k]
        target = wp.vec3f(clip_body_pos[frame, b, 0], clip_body_pos[frame, b, 1], clip_body_pos[frame, b, 2])
        position = wp.transform_get_translation(body_link_pose_w[world, b]) - env_origins[world]
        total += wp.length_sq(position - drift - target)
    return total / float(body_ids.shape[0])


local_pos_error = mdp2.Term(warp=_local_pos_error)
"""Mean squared distance of the ``body_ids`` bodies to the tracked frame, with the clip moved horizontally onto the
robot's root [m^2]: the error of the pose, the height and the heading, and not of where the robot stands.
"""


@wp.func
def _root_drift(
    world: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_root_pose: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    end: wp.array(dtype=float),
) -> float:
    frame = reference_frame(world, start, episode_length, end, 0)
    root = wp.transform_get_translation(root_link_pose_w[world]) - env_origins[world]
    x, y = root[0] - clip_root_pose[frame, 0], root[1] - clip_root_pose[frame, 1]
    return x * x + y * y


root_drift = mdp2.Term(warp=_root_drift)
"""Squared horizontal distance of the robot's root to the root of the tracked frame [m^2]."""


@wp.func
def _subset_vel_error(
    world: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    clip_body_pos: wp.array3d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    end: wp.array(dtype=float),
    body_ids: wp.array(dtype=wp.int32),
    previous: wp.array3d(dtype=float),
    step_dt: float,
) -> float:
    frame = reference_frame(world, start, episode_length, end, 0)
    before = wp.max(frame - 1, 0)
    total = float(0.0)
    for k in range(body_ids.shape[0]):
        b = body_ids[k]
        position = wp.transform_get_translation(body_link_pose_w[world, b])
        for axis in range(3):
            moved = position[axis] - previous[world, k, axis]
            target = clip_body_pos[frame, b, axis] - clip_body_pos[before, b, axis]
            total += (moved - target) * (moved - target)
    return total / float(body_ids.shape[0]) / (step_dt * step_dt)


@wp.func
def _remember_positions(
    world: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    body_ids: wp.array(dtype=wp.int32),
    previous: wp.array3d(dtype=float),
):
    for k in range(body_ids.shape[0]):
        position = wp.transform_get_translation(body_link_pose_w[world, body_ids[k]])
        for axis in range(3):
            previous[world, k, axis] = position[axis]


subset_vel_error = mdp2.Term(
    warp=mdp2.Implementation(compute=_subset_vel_error, update=_remember_positions, reset=_remember_positions),
    state={"previous": mdp2.State(("worlds", "body_ids", 3))},
)
"""Mean squared error of the velocity of the ``body_ids`` bodies to the clip's, over the last step [m^2/s^2].

Both velocities are differences of positions over ``step_dt``: a body that the clip holds still, such as a planted
foot, is in error by its speed.
"""


@wp.func
def _root_vel_error(
    world: int,
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    clip_root_vel: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    end: wp.array(dtype=float),
    angular_scale: float,
) -> float:
    frame = reference_frame(world, start, episode_length, end, 0)
    total = float(0.0)
    for k in range(6):
        error = root_com_vel_w[world][k] - clip_root_vel[frame, k]
        total += wp.where(k < 3, 1.0, angular_scale) * error * error
    return total


root_vel_error = mdp2.Term(warp=_root_vel_error)
"""Squared error of the root velocity to the tracked frame, the angular part scaled by ``angular_scale``."""


@wp.func
def _episode_steps(world: int, episode_length: wp.array(dtype=wp.int32)) -> float:
    return float(episode_length[world])


episode_steps = mdp2.Term(warp=_episode_steps)
"""Steps of the episode so far. A reset event reads it as the length of the episode that ended, which
``mdp2.episode_length`` no longer holds then."""
