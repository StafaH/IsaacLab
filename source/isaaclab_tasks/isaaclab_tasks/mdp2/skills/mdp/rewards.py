# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms of the multi-skill tasks."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2


@wp.func
def _axis_alignment(
    world: int, root_link_pose_w: wp.array(dtype=wp.transformf), axis: wp.vec3f, direction: wp.vec3f
) -> float:
    return 0.5 + 0.5 * wp.dot(wp.quat_rotate(wp.transform_get_rotation(root_link_pose_w[world]), axis), direction)


axis_alignment = mdp2.Term(warp=_axis_alignment)
"""How well the ``axis`` of the root points along the world ``direction``: 1 aligned, 0 opposed."""


@wp.func
def _height_reward(world: int, root_link_pose_w: wp.array(dtype=wp.transformf), target: float) -> float:
    return wp.clamp(wp.transform_get_translation(root_link_pose_w[world])[2] / target, 0.0, 1.0)


height_reward = mdp2.Term(warp=_height_reward)
"""The height of the root as a fraction of ``target`` [m], at most 1."""


@wp.func
def _bodies_height_reward(
    world: int, body_link_pose_w: wp.array2d(dtype=wp.transformf), body_ids: wp.array(dtype=wp.int32), target: float
) -> float:
    total = float(0.0)
    for k in range(body_ids.shape[0]):
        height = wp.transform_get_translation(body_link_pose_w[world, body_ids[k]])[2]
        total += wp.clamp(height / target, 0.0, 1.0)
    return total / float(body_ids.shape[0])


bodies_height_reward = mdp2.Term(warp=_bodies_height_reward)
"""The mean height of the bodies ``body_ids`` as a fraction of ``target`` [m], at most 1 each."""


@wp.func
def _axis_velocity_exp(
    world: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    axis: wp.vec3f,
    speed: float,
    std: float,
) -> float:
    pointing = wp.quat_rotate(wp.transform_get_rotation(root_link_pose_w[world]), axis)
    heading = wp.normalize(wp.vec2f(pointing[0], pointing[1]))
    velocity = root_com_vel_w[world]
    error = speed - (heading[0] * velocity[0] + heading[1] * velocity[1])
    return wp.exp(-error * error / (std * std))


axis_velocity_exp = mdp2.Term(warp=_axis_velocity_exp)
"""Moving over the ground at ``speed`` [m/s] the way the ``axis`` of the root points, ``exp(-error^2 / std^2)``."""
