# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quantity terms of the multi-task manipulation tasks."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2


@wp.func
def _orientation_error(
    world: int,
    command: wp.array2d(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    object_pose_w: wp.array(dtype=wp.transformf),
) -> float:
    goal = wp.quatf(command[world, 3], command[world, 4], command[world, 5], command[world, 6])
    root = wp.transform_get_rotation(root_link_pose_w[world])
    rotation = wp.quat_inverse(root) * wp.transform_get_rotation(object_pose_w[world])
    return 2.0 * wp.acos(wp.min(wp.abs(wp.dot(goal, rotation)), 1.0))


orientation_error = mdp2.Term(warp=_orientation_error)
"""Angle [rad] from the object's orientation to the commanded one, which is in the robot's root frame."""


@wp.func
def _pose_succeeded(
    world: int,
    goal_error: wp.array(dtype=float),
    orientation_error: wp.array(dtype=float),
    thumb_force: wp.array(dtype=float),
    finger_force: wp.array(dtype=float),
    succeeded: wp.array(dtype=float),
    distance: float,
    angle: float,
    contact_threshold: float,
) -> float:
    grasped = thumb_force[world] > contact_threshold and finger_force[world] > contact_threshold
    if grasped and goal_error[world] < distance and orientation_error[world] < angle:
        succeeded[world] = 1.0
    return succeeded[world]


pose_succeeded = mdp2.Term(warp=_pose_succeeded, state={"succeeded": mdp2.State()})
"""1 once the grasped object has been within ``distance`` and ``angle`` of the goal pose in the episode, else 0."""
