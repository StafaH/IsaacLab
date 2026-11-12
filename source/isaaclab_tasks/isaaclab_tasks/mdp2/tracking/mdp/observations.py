# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms of the motion tracking tasks."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2

from .events import reference_frame


@wp.func
def _root_height(world: int, root_link_pose_w: wp.array(dtype=wp.transformf)) -> float:
    return wp.transform_get_translation(root_link_pose_w[world])[2]


root_height = mdp2.Term(warp=_root_height)
"""Height of the root [m]."""


@wp.func
def _body_pos_b(
    world: int,
    body: int,
    axis: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
) -> float:
    position = wp.transform_get_translation(body_link_pose_w[world, body])
    return wp.transform_point(wp.transform_inverse(root_link_pose_w[world]), position)[axis]


body_pos_b = mdp2.Term(warp=_body_pos_b, shape=("worlds", "bodies", 3))
"""Positions of the bodies in the root frame [m]."""


@wp.func
def _reference_body_pos_b(
    world: int,
    body: int,
    axis: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_body_pos: wp.array3d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    end: wp.array(dtype=float),
) -> float:
    frame = reference_frame(world, start, episode_length, end, 1)
    target = wp.vec3f(clip_body_pos[frame, body, 0], clip_body_pos[frame, body, 1], clip_body_pos[frame, body, 2])
    return wp.transform_point(wp.transform_inverse(root_link_pose_w[world]), target + env_origins[world])[axis]


reference_body_pos_b = mdp2.Term(warp=_reference_body_pos_b, shape=("worlds", "bodies", 3))
"""Positions of the bodies in the frame that the next step tracks, in the robot's root frame [m]."""
