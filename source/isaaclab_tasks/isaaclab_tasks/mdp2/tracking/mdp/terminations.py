# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms of the motion tracking tasks."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2


@wp.func
def _is_above(world: int, value: wp.array(dtype=float), threshold: float) -> bool:
    return value[world] > threshold


is_above = mdp2.Term(warp=_is_above)
"""Whether the ``value`` input, one float per world, is above ``threshold``."""


@wp.func
def _motion_ended(
    world: int,
    clip_joint_pos: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
) -> bool:
    return int(start[world]) + episode_length[world] >= clip_joint_pos.shape[0] - 2


motion_ended = mdp2.Term(warp=_motion_ended)
"""Whether the world tracked the clip to its end."""
