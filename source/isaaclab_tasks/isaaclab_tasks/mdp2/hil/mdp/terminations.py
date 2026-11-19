# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms of the hybrid imitation learning tasks."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2


@wp.func
def _root_height_outside(world: int, root_link_pose_w: wp.array(dtype=wp.transformf), bounds: wp.vec2f) -> bool:
    height = wp.transform_get_translation(root_link_pose_w[world])[2]
    return not (height > bounds[0] and height < bounds[1])


root_height_outside = mdp2.Term(warp=_root_height_outside)
"""Whether the root is outside the height ``bounds`` [m]: fallen, launched, or not a number after a solver failure."""
