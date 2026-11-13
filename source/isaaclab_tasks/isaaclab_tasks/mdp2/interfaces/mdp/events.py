# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms of the multi-interface tracking tasks."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2.mdp.events import _uniform_spatial


@wp.func
def _scaled_push(
    world: int,
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    scale: wp.array(dtype=float),
    velocity_range: wp.types.matrix(shape=(6, 2), dtype=wp.float32),
    rng: wp.int32,
):
    root_com_vel_w[world] = root_com_vel_w[world] + scale[world] * _uniform_spatial(rng, 0, velocity_range)


scaled_push = mdp2.Term(warp=_scaled_push, random=True)
"""Add a uniform sample from ``velocity_range``, times the ``scale`` input of the world, to the root velocity."""
