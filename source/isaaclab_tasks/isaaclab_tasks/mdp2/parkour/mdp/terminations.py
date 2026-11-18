# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terminations of the parkour task."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2


@wp.func
def _after_steps(
    world: int, episode_length: wp.array(dtype=wp.int32), episode: wp.array2d(dtype=float), column: int
) -> bool:
    return float(episode_length[world]) >= episode[world, column]


after_steps = mdp2.Term(warp=_after_steps)
"""Whether the episode has run for the steps in ``column`` of the ``episode`` input."""


@wp.func
def _above_limit(world: int, value: wp.array(dtype=float), episode: wp.array2d(dtype=float), column: int) -> bool:
    return value[world] > episode[world, column]


above_limit = mdp2.Term(warp=_above_limit)
"""Whether the ``value`` input, one float per world, is above ``column`` of the ``episode`` input."""
