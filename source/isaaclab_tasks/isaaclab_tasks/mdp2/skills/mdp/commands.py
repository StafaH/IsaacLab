# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms of the multi-skill tasks."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2


@wp.func
def _one_hot(world: int, command: wp.array2d(dtype=float), index: int):
    for k in range(command.shape[1]):
        command[world, k] = wp.where(k == index, 1.0, 0.0)


one_hot_command = mdp2.Term(warp=_one_hot, shape=("worlds", "skills"))
"""One at ``index`` and zero elsewhere, of ``skills`` values: which skill a world practices."""
