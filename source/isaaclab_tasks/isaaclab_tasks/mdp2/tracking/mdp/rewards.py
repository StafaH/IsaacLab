# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms of the motion tracking tasks."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2


@wp.func
def _exp_kernel(world: int, value: wp.array(dtype=float), scale: float) -> float:
    return wp.exp(-scale * value[world])


exp_kernel = mdp2.Term(warp=_exp_kernel)
"""``exp(-scale * value)`` of the ``value`` input, one float per world, such as a tracking error."""
