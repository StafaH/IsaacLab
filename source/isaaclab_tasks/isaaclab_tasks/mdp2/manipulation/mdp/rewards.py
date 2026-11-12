# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms of the multi-task manipulation tasks."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2


@wp.func
def _orientation_tracking(
    world: int,
    orientation_error: wp.array(dtype=float),
    thumb_force: wp.array(dtype=float),
    finger_force: wp.array(dtype=float),
    std: float,
    contact_threshold: float,
) -> float:
    if thumb_force[world] > contact_threshold and finger_force[world] > contact_threshold:
        return 1.0 - wp.tanh(orientation_error[world] / std)
    return 0.0


orientation_tracking = mdp2.Term(warp=_orientation_tracking)
"""``1 - tanh(e / std)`` of the goal orientation error ``e``, while the object is grasped."""
