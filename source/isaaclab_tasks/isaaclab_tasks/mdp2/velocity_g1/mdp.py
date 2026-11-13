# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terms of the Unitree G1 velocity task."""

import torch
import warp as wp
from isaaclab_experimental.mdp2 import Term


@wp.func
def _feet_air_time_biped(
    world: int,
    current_air_time: wp.array2d(dtype=float),
    current_contact_time: wp.array2d(dtype=float),
    command: wp.array2d(dtype=float),
    body_ids: wp.array(dtype=wp.int32),
    threshold: float,
) -> float:
    if wp.length(wp.vec2f(command[world, 0], command[world, 1])) <= 0.1:
        return 0.0

    feet_in_contact = int(0)
    shortest = float(threshold)
    for k in range(body_ids.shape[0]):
        body = body_ids[k]
        contact_time = current_contact_time[world, body]
        if contact_time > 0.0:
            feet_in_contact += 1
            shortest = wp.min(shortest, contact_time)
        else:
            shortest = wp.min(shortest, current_air_time[world, body])
    return wp.where(feet_in_contact == 1, shortest, 0.0)


def _feet_air_time_biped_torch(current_air_time, current_contact_time, command, body_ids, *, threshold):
    contact_time = current_contact_time[:, body_ids]
    in_contact = contact_time > 0.0
    time = torch.where(in_contact, contact_time, current_air_time[:, body_ids])
    reward = time.amin(dim=-1).clamp(max=threshold) * (in_contact.sum(-1) == 1)
    return reward * (command[:, :2].norm(dim=-1) > 0.1)


feet_air_time_biped = Term(torch=_feet_air_time_biped_torch, warp=_feet_air_time_biped)
"""Time that the ``body_ids`` feet have spent in their state, one in the air and one on the ground, up to ``threshold``.

Zero unless exactly one foot is on the ground, and for worlds whose velocity ``command`` is at most 0.1, like MDP1's
``feet_air_time_positive_biped``.
"""
