# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms of the multi-interface tracking tasks."""

from __future__ import annotations

import torch
import warp as wp
from isaaclab_experimental import mdp2


def _load_interfaces(ctx, *, interfaces, **params):
    patterns = torch.tensor(interfaces, dtype=torch.float32, device=ctx.state["patterns"].device)
    ctx.state["patterns"].copy_(patterns)
    ctx.state["shares"].fill_(1.0 / len(interfaces))


@wp.func
def _sampled_interface(
    world: int,
    command: wp.array2d(dtype=float),
    patterns: wp.array2d(dtype=float),
    shares: wp.array(dtype=float),
    fixed: wp.array(dtype=float),
    interface: wp.array(dtype=float),
    rng: wp.int32,
):
    chosen = int(fixed[world])
    if chosen < 0:
        draw = wp.randf(wp.rand_init(rng, 0))
        chosen = int(0)
        cumulative = shares[0]
        while chosen < shares.shape[0] - 1 and cumulative <= draw:
            chosen += 1
            cumulative += shares[chosen]
    interface[world] = float(chosen)
    for k in range(command.shape[1]):
        command[world, k] = patterns[chosen, k]


sampled_interface = mdp2.Term(
    warp=_sampled_interface,
    shape=("worlds", "groups"),
    state={
        "patterns": mdp2.State(("count", "groups")),
        "shares": mdp2.State(("count",)),
        "fixed": mdp2.State(initial=-1.0, persistent=True),
        "interface": mdp2.State(persistent=True),
    },
    random=True,
    init=_load_interfaces,
)
"""Which of ``groups`` groups of targets a world is told, as ones and zeros: one of the ``count`` rows of
``interfaces``, drawn on reset.

The ``shares`` state holds the probability of each row, which a learner may rewrite while training. A world whose
``fixed`` state is the index of a row always gets that row. ``interface`` is the index of the row a world has.
"""


@wp.func
def _ones(world: int, command: wp.array2d(dtype=float)):
    for k in range(command.shape[1]):
        command[world, k] = 1.0


ones_command = mdp2.Term(warp=_ones, shape=("worlds", "groups"))
"""Ones for all ``groups`` groups: every target is told."""
