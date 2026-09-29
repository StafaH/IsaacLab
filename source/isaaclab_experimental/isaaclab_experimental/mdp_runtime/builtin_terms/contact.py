# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-sensor terms.

Fields: ``contact_net_normal_forces_history`` ``(N, T, S, 3)`` (index 0 newest), and ``contact_current_air_time``,
``contact_last_air_time``, ``contact_current_contact_time`` ``(N, S)`` [s]. ``contact_bodies`` selects sensor
bodies by name.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from ..terms import REQUIRED, Stage, TermContext, define_term, implement
from .joint import f32, ids_vector

define_term(
    "feet_air_time",
    Stage.REWARD,
    params={"command": REQUIRED, "threshold": REQUIRED, "contact_bodies": REQUIRED},
    reads=("contact_current_contact_time", "contact_last_air_time", "commands"),
    doc="Sum over bodies that just touched down of (last air time - threshold), zero for |v_cmd_xy| <= 0.1. A body "
    "just touched down if 0 < contact time < step_dt + physics_dt / 2.",
)


@implement("feet_air_time", "warp")
def _(ctx: TermContext):
    ids, n = ids_vector(ctx.params["contact_ids"]), len(ctx.params["contact_ids"])
    col, threshold = ctx.params["command_columns"][0], float(ctx.params["threshold"])
    window = float(ctx.info.step_dt + 0.5 * ctx.info.physics_dt)

    @wp.func
    def term(env: int, f: Any) -> float:
        total = float(0.0)
        for k in range(wp.static(n)):
            t = f.contact_current_contact_time[env, ids[k]]
            if t > 0.0 and t < wp.static(window):
                total = total + f.contact_last_air_time[env, ids[k]] - wp.static(threshold)
        vx = f.commands[env, col]
        vy = f.commands[env, col + 1]
        return wp.where(wp.sqrt(vx * vx + vy * vy) > 0.1, total, 0.0)

    return term


@implement("feet_air_time", "torch")
def _(ctx: TermContext):
    fields, ids, out = ctx.fields, ctx.indices["contact_ids"], ctx.out
    col, threshold = ctx.params["command_columns"][0], f32(ctx.params["threshold"])
    window = f32(ctx.info.step_dt + 0.5 * ctx.info.physics_dt)

    def run():
        t = fields["contact_current_contact_time"][:, ids]
        first = (t > 0.0) & (t < window)
        reward = torch.sum((fields["contact_last_air_time"][:, ids] - threshold) * first, dim=1)
        moving = torch.linalg.vector_norm(fields["commands"][:, col : col + 2], dim=1) > 0.1
        torch.mul(reward, moving, out=out)

    return run


define_term(
    "illegal_contact",
    Stage.TERMINATION,
    params={"threshold": REQUIRED, "contact_bodies": REQUIRED},
    reads=("contact_net_normal_forces_history",),
    doc="Any selected body had a normal contact force above the threshold in the sensor history [N].",
)


@implement("illegal_contact", "warp")
def _(ctx: TermContext):
    ids, n = ids_vector(ctx.params["contact_ids"]), len(ctx.params["contact_ids"])
    threshold = float(ctx.params["threshold"])

    @wp.func
    def term(env: int, f: Any) -> bool:
        history = f.contact_net_normal_forces_history
        hit = bool(False)
        for h in range(history.shape[1]):
            for k in range(wp.static(n)):
                b = ids[k]
                x = history[env, h, b, 0]
                y = history[env, h, b, 1]
                z = history[env, h, b, 2]
                if wp.sqrt(x * x + y * y + z * z) > wp.static(threshold):
                    hit = True
        return hit

    return term


@implement("illegal_contact", "torch")
def _(ctx: TermContext):
    history, ids, out = ctx.fields["contact_net_normal_forces_history"], ctx.indices["contact_ids"], ctx.out
    threshold = f32(ctx.params["threshold"])

    def run():
        force = torch.linalg.vector_norm(history[:, :, ids], dim=-1)
        torch.any(torch.amax(force, dim=1) > threshold, dim=1, out=out)

    return run
