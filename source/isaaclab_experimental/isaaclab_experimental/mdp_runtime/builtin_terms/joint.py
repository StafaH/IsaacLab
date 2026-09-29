# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-space, action, episode, and termination-derived terms."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import warp as wp

from isaaclab.utils.math import wrap_to_pi as torch_wrap_to_pi

from .. import torch_backend as tb
from ..terms import REQUIRED, Stage, TermContext, define_term, implement
from ..wp_math import rng_next, rng_uniform, wrap_to_pi


def ids_vector(ids) -> Any:
    """A compile-time Warp vector constant holding indices."""
    return wp.types.vector(length=len(ids), dtype=wp.int32)(*ids)


def f32(value: float) -> float:
    """Round a Python float to float32, as a Warp constant is."""
    return float(np.float32(value))


def _count(key: str):
    return lambda params, info: len(params[key])


# -- actions -------------------------------------------------------------------------------------------

define_term(
    "joint_effort",
    Stage.ACTION,
    params={"joints": ".*"},
    writes=("joint_effort_target",),
    width=_count("joint_ids"),
    doc="Write processed actions to the effort targets of the selected joints.",
)


@implement("joint_effort", "warp")
def _(ctx: TermContext):
    ids, col, n = ids_vector(ctx.params["joint_ids"]), ctx.columns[0], len(ctx.params["joint_ids"])

    @wp.func
    def term(env: int, f: Any):
        for k in range(wp.static(n)):
            f.joint_effort_target[env, ids[k]] = f.processed_actions[env, col + k]

    return term


@implement("joint_effort", "torch")
def _(ctx: TermContext):
    ids, target, action = ctx.indices["joint_ids"], ctx.fields["joint_effort_target"], ctx.out

    def run():
        target[:, ids] = action

    return run


define_term(
    "joint_position",
    Stage.ACTION,
    params={"joints": ".*", "use_default_offset": True},
    reads=("default_joint_pos",),
    writes=("joint_pos_target",),
    width=_count("joint_ids"),
    doc="Position targets: processed action, plus the default joint position if use_default_offset.",
)


@implement("joint_position", "warp")
def _(ctx: TermContext):
    ids, col, n = ids_vector(ctx.params["joint_ids"]), ctx.columns[0], len(ctx.params["joint_ids"])
    offset = bool(ctx.params["use_default_offset"])

    @wp.func
    def term(env: int, f: Any):
        for k in range(wp.static(n)):
            j = ids[k]
            target = f.processed_actions[env, col + k]
            if wp.static(offset):
                target = target + f.default_joint_pos[env, j]
            f.joint_pos_target[env, j] = target

    return term


@implement("joint_position", "torch")
def _(ctx: TermContext):
    ids, action = ctx.indices["joint_ids"], ctx.out
    target, default = ctx.fields["joint_pos_target"], ctx.fields["default_joint_pos"]
    offset = bool(ctx.params["use_default_offset"])

    def run():
        target[:, ids] = action + default[:, ids] if offset else action

    return run


# -- observations --------------------------------------------------------------------------------------

for _name, _field in (("joint_pos_rel", "joint_pos"), ("joint_vel_rel", "joint_vel")):
    define_term(
        _name,
        Stage.OBSERVATION,
        params={"joints": ".*"},
        reads=(_field, f"default_{_field}"),
        width=_count("joint_ids"),
        doc=f"Selected {_field} relative to the default values.",
    )


@implement("joint_pos_rel", "warp")
def _(ctx: TermContext):
    ids, col, n = ids_vector(ctx.params["joint_ids"]), ctx.columns[0], len(ctx.params["joint_ids"])

    @wp.func
    def term(env: int, f: Any, out: wp.array2d(dtype=wp.float32)):
        for k in range(wp.static(n)):
            out[env, col + k] = f.joint_pos[env, ids[k]] - f.default_joint_pos[env, ids[k]]

    return term


@implement("joint_vel_rel", "warp")
def _(ctx: TermContext):
    ids, col, n = ids_vector(ctx.params["joint_ids"]), ctx.columns[0], len(ctx.params["joint_ids"])

    @wp.func
    def term(env: int, f: Any, out: wp.array2d(dtype=wp.float32)):
        for k in range(wp.static(n)):
            out[env, col + k] = f.joint_vel[env, ids[k]] - f.default_joint_vel[env, ids[k]]

    return term


def _torch_relative(field: str):
    def binder(ctx: TermContext):
        value, default, ids, out = ctx.fields[field], ctx.fields[f"default_{field}"], ctx.indices["joint_ids"], ctx.out
        return lambda: torch.sub(value[:, ids], default[:, ids], out=out)

    return binder


implement("joint_pos_rel", "torch")(_torch_relative("joint_pos"))
implement("joint_vel_rel", "torch")(_torch_relative("joint_vel"))

define_term(
    "last_action",
    Stage.OBSERVATION,
    reads=("action",),
    width=lambda params, info: info.num_actions,
    doc="Raw action of the current step.",
)


@implement("last_action", "warp")
def _(ctx: TermContext):
    col, n = ctx.columns[0], ctx.info.num_actions

    @wp.func
    def term(env: int, f: Any, out: wp.array2d(dtype=wp.float32)):
        for k in range(wp.static(n)):
            out[env, col + k] = f.action[env, k]

    return term


@implement("last_action", "torch")
def _(ctx: TermContext):
    action, out = ctx.fields["action"], ctx.out
    return lambda: out.copy_(action)


# -- rewards -------------------------------------------------------------------------------------------

define_term("is_alive", Stage.REWARD, reads=("terminated",), doc="1 unless the environment terminated.")
define_term("is_terminated", Stage.REWARD, reads=("terminated",), doc="1 if the environment terminated.")


@implement("is_alive", "warp")
def _(ctx: TermContext):
    @wp.func
    def term(env: int, f: Any) -> float:
        return wp.where(f.terminated[env], 0.0, 1.0)

    return term


@implement("is_terminated", "warp")
def _(ctx: TermContext):
    @wp.func
    def term(env: int, f: Any) -> float:
        return wp.where(f.terminated[env], 1.0, 0.0)

    return term


@implement("is_alive", "torch")
def _(ctx: TermContext):
    terminated, out = ctx.fields["terminated"], ctx.out
    return lambda: out.copy_(~terminated)


@implement("is_terminated", "torch")
def _(ctx: TermContext):
    terminated, out = ctx.fields["terminated"], ctx.out
    return lambda: out.copy_(terminated)


define_term(
    "termination_term",
    Stage.REWARD,
    params={"terms": REQUIRED},
    reads=("termination_values", "truncated"),
    doc="Number of the given termination terms that are set, unless truncated (stable is_terminated_term).",
)


@implement("termination_term", "warp")
def _(ctx: TermContext):
    ids, n = ids_vector(ctx.params["term_ids"]), len(ctx.params["term_ids"])

    @wp.func
    def term(env: int, f: Any) -> float:
        total = float(0.0)
        for k in range(wp.static(n)):
            total = total + wp.where(f.termination_values[ids[k], env], 1.0, 0.0)
        return wp.where(f.truncated[env], 0.0, total)

    return term


@implement("termination_term", "torch")
def _(ctx: TermContext):
    values, truncated, ids, out = (
        ctx.fields["termination_values"],
        ctx.fields["truncated"],
        ctx.indices["term_ids"],
        ctx.out,
    )
    return lambda: torch.mul(values[ids].sum(dim=0), ~truncated, out=out)


define_term(
    "joint_pos_target_l2",
    Stage.REWARD,
    params={"target": REQUIRED, "joints": ".*"},
    reads=("joint_pos",),
    doc="Sum of squared deviations of wrapped joint positions from a target [rad^2].",
)


@implement("joint_pos_target_l2", "warp")
def _(ctx: TermContext):
    ids, n, target = ids_vector(ctx.params["joint_ids"]), len(ctx.params["joint_ids"]), float(ctx.params["target"])

    @wp.func
    def term(env: int, f: Any) -> float:
        total = float(0.0)
        for k in range(wp.static(n)):
            error = wrap_to_pi(f.joint_pos[env, ids[k]]) - wp.static(target)
            total = total + error * error
        return total

    return term


@implement("joint_pos_target_l2", "torch")
def _(ctx: TermContext):
    joint_pos, ids, target, out = ctx.fields["joint_pos"], ctx.indices["joint_ids"], f32(ctx.params["target"]), ctx.out
    return lambda: torch.sum(torch.square(torch_wrap_to_pi(joint_pos[:, ids]) - target), dim=1, out=out)


@wp.func
def _joint_vel(f: Any, env: int, j: int) -> float:
    return f.joint_vel[env, j]


@wp.func
def _joint_acc(f: Any, env: int, j: int) -> float:
    return f.joint_acc[env, j]


@wp.func
def _applied_effort(f: Any, env: int, j: int) -> float:
    return f.applied_effort[env, j]


def _define_joint_norm(name: str, field: str, read, squared: bool, doc: str):
    define_term(name, Stage.REWARD, params={"joints": ".*"}, reads=(field,), doc=doc)

    @implement(name, "warp")
    def _(ctx: TermContext):
        ids, n = ids_vector(ctx.params["joint_ids"]), len(ctx.params["joint_ids"])

        @wp.func
        def term(env: int, f: Any) -> float:
            total = float(0.0)
            for k in range(wp.static(n)):
                v = wp.static(read)(f, env, ids[k])
                if wp.static(squared):
                    total = total + v * v
                else:
                    total = total + wp.abs(v)
            return total

        return term

    @implement(name, "torch")
    def _(ctx: TermContext):
        value, ids, out = ctx.fields[field], ctx.indices["joint_ids"], ctx.out
        op = torch.square if squared else torch.abs
        return lambda: torch.sum(op(value[:, ids]), dim=1, out=out)


_define_joint_norm("joint_vel_l1", "joint_vel", _joint_vel, False, "L1 norm of the selected joint velocities.")
_define_joint_norm("joint_vel_l2", "joint_vel", _joint_vel, True, "Squared L2 norm of the selected joint velocities.")
_define_joint_norm("joint_acc_l2", "joint_acc", _joint_acc, True, "Squared L2 norm of joint accelerations.")
_define_joint_norm("joint_torques_l2", "applied_effort", _applied_effort, True, "Squared L2 norm of applied efforts.")

define_term("action_l2", Stage.REWARD, reads=("action",), doc="Squared L2 norm of the raw action.")
define_term(
    "action_rate_l2", Stage.REWARD, reads=("action", "prev_action"), doc="Squared L2 norm of the raw action change."
)


@implement("action_l2", "warp")
def _(ctx: TermContext):
    n = ctx.info.num_actions

    @wp.func
    def term(env: int, f: Any) -> float:
        total = float(0.0)
        for k in range(wp.static(n)):
            total = total + f.action[env, k] * f.action[env, k]
        return total

    return term


@implement("action_rate_l2", "warp")
def _(ctx: TermContext):
    n = ctx.info.num_actions

    @wp.func
    def term(env: int, f: Any) -> float:
        total = float(0.0)
        for k in range(wp.static(n)):
            d = f.action[env, k] - f.prev_action[env, k]
            total = total + d * d
        return total

    return term


@implement("action_l2", "torch")
def _(ctx: TermContext):
    action, out = ctx.fields["action"], ctx.out
    return lambda: torch.sum(torch.square(action), dim=1, out=out)


@implement("action_rate_l2", "torch")
def _(ctx: TermContext):
    action, prev_action, out = ctx.fields["action"], ctx.fields["prev_action"], ctx.out
    return lambda: torch.sum(torch.square(action - prev_action), dim=1, out=out)


# -- terminations --------------------------------------------------------------------------------------

define_term("time_out", Stage.TERMINATION, reads=("episode_length",), doc="Episode length reached the maximum.")


@implement("time_out", "warp")
def _(ctx: TermContext):
    limit = int(ctx.info.max_episode_length)

    @wp.func
    def term(env: int, f: Any) -> bool:
        return f.episode_length[env] >= wp.static(limit)

    return term


@implement("time_out", "torch")
def _(ctx: TermContext):
    episode_length, limit, out = ctx.fields["episode_length"], ctx.info.max_episode_length, ctx.out
    return lambda: torch.ge(episode_length, limit, out=out)


define_term(
    "joint_pos_out_of_manual_limit",
    Stage.TERMINATION,
    params={"bounds": REQUIRED, "joints": ".*"},
    reads=("joint_pos",),
    doc="Any selected joint position is outside the given bounds.",
)


@implement("joint_pos_out_of_manual_limit", "warp")
def _(ctx: TermContext):
    ids, n = ids_vector(ctx.params["joint_ids"]), len(ctx.params["joint_ids"])
    lower, upper = (float(b) for b in ctx.params["bounds"])

    @wp.func
    def term(env: int, f: Any) -> bool:
        violated = bool(False)
        for k in range(wp.static(n)):
            q = f.joint_pos[env, ids[k]]
            if q > wp.static(upper) or q < wp.static(lower):
                violated = True
        return violated

    return term


@implement("joint_pos_out_of_manual_limit", "torch")
def _(ctx: TermContext):
    joint_pos, ids, out = ctx.fields["joint_pos"], ctx.indices["joint_ids"], ctx.out
    lower, upper = (f32(b) for b in ctx.params["bounds"])

    def run():
        q = joint_pos[:, ids]
        torch.any((q > upper) | (q < lower), dim=1, out=out)

    return run


# -- events --------------------------------------------------------------------------------------------

_RESET_READS = ("default_joint_pos", "default_joint_vel", "soft_joint_pos_limits", "soft_joint_vel_limits")

define_term(
    "reset_joints_by_offset",
    Stage.EVENT,
    params={"position_range": REQUIRED, "velocity_range": REQUIRED, "joints": ".*"},
    reads=_RESET_READS,
    writes=("joint_pos", "joint_vel"),
    doc="Default joint state plus uniform offsets, clamped to the soft limits. Draws positions, then velocities.",
)
define_term(
    "reset_joints_by_scale",
    Stage.EVENT,
    params={"position_range": REQUIRED, "velocity_range": REQUIRED, "joints": ".*"},
    reads=_RESET_READS,
    writes=("joint_pos", "joint_vel"),
    doc="Default joint state times uniform factors, clamped to the soft limits. Draws positions, then velocities.",
)


def _warp_reset_joints(scale: bool):
    def factory(ctx: TermContext):
        ids, n = ids_vector(ctx.params["joint_ids"]), len(ctx.params["joint_ids"])
        (pos_lo, pos_hi), (vel_lo, vel_hi) = ctx.params["position_range"], ctx.params["velocity_range"]
        pos_lo, pos_hi, vel_lo, vel_hi = float(pos_lo), float(pos_hi), float(vel_lo), float(vel_hi)

        @wp.func
        def term(env: int, f: Any, s: wp.uint32) -> wp.uint32:
            for k in range(wp.static(n)):
                j = ids[k]
                s = rng_next(s)
                u = rng_uniform(s, wp.static(pos_lo), wp.static(pos_hi))
                q = wp.where(wp.static(scale), f.default_joint_pos[env, j] * u, f.default_joint_pos[env, j] + u)
                f.joint_pos[env, j] = wp.clamp(
                    q, f.soft_joint_pos_limits[env, j, 0], f.soft_joint_pos_limits[env, j, 1]
                )
            for k in range(wp.static(n)):
                j = ids[k]
                s = rng_next(s)
                u = rng_uniform(s, wp.static(vel_lo), wp.static(vel_hi))
                v = wp.where(wp.static(scale), f.default_joint_vel[env, j] * u, f.default_joint_vel[env, j] + u)
                limit = f.soft_joint_vel_limits[env, j]
                f.joint_vel[env, j] = wp.clamp(v, -limit, limit)
            return s

        return term

    return factory


def _draw_columns(rng: torch.Tensor, mask: torch.Tensor, count: int, lo: float, hi: float) -> torch.Tensor:
    """Draw ``count`` uniforms per environment in stream order, advancing only the masked streams."""
    return torch.stack([tb.draw(rng, mask, lo, hi) for _ in range(count)], dim=1)


def _torch_reset_joints(scale: bool):
    def binder(ctx: TermContext):
        f, ids, rng = ctx.fields, ctx.indices["joint_ids"], ctx.rng
        count = len(ctx.params["joint_ids"])
        (pos_lo, pos_hi), (vel_lo, vel_hi) = ctx.params["position_range"], ctx.params["velocity_range"]
        combine = torch.mul if scale else torch.add

        def run(mask: torch.Tensor):
            rows = mask[:, None]
            pos = _draw_columns(rng, mask, count, pos_lo, pos_hi)
            vel = _draw_columns(rng, mask, count, vel_lo, vel_hi)
            limits = f["soft_joint_pos_limits"][:, ids]
            q = torch.clamp(combine(f["default_joint_pos"][:, ids], pos), limits[..., 0], limits[..., 1])
            vel_limit = f["soft_joint_vel_limits"][:, ids]
            v = torch.clamp(combine(f["default_joint_vel"][:, ids], vel), -vel_limit, vel_limit)
            f["joint_pos"][:, ids] = torch.where(rows, q, f["joint_pos"][:, ids])
            f["joint_vel"][:, ids] = torch.where(rows, v, f["joint_vel"][:, ids])

        return run

    return binder


implement("reset_joints_by_offset", "warp")(_warp_reset_joints(False))
implement("reset_joints_by_scale", "warp")(_warp_reset_joints(True))
implement("reset_joints_by_offset", "torch")(_torch_reset_joints(False))
implement("reset_joints_by_scale", "torch")(_torch_reset_joints(True))

define_term(
    "push_joints_by_velocity",
    Stage.EVENT,
    params={"velocity_range": REQUIRED, "joints": ".*"},
    reads=("joint_vel",),
    writes=("joint_vel",),
    doc="Add uniform offsets to the selected joint velocities.",
)


@implement("push_joints_by_velocity", "warp")
def _(ctx: TermContext):
    ids, n = ids_vector(ctx.params["joint_ids"]), len(ctx.params["joint_ids"])
    lo, hi = (float(v) for v in ctx.params["velocity_range"])

    @wp.func
    def term(env: int, f: Any, s: wp.uint32) -> wp.uint32:
        for k in range(wp.static(n)):
            s = rng_next(s)
            f.joint_vel[env, ids[k]] = f.joint_vel[env, ids[k]] + rng_uniform(s, wp.static(lo), wp.static(hi))
        return s

    return term


@implement("push_joints_by_velocity", "torch")
def _(ctx: TermContext):
    ids, rng, joint_vel = ctx.indices["joint_ids"], ctx.rng, ctx.fields["joint_vel"]
    count, (lo, hi) = len(ctx.params["joint_ids"]), ctx.params["velocity_range"]

    def run(mask: torch.Tensor):
        offset = _draw_columns(rng, mask, count, lo, hi)
        joint_vel[:, ids] = torch.where(mask[:, None], joint_vel[:, ids] + offset, joint_vel[:, ids])

    return run
