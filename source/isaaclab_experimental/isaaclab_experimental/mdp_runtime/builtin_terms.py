# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Built-in joint-space terms with Warp and Torch implementations.

The semantics follow the stable :mod:`isaaclab.envs.mdp` functions of the same name. Terms operate on the
joint-space physics fields documented in :mod:`~isaaclab_experimental.mdp_runtime.physics`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp

from isaaclab.utils.math import wrap_to_pi as torch_wrap_to_pi

from . import torch_backend as tb
from . import warp_backend as wb
from .terms import REQUIRED, Stage, TermContext, define_term, implement


def _joint_count(params) -> int:
    return len(params["joint_ids"])


def _record(kernel, dim, inputs, ctx: TermContext):
    return wp.launch(kernel, dim=dim, inputs=inputs, device=ctx.device, record_cmd=True).launch


def _f32(value: float) -> float:
    """Round a Python float to float32, as a Warp kernel argument is."""
    return float(np.float32(value))


@wp.func
def wrap_to_pi(angle: wp.float32) -> wp.float32:
    """Wrap an angle to ``[-pi, pi]`` like :func:`isaaclab.utils.math.wrap_to_pi`."""
    two_pi = wp.float32(2.0 * math.pi)
    shifted = angle + wp.float32(math.pi)
    wrapped = shifted - two_pi * wp.floor(shifted / two_pi)
    if wrapped == 0.0 and angle > 0.0:
        return wp.float32(math.pi)
    return wrapped - wp.float32(math.pi)


# -- actions -------------------------------------------------------------------------------------------

define_term(
    "joint_effort",
    Stage.ACTION,
    params={"joints": ".*"},
    writes=("joint_effort_target",),
    width=_joint_count,
    doc="Write processed actions to the effort targets of the selected joints.",
)


@wp.kernel
def _joint_effort(
    action: wp.array2d(dtype=wp.float32), joint_ids: wp.array(dtype=wp.int32), target: wp.array2d(dtype=wp.float32)
):
    i, j = wp.tid()
    target[i, joint_ids[j]] = action[i, j]


@implement("joint_effort", "warp")
def _(ctx: TermContext):
    ids = ctx.indices["joint_ids"]
    return _record(_joint_effort, ctx.action.shape, [ctx.action, ids, ctx.fields["joint_effort_target"]], ctx)


@implement("joint_effort", "torch")
def _(ctx: TermContext):
    ids, target, action = ctx.indices["joint_ids"], ctx.fields["joint_effort_target"], ctx.action

    def run():
        target[:, ids] = action

    return run


# -- observations --------------------------------------------------------------------------------------

for _name, _field in (("joint_pos_rel", "joint_pos"), ("joint_vel_rel", "joint_vel")):
    define_term(
        _name,
        Stage.OBSERVATION,
        params={"joints": ".*"},
        reads=(_field, f"default_{_field}"),
        width=_joint_count,
        doc=f"Selected {_field} relative to the default values.",
    )


@wp.kernel
def _relative(
    value: wp.array2d(dtype=wp.float32),
    default: wp.array2d(dtype=wp.float32),
    joint_ids: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    out[i, j] = value[i, joint_ids[j]] - default[i, joint_ids[j]]


def _bind_relative(field: str):
    def warp_binder(ctx: TermContext):
        inputs = [ctx.fields[field], ctx.fields[f"default_{field}"], ctx.indices["joint_ids"], ctx.out]
        return _record(_relative, ctx.out.shape, inputs, ctx)

    def torch_binder(ctx: TermContext):
        value, default, ids, out = ctx.fields[field], ctx.fields[f"default_{field}"], ctx.indices["joint_ids"], ctx.out
        return lambda: torch.sub(value[:, ids], default[:, ids], out=out)

    implement(f"{field}_rel", "warp")(warp_binder)
    implement(f"{field}_rel", "torch")(torch_binder)


_bind_relative("joint_pos")
_bind_relative("joint_vel")


# -- rewards -------------------------------------------------------------------------------------------

define_term("is_alive", Stage.REWARD, reads=("terminated",), doc="1 unless the environment terminated.")
define_term("is_terminated", Stage.REWARD, reads=("terminated",), doc="1 if the environment terminated.")


@wp.kernel
def _flag(flags: wp.array(dtype=wp.bool), value_if_set: wp.float32, out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.where(flags[i], value_if_set, 1.0 - value_if_set)


@implement("is_alive", "warp")
def _(ctx: TermContext):
    return _record(_flag, ctx.num_envs, [ctx.fields["terminated"], 0.0, ctx.out], ctx)


@implement("is_terminated", "warp")
def _(ctx: TermContext):
    return _record(_flag, ctx.num_envs, [ctx.fields["terminated"], 1.0, ctx.out], ctx)


@implement("is_alive", "torch")
def _(ctx: TermContext):
    terminated, out = ctx.fields["terminated"], ctx.out
    return lambda: out.copy_(~terminated)


@implement("is_terminated", "torch")
def _(ctx: TermContext):
    terminated, out = ctx.fields["terminated"], ctx.out
    return lambda: out.copy_(terminated)


define_term(
    "joint_pos_target_l2",
    Stage.REWARD,
    params={"target": REQUIRED, "joints": ".*"},
    reads=("joint_pos",),
    doc="Sum of squared deviations of wrapped joint positions from a target [rad^2].",
)


@wp.kernel
def _joint_pos_target_l2(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_ids: wp.array(dtype=wp.int32),
    target: wp.float32,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    total = wp.float32(0.0)
    for j in range(joint_ids.shape[0]):
        error = wrap_to_pi(joint_pos[i, joint_ids[j]]) - target
        total = total + error * error
    out[i] = total


@implement("joint_pos_target_l2", "warp")
def _(ctx: TermContext):
    inputs = [ctx.fields["joint_pos"], ctx.indices["joint_ids"], ctx.params["target"], ctx.out]
    return _record(_joint_pos_target_l2, ctx.num_envs, inputs, ctx)


@implement("joint_pos_target_l2", "torch")
def _(ctx: TermContext):
    joint_pos, ids, target, out = ctx.fields["joint_pos"], ctx.indices["joint_ids"], _f32(ctx.params["target"]), ctx.out
    return lambda: torch.sum(torch.square(torch_wrap_to_pi(joint_pos[:, ids]) - target), dim=1, out=out)


for _name, _power in (("joint_vel_l1", 1), ("joint_vel_l2", 2)):
    define_term(
        _name,
        Stage.REWARD,
        params={"joints": ".*"},
        reads=("joint_vel",),
        doc=f"L{_power} norm{' squared' if _power == 2 else ''} of the selected joint velocities.",
    )


@wp.kernel
def _joint_vel_norm(
    joint_vel: wp.array2d(dtype=wp.float32),
    joint_ids: wp.array(dtype=wp.int32),
    squared: wp.bool,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    total = wp.float32(0.0)
    for j in range(joint_ids.shape[0]):
        v = joint_vel[i, joint_ids[j]]
        total = total + wp.where(squared, v * v, wp.abs(v))
    out[i] = total


def _bind_joint_vel_norm(name: str, squared: bool):
    def warp_binder(ctx: TermContext):
        inputs = [ctx.fields["joint_vel"], ctx.indices["joint_ids"], squared, ctx.out]
        return _record(_joint_vel_norm, ctx.num_envs, inputs, ctx)

    def torch_binder(ctx: TermContext):
        joint_vel, ids, out = ctx.fields["joint_vel"], ctx.indices["joint_ids"], ctx.out
        op = torch.square if squared else torch.abs
        return lambda: torch.sum(op(joint_vel[:, ids]), dim=1, out=out)

    implement(name, "warp")(warp_binder)
    implement(name, "torch")(torch_binder)


_bind_joint_vel_norm("joint_vel_l1", False)
_bind_joint_vel_norm("joint_vel_l2", True)


define_term(
    "action_rate_l2",
    Stage.REWARD,
    reads=("action", "prev_action"),
    doc="Squared L2 norm of the change of the raw action.",
)


@wp.kernel
def _action_rate_l2(
    action: wp.array2d(dtype=wp.float32), prev_action: wp.array2d(dtype=wp.float32), out: wp.array(dtype=wp.float32)
):
    i = wp.tid()
    total = wp.float32(0.0)
    for j in range(action.shape[1]):
        d = action[i, j] - prev_action[i, j]
        total = total + d * d
    out[i] = total


@implement("action_rate_l2", "warp")
def _(ctx: TermContext):
    return _record(_action_rate_l2, ctx.num_envs, [ctx.fields["action"], ctx.fields["prev_action"], ctx.out], ctx)


@implement("action_rate_l2", "torch")
def _(ctx: TermContext):
    action, prev_action, out = ctx.fields["action"], ctx.fields["prev_action"], ctx.out
    return lambda: torch.sum(torch.square(action - prev_action), dim=1, out=out)


# -- terminations --------------------------------------------------------------------------------------

define_term(
    "time_out",
    Stage.TERMINATION,
    reads=("episode_length",),
    doc="Episode length reached the maximum episode length.",
)


@wp.kernel
def _time_out(episode_length: wp.array(dtype=wp.int32), limit: wp.int32, out: wp.array(dtype=wp.bool)):
    i = wp.tid()
    out[i] = episode_length[i] >= limit


@implement("time_out", "warp")
def _(ctx: TermContext):
    return _record(_time_out, ctx.num_envs, [ctx.fields["episode_length"], ctx.max_episode_length, ctx.out], ctx)


@implement("time_out", "torch")
def _(ctx: TermContext):
    episode_length, limit, out = ctx.fields["episode_length"], ctx.max_episode_length, ctx.out
    return lambda: torch.ge(episode_length, limit, out=out)


define_term(
    "joint_pos_out_of_manual_limit",
    Stage.TERMINATION,
    params={"bounds": REQUIRED, "joints": ".*"},
    reads=("joint_pos",),
    doc="Any selected joint position is outside the given bounds.",
)


@wp.kernel
def _joint_pos_out_of_manual_limit(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_ids: wp.array(dtype=wp.int32),
    lower: wp.float32,
    upper: wp.float32,
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    violated = bool(False)
    for j in range(joint_ids.shape[0]):
        q = joint_pos[i, joint_ids[j]]
        if q > upper or q < lower:
            violated = True
    out[i] = violated


@implement("joint_pos_out_of_manual_limit", "warp")
def _(ctx: TermContext):
    lower, upper = ctx.params["bounds"]
    inputs = [ctx.fields["joint_pos"], ctx.indices["joint_ids"], lower, upper, ctx.out]
    return _record(_joint_pos_out_of_manual_limit, ctx.num_envs, inputs, ctx)


@implement("joint_pos_out_of_manual_limit", "torch")
def _(ctx: TermContext):
    joint_pos, ids, out = ctx.fields["joint_pos"], ctx.indices["joint_ids"], ctx.out
    lower, upper = (_f32(b) for b in ctx.params["bounds"])

    def run():
        q = joint_pos[:, ids]
        torch.any((q > upper) | (q < lower), dim=1, out=out)

    return run


# -- events --------------------------------------------------------------------------------------------

define_term(
    "reset_joints_by_offset",
    Stage.EVENT,
    params={"position_range": REQUIRED, "velocity_range": REQUIRED, "joints": ".*"},
    reads=("default_joint_pos", "default_joint_vel", "soft_joint_pos_limits", "soft_joint_vel_limits"),
    writes=("joint_pos", "joint_vel"),
    doc="Default joint state plus uniform offsets, clamped to the soft limits. Draws positions, then velocities.",
)


@wp.kernel
def _reset_joints_by_offset(
    mask: wp.array(dtype=wp.bool),
    rng: wp.array(dtype=wp.uint32),
    joint_ids: wp.array(dtype=wp.int32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_vel: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array3d(dtype=wp.float32),
    soft_joint_vel_limits: wp.array2d(dtype=wp.float32),
    pos_lo: wp.float32,
    pos_hi: wp.float32,
    vel_lo: wp.float32,
    vel_hi: wp.float32,
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    if not mask[i]:
        return
    state = rng[i]
    for k in range(joint_ids.shape[0]):
        j = joint_ids[k]
        state = wb.rng_next(state)
        q = default_joint_pos[i, j] + wb.rng_uniform(state, pos_lo, pos_hi)
        joint_pos[i, j] = wp.clamp(q, soft_joint_pos_limits[i, j, 0], soft_joint_pos_limits[i, j, 1])
    for k in range(joint_ids.shape[0]):
        j = joint_ids[k]
        state = wb.rng_next(state)
        v = default_joint_vel[i, j] + wb.rng_uniform(state, vel_lo, vel_hi)
        joint_vel[i, j] = wp.clamp(v, -soft_joint_vel_limits[i, j], soft_joint_vel_limits[i, j])
    rng[i] = state


@implement("reset_joints_by_offset", "warp")
def _(ctx: TermContext):
    f = ctx.fields
    inputs = [
        ctx.mask,
        ctx.rng,
        ctx.indices["joint_ids"],
        f["default_joint_pos"],
        f["default_joint_vel"],
        f["soft_joint_pos_limits"],
        f["soft_joint_vel_limits"],
        *ctx.params["position_range"],
        *ctx.params["velocity_range"],
        f["joint_pos"],
        f["joint_vel"],
    ]
    return _record(_reset_joints_by_offset, ctx.num_envs, inputs, ctx)


def _draw_columns(state: torch.Tensor, count: int, lo: float, hi: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``count`` uniforms per environment in stream order; return the samples and the final state."""
    samples = []
    for _ in range(count):
        state = tb.rng_next(state)
        samples.append(tb.rng_uniform(state, lo, hi))
    return torch.stack(samples, dim=1), state


@implement("reset_joints_by_offset", "torch")
def _(ctx: TermContext):
    f, ids, mask, rng = ctx.fields, ctx.indices["joint_ids"], ctx.mask, ctx.rng
    rows, count = mask[:, None], len(ctx.params["joint_ids"])
    (pos_lo, pos_hi), (vel_lo, vel_hi) = ctx.params["position_range"], ctx.params["velocity_range"]

    def run():
        pos_offset, state = _draw_columns(rng, count, pos_lo, pos_hi)
        vel_offset, state = _draw_columns(state, count, vel_lo, vel_hi)
        limits = f["soft_joint_pos_limits"][:, ids]
        q = torch.clamp(f["default_joint_pos"][:, ids] + pos_offset, limits[..., 0], limits[..., 1])
        vel_limit = f["soft_joint_vel_limits"][:, ids]
        v = torch.clamp(f["default_joint_vel"][:, ids] + vel_offset, -vel_limit, vel_limit)
        f["joint_pos"][:, ids] = torch.where(rows, q, f["joint_pos"][:, ids])
        f["joint_vel"][:, ids] = torch.where(rows, v, f["joint_vel"][:, ids])
        rng.copy_(torch.where(mask, state, rng))

    return run


define_term(
    "push_joints_by_velocity",
    Stage.EVENT,
    params={"velocity_range": REQUIRED, "joints": ".*"},
    reads=("joint_vel",),
    writes=("joint_vel",),
    doc="Add uniform offsets to the selected joint velocities.",
)


@wp.kernel
def _push_joints_by_velocity(
    mask: wp.array(dtype=wp.bool),
    rng: wp.array(dtype=wp.uint32),
    joint_ids: wp.array(dtype=wp.int32),
    lo: wp.float32,
    hi: wp.float32,
    joint_vel: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    if not mask[i]:
        return
    state = rng[i]
    for k in range(joint_ids.shape[0]):
        j = joint_ids[k]
        state = wb.rng_next(state)
        joint_vel[i, j] = joint_vel[i, j] + wb.rng_uniform(state, lo, hi)
    rng[i] = state


@implement("push_joints_by_velocity", "warp")
def _(ctx: TermContext):
    inputs = [ctx.mask, ctx.rng, ctx.indices["joint_ids"], *ctx.params["velocity_range"], ctx.fields["joint_vel"]]
    return _record(_push_joints_by_velocity, ctx.num_envs, inputs, ctx)


@implement("push_joints_by_velocity", "torch")
def _(ctx: TermContext):
    ids, mask, rng, joint_vel = ctx.indices["joint_ids"], ctx.mask, ctx.rng, ctx.fields["joint_vel"]
    rows, count = mask[:, None], len(ctx.params["joint_ids"])
    lo, hi = ctx.params["velocity_range"]

    def run():
        offset, state = _draw_columns(rng, count, lo, hi)
        joint_vel[:, ids] = torch.where(rows, joint_vel[:, ids] + offset, joint_vel[:, ids])
        rng.copy_(torch.where(mask, state, rng))

    return run
