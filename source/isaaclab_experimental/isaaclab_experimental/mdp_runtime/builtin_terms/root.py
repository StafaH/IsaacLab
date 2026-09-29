# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Root-state terms of floating-base articulations.

Fields: ``root_pose_w`` ``(N, 7)`` is the root link pose, ``root_vel_w`` ``(N, 6)`` the root COM velocity
``[lin, ang]`` in the world frame. Body-frame quantities rotate by the root link orientation, as Isaac Lab's
``root_lin_vel_b`` and ``root_ang_vel_b`` do. Projected gravity assumes gravity along ``-z``.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul

from .. import torch_backend as tb
from ..terms import REQUIRED, Stage, TermContext, define_term, implement
from ..wp_math import (
    projected_gravity_b,
    rng_next,
    rng_uniform,
    root_ang_vel_b,
    root_lin_vel_b,
    row_quat,
    row_vec3,
)
from ..wp_math import (
    quat_from_euler_xyz as wp_quat_from_euler_xyz,
)

_ROOT = ("root_pose_w", "root_vel_w")
_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def _gravity(pose: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    return quat_apply_inverse(pose[:, 3:7], down.expand(pose.shape[0], 3))


def _down(ctx: TermContext) -> torch.Tensor:
    """The gravity direction, allocated at bind time (no host copy during capture)."""
    return torch.tensor([0.0, 0.0, -1.0], device=ctx.info.device)


def _torch_lin_vel_b(f) -> torch.Tensor:
    return quat_apply_inverse(f["root_pose_w"][:, 3:7], f["root_vel_w"][:, 0:3])


def _torch_ang_vel_b(f) -> torch.Tensor:
    return quat_apply_inverse(f["root_pose_w"][:, 3:7], f["root_vel_w"][:, 3:6])


# -- observations --------------------------------------------------------------------------------------

_VECTORS = {
    "base_lin_vel": (
        root_lin_vel_b,
        lambda f, down: _torch_lin_vel_b(f),
        "Root COM linear velocity in the root frame.",
    ),
    "base_ang_vel": (root_ang_vel_b, lambda f, down: _torch_ang_vel_b(f), "Root angular velocity in the root frame."),
    "projected_gravity": (
        projected_gravity_b,
        lambda f, down: _gravity(f["root_pose_w"], down),
        "Gravity in the root frame.",
    ),
}


def _define_vector(name: str, wp_value, torch_value, doc: str):
    define_term(name, Stage.OBSERVATION, reads=_ROOT, width=3, doc=doc)

    @implement(name, "warp")
    def _(ctx: TermContext):
        col = ctx.columns[0]

        @wp.func
        def term(env: int, f: Any, out: wp.array2d(dtype=wp.float32)):
            v = wp.static(wp_value)(f, env)
            out[env, col] = v[0]
            out[env, col + 1] = v[1]
            out[env, col + 2] = v[2]

        return term

    @implement(name, "torch")
    def _(ctx: TermContext):
        fields, out, down = ctx.fields, ctx.out, _down(ctx)
        return lambda: out.copy_(torch_value(fields, down))


for _name, (_wp, _torch, _doc) in _VECTORS.items():
    _define_vector(_name, _wp, _torch, _doc)


# -- rewards -------------------------------------------------------------------------------------------

define_term("lin_vel_z_l2", Stage.REWARD, reads=_ROOT, doc="Squared vertical base velocity (root frame).")
define_term("ang_vel_xy_l2", Stage.REWARD, reads=_ROOT, doc="Squared roll and pitch rates (root frame).")
define_term("flat_orientation_l2", Stage.REWARD, reads=_ROOT, doc="Squared xy components of projected gravity.")


@implement("lin_vel_z_l2", "warp")
def _(ctx: TermContext):
    @wp.func
    def term(env: int, f: Any) -> float:
        v = root_lin_vel_b(f, env)
        return v[2] * v[2]

    return term


@implement("ang_vel_xy_l2", "warp")
def _(ctx: TermContext):
    @wp.func
    def term(env: int, f: Any) -> float:
        w = root_ang_vel_b(f, env)
        return w[0] * w[0] + w[1] * w[1]

    return term


@implement("flat_orientation_l2", "warp")
def _(ctx: TermContext):
    @wp.func
    def term(env: int, f: Any) -> float:
        g = projected_gravity_b(f, env)
        return g[0] * g[0] + g[1] * g[1]

    return term


@implement("lin_vel_z_l2", "torch")
def _(ctx: TermContext):
    fields, out = ctx.fields, ctx.out
    return lambda: torch.square(_torch_lin_vel_b(fields)[:, 2], out=out)


@implement("ang_vel_xy_l2", "torch")
def _(ctx: TermContext):
    fields, out = ctx.fields, ctx.out
    return lambda: torch.sum(torch.square(_torch_ang_vel_b(fields)[:, :2]), dim=1, out=out)


@implement("flat_orientation_l2", "torch")
def _(ctx: TermContext):
    fields, out, down = ctx.fields, ctx.out, _down(ctx)
    return lambda: torch.sum(torch.square(_gravity(fields["root_pose_w"], down)[:, :2]), dim=1, out=out)


define_term(
    "track_lin_vel_xy_exp",
    Stage.REWARD,
    params={"std": REQUIRED, "command": REQUIRED},
    reads=(*_ROOT, "commands"),
    doc="exp(-|v_cmd_xy - v_xy|^2 / std^2) with the root-frame linear velocity.",
)
define_term(
    "track_ang_vel_z_exp",
    Stage.REWARD,
    params={"std": REQUIRED, "command": REQUIRED},
    reads=(*_ROOT, "commands"),
    doc="exp(-(w_cmd_z - w_z)^2 / std^2) with the root-frame angular velocity.",
)


@implement("track_lin_vel_xy_exp", "warp")
def _(ctx: TermContext):
    col, variance = ctx.params["command_columns"][0], float(ctx.params["std"]) ** 2

    @wp.func
    def term(env: int, f: Any) -> float:
        v = root_lin_vel_b(f, env)
        dx = f.commands[env, col] - v[0]
        dy = f.commands[env, col + 1] - v[1]
        return wp.exp(-(dx * dx + dy * dy) / wp.static(variance))

    return term


@implement("track_ang_vel_z_exp", "warp")
def _(ctx: TermContext):
    col, variance = ctx.params["command_columns"][0], float(ctx.params["std"]) ** 2

    @wp.func
    def term(env: int, f: Any) -> float:
        d = f.commands[env, col + 2] - root_ang_vel_b(f, env)[2]
        return wp.exp(-(d * d) / wp.static(variance))

    return term


@implement("track_lin_vel_xy_exp", "torch")
def _(ctx: TermContext):
    fields, out, col = ctx.fields, ctx.out, ctx.params["command_columns"][0]
    variance = float(ctx.params["std"]) ** 2

    def run():
        error = torch.sum(torch.square(fields["commands"][:, col : col + 2] - _torch_lin_vel_b(fields)[:, :2]), dim=1)
        torch.exp(-error / variance, out=out)

    return run


@implement("track_ang_vel_z_exp", "torch")
def _(ctx: TermContext):
    fields, out, col = ctx.fields, ctx.out, ctx.params["command_columns"][0]
    variance = float(ctx.params["std"]) ** 2

    def run():
        error = torch.square(fields["commands"][:, col + 2] - _torch_ang_vel_b(fields)[:, 2])
        torch.exp(-error / variance, out=out)

    return run


# -- events --------------------------------------------------------------------------------------------


def _ranges(value: dict) -> list[tuple[float, float]]:
    return [tuple(float(b) for b in value.get(axis, (0.0, 0.0))) for axis in _AXES]


def _check_axes(ctx: TermContext, *keys: str) -> None:
    for key in keys:
        unknown = set(ctx.params[key]) - set(_AXES)
        if unknown:
            raise ValueError(f"'{key}' has unknown axes {sorted(unknown)}; expected a subset of {_AXES}.")


define_term(
    "reset_root_state_uniform",
    Stage.EVENT,
    params={"pose_range": REQUIRED, "velocity_range": REQUIRED},
    reads=("default_root_pose", "default_root_vel", "env_origins"),
    writes=_ROOT,
    doc="Default root pose (plus env origin) offset by uniform xyz and XYZ-Euler samples; default velocity plus "
    "uniform samples. Draws 6 pose then 6 velocity values; missing axes draw from (0, 0).",
)


@implement("reset_root_state_uniform", "warp")
def _(ctx: TermContext):
    _check_axes(ctx, "pose_range", "velocity_range")
    pose = _ranges(ctx.params["pose_range"])
    vel = _ranges(ctx.params["velocity_range"])
    pose_lo = wp.types.vector(length=6, dtype=wp.float32)(*[p[0] for p in pose])
    pose_hi = wp.types.vector(length=6, dtype=wp.float32)(*[p[1] for p in pose])
    vel_lo = wp.types.vector(length=6, dtype=wp.float32)(*[v[0] for v in vel])
    vel_hi = wp.types.vector(length=6, dtype=wp.float32)(*[v[1] for v in vel])

    @wp.func
    def term(env: int, f: Any, s: wp.uint32) -> wp.uint32:
        u = wp.vector(dtype=wp.float32, length=6)
        for k in range(6):
            s = rng_next(s)
            u[k] = rng_uniform(s, pose_lo[k], pose_hi[k])
        position = row_vec3(f.default_root_pose, env, 0) + row_vec3(f.env_origins, env, 0) + wp.vec3(u[0], u[1], u[2])
        orientation = row_quat(f.default_root_pose, env, 3) * wp_quat_from_euler_xyz(u[3], u[4], u[5])
        for k in range(3):
            f.root_pose_w[env, k] = position[k]
        for k in range(4):
            f.root_pose_w[env, 3 + k] = orientation[k]
        for k in range(6):
            s = rng_next(s)
            f.root_vel_w[env, k] = f.default_root_vel[env, k] + rng_uniform(s, vel_lo[k], vel_hi[k])
        return s

    return term


def _torch_samples(rng: torch.Tensor, mask: torch.Tensor, ranges: list[tuple[float, float]]) -> torch.Tensor:
    return torch.stack([tb.draw(rng, mask, lo, hi) for lo, hi in ranges], dim=1)


@implement("reset_root_state_uniform", "torch")
def _(ctx: TermContext):
    _check_axes(ctx, "pose_range", "velocity_range")
    f, rng = ctx.fields, ctx.rng
    pose, vel = _ranges(ctx.params["pose_range"]), _ranges(ctx.params["velocity_range"])

    def run(mask: torch.Tensor):
        rows = mask[:, None]
        u = _torch_samples(rng, mask, pose)
        position = f["default_root_pose"][:, :3] + f["env_origins"] + u[:, :3]
        orientation = quat_mul(f["default_root_pose"][:, 3:7], quat_from_euler_xyz(u[:, 3], u[:, 4], u[:, 5]))
        velocity = f["default_root_vel"] + _torch_samples(rng, mask, vel)
        f["root_pose_w"].copy_(torch.where(rows, torch.cat([position, orientation], dim=1), f["root_pose_w"]))
        f["root_vel_w"].copy_(torch.where(rows, velocity, f["root_vel_w"]))

    return run


define_term(
    "push_by_setting_velocity",
    Stage.EVENT,
    params={"velocity_range": REQUIRED},
    reads=("root_vel_w",),
    writes=("root_vel_w",),
    doc="Add uniform samples to the root COM velocity. Draws 6 values; missing axes draw from (0, 0).",
)


@implement("push_by_setting_velocity", "warp")
def _(ctx: TermContext):
    _check_axes(ctx, "velocity_range")
    vel = _ranges(ctx.params["velocity_range"])
    lo = wp.types.vector(length=6, dtype=wp.float32)(*[v[0] for v in vel])
    hi = wp.types.vector(length=6, dtype=wp.float32)(*[v[1] for v in vel])

    @wp.func
    def term(env: int, f: Any, s: wp.uint32) -> wp.uint32:
        for k in range(6):
            s = rng_next(s)
            f.root_vel_w[env, k] = f.root_vel_w[env, k] + rng_uniform(s, lo[k], hi[k])
        return s

    return term


@implement("push_by_setting_velocity", "torch")
def _(ctx: TermContext):
    _check_axes(ctx, "velocity_range")
    root_vel, rng, vel = ctx.fields["root_vel_w"], ctx.rng, _ranges(ctx.params["velocity_range"])

    def run(mask: torch.Tensor):
        offset = _torch_samples(rng, mask, vel)
        root_vel.copy_(torch.where(mask[:, None], root_vel + offset, root_vel))

    return run
