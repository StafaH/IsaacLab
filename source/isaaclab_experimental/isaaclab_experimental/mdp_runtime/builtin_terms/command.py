# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms and the observation that exposes them.

A command term owns columns of the ``commands`` buffer and, optionally, private state columns. The runtime
resamples it on reset and when its per-environment timer expires, then calls its update every step.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, wrap_to_pi

from .. import torch_backend as tb
from ..terms import REQUIRED, Stage, TermContext, define_term, implement
from ..wp_math import heading_w, rng_next, rng_uniform
from ..wp_math import quat_from_euler_xyz as wp_quat_from_euler_xyz
from ..wp_math import wrap_to_pi as wp_wrap_to_pi
from .joint import f32

define_term(
    "generated_commands",
    Stage.OBSERVATION,
    params={"command": REQUIRED},
    reads=("commands",),
    width=lambda params, info: params["command_columns"][1] - params["command_columns"][0],
    doc="The current value of a command.",
)


@implement("generated_commands", "warp")
def _(ctx: TermContext):
    col, (start, stop) = ctx.columns[0], ctx.params["command_columns"]

    @wp.func
    def term(env: int, f: Any, out: wp.array2d(dtype=wp.float32)):
        for k in range(wp.static(stop - start)):
            out[env, col + k] = f.commands[env, start + k]

    return term


@implement("generated_commands", "torch")
def _(ctx: TermContext):
    commands, out, (start, stop) = ctx.fields["commands"], ctx.out, ctx.params["command_columns"]
    return lambda: out.copy_(commands[:, start:stop])


# -- base velocity command -----------------------------------------------------------------------------

define_term(
    "uniform_velocity",
    Stage.COMMAND,
    params={
        "lin_vel_x": REQUIRED,
        "lin_vel_y": REQUIRED,
        "ang_vel_z": REQUIRED,
        "heading": (-math.pi, math.pi),
        "heading_command": False,
        "heading_control_stiffness": 1.0,
        "rel_heading_envs": 1.0,
        "rel_standing_envs": 0.0,
    },
    reads=("root_pose_w",),
    width=3,
    state_width=3,
    doc="Root-frame velocity command (vx, vy, wz), like the stable UniformVelocityCommand. Resampling draws vx, vy, "
    "wz, then the heading target, heading flag, and standing flag. Heading environments steer wz toward the "
    "target heading every step; standing environments command zero.",
)


def _velocity_constants(params):
    return (
        [float(b) for b in params["lin_vel_x"]],
        [float(b) for b in params["lin_vel_y"]],
        [float(b) for b in params["ang_vel_z"]],
        [float(b) for b in params["heading"]],
    )


@implement("uniform_velocity", "warp")
def _(ctx: TermContext):
    p = ctx.params
    (vx_lo, vx_hi), (vy_lo, vy_hi), (wz_lo, wz_hi), (h_lo, h_hi) = _velocity_constants(p)
    heading_command = bool(p["heading_command"])
    stiffness = float(p["heading_control_stiffness"])
    rel_heading, rel_standing = float(p["rel_heading_envs"]), float(p["rel_standing_envs"])
    col, state = ctx.columns[0], ctx.state_columns[0]

    @wp.func
    def resample(env: int, f: Any, s: wp.uint32) -> wp.uint32:
        s = rng_next(s)
        f.commands[env, col] = rng_uniform(s, wp.static(vx_lo), wp.static(vx_hi))
        s = rng_next(s)
        f.commands[env, col + 1] = rng_uniform(s, wp.static(vy_lo), wp.static(vy_hi))
        s = rng_next(s)
        f.commands[env, col + 2] = rng_uniform(s, wp.static(wz_lo), wp.static(wz_hi))
        if wp.static(heading_command):
            s = rng_next(s)
            f.command_state[env, state] = rng_uniform(s, wp.static(h_lo), wp.static(h_hi))
            s = rng_next(s)
            f.command_state[env, state + 1] = wp.where(rng_uniform(s, 0.0, 1.0) <= wp.static(rel_heading), 1.0, 0.0)
        s = rng_next(s)
        f.command_state[env, state + 2] = wp.where(rng_uniform(s, 0.0, 1.0) <= wp.static(rel_standing), 1.0, 0.0)
        return s

    @wp.func
    def update(env: int, f: Any):
        if wp.static(heading_command):
            if f.command_state[env, state + 1] > 0.5:
                error = wp_wrap_to_pi(f.command_state[env, state] - heading_w(f, env))
                f.commands[env, col + 2] = wp.clamp(wp.static(stiffness) * error, wp.static(wz_lo), wp.static(wz_hi))
        if f.command_state[env, state + 2] > 0.5:
            f.commands[env, col] = 0.0
            f.commands[env, col + 1] = 0.0
            f.commands[env, col + 2] = 0.0

    return resample, update


@implement("uniform_velocity", "torch")
def _(ctx: TermContext):
    p, out, state, rng, pose = ctx.params, ctx.out, ctx.state, ctx.rng, ctx.fields["root_pose_w"]
    ranges = _velocity_constants(p)
    heading_command = bool(p["heading_command"])
    stiffness = f32(p["heading_control_stiffness"])
    rel_heading, rel_standing = f32(p["rel_heading_envs"]), f32(p["rel_standing_envs"])
    wz_lo, wz_hi = f32(ranges[2][0]), f32(ranges[2][1])
    forward = torch.tensor([1.0, 0.0, 0.0], device=out.device)

    def resample(mask: torch.Tensor):
        for k in range(3):
            out[:, k] = torch.where(mask, tb.draw(rng, mask, *ranges[k]), out[:, k])
        if heading_command:
            state[:, 0] = torch.where(mask, tb.draw(rng, mask, *ranges[3]), state[:, 0])
            flag = (tb.draw(rng, mask, 0.0, 1.0) <= rel_heading).float()
            state[:, 1] = torch.where(mask, flag, state[:, 1])
        flag = (tb.draw(rng, mask, 0.0, 1.0) <= rel_standing).float()
        state[:, 2] = torch.where(mask, flag, state[:, 2])

    def update():
        if heading_command:
            direction = quat_apply(pose[:, 3:7], forward.expand(pose.shape[0], 3))
            heading = torch.atan2(direction[:, 1], direction[:, 0])
            steer = torch.clamp(stiffness * wrap_to_pi(state[:, 0] - heading), wz_lo, wz_hi)
            out[:, 2] = torch.where(state[:, 1] > 0.5, steer, out[:, 2])
        out.masked_fill_(state[:, 2:3] > 0.5, 0.0)

    return resample, update


# -- pose command --------------------------------------------------------------------------------------

_POSE_AXES = ("pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw")

define_term(
    "uniform_pose",
    Stage.COMMAND,
    params={**{axis: REQUIRED for axis in _POSE_AXES}, "make_quat_unique": False},
    width=7,
    doc="Pose command [x, y, z, qx, qy, qz, qw] in the root frame, like the stable UniformPoseCommand. Resampling "
    "draws x, y, z, roll, pitch, yaw.",
)


@implement("uniform_pose", "warp")
def _(ctx: TermContext):
    bounds = [[float(b) for b in ctx.params[axis]] for axis in _POSE_AXES]
    lo = wp.types.vector(length=6, dtype=wp.float32)(*[b[0] for b in bounds])
    hi = wp.types.vector(length=6, dtype=wp.float32)(*[b[1] for b in bounds])
    unique, col = bool(ctx.params["make_quat_unique"]), ctx.columns[0]

    @wp.func
    def resample(env: int, f: Any, s: wp.uint32) -> wp.uint32:
        u = wp.vector(dtype=wp.float32, length=6)
        for k in range(6):
            s = rng_next(s)
            u[k] = rng_uniform(s, lo[k], hi[k])
        q = wp_quat_from_euler_xyz(u[3], u[4], u[5])
        if wp.static(unique):
            if q[3] < 0.0:
                q = wp.quat(-q[0], -q[1], -q[2], -q[3])
        for k in range(3):
            f.commands[env, col + k] = u[k]
        for k in range(4):
            f.commands[env, col + 3 + k] = q[k]
        return s

    @wp.func
    def update(env: int, f: Any):
        pass

    return resample, update


@implement("uniform_pose", "torch")
def _(ctx: TermContext):
    out, rng = ctx.out, ctx.rng
    bounds = [ctx.params[axis] for axis in _POSE_AXES]
    unique = bool(ctx.params["make_quat_unique"])

    def resample(mask: torch.Tensor):
        u = torch.stack([tb.draw(rng, mask, *b) for b in bounds], dim=1)
        q = quat_from_euler_xyz(u[:, 3], u[:, 4], u[:, 5])
        if unique:
            q = torch.where(q[:, 3:4] < 0.0, -q, q)
        out.copy_(torch.where(mask[:, None], torch.cat([u[:, :3], q], dim=1), out))

    return resample, lambda: None
