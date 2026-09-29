# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Body-pose tracking terms for manipulation.

Fields: ``body_pose_w`` ``(N, B, 7)`` link poses and ``root_pose_w`` ``(N, 7)``. Pose commands are in the root
frame: the desired world pose is ``root_pose_w * command``.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_mul

from ..terms import REQUIRED, Stage, TermContext, define_term, implement
from ..wp_math import body_quat, body_vec3, row_quat, row_vec3
from ..wp_math import quat_error_magnitude as wp_quat_error_magnitude
from .joint import f32

_READS = ("body_pose_w", "root_pose_w", "commands")


def _single_body(ctx: TermContext) -> int:
    ids = ctx.params["body_ids"]
    if len(ids) != 1:
        raise ValueError(f"'bodies' must select exactly one body, got {len(ids)}.")
    return ids[0]


@wp.func
def _position_error(f: Any, env: int, body: int, col: int) -> float:
    root_q = row_quat(f.root_pose_w, env, 3)
    desired = row_vec3(f.root_pose_w, env, 0) + wp.quat_rotate(root_q, row_vec3(f.commands, env, col))
    return wp.length(body_vec3(f.body_pose_w, env, body, 0) - desired)


@wp.func
def _orientation_error(f: Any, env: int, body: int, col: int) -> float:
    desired = row_quat(f.root_pose_w, env, 3) * row_quat(f.commands, env, col + 3)
    return wp_quat_error_magnitude(body_quat(f.body_pose_w, env, body), desired)


def _torch_errors(fields, body: int, col: int) -> tuple[torch.Tensor, torch.Tensor]:
    root, command, pose = fields["root_pose_w"], fields["commands"], fields["body_pose_w"][:, body]
    desired_pos = root[:, :3] + quat_apply(root[:, 3:7], command[:, col : col + 3])
    desired_quat = quat_mul(root[:, 3:7], command[:, col + 3 : col + 7])
    position = torch.linalg.vector_norm(pose[:, :3] - desired_pos, dim=1)
    return position, quat_error_magnitude(pose[:, 3:7], desired_quat)


for _name, _doc in (
    ("position_command_error", "Distance between a body and the commanded position [m]."),
    ("orientation_command_error", "Rotation angle between a body and the commanded orientation [rad]."),
):
    define_term(_name, Stage.REWARD, params={"command": REQUIRED, "bodies": REQUIRED}, reads=_READS, doc=_doc)


@implement("position_command_error", "warp")
def _(ctx: TermContext):
    body, col = _single_body(ctx), ctx.params["command_columns"][0]

    @wp.func
    def term(env: int, f: Any) -> float:
        return _position_error(f, env, body, col)

    return term


@implement("orientation_command_error", "warp")
def _(ctx: TermContext):
    body, col = _single_body(ctx), ctx.params["command_columns"][0]

    @wp.func
    def term(env: int, f: Any) -> float:
        return _orientation_error(f, env, body, col)

    return term


@implement("position_command_error", "torch")
def _(ctx: TermContext):
    fields, out, body, col = ctx.fields, ctx.out, _single_body(ctx), ctx.params["command_columns"][0]
    return lambda: out.copy_(_torch_errors(fields, body, col)[0])


@implement("orientation_command_error", "torch")
def _(ctx: TermContext):
    fields, out, body, col = ctx.fields, ctx.out, _single_body(ctx), ctx.params["command_columns"][0]
    return lambda: out.copy_(_torch_errors(fields, body, col)[1])


define_term(
    "pose_command_success",
    Stage.TERMINATION,
    params={"command": REQUIRED, "bodies": REQUIRED, "position_threshold": REQUIRED, "orientation_threshold": REQUIRED},
    reads=_READS,
    doc="The body is within both thresholds of the commanded pose. Unlike the stable term, which reads metrics "
    "from the previous command update, it uses the current state.",
)


@implement("pose_command_success", "warp")
def _(ctx: TermContext):
    body, col = _single_body(ctx), ctx.params["command_columns"][0]
    pos_thr, rot_thr = float(ctx.params["position_threshold"]), float(ctx.params["orientation_threshold"])

    @wp.func
    def term(env: int, f: Any) -> bool:
        return _position_error(f, env, body, col) < wp.static(pos_thr) and _orientation_error(
            f, env, body, col
        ) < wp.static(rot_thr)

    return term


@implement("pose_command_success", "torch")
def _(ctx: TermContext):
    fields, out, body, col = ctx.fields, ctx.out, _single_body(ctx), ctx.params["command_columns"][0]
    pos_thr, rot_thr = f32(ctx.params["position_threshold"]), f32(ctx.params["orientation_threshold"])

    def run():
        position, orientation = _torch_errors(fields, body, col)
        torch.logical_and(position < pos_thr, orientation < rot_thr, out=out)

    return run
