# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terms of the humanoid motion tracking task.

The reset event keeps the motion clip as state, sampled at the control rate: every world starts at a frame of the
clip and follows it one frame per step. The other terms read the clip from that state, so looking up the reference
is an array read inside the step's kernels.
"""

from __future__ import annotations

import re

import numpy as np
import torch
import warp as wp
from isaaclab_experimental import mdp2

from isaaclab.utils import math as math_utils

CLIP_JOINTS = {"thigh": "hip", "foot": "ankle", "torso": "abdomen", "head": "neck", "upper_arm": "shoulder"}
"""Names that the clips give the ball joints of the robot, by the name of the joint's child body."""


@wp.func
def reference_frame(
    world: int, start: wp.array(dtype=float), episode_length: wp.array(dtype=wp.int32), frames: int
) -> int:
    """Return the frame of the clip that a world tracks at its current step."""
    return wp.min(int(start[world]) + episode_length[world], frames - 1)


def _load_motion(ctx, *, motion_file, frames, bodies, start_range, noise):
    robot = ctx.entity("joint_pos")
    clip = np.load(motion_file)
    stride = round(float(clip["fps"]) * ctx.step_dt)
    clip_joints, clip_bodies = list(clip["dof_names"]), list(clip["body_names"])
    device = ctx.state["start"].device

    # The clips hold ball joints as rotation vectors; the solver's coordinates are their intrinsic XYZ Euler angles.
    joint_pos = torch.zeros(len(clip["dof_positions"]), len(robot.joint_names))
    for j, name in enumerate(robot.joint_names):
        body, _, axis = name.partition(":")
        if not axis:
            joint_pos[:, j] = torch.from_numpy(clip["dof_positions"][:, clip_joints.index(name)])
        elif axis == "0":
            side, part = re.fullmatch("(left_|right_)?(.*)", body).groups("")
            columns = [clip_joints.index(f"{side}{CLIP_JOINTS[part]}_{x}") for x in "xyz"]
            vector = torch.from_numpy(clip["dof_positions"][:, columns])
            angle = vector.norm(dim=-1)
            quat = math_utils.quat_from_angle_axis(angle, vector / angle.clamp(min=1.0e-9)[:, None])
            matrix = math_utils.matrix_from_quat(quat)
            joint_pos[:, j] = torch.atan2(-matrix[:, 1, 2], matrix[:, 2, 2])
            joint_pos[:, j + 1] = torch.asin(matrix[:, 0, 2].clamp(-1.0, 1.0))
            joint_pos[:, j + 2] = torch.atan2(-matrix[:, 0, 1], matrix[:, 0, 0])

    # The root is the first body. Clip quaternions are wxyz, and the solver's are xyzw. Velocities are differences
    # of the frames at the control rate: the clips' own start from rest and repeat frames.
    body_ids = [clip_bodies.index(name) for name in robot.body_names]
    root = body_ids[0]
    joint_pos = torch.from_numpy(np.unwrap(joint_pos[::stride].numpy(), axis=0))
    root_pos = clip["body_positions"][::stride, root]
    root_quat = clip["body_rotations"][::stride, root][:, [1, 2, 3, 0]]
    root_vel = np.concatenate(
        [np.gradient(root_pos, ctx.step_dt, axis=0), clip["body_angular_velocities"][::stride, root]], axis=-1
    )
    values = {
        "clip_joint_pos": joint_pos,
        "clip_joint_vel": torch.gradient(joint_pos, spacing=ctx.step_dt, dim=0)[0],
        "clip_root_pose": torch.from_numpy(np.concatenate([root_pos, root_quat], axis=-1)),
        "clip_root_vel": torch.from_numpy(root_vel),
        "clip_body_pos": torch.from_numpy(clip["body_positions"][::stride, body_ids]),
    }
    for name, value in values.items():
        if value.shape != ctx.state[name].shape:
            raise ValueError(
                f"The clip has {len(joint_pos)} frames at the control rate and {len(body_ids)} bodies, not the"
                f" {frames} frames and {bodies} bodies of the term's params."
            )
        ctx.state[name].copy_(value.to(device))


@wp.func
def _reset_to_motion(
    world: int,
    joint_pos: wp.array2d(dtype=float),
    joint_vel: wp.array2d(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    env_origins: wp.array(dtype=wp.vec3f),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    clip_joint_pos: wp.array2d(dtype=float),
    clip_joint_vel: wp.array2d(dtype=float),
    clip_root_pose: wp.array2d(dtype=float),
    clip_root_vel: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    start_range: wp.vec2f,
    noise: float,
    rng: wp.int32,
):
    frames = clip_joint_pos.shape[0]
    fraction = wp.randf(wp.rand_init(rng, 0), start_range[0], start_range[1])
    frame = wp.min(int(fraction * float(frames)), frames - 2)
    start[world] = float(frame)

    for j in range(joint_pos.shape[1]):
        limits = soft_joint_pos_limits[world, j]
        position = clip_joint_pos[frame, j] + wp.randf(wp.rand_init(rng, 1 + j), -noise, noise)
        joint_pos[world, j] = wp.clamp(position, limits[0], limits[1])
        joint_vel[world, j] = clip_joint_vel[frame, j]

    pose = wp.transformf()
    for k in range(7):
        pose[k] = clip_root_pose[frame, k]
    for k in range(3):
        pose[k] = pose[k] + env_origins[world][k]
    velocity = wp.spatial_vectorf()
    for k in range(6):
        velocity[k] = clip_root_vel[frame, k]
    root_link_pose_w[world] = pose
    root_com_vel_w[world] = velocity


reset_to_motion = mdp2.Term(
    warp=_reset_to_motion,
    state={
        "clip_joint_pos": mdp2.State(("frames", "joint_pos")),
        "clip_joint_vel": mdp2.State(("frames", "joint_pos")),
        "clip_root_pose": mdp2.State(("frames", 7)),
        "clip_root_vel": mdp2.State(("frames", 6)),
        "clip_body_pos": mdp2.State(("frames", "bodies", 3)),
        "start": mdp2.State(persistent=True),
    },
    random=True,
    init=_load_motion,
)
"""Start the reset worlds on the clip in ``motion_file``, at a frame drawn from the fraction ``start_range`` of it.

The clip is stored as state, sampled at the control rate, ``frames`` frames of ``bodies`` bodies, in the robot's
joint and body order, and ``start`` is the frame each world started at. Joint positions get uniform ``noise`` [rad].
"""


@wp.func
def _reference_joint_pos(
    world: int,
    joint: int,
    clip_joint_pos: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
) -> float:
    frame = reference_frame(world, start, episode_length, clip_joint_pos.shape[0] - 1) + 1
    return clip_joint_pos[frame, joint_ids[joint]]


reference_joint_pos = mdp2.Term(warp=_reference_joint_pos, shape=("worlds", "joint_ids"))
"""Joint positions of the frame that the next step tracks."""


@wp.func
def _joint_pos_error(
    world: int,
    joint_pos: wp.array2d(dtype=float),
    clip_joint_pos: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
) -> float:
    frame = reference_frame(world, start, episode_length, clip_joint_pos.shape[0])
    total = float(0.0)
    for j in range(joint_pos.shape[1]):
        error = joint_pos[world, j] - clip_joint_pos[frame, j]
        total += error * error
    return total / float(joint_pos.shape[1])


joint_pos_error = mdp2.Term(warp=_joint_pos_error)
"""Mean squared error of the joint positions to the tracked frame [rad^2]."""


@wp.func
def _joint_vel_error(
    world: int,
    joint_vel: wp.array2d(dtype=float),
    clip_joint_vel: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
) -> float:
    frame = reference_frame(world, start, episode_length, clip_joint_vel.shape[0])
    total = float(0.0)
    for j in range(joint_vel.shape[1]):
        error = joint_vel[world, j] - clip_joint_vel[frame, j]
        total += error * error
    return total / float(joint_vel.shape[1])


joint_vel_error = mdp2.Term(warp=_joint_vel_error)
"""Mean squared error of the joint velocities to the tracked frame [rad^2/s^2]."""


@wp.func
def _body_pos_error(
    world: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_body_pos: wp.array3d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
) -> float:
    frame = reference_frame(world, start, episode_length, clip_body_pos.shape[0])
    total = float(0.0)
    for b in range(body_link_pose_w.shape[1]):
        target = wp.vec3f(clip_body_pos[frame, b, 0], clip_body_pos[frame, b, 1], clip_body_pos[frame, b, 2])
        position = wp.transform_get_translation(body_link_pose_w[world, b]) - env_origins[world]
        total += wp.length_sq(position - target)
    return total / float(body_link_pose_w.shape[1])


body_pos_error = mdp2.Term(warp=_body_pos_error)
"""Mean squared distance of the bodies to their positions in the tracked frame [m^2]."""


@wp.func
def _root_vel_error(
    world: int,
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    clip_root_vel: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    angular_scale: float,
) -> float:
    frame = reference_frame(world, start, episode_length, clip_root_vel.shape[0])
    total = float(0.0)
    for k in range(6):
        error = root_com_vel_w[world][k] - clip_root_vel[frame, k]
        total += wp.where(k < 3, 1.0, angular_scale) * error * error
    return total


root_vel_error = mdp2.Term(warp=_root_vel_error)
"""Squared error of the root velocity to the tracked frame, the angular part scaled by ``angular_scale``."""


@wp.func
def _exp_kernel(world: int, value: wp.array(dtype=float), scale: float) -> float:
    return wp.exp(-scale * value[world])


exp_kernel = mdp2.Term(warp=_exp_kernel)
"""``exp(-scale * value)`` of the ``value`` input, one float per world, such as a tracking error."""


@wp.func
def _is_above(world: int, value: wp.array(dtype=float), threshold: float) -> bool:
    return value[world] > threshold


is_above = mdp2.Term(warp=_is_above)
"""Whether the ``value`` input, one float per world, is above ``threshold``."""


@wp.func
def _motion_ended(
    world: int,
    clip_joint_pos: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
) -> bool:
    return int(start[world]) + episode_length[world] >= clip_joint_pos.shape[0] - 2


motion_ended = mdp2.Term(warp=_motion_ended)
"""Whether the world tracked the clip to its end."""


@wp.func
def _root_height(world: int, root_link_pose_w: wp.array(dtype=wp.transformf)) -> float:
    return wp.transform_get_translation(root_link_pose_w[world])[2]


root_height = mdp2.Term(warp=_root_height)
"""Height of the root [m]."""


@wp.func
def _body_pos_b(
    world: int,
    body: int,
    axis: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
) -> float:
    position = wp.transform_get_translation(body_link_pose_w[world, body])
    return wp.transform_point(wp.transform_inverse(root_link_pose_w[world]), position)[axis]


body_pos_b = mdp2.Term(warp=_body_pos_b, shape=("worlds", "bodies", 3))
"""Positions of the bodies in the root frame [m]."""


@wp.func
def _reference_body_pos_b(
    world: int,
    body: int,
    axis: int,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_body_pos: wp.array3d(dtype=float),
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
) -> float:
    frame = reference_frame(world, start, episode_length, clip_body_pos.shape[0] - 1) + 1
    target = wp.vec3f(clip_body_pos[frame, body, 0], clip_body_pos[frame, body, 1], clip_body_pos[frame, body, 2])
    return wp.transform_point(wp.transform_inverse(root_link_pose_w[world]), target + env_origins[world])[axis]


reference_body_pos_b = mdp2.Term(warp=_reference_body_pos_b, shape=("worlds", "bodies", 3))
"""Positions of the bodies in the frame that the next step tracks, in the robot's root frame [m]."""
