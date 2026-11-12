# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset event of the motion tracking tasks, which owns the motion clip.

The reset event keeps the clip as state, sampled at the control rate: every world starts at a frame of the clip and
follows it one frame per step. The other terms read the clip from that state, so looking up the reference is an
array read inside the step's kernels.
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
    world: int,
    start: wp.array(dtype=float),
    episode_length: wp.array(dtype=wp.int32),
    end: wp.array(dtype=float),
    ahead: int,
) -> int:
    """Return the frame that a world tracks ``ahead`` steps from now, which stays on the world's clip."""
    return wp.min(int(start[world]) + episode_length[world] + ahead, int(end[world]) - 1)


def _load_motions(ctx, *, motion_files, frames, bodies, clips, **reset_params):
    robot = ctx.entity("joint_pos")
    device = ctx.state["start"].device
    values = {
        name: [] for name in ("clip_joint_pos", "clip_joint_vel", "clip_root_pose", "clip_root_vel", "clip_body_pos")
    }
    ranges, begin = [], 0
    for motion_file in motion_files:
        clip = np.load(motion_file)
        stride = round(float(clip["fps"]) * ctx.step_dt)
        clip_joints, clip_bodies = list(clip["dof_names"]), list(clip["body_names"])

        # The clips hold ball joints as rotation vectors; the solver's coordinates are their intrinsic XYZ Euler
        # angles.
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

        # The root is the first body. Clip quaternions are wxyz, and the solver's are xyzw. Velocities are
        # differences of the frames at the control rate: the clips' own start from rest and repeat frames.
        body_ids = [clip_bodies.index(name) for name in robot.body_names]
        root = body_ids[0]
        joint_pos = torch.from_numpy(np.unwrap(joint_pos[::stride].numpy(), axis=0))
        root_pos = clip["body_positions"][::stride, root]
        root_quat = clip["body_rotations"][::stride, root][:, [1, 2, 3, 0]]
        root_vel = np.concatenate(
            [np.gradient(root_pos, ctx.step_dt, axis=0), clip["body_angular_velocities"][::stride, root]], axis=-1
        )
        values["clip_joint_pos"].append(joint_pos)
        values["clip_joint_vel"].append(torch.gradient(joint_pos, spacing=ctx.step_dt, dim=0)[0])
        values["clip_root_pose"].append(torch.from_numpy(np.concatenate([root_pos, root_quat], axis=-1)))
        values["clip_root_vel"].append(torch.from_numpy(root_vel))
        values["clip_body_pos"].append(torch.from_numpy(clip["body_positions"][::stride, body_ids]))
        ranges.append((begin, begin + len(joint_pos)))
        begin += len(joint_pos)

    for name, parts in values.items():
        value = torch.cat(parts)
        if value.shape != ctx.state[name].shape:
            raise ValueError(
                f"The clips have {begin} frames at the control rate and {value.shape[1]} values per frame, not the"
                f" {frames} frames and {bodies} bodies of the term's params."
            )
        ctx.state[name].copy_(value.to(device))
    ctx.state["clip_range"].copy_(torch.tensor(ranges, dtype=torch.float32, device=device))


@wp.func
def _start_on_frame(
    world: int,
    frame: int,
    last: int,
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
    end: wp.array(dtype=float),
    noise: float,
    rng: wp.int32,
):
    """Put a world's robot on a frame of a clip whose frames end before ``last``."""
    start[world] = float(frame)
    end[world] = float(last)

    for j in range(joint_pos.shape[1]):
        limits = soft_joint_pos_limits[world, j]
        position = clip_joint_pos[frame, j] + wp.randf(wp.rand_init(rng, 3 + j), -noise, noise)
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


@wp.func
def _uniform_frame(clip_range: wp.array2d(dtype=float), start_range: wp.vec2f, rng: wp.int32) -> wp.vec2i:
    """Return a frame in the fraction ``start_range`` of a clip drawn uniformly, and the frame after the clip."""
    clip = wp.min(int(wp.randf(wp.rand_init(rng, 0)) * float(clip_range.shape[0])), clip_range.shape[0] - 1)
    first, last = int(clip_range[clip, 0]), int(clip_range[clip, 1])
    fraction = wp.randf(wp.rand_init(rng, 1), start_range[0], start_range[1])
    return wp.vec2i(first + wp.min(int(fraction * float(last - first)), last - first - 2), last)


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
    clip_range: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    end: wp.array(dtype=float),
    start_range: wp.vec2f,
    noise: float,
    rng: wp.int32,
):
    frame = _uniform_frame(clip_range, start_range, rng)
    _start_on_frame(
        world,
        frame[0],
        frame[1],
        joint_pos,
        joint_vel,
        root_link_pose_w,
        root_com_vel_w,
        env_origins,
        soft_joint_pos_limits,
        clip_joint_pos,
        clip_joint_vel,
        clip_root_pose,
        clip_root_vel,
        start,
        end,
        noise,
        rng,
    )


CLIP_STATE = {
    "clip_joint_pos": mdp2.State(("frames", "joint_pos")),
    "clip_joint_vel": mdp2.State(("frames", "joint_pos")),
    "clip_root_pose": mdp2.State(("frames", 7)),
    "clip_root_vel": mdp2.State(("frames", 6)),
    "clip_body_pos": mdp2.State(("frames", "bodies", 3)),
    "clip_range": mdp2.State(("clips", 2)),
    "start": mdp2.State(persistent=True),
    "end": mdp2.State(persistent=True),
}
"""State of the reset events: the clips, and the frames each world started at and ends before."""

reset_to_motion = mdp2.Term(
    warp=_reset_to_motion,
    state=CLIP_STATE,
    random=True,
    init=_load_motions,
)
"""Start the reset worlds on one of the clips in ``motion_files``, at a frame drawn from the fraction ``start_range``
of it.

The clips are stored as state, one after the other, sampled at the control rate: ``frames`` frames in all of
``bodies`` bodies, in the robot's joint and body order. ``clip_range`` holds the first frame of each of the ``clips``
clips and the frame after its last; ``start`` is the frame each world started at, and ``end`` the frame after the
last of its clip. Joint positions get uniform ``noise`` [rad].
"""


@wp.kernel
def _credit_failures(
    mask: wp.array(dtype=wp.bool),
    terminated: wp.array(dtype=wp.bool),
    start: wp.array(dtype=float),
    end: wp.array(dtype=float),
    episode_steps: wp.array(dtype=float),
    bin_frames: int,
    failures: wp.array(dtype=float),
):
    world = wp.tid()
    if mask[world] and terminated[world]:
        frame = wp.min(int(start[world]) + int(episode_steps[world]), int(end[world]) - 1)
        wp.atomic_add(failures, frame / bin_frames, 1.0)


@wp.kernel
def _forget_failures(failures: wp.array(dtype=float), decay: float):
    failures[wp.tid()] = decay * failures[wp.tid()]


@wp.kernel
def _start_where_it_fails(
    mask: wp.array(dtype=wp.bool),
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
    clip_range: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    end: wp.array(dtype=float),
    failures: wp.array(dtype=float),
    rng: wp.array(dtype=wp.int32),
    start_range: wp.vec2f,
    noise: float,
    adaptive: float,
    bin_frames: int,
):
    world = wp.tid()
    if not mask[world]:
        return

    seed = rng[world]
    frame = _uniform_frame(clip_range, start_range, seed)
    total = float(0.0)
    for b in range(failures.shape[0]):
        total += failures[b]
    if total > 0.0 and wp.randf(wp.rand_init(seed, 2)) < adaptive:
        # Draw a bin by its failures, then a frame in it, a few frames before where the episodes failed.
        draw = wp.randf(wp.rand_init(seed, 0)) * total
        chosen = int(0)
        cumulative = failures[0]
        while chosen < failures.shape[0] - 1 and cumulative <= draw:
            chosen += 1
            cumulative += failures[chosen]
        drawn = chosen * bin_frames + int(wp.randf(wp.rand_init(seed, 1)) * float(bin_frames)) - bin_frames
        for c in range(clip_range.shape[0]):
            first, last = int(clip_range[c, 0]), int(clip_range[c, 1])
            if chosen * bin_frames >= first and chosen * bin_frames < last:
                frame = wp.vec2i(wp.clamp(drawn, first, last - 2), last)

    _start_on_frame(
        world,
        frame[0],
        frame[1],
        joint_pos,
        joint_vel,
        root_link_pose_w,
        root_com_vel_w,
        env_origins,
        soft_joint_pos_limits,
        clip_joint_pos,
        clip_joint_vel,
        clip_root_pose,
        clip_root_vel,
        start,
        end,
        noise,
        seed,
    )


def _reset_to_failures(
    mask,
    joint_pos,
    joint_vel,
    root_link_pose_w,
    root_com_vel_w,
    env_origins,
    soft_joint_pos_limits,
    terminated,
    episode_steps,
    clip_joint_pos,
    clip_joint_vel,
    clip_root_pose,
    clip_root_vel,
    clip_range,
    start,
    end,
    failures,
    rng,
    *,
    start_range,
    noise,
    adaptive,
    bin_frames,
    decay,
):
    device, num_worlds = mask.device, mask.shape[0]
    wp.launch(_forget_failures, dim=failures.shape[0], inputs=[failures, decay], device=device)
    wp.launch(
        _credit_failures,
        dim=num_worlds,
        inputs=[mask, terminated, start, end, episode_steps, bin_frames, failures],
        device=device,
    )
    fields = [joint_pos, joint_vel, root_link_pose_w, root_com_vel_w, env_origins, soft_joint_pos_limits]
    clips = [clip_joint_pos, clip_joint_vel, clip_root_pose, clip_root_vel, clip_range, start, end, failures, rng]
    params = [wp.vec2f(*start_range), noise, adaptive, bin_frames]
    wp.launch(_start_where_it_fails, dim=num_worlds, inputs=[mask, *fields, *clips, *params], device=device)


reset_to_failures = mdp2.Term(
    warp=_reset_to_failures,
    state={**CLIP_STATE, "failures": mdp2.State(("bins",))},
    random=True,
    init=_load_motions,
)
"""Like :data:`reset_to_motion`, but a fraction ``adaptive`` of the resets start where episodes fail.

The clips are divided into ``bins`` bins of ``bin_frames`` frames. A reset counts the episode that ended, if it was
terminated, in the bin of its last frame, from the ``episode_steps`` input; the counts fade by ``decay`` per step. A
world then starts in a bin drawn by its count, up to ``bin_frames`` frames before it, so that it practices what leads
to the
failure (as the adaptive sampling of BeyondMimic, arXiv 2508.08241).
"""
