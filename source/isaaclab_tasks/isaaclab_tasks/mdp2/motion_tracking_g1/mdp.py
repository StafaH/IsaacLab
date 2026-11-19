# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from HybridRobotics/whole_body_tracking (MIT).
# See docs/licenses/dependencies/motion-tracking-license.txt.

"""Terms of the MDP2 motion tracking task, the counterparts of MDP1's motion command and its terms.

MDP1's ``MotionCommand`` becomes the state of the reset event :data:`reset_to_motion`: the clip, loaded once by its
``init`` in the robot's joint and body order, the frame each world started on, and the failure counts of the adaptive
sampling. The frame a world tracks is the quantity :data:`motion_frame`, ``start`` plus the steps of the episode, as
MDP1 advances its time step once per step; observations read the frame after it, as MDP1 observes after advancing.

Clip bodies are indexed by robot body, so the body subsets of the robot select both.
"""

from __future__ import annotations

import math

import torch
import warp as wp
from isaaclab_experimental import mdp2

from isaaclab.utils.datasets import MotionLoader
from isaaclab.utils.string import resolve_matching_names_values

##
# Helpers.
##


@wp.func
def _clip_vec(values: wp.array3d(dtype=float), frame: int, body: int) -> wp.vec3f:
    return wp.vec3f(values[frame, body, 0], values[frame, body, 1], values[frame, body, 2])


@wp.func
def _clip_quat(values: wp.array3d(dtype=float), frame: int, body: int) -> wp.quatf:
    return wp.quatf(values[frame, body, 0], values[frame, body, 1], values[frame, body, 2], values[frame, body, 3])


@wp.func
def _quat_error(q1: wp.quatf, q2: wp.quatf) -> float:
    """Angle of the rotation between two orientations [rad], as ``quat_error_magnitude``."""
    delta = q1 * wp.quat_inverse(q2)
    return 2.0 * wp.atan2(wp.length(wp.vec3f(delta[0], delta[1], delta[2])), wp.abs(delta[3]))


@wp.func
def _yaw_quat(q: wp.quatf) -> wp.quatf:
    """The yaw of an orientation, as ``yaw_quat``."""
    yaw = wp.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
    return wp.quatf(0.0, 0.0, wp.sin(0.5 * yaw), wp.cos(0.5 * yaw))


@wp.func
def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> wp.quatf:
    """The orientation of XYZ Euler angles, as ``quat_from_euler_xyz``."""
    cy, sy = wp.cos(0.5 * yaw), wp.sin(0.5 * yaw)
    cr, sr = wp.cos(0.5 * roll), wp.sin(0.5 * roll)
    cp, sp = wp.cos(0.5 * pitch), wp.sin(0.5 * pitch)
    return wp.quatf(
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
        cy * cr * cp + sy * sr * sp,
    )


@wp.func
def _yaw_delta(robot_anchor: wp.quatf, reference_anchor: wp.quatf) -> wp.quatf:
    """The yaw that turns the reference anchor to the robot's, which MDP1 aligns the reference bodies with."""
    return _yaw_quat(robot_anchor * wp.quat_inverse(reference_anchor))


@wp.func
def _relative_body_pos(
    world: int,
    frame: int,
    body: int,
    anchor: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_body_pos: wp.array3d(dtype=float),
    clip_body_quat: wp.array3d(dtype=float),
) -> wp.vec3f:
    """Reference body position aligned to the robot anchor's horizontal position and yaw, as MDP1's
    ``body_pos_relative_w``."""
    robot_anchor = body_link_pose_w[world, anchor]
    reference_anchor = _clip_vec(clip_body_pos, frame, anchor) + env_origins[world]
    position = wp.transform_get_translation(robot_anchor)
    position = wp.vec3f(position[0], position[1], reference_anchor[2])
    yaw = _yaw_delta(wp.transform_get_rotation(robot_anchor), _clip_quat(clip_body_quat, frame, anchor))
    offset = _clip_vec(clip_body_pos, frame, body) - _clip_vec(clip_body_pos, frame, anchor)
    return position + wp.quat_rotate(yaw, offset)


##
# Reset event: the clip, the start of each world, and the adaptive sampling.
##


def _load_motion(ctx, *, motion_file: str, frames: int, adaptive_kernel_size: int, adaptive_lambda: float, **params):
    """Load the clip in the robot's joint and body order, and the sampling kernel."""
    robot = ctx.entity("joint_pos")
    device = ctx.state["start"].device
    motion = MotionLoader(motion_file, body_names=robot.body_names, joint_names=robot.joint_names, device=str(device))
    if not math.isclose(motion.fps * ctx.step_dt, 1.0, rel_tol=1e-5):
        raise ValueError("Motion FPS must match the policy rate; resample the reference before training.")
    if motion.time_step_total != frames:
        raise ValueError(f"The clip has {motion.time_step_total} frames, not the {frames} of the event's params.")
    clip = {
        "clip_joint_pos": motion.joint_pos,
        "clip_joint_vel": motion.joint_vel,
        "clip_body_pos": motion.body_pos_w,
        "clip_body_quat": motion.body_quat_w,
        "clip_body_lin_vel": motion.body_lin_vel_w,
        "clip_body_ang_vel": motion.body_ang_vel_w,
    }
    for name, values in clip.items():
        ctx.state[name].copy_(values)
    kernel = torch.tensor([adaptive_lambda**i for i in range(adaptive_kernel_size)], device=device)
    ctx.state["kernel"].copy_(kernel / kernel.sum())


@wp.kernel
def _credit_failures(
    mask: wp.array(dtype=wp.bool),
    terminated: wp.array(dtype=wp.bool),
    episode_steps: wp.array(dtype=float),
    start: wp.array(dtype=float),
    frames: int,
    step_failures: wp.array(dtype=float),
):
    world = wp.tid()
    if mask[world] and terminated[world]:
        # The frame that the episode failed on: MDP1's time step before it resamples.
        frame = wp.min(int(start[world]) + int(episode_steps[world]), frames - 1)
        bins = step_failures.shape[0]
        failed = wp.clamp(frame * bins // frames, 0, bins - 1)
        wp.atomic_add(step_failures, failed, 1.0)


@wp.func
def _sampling_weight(failures: wp.array(dtype=float), kernel: wp.array(dtype=float), uniform: float, b: int) -> float:
    """Unnormalized probability of a bin: its failures and a uniform share, smoothed over the bins after it."""
    bins = failures.shape[0]
    weight = float(0.0)
    for k in range(kernel.shape[0]):
        weight += kernel[k] * (failures[wp.min(b + k, bins - 1)] + uniform)
    return weight


@wp.kernel
def _start_on_motion(
    mask: wp.array(dtype=wp.bool),
    rng: wp.array(dtype=wp.int32),
    joint_pos: wp.array2d(dtype=float),
    joint_vel: wp.array2d(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    env_origins: wp.array(dtype=wp.vec3f),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    clip_joint_pos: wp.array2d(dtype=float),
    clip_joint_vel: wp.array2d(dtype=float),
    clip_body_pos: wp.array3d(dtype=float),
    clip_body_quat: wp.array3d(dtype=float),
    clip_body_lin_vel: wp.array3d(dtype=float),
    clip_body_ang_vel: wp.array3d(dtype=float),
    failures: wp.array(dtype=float),
    kernel: wp.array(dtype=float),
    start: wp.array(dtype=float),
    frames: int,
    uniform_ratio: float,
    pose_range: wp.types.matrix(shape=(6, 2), dtype=wp.float32),
    velocity_range: wp.types.matrix(shape=(6, 2), dtype=wp.float32),
    joint_position_range: wp.vec2f,
):
    world = wp.tid()
    if not mask[world]:
        return
    seed = rng[world]

    # Draw a bin by its smoothed failure rate, then a frame in it, as MDP1's adaptive sampling.
    bins = failures.shape[0]
    uniform = uniform_ratio / float(bins)
    total = float(0.0)
    for b in range(bins):
        total += _sampling_weight(failures, kernel, uniform, b)
    draw = wp.randf(wp.rand_init(seed, 0)) * total
    chosen = int(0)
    cumulative = _sampling_weight(failures, kernel, uniform, 0)
    while chosen < bins - 1 and cumulative <= draw:
        chosen += 1
        cumulative += _sampling_weight(failures, kernel, uniform, chosen)
    fraction = (float(chosen) + wp.randf(wp.rand_init(seed, 1))) / float(bins)
    frame = wp.min(int(fraction * float(frames - 1)), frames - 1)
    start[world] = float(frame)

    # The root of the frame, offset by uniform samples from the ranges; the root is the robot's first body.
    offsets = wp.spatial_vectorf()
    velocities = wp.spatial_vectorf()
    for k in range(6):
        offsets[k] = wp.randf(wp.rand_init(seed, 2 + k), pose_range[k, 0], pose_range[k, 1])
        velocities[k] = wp.randf(wp.rand_init(seed, 8 + k), velocity_range[k, 0], velocity_range[k, 1])
    position = _clip_vec(clip_body_pos, frame, 0) + env_origins[world]
    position += wp.vec3f(offsets[0], offsets[1], offsets[2])
    orientation = _quat_from_euler_xyz(offsets[3], offsets[4], offsets[5]) * _clip_quat(clip_body_quat, frame, 0)
    root_link_pose_w[world] = wp.transformf(position, orientation)
    linear = _clip_vec(clip_body_lin_vel, frame, 0) + wp.vec3f(velocities[0], velocities[1], velocities[2])
    angular = _clip_vec(clip_body_ang_vel, frame, 0) + wp.vec3f(velocities[3], velocities[4], velocities[5])
    root_com_vel_w[world] = wp.spatial_vector(linear, angular)

    # The joints of the frame, with uniform noise on the positions, within their soft limits.
    for j in range(joint_pos.shape[1]):
        noise = wp.randf(wp.rand_init(seed, 14 + j), joint_position_range[0], joint_position_range[1])
        limits = soft_joint_pos_limits[world, j]
        joint_pos[world, j] = wp.clamp(clip_joint_pos[frame, j] + noise, limits[0], limits[1])
        joint_vel[world, j] = clip_joint_vel[frame, j]


@wp.kernel
def _blend_failures(failures: wp.array(dtype=float), step_failures: wp.array(dtype=float), alpha: float):
    b = wp.tid()
    failures[b] = failures[b] + alpha * (step_failures[b] - failures[b])
    step_failures[b] = 0.0


def _ranges(ranges: dict[str, tuple[float, float]]) -> wp.types.matrix:
    """The ``(low, high)`` ranges of x, y, z, roll, pitch, and yaw, zero where not given, as a 6 x 2 matrix."""
    values = [ranges.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")]
    return wp.types.matrix(shape=(6, 2), dtype=wp.float32)(*(bound for pair in values for bound in pair))


def _reset_to_motion(
    mask,
    rng,
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
    clip_body_pos,
    clip_body_quat,
    clip_body_lin_vel,
    clip_body_ang_vel,
    failures,
    step_failures,
    kernel,
    start,
    *,
    frames,
    pose_range,
    velocity_range,
    joint_position_range,
    adaptive_uniform_ratio,
    adaptive_alpha,
    **params,
):
    # It runs every step, also without resets, so the failure rates blend every step, as in MDP1's command update.
    device, num_worlds = mask.device, mask.shape[0]
    wp.launch(
        _credit_failures,
        dim=num_worlds,
        inputs=[mask, terminated, episode_steps, start, frames, step_failures],
        device=device,
    )
    clip = [clip_joint_pos, clip_joint_vel, clip_body_pos, clip_body_quat, clip_body_lin_vel, clip_body_ang_vel]
    wp.launch(
        _start_on_motion,
        dim=num_worlds,
        inputs=[
            mask,
            rng,
            joint_pos,
            joint_vel,
            root_link_pose_w,
            root_com_vel_w,
            env_origins,
            soft_joint_pos_limits,
            *clip,
            failures,
            kernel,
            start,
            frames,
            adaptive_uniform_ratio,
            _ranges(pose_range),
            _ranges(velocity_range),
            wp.vec2f(*joint_position_range),
        ],
        device=device,
    )
    wp.launch(_blend_failures, dim=failures.shape[0], inputs=[failures, step_failures, adaptive_alpha], device=device)


reset_to_motion = mdp2.Term(
    warp=_reset_to_motion,
    state={
        "clip_joint_pos": mdp2.State(("frames", "joints")),
        "clip_joint_vel": mdp2.State(("frames", "joints")),
        "clip_body_pos": mdp2.State(("frames", "bodies", 3)),
        "clip_body_quat": mdp2.State(("frames", "bodies", 4)),
        "clip_body_lin_vel": mdp2.State(("frames", "bodies", 3)),
        "clip_body_ang_vel": mdp2.State(("frames", "bodies", 3)),
        "failures": mdp2.State(("bins",)),
        "step_failures": mdp2.State(("bins",)),
        "kernel": mdp2.State(("adaptive_kernel_size",)),
        "start": mdp2.State(persistent=True),
    },
    random=True,
    init=_load_motion,
)
"""Start the reset worlds on a frame of the clip ``motion_file``, drawn by MDP1's adaptive sampling, with the root
and joints offset by uniform samples from ``pose_range``, ``velocity_range``, and ``joint_position_range``.

The clip has ``frames`` frames of ``bodies`` bodies, at the control rate. Its ``bins`` bins of about a second each
count the episodes that terminated in them, through the ``terminated`` and ``episode_steps`` inputs, and their failure
rates blend in by ``adaptive_alpha`` every step. A bin is drawn by its rate plus ``adaptive_uniform_ratio / bins``,
smoothed over the ``adaptive_kernel_size`` bins after it with weights ``adaptive_lambda ** k``.
"""

clip_names = {
    name: mdp2.term_state("events.reset_to_motion", name)
    for name in (
        "clip_joint_pos",
        "clip_joint_vel",
        "clip_body_pos",
        "clip_body_quat",
        "clip_body_lin_vel",
        "clip_body_ang_vel",
        "start",
    )
}
"""The state of :data:`reset_to_motion` that the other terms read, by input name."""


##
# Quantities.
##


@wp.func
def _episode_steps(world: int, steps: wp.array(dtype=float)) -> float:
    steps[world] = steps[world] + 1.0
    return steps[world]


episode_steps = mdp2.Term(warp=_episode_steps, state={"steps": mdp2.State()})
"""Steps of the episode so far, counted in its own state: the training scripts start episodes at random lengths,
which MDP1's motion command does not see. The reset event reads it as the length of the episode that ended."""


@wp.func
def _motion_frame(
    world: int, start: wp.array(dtype=float), steps: wp.array(dtype=float), ahead: int, frames: int
) -> float:
    return float(wp.min(int(start[world]) + int(steps[world]) + ahead, frames - 1))


motion_frame = mdp2.Term(warp=_motion_frame)
"""The frame ``ahead`` steps after the one the episode reached, on the clip of ``frames`` frames, from the ``steps``
state of :data:`episode_steps`.

With ``ahead`` 0, it is MDP1's ``time_steps`` as the terminations and rewards see it; with 1, as the observations
see it, after the command's update.
"""


##
# Actions.
##


def _init_action_scale(ctx, *, scale: dict[str, float]):
    joint_names = [ctx.entity("joint_ids").joint_names[j] for j in ctx.inputs["joint_ids"].tolist()]
    indices, _, values = resolve_matching_names_values(scale, joint_names, preserve_order=False)
    ctx.state["joint_scale"][indices] = torch.tensor(values, device=ctx.state["joint_scale"].device)


@wp.func
def _joint_position(
    world: int,
    action: wp.array2d(dtype=float),
    joint_pos_target: wp.array2d(dtype=float),
    default_joint_pos: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
    joint_scale: wp.array(dtype=float),
):
    for k in range(joint_ids.shape[0]):
        joint = joint_ids[k]
        joint_pos_target[world, joint] = default_joint_pos[world, joint] + joint_scale[k] * action[world, k]


joint_position = mdp2.Term(
    warp=_joint_position,
    shape=("worlds", "joint_ids"),
    state={"joint_scale": mdp2.State(("joint_ids",))},
    init=_init_action_scale,
)
"""Joint position targets relative to the default joint positions, scaled per joint by the ``scale`` dict of joint
name patterns: MDP1's ``JointPositionActionCfg`` with ``use_default_offset``."""


##
# Observations.
##


@wp.func
def _motion_command(
    world: int,
    kind: int,
    k: int,
    motion_frame: wp.array(dtype=float),
    clip_joint_pos: wp.array2d(dtype=float),
    clip_joint_vel: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
) -> float:
    frame = int(motion_frame[world])
    if kind == 0:
        return clip_joint_pos[frame, joint_ids[k]]
    return clip_joint_vel[frame, joint_ids[k]]


motion_command = mdp2.Term(warp=_motion_command, shape=("worlds", 2, "joint_ids"))
"""The joint positions, then the joint velocities, of the reference frame: MDP1's motion command."""


@wp.func
def _motion_anchor_pos_b(
    world: int,
    axis: int,
    motion_frame: wp.array(dtype=float),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_body_pos: wp.array3d(dtype=float),
    anchor_ids: wp.array(dtype=wp.int32),
) -> float:
    anchor = anchor_ids[0]
    robot_anchor = body_link_pose_w[world, anchor]
    reference = _clip_vec(clip_body_pos, int(motion_frame[world]), anchor) + env_origins[world]
    offset = reference - wp.transform_get_translation(robot_anchor)
    return wp.quat_rotate_inv(wp.transform_get_rotation(robot_anchor), offset)[axis]


motion_anchor_pos_b = mdp2.Term(warp=_motion_anchor_pos_b, shape=("worlds", 3))
"""Reference anchor position in the robot anchor's frame [m]."""


@wp.func
def _motion_anchor_ori_b(
    world: int,
    row: int,
    column: int,
    motion_frame: wp.array(dtype=float),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    clip_body_quat: wp.array3d(dtype=float),
    anchor_ids: wp.array(dtype=wp.int32),
) -> float:
    anchor = anchor_ids[0]
    robot = wp.transform_get_rotation(body_link_pose_w[world, anchor])
    reference = _clip_quat(clip_body_quat, int(motion_frame[world]), anchor)
    return wp.quat_to_matrix(wp.quat_inverse(robot) * reference)[row, column]


motion_anchor_ori_b = mdp2.Term(warp=_motion_anchor_ori_b, shape=("worlds", 3, 2))
"""Reference anchor orientation in the robot anchor's frame, as the first two columns of its rotation matrix."""


@wp.func
def _robot_body_pos_b(
    world: int,
    k: int,
    axis: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    anchor_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
) -> float:
    anchor = body_link_pose_w[world, anchor_ids[0]]
    offset = wp.transform_get_translation(body_link_pose_w[world, body_ids[k]]) - wp.transform_get_translation(anchor)
    return wp.quat_rotate_inv(wp.transform_get_rotation(anchor), offset)[axis]


robot_body_pos_b = mdp2.Term(warp=_robot_body_pos_b, shape=("worlds", "body_ids", 3))
"""Positions of the tracked bodies in the robot anchor's frame [m]."""


@wp.func
def _robot_body_ori_b(
    world: int,
    k: int,
    row: int,
    column: int,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    anchor_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
) -> float:
    anchor = wp.transform_get_rotation(body_link_pose_w[world, anchor_ids[0]])
    body = wp.transform_get_rotation(body_link_pose_w[world, body_ids[k]])
    return wp.quat_to_matrix(wp.quat_inverse(anchor) * body)[row, column]


robot_body_ori_b = mdp2.Term(warp=_robot_body_ori_b, shape=("worlds", "body_ids", 3, 2))
"""Orientations of the tracked bodies in the robot anchor's frame, as the first two columns of their rotation
matrices."""


##
# Rewards.
##


@wp.func
def _anchor_pos_error_exp(
    world: int,
    motion_frame: wp.array(dtype=float),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_body_pos: wp.array3d(dtype=float),
    anchor_ids: wp.array(dtype=wp.int32),
    std: float,
) -> float:
    anchor = anchor_ids[0]
    reference = _clip_vec(clip_body_pos, int(motion_frame[world]), anchor) + env_origins[world]
    error = wp.length_sq(reference - wp.transform_get_translation(body_link_pose_w[world, anchor]))
    return wp.exp(-error / (std * std))


motion_global_anchor_position_error_exp = mdp2.Term(warp=_anchor_pos_error_exp)
"""``exp(-|anchor position error|^2 / std^2)``, in the world frame."""


@wp.func
def _anchor_ori_error_exp(
    world: int,
    motion_frame: wp.array(dtype=float),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    clip_body_quat: wp.array3d(dtype=float),
    anchor_ids: wp.array(dtype=wp.int32),
    std: float,
) -> float:
    anchor = anchor_ids[0]
    reference = _clip_quat(clip_body_quat, int(motion_frame[world]), anchor)
    error = _quat_error(reference, wp.transform_get_rotation(body_link_pose_w[world, anchor]))
    return wp.exp(-error * error / (std * std))


motion_global_anchor_orientation_error_exp = mdp2.Term(warp=_anchor_ori_error_exp)
"""``exp(-angle_error^2 / std^2)`` of the anchor orientation."""


@wp.func
def _body_pos_error_exp(
    world: int,
    motion_frame: wp.array(dtype=float),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_body_pos: wp.array3d(dtype=float),
    clip_body_quat: wp.array3d(dtype=float),
    anchor_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    std: float,
) -> float:
    frame = int(motion_frame[world])
    total = float(0.0)
    for k in range(body_ids.shape[0]):
        body = body_ids[k]
        reference = _relative_body_pos(
            world, frame, body, anchor_ids[0], body_link_pose_w, env_origins, clip_body_pos, clip_body_quat
        )
        total += wp.length_sq(reference - wp.transform_get_translation(body_link_pose_w[world, body]))
    return wp.exp(-total / float(body_ids.shape[0]) / (std * std))


motion_relative_body_position_error_exp = mdp2.Term(warp=_body_pos_error_exp)
"""``exp(-mean |body position error|^2 / std^2)``, with the reference aligned to the robot anchor's horizontal
position and yaw."""


@wp.func
def _body_ori_error_exp(
    world: int,
    motion_frame: wp.array(dtype=float),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    clip_body_quat: wp.array3d(dtype=float),
    anchor_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    std: float,
) -> float:
    frame = int(motion_frame[world])
    anchor = anchor_ids[0]
    yaw = _yaw_delta(
        wp.transform_get_rotation(body_link_pose_w[world, anchor]), _clip_quat(clip_body_quat, frame, anchor)
    )
    total = float(0.0)
    for k in range(body_ids.shape[0]):
        body = body_ids[k]
        reference = yaw * _clip_quat(clip_body_quat, frame, body)
        error = _quat_error(reference, wp.transform_get_rotation(body_link_pose_w[world, body]))
        total += error * error
    return wp.exp(-total / float(body_ids.shape[0]) / (std * std))


motion_relative_body_orientation_error_exp = mdp2.Term(warp=_body_ori_error_exp)
"""``exp(-mean angle_error^2 / std^2)`` of the body orientations, with the reference aligned to the robot anchor's
yaw."""


@wp.func
def _body_vel_error_exp(
    world: int,
    motion_frame: wp.array(dtype=float),
    body_com_vel_w: wp.array2d(dtype=wp.spatial_vectorf),
    clip_body_vel: wp.array3d(dtype=float),
    body_ids: wp.array(dtype=wp.int32),
    angular: bool,
    std: float,
) -> float:
    frame = int(motion_frame[world])
    total = float(0.0)
    for k in range(body_ids.shape[0]):
        body = body_ids[k]
        velocity = body_com_vel_w[world, body]
        robot = wp.spatial_top(velocity)
        if angular:
            robot = wp.spatial_bottom(velocity)
        total += wp.length_sq(_clip_vec(clip_body_vel, frame, body) - robot)
    return wp.exp(-total / float(body_ids.shape[0]) / (std * std))


motion_global_body_velocity_error_exp = mdp2.Term(warp=_body_vel_error_exp)
"""``exp(-mean |body velocity error|^2 / std^2)`` in the world frame, of the linear velocities, or of the angular
ones if ``angular``; ``clip_body_vel`` is the matching velocity of the clip."""


@wp.func
def _self_collision_cost(
    world: int,
    force_matrix_w_history: wp.array4d(dtype=wp.vec3f),
    body_ids: wp.array(dtype=wp.int32),
    threshold: float,
) -> float:
    count = float(0.0)
    for step in range(force_matrix_w_history.shape[1]):
        peak = float(0.0)
        for k in range(body_ids.shape[0]):
            for f in range(force_matrix_w_history.shape[3]):
                peak = wp.max(peak, wp.length(force_matrix_w_history[world, step, body_ids[k], f]))
        if peak > threshold:
            count += 1.0
    return count


self_collision_cost = mdp2.Term(warp=_self_collision_cost)
"""Physics steps of the history with a filtered contact force above ``threshold`` [N] on any of the bodies, as
MDP1's ``self_collision_cost``."""


##
# Terminations.
##


@wp.func
def _motion_end(world: int, motion_frame: wp.array(dtype=float), frames: int) -> bool:
    return int(motion_frame[world]) >= frames - 1


motion_end = mdp2.Term(warp=_motion_end)
"""Whether the world reached the last frame of the clip."""


@wp.func
def _bad_anchor_pos_z_only(
    world: int,
    motion_frame: wp.array(dtype=float),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_body_pos: wp.array3d(dtype=float),
    anchor_ids: wp.array(dtype=wp.int32),
    threshold: float,
) -> bool:
    anchor = anchor_ids[0]
    reference = _clip_vec(clip_body_pos, int(motion_frame[world]), anchor) + env_origins[world]
    return wp.abs(reference[2] - wp.transform_get_translation(body_link_pose_w[world, anchor])[2]) > threshold


bad_anchor_pos_z_only = mdp2.Term(warp=_bad_anchor_pos_z_only)
"""Whether the anchor height is more than ``threshold`` [m] from the reference's."""


@wp.func
def _bad_anchor_ori(
    world: int,
    motion_frame: wp.array(dtype=float),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    gravity_w: wp.array(dtype=wp.vec3f),
    clip_body_quat: wp.array3d(dtype=float),
    anchor_ids: wp.array(dtype=wp.int32),
    threshold: float,
) -> bool:
    anchor = anchor_ids[0]
    gravity = wp.normalize(gravity_w[world])
    reference = wp.quat_rotate_inv(_clip_quat(clip_body_quat, int(motion_frame[world]), anchor), gravity)
    robot = wp.quat_rotate_inv(wp.transform_get_rotation(body_link_pose_w[world, anchor]), gravity)
    return wp.abs(reference[2] - robot[2]) > threshold


bad_anchor_ori = mdp2.Term(warp=_bad_anchor_ori)
"""Whether the vertical part of the gravity direction in the anchor frame differs from the reference's by more than
``threshold``."""


@wp.func
def _bad_motion_body_pos_z_only(
    world: int,
    motion_frame: wp.array(dtype=float),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    clip_body_pos: wp.array3d(dtype=float),
    body_ids: wp.array(dtype=wp.int32),
    threshold: float,
) -> bool:
    # The yaw alignment of the reference keeps the heights: the aligned height is the reference's.
    frame = int(motion_frame[world])
    for k in range(body_ids.shape[0]):
        body = body_ids[k]
        reference = clip_body_pos[frame, body, 2] + env_origins[world][2]
        if wp.abs(reference - wp.transform_get_translation(body_link_pose_w[world, body])[2]) > threshold:
            return True
    return False


bad_motion_body_pos_z_only = mdp2.Term(warp=_bad_motion_body_pos_z_only)
"""Whether the height of any of the bodies is more than ``threshold`` [m] from the reference's."""
