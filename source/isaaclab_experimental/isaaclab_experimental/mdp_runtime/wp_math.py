# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp functions shared by the runtime kernels and the built-in terms.

Poses are float32 rows ``[px, py, pz, qx, qy, qz, qw]`` and velocities are ``[vx, vy, vz, wx, wy, wz]``, the
layout of Isaac Lab's ``wp.transformf`` and ``wp.spatial_vectorf`` arrays. Quaternions are ``xyzw``, as in
:class:`warp.quat`.
"""

from __future__ import annotations

import math
from typing import Any

import warp as wp


@wp.func
def rng_next(state: wp.uint32) -> wp.uint32:
    """Advance a PCG32 stream state."""
    return state * wp.uint32(747796405) + wp.uint32(2891336453)


@wp.func
def rng_uniform(state: wp.uint32, lo: wp.float32, hi: wp.float32) -> wp.float32:
    """Map an advanced PCG32 state to ``[lo, hi)`` with 24 random bits (bitwise reproducible in Torch)."""
    word = ((state >> ((state >> wp.uint32(28)) + wp.uint32(4))) ^ state) * wp.uint32(277803737)
    word = (word >> wp.uint32(22)) ^ word
    u = wp.float32(word >> wp.uint32(8)) * wp.float32(5.9604644775390625e-08)
    return lo + (hi - lo) * u


@wp.func
def wrap_to_pi(angle: wp.float32) -> wp.float32:
    """Wrap an angle to ``[-pi, pi]`` like :func:`isaaclab.utils.math.wrap_to_pi`."""
    two_pi = wp.float32(2.0 * math.pi)
    shifted = angle + wp.float32(math.pi)
    wrapped = shifted - two_pi * wp.floor(shifted / two_pi)
    if wrapped == 0.0 and angle > 0.0:
        return wp.float32(math.pi)
    return wrapped - wp.float32(math.pi)


@wp.func
def row_vec3(a: Any, env: int, start: int) -> wp.vec3:
    """Three consecutive values of a 2D row."""
    return wp.vec3(a[env, start], a[env, start + 1], a[env, start + 2])


@wp.func
def row_quat(a: Any, env: int, start: int) -> wp.quat:
    """Four consecutive values ``xyzw`` of a 2D row."""
    return wp.quat(a[env, start], a[env, start + 1], a[env, start + 2], a[env, start + 3])


@wp.func
def body_vec3(a: Any, env: int, body: int, start: int) -> wp.vec3:
    """Three consecutive values of a ``(N, B, 7)`` pose row."""
    return wp.vec3(a[env, body, start], a[env, body, start + 1], a[env, body, start + 2])


@wp.func
def body_quat(a: Any, env: int, body: int) -> wp.quat:
    """Orientation ``xyzw`` of a ``(N, B, 7)`` pose row."""
    return wp.quat(a[env, body, 3], a[env, body, 4], a[env, body, 5], a[env, body, 6])


@wp.func
def quat_from_euler_xyz(roll: wp.float32, pitch: wp.float32, yaw: wp.float32) -> wp.quat:
    """Quaternion ``xyzw`` from XYZ Euler angles, like :func:`isaaclab.utils.math.quat_from_euler_xyz`."""
    cy = wp.cos(yaw * 0.5)
    sy = wp.sin(yaw * 0.5)
    cr = wp.cos(roll * 0.5)
    sr = wp.sin(roll * 0.5)
    cp = wp.cos(pitch * 0.5)
    sp = wp.sin(pitch * 0.5)
    return wp.quat(
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
        cy * cr * cp + sy * sr * sp,
    )


@wp.func
def quat_error_magnitude(q1: wp.quat, q2: wp.quat) -> wp.float32:
    """Rotation angle between two quaternions, like :func:`isaaclab.utils.math.quat_error_magnitude`."""
    d = q1 * wp.quat_inverse(q2)
    xyz = wp.vec3(d[0], d[1], d[2])
    w = d[3]
    if w < 0.0:
        xyz = -xyz
        w = -w
    return 2.0 * wp.atan2(wp.length(xyz), w)


@wp.func
def root_quat(f: Any, env: int) -> wp.quat:
    """Root link orientation in the world frame."""
    return row_quat(f.root_pose_w, env, 3)


@wp.func
def root_lin_vel_b(f: Any, env: int) -> wp.vec3:
    """Root COM linear velocity in the root link frame (Isaac Lab's ``root_lin_vel_b``)."""
    return wp.quat_rotate_inv(root_quat(f, env), row_vec3(f.root_vel_w, env, 0))


@wp.func
def root_ang_vel_b(f: Any, env: int) -> wp.vec3:
    """Root angular velocity in the root link frame (Isaac Lab's ``root_ang_vel_b``)."""
    return wp.quat_rotate_inv(root_quat(f, env), row_vec3(f.root_vel_w, env, 3))


@wp.func
def projected_gravity_b(f: Any, env: int) -> wp.vec3:
    """Unit gravity direction ``(0, 0, -1)`` in the root link frame."""
    return wp.quat_rotate_inv(root_quat(f, env), wp.vec3(0.0, 0.0, -1.0))


@wp.func
def heading_w(f: Any, env: int) -> wp.float32:
    """Yaw of the root link's x axis in the world frame (Isaac Lab's ``heading_w``)."""
    forward = wp.quat_rotate(root_quat(f, env), wp.vec3(1.0, 0.0, 0.0))
    return wp.atan2(forward[1], forward[0])
