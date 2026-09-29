# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics bindings: the simulator side of an MDP program.

A binding exposes named ``wp.array`` fields at fixed addresses and capture-safe operations. The built-in
terms use this schema (``N`` environments, ``J`` joints, ``B`` bodies, ``S`` contact bodies, ``T`` contact
history; float32; poses ``[px, py, pz, qx, qy, qz, qw]``, velocities ``[lin, ang]``):

================================== ================ ==========================================================
Field                              Shape            Meaning
================================== ================ ==========================================================
``joint_pos``, ``joint_vel``       ``(N, J)``       Joint state. Reset events write it.
``joint_acc``                      ``(N, J)``       Joint acceleration.
``default_joint_pos``/``_vel``     ``(N, J)``       Default joint state.
``soft_joint_pos_limits``          ``(N, J, 2)``    Soft position limits ``[lower, upper]``.
``soft_joint_vel_limits``          ``(N, J)``       Soft velocity limits.
``joint_pos_target``               ``(N, J)``       Position command applied by :meth:`PhysicsBinding.step`.
``joint_effort_target``            ``(N, J)``       Effort command applied by :meth:`PhysicsBinding.step`.
``applied_effort``                 ``(N, J)``       Effort applied by the actuators.
``root_pose_w``                    ``(N, 7)``       Root link pose in the world frame. Events write it.
``root_vel_w``                     ``(N, 6)``       Root COM velocity in the world frame. Events write it.
``default_root_pose``/``_vel``     ``(N, 7)``/(N,6) Default root state, pose relative to the env origin.
``env_origins``                    ``(N, 3)``       Environment origins in the world frame.
``body_pose_w``                    ``(N, B, 7)``    Link poses in the world frame.
``contact_net_normal_forces_history`` ``(N,T,S,3)`` Normal contact forces, index 0 newest.
``contact_current_air_time``       ``(N, S)``       Time since the body left contact [s].
``contact_last_air_time``          ``(N, S)``       Duration of the last air phase [s].
``contact_current_contact_time``   ``(N, S)``       Time since the body touched down [s].
================================== ================ ==========================================================

A binding provides the subset its simulator supports; compilation rejects terms reading missing fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import warp as wp


class PhysicsBinding(Protocol):
    """Simulator interface used by :class:`~isaaclab_experimental.mdp_runtime.MdpProgram`.

    Fields must keep their addresses for the binding's lifetime. :meth:`step`, :meth:`reset`, and :meth:`commit`
    must only enqueue capture-safe work on the current Warp stream: no allocation, host synchronization, or
    branches on device data.
    """

    num_envs: int
    device: str
    physics_dt: float
    """Physics step [s]."""
    step_dt: float
    """Control period [s]: one :meth:`step` advances the simulation by this duration."""
    joint_names: Sequence[str]
    body_names: Sequence[str]
    contact_body_names: Sequence[str]
    fields: Mapping[str, wp.array]

    def prepare(self, reads: set[str], writes: set[str]) -> None:
        """Receive the fields the program's terms read and write, before any step (outside capture)."""
        ...

    def step(self) -> None:
        """Apply the command fields and advance one control period."""
        ...

    def reset(self, mask: wp.array) -> None:
        """Reset simulator-internal state (e.g. sensors) of the masked environments."""
        ...

    def commit(self, mask: wp.array) -> None:
        """Make event writes to state fields of the masked environments effective."""
        ...


@wp.kernel
def _integrate(
    effort: wp.array2d(dtype=wp.float32),
    inv_mass: wp.float32,
    dt: wp.float32,
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    v = joint_vel[i, j] + effort[i, j] * inv_mass * dt
    joint_vel[i, j] = v
    joint_pos[i, j] = joint_pos[i, j] + v * dt


class PointMassPhysics:
    """Analytic unit point masses on independent prismatic joints, integrated with semi-implicit Euler.

    It needs no simulator, so it serves physics-free tests and the heterogeneous example. Fields are the
    simulation state, so :meth:`reset` and :meth:`commit` have nothing to do.
    """

    def __init__(
        self,
        num_envs: int,
        joint_names: Sequence[str],
        physics_dt: float,
        decimation: int,
        device: str,
        mass: float = 1.0,
        position_limit: float = 10.0,
        velocity_limit: float = 10.0,
    ):
        self.num_envs = num_envs
        self.device = device
        self.joint_names = tuple(joint_names)
        self.body_names = ()
        self.contact_body_names = ()
        self.physics_dt = physics_dt
        self.step_dt = physics_dt * decimation
        self._decimation = decimation
        shape = (num_envs, len(self.joint_names))
        limits = np.empty((*shape, 2), dtype=np.float32)
        limits[..., 0], limits[..., 1] = -position_limit, position_limit
        self.fields = {
            "joint_pos": wp.zeros(shape, dtype=wp.float32, device=device),
            "joint_vel": wp.zeros(shape, dtype=wp.float32, device=device),
            "default_joint_pos": wp.zeros(shape, dtype=wp.float32, device=device),
            "default_joint_vel": wp.zeros(shape, dtype=wp.float32, device=device),
            "soft_joint_pos_limits": wp.array(limits, dtype=wp.float32, device=device),
            "soft_joint_vel_limits": wp.full(shape, velocity_limit, dtype=wp.float32, device=device),
            "joint_effort_target": wp.zeros(shape, dtype=wp.float32, device=device),
        }
        f = self.fields
        self._substep = wp.launch(
            _integrate,
            dim=shape,
            inputs=[f["joint_effort_target"], 1.0 / mass, physics_dt, f["joint_pos"], f["joint_vel"]],
            device=device,
            record_cmd=True,
        )

    def prepare(self, reads: set[str], writes: set[str]) -> None:
        pass

    def step(self) -> None:
        for _ in range(self._decimation):
            self._substep.launch()

    def reset(self, mask: wp.array) -> None:
        pass

    def commit(self, mask: wp.array) -> None:
        pass
