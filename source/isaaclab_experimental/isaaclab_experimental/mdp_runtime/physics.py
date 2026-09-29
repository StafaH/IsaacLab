# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics bindings: the simulator side of an MDP program.

A binding exposes named ``wp.array`` fields at fixed addresses and two capture-safe operations. The built-in
terms use this joint-space schema (``N`` environments, ``J`` joints, float32):

========================== ============== =====================================================
Field                      Shape          Meaning
========================== ============== =====================================================
``joint_pos``              ``(N, J)``     Joint positions. Reset events may write them.
``joint_vel``              ``(N, J)``     Joint velocities. Events may write them.
``default_joint_pos``      ``(N, J)``     Default joint positions.
``default_joint_vel``      ``(N, J)``     Default joint velocities.
``soft_joint_pos_limits``  ``(N, J, 2)``  Soft position limits ``[lower, upper]``.
``soft_joint_vel_limits``  ``(N, J)``     Soft velocity limits.
``joint_effort_target``    ``(N, J)``     Effort command read by :meth:`PhysicsBinding.step`.
========================== ============== =====================================================
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import warp as wp


class PhysicsBinding(Protocol):
    """Simulator interface used by :class:`~isaaclab_experimental.mdp_runtime.MdpProgram`.

    Fields must keep their addresses for the binding's lifetime. :meth:`step` and :meth:`commit` must only
    enqueue capture-safe work on the current Warp stream: no allocation, host synchronization, or branches on
    device data.
    """

    num_envs: int
    device: str
    step_dt: float
    """Control period [s]: one :meth:`step` advances the simulation by this duration."""
    joint_names: Sequence[str]
    fields: Mapping[str, wp.array]

    def step(self) -> None:
        """Apply ``joint_effort_target`` and advance one control period."""
        ...

    def commit(self, mask: wp.array) -> None:
        """Make event writes to ``joint_pos``/``joint_vel`` of the masked environments effective."""
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
    simulation state, so :meth:`commit` has nothing to do.
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
        self.step_dt = physics_dt * decimation
        self._physics_dt = physics_dt
        self._decimation = decimation
        self._inv_mass = 1.0 / mass
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
            inputs=[f["joint_effort_target"], self._inv_mass, physics_dt, f["joint_pos"], f["joint_vel"]],
            device=device,
            record_cmd=True,
        )

    def step(self) -> None:
        for _ in range(self._decimation):
            self._substep.launch()

    def commit(self, mask: wp.array) -> None:
        pass
