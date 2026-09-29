# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics binding for an Isaac Lab articulation simulated by Newton.

.. warning::
    Isaac Lab exposes no public capture-safe stepping API. This binding calls
    ``NewtonManager._simulate_full`` (actuators, solver substeps, sensors) and ``NewtonManager.forward`` so
    that the physics launches are recorded into the caller's graph, as ``NewtonManager.step`` would otherwise
    launch its own graph. Revalidate it when the Newton manager changes.
"""

from __future__ import annotations

import warp as wp

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene


class NewtonArticulationPhysics:
    """Bind one articulation of a Newton-backed scene to the joint-space field schema.

    ``joint_pos`` and ``joint_vel`` are the articulation data arrays; events write them and :meth:`commit`
    writes the masked rows back to the solver. Host-side simulation time counters do not advance.

    Limitations: only stateless implicit actuators are accepted, because actuator state resets need
    environment indices; the model must not change at run time.
    """

    def __init__(self, scene: InteractiveScene, asset_name: str, decimation: int):
        from isaaclab_newton.physics import NewtonManager

        self._manager = NewtonManager
        self._scene = scene
        self._robot = robot = scene[asset_name]
        explicit = [name for name, cfg in robot.cfg.actuators.items() if not isinstance(cfg, ImplicitActuatorCfg)]
        if explicit:
            raise ValueError(f"Only stateless implicit actuators are supported; '{asset_name}' has {explicit}.")
        if decimation % NewtonManager._decimation:
            raise ValueError(f"decimation {decimation} is not a multiple of Newton's {NewtonManager._decimation}.")
        self._repeats = decimation // NewtonManager._decimation
        self._physics_dt = scene.physics_dt
        self.num_envs = scene.num_envs
        self.device = str(robot.device)
        self.step_dt = self._physics_dt * decimation
        self.joint_names = tuple(robot.joint_names)
        data = robot.data
        self.fields = {
            "joint_pos": data.joint_pos.warp,
            "joint_vel": data.joint_vel.warp,
            "default_joint_pos": data.default_joint_pos.warp,
            "default_joint_vel": data.default_joint_vel.warp,
            "soft_joint_pos_limits": data.soft_joint_pos_limits.warp.view(wp.float32),
            "soft_joint_vel_limits": data.soft_joint_vel_limits.warp,
            "joint_effort_target": wp.zeros(data.joint_pos.warp.shape, dtype=wp.float32, device=self.device),
        }

    def step(self) -> None:
        effort = self.fields["joint_effort_target"]
        for _ in range(self._repeats):
            self._robot.actuators.target_command.set_effort_mask(value=effort)
            self._robot.write_data_to_sim()
            self._manager._simulate_full()
            self._scene.update(dt=self._physics_dt * self._manager._decimation)

    def commit(self, mask: wp.array) -> None:
        f = self.fields
        self._robot.write_joint_state_to_sim_mask(position=f["joint_pos"], velocity=f["joint_vel"], env_mask=mask)
        self._manager.forward()
