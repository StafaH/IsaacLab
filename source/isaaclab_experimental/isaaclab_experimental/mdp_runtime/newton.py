# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics binding for an Isaac Lab articulation, and optionally a contact sensor, simulated by Newton.

.. warning::
    Isaac Lab exposes no public capture-safe stepping API. This binding calls
    ``NewtonManager._simulate_full`` (actuators, solver substeps, sensors) and ``NewtonManager.forward`` so
    that the physics launches are recorded into the caller's graph, as ``NewtonManager.step`` would otherwise
    launch its own graph. Revalidate it when the Newton manager changes.
"""

from __future__ import annotations

import warp as wp

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene

_CONTACT_FIELDS = (
    "contact_net_normal_forces_history",
    "contact_current_air_time",
    "contact_last_air_time",
    "contact_current_contact_time",
)


class NewtonPhysics:
    """Bind one articulation (and optionally one contact sensor) of a Newton scene to the field schema.

    State fields are the articulation's data arrays: events write them, and :meth:`commit` writes the masked
    rows back to the solver. Derived quantities (body-frame velocities, projected gravity, heading) are computed
    by the terms from the raw root state, not read from Isaac Lab's lazily computed properties. Host-side
    simulation time counters do not advance.

    Limitations: actuators must be stateless (implicit or DC motor), because actuator state resets need
    environment indices; the model must not change at run time.

    Args:
        scene: The scene, after :meth:`~isaaclab.sim.SimulationContext.reset`.
        articulation: Name of the articulation in the scene.
        decimation: Physics steps per control step.
        contact_sensor: Name of a contact sensor with ``track_air_time=True``, or None.
    """

    def __init__(self, scene: InteractiveScene, articulation: str, decimation: int, contact_sensor: str | None = None):
        from isaaclab_newton.physics import NewtonManager

        self._manager = NewtonManager
        self._scene = scene
        self._robot = robot = scene[articulation]
        stateful = [
            name for name, cfg in robot.cfg.actuators.items() if not isinstance(cfg, (ImplicitActuatorCfg, DCMotorCfg))
        ]
        if stateful:
            raise ValueError(
                f"Only stateless actuators (implicit, DC motor) are supported; '{articulation}' has {stateful}."
            )
        # Step physics like Isaac Lab environments do: when Newton runs every actuator inside the solver step, it
        # executes the whole decimation loop per call, and data is updated once per control step; otherwise
        # actions are applied, physics stepped, and data updated once per physics step.
        NewtonManager.set_decimation(decimation)
        self._fused = NewtonManager.handles_decimation()
        if not self._fused:
            NewtonManager.set_decimation(1)
        self._repeats = 1 if self._fused else decimation
        self.physics_dt = scene.physics_dt
        self.step_dt = self.physics_dt * decimation
        self.num_envs = scene.num_envs
        self.device = str(robot.device)
        self.joint_names = tuple(robot.joint_names)
        self.body_names = tuple(robot.body_names)
        data = robot.data
        shape = data.joint_pos.warp.shape
        self._root_pose = data.root_link_pose_w.warp
        self._root_vel = data.root_com_vel_w.warp
        self.fields = {
            "joint_pos": data.joint_pos.warp,
            "joint_vel": data.joint_vel.warp,
            "joint_acc": data.joint_acc.warp,
            "default_joint_pos": data.default_joint_pos.warp,
            "default_joint_vel": data.default_joint_vel.warp,
            "soft_joint_pos_limits": data.soft_joint_pos_limits.warp.view(wp.float32),
            "soft_joint_vel_limits": data.soft_joint_vel_limits.warp,
            "joint_pos_target": wp.clone(data.default_joint_pos.warp),
            "joint_effort_target": wp.zeros(shape, dtype=wp.float32, device=self.device),
            "applied_effort": robot.actuators.applied_effort.warp,
            "root_pose_w": self._root_pose.view(wp.float32),
            "root_vel_w": self._root_vel.view(wp.float32),
            "default_root_pose": data.default_root_pose.warp.view(wp.float32),
            "default_root_vel": data.default_root_vel.warp.view(wp.float32),
            "env_origins": wp.from_torch(scene.env_origins.contiguous()),
            "body_pose_w": data.body_link_pose_w.warp.view(wp.float32),
        }
        self._sensor = None
        self.contact_body_names = ()
        if contact_sensor is not None:
            self._sensor = sensor = scene[contact_sensor]
            if not sensor.cfg.track_air_time:
                raise ValueError(f"Contact sensor '{contact_sensor}' must track air time.")
            self.contact_body_names = tuple(sensor.body_names)
            sd = sensor.data
            self.fields.update(
                contact_net_normal_forces_history=sd.net_normal_forces_w_history.warp.view(wp.float32),
                contact_current_air_time=sd.current_air_time.warp,
                contact_last_air_time=sd.last_air_time.warp,
                contact_current_contact_time=sd.current_contact_time.warp,
            )
        self._apply_position = self._apply_effort = self._read_contacts = self._read_bodies = False
        self._write_root_pose = self._write_root_vel = self._write_joints = False

    def prepare(self, reads: set[str], writes: set[str]) -> None:
        """Apply only the commands that terms write, refresh only the data that terms read."""
        self._apply_position = "joint_pos_target" in writes
        self._apply_effort = "joint_effort_target" in writes
        self._read_contacts = self._sensor is not None and bool(set(_CONTACT_FIELDS) & reads)
        self._read_bodies = "body_pose_w" in reads
        self._write_root_pose = "root_pose_w" in writes
        self._write_root_vel = "root_vel_w" in writes
        self._write_joints = bool({"joint_pos", "joint_vel"} & writes)

    def step(self) -> None:
        dt = self.physics_dt * self._manager._decimation
        for _ in range(self._repeats):
            self._submit()
            self._manager._simulate_full()
            self._scene.update(dt=dt)
        self._refresh()

    def _submit(self) -> None:
        commands = self._robot.actuators.target_command
        if self._apply_position:
            commands.set_position_mask(value=self.fields["joint_pos_target"])
        if self._apply_effort:
            commands.set_effort_mask(value=self.fields["joint_effort_target"])
        self._robot.write_data_to_sim()

    def reset(self, mask: wp.array) -> None:
        if self._sensor is not None:
            self._sensor.reset(env_mask=mask)

    def commit(self, mask: wp.array) -> None:
        robot, f = self._robot, self.fields
        if self._write_root_pose:
            robot.write_root_pose_to_sim_mask(root_pose=self._root_pose, env_mask=mask)
        if self._write_root_vel:
            robot.write_root_velocity_to_sim_mask(root_velocity=self._root_vel, env_mask=mask)
        if self._write_joints:
            robot.write_joint_state_to_sim_mask(position=f["joint_pos"], velocity=f["joint_vel"], env_mask=mask)
        self._manager.forward()

    def _refresh(self) -> None:
        # Accessing the lazily updated data records its refresh kernels into the current capture.
        if self._read_contacts:
            self._sensor.data  # noqa: B018
        if self._read_bodies:
            self._robot.data.body_link_pose_w  # noqa: B018
