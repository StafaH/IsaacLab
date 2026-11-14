# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A MuJoCo model of a robot behind the calls of a real one, for the policy of ``interface_export.py``."""

import mujoco
import numpy as np
from interface_export import to_root, turn


class MujocoRobot:
    """A robot in MuJoCo that is read and written as a real one: its joints are those that the policy moves.

    ``scene`` is a MuJoCo scene with the robot on a floor and one actuator per joint that the policy moves. The
    actuators become position actuators with the gains of every ``write``. ``armature`` gives the joints the
    armature of the export instead of the model's, ``timestep`` is the physics step, rounded to divide the control
    step, and ``effort_limits`` holds the torques within the limits of the export.
    """

    def __init__(self, scene, policy, armature=False, timestep=None, effort_limits=False):
        spec = mujoco.MjSpec.from_file(scene)
        try:
            self.model = spec.compile()
        except ValueError:
            # The meshes of the model are not here: they are what it looks like, so the robot is simulated without.
            for geom in [geom for geom in spec.geoms if geom.type == mujoco.mjtGeom.mjGEOM_MESH]:
                spec.delete(geom)
            for mesh in list(spec.meshes):
                spec.delete(mesh)
            self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.clip = policy.clip
        self.hung = None

        model, export = self.model, policy.spec
        joints = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, export["joints"][k]) for k in policy.action_joints
        ]
        self.actuators = []
        for joint, k in zip(joints, policy.action_joints):
            owners = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
            if joint < 0 or len(owners) != 1:
                raise ValueError(f"The model needs one actuator for the joint {export['joints'][k]}")
            self.actuators.append(owners[0])
        self.qpos_ids, self.dof_ids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.clip_joints = np.array(policy.action_joints)
        # Position actuators without a range for their targets.
        model.actuator_gaintype[self.actuators] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[self.actuators] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[self.actuators, :3] = 0.0
        model.actuator_biasprm[self.actuators, :3] = 0.0
        model.actuator_ctrllimited[self.actuators] = 0
        model.actuator_forcelimited[self.actuators] = int(effort_limits)
        limits = np.array(export["effort_limits"])[self.clip_joints]
        model.actuator_forcerange[self.actuators] = np.stack([-limits, limits], axis=1)
        if armature:
            model.dof_armature[self.dof_ids] = np.array(export["armature"])[self.clip_joints]
        self.substeps = max(1, round(export["step_dt"] / (timestep or model.opt.timestep)))
        model.opt.timestep = export["step_dt"] / self.substeps

    def reset(self, frame, yaw=0.0, hung=False):
        """Put the robot on a frame of the clip, turned by ``yaw`` [rad] about the vertical axis through its root.

        It moves as the clip does there. With ``hung`` it is at rest instead, and its root stays where it is until
        ``hung`` is set to None: a robot on a hoist.
        """
        data, clip = self.data, self.clip
        quat = clip["root_pose"][frame, 3:]
        turned = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
        mujoco.mj_resetData(self.model, data)
        data.qpos[:3] = clip["root_pose"][frame, :3]
        mujoco.mju_mulQuat(data.qpos[3:7], turned, quat[[3, 0, 1, 2]])
        data.qpos[self.qpos_ids] = clip["joint_pos"][frame, self.clip_joints]
        if not hung:
            data.qvel[:3] = turn(clip["root_vel"][frame, :3], yaw)
            data.qvel[3:6] = to_root(quat, clip["root_vel"][frame, 3:])
            data.qvel[self.dof_ids] = clip["joint_vel"][frame, self.clip_joints]
        self.hung = data.qpos[:7].copy() if hung else None
        mujoco.mj_forward(self.model, data)

    def read(self):
        """Return the orientation (x, y, z, w) and the angular velocity of the root, in its own axes, and the
        positions and velocities of the joints."""
        data = self.data
        return data.qpos[[4, 5, 6, 3]], data.qvel[3:6].copy(), data.qpos[self.qpos_ids], data.qvel[self.dof_ids]

    def write(self, targets, stiffness, damping):
        """Hold the targets [rad] of the joints with the gains for one control step."""
        self.model.actuator_gainprm[self.actuators, 0] = stiffness
        self.model.actuator_biasprm[self.actuators, 1] = -np.asarray(stiffness)
        self.model.actuator_biasprm[self.actuators, 2] = -np.asarray(damping)
        self.data.ctrl[self.actuators] = targets
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            if self.hung is not None:
                self.data.qpos[:7], self.data.qvel[:6] = self.hung, 0.0
