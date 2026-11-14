# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The policy of a folder of ``export_interfaces.py``, from what a robot measures to the targets of its joints.

It needs only NumPy and the ONNX runtime, so that it runs on the computer of a robot.
"""

import json
import os

import numpy as np
import onnxruntime


def to_root(quat, vectors):
    """Return world ``vectors`` in the axes of a root with the orientation ``quat`` (x, y, z, w)."""
    cross = np.cross(quat[:3], vectors)
    return vectors - 2.0 * quat[3] * cross + 2.0 * np.cross(quat[:3], cross)


def turn(vectors, angle):
    """Return ``vectors`` turned by ``angle`` [rad] about the vertical axis."""
    cos, sin = np.cos(angle), np.sin(angle)
    return vectors @ np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]])


def heading(quat):
    """Return the angle [rad] about the vertical axis that a root with the orientation ``quat`` (x, y, z, w) faces."""
    x, y, z, w = quat
    return np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


class ExportedPolicy:
    """The policy of an export on its clip.

    ``reset`` puts it on a frame of the clip, and every ``step`` takes what the robot measures, answers the targets
    of the joints that the policy moves and goes to the next frame; on the last frame it stays there.
    """

    def __init__(self, folder):
        with open(os.path.join(folder, "spec.json")) as file:
            self.spec = json.load(file)
        self.clip = dict(np.load(os.path.join(folder, "clip.npz")))
        self.session = onnxruntime.InferenceSession(os.path.join(folder, "policy.onnx"))
        self.frames = len(self.clip["joint_pos"])
        self.interface = np.array(self.spec["interface"])
        self.action_joints = self.spec["action_joints"]
        self.default = np.array(self.spec["default_joint_pos"])
        self.reset(0, self.clip["root_pose"][0, 3:])

    def reset(self, frame, quat):
        """Start on a frame of the clip, with the heading of the robot as the heading of the clip there.

        The clip is turned about the vertical axis onto the robot with the orientation ``quat`` (x, y, z, w), so
        the robot may face anywhere, and what its orientation sensor calls north does not matter.
        """
        self.frame = frame
        self.turned = heading(quat) - heading(self.clip["root_pose"][frame, 3:])
        self.last_action = np.zeros(len(self.action_joints), dtype=np.float32)

    def observe(self, quat, ang_vel, joint_pos, joint_vel):
        """Return the observations of the policy on the current frame.

        ``quat`` (x, y, z, w) is the orientation of the root, ``ang_vel`` its angular velocity in its own axes, and
        ``joint_pos`` and ``joint_vel`` are in the order of the export's joints.
        """
        measured = {
            "base_ang_vel": ang_vel,
            "projected_gravity": to_root(quat, np.array([0.0, 0.0, -1.0])),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "actions": self.last_action,
            "interface": self.interface,
        }
        values = []
        for term in self.spec["observations"]:
            if term["name"] in measured:
                value = measured[term["name"]]
            elif "bodies" in term:
                # Where the bodies are some steps ahead, from the root of the clip now, in the axes of the robot's root.
                ahead = min(self.frame + term["ahead"], self.frames - 1)
                offsets = self.clip["body_pos"][ahead, term["bodies"]] - self.clip["root_pose"][self.frame, :3]
                value = self.interface[term["group"]] * to_root(quat, turn(offsets, self.turned)).ravel()
            else:
                ahead = min(self.frame + term["ahead"], self.frames - 1)
                value = self.interface[term["group"]] * self.clip["joint_pos"][ahead, term["joints"]]
            values.append(term["scale"] * np.asarray(value))
        return np.concatenate(values).astype(np.float32)

    def act(self, obs):
        """Return the action of the policy for its observations."""
        return self.session.run(None, {"obs": obs[None]})[0][0]

    def step(self, quat, ang_vel, joint_pos, joint_vel):
        """Return the targets [rad] of the joints that the policy moves, and go to the next frame."""
        self.last_action = self.act(self.observe(quat, ang_vel, joint_pos, joint_vel))
        self.frame = min(self.frame + 1, self.frames - 1)
        return self.default[self.action_joints] + self.spec["action_scale"] * self.last_action
