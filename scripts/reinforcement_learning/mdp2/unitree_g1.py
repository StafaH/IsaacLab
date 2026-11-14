# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A Unitree G1 with 29 joints over the low-level messages of ``unitree_sdk2py``, for ``deploy_interfaces.py``.

This file has not run on a robot yet. The first run belongs on a hoist, with the feet off the ground.
"""

import struct
import time

import numpy as np
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
MOTORS = (
    *(f"left_{joint}_joint" for joint in LEG),
    *(f"right_{joint}_joint" for joint in LEG),
    *(f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")),
    *(f"left_{joint}_joint" for joint in ARM),
    *(f"right_{joint}_joint" for joint in ARM),
)
"""The joints in the order of the robot's motors."""

BUTTONS = {
    "R1": 0,
    "L1": 1,
    "start": 2,
    "select": 3,
    "R2": 4,
    "L2": 5,
    "F1": 6,
    "F2": 7,
    "A": 8,
    "B": 9,
    "X": 10,
    "Y": 11,
}
"""The bit of every button of the remote in its message."""


class UnitreeG1:
    """The robot on the network interface ``net``, read and written at the level of its motors.

    The robot's own controller is told to let go of the motors, so the robot must hang or be held before this.
    """

    def __init__(self, net):
        ChannelFactoryInitialize(0, net)
        switcher = MotionSwitcherClient()
        switcher.SetTimeout(5.0)
        switcher.Init()
        _, mode = switcher.CheckMode()
        while mode["name"]:
            switcher.ReleaseMode()
            time.sleep(1.0)
            _, mode = switcher.CheckMode()
        self.state = None
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._receive, 10)
        while self.state is None:
            time.sleep(0.01)
        self.command = unitree_hg_msg_dds__LowCmd_()
        # The ankles and the waist are commanded as pitch and roll, not as the two motors of each.
        self.command.mode_pr = 0
        self.command.mode_machine = self.state.mode_machine
        self.crc = CRC()

    def _receive(self, state):
        self.state = state

    def read(self):
        """Return the orientation (x, y, z, w) and the angular velocity of the pelvis, in its own axes, and the
        positions and velocities of the joints."""
        state = self.state
        motors = state.motor_state[: len(MOTORS)]
        return (
            np.array(state.imu_state.quaternion)[[1, 2, 3, 0]],
            np.array(state.imu_state.gyroscope),
            np.array([motor.q for motor in motors]),
            np.array([motor.dq for motor in motors]),
        )

    def write(self, targets, stiffness, damping):
        """Send the targets [rad] of the joints with the gains of their motors."""
        for k, motor in enumerate(self.command.motor_cmd[: len(MOTORS)]):
            motor.mode = 1
            motor.q, motor.dq, motor.tau = float(targets[k]), 0.0, 0.0
            motor.kp, motor.kd = float(stiffness[k]), float(damping[k])
        self.command.crc = self.crc.Crc(self.command)
        self.publisher.Write(self.command)

    def pressed(self, button):
        """Return whether a button of the remote is down."""
        keys = struct.unpack("H", bytes(self.state.wireless_remote[2:4]))[0]
        return bool(keys >> BUTTONS[button] & 1)
