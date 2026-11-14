# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an export of ``export_interfaces.py`` on a Unitree G1 with 29 joints, or rehearse the run in MuJoCo.

    python deploy_interfaces.py --export export/dance1 --net enp3s0
    python deploy_interfaces.py --export export/dance1 --mujoco scene.xml

The script needs NumPy and the ONNX runtime, with ``unitree_sdk2py`` for the robot or MuJoCo for the rehearsal. A
run has four stages:

1. The motors only damp. The robot hangs from a hoist or is held, and START goes on.
2. The joints move in ``--ramp`` seconds to the pose of the clip on ``--frame`` and hold it. The robot is set on
   the ground in that pose, facing anywhere: the clip is turned onto it. A goes on.
3. The policy tracks the clip up to ``--last`` and then holds that frame.
4. SELECT ends the run at any time, and so does a lean of the root past ``--lean`` while the policy runs: the
   motors damp again and the script exits. The robot sinks then, so it is held or on the hoist.

The robot starts at rest, so ``--frame`` and ``--last`` are frames where the clip stands still on both feet. The
policy does not know where the robot is: it follows the motion of the clip, not its path, and the robot ends up
to a few meters from where the clip does. The targets of the joints stay within what the torque limits of the
export allow at its gains.

The rehearsal goes through the same stages on a MuJoCo scene with one actuator per joint, with the buttons pressed
at once, the robot hung by its root until the policy starts and turned by ``--yaw`` from the clip. It ends two
seconds after the clip and reports how far the robot leaned and went.
"""

import argparse
import sys
import time

import numpy as np
from interface_export import ExportedPolicy, to_root

parser = argparse.ArgumentParser()
parser.add_argument("--export", required=True, help="Folder written by export_interfaces.py")
parser.add_argument("--net", default=None, help="Network interface of the robot")
parser.add_argument("--mujoco", default=None, help="MuJoCo scene of the robot on a floor, to rehearse the run in")
parser.add_argument("--frame", type=int, default=0, help="Frame of the clip on which the policy starts")
parser.add_argument("--last", type=int, default=None, help="Frame that the policy holds at the end; the last one")
parser.add_argument("--ramp", type=float, default=3.0, help="Seconds of the move to the first pose")
parser.add_argument("--lean", type=float, default=1.0, help="Lean of the root [rad] that ends the run")
parser.add_argument("--yaw", type=float, default=0.0, help="Turn of the rehearsed robot from the clip [rad]")
args = parser.parse_args()

policy = ExportedPolicy(args.export)
JOINTS = policy.action_joints
NAMES = tuple(policy.spec["joints"][joint] for joint in JOINTS)
STEP = policy.spec["step_dt"]
STIFFNESS, DAMPING, LIMITS = (np.array(policy.spec[name])[JOINTS] for name in ("stiffness", "damping", "effort_limits"))
SLACK = np.zeros(len(JOINTS)), np.full(len(JOINTS), 8.0)
"""The gains of motors that only damp."""
# The policy stays on the last frame that it knows of.
policy.frames = policy.frames if args.last is None else args.last + 1
if np.abs(policy.clip["joint_vel"][args.frame]).max() > 0.5:
    raise ValueError(f"The clip moves on the frame {args.frame}: the robot starts at rest, so choose a still frame")

# A run needs the messages of the robot and a rehearsal needs MuJoCo: neither needs the other.
if args.mujoco:
    from mujoco_robot import MujocoRobot

    robot = MujocoRobot(args.mujoco, policy, effort_limits=True)
    robot.reset(args.frame, args.yaw, hung=True)

    def pressed(button):
        return button != "select"
else:
    from unitree_g1 import MOTORS, UnitreeG1

    if NAMES != MOTORS:
        raise ValueError("The policy moves other joints than the motors of the robot, or in another order")
    robot = UnitreeG1(args.net)
    pressed = robot.pressed

# Joints that the policy does not move, such as fingers, read as their default pose.
joint_pos, joint_vel = policy.default.copy(), np.zeros(len(policy.default))
quat, ang_vel, joint_pos[JOINTS], joint_vel[JOINTS] = robot.read()
clock = time.perf_counter()


def step(targets, stiffness=STIFFNESS, damping=DAMPING):
    """Hold the targets of the joints for one control step, and read the robot after it."""
    global quat, ang_vel, clock
    reach = LIMITS / np.maximum(stiffness, 1e-6)
    robot.write(np.clip(targets, joint_pos[JOINTS] - reach, joint_pos[JOINTS] + reach), stiffness, damping)
    if not args.mujoco:
        clock = max(clock + STEP, time.perf_counter())
        time.sleep(clock - time.perf_counter())
    quat, ang_vel, joint_pos[JOINTS], joint_vel[JOINTS] = robot.read()
    if pressed("select"):
        sys.exit("SELECT: the run ends.")


try:
    print("The motors damp. Hang or hold the robot, then press START to move to the first pose of the clip.")
    while not pressed("start"):
        step(joint_pos[JOINTS], *SLACK)
    first, goal = joint_pos[JOINTS].copy(), policy.clip["joint_pos"][args.frame, JOINTS]
    for share in np.linspace(0.0, 1.0, round(args.ramp / STEP)):
        step(first + share * (goal - first))
    print("The robot holds the first pose. Set it on the ground, then press A to start the clip.")
    while not pressed("A"):
        step(goal)

    policy.reset(args.frame, quat)
    if args.mujoco:
        robot.hung, start = None, robot.data.qpos[:2].copy()
    leaned, held = 0.0, 0
    while not (args.mujoco and held * STEP >= 2.0):
        step(policy.step(quat, ang_vel, joint_pos, joint_vel))
        lean = np.arccos(np.clip(-to_root(quat, np.array([0.0, 0.0, -1.0]))[2], -1.0, 1.0))
        leaned = max(leaned, lean)
        if lean > args.lean:
            sys.exit(f"The robot leans by {lean:.2f} rad on frame {policy.frame}: the run ends.")
        held += policy.frame == policy.frames - 1
        if held == 1:
            print("The clip is over and the policy holds its last frame. Press SELECT to end the run.")
    travel = np.linalg.norm(robot.data.qpos[:2] - start)
    print(
        f"REHEARSAL frames {args.frame}..{policy.frame}, largest lean {leaned:.2f} rad, {travel:.2f} m from the start"
    )
finally:
    for _ in range(round(2.0 / STEP)):
        robot.write(joint_pos[JOINTS], *SLACK)
        time.sleep(0.0 if args.mujoco else STEP)
