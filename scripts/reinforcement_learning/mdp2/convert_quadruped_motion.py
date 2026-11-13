# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a looping quadruped clip to the rows that the MDP2 motion tracking terms read.

    uv run python scripts/reinforcement_learning/mdp2/convert_quadruped_motion.py trot0.txt trot0.csv

The input is a clip of a dog, retargeted to the Unitree A1, as in the ``datasets/mocap_motions`` folder of
https://github.com/escontra/AMP_for_hardware: a JSON file whose frames hold the position of the root [m], its
orientation as a quaternion (x, y, z, w), and twelve joint angles [rad], for the legs FR, FL, RR, and RL. The clip
is one cycle of a gait, which the script repeats for ``--seconds``: every cycle starts where the last one ended.

The output has one row per frame with those nineteen values. ``--scale`` scales the positions of the root, for a
robot whose legs are longer than the A1's: 1.065 for the Unitree Go2.
"""

import argparse
import json

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("destination")
parser.add_argument("--seconds", type=float, default=20.0, help="Duration of the output")
parser.add_argument("--scale", type=float, default=1.065, help="Scale of the root positions")
args = parser.parse_args()


def multiply(a, b):
    """Return the product of two quaternions (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def rotate(q, v):
    """Return the vector ``v`` rotated by the quaternion ``q`` (x, y, z, w)."""
    return multiply(multiply(q, np.array([*v, 0.0])), q * np.array([-1.0, -1.0, -1.0, 1.0]))[:3]


with open(args.source) as file:
    clip = json.load(file)
frames = np.array(clip["Frames"])[:, :19]
frames[:, :3] *= args.scale

# The last frame of a cycle is the first frame of the next: the turn about the vertical axis and the position
# that take the first frame to the last carry every cycle on from the one before.
first, last = frames[0], frames[-1]
turn = multiply(last[3:7], first[3:7] * np.array([-1.0, -1.0, -1.0, 1.0]))
turn = np.array([0.0, 0.0, turn[2], turn[3]]) / np.hypot(turn[2], turn[3])

rows, rotation, origin = [], np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3)
while len(rows) * clip["FrameDuration"] < args.seconds:
    for frame in frames[:-1]:
        position = origin + rotate(rotation, frame[:3] - first[:3] * np.array([1.0, 1.0, 0.0]))
        rows.append([*position, *multiply(rotation, frame[3:7]), *frame[7:]])
    origin = origin + rotate(rotation, (last[:3] - first[:3]) * np.array([1.0, 1.0, 0.0]))
    rotation = multiply(rotation, turn)

np.savetxt(args.destination, np.array(rows), delimiter=",", fmt="%.6f")
print(f"{args.destination}: {len(rows)} frames at {1.0 / clip['FrameDuration']:.2f} fps")
