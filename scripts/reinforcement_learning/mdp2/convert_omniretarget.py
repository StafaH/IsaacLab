# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the terrain clips of the OmniRetarget dataset to the files that the MDP2 parkour task reads.

    uv run python scripts/reinforcement_learning/mdp2/convert_omniretarget.py <dataset> <motions folder>

The input is a copy of https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset with ``robot-terrain.zip``
unpacked: clips of the Unitree G1 climbing boxes, ``robot-terrain/<clip>_z_scale_<s>.npz``, and the boxes of each,
``models/terrain/<clip>/``. A clip holds one row per frame with the orientation of the root as a quaternion
(w, x, y, z), its position [m], and the positions of the 29 joints [rad].

The output has, per clip and scale, a CSV file with one row per frame, holding the root position, the root quaternion
(x, y, z, w), and the joints, and a JSON file with the boxes: their centre [m], size [m], and turn about the vertical
axis [rad].
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.interpolate import CubicSpline

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("destination")
parser.add_argument("--fps", type=float, default=50.0, help="Rate of the converted clips, that of the controller")
args = parser.parse_args()

os.makedirs(args.destination, exist_ok=True)
for path in sorted(glob.glob(os.path.join(args.source, "robot-terrain", "*.npz"))):
    name, scale = re.fullmatch(r"(.*)_z_scale_(.*)\.npz", os.path.basename(path)).groups()
    with np.load(path) as clip:
        qpos, fps = clip["qpos"], float(clip["fps"])
    rows = np.concatenate([qpos[:, 4:7], qpos[:, 1:4], qpos[:, :1], qpos[:, 7:]], axis=1)
    # A spline through the frames gives the clip at the rate of the controller with smooth velocities. The
    # quaternions keep one sign along the clip, and are normalized after.
    flips = np.sign(np.sum(rows[1:, 3:7] * rows[:-1, 3:7], axis=1))
    rows[1:, 3:7] *= np.cumprod(np.where(flips == 0.0, 1.0, flips))[:, None]
    times = np.arange(len(rows)) / fps
    rows = CubicSpline(times, rows)(np.arange(round(times[-1] * args.fps) + 1) / args.fps)
    rows[:, 3:7] /= np.linalg.norm(rows[:, 3:7], axis=1, keepdims=True)
    fps = args.fps

    # A box is a file of eight vertices, bottom and top of each corner in turn, which the terrain file scales.
    terrain = os.path.join(args.source, "models", "terrain", name)
    with open(os.path.join(terrain, f"multi_boxes_z_scale_{scale}.urdf")) as file:
        scales = dict(re.findall(r'filename="box_models/(.*?)\.obj" scale="(.*?)"', file.read()))
    boxes = []
    for box, factors in sorted(scales.items()):
        with open(os.path.join(terrain, "box_models", f"{box}.obj")) as file:
            vertices = np.array([line.split()[1:4] for line in file if line.startswith("v ")], dtype=float)
        vertices *= np.array(factors.split(), dtype=float)
        corner, side, front, height = vertices[0], vertices[2] - vertices[0], vertices[4] - vertices[0], vertices[1, 2]
        centre = corner + 0.5 * (side + front)
        boxes.append(
            {
                "centre": [centre[0], centre[1], 0.5 * height],
                "size": [np.linalg.norm(front), np.linalg.norm(side), height],
                "yaw": np.arctan2(front[1], front[0]),
            }
        )

    stem = os.path.join(args.destination, f"{name}_z{scale}")
    np.savetxt(f"{stem}.csv", rows, delimiter=",", fmt="%.6f")
    with open(f"{stem}.json", "w") as file:
        json.dump({"fps": fps, "frames": len(rows), "boxes": boxes}, file, indent=1)
    print(f"{stem}: {len(rows)} frames at {fps:.0f} fps, {len(boxes)} boxes")
