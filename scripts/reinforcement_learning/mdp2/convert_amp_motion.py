# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a poselib motion of the AMP humanoid (``.npy``, as in IsaacGymEnvs ``assets/amp/motions``) to the motion
format of the humanoid AMP task (``.npz``), resampled to 60 fps.

    python convert_amp_motion.py amp_humanoid_backflip.npy humanoid_backflip.npz

The source files are pickles, so they are read with an unpickler that only builds NumPy arrays and ordered
dictionaries. The conversion reproduces ``humanoid_walk.npz`` of the AMP task from ``amp_humanoid_walk.npy``.
"""

import collections  # noqa: F401
import pickle
import sys

import numpy as np

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("collections", "OrderedDict"),
}


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"refused {module}.{name}")
        return super().find_class(module, name)


def load(path):
    np.lib.format  # the file is a pickled object array: read the header, then unpickle with the allow list
    with open(path, "rb") as file:
        version = np.lib.format.read_magic(file)
        np.lib.format.read_array_header_1_0(file) if version == (1, 0) else np.lib.format.read_array_header_2_0(file)
        return Unpickler(file).load()


FPS = 60
JOINTS = {  # child body -> clip joint name, and whether it is a ball joint
    "torso": ("abdomen", True),
    "head": ("neck", True),
    "right_upper_arm": ("right_shoulder", True),
    "right_lower_arm": ("right_elbow", False),
    "left_upper_arm": ("left_shoulder", True),
    "left_lower_arm": ("left_elbow", False),
    "right_thigh": ("right_hip", True),
    "right_shin": ("right_knee", False),
    "right_foot": ("right_ankle", True),
    "left_thigh": ("left_hip", True),
    "left_shin": ("left_knee", False),
    "left_foot": ("left_ankle", True),
}


def mul(a, b):  # xyzw
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        -1,
    )


def rotate(q, v):
    u, w = q[..., :3], q[..., 3:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def conj(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def rotvec(q):
    q = np.where(q[..., 3:] < 0, -q, q)
    norm = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, q[..., 3:])
    return np.where(norm > 1e-9, q[..., :3] / np.maximum(norm, 1e-9) * angle, 2.0 * q[..., :3])


def slerp(a, b, t):
    b = np.where((a * b).sum(-1, keepdims=True) < 0, -b, b)
    dot = np.clip((a * b).sum(-1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(dot)
    near = theta < 1e-5
    s = np.where(near, 1.0, np.sin(theta))
    out = np.where(near, (1 - t) * a + t * b, (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / s)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


data = load(sys.argv[1])
data = data.item() if isinstance(data, np.ndarray) else data
names = list(data["skeleton_tree"]["node_names"])
parents = data["skeleton_tree"]["parent_indices"]["arr"]
offsets = data["skeleton_tree"]["local_translation"]["arr"].astype(np.float64)
local, root, fps = data["rotation"]["arr"], data["root_translation"]["arr"], float(data["fps"])
assert data["is_local"]

# Resample to 60 fps.
duration = (len(local) - 1) / fps
times = np.arange(int(np.floor(duration * FPS)) + 1) / FPS
index = np.minimum((times * fps).astype(int), len(local) - 2)
blend = (times * fps - index)[:, None, None]
local = slerp(local[index], local[index + 1], blend)
root = (1 - blend[:, 0]) * root[index] + blend[:, 0] * root[index + 1]

# Forward kinematics of the skeleton tree.
rotations, positions = np.zeros_like(local), np.zeros((*local.shape[:2], 3))
for b, parent in enumerate(parents):
    if parent < 0:
        rotations[:, b], positions[:, b] = local[:, b], root
    else:
        rotations[:, b] = mul(rotations[:, parent], local[:, b])
        positions[:, b] = positions[:, parent] + rotate(rotations[:, parent], offsets[b])

# Joint coordinates of the clip format: rotation vectors of ball joints, the angle about y of hinges.
dof_names, dof_positions = [], []
for body in names[1:]:
    if body not in JOINTS:
        continue
    joint, ball = JOINTS[body]
    vector = rotvec(local[:, names.index(body)])
    if ball:
        dof_names += [f"{joint}_{axis}" for axis in "xyz"]
        dof_positions += [vector[:, 0], vector[:, 1], vector[:, 2]]
    else:
        dof_names.append(joint)
        dof_positions.append(vector[:, 1])
dof_positions = np.stack(dof_positions, -1)

linear = np.gradient(positions, 1.0 / FPS, axis=0)
delta = mul(np.roll(rotations, -1, axis=0), conj(rotations))
angular = rotvec(delta) * FPS
angular[-1] = angular[-2]
np.savez(
    sys.argv[2],
    fps=np.int64(FPS),
    dof_names=np.array(dof_names),
    body_names=np.array(names),
    dof_positions=dof_positions.astype(np.float32),
    dof_velocities=np.gradient(dof_positions, 1.0 / FPS, axis=0).astype(np.float32),
    body_positions=positions.astype(np.float32),
    body_rotations=rotations[..., [3, 0, 1, 2]].astype(np.float32),
    body_linear_velocities=linear.astype(np.float32),
    body_angular_velocities=angular.astype(np.float32),
)
print(
    sys.argv[2],
    "frames",
    len(times),
    "duration",
    round(duration, 2),
    "s, source fps",
    fps,
    "min height",
    positions[..., 2].min().round(3),
)
