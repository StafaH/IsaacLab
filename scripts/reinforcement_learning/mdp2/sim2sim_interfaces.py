# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an export of ``export_interfaces.py`` in MuJoCo: the policy on another simulator, before a robot.

    uv run python scripts/reinforcement_learning/mdp2/sim2sim_interfaces.py --export export/dance1 --model scene.xml

``--model`` is a MuJoCo scene with the robot on a floor, with one actuator per joint that the policy moves. The
script gives the actuators the gains of the export, starts the robot on ``--starts`` frames spread over the clip,
and reports the share of the episodes that track for their whole length, as the training task counts it: until the
mean distance of the bodies to the clip, moved horizontally onto the robot, passes 0.25 m.

The observations are computed here from what a robot measures: its joints, the orientation and the angular velocity
of its root. Before the episodes, the script computes them for the states of the exported episode and prints how
far they are from what the policy saw there, which must be zero up to rounding. Joints of the export that the model
does not have, such as fingers, read as their default pose. ``--armature trained`` gives the joints the armature of
the export instead of the model's, and ``--timestep`` sets the physics step, which is rounded to divide the
control step.
"""

import argparse
import json
import os

import mujoco
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--export", required=True, help="Folder written by export_interfaces.py")
parser.add_argument("--model", required=True, help="MuJoCo scene with the robot on a floor")
parser.add_argument("--starts", type=int, default=20, help="Episodes, on start frames spread over the clip")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--armature", default="model", choices=["model", "trained"])
parser.add_argument("--timestep", type=float, default=None, help="Physics step [s]; the model's by default")
parser.add_argument("--out", default=None, help="Path of a JSON file with the result")
args = parser.parse_args()

with open(os.path.join(args.export, "spec.json")) as file:
    SPEC = json.load(file)
CLIP = dict(np.load(os.path.join(args.export, "clip.npz")))
FRAMES = len(CLIP["joint_pos"])
INTERFACE = np.array(SPEC["interface"])
DEFAULT = np.array(SPEC["default_joint_pos"])
ACTION_JOINTS = SPEC["action_joints"]


def to_root(quat, vectors):
    """Return world ``vectors`` in the axes of a root with the orientation ``quat`` (x, y, z, w)."""
    cross = np.cross(quat[:3], vectors)
    return vectors - 2.0 * quat[3] * cross + 2.0 * np.cross(quat[:3], cross)


def observe(frame, quat, ang_vel, joint_pos, joint_vel, last_action):
    """Return the observations of the policy on a frame of the clip.

    ``quat`` (x, y, z, w) is the orientation of the root, ``ang_vel`` its angular velocity in its own axes, and
    ``joint_pos`` and ``joint_vel`` are in the order of the export's joints.
    """
    measured = {
        "base_ang_vel": ang_vel,
        "projected_gravity": to_root(quat, np.array([0.0, 0.0, -1.0])),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "actions": last_action,
        "interface": INTERFACE,
    }
    values = []
    for term in SPEC["observations"]:
        if term["name"] in measured:
            value = measured[term["name"]]
        elif "bodies" in term:
            # Where the bodies are some steps ahead, from the root of the clip now, in the axes of the robot's root.
            ahead = min(frame + term["ahead"], FRAMES - 1)
            offsets = CLIP["body_pos"][ahead, term["bodies"]] - CLIP["root_pose"][frame, :3]
            value = INTERFACE[term["group"]] * to_root(quat, offsets).ravel()
        else:
            ahead = min(frame + term["ahead"], FRAMES - 1)
            value = INTERFACE[term["group"]] * CLIP["joint_pos"][ahead, term["joints"]]
        values.append(term["scale"] * np.asarray(value))
    return np.concatenate(values).astype(np.float32)


# The observations of the exported episode, computed here from its states, against what the policy saw.
rollout = np.load(os.path.join(args.export, "rollout.npz"))
difference, last_action = 0.0, np.zeros(len(ACTION_JOINTS))
for k in range(len(rollout["frame"])):
    quat = rollout["root_link_pose_w"][k, 3:]
    ang_vel = to_root(quat, rollout["root_com_vel_w"][k, 3:])
    frame = int(rollout["frame"][k] - rollout["first"])
    obs = observe(frame, quat, ang_vel, rollout["joint_pos"][k], rollout["joint_vel"][k], last_action)
    difference = max(difference, np.abs(obs - rollout["obs"][k]).max())
    last_action = rollout["action"][k]
print(f"CHECK observations of {len(rollout['frame'])} exported steps differ by at most {difference:.2e}")

policy = torch.jit.load(os.path.join(args.export, "policy.pt"))
difference = np.abs(policy(torch.from_numpy(rollout["obs"])).detach().numpy() - rollout["action"]).max()
print(f"CHECK actions of the exported policy differ by at most {difference:.2e}")

spec = mujoco.MjSpec.from_file(args.model)
try:
    model = spec.compile()
except ValueError:
    # The meshes of the model are not here: they are what it looks like, so the robot is simulated without them.
    for geom in [geom for geom in spec.geoms if geom.type == mujoco.mjtGeom.mjGEOM_MESH]:
        spec.delete(geom)
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    model = spec.compile()
data = mujoco.MjData(model)

# The joints and bodies of the export in the model, and the actuator of every joint that the policy moves.
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in SPEC["joints"]]
has = np.array([joint >= 0 for joint in joint_ids])
qpos_ids = model.jnt_qposadr[[joint for joint in joint_ids if joint >= 0]]
dof_ids = model.jnt_dofadr[[joint for joint in joint_ids if joint >= 0]]
body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in SPEC["bodies"]]
tracked = np.array([k for k, body in enumerate(body_ids) if body >= 0])
tracked_ids = np.array([body_ids[k] for k in tracked])
actuators = []
for joint in ACTION_JOINTS:
    owners = np.flatnonzero(model.actuator_trnid[:, 0] == joint_ids[joint])
    if joint_ids[joint] < 0 or len(owners) != 1:
        raise ValueError(f"The model needs one actuator for the joint {SPEC['joints'][joint]}")
    actuator = owners[0]
    actuators.append(actuator)
    # A position actuator with the gains of the export, without a range for its target.
    model.actuator_gaintype[actuator] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[actuator] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_gainprm[actuator, :3] = (SPEC["stiffness"][joint], 0.0, 0.0)
    model.actuator_biasprm[actuator, :3] = (0.0, -SPEC["stiffness"][joint], -SPEC["damping"][joint])
    model.actuator_ctrllimited[actuator] = 0
if args.armature == "trained":
    model.dof_armature[dof_ids] = np.array(SPEC["armature"])[has]
substeps = max(1, round(SPEC["step_dt"] / (args.timestep or model.opt.timestep)))
model.opt.timestep = SPEC["step_dt"] / substeps
print(
    f"MODEL joints {int(has.sum())}/{len(has)} bodies {len(tracked)}/{len(body_ids)} step {model.opt.timestep:.5f} s"
    f" armature {model.dof_armature[dof_ids].min():.4f}..{model.dof_armature[dof_ids].max():.4f}"
)

episode_frames = round(args.seconds / SPEC["step_dt"])
starts = np.linspace(0, max(0, FRAMES - episode_frames - 1), args.starts).astype(int)
steps, errors, drifts, completed = [], [], [], []
for start in starts:
    # The robot starts on the frame of the clip, as it does in training.
    quat = CLIP["root_pose"][start, 3:]
    mujoco.mj_resetData(model, data)
    data.qpos[:3], data.qpos[3:7] = CLIP["root_pose"][start, :3], quat[[3, 0, 1, 2]]
    data.qvel[:3], data.qvel[3:6] = CLIP["root_vel"][start, :3], to_root(quat, CLIP["root_vel"][start, 3:])
    data.qpos[qpos_ids], data.qvel[dof_ids] = CLIP["joint_pos"][start, has], CLIP["joint_vel"][start, has]
    mujoco.mj_forward(model, data)
    last_action = np.zeros(len(ACTION_JOINTS), dtype=np.float32)
    joint_pos, joint_vel = DEFAULT.copy(), np.zeros(len(DEFAULT))
    target = min(episode_frames, FRAMES - 2 - start)
    step, error_sum, drift = 0, 0.0, 0.0
    while step < target:
        frame = min(start + step, FRAMES - 1)
        joint_pos[has], joint_vel[has] = data.qpos[qpos_ids], data.qvel[dof_ids]
        obs = observe(frame, data.qpos[[4, 5, 6, 3]], data.qvel[3:6], joint_pos, joint_vel, last_action)
        with torch.no_grad():
            last_action = policy(torch.from_numpy(obs)[None])[0].numpy()
        data.ctrl[actuators] = DEFAULT[ACTION_JOINTS] + SPEC["action_scale"] * last_action
        mujoco.mj_step(model, data, substeps)
        step += 1
        # The distance of the bodies to the clip, moved horizontally onto the robot.
        frame = min(start + step, FRAMES - 1)
        offset = np.array([*(data.qpos[:2] - CLIP["root_pose"][frame, :2]), 0.0])
        error = np.square(data.xpos[tracked_ids] - offset - CLIP["body_pos"][frame, tracked]).sum(-1).mean()
        if not np.isfinite(error) or error > 0.0625:
            break
        error_sum, drift = error_sum + np.sqrt(error), np.linalg.norm(offset)
    else:
        completed.append(start)
    steps.append(step / target)
    errors.append(error_sum / max(step, 1))
    drifts.append(drift)
    print(f"EPISODE frame {start} tracked {step}/{target} steps, error {1e3 * errors[-1]:.0f} mm, drift {drift:.2f} m")

result = {
    "success": len(completed) / len(starts),
    "tracked": float(np.mean(steps)),
    "error_mm": 1e3 * float(np.mean(errors)),
    "drift_m": float(np.mean(drifts)),
    "timestep": model.opt.timestep,
    "armature": args.armature,
}
print(f"RESULT success {result['success']:.2f} tracked {result['tracked']:.2f} error {result['error_mm']:.0f} mm")
if args.out:
    with open(args.out, "w") as file:
        json.dump(result, file)
