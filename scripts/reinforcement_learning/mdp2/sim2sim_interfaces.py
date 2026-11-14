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
far they are from what the policy saw there, which must be zero up to rounding. Joints that the policy does not
move, such as fingers, read as their default pose. ``--armature trained`` gives the joints the armature of
the export instead of the model's, ``--timestep`` sets the physics step, which is rounded to divide the control
step, and ``--effort_limits`` holds the torque of every joint within the limit of the export. The result has the
largest torque of the episodes as a share of that limit. ``--yaw`` turns the robot about the vertical axis at the
start, as a robot that does not face where the clip does: the policy turns the clip onto it.
"""

import argparse
import json
import os

import mujoco
import numpy as np
from interface_export import ExportedPolicy, to_root, turn
from mujoco_robot import MujocoRobot

parser = argparse.ArgumentParser()
parser.add_argument("--export", required=True, help="Folder written by export_interfaces.py")
parser.add_argument("--model", required=True, help="MuJoCo scene with the robot on a floor")
parser.add_argument("--starts", type=int, default=20, help="Episodes, on start frames spread over the clip")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--armature", default="model", choices=["model", "trained"])
parser.add_argument("--timestep", type=float, default=None, help="Physics step [s]; the model's by default")
parser.add_argument("--effort_limits", action="store_true", help="Hold the torques within the limits of the export")
parser.add_argument("--yaw", type=float, default=0.0, help="Turn of the robot from the clip about the vertical [rad]")
parser.add_argument("--out", default=None, help="Path of a JSON file with the result")
args = parser.parse_args()

policy = ExportedPolicy(args.export)
SPEC, CLIP, FRAMES = policy.spec, policy.clip, policy.frames
DEFAULT, ACTION_JOINTS = policy.default, policy.action_joints

# The observations of the exported episode, computed here from its states, against what the policy saw.
rollout = np.load(os.path.join(args.export, "rollout.npz"))
observed, answered = 0.0, 0.0
policy.reset(0, CLIP["root_pose"][0, 3:])
for k in range(len(rollout["frame"])):
    quat = rollout["root_link_pose_w"][k, 3:]
    ang_vel = to_root(quat, rollout["root_com_vel_w"][k, 3:])
    policy.frame = int(rollout["frame"][k] - rollout["first"])
    obs = policy.observe(quat, ang_vel, rollout["joint_pos"][k], rollout["joint_vel"][k])
    observed = max(observed, np.abs(obs - rollout["obs"][k]).max())
    answered = max(answered, np.abs(policy.act(rollout["obs"][k]) - rollout["action"][k]).max())
    policy.last_action = rollout["action"][k]
print(f"CHECK observations of {len(rollout['frame'])} exported steps differ by at most {observed:.2e}")
print(f"CHECK actions of the exported policy differ by at most {answered:.2e}")

robot = MujocoRobot(args.model, policy, args.armature == "trained", args.timestep, args.effort_limits)
model, data = robot.model, robot.data
body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in SPEC["bodies"]]
tracked = np.array([k for k, body in enumerate(body_ids) if body >= 0])
tracked_ids = np.array([body_ids[k] for k in tracked])
print(
    f"MODEL joints {len(ACTION_JOINTS)}/{len(DEFAULT)} bodies {len(tracked)}/{len(body_ids)}"
    f" step {model.opt.timestep:.5f} s"
    f" armature {model.dof_armature[robot.dof_ids].min():.4f}..{model.dof_armature[robot.dof_ids].max():.4f}"
)

STIFFNESS, DAMPING = (np.array(SPEC[name])[ACTION_JOINTS] for name in ("stiffness", "damping"))
episode_frames = round(args.seconds / SPEC["step_dt"])
starts = np.linspace(0, max(0, FRAMES - episode_frames - 1), args.starts).astype(int)
steps, errors, drifts, completed = [], [], [], []
limits, torque = np.array(SPEC["effort_limits"])[ACTION_JOINTS], np.zeros(len(ACTION_JOINTS))
for start in starts:
    # The robot starts on the frame of the clip, as it does in training, turned about its root by ``--yaw``.
    robot.reset(start, args.yaw)
    origin = CLIP["root_pose"][start, :3]
    policy.reset(start, robot.read()[0])
    joint_pos, joint_vel = DEFAULT.copy(), np.zeros(len(DEFAULT))
    target = min(episode_frames, FRAMES - 2 - start)
    step, error_sum, drift = 0, 0.0, 0.0
    while step < target:
        # Joints that the policy does not move, such as fingers, read as their default pose.
        quat, ang_vel, joint_pos[ACTION_JOINTS], joint_vel[ACTION_JOINTS] = robot.read()
        robot.write(policy.step(quat, ang_vel, joint_pos, joint_vel), STIFFNESS, DAMPING)
        step += 1
        torque = np.maximum(torque, np.abs(data.actuator_force[robot.actuators]) / limits)
        # The distance of the bodies, turned back to the clip, to the clip moved horizontally onto the robot.
        frame = min(start + step, FRAMES - 1)
        bodies = origin + turn(data.xpos[tracked_ids] - origin, -args.yaw)
        root = origin + turn(data.qpos[:3] - origin, -args.yaw)
        offset = np.array([*(root[:2] - CLIP["root_pose"][frame, :2]), 0.0])
        error = np.square(bodies - offset - CLIP["body_pos"][frame, tracked]).sum(-1).mean()
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
    "effort_limits": args.effort_limits,
    "yaw": args.yaw,
    "torque": {SPEC["joints"][joint]: round(float(share), 2) for joint, share in zip(ACTION_JOINTS, torque)},
}
print(f"RESULT success {result['success']:.2f} tracked {result['tracked']:.2f} error {result['error_mm']:.0f} mm")
joint = int(torque.argmax())
print(f"RESULT largest torque {torque[joint]:.2f} of its limit, on {SPEC['joints'][ACTION_JOINTS[joint]]}")
if args.out:
    with open(args.out, "w") as file:
        json.dump(result, file)
