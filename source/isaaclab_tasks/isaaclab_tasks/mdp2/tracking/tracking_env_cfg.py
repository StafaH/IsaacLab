# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion tracking: a robot follows a motion clip, frame by frame."""

import numpy as np
from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from isaaclab.utils import replace

from . import mdp as tracking

robot = mdp2.Entity("robot")
scene = mdp2.Entity("scene")
joints = robot.joints(".*")

root_state = {"root_link_pose_w": robot.root_link_pose_w, "root_com_vel_w": robot.root_com_vel_w}
RESET = "events.reset_to_motion"
progress = {
    "start": mdp2.term_state(RESET, "start"),
    "end": mdp2.term_state(RESET, "end"),
    "episode_length": mdp2.episode_length,
}
clip_joint_pos = {"clip_joint_pos": mdp2.term_state(RESET, "clip_joint_pos"), **progress}
clip_body_pos = {
    "clip_body_pos": mdp2.term_state(RESET, "clip_body_pos"),
    "env_origins": scene.env_origins,
    **progress,
}


def tracking_task(
    motion_files: list[str],
    step_dt: float,
    num_bodies: int,
    start_range: tuple[float, float] = (0.0, 0.9),
    adaptive: float = 0.0,
) -> mdp2.TaskCfg:
    """Return the MDP in which the ``robot`` tracks the clips in ``motion_files``, frame by frame.

    Every world tracks one clip per episode. The clips are sampled every ``step_dt`` seconds, and episodes start in
    the fraction ``start_range`` of a clip. A fraction ``adaptive`` of them start where episodes fail instead.
    """
    lengths = []
    for motion_file in motion_files:
        clip = np.load(motion_file)
        lengths.append(len(clip["dof_positions"][:: round(float(clip["fps"]) * step_dt)]))
    frames = sum(lengths)
    reset = mdp2.Event(
        tracking.reset_to_motion,
        inputs={"env_origins": scene.env_origins, "soft_joint_pos_limits": robot.soft_joint_pos_limits},
        outputs={"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel, **root_state},
        params={
            "motion_files": tuple(motion_files),
            "frames": frames,
            "clips": len(motion_files),
            "bodies": num_bodies,
            "start_range": start_range,
            "noise": 0.02,
        },
    )
    if adaptive > 0.0:
        reset = replace(
            reset,
            term=tracking.reset_to_failures,
            inputs={**reset.inputs, "terminated": mdp2.terminated, "episode_steps": mdp2.quantities.episode_steps},
            params={**reset.params, "adaptive": adaptive, "bin_frames": 15, "bins": -(-frames // 15), "decay": 0.999},
        )

    return mdp2.TaskCfg(
        # The policy corrects the pose of the frame it tracks next.
        actions={
            "joint_pos": mdp2.DirectAction(
                robot.joint_pos_target, scale=0.25, offset=mdp2.quantities.reference_joint_pos
            ),
        },
        quantities={
            "reference_joint_pos": mdp2.Quantity(
                tracking.reference_joint_pos, inputs={**clip_joint_pos, "joint_ids": joints}
            ),
            "episode_steps": mdp2.Quantity(tracking.episode_steps, inputs={"episode_length": mdp2.episode_length}),
            "joint_pos_error": mdp2.Quantity(
                tracking.joint_pos_error, inputs={"joint_pos": robot.joint_pos, **clip_joint_pos}, log=True
            ),
            "joint_vel_error": mdp2.Quantity(
                tracking.joint_vel_error,
                inputs={
                    "joint_vel": robot.joint_vel,
                    "clip_joint_vel": mdp2.term_state(RESET, "clip_joint_vel"),
                    **progress,
                },
            ),
            "body_pos_error": mdp2.Quantity(
                tracking.body_pos_error, inputs={"body_link_pose_w": robot.body_link_pose_w, **clip_body_pos}, log=True
            ),
            "root_vel_error": mdp2.Quantity(
                tracking.root_vel_error,
                inputs={
                    "root_com_vel_w": robot.root_com_vel_w,
                    "clip_root_vel": mdp2.term_state(RESET, "clip_root_vel"),
                    **progress,
                },
                params={"angular_scale": 0.1},
            ),
        },
        observations={
            "policy": mdp2.ObservationGroup(
                root_height=mdp2.Observation(tracking.root_height, inputs={"root_link_pose_w": robot.root_link_pose_w}),
                base_lin_vel=mdp2.Observation(mdp.base_lin_vel, inputs=root_state),
                base_ang_vel=mdp2.Observation(mdp.base_ang_vel, inputs=root_state),
                projected_gravity=mdp2.Observation(
                    mdp.projected_gravity,
                    inputs={"root_link_pose_w": robot.root_link_pose_w, "gravity_w": scene.gravity_w},
                ),
                joint_pos=mdp2.Observation(mdp.joint_pos, inputs={"joint_pos": robot.joint_pos, "joint_ids": joints}),
                joint_vel=mdp2.Observation(
                    mdp.joint_vel, inputs={"joint_vel": robot.joint_vel, "joint_ids": joints}, scale=0.1
                ),
                body_pos=mdp2.Observation(
                    tracking.body_pos_b,
                    inputs={
                        "body_link_pose_w": robot.body_link_pose_w,
                        "root_link_pose_w": robot.root_link_pose_w,
                    },
                ),
                reference_joint_pos=mdp2.Observation(
                    mdp.generated_commands, inputs={"command": mdp2.quantities.reference_joint_pos}
                ),
                reference_body_pos=mdp2.Observation(
                    tracking.reference_body_pos_b,
                    inputs={"root_link_pose_w": robot.root_link_pose_w, **clip_body_pos},
                    params={"bodies": num_bodies},
                ),
                actions=mdp2.Observation(mdp.last_action, inputs={"actions": mdp2.last_action}),
            )
        },
        rewards={
            "joint_pos": mdp2.Reward(
                tracking.exp_kernel,
                inputs={"value": mdp2.quantities.joint_pos_error},
                params={"scale": 25.0},
                weight=15.0,
            ),
            "joint_vel": mdp2.Reward(
                tracking.exp_kernel,
                inputs={"value": mdp2.quantities.joint_vel_error},
                params={"scale": 0.05},
                weight=3.0,
            ),
            "body_pos": mdp2.Reward(
                tracking.exp_kernel,
                inputs={"value": mdp2.quantities.body_pos_error},
                params={"scale": 100.0},
                weight=9.0,
            ),
            "root_vel": mdp2.Reward(
                tracking.exp_kernel,
                inputs={"value": mdp2.quantities.root_vel_error},
                params={"scale": 1.0},
                weight=3.0,
            ),
            "action_rate_l2": mdp2.Reward(mdp.action_rate_l2, inputs={"actions": mdp2.last_action}, weight=-0.03),
        },
        terminations={
            "time_out": mdp2.TimeLimit(episode_length_s=max(lengths) * step_dt),
            "motion_ended": mdp2.Termination(tracking.motion_ended, inputs=progress, time_out=True),
            # A mean distance of 0.25 m to the tracked bodies.
            "tracking_lost": mdp2.Termination(
                tracking.is_above, inputs={"value": mdp2.quantities.body_pos_error}, params={"threshold": 0.0625}
            ),
        },
        events={
            "reset_to_motion": reset,
        },
    )
