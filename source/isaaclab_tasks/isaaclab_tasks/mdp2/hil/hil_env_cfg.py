# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hybrid imitation learning: one policy for worlds that track a clip and worlds that follow goals in its style.

Both kinds of worlds give the policy the same observations, a goal instead of the pose to track, so what it learns
from tracking carries over to the goals (HIL, arXiv 2505.12619). They differ in their goals, rewards, and
terminations, and share a style reward from a discriminator that the learner trains on the clip.
"""

import numpy as np
from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from isaaclab.utils import replace

from isaaclab_tasks.mdp2.tracking import mdp as tracking

from . import mdp as hil

robot = mdp2.Entity("robot")
scene = mdp2.Entity("scene")
joints = robot.joints(".*")

root_state = {"root_link_pose_w": robot.root_link_pose_w, "root_com_vel_w": robot.root_com_vel_w}
joint_state = {"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel}
RESET = "events.reset_to_motion"
progress = {"start": mdp2.term_state(RESET, "start"), "episode_length": mdp2.episode_length}


def hil_tasks(
    motion_file: str,
    step_dt: float,
    num_bodies: int,
    discriminator: hil.Discriminator,
    episode_length_s: float = 10.0,
    style_weight: float = 15.0,
) -> list[mdp2.TaskCfg]:
    """Return the MDP of the worlds that track the clip in motion_file, and of the worlds that follow goals."""
    clip = np.load(motion_file)
    frames = len(clip["dof_positions"][:: round(float(clip["fps"]) * step_dt)])
    num_features = discriminator.mean.numel()

    following = mdp2.TaskCfg(
        actions={
            "joint_pos": mdp2.DirectAction(robot.joint_pos_target, scale=0.5, offset=robot.default_joint_pos),
        },
        commands={
            "goal": mdp2.Command(
                hil.heading_goal,
                inputs={"root_link_pose_w": robot.root_link_pose_w},
                resampling_time_range=(2.0, 5.0),
                params={"max_turn": 1.0, "speed": (0.8, 1.6)},
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
                goal=mdp2.Observation(
                    hil.goal_b,
                    inputs={"command": mdp2.commands.goal, "root_link_pose_w": robot.root_link_pose_w},
                ),
                actions=mdp2.Observation(mdp.last_action, inputs={"actions": mdp2.last_action}),
            )
        },
        rewards={
            "goal": mdp2.Reward(
                hil.heading_velocity,
                inputs={"command": mdp2.commands.goal, **root_state},
                params={"scale": 0.25, "facing_weight": 0.3},
                weight=15.0,
            ),
            "style": mdp2.Reward(
                hil.style,
                inputs={**joint_state, **root_state},
                params={"discriminator": discriminator, "features": num_features},
                weight=style_weight,
            ),
        },
        terminations={
            "time_out": mdp2.TimeLimit(episode_length_s=episode_length_s),
            "fallen": mdp2.Termination(
                hil.root_height_outside,
                inputs={"root_link_pose_w": robot.root_link_pose_w},
                params={"bounds": (0.5, 1.5)},
            ),
        },
        events={
            "reset_to_motion": mdp2.Event(
                tracking.reset_to_motion,
                inputs={"env_origins": scene.env_origins, "soft_joint_pos_limits": robot.soft_joint_pos_limits},
                outputs={**joint_state, **root_state},
                params={
                    "motion_file": motion_file,
                    "frames": frames,
                    "bodies": num_bodies,
                    "start_range": (0.0, 0.9),
                    "noise": 0.05,
                },
            ),
        },
    )

    clip_state = {name: mdp2.term_state(RESET, name) for name in ("clip_joint_pos", "clip_joint_vel")}
    errors = {
        "joint_pos_error": mdp2.Quantity(
            tracking.joint_pos_error,
            inputs={"joint_pos": robot.joint_pos, "clip_joint_pos": clip_state["clip_joint_pos"], **progress},
        ),
        "joint_vel_error": mdp2.Quantity(
            tracking.joint_vel_error,
            inputs={"joint_vel": robot.joint_vel, "clip_joint_vel": clip_state["clip_joint_vel"], **progress},
        ),
        "body_pos_error": mdp2.Quantity(
            tracking.body_pos_error,
            inputs={
                "body_link_pose_w": robot.body_link_pose_w,
                "clip_body_pos": mdp2.term_state(RESET, "clip_body_pos"),
                "env_origins": scene.env_origins,
                **progress,
            },
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
    }
    kernels = {"joint_pos": (25.0, 15.0), "joint_vel": (0.05, 3.0), "body_pos": (100.0, 9.0), "root_vel": (1.0, 3.0)}
    tracking_rewards = {
        name: mdp2.Reward(
            tracking.exp_kernel,
            inputs={"value": getattr(mdp2.quantities, f"{name}_error")},
            params={"scale": scale},
            weight=weight,
        )
        for name, (scale, weight) in kernels.items()
    }
    tracking_mode = replace(
        following,
        commands={
            "goal": mdp2.Command(
                hil.reference_goal,
                inputs={"clip_root_vel": mdp2.term_state(RESET, "clip_root_vel"), **progress},
                params={"lookahead": 15},
            ),
        },
        quantities=errors,
        rewards={**tracking_rewards, "style": following.rewards["style"]},
        terminations={
            "time_out": following.terminations["time_out"],
            "motion_ended": mdp2.Termination(
                tracking.motion_ended,
                inputs={"clip_joint_pos": clip_state["clip_joint_pos"], **progress},
                time_out=True,
            ),
            "tracking_lost": mdp2.Termination(
                tracking.is_above, inputs={"value": mdp2.quantities.body_pos_error}, params={"threshold": 0.0625}
            ),
        },
    )
    return [tracking_mode, following]
