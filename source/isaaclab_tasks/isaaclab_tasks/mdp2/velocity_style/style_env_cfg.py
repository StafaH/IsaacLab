# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity tracking in the style of motion clips: a second task for some of the worlds of a velocity task.

The worlds of the style task follow the same velocity commands with the same observations, and start on frames of
the clips. In place of the rewards that shape the gait by hand, they get a style reward from a discriminator, which
the learner trains to tell the motion of the robot from that of the clips (adversarial motion priors, as in hybrid
imitation learning, arXiv 2505.12619). One policy learns from both tasks, and moves like the clips in both.
"""

from typing import Any

from isaaclab_experimental import mdp2

from isaaclab.utils import replace

from isaaclab_tasks.mdp2.hil import mdp as hil
from isaaclab_tasks.mdp2.tracking import mdp as tracking

robot = mdp2.Entity("robot")
scene = mdp2.Entity("scene")

KEPT_REWARDS = (
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "termination_penalty",
    "action_rate_l2",
    "dof_torques_l2",
    "dof_acc_l2",
)
"""Rewards of a velocity task that its style task keeps: following the command, and moving smoothly."""


def style_task(
    task: mdp2.TaskCfg, clips: dict[str, Any], discriminator: hil.Discriminator, style_weight: float
) -> mdp2.TaskCfg:
    """Return the style task of the velocity task ``task``, for the worlds that move in the style of the clips.

    ``task`` is a velocity task of the entities ``robot`` and ``scene``. ``clips`` are the params of
    :data:`~isaaclab_tasks.mdp2.tracking.mdp.reset_to_csv_motion` that name the clips. ``discriminator`` scores
    pairs of consecutive :func:`~isaaclab_tasks.mdp2.hil.mdp.motion_features`, and the style reward of its score
    weighs ``style_weight``.
    """
    joint_state = {"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel}
    root_state = {"root_link_pose_w": robot.root_link_pose_w, "root_com_vel_w": robot.root_com_vel_w}
    style = mdp2.Reward(
        hil.style,
        inputs={**joint_state, **root_state},
        params={"discriminator": discriminator, "features": discriminator.mean.numel()},
        weight=style_weight,
    )
    # The worlds start on a frame of a clip, in the middle of a step, where the motion of the clips is at hand.
    reset = mdp2.Event(
        tracking.reset_to_csv_motion,
        inputs={
            "env_origins": scene.env_origins,
            "soft_joint_pos_limits": robot.soft_joint_pos_limits,
            "body_link_pose_w": robot.body_link_pose_w,
        },
        outputs={**joint_state, **root_state},
        params={"start_range": (0.0, 0.95), "noise": 0.02, **clips},
    )
    events = {name: event for name, event in task.events.items() if event.mode is not mdp2.EventMode.RESET}
    rewards = {name: reward for name, reward in task.rewards.items() if name in KEPT_REWARDS}
    return replace(task, rewards={**rewards, "style": style}, events={**events, "reset_to_motion": reset})
