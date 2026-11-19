# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-task manipulation: variants of a task that carries a grasped object to a commanded pose.

The variants change what the task commands and what counts as success, and keep its observations, so one policy
can learn them together when :func:`~isaaclab_tasks.mdp2.skills.skills_env_cfg.skill_tasks` tells it the task.
"""

from isaaclab_experimental import mdp2

from isaaclab.utils import replace

from . import mdp as manipulation_mdp


def placing(task: mdp2.TaskCfg, goal_range: tuple, command: str = "object_pose") -> mdp2.TaskCfg:
    """Return ``task`` with the positions of its ``command`` drawn from ``goal_range``, three ``(low, high)``."""
    term = task.commands[command]
    pose_range = (*goal_range, *term.params["pose_range"][3:])
    term = replace(term, params={**term.params, "pose_range": pose_range})
    return replace(task, commands={**task.commands, command: term})


def reorienting(
    task: mdp2.TaskCfg, object_pose_w, root_link_pose_w, angle: float = 0.4, weight: float = 4.0
) -> mdp2.TaskCfg:
    """Return ``task`` with the commanded orientation part of the goal: success needs it within ``angle`` [rad].

    The task's ``success`` quantity is replaced, so what reads it, such as a curriculum, follows the new goal.
    """
    success = task.quantities["success"]
    grasp = {name: success.inputs[name] for name in ("thumb_force", "finger_force")}
    error = {"orientation_error": mdp2.quantities.orientation_error}
    return replace(
        task,
        # Quantities are computed in order, and the success reads the orientation error.
        quantities={
            "orientation_error": mdp2.Quantity(
                manipulation_mdp.orientation_error,
                inputs={
                    "command": mdp2.commands.object_pose,
                    "root_link_pose_w": root_link_pose_w,
                    "object_pose_w": object_pose_w,
                },
            ),
            **task.quantities,
            "success": replace(
                success,
                term=manipulation_mdp.pose_succeeded,
                inputs={**success.inputs, **error},
                params={**success.params, "angle": angle},
            ),
        },
        rewards={
            **task.rewards,
            "orientation_tracking": mdp2.Reward(
                manipulation_mdp.orientation_tracking,
                inputs={**error, **grasp},
                params={"std": 1.5, "contact_threshold": 0.1},
                weight=weight,
            ),
        },
    )
