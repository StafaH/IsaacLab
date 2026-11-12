# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-task Kuka Allegro: lifting and placing an object, with one policy.

The worlds are split between the tasks. Lifting carries the object to a position above the table, and placing to a
position just above the table top. Reorienting, to a position and an orientation, is defined too, but it is not
among the tasks of the environment: PPO did not learn it next to the others in 4000 iterations.
"""

from isaaclab_experimental import mdp2

from isaaclab.utils import configclass

from isaaclab_tasks.mdp2.lift_kuka_allegro.lift_env_cfg import GOAL_RANGE, TASK, KukaAllegroLiftEnvCfg
from isaaclab_tasks.mdp2.manipulation.manipulation_env_cfg import placing, reorienting
from isaaclab_tasks.mdp2.skills.skills_env_cfg import skill_tasks

robot = mdp2.Entity("robot")
target = mdp2.Entity("object")

PLACE_RANGE = (*GOAL_RANGE[:2], (0.34, 0.42))
"""Ranges of the commanded object position when placing: over the table, just above its top [m]."""

SKILLS = {
    "lift": TASK,
    "place": placing(TASK, PLACE_RANGE),
    "reorient": reorienting(TASK, target.root_link_pose_w, robot.root_link_pose_w),
}
"""The task of every skill."""


@configclass
class KukaAllegroManipulationEnvCfg(KukaAllegroLiftEnvCfg):
    """The Kuka Allegro lift environment with its worlds split between the skills."""

    task: list[mdp2.TaskCfg] = skill_tasks({name: SKILLS[name] for name in ("lift", "place")})
