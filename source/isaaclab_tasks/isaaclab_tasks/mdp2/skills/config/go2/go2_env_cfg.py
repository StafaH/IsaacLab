# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-skill Unitree Go2: walking, standing and walking on the hind legs, and getting up, with one policy.

A handstand is defined too, but it is not among the skills of the environment: PPO did not find it from these
rewards in 600 million samples.
"""

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from isaaclab.utils import configclass, replace

from isaaclab_tasks.mdp2.skills import mdp as skills_mdp
from isaaclab_tasks.mdp2.skills.skills_env_cfg import skill_tasks
from isaaclab_tasks.mdp2.velocity_go2.flat_env_cfg import STAND_TASK, TASK, UnitreeGo2FlatEnvCfg

robot = mdp2.Entity("robot")
contact_forces = mdp2.Entity("contact_forces")
joints = robot.joints(".*")
root_pose = {"root_link_pose_w": robot.root_link_pose_w}
contacts = {"net_normal_forces_w_history": contact_forces.net_normal_forces_w_history}

# Every skill but walking stands in place, and is regularized alike.
IN_PLACE = replace(STAND_TASK, rewards={name: TASK.rewards[name] for name in ("dof_torques_l2", "dof_acc_l2")})
REGULARIZATION = {**IN_PLACE.rewards, "action_rate_l2": replace(TASK.rewards["action_rate_l2"], weight=-0.01)}


def balancing(direction: tuple[float, float, float], height: float, feet_up: str) -> mdp2.TaskCfg:
    """Return the task of balancing with the base's forward axis along ``direction`` and the ``feet_up`` feet off
    the ground, at ``height`` [m]."""
    return replace(
        IN_PLACE,
        rewards={
            "posture": mdp2.Reward(
                skills_mdp.axis_alignment,
                inputs=root_pose,
                params={"axis": (1.0, 0.0, 0.0), "direction": direction},
                weight=4.0,
            ),
            "height": mdp2.Reward(skills_mdp.height_reward, inputs=root_pose, params={"target": height}, weight=2.0),
            "feet_up": mdp2.Reward(
                skills_mdp.bodies_height_reward,
                inputs={"body_link_pose_w": robot.body_link_pose_w, "body_ids": robot.bodies(feet_up)},
                params={"target": height},
                weight=3.0,
            ),
            # Touching the ground with the base costs, but does not end the episode: the way up passes close to it.
            "base_contact": mdp2.Reward(
                mdp.undesired_contacts,
                inputs={**contacts, "body_ids": contact_forces.bodies("base")},
                params={"threshold": 1.0},
                weight=-2.0,
            ),
            **REGULARIZATION,
        },
        terminations={"time_out": TASK.terminations["time_out"]},
    )


GET_UP = replace(
    IN_PLACE,
    rewards={
        "posture": mdp2.Reward(
            skills_mdp.axis_alignment,
            inputs=root_pose,
            params={"axis": (0.0, 0.0, 1.0), "direction": (0.0, 0.0, 1.0)},
            weight=4.0,
        ),
        "height": mdp2.Reward(skills_mdp.height_reward, inputs=root_pose, params={"target": 0.34}, weight=2.0),
        "joint_deviation_l1": mdp2.Reward(
            mdp.joint_deviation_l1,
            inputs={"joint_pos": robot.joint_pos, "default_joint_pos": robot.default_joint_pos, "joint_ids": joints},
            weight=-0.3,
        ),
        **REGULARIZATION,
    },
    # A fallen robot lies on its base.
    terminations={"time_out": TASK.terminations["time_out"]},
    events={
        **TASK.events,
        "reset_base": replace(
            TASK.events["reset_base"],
            params={
                "pose_range": ((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.2), (-1.6, 1.6), (-0.5, 0.5), (-3.14, 3.14)),
                "velocity_range": ((-0.5, 0.5),) * 6,
            },
        ),
        "push_robot": None,
    },
)
"""Getting up: the robot is dropped on either side and rewarded for standing in its default pose."""

HIND_LEGS = balancing((0.0, 0.0, 1.0), 0.55, "F[LR]_foot")
"""Standing on the hind legs."""

HIND_WALK = replace(
    HIND_LEGS,
    rewards={
        **HIND_LEGS.rewards,
        # Upright, the belly faces where the base's top axis points away from.
        "walk": mdp2.Reward(
            skills_mdp.axis_velocity_exp,
            inputs={"root_link_pose_w": robot.root_link_pose_w, "root_com_vel_w": robot.root_com_vel_w},
            params={"axis": (0.0, 0.0, -1.0), "speed": 0.4, "std": 0.25},
            weight=3.0,
        ),
    },
)
"""Walking on the hind legs, belly first, at 0.4 m/s."""

SKILLS = {
    "walk": TASK,
    "hind_legs": HIND_LEGS,
    "hind_walk": HIND_WALK,
    "get_up": GET_UP,
    "handstand": balancing((0.0, 0.0, -1.0), 0.5, "R[LR]_foot"),
}
"""The task of every skill."""


@configclass
class UnitreeGo2SkillsEnvCfg(UnitreeGo2FlatEnvCfg):
    """The Go2 flat environment with its worlds split between the skills."""

    task: list[mdp2.TaskCfg] = skill_tasks(
        {name: SKILLS[name] for name in ("walk", "hind_legs", "hind_walk", "get_up")}
    )
