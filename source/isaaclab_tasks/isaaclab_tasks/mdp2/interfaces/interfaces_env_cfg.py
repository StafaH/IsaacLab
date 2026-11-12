# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-interface motion tracking: the tracking task, told to the policy through one of several interfaces.

An interface is a set of target groups: the positions of some bodies, or of some joints, in the frames ahead. Every
world tracks the whole clip and is rewarded for all of it, and its policy sees the groups of its interface. The
``teacher`` observations are the policy's with every group, and the ``critic`` observations are privileged.
"""

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from isaaclab.utils import replace

from isaaclab_tasks.mdp2.tracking import mdp as tracking
from isaaclab_tasks.mdp2.tracking.tracking_env_cfg import clip_body_pos, clip_joint_pos, tracking_task

from . import mdp as interfaces_mdp

robot = mdp2.Entity("robot")

INTERFACE = "commands.interface"
"""The command whose state a learner sets: ``shares`` of the interfaces, and the ``fixed`` interface of a world."""


def interface_task(
    motion_files: list[str],
    lengths: list[int],
    step_dt: float,
    num_bodies: int,
    body_groups: dict[str, mdp2.SubsetSelector],
    joint_groups: dict[str, mdp2.SubsetSelector],
    interfaces: dict[str, tuple[str, ...]],
    train_clips: int,
    horizons: tuple[int, ...] = (1, 5, 15),
    adaptive: float = 0.5,
    episode_length_s: float = 10.0,
) -> mdp2.TaskCfg:
    """Return the MDP in which the ``robot`` tracks the clips, told through one of ``interfaces`` per episode.

    ``body_groups`` and ``joint_groups`` name the groups of targets, and an interface is the names of its groups.
    The first ``train_clips`` clips are for training; evaluation worlds start on frames that the learner fixes.
    Targets are given ``horizons`` steps ahead, and episodes last ``episode_length_s`` at most.
    """
    task = tracking_task(motion_files, step_dt, num_bodies, lengths=lengths)
    groups = [*body_groups, *joint_groups]
    patterns = tuple(tuple(float(group in members) for group in groups) for members in interfaces.values())
    frames = sum(lengths)

    reset = task.events["reset_to_motion"]
    reset = replace(
        reset,
        term=tracking.reset_to_clips,
        inputs={
            **reset.inputs,
            "body_link_pose_w": robot.body_link_pose_w,
            "terminated": mdp2.terminated,
            "episode_steps": mdp2.quantities.episode_steps,
        },
        params={
            "motion_files": tuple(motion_files),
            "windows": tuple((0, length) for length in lengths),
            "frames": frames,
            "clips": len(motion_files),
            "bodies": num_bodies,
            "noise": 0.02,
            "ground_clearance": 0.03,
            "adaptive": adaptive,
            "bin_frames": 15,
            "bins": -(-frames // 15),
            "decay": 0.999,
            "train_clips": train_clips,
        },
    )

    def targets(command) -> dict[str, mdp2.Observation]:
        """The targets of every group and horizon, shown where ``command`` has the group."""
        terms = {}
        for ahead in horizons:
            for index, (name, bodies) in enumerate(body_groups.items()):
                terms[f"{name}_{ahead}"] = mdp2.Observation(
                    interfaces_mdp.commanded_body_pos_b,
                    inputs={
                        "root_link_pose_w": robot.root_link_pose_w,
                        **clip_body_pos,
                        "body_ids": bodies,
                        "command": command,
                    },
                    params={"group": index, "ahead": ahead},
                )
            for index, (name, joints) in enumerate(joint_groups.items(), start=len(body_groups)):
                terms[f"{name}_{ahead}"] = mdp2.Observation(
                    interfaces_mdp.commanded_joint_pos,
                    inputs={**clip_joint_pos, "joint_ids": joints, "command": command},
                    params={"group": index, "ahead": ahead},
                )
        terms["interface"] = mdp2.Observation(mdp.generated_commands, inputs={"command": command})
        return terms

    # What the robot measures: no base velocity and no height.
    policy = task.observations["policy"].terms
    measured = {
        name: policy[name] for name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")
    }
    interface = mdp2.Observation(mdp.generated_commands, inputs={"command": mdp2.commands.interface})
    return replace(
        task,
        # Targets for the joints are offsets from the default pose: the policy is not given the clip's pose.
        actions={"joint_pos": mdp2.DirectAction(robot.joint_pos_target, scale=0.25, offset=robot.default_joint_pos)},
        commands={
            "interface": mdp2.Command(
                interfaces_mdp.sampled_interface,
                params={"interfaces": patterns, "count": len(patterns), "groups": len(groups)},
            ),
            "every_group": mdp2.Command(interfaces_mdp.ones_command, params={"groups": len(groups)}),
        },
        observations={
            "policy": mdp2.ObservationGroup(**measured, **targets(mdp2.commands.interface)),
            "teacher": mdp2.ObservationGroup(**measured, **targets(mdp2.commands.every_group)),
            "critic": mdp2.ObservationGroup(**policy, interface=interface),
        },
        terminations={
            **task.terminations,
            "time_out": mdp2.TimeLimit(episode_length_s=episode_length_s),
            "diverged": mdp2.Termination(tracking.is_not_finite, inputs={"value": mdp2.quantities.body_pos_error}),
        },
        events={"reset_to_motion": reset},
    )
