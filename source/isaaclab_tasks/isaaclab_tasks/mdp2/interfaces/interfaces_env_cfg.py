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
    count_failures: bool = True,
    episode_length_s: float = 10.0,
    pushes: dict[str, float] | None = None,
    push_velocity: tuple[float, float, float] = (1.0, 1.0, 0.3),
    push_interval_s: tuple[float, float] = (1.0, 3.0),
    feet: mdp2.SubsetSelector | None = None,
    feet_weight: float = 0.0,
    action_rate_weight: float = 0.03,
) -> mdp2.TaskCfg:
    """Return the MDP in which the ``robot`` tracks the clips, told through one of ``interfaces`` per episode.

    ``body_groups`` and ``joint_groups`` name the groups of targets, and an interface is the names of its groups.
    The first ``train_clips`` clips are for training; evaluation worlds start on frames that the learner fixes.
    Targets are given ``horizons`` steps ahead, and episodes last ``episode_length_s`` at most. A share ``adaptive``
    of the training worlds starts where episodes fail; without ``count_failures`` a learner sets where they start,
    see :data:`~isaaclab_tasks.mdp2.tracking.mdp.reset_to_clips`.

    The worlds of the interfaces in ``pushes`` are pushed every ``push_interval_s``: their root gets a velocity of up
    to ``push_velocity`` [m/s] along x, y and z, times the value of the interface in ``pushes``.

    With a ``feet_weight``, two rewards of that weight each ask the ``feet`` bodies for the clip's positions and
    velocities. The tracking rewards average over every body, most of which are fingers, so without them a policy
    may step where the clip stands still. ``action_rate_weight`` weighs the penalty on the change of the actions:
    the default is small next to the tracking rewards, and lets the targets of the joints jump from step to step.
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
            "count_failures": count_failures,
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
    command = {"interfaces": patterns, "count": len(patterns), "groups": len(groups)}
    critic, events = {**policy, "interface": interface}, {"reset_to_motion": reset}
    if pushes:
        # The critic sees how hard its world is pushed; the policy does not.
        push = mdp2.term_state(INTERFACE, "push")
        command["push_scales"] = tuple(pushes.get(name, 0.0) for name in interfaces)
        critic["push"] = mdp2.Observation(interfaces_mdp.world_value, inputs={"value": push})
        events["push"] = mdp2.Event(
            interfaces_mdp.scaled_push,
            inputs={"scale": push},
            outputs={"root_com_vel_w": robot.root_com_vel_w},
            mode=mdp2.EventMode.INTERVAL,
            interval_range_s=push_interval_s,
            params={"velocity_range": (*((-speed, speed) for speed in push_velocity), *(((0.0, 0.0),) * 3))},
        )
    quantities, rewards = dict(task.quantities), dict(task.rewards)
    rewards["action_rate_l2"] = replace(rewards["action_rate_l2"], weight=-action_rate_weight)
    if feet_weight > 0.0:
        pose = {"body_link_pose_w": robot.body_link_pose_w, "body_ids": feet}
        quantities["feet_pos_error"] = mdp2.Quantity(tracking.subset_pos_error, inputs={**pose, **clip_body_pos})
        quantities["feet_vel_error"] = mdp2.Quantity(
            tracking.subset_vel_error,
            inputs={**pose, **{name: value for name, value in clip_body_pos.items() if name != "env_origins"}},
            params={"step_dt": step_dt},
        )
        for name, scale in (("feet_pos", 100.0), ("feet_vel", 2.0)):
            rewards[name] = mdp2.Reward(
                tracking.exp_kernel,
                inputs={"value": getattr(mdp2.quantities, f"{name}_error")},
                params={"scale": scale},
                weight=feet_weight,
            )
    return replace(
        task,
        quantities=quantities,
        rewards=rewards,
        # Targets for the joints are offsets from the default pose: the policy is not given the clip's pose.
        actions={"joint_pos": mdp2.DirectAction(robot.joint_pos_target, scale=0.25, offset=robot.default_joint_pos)},
        commands={
            "interface": mdp2.Command(interfaces_mdp.sampled_interface, params=command),
            "every_group": mdp2.Command(interfaces_mdp.ones_command, params={"groups": len(groups)}),
        },
        observations={
            "policy": mdp2.ObservationGroup(**measured, **targets(mdp2.commands.interface)),
            "teacher": mdp2.ObservationGroup(**measured, **targets(mdp2.commands.every_group)),
            "critic": mdp2.ObservationGroup(**critic),
        },
        terminations={
            **task.terminations,
            "time_out": mdp2.TimeLimit(episode_length_s=episode_length_s),
            "diverged": mdp2.Termination(tracking.is_not_finite, inputs={"value": mdp2.quantities.body_pos_error}),
        },
        events=events,
    )
