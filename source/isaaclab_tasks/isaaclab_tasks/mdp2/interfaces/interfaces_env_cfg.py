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
from isaaclab_tasks.mdp2.tracking.tracking_env_cfg import RESET, clip_body_pos, clip_joint_pos, tracking_task

from . import mdp as interfaces_mdp

robot = mdp2.Entity("robot")

MEASUREMENT_NOISE = {"base_ang_vel": 0.2, "projected_gravity": 0.05, "joint_pos": 0.01, "joint_vel": 0.5}
"""The half width of the uniform noise on what the robot measures [rad/s, -, rad, rad/s], before any scaling."""

INTERFACE = "commands.interface"
"""The command whose state a learner sets: ``shares`` of the interfaces, and the ``fixed`` interface of a world."""


def robot_randomization(torso: mdp2.SubsetSelector) -> dict[str, mdp2.Event]:
    """Return the events that give every world its own robot when the simulation starts."""
    startup = mdp2.EventMode.STARTUP
    return {
        "friction": mdp2.Event(
            mdp.randomize_rigid_body_material,
            mode=startup,
            inputs={"body_ids": robot.bodies(".*")},
            params={"friction_range": (0.3, 1.6), "restitution_range": (0.0, 0.0)},
        ),
        "mass": mdp2.Event(
            mdp.randomize_rigid_body_mass,
            mode=startup,
            inputs={"body_ids": robot.bodies(".*")},
            params={"mass_distribution_params": (0.9, 1.1), "operation": "scale"},
        ),
        "torso_com": mdp2.Event(
            mdp.randomize_rigid_body_com,
            mode=startup,
            inputs={"body_ids": torso},
            params={"com_range": ((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05))},
        ),
        "gains": mdp2.Event(
            mdp.randomize_actuator_gains,
            mode=startup,
            inputs={"joint_ids": robot.joints(".*")},
            params={
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
        ),
    }


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
    local: bool = False,
    randomize: bool = False,
    torso: mdp2.SubsetSelector | None = None,
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

    With ``local``, the task asks for nothing that a robot cannot measure: the policy sees the targets of the bodies
    from the root of the clip, in the axes of its own root, which takes its orientation and not its position. The
    distance of the bodies to the clip, in the reward and in the termination, is taken with the clip moved
    horizontally onto the robot, and a wide reward for the horizontal position of the root keeps the robot from
    drifting away.

    With ``randomize``, every world gets its own robot when the simulation starts: the friction of its bodies,
    their masses, the center of mass of its torso and the gains of its joints differ from world to world, and the
    policy sees its measurements with noise. The critic sees them without. ``torso`` is the body whose center of
    mass moves.
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
            "blocks": -(-frames // (15 * tracking.BLOCK_BINS)),
            "decay": 0.999,
            "train_clips": train_clips,
            "count_failures": count_failures,
        },
    )

    clip_root = {"root_link_pose_w": robot.root_link_pose_w, "clip_root_pose": mdp2.term_state(RESET, "clip_root_pose")}
    seen_from_clip = {name: value for name, value in clip_body_pos.items() if name != "env_origins"} | clip_root

    def targets(command) -> dict[str, mdp2.Observation]:
        """The targets of every group and horizon, shown where ``command`` has the group."""
        terms = {}
        for ahead in horizons:
            for index, (name, bodies) in enumerate(body_groups.items()):
                terms[f"{name}_{ahead}"] = mdp2.Observation(
                    interfaces_mdp.commanded_body_pos_local if local else interfaces_mdp.commanded_body_pos_b,
                    inputs={
                        "root_link_pose_w": robot.root_link_pose_w,
                        **(seen_from_clip if local else clip_body_pos),
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
    if randomize:
        measured = {
            name: replace(term, noise=mdp2.UniformNoise(-MEASUREMENT_NOISE[name], MEASUREMENT_NOISE[name]))
            if name in MEASUREMENT_NOISE
            else term
            for name, term in measured.items()
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
    if randomize:
        events |= robot_randomization(torso)
    quantities, rewards = dict(task.quantities), dict(task.rewards)
    rewards["action_rate_l2"] = replace(rewards["action_rate_l2"], weight=-action_rate_weight)
    pos_error, pos_inputs = tracking.subset_pos_error, clip_body_pos
    if local:
        pos_error, pos_inputs = tracking.local_pos_error, {**clip_body_pos, **clip_root}
        quantities["body_pos_error"] = mdp2.Quantity(
            pos_error,
            inputs={"body_link_pose_w": robot.body_link_pose_w, "body_ids": robot.bodies(".*"), **pos_inputs},
            log=True,
        )
        drift = {name: value for name, value in pos_inputs.items() if name != "clip_body_pos"}
        quantities["root_drift"] = mdp2.Quantity(tracking.root_drift, inputs=drift, log=True)
        rewards["root_pos"] = mdp2.Reward(
            tracking.exp_kernel, inputs={"value": mdp2.quantities.root_drift}, params={"scale": 4.0}, weight=3.0
        )
    if feet_weight > 0.0:
        pose = {"body_link_pose_w": robot.body_link_pose_w, "body_ids": feet}
        quantities["feet_pos_error"] = mdp2.Quantity(pos_error, inputs={**pose, **pos_inputs})
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
