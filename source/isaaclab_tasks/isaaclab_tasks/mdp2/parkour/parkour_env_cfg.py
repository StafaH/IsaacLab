# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour: one policy for worlds that track a clip of a robot climbing boxes and worlds that reach a goal on them.

Every world gives the policy the same observations: the robot, the terrain around it, and a goal, the pose of the
clip's root at a frame. A learner writes the episode of a world, see :data:`mdp.start_episode`, and the weights of
its rewards. A world that tracks a clip has a goal that runs ahead of the tracked frame, the tracking rewards, and
ends when it loses the clip. A world that reaches a goal has the goal of one frame, and the reward of having reached
it when its episode ends.
"""

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from isaaclab_tasks.mdp2.tracking import mdp as tracking

from . import mdp as parkour

robot = mdp2.Entity("robot")
scene = mdp2.Entity("scene")
height_scanner = mdp2.Entity("height_scanner")
joints = robot.joints(".*")

root_state = {"root_link_pose_w": robot.root_link_pose_w, "root_com_vel_w": robot.root_com_vel_w}
START = "events.start_episode"
episode = mdp2.term_state(START, "episode")
progress = {
    "start": mdp2.term_state(START, "start"),
    "end": mdp2.term_state(START, "end"),
    "episode_length": mdp2.episode_length,
}
clip_body_pos = {
    "clip_body_pos": mdp2.term_state(START, "clip_body_pos"),
    "env_origins": scene.env_origins,
    **progress,
}


def parkour_task(
    motion_files: list[str],
    lengths: list[int],
    step_dt: float,
    num_bodies: int,
    fps: float = 30.0,
    reference_input: bool = False,
) -> mdp2.TaskCfg:
    """Return the MDP in which the ``robot`` tracks the clips in ``motion_files`` or reaches goals on their terrain.

    ``lengths`` gives the frames of each clip, which are sampled every ``step_dt`` seconds. With
    ``reference_input``, the policy observes the frame that its world tracks in place of its last action, and
    corrects the joint positions of that frame: a tracker, which needs the clip to act.
    """
    frames = sum(round((length - 1) / (fps * step_dt)) + 1 for length in lengths)
    errors = {
        "joint_pos_error": mdp2.Quantity(
            tracking.joint_pos_error,
            inputs={
                "joint_pos": robot.joint_pos,
                "clip_joint_pos": mdp2.term_state(START, "clip_joint_pos"),
                **progress,
            },
            log=True,
        ),
        "joint_vel_error": mdp2.Quantity(
            tracking.joint_vel_error,
            inputs={
                "joint_vel": robot.joint_vel,
                "clip_joint_vel": mdp2.term_state(START, "clip_joint_vel"),
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
                "clip_root_vel": mdp2.term_state(START, "clip_root_vel"),
                **progress,
            },
            params={"angular_scale": 0.1},
        ),
    }
    kernels = {"joint_pos": (25.0, 15.0), "joint_vel": (0.05, 3.0), "body_pos": (100.0, 9.0), "root_vel": (1.0, 3.0)}
    observations = {
        "root_height": mdp2.Observation(tracking.root_height, inputs={"root_link_pose_w": robot.root_link_pose_w}),
        "base_lin_vel": mdp2.Observation(mdp.base_lin_vel, inputs=root_state),
        "base_ang_vel": mdp2.Observation(mdp.base_ang_vel, inputs=root_state),
        "projected_gravity": mdp2.Observation(
            mdp.projected_gravity, inputs={"root_link_pose_w": robot.root_link_pose_w, "gravity_w": scene.gravity_w}
        ),
        "joint_pos": mdp2.Observation(mdp.joint_pos, inputs={"joint_pos": robot.joint_pos, "joint_ids": joints}),
        "joint_vel": mdp2.Observation(
            mdp.joint_vel, inputs={"joint_vel": robot.joint_vel, "joint_ids": joints}, scale=0.1
        ),
        "body_pos": mdp2.Observation(
            tracking.body_pos_b,
            inputs={"body_link_pose_w": robot.body_link_pose_w, "root_link_pose_w": robot.root_link_pose_w},
        ),
        "height_scan": mdp2.Observation(
            mdp.height_scan,
            inputs={"pos_w": height_scanner.pos_w, "ray_hits_w": height_scanner.ray_hits_w},
            params={"offset": 0.8},
            clip=(-1.5, 1.5),
        ),
        "goal": mdp2.Observation(
            parkour.goal_b, inputs={"goal": mdp2.quantities.goal, "root_link_pose_w": robot.root_link_pose_w}
        ),
        "actions": mdp2.Observation(mdp.last_action, inputs={"actions": mdp2.last_action}),
    }
    reference_body_pos = mdp2.Observation(
        tracking.reference_body_pos_b,
        inputs={"root_link_pose_w": robot.root_link_pose_w, **clip_body_pos},
        params={"bodies": num_bodies},
    )
    errors["reference_joint_pos"] = mdp2.Quantity(
        tracking.reference_joint_pos,
        inputs={"clip_joint_pos": mdp2.term_state(START, "clip_joint_pos"), "joint_ids": joints, **progress},
    )
    reference_joint_pos = mdp2.Observation(
        mdp.generated_commands, inputs={"command": mdp2.quantities.reference_joint_pos}
    )
    # Actions are in radians: a learner scales what its network puts out, joint by joint.
    action = mdp2.DirectAction(robot.joint_pos_target, scale=1.0, offset=robot.default_joint_pos)
    policy = dict(observations)
    if reference_input:
        action = mdp2.DirectAction(robot.joint_pos_target, scale=1.0, offset=mdp2.quantities.reference_joint_pos)
        del policy["actions"]
        policy |= {"reference_joint_pos": reference_joint_pos, "reference_body_pos": reference_body_pos}

    return mdp2.TaskCfg(
        actions={"joint_pos": action},
        quantities={
            "goal": mdp2.Quantity(
                parkour.goal,
                inputs={
                    "clip_root_pose": mdp2.term_state(START, "clip_root_pose"),
                    "env_origins": scene.env_origins,
                    "start": progress["start"],
                    "episode_length": mdp2.episode_length,
                    "episode": episode,
                },
            ),
            "at_goal": mdp2.Quantity(
                parkour.at_goal,
                inputs={
                    "goal": mdp2.quantities.goal,
                    "root_link_pose_w": robot.root_link_pose_w,
                    "episode": episode,
                },
                params={"height_tolerance": 0.15, "min_upright": 0.9},
                log=True,
            ),
            **errors,
        },
        observations={
            "policy": mdp2.ObservationGroup(**policy),
            # The critic also sees the kind of episode, its remaining time, and the frame that the world tracks,
            # which is also what a tracker observes on top of the policy's observations.
            "critic": mdp2.ObservationGroup(
                **observations,
                episode=mdp2.Observation(
                    parkour.episode_b,
                    inputs={
                        "episode": episode,
                        "episode_length": mdp2.episode_length,
                        "at_goal": mdp2.quantities.at_goal,
                    },
                ),
                reference_joint_pos=reference_joint_pos,
                reference_body_pos=reference_body_pos,
            ),
        },
        rewards={
            **{
                name: mdp2.Reward(
                    tracking.exp_kernel,
                    inputs={"value": getattr(mdp2.quantities, f"{name}_error")},
                    params={"scale": scale},
                    weight=weight,
                )
                for name, (scale, weight) in kernels.items()
            },
            "is_alive": mdp2.Reward(mdp.is_alive, inputs={"terminated": mdp2.terminated}, weight=0.5),
            "upright": mdp2.Reward(
                mdp.flat_orientation_l2,
                inputs={"root_link_pose_w": robot.root_link_pose_w, "gravity_w": scene.gravity_w},
                weight=-0.5,
            ),
            # The reward of a step is its weight times the step, so reaching a goal is worth 20.
            "goal_reached": mdp2.Reward(
                parkour.goal_reached,
                inputs={
                    "at_goal": mdp2.quantities.at_goal,
                    "episode_length": mdp2.episode_length,
                    "episode": episode,
                },
                weight=20.0 / step_dt,
            ),
            # Ending an episode before its time costs half of what reaching the goal is worth.
            "failed": mdp2.Reward(
                parkour.failed,
                inputs={"terminated": mdp2.terminated, "episode_length": mdp2.episode_length, "episode": episode},
                weight=-10.0 / step_dt,
            ),
            # The cost of control, in every kind of world: the rate of the joint targets [rad], the mechanical
            # power of the joints, and the joints near their limits.
            "action_rate_l2": mdp2.Reward(mdp.action_rate_l2, inputs={"actions": mdp2.last_action}, weight=-10.0),
            "joint_power": mdp2.Reward(
                parkour.joint_power_l1,
                inputs={"applied_effort": robot.applied_effort, "joint_vel": robot.joint_vel, "joint_ids": joints},
                weight=-0.003,
            ),
        },
        terminations={
            "time_out": mdp2.TimeLimit(episode_length_s=60.0),
            "clip_ended": mdp2.Termination(
                parkour.after_steps,
                inputs={"episode_length": mdp2.episode_length, "episode": episode},
                params={"column": 6},
                time_out=True,
            ),
            "episode_ended": mdp2.Termination(
                parkour.after_steps,
                inputs={"episode_length": mdp2.episode_length, "episode": episode},
                params={"column": 7},
            ),
            "tracking_lost": mdp2.Termination(
                parkour.above_limit,
                inputs={"value": mdp2.quantities.body_pos_error, "episode": episode},
                params={"column": 4},
            ),
            # On the ground beside the boxes, where no clip takes the root.
            "fallen": mdp2.Termination(
                mdp.root_height_below_minimum,
                inputs={"root_link_pose_w": robot.root_link_pose_w},
                params={"minimum_height": 0.3},
            ),
            "diverged": mdp2.Termination(tracking.is_not_finite, inputs={"value": mdp2.quantities.body_pos_error}),
            "left_terrain": mdp2.Termination(
                mdp.root_beyond_distance,
                inputs={"root_link_pose_w": robot.root_link_pose_w, "env_origins": scene.env_origins},
                params={"distance": 3.5},
            ),
        },
        events={
            # Before the start of the episode, which reads the origin.
            "terrain_origin": mdp2.Event(mdp.move_env_origin, outputs={"env_origins": scene.env_origins}),
            "start_episode": mdp2.Event(
                parkour.start_episode,
                inputs={
                    "env_origins": scene.env_origins,
                    "soft_joint_pos_limits": robot.soft_joint_pos_limits,
                    "body_link_pose_w": robot.body_link_pose_w,
                },
                outputs={"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel, **root_state},
                params={
                    "motion_files": tuple(motion_files),
                    "windows": tuple((0, length) for length in lengths),
                    "frames": frames,
                    "clips": len(motion_files),
                    "bodies": num_bodies,
                    "fps": fps,
                    "noise": 0.02,
                },
            ),
        },
    )
