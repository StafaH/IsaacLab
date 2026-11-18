# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset event of the parkour task."""

from __future__ import annotations

import warp as wp
from isaaclab_experimental import mdp2

from isaaclab_tasks.mdp2.tracking.mdp.events import CLIP_STATE, _load_csv_motions, _start_on_frame


def _load_clips(ctx, **params):
    _load_csv_motions(ctx, **params)
    # Until a learner writes the episodes, every world tracks the first clip from its first frame to its end.
    last = float(ctx.state["clip_range"][0, 1])
    for column, value in enumerate((0.0, last, last - 1.0, 45.0, 0.25, 0.0, last - 2.0, 1.0e9, 0.0, 0.0, 0.0)):
        ctx.state["pending"][:, column] = value
        ctx.state["episode"][:, column] = value


@wp.func
def _start_episode(
    world: int,
    joint_pos: wp.array2d(dtype=float),
    joint_vel: wp.array2d(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    env_origins: wp.array(dtype=wp.vec3f),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    clip_joint_pos: wp.array2d(dtype=float),
    clip_joint_vel: wp.array2d(dtype=float),
    clip_root_pose: wp.array2d(dtype=float),
    clip_root_vel: wp.array2d(dtype=float),
    start: wp.array(dtype=float),
    end: wp.array(dtype=float),
    pending: wp.array2d(dtype=float),
    episode: wp.array2d(dtype=float),
    noise: float,
    rng: wp.int32,
):
    for k in range(episode.shape[1]):
        episode[world, k] = pending[world, k]
    _start_on_frame(
        world,
        int(episode[world, 0]),
        int(episode[world, 1]),
        joint_pos,
        joint_vel,
        root_link_pose_w,
        root_com_vel_w,
        env_origins,
        soft_joint_pos_limits,
        clip_joint_pos,
        clip_joint_vel,
        clip_root_pose,
        clip_root_vel,
        start,
        end,
        noise,
        rng,
    )

    # Move the root in the ground plane and turn it about the vertical axis through it.
    pose = root_link_pose_w[world]
    turn = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), episode[world, 10])
    position = wp.transform_get_translation(pose) + wp.vec3f(episode[world, 8], episode[world, 9], 0.0)
    root_link_pose_w[world] = wp.transformf(position, turn * wp.transform_get_rotation(pose))
    velocity = root_com_vel_w[world]
    linear = wp.quat_rotate(turn, wp.spatial_top(velocity))
    angular = wp.quat_rotate(turn, wp.spatial_bottom(velocity))
    root_com_vel_w[world] = wp.spatial_vector(linear, angular)


start_episode = mdp2.Term(
    warp=_start_episode,
    state={
        **CLIP_STATE,
        "pending": mdp2.State(("worlds", 11), persistent=True),
        "episode": mdp2.State(("worlds", 11), persistent=True),
    },
    random=True,
    init=_load_clips,
)
"""Start the reset worlds on the frame of a clip that a learner chose, with the episode that it chose.

The clips are those of :data:`~isaaclab_tasks.mdp2.tracking.mdp.reset_to_csv_motion`. A learner writes the next
episode of a world into the ``pending`` state, and a reset copies it to the ``episode`` state, which the other
terms read. Its columns are: the frame to start on; the frame after the last of that clip; the frame whose root pose
is the goal; the frames that the goal runs ahead of the tracked frame until it gets there; the mean squared distance
of the bodies to the tracked frame that ends the episode [m^2]; the distance to the goal that counts as reaching it
[m]; the steps after which the episode ends as a time-out; the steps after which it ends for good; and how far the
root starts from the frame in the ground plane, along x and y [m] and turned about the vertical axis [rad].
"""
