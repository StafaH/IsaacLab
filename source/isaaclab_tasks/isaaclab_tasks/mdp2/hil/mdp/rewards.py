# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms of the hybrid imitation learning tasks: following the goal, and moving like the clip."""

from __future__ import annotations

import torch
import warp as wp
from isaaclab_experimental import mdp2

from isaaclab.utils import math as math_utils


@wp.func
def _heading_velocity(
    world: int,
    command: wp.array2d(dtype=float),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_vel_w: wp.array(dtype=wp.spatial_vectorf),
    scale: float,
    facing_weight: float,
) -> float:
    direction = wp.vec2f(command[world, 0], command[world, 1])
    velocity = root_com_vel_w[world]
    error = command[world, 2] - (direction[0] * velocity[0] + direction[1] * velocity[1])
    forward = wp.quat_rotate(wp.transform_get_rotation(root_link_pose_w[world]), wp.vec3f(1.0, 0.0, 0.0))
    facing = wp.dot(wp.normalize(wp.vec2f(forward[0], forward[1])), direction)
    return (1.0 - facing_weight) * wp.exp(-scale * error * error) + facing_weight * facing


heading_velocity = mdp2.Term(warp=_heading_velocity)
"""Moving at the goal's speed along its direction, ``exp(-scale * error^2)``, and facing it, by ``facing_weight``."""


class Discriminator(torch.nn.Module):
    """Scores a transition, two consecutive :func:`motion_features`, near 1 if it looks like the clip and -1 if not."""

    def __init__(self, num_features: int, hidden: tuple[int, ...] = (512, 256)):
        super().__init__()
        self.register_buffer("mean", torch.zeros(2 * num_features))
        self.register_buffer("std", torch.ones(2 * num_features))
        sizes = (2 * num_features, *hidden)
        layers = []
        for k in range(len(hidden)):
            layers += [torch.nn.Linear(sizes[k], sizes[k + 1]), torch.nn.ReLU()]
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(sizes[-1], 1))

    def forward(self, transition: torch.Tensor) -> torch.Tensor:
        return self.net((transition - self.mean) / self.std).squeeze(-1)


def motion_features(joint_pos, joint_vel, root_link_pose_w, root_com_vel_w):
    """Return what the discriminator sees of a state: the root's height, velocity and tilt in its own frame, and the
    joint positions and velocities."""
    quat = root_link_pose_w[:, 3:7]
    x, y, z, w = quat.unbind(-1)
    gravity_b = -torch.stack([2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)], dim=-1)
    linear = math_utils.quat_apply_inverse(quat, root_com_vel_w[:, :3])
    angular = math_utils.quat_apply_inverse(quat, root_com_vel_w[:, 3:])
    return torch.cat([root_link_pose_w[:, 2:3], linear, angular, gravity_b, joint_pos, joint_vel], dim=-1)


def _move_discriminator(ctx, *, discriminator, features):
    discriminator.to(ctx.state["transition"].device)


def _style(joint_pos, joint_vel, root_link_pose_w, root_com_vel_w, transition, *, discriminator):
    size = transition.shape[1] // 2
    transition[:, :size] = transition[:, size:]
    transition[:, size:] = motion_features(joint_pos, joint_vel, root_link_pose_w, root_com_vel_w)
    with torch.no_grad():
        score = discriminator(transition)
    return (1.0 - 0.25 * (score - 1.0).square()).clamp(min=0.0)


def _restart_style(mask, joint_pos, joint_vel, root_link_pose_w, root_com_vel_w, transition):
    size = transition.shape[1] // 2
    features = motion_features(joint_pos, joint_vel, root_link_pose_w, root_com_vel_w)
    transition[:, size:] = torch.where(mask[:, None], features, transition[:, size:])


style = mdp2.Term(
    torch=mdp2.Implementation(compute=_style, reset=_restart_style),
    state={"transition": mdp2.State(("worlds", "features"))},
    init=_move_discriminator,
)
"""How much the last step looks like the clip to ``discriminator``, a network that the learner trains: ``max(0, 1 -
0.25 (score - 1)^2)``, as in adversarial motion priors.

The state ``transition`` holds the :func:`motion_features` before and after the step, ``features`` values in all,
which the learner reads to train the discriminator. The network runs inside the step's CUDA graph.
"""
