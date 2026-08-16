# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands.velocity_command import NormalVelocityCommand, UniformVelocityCommand

pytestmark = pytest.mark.unit


def test_uniform_velocity_update_preserves_heading_and_standing_combinations():
    """Heading control and standing masks produce the established seeded command values."""
    generator = torch.Generator().manual_seed(42)
    command = object.__new__(UniformVelocityCommand)
    command.cfg = SimpleNamespace(
        heading_command=True,
        heading_control_stiffness=0.75,
        ranges=SimpleNamespace(ang_vel_z=(-1.0, 0.8)),
    )
    command.vel_command_b = torch.randn((8, 3), generator=generator)
    command.heading_target = 4.0 * torch.randn(8, generator=generator)
    heading_w = 4.0 * torch.randn(8, generator=generator)
    command.robot = SimpleNamespace(data=SimpleNamespace(heading_w=SimpleNamespace(torch=heading_w)))
    command.is_heading_env = torch.tensor([False, True, False, True, False, True, False, True])
    command.is_standing_env = torch.tensor([False, False, True, True, False, False, True, True])

    expected = command.vel_command_b.clone()
    heading_env_ids = command.is_heading_env.nonzero(as_tuple=False).flatten()
    heading_error = math_utils.wrap_to_pi(command.heading_target[heading_env_ids] - heading_w[heading_env_ids])
    expected[heading_env_ids, 2] = torch.clip(
        command.cfg.heading_control_stiffness * heading_error,
        min=command.cfg.ranges.ang_vel_z[0],
        max=command.cfg.ranges.ang_vel_z[1],
    )
    expected[command.is_standing_env.nonzero(as_tuple=False).flatten()] = 0.0

    command._update_command()

    torch.testing.assert_close(command.vel_command_b, expected)


def test_uniform_velocity_update_ignores_heading_mask_when_disabled():
    """A disabled heading command leaves moving yaw commands unchanged and zeros standing commands."""
    generator = torch.Generator().manual_seed(7)
    command = object.__new__(UniformVelocityCommand)
    command.cfg = SimpleNamespace(heading_command=False)
    command.vel_command_b = torch.randn((4, 3), generator=generator)
    command.is_heading_env = torch.ones(4, dtype=torch.bool)
    command.is_standing_env = torch.tensor([False, True, False, True])

    expected = command.vel_command_b.clone()
    expected[command.is_standing_env.nonzero(as_tuple=False).flatten()] = 0.0

    command._update_command()

    torch.testing.assert_close(command.vel_command_b, expected)


def test_normal_velocity_update_preserves_standing_and_component_masks():
    """Standing and component masks produce the established seeded command values."""
    generator = torch.Generator().manual_seed(123)
    command = object.__new__(NormalVelocityCommand)
    command.vel_command_b = torch.randn((8, 3), generator=generator)
    command.is_standing_env = torch.tensor([False, False, False, False, True, True, True, True])
    command.is_zero_vel_x_env = torch.tensor([False, True, False, True, False, True, False, True])
    command.is_zero_vel_y_env = torch.tensor([False, False, True, True, False, False, True, True])
    command.is_zero_vel_yaw_env = torch.tensor([True, False, True, False, True, False, True, False])
    original_masks = (
        command.is_standing_env.clone(),
        command.is_zero_vel_x_env.clone(),
        command.is_zero_vel_y_env.clone(),
        command.is_zero_vel_yaw_env.clone(),
    )

    expected = command.vel_command_b.clone()
    expected[command.is_standing_env.nonzero(as_tuple=False).flatten()] = 0.0
    expected[command.is_zero_vel_x_env.nonzero(as_tuple=False).flatten(), 0] = 0.0
    expected[command.is_zero_vel_y_env.nonzero(as_tuple=False).flatten(), 1] = 0.0
    expected[command.is_zero_vel_yaw_env.nonzero(as_tuple=False).flatten(), 2] = 0.0

    command._update_command()

    torch.testing.assert_close(command.vel_command_b, expected)
    for mask, original_mask in zip(
        (
            command.is_standing_env,
            command.is_zero_vel_x_env,
            command.is_zero_vel_y_env,
            command.is_zero_vel_yaw_env,
        ),
        original_masks,
    ):
        torch.testing.assert_close(mask, original_mask)
