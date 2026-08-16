# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for the unified dexterous Lift and Reorient tasks."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.mdp.rewards import _ProgressReward


def test_camera_normalization_is_stationary() -> None:
    """RGB and depth normalization must not depend on per-frame statistics."""
    rgb = torch.tensor([0.0, 127.5, 255.0])
    depth = torch.tensor([0.0, 2.0])

    assert torch.allclose(mdp.vision_camera._rgb_norm(None, rgb), torch.tensor([-0.5, 0.0, 0.5]))
    assert torch.allclose(mdp.vision_camera._depth_norm(None, depth), torch.tanh(depth / 2) - 0.5)


def _reference_progress(
    best_error: torch.Tensor,
    prev_command: torch.Tensor | None,
    error: torch.Tensor,
    gate: torch.Tensor,
    min_improvement: float,
    command: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference the original masked bookkeeping implementation."""
    if prev_command is None:
        prev_command = command.clone()
    else:
        best_error[(prev_command != command).any(dim=1)] = float("inf")
        prev_command.copy_(command)
    unseeded = torch.isinf(best_error)
    best_error[unseeded] = error[unseeded]
    improved = gate & (error < best_error - min_improvement)
    best_error[improved] = error[improved]
    return improved.float(), prev_command


def test_progress_reward_full_shape_bookkeeping_matches_masked_reference() -> None:
    """First steps, command changes, contact gates, and partial resets preserve exact behavior."""
    env = SimpleNamespace(num_envs=4, device="cpu")
    reward = _ProgressReward(cfg=None, env=env)
    reference_best = torch.full((env.num_envs,), float("inf"))
    reference_command = None
    command = torch.zeros((env.num_envs, 3))
    min_improvement = 0.5
    steps = (
        (torch.tensor([10.0, 8.0, 6.0, 4.0]), torch.tensor([True, True, False, True])),
        (torch.tensor([8.0, 7.8, 4.0, 2.5]), torch.tensor([True, True, False, False])),
        (torch.tensor([7.7, 7.0, 4.0, 3.0]), torch.tensor([True, True, True, True])),
    )

    for error, gate in steps:
        expected, reference_command = _reference_progress(
            reference_best, reference_command, error, gate, min_improvement, command
        )
        actual = reward._progress(error, gate, min_improvement, command)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(reward.best_error, reference_best, rtol=0.0, atol=0.0)

    command[[0, 2], 1] = 1.0
    error = torch.tensor([1.0, 6.0, 10.0, 2.0])
    gate = torch.ones(env.num_envs, dtype=torch.bool)
    expected, reference_command = _reference_progress(
        reference_best, reference_command, error, gate, min_improvement, command
    )
    actual = reward._progress(error, gate, min_improvement, command)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reward.best_error, reference_best, rtol=0.0, atol=0.0)

    reset_ids = torch.tensor([1, 3])
    reward.reset(reset_ids)
    reference_best[reset_ids] = float("inf")
    error = torch.tensor([0.4, 3.0, 8.0, 1.0])
    gate = torch.tensor([True, False, True, True])
    expected, reference_command = _reference_progress(
        reference_best, reference_command, error, gate, min_improvement, command
    )
    actual = reward._progress(error, gate, min_improvement, command)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reward.best_error, reference_best, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reward._prev_command, reference_command, rtol=0.0, atol=0.0)
