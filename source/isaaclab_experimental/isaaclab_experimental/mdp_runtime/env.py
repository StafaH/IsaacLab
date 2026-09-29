# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-buffer environment view of a program for graph-captured learners."""

from __future__ import annotations

import warp as wp

from .plan import MdpProgram


@wp.kernel
def _clamp_copy(src: wp.array2d(dtype=wp.float32), lo: wp.float32, hi: wp.float32, dst: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    dst[i, j] = wp.clamp(src[i, j], lo, hi)


class MdpEnv:
    """Expose one observation group of a program through a fixed-buffer environment contract.

    The attributes match the ``WarpEnv`` protocol of the ``warp-rl`` learner: ``observations``,
    ``final_observations`` (pre-reset), ``rewards``, ``terminated``, and ``truncated`` are ``wp.array`` views of
    the program outputs with fixed addresses. :meth:`step` copies the caller's actions into the program input,
    so the caller may pass a different array every step, then enqueues one program step.

    Args:
        program: The program to expose.
        group: The observation group returned as ``observations``.
        action_bounds: Bounds applied while copying the learner's actions, e.g. ``(-5.0, 5.0)`` for unbounded
            Gaussian policies. The program then sees, stores, and observes the bounded raw action; the learner
            keeps its unbounded sample for likelihoods. None copies unchanged.
    """

    def __init__(self, program: MdpProgram, group: str = "policy", action_bounds: tuple[float, float] | None = None):
        plan, be = program.plan, program.backend
        if group not in plan.observation_widths:
            raise ValueError(f"Unknown observation group '{group}'. Available: {sorted(plan.observation_widths)}.")
        if not plan.compute_final_observations:
            raise ValueError("MdpEnv requires compute_final_observations=True for truncation bootstrapping.")
        self.program = program
        self.num_envs = plan.num_envs
        self.num_observations = plan.observation_widths[group]
        self.num_actions = plan.num_actions
        self.device = wp.get_device(plan.device)
        outputs = program.outputs
        self.observations = be.to_warp(outputs.observations[group])
        self.final_observations = be.to_warp(outputs.final_observations[group])
        self.rewards = be.to_warp(outputs.reward)
        self.terminated = be.to_warp(outputs.terminated)
        self.truncated = be.to_warp(outputs.truncated)
        self._actions = be.to_warp(program.inputs.actions)
        self._action_bounds = action_bounds

    def reset(self) -> None:
        """Reset every environment (host-side mask; outside capture)."""
        self.program.reset()

    def step(self, actions: wp.array) -> None:
        """Copy ``actions`` into the program input and enqueue one step."""
        if actions.shape != self._actions.shape or actions.dtype != wp.float32:
            raise ValueError(
                f"Expected float32 actions of shape {self._actions.shape}, got {actions.dtype} {actions.shape}."
            )
        if self._action_bounds is None:
            wp.copy(self._actions, actions)
        else:
            lo, hi = self._action_bounds
            wp.launch(_clamp_copy, dim=actions.shape, inputs=[actions, lo, hi, self._actions], device=self.device)
        self.program.step()
