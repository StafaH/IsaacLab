# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heterogeneous populations: agent types with different terms, observation widths, and action widths.

Layout: the population is partitioned into contiguous per-type blocks. Every type keeps its own dense
``(N_g, width_g)`` observation and ``(N_g, A_g)`` action buffers and its own compiled plan. Per-environment
scalars (reward, terminated, truncated, reset mask) are population-wide ``(N,)`` buffers; each type writes
its contiguous row range ``[offset_g, offset_g + N_g)`` through a view. Nothing is padded.

:meth:`HeterogeneousProgram.packed_observations` optionally adds a padded ``(N, max_width)`` copy for learners
that need one tensor. It is an explicit extra operation, not the storage layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .cfg import MdpCfg
from .physics import PhysicsBinding
from .plan import CapturedStep, ExecutionPlan, MdpConfigError, MdpProgram, capture_step, compile_plan


class HeterogeneousProgram:
    """Programs for several agent types sharing population-wide per-environment outputs.

    Each type has its own physics binding; bindings must be distinct objects. A step runs the types in
    declaration order on one stream, so the whole population can be captured in one graph.
    """

    def __init__(self, types: Mapping[str, tuple[MdpCfg, PhysicsBinding]], backend: str = "warp"):
        if not types:
            raise MdpConfigError(["at least one agent type is required."])
        errors, plans = [], {}
        for name, (cfg, physics) in types.items():
            try:
                plans[name] = compile_plan(cfg, physics, backend)
            except MdpConfigError as e:
                errors += [f"{name}: {message}" for message in e.errors]
        bindings = [physics for _, physics in types.values()]
        if len({id(p) for p in bindings}) != len(bindings):
            errors.append("agent types must use distinct physics bindings.")
        if len({p.device for p in bindings}) != 1:
            errors.append("agent types must share one device.")
        if errors:
            raise MdpConfigError(errors)

        self.plans: dict[str, ExecutionPlan] = plans
        first = next(iter(plans.values()))
        self.backend = be = first.make_backend()
        self.num_envs = sum(p.num_envs for p in plans.values())
        self.reward = be.zeros((self.num_envs,), "float32")
        self.terminated = be.zeros((self.num_envs,), "bool")
        self.truncated = be.zeros((self.num_envs,), "bool")
        self.reset_mask = be.zeros((self.num_envs,), "bool")
        self.slices: dict[str, slice] = {}
        self.programs: dict[str, MdpProgram] = {}
        offset = 0
        for name, plan in plans.items():
            rows = slice(offset, offset + plan.num_envs)
            outputs = plan.allocate_outputs(
                **{
                    key: be.rows(array, rows.start, rows.stop)
                    for key, array in (
                        ("reward", self.reward),
                        ("terminated", self.terminated),
                        ("truncated", self.truncated),
                        ("reset_mask", self.reset_mask),
                    )
                }
            )
            self.programs[name] = plan.bind(plan.allocate_inputs(), plan.allocate_state(), outputs)
            self.slices[name] = rows
            offset = rows.stop
        self._pack_ops: list = []

    def packed_observations(self, group: str = "policy", pad: float = 0.0) -> tuple[Any, dict[str, int]]:
        """Add a step operation that copies every type's ``group`` into one zero-padded ``(N, max_width)`` buffer.

        Returns:
            The padded buffer and the valid width of each type. Padding columns carry no information and a
            learner must mask them; this view does not make the population homogeneous.
        """
        widths = {}
        for name, plan in self.plans.items():
            if group not in plan.observation_widths:
                raise ValueError(f"Agent type '{name}' has no observation group '{group}'.")
            widths[name] = plan.observation_widths[group]
        packed = self.backend.zeros((self.num_envs, max(widths.values())), "float32")
        for name, program in self.programs.items():
            op = self.backend.bind_pack_rows(program.outputs.observations[group], packed, self.slices[name].start, pad)
            self._pack_ops.append(op)
        return packed, widths

    def step(self, include_physics: bool = True) -> None:
        """Step every type, then refresh packed views."""
        for program in self.programs.values():
            program.step(include_physics)
        with self.backend.stream_scope():
            for op in self._pack_ops:
                op()

    def reset(self) -> None:
        """Reset every environment of every type (host-side mask; outside capture)."""
        for program in self.programs.values():
            program.reset()
        with self.backend.stream_scope():
            for op in self._pack_ops:
                op()

    def capture(self, include_physics: bool = True, warmup: bool = True) -> CapturedStep:
        """Capture one population step into a single CUDA graph."""
        return capture_step(self.backend, lambda: self.step(include_physics), warmup=warmup, owner=self)
