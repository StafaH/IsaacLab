# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch backend: buffers are ``torch.Tensor`` and the step is a fixed list of vectorized tensor functions.

The executor follows the Warp backend's schedule, arithmetic order, and random-draw order per environment.
It uses masked updates instead of index selection, so no operation synchronizes with the host. Random stream
states are ``int64`` holding unsigned 32-bit values, so the PCG32 stream matches Warp bit for bit.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp

from .terms import TermContext, get_impl

if TYPE_CHECKING:
    from .plan import MdpProgram, ResolvedTerm

_DTYPES = {"float32": torch.float32, "int32": torch.int32, "bool": torch.bool, "rng": torch.int64, "index": torch.long}
_MASK32 = 0xFFFFFFFF
_WARP_STREAMS: dict[int, wp.Stream] = {}


def warp_stream(stream: torch.cuda.Stream) -> wp.Stream:
    """Return one persistent Warp stream object per Torch stream.

    Warp tracks capture state per :class:`warp.Stream` object, so capture registration and the launches
    inside the capture must use the same object.
    """
    if stream.cuda_stream not in _WARP_STREAMS:
        _WARP_STREAMS[stream.cuda_stream] = wp.stream_from_torch(stream)
    return _WARP_STREAMS[stream.cuda_stream]


def rng_next(state: torch.Tensor) -> torch.Tensor:
    """Advance PCG32 stream states (``int64`` tensors holding ``uint32`` values)."""
    return (state * 747796405 + 2891336453) & _MASK32


def rng_uniform(state: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Map advanced PCG32 states to ``[lo, hi)`` exactly as the Warp ``rng_uniform`` function does."""
    word = (((state >> ((state >> 28) + 4)) ^ state) * 277803737) & _MASK32
    word = (word >> 22) ^ word
    u = (word >> 8).to(torch.float32) * 5.9604644775390625e-08
    # Warp receives the bounds as float32 kernel arguments and subtracts them in float32.
    lo32 = np.float32(lo)
    return float(lo32) + float(np.float32(hi) - lo32) * u


def draw(rng: torch.Tensor, mask: torch.Tensor | None, lo: float, hi: float) -> torch.Tensor:
    """Advance the streams of the masked environments (all if None) and return one uniform draw per env."""
    state = rng_next(rng)
    if mask is None:
        rng.copy_(state)
    else:
        rng.copy_(torch.where(mask, state, rng))
    return rng_uniform(state, lo, hi)


class TorchBackend:
    """Allocation helpers and executor factory for Torch programs.

    Physics fields are zero-copy views of the binding's Warp arrays. Work is enqueued on the current Torch
    stream and Warp physics launches are redirected to that stream (:meth:`stream_scope`).
    """

    name = "torch"

    def __init__(self, device: str):
        self.device = device

    def zeros(self, shape: tuple[int, ...], dtype: str) -> torch.Tensor:
        return torch.zeros(shape, dtype=_DTYPES[dtype], device=self.device)

    def constant(self, values: Sequence[float], dtype: str) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values), dtype=_DTYPES[dtype], device=self.device)

    def adopt(self, array: wp.array) -> torch.Tensor:
        """Return a physics field in this backend's array type (zero-copy)."""
        return wp.to_torch(array)

    def to_warp(self, array: torch.Tensor) -> wp.array:
        return wp.from_torch(array)

    def to_numpy(self, array: torch.Tensor) -> np.ndarray:
        return array.cpu().numpy()

    def rows(self, array: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        return array[start:stop]

    def columns(self, array: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        return array[:, start:stop]

    def copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        dst.copy_(src)

    def stream_scope(self):
        if torch.device(self.device).type != "cuda":
            return contextlib.nullcontext()
        stream = torch.cuda.current_stream(self.device)
        # Synchronizing with the previous Warp stream is required eagerly but invalid during capture.
        sync = not torch.cuda.is_current_stream_capturing()
        return wp.ScopedStream(warp_stream(stream), sync_enter=sync, sync_exit=sync)

    def is_capturing(self) -> bool:
        return torch.device(self.device).type == "cuda" and torch.cuda.is_current_stream_capturing()

    def seed(self, rng: torch.Tensor, seed: int) -> None:
        ids = torch.arange(rng.shape[0], dtype=torch.int64, device=self.device)
        rng.copy_(rng_next(rng_next((ids * 2654435761 + (seed & _MASK32)) & _MASK32)))

    def assign_mask(self, mask: torch.Tensor, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            mask.fill_(True)
            return
        mask.fill_(False)
        mask[torch.as_tensor(env_ids, dtype=torch.long, device=self.device)] = True

    def bind_pack_rows(self, src, dst, offset, pad) -> Callable[[], None]:
        rows, width = src.shape
        target, padding = dst[offset : offset + rows, :width], dst[offset : offset + rows, width:]

        def run():
            target.copy_(src)
            padding.fill_(pad)

        return run

    def build_executor(self, program: MdpProgram) -> TorchExecutor:
        return TorchExecutor(program)


class TorchExecutor:
    """Binds every term as a tensor function and runs them in the program schedule."""

    def __init__(self, program: MdpProgram):
        self.program = program
        plan, s, o = program.plan, program.state, program.outputs
        be = program.backend
        fields = {name: be.adopt(array) for name, array in plan.physics.fields.items()}
        fields.update(
            action=s.action,
            prev_action=s.prev_action,
            episode_length=s.episode_length,
            terminated=o.terminated,
            truncated=o.truncated,
            termination_values=s.termination_values,
            commands=s.commands,
        )

        def bind(term: ResolvedTerm, out: Any = None, state: Any = None):
            indices = {
                key: be.constant(term.params[key], "index")
                for key in ("joint_ids", "body_ids", "contact_ids", "term_ids")
                if key in term.params
            }
            declared = term.spec.reads + term.spec.writes
            ctx = TermContext(
                params=term.params,
                columns=term.columns,
                state_columns=term.state_columns,
                info=plan.info,
                fields={name: fields[name] for name in declared},
                out=out,
                state=state,
                rng=s.rng,
                indices=indices,
            )
            return get_impl(term.spec.name, "torch")(ctx)

        # actions
        scale, offset, lo, hi = [], [], [], []
        for term in plan.actions:
            clip = term.cfg.clip or (-np.inf, np.inf)
            scale += [term.cfg.scale] * term.width
            offset += [term.cfg.offset] * term.width
            lo += [clip[0]] * term.width
            hi += [clip[1]] * term.width
        self._action_constants = [be.constant(v, "float32") for v in (scale, offset, lo, hi)]
        self._actions = [bind(t, out=be.columns(s.processed_actions, *t.columns)) for t in plan.actions]
        # post-physics terms
        self._terminations = [bind(t, out=s.termination_values[i]) for i, t in enumerate(plan.terminations)]
        self._time_out = be.constant([t.cfg.time_out for t in plan.terminations], "bool")[:, None]
        self._rewards = [bind(t, out=s.reward_values[i]) for i, t in enumerate(plan.rewards)]
        self._reset_events = [bind(t) for t in plan.reset_events]
        self._interval_events = [bind(t) for t in plan.interval_events]
        self._reset_writes = any(t.spec.writes for t in plan.reset_events)
        self._commands = [
            bind(t, out=be.columns(s.commands, *t.columns), state=be.columns(s.command_state, *t.state_columns))
            for t in plan.commands
        ]
        # observations: (term function, noise, clip, scale, columns) per term, per output buffer
        terms = [t for group in plan.observations.values() for t in group]
        self._observe = {
            "obs": [(bind(t, out=be.columns(o.observation_buffer, *t.columns)), t) for t in terms],
        }
        if o.final_observation_buffer is not None:
            self._observe["final"] = [
                (bind(t, out=be.columns(o.final_observation_buffer, *t.columns)), t) for t in terms
            ]
        self._buffers = {"obs": o.observation_buffer, "final": o.final_observation_buffer}
        self._wp_reset_mask = be.to_warp(o.reset_mask)
        self._wp_commit_mask = be.to_warp(s.commit_mask)
        self._wp_reset_request = be.to_warp(s.reset_request)

    # -- stages --------------------------------------------------------------------------------------------

    def _observe_into(self, key: str) -> None:
        rng, buffer = self.program.state.rng, self._buffers[key]
        for run, term in self._observe[key]:
            run()
            start, stop = term.columns
            if term.cfg.noise is not None:
                for c in range(start, stop):
                    buffer[:, c] += draw(rng, None, *term.cfg.noise)
            if term.cfg.clip is not None or term.cfg.scale != 1.0:
                view = buffer[:, start:stop]
                clip = term.cfg.clip or (-np.inf, np.inf)
                torch.mul(
                    torch.clamp(view, float(np.float32(clip[0])), float(np.float32(clip[1]))), term.cfg.scale, out=view
                )

    def _reset_block(self, mask: torch.Tensor) -> None:
        plan, s = self.program.plan, self.program.state
        for run in self._reset_events:
            run(mask)
        s.episode_length.masked_fill_(mask, 0)
        s.action.masked_fill_(mask[:, None], 0.0)
        s.prev_action.masked_fill_(mask[:, None], 0.0)
        s.episode_sums.masked_fill_(mask, 0.0)
        for e, term in enumerate(plan.interval_events):
            value = draw(s.rng, mask, *term.cfg.interval_range_s)
            s.interval_time_left[e] = torch.where(mask, value, s.interval_time_left[e])
        for c, term in enumerate(plan.commands):
            value = draw(s.rng, mask, *term.cfg.resampling_time_range)
            s.command_time_left[c] = torch.where(mask, value, s.command_time_left[c])
            self._commands[c][0](mask)

    def step(self, include_physics: bool) -> None:
        program = self.program
        plan, s, o = program.plan, program.state, program.outputs
        physics = plan.physics
        # actions
        scale, offset, lo, hi = self._action_constants
        raw = program.inputs.actions
        s.prev_action.copy_(s.action)
        s.action.copy_(raw)
        torch.clamp(raw * scale + offset, lo, hi, out=s.processed_actions)
        for run in self._actions:
            run()
        if include_physics:
            physics.step()
        # terminations and rewards
        s.episode_length.add_(1)
        for run in self._terminations:
            run()
        torch.any(s.termination_values & ~self._time_out, dim=0, out=o.terminated)
        torch.any(s.termination_values & self._time_out, dim=0, out=o.truncated)
        torch.logical_or(o.terminated, o.truncated, out=o.reset_mask)
        o.reward.zero_()
        for k, run in enumerate(self._rewards):
            run()
            weighted = s.reward_values[k] * s.reward_weights[k] * plan.step_dt
            s.episode_sums[k] += weighted
            o.reward.add_(weighted)
        if plan.compute_final_observations:
            self._observe_into("final")
        # resets
        self._reset_block(o.reset_mask)
        commit = o.reset_mask & self._reset_writes
        # commands
        for c, term in enumerate(plan.commands):
            remaining = s.command_time_left[c] - plan.step_dt
            expired = remaining <= 0.0
            value = draw(s.rng, expired, *term.cfg.resampling_time_range)
            s.command_time_left[c] = torch.where(expired, value, remaining)
            self._commands[c][0](expired)
            self._commands[c][1]()
        # interval events
        for e, term in enumerate(plan.interval_events):
            remaining = s.interval_time_left[e] - plan.step_dt
            fired = remaining < 1.0e-6
            value = draw(s.rng, fired, *term.cfg.interval_range_s)
            s.interval_time_left[e] = torch.where(fired, value, remaining)
            s.interval_fired[e] = fired
            self._interval_events[e](fired)
            if term.spec.writes:
                commit = commit | fired
        s.commit_mask.copy_(commit)
        physics.reset(self._wp_reset_mask)
        if program.events_write_physics:
            physics.commit(self._wp_commit_mask)
        self._observe_into("obs")

    def reset(self) -> None:
        program = self.program
        physics = program.plan.physics
        self._reset_block(program.state.reset_request)
        physics.reset(self._wp_reset_request)
        if program.events_write_physics:
            physics.commit(self._wp_reset_request)
        self._observe_into("obs")
