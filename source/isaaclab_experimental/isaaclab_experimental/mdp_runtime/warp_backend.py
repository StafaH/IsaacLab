# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp backend: buffers are ``wp.array`` and the step is three generated, fused kernels.

Every array a term may touch is a member of one generated :func:`warp.struct` (the program's *field struct*).
Term implementations return per-environment :func:`warp.func` objects with their parameters baked in as
constants. The executor inlines them, with the runtime's own work, into:

* ``pre``: action processing and action terms;
* ``post``: episode length, terminations, rewards, final observations, resets, commands, interval events;
* ``observe``: observations.

The physics step runs between ``pre`` and ``post``; the physics reset and commit run between ``post`` and
``observe``. Each kernel is recorded once with ``record_cmd=True``, so a step replays a fixed launch list.
"""

# No ``from __future__ import annotations``: Warp resolves kernel annotations, and the generated kernels are
# annotated with the closure variable ``fields_type``.
import contextlib
import hashlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from .terms import TermContext, get_impl
from .wp_math import rng_next, rng_uniform

if TYPE_CHECKING:
    from .plan import MdpProgram, ResolvedTerm

_DTYPES = {"float32": wp.float32, "int32": wp.int32, "bool": wp.bool, "rng": wp.uint32, "index": wp.int32}
_BIG = 3.0e38
"""Finite stand-in for an absent clip bound (float32 max is about 3.4e38)."""


@wp.kernel
def _seed(rng: wp.array(dtype=wp.uint32), seed: wp.uint32):
    i = wp.tid()
    rng[i] = rng_next(rng_next(wp.uint32(i) * wp.uint32(2654435761) + seed))


@wp.kernel
def _pack_rows(src: wp.array2d(dtype=wp.float32), offset: wp.int32, pad: wp.float32, dst: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    dst[offset + i, j] = wp.where(j < src.shape[1], src[i, wp.min(j, src.shape[1] - 1)], pad)


class WarpBackend:
    """Allocation helpers and executor factory for Warp programs."""

    name = "warp"

    def __init__(self, device: str):
        self.device = device

    def zeros(self, shape: tuple[int, ...], dtype: str) -> wp.array:
        return wp.zeros(shape, dtype=_DTYPES[dtype], device=self.device)

    def constant(self, values: Sequence[float], dtype: str) -> wp.array:
        return wp.array(np.asarray(values), dtype=_DTYPES[dtype], device=self.device)

    def adopt(self, array: wp.array) -> wp.array:
        """Return a physics field in this backend's array type (zero-copy)."""
        return array

    def to_warp(self, array: wp.array) -> wp.array:
        return array

    def to_numpy(self, array: wp.array) -> np.ndarray:
        return array.numpy()

    def rows(self, array: wp.array, start: int, stop: int) -> wp.array:
        return array[start:stop]

    def columns(self, array: wp.array, start: int, stop: int) -> wp.array:
        return array[:, start:stop]

    def copy(self, dst: wp.array, src: wp.array) -> None:
        wp.copy(dst, src)

    def stream_scope(self):
        return contextlib.nullcontext()

    def is_capturing(self) -> bool:
        device = wp.get_device(self.device)
        return device.is_cuda and device.stream.is_capturing

    def seed(self, rng: wp.array, seed: int) -> None:
        wp.launch(_seed, dim=rng.shape[0], inputs=[rng, wp.uint32(seed & 0xFFFFFFFF)], device=self.device)

    def assign_mask(self, mask: wp.array, env_ids: Sequence[int] | None) -> None:
        values = np.ones(mask.shape[0], dtype=bool)
        if env_ids is not None:
            values[:] = False
            values[np.asarray(env_ids, dtype=np.int64)] = True
        mask.assign(values)

    def bind_pack_rows(self, src, dst, offset, pad) -> Callable[[], None]:
        dim = (src.shape[0], dst.shape[1])
        return wp.launch(
            _pack_rows, dim=dim, inputs=[src, offset, pad, dst], device=self.device, record_cmd=True
        ).launch

    def build_executor(self, program: "MdpProgram") -> "WarpExecutor":
        return WarpExecutor(program)


# -- code generation ---------------------------------------------------------------------------------------


def _field_struct(arrays: dict[str, wp.array]):
    """Generate the field struct type; its name is derived from the member types so kernels cache on disk.

    Returns:
        The struct type, an instance holding the arrays, and the layout key.
    """
    annotations = {name: wp.array(dtype=a.dtype, ndim=a.ndim) for name, a in arrays.items()}
    key = ",".join(f"{n}:{wp.types.type_repr(a.dtype)}:{a.ndim}" for n, a in arrays.items())
    name = "MdpFields_" + hashlib.sha256(key.encode()).hexdigest()[:16]
    struct_type = wp.struct(type(name, (), {"__annotations__": annotations}))
    instance = struct_type()
    for member, array in arrays.items():
        setattr(instance, member, array)
    return struct_type, instance, key


def _module_name(plan, layout_key: str) -> str:
    """Name the Warp module of a program's kernels after everything baked into them.

    Term functions capture parameters, columns, and flags as compile-time constants through ``wp.static``. Warp's
    module hash does not cover those values in nested functions, so kernels of programs that differ only in
    constants would share a cache entry. A module per program signature keeps them apart, while identical
    programs still reuse the kernel cache.
    """
    info = plan.info
    parts = [
        layout_key,
        repr((info.step_dt, info.physics_dt, info.max_episode_length, info.num_actions)),
        repr((sorted(info.command_columns.items()), info.termination_names, plan.compute_final_observations)),
        repr(sorted(plan.observation_columns.items())),
    ]
    for term in plan.terms:
        params = sorted((k, repr(v)) for k, v in term.params.items())
        parts.append(repr((term.path, term.spec.name, params, term.columns, term.state_columns, repr(term.cfg))))
    return "isaaclab_mdp_runtime_" + hashlib.sha256("\n".join(parts).encode()).hexdigest()[:24]


def _make_observe(terms: "list[ResolvedTerm]", funcs: list):
    """Generate ``observe(env, f, out, s) -> s``: observation terms, then noise, clip, and scale per term."""
    count = len(terms)
    columns = [t.columns for t in terms]
    noise = [t.cfg.noise for t in terms]
    has_noise = [n is not None for n in noise]
    noise_lo = [float(n[0]) if n is not None else 0.0 for n in noise]
    noise_hi = [float(n[1]) if n is not None else 0.0 for n in noise]
    has_post = [t.cfg.clip is not None or t.cfg.scale != 1.0 for t in terms]
    clip_lo = [float(t.cfg.clip[0]) if t.cfg.clip is not None else -_BIG for t in terms]
    clip_hi = [float(t.cfg.clip[1]) if t.cfg.clip is not None else _BIG for t in terms]
    scale = [float(t.cfg.scale) for t in terms]

    @wp.func
    def observe(env: int, f: Any, out: wp.array2d(dtype=wp.float32), s: wp.uint32) -> wp.uint32:
        for i in range(wp.static(count)):
            wp.static(funcs[i])(env, f, out)
            if wp.static(has_noise[i]):
                for c in range(wp.static(columns[i][0]), wp.static(columns[i][1])):
                    s = rng_next(s)
                    out[env, c] = out[env, c] + rng_uniform(s, wp.static(noise_lo[i]), wp.static(noise_hi[i]))
            if wp.static(has_post[i]):
                for c in range(wp.static(columns[i][0]), wp.static(columns[i][1])):
                    value = wp.clamp(out[env, c], wp.static(clip_lo[i]), wp.static(clip_hi[i]))
                    out[env, c] = value * wp.static(scale[i])
        return s

    return observe


class WarpExecutor:
    """Generates and records the fused kernels of one program."""

    def __init__(self, program: "MdpProgram"):
        self.program = program
        plan, s, o = program.plan, program.state, program.outputs
        runtime = {
            "actions_in": program.inputs.actions,
            "action": s.action,
            "prev_action": s.prev_action,
            "processed_actions": s.processed_actions,
            "episode_length": s.episode_length,
            "rng": s.rng,
            "commands": s.commands,
            "command_state": s.command_state,
            "command_time_left": s.command_time_left,
            "termination_values": s.termination_values,
            "reward_values": s.reward_values,
            "episode_sums": s.episode_sums,
            "reward_weights": s.reward_weights,
            "interval_time_left": s.interval_time_left,
            "interval_fired": s.interval_fired,
            "reset_request": s.reset_request,
            "commit_mask": s.commit_mask,
            "observations": o.observation_buffer,
            # Without final observations the member aliases the observation buffer and is never written.
            "final_observations": o.final_observation_buffer
            if o.final_observation_buffer is not None
            else o.observation_buffer,
            "reward": o.reward,
            "terminated": o.terminated,
            "truncated": o.truncated,
            "reset_mask": o.reset_mask,
        }
        collisions = sorted(set(runtime) & set(plan.physics.fields))
        if collisions:
            raise ValueError(f"Physics fields {collisions} collide with runtime field names.")
        self.fields_type, self.fields, layout_key = _field_struct({**plan.physics.fields, **runtime})
        self.module = _module_name(plan, layout_key)

        def build(term: "ResolvedTerm"):
            ctx = TermContext(
                params=term.params, columns=term.columns, state_columns=term.state_columns, info=plan.info
            )
            return get_impl(term.spec.name, "warp")(ctx)

        observation_terms = [t for terms in plan.observations.values() for t in terms]
        observe = _make_observe(observation_terms, [build(t) for t in observation_terms])
        commands = [build(t) for t in plan.commands]
        reset_env = self._make_reset(plan, [build(t) for t in plan.reset_events], commands)
        reset_writes = any(t.spec.writes for t in plan.reset_events)
        device, n, fields = plan.device, plan.num_envs, self.fields

        def record(kernel):
            return wp.launch(kernel, dim=n, inputs=[fields], device=device, record_cmd=True).launch

        self._pre = record(self._make_pre(plan, [build(t) for t in plan.actions]))
        self._post = record(
            self._make_post(
                plan,
                [build(t) for t in plan.terminations],
                [build(t) for t in plan.rewards],
                commands,
                [build(t) for t in plan.interval_events],
                observe,
                reset_env,
                reset_writes,
            )
        )
        self._observe = record(self._make_observe_kernel(observe))
        self._reset = record(self._make_reset_kernel(reset_env))

    # -- execution -----------------------------------------------------------------------------------------

    def step(self, include_physics: bool) -> None:
        program = self.program
        physics = program.plan.physics
        self._pre()
        if include_physics:
            physics.step()
        self._post()
        physics.reset(program.outputs.reset_mask)
        if program.events_write_physics:
            physics.commit(program.state.commit_mask)
        self._observe()

    def reset(self) -> None:
        program = self.program
        physics = program.plan.physics
        self._reset()
        physics.reset(program.state.reset_request)
        if program.events_write_physics:
            physics.commit(program.state.reset_request)
        self._observe()

    # -- kernels -------------------------------------------------------------------------------------------

    def _make_pre(self, plan, action_funcs: list):
        fields_type = self.fields_type
        scale, offset, lo, hi = [], [], [], []
        for term in plan.actions:
            clip = term.cfg.clip or (-_BIG, _BIG)
            scale += [float(term.cfg.scale)] * term.width
            offset += [float(term.cfg.offset)] * term.width
            lo += [float(clip[0])] * term.width
            hi += [float(clip[1])] * term.width
        vec = wp.types.vector(length=len(scale), dtype=wp.float32)
        scale, offset, lo, hi = vec(*scale), vec(*offset), vec(*lo), vec(*hi)
        num_actions = plan.num_actions

        @wp.kernel(module=self.module)
        def pre(f: fields_type):
            env = wp.tid()
            for c in range(num_actions):
                raw = f.actions_in[env, c]
                f.prev_action[env, c] = f.action[env, c]
                f.action[env, c] = raw
                f.processed_actions[env, c] = wp.clamp(raw * scale[c] + offset[c], lo[c], hi[c])
            for i in range(wp.static(len(action_funcs))):
                wp.static(action_funcs[i])(env, f)

        return pre

    @staticmethod
    def _make_reset(plan, event_funcs: list, command_funcs: list):
        """Generate ``reset_env(env, f, s) -> s`` for one resetting environment."""
        interval_lo = [float(t.cfg.interval_range_s[0]) for t in plan.interval_events]
        interval_hi = [float(t.cfg.interval_range_s[1]) for t in plan.interval_events]
        command_lo = [float(t.cfg.resampling_time_range[0]) for t in plan.commands]
        command_hi = [float(t.cfg.resampling_time_range[1]) for t in plan.commands]
        resample = [pair[0] for pair in command_funcs]

        @wp.func
        def reset_env(env: int, f: Any, s: wp.uint32) -> wp.uint32:
            for i in range(wp.static(len(event_funcs))):
                s = wp.static(event_funcs[i])(env, f, s)
            f.episode_length[env] = 0
            for c in range(f.action.shape[1]):
                f.action[env, c] = 0.0
                f.prev_action[env, c] = 0.0
            for k in range(f.episode_sums.shape[0]):
                f.episode_sums[k, env] = 0.0
            for e in range(wp.static(len(interval_lo))):
                s = rng_next(s)
                f.interval_time_left[e, env] = rng_uniform(s, wp.static(interval_lo[e]), wp.static(interval_hi[e]))
            for c in range(wp.static(len(resample))):
                s = rng_next(s)
                f.command_time_left[c, env] = rng_uniform(s, wp.static(command_lo[c]), wp.static(command_hi[c]))
                s = wp.static(resample[c])(env, f, s)
            return s

        return reset_env

    def _make_post(
        self, plan, termination_funcs, reward_funcs, command_funcs, interval_funcs, observe, reset_env, reset_writes
    ):
        fields_type = self.fields_type
        dt = float(plan.step_dt)
        time_out = [bool(t.cfg.time_out) for t in plan.terminations]
        final = plan.compute_final_observations
        resample = [pair[0] for pair in command_funcs]
        update = [pair[1] for pair in command_funcs]
        command_lo = [float(t.cfg.resampling_time_range[0]) for t in plan.commands]
        command_hi = [float(t.cfg.resampling_time_range[1]) for t in plan.commands]
        interval_lo = [float(t.cfg.interval_range_s[0]) for t in plan.interval_events]
        interval_hi = [float(t.cfg.interval_range_s[1]) for t in plan.interval_events]
        interval_writes = [bool(t.spec.writes) for t in plan.interval_events]

        @wp.kernel(module=self.module)
        def post(f: fields_type):
            env = wp.tid()
            s = f.rng[env]
            f.episode_length[env] = f.episode_length[env] + 1
            # terminations
            terminated = bool(False)
            truncated = bool(False)
            for i in range(wp.static(len(termination_funcs))):
                flag = wp.static(termination_funcs[i])(env, f)
                f.termination_values[i, env] = flag
                if flag:
                    if wp.static(time_out[i]):
                        truncated = True
                    else:
                        terminated = True
            reset = terminated or truncated
            f.terminated[env] = terminated
            f.truncated[env] = truncated
            f.reset_mask[env] = reset
            # rewards
            total = float(0.0)
            for i in range(wp.static(len(reward_funcs))):
                value = wp.static(reward_funcs[i])(env, f)
                f.reward_values[i, env] = value
                weighted = value * f.reward_weights[i] * wp.static(dt)
                f.episode_sums[i, env] = f.episode_sums[i, env] + weighted
                total = total + weighted
            f.reward[env] = total
            if wp.static(final):
                s = observe(env, f, f.final_observations, s)
            # resets
            commit = bool(False)
            if reset:
                s = reset_env(env, f, s)
                commit = wp.static(reset_writes)
            # commands
            for c in range(wp.static(len(resample))):
                remaining = f.command_time_left[c, env] - wp.static(dt)
                if remaining <= 0.0:
                    s = rng_next(s)
                    remaining = rng_uniform(s, wp.static(command_lo[c]), wp.static(command_hi[c]))
                    s = wp.static(resample[c])(env, f, s)
                f.command_time_left[c, env] = remaining
                wp.static(update[c])(env, f)
            # interval events
            for e in range(wp.static(len(interval_funcs))):
                remaining = f.interval_time_left[e, env] - wp.static(dt)
                fired = remaining < 1.0e-6
                if fired:
                    s = rng_next(s)
                    remaining = rng_uniform(s, wp.static(interval_lo[e]), wp.static(interval_hi[e]))
                    s = wp.static(interval_funcs[e])(env, f, s)
                    commit = commit or wp.static(interval_writes[e])
                f.interval_time_left[e, env] = remaining
                f.interval_fired[e, env] = fired
            f.commit_mask[env] = commit
            f.rng[env] = s

        return post

    def _make_observe_kernel(self, observe):
        fields_type = self.fields_type

        @wp.kernel(module=self.module)
        def observe_all(f: fields_type):
            env = wp.tid()
            f.rng[env] = observe(env, f, f.observations, f.rng[env])

        return observe_all

    def _make_reset_kernel(self, reset_env):
        fields_type = self.fields_type

        @wp.kernel(module=self.module)
        def reset_requested(f: fields_type):
            env = wp.tid()
            if f.reset_request[env]:
                f.rng[env] = reset_env(env, f, f.rng[env])

        return reset_requested
