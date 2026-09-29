# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp implementation of the runtime-owned operations.

Every ``bind_*`` method validates nothing and allocates nothing per step: it records one launch with
``record_cmd=True`` and returns its ``launch`` method, so the step replays a fixed launch list.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence

import numpy as np
import warp as wp

_DTYPES = {"float32": wp.float32, "int32": wp.int32, "bool": wp.bool, "rng": wp.uint32, "index": wp.int32}


@wp.func
def rng_next(state: wp.uint32) -> wp.uint32:
    """Advance a PCG32 stream state."""
    return state * wp.uint32(747796405) + wp.uint32(2891336453)


@wp.func
def rng_uniform(state: wp.uint32, lo: wp.float32, hi: wp.float32) -> wp.float32:
    """Map an advanced PCG32 state to ``[lo, hi)`` with 24 random bits (bitwise reproducible in Torch)."""
    word = ((state >> ((state >> wp.uint32(28)) + wp.uint32(4))) ^ state) * wp.uint32(277803737)
    word = (word >> wp.uint32(22)) ^ word
    u = wp.float32(word >> wp.uint32(8)) * wp.float32(5.9604644775390625e-08)
    return lo + (hi - lo) * u


@wp.kernel
def _seed(rng: wp.array(dtype=wp.uint32), seed: wp.uint32):
    i = wp.tid()
    rng[i] = rng_next(rng_next(wp.uint32(i) * wp.uint32(2654435761) + seed))


@wp.kernel
def _process_actions(
    raw: wp.array2d(dtype=wp.float32),
    scale: wp.array(dtype=wp.float32),
    offset: wp.array(dtype=wp.float32),
    lo: wp.array(dtype=wp.float32),
    hi: wp.array(dtype=wp.float32),
    action: wp.array2d(dtype=wp.float32),
    prev_action: wp.array2d(dtype=wp.float32),
    processed: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    value = raw[i, j]
    prev_action[i, j] = action[i, j]
    action[i, j] = value
    processed[i, j] = wp.clamp(value * scale[j] + offset[j], lo[j], hi[j])


@wp.kernel
def _increment(episode_length: wp.array(dtype=wp.int32)):
    i = wp.tid()
    episode_length[i] = episode_length[i] + 1


@wp.kernel
def _reduce_terminations(
    values: wp.array2d(dtype=wp.bool),
    time_out: wp.array(dtype=wp.bool),
    terminated: wp.array(dtype=wp.bool),
    truncated: wp.array(dtype=wp.bool),
    reset_mask: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    term = bool(False)
    trunc = bool(False)
    for k in range(values.shape[0]):
        if values[k, i]:
            if time_out[k]:
                trunc = True
            else:
                term = True
    terminated[i] = term
    truncated[i] = trunc
    reset_mask[i] = term or trunc


@wp.kernel
def _reduce_rewards(
    values: wp.array2d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    dt: wp.float32,
    reward: wp.array(dtype=wp.float32),
    episode_sums: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    total = wp.float32(0.0)
    for k in range(values.shape[0]):
        value = values[k, i] * weights[k] * dt
        episode_sums[k, i] = episode_sums[k, i] + value
        total = total + value
    reward[i] = total


@wp.kernel
def _post_observations(
    obs: wp.array2d(dtype=wp.float32),
    lo: wp.array(dtype=wp.float32),
    hi: wp.array(dtype=wp.float32),
    scale: wp.array(dtype=wp.float32),
):
    i, j = wp.tid()
    obs[i, j] = wp.clamp(obs[i, j], lo[j], hi[j]) * scale[j]


@wp.kernel
def _reset_state(
    mask: wp.array(dtype=wp.bool),
    episode_length: wp.array(dtype=wp.int32),
    action: wp.array2d(dtype=wp.float32),
    prev_action: wp.array2d(dtype=wp.float32),
    episode_sums: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    if mask[i]:
        episode_length[i] = 0
        for j in range(action.shape[1]):
            action[i, j] = 0.0
            prev_action[i, j] = 0.0
        for k in range(episode_sums.shape[0]):
            episode_sums[k, i] = 0.0


@wp.kernel
def _sample_timers(
    mask: wp.array(dtype=wp.bool),
    rng: wp.array(dtype=wp.uint32),
    time_left: wp.array(dtype=wp.float32),
    lo: wp.float32,
    hi: wp.float32,
):
    i = wp.tid()
    if mask[i]:
        state = rng_next(rng[i])
        time_left[i] = rng_uniform(state, lo, hi)
        rng[i] = state


@wp.kernel
def _tick_timers(
    time_left: wp.array(dtype=wp.float32),
    dt: wp.float32,
    rng: wp.array(dtype=wp.uint32),
    lo: wp.float32,
    hi: wp.float32,
    fired: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    remaining = time_left[i] - dt
    fire = remaining < 1.0e-6
    if fire:
        state = rng_next(rng[i])
        remaining = rng_uniform(state, lo, hi)
        rng[i] = state
    time_left[i] = remaining
    fired[i] = fire


@wp.kernel
def _pack_rows(src: wp.array2d(dtype=wp.float32), offset: wp.int32, pad: wp.float32, dst: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    dst[offset + i, j] = wp.where(j < src.shape[1], src[i, wp.min(j, src.shape[1] - 1)], pad)


def _record(kernel, dim, inputs, device) -> Callable[[], None]:
    return wp.launch(kernel, dim=dim, inputs=inputs, device=device, record_cmd=True).launch


class WarpBackend:
    """Allocates ``wp.array`` buffers and binds runtime operations as recorded Warp launches."""

    name = "warp"

    def __init__(self, device: str):
        self.device = device

    # -- allocation ----------------------------------------------------------------------------------

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

    def stream_scope(self):
        return contextlib.nullcontext()

    def is_capturing(self) -> bool:
        device = wp.get_device(self.device)
        return device.is_cuda and device.stream.is_capturing

    # -- host-side operations (outside capture) --------------------------------------------------------

    def seed(self, rng: wp.array, seed: int) -> None:
        wp.launch(_seed, dim=rng.shape[0], inputs=[rng, wp.uint32(seed & 0xFFFFFFFF)], device=self.device)

    def assign_mask(self, mask: wp.array, env_ids: Sequence[int] | None) -> None:
        values = np.ones(mask.shape[0], dtype=bool)
        if env_ids is not None:
            values[:] = False
            values[np.asarray(env_ids, dtype=np.int64)] = True
        mask.assign(values)

    # -- bound operations ----------------------------------------------------------------------------

    def bind_copy(self, dst: wp.array, src: wp.array) -> Callable[[], None]:
        return lambda: wp.copy(dst, src)

    def bind_process_actions(self, raw, scale, offset, lo, hi, action, prev_action, processed):
        return _record(
            _process_actions, raw.shape, [raw, scale, offset, lo, hi, action, prev_action, processed], self.device
        )

    def bind_increment(self, episode_length):
        return _record(_increment, episode_length.shape[0], [episode_length], self.device)

    def bind_reduce_terminations(self, values, time_out, terminated, truncated, reset_mask):
        return _record(
            _reduce_terminations,
            terminated.shape[0],
            [values, time_out, terminated, truncated, reset_mask],
            self.device,
        )

    def bind_reduce_rewards(self, values, weights, dt, reward, episode_sums):
        return _record(_reduce_rewards, reward.shape[0], [values, weights, dt, reward, episode_sums], self.device)

    def bind_post_observations(self, obs, lo, hi, scale):
        return _record(_post_observations, obs.shape, [obs, lo, hi, scale], self.device)

    def bind_reset_state(self, mask, episode_length, action, prev_action, episode_sums):
        return _record(
            _reset_state, mask.shape[0], [mask, episode_length, action, prev_action, episode_sums], self.device
        )

    def bind_sample_timers(self, mask, rng, time_left, lo, hi):
        return _record(_sample_timers, mask.shape[0], [mask, rng, time_left, lo, hi], self.device)

    def bind_tick_timers(self, time_left, dt, rng, lo, hi, fired):
        return _record(_tick_timers, fired.shape[0], [time_left, dt, rng, lo, hi, fired], self.device)

    def bind_pack_rows(self, src, dst, offset, pad):
        return _record(_pack_rows, (src.shape[0], dst.shape[1]), [src, offset, pad, dst], self.device)
