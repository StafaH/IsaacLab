# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch implementation of the runtime-owned operations.

The operations mirror :mod:`~isaaclab_experimental.mdp_runtime.warp_backend` in arithmetic order. They use
masked updates instead of index selection, so no operation synchronizes with the host. Random stream states
are stored as ``int64`` holding unsigned 32-bit values, so the PCG32 stream matches Warp bit for bit.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence

import numpy as np
import torch
import warp as wp

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


class TorchBackend:
    """Allocates ``torch.Tensor`` buffers and binds runtime operations as tensor functions.

    Physics fields are zero-copy views of the binding's Warp arrays. Work is enqueued on the current Torch
    stream, and Warp physics launches are redirected to that stream (:meth:`stream_scope`), so a step can be
    captured with :class:`torch.cuda.CUDAGraph`.
    """

    name = "torch"

    def __init__(self, device: str):
        self.device = device

    # -- allocation ----------------------------------------------------------------------------------

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

    def stream_scope(self):
        if torch.device(self.device).type != "cuda":
            return contextlib.nullcontext()
        stream = torch.cuda.current_stream(self.device)
        # Synchronizing with the previous Warp stream is required eagerly but invalid during capture.
        sync = not torch.cuda.is_current_stream_capturing()
        return wp.ScopedStream(warp_stream(stream), sync_enter=sync, sync_exit=sync)

    def is_capturing(self) -> bool:
        return torch.device(self.device).type == "cuda" and torch.cuda.is_current_stream_capturing()

    # -- host-side operations (outside capture) --------------------------------------------------------

    def seed(self, rng: torch.Tensor, seed: int) -> None:
        ids = torch.arange(rng.shape[0], dtype=torch.int64, device=self.device)
        rng.copy_(rng_next(rng_next((ids * 2654435761 + (seed & _MASK32)) & _MASK32)))

    def assign_mask(self, mask: torch.Tensor, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            mask.fill_(True)
            return
        mask.fill_(False)
        mask[torch.as_tensor(env_ids, dtype=torch.long, device=self.device)] = True

    # -- bound operations ----------------------------------------------------------------------------

    def bind_copy(self, dst, src) -> Callable[[], None]:
        return lambda: dst.copy_(src)

    def bind_process_actions(self, raw, scale, offset, lo, hi, action, prev_action, processed):
        def run():
            prev_action.copy_(action)
            action.copy_(raw)
            torch.clamp(raw * scale + offset, lo, hi, out=processed)

        return run

    def bind_increment(self, episode_length):
        return lambda: episode_length.add_(1)

    def bind_reduce_terminations(self, values, time_out, terminated, truncated, reset_mask):
        timeout_rows = time_out[:, None]

        def run():
            torch.any(values & ~timeout_rows, dim=0, out=terminated)
            torch.any(values & timeout_rows, dim=0, out=truncated)
            torch.logical_or(terminated, truncated, out=reset_mask)

        return run

    def bind_reduce_rewards(self, values, weights, dt, reward, episode_sums):
        def run():
            reward.zero_()
            for k in range(values.shape[0]):
                value = values[k] * weights[k] * dt
                episode_sums[k] += value
                reward.add_(value)

        return run

    def bind_post_observations(self, obs, lo, hi, scale):
        return lambda: torch.mul(torch.clamp(obs, lo, hi), scale, out=obs)

    def bind_reset_state(self, mask, episode_length, action, prev_action, episode_sums):
        rows = mask[:, None]

        def run():
            episode_length.masked_fill_(mask, 0)
            action.masked_fill_(rows, 0.0)
            prev_action.masked_fill_(rows, 0.0)
            episode_sums.masked_fill_(mask, 0.0)

        return run

    def bind_sample_timers(self, mask, rng, time_left, lo, hi):
        def run():
            state = rng_next(rng)
            time_left.copy_(torch.where(mask, rng_uniform(state, lo, hi), time_left))
            rng.copy_(torch.where(mask, state, rng))

        return run

    def bind_tick_timers(self, time_left, dt, rng, lo, hi, fired):
        def run():
            remaining = time_left - dt
            torch.lt(remaining, 1.0e-6, out=fired)
            state = rng_next(rng)
            time_left.copy_(torch.where(fired, rng_uniform(state, lo, hi), remaining))
            rng.copy_(torch.where(fired, state, rng))

        return run

    def bind_pack_rows(self, src, dst, offset, pad):
        rows, width = src.shape
        target, padding = dst[offset : offset + rows, :width], dst[offset : offset + rows, width:]

        def run():
            target.copy_(src)
            padding.fill_(pad)

        return run
