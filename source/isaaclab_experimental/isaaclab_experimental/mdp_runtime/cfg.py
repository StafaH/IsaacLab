# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-independent configuration of an MDP specification.

A configuration names registered terms and their parameters. It never refers to a Warp kernel or a
Torch function: :func:`~isaaclab_experimental.mdp_runtime.compile_plan` selects the implementation for
the requested backend and validates the configuration against the physics binding.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any, Literal

from isaaclab.utils import configclass


@configclass
class TermCfg:
    """Declaration of one term: a registered term name and its parameters."""

    term: str = MISSING
    """Name of a term registered with :func:`~isaaclab_experimental.mdp_runtime.define_term`."""

    params: dict[str, Any] = {}
    """Keyword parameters of the term. Every key must be declared by the term specification.

    The ``joints`` parameter is special: it holds joint-name regular expressions that are resolved
    against the physics binding's joint names at compile time.
    """


@configclass
class ActionTermCfg(TermCfg):
    """Action term. The runtime owns the shared processing ``clamp(raw * scale + offset, *clip)``."""

    scale: float = 1.0
    """Multiplier applied to the raw action columns of this term."""

    offset: float = 0.0
    """Offset added after scaling."""

    clip: tuple[float, float] | None = None
    """Bounds applied to the processed action. None disables clipping."""


@configclass
class ObservationTermCfg(TermCfg):
    """Observation term. The runtime owns the shared post-processing ``clamp(value, *clip) * scale``."""

    scale: float = 1.0
    """Multiplier applied after clipping."""

    clip: tuple[float, float] | None = None
    """Bounds applied to the raw term value. None disables clipping."""


@configclass
class ObservationGroupCfg:
    """An ordered set of observation terms concatenated along the last dimension."""

    terms: dict[str, ObservationTermCfg] = {}
    """Terms in column order."""


@configclass
class RewardTermCfg(TermCfg):
    """Reward term. The step reward is ``sum(weight * value) * step_dt`` in declaration order."""

    weight: float = MISSING
    """Weight of the term. It is stored on the device and can be changed without recapturing."""


@configclass
class TerminationTermCfg(TermCfg):
    """Termination term."""

    time_out: bool = False
    """Whether the term reports truncation (time limit) instead of a terminal state."""


@configclass
class EventTermCfg(TermCfg):
    """Event term applied to a capture-safe environment mask."""

    mode: Literal["reset", "interval"] = "reset"
    """``"reset"`` runs for every environment that resets. ``"interval"`` runs per environment when its
    timer expires; the timer is resampled from :attr:`interval_range_s` on expiry and on reset."""

    interval_range_s: tuple[float, float] | None = None
    """Interval bounds [s]. Required for, and only allowed with, ``mode="interval"``."""


@configclass
class MdpCfg:
    """An MDP specification. Dictionary order is execution and column order."""

    episode_length_s: float = MISSING
    """Episode duration [s]. The maximum episode length is ``ceil(episode_length_s / step_dt)`` steps."""

    actions: dict[str, ActionTermCfg] = {}
    """Action terms, in action-column order."""

    observations: dict[str, ObservationGroupCfg] = {}
    """Observation groups. Each group owns one ``(num_envs, width)`` output buffer."""

    rewards: dict[str, RewardTermCfg] = {}
    """Reward terms, summed in declaration order."""

    terminations: dict[str, TerminationTermCfg] = {}
    """Termination terms."""

    events: dict[str, EventTermCfg] = {}
    """Reset and interval events, run in declaration order within each mode."""

    compute_final_observations: bool = True
    """Whether the step also writes pre-reset observations to the ``final_observations`` outputs."""

    seed: int = 0
    """Seed of the per-environment random streams used by events and interval timers."""
