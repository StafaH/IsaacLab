# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Term specifications and backend implementation registry.

A term has one backend-independent :class:`TermSpec` (kind, parameters, fields it reads and writes, output
width) and one implementation per backend. An implementation is a *binder*: it receives a
:class:`TermContext` with the arrays it may touch and returns a zero-argument callable that enqueues the
term's work. Binders run once, before capture; the returned callables run every step.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

BACKENDS = ("warp", "torch")
"""Supported backend names."""

REQUIRED = object()
"""Sentinel default marking a required term parameter."""


class Stage(IntEnum):
    """Execution stages of one step, in order. Physics runs between ``ACTION`` and ``TERMINATION``."""

    ACTION = 0
    TERMINATION = 1
    REWARD = 2
    EVENT = 3
    OBSERVATION = 4


RUNTIME_FIELDS: Mapping[str, Stage] = {
    "action": Stage.ACTION,
    "prev_action": Stage.ACTION,
    "episode_length": Stage.ACTION,
    "terminated": Stage.TERMINATION,
    "truncated": Stage.TERMINATION,
}
"""Runtime-owned fields readable by terms, mapped to the stage that produces them.

A term may read a runtime field only if its own stage comes strictly after the producing stage.
``episode_length`` is incremented right after physics, so every post-physics stage sees the new value.
"""


@dataclass(frozen=True)
class TermSpec:
    """Backend-independent semantics of a term."""

    name: str
    """Registered name used by :attr:`TermCfg.term`."""

    stage: Stage
    """Kind of the term. Events use :attr:`Stage.EVENT` for both reset and interval mode."""

    params: Mapping[str, Any]
    """Parameter names mapped to defaults, or :data:`REQUIRED`."""

    reads: tuple[str, ...]
    """Physics or runtime fields the term reads."""

    writes: tuple[str, ...]
    """Physics fields the term writes. Only action and event terms may write."""

    width: Callable[[Mapping[str, Any]], int] | None
    """Output columns for observation terms, or consumed action columns for action terms, from the resolved
    parameters. Rewards and terminations produce one value per environment."""

    doc: str
    """One-line description."""


@dataclass(frozen=True)
class TermContext:
    """Everything a binder may use. Arrays are native to the backend (``wp.array`` or ``torch.Tensor``)."""

    params: Mapping[str, Any]
    """Validated parameters. ``joints`` is resolved into the tuple ``joint_ids``."""

    fields: Mapping[str, Any]
    """Only the fields declared in :attr:`TermSpec.reads` and :attr:`TermSpec.writes`."""

    out: Any
    """Output view: ``(N, width)`` for observations, ``(N,)`` float32 for rewards, ``(N,)`` bool for
    terminations, ``None`` otherwise."""

    action: Any
    """Processed action columns ``(N, width)`` of an action term, else ``None``."""

    mask: Any
    """Environment mask ``(N,)`` bool of an event term, else ``None``."""

    rng: Any
    """Per-environment random stream state ``(N,)`` of an event term, else ``None``."""

    indices: Mapping[str, Any]
    """Device index arrays prepared from the parameters, e.g. ``joint_ids``."""

    num_envs: int
    step_dt: float
    max_episode_length: int
    device: str


_SPECS: dict[str, TermSpec] = {}
_IMPLS: dict[tuple[str, str], Callable[[TermContext], Callable[[], None]]] = {}


def define_term(
    name: str,
    stage: Stage,
    *,
    params: Mapping[str, Any] | None = None,
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
    width: Callable[[Mapping[str, Any]], int] | None = None,
    doc: str = "",
) -> TermSpec:
    """Register the backend-independent specification of a term.

    Raises:
        ValueError: If the name is already registered or the declaration is inconsistent.
    """
    if name in _SPECS:
        raise ValueError(f"Term '{name}' is already defined.")
    if writes and stage not in (Stage.ACTION, Stage.EVENT):
        raise ValueError(f"Term '{name}': only action and event terms may write fields.")
    if (width is None) != (stage not in (Stage.ACTION, Stage.OBSERVATION)):
        raise ValueError(f"Term '{name}': action and observation terms, and only those, declare a width.")
    spec = TermSpec(name, stage, dict(params or {}), tuple(reads), tuple(writes), width, doc)
    _SPECS[name] = spec
    return spec


def implement(name: str, backend: str):
    """Decorator registering the binder of a defined term for one backend."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Expected one of {BACKENDS}.")

    def decorator(binder: Callable[[TermContext], Callable[[], None]]):
        if name not in _SPECS:
            raise ValueError(f"Term '{name}' must be defined before it is implemented.")
        if (name, backend) in _IMPLS:
            raise ValueError(f"Term '{name}' already has a {backend} implementation.")
        _IMPLS[(name, backend)] = binder
        return binder

    return decorator


def get_spec(name: str) -> TermSpec | None:
    """Return the specification of a registered term, or None."""
    return _SPECS.get(name)


def get_impl(name: str, backend: str) -> Callable[[TermContext], Callable[[], None]] | None:
    """Return the binder of a term for a backend, or None."""
    return _IMPLS.get((name, backend))


def registered_terms() -> dict[str, tuple[str, ...]]:
    """Map every registered term to the backends that implement it."""
    return {name: tuple(b for b in BACKENDS if (name, b) in _IMPLS) for name in _SPECS}
