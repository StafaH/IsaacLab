# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Term specifications and backend implementation registry.

A term has one backend-independent :class:`TermSpec` (kind, parameters, fields it reads and writes, output
width) and one implementation per backend:

* **Warp:** a factory ``factory(ctx) -> wp.func`` evaluated once at bind time. The returned per-environment
  function is inlined into the runtime's fused stage kernels, with parameters baked in as compile-time
  constants. Signatures by stage (``f`` is the program's field struct):

  ========================= =========================================================================
  Stage                     Function signature
  ========================= =========================================================================
  action                    ``(env: int, f: Any)`` -- reads ``f.processed_actions`` columns, writes fields
  observation               ``(env: int, f: Any, out: wp.array2d(dtype=float))`` -- writes its columns
  reward                    ``(env: int, f: Any) -> float``
  termination               ``(env: int, f: Any) -> bool``
  event                     ``(env: int, f: Any, state: wp.uint32) -> wp.uint32`` -- returns the stream
  command                   a pair ``(resample(env, f, state) -> wp.uint32, update(env, f))``
  ========================= =========================================================================

* **Torch:** a binder ``binder(ctx) -> Callable`` evaluated once at bind time. It returns a vectorized callable
  over all environments. Event callables take the environment mask: ``run(mask)``. Command binders return
  ``(resample(mask), update())``. Masked callables must leave unmasked environments and their random streams
  unchanged.
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
    """Term kinds, in step order. Physics runs between ``ACTION`` and ``TERMINATION``."""

    ACTION = 0
    TERMINATION = 1
    REWARD = 2
    EVENT = 3
    COMMAND = 4
    OBSERVATION = 5


STATE = -1
"""Producer marker for runtime fields that persist across steps and are readable in every stage."""

RUNTIME_FIELDS: Mapping[str, int] = {
    "action": Stage.ACTION,
    "prev_action": Stage.ACTION,
    "episode_length": Stage.ACTION,
    "terminated": Stage.TERMINATION,
    "truncated": Stage.TERMINATION,
    "termination_values": Stage.TERMINATION,
    "commands": STATE,
}
"""Runtime-owned fields readable by terms, mapped to the stage that produces them.

A term may read a runtime field only if its stage comes strictly after the producing stage. ``episode_length``
is incremented right after physics. ``commands`` is state: terms before the command stage read the command of
the previous step, as in the stable ``ManagerBasedRLEnv``; only command terms write it.
"""


@dataclass(frozen=True)
class CompileInfo:
    """Plan-wide constants available to width functions and implementations."""

    num_envs: int
    num_actions: int
    step_dt: float
    physics_dt: float
    max_episode_length: int
    command_columns: Mapping[str, tuple[int, int]]
    termination_names: tuple[str, ...]
    device: str


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

    width: Callable[[Mapping[str, Any], CompileInfo], int] | None
    """Columns of observation, action, and command terms, from the resolved parameters. Rewards and terminations
    produce one value per environment."""

    state_width: int
    """Per-environment private state columns of a command term."""

    doc: str
    """One-line description."""


@dataclass(frozen=True)
class TermContext:
    """What an implementation receives at bind time.

    Both backends receive the resolved parameters and the layout constants. Torch binders also receive
    arrays: only the declared fields, plus the views listed below.
    """

    params: Mapping[str, Any]
    """Validated parameters. Name parameters are resolved: ``joint_ids``, ``body_ids``, ``contact_ids``,
    ``command_columns``, ``term_ids``."""

    columns: tuple[int, int] | None
    """Column range of an action, observation, or command term."""

    state_columns: tuple[int, int] | None
    """Column range of a command term's private state."""

    info: CompileInfo

    # Torch only --------------------------------------------------------------------------------------------
    fields: Mapping[str, Any] | None = None
    """Declared fields as tensors."""
    out: Any = None
    """Output view: ``(N, width)`` observation or command columns, processed action columns of an action term,
    or the ``(N,)`` value row of a reward or termination."""
    state: Any = None
    """Command private state view ``(N, state_width)``."""
    rng: Any = None
    """Per-environment random stream state."""
    indices: Mapping[str, Any] | None = None
    """Device index tensors of the ``*_ids`` parameters."""


_SPECS: dict[str, TermSpec] = {}
_IMPLS: dict[tuple[str, str], Callable[[TermContext], Any]] = {}


def define_term(
    name: str,
    stage: Stage,
    *,
    params: Mapping[str, Any] | None = None,
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
    width: Callable[[Mapping[str, Any], CompileInfo], int] | int | None = None,
    state_width: int = 0,
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
    if (width is None) != (stage not in (Stage.ACTION, Stage.OBSERVATION, Stage.COMMAND)):
        raise ValueError(f"Term '{name}': action, observation, and command terms, and only those, declare a width.")
    if state_width and stage != Stage.COMMAND:
        raise ValueError(f"Term '{name}': only command terms declare private state.")
    if isinstance(width, int):
        constant = width
        width = lambda params, info: constant  # noqa: E731
    spec = TermSpec(name, stage, dict(params or {}), tuple(reads), tuple(writes), width, state_width, doc)
    _SPECS[name] = spec
    return spec


def implement(name: str, backend: str):
    """Decorator registering the implementation of a defined term for one backend."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Expected one of {BACKENDS}.")

    def decorator(impl: Callable[[TermContext], Any]):
        if name not in _SPECS:
            raise ValueError(f"Term '{name}' must be defined before it is implemented.")
        if (name, backend) in _IMPLS:
            raise ValueError(f"Term '{name}' already has a {backend} implementation.")
        _IMPLS[(name, backend)] = impl
        return impl

    return decorator


def get_spec(name: str) -> TermSpec | None:
    """Return the specification of a registered term, or None."""
    return _SPECS.get(name)


def get_impl(name: str, backend: str) -> Callable[[TermContext], Any] | None:
    """Return the implementation of a term for a backend, or None."""
    return _IMPLS.get((name, backend))


def registered_terms() -> dict[str, tuple[str, ...]]:
    """Map every registered term to the backends that implement it."""
    return {name: tuple(b for b in BACKENDS if (name, b) in _IMPLS) for name in _SPECS}
