# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration compilation, buffer allocation, and program binding.

The lifecycle has three explicit phases:

1. :func:`compile_plan` validates an :class:`~isaaclab_experimental.mdp_runtime.MdpCfg` against a physics
   binding and a backend, and resolves it into an immutable :class:`ExecutionPlan` (no device memory).
2. :meth:`ExecutionPlan.allocate` creates the input, state, and output buffers.
3. :meth:`ExecutionPlan.bind` builds the backend executor for those buffers and returns an :class:`MdpProgram`.
   The Warp executor generates three fused kernels; the Torch executor records a list of tensor functions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, dataclass, field
from typing import Any

import torch
import warp as wp

from isaaclab.utils.string import resolve_matching_names

from . import builtin_terms  # noqa: F401  (registers the built-in terms)
from .cfg import (
    ActionTermCfg,
    CommandTermCfg,
    EventTermCfg,
    MdpCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TermCfg,
    TerminationTermCfg,
)
from .physics import PhysicsBinding
from .terms import BACKENDS, REQUIRED, RUNTIME_FIELDS, STATE, CompileInfo, Stage, TermSpec, get_impl, get_spec
from .torch_backend import TorchBackend, warp_stream
from .warp_backend import WarpBackend


class MdpConfigError(ValueError):
    """Raised by :func:`compile_plan` with every configuration problem found, not only the first."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("Invalid MDP configuration:\n" + "\n".join(f"  - {e}" for e in self.errors))


@dataclass(frozen=True)
class ResolvedTerm:
    """A validated term with resolved parameters and its columns."""

    path: str
    """Configuration path, e.g. ``"rewards.alive"``. Used in errors and the schedule."""
    name: str
    spec: TermSpec
    cfg: TermCfg
    params: Mapping[str, Any]
    columns: tuple[int, int] | None = None
    """Column range in the action, observation, or command buffer."""
    state_columns: tuple[int, int] | None = None
    """Column range of a command term's private state."""

    @property
    def width(self) -> int:
        return self.columns[1] - self.columns[0]


@dataclass
class MdpInputs:
    """Buffers written by the caller before a step."""

    actions: Any
    """Raw actions ``(N, num_actions)`` float32."""


@dataclass
class MdpState:
    """Runtime-owned state that persists across steps."""

    action: Any
    """Raw action of the current step ``(N, A)``."""
    prev_action: Any
    """Raw action of the previous step ``(N, A)``."""
    processed_actions: Any
    """``clamp(raw * scale + offset)`` per column ``(N, A)``."""
    episode_length: Any
    """Steps since the last reset ``(N,)`` int32."""
    rng: Any
    """Per-environment PCG32 stream ``(N,)``: ``uint32`` (Warp) or ``int64`` holding ``uint32`` (Torch)."""
    commands: Any
    """Command values ``(N, C)``; each command term owns a column range."""
    command_state: Any
    """Private command state ``(N, S)``."""
    command_time_left: Any
    """Time until each command resamples ``(num_commands, N)`` [s]."""
    termination_values: Any
    """Per-term termination flags ``(T, N)`` bool."""
    reward_values: Any
    """Per-term unweighted reward values ``(K, N)`` float32."""
    episode_sums: Any
    """Per-term weighted episode return ``(K, N)`` float32."""
    reward_weights: Any
    """Reward weights ``(K,)`` float32 on the device."""
    interval_time_left: Any
    """Time until each interval event fires ``(E, N)`` [s]."""
    interval_fired: Any
    """Interval events that fired this step ``(E, N)`` bool."""
    reset_request: Any
    """Mask used by :meth:`MdpProgram.reset` ``(N,)`` bool."""
    commit_mask: Any
    """Environments whose physics state events changed this step ``(N,)`` bool."""


@dataclass
class MdpOutputs:
    """Buffers written by a step."""

    observation_buffer: Any
    """All observation groups side by side ``(N, W)`` float32."""
    final_observation_buffer: Any
    """Pre-reset observations ``(N, W)``, or None unless ``compute_final_observations``."""
    observations: dict[str, Any]
    """Post-reset observations per group: column views of :attr:`observation_buffer`."""
    final_observations: dict[str, Any]
    """Pre-reset observations per group: column views of :attr:`final_observation_buffer`."""
    reward: Any
    """``(N,)`` float32."""
    terminated: Any
    """Terminal state reached ``(N,)`` bool."""
    truncated: Any
    """Time limit reached ``(N,)`` bool. Both flags may be set."""
    reset_mask: Any
    """``terminated | truncated``: environments reset during this step ``(N,)`` bool."""


@dataclass(frozen=True)
class ExecutionPlan:
    """A validated, resolved MDP. It owns no device memory and can be bound to several buffer sets."""

    backend: str
    physics: PhysicsBinding = field(repr=False)
    info: CompileInfo
    actions: tuple[ResolvedTerm, ...]
    commands: tuple[ResolvedTerm, ...]
    observations: Mapping[str, tuple[ResolvedTerm, ...]]
    observation_columns: Mapping[str, tuple[int, int]]
    terminations: tuple[ResolvedTerm, ...]
    rewards: tuple[ResolvedTerm, ...]
    reset_events: tuple[ResolvedTerm, ...]
    interval_events: tuple[ResolvedTerm, ...]
    compute_final_observations: bool
    seed: int

    @property
    def num_envs(self) -> int:
        return self.info.num_envs

    @property
    def num_actions(self) -> int:
        return self.info.num_actions

    @property
    def step_dt(self) -> float:
        return self.info.step_dt

    @property
    def max_episode_length(self) -> int:
        return self.info.max_episode_length

    @property
    def device(self) -> str:
        return self.info.device

    @property
    def observation_widths(self) -> dict[str, int]:
        return {g: stop - start for g, (start, stop) in self.observation_columns.items()}

    @property
    def observation_width(self) -> int:
        return max((stop for _, stop in self.observation_columns.values()), default=0)

    @property
    def num_command_columns(self) -> int:
        return self.commands[-1].columns[1] if self.commands else 0

    @property
    def num_command_state(self) -> int:
        return self.commands[-1].state_columns[1] if self.commands else 0

    @property
    def terms(self) -> tuple[ResolvedTerm, ...]:
        """Every term of the plan."""
        groups = tuple(t for terms in self.observations.values() for t in terms)
        return (
            self.actions
            + self.commands
            + groups
            + self.terminations
            + self.rewards
            + self.reset_events
            + self.interval_events
        )

    def make_backend(self) -> WarpBackend | TorchBackend:
        return (WarpBackend if self.backend == "warp" else TorchBackend)(self.device)

    def allocate_inputs(self) -> MdpInputs:
        return MdpInputs(actions=self.make_backend().zeros((self.num_envs, self.num_actions), "float32"))

    def allocate_state(self) -> MdpState:
        be, n, a = self.make_backend(), self.num_envs, self.num_actions
        k, t, e = len(self.rewards), len(self.terminations), len(self.interval_events)
        state = MdpState(
            action=be.zeros((n, a), "float32"),
            prev_action=be.zeros((n, a), "float32"),
            processed_actions=be.zeros((n, a), "float32"),
            episode_length=be.zeros((n,), "int32"),
            rng=be.zeros((n,), "rng"),
            commands=be.zeros((n, self.num_command_columns), "float32"),
            command_state=be.zeros((n, self.num_command_state), "float32"),
            command_time_left=be.zeros((len(self.commands), n), "float32"),
            termination_values=be.zeros((t, n), "bool"),
            reward_values=be.zeros((k, n), "float32"),
            episode_sums=be.zeros((k, n), "float32"),
            reward_weights=be.constant([term.cfg.weight for term in self.rewards], "float32"),
            interval_time_left=be.zeros((e, n), "float32"),
            interval_fired=be.zeros((e, n), "bool"),
            reset_request=be.zeros((n,), "bool"),
            commit_mask=be.zeros((n,), "bool"),
        )
        be.seed(state.rng, self.seed)
        return state

    def allocate_outputs(self, **shared: Any) -> MdpOutputs:
        """Allocate outputs. ``reward``, ``terminated``, ``truncated``, ``reset_mask`` may be passed as
        caller-owned ``(N,)`` views, e.g. slices of population-wide buffers."""
        be, n, w = self.make_backend(), self.num_envs, self.observation_width
        unknown = set(shared) - {"reward", "terminated", "truncated", "reset_mask"}
        if unknown:
            raise ValueError(f"Unknown shared outputs: {sorted(unknown)}")
        obs = be.zeros((n, w), "float32")
        final = be.zeros((n, w), "float32") if self.compute_final_observations else None
        return MdpOutputs(
            observation_buffer=obs,
            final_observation_buffer=final,
            observations={g: be.columns(obs, *c) for g, c in self.observation_columns.items()},
            final_observations={g: be.columns(final, *c) for g, c in self.observation_columns.items()}
            if final is not None
            else {},
            reward=shared["reward"] if "reward" in shared else be.zeros((n,), "float32"),
            terminated=shared["terminated"] if "terminated" in shared else be.zeros((n,), "bool"),
            truncated=shared["truncated"] if "truncated" in shared else be.zeros((n,), "bool"),
            reset_mask=shared["reset_mask"] if "reset_mask" in shared else be.zeros((n,), "bool"),
        )

    def allocate(self) -> tuple[MdpInputs, MdpState, MdpOutputs]:
        return self.allocate_inputs(), self.allocate_state(), self.allocate_outputs()

    def bind(self, inputs: MdpInputs, state: MdpState, outputs: MdpOutputs) -> MdpProgram:
        """Build the executor for the given buffers. Buffers must outlive the program."""
        return MdpProgram(self, inputs, state, outputs)


# -- compilation ---------------------------------------------------------------------------------------


def _check_range(path: str, name: str, value: Any, errors: list[str]) -> None:
    if not (isinstance(value, Sequence) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)):
        errors.append(f"{path}: '{name}' must be a (lower, upper) pair, got {value!r}.")
    elif not value[0] <= value[1]:
        errors.append(f"{path}: '{name}' lower bound {value[0]} exceeds upper bound {value[1]}.")


def _resolve_names(path: str, params: dict, physics: PhysicsBinding, info: CompileInfo, stage: Stage, errors):
    """Resolve the name parameters documented in :class:`~isaaclab_experimental.mdp_runtime.TermCfg`."""
    for key, target, names in (
        ("joints", "joint_ids", physics.joint_names),
        ("bodies", "body_ids", physics.body_names),
        ("contact_bodies", "contact_ids", physics.contact_body_names),
    ):
        if key not in params:
            continue
        try:
            ids, _ = resolve_matching_names(params[key], names, preserve_order=True)
            params[target] = tuple(ids)
        except ValueError as e:
            errors.append(f"{path}: {e}")
    if "command" in params:
        if params["command"] not in info.command_columns:
            errors.append(f"{path}: unknown command '{params['command']}'. Defined: {sorted(info.command_columns)}.")
        else:
            params["command_columns"] = info.command_columns[params["command"]]
    if "terms" in params:
        if stage != Stage.REWARD:
            errors.append(f"{path}: the 'terms' parameter is only resolved for reward terms.")
        try:
            ids, _ = resolve_matching_names(params["terms"], info.termination_names, preserve_order=True)
            params["term_ids"] = tuple(ids)
        except ValueError as e:
            errors.append(f"{path}: {e}")


def _resolve_term(
    path: str, name: str, cfg: TermCfg, stage: Stage, physics: PhysicsBinding, info: CompileInfo, backend, errors
) -> ResolvedTerm | None:
    if cfg.term is MISSING or not isinstance(cfg.term, str):
        errors.append(f"{path}: 'term' must name a registered term.")
        return None
    spec = get_spec(cfg.term)
    if spec is None:
        errors.append(f"{path}: unknown term '{cfg.term}'.")
        return None
    count = len(errors)
    if spec.stage != stage:
        errors.append(f"{path}: term '{spec.name}' is a {spec.stage.name.lower()} term, not {stage.name.lower()}.")
    if get_impl(spec.name, backend) is None:
        errors.append(f"{path}: term '{spec.name}' has no {backend} implementation.")
    unknown = sorted(set(cfg.params) - set(spec.params))
    if unknown:
        errors.append(f"{path}: unknown parameters {unknown} for term '{spec.name}'.")
    missing = sorted(k for k, v in spec.params.items() if v is REQUIRED and k not in cfg.params)
    if missing:
        errors.append(f"{path}: missing required parameters {missing} for term '{spec.name}'.")
    params = {k: v for k, v in spec.params.items() if v is not REQUIRED}
    params.update(cfg.params)
    for key, value in params.items():
        if key.endswith("_range") or key == "bounds":
            if isinstance(value, Mapping):
                for axis, bounds in value.items():
                    _check_range(path, f"{key}.{axis}", bounds, errors)
            else:
                _check_range(path, key, value, errors)
    _resolve_names(path, params, physics, info, stage, errors)
    for field_name in spec.reads:
        producer = RUNTIME_FIELDS.get(field_name)
        if producer is not None and producer != STATE and producer >= stage:
            errors.append(
                f"{path}: reads '{field_name}', which the {Stage(producer).name.lower()} stage produces; only later"
                " stages may read it."
            )
        elif producer is None and field_name not in physics.fields:
            errors.append(f"{path}: reads '{field_name}', which the physics binding does not provide.")
    for field_name in spec.writes:
        if field_name not in physics.fields:
            errors.append(f"{path}: writes '{field_name}', which the physics binding does not provide.")
    if len(errors) > count:
        return None
    return ResolvedTerm(path, name, spec, cfg, params)


def _with_columns(terms: list[ResolvedTerm], info: CompileInfo, errors: list[str], start: int = 0):
    result, state_start = [], 0
    for term in terms:
        width = term.spec.width(term.params, info)
        if width <= 0:
            errors.append(f"{term.path}: term '{term.spec.name}' has width {width}; it must be positive.")
        state = (state_start, state_start + term.spec.state_width)
        result.append(
            ResolvedTerm(term.path, term.name, term.spec, term.cfg, term.params, (start, start + width), state)
        )
        start += width
        state_start = state[1]
    return tuple(result), start


def compile_plan(cfg: MdpCfg, physics: PhysicsBinding, backend: str = "warp") -> ExecutionPlan:
    """Validate a configuration against a physics binding and resolve it for one backend.

    Raises:
        MdpConfigError: With every problem found.
    """
    if backend not in BACKENDS:
        raise MdpConfigError([f"unknown backend '{backend}'; expected one of {BACKENDS}."])
    errors: list[str] = []
    if physics.num_envs <= 0:
        errors.append(f"physics binding has {physics.num_envs} environments.")
    episode_ok = isinstance(cfg.episode_length_s, (int, float)) and cfg.episode_length_s > 0
    if not episode_ok:
        errors.append(f"episode_length_s must be positive, got {cfg.episode_length_s!r}.")

    def make_info(num_actions: int, command_columns: Mapping[str, tuple[int, int]]) -> CompileInfo:
        return CompileInfo(
            num_envs=physics.num_envs,
            num_actions=num_actions,
            step_dt=physics.step_dt,
            physics_dt=physics.physics_dt,
            max_episode_length=math.ceil(cfg.episode_length_s / physics.step_dt) if episode_ok else 0,
            command_columns=command_columns,
            termination_names=tuple(cfg.terminations),
            device=physics.device,
        )

    info = make_info(0, {})

    def resolve(section: str, terms: Mapping[str, TermCfg], stage: Stage, cfg_type: type) -> list[ResolvedTerm]:
        out = []
        for name, term_cfg in terms.items():
            path = f"{section}.{name}"
            if not isinstance(term_cfg, cfg_type):
                errors.append(f"{path}: expected {cfg_type.__name__}, got {type(term_cfg).__name__}.")
                continue
            resolved = _resolve_term(path, name, term_cfg, stage, physics, info, backend, errors)
            if resolved is not None:
                out.append(resolved)
        return out

    # Actions and commands are resolved first: their widths define the layout that later terms read.
    if not cfg.actions:
        errors.append("actions: at least one action term is required.")
    actions = resolve("actions", cfg.actions, Stage.ACTION, ActionTermCfg)
    for term in actions:
        if term.cfg.clip is not None:
            _check_range(term.path, "clip", term.cfg.clip, errors)
    actions, num_actions = _with_columns(actions, info, errors)
    commands = resolve("commands", cfg.commands, Stage.COMMAND, CommandTermCfg)
    for term in commands:
        _check_range(term.path, "resampling_time_range", term.cfg.resampling_time_range, errors)
    commands, _ = _with_columns(commands, info, errors)
    for term in actions + commands:
        if "command" in term.params:
            errors.append(f"{term.path}: action and command terms cannot read commands.")
    info = make_info(num_actions, {t.name: t.columns for t in commands})

    if not cfg.observations:
        errors.append("observations: at least one observation group is required.")
    observations, observation_columns, start = {}, {}, 0
    for group, group_cfg in cfg.observations.items():
        if not group_cfg.terms:
            errors.append(f"observations.{group}: the group has no terms.")
        terms = resolve(f"observations.{group}", group_cfg.terms, Stage.OBSERVATION, ObservationTermCfg)
        for term in terms:
            if term.cfg.clip is not None:
                _check_range(term.path, "clip", term.cfg.clip, errors)
            if term.cfg.noise is not None:
                _check_range(term.path, "noise", term.cfg.noise, errors)
        observations[group], stop = _with_columns(terms, info, errors, start)
        observation_columns[group] = (start, stop)
        start = stop

    terminations = resolve("terminations", cfg.terminations, Stage.TERMINATION, TerminationTermCfg)
    rewards = resolve("rewards", cfg.rewards, Stage.REWARD, RewardTermCfg)
    for term in rewards:
        if not (isinstance(term.cfg.weight, (int, float)) and math.isfinite(term.cfg.weight)):
            errors.append(f"{term.path}: weight must be a finite number, got {term.cfg.weight!r}.")
    events = resolve("events", cfg.events, Stage.EVENT, EventTermCfg)
    for term in events:
        if term.cfg.mode not in ("reset", "interval"):
            errors.append(f"{term.path}: mode must be 'reset' or 'interval', got {term.cfg.mode!r}.")
        elif term.cfg.mode == "interval":
            if term.cfg.interval_range_s is None:
                errors.append(f"{term.path}: interval events require interval_range_s.")
            else:
                _check_range(term.path, "interval_range_s", term.cfg.interval_range_s, errors)
                if term.cfg.interval_range_s[1] <= 0:
                    errors.append(f"{term.path}: interval_range_s must have a positive upper bound.")
        elif term.cfg.interval_range_s is not None:
            errors.append(f"{term.path}: interval_range_s is only valid for mode='interval'.")

    if errors:
        raise MdpConfigError(errors)
    return ExecutionPlan(
        backend=backend,
        physics=physics,
        info=info,
        actions=actions,
        commands=commands,
        observations=observations,
        observation_columns=observation_columns,
        terminations=tuple(terminations),
        rewards=tuple(rewards),
        reset_events=tuple(t for t in events if t.cfg.mode == "reset"),
        interval_events=tuple(t for t in events if t.cfg.mode == "interval"),
        compute_final_observations=cfg.compute_final_observations,
        seed=cfg.seed,
    )


# -- programs ------------------------------------------------------------------------------------------


def _check_shape(name: str, array: Any, shape: tuple[int, ...]) -> None:
    if tuple(array.shape) != shape:
        raise ValueError(f"Buffer '{name}' has shape {tuple(array.shape)}, expected {shape}.")


class MdpProgram:
    """An execution plan bound to fixed buffers.

    :meth:`step` runs the work listed by :attr:`schedule`, in this order:

    1. action processing and action terms;
    2. the physics step (skipped with ``include_physics=False``);
    3. episode-length increment, terminations, rewards, and, with ``compute_final_observations``, pre-reset
       observations;
    4. for resetting environments: reset events, runtime-state reset, interval timers and commands resampled;
    5. command timers and updates, then interval events;
    6. physics reset of the resetting environments, and commit of every environment events changed;
    7. observations.

    On the Warp backend, 1, 3-5, and 7 are one generated kernel each. Every argument is fixed when the program
    is created, so a step can be captured into a CUDA graph and replayed.
    """

    def __init__(self, plan: ExecutionPlan, inputs: MdpInputs, state: MdpState, outputs: MdpOutputs):
        self.plan = plan
        self.inputs = inputs
        self.state = state
        self.outputs = outputs
        self.backend = plan.make_backend()
        self._validate_buffers()
        self._host_weights = [float(term.cfg.weight) for term in plan.rewards]
        self._reward_index = {term.name: k for k, term in enumerate(plan.rewards)}
        reads = {f for t in plan.terms for f in t.spec.reads if f in plan.physics.fields}
        writes = {f for t in plan.terms for f in t.spec.writes}
        plan.physics.prepare(reads, writes)
        self.events_write_physics = any(t.spec.writes for t in plan.reset_events + plan.interval_events)
        self._executor = self.backend.build_executor(self)

    @property
    def schedule(self) -> tuple[str, ...]:
        """The logical order of work in one step (identical on both backends)."""
        p = self.plan
        names = ["action.process", *(t.path for t in p.actions), "physics.step", "episode_length"]
        names += [t.path for t in p.terminations] + [t.path for t in p.rewards]
        observe = [t.path for terms in p.observations.values() for t in terms]
        if p.compute_final_observations:
            names += [f"final.{n}" for n in observe]
        names += [f"reset.{t.path}" for t in p.reset_events] + ["reset.runtime_state"]
        names += [f"reset.timer.{t.path}" for t in p.interval_events] + [f"reset.{t.path}" for t in p.commands]
        names += [t.path for t in p.commands] + [t.path for t in p.interval_events]
        names += ["physics.reset", "physics.commit", *observe]
        return tuple(names)

    def step(self, include_physics: bool = True) -> None:
        """Enqueue one control step reading :attr:`inputs` and writing :attr:`outputs`. Never synchronizes."""
        with self.backend.stream_scope():
            self._executor.step(include_physics)

    def reset(self, env_ids: Sequence[int] | None = None, mask: Any = None) -> None:
        """Reset environments and recompute their observations.

        Args:
            env_ids: Environment indices, converted to a mask on the host. Not allowed during capture.
            mask: A ``(N,)`` bool array of this program's backend, copied on the device (capture-safe).
                If neither is given, all environments reset.
        """
        be = self.backend
        if mask is not None:
            if env_ids is not None:
                raise ValueError("Pass either env_ids or mask, not both.")
            _check_shape("mask", mask, (self.plan.num_envs,))
        elif be.is_capturing():
            raise RuntimeError("reset(env_ids) converts indices on the host; pass a device mask during capture.")
        else:
            be.assign_mask(self.state.reset_request, env_ids)
        with be.stream_scope():
            if mask is not None:
                be.copy(self.state.reset_request, mask)
            self._executor.reset()

    def set_reward_weight(self, name: str, weight: float) -> None:
        """Change a reward weight in device memory. Captured graphs see the new value on the next replay."""
        if self.backend.is_capturing():
            raise RuntimeError("Reward weights are host-written; change them outside capture.")
        if not math.isfinite(weight):
            raise ValueError(f"Reward weight must be finite, got {weight}.")
        self._host_weights[self._reward_index[name]] = float(weight)
        self.backend.copy(self.state.reward_weights, self.backend.constant(self._host_weights, "float32"))

    def capture(self, include_physics: bool = True, warmup: bool = True) -> CapturedStep:
        """Capture one :meth:`step` into a CUDA graph.

        Warp programs use :class:`warp.ScopedCapture`; Torch programs use :class:`torch.cuda.CUDAGraph` with
        Warp registered as an external capture. With ``warmup`` one eager step runs first to compile kernels
        and settle lazy allocations; it advances the environment state.
        """
        return capture_step(self.backend, lambda: self.step(include_physics), warmup=warmup, owner=self)

    def _validate_buffers(self) -> None:
        p, n, a = self.plan, self.plan.num_envs, self.plan.num_actions
        k, t, e = len(p.rewards), len(p.terminations), len(p.interval_events)
        s, o = self.state, self.outputs
        for name, array, shape in (
            ("inputs.actions", self.inputs.actions, (n, a)),
            ("state.action", s.action, (n, a)),
            ("state.prev_action", s.prev_action, (n, a)),
            ("state.processed_actions", s.processed_actions, (n, a)),
            ("state.episode_length", s.episode_length, (n,)),
            ("state.rng", s.rng, (n,)),
            ("state.commands", s.commands, (n, p.num_command_columns)),
            ("state.command_state", s.command_state, (n, p.num_command_state)),
            ("state.command_time_left", s.command_time_left, (len(p.commands), n)),
            ("state.termination_values", s.termination_values, (t, n)),
            ("state.reward_values", s.reward_values, (k, n)),
            ("state.episode_sums", s.episode_sums, (k, n)),
            ("state.reward_weights", s.reward_weights, (k,)),
            ("state.interval_time_left", s.interval_time_left, (e, n)),
            ("state.interval_fired", s.interval_fired, (e, n)),
            ("state.reset_request", s.reset_request, (n,)),
            ("state.commit_mask", s.commit_mask, (n,)),
            ("outputs.observation_buffer", o.observation_buffer, (n, p.observation_width)),
            ("outputs.reward", o.reward, (n,)),
            ("outputs.terminated", o.terminated, (n,)),
            ("outputs.truncated", o.truncated, (n,)),
            ("outputs.reset_mask", o.reset_mask, (n,)),
        ):
            _check_shape(name, array, shape)
        if p.compute_final_observations:
            _check_shape("outputs.final_observation_buffer", o.final_observation_buffer, (n, p.observation_width))


# -- capture -------------------------------------------------------------------------------------------


class CapturedStep:
    """A CUDA graph of one step. It references its owner so the captured buffers stay alive.

    The graph is valid while the owner's buffers and the physics binding's arrays keep their addresses.
    Configuration or buffer changes require a new program and a new capture; reward weights may change.
    """

    def __init__(self, graph: Any, replay: Callable[[], None], owner: Any):
        self.graph = graph
        self.owner = owner
        self._replay = replay

    def replay(self) -> None:
        """Enqueue the graph on the current stream without synchronizing."""
        self._replay()


def capture_step(backend: Any, step: Callable[[], None], warmup: bool = True, owner: Any = None) -> CapturedStep:
    """Capture ``step`` into a CUDA graph using the backend's capture mechanism."""
    device = backend.device
    if torch.device(device).type != "cuda":
        raise RuntimeError(f"CUDA graph capture requires a CUDA device, got '{device}'.")
    if warmup:
        step()
    if backend.name == "warp":
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            step()
        return CapturedStep(capture.graph, lambda: wp.capture_launch(capture.graph), owner)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Register the Torch capture with Warp so capture-aware Warp code (e.g. conditional graph nodes in the
        # physics solver) records instead of synchronizing. torch.cuda.graph uses the global capture mode.
        stream = warp_stream(torch.cuda.current_stream(device))
        wp.capture_begin(device, stream=stream, external=True, capture_mode=wp.CaptureMode.GLOBAL)
        try:
            step()
        finally:
            wp.capture_end(device, stream=stream)
    return CapturedStep(graph, graph.replay, owner)
