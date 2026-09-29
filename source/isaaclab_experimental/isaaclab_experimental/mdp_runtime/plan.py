# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration compilation, buffer allocation, and the fixed per-step operation list.

The lifecycle has three explicit phases:

1. :func:`compile_plan` validates an :class:`~isaaclab_experimental.mdp_runtime.MdpCfg` against a physics
   binding and a backend, and resolves it into an immutable :class:`ExecutionPlan` (no device memory).
2. :meth:`ExecutionPlan.allocate` creates the input, state, and output buffers.
3. :meth:`ExecutionPlan.bind` records every operation against those buffers and returns an
   :class:`MdpProgram` whose step replays the same operation list with the same arguments every time.
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
from .cfg import ActionTermCfg, EventTermCfg, MdpCfg, ObservationTermCfg, RewardTermCfg, TermCfg, TerminationTermCfg
from .physics import PhysicsBinding
from .terms import BACKENDS, REQUIRED, RUNTIME_FIELDS, Stage, TermContext, TermSpec, get_impl, get_spec
from .torch_backend import TorchBackend, warp_stream
from .warp_backend import WarpBackend


class MdpConfigError(ValueError):
    """Raised by :func:`compile_plan` with every configuration problem found, not only the first."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("Invalid MDP configuration:\n" + "\n".join(f"  - {e}" for e in self.errors))


@dataclass(frozen=True)
class ResolvedTerm:
    """A validated term with resolved parameters and its output columns."""

    path: str
    """Configuration path, e.g. ``"rewards.alive"``. Used in operation names and errors."""
    name: str
    spec: TermSpec
    cfg: TermCfg
    params: Mapping[str, Any]
    columns: tuple[int, int] | None
    """Column range in the action or observation-group buffer, else None."""


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
    termination_values: Any
    """Per-term termination flags ``(T, N)`` bool."""
    reward_values: Any
    """Per-term unweighted reward values ``(K, N)`` float32."""
    episode_sums: Any
    """Per-term weighted episode return ``(K, N)`` float32."""
    reward_weights: Any
    """Reward weights ``(K,)`` float32 on the device."""
    interval_time_left: Any
    """Time until each interval event fires ``(E, N)`` float32 [s]."""
    interval_fired: Any
    """Interval events that fired this step ``(E, N)`` bool."""
    reset_request: Any
    """Mask used by :meth:`MdpProgram.reset` ``(N,)`` bool."""


@dataclass
class MdpOutputs:
    """Buffers written by a step."""

    observations: dict[str, Any]
    """Post-reset observations per group ``(N, width)`` float32."""
    final_observations: dict[str, Any]
    """Pre-reset observations per group, empty unless ``compute_final_observations``."""
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
    num_envs: int
    step_dt: float
    max_episode_length: int
    num_actions: int
    actions: tuple[ResolvedTerm, ...]
    observations: Mapping[str, tuple[ResolvedTerm, ...]]
    observation_widths: Mapping[str, int]
    terminations: tuple[ResolvedTerm, ...]
    rewards: tuple[ResolvedTerm, ...]
    reset_events: tuple[ResolvedTerm, ...]
    interval_events: tuple[ResolvedTerm, ...]
    compute_final_observations: bool
    seed: int

    @property
    def device(self) -> str:
        return self.physics.device

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
            termination_values=be.zeros((t, n), "bool"),
            reward_values=be.zeros((k, n), "float32"),
            episode_sums=be.zeros((k, n), "float32"),
            reward_weights=be.constant([term.cfg.weight for term in self.rewards], "float32"),
            interval_time_left=be.zeros((e, n), "float32"),
            interval_fired=be.zeros((e, n), "bool"),
            reset_request=be.zeros((n,), "bool"),
        )
        be.seed(state.rng, self.seed)
        return state

    def allocate_outputs(self, **shared: Any) -> MdpOutputs:
        """Allocate outputs. ``reward``, ``terminated``, ``truncated``, ``reset_mask`` may be passed as
        caller-owned ``(N,)`` views, e.g. slices of population-wide buffers."""
        be, n = self.make_backend(), self.num_envs
        unknown = set(shared) - {"reward", "terminated", "truncated", "reset_mask"}
        if unknown:
            raise ValueError(f"Unknown shared outputs: {sorted(unknown)}")
        obs = {g: be.zeros((n, w), "float32") for g, w in self.observation_widths.items()}
        final = (
            {g: be.zeros((n, w), "float32") for g, w in self.observation_widths.items()}
            if self.compute_final_observations
            else {}
        )
        return MdpOutputs(
            observations=obs,
            final_observations=final,
            reward=shared["reward"] if "reward" in shared else be.zeros((n,), "float32"),
            terminated=shared["terminated"] if "terminated" in shared else be.zeros((n,), "bool"),
            truncated=shared["truncated"] if "truncated" in shared else be.zeros((n,), "bool"),
            reset_mask=shared["reset_mask"] if "reset_mask" in shared else be.zeros((n,), "bool"),
        )

    def allocate(self) -> tuple[MdpInputs, MdpState, MdpOutputs]:
        return self.allocate_inputs(), self.allocate_state(), self.allocate_outputs()

    def bind(self, inputs: MdpInputs, state: MdpState, outputs: MdpOutputs) -> MdpProgram:
        """Record every operation against the given buffers. Buffers must outlive the program."""
        return MdpProgram(self, inputs, state, outputs)


# -- compilation ---------------------------------------------------------------------------------------


def _check_range(path: str, name: str, value: Any, errors: list[str]) -> None:
    if not (isinstance(value, Sequence) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)):
        errors.append(f"{path}: '{name}' must be a (lower, upper) pair, got {value!r}.")
    elif not value[0] <= value[1]:
        errors.append(f"{path}: '{name}' lower bound {value[0]} exceeds upper bound {value[1]}.")


def _resolve_term(
    path: str, name: str, cfg: TermCfg, stage: Stage, physics: PhysicsBinding, backend: str, errors: list[str]
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
            _check_range(path, key, value, errors)
    if "joints" in spec.params and "joints" in params:
        try:
            ids, _ = resolve_matching_names(params["joints"], physics.joint_names, preserve_order=True)
            params["joint_ids"] = tuple(ids)
        except ValueError as e:
            errors.append(f"{path}: {e}")
    for field_name in spec.reads:
        producer = RUNTIME_FIELDS.get(field_name)
        if producer is not None and producer >= stage:
            errors.append(
                f"{path}: reads '{field_name}', which the {producer.name.lower()} stage produces; only later"
                " stages may read it."
            )
        elif producer is None and field_name not in physics.fields:
            errors.append(f"{path}: reads '{field_name}', which the physics binding does not provide.")
    for field_name in spec.writes:
        if field_name not in physics.fields:
            errors.append(f"{path}: writes '{field_name}', which the physics binding does not provide.")
    if len(errors) > count:
        return None
    return ResolvedTerm(path, name, spec, cfg, params, None)


def _with_columns(terms: list[ResolvedTerm], errors: list[str]) -> tuple[tuple[ResolvedTerm, ...], int]:
    start, result = 0, []
    for term in terms:
        width = term.spec.width(term.params)
        if width <= 0:
            errors.append(f"{term.path}: term '{term.spec.name}' has width {width}; it must be positive.")
        result.append(ResolvedTerm(term.path, term.name, term.spec, term.cfg, term.params, (start, start + width)))
        start += width
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
    if not (isinstance(cfg.episode_length_s, (int, float)) and cfg.episode_length_s > 0):
        errors.append(f"episode_length_s must be positive, got {cfg.episode_length_s!r}.")
    if backend == "torch" and not isinstance(physics.device, str):
        errors.append("physics device must be a device string.")

    def resolve(section: str, terms: Mapping[str, TermCfg], stage: Stage, cfg_type: type) -> list[ResolvedTerm]:
        out = []
        for name, term_cfg in terms.items():
            path = f"{section}.{name}"
            if not isinstance(term_cfg, cfg_type):
                errors.append(f"{path}: expected {cfg_type.__name__}, got {type(term_cfg).__name__}.")
                continue
            resolved = _resolve_term(path, name, term_cfg, stage, physics, backend, errors)
            if resolved is not None:
                out.append(resolved)
        return out

    if not cfg.actions:
        errors.append("actions: at least one action term is required.")
    actions = resolve("actions", cfg.actions, Stage.ACTION, ActionTermCfg)
    for term in actions:
        if term.cfg.clip is not None:
            _check_range(term.path, "clip", term.cfg.clip, errors)
    actions, num_actions = _with_columns(actions, errors)

    if not cfg.observations:
        errors.append("observations: at least one observation group is required.")
    observations, widths = {}, {}
    for group, group_cfg in cfg.observations.items():
        if not group_cfg.terms:
            errors.append(f"observations.{group}: the group has no terms.")
        terms = resolve(f"observations.{group}", group_cfg.terms, Stage.OBSERVATION, ObservationTermCfg)
        for term in terms:
            if term.cfg.clip is not None:
                _check_range(term.path, "clip", term.cfg.clip, errors)
        observations[group], widths[group] = _with_columns(terms, errors)

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
        num_envs=physics.num_envs,
        step_dt=physics.step_dt,
        max_episode_length=math.ceil(cfg.episode_length_s / physics.step_dt),
        num_actions=num_actions,
        actions=actions,
        observations=observations,
        observation_widths=widths,
        terminations=tuple(terminations),
        rewards=tuple(rewards),
        reset_events=tuple(t for t in events if t.cfg.mode == "reset"),
        interval_events=tuple(t for t in events if t.cfg.mode == "interval"),
        compute_final_observations=cfg.compute_final_observations,
        seed=cfg.seed,
    )


# -- binding and execution -----------------------------------------------------------------------------


Op = tuple[str, Callable[[], None]]


def _check_shape(name: str, array: Any, shape: tuple[int, ...]) -> None:
    if tuple(array.shape) != shape:
        raise ValueError(f"Buffer '{name}' has shape {tuple(array.shape)}, expected {shape}.")


class MdpProgram:
    """An execution plan bound to fixed buffers.

    :meth:`step` runs this operation list, in order, on the current stream:

    1. ``action.process`` then one ``action.<term>`` per action term
    2. ``physics.step`` (skipped when ``include_physics=False``)
    3. ``runtime.episode_length``; ``termination.<term>`` ...; ``termination.reduce``
    4. ``reward.<term>`` ...; ``reward.reduce``
    5. with final observations: ``observation.<group>.<term>`` ...; ``final_observation.<group>``
    6. ``event.reset.<term>`` ...; ``physics.commit.reset``; ``runtime.reset_state``; interval timer resampling
    7. per interval event: ``event.interval.<term>.tick``, the term, ``physics.commit.interval.<term>``
    8. ``observation.<group>.<term>`` ...; ``observation.<group>.post`` when a term scales or clips

    Every array argument is fixed when the program is created, so the list can be captured into a CUDA graph
    and replayed.
    """

    def __init__(self, plan: ExecutionPlan, inputs: MdpInputs, state: MdpState, outputs: MdpOutputs):
        self.plan = plan
        self.inputs = inputs
        self.state = state
        self.outputs = outputs
        self.backend = be = plan.make_backend()
        self._validate_buffers()
        self._host_weights = [float(term.cfg.weight) for term in plan.rewards]
        self._reward_index = {term.name: k for k, term in enumerate(plan.rewards)}
        physics = plan.physics
        self._fields = {name: be.adopt(array) for name, array in physics.fields.items()}
        self._fields.update(
            action=state.action,
            prev_action=state.prev_action,
            episode_length=state.episode_length,
            terminated=outputs.terminated,
            truncated=outputs.truncated,
        )
        self._indices: dict[str, dict[str, Any]] = {}

        actions = self._bind_actions()
        terminations = self._bind_terminations()
        rewards = self._bind_rewards()
        observations = self._bind_observations()
        final = (
            [
                (f"final_observation.{g}", be.bind_copy(outputs.final_observations[g], obs))
                for g, obs in outputs.observations.items()
            ]
            if plan.compute_final_observations
            else []
        )

        self.pre_physics_ops: tuple[Op, ...] = tuple(actions)
        self.physics_ops: tuple[Op, ...] = (("physics.step", physics.step),)
        self.post_physics_ops: tuple[Op, ...] = (
            ("runtime.episode_length", be.bind_increment(state.episode_length)),
            *terminations,
            *rewards,
            *(observations + final if final else []),
            *self._bind_reset(outputs.reset_mask),
            *self._bind_intervals(),
            *observations,
        )
        self.reset_ops: tuple[Op, ...] = (*self._bind_reset(state.reset_request), *observations)

    # -- public API ----------------------------------------------------------------------------------

    @property
    def op_names(self) -> tuple[str, ...]:
        """Names of the step operations in execution order."""
        return tuple(name for name, _ in self.pre_physics_ops + self.physics_ops + self.post_physics_ops)

    def step(self, include_physics: bool = True) -> None:
        """Enqueue one control step reading :attr:`inputs` and writing :attr:`outputs`. Never synchronizes."""
        with self.backend.stream_scope():
            for _, op in self.pre_physics_ops:
                op()
            if include_physics:
                for _, op in self.physics_ops:
                    op()
            for _, op in self.post_physics_ops:
                op()

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
            copy = be.bind_copy(self.state.reset_request, mask)
        else:
            if be.is_capturing():
                raise RuntimeError("reset(env_ids) converts indices on the host; pass a device mask during capture.")
            be.assign_mask(self.state.reset_request, env_ids)
            copy = None
        with be.stream_scope():
            if copy is not None:
                copy()
            for _, op in self.reset_ops:
                op()

    def set_reward_weight(self, name: str, weight: float) -> None:
        """Change a reward weight in device memory. Captured graphs see the new value on the next replay."""
        if self.backend.is_capturing():
            raise RuntimeError("Reward weights are host-written; change them outside capture.")
        if not math.isfinite(weight):
            raise ValueError(f"Reward weight must be finite, got {weight}.")
        self._host_weights[self._reward_index[name]] = float(weight)
        weights = self.backend.constant(self._host_weights, "float32")
        self.backend.bind_copy(self.state.reward_weights, weights)()

    def capture(self, include_physics: bool = True, warmup: bool = True) -> CapturedStep:
        """Capture one :meth:`step` into a CUDA graph.

        Warp programs use :class:`warp.ScopedCapture`; Torch programs use :class:`torch.cuda.CUDAGraph` with
        Warp physics launches redirected to the capture stream. With ``warmup`` one eager step runs first to
        compile kernels and settle lazy allocations; it advances the environment state.
        """
        return capture_step(self.backend, lambda: self.step(include_physics), warmup=warmup, owner=self)

    # -- binding helpers -------------------------------------------------------------------------------

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
            ("state.termination_values", s.termination_values, (t, n)),
            ("state.reward_values", s.reward_values, (k, n)),
            ("state.episode_sums", s.episode_sums, (k, n)),
            ("state.reward_weights", s.reward_weights, (k,)),
            ("state.interval_time_left", s.interval_time_left, (e, n)),
            ("state.interval_fired", s.interval_fired, (e, n)),
            ("state.reset_request", s.reset_request, (n,)),
            ("outputs.reward", o.reward, (n,)),
            ("outputs.terminated", o.terminated, (n,)),
            ("outputs.truncated", o.truncated, (n,)),
            ("outputs.reset_mask", o.reset_mask, (n,)),
        ):
            _check_shape(name, array, shape)
        for group, width in p.observation_widths.items():
            _check_shape(f"outputs.observations.{group}", o.observations[group], (n, width))
            if p.compute_final_observations:
                _check_shape(f"outputs.final_observations.{group}", o.final_observations[group], (n, width))

    def _context(self, term: ResolvedTerm, *, out=None, action=None, mask=None, rng=None) -> TermContext:
        if term.path not in self._indices:
            self._indices[term.path] = (
                {"joint_ids": self.backend.constant(term.params["joint_ids"], "index")}
                if "joint_ids" in term.params
                else {}
            )
        declared = term.spec.reads + term.spec.writes
        return TermContext(
            params=term.params,
            fields={name: self._fields[name] for name in declared},
            out=out,
            action=action,
            mask=mask,
            rng=rng,
            indices=self._indices[term.path],
            num_envs=self.plan.num_envs,
            step_dt=self.plan.step_dt,
            max_episode_length=self.plan.max_episode_length,
            device=self.plan.device,
        )

    def _bind(self, term: ResolvedTerm, **kwargs) -> Callable[[], None]:
        return get_impl(term.spec.name, self.plan.backend)(self._context(term, **kwargs))

    def _bind_actions(self) -> list[Op]:
        be, s, terms = self.backend, self.state, self.plan.actions
        scale, offset, lo, hi = [], [], [], []
        for term in terms:
            width = term.columns[1] - term.columns[0]
            clip = term.cfg.clip or (-math.inf, math.inf)
            scale += [term.cfg.scale] * width
            offset += [term.cfg.offset] * width
            lo += [clip[0]] * width
            hi += [clip[1]] * width
        constants = [be.constant(values, "float32") for values in (scale, offset, lo, hi)]
        ops = [
            (
                "action.process",
                be.bind_process_actions(self.inputs.actions, *constants, s.action, s.prev_action, s.processed_actions),
            )
        ]
        for term in terms:
            action = be.columns(s.processed_actions, *term.columns)
            ops.append((f"action.{term.name}", self._bind(term, action=action)))
        return ops

    def _bind_terminations(self) -> list[Op]:
        be, s, o, terms = self.backend, self.state, self.outputs, self.plan.terminations
        ops = [(f"termination.{t.name}", self._bind(t, out=s.termination_values[k])) for k, t in enumerate(terms)]
        time_out = be.constant([t.cfg.time_out for t in terms], "bool")
        ops.append(
            (
                "termination.reduce",
                be.bind_reduce_terminations(s.termination_values, time_out, o.terminated, o.truncated, o.reset_mask),
            )
        )
        return ops

    def _bind_rewards(self) -> list[Op]:
        be, s, o, terms = self.backend, self.state, self.outputs, self.plan.rewards
        ops = [(f"reward.{t.name}", self._bind(t, out=s.reward_values[k])) for k, t in enumerate(terms)]
        ops.append(
            (
                "reward.reduce",
                be.bind_reduce_rewards(s.reward_values, s.reward_weights, self.plan.step_dt, o.reward, s.episode_sums),
            )
        )
        return ops

    def _bind_observations(self) -> list[Op]:
        be, ops = self.backend, []
        for group, terms in self.plan.observations.items():
            buffer = self.outputs.observations[group]
            lo, hi, scale = [], [], []
            for term in terms:
                ops.append(
                    (f"observation.{group}.{term.name}", self._bind(term, out=be.columns(buffer, *term.columns)))
                )
                width = term.columns[1] - term.columns[0]
                clip = term.cfg.clip or (-math.inf, math.inf)
                lo += [clip[0]] * width
                hi += [clip[1]] * width
                scale += [term.cfg.scale] * width
            if any(t.cfg.clip is not None or t.cfg.scale != 1.0 for t in terms):
                constants = [be.constant(values, "float32") for values in (lo, hi, scale)]
                ops.append((f"observation.{group}.post", be.bind_post_observations(buffer, *constants)))
        return ops

    def _bind_reset(self, mask: Any) -> list[Op]:
        """Reset events, physics commit, runtime state, and interval timers for the masked environments."""
        be, s, plan = self.backend, self.state, self.plan
        ops = [(f"event.reset.{t.name}", self._bind(t, mask=mask, rng=s.rng)) for t in plan.reset_events]
        if any(t.spec.writes for t in plan.reset_events):
            wp_mask = be.to_warp(mask)
            ops.append(("physics.commit.reset", lambda: plan.physics.commit(wp_mask)))
        ops.append(
            (
                "runtime.reset_state",
                be.bind_reset_state(mask, s.episode_length, s.action, s.prev_action, s.episode_sums),
            )
        )
        for k, t in enumerate(plan.interval_events):
            lo, hi = t.cfg.interval_range_s
            ops.append(
                (
                    f"event.interval.{t.name}.resample",
                    be.bind_sample_timers(mask, s.rng, s.interval_time_left[k], lo, hi),
                )
            )
        return ops

    def _bind_intervals(self) -> list[Op]:
        be, s, plan, ops = self.backend, self.state, self.plan, []
        for k, t in enumerate(plan.interval_events):
            lo, hi = t.cfg.interval_range_s
            fired = s.interval_fired[k]
            ops.append(
                (
                    f"event.interval.{t.name}.tick",
                    be.bind_tick_timers(s.interval_time_left[k], plan.step_dt, s.rng, lo, hi, fired),
                )
            )
            ops.append((f"event.interval.{t.name}", self._bind(t, mask=fired, rng=s.rng)))
            if t.spec.writes:
                wp_fired = be.to_warp(fired)
                ops.append((f"physics.commit.interval.{t.name}", lambda m=wp_fired: plan.physics.commit(m)))
        return ops


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
