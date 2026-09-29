# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts of the experimental MDP runtime on analytic point-mass physics (no simulator)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_experimental.mdp_runtime import (
    EventTermCfg,
    HeterogeneousProgram,
    MdpConfigError,
    ObservationGroupCfg,
    ObservationTermCfg,
    PointMassPhysics,
    RewardTermCfg,
    TerminationTermCfg,
    compile_plan,
)
from isaaclab_tasks_experimental.mdp_runtime.point_mass import (
    DECIMATION,
    PHYSICS_DT,
    GantryMdpCfg,
    SliderMdpCfg,
    make_point_mass_population,
)

DEVICE = "cuda:0"
NUM_ENVS = 64
NUM_STEPS = 60

TORCH_WARP_ATOL = 1.0e-5
"""Stated Torch/Warp agreement tolerance for float outputs. Random draws are bitwise identical; float results
differ only by operation contraction, e.g. fused multiply-add in Warp kernels."""


def _cfg() -> GantryMdpCfg:
    """Gantry MDP with short episodes and a term that reports ``terminated`` so resets happen often."""
    cfg = GantryMdpCfg(seed=7, episode_length_s=0.3)
    cfg.rewards["terminated"] = RewardTermCfg(term="is_terminated", weight=-2.0)
    return cfg


def _program(backend: str, cfg=None):
    physics = PointMassPhysics(NUM_ENVS, ["x", "y", "z"], PHYSICS_DT, DECIMATION, DEVICE)
    plan = compile_plan(cfg or _cfg(), physics, backend)
    return plan.bind(*plan.allocate())


def _actions(num_actions: int = 3) -> np.ndarray:
    return np.random.default_rng(0).normal(scale=2.0, size=(NUM_STEPS, NUM_ENVS, num_actions)).astype(np.float32)


def _set_actions(program, action: np.ndarray) -> None:
    if program.backend.name == "warp":
        program.inputs.actions.assign(action)
    else:
        program.inputs.actions.copy_(torch.from_numpy(action))


def _snapshot(program) -> dict[str, np.ndarray]:
    be, o, s = program.backend, program.outputs, program.state
    return {
        "obs": be.to_numpy(o.observations["policy"]),
        "final_obs": be.to_numpy(o.final_observations["policy"]),
        "reward": be.to_numpy(o.reward),
        "terminated": be.to_numpy(o.terminated),
        "truncated": be.to_numpy(o.truncated),
        "episode_length": be.to_numpy(s.episode_length),
        "joint_pos": program.plan.physics.fields["joint_pos"].numpy(),
    }


def _run(program, actions: np.ndarray, replay=None) -> list[dict[str, np.ndarray]]:
    history = []
    for action in actions:
        _set_actions(program, action)
        replay() if replay is not None else program.step()
        history.append(_snapshot(program))
    return history


def test_compile_reports_every_configuration_error():
    """One compile reports all problems: unknown term, wrong kind, parameters, ordering, ranges, event mode."""
    cfg = _cfg()
    cfg.observations["policy"].terms["bad_kind"] = ObservationTermCfg(term="is_alive")
    cfg.rewards["unknown"] = RewardTermCfg(term="no_such_term", weight=1.0)
    cfg.rewards["bad_param"] = RewardTermCfg(term="joint_vel_l1", weight=1.0, params={"asset": "robot"})
    cfg.rewards["missing_param"] = RewardTermCfg(term="joint_pos_target_l2", weight=1.0)
    cfg.rewards["no_joint"] = RewardTermCfg(term="joint_vel_l1", weight=1.0, params={"joints": ["w"]})
    cfg.terminations["reads_later_stage"] = TerminationTermCfg(term="is_alive")
    cfg.events["bad_range"] = EventTermCfg(
        term="reset_joints_by_offset", params={"position_range": (1.0, -1.0), "velocity_range": (0.0, 0.0)}
    )
    cfg.events["no_interval"] = EventTermCfg(
        term="push_joints_by_velocity", mode="interval", params={"velocity_range": (0.0, 1.0)}
    )
    with pytest.raises(MdpConfigError) as info:
        _program("warp", cfg)
    message = str(info.value)
    for expected in (
        "observations.policy.bad_kind: term 'is_alive' is a reward term, not observation",
        "rewards.unknown: unknown term 'no_such_term'",
        "rewards.bad_param: unknown parameters ['asset']",
        "rewards.missing_param: missing required parameters ['target']",
        "rewards.no_joint:",
        "terminations.reads_later_stage: term 'is_alive' is a reward term",
        "terminations.reads_later_stage: reads 'terminated', which the termination stage produces",
        "events.bad_range: 'position_range' lower bound 1.0 exceeds upper bound -1.0",
        "events.no_interval: interval events require interval_range_s",
    ):
        assert expected in message
    assert len(info.value.errors) == 9


def test_term_ordering():
    """Rewards see this step's terminations; final observations are pre-reset; observations are post-reset."""
    program = _program("warp")
    names = program.schedule
    post_reset_obs = len(names) - 1 - names[::-1].index("observations.policy.pos")
    assert names.index("physics.step") < names.index("terminations.time_out") < names.index("rewards.terminated")
    assert names.index("final.observations.policy.pos") < names.index("reset.events.reset")
    assert names.index("reset.events.reset") < names.index("physics.commit") < post_reset_obs
    program.reset()
    history = _run(program, _actions())
    dt = PHYSICS_DT * DECIMATION
    num_terminated = 0
    for snap in history:
        reset = snap["terminated"] | snap["truncated"]
        num_terminated += int(np.sum(snap["terminated"]))
        np.testing.assert_array_equal(snap["episode_length"][reset], 0)
        assert np.all(snap["episode_length"][~reset] > 0)
        # Environments that did not reset report identical pre- and post-reset observations.
        np.testing.assert_array_equal(snap["obs"][~reset], snap["final_obs"][~reset])
        assert np.all(np.any(snap["obs"][reset] != snap["final_obs"][reset], axis=1))
        # All other terms are penalties, and is_alive is 0 once terminated, so the -2 is_terminated
        # contribution must be visible in the reward of the step that terminated.
        assert np.all(snap["reward"][snap["terminated"]] <= -2.0 * dt + 1e-6)
    assert num_terminated > 0


def test_subset_reset_touches_only_masked_environments():
    """reset(env_ids) and reset(mask) change exactly the selected rows; host ids are rejected during capture."""
    program = _program("warp")
    program.reset()
    for action in _actions()[:5]:
        _set_actions(program, action)
        program.step()
    before = _snapshot(program)
    selected = np.zeros(NUM_ENVS, dtype=bool)
    selected[[1, 5, 60]] = True
    program.reset(env_ids=[1, 5, 60])
    after = _snapshot(program)
    for key in ("obs", "episode_length", "joint_pos"):
        np.testing.assert_array_equal(after[key][~selected], before[key][~selected])
        assert np.all(np.any((after[key] != before[key]).reshape(NUM_ENVS, -1)[selected], axis=1))
    np.testing.assert_array_equal(after["episode_length"][selected], 0)

    mask = wp.array(~selected, dtype=wp.bool, device=DEVICE)
    program.reset(mask=mask)
    np.testing.assert_array_equal(_snapshot(program)["episode_length"], 0)

    with wp.ScopedCapture(device=DEVICE):
        with pytest.raises(RuntimeError, match="pass a device mask during capture"):
            program.reset(env_ids=[0])


@pytest.mark.parametrize("backend", ["warp", "torch"])
def test_graph_replay_matches_eager_and_reuses_buffers(backend):
    """Repeated replays equal eager steps bitwise, keep buffer addresses, and allocate nothing per step."""
    actions = _actions()
    eager_program = _program(backend)
    eager_program.reset()
    eager = _run(eager_program, actions)

    program = _program(backend)
    program.reset()
    be = program.backend
    buffers = [program.inputs.actions, program.outputs.observations["policy"], program.outputs.reward]
    buffers += list(vars(program.state).values()) + list(program.plan.physics.fields.values())
    addresses = [b.ptr if isinstance(b, wp.array) else b.data_ptr() for b in buffers]
    graph = program.capture(warmup=False)
    torch.cuda.synchronize()
    memory = torch.cuda.memory_allocated(DEVICE), wp.get_mempool_used_mem_current(DEVICE)
    captured = _run(program, actions, replay=graph.replay)
    torch.cuda.synchronize()
    assert (torch.cuda.memory_allocated(DEVICE), wp.get_mempool_used_mem_current(DEVICE)) == memory
    assert [b.ptr if isinstance(b, wp.array) else b.data_ptr() for b in buffers] == addresses
    assert sum(int(np.sum(s["terminated"] | s["truncated"])) for s in captured) > NUM_ENVS
    for step, (a, b) in enumerate(zip(eager, captured)):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"{key} at step {step}")

    # Reward weights live on the device: a captured graph sees a new weight without recapture.
    program.set_reward_weight("alive", 0.0)
    program.set_reward_weight("position", 0.0)
    program.set_reward_weight("velocity", 0.0)
    program.set_reward_weight("action_rate", 0.0)
    program.set_reward_weight("terminated", 0.0)
    graph.replay()
    np.testing.assert_array_equal(be.to_numpy(program.outputs.reward), 0.0)


def test_torch_and_warp_agree():
    """Both backends produce the same trajectory, including random resets and interval events."""
    actions = _actions()
    results = {}
    for backend in ("warp", "torch"):
        program = _program(backend)
        program.reset()
        results[backend] = _run(program, actions)
    for step, (w, t) in enumerate(zip(results["warp"], results["torch"])):
        for key in ("terminated", "truncated", "episode_length"):
            np.testing.assert_array_equal(w[key], t[key], err_msg=f"{key} at step {step}")
        for key in ("obs", "final_obs", "reward", "joint_pos"):
            np.testing.assert_allclose(w[key], t[key], rtol=0.0, atol=TORCH_WARP_ATOL, err_msg=f"{key} at {step}")


def test_heterogeneous_population():
    """Types keep their own widths and terms, write disjoint rows of shared outputs, and replay in one graph."""
    population = make_point_mass_population(num_sliders=48, num_gantries=80)
    slider, gantry = population.programs["slider"], population.programs["gantry"]
    assert (slider.plan.observation_widths["policy"], slider.plan.num_actions) == (2, 1)
    assert (gantry.plan.observation_widths["policy"], gantry.plan.num_actions) == (6, 3)
    assert "events.push" in gantry.schedule and "events.push" not in slider.schedule
    packed, widths = population.packed_observations()
    assert widths == {"slider": 2, "gantry": 6} and packed.shape == (128, 6)

    population.reset()
    graph = population.capture(warmup=False)
    rng = np.random.default_rng(0)
    for _ in range(40):
        slider.inputs.actions.assign(rng.normal(size=(48, 1)).astype(np.float32))
        gantry.inputs.actions.assign(rng.normal(size=(80, 3)).astype(np.float32))
        graph.replay()
    reward = population.reward.numpy()
    np.testing.assert_array_equal(reward[:48], slider.outputs.reward.numpy())
    np.testing.assert_array_equal(reward[48:], gantry.outputs.reward.numpy())
    rows = packed.numpy()
    np.testing.assert_array_equal(rows[:48, :2], slider.outputs.observations["policy"].numpy())
    np.testing.assert_array_equal(rows[:48, 2:], 0.0)
    np.testing.assert_array_equal(rows[48:], gantry.outputs.observations["policy"].numpy())
    assert population.terminated.numpy().shape == (128,)
    bad = GantryMdpCfg()
    bad.observations["policy"] = ObservationGroupCfg(terms={"pos": ObservationTermCfg(term="nope")})
    physics = PointMassPhysics(8, ["x", "y", "z"], PHYSICS_DT, DECIMATION, DEVICE)
    with pytest.raises(MdpConfigError) as info:
        HeterogeneousProgram({"slider": (SliderMdpCfg(), physics), "gantry": (bad, physics)})
    assert info.value.errors == [
        "gantry: observations.policy.pos: unknown term 'nope'.",
        "agent types must use distinct physics bindings.",
    ]
