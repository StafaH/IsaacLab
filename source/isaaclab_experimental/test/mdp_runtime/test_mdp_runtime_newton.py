# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime tasks on Newton/MJWarp: graph replay against eager steps, and Torch/Warp agreement."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch
from isaaclab_experimental.mdp_runtime import NewtonPhysics, RewardTermCfg, TerminationTermCfg, compile_plan
from isaaclab_tasks_experimental.mdp_runtime.stable import stable_physics_cfgs

from isaaclab.sim import build_simulation_context
from isaaclab.utils import instantiate

NUM_ENVS = 64
NUM_STEPS = 40

TORCH_WARP_ATOL = 1.0e-4
"""Stated Torch/Warp agreement tolerance on Newton. Random draws are bitwise identical; float results differ by
operation contraction in Warp kernels, amplified by the dynamics over the steps."""


def _outputs(program) -> np.ndarray:
    be, o = program.backend, program.outputs
    parts = [be.to_numpy(o.observation_buffer), be.to_numpy(o.reward), be.to_numpy(o.terminated)]
    parts += [be.to_numpy(o.truncated), be.to_numpy(program.state.commands)]
    if o.final_observation_buffer is not None:
        parts.append(be.to_numpy(o.final_observation_buffer))
    return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])


def _run(plan, actions: np.ndarray, captured: bool, include_physics: bool) -> list[np.ndarray]:
    """Run from a fresh reset. Fresh state per run gives identical random resets."""
    program = plan.bind(*plan.allocate())
    program.reset()
    graph = program.capture(include_physics=include_physics, warmup=False) if captured else None
    history = []
    for action in actions:
        if program.backend.name == "warp":
            program.inputs.actions.assign(action)
        else:
            program.inputs.actions.copy_(torch.from_numpy(action))
        graph.replay() if captured else program.step(include_physics)
        history.append(_outputs(program))
    return history


def _scene(task: str):
    module = importlib.import_module(f"isaaclab_tasks_experimental.mdp_runtime.{task}")
    return module, stable_physics_cfgs(module.STABLE_TASK, NUM_ENVS)


def _check(physics, cfg, include_physics: bool) -> dict[str, list[np.ndarray]]:
    results = {}
    for backend in ("warp", "torch"):
        plan = compile_plan(cfg, physics, backend)
        warm = plan.bind(*plan.allocate())
        warm.reset()
        warm.step(include_physics)
        actions = np.random.default_rng(0).uniform(-1.0, 1.0, (NUM_STEPS, NUM_ENVS, plan.num_actions))
        actions = actions.astype(np.float32)
        results[backend] = _run(plan, actions, False, include_physics)
        replayed = _run(plan, actions, True, include_physics)
        for step, (eager, replay) in enumerate(zip(results[backend], replayed)):
            np.testing.assert_array_equal(eager, replay, err_msg=f"{backend} step {step}")
    for step, (w, t) in enumerate(zip(results["warp"], results["torch"])):
        np.testing.assert_allclose(w, t, rtol=0.0, atol=TORCH_WARP_ATOL, err_msg=f"step {step}")
    return results


def test_newton_cartpole_capture_and_backend_agreement():
    """Physics + MDP graphs replay like eager steps on both backends, and the backends agree."""
    module, (sim_cfg, scene_cfg, decimation) = _scene("cartpole")
    cfg = module.CartpoleMdpCfg(seed=3, episode_length_s=0.4)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = instantiate(scene_cfg)
        sim.reset()
        physics = NewtonPhysics(scene, "robot", decimation)
        _check(physics, cfg, include_physics=True)

        # The episodes are short enough that every environment resets during the checked steps.
        plan = compile_plan(cfg, physics, "warp")
        program = plan.bind(*plan.allocate())
        program.reset()
        resets = 0
        for _ in range(NUM_STEPS):
            program.step()
            resets += int(program.outputs.reset_mask.numpy().sum())
        assert resets >= NUM_ENVS

        # Reset writes reached the solver: one physics step continues from the sampled state instead of the
        # pre-reset solver state (|velocity| <= 0.8 rad/s over 1/60 s moves joints by far less than 0.1).
        program.reset()
        sampled = physics.fields["joint_pos"].numpy().copy()
        assert np.std(sampled[:, 0]) > 0.3  # cart positions sampled from (-1, 1)
        physics.fields["joint_effort_target"].zero_()
        physics.step()
        assert np.abs(physics.fields["joint_pos"].numpy() - sampled).max() < 0.1


@pytest.mark.parametrize(
    ("task", "cfg_name", "sensor"),
    [("go2_velocity", "Go2FlatVelocityMdpCfg", "contact_forces"), ("franka_reach", "FrankaReachMdpCfg", None)],
)
def test_newton_mdp_capture_and_backend_agreement(task, cfg_name, sensor):
    """Root, body, contact, and command terms: MDP graphs replay like eager steps and the backends agree.

    Physics is excluded: with contacts, Newton's collision pipeline is not bitwise reproducible between runs.
    Short episodes make every step exercise resets, command resampling, and events on the real scene state.
    """
    module, (sim_cfg, scene_cfg, decimation) = _scene(task)
    cfg = getattr(module, cfg_name)(seed=3, episode_length_s=0.2)
    if task == "go2_velocity":
        # Root terms that the stable Go2 task does not use.
        cfg.rewards["height"] = RewardTermCfg(term="base_height_l2", weight=-5.0, params={"target_height": 0.34})
        cfg.terminations["low"] = TerminationTermCfg(term="root_height_below_minimum", params={"minimum_height": 0.2})
        cfg.terminations["tilt"] = TerminationTermCfg(term="bad_orientation", params={"limit_angle": 0.5})
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = instantiate(scene_cfg)
        sim.reset()
        physics = NewtonPhysics(scene, "robot", decimation, contact_sensor=sensor)
        _check(physics, cfg, include_physics=False)
