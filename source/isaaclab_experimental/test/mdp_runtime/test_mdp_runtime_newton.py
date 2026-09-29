# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics + MDP capture of the runtime Cartpole task on Newton/MJWarp."""

from __future__ import annotations

import numpy as np
import torch
from isaaclab_experimental.mdp_runtime import NewtonArticulationPhysics, compile_plan
from isaaclab_tasks_experimental.mdp_runtime.cartpole import DECIMATION, CartpoleMdpCfg, cartpole_scene_cfgs

from isaaclab.sim import build_simulation_context
from isaaclab.utils import instantiate

NUM_ENVS = 64
NUM_STEPS = 40

TORCH_WARP_ATOL = 1.0e-4
"""Stated Torch/Warp agreement tolerance on Newton. Both backends feed bitwise-equal resets to the solver; the
remaining differences come from float contraction in the MDP terms, amplified by 40 steps of pole dynamics."""


def _outputs(program) -> np.ndarray:
    be, o = program.backend, program.outputs
    return np.concatenate(
        [
            be.to_numpy(o.observations["policy"]).ravel(),
            be.to_numpy(o.final_observations["policy"]).ravel(),
            be.to_numpy(o.reward),
            be.to_numpy(o.terminated),
            be.to_numpy(o.truncated),
        ]
    )


def _run(plan, actions: np.ndarray, captured: bool) -> list[np.ndarray]:
    """Run from a fresh reset. Fresh state per run gives identical random resets."""
    program = plan.bind(*plan.allocate())
    program.reset()
    graph = program.capture(warmup=False) if captured else None
    history = []
    for action in actions:
        if program.backend.name == "warp":
            program.inputs.actions.assign(action)
        else:
            program.inputs.actions.copy_(torch.from_numpy(action))
        graph.replay() if captured else program.step()
        history.append(_outputs(program))
    return history


def test_newton_cartpole_capture_and_backend_agreement():
    """Physics + MDP graphs replay like eager steps on both backends, and the backends agree."""
    sim_cfg, scene_cfg = cartpole_scene_cfgs(NUM_ENVS)
    cfg = CartpoleMdpCfg(seed=3, episode_length_s=0.4)
    actions = np.random.default_rng(0).uniform(-1.0, 1.0, (NUM_STEPS, NUM_ENVS, 1)).astype(np.float32)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = instantiate(scene_cfg)
        sim.reset()
        physics = NewtonArticulationPhysics(scene, "robot", DECIMATION)
        results = {}
        for backend in ("warp", "torch"):
            plan = compile_plan(cfg, physics, backend)
            warm = plan.bind(*plan.allocate())
            warm.reset()
            warm.step()
            results[backend] = _run(plan, actions, captured=False)
            for step, (eager, replayed) in enumerate(zip(results[backend], _run(plan, actions, captured=True))):
                np.testing.assert_array_equal(eager, replayed, err_msg=f"{backend} step {step}")

        # Reset writes reached the solver: one physics step continues from the sampled state instead of the
        # pre-reset solver state (|velocity| <= 0.8 rad/s over 1/60 s moves joints by far less than 0.1).
        plan = compile_plan(cfg, physics, "warp")
        program = plan.bind(*plan.allocate())
        program.reset()
        sampled = physics.fields["joint_pos"].numpy().copy()
        assert np.std(sampled[:, 0]) > 0.3  # cart positions sampled from (-1, 1)
        physics.fields["joint_effort_target"].zero_()
        physics.step()
        assert np.abs(physics.fields["joint_pos"].numpy() - sampled).max() < 0.1

    resets = sum(int(o[-2 * NUM_ENVS :].sum()) for o in results["warp"])
    assert resets >= NUM_ENVS
    for step, (w, t) in enumerate(zip(results["warp"], results["torch"])):
        np.testing.assert_allclose(w, t, rtol=0.0, atol=TORCH_WARP_ATOL, err_msg=f"step {step}")
