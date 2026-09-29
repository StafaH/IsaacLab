# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure CUDA-graph capture boundaries of the experimental MDP runtime on Newton Cartpole.

Boundaries:

* ``mdp``: one graph with the MDP step only (actions, terminations, rewards, resets, events, observations).
  Physics is bound but not stepped.
* ``physics_mdp``: one graph with the MDP step and the Newton/MJWarp physics step.
* ``training``: one graph with a full warp-rl PPO iteration: ``horizon`` physics+MDP steps plus every learning
  epoch. Requires the sibling ``warp-rl`` checkout.

Each boundary replays its graph and checks it against eager execution before timing it.

.. code-block:: bash

    uv run python scripts/benchmarks/benchmark_mdp_runtime_capture.py --boundary mdp physics_mdp --backend warp torch
    uv run --with-editable ../warp-rl python scripts/benchmarks/benchmark_mdp_runtime_capture.py --boundary training
    uv run --with-editable ../warp-rl python scripts/benchmarks/benchmark_mdp_runtime_capture.py --boundary training \\
        --backend torch --graph_owner torch

"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from isaaclab_experimental.mdp_runtime import MdpEnv, NewtonPhysics, capture_step, compile_plan
from isaaclab_tasks_experimental.mdp_runtime.cartpole import STABLE_TASK, CartpoleMdpCfg
from isaaclab_tasks_experimental.mdp_runtime.stable import stable_physics_cfgs

from isaaclab.app import launch_simulation
from isaaclab.sim import SimulationContext
from isaaclab.utils import instantiate


def time_replays(replay, device: str, replays: int, samples: int = 5) -> float:
    """Median wall time [ms] of one replay, from CUDA events around ``replays`` launches."""
    for _ in range(3):
        replay()
    times = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(torch.cuda.current_stream(device)):
            wp.synchronize_device(device)
            start.record()
            for _ in range(replays):
                replay()
            wp.synchronize_device(device)
            end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / replays)
    return float(np.median(times))


def time_eager(step, device: str, steps: int, samples: int = 5) -> float:
    """Median host wall time [ms] of one eager step, synchronized per sample."""
    step()
    times = []
    for _ in range(samples):
        wp.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(steps):
            step()
        wp.synchronize_device(device)
        times.append((time.perf_counter() - start) * 1000.0 / steps)
    return float(np.median(times))


def trajectory(program, actions: np.ndarray, include_physics: bool, replay=None) -> list[np.ndarray]:
    """Run ``actions`` through eager steps, or through ``replay`` if given; return outputs per step."""
    be, out = program.backend, []
    for action in actions:
        if be.name == "warp":
            program.inputs.actions.assign(action)
        else:
            program.inputs.actions.copy_(torch.from_numpy(action))
        if replay is None:
            program.step(include_physics)
        else:
            replay()
        o = program.outputs
        out.append(np.concatenate([be.to_numpy(o.observations["policy"]).ravel(), be.to_numpy(o.reward)]))
    return out


def measure_step_boundary(plan, include_physics: bool, args) -> dict:
    """Validate replay against eager execution from identical state, then time both."""
    actions = np.random.default_rng(0).uniform(-1.0, 1.0, (args.check_steps, plan.num_envs, 1)).astype(np.float32)
    warm = plan.bind(*plan.allocate())
    warm.reset()
    warm.step(include_physics)  # compile kernels outside the measured runs

    # Fresh state per run: the random streams then produce identical resets.
    eager_program = plan.bind(*plan.allocate())
    eager_program.reset()
    eager = trajectory(eager_program, actions, include_physics)
    captured_program = plan.bind(*plan.allocate())
    captured_program.reset()
    graph = captured_program.capture(include_physics=include_physics, warmup=False)
    captured = trajectory(captured_program, actions, include_physics, replay=graph.replay)
    max_diff = max(float(np.abs(a - b).max()) for a, b in zip(eager, captured))
    replay_ms = time_replays(graph.replay, plan.device, args.replays)
    eager_ms = time_eager(lambda: captured_program.step(include_physics), plan.device, args.replays)
    return {
        "replay_matches_eager_max_abs_diff": max_diff,
        "check_steps": args.check_steps,
        "graph_replay_ms_per_step": replay_ms,
        "eager_ms_per_step": eager_ms,
        "graph_env_steps_per_s": plan.num_envs * 1000.0 / replay_ms,
        "eager_env_steps_per_s": plan.num_envs * 1000.0 / eager_ms,
    }


def train(physics, backend: str, args) -> dict:
    """Capture physics + MDP + PPO learning into one graph and train.

    ``--graph_owner warp_rl`` uses warp-rl's :class:`OnPolicyRunner`, which captures with ``wp.ScopedCapture``.
    ``--graph_owner torch`` captures the same PPO iteration with :func:`capture_step`: a Torch-owned graph with
    Warp registered as an external capture, all launches on the Torch capture stream.
    """
    from warp_rl import PPO, OnPolicyRunner, PPOConfig
    from warp_rl.evaluation import evaluate

    cfg = CartpoleMdpCfg(seed=args.seed)
    # warp-rl samples unbounded Gaussian actions; clip the processed effort to the raw range [-1, 1].
    cfg.actions["joint_effort"].clip = (-100.0, 100.0)
    plan = compile_plan(cfg, physics, backend)
    program = plan.bind(*plan.allocate())
    env = MdpEnv(program)
    config = PPOConfig(hidden_sizes=(64, 64), reward_scale=1.0 / plan.step_dt, seed=args.seed)
    ppo = PPO(env, config)
    start = time.perf_counter()
    if args.graph_owner == "warp_rl":
        runner = OnPolicyRunner(ppo, capture=not args.eager)
        launch = runner.step
    else:
        # The runner only provides metrics here; the warm-up iteration is a real (kept) PPO update.
        runner = OnPolicyRunner(ppo, capture=False)

        def iteration():
            with program.backend.stream_scope():
                ppo.iteration()

        graph = capture_step(program.backend, iteration, warmup=True, owner=ppo)
        runner.iterations += 1
        launch = graph.replay
    setup_s = time.perf_counter() - start
    # warp-rl's evaluation captures env.step with wp.ScopedCapture, which requires Warp-owned capture.
    evaluable = args.graph_owner == "warp_rl"
    initial = evaluate(ppo, 600) if evaluable else None
    env.reset()
    history = []
    wp.synchronize()
    start = time.perf_counter()
    for i in range(args.iterations):
        launch()
        if args.graph_owner == "torch":
            runner.iterations += 1
        if (i + 1) % 25 == 0 or i == 0:
            metrics = runner.metrics()
            history.append({"iteration": runner.iterations, **metrics})
            print(json.dumps(history[-1]), flush=True)
            if not np.isfinite(list(metrics.values())).all():
                raise RuntimeError("Non-finite training metrics.")
    wp.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "graph_owner": args.graph_owner,
        "capture": not args.eager,
        "iterations": args.iterations,
        "ppo_config": {"horizon": config.horizon, "epochs": config.epochs, "minibatches": config.minibatches},
        "capture_setup_s": setup_s,
        "train_s": elapsed,
        "transitions_per_s": args.iterations * config.horizon * plan.num_envs / elapsed,
        "initial_evaluation": initial,
        "trained_evaluation": evaluate(ppo, 600) if evaluable else None,
        "final_training_metrics": history[-1],
        "history": history,
    }


def revision(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "unavailable (no commits)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boundary", nargs="+", choices=["mdp", "physics_mdp", "training"], default=["mdp"])
    parser.add_argument("--backend", nargs="+", choices=["warp", "torch"], default=["warp"])
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--check_steps", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--eager", action="store_true", help="Train without capturing (warp_rl graph owner only).")
    parser.add_argument(
        "--graph_owner",
        choices=["warp_rl", "torch"],
        default="warp_rl",
        help="Who captures the training graph: warp-rl's OnPolicyRunner or a Torch CUDA graph.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Write the results as JSON.")
    args = parser.parse_args()
    if args.graph_owner == "torch" and args.backend != ["torch"]:
        parser.error("--graph_owner torch captures Torch MDP programs only; use --backend torch.")

    sim_cfg, scene_cfg, decimation = stable_physics_cfgs(STABLE_TASK, args.num_envs)
    root = Path(__file__).resolve().parents[2]
    results = {
        "hardware": {"gpu": torch.cuda.get_device_name(0), "platform": platform.platform()},
        "dependencies": {
            "isaaclab": revision(root),
            "warp": wp.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "num_envs": args.num_envs,
        "boundaries": {},
    }
    with launch_simulation(sim_cfg):
        import newton

        results["dependencies"]["newton"] = newton.__version__
        sim = SimulationContext(sim_cfg)
        scene = instantiate(scene_cfg)
        sim.reset()
        physics = NewtonPhysics(scene, "robot", decimation)
        for backend in args.backend:
            for boundary in args.boundary:
                key = f"{boundary}/{backend}" + (f"/{args.graph_owner}" if boundary == "training" else "")
                print(f"== {key}", flush=True)
                try:
                    if boundary == "training":
                        import warp_rl

                        results["dependencies"]["warp_rl"] = revision(Path(warp_rl.__file__).resolve().parents[2])
                        result = train(physics, backend, args)
                    else:
                        plan = compile_plan(CartpoleMdpCfg(seed=args.seed), physics, backend)
                        result = measure_step_boundary(plan, boundary == "physics_mdp", args)
                    result["status"] = "demonstrated"
                except Exception as e:  # noqa: BLE001 - report the blocked boundary with the failing operation
                    result = {"status": "blocked", "error": f"{type(e).__name__}: {e}".splitlines()[0]}
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception as sync_error:  # noqa: BLE001
                            result["error_after_sync"] = str(sync_error).splitlines()[0]
                print(json.dumps({key: {k: v for k, v in result.items() if k != "history"}}, indent=2), flush=True)
                results["boundaries"][key] = result
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
