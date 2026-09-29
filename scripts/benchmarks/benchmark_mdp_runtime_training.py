# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Throughput ceilings of graph-captured warp-rl PPO training on runtime tasks and on warp-rl's own Go2.

Probes follow warp-rl's ``examples/speed_of_light.py``: CUDA-event GPU time of graph replays, median of samples,
FPS = environment transitions per second (``num_envs * horizon`` per iteration).

* ``full_training``: one PPO iteration (rollout + all epochs) as one graph (warp-rl ``OnPolicyRunner``).
* ``collection``: the rollout only; ``learning``: the update only.
* ``environment_action_tape``: ``horizon`` environment steps replaying the recorded policy actions.
* ``environment_zero_actions``: the same with zero actions.

Environments (``--env``):

* ``runtime_go2`` / ``runtime_reach`` / ``runtime_cartpole``: the experimental MDP runtime (Warp backend) on the
  stable task scene, exposed through :class:`~isaaclab_experimental.mdp_runtime.MdpEnv`.
* ``warprl_isaac_go2``: warp-rl's Isaac-Lab-hosted direct Go2 (``warp_rl.integrations.go2``).

Requires the sibling warp-rl checkout:

.. code-block:: bash

    uv run --with-editable ../warp-rl python scripts/benchmarks/benchmark_mdp_runtime_training.py --env runtime_go2
    uv run --with-editable ../warp-rl python scripts/benchmarks/benchmark_mdp_runtime_training.py --env runtime_go2 \\
        --train_iterations 1000

"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np

RUNTIME_TASKS = {
    "runtime_cartpole": ("isaaclab_tasks_experimental.mdp_runtime.cartpole", "CartpoleMdpCfg", None),
    "runtime_go2": ("isaaclab_tasks_experimental.mdp_runtime.go2_velocity", "Go2FlatVelocityMdpCfg", "contact_forces"),
    "runtime_reach": ("isaaclab_tasks_experimental.mdp_runtime.franka_reach", "FrankaReachMdpCfg", None),
}


def capture(fn):
    import warp as wp

    fn()
    wp.synchronize()
    with wp.ScopedCapture() as scope:
        fn()
    return lambda: wp.capture_launch(scope.graph)


def measure(fn, count: int, iterations: int, samples: int) -> dict:
    import warp as wp

    for _ in range(3):
        fn()
    wp.synchronize()
    gpu_ms = []
    start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    for _ in range(samples):
        wp.record_event(start)
        for _ in range(iterations):
            fn()
        wp.record_event(end)
        wp.synchronize()
        gpu_ms.append(wp.get_event_elapsed_time(start, end) / iterations)
    median = float(np.median(gpu_ms))
    return {"gpu_ms": gpu_ms, "median_gpu_ms": median, "fps": count * 1000.0 / median}


def build_env(args):
    """Return (env, step_dt, close) for the selected environment inside a running simulation."""
    if args.env == "warprl_isaac_go2":
        from warp_rl.integrations.factory import make_config, make_env

        cfg, task_id = make_config("go2", args.num_envs, args.seed)
        env = make_env("go2", cfg, task_id)
        return env, env.step_dt, env.close

    from isaaclab_experimental.mdp_runtime import MdpEnv, NewtonPhysics, compile_plan

    from isaaclab.sim import SimulationContext
    from isaaclab.utils import instantiate

    module_name, cfg_name, sensor = RUNTIME_TASKS[args.env]
    module = importlib.import_module(module_name)
    sim_cfg, scene_cfg = args.sim_cfg, args.scene_cfg
    sim = SimulationContext(sim_cfg)
    scene = instantiate(scene_cfg)
    sim.reset()
    physics = NewtonPhysics(scene, "robot", args.decimation, contact_sensor=sensor)
    cfg = getattr(module, cfg_name)(seed=args.seed)
    for term in cfg.actions.values():
        # warp-rl samples unbounded Gaussian actions; bound the raw action to [-5, 5] as warp-rl's Go2 does.
        term.clip = tuple(term.scale * bound + term.offset for bound in (-5.0, 5.0))
    plan = compile_plan(cfg, physics, "warp")
    return MdpEnv(plan.bind(*plan.allocate())), plan.step_dt, lambda: None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=[*RUNTIME_TASKS, "warprl_isaac_go2"], default="runtime_go2")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20, help="Graph replays per timing sample.")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--num_substeps", type=int, default=None, help="Override the Newton solver substeps.")
    parser.add_argument("--train_iterations", type=int, default=0, help="Also train and report learning metrics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import warp as wp
    from warp_rl import PPO, OnPolicyRunner, PPOConfig

    from isaaclab.app import launch_simulation

    if args.env == "warprl_isaac_go2":
        from warp_rl.integrations.factory import make_config

        launch_cfg, _ = make_config("go2", args.num_envs, args.seed)
    else:
        from isaaclab_tasks_experimental.mdp_runtime.stable import stable_physics_cfgs

        module = importlib.import_module(RUNTIME_TASKS[args.env][0])
        args.sim_cfg, args.scene_cfg, args.decimation = stable_physics_cfgs(module.STABLE_TASK, args.num_envs)
        if args.num_substeps is not None:
            args.sim_cfg.physics.num_substeps = args.num_substeps
        launch_cfg = args.sim_cfg

    report = {
        "arguments": {k: str(v) for k, v in vars(args).items() if k not in ("sim_cfg", "scene_cfg")},
        "phases": {},
    }
    with launch_simulation(launch_cfg):
        env, step_dt, close = build_env(args)
        try:
            config = PPOConfig(hidden_sizes=(args.width, args.width), horizon=args.horizon, reward_scale=1.0 / step_dt)
            ppo = PPO(env, config)
            start = time.perf_counter()
            runner = OnPolicyRunner(ppo)
            report["capture_s"] = time.perf_counter() - start
            count = args.num_envs * args.horizon

            def probe(name, fn):
                result = measure(fn, count, args.iterations, args.samples)
                report["phases"][name] = result
                print(f"{name}: {result['median_gpu_ms']:.3f} ms, {result['fps']:,.0f} FPS", flush=True)

            probe("full_training", runner.step)
            probe("collection", capture(ppo.collect))
            actions = [wp.clone(a) for a in ppo.storage.action_views]
            probe("learning", capture(ppo.update))

            def environment():
                for t in range(args.horizon):
                    env.step(actions[t])

            env.reset()
            graph = capture(environment)
            probe("environment_action_tape", graph)
            for a in actions:
                a.zero_()
            env.reset()
            probe("environment_zero_actions", graph)
            phases = report["phases"]
            report["interpretation"] = {
                "training_fraction_of_action_tape_ceiling": phases["environment_action_tape"]["median_gpu_ms"]
                / phases["full_training"]["median_gpu_ms"],
                "speedup_if_learning_were_free": phases["full_training"]["median_gpu_ms"]
                / phases["collection"]["median_gpu_ms"],
            }
            print(json.dumps(report["interpretation"], indent=2))

            if args.train_iterations:
                # Fresh learner state is not restored: training continues from the probes' updates.
                env.reset()
                history = []
                wp.synchronize()
                start = time.perf_counter()
                for i in range(args.train_iterations):
                    runner.step()
                    if (i + 1) % 100 == 0 or i == 0:
                        history.append({"iteration": i + 1, **runner.metrics()})
                        print(json.dumps(history[-1]), flush=True)
                wp.synchronize()
                elapsed = time.perf_counter() - start
                report["training"] = {
                    "iterations": args.train_iterations,
                    "seconds": elapsed,
                    "transitions_per_s_wall": args.train_iterations * count / elapsed,
                    "history": history,
                }
                print(f"training: {report['training']['transitions_per_s_wall']:,.0f} transitions/s (wall)")
        finally:
            close()
    report["gpu"] = str(wp.get_device().name)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
