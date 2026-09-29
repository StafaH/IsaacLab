# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare environment step throughput of the stable MDP, the Warp frontend, and the experimental MDP runtime.

Every system simulates the same stable task scene on Newton/MJWarp with the same random actions. One process
measures one (task, system) pair; ``--system all`` runs every system in its own subprocess and prints a table.

Systems:

* ``stable``: :class:`~isaaclab.envs.ManagerBasedRLEnv` (Torch managers), ``gym.make``.
* ``warp_frontend``: ``ManagerBasedRLEnvWarp`` via ``--frontend warp`` (captured Warp managers).
* ``runtime_warp`` / ``runtime_torch``: the experimental MDP runtime, one CUDA graph per step.
* ``runtime_warp_eager`` / ``runtime_torch_eager``: the same programs without a graph.
* ``physics_only``: a CUDA graph of the physics step alone with the default joint targets (robots at rest). Physics
  cost depends on the state (contacts, solver iterations), so this is a reference, not a strict ceiling.
* ``mdp_only``: a CUDA graph of the Warp runtime step without physics.

``--profile`` adds, for ``runtime_warp``, the GPU time per step of the runtime's generated MDP kernels, measured
with Warp's CUDA activity timer over steps from the same state distribution, and its share of the graph step
time. The rest of the step is physics and binding work, so ``1 - share`` bounds what any faster MDP could save.

.. code-block:: bash

    uv run python scripts/benchmarks/benchmark_mdp_runtime_compare.py --task go2 --system all --num_envs 4096

"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SYSTEMS = (
    "stable",
    "warp_frontend",
    "runtime_warp",
    "runtime_torch",
    "runtime_warp_eager",
    "runtime_torch_eager",
    "physics_only",
    "mdp_only",
)
TASKS = {
    "cartpole": "isaaclab_tasks_experimental.mdp_runtime.cartpole:CartpoleMdpCfg",
    "go2": "isaaclab_tasks_experimental.mdp_runtime.go2_velocity:Go2FlatVelocityMdpCfg",
    "reach": "isaaclab_tasks_experimental.mdp_runtime.franka_reach:FrankaReachMdpCfg",
}
SENSORS = {"go2": "contact_forces"}


def _load(path: str):
    import importlib

    module, name = path.split(":")
    module = importlib.import_module(module)
    return module, getattr(module, name)


def _timed(step, sync, steps: int, samples: int) -> list[float]:
    """Wall time [ms] per step of ``samples`` batches of ``steps`` steps, synchronized per batch."""
    times = []
    for _ in range(samples):
        sync()
        start = time.perf_counter()
        for i in range(steps):
            step(i)
        sync()
        times.append((time.perf_counter() - start) * 1000.0 / steps)
    return times


def measure(task: str, system: str, args) -> dict:
    import torch
    import warp as wp

    from isaaclab.app import launch_simulation

    module, mdp_cfg_class = _load(TASKS[task])
    stable_task = module.STABLE_TASK
    rng = np.random.default_rng(args.seed)

    if system in ("stable", "warp_frontend"):
        import gymnasium as gym

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(
            stable_task, device="cuda:0", num_envs=args.num_envs, overrides=["presets=newton_mjwarp"]
        )
        env_cfg.seed = args.seed
        with launch_simulation(env_cfg):
            if system == "stable":
                env = gym.make(stable_task, cfg=env_cfg)
            else:
                from isaaclab_experimental.envs.frontend import WarpFrontend

                env = WarpFrontend.build_env(env_cfg, stable_task)
            unwrapped = env.unwrapped
            action_dim = unwrapped.single_action_space.shape[0]
            ring = torch.as_tensor(
                rng.uniform(-1.0, 1.0, (args.ring, args.num_envs, action_dim)), dtype=torch.float32, device="cuda:0"
            )
            env.reset(seed=args.seed)
            with torch.inference_mode():
                for i in range(args.warmup):
                    env.step(ring[i % args.ring])
                times = _timed(
                    lambda i: env.step(ring[i % args.ring]), torch.cuda.synchronize, args.steps, args.samples
                )
            env.close()
        return {"ms_per_step": times}

    from isaaclab_experimental.mdp_runtime import NewtonPhysics, capture_step, compile_plan
    from isaaclab_tasks_experimental.mdp_runtime.stable import stable_physics_cfgs

    from isaaclab.sim import SimulationContext
    from isaaclab.utils import instantiate

    sim_cfg, scene_cfg, decimation = stable_physics_cfgs(stable_task, args.num_envs)
    with launch_simulation(sim_cfg):
        sim = SimulationContext(sim_cfg)
        scene = instantiate(scene_cfg)
        sim.reset()
        physics = NewtonPhysics(scene, "robot", decimation, contact_sensor=SENSORS.get(task))
        backend = "torch" if system.startswith("runtime_torch") else "warp"
        cfg = mdp_cfg_class(seed=args.seed)
        # The stable environment does not compute pre-reset observations by default.
        cfg.compute_final_observations = False
        plan = compile_plan(cfg, physics, backend)
        program = plan.bind(*plan.allocate())
        be = program.backend
        ring = [
            be.constant(rng.uniform(-1.0, 1.0, (args.num_envs, plan.num_actions)), "float32") for _ in range(args.ring)
        ]
        program.reset()
        sync = torch.cuda.synchronize if backend == "torch" else wp.synchronize

        def feed(i: int) -> None:
            be.copy(program.inputs.actions, ring[i % args.ring])

        if system == "physics_only":
            graph = capture_step(be, physics.step, owner=physics)
            step = lambda i: graph.replay()  # noqa: E731
        elif system.endswith("_eager"):
            step = lambda i: (feed(i), program.step())  # noqa: E731
        else:
            graph = program.capture(include_physics=system != "mdp_only")
            step = lambda i: (feed(i), graph.replay())  # noqa: E731
        for i in range(args.warmup):
            step(i)
        times = _timed(step, sync, args.steps, args.samples)
        result = {"ms_per_step": times, "schedule_length": len(program.schedule)}
        if args.profile and system == "runtime_warp":
            result["mdp_gpu_ms_per_step"] = _profile(lambda i: (feed(i), program.step()), args.profile_steps)
        return result


def _profile(step, steps: int) -> float:
    """GPU time per step [ms] of the runtime's generated MDP kernels."""
    import warp as wp

    wp.synchronize()
    with wp.ScopedTimer("profile", cuda_filter=wp.TIMING_KERNEL, print=False, synchronize=True) as timer:
        for i in range(steps):
            step(i)
    return sum(a.elapsed for a in timer.timing_results if "WarpExecutor" in a.name) / steps


def run_all(args) -> dict:
    results = {}
    for system in args.systems:
        command = [sys.executable, __file__, "--task", args.task, "--system", system, "--json"]
        command += ["--num_envs", str(args.num_envs), "--steps", str(args.steps), "--samples", str(args.samples)]
        command += ["--warmup", str(args.warmup), "--seed", str(args.seed)]
        command += ["--profile"] if args.profile else []
        completed = subprocess.run(command, capture_output=True, text=True)
        lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
        if completed.returncode != 0 or not lines:
            tail = (completed.stderr or completed.stdout).strip().splitlines()[-3:]
            results[system] = {"status": "failed", "error": " | ".join(tail)}
        else:
            results[system] = json.loads(lines[-1][len("RESULT ") :])
        print(system, json.dumps(results[system]), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), default="go2")
    parser.add_argument("--system", choices=[*SYSTEMS, "all"], default="all")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS), help="Systems for 'all'.")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--ring", type=int, default=8, help="Distinct random action batches cycled through.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true", help="Add the GPU time breakdown of runtime_warp.")
    parser.add_argument("--profile_steps", type=int, default=50)
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.system != "all":
        result = measure(args.task, args.system, args)
        median = float(np.median(result["ms_per_step"]))
        result.update(median_ms_per_step=median, env_steps_per_s=args.num_envs * 1000.0 / median, status="ok")
        print("RESULT " + json.dumps(result), flush=True)
        return

    results = run_all(args)
    rows = [(s, r) for s, r in results.items() if r.get("status") == "ok"]
    base = results.get("stable", {}).get("median_ms_per_step")
    print(f"\n{args.task}, {args.num_envs} envs")
    print(f"{'system':<22}{'ms/step':>10}{'env-steps/s':>14}{'vs stable':>11}")
    for system, r in rows:
        speedup = f"{base / r['median_ms_per_step']:.2f}x" if base else "-"
        print(f"{system:<22}{r['median_ms_per_step']:>10.3f}{r['env_steps_per_s']:>14,.0f}{speedup:>11}")
        if "mdp_gpu_ms_per_step" in r:
            mdp = r["mdp_gpu_ms_per_step"]
            print(f"{'  MDP kernels (GPU)':<22}{mdp:>10.3f}  = {mdp / r['median_ms_per_step']:.1%} of the step")
    if args.output:
        import torch
        import warp as wp

        report = {
            "task": args.task,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "samples": args.samples,
            "gpu": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
            "warp": wp.__version__,
            "torch": torch.__version__,
            "isaaclab": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
