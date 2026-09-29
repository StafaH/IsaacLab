# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that a runtime task computes the same terms as its stable manager-based task.

The stable environment steps with random actions. After each step, the runtime's Torch terms are evaluated on
the same scene state, with the runtime's action, command, and episode buffers copied from the stable managers,
and compared with the stable per-term termination and reward values and the (noise-free) observations.
Environments that reset during the step are excluded from the reward and termination comparison, because the
stable values were computed before their reset.

.. code-block:: bash

    uv run python scripts/benchmarks/validate_mdp_runtime_parity.py --task go2

"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
import torch

TASKS = {
    "cartpole": ("isaaclab_tasks_experimental.mdp_runtime.cartpole", "CartpoleMdpCfg", None),
    "go2": ("isaaclab_tasks_experimental.mdp_runtime.go2_velocity", "Go2FlatVelocityMdpCfg", "contact_forces"),
    "reach": ("isaaclab_tasks_experimental.mdp_runtime.franka_reach", "FrankaReachMdpCfg", None),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), default="go2")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import gymnasium as gym
    from isaaclab_experimental.mdp_runtime import NewtonPhysics, compile_plan

    from isaaclab.app import launch_simulation

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    module_name, cfg_name, sensor = TASKS[args.task]
    module = importlib.import_module(module_name)
    env_cfg = parse_env_cfg(
        module.STABLE_TASK, device="cuda:0", num_envs=args.num_envs, overrides=["presets=newton_mjwarp"]
    )
    env_cfg.observations.policy.enable_corruption = False
    with launch_simulation(env_cfg):
        env = gym.make(module.STABLE_TASK, cfg=env_cfg).unwrapped
        env.reset(seed=args.seed)
        cfg = getattr(module, cfg_name)()
        for term in cfg.observations["policy"].terms.values():
            term.noise = None
        physics = NewtonPhysics(env.scene, "robot", env_cfg.decimation, contact_sensor=sensor)
        plan = compile_plan(cfg, physics, "torch")
        program = plan.bind(*plan.allocate())
        executor, state, outputs = program._executor, program.state, program.outputs
        errors: dict[str, float] = {}
        scales: dict[str, float] = {}
        rng = np.random.default_rng(args.seed)
        for _ in range(args.steps):
            commands_before = {c.name: env.command_manager.get_command(c.name).clone() for c in plan.commands}
            action = torch.as_tensor(rng.uniform(-1, 1, (args.num_envs, plan.num_actions)), dtype=torch.float32)
            obs, _, terminated, truncated, _ = env.step(action.to(env.device))
            kept = ~(terminated | truncated)
            # Mirror the stable runtime buffers that terms read.
            state.action.copy_(env.action_manager.action)
            state.prev_action.copy_(env.action_manager.prev_action)
            state.episode_length.copy_(env.episode_length_buf)
            outputs.terminated.copy_(env.termination_manager.terminated)
            outputs.truncated.copy_(env.termination_manager.time_outs)
            with program.backend.stream_scope():
                for c in plan.commands:
                    state.commands[:, slice(*c.columns)] = commands_before[c.name]
                for k, run in enumerate(executor._terminations):
                    run()
                    name = plan.terminations[k].name
                    stable = env.termination_manager.get_term(name)
                    mismatch = (state.termination_values[k] != stable)[kept].float().mean().item()
                    errors[f"terminations.{name} (mismatch rate)"] = max(
                        errors.get(f"terminations.{name} (mismatch rate)", 0), mismatch
                    )
                for k, run in enumerate(executor._rewards):
                    run()
                    term = plan.rewards[k]
                    index = env.reward_manager.active_terms.index(term.name)
                    stable = env.reward_manager._step_reward[:, index] / term.cfg.weight
                    error = (state.reward_values[k] - stable)[kept].abs().max().item()
                    scale = stable[kept].abs().max().item()
                    errors[f"rewards.{term.name}"] = max(errors.get(f"rewards.{term.name}", 0.0), error)
                    scales[term.name] = max(scales.get(term.name, 0.0), scale)
                for c in plan.commands:
                    state.commands[:, slice(*c.columns)] = env.command_manager.get_command(c.name)
                executor._observe_into("obs")
            error = (outputs.observations["policy"] - obs["policy"]).abs().max().item()
            errors["observations.policy"] = max(errors.get("observations.policy", 0.0), error)
        env.close()
    print(f"{args.task}: max abs difference to the stable task over {args.steps} steps, {args.num_envs} envs")
    print(f"  {'term':<55} {'max abs':>10} {'max |stable|':>13}")
    for name, value in errors.items():
        scale = scales.get(name.removeprefix("rewards."))
        print(f"  {name:<55} {value:>10.3g} {scale if scale is not None else float('nan'):>13.3g}")


if __name__ == "__main__":
    main()
