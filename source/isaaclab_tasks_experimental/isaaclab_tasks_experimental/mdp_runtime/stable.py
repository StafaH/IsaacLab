# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reuse the simulation and scene of stable tasks, so runtime tasks simulate exactly the same world."""

from __future__ import annotations

from collections.abc import Sequence


def stable_physics_cfgs(task_id: str, num_envs: int, device: str = "cuda:0", overrides: Sequence[str] = ()):
    """Resolve a stable task on Newton/MJWarp and return its simulation, scene, and decimation.

    Args:
        task_id: Registered stable task, e.g. ``"Isaac-Velocity-Flat-UnitreeGo2"``.
        num_envs: Number of environments.
        device: Simulation device.
        overrides: Extra Hydra overrides, applied after ``presets=newton_mjwarp``.

    Returns:
        The :class:`~isaaclab.sim.SimulationCfg`, the scene configuration, and the decimation.
    """
    import isaaclab_tasks  # noqa: F401  (registers the stable tasks)
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs, overrides=["presets=newton_mjwarp", *overrides])
    return cfg.sim, cfg.scene, cfg.decimation
