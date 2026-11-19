# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 velocity tracking on flat ground, written with MDP2."""

import gymnasium as gym

AGENTS = "isaaclab_tasks.core.velocity.config.g1.agents"

gym.register(
    id="Isaac-Velocity-Flat-G1-MDP2",
    entry_point="isaaclab_experimental.envs:MDP2RLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1FlatEnvCfg",
        # The agents of the MDP1 task, so that both runtimes train with the same settings.
        "rsl_rl_cfg_entry_point": f"{AGENTS}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "rsl_rl_sac_cfg_entry_point": f"{AGENTS}.rsl_rl_sac_cfg:G1FlatSACRunnerCfg",
        "default_agent": "rsl_rl",
    },
)
