# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hybrid imitation learning with the 28-DoF humanoid, written with MDP2."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-HIL-Humanoid-Walk-MDP2",
    entry_point="isaaclab_experimental.envs:MDP2RLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_env_cfg:HumanoidHILEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidHILPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
