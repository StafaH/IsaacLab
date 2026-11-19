# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 whole-body motion tracking, the MDP1 task ``Isaac-Motion-Tracking-G1`` written with MDP2."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Motion-Tracking-G1-MDP2",
    entry_point="isaaclab_experimental.envs:MDP2RLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1MotionTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MotionTrackingPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
