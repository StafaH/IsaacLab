# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward search on humanoid motion tracking, written with MDP2."""

import gymnasium as gym

gym.register(
    id="Isaac-RewardSearch-Humanoid-Walk-MDP2",
    entry_point="isaaclab_experimental.envs:MDP2RLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.humanoid_env_cfg:HumanoidRewardSearchEnvCfg"},
)
