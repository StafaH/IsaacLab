# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity tracking of the robots of a family in the worlds of one simulation, written with MDP2."""

import gymnasium as gym

gym.register(
    id="Isaac-Velocity-Flat-Quadrupeds-MDP2",
    entry_point="isaaclab_experimental.envs:MDP2RLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.quadrupeds_env_cfg:QuadrupedsFlatEnvCfg"},
)

gym.register(
    id="Isaac-Velocity-Flat-Humanoids-MDP2",
    entry_point="isaaclab_experimental.envs:MDP2RLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.humanoids_env_cfg:HumanoidsFlatEnvCfg"},
)
