# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree Go2 velocity tracking with the style of dog clips, written with MDP2."""

import gymnasium as gym

gym.register(
    id="Isaac-Velocity-Flat-UnitreeGo2-Style-MDP2",
    entry_point="isaaclab_experimental.envs:MDP2RLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.go2_env_cfg:UnitreeGo2FlatStyleEnvCfg"},
)
