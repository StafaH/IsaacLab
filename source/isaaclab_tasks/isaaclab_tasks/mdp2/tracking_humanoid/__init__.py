# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Humanoid motion tracking, written with MDP2."""

import gymnasium as gym

from . import agents

for motion in ("Walk", "Run", "Dance"):
    gym.register(
        id=f"Isaac-Tracking-Humanoid-{motion}-MDP2",
        entry_point="isaaclab_experimental.envs:MDP2RLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.tracking_env_cfg:HumanoidTracking{'' if motion == 'Walk' else motion}EnvCfg"
            ),
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidTrackingPPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
