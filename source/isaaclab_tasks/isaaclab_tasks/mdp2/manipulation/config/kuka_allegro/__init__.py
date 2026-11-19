# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-task manipulation with the Kuka Allegro, written with MDP2."""

import gymnasium as gym

from isaaclab_tasks.mdp2.lift_kuka_allegro import agents

gym.register(
    id="Isaac-Manipulation-KukaAllegro-MDP2",
    entry_point="isaaclab_experimental.envs:MDP2RLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kuka_allegro_env_cfg:KukaAllegroManipulationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KukaAllegroLiftPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
