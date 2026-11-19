# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kuka Allegro lift, from the state or from a base camera, written with MDP2."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Lift-KukaAllegro-MDP2",
    entry_point="isaaclab_experimental.envs:MDP2RLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_env_cfg:KukaAllegroLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KukaAllegroLiftPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Lift-KukaAllegro-Camera-MDP2",
    entry_point="isaaclab_experimental.envs:MDP2RLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:KukaAllegroLiftCameraEnvCfg",
        # MDP1's single-camera runner: a spatial-softmax CNN actor on the image and the state, and a state critic.
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.core.lift.config.kuka_allegro.agents.rsl_rl_ppo_cfg:KukaAllegroSingleCameraPPORunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)
