# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configurations of the Unitree Go2 velocity tasks."""

from isaaclab.utils import configclass, replace

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 50
    experiment_name = "unitree_go2_flat"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[128, 128, 128], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeGo2FlatWalkStandPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    # With half of the worlds standing, standing still is a local optimum that walking leaves after 300 iterations.
    max_iterations = 1000
    experiment_name = "unitree_go2_flat_walk_stand"


@configclass
class UnitreeGo2RoughPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    max_iterations = 1500
    experiment_name = "unitree_go2_rough"
    actor = replace(UnitreeGo2FlatPPORunnerCfg.actor, hidden_dims=[512, 256, 128])
    critic = replace(UnitreeGo2FlatPPORunnerCfg.critic, hidden_dims=[512, 256, 128])
