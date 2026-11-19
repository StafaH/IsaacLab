# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration of the MDP2 G1 motion tracking task: the MDP1 task's, logged apart."""

from isaaclab.utils import configclass

from isaaclab_tasks.core.motion_tracking.config.g1.agents.rsl_rl_ppo_cfg import (
    G1MotionTrackingPPORunnerCfg as MDP1G1MotionTrackingPPORunnerCfg,
)


@configclass
class G1MotionTrackingPPORunnerCfg(MDP1G1MotionTrackingPPORunnerCfg):
    experiment_name = "g1_motion_tracking_mdp2"
