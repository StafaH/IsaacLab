# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push recovery for humanoid motion tracking: half of the worlds track the dance clip while they are pushed."""

from isaaclab_experimental import mdp2

from isaaclab.utils import configclass

from isaaclab_tasks.mdp2.recovery.recovery_env_cfg import pushed
from isaaclab_tasks.mdp2.tracking.config.humanoid.humanoid_env_cfg import HumanoidTrackingEnvCfg, humanoid_task


def recovery_task(motions: str, start_range: tuple[float, float] = (0.0, 0.9)) -> mdp2.TaskCfg:
    """Return the tracking task for the clips motions that tolerates being off the clip by half a meter."""
    task = humanoid_task(motions, start_range)
    task.terminations["tracking_lost"].params["threshold"] = 0.25
    return task


@configclass
class HumanoidRecoveryEnvCfg(HumanoidTrackingEnvCfg):
    """The humanoid tracking environment with pushes in half of its worlds."""

    task: list[mdp2.TaskCfg] = [recovery_task("dance"), pushed(recovery_task("dance"))]
