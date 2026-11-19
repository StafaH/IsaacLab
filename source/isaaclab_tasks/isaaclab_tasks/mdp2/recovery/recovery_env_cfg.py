# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recovery: some worlds run a task undisturbed, and the others run it while their robot is pushed.

One policy trains on both, so it keeps the precision of the undisturbed worlds and learns to recover in the pushed
ones. The pushes are an interval event of the pushed worlds only.
"""

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from isaaclab.utils import replace

robot = mdp2.Entity("robot")


def pushed(task: mdp2.TaskCfg, speed: float = 1.5, interval_s: tuple[float, float] = (1.0, 3.0)) -> mdp2.TaskCfg:
    """Return task with a push of the robot's root, up to speed [m/s] sideways, every interval_s."""
    ranges = ((-speed, speed), (-speed, speed), (-0.3 * speed, 0.3 * speed), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    push = mdp2.Event(
        mdp.push_by_setting_velocity,
        outputs={"root_com_vel_w": robot.root_com_vel_w},
        mode=mdp2.EventMode.INTERVAL,
        interval_range_s=interval_s,
        params={"velocity_range": ranges},
    )
    return replace(task, events={**task.events, "push": push})
