# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terms of the parkour task."""

from .events import start_episode
from .observations import episode_b, goal_b
from .quantities import at_goal, goal
from .rewards import failed, goal_reached, joint_power_l1
from .terminations import above_limit, after_steps
from .terrains import ClipBoxesTerrainCfg
