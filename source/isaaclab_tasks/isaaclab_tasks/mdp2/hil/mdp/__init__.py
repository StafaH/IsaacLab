# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terms of the hybrid imitation learning tasks."""

from .commands import heading_goal, reference_goal
from .observations import goal_b
from .rewards import Discriminator, heading_velocity, motion_features, style
from .terminations import root_height_outside
