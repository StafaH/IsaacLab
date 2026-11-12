# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terms of the motion tracking tasks."""

from .events import reference_frame, reset_to_csv_motion, reset_to_failures, reset_to_motion
from .observations import body_pos_b, reference_body_pos_b, root_height
from .quantities import body_pos_error, episode_steps, joint_pos_error, joint_vel_error, reference_joint_pos, root_vel_error
from .rewards import exp_kernel
from .terminations import is_above, motion_ended
