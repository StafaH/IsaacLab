# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher and student for Unitree Go2 velocity tracking on flat ground.

The teacher observes the task's observations without noise. The student observes what a robot can measure: no base
linear velocity, the task's noise on the rest, and the last five steps of it.
"""

from isaaclab_experimental import mdp2

from isaaclab.utils import configclass, replace

from isaaclab_tasks.mdp2.distillation.distillation_env_cfg import teacher_student_tasks
from isaaclab_tasks.mdp2.velocity_go2.flat_env_cfg import TASK, UnitreeGo2FlatEnvCfg

HISTORY_LENGTH = 5

terms = TASK.observations["policy"].terms
PRIVILEGED = mdp2.ObservationGroup(**{name: replace(term, noise=None) for name, term in terms.items()})
"""The observations of the task, without noise."""

STUDENT = mdp2.ObservationGroup(
    **{name: replace(term, history_length=HISTORY_LENGTH) for name, term in terms.items() if name != "base_lin_vel"}
)
"""The noisy observations of the task without the base linear velocity, over the last steps."""


@configclass
class UnitreeGo2DistillationEnvCfg(UnitreeGo2FlatEnvCfg):
    """The Go2 flat environment with teacher worlds and student worlds, half of each."""

    task: list[mdp2.TaskCfg] = teacher_student_tasks(TASK, PRIVILEGED, STUDENT)
