# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distillation: a teacher that sees privileged observations and a student that does not, in one simulation.

The teacher worlds compute only the privileged observations. The student worlds also compute the student's, so
that a learner can label the student's states with the teacher's actions while the teacher is still learning.
The worlds share everything else of the task.
"""

from isaaclab_experimental import mdp2

from isaaclab.utils import replace


def teacher_student_tasks(
    task: mdp2.TaskCfg, privileged: mdp2.ObservationGroup, student: mdp2.ObservationGroup
) -> list[mdp2.TaskCfg]:
    """Return task for the teacher worlds, observing privileged, and for the student worlds, which also
    observe student."""
    teacher = replace(task, observations={"privileged": privileged})
    return [teacher, replace(task, observations={"privileged": privileged, "student": student})]
