# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-skill tasks: the worlds are split between skills, and one policy is told which skill each world practices.

A skill is a task of its own: its rewards, terminations, commands, and resets. Every skill observes the same
values plus a ``skill`` command, one at the index of the skill, so one policy can learn them all.
"""

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import mdp

from isaaclab.utils import replace

from . import mdp as skills_mdp


def skill_tasks(skills: dict[str, mdp2.TaskCfg], group: str = "policy") -> list[mdp2.TaskCfg]:
    """Return the task of each skill with the ``skill`` command, observed at the end of ``group``."""
    tasks = []
    for index, task in enumerate(skills.values()):
        command = mdp2.Command(skills_mdp.one_hot_command, params={"index": index, "skills": len(skills)})
        observation = mdp2.Observation(mdp.generated_commands, inputs={"command": mdp2.commands.skill})
        terms = {**task.observations[group].terms, "skill": observation}
        tasks.append(
            replace(
                task,
                commands={**task.commands, "skill": command},
                observations={**task.observations, group: mdp2.ObservationGroup(**terms)},
            )
        )
    return tasks
