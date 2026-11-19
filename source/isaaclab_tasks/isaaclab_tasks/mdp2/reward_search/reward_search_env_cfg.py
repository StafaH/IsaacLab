# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward search: the worlds are split between candidates that weight the rewards of one task differently.

Every candidate has the terms of the task, so the worlds share one set of kernels, and a candidate is a row of the
per-world reward weights, which are state: a learner can also change them while it trains, as population-based
training does. Each candidate is meant to train its own policy; what they are ranked by is up to the learner.
"""

from isaaclab_experimental import mdp2

from isaaclab.utils import replace


def reward_candidates(task: mdp2.TaskCfg, weights: list[dict[str, float]]) -> list[mdp2.TaskCfg]:
    """Return one copy of ``task`` per candidate, with the reward weights that the candidate gives by term name."""
    return [
        replace(
            task,
            rewards={
                name: replace(reward, weight=candidate.get(name, reward.weight))
                for name, reward in task.rewards.items()
            },
        )
        for candidate in weights
    ]
