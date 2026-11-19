# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward search on humanoid motion tracking: sixteen weightings of the tracking rewards in one simulation."""

import random

from isaaclab_experimental import mdp2

from isaaclab.utils import configclass

from isaaclab_tasks.mdp2.reward_search.reward_search_env_cfg import reward_candidates
from isaaclab_tasks.mdp2.tracking.config.humanoid.humanoid_env_cfg import HumanoidTrackingEnvCfg, humanoid_task

BASE_WEIGHTS = {"joint_pos": 15.0, "joint_vel": 3.0, "body_pos": 9.0, "root_vel": 3.0}
"""Weights of the tracking rewards in the tracking task."""


def humanoid_candidates(motion: str, seed: int = 0) -> list[dict[str, float]]:
    """Return sixteen weightings of the tracking rewards: the task's own, each term dropped, each term doubled, and
    seven with every weight scaled by a random factor between a quarter and four."""
    generator = random.Random(seed)
    candidates = [dict(BASE_WEIGHTS)]
    candidates += [{**BASE_WEIGHTS, name: 0.0} for name in BASE_WEIGHTS]
    candidates += [{**BASE_WEIGHTS, name: 2.0 * weight} for name, weight in BASE_WEIGHTS.items()]
    candidates += [
        {name: weight * 4.0 ** generator.uniform(-1.0, 1.0) for name, weight in BASE_WEIGHTS.items()} for _ in range(7)
    ]
    return candidates


CANDIDATES = humanoid_candidates("walk")
"""The reward weights of each candidate, by term name."""


@configclass
class HumanoidRewardSearchEnvCfg(HumanoidTrackingEnvCfg):
    """The humanoid tracking environment with its worlds split between sixteen reward candidates."""

    task: list[mdp2.TaskCfg] = reward_candidates(humanoid_task("walk"), CANDIDATES)
