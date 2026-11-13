# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Allocators: rules that turn a signal per arm into the shares of the training worlds, without a simulation.

An arm is a variant of the MDP that a world can run, such as an interface. A learner calls :meth:`update` with a
signal per arm at a fixed period and writes the returned shares to the task.
"""

from __future__ import annotations

import numpy as np


class Allocator:
    """Shares from the competence of every arm, a number in [0, 1] that the evaluation worlds measure.

    The priority of an arm is its absolute learning progress, the distance between a fast and a slow average of its
    competence: an arm that improves and an arm that is being forgotten both attract worlds, and an arm that stays
    where it is, mastered or impossible, does not. ``rule`` replaces it by ``"failure"``, one minus the competence,
    by ``"headroom"``, one minus the competence for an arm whose progress reaches ``moving`` and proportionally less
    for an arm that moves less, by ``"frontier"``, a priority that the caller passes to :meth:`update`, or by
    ``"uniform"``, the same priority for every open arm.

    An arm is open once every arm in its ``parents`` has a competence of ``open_at``, and closes when one falls
    below ``close_at``; arms without parents are always open. Every open arm keeps the share ``floor``, the arms in
    ``anchors`` keep ``anchor_share``, and the rest is divided by priority. The shares move towards that target by
    ``rate`` per update.
    """

    def __init__(
        self,
        parents: list[list[int]],
        anchors: tuple[int, ...] = (0,),
        rule: str = "progress",
        fast: float = 0.3,
        slow: float = 0.05,
        open_at: float = 0.5,
        close_at: float = 0.4,
        floor: float = 0.05,
        anchor_share: float = 0.2,
        rate: float = 0.2,
        epsilon: float = 0.002,
        moving: float = 0.01,
    ):
        self.parents, self.rule, self.moving = parents, rule, moving
        self.fast_rate, self.slow_rate, self.rate, self.epsilon = fast, slow, rate, epsilon
        self.open_at, self.close_at = open_at, close_at
        count = len(parents)
        self.floors = np.full(count, floor)
        self.floors[list(anchors)] = np.maximum(anchor_share, floor)
        self.fast, self.slow, self.frontier = None, None, None
        self.open = np.array([not arm_parents for arm_parents in parents])
        self.shares = self.target()

    def target(self) -> np.ndarray:
        """Return the shares that the open arms should have now."""
        if self.fast is None or self.rule == "uniform":
            priority = np.ones(len(self.parents))
        elif self.rule == "failure":
            priority = 1.0 - self.fast
        elif self.rule == "headroom":
            priority = (1.0 - self.fast) * np.minimum(np.abs(self.fast - self.slow) / self.moving, 1.0)
        elif self.rule == "frontier":
            priority = self.frontier
        else:
            priority = np.abs(self.fast - self.slow)
        # A small constant, so that open arms share the worlds when no arm has any priority.
        priority = np.where(self.open, priority + self.epsilon, 0.0)
        floors = np.where(self.open, self.floors, 0.0)
        floors = floors / max(floors.sum(), 1.0)
        return floors + (1.0 - floors.sum()) * priority / priority.sum()

    def update(self, competence, frontier=None) -> np.ndarray:
        """Take the competence of every arm, and its priority for the rule ``"frontier"``, and return the new shares."""
        competence = np.asarray(competence, dtype=float)
        self.frontier = None if frontier is None else np.asarray(frontier, dtype=float)
        if self.fast is None:
            self.fast, self.slow = competence.copy(), competence.copy()
        self.fast += self.fast_rate * (competence - self.fast)
        self.slow += self.slow_rate * (competence - self.slow)
        for arm, arm_parents in enumerate(self.parents):
            if arm_parents:
                level = self.close_at if self.open[arm] else self.open_at
                self.open[arm] = all(self.fast[parent] >= level for parent in arm_parents)
        self.shares = (1.0 - self.rate) * self.shares + self.rate * self.target()
        self.shares = np.where(self.open | (self.shares > 1.0e-3), self.shares, 0.0)
        self.shares /= self.shares.sum()
        return self.shares


class TSCL:
    """Teacher-student curriculum learning (arXiv 1707.00183), the online algorithm, for arms sampled in parallel.

    The score of an arm is a running average of the change of its signal between updates, and the arm with the
    largest absolute score gets the worlds, except for the share ``explore`` that is spread over all arms.
    """

    def __init__(self, count: int, rate: float = 0.1, explore: float = 0.1):
        self.rate, self.explore = rate, explore
        self.score, self.last = np.zeros(count), None
        self.shares = np.full(count, 1.0 / count)

    def update(self, signal) -> np.ndarray:
        signal = np.asarray(signal, dtype=float)
        if self.last is not None:
            self.score += self.rate * (signal - self.last - self.score)
        self.last = signal
        self.shares = np.full(len(signal), self.explore / len(signal))
        self.shares[np.abs(self.score).argmax()] += 1.0 - self.explore
        return self.shares


class ALP:
    """Absolute learning progress on discrete arms, as the teachers of arXiv 1910.07224 use it.

    The progress of an arm is the absolute change of its signal over ``window`` updates. A share ``explore`` of the
    worlds is spread over all arms and the rest in proportion to the progress.
    """

    def __init__(self, count: int, window: int = 10, explore: float = 0.2):
        self.window, self.explore = window, explore
        self.history = []
        self.shares = np.full(count, 1.0 / count)

    def update(self, signal) -> np.ndarray:
        self.history = [*self.history, np.asarray(signal, dtype=float)][-2 * self.window :]
        if len(self.history) >= 2:
            half = len(self.history) // 2
            progress = np.abs(np.mean(self.history[half:], axis=0) - np.mean(self.history[:half], axis=0))
            uniform = np.full(len(progress), 1.0 / len(progress))
            proportional = progress / progress.sum() if progress.sum() > 0.0 else uniform
            self.shares = self.explore * uniform + (1.0 - self.explore) * proportional
        return self.shares


class RankedScore:
    """The rank prioritization of prioritized level replay (arXiv 2010.03934), with arms in place of levels.

    The signal is a score per arm, such as the mean absolute advantage of its samples. The share of an arm is
    proportional to ``(1 / rank) ** (1 / temperature)``, with rank 1 for the largest score. A share ``explore``
    is spread over all arms, which stands for the staleness term of the paper: every arm is sampled in every update
    here, so none is stale.
    """

    def __init__(self, count: int, temperature: float = 0.3, explore: float = 0.1, rate: float = 0.3):
        self.temperature, self.explore, self.rate = temperature, explore, rate
        self.score = None
        self.shares = np.full(count, 1.0 / count)

    def update(self, signal) -> np.ndarray:
        signal = np.asarray(signal, dtype=float)
        self.score = signal.copy() if self.score is None else self.score + self.rate * (signal - self.score)
        rank = np.empty(len(signal))
        rank[np.argsort(-self.score)] = np.arange(1, len(signal) + 1)
        weight = (1.0 / rank) ** (1.0 / self.temperature)
        self.shares = self.explore / len(signal) + (1.0 - self.explore) * weight / weight.sum()
        return self.shares
