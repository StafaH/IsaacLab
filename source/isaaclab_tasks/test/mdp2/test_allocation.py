# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the allocators of the MDP2 multi-interface tracking task, on synthetic competence curves."""

import numpy as np

from isaaclab_tasks.mdp2.interfaces.allocation import Allocator


def test_progress_attracts_worlds_and_closed_or_stalled_arms_do_not():
    # Arm 0 is the anchor and is mastered early. Arm 1 never improves, arm 2 rises, arm 3 opens after arm 2,
    # and arm 4 waits for arm 1, which never gets there.
    allocator = Allocator(parents=[[], [0], [0], [2], [1]])
    assert allocator.shares.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]

    competence = np.zeros(5)
    for step in range(100):
        competence[0] = min(1.0, 0.1 * step)
        competence[2] = min(0.9, 0.02 * max(step - 30, 0))
        competence[3] = 0.6 if step > 85 else 0.0
        shares = allocator.update(competence)
        if step == 50:
            # Arm 3 stays closed until its parent is competent.
            assert shares[3] == 0.0
        if step == 70:
            # The rising arm has the largest share, the stalled one its floor, and the anchor its minimum.
            assert shares[2] == shares.max() and shares[1] < 0.1 and shares[0] >= 0.2
    assert abs(shares.sum() - 1.0) < 1.0e-6
    assert shares[3] == shares.max() and shares[4] == 0.0

    # Once nothing changes, an arm that is being forgotten regains worlds.
    for step in range(100):
        shares = allocator.update(competence)
    before = shares[2]
    for step in range(8):
        competence[2] -= 0.05
        shares = allocator.update(competence)
    assert shares[2] > before


def test_frontier_gives_the_worlds_to_the_arm_whose_outcome_is_open():
    # Arm 0 is mastered, arm 1 succeeds on half of its tries, and arm 2 never succeeds.
    allocator = Allocator(parents=[[], [], []], anchors=(), rule="frontier")
    for _ in range(50):
        shares = allocator.update([0.95, 0.6, 0.1], frontier=[0.02, 0.9, 0.0])
    assert abs(shares.sum() - 1.0) < 1.0e-6
    assert shares[1] > 0.8 and 0.05 <= shares[2] < shares[0] < 0.1


def test_headroom_gives_the_worlds_to_the_weak_arm_that_moves_and_not_to_the_one_that_is_stuck():
    # Arm 0 is mastered, arm 1 is weak and improves slowly, and arm 2 is weak and stays where it is.
    allocator = Allocator(parents=[[], [], []], anchors=(), rule="headroom")
    for step in range(100):
        shares = allocator.update([0.95, 0.3 + 0.002 * step, 0.2])
    assert shares[1] > 0.8 and shares[0] < 0.1 and shares[2] < 0.1
