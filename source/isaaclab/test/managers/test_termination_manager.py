# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch

from isaaclab.managers import TerminationManager, TerminationTermCfg
from isaaclab.sim import SimulationContext

pytestmark = pytest.mark.integration


class DummyEnv:
    """Minimal mutable env stub for the termination manager tests."""

    def __init__(self, num_envs: int, device: str, sim: SimulationContext):
        self.num_envs = num_envs
        self.device = device
        self.sim = sim
        self.counter = 0  # mutable step counter used by test terms


def fail_every_5_steps(env) -> torch.Tensor:
    """Returns True for all envs when counter is a positive multiple of 5."""
    cond = env.counter > 0 and (env.counter % 5 == 0)
    return torch.full((env.num_envs,), cond, dtype=torch.bool, device=env.device)


def fail_every_10_steps(env) -> torch.Tensor:
    """Returns True for all envs when counter is a positive multiple of 10."""
    cond = env.counter > 0 and (env.counter % 10 == 0)
    return torch.full((env.num_envs,), cond, dtype=torch.bool, device=env.device)


def fail_every_3_steps(env) -> torch.Tensor:
    """Returns True for all envs when counter is a positive multiple of 3."""
    cond = env.counter > 0 and (env.counter % 3 == 0)
    return torch.full((env.num_envs,), cond, dtype=torch.bool, device=env.device)


def fail_even_envs(env) -> torch.Tensor:
    """Returns True for even-numbered environments."""
    return torch.arange(env.num_envs, device=env.device) % 2 == 0


def fail_first_half(env) -> torch.Tensor:
    """Returns True for environments in the first half of the batch."""
    return torch.arange(env.num_envs, device=env.device) < env.num_envs // 2


@pytest.fixture
def env():
    sim = SimulationContext()
    return DummyEnv(num_envs=20, device="cpu", sim=sim)


def test_initial_state_and_shapes(env):
    cfg = {
        "term_5": TerminationTermCfg(func=fail_every_5_steps),
        "term_10": TerminationTermCfg(func=fail_every_10_steps),
    }
    tm = TerminationManager(cfg, env)

    # Active term names
    assert tm.active_terms == ["term_5", "term_10"]

    # Internal buffers have expected shapes and start as all False
    assert tm._term_dones.shape == (env.num_envs, 2)
    assert tm._last_episode_dones.shape == (env.num_envs, 2)
    assert tm.dones.shape == (env.num_envs,)
    assert tm.time_outs.shape == (env.num_envs,)
    assert tm.terminated.shape == (env.num_envs,)
    assert torch.all(~tm._term_dones) and torch.all(~tm._last_episode_dones)


def test_term_transitions_and_persistence(env):
    """Concise transitions: single fire, persist, switch, both, persist.

    Uses 3-step and 5-step terms and verifies current-step values and last-episode persistence.
    """
    cfg = {
        "term_3": TerminationTermCfg(func=fail_every_3_steps, time_out=False),
        "term_5": TerminationTermCfg(func=fail_every_5_steps, time_out=False),
    }
    tm = TerminationManager(cfg, env)

    # step 3: only term_3 -> last_episode [True, False]
    env.counter = 3
    out = tm.compute()
    tm.record_episode_terminations(out.nonzero(as_tuple=True)[0])
    assert torch.all(tm.get_term("term_3")) and torch.all(~tm.get_term("term_5"))
    assert torch.all(out)
    assert torch.all(tm._last_episode_dones[:, 0]) and torch.all(~tm._last_episode_dones[:, 1])

    # step 4: none -> last_episode persists [True, False]
    env.counter = 4
    out = tm.compute()
    assert torch.all(~out)
    assert torch.all(~tm.get_term("term_3")) and torch.all(~tm.get_term("term_5"))
    assert torch.all(tm._last_episode_dones[:, 0]) and torch.all(~tm._last_episode_dones[:, 1])

    # step 5: only term_5 -> last_episode [False, True]
    env.counter = 5
    out = tm.compute()
    tm.record_episode_terminations(out.nonzero(as_tuple=True)[0])
    assert torch.all(~tm.get_term("term_3")) and torch.all(tm.get_term("term_5"))
    assert torch.all(out)
    assert torch.all(~tm._last_episode_dones[:, 0]) and torch.all(tm._last_episode_dones[:, 1])

    # step 15: both -> last_episode [True, True]
    env.counter = 15
    out = tm.compute()
    tm.record_episode_terminations(out.nonzero(as_tuple=True)[0])
    assert torch.all(tm.get_term("term_3")) and torch.all(tm.get_term("term_5"))
    assert torch.all(out)
    assert torch.all(tm._last_episode_dones[:, 0]) and torch.all(tm._last_episode_dones[:, 1])

    # step 16: none -> persist [True, True]
    env.counter = 16
    out = tm.compute()
    assert torch.all(~out)
    assert torch.all(~tm.get_term("term_3")) and torch.all(~tm.get_term("term_5"))
    assert torch.all(tm._last_episode_dones[:, 0]) and torch.all(tm._last_episode_dones[:, 1])


def test_last_episode_dones_update_only_terminated_environments(env):
    cfg = {
        "even": TerminationTermCfg(func=fail_even_envs),
        "first_half": TerminationTermCfg(func=fail_first_half),
    }
    tm = TerminationManager(cfg, env)
    tm._last_episode_dones[:, 1] = True

    tm.compute()
    tm.record_episode_terminations(tm.dones.nonzero(as_tuple=True)[0])

    expected = torch.zeros_like(tm._last_episode_dones)
    expected[:, 1] = True
    current_dones = torch.stack((fail_even_envs(env), fail_first_half(env)), dim=1)
    expected[current_dones.any(dim=1)] = current_dones[current_dones.any(dim=1)]
    assert torch.equal(tm._last_episode_dones, expected)

    # Explicit/manual resets are not termination events and must not overwrite the last episode cause.
    previous = tm._last_episode_dones.clone()
    tm._term_dones.zero_()
    tm.reset(torch.tensor([1, 3], device=env.device))
    assert torch.equal(tm._last_episode_dones, previous)


def test_reset_records_current_terminations_for_direct_manager_callers(env):
    """Direct compute/reset use preserves episode-cause recording without an environment loop."""
    tm = TerminationManager({"term_3": TerminationTermCfg(func=fail_every_3_steps)}, env)

    env.counter = 3
    tm.compute()
    extras = tm.reset(torch.arange(env.num_envs, device=env.device))

    assert torch.all(tm._last_episode_dones)
    assert extras["Episode_Termination/term_3"] == 1.0


def test_time_out_vs_terminated_split(env):
    cfg = {
        "term_5": TerminationTermCfg(func=fail_every_5_steps, time_out=False),  # terminated
        "term_10": TerminationTermCfg(func=fail_every_10_steps, time_out=True),  # timeout
    }
    tm = TerminationManager(cfg, env)

    # Step 5: terminated fires, not timeout
    env.counter = 5
    out = tm.compute()
    assert torch.all(out)
    assert torch.all(tm.terminated) and torch.all(~tm.time_outs)

    # Step 10: both fire; timeout and terminated both True
    env.counter = 10
    out = tm.compute()
    assert torch.all(out)
    assert torch.all(tm.terminated) and torch.all(tm.time_outs)
