# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity of the MDP2 Unitree G1 flat velocity task with MDP1's ``Isaac-Velocity-Flat-G1`` on Newton MJWarp."""

from dataclasses import replace

import pytest
import torch
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv, Stage, Step

from isaaclab.managers import ManagerTermBaseCfg, SceneEntityCfg

from isaaclab_tasks.mdp2.velocity_g1.flat_env_cfg import TASK, UnitreeG1FlatEnvCfg
from isaaclab_tasks.utils import resolve_task_config

pytestmark = [pytest.mark.kitless, pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")]

NUM_ENVS = 32
NUM_STEPS = 150
AXES = ("x", "y", "z", "roll", "pitch", "yaw")


@pytest.fixture
def mdp1_cfg():
    cfg, _ = resolve_task_config("Isaac-Velocity-Flat-G1", None, overrides=["physics=newton_mjwarp"])
    return cfg


def terms(group) -> dict:
    """Return the terms of an MDP1 manager config, without the removed ones."""
    return {name: term for name, term in vars(group).items() if isinstance(term, ManagerTermBaseCfg)}


def ranges(cfg: dict[str, tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Return an MDP1 pose or velocity range dict as the MDP2 tuple of ``(low, high)`` per axis."""
    return tuple(cfg.get(axis, (0.0, 0.0)) for axis in AXES)


def test_config_matches_mdp1(mdp1_cfg):
    cfg = UnitreeG1FlatEnvCfg()
    physics = cfg.sim.physics.newton_mjwarp

    # Simulation and scene.
    assert physics.to_dict() == mdp1_cfg.sim.physics.to_dict()
    assert (cfg.sim.dt, cfg.decimation, cfg.sim.render_interval) == (
        mdp1_cfg.sim.dt,
        mdp1_cfg.decimation,
        mdp1_cfg.sim.render_interval,
    )
    assert cfg.sim.use_newton_actuators == mdp1_cfg.sim.use_newton_actuators
    assert cfg.sim.physics_material.to_dict() == mdp1_cfg.sim.physics_material.to_dict()
    assert cfg.scene.robot.to_dict() == mdp1_cfg.scene.robot.to_dict()
    assert cfg.scene.terrain.physics_material.to_dict() == mdp1_cfg.scene.terrain.physics_material.to_dict()
    sensor, mdp1_sensor = cfg.scene.contact_forces, mdp1_cfg.scene.contact_forces
    assert (sensor.prim_path, sensor.history_length, sensor.track_air_time, sensor.update_period) == (
        mdp1_sensor.prim_path,
        mdp1_sensor.history_length,
        mdp1_sensor.track_air_time,
        mdp1_sensor.update_period,
    )

    # Actions, commands, and terms.
    action, mdp1_action = TASK.actions["joint_pos"], mdp1_cfg.actions.joint_pos
    assert action.scale == mdp1_action.scale and mdp1_action.use_default_offset
    command, mdp1_command = TASK.commands["base_velocity"].params, mdp1_cfg.commands.base_velocity
    assert mdp1_command.heading_command and TASK.commands["base_velocity"].resampling_time_range == (
        mdp1_command.resampling_time_range
    )
    assert command == {
        "lin_vel_x": mdp1_command.ranges.lin_vel_x,
        "lin_vel_y": mdp1_command.ranges.lin_vel_y,
        "ang_vel_z": mdp1_command.ranges.ang_vel_z,
        "heading": mdp1_command.ranges.heading,
        "rel_standing_envs": mdp1_command.rel_standing_envs,
        "rel_heading_envs": mdp1_command.rel_heading_envs,
        "heading_control_stiffness": mdp1_command.heading_control_stiffness,
    }

    assert list(TASK.observations["policy"]) == list(terms(mdp1_cfg.observations.policy))
    for name, term in terms(mdp1_cfg.observations.policy).items():
        noise = TASK.observations["policy"][name].noise
        assert (noise is None) == (term.noise is None), name
        if noise is not None:
            assert (noise.n_min, noise.n_max) == (term.noise.n_min, term.noise.n_max), name

    rewards = {name: reward.weight for name, reward in TASK.rewards.items() if reward is not None}
    assert rewards == {name: term.weight for name, term in terms(mdp1_cfg.rewards).items()}

    terminations = {name for name, term in TASK.terminations.items() if term is not None}
    assert terminations == set(terms(mdp1_cfg.terminations))
    assert TASK.terminations["time_out"].episode_length_s == mdp1_cfg.episode_length_s

    # Events; MDP1's zero external force applies no force, and Newton has no dynamic friction or material buckets.
    events = terms(mdp1_cfg.events)
    assert set(TASK.events) == set(events) - {"base_external_force_torque"}
    assert events["base_external_force_torque"].params["force_range"] == (0.0, 0.0)
    assert events["base_external_force_torque"].params["torque_range"] == (0.0, 0.0)
    for name, event in TASK.events.items():
        mdp1_event = events[name]
        assert event.mode.name.lower() == mdp1_event.mode, name
        assert event.interval_range_s == mdp1_event.interval_range_s, name
    material = events["physics_material"].params
    assert TASK.events["physics_material"].params == {
        "friction_range": material["static_friction_range"],
        "restitution_range": material["restitution_range"],
    }
    params = events["add_base_mass"].params
    assert TASK.events["add_base_mass"].params == {
        name: params[name] for name in ("mass_distribution_params", "operation", "distribution")
    }
    params = events["reset_base"].params
    assert TASK.events["reset_base"].params == {
        "pose_range": ranges(params["pose_range"]),
        "velocity_range": ranges(params["velocity_range"]),
    }
    assert TASK.events["reset_robot_joints"].params == events["reset_robot_joints"].params
    velocity_range = ranges(events["push_robot"].params["velocity_range"])
    assert TASK.events["push_robot"].params == {"velocity_range": velocity_range}
    assert not terms(mdp1_cfg.curriculum)


def relative_difference(value: torch.Tensor, expected: torch.Tensor) -> float:
    """Return the largest difference of two values relative to the largest magnitude of the expected one."""
    return ((value - expected).abs().max() / expected.abs().max().clamp(min=1e-6)).item()


class MDP1View:
    """What the MDP1 terms read from an environment, from an MDP2 environment."""

    def __init__(self, env: MDP2RLEnv):
        self.scene = env.scene
        self.step_dt = env.step_dt
        self.command_manager = self.action_manager = self.termination_manager = self
        self.command = self.action = self.prev_action = self.terminated = None

    def get_command(self, name: str) -> torch.Tensor:
        return self.command


def resolve(terms: dict, env: MDP2RLEnv) -> None:
    """Resolve the scene entities of the params of MDP1 terms."""
    for term in terms.values():
        for value in term.params.values():
            if isinstance(value, SceneEntityCfg):
                value.resolve(env.scene)


@pytest.mark.parametrize("backend", ["torch", "warp"])
def test_terms_match_mdp1_on_the_same_states(mdp1_cfg, backend):
    """The rewards, observations, and terminations of rollouts match MDP1's terms on the same states.

    A stage after the evaluation, before the resets, evaluates the MDP1 terms on the state that the MDP2 terms saw.
    Observations have no noise, and there are no pushes, which run after the rewards and change the state.
    """
    task = replace(
        TASK,
        observations={"policy": {name: replace(use, noise=None) for name, use in TASK.observations["policy"].items()}},
        events={**TASK.events, "push_robot": None},
    )
    cfg = UnitreeG1FlatEnvCfg(task=task, backend=mdp2.Backend(backend), seed=0)
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = NUM_ENVS

    rewards, observations = terms(mdp1_cfg.rewards), terms(mdp1_cfg.observations.policy)
    base_contact = mdp1_cfg.terminations.base_contact
    differences = dict.fromkeys([*rewards, *observations, "base_contact"], 0.0)
    nonzero = dict.fromkeys(rewards, False)
    terminated = []

    def compare(env: MDP2RLEnv) -> None:
        transition = env.transition
        view.terminated = transition.terminated.torch
        for name, term in rewards.items():
            # The rewards saw the command before the step updated it.
            view.command = command
            # Unweighted, so that the tolerance is relative to the term, up to the rounding of the weighting.
            value = term.func(view, **term.params).float()
            mdp2_value = transition.reward_terms.torch[:, reward_index[name]] / (term.weight * env.step_dt)
            torch.testing.assert_close(mdp2_value, value, rtol=1e-5, atol=1e-5, msg=name)
            differences[name] = max(differences[name], relative_difference(mdp2_value, value))
            nonzero[name] |= bool(value.abs().sum() > 0.0)

        view.command = env.state.arrays[mdp2.commands.base_velocity].torch
        values = [term.func(view, **term.params) for term in observations.values()]
        mdp2_values = transition.observations["policy"].torch.split([value.shape[1] for value in values], -1)
        for name, value, mdp2_value in zip(observations, values, mdp2_values):
            torch.testing.assert_close(mdp2_value, value, msg=name)
            differences[name] = max(differences[name], relative_difference(mdp2_value, value))

        value = base_contact.func(view, **base_contact.params)
        assert torch.equal(transition.termination_terms.torch[:, termination_index["base_contact"]], value)
        terminated.append(value)

    cfg.step = [Step.PHYSICS, Step.EVALUATE, Stage(compare, capturable=False), Step.RESET]
    env = MDP2RLEnv(cfg)
    try:
        view = MDP1View(env)
        resolve({**rewards, **observations, "base_contact": base_contact}, env)
        reward_index = {op.name.split(".", 1)[1]: op.index for op in env.mdp.ops[mdp2.Phase.REWARDS]}
        termination_index = {op.name.split(".", 1)[1]: op.index for op in env.mdp.ops[mdp2.Phase.TERMINATIONS]}
        assert set(reward_index) == set(rewards)

        generator = torch.Generator(device=env.device).manual_seed(0)
        env.reset()
        command = env.state.arrays[mdp2.commands.base_velocity].torch.clone()
        view.action = torch.zeros(NUM_ENVS, env.mdp.action_dim, device=env.device)
        for _ in range(NUM_STEPS):
            view.prev_action = view.action
            view.action = 2.0 * torch.rand(NUM_ENVS, env.mdp.action_dim, device=env.device, generator=generator) - 1.0
            env.step(view.action)
            # The commands and actions of the worlds that reset start over.
            command = env.state.arrays[mdp2.commands.base_velocity].torch.clone()
            view.action = torch.where(env.transition.done.torch[:, None], 0.0, view.action)
    finally:
        env.close()

    print(f"\nMax relative differences, {backend}:", {name: f"{value:.2e}" for name, value in differences.items()})
    # The rollouts exercise the terms: every reward is nonzero somewhere, and the torso touches the ground.
    assert all(nonzero.values()), [name for name, value in nonzero.items() if not value]
    terminated = torch.stack(terminated)
    assert terminated.any() and not terminated.all()
