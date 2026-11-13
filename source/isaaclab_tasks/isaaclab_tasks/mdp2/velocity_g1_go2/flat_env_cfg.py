# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity tracking of a Unitree G1 in half of the worlds and of a Unitree Go2 in the other half.

The worlds differ in their scene and in their task. The scene declares both robots and the combinations that a
world may hold, one robot each. MuJoCo Warp simulates worlds of one structure, so every world holds both robots and
keeps the one outside its combination always asleep, see
:attr:`~isaaclab_newton.physics.MJWarpSolverCfg.sleep_inactive_assets`. Each half then runs the task of its robot:
the tasks of :mod:`..velocity_g1` and :mod:`..velocity_go2` as they are, bound to the robots of this scene.
"""

import dataclasses

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG, UNITREE_GO2_CFG

from ..velocity_g1.flat_env_cfg import DECIMATION, GROUND_MATERIAL, SIM_DT
from ..velocity_g1.flat_env_cfg import TASK as G1_TASK
from ..velocity_go2.flat_env_cfg import TASK as GO2_TASK
from ..velocity_style.config.g1.g1_env_cfg import G1_STYLE_TASKS
from ..velocity_style.config.go2.go2_env_cfg import GO2_STYLE_TASKS

CONFIGS = (mdp2.TaskCfg, mdp2.TermCfg, mdp2.DirectAction, mdp2.UniformNoise)


def bound(config, entities: dict[str, str]):
    """Return a task configuration that reads and writes other scene entities, as in ``{"robot": "go2"}``."""
    if isinstance(config, (mdp2.FieldSelector, mdp2.SubsetSelector)):
        return dataclasses.replace(config, entity=entities.get(config.entity, config.entity))
    if isinstance(config, dict):
        return {name: bound(value, entities) for name, value in config.items()}
    if isinstance(config, mdp2.ObservationGroup):
        return mdp2.ObservationGroup(**bound(config.terms, entities))
    if isinstance(config, CONFIGS):
        fields = {field.name: bound(getattr(config, field.name), entities) for field in dataclasses.fields(config)}
        return dataclasses.replace(config, **fields)
    return config


@configclass
class G1Go2FlatSceneCfg(InteractiveSceneCfg):
    """A ground plane with a Unitree G1 or a Unitree Go2 in each world, and contact sensors on their bodies."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=GROUND_MATERIAL,
    )
    g1: ArticulationCfg = replace(G1_MINIMAL_CFG, prim_path="{ENV_REGEX_NS}/G1")
    go2: ArticulationCfg = replace(UNITREE_GO2_CFG, prim_path="{ENV_REGEX_NS}/Go2")
    g1_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/G1/[^/]*", history_length=3, track_air_time=True, update_period=SIM_DT
    )
    go2_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Go2/[^/]*", history_length=3, track_air_time=True, update_period=SIM_DT
    )
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))
    # A world holds one of the robots: the first half of the worlds the G1, the second half the Go2.
    clone_cfg: CloneCfg = CloneCfg(clone_combinations=[InclusionSet(assets=["g1"]), InclusionSet(assets=["go2"])])


@configclass
class G1Go2FlatPhysicsCfg(PresetCfg):
    """Newton physics presets of the task. The robot that a world does not hold sleeps in it."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=150,
            nconmax=60,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            sleep_inactive_assets=True,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp


def robot_task(task: mdp2.TaskCfg, name: str) -> mdp2.TaskCfg:
    """Return the task of a robot for this scene, with its rewards, terminations, and events named after the robot.

    Uses of one name share their state and their log entry between the tasks, which the robots cannot: their
    actions and their clips differ in size.
    """
    entities = {"robot": name, "contact_forces": f"{name}_contact_forces"}
    entities |= {f"events.{event}": f"events.{name}_{event}" for event in task.events}
    task = bound(task, entities)
    rewards = {f"{name}_{term}": reward for term, reward in task.rewards.items()}
    terminations = {f"{name}_{term}": termination for term, termination in task.terminations.items()}
    events = {f"{name}_{term}": event for term, event in task.events.items()}
    return replace(task, rewards=rewards, terminations=terminations, events=events)


G1_GO2_TASKS = [robot_task(G1_TASK, "g1"), robot_task(GO2_TASK, "go2")]
"""The MDP of the worlds with a G1 and the MDP of the worlds with a Go2."""

G1_GO2_STYLE_TASKS = [
    *(robot_task(task, "g1") for task in G1_STYLE_TASKS),
    *(robot_task(task, "go2") for task in GO2_STYLE_TASKS),
]
"""The MDPs of the worlds with a G1 and of the worlds with a Go2, two each: the style task and the velocity task."""


@configclass
class G1Go2FlatEnvCfg(MDP2RLEnvCfg):
    """Configuration of the environment where half of the worlds hold a G1 and the other half a Go2."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics=G1Go2FlatPhysicsCfg(),
        physics_material=GROUND_MATERIAL,
        use_newton_actuators=True,
    )
    scene: G1Go2FlatSceneCfg = G1Go2FlatSceneCfg(num_envs=8192, env_spacing=2.5)
    task: list[mdp2.TaskCfg] = G1_GO2_TASKS
    decimation: int = DECIMATION
