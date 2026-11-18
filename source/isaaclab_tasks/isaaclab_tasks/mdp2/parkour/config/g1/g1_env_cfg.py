# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 parkour: the 29-DoF G1 climbs the boxes of the terrain clips of the OmniRetarget dataset."""

import json
import math
import os

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.mdp2.parkour import mdp as parkour
from isaaclab_tasks.mdp2.parkour.parkour_env_cfg import parkour_task
from isaaclab_tasks.mdp2.tracking.config.g1.g1_env_cfg import G1TrackingSceneCfg
from isaaclab_tasks.utils import PresetCfg

TRACKING_ROBOT = G1TrackingSceneCfg().robot

MOTIONS_DIR = os.path.join(os.path.dirname(__file__), "motions")
"""Folder of the converted clips, which are downloaded, see its README."""

CLIPS = tuple(f"climb_{clip:02d}" for clip in range(29))
LEVELS = ("0.8", "0.9", "1.0", "1.1", "1.2")
"""Scales of the height of the boxes: every clip comes retargeted to each."""

SIM_DT = 0.005
DECIMATION = 4
NUM_BODIES = 46

# The reflected inertia of the motors, and gains that give every joint a natural frequency of 10 Hz with a damping
# ratio of two on that inertia, as trackers of this robot on hardware do.
_GROUPS = {
    (".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"): 0.01018,
    (".*_hip_roll_joint", ".*_knee_joint"): 0.0251,
    (".*_ankle_.*", "waist_roll_joint", "waist_pitch_joint"): 0.00722,
    (".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_roll_joint"): 0.00361,
    (".*_wrist_pitch_joint", ".*_wrist_yaw_joint"): 0.00425,
    (".*_hand_.*",): 0.001,
}
_OMEGA = 2.0 * math.pi * 10.0
ARMATURE = {joint: inertia for joints, inertia in _GROUPS.items() for joint in joints}
STIFFNESS = {joint: inertia * _OMEGA**2 for joint, inertia in ARMATURE.items()}
DAMPING = {joint: 2.0 * 2.0 * inertia * _OMEGA for joint, inertia in ARMATURE.items()}
EFFORT = {
    ".*_hip_pitch_joint": 88.0, ".*_hip_yaw_joint": 88.0, "waist_yaw_joint": 88.0, ".*_hip_roll_joint": 139.0,
    ".*_knee_joint": 139.0, ".*_ankle_.*": 50.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0,
    ".*_shoulder_.*": 25.0, ".*_elbow_joint": 25.0, ".*_wrist_roll_joint": 25.0, ".*_wrist_pitch_joint": 5.0,
    ".*_wrist_yaw_joint": 5.0,
}  # fmt: skip
"""Largest torque of the joints of the body [N m], by which a learner scales its actions."""
TILE_SIZE = 8.0


def g1_terrain(clips: tuple[str, ...] = CLIPS, levels: tuple[str, ...] = LEVELS) -> TerrainGeneratorCfg:
    """Return a terrain with a column per clip and a row per level, each tile with the boxes of its clip."""
    return TerrainGeneratorCfg(
        size=(TILE_SIZE, TILE_SIZE),
        num_rows=len(levels),
        num_cols=len(clips),
        curriculum=True,
        use_cache=False,
        sub_terrains={
            clip: parkour.ClipBoxesTerrainCfg(
                files=tuple(os.path.join(MOTIONS_DIR, f"{clip}_z{level}.json") for level in levels)
            )
            for clip in clips
        },
    )


def g1_task(
    clips: tuple[str, ...] = CLIPS, levels: tuple[str, ...] = LEVELS, reference_input: bool = False
) -> mdp2.TaskCfg:
    """Return the MDP of the G1 on the terrain of :func:`g1_terrain`: the clips of a column follow each other."""
    stems = [os.path.join(MOTIONS_DIR, f"{clip}_z{level}") for clip in clips for level in levels]
    lengths, rates = [], set()
    for stem in stems:
        with open(f"{stem}.json") as file:
            clip = json.load(file)
        lengths.append(clip["frames"])
        rates.add(clip["fps"])
    (fps,) = rates
    task = parkour_task(
        [f"{stem}.csv" for stem in stems], lengths, SIM_DT * DECIMATION, NUM_BODIES, fps, reference_input
    )
    # The policy moves the 29 joints of the body; the fingers hold the pose of the clip, which is zero.
    action = replace(task.actions["joint_pos"], subset=mdp2.Entity("robot").joints("(?!.*_hand_).*"))
    return replace(task, actions={"joint_pos": action})


@configclass
class G1ParkourSceneCfg(G1TrackingSceneCfg):
    """A Unitree G1 on the boxes of the clips, with a scanner of the terrain height around its pelvis."""

    # Climbing takes the collision shapes of every link, which the asset holds as a variant.
    robot: ArticulationCfg = replace(
        TRACKING_ROBOT,
        spawn=replace(TRACKING_ROBOT.spawn, variants={"Physics": "PhysX"}),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=STIFFNESS, damping=DAMPING, armature=ARMATURE
            )
        },
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", terrain_generator=g1_terrain(), collision_group=-1
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 1.2]),
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        update_period=DECIMATION * SIM_DT,
    )


@configclass
class G1ParkourPhysicsCfg(PresetCfg):
    """Newton physics presets of the G1 parkour task, with room for the contacts of the whole body."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=600,
            nconmax=200,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class G1ParkourEnvCfg(MDP2RLEnvCfg):
    """Configuration of the G1 parkour environment.

    The tile of a clip and a level is ``scene.terrain.terrain_origins[level, clip]``. A learner that starts a world
    on a clip writes that origin into the ``origin`` state of the ``terrain_origin`` event and the episode into
    the ``pending`` state of the ``start_episode`` event.
    """

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT, render_interval=DECIMATION, physics=G1ParkourPhysicsCfg(), use_newton_actuators=True
    )
    scene: G1ParkourSceneCfg = G1ParkourSceneCfg(num_envs=4096, env_spacing=4.0)
    task: mdp2.TaskCfg = g1_task()
    decimation: int = DECIMATION
