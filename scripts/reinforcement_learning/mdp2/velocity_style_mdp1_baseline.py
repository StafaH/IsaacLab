# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The velocity tasks in the style of motion clips of ``isaaclab_tasks.mdp2.velocity_style`` written with MDP1.

A baseline for ``train_velocity_style.py --runtime mdp1``: the scene, physics, clips, commands, observations,
actions, rewards, terminations, and events of the MDP2 G1 or Go2 style environment, as a ``ManagerBasedRLEnv``.
The MDP2 environment runs the style task in the first half of its worlds and the velocity task in the second.
MDP1 has one set of terms for every environment, so this one runs every term of both tasks in all environments:

- the rewards that the style task drops are multiplied by a mask of the second half;
- the style reward runs its discriminator in every environment and is multiplied by a mask of the first half;
- the reset events filter their environments: the first half starts on frames of the clips, the second half as
  the velocity task does.
"""

from __future__ import annotations

import functools
import inspect
import math
from types import SimpleNamespace

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ManagerTermBase,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from isaaclab_tasks.core.velocity.mdp import feet_air_time, feet_air_time_positive_biped
from isaaclab_tasks.mdp2.hil import mdp as hil
from isaaclab_tasks.mdp2.tracking.mdp import events as tracking
from isaaclab_tasks.mdp2.velocity_g1 import flat_env_cfg as g1_velocity
from isaaclab_tasks.mdp2.velocity_go2 import flat_env_cfg as go2_velocity
from isaaclab_tasks.mdp2.velocity_style.config.g1 import g1_env_cfg as g1_style
from isaaclab_tasks.mdp2.velocity_style.config.go2 import go2_env_cfg as go2_style
from isaaclab_tasks.mdp2.velocity_style.style_env_cfg import KEPT_REWARDS


def first_half(env) -> torch.Tensor:
    """The environments of the style task."""
    return torch.arange(env.num_envs, device=env.device) < env.num_envs // 2


def second_half(env) -> torch.Tensor:
    """The environments of the velocity task."""
    return ~first_half(env)


def resolve(env, env_ids) -> torch.Tensor:
    """Return the ids of ``env_ids``, which events may also get as ``None`` or a slice for all environments."""
    if isinstance(env_ids, torch.Tensor):
        return env_ids
    return torch.arange(env.num_envs, device=env.device)


def where_second_half(func):
    """Return the reward ``func`` with its output zeroed in the environments of the style task."""

    @functools.wraps(func)
    def wrapper(env, *args, **kwargs):
        return func(env, *args, **kwargs) * second_half(env)

    return wrapper


def in_second_half(term):
    """Return the event ``term``, a function or a term class, applied only to the environments of the velocity task."""
    if inspect.isclass(term):

        class Filtered(term):
            @functools.wraps(term.__call__)
            def __call__(self, env, env_ids, *args, **kwargs):
                env_ids = resolve(env, env_ids)
                env_ids = env_ids[env_ids >= env.num_envs // 2]
                if len(env_ids):
                    super().__call__(env, env_ids, *args, **kwargs)

        Filtered.__name__ = term.__name__
        return Filtered

    @functools.wraps(term)
    def wrapper(env, env_ids, *args, **kwargs):
        env_ids = resolve(env, env_ids)
        env_ids = env_ids[env_ids >= env.num_envs // 2]
        if len(env_ids):
            term(env, env_ids, *args, **kwargs)

    return wrapper


class ResetToClip(ManagerTermBase):
    """Start the environments of the style task on frames of the clips, as the MDP2 ``reset_to_csv_motion`` does.

    The clips are loaded with the MDP2 loader on the first call, when the robot can be posed to read the positions
    of its bodies, so both runtimes start on the same data. ``clip`` holds them, for the learner to read.
    """

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.clip = None

    def _load(self, motion):
        robot, env, device = self.robot, self._env, self.device
        num_joints, num_bodies = len(robot.joint_names), len(robot.body_names)
        shapes = {
            "clip_joint_pos": (motion["frames"], num_joints),
            "clip_joint_vel": (motion["frames"], num_joints),
            "clip_root_pose": (motion["frames"], 7),
            "clip_root_vel": (motion["frames"], 6),
            "clip_body_pos": (motion["frames"], num_bodies, 3),
            "clip_range": (motion["clips"], 2),
            "start": (self.num_envs,),
        }
        state = {name: torch.zeros(shape, device=device) for name, shape in shapes.items()}
        outputs = {
            "joint_pos": torch.zeros(self.num_envs, num_joints, device=device),
            "root_link_pose_w": torch.zeros(self.num_envs, 7, device=device),
        }
        inputs = {}

        def commit(worlds, selectors):
            # Pose every robot on its frame, and read the positions of its bodies after forward kinematics.
            robot.write_joint_position_to_sim_index(position=outputs["joint_pos"])
            robot.write_root_pose_to_sim_index(root_pose=outputs["root_link_pose_w"])
            env.sim.forward()
            inputs["body_link_pose_w"] = robot.data.body_link_pose_w.torch

        context = SimpleNamespace(
            entity=lambda name: robot,
            step_dt=env.step_dt,
            state=state,
            num_worlds=self.num_envs,
            outputs=outputs,
            inputs=inputs,
            selectors={"joint_pos": "joint_pos", "root_link_pose_w": "root_link_pose_w"},
            scene_data=SimpleNamespace(commit=commit),
        )
        tracking._load_csv_motions(context, **motion)
        return state

    def __call__(self, env, env_ids, motion: dict, start_range: tuple[float, float], noise: float):
        if self.clip is None:
            self.clip = self._load(motion)
        env_ids = resolve(env, env_ids)
        env_ids = env_ids[env_ids < self.num_envs // 2]
        if not len(env_ids):
            return
        clip, robot = self.clip, self.robot
        ranges = clip["clip_range"].long()
        index = torch.randint(len(ranges), (len(env_ids),), device=self.device)
        first, last = ranges[index, 0], ranges[index, 1]
        low, high = start_range
        fraction = torch.rand(len(env_ids), device=self.device) * (high - low) + low
        frame = first + torch.minimum((fraction * (last - first)).long(), last - first - 2)
        clip["start"][env_ids] = frame.float()

        limits = robot.data.soft_joint_pos_limits.torch[env_ids]
        jitter = (2.0 * torch.rand(len(env_ids), limits.shape[1], device=self.device) - 1.0) * noise
        joint_pos = (clip["clip_joint_pos"][frame] + jitter).clamp(limits[..., 0], limits[..., 1])
        root_pose = clip["clip_root_pose"][frame].clone()
        root_pose[:, :3] += env.scene.env_origins[env_ids]
        robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=clip["clip_joint_vel"][frame], env_ids=env_ids)
        robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim_index(root_velocity=clip["clip_root_vel"][frame], env_ids=env_ids)


class StyleReward(ManagerTermBase):
    """The MDP2 ``style`` reward: a discriminator scores the motion features before and after the step.

    It runs in every environment and is zero outside the style task's. ``transition`` holds the features, and
    ``discriminator`` the network, for the learner to train.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.data = env.scene["robot"].data
        self.size = 10 + 2 * self.data.joint_pos.torch.shape[1]
        self.discriminator = hil.Discriminator(self.size).to(self.device)
        self.transition = torch.zeros(self.num_envs, 2 * self.size, device=self.device)

    def features(self) -> torch.Tensor:
        data = self.data
        return hil.motion_features(
            data.joint_pos.torch, data.joint_vel.torch, data.root_link_pose_w.torch, data.root_com_vel_w.torch
        )

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self.transition[env_ids, self.size :] = self.features()[env_ids]

    def __call__(self, env) -> torch.Tensor:
        self.transition[:, : self.size] = self.transition[:, self.size :]
        self.transition[:, self.size :] = self.features()
        with torch.no_grad():
            score = self.discriminator(self.transition)
        return (1.0 - 0.25 * (score - 1.0).square()).clamp(min=0.0) * first_half(env)


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5))
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class VelocityStyleMDP1EnvCfg(ManagerBasedRLEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: dict = {}  # Built by style_cfg, by name
    terminations: TerminationsCfg = TerminationsCfg()
    events: dict = {}  # Built by style_cfg, by name
    episode_length_s: float = 20.0


def joints(*names: str) -> dict:
    return {"asset_cfg": SceneEntityCfg("robot", joint_names=list(names))}


def contacts(*names: str) -> dict:
    return {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(names))}


def g1_rewards() -> dict[str, RewardTermCfg]:
    """The rewards of the MDP2 G1 velocity task, by name."""
    legs = joints(".*_hip_.*", ".*_knee_joint")
    fingers = [f".*_{n}_joint" for n in ("zero", "one", "two", "three", "four", "five", "six")]
    command = {"command_name": "base_velocity", "std": 0.5}
    return {
        "track_lin_vel_xy_exp": RewardTermCfg(func=mdp.track_lin_vel_xy_exp, weight=2.0, params=command),
        "track_ang_vel_z_exp": RewardTermCfg(func=mdp.track_ang_vel_z_exp, weight=2.0, params=command),
        "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
        "lin_vel_z_l2": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.2),
        "ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
        "flat_orientation_l2": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0),
        "base_height_l2": RewardTermCfg(func=mdp.base_height_l2, weight=-20.0, params={"target_height": 0.72}),
        "dof_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2.0e-6, params=legs),
        "dof_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.0e-7, params=legs),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
        "feet_air_time": RewardTermCfg(
            func=feet_air_time_positive_biped,
            weight=1.0,
            params={"command_name": "base_velocity", "threshold": 0.4, **contacts(".*_ankle_roll_link")},
        ),
        "dof_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits, weight=-1.0, params=joints(".*_ankle_pitch_joint", ".*_ankle_roll_joint")
        ),
        "joint_deviation_hip": RewardTermCfg(
            func=mdp.joint_deviation_l1, weight=-0.1, params=joints(".*_hip_yaw_joint", ".*_hip_roll_joint")
        ),
        "joint_deviation_arms": RewardTermCfg(
            func=mdp.joint_deviation_l1, weight=-0.1, params=joints(".*_shoulder_.*", ".*_elbow_.*")
        ),
        "joint_deviation_fingers": RewardTermCfg(func=mdp.joint_deviation_l1, weight=-0.05, params=joints(*fingers)),
        "joint_deviation_torso": RewardTermCfg(func=mdp.joint_deviation_l1, weight=-0.1, params=joints("torso_joint")),
    }


def go2_rewards() -> dict[str, RewardTermCfg]:
    """The rewards of the MDP2 Go2 velocity task, by name."""
    command = {"command_name": "base_velocity", "std": math.sqrt(0.25)}
    return {
        "track_lin_vel_xy_exp": RewardTermCfg(func=mdp.track_lin_vel_xy_exp, weight=1.5, params=command),
        "track_ang_vel_z_exp": RewardTermCfg(func=mdp.track_ang_vel_z_exp, weight=0.75, params=command),
        "lin_vel_z_l2": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0),
        "ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
        "dof_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2.0e-4),
        "dof_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
        "feet_air_time": RewardTermCfg(
            func=feet_air_time,
            weight=0.25,
            params={"command_name": "base_velocity", "threshold": 0.5, **contacts(".*_foot")},
        ),
        "flat_orientation_l2": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-2.5),
        "base_height_l2": RewardTermCfg(func=mdp.base_height_l2, weight=-30.0, params={"target_height": 0.4}),
    }


ROBOTS = {
    "g1": SimpleNamespace(
        velocity=g1_velocity,
        scene=g1_velocity.UnitreeG1FlatSceneCfg,
        physics=g1_velocity.UnitreeG1FlatPhysicsCfg,
        style=g1_style,
        style_tasks=g1_style.G1_STYLE_TASKS,
        rewards=g1_rewards,
        action_scale=0.5,
        lin_vel=((0.0, 1.0), (-0.5, 0.5)),
        base="torso_link",
        reset_velocity=0.0,
    ),
    "go2": SimpleNamespace(
        velocity=go2_velocity,
        scene=go2_velocity.UnitreeGo2FlatSceneCfg,
        physics=go2_velocity.UnitreeGo2FlatPhysicsCfg,
        style=go2_style,
        style_tasks=go2_style.GO2_STYLE_TASKS,
        rewards=go2_rewards,
        action_scale=0.25,
        lin_vel=((-1.0, 1.0), (-1.0, 1.0)),
        base="base",
        reset_velocity=0.5,
    ),
}


def style_cfg(robot: str) -> VelocityStyleMDP1EnvCfg:
    """Return the MDP1 environment config of the MDP2 style environment of ``robot``, ``"g1"`` or ``"go2"``."""
    spec = ROBOTS[robot]
    velocity, style_task = spec.velocity, spec.style_tasks[0]
    cfg = VelocityStyleMDP1EnvCfg()
    # The scene and physics of the MDP2 task.
    cfg.scene = spec.scene(num_envs=4096, env_spacing=2.5)
    cfg.sim = sim_utils.SimulationCfg(
        dt=velocity.SIM_DT,
        render_interval=velocity.DECIMATION,
        physics=spec.physics().newton_mjwarp,
        physics_material=velocity.GROUND_MATERIAL,
        use_newton_actuators=True,
    )
    cfg.decimation = velocity.DECIMATION
    cfg.actions.joint_pos.scale = spec.action_scale
    ranges = cfg.commands.base_velocity.ranges
    ranges.lin_vel_x, ranges.lin_vel_y = spec.lin_vel

    # Rewards: the velocity task's, with those that the style task drops masked to the velocity task's half.
    cfg.rewards = {}
    for name, term in spec.rewards().items():
        if name not in KEPT_REWARDS:
            term.func = where_second_half(term.func)
        cfg.rewards[name] = term
    cfg.rewards["style"] = RewardTermCfg(func=StyleReward, weight=style_task.rewards["style"].weight)

    cfg.terminations.base_contact = TerminationTermCfg(
        func=mdp.illegal_contact, params={"threshold": 1.0, **contacts(spec.base)}
    )
    if robot == "g1":
        cfg.terminations.bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": 1.0})

    # Events: the clip reset first, whose first call poses every robot to load the clips.
    reset_params = style_task.events["reset_to_motion"].params
    motion = {key: value for key, value in reset_params.items() if key not in ("start_range", "noise")}
    events = cfg.events = {}
    events["reset_to_motion"] = EventTermCfg(
        func=ResetToClip,
        mode="reset",
        params={"motion": motion, "start_range": reset_params["start_range"], "noise": reset_params["noise"]},
    )
    # Newton has one friction coefficient, as in the MDP2 task.
    events["physics_material"] = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.8, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    events["add_base_mass"] = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=spec.base),
            "mass_distribution_params": (1 / 1.25, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    if robot == "go2":
        events["base_com"] = EventTermCfg(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
            },
        )
    speed = (-spec.reset_velocity, spec.reset_velocity)
    events["reset_base"] = EventTermCfg(
        func=in_second_half(mdp.reset_root_state_uniform),
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {axis: speed for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    position_range = (1.0, 1.0) if robot == "g1" else (0.5, 1.5)
    events["reset_robot_joints"] = EventTermCfg(
        func=in_second_half(mdp.reset_joints_by_scale),
        mode="reset",
        params={"position_range": position_range, "velocity_range": (0.0, 0.0)},
    )
    events["push_robot"] = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    return cfg
