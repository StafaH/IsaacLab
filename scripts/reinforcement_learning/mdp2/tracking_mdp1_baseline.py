# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The humanoid motion tracking task of ``isaaclab_tasks.mdp2.tracking_humanoid`` written with MDP1 managers.

A baseline for ``train_tracking.py --runtime mdp1``: the same scene, physics, clip data, observations, rewards,
terminations, resets, and residual actions, as a ``ManagerBasedRLEnv``. The motion is a command term that owns the
clip, starts the reset worlds on it, and gives the terms the frame they track, as MDP1 tracking tasks do.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    CommandTerm,
    CommandTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from isaaclab_tasks.mdp2.tracking.config.humanoid.humanoid_env_cfg import (
    DECIMATION,
    NUM_BODIES,
    SIM_DT,
    HumanoidTrackingPhysicsCfg,
    HumanoidTrackingSceneCfg,
    humanoid_task,
)
from isaaclab_tasks.mdp2.tracking.mdp import events as tracking


class MotionCommand(CommandTerm):
    """The clip, the frame each world started at, and the reference of the frames it tracks."""

    def __init__(self, cfg: MotionCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        num_joints = len(self.robot.joint_names)
        shapes = {
            "clip_joint_pos": (cfg.frames, num_joints),
            "clip_joint_vel": (cfg.frames, num_joints),
            "clip_root_pose": (cfg.frames, 7),
            "clip_root_vel": (cfg.frames, 6),
            "clip_body_pos": (cfg.frames, NUM_BODIES, 3),
            "clip_range": (len(cfg.motion["motion_files"]), 2),
        }
        self.clip = {name: torch.zeros(shape, device=self.device) for name, shape in shapes.items()}
        self.start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # The MDP2 task's loader, so both tasks track the same data.
        context = SimpleNamespace(
            entity=lambda name: self.robot, step_dt=env.step_dt, state={**self.clip, "start": self.start}
        )
        tracking._load_motions(context, **cfg.motion)
        self.end = torch.zeros_like(self.start)

    @property
    def command(self) -> torch.Tensor:
        """Joint positions of the frame that the next step tracks."""
        return self.clip["clip_joint_pos"][self.frame(next_step=True)]

    def frame(self, next_step: bool = False) -> torch.Tensor:
        """The frame each world tracks at its current step, or at the next one."""
        frame = self.start + self._env.episode_length_buf
        if next_step:
            return torch.minimum(frame + 1, self.end - 1)
        return torch.minimum(frame, self.end - 1)

    def body_pos_error(self) -> torch.Tensor:
        """Mean squared distance of the bodies to their positions in the tracked frame [m^2]."""
        position = self.robot.data.body_link_pose_w.torch[..., :3] - self._env.scene.env_origins[:, None]
        return (position - self.clip["clip_body_pos"][self.frame()]).square().sum(-1).mean(-1)

    def _resample_command(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        ranges = self.clip["clip_range"].long()
        clip = torch.randint(len(ranges), (len(env_ids),), device=self.device)
        first, last = ranges[clip, 0], ranges[clip, 1]
        low, high = self.cfg.start_range
        fraction = torch.rand(len(env_ids), device=self.device) * (high - low) + low
        frame = first + torch.minimum((fraction * (last - first)).long(), last - first - 2)
        self.start[env_ids] = frame
        self.end[env_ids] = last

        limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        noise = (2.0 * torch.rand(len(env_ids), limits.shape[1], device=self.device) - 1.0) * self.cfg.noise
        joint_pos = (self.clip["clip_joint_pos"][frame] + noise).clamp(limits[..., 0], limits[..., 1])
        root_pose = self.clip["clip_root_pose"][frame].clone()
        root_pose[:, :3] += self._env.scene.env_origins[env_ids]
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=self.clip["clip_joint_vel"][frame], env_ids=env_ids)
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=self.clip["clip_root_vel"][frame], env_ids=env_ids)

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type: type = MotionCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # Only on reset
    motion: dict = {}  # The params of the MDP2 task's clip loader
    frames: int = 0
    start_range: tuple[float, float] = (0.0, 0.9)
    noise: float = 0.02


class ReferenceJointPositionAction(ActionTerm):
    """Joint position targets that correct the pose of the frame that the next step tracks."""

    def __init__(self, cfg: ReferenceJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return len(self._asset.joint_names)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions * self.cfg.scale

    def apply_actions(self):
        reference = self._env.command_manager.get_command("motion")
        self._asset.set_joint_position_target_index(target=reference + self._processed_actions)


@configclass
class ReferenceJointPositionActionCfg(ActionTermCfg):
    class_type: type = ReferenceJointPositionAction
    asset_name: str = "robot"
    scale: float = 0.25


def root_height(env) -> torch.Tensor:
    return env.scene["robot"].data.root_link_pose_w.torch[:, 2:3]


def body_pos_b(env) -> torch.Tensor:
    data = env.scene["robot"].data
    root = data.root_link_pose_w.torch
    offset = data.body_link_pose_w.torch[..., :3] - root[:, None, :3]
    return math_utils.quat_apply_inverse(root[:, None, 3:7].expand(-1, NUM_BODIES, -1), offset).flatten(1)


def reference_body_pos_b(env) -> torch.Tensor:
    motion = env.command_manager.get_term("motion")
    root = env.scene["robot"].data.root_link_pose_w.torch
    target = motion.clip["clip_body_pos"][motion.frame(next_step=True)] + env.scene.env_origins[:, None]
    offset = target - root[:, None, :3]
    return math_utils.quat_apply_inverse(root[:, None, 3:7].expand(-1, NUM_BODIES, -1), offset).flatten(1)


def joint_pos_reward(env, scale: float) -> torch.Tensor:
    motion = env.command_manager.get_term("motion")
    error = env.scene["robot"].data.joint_pos.torch - motion.clip["clip_joint_pos"][motion.frame()]
    return torch.exp(-scale * error.square().mean(-1))


def joint_vel_reward(env, scale: float) -> torch.Tensor:
    motion = env.command_manager.get_term("motion")
    error = env.scene["robot"].data.joint_vel.torch - motion.clip["clip_joint_vel"][motion.frame()]
    return torch.exp(-scale * error.square().mean(-1))


def body_pos_reward(env, scale: float) -> torch.Tensor:
    return torch.exp(-scale * env.command_manager.get_term("motion").body_pos_error())


def root_vel_reward(env, scale: float, angular_scale: float) -> torch.Tensor:
    motion = env.command_manager.get_term("motion")
    error = env.scene["robot"].data.root_com_vel_w.torch - motion.clip["clip_root_vel"][motion.frame()]
    error = error[:, :3].square().sum(-1) + angular_scale * error[:, 3:].square().sum(-1)
    return torch.exp(-scale * error)


def motion_ended(env) -> torch.Tensor:
    motion = env.command_manager.get_term("motion")
    return motion.start + env.episode_length_buf >= motion.end - 2


def tracking_lost(env, threshold: float) -> torch.Tensor:
    return env.command_manager.get_term("motion").body_pos_error() > threshold


@configclass
class ActionsCfg:
    joint_pos = ReferenceJointPositionActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        root_height = ObservationTermCfg(func=root_height)
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, scale=0.1)
        body_pos = ObservationTermCfg(func=body_pos_b)
        reference_joint_pos = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "motion"})
        reference_body_pos = ObservationTermCfg(func=reference_body_pos_b)
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    joint_pos = RewardTermCfg(func=joint_pos_reward, params={"scale": 25.0}, weight=15.0)
    joint_vel = RewardTermCfg(func=joint_vel_reward, params={"scale": 0.05}, weight=3.0)
    body_pos = RewardTermCfg(func=body_pos_reward, params={"scale": 100.0}, weight=9.0)
    root_vel = RewardTermCfg(func=root_vel_reward, params={"scale": 1.0, "angular_scale": 0.1}, weight=3.0)
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.03)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    motion_ended = TerminationTermCfg(func=motion_ended, time_out=True)
    tracking_lost = TerminationTermCfg(func=tracking_lost, params={"threshold": 0.0625})


@configclass
class CommandsCfg:
    motion: MotionCommandCfg = MotionCommandCfg()


@configclass
class EventsCfg:
    pass


@configclass
class HumanoidTrackingMDP1EnvCfg(ManagerBasedRLEnvCfg):
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics=HumanoidTrackingPhysicsCfg().newton_mjwarp,
        use_newton_actuators=True,
    )
    scene: HumanoidTrackingSceneCfg = HumanoidTrackingSceneCfg(num_envs=2048, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    decimation: int = DECIMATION


def tracking_cfg(motions: str, start_range: tuple[float, float] = (0.0, 0.9), noise: float = 0.02):
    """Return the MDP1 environment config that tracks the clips ``motions``, like the MDP2 ``humanoid_task``."""
    params = humanoid_task(motions).events["reset_to_motion"].params
    cfg = HumanoidTrackingMDP1EnvCfg()
    cfg.episode_length_s = humanoid_task(motions).terminations["time_out"].episode_length_s
    cfg.commands.motion.motion = {**params, "start_range": start_range, "noise": noise}
    cfg.commands.motion.frames = params["frames"]
    cfg.commands.motion.start_range = start_range
    cfg.commands.motion.noise = noise
    return cfg
