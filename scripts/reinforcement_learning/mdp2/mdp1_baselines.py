# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP1 versions of the MDP2 demos that split their worlds between tasks, for timing ``env.step``.

    uv run python scripts/reinforcement_learning/mdp2/mdp1_baselines.py recovery 4096

MDP1 has one set of terms for every environment. Each demo is written the way MDP1 allows: terms that only some
environments need run in all of them and are multiplied by a mask, events filter their environments, and reward
weights per environment are a tensor that a wrapper multiplies in. The baselines build on the MDP1 tracking task of
``tracking_mdp1_baseline.py`` and on the MDP1 Go2 velocity task.

``recovery``: pushes in the second half of the environments. ``reward_search``: sixteen weightings of the tracking
rewards. ``hil``: tracking in the first half, heading goals in the second, and a discriminator's style reward in
all. ``distillation``: the Go2 task with a second, noisy observation group with history, in every environment.
"""

import functools
import sys
import time

import torch
import warp as wp
from tracking_mdp1_baseline import tracking_cfg

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.noise import UniformNoiseCfg

from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.mdp2.hil import mdp as hil
from isaaclab_tasks.mdp2.reward_search.config.humanoid.humanoid_env_cfg import CANDIDATES
from isaaclab_tasks.utils.hydra import resolve_presets


def second_half(env) -> torch.Tensor:
    return torch.arange(env.num_envs, device=env.device) >= env.num_envs // 2


def where(func, mask):
    """Return the term ``func`` with its output zeroed outside the environments that ``mask(env)`` selects."""

    @functools.wraps(func)
    def wrapper(env, *args, **kwargs):
        value = func(env, *args, **kwargs)
        return value * mask(env).view(-1, *[1] * (value.ndim - 1))

    return wrapper


def push_second_half(env, env_ids, speed: float):
    """Push the roots of the environments of the second half, as ``push_by_setting_velocity`` does for all."""
    env_ids = env_ids[env_ids >= env.num_envs // 2]
    if not len(env_ids):
        return
    robot = env.scene["robot"]
    velocity = robot.data.root_com_vel_w.torch[env_ids].clone()
    scale = torch.tensor([speed, speed, 0.3 * speed], device=env.device)
    velocity[:, :3] += (2.0 * torch.rand(len(env_ids), 3, device=env.device) - 1.0) * scale
    robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


def recovery_cfg():
    cfg = tracking_cfg("dance")
    cfg.terminations.tracking_lost.params["threshold"] = 0.25
    cfg.events.push = EventTermCfg(
        func=push_second_half, mode="interval", interval_range_s=(2.0, 4.0), params={"speed": 0.8}
    )
    return cfg


def weighted(func, name):
    """Return the reward ``func`` times the weight that each environment's candidate gives the term ``name``."""

    @functools.wraps(func)
    def wrapper(env, *args, **kwargs):
        weights = torch.tensor([candidate[name] for candidate in CANDIDATES], device=env.device)
        candidate = torch.arange(env.num_envs, device=env.device) * len(CANDIDATES) // env.num_envs
        return func(env, *args, **kwargs) * weights[candidate]

    return wrapper


def reward_search_cfg():
    cfg = tracking_cfg("walk")
    for name in CANDIDATES[0]:
        term = getattr(cfg.rewards, name)
        term.func, term.weight = weighted(term.func, name), 1.0
    return cfg


GOALS = {}


def goal(env) -> torch.Tensor:
    """A direction and a speed per environment, drawn once here; the task resamples it every few seconds."""
    if env.num_envs not in GOALS:
        angle = torch.rand(env.num_envs, device=env.device) * 6.283
        GOALS[env.num_envs] = torch.stack([angle.cos(), angle.sin(), torch.full_like(angle, 1.2)], dim=-1)
    return GOALS[env.num_envs]


def goal_b(env) -> torch.Tensor:
    quat = env.scene["robot"].data.root_link_pose_w.torch[:, 3:7]
    target = goal(env)
    direction = torch.cat([target[:, :2], torch.zeros_like(target[:, :1])], dim=-1)
    return torch.cat(
        [math_utils.quat_apply_inverse(math_utils.yaw_quat(quat), direction)[:, :2], target[:, 2:]], dim=-1
    )


def heading_velocity(env, scale: float, facing_weight: float) -> torch.Tensor:
    data = env.scene["robot"].data
    target = goal(env)
    error = target[:, 2] - (target[:, :2] * data.root_com_vel_w.torch[:, :2]).sum(-1)
    forward = math_utils.quat_apply(
        data.root_link_pose_w.torch[:, 3:7], torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    )
    facing = (torch.nn.functional.normalize(forward[:, :2], dim=-1) * target[:, :2]).sum(-1)
    return (1.0 - facing_weight) * torch.exp(-scale * error.square()) + facing_weight * facing


class Style:
    """The style reward of the MDP2 task: a discriminator scores the motion features before and after the step."""

    def __init__(self):
        self.discriminator, self.previous = None, None

    def __call__(self, env) -> torch.Tensor:
        data = env.scene["robot"].data
        features = hil.motion_features(
            data.joint_pos.torch, data.joint_vel.torch, data.root_link_pose_w.torch, data.root_com_vel_w.torch
        )
        if self.discriminator is None:
            self.discriminator = hil.Discriminator(features.shape[1]).to(env.device)
            self.previous = features.clone()
        with torch.no_grad():
            score = self.discriminator(torch.cat([self.previous, features], dim=-1))
        self.previous = features.clone()
        return (1.0 - 0.25 * (score - 1.0).square()).clamp(min=0.0)


def fallen(env, minimum: float, maximum: float) -> torch.Tensor:
    height = env.scene["robot"].data.root_link_pose_w.torch[:, 2]
    return ~((height > minimum) & (height < maximum))


def hil_cfg():
    cfg = tracking_cfg("walk")
    first_half = lambda env: ~second_half(env)  # noqa: E731
    for name in ("joint_pos", "joint_vel", "body_pos", "root_vel"):
        term = getattr(cfg.rewards, name)
        term.func = where(term.func, first_half)
    params = {"scale": 0.25, "facing_weight": 0.3}
    cfg.rewards.goal = RewardTermCfg(func=where(heading_velocity, second_half), params=params, weight=15.0)
    cfg.rewards.style = RewardTermCfg(func=Style(), weight=15.0)
    for name in ("motion_ended", "tracking_lost"):
        term = getattr(cfg.terminations, name)
        term.func = where(term.func, first_half)
    params = {"minimum": 0.5, "maximum": 1.5}
    cfg.terminations.fallen = TerminationTermCfg(func=where(fallen, second_half), params=params)
    cfg.observations.policy.goal = ObservationTermCfg(func=goal_b)
    return cfg


def distillation_cfg():
    cfg = resolve_presets(UnitreeGo2FlatEnvCfg(), selected=("newton_mjwarp",))
    policy = cfg.observations.policy

    @configclass
    class StudentCfg(ObservationGroupCfg):
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = policy.velocity_commands
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5))
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    # The teacher's group is the task's without noise; the student's is computed in every environment.
    policy.enable_corruption = False
    cfg.observations.student = StudentCfg()
    return cfg


if __name__ == "__main__":
    name, num_envs = sys.argv[1], int(sys.argv[2])
    cfg = {
        "recovery": recovery_cfg,
        "reward_search": reward_search_cfg,
        "hil": hil_cfg,
        "distillation": distillation_cfg,
    }[name]()
    cfg.scene.num_envs, cfg.sim.device = num_envs, "cuda:0"
    env = ManagerBasedRLEnv(cfg)
    env.reset()
    dim = env.action_manager.total_action_dim
    actions = [torch.rand(num_envs, dim, device="cuda:0") * 0.2 - 0.1 for _ in range(16)]
    with torch.inference_mode():
        for i in range(60):
            env.step(actions[i % 16])
        wp.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(300):
            env.step(actions[i % 16])
        wp.synchronize()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 300
    print(f"MDP1 {name}: {num_envs} worlds, step {elapsed * 1e6:.0f} us, {num_envs / elapsed / 1e6:.3f} M steps/s")
    env.close()
