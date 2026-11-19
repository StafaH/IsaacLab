# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The multi-skill Go2 task of ``isaaclab_tasks.mdp2.skills`` written with MDP1 managers, for timing.

    uv run python scripts/reinforcement_learning/mdp2/skills_mdp1_baseline.py 4096

MDP1 has one set of terms for every environment, so the skills share the managers: every reward and termination of
every skill is computed in every environment and multiplied by a mask of the environments that practice the skill,
and the reset of the get-up skill is an event that filters its environments. The script times ``env.step``.
"""

import functools
import sys
import time

import torch
import warp as wp

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import math as math_utils

from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

SKILLS = ("walk", "hind_legs", "handstand", "get_up")


def skill_ids(env) -> torch.Tensor:
    """The skill of every environment: equal shares, in order."""
    return torch.arange(env.num_envs, device=env.device) * len(SKILLS) // env.num_envs


def masked(func, *skills):
    """Return ``func`` with its output zeroed in the environments that practice none of ``skills``."""
    indices = [SKILLS.index(skill) for skill in skills]

    @functools.wraps(func)
    def wrapper(env, *args, **kwargs):
        mask = torch.isin(skill_ids(env), torch.tensor(indices, device=env.device))
        value = func(env, *args, **kwargs)
        return value * mask.view(-1, *[1] * (value.ndim - 1))

    return wrapper


def reset_fallen(env, env_ids, pose_range: dict[str, tuple[float, float]]):
    """Drop the environments that practice getting up on their side, after the task's reset of every environment."""
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids[skill_ids(env)[env_ids] == SKILLS.index("get_up")]
    if not len(env_ids):
        return
    robot = env.scene["robot"]
    ranges = torch.tensor([pose_range[key] for key in ("x", "y", "z", "roll", "pitch", "yaw")], device=env.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    pose = robot.data.default_root_pose.torch[env_ids].clone()
    pose[:, :3] += env.scene.env_origins[env_ids] + samples[:, :3]
    pose[:, 3:7] = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)


def axis_alignment(env, axis: tuple[float, float, float], direction: tuple[float, float, float]) -> torch.Tensor:
    quat = env.scene["robot"].data.root_link_pose_w.torch[:, 3:7]
    rotated = math_utils.quat_apply(quat, torch.tensor(axis, device=env.device).expand(env.num_envs, 3))
    return 0.5 + 0.5 * (rotated * torch.tensor(direction, device=env.device)).sum(-1)


def height_reward(env, target: float) -> torch.Tensor:
    return (env.scene["robot"].data.root_link_pose_w.torch[:, 2] / target).clamp(0.0, 1.0)


def bodies_height_reward(env, asset_cfg: SceneEntityCfg, target: float) -> torch.Tensor:
    height = env.scene["robot"].data.body_link_pose_w.torch[:, asset_cfg.body_ids, 2]
    return (height / target).clamp(0.0, 1.0).mean(-1)


def skill_one_hot(env) -> torch.Tensor:
    return torch.nn.functional.one_hot(skill_ids(env), len(SKILLS)).float()


def skills_cfg(num_envs: int):
    cfg = resolve_presets(UnitreeGo2FlatEnvCfg(), selected=("newton_mjwarp",))
    cfg.scene.num_envs, cfg.sim.device = num_envs, "cuda:0"
    rewards, contact = cfg.rewards, SceneEntityCfg("contact_forces", body_names="base")

    # Walking: the task's rewards, where the environment walks. The regularizers stay on for every skill.
    regularizers = ("dof_torques_l2", "dof_acc_l2", "action_rate_l2")
    for name, term in vars(rewards).items():
        if isinstance(term, RewardTermCfg) and name not in regularizers:
            term.func = masked(term.func, "walk")

    # Balancing on the hind legs and on the front legs.
    for skill, direction, height, feet in (
        ("hind_legs", (0.0, 0.0, 1.0), 0.55, "F[LR]_foot"),
        ("handstand", (0.0, 0.0, -1.0), 0.5, "R[LR]_foot"),
    ):
        params = {"axis": (1.0, 0.0, 0.0), "direction": direction}
        setattr(
            rewards, f"{skill}_posture", RewardTermCfg(func=masked(axis_alignment, skill), params=params, weight=4.0)
        )
        params = {"target": height}
        setattr(rewards, f"{skill}_height", RewardTermCfg(func=masked(height_reward, skill), params=params, weight=2.0))
        params = {"asset_cfg": SceneEntityCfg("robot", body_names=feet), "target": height}
        term = RewardTermCfg(func=masked(bodies_height_reward, skill), params=params, weight=3.0)
        setattr(rewards, f"{skill}_feet_up", term)
        params = {"sensor_cfg": contact, "threshold": 1.0}
        term = RewardTermCfg(func=masked(mdp.undesired_contacts, skill), params=params, weight=-2.0)
        setattr(rewards, f"{skill}_base_contact", term)

    # Getting up.
    params = {"axis": (0.0, 0.0, 1.0), "direction": (0.0, 0.0, 1.0)}
    rewards.get_up_posture = RewardTermCfg(func=masked(axis_alignment, "get_up"), params=params, weight=4.0)
    rewards.get_up_height = RewardTermCfg(func=masked(height_reward, "get_up"), params={"target": 0.34}, weight=2.0)
    rewards.get_up_joints = RewardTermCfg(func=masked(mdp.joint_deviation_l1, "get_up"), weight=-0.3)

    # Only walking ends on base contact.
    cfg.terminations.base_contact = TerminationTermCfg(
        func=masked(cfg.terminations.base_contact.func, "walk"), params=cfg.terminations.base_contact.params
    )

    # The get-up environments are dropped on their side, after the task's reset.
    pose_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2), "roll": (-1.6, 1.6), "pitch": (-0.5, 0.5)}
    params = {"pose_range": {**pose_range, "yaw": (-3.14, 3.14)}}
    cfg.events.reset_fallen = EventTermCfg(func=reset_fallen, mode="reset", params=params)

    # The policy is told the skill, and only walking environments get velocity commands.
    policy = cfg.observations.policy
    policy.velocity_commands.func = masked(policy.velocity_commands.func, "walk")
    policy.skill = ObservationTermCfg(func=skill_one_hot)
    return cfg


if __name__ == "__main__":
    num_envs = int(sys.argv[1])
    env = ManagerBasedRLEnv(skills_cfg(num_envs))
    env.reset()
    actions = [torch.rand(num_envs, 12, device="cuda:0") * 0.2 - 0.1 for _ in range(16)]
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
    print(f"MDP1 skills: {num_envs} worlds, step {elapsed * 1e6:.0f} us, {num_envs / elapsed / 1e6:.3f} M steps/s")
    env.close()
