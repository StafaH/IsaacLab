# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train motion tracking off-policy with few worlds: TD3 with a replay buffer on the device.

    uv run python scripts/reinforcement_learning/mdp2/train_tracking_offpolicy.py --num_envs 512 --seconds 60

Off-policy learners reuse every transition many times, so they run few worlds and step them often, which is where
the time of an environment step matters most. ``--runtime mdp1`` steps the same task written with MDP1 managers.
The script trains for ``--seconds`` of wall time and reports how much of the clip the policy tracks then.
"""

import argparse
import copy
import time

import gymnasium as gym
import torch
import warp as wp
from ppo import Normalizer, mlp

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.tracking.config.humanoid.humanoid_env_cfg import humanoid_task
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--motion", default="walk")
parser.add_argument("--runtime", default="mdp2", choices=["mdp2", "mdp1"])
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--seconds", type=float, default=60.0, help="Wall time to train for")
parser.add_argument("--buffer_steps", type=int, default=1024, help="Steps of every world that the buffer holds")
parser.add_argument("--batch", type=int, default=8192)
parser.add_argument("--updates", type=int, default=2, help="Updates per environment step")
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--tau", type=float, default=0.02)
parser.add_argument("--lr", type=float, default=3.0e-4)
parser.add_argument("--noise", type=float, default=0.15, help="Exploration noise, in actions")
parser.add_argument("--action_limit", type=float, default=1.5)
parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
args = parser.parse_args()

DEVICE, TASK = "cuda:0", "Isaac-Tracking-Humanoid-Walk-MDP2"
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
if args.runtime == "mdp1":
    from tracking_mdp1_baseline import tracking_cfg

    from isaaclab.envs import ManagerBasedRLEnv

    cfg = tracking_cfg(args.motion)
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, args.num_envs, args.seed
    env = ManagerBasedRLEnv(cfg)
else:
    cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, args.num_envs, args.seed
    cfg.task = humanoid_task(args.motion)
    env = gym.make(TASK, cfg=cfg).unwrapped
obs = env.reset()[0]["policy"]
N, obs_dim, action_dim = obs.shape[0], obs.shape[1], env.single_action_space.shape[0]

normalizer = Normalizer(obs_dim).to(DEVICE)
actor = mlp([obs_dim, *args.hidden, action_dim]).to(DEVICE)
critics = torch.nn.ModuleList([mlp([obs_dim + action_dim, *args.hidden, 1]) for _ in range(2)]).to(DEVICE)
target_actor, target_critics = copy.deepcopy(actor), copy.deepcopy(critics)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr, fused=True)
critic_optimizer = torch.optim.Adam(critics.parameters(), lr=args.lr, fused=True)


def act(observations, network=actor):
    return args.action_limit * torch.tanh(network(normalizer(observations)))


def value(observations, actions, networks):
    inputs = torch.cat([normalizer(observations), actions], dim=-1)
    return torch.cat([network(inputs) for network in networks], dim=-1)


# The replay buffer: the last steps of every world, written in a ring.
size = args.buffer_steps
buffer = {
    "obs": torch.zeros(size, N, obs_dim, device=DEVICE),
    "next_obs": torch.zeros(size, N, obs_dim, device=DEVICE),
    "actions": torch.zeros(size, N, action_dim, device=DEVICE),
    "rewards": torch.zeros(size, N, device=DEVICE),
    "terminated": torch.zeros(size, N, device=DEVICE),
}
steps = 0


def update(step):
    filled = min(steps, size)
    rows = torch.randint(filled, (args.batch,), device=DEVICE)
    columns = torch.randint(N, (args.batch,), device=DEVICE)
    observations, next_observations = buffer["obs"][rows, columns], buffer["next_obs"][rows, columns]
    actions = buffer["actions"][rows, columns]
    with torch.no_grad():
        noise = (0.2 * torch.randn_like(actions)).clamp(-0.5, 0.5)
        next_actions = (act(next_observations, target_actor) + noise).clamp(-args.action_limit, args.action_limit)
        next_value = value(next_observations, next_actions, target_critics).min(dim=-1).values
        target = (
            buffer["rewards"][rows, columns] + args.gamma * (1.0 - buffer["terminated"][rows, columns]) * next_value
        )
    critic_loss = (value(observations, actions, critics) - target[:, None]).square().mean()
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()
    if step % 2 == 0:
        actor_loss = -value(observations, act(observations), critics)[:, 0].mean()
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optimizer.step()
        with torch.no_grad():
            for network, target_network in ((actor, target_actor), (critics, target_critics)):
                for param, target_param in zip(network.parameters(), target_network.parameters()):
                    target_param.lerp_(param, args.tau)


completed = torch.zeros(N, device=DEVICE)
updates = 0
log = []
wp.synchronize()
start = time.perf_counter()
next_log = 10.0
while True:
    with torch.no_grad():
        normalizer.update(obs)
        action = act(obs) + args.noise * torch.randn(N, action_dim, device=DEVICE)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = next_obs["policy"]
        row = steps % size
        buffer["obs"][row], buffer["next_obs"][row], buffer["actions"][row] = obs, next_obs, action
        buffer["rewards"][row], buffer["terminated"][row] = reward, terminated
        # The observation after a reset belongs to the next episode; a terminated step has no successor in its target.
        done = terminated | truncated
        completed.copy_(torch.where(done, truncated.float(), completed))
        obs = next_obs
    steps += 1
    if steps >= 16:
        for _ in range(args.updates):
            update(updates)
            updates += 1
    if steps % 64 == 0:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if elapsed >= next_log or elapsed >= args.seconds:
            log.append((elapsed, steps, completed.mean().item(), buffer["rewards"][: min(steps, size)].mean().item()))
            print(
                f"LOG {elapsed:6.1f} s  steps {steps}  completion {log[-1][2]:.3f}  reward {log[-1][3]:.3f}", flush=True
            )
            next_log += 10.0
        if elapsed >= args.seconds:
            break
elapsed = time.perf_counter() - start
print(
    f"OFFPOLICY {args.runtime} {N} worlds: {steps} steps in {elapsed:.1f} s,"
    f" {1e3 * elapsed / steps:.2f} ms per step with"
    f" {args.updates} updates, {steps * N / elapsed / 1e6:.3f} M steps/s, completion {completed.mean().item():.3f}"
)
env.close()
