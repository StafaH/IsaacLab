# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train one policy per reward candidate of an MDP2 reward search task, all in one simulation.

    uv run python scripts/reinforcement_learning/mdp2/train_reward_search.py --budget 60

The worlds are split between candidates that weight the rewards differently. Every candidate has its own policy,
critic, normalizers, and learning rate, held as one batched network, so one rollout and one captured update train
all of them. The candidates are ranked by what the task is for, not by their own rewards: how much of the clip
their worlds track, and how closely. ``--pbt`` also runs population-based training: every ``--pbt_interval``
iterations the worst candidates copy the policy and the reward weights of the best, with the weights perturbed,
by writing the per-world reward weights of the running environment.
"""

import argparse
import json
import time

import gymnasium as gym
import torch
import warp as wp
from isaaclab_experimental import mdp2

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-RewardSearch-Humanoid-Walk-MDP2")
parser.add_argument("--num_envs", type=int, default=16384)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=float, default=60.0, help="Samples to train for, in millions, over all candidates")
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument("--max_lr", type=float, default=1.0e-2)
parser.add_argument("--kl", type=float, default=0.15)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--clip", type=float, default=0.3)
parser.add_argument("--value_coef", type=float, default=4.0)
parser.add_argument("--init_std", type=float, default=0.3)
parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 128])
parser.add_argument("--pbt", action="store_true")
parser.add_argument("--pbt_interval", type=int, default=40)
parser.add_argument("--pbt_fraction", type=float, default=0.25)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--out", default=None)
args = parser.parse_args()

DEVICE = "cuda:0"
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
cfg = resolve_presets(load_cfg_from_registry(args.task, "env_cfg_entry_point"), selected=("newton_mjwarp",))
cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, args.num_envs, args.seed
env = gym.make(args.task, cfg=cfg).unwrapped
obs = env.reset()[0]["policy"]
K = len(env.mdp.world_configs)
num_envs, obs_dim, action_dim = obs.shape[0], obs.shape[1], env.mdp.action_dim
worlds = num_envs // K
assert num_envs == K * worlds, "The number of worlds must be a multiple of the number of candidates."
reward_names = [op.name.split(".", 1)[1] for op in env.mdp.ops[mdp2.Phase.REWARDS]]
termination_names = [op.name.split(".", 1)[1] for op in env.mdp.ops[mdp2.Phase.TERMINATIONS]]
reward_weights = env.state.reward_weights.torch
body_error = env.state.arrays[mdp2.quantities.body_pos_error].torch

# One actor and one critic per candidate, as rows of a parameter matrix: weights, biases, then the log std.
sizes = {"actor": [obs_dim, *args.hidden, action_dim], "critic": [obs_dim, *args.hidden, 1]}
layout, offset = {}, 0
for net, dims in sizes.items():
    for k in range(len(dims) - 1):
        layout[net, k] = (offset, dims[k], dims[k + 1])
        offset += dims[k] * dims[k + 1] + dims[k + 1]
layout["log_std"] = offset
num_params = offset + action_dim
params = torch.zeros(K, num_params, device=DEVICE)
for begin, fan_in, fan_out in (value for key, value in layout.items() if key != "log_std"):
    params[:, begin : begin + fan_in * fan_out + fan_out].uniform_(-(fan_in**-0.5), fan_in**-0.5)
params[:, layout["log_std"] :] = float(torch.tensor(args.init_std).log())
params.requires_grad_()
normalizer = {
    "mean": torch.zeros(K, 1, obs_dim, device=DEVICE),
    "var": torch.ones(K, 1, obs_dim, device=DEVICE),
    "ret_mean": torch.zeros(K, 1, device=DEVICE),
    "ret_var": torch.ones(K, 1, device=DEVICE),
}
count = torch.full((), 1.0e-4, device=DEVICE)


def forward(net, vector, x):
    """Run the network ``net`` of every candidate on its own inputs, of shape ``(K, batch, inputs)``."""
    depth = len(sizes[net]) - 1
    for k in range(depth):
        begin, fan_in, fan_out = layout[net, k]
        weight = vector[:, begin : begin + fan_in * fan_out].view(K, fan_in, fan_out)
        bias = vector[:, begin + fan_in * fan_out : begin + fan_in * fan_out + fan_out].view(K, 1, fan_out)
        x = torch.baddbmm(bias, x, weight)
        if k < depth - 1:
            x = torch.nn.functional.elu(x)
    return x


def normalize(x):
    return (x - normalizer["mean"]) / (normalizer["var"].sqrt() + 1.0e-8)


def update_running(mean, var, values, dims, batch_count):
    batch_mean, batch_var = values.mean(dims, keepdim=True), values.var(dims, keepdim=True)
    total = count + batch_count
    delta = batch_mean - mean
    var.copy_((var * count + batch_var * batch_count + delta**2 * count * batch_count / total) / total)
    mean.add_(delta * batch_count / total)


T = args.steps
batch = T * worlds
iterations = round(args.budget * 1.0e6 / (T * num_envs))
buffer = {
    "obs": torch.zeros(T, K, worlds, obs_dim, device=DEVICE),
    "actions": torch.zeros(T, K, worlds, action_dim, device=DEVICE),
    "means": torch.zeros(T, K, worlds, action_dim, device=DEVICE),
    "log_probs": torch.zeros(T, K, worlds, device=DEVICE),
    "values": torch.zeros(T + 1, K, worlds, device=DEVICE),
    "rewards": torch.zeros(T, K, worlds, device=DEVICE),
    "done": torch.zeros(T, K, worlds, device=DEVICE),
}
# Per candidate, the samples of a rollout: (K, batch, ...).
flat = {name: torch.zeros(K, batch, *buffer[name].shape[3:], device=DEVICE) for name in ("obs", "actions", "means")}
flat |= {name: torch.zeros(K, batch, device=DEVICE) for name in ("log_probs", "returns", "advantages")}
old_std = torch.zeros(K, 1, action_dim, device=DEVICE)
# What the candidates are ranked by: whether each world's last episode reached the end of the clip, and its error.
completed = torch.zeros(K, worlds, device=DEVICE)
motion_ended = termination_names.index("motion_ended")


def rollout():
    """Collect a rollout of every candidate, and its returns and advantages, into the buffers."""
    global obs
    with torch.no_grad():
        for t in range(T):
            grouped = obs.view(K, worlds, obs_dim)
            update_running(normalizer["mean"], normalizer["var"], grouped, 1, worlds)
            count.add_(worlds)
            normalized = normalize(grouped)
            mean = forward("actor", params, normalized)
            value = (
                forward("critic", params, normalized).squeeze(-1) * normalizer["ret_var"].sqrt()
                + normalizer["ret_mean"]
            )
            std = params[:, None, layout["log_std"] :].exp()
            action = mean + std * torch.randn_like(mean)
            buffer["obs"][t], buffer["actions"][t], buffer["values"][t], buffer["means"][t] = (
                grouped,
                action,
                value,
                mean,
            )
            buffer["log_probs"][t] = (-0.5 * ((action - mean) / std) ** 2 - std.log()).sum(-1)
            obs, reward, terminated, truncated, _ = env.step(action.view(num_envs, action_dim))
            obs = obs["policy"]
            done = (terminated | truncated).view(K, worlds)
            ended = env.transition.termination_terms.torch[:, motion_ended].view(K, worlds).float()
            completed.copy_(torch.where(done, ended, completed))
            # A time-out is not a failure: bootstrap it with the value of the state it left.
            buffer["rewards"][t] = reward.view(K, worlds) + args.gamma * value * truncated.view(K, worlds)
            buffer["done"][t] = done
        last_value = forward("critic", params, normalize(obs.view(K, worlds, obs_dim))).squeeze(-1)
        buffer["values"][T] = last_value * normalizer["ret_var"].sqrt() + normalizer["ret_mean"]

        advantages = torch.zeros(T, K, worlds, device=DEVICE)
        last = torch.zeros(K, worlds, device=DEVICE)
        for t in reversed(range(T)):
            running = 1.0 - buffer["done"][t]
            delta = buffer["rewards"][t] + args.gamma * buffer["values"][t + 1] * running - buffer["values"][t]
            last = delta + args.gamma * args.lam * running * last
            advantages[t] = last
        for name in ("obs", "actions", "means", "log_probs"):
            flat[name].copy_(buffer[name].transpose(0, 1).reshape(flat[name].shape))
        returns = (advantages + buffer["values"][:T]).transpose(0, 1).reshape(K, batch)
        update_running(normalizer["ret_mean"], normalizer["ret_var"], returns, 1, batch)
        flat["returns"].copy_((returns - normalizer["ret_mean"]) / normalizer["ret_var"].sqrt())
        advantages = advantages.transpose(0, 1).reshape(K, batch)
        flat["advantages"].copy_(
            (advantages - advantages.mean(1, keepdim=True)) / (advantages.std(1, keepdim=True) + 1.0e-8)
        )
        old_std.copy_(params[:, None, layout["log_std"] :].exp())


class GraphUpdate:
    """The policy update of every candidate as one CUDA graph, each with its own clipping and learning rate."""

    def __init__(self):
        self.first = torch.zeros_like(params)
        self.second = torch.zeros_like(params)
        self.count = torch.zeros((), device=DEVICE)
        self.lr = torch.full((K, 1), args.lr, device=DEVICE)
        self.perms = torch.zeros(args.epochs, batch, dtype=torch.long, device=DEVICE)
        self.graph = None
        self.calls = 0

    def __call__(self):
        self.perms.copy_(torch.rand(args.epochs, batch, device=DEVICE).argsort(dim=-1))
        self.calls += 1
        if self.graph is not None:
            self.graph.replay()
        elif self.calls < 3:
            self.steps()
        else:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.steps()

    def steps(self):
        size = batch // args.minibatches
        for epoch in range(args.epochs):
            for k in range(args.minibatches):
                self.step(self.perms[epoch, k * size : (k + 1) * size])

    def step(self, indices):
        normalized = normalize(flat["obs"][:, indices])
        mean = forward("actor", params, normalized)
        value = forward("critic", params, normalized).squeeze(-1)
        log_std = params[:, None, layout["log_std"] :]
        std = log_std.exp()
        log_prob = (-0.5 * ((flat["actions"][:, indices] - mean) / std) ** 2 - log_std).sum(-1)
        ratio = (log_prob - flat["log_probs"][:, indices]).exp()
        advantage = flat["advantages"][:, indices]
        policy_loss = -torch.min(ratio * advantage, ratio.clamp(1.0 - args.clip, 1.0 + args.clip) * advantage).mean(1)
        value_loss = (value - flat["returns"][:, indices]).square().mean(1)
        # The candidates share no parameters, so the gradient of the sum is each candidate's own gradient.
        (gradient,) = torch.autograd.grad((policy_loss + args.value_coef * value_loss).sum(), params)
        with torch.no_grad():
            kl = flat["means"][:, indices] - mean
            kl = (kl.square() + old_std.square()) / (2.0 * std.square()) + (std / old_std).log() - 0.5
            kl = kl.sum(-1).mean(1, keepdim=True)
            gradient = gradient * (1.0 / (gradient.norm(dim=1, keepdim=True) + 1.0e-6)).clamp(max=1.0)
            self.count += 1.0
            self.first.mul_(0.9).add_(gradient, alpha=0.1)
            self.second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
            first = self.first / (1.0 - 0.9**self.count)
            second = self.second / (1.0 - 0.999**self.count)
            params.sub_(self.lr * first / (second.sqrt() + 1.0e-5))
            slower, faster = (self.lr / 1.5).clamp(min=1.0e-5), (self.lr * 1.5).clamp(max=args.max_lr)
            self.lr.copy_(torch.where(kl > 2.0 * args.kl, slower, torch.where(kl < 0.5 * args.kl, faster, self.lr)))


def exploit(scores):
    """Have the worst candidates copy the policy and the reward weights of the best, with the weights perturbed."""
    order = scores.argsort(descending=True)
    cut = max(1, round(args.pbt_fraction * K))
    with torch.no_grad():
        for loser, winner in zip(order[-cut:].tolist(), order[:cut].tolist()):
            params[loser] = params[winner]
            for value in (update.first, update.second, update.lr, *normalizer.values()):
                value[loser] = value[winner]
            factors = 1.25 ** (2.0 * torch.rand(len(reward_names), device=DEVICE) - 1.0)
            reward_weights[loser * worlds : (loser + 1) * worlds] = reward_weights[winner * worlds] * factors
    return order[-cut:].tolist(), order[:cut].tolist()


update = GraphUpdate()
history = []
wp.synchronize()
start = time.perf_counter()
for k in range(1, iterations + 1):
    rollout()
    update()
    if k % args.log_interval == 0 or k == iterations:
        torch.cuda.synchronize()
        completion = completed.mean(1)
        error = 100.0 * body_error.view(K, worlds).sqrt().mean(1)
        history.append(
            {
                "iteration": k,
                "time": time.perf_counter() - start,
                "completion": completion.tolist(),
                "error_cm": error.tolist(),
                "weights": reward_weights[::worlds].tolist(),
            }
        )
        if args.pbt and k % args.pbt_interval == 0 and k < iterations:
            losers, winners = exploit(completion - 0.01 * error)
            print(f"PBT iteration {k}: {losers} copy {winners}", flush=True)
train_time = time.perf_counter() - start
samples = iterations * T * num_envs
print(
    f"TRAIN {train_time:.1f} s for {K} candidates, {samples / train_time / 1e6:.2f} M steps/s, graphs {env.num_graphs}"
)

# Rank the candidates by how much of the clip their worlds track, then by how closely.
final = history[-1]
order = sorted(range(K), key=lambda c: (-final["completion"][c], final["error_cm"][c]))
for rank, c in enumerate(order):
    reached = next((row["time"] for row in history if row["completion"][c] >= 0.9), None)
    weights = " ".join(f"{name}={final['weights'][c][i]:.2f}" for i, name in enumerate(reward_names))
    print(
        f"RANK {rank + 1:2d} candidate {c:2d} completion {final['completion'][c]:.3f}"
        f" error {final['error_cm'][c]:.1f} cm"
        f" reached 90% at {'never' if reached is None else f'{reached:.1f} s'}  {weights}"
    )
if args.out:
    summary = {"args": vars(args), "train_time": train_time, "reward_names": reward_names, "history": history}
    with open(args.out, "w") as file:
        json.dump(summary, file)
env.close()
