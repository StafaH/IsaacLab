# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the MDP2 hybrid imitation task: one policy for worlds that track a clip and worlds that follow goals.

    uv run python scripts/reinforcement_learning/mdp2/train_hil.py --mode hybrid --budget 100

PPO with a discriminator, as in HIL (arXiv 2505.12619) and adversarial motion priors: the discriminator learns to
tell the transitions of the policy from those of the clip, and its score is the style reward of every world, which
the environment computes inside its step graph. The critic also sees which task its world runs; the policy does not.
``--mode tracking`` and ``--mode goals`` train on one kind of world only, and ``--no-style`` drops the style reward.
"""

import argparse
import json
import time

import gymnasium as gym
import torch
import warp as wp
from isaaclab_experimental import mdp2
from video import add_camera, record

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.mdp2.hil import mdp as hil
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="hybrid", choices=["hybrid", "tracking", "goals"])
parser.add_argument("--style", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--style_weight", type=float, default=None)
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=float, default=100.0, help="Samples to train for, in millions")
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument("--max_lr", type=float, default=1.0e-2)
parser.add_argument("--kl", type=float, default=0.02)
parser.add_argument("--gamma", type=float, default=0.97)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--clip", type=float, default=0.2)
parser.add_argument("--value_coef", type=float, default=4.0)
parser.add_argument("--init_std", type=float, default=0.3)
parser.add_argument("--action_scale", type=float, default=None)
parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
parser.add_argument("--disc_lr", type=float, default=1.0e-4)
parser.add_argument("--disc_steps", type=int, default=2)
parser.add_argument("--disc_batch", type=int, default=4096)
parser.add_argument("--penalty", type=float, default=5.0)
parser.add_argument("--log_interval", type=int, default=20)
parser.add_argument("--out", default=None)
parser.add_argument("--video", default=None, help="Path of an MP4 of the final policy in four worlds")
args = parser.parse_args()

DEVICE = "cuda:0"
TASK = "Isaac-HIL-Humanoid-Walk-MDP2"
STYLE, RESET = "rewards.style", "events.reset_to_motion"

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True


def make_env(num_envs, camera=False):
    """Create the environment of the mode, with a camera in every world for recording."""
    cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, num_envs, args.seed
    tasks = {"hybrid": cfg.task, "tracking": cfg.task[:1], "goals": cfg.task[1:]}[args.mode]
    for task in tasks:
        if not args.style:
            task.rewards["style"] = None
        elif args.style_weight is not None:
            task.rewards["style"].weight = args.style_weight
        if args.action_scale is not None:
            task.actions["joint_pos"].scale = args.action_scale
    cfg.task = tasks if len(tasks) > 1 else tasks[0]
    if camera:
        add_camera(cfg)
    return gym.make(TASK, cfg=cfg).unwrapped


env = make_env(args.num_envs)
obs = env.reset()[0]["policy"]
num_envs, obs_dim, action_dim = obs.shape[0], obs.shape[1], env.mdp.action_dim
arrays = env.state.arrays

# The worlds of each task. With one task, every world runs it.
tracks = torch.zeros(num_envs, dtype=torch.bool, device=DEVICE)
if args.mode == "hybrid":
    tracks[: env.mdp.world_configs[0].worlds.stop] = True
elif args.mode == "tracking":
    tracks[:] = True
follows = ~tracks
task_id = follows.float()[:, None]

# The clip, as the transitions that the discriminator should score as real.
clip = {
    name: arrays[mdp2.term_state(RESET, f"clip_{name}")].torch
    for name in ("joint_pos", "joint_vel", "root_pose", "root_vel")
}
features = hil.motion_features(clip["joint_pos"], clip["joint_vel"], clip["root_pose"], clip["root_vel"])
reference = torch.cat([features[:-1], features[1:]], dim=-1)
discriminator = None
if args.style:
    discriminator = next(op for op in env.mdp.term_ops if op.name == STYLE).params["discriminator"]
    discriminator.mean.copy_(reference.mean(0))
    discriminator.std.copy_(reference.std(0).clamp(min=0.05))
    transition = arrays[mdp2.term_state(STYLE, "transition")].torch
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr)


def mlp(sizes):
    layers = []
    for k in range(len(sizes) - 1):
        layers += [torch.nn.Linear(sizes[k], sizes[k + 1]), torch.nn.ELU()]
    return torch.nn.Sequential(*layers[:-1])


class Agent(torch.nn.Module):
    """A Gaussian policy and a critic that also sees the task of its world, with running normalizers."""

    def __init__(self):
        super().__init__()
        self.actor = mlp([obs_dim, *args.hidden, action_dim])
        self.critic = mlp([obs_dim + 1, *args.hidden, 1])
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), float(torch.tensor(args.init_std).log())))
        for name, size in (("obs", obs_dim), ("ret", ())):
            self.register_buffer(f"{name}_mean", torch.zeros(size))
            self.register_buffer(f"{name}_var", torch.ones(size))
            self.register_buffer(f"{name}_count", torch.tensor(1.0e-4))

    def update_normalizer(self, name, values):
        mean, var, count = (getattr(self, f"{name}_{stat}") for stat in ("mean", "var", "count"))
        batch_mean, batch_var, batch_count = values.mean(0), values.var(0), values.shape[0]
        total = count + batch_count
        delta = batch_mean - mean
        var.copy_((var * count + batch_var * batch_count + delta**2 * count * batch_count / total) / total)
        mean.add_(delta * batch_count / total)
        count.copy_(total)

    def normalize(self, obs):
        return (obs - self.obs_mean) / (self.obs_var.sqrt() + 1.0e-8)

    def value(self, normalized, task):
        """Return the value of normalized observations, in reward units."""
        value = self.critic(torch.cat([normalized, task], dim=-1)).squeeze(-1)
        return value * self.ret_var.sqrt() + self.ret_mean

    def forward(self, obs, task, actions):
        obs = self.normalize(obs)
        mean, std = self.actor(obs), self.log_std.exp()
        log_prob = (-0.5 * ((actions - mean) / std) ** 2 - self.log_std).sum(-1)
        return log_prob, self.critic(torch.cat([obs, task], dim=-1)).squeeze(-1), mean, std


agent = Agent().to(DEVICE)
T = args.steps
batch = T * num_envs
iterations = round(args.budget * 1.0e6 / batch)
buffer = {
    "obs": torch.zeros(T, num_envs, obs_dim, device=DEVICE),
    "task": task_id.expand(T, num_envs, 1).contiguous(),
    "actions": torch.zeros(T, num_envs, action_dim, device=DEVICE),
    "means": torch.zeros(T, num_envs, action_dim, device=DEVICE),
    "log_probs": torch.zeros(T, num_envs, device=DEVICE),
    "values": torch.zeros(T + 1, num_envs, device=DEVICE),
    "rewards": torch.zeros(T, num_envs, device=DEVICE),
    "done": torch.zeros(T, num_envs, device=DEVICE),
}
if args.style:
    buffer["transitions"] = torch.zeros(T, num_envs, reference.shape[1], device=DEVICE)
flat = {name: buffer[name][:T].reshape(batch, -1).squeeze(-1) for name in ("obs", "actions", "means", "log_probs")}
flat["task"] = buffer["task"].reshape(batch, 1)
flat["returns"] = torch.zeros(batch, device=DEVICE)
flat["advantages"] = torch.zeros(batch, device=DEVICE)
advantages = torch.zeros(T, num_envs, device=DEVICE)
old_std = torch.zeros(action_dim, device=DEVICE)


def rollout():
    """Collect a rollout and its returns and advantages into the buffers."""
    global obs
    with torch.no_grad():
        for t in range(T):
            agent.update_normalizer("obs", obs)
            normalized = agent.normalize(obs)
            mean, value = agent.actor(normalized), agent.value(normalized, task_id)
            action = mean + agent.log_std.exp() * torch.randn_like(mean)
            buffer["obs"][t], buffer["actions"][t], buffer["values"][t], buffer["means"][t] = obs, action, value, mean
            buffer["log_probs"][t] = (-0.5 * ((action - mean) / agent.log_std.exp()) ** 2 - agent.log_std).sum(-1)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = obs["policy"]
            if args.style:
                buffer["transitions"][t] = transition
            # A time-out is not a failure: bootstrap it with the value of the state it left.
            buffer["rewards"][t] = reward + args.gamma * value * truncated
            buffer["done"][t] = terminated | truncated
        buffer["values"][T] = agent.value(agent.normalize(obs), task_id)

        last = torch.zeros(num_envs, device=DEVICE)
        for t in reversed(range(T)):
            running = 1.0 - buffer["done"][t]
            delta = buffer["rewards"][t] + args.gamma * buffer["values"][t + 1] * running - buffer["values"][t]
            last = delta + args.gamma * args.lam * running * last
            advantages[t] = last
        returns = (advantages + buffer["values"][:T]).reshape(batch)
        agent.update_normalizer("ret", returns)
        flat["returns"].copy_((returns - agent.ret_mean) / agent.ret_var.sqrt())
        flat_advantages = advantages.reshape(batch)
        flat["advantages"].copy_((flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1.0e-8))
        old_std.copy_(agent.log_std.exp())


class GraphUpdate:
    """The policy update of an iteration as one CUDA graph, see ``train_tracking.py``."""

    def __init__(self):
        self.vector = torch.nn.utils.parameters_to_vector(agent.parameters()).detach().clone().requires_grad_()
        self.first = torch.zeros_like(self.vector)
        self.second = torch.zeros_like(self.vector)
        self.count = torch.zeros((), device=DEVICE)
        self.lr = torch.tensor(args.lr, device=DEVICE)
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
        with torch.no_grad():
            self.vector.copy_(torch.nn.utils.parameters_to_vector(agent.parameters()))
        size = batch // args.minibatches
        for epoch in range(args.epochs):
            for k in range(args.minibatches):
                self.step(self.perms[epoch, k * size : (k + 1) * size])
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(self.vector, agent.parameters())

    def step(self, indices):
        views, offset = {}, 0
        for name, param in agent.named_parameters():
            views[name] = self.vector[offset : offset + param.numel()].view(param.shape)
            offset += param.numel()
        inputs = (flat["obs"][indices], flat["task"][indices], flat["actions"][indices])
        log_prob, value, mean, std = torch.func.functional_call(agent, views, inputs)
        ratio = (log_prob - flat["log_probs"][indices]).exp()
        advantage = flat["advantages"][indices]
        policy_loss = -torch.min(ratio * advantage, ratio.clamp(1.0 - args.clip, 1.0 + args.clip) * advantage).mean()
        value_loss = (value - flat["returns"][indices]).square().mean()
        (gradient,) = torch.autograd.grad(policy_loss + args.value_coef * value_loss, self.vector)
        with torch.no_grad():
            kl = flat["means"][indices] - mean
            kl = ((kl.square() + old_std.square()) / (2.0 * std.square()) + (std / old_std).log() - 0.5).sum(-1).mean()
            gradient = gradient * (1.0 / (gradient.norm() + 1.0e-6)).clamp(max=1.0)
            self.count += 1.0
            self.first.mul_(0.9).add_(gradient, alpha=0.1)
            self.second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
            first = self.first / (1.0 - 0.9**self.count)
            second = self.second / (1.0 - 0.999**self.count)
            self.vector.sub_(self.lr * first / (second.sqrt() + 1.0e-5))
            slower, faster = (self.lr / 1.5).clamp(min=1.0e-5), (self.lr * 1.5).clamp(max=args.max_lr)
            self.lr.copy_(torch.where(kl > 2.0 * args.kl, slower, torch.where(kl < 0.5 * args.kl, faster, self.lr)))


def update_discriminator():
    """Teach the discriminator to score the clip's transitions 1 and the policy's -1, with a gradient penalty."""
    policy_transitions = buffer["transitions"].reshape(batch, -1)
    for _ in range(args.disc_steps):
        fake = policy_transitions[torch.randint(batch, (args.disc_batch,), device=DEVICE)]
        real = reference[torch.randint(len(reference), (args.disc_batch,), device=DEVICE)].clone().requires_grad_()
        real_score = discriminator(real)
        loss = (real_score - 1.0).square().mean() + (discriminator(fake) + 1.0).square().mean()
        (slope,) = torch.autograd.grad(real_score.sum(), real, create_graph=True)
        loss = loss + args.penalty * slope.square().sum(-1).mean()
        disc_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        disc_optimizer.step()


def measure():
    """Return how each kind of world is doing, from the state after the last step."""
    log = {name: float(value) for name, value in env.extras["log"].items()}
    result = {
        "reward": buffer["rewards"].mean().item(),
        "lr": float(update.lr),
        "std": agent.log_std.exp().mean().item(),
    }
    result["episode_steps"] = 1.0 / max(buffer["done"].mean().item(), 1.0e-6)
    robot = env.scene["robot"].data
    goal = arrays[mdp2.commands.goal].torch
    speed_error = (goal[:, 2] - (goal[:, :2] * robot.root_com_vel_w.torch[:, :2]).sum(-1)).abs()
    if tracks.any():
        result["tracking_completion"] = log["Episode_Termination/motion_ended"]
        result["tracking_speed_error"] = speed_error[tracks].mean().item()
        result["tracking_error_cm"] = 100.0 * arrays[mdp2.quantities.body_pos_error].torch[tracks].sqrt().mean().item()
    if follows.any():
        result["goal_speed_error"] = speed_error[follows].mean().item()
        result["goal_falls"] = log["Episode_Termination/fallen"]
        # How far the joint pose of each world is from the nearest frame of the clip [rad, RMS over joints].
        distance = torch.cdist(robot.joint_pos.torch[follows], clip["joint_pos"]).min(dim=-1).values
        result["goal_pose_distance"] = (distance / action_dim**0.5).mean().item()
    return result


update = GraphUpdate()
history = []
wp.synchronize()
start = time.perf_counter()
for k in range(1, iterations + 1):
    rollout()
    update()
    if args.style:
        update_discriminator()
    if k % args.log_interval == 0 or k == iterations:
        torch.cuda.synchronize()
        row = {"iteration": k, "samples_m": k * batch / 1.0e6, "time": time.perf_counter() - start, **measure()}
        history.append(row)
        print("LOG " + " ".join(f"{name} {value:.3f}" for name, value in row.items()), flush=True)
train_time = time.perf_counter() - start
print(f"TRAIN {train_time:.1f} s, {iterations * batch / train_time / 1e6:.2f} M steps/s, graphs {env.num_graphs}")

# Run the policy without exploration noise, and average the measures over the steps.
totals, steps = {}, 300
with torch.no_grad():
    for _ in range(steps):
        obs = env.step(agent.actor(agent.normalize(obs)))[0]["policy"]
        for name, value in measure().items():
            totals[name] = totals.get(name, 0.0) + value / steps
print(
    "EVAL " + " ".join(f"{name} {value:.3f}" for name, value in totals.items() if name not in ("reward", "lr", "std"))
)
if args.out:
    summary = {"args": vars(args), "train_time": train_time, "history": history, "evaluation": totals}
    with open(args.out, "w") as file:
        json.dump(summary, file)
env.close()

if args.video:
    # Four worlds: with both tasks, two that track the clip, then two that follow goals.
    env = make_env(4, camera=True)
    record(env, lambda obs: agent.actor(agent.normalize(obs)), args.video, 300, 4)
    env.close()
