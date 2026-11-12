# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A lean PPO learner whose data stays on the device and whose update is one CUDA graph."""

import torch


def mlp(sizes):
    layers = []
    for k in range(len(sizes) - 1):
        layers += [torch.nn.Linear(sizes[k], sizes[k + 1]), torch.nn.ELU()]
    return torch.nn.Sequential(*layers[:-1])


class Normalizer(torch.nn.Module):
    """Running mean and variance of a stream of batches."""

    def __init__(self, size):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1.0e-4))

    def update(self, values):
        batch_mean, batch_var, batch_count = values.mean(0), values.var(0), values.shape[0]
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.var.copy_(
            (self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total) / total
        )
        self.mean.add_(delta * batch_count / total)
        self.count.copy_(total)

    def forward(self, values):
        return (values - self.mean) / (self.var.sqrt() + 1.0e-8)


class Agent(torch.nn.Module):
    """A Gaussian policy and a critic with normalized inputs and returns."""

    def __init__(self, obs_dim, action_dim, hidden, init_std):
        super().__init__()
        self.actor = mlp([obs_dim, *hidden, action_dim])
        self.critic = mlp([obs_dim, *hidden, 1])
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), float(torch.tensor(init_std).log())))
        self.obs = Normalizer(obs_dim)
        self.returns = Normalizer(())

    def value(self, normalized):
        """Return the value of normalized observations, in reward units."""
        return self.critic(normalized).squeeze(-1) * self.returns.var.sqrt() + self.returns.mean

    def forward(self, obs, actions):
        obs = self.obs(obs)
        mean, std = self.actor(obs), self.log_std.exp()
        log_prob = (-0.5 * ((actions - mean) / std) ** 2 - self.log_std).sum(-1)
        return log_prob, self.critic(obs).squeeze(-1), mean, std


class PPO:
    """PPO for ``num_envs`` environments: call :meth:`act` and :meth:`record` every step, then :meth:`update`."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        num_envs,
        device,
        hidden=(256, 256, 128),
        steps=24,
        epochs=5,
        minibatches=4,
        lr=1.0e-3,
        max_lr=1.0e-2,
        kl=0.01,
        gamma=0.99,
        lam=0.95,
        clip=0.2,
        value_coef=1.0,
        entropy=0.0,
        init_std=1.0,
    ):
        self.agent = Agent(obs_dim, action_dim, hidden, init_std).to(device)
        self.steps, self.epochs, self.minibatches, self.batch = steps, epochs, minibatches, steps * num_envs
        self.max_lr, self.kl, self.gamma, self.lam, self.clip = max_lr, kl, gamma, lam, clip
        self.value_coef, self.entropy = value_coef, entropy
        zeros = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
        self.buffer = {
            "obs": zeros(steps, num_envs, obs_dim),
            "actions": zeros(steps, num_envs, action_dim),
            "means": zeros(steps, num_envs, action_dim),
            "log_probs": zeros(steps, num_envs),
            "values": zeros(steps + 1, num_envs),
            "rewards": zeros(steps, num_envs),
            "done": zeros(steps, num_envs),
        }
        names = ("obs", "actions", "means", "log_probs")
        self.flat = {name: self.buffer[name].reshape(self.batch, -1).squeeze(-1) for name in names}
        self.flat |= {"returns": zeros(self.batch), "advantages": zeros(self.batch)}
        self.old_std = zeros(action_dim)
        self.t = 0

        # The update: one parameter vector that the networks view, Adam moments, and the learning rate.
        self.vector = torch.nn.utils.parameters_to_vector(self.agent.parameters()).detach().clone().requires_grad_()
        self.first, self.second = torch.zeros_like(self.vector), torch.zeros_like(self.vector)
        self.count = zeros(())
        self.lr = torch.tensor(lr, device=device)
        self.perms = torch.zeros(epochs, self.batch, dtype=torch.long, device=device)
        self.graph, self.calls = None, 0
        self.initial = {name: value.clone() for name, value in self.agent.state_dict().items()}
        self.initial_lr = lr

    def reset(self):
        """Forget everything learned, and keep the captured update: for warming up outside a clock."""
        self.agent.load_state_dict(self.initial)
        for value in (self.first, self.second, self.count):
            value.zero_()
        self.lr.fill_(self.initial_lr)
        self.t = 0

    def policy(self, obs):
        """Return the actions of the policy without exploration noise."""
        return self.agent.actor(self.agent.obs(obs))

    def act(self, obs):
        """Return exploring actions for ``obs``, and store the step."""
        agent, buffer, t = self.agent, self.buffer, self.t
        with torch.no_grad():
            agent.obs.update(obs)
            normalized = agent.obs(obs)
            mean, value = agent.actor(normalized), agent.value(normalized)
            std = agent.log_std.exp()
            action = mean + std * torch.randn_like(mean)
            buffer["obs"][t], buffer["actions"][t], buffer["values"][t], buffer["means"][t] = obs, action, value, mean
            buffer["log_probs"][t] = (-0.5 * ((action - mean) / std) ** 2 - agent.log_std).sum(-1)
        return action

    def record(self, reward, terminated, truncated):
        """Store the outcome of the step that followed :meth:`act`."""
        # A time-out is not a failure: bootstrap it with the value of the state it left.
        self.buffer["rewards"][self.t] = reward + self.gamma * self.buffer["values"][self.t] * truncated
        self.buffer["done"][self.t] = terminated | truncated
        self.t += 1

    def update(self, obs):
        """Compute returns and advantages with the value of the observations after the rollout, and update."""
        agent, buffer, flat, steps = self.agent, self.buffer, self.flat, self.steps
        with torch.no_grad():
            buffer["values"][steps] = agent.value(agent.obs(obs))
            advantages = torch.zeros_like(buffer["rewards"])
            last = torch.zeros_like(buffer["rewards"][0])
            for t in reversed(range(steps)):
                running = 1.0 - buffer["done"][t]
                delta = buffer["rewards"][t] + self.gamma * buffer["values"][t + 1] * running - buffer["values"][t]
                last = delta + self.gamma * self.lam * running * last
                advantages[t] = last
            returns = (advantages + buffer["values"][:steps]).reshape(self.batch)
            agent.returns.update(returns)
            flat["returns"].copy_((returns - agent.returns.mean) / agent.returns.var.sqrt())
            advantages = advantages.reshape(self.batch)
            flat["advantages"].copy_((advantages - advantages.mean()) / (advantages.std() + 1.0e-8))
            self.old_std.copy_(agent.log_std.exp())
            self.perms.copy_(torch.rand(self.epochs, self.batch, device=obs.device).argsort(dim=-1))
        self.t = 0

        # The first calls run eagerly, the third is captured, and the rest replay the graph.
        self.calls += 1
        if self.graph is not None:
            self.graph.replay()
        elif self.calls < 3:
            self._update()
        else:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._update()

    def _update(self):
        agent, flat = self.agent, self.flat
        with torch.no_grad():
            self.vector.copy_(torch.nn.utils.parameters_to_vector(agent.parameters()))
        size = self.batch // self.minibatches
        for epoch in range(self.epochs):
            for k in range(self.minibatches):
                indices = self.perms[epoch, k * size : (k + 1) * size]
                views, offset = {}, 0
                for name, param in agent.named_parameters():
                    views[name] = self.vector[offset : offset + param.numel()].view(param.shape)
                    offset += param.numel()
                inputs = (flat["obs"][indices], flat["actions"][indices])
                log_prob, value, mean, std = torch.func.functional_call(agent, views, inputs)
                ratio = (log_prob - flat["log_probs"][indices]).exp()
                advantage = flat["advantages"][indices]
                clipped = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage
                loss = -torch.min(ratio * advantage, clipped).mean()
                loss = loss + self.value_coef * (value - flat["returns"][indices]).square().mean()
                loss = loss - self.entropy * views["log_std"].sum()
                (gradient,) = torch.autograd.grad(loss, self.vector)
                with torch.no_grad():
                    kl = flat["means"][indices] - mean
                    kl = (kl.square() + self.old_std.square()) / (2.0 * std.square()) + (std / self.old_std).log() - 0.5
                    kl = kl.sum(-1).mean()
                    gradient = gradient * (1.0 / (gradient.norm() + 1.0e-6)).clamp(max=1.0)
                    self.count += 1.0
                    self.first.mul_(0.9).add_(gradient, alpha=0.1)
                    self.second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
                    first = self.first / (1.0 - 0.9**self.count)
                    second = self.second / (1.0 - 0.999**self.count)
                    self.vector.sub_(self.lr * first / (second.sqrt() + 1.0e-5))
                    slower, faster = (self.lr / 1.5).clamp(min=1.0e-5), (self.lr * 1.5).clamp(max=self.max_lr)
                    lr = torch.where(kl > 2.0 * self.kl, slower, torch.where(kl < 0.5 * self.kl, faster, self.lr))
                    self.lr.copy_(lr)
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(self.vector, agent.parameters())
