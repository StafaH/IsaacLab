"""PPO for worlds of several kinds, whose advantages are normalized per kind."""

import torch
from ppo import PPO


class HybridPPO(PPO):
    """PPO whose environments have a kind each, in ``kinds``, with a discount per environment in ``gamma``.

    The advantages of a kind are normalized over the steps of its environments, so that a kind with rare rewards
    counts as much in the policy loss as one with a reward every step.
    """

    def __init__(self, *args, kinds, num_kinds=2, kind_norm=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.kinds, self.num_kinds, self.kind_norm = kinds, num_kinds, kind_norm

    def update(self, obs, critic_obs=None):
        agent, buffer, flat, steps = self.agent, self.buffer, self.flat, self.steps
        with torch.no_grad():
            buffer["values"][steps] = self._value(agent.obs(obs), critic_obs)
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

            if self.kind_norm:
                # Mean and deviation of the advantages of each kind, over the environments that have it.
                one_hot = torch.nn.functional.one_hot(self.kinds, self.num_kinds).float()
                count = (one_hot.sum(0) * steps).clamp(min=1.0)
                mean = (advantages @ one_hot).sum(0) / count
                deviation = ((advantages.square() @ one_hot).sum(0) / count - mean.square()).clamp(min=0.0).sqrt()
                advantages = (advantages - mean[self.kinds]) / (deviation[self.kinds] + 1.0e-8)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
            flat["advantages"].copy_(advantages.reshape(self.batch))
            self.old_std.copy_(agent.log_std.exp())
            self.perms.copy_(torch.rand(self.epochs, self.batch, device=obs.device).argsort(dim=-1))
        self.t = 0

        self.calls += 1
        if self.graph is not None:
            self.graph.replay()
        elif self.calls < 3:
            self._update()
        else:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._update()
