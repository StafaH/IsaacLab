# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an MDP2 motion tracking task in seconds with a lean PPO loop, then evaluate its snapshots.

    uv run python scripts/reinforcement_learning/mdp2/train_tracking.py --motion walk --seed 0

The loop keeps everything on the device: the environment step replays one CUDA graph, the policy update another,
and the learner reads no value back while it trains. The reported time is the active training time (rollouts and
updates), as in InstantMimic (arXiv 2609.09821): startup, kernel compilation, and the evaluation are outside it.
Every saved policy is then run without exploration noise from the first frame of the clip to its end.
``--runtime mdp1`` trains the same task written with MDP1 managers, see ``tracking_mdp1_baseline.py``.
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
from isaaclab_tasks.mdp2.tracking.config.humanoid.humanoid_env_cfg import humanoid_task
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import resolve_presets

parser = argparse.ArgumentParser()
parser.add_argument("--motion", default="walk")
parser.add_argument("--runtime", default="mdp2", choices=["mdp2", "mdp1"])
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--iterations", type=int, default=150)
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--clip", type=float, default=0.3)
parser.add_argument("--kl", type=float, default=0.15)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--value_coef", type=float, default=4.0)
parser.add_argument("--entropy", type=float, default=0.0)
parser.add_argument("--action_scale", type=float, default=None)
parser.add_argument("--max_lr", type=float, default=1.0e-2)
parser.add_argument("--tag", default="")
parser.add_argument("--adaptive", type=float, default=0.0, help="Share of episodes that start where episodes fail")
parser.add_argument("--lost_distance", type=float, default=None, help="RMS body distance that ends an episode [m]")
parser.add_argument("--activation", default="elu", choices=["elu", "silu"])
parser.add_argument("--optimizer", default="adam", choices=["adam", "muon"])
parser.add_argument("--value_norm", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--graph_update", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--profile", action="store_true")
parser.add_argument(
    "--budget", type=float, default=None, help="Samples to train for, in millions, instead of --iterations"
)
parser.add_argument("--init_std", type=float, default=0.3)
parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 128])
parser.add_argument("--snapshot_interval", type=int, default=5)
parser.add_argument("--eval_envs", type=int, default=256)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--out", default=None)
parser.add_argument("--video", default=None, help="Path of an MP4 of the final policy, from the first frame")
args = parser.parse_args()

DEVICE = "cuda:0"
TASK = "Isaac-Tracking-Humanoid-Walk-MDP2"


def make_env(num_envs, evaluation=False, camera=False):
    """Create the training environment, or the one that starts on the first frame of the clip without noise."""
    if args.runtime == "mdp1":
        from tracking_mdp1_baseline import tracking_cfg

        from isaaclab.envs import ManagerBasedRLEnv

        cfg = tracking_cfg(args.motion, (0.0, 0.0), 0.0) if evaluation else tracking_cfg(args.motion)
        cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, num_envs, args.seed
        return ManagerBasedRLEnv(cfg)

    cfg = resolve_presets(load_cfg_from_registry(TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    cfg.sim.device, cfg.scene.num_envs, cfg.seed = DEVICE, num_envs, args.seed
    cfg.task = humanoid_task(args.motion, adaptive=args.adaptive)
    if evaluation:
        cfg.task = humanoid_task(args.motion, start_range=(0.0, 0.0))
        cfg.task.events["reset_to_motion"].params["noise"] = 0.0
    if args.action_scale is not None:
        cfg.task.actions["joint_pos"].scale = args.action_scale
    if args.lost_distance is not None:
        cfg.task.terminations["tracking_lost"].params["threshold"] = args.lost_distance**2
    if camera:
        add_camera(cfg)
    return gym.make(TASK, cfg=cfg).unwrapped


def body_error(env):
    """Root mean squared distance of the bodies to the tracked frame [m], per world."""
    if args.runtime == "mdp1":
        return env.command_manager.get_term("motion").body_pos_error().sqrt()
    return env.state.arrays[mdp2.quantities.body_pos_error].torch.sqrt()


def mlp(sizes):
    layers = []
    for k in range(len(sizes) - 1):
        layers += [
            torch.nn.Linear(sizes[k], sizes[k + 1]),
            torch.nn.SiLU() if args.activation == "silu" else torch.nn.ELU(),
        ]
    return torch.nn.Sequential(*layers[:-1])


class Agent(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor = mlp([obs_dim, *args.hidden, action_dim])
        self.critic = mlp([obs_dim, *args.hidden, 1])
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), float(torch.tensor(args.init_std).log())))
        self.register_buffer("mean", torch.zeros(obs_dim))
        self.register_buffer("var", torch.ones(obs_dim))
        self.register_buffer("count", torch.tensor(1.0e-4))
        self.register_buffer("return_mean", torch.zeros(()))
        self.register_buffer("return_var", torch.ones(()))
        self.register_buffer("return_count", torch.tensor(1.0e-4))

    def normalize(self, obs):
        return (obs - self.mean) / (self.var.sqrt() + 1.0e-8)

    def update_normalizer(self, obs):
        batch_mean, batch_var, batch_count = obs.mean(0), obs.var(0), obs.shape[0]
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.var.copy_(
            (self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total) / total
        )
        self.mean.add_(delta * batch_count / total)
        self.count.copy_(total)

    def value(self, normalized):
        """Return the value of normalized observations, in reward units."""
        value = self.critic(normalized).squeeze(-1)
        return value * self.return_var.sqrt() + self.return_mean if args.value_norm else value

    def update_return_normalizer(self, returns):
        batch_mean, batch_var, batch_count = returns.mean(), returns.var(), returns.numel()
        total = self.return_count + batch_count
        delta = batch_mean - self.return_mean
        self.return_var.copy_(
            (
                self.return_var * self.return_count
                + batch_var * batch_count
                + delta**2 * self.return_count * batch_count / total
            )
            / total
        )
        self.return_mean.add_(delta * batch_count / total)
        self.return_count.copy_(total)

    def forward(self, obs, actions):
        return self.evaluate(obs, actions)

    def evaluate(self, obs, actions):
        obs = self.normalize(obs)
        mean, std = self.actor(obs), self.log_std.exp()
        log_prob = (-0.5 * ((actions - mean) / std) ** 2 - self.log_std).sum(-1)
        return log_prob, self.critic(obs).squeeze(-1), mean, std


torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
env = make_env(args.num_envs)
obs = env.reset()[0]["policy"]
num_envs, obs_dim, action_dim = obs.shape[0], obs.shape[1], env.single_action_space.shape[0]
agent = Agent(obs_dim, action_dim).to(DEVICE)


def make_optimizers():
    """Adam for every parameter, or Muon for the hidden weight matrices and Adam for the rest."""
    if args.optimizer == "adam":
        return [torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1.0e-5, fused=True)]
    hidden = [layer.weight for net in (agent.actor, agent.critic) for layer in list(net)[2:-1:2]]
    others = [p for p in agent.parameters() if all(p is not weight for weight in hidden)]
    return [
        torch.optim.Muon(
            hidden, lr=args.lr, momentum=0.95, nesterov=True, weight_decay=3.0e-5, adjust_lr_fn="match_rms_adamw"
        ),
        torch.optim.Adam(others, lr=args.lr, eps=1.0e-5, fused=True),
    ]


optimizers = make_optimizers()
evaluate = torch.compile(agent.evaluate) if args.compile else agent.evaluate

T = args.steps
if args.budget is not None:
    args.iterations = round(args.budget * 1.0e6 / (T * num_envs))
    args.snapshot_interval = max(1, args.iterations // 30)
buffer = {
    "obs": torch.zeros(T, num_envs, obs_dim, device=DEVICE),
    "actions": torch.zeros(T, num_envs, action_dim, device=DEVICE),
    "means": torch.zeros(T, num_envs, action_dim, device=DEVICE),
    "log_probs": torch.zeros(T, num_envs, device=DEVICE),
    "values": torch.zeros(T + 1, num_envs, device=DEVICE),
    "rewards": torch.zeros(T, num_envs, device=DEVICE),
    "terminated": torch.zeros(T, num_envs, device=DEVICE),
    "done": torch.zeros(T, num_envs, device=DEVICE),
}
advantages = torch.zeros(T, num_envs, device=DEVICE)
batch = T * num_envs


flat = {name: buffer[name][:T].reshape(batch, -1).squeeze(-1) for name in ("obs", "actions", "means", "log_probs")}
flat["returns"] = torch.zeros(batch, device=DEVICE)
flat["advantages"] = torch.zeros(batch, device=DEVICE)
old_std = torch.zeros(action_dim, device=DEVICE)


def rollout():
    """Collect a rollout and its returns and advantages into the buffers."""
    global obs
    with torch.no_grad():
        for t in range(T):
            agent.update_normalizer(obs)
            normalized = agent.normalize(obs)
            mean, value = agent.actor(normalized), agent.value(normalized)
            action = mean + agent.log_std.exp() * torch.randn_like(mean)
            buffer["obs"][t], buffer["actions"][t], buffer["values"][t] = obs, action, value
            buffer["means"][t] = mean
            buffer["log_probs"][t] = (-0.5 * ((action - mean) / agent.log_std.exp()) ** 2 - agent.log_std).sum(-1)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = obs["policy"]
            # A time-out is not a failure: bootstrap it with the value of the state it left.
            buffer["rewards"][t] = reward + args.gamma * value * truncated
            buffer["terminated"][t] = terminated
            buffer["done"][t] = terminated | truncated
        buffer["values"][T] = agent.value(agent.normalize(obs))

        last = torch.zeros(num_envs, device=DEVICE)
        for t in reversed(range(T)):
            running = 1.0 - buffer["done"][t]
            delta = buffer["rewards"][t] + args.gamma * buffer["values"][t + 1] * running - buffer["values"][t]
            last = delta + args.gamma * args.lam * running * last
            advantages[t] = last
        returns = (advantages + buffer["values"][:T]).reshape(batch)
        if args.value_norm:
            agent.update_return_normalizer(returns)
            returns = (returns - agent.return_mean) / agent.return_var.sqrt()
        flat["returns"].copy_(returns)
        flat_advantages = advantages.reshape(batch)
        flat["advantages"].copy_((flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1.0e-8))
        old_std.copy_(agent.log_std.exp())


def losses(indices, log_prob, value, mean, std, log_std):
    """Return the PPO loss of a minibatch and the KL divergence from the policy that collected it."""
    ratio = (log_prob - flat["log_probs"][indices]).exp()
    advantage = flat["advantages"][indices]
    policy_loss = -torch.min(ratio * advantage, ratio.clamp(1.0 - args.clip, 1.0 + args.clip) * advantage).mean()
    value_loss = (value - flat["returns"][indices]).square().mean()
    kl = flat["means"][indices] - mean.detach()
    kl = (kl.square() + old_std.square()) / (2.0 * std.detach().square()) + (std.detach() / old_std).log() - 0.5
    return policy_loss + args.value_coef * value_loss - args.entropy * log_std.sum(), kl.sum(-1).mean()


def update():
    """Update the policy with the optimizers of Torch, reading the KL divergence back after every minibatch."""
    for _ in range(args.epochs):
        for indices in torch.randperm(batch, device=DEVICE).chunk(args.minibatches):
            outputs = evaluate(flat["obs"][indices], flat["actions"][indices])
            loss, kl = losses(indices, *outputs, agent.log_std)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            for optimizer in optimizers:
                optimizer.step()

            # Keep the policy change per update near the target, as RSL-RL's adaptive schedule does.
            kl = kl.item()
            lr = optimizers[0].param_groups[0]["lr"]
            if kl > 2.0 * args.kl:
                lr = max(lr / 1.5, 1.0e-5)
            elif kl < 0.5 * args.kl:
                lr = min(lr * 1.5, args.max_lr)
            for optimizer in optimizers:
                optimizer.param_groups[0]["lr"] = lr


class GraphUpdate:
    """The whole policy update of an iteration as one CUDA graph: every minibatch's forward and backward pass, the
    gradient clipping, Adam, and the adaptive learning rate, with nothing read back.

    The parameters are one vector that the networks view, so the optimizer is a few kernels for all of them.
    """

    def __init__(self):
        self.names = [name for name, _ in agent.named_parameters()]
        self.vector = torch.nn.utils.parameters_to_vector(agent.parameters()).detach().clone().requires_grad_()
        self.first = torch.zeros_like(self.vector)
        self.second = torch.zeros_like(self.vector)
        self.count = torch.zeros((), device=DEVICE)
        self.lr = torch.tensor(args.lr, device=DEVICE)
        self.perms = torch.zeros(args.epochs, batch, dtype=torch.long, device=DEVICE)
        self.graph = None
        self.calls = 0

    def reset(self):
        for value in (self.first, self.second, self.count):
            value.zero_()
        self.lr.fill_(args.lr)

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
        log_prob, value, mean, std = torch.func.functional_call(
            agent, views, (flat["obs"][indices], flat["actions"][indices])
        )
        loss, kl = losses(indices, log_prob, value, mean, std, views["log_std"])
        (gradient,) = torch.autograd.grad(loss, self.vector)
        with torch.no_grad():
            gradient = gradient * (1.0 / (gradient.norm() + 1.0e-6)).clamp(max=1.0)
            self.count += 1.0
            self.first.mul_(0.9).add_(gradient, alpha=0.1)
            self.second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
            first = self.first / (1.0 - 0.9**self.count)
            second = self.second / (1.0 - 0.999**self.count)
            self.vector.sub_(self.lr * first / (second.sqrt() + 1.0e-5))
            slower, faster = (self.lr / 1.5).clamp(min=1.0e-5), (self.lr * 1.5).clamp(max=args.max_lr)
            self.lr.copy_(torch.where(kl > 2.0 * args.kl, slower, torch.where(kl < 0.5 * args.kl, faster, self.lr)))


if args.graph_update:
    update = GraphUpdate()


def iteration():
    rollout()
    update()


# Warm up outside the clock: the first steps capture the CUDA graph, and the first update compiles.
state = {name: value.clone() for name, value in agent.state_dict().items()}
for _ in range(3):
    iteration()
agent.load_state_dict(state)
optimizers = make_optimizers()
if args.graph_update:
    update.reset()
obs = env.reset()[0]["policy"]

snapshots, times, curve = [], [], []
wp.synchronize()
start = time.perf_counter()
for k in range(1, args.iterations + 1):
    iteration()
    if k % args.snapshot_interval == 0:
        snapshots.append({name: value.clone() for name, value in agent.state_dict().items()})
        wp.synchronize()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        curve.append(env.extras.get("log", {}).get("Episode_Termination/motion_ended", 0.0))
        if args.debug:
            print(
                k,
                "reward",
                buffer["rewards"].mean().item(),
                "std",
                agent.log_std.exp().mean().item(),
                "lr",
                float(update.lr) if args.graph_update else optimizers[0].param_groups[0]["lr"],
                "value",
                buffer["values"].mean().item(),
                "done",
                buffer["done"].mean().item(),
                "completion",
                float(curve[-1]),
            )
train_time = times[-1]
if args.profile:
    spent = {"rollout": 0.0, "update": 0.0}
    for _ in range(20):
        for name, part in (("rollout", rollout), ("update", update)):
            wp.synchronize()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            part()
            wp.synchronize()
            torch.cuda.synchronize()
            spent[name] += (time.perf_counter() - begin) / 20
    print(f"PROFILE rollout {1e3 * spent['rollout']:.1f} ms update {1e3 * spent['update']:.1f} ms per iteration")
samples = args.iterations * batch
print(f"TRAIN {train_time:.2f} s for {args.iterations} iterations, {samples / train_time / 1e6:.2f} M steps/s")
env.close()

# Evaluate every snapshot without exploration noise, from the first frame of the clip to its end.
env = make_env(args.eval_envs, evaluation=True)
horizon = int(env.max_episode_length)
motions = args.motion.split(",")
if args.runtime == "mdp1":
    motion_command = env.command_manager.get_term("motion")
    clip_ends, world_end = motion_command.clip["clip_range"][:, 1], lambda: motion_command.end
else:
    reset_state = {
        name: env.state.arrays[mdp2.term_state("events.reset_to_motion", name)].torch for name in ("clip_range", "end")
    }
    clip_ends, world_end = reset_state["clip_range"][:, 1], lambda: reset_state["end"]
results = []
with torch.no_grad():
    for index, snapshot in enumerate(snapshots):
        agent.load_state_dict(snapshot)
        obs = env.reset()[0]["policy"]
        # A world succeeds if its first episode reaches the end of its clip. Worlds draw their clips at random.
        clip = (world_end().float()[:, None] == clip_ends[None, :]).float().argmax(dim=-1)
        finished = torch.zeros(args.eval_envs, dtype=torch.bool, device=DEVICE)
        succeeded = torch.zeros_like(finished)
        error, steps = torch.zeros(args.eval_envs, device=DEVICE), torch.zeros(args.eval_envs, device=DEVICE)
        for _ in range(horizon):
            error += body_error(env) * ~finished
            steps += ~finished
            obs, _, terminated, truncated, _ = env.step(agent.actor(agent.normalize(obs)))
            obs = obs["policy"]
            succeeded |= truncated & ~finished
            finished |= terminated | truncated
        per_clip = [succeeded[clip == c].float().mean().item() for c in range(len(motions))]
        results.append(
            {
                "iteration": (index + 1) * args.snapshot_interval,
                "time": times[index],
                "success": min(per_clip),
                "per_clip": per_clip,
                "body_error_m": (error.sum() / steps.sum().clamp(min=1)).item(),
                "train_completion": float(curve[index]),
            }
        )
env.close()

first = next((result for result in results if result["success"] >= 0.9), None)
for result in results:
    print(
        f"  it {result['iteration']:4d}  {result['time']:6.2f} s  success {result['success']:.3f}"
        f"  body error {100 * result['body_error_m']:.1f} cm  train completion {result['train_completion']:.3f}"
    )
print("PER_CLIP " + " ".join(f"{motion} {value:.2f}" for motion, value in zip(motions, results[-1]["per_clip"])))
print("TIME_TO_90", "none" if first is None else f"{first['time']:.2f} s at iteration {first['iteration']}")
print(
    f"SUMMARY {args.tag} t90 {'none' if first is None else round(first['time'], 2)} final {results[-1]['success']:.3f}"
    f" samples90 {'none' if first is None else round(first['iteration'] * batch / 1e6, 2)}M"
    f" err {100 * results[-1]['body_error_m']:.1f}cm fps {samples / train_time / 1e6:.2f}M success@ "
    + " ".join(f"{r['iteration']}:{r['success']:.2f}" for r in results[:: max(1, len(results) // 8)])
)
if args.out:
    summary = {"args": vars(args), "train_time": train_time, "steps_per_s": samples / train_time, "results": results}
    with open(args.out, "w") as file:
        json.dump(summary, file)

if args.video:
    # The final policy from the first frame of its clips: one world for a single clip, a row of worlds for many.
    tiles = 1 if len(motions) == 1 else 6
    env = make_env(tiles, evaluation=True, camera=True)
    steps = 3 * int(env.max_episode_length) if len(motions) == 1 else 450
    record(env, lambda obs: agent.actor(agent.normalize(obs)), args.video, steps, min(tiles, 3))
    env.close()
