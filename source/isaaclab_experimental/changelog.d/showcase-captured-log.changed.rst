Changed :class:`~isaaclab_experimental.envs.MDP2RLEnv` to sum the episode rewards and average the logged terms inside
the captured step, so that a step launches one copy of the averages for ``extras["log"]`` instead of a dozen Torch
operations.
