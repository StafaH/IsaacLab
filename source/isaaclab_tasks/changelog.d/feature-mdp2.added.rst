* Added ``Isaac-Cartpole-MDP2``, the Cartpole task written with :mod:`isaaclab_experimental.mdp2` and run by
  :class:`~isaaclab_experimental.envs.MDP2RLEnv` on Newton, in :mod:`isaaclab_tasks.mdp2`.
* Added ``Isaac-Velocity-Flat-UnitreeGo2-MDP2``, the Unitree Go2 flat velocity task written with
  :mod:`isaaclab_experimental.mdp2`, with its startup randomization and heading commands.
* Added ``Isaac-Velocity-Flat-WalkStand-UnitreeGo2-MDP2``, which trains one Go2 policy in a scene where half of the
  worlds follow velocity commands and the other half stand still, with their own rewards.
* Added ``Isaac-Lift-KukaAllegro-MDP2``, the Kuka Allegro lift task written with :mod:`isaaclab_experimental.mdp2`,
  with its adaptive difficulty curriculum, curriculum gravity, and a reset bank of valid start states per world, all
  inside the step's CUDA graph.
* Added ``Isaac-Tracking-Humanoid-Walk-MDP2``, ``Isaac-Tracking-Humanoid-Run-MDP2``, and
  ``Isaac-Tracking-Humanoid-Dance-MDP2``, in which a 28-DoF humanoid tracks a motion clip frame by frame. The clip
  is term state that the step's kernels index, resets start on a random frame of it, and actions correct the pose
  of the frame that the next step tracks.
