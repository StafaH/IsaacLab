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
* Added ``Isaac-HIL-Humanoid-Walk-MDP2``, hybrid imitation learning with the 28-DoF humanoid: half of the worlds
  track a clip and the other half follow heading goals, with shared goal-conditioned observations and a style reward
  from a discriminator that runs inside the step's CUDA graph.
* Added ``Isaac-RewardSearch-Humanoid-Walk-MDP2``, which splits the worlds of the humanoid tracking task between
  sixteen weightings of its rewards that share one set of kernels, for training one policy per candidate in one
  simulation and for population-based training on the per-world reward weights.
* Added ``Isaac-Tracking-Humanoid-{Hop,Cartwheel,Backflip,All}-MDP2``: the motion tracking task takes several clips
  for one policy, with three more clips of the humanoid, and can start episodes where earlier episodes failed, with a
  reset event that counts failures per segment of the clips inside the step's CUDA graph.
* Added ``Isaac-Recovery-Humanoid-Dance-MDP2``, in which half of the worlds track a clip while their robot is
  pushed, and ``Isaac-Distillation-Flat-UnitreeGo2-MDP2``, in which teacher worlds observe the Go2 velocity task
  without noise and student worlds also compute the noisy, history-only observations of a student.
* Added ``Isaac-Skills-UnitreeGo2-MDP2``, in which the worlds are split between four skills of the Go2 (walking,
  standing and walking on its hind legs, and getting up), each with its own rewards, terminations, and resets, for
  one policy that observes which skill its world practices.
* Added ``Isaac-Tracking-Humanoid-LiveEval-MDP2``, the walk tracking task with evaluation worlds that start on
  the first frame without noise next to the training worlds, for an evaluation read from the training simulation.
* Added ``Isaac-Tracking-G1-MDP2``, motion tracking with the 29-DoF Unitree G1 on clips of the LAFAN1 retargeting
  dataset, which are downloaded and not part of the repository.
