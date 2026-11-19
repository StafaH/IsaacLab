* Added the ``body_com_vel_w`` field of articulations and rigid objects, and the ``force_matrix_w_history`` field of
  contact sensors with filter objects, to the MDP2 scene data, so that MDP2 terms read body velocities and the
  per-physics-step history of filtered contacts, such as self-collisions.
* Added :attr:`~isaaclab_experimental.mdp2.Event.resample_interval_on_reset`, which keeps the countdown of an MDP2
  interval event across resets, as MDP1's flag of that name does, for intervals longer than the episodes.
