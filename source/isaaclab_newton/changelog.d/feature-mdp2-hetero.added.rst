* Added :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.sleep_inactive_assets`, with which every environment
  holds the assets of every clone combination and keeps those outside its own combination always asleep, so that
  :attr:`~isaaclab.cloner.CloneCfg.clone_combinations` can give the environments of MuJoCo Warp different
  articulations. It needs a Newton and a MuJoCo Warp with per-world sleep policies.
