* Added opt-in ``MJWarpSolverCfg.defer_body_state`` to publish body state only after
  the final solver substep when no intermediate consumer needs it. Requires a Newton
  version supporting ``SolverMuJoCo.step(update_body_state=False)``; external collision,
  native actuators, force callbacks, and body-state sensors retain full updates.
