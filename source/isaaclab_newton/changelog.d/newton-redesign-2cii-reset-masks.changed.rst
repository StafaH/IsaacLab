* **Breaking:** Changed :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_fk` and
  :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_body_state` to accept environment masks instead of
  environment indices. Callers authoring native state must pass a boolean ``env_mask``.
* Fused solver-reset and forward-kinematics flagging into Newton state writers, separated asset cache
  invalidation from backend notification, and removed per-asset scratch masks and their transfer launches.
* Added direct masked rigid-object-collection state writes and limited FK invalidation to the selected objects.
* Avoided refreshing derived state while obtaining authoritative write destinations.
