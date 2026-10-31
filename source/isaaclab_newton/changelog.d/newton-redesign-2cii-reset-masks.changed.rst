* **Breaking:** Changed the positional selection arguments of
  :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_fk` and
  :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_body_state` (and their backend counterparts).
  Native-state callers should pass selections explicitly with ``env_ids=`` or ``env_mask=``, and
  articulation mappings with ``articulation_ids=``.
* Centralized solver-reset and forward-kinematics flagging in the Newton backend, accepting indices or masks
  directly without per-asset scratch masks. Asset data classes now invalidate only their derived caches.
* Reused collection state-writing kernels for both indexed and masked selections without converting masks
  to indices, and limited FK invalidation to the selected objects.
* Avoided refreshing derived state while obtaining authoritative write destinations.
