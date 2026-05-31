Added
^^^^^

* Added :class:`~isaaclab_newton.sim.BatchedModelBuilder`, a
  :class:`~newton.ModelBuilder` subclass that replicates a prototype into many worlds in a
  single vectorized pass via :meth:`~isaaclab_newton.sim.BatchedModelBuilder.replicate_grouped`,
  replacing the per-world :meth:`~newton.ModelBuilder.add_builder` loop for large environment
  counts. It keeps the per-world transform arrays as NumPy and overrides
  :meth:`~newton.ModelBuilder.find_shape_contact_pairs` to tile the prototype's contact pairs
  instead of recomputing them per world, speeding up finalization. The cloner and
  :class:`~isaaclab_newton.physics.NewtonManager` use it automatically for rigid scenes and
  fall back to the sequential loop for particle/deformable content.
