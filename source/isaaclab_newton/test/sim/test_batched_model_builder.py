# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Equivalence tests for :class:`BatchedModelBuilder`.

The contract under test: replicating a prototype with
:meth:`BatchedModelBuilder.replicate_grouped` produces a builder (and a finalized
:class:`~newton.Model`) identical to the one produced by the stock per-world
:meth:`~newton.ModelBuilder.add_builder` loop. We assert equivalence both on the builder
arrays (before finalize) and on the finalized model (after finalize), across:

  * homogeneous replication (one prototype for every world),
  * a global (world ``-1``) ground plane added before replication,
  * heterogeneous replication with interleaved prototypes.
"""

import math
import unittest

import numpy as np
import warp as wp
import newton
from isaaclab_newton.sim.batched_model_builder import BatchedModelBuilder
from newton import ModelBuilder
from newton.solvers import SolverMuJoCo


def _make_proto(seed: int = 0, *, with_mujoco: bool = True) -> ModelBuilder:
    """Build a small rigid prototype: a free-floating root + a revolute child, two shapes."""
    b = ModelBuilder()
    if with_mujoco:
        SolverMuJoCo.register_custom_attributes(b)
    rng = np.random.default_rng(seed)
    root = b.add_body(mass=1.5 + seed, com=wp.vec3(0.0, 0.0, 0.1))
    b.add_shape_box(body=root, hx=0.1, hy=0.2, hz=0.3)
    child = b.add_body(mass=0.5)
    b.add_shape_sphere(body=child, radius=0.05)
    jf = b.add_joint_free(child=root)
    jr = b.add_joint_revolute(parent=root, child=child, axis=wp.vec3(0.0, 0.0, 1.0))
    b.add_articulation([jf, jr])
    # nudge the free-joint root pose so transforms are exercised non-trivially
    qi = b.joint_q_start[jf]
    b.joint_q[qi : qi + 3] = [0.3 * seed, -0.2, 0.5]
    return b


def _inject_custom_attr_values(proto: ModelBuilder) -> None:
    """Populate representative mujoco custom-attribute values that exercise index/value offsets.

    Covers a BODY-enum-frequency attr (keyed by body index), string-frequency attrs whose
    values reference ``joint``/``world`` entities, and a string label column — i.e. the
    per-world index/value offset paths that empty schemas never trigger.
    """
    ca = proto.custom_attributes
    # BODY-enum frequency: dict keyed by body index -> exercises per-world index_offset.
    ca["mujoco:gravcomp"].values = {0: 0.5, 1: 0.25}
    # mujoco:tendon string-frequency (one tendon): world reference + label.
    ca["mujoco:tendon_world"].values = [0]
    ca["mujoco:tendon_label"].values = ["tendon0"]
    proto._custom_frequency_counts["mujoco:tendon"] = 1
    # mujoco:tendon_joint string-frequency referencing joint indices -> exercises value_offset.
    ca["mujoco:tendon_joint"].values = [0, 1]
    ca["mujoco:tendon_coef"].values = [0.7, 0.3]
    proto._custom_frequency_counts["mujoco:tendon_joint"] = 2


def _xforms(n: int) -> np.ndarray:
    """``n`` distinct non-identity transforms (translation + rotation) as an ``[n, 7]`` array."""
    out = []
    for i in range(n):
        ang = 0.3 * (i + 1)
        axis = np.array([0.0, 0.0, 1.0])
        q = wp.quat_from_axis_angle(wp.vec3(*axis), ang)
        out.append([i * 1.0, -i * 0.5, 0.25 * i, q[0], q[1], q[2], q[3]])
    return np.array(out, dtype=np.float32)


def _sequential(protos, signatures, xforms, *, global_builder_fn=None) -> ModelBuilder:
    b = ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    if global_builder_fn is not None:
        global_builder_fn(b)
    for col, sig in enumerate(signatures):
        b.begin_world()
        b.add_builder(protos[sig], xform=wp.transform(*xforms[col].tolist()))
        b.end_world()
    return b


def _batched(protos, signatures, xforms, *, global_builder_fn=None) -> BatchedModelBuilder:
    b = BatchedModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    if global_builder_fn is not None:
        global_builder_fn(b)
    b.replicate_grouped(protos, signatures, xforms)
    return b


# Builder array names to compare directly (before finalize).
_INT_LIST_ATTRS = (
    "shape_body",
    "joint_parent",
    "joint_child",
    "joint_q_start",
    "joint_qd_start",
    "joint_cts_start",
    "articulation_start",
    "joint_articulation",
    "body_world",
    "shape_world",
    "joint_world",
    "articulation_world",
    "equality_constraint_world",
    "constraint_mimic_world",
    "shape_type",
)
_FLOAT_LIST_ATTRS = (
    "body_q",
    "shape_transform",
    "joint_X_p",
    "joint_q",
    "body_mass",
    "body_com",
    "body_inertia",
    "shape_scale",
)
_STR_LIST_ATTRS = ("body_label", "joint_label", "shape_label", "articulation_label")


class TestBatchedModelBuilder(unittest.TestCase):
    def _assert_builders_equal(self, a: ModelBuilder, b: ModelBuilder):
        self.assertEqual(a.world_count, b.world_count)
        self.assertEqual(a.joint_dof_count, b.joint_dof_count)
        self.assertEqual(a.joint_coord_count, b.joint_coord_count)
        self.assertEqual(a.joint_constraint_count, b.joint_constraint_count)
        for attr in _INT_LIST_ATTRS:
            self.assertEqual(
                np.asarray(getattr(a, attr)).tolist(),
                np.asarray(getattr(b, attr)).tolist(),
                msg=f"int array mismatch: {attr}",
            )
        for attr in _STR_LIST_ATTRS:
            self.assertEqual(list(getattr(a, attr)), list(getattr(b, attr)), msg=f"label mismatch: {attr}")
        for attr in _FLOAT_LIST_ATTRS:
            va = np.array([np.asarray(x, dtype=np.float64).ravel() for x in getattr(a, attr)]) if len(getattr(a, attr)) else np.array([])
            vb = np.array([np.asarray(x, dtype=np.float64).ravel() for x in getattr(b, attr)]) if len(getattr(b, attr)) else np.array([])
            self.assertEqual(va.shape, vb.shape, msg=f"shape mismatch: {attr}")
            if va.size:
                np.testing.assert_allclose(va, vb, atol=1e-5, err_msg=f"float array mismatch: {attr}")
        # gravity, collision groups, filter pairs
        self.assertEqual([tuple(g) for g in a.world_gravity], [tuple(g) for g in b.world_gravity])
        self.assertEqual(list(a.shape_collision_group), list(b.shape_collision_group))
        self.assertEqual(
            sorted(map(tuple, a.shape_collision_filter_pairs)),
            sorted(map(tuple, b.shape_collision_filter_pairs)),
        )
        # body_shapes dict
        self.assertEqual({k: list(v) for k, v in a.body_shapes.items()}, {k: list(v) for k, v in b.body_shapes.items()})
        # custom attributes
        self.assertEqual(set(a.custom_attributes), set(b.custom_attributes))
        self.assertEqual(dict(a._custom_frequency_counts), dict(b._custom_frequency_counts))
        for key in a.custom_attributes:
            self.assertEqual(
                a.custom_attributes[key].values,
                b.custom_attributes[key].values,
                msg=f"custom attr values mismatch: {key}",
            )
        # actuator entries
        self.assertEqual(set(a.actuator_entries), set(b.actuator_entries))
        for key in a.actuator_entries:
            self.assertEqual(list(a.actuator_entries[key].indices), list(b.actuator_entries[key].indices))
            self.assertEqual(list(a.actuator_entries[key].pos_indices), list(b.actuator_entries[key].pos_indices))

    def _assert_models_equal(self, ma: newton.Model, mb: newton.Model):
        self.assertEqual(ma.world_count, mb.world_count)
        self.assertEqual(ma.body_count, mb.body_count)
        self.assertEqual(ma.shape_count, mb.shape_count)
        self.assertEqual(ma.joint_count, mb.joint_count)
        self.assertEqual(ma.articulation_count, mb.articulation_count)
        int_arrays = (
            "shape_body",
            "shape_world",
            "body_world",
            "joint_world",
            "joint_type",
            "joint_parent",
            "joint_child",
            "joint_q_start",
            "joint_qd_start",
            "articulation_start",
        )
        for attr in int_arrays:
            self.assertEqual(
                getattr(ma, attr).numpy().tolist(),
                getattr(mb, attr).numpy().tolist(),
                msg=f"model int array mismatch: {attr}",
            )
        float_arrays = ("body_q", "body_mass", "body_com", "shape_transform", "joint_q", "gravity")
        for attr in float_arrays:
            np.testing.assert_allclose(
                getattr(ma, attr).numpy(), getattr(mb, attr).numpy(), atol=1e-5, err_msg=f"model float mismatch: {attr}"
            )
        # contact pairs (order-independent)
        pa = sorted(map(tuple, ma.shape_contact_pairs.numpy().tolist()))
        pb = sorted(map(tuple, mb.shape_contact_pairs.numpy().tolist()))
        self.assertEqual(pa, pb)

    def test_homogeneous_matches_sequential(self):
        proto = _make_proto(0)
        n = 4
        xf = _xforms(n)
        sigs = [0] * n
        ref = _sequential([proto], sigs, xf)
        bat = _batched([proto], sigs, xf)
        self._assert_builders_equal(ref, bat)
        self._assert_models_equal(ref.finalize(device="cpu"), bat.finalize(device="cpu"))

    def test_global_ground_plane_front(self):
        proto = _make_proto(1)
        n = 3
        xf = _xforms(n)
        sigs = [0] * n

        def add_ground(b):
            b.add_ground_plane()

        ref = _sequential([proto], sigs, xf, global_builder_fn=add_ground)
        bat = _batched([proto], sigs, xf, global_builder_fn=add_ground)
        self._assert_builders_equal(ref, bat)
        self._assert_models_equal(ref.finalize(device="cpu"), bat.finalize(device="cpu"))

    def test_custom_attr_values_match(self):
        proto = _make_proto(0)
        _inject_custom_attr_values(proto)
        n = 4
        xf = _xforms(n)
        sigs = [0] * n
        ref = _sequential([proto], sigs, xf)
        bat = _batched([proto], sigs, xf)
        # Compare the offset-sensitive custom attributes directly.
        for key in ("mujoco:gravcomp", "mujoco:tendon_joint", "mujoco:tendon_world", "mujoco:tendon_label"):
            self.assertEqual(
                ref.custom_attributes[key].values,
                bat.custom_attributes[key].values,
                msg=f"custom attr values mismatch: {key}",
            )
        self.assertEqual(dict(ref._custom_frequency_counts), dict(bat._custom_frequency_counts))

    def test_intra_world_contact_pairs_tiled(self):
        # Two unconnected free bodies in the same positive collision group -> one intra-world
        # contact pair per world, exercising the tiled proto-pair path (not just global pairs).
        proto = ModelBuilder()
        b0 = proto.add_body(mass=1.0)
        proto.add_shape_box(body=b0, hx=0.1, hy=0.1, hz=0.1)
        proto.add_joint_free(child=b0)
        b1 = proto.add_body(mass=1.0)
        proto.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        proto.add_joint_free(child=b1)
        proto.shape_collision_group = [1, 1]
        n = 3
        xf = _xforms(n)
        ref = _sequential([proto], [0] * n, xf)
        bat = _batched([proto], [0] * n, xf)
        mr, mb = ref.finalize(device="cpu"), bat.finalize(device="cpu")
        pr = sorted(map(tuple, mr.shape_contact_pairs.numpy().tolist()))
        pb = sorted(map(tuple, mb.shape_contact_pairs.numpy().tolist()))
        self.assertEqual(pr, pb)
        self.assertEqual(len(pb), n, "expected one intra-world pair per world")

    def test_heterogeneous_interleaved(self):
        protos = [_make_proto(0), _make_proto(2)]
        sigs = [0, 1, 0, 1, 1, 0]
        n = len(sigs)
        xf = _xforms(n)
        ref = _sequential(protos, sigs, xf)
        bat = _batched(protos, sigs, xf)
        self._assert_builders_equal(ref, bat)
        self._assert_models_equal(ref.finalize(device="cpu"), bat.finalize(device="cpu"))


if __name__ == "__main__":
    unittest.main()
