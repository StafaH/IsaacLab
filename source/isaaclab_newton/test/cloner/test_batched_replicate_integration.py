# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the batched replication path in ``newton_replicate``.

Verifies that :func:`_replicate_into_builder` (which selects the batched
:class:`BatchedModelBuilder` fast path) produces the same per-world site-index map and the
same finalized :class:`~newton.Model` as the sequential fallback, for both homogeneous and
heterogeneous source mappings.
"""

import unittest

import numpy as np
import torch
import warp as wp
from isaaclab_newton.cloner.newton_replicate import _replicate_into_builder, _replicate_sequential
from isaaclab_newton.sim.batched_model_builder import BatchedModelBuilder
from newton import ModelBuilder
from newton.solvers import SolverMuJoCo


def _make_proto(seed: int) -> ModelBuilder:
    b = ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    root = b.add_body(mass=1.0 + seed)
    b.add_shape_box(body=root, hx=0.1, hy=0.1, hz=0.1)
    j = b.add_joint_free(child=root)
    b.add_articulation([j])
    return b


def _inject_site(proto: ModelBuilder, label: str) -> dict:
    """Add a site to a prototype's first body and return its proto_sites entry."""
    idx = proto.add_site(body=0, xform=wp.transform_identity(), label=label)
    return {label: [idx]}


def _poses(n: int):
    positions = torch.zeros((n, 3), dtype=torch.float32)
    positions[:, 0] = torch.arange(n, dtype=torch.float32)
    quaternions = torch.zeros((n, 4), dtype=torch.float32)
    quaternions[:, 3] = 1.0
    return positions, quaternions


class TestBatchedReplicateIntegration(unittest.TestCase):
    def _run(self, sources, protos, proto_sites, mapping):
        n = mapping.size(1)
        env_ids = torch.arange(n)
        positions, quaternions = _poses(n)
        env0_pos = positions[0]

        seq_builder = ModelBuilder(up_axis="Z")
        seq_map = _replicate_sequential(
            seq_builder, sources, protos, env_ids, mapping, positions, quaternions, env0_pos, proto_sites
        )

        bat_builder = BatchedModelBuilder(up_axis="Z")
        bat_map = _replicate_into_builder(
            bat_builder, sources, protos, env_ids, mapping, positions, quaternions, env0_pos, proto_sites, "Z"
        )
        return seq_builder, seq_map, bat_builder, bat_map

    def _assert_equal(self, seq_builder, seq_map, bat_builder, bat_map):
        # the batched path must actually have been taken (no worlds added sequentially)
        self.assertIsInstance(bat_builder, BatchedModelBuilder)
        self.assertEqual(seq_map, bat_map)
        ms = seq_builder.finalize(device="cpu")
        mb = bat_builder.finalize(device="cpu")
        self.assertEqual(ms.world_count, mb.world_count)
        self.assertEqual(ms.shape_count, mb.shape_count)
        np.testing.assert_allclose(ms.body_q.numpy(), mb.body_q.numpy(), atol=1e-5)
        self.assertEqual(ms.shape_world.numpy().tolist(), mb.shape_world.numpy().tolist())

    def test_homogeneous_single_source(self):
        a = _make_proto(0)
        sources = ["/A"]
        protos = {"/A": a}
        proto_sites = {id(a): _inject_site(a, "ftA")}
        mapping = torch.ones((1, 5), dtype=torch.bool)
        self._assert_equal(*self._run(sources, protos, proto_sites, mapping))

    def test_heterogeneous_two_sources(self):
        a = _make_proto(0)
        b = _make_proto(3)
        sources = ["/A", "/B"]
        protos = {"/A": a, "/B": b}
        proto_sites = {id(a): _inject_site(a, "ftA"), id(b): _inject_site(b, "ftB")}
        # columns: A, B, A, B, B  (interleaved single-source signatures)
        mapping = torch.tensor(
            [
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 1],
            ],
            dtype=torch.bool,
        )
        self._assert_equal(*self._run(sources, protos, proto_sites, mapping))


if __name__ == "__main__":
    unittest.main()
