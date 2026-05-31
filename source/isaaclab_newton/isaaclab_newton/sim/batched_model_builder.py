# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched construction of Newton models for reinforcement learning.

Reinforcement learning spawns ``N`` near-identical "worlds" (environments), often in the
thousands. Newton's :class:`~newton.ModelBuilder` builds these by replicating a per-world
*prototype* builder ``N`` times through :meth:`~newton.ModelBuilder.add_builder`, called once
per world. Each call performs hundreds of pure-Python list operations, so the cost grows
linearly with the world count and dominates scene-build time at large ``N``.

:class:`BatchedModelBuilder` replaces that per-world loop with vectorized NumPy tiling: a
prototype is replicated into many worlds in a single pass, computing per-world index offsets
and rigid transforms with array operations instead of Python iteration. The result is an
exact equivalent of the builder that the sequential :meth:`~newton.ModelBuilder.add_builder`
loop would produce, so Newton's stock :meth:`~newton.ModelBuilder.finalize` runs unchanged on
top of it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp
from newton import JointType, ModelBuilder

# ``ModelBuilder.add_builder`` copies these per-entity arrays verbatim (no index offset and no
# transform). Mirrors the ``more_builder_attrs`` list in Newton's ``add_builder`` plus the
# pure-value columns it handles separately (labels, collision groups, constraint payloads).
# Tiling these is a plain repeat of the prototype's values.
_VALUE_ATTRS: tuple[str, ...] = (
    # bodies
    "body_inertia",
    "body_mass",
    "body_inv_inertia",
    "body_inv_mass",
    "body_com",
    "body_lock_inertia",
    "body_flags",
    "body_qd",
    # joints (dynamics, all expressed in joint/child frames -> no world transform)
    "joint_type",
    "joint_enabled",
    "joint_collision_filter_parent",
    "joint_X_c",
    "joint_armature",
    "joint_axis",
    "joint_dof_dim",
    "joint_qd",
    "joint_cts",
    "joint_f",
    "joint_act",
    "joint_target_pos",
    "joint_target_vel",
    "joint_limit_lower",
    "joint_limit_upper",
    "joint_limit_ke",
    "joint_limit_kd",
    "joint_target_ke",
    "joint_target_kd",
    "joint_target_mode",
    "joint_effort_limit",
    "joint_velocity_limit",
    "joint_friction",
    # shapes (materials + parameters)
    "shape_flags",
    "shape_type",
    "shape_scale",
    "shape_source",
    "shape_color",
    "shape_is_solid",
    "shape_margin",
    "shape_material_ke",
    "shape_material_kd",
    "shape_material_kf",
    "shape_material_ka",
    "shape_material_mu",
    "shape_material_restitution",
    "shape_material_mu_torsional",
    "shape_material_mu_rolling",
    "shape_material_kh",
    "shape_collision_radius",
    "shape_gap",
    "shape_sdf_narrow_band_range",
    "shape_sdf_max_resolution",
    "shape_sdf_target_voxel_size",
    "shape_sdf_texture_format",
    "shape_collision_group",
    # labels
    "body_label",
    "joint_label",
    "shape_label",
    "articulation_label",
    # equality constraints (value columns)
    "equality_constraint_type",
    "equality_constraint_anchor",
    "equality_constraint_torquescale",
    "equality_constraint_relpose",
    "equality_constraint_polycoef",
    "equality_constraint_label",
    "equality_constraint_enabled",
    # mimic constraints (value columns)
    "constraint_mimic_coef0",
    "constraint_mimic_coef1",
    "constraint_mimic_enabled",
    "constraint_mimic_label",
)

# Attributes that signal the prototype contains particles / deformables. The batched path
# only supports rigid-body content; presence of any of these forces a sequential fallback.
_DEFORMABLE_ATTRS: tuple[str, ...] = (
    "particle_q",
    "spring_rest_length",
    "edge_rest_angle",
    "tri_poses",
    "tet_poses",
    "muscle_start",
)

# Entity types whose per-world index offset the batched path knows how to apply when a custom
# attribute's frequency or ``references`` targets them.
_KNOWN_ENTITY_REFERENCES: frozenset[str] = frozenset(
    {
        "body",
        "shape",
        "joint",
        "joint_dof",
        "joint_coord",
        "joint_constraint",
        "articulation",
        "equality_constraint",
        "constraint_mimic",
        "particle",
        "edge",
        "triangle",
        "tetrahedron",
        "spring",
    }
)


class BatchedReplicationError(RuntimeError):
    """Raised when a prototype cannot be replicated through the batched fast path.

    Callers should catch this and fall back to the sequential
    :meth:`~newton.ModelBuilder.add_builder` loop, which handles every case the batched path
    declines (particles/deformables, exotic custom attributes, etc.).
    """


class BatchedModelBuilder(ModelBuilder):
    """A :class:`~newton.ModelBuilder` that replicates prototypes into many worlds at once.

    Use :meth:`replicate_grouped` to populate the builder with ``N`` worlds from one or more
    prototype builders, then call :meth:`~newton.ModelBuilder.finalize` exactly as with a
    plain :class:`~newton.ModelBuilder`. Any global (world ``-1``) entities, such as a ground
    plane, must be added to this builder before :meth:`replicate_grouped` so they remain at
    the front of the entity arrays.
    """

    def replicate_grouped(
        self,
        protos: Sequence[ModelBuilder],
        signatures: Sequence[int],
        xforms: np.ndarray,
    ) -> None:
        """Replicate prototype builders into ``N`` worlds in column order.

        Worlds are appended in the order given by ``signatures`` so that the resulting world
        indices are contiguous and monotonic (world ``i`` is column ``i``), matching the
        invariant Newton's :meth:`~newton.ModelBuilder.finalize` validates. Contiguous runs of
        the same prototype are tiled in a single vectorized pass; the homogeneous case (one
        prototype for every world) is therefore a single pass.

        Args:
            protos: Prototype builders indexed by signature id. Each prototype holds one
                world's worth of rigid bodies, shapes and joints, already positioned at its
                env-local rest pose (the per-world ``xforms`` are composed on top).
            signatures: Length-``N`` sequence mapping each world (in column/world order) to
                the index of its prototype in ``protos``.
            xforms: Per-world rigid transforms, shape ``[N, 7]`` as ``[p_xyz, q_xyzw]``,
                applied to each world's root entities (matching ``add_builder``'s ``xform``).

        Raises:
            BatchedReplicationError: If any prototype contains particles/deformables or a
                custom attribute the batched path does not support.
        """
        signatures = np.asarray(signatures, dtype=np.int64)
        xforms = np.asarray(xforms, dtype=np.float32).reshape(-1, 7)
        if signatures.shape[0] != xforms.shape[0]:
            raise ValueError(f"signatures ({signatures.shape[0]}) and xforms ({xforms.shape[0]}) length mismatch")

        for proto in protos:
            self._check_proto_supported(proto)

        # Transform arrays are computed as NumPy and kept as NumPy on the builder (rather than
        # Python lists): finalize consumes them with a near-free ``wp.array`` instead of a costly
        # nested ``np.asarray`` over millions of floats. Convert the existing front (global,
        # world ``-1``) entries once so per-world blocks can be concatenated onto them.
        self.body_q = self._to_transform_array(self.body_q)
        self.shape_transform = self._to_transform_array(self.shape_transform)
        self.joint_X_p = self._to_transform_array(self.joint_X_p)
        self.joint_q = np.asarray(self.joint_q, dtype=np.float32)

        # Record state for the tiled :meth:`find_shape_contact_pairs` fast path. The contact
        # pairs of every world are the prototype's intra-world pairs offset per world, so they
        # can be tiled instead of recomputed by the O(N * shapes^2) base-class loop. Only the
        # homogeneous case (one prototype for every world) is tiled; otherwise we fall back.
        n = signatures.shape[0]
        self._cp_homogeneous = bool(len(protos) >= 1 and np.unique(signatures).size == 1)
        self._cp_proto = protos[int(signatures[0])] if (self._cp_homogeneous and n) else None
        self._cp_front_shapes = self.shape_count
        self._cp_num_worlds = n

        # Split into maximal contiguous runs of identical signature; each run is one
        # vectorized append. A single signature collapses to one run (the homogeneous case).
        if n == 0:
            return
        boundaries = np.flatnonzero(signatures[1:] != signatures[:-1]) + 1
        run_starts = np.concatenate(([0], boundaries))
        run_ends = np.concatenate((boundaries, [n]))
        for start, end in zip(run_starts.tolist(), run_ends.tolist()):
            self._append_worlds(protos[int(signatures[start])], xforms[start:end])

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def proto_has_deformables(proto: ModelBuilder) -> bool:
        """Return whether a prototype contains particles or deformable elements."""
        return any(getattr(proto, attr, None) for attr in _DEFORMABLE_ATTRS)

    @classmethod
    def can_replicate(cls, protos: Sequence[ModelBuilder]) -> bool:
        """Return whether all prototypes can be replicated through the batched fast path.

        Performs the same checks as :meth:`replicate_grouped` without mutating any builder, so
        callers can decide between the batched path and the sequential
        :meth:`~newton.ModelBuilder.add_builder` loop up front.
        """
        for proto in protos:
            if cls.proto_has_deformables(proto):
                return False
            for attr in proto.custom_attributes.values():
                if not cls._custom_attr_reference_supported(proto, attr):
                    return False
        return True

    @staticmethod
    def _custom_attr_reference_supported(proto: ModelBuilder, attr) -> bool:
        ref = attr.references
        if ref is None or ref == "world" or ref in _KNOWN_ENTITY_REFERENCES:
            return True
        return ref in proto._custom_frequency_counts

    def _check_proto_supported(self, proto: ModelBuilder) -> None:
        if self.proto_has_deformables(proto):
            raise BatchedReplicationError("batched replication does not support particles/deformables")

    def _append_worlds(self, proto: ModelBuilder, xforms: np.ndarray) -> None:
        """Append ``K = len(xforms)`` copies of ``proto`` as new worlds, fully vectorized over K."""
        k = xforms.shape[0]
        if k == 0:
            return

        # -- snapshot base counts BEFORE mutating any array (add_builder captures these too) --
        base_body = self.body_count
        base_shape = self.shape_count
        base_joint = self.joint_count
        base_articulation = self.articulation_count
        base_coord = self.joint_coord_count
        base_dof = self.joint_dof_count
        base_cts = self.joint_constraint_count
        base_world = self.world_count
        # Pre-extension entity counts for custom-attribute index offsets (keyed as add_builder
        # keys its ``entity_offsets`` table). Captured here because the arrays below mutate.
        entity_bases = {
            "body": base_body,
            "shape": base_shape,
            "joint": base_joint,
            "joint_dof": base_dof,
            "joint_coord": base_coord,
            "joint_constraint": base_cts,
            "articulation": base_articulation,
            "equality_constraint": len(self.equality_constraint_type),
            "constraint_mimic": len(self.constraint_mimic_joint0),
            "particle": self.particle_count,
            "edge": self.edge_count,
            "triangle": self.tri_count,
            "tetrahedron": self.tet_count,
            "spring": self.spring_count,
        }

        n_body = proto.body_count
        n_shape = proto.shape_count
        n_joint = proto.joint_count
        n_articulation = proto.articulation_count
        n_coord = proto.joint_coord_count
        n_dof = proto.joint_dof_count
        n_cts = proto.joint_constraint_count
        n_eq = len(proto.equality_constraint_type)
        n_mimic = len(proto.constraint_mimic_joint0)

        world_ids = np.arange(base_world, base_world + k, dtype=np.int32)

        # -- value arrays: plain tile (no offset, no transform), preserving element types -----
        for attr in _VALUE_ATTRS:
            src = getattr(proto, attr)
            if src:
                getattr(self, attr).extend(src * k)

        # -- index arrays: tile + per-world additive offset (preserve -1 sentinels) -----------
        self.shape_body.extend(self._tile_offset(proto.shape_body, base_body, n_body, k, sentinel=True))
        self.joint_parent.extend(self._tile_offset(proto.joint_parent, base_body, n_body, k, sentinel=True))
        self.joint_child.extend(self._tile_offset(proto.joint_child, base_body, n_body, k, sentinel=False))
        self.joint_q_start.extend(self._tile_offset(proto.joint_q_start, base_coord, n_coord, k, sentinel=False))
        self.joint_qd_start.extend(self._tile_offset(proto.joint_qd_start, base_dof, n_dof, k, sentinel=False))
        self.joint_cts_start.extend(self._tile_offset(proto.joint_cts_start, base_cts, n_cts, k, sentinel=False))
        self.articulation_start.extend(self._tile_offset(proto.articulation_start, base_joint, n_joint, k, sentinel=False))
        self.joint_articulation.extend(
            self._tile_offset(proto.joint_articulation, base_articulation, n_articulation, k, sentinel=True)
        )

        if n_eq:
            self.equality_constraint_body1.extend(self._tile_offset(proto.equality_constraint_body1, base_body, n_body, k, sentinel=True))
            self.equality_constraint_body2.extend(self._tile_offset(proto.equality_constraint_body2, base_body, n_body, k, sentinel=True))
            self.equality_constraint_joint1.extend(self._tile_offset(proto.equality_constraint_joint1, base_joint, n_joint, k, sentinel=True))
            self.equality_constraint_joint2.extend(self._tile_offset(proto.equality_constraint_joint2, base_joint, n_joint, k, sentinel=True))
        if n_mimic:
            self.constraint_mimic_joint0.extend(self._tile_offset(proto.constraint_mimic_joint0, base_joint, n_joint, k, sentinel=True))
            self.constraint_mimic_joint1.extend(self._tile_offset(proto.constraint_mimic_joint1, base_joint, n_joint, k, sentinel=True))

        # -- collision filter pairs: offset both shape indices per world ----------------------
        if proto.shape_collision_filter_pairs:
            pairs = np.asarray(proto.shape_collision_filter_pairs, dtype=np.int64)  # [P, 2]
            off = (base_shape + np.arange(k, dtype=np.int64) * n_shape).repeat(pairs.shape[0])[:, None]
            tiled = np.tile(pairs, (k, 1)) + off
            self.shape_collision_filter_pairs.extend(map(tuple, tiled.tolist()))

        # -- transform arrays: tile + per-world rigid transform on the relevant subset --------
        # Kept as NumPy (see replicate_grouped); concatenated per run so each is a single
        # contiguous array that finalize feeds to wp.array without a nested asarray conversion.
        self.body_q = np.concatenate([self.body_q, self._tile_transform_all(proto.body_q, xforms, n_body)])
        self.shape_transform = np.concatenate(
            [self.shape_transform, self._tile_transform_masked(proto.shape_transform, xforms, n_shape, np.asarray(proto.shape_body) < 0)]
        )
        if n_joint:
            joint_parent_arr = np.asarray(proto.joint_parent)
            joint_type_arr = np.asarray(proto.joint_type)
            root_non_free = (joint_parent_arr == -1) & (joint_type_arr != int(JointType.FREE))
            self.joint_X_p = np.concatenate(
                [self.joint_X_p, self._tile_transform_masked(proto.joint_X_p, xforms, n_joint, root_non_free)]
            )
            self.joint_q = np.concatenate([self.joint_q, self._tile_free_joint_q(proto, xforms, n_coord)])

        # -- world-indexed arrays -------------------------------------------------------------
        self.body_world.extend(world_ids.repeat(n_body).tolist())
        self.shape_world.extend(world_ids.repeat(n_shape).tolist())
        self.joint_world.extend(world_ids.repeat(n_joint).tolist())
        self.articulation_world.extend(world_ids.repeat(n_articulation).tolist())
        if n_eq:
            self.equality_constraint_world.extend(world_ids.repeat(n_eq).tolist())
        if n_mimic:
            self.constraint_mimic_world.extend(world_ids.repeat(n_mimic).tolist())

        # -- body_shapes dict (per-world rekey + shape offset) --------------------------------
        for w in range(k):
            body_off = base_body + w * n_body
            shape_off = base_shape + w * n_shape
            for b, shapes in proto.body_shapes.items():
                offset_shapes = [s + shape_off for s in shapes]
                if b == -1:
                    self.body_shapes[-1].extend(offset_shapes)
                else:
                    self.body_shapes[b + body_off] = offset_shapes

        # -- per-world gravity ----------------------------------------------------------------
        up = proto.up_vector
        gravity_vec = (up[0] * proto.gravity, up[1] * proto.gravity, up[2] * proto.gravity)
        self.world_gravity.extend([gravity_vec] * k)

        # -- requested attributes (set union, world-independent) ------------------------------
        self._requested_contact_attributes.update(proto._requested_contact_attributes)
        self._requested_state_attributes.update(proto._requested_state_attributes)

        # -- custom attributes + actuators (small data; per-world replay of add_builder logic)-
        self._append_custom_attributes(proto, k, base_world, entity_bases)
        self._append_actuator_entries(proto, k, base_dof, base_coord)

        # -- maintained scalar counts ---------------------------------------------------------
        self.joint_dof_count += k * n_dof
        self.joint_coord_count += k * n_coord
        self.joint_constraint_count += k * n_cts
        self.world_count += k

    # --------------------------------------------------------------- numeric tiling

    @staticmethod
    def _tile_offset(values: Sequence[int], base: int, stride: int, k: int, *, sentinel: bool) -> list[int]:
        """Tile an integer index array ``k`` times, adding ``base + w*stride`` to world ``w``.

        When ``sentinel`` is True, ``-1`` entries are preserved (they mark "no parent"/invalid).
        """
        if not values:
            return []
        arr = np.asarray(values, dtype=np.int64)
        offsets = (base + np.arange(k, dtype=np.int64) * stride).repeat(arr.shape[0])
        tiled = np.tile(arr, k)
        if sentinel:
            tiled = np.where(tiled >= 0, tiled + offsets, tiled)
        else:
            tiled = tiled + offsets
        return tiled.tolist()

    @staticmethod
    def _to_transform_array(transforms: Sequence) -> np.ndarray:
        """Convert a list of ``wp.transform`` (or 7-tuples) into an ``[M, 7]`` float32 array."""
        return np.array([tuple(t) for t in transforms], dtype=np.float32).reshape(-1, 7)

    @staticmethod
    def _transform_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Batched ``wp.transform_multiply``: ``a ∘ b`` for ``[M, 7]`` arrays (xyzw quats)."""
        ap, aq = a[:, :3], a[:, 3:]
        bp, bq = b[:, :3], b[:, 3:]
        # rotate b's translation by a's quaternion
        qv = aq[:, :3]
        qw = aq[:, 3:4]
        t = 2.0 * np.cross(qv, bp)
        rotated = bp + qw * t + np.cross(qv, t)
        # quaternion product a.q * b.q (xyzw)
        ax, ay, az, aw = aq[:, 0], aq[:, 1], aq[:, 2], aq[:, 3]
        bx, by, bz, bw = bq[:, 0], bq[:, 1], bq[:, 2], bq[:, 3]
        out_q = np.stack(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            axis=-1,
        )
        return np.concatenate([ap + rotated, out_q], axis=-1)

    def _tile_transform_all(self, transforms: Sequence, xforms: np.ndarray, n: int) -> np.ndarray:
        """Tile ``n`` transforms across ``K`` worlds, composing each world's xform onto all."""
        if n == 0:
            return np.empty((0, 7), dtype=np.float32)
        proto = self._to_transform_array(transforms)
        k = xforms.shape[0]
        per_entity_xform = np.repeat(xforms, n, axis=0)
        tiled = np.tile(proto, (k, 1))
        return self._transform_mul(per_entity_xform, tiled)

    def _tile_transform_masked(self, transforms: Sequence, xforms: np.ndarray, n: int, mask: np.ndarray) -> np.ndarray:
        """Like :meth:`_tile_transform_all`, but only transform entities where ``mask`` is True."""
        if n == 0:
            return np.empty((0, 7), dtype=np.float32)
        proto = self._to_transform_array(transforms)
        k = xforms.shape[0]
        tiled = np.tile(proto, (k, 1))
        mask = np.asarray(mask, dtype=bool)
        full_mask = np.tile(mask, k)
        if full_mask.any():
            per_entity_xform = np.repeat(xforms, n, axis=0)[full_mask]
            tiled[full_mask] = self._transform_mul(per_entity_xform, tiled[full_mask])
        return tiled

    def _tile_free_joint_q(self, proto: ModelBuilder, xforms: np.ndarray, n_coord: int) -> np.ndarray:
        """Tile the flat ``joint_q`` coordinate array, transforming each FREE joint's 7-slice."""
        if n_coord == 0:
            return np.empty((0,), dtype=np.float32)
        proto_q = np.asarray(proto.joint_q, dtype=np.float32)
        k = xforms.shape[0]
        tiled = np.tile(proto_q, k)
        joint_type = np.asarray(proto.joint_type)
        free_joints = np.flatnonzero(joint_type == int(JointType.FREE))
        if free_joints.size:
            q_starts = np.asarray(proto.joint_q_start, dtype=np.int64)[free_joints]
            # absolute slice start for every (world, free-joint) pair, shape [K*F]
            world_base = (np.arange(k, dtype=np.int64) * n_coord)[:, None] + q_starts[None, :]
            slice_starts = world_base.reshape(-1)
            slice_idx = slice_starts[:, None] + np.arange(7, dtype=np.int64)[None, :]  # [K*F, 7]
            world_xform = np.repeat(xforms, free_joints.size, axis=0)  # [K*F, 7]
            transformed = self._transform_mul(world_xform, tiled[slice_idx])
            tiled[slice_idx] = transformed
        return tiled

    # --------------------------------------------------------------- custom attrs

    def _append_custom_attributes(self, proto: ModelBuilder, k: int, base_world: int, entity_bases: dict[str, int]) -> None:
        """Replicate the prototype's custom attributes across ``k`` worlds.

        Mirrors the per-attribute offset/transform semantics of
        ``ModelBuilder.add_builder``, applied once per world. Custom-attribute payloads are
        small relative to the bulk arrays, so a per-world Python loop here is inexpensive.

        Args:
            proto: Prototype whose custom attributes are replicated.
            k: Number of worlds to append.
            base_world: World id of the first appended world.
            entity_bases: Per-entity-type counts captured *before* this run's arrays were
                extended, used as the additive base for index offsets.
        """
        from dataclasses import replace  # noqa: PLC0415

        from newton import Model  # noqa: PLC0415

        # Carry over frequency registrations (idempotent).
        for freq_key, freq_obj in proto.custom_frequencies.items():
            self.custom_frequencies.setdefault(freq_key, freq_obj)

        entity_counts = {
            "body": proto.body_count,
            "shape": proto.shape_count,
            "joint": proto.joint_count,
            "joint_dof": proto.joint_dof_count,
            "joint_coord": proto.joint_coord_count,
            "joint_constraint": proto.joint_constraint_count,
            "articulation": proto.articulation_count,
            "equality_constraint": len(proto.equality_constraint_type),
            "constraint_mimic": len(proto.constraint_mimic_joint0),
            "particle": proto.particle_count,
            "edge": proto.edge_count,
            "triangle": proto.tri_count,
            "tetrahedron": proto.tet_count,
            "spring": proto.spring_count,
        }

        def base_for(key: str | None) -> int:
            if key is None:
                return 0
            if key in entity_bases:
                return entity_bases[key]
            return self._custom_frequency_counts.get(key, 0)

        def count_for(key: str | None) -> int:
            if key is None:
                return 0
            if key in entity_counts:
                return entity_counts[key]
            return proto._custom_frequency_counts.get(key, 0)

        for full_key, attr in proto.custom_attributes.items():
            freq_key = attr.frequency
            is_str_freq = isinstance(freq_key, str)
            self._validate_custom_attr_supported(proto, full_key, attr)

            use_world = attr.references == "world"
            if is_str_freq:
                freq_name = freq_key
            elif freq_key == Model.AttributeFrequency.ONCE:
                freq_name = None
            elif freq_key == Model.AttributeFrequency.WORLD:
                freq_name = "world"
            else:
                freq_name = freq_key.name.lower()

            ref_base = base_for(attr.references)
            ref_count = count_for(attr.references)

            # Ensure the attribute exists on this builder (declare empty on first sight).
            merged = self.custom_attributes.get(full_key)
            if merged is None:
                merged = replace(attr, values=([] if is_str_freq else {}))
                self.custom_attributes[full_key] = merged
            if merged.values is None:
                merged.values = [] if is_str_freq else {}

            if not attr.values:
                continue

            for w in range(k):
                value_offset = 0 if use_world else (ref_base + w * ref_count)
                cur_world = base_world + w

                def transform_value(v, _off=value_offset, _world=use_world, _cw=cur_world):
                    if _world:
                        return _cw
                    if _off == 0:
                        return v
                    if isinstance(v, int):
                        return v + _off if v >= 0 else v
                    if isinstance(v, (list, tuple)):
                        return type(v)([x + _off if isinstance(x, int) and x >= 0 else x for x in v])
                    try:
                        return v + _off
                    except TypeError:
                        return v

                if is_str_freq:
                    merged.values.extend(transform_value(v) for v in attr.values)
                elif freq_name == "world":
                    # WORLD frequency: indices keyed by world id.
                    merged.values.update({cur_world + idx: transform_value(v) for idx, v in attr.values.items()})
                elif freq_name is None:
                    # ONCE frequency: a single shared row (offset 0). ``add_builder`` rewrites it
                    # for every world, so the last prototype wins; replay once per run to match.
                    if w == 0:
                        merged.values.update({idx: transform_value(v) for idx, v in attr.values.items()})
                else:
                    index_offset = entity_bases[freq_name] + w * entity_counts[freq_name]
                    merged.values.update({index_offset + idx: transform_value(v) for idx, v in attr.values.items()})

        # Update frequency counts once per frequency (base + k * proto_count).
        for freq_key, proto_count in proto._custom_frequency_counts.items():
            self._custom_frequency_counts[freq_key] = self._custom_frequency_counts.get(freq_key, 0) + k * proto_count

    def _validate_custom_attr_supported(self, proto: ModelBuilder, full_key, attr) -> None:
        """Reject custom attributes whose ``references`` target the batched path cannot offset."""
        if not self._custom_attr_reference_supported(proto, attr):
            raise BatchedReplicationError(f"custom attribute '{full_key}' references unsupported target '{attr.references}'")

    def _append_actuator_entries(self, proto: ModelBuilder, k: int, base_dof: int, base_coord: int) -> None:
        """Replicate the prototype's actuator entries across ``k`` worlds with DOF/coord offsets."""
        n_dof = proto.joint_dof_count
        n_coord = proto.joint_coord_count
        for entry_key, sub_entry in proto.actuator_entries.items():
            entry = self.actuator_entries.setdefault(
                entry_key,
                ModelBuilder.ActuatorEntry(
                    controller_class=sub_entry.controller_class,
                    clamping_classes=sub_entry.clamping_classes,
                    clamping_shared_kwargs=sub_entry.clamping_shared_kwargs,
                    controller_shared_kwargs=sub_entry.controller_shared_kwargs,
                    indices=[],
                    pos_indices=[],
                    controller_args=[],
                    delay_args=[],
                    clamping_args=[],
                ),
            )
            for w in range(k):
                dof_off = base_dof + w * n_dof
                coord_off = base_coord + w * n_coord
                entry.indices.extend(idx + dof_off for idx in sub_entry.indices)
                entry.pos_indices.extend(idx + coord_off for idx in sub_entry.pos_indices)
                entry.controller_args.extend(sub_entry.controller_args)
                entry.delay_args.extend(sub_entry.delay_args)
                entry.clamping_args.extend(sub_entry.clamping_args)

    # --------------------------------------------------------------- contact pairs

    @staticmethod
    def _test_group_pair(group_a: int, group_b: int) -> bool:
        """Whether two collision groups interact (mirrors ``ModelBuilder._test_group_pair``)."""
        if group_a == 0 or group_b == 0:
            return False
        if group_a > 0:
            return group_a == group_b or group_b < 0
        return group_a != group_b

    def find_shape_contact_pairs(self, model) -> None:
        """Compute candidate shape contact pairs, tiling the prototype's pairs across worlds.

        Worlds never collide with each other (the base-class filter rejects cross-world pairs),
        so every world's intra-world contact pairs are the prototype's pairs offset by that
        world's shape stride. Computing the prototype's pairs once and tiling them avoids the
        O(N * shapes^2) double loop in :meth:`~newton.ModelBuilder.find_shape_contact_pairs`.
        Falls back to the base-class implementation for heterogeneous worlds.
        """
        from newton import ShapeFlags  # noqa: PLC0415

        if not getattr(self, "_cp_homogeneous", False) or self._cp_proto is None:
            return super().find_shape_contact_pairs(model)

        proto = self._cp_proto
        front = self._cp_front_shapes
        num_worlds = self._cp_num_worlds
        n_shape = proto.shape_count
        collide = int(ShapeFlags.COLLIDE_SHAPES)
        filters: set[tuple[int, int]] = model.shape_collision_filter_pairs

        # Colliding shapes (prototype-local and global/world -1) with their collision groups.
        proto_groups = proto.shape_collision_group
        proto_colliding = [i for i in range(n_shape) if proto.shape_flags[i] & collide]
        global_colliding = [i for i in range(front) if self.shape_flags[i] & collide]
        proto_filters = {(min(a, b), max(a, b)) for a, b in proto.shape_collision_filter_pairs}

        # Prototype intra-world pairs (prototype-local, i < j), pre-filtered by prototype filters.
        proto_pairs: list[tuple[int, int]] = []
        for ai in range(len(proto_colliding)):
            sa = proto_colliding[ai]
            ga = proto_groups[sa]
            for bi in range(ai + 1, len(proto_colliding)):
                sb = proto_colliding[bi]
                if self._test_group_pair(ga, proto_groups[sb]):
                    pair = (sa, sb) if sa < sb else (sb, sa)
                    if pair not in proto_filters:
                        proto_pairs.append(pair)

        blocks: list[np.ndarray] = []
        # Tile the prototype intra-world pairs across all worlds.
        if proto_pairs and num_worlds:
            pp = np.asarray(proto_pairs, dtype=np.int64)  # [P, 2]
            offsets = (front + np.arange(num_worlds, dtype=np.int64) * n_shape)[:, None, None]
            blocks.append((pp[None, :, :] + offsets).reshape(-1, 2))

        # Global (world -1) shapes collide with every world's shapes (and each other). These are
        # few (e.g. a ground plane), so filtering against the global-involving filters is cheap.
        if global_colliding:
            global_filters = {(a, b) for a, b in filters if a < front or b < front}
            gw = [
                (g, s)
                for g in global_colliding
                for s in proto_colliding
                if self._test_group_pair(self.shape_collision_group[g], proto_groups[s])
            ]
            if gw and num_worlds:
                gw_arr = np.asarray(gw, dtype=np.int64)  # [Q, 2]: (global, proto-local)
                world_off = (front + np.arange(num_worlds, dtype=np.int64) * n_shape)[:, None]
                g_col = np.tile(gw_arr[:, 0], num_worlds)
                s_col = (gw_arr[:, 1][None, :] + world_off).reshape(-1)
                gw_pairs = np.stack([g_col, s_col], axis=1)  # g < s always
                if global_filters:
                    gw_pairs = np.array([row for row in gw_pairs if tuple(row.tolist()) not in global_filters], dtype=np.int64).reshape(-1, 2)
                blocks.append(gw_pairs)
            # global-vs-global pairs
            gg = []
            for ai in range(len(global_colliding)):
                ga_idx = global_colliding[ai]
                ga = self.shape_collision_group[ga_idx]
                for bi in range(ai + 1, len(global_colliding)):
                    gb_idx = global_colliding[bi]
                    if self._test_group_pair(ga, self.shape_collision_group[gb_idx]):
                        pair = (min(ga_idx, gb_idx), max(ga_idx, gb_idx))
                        if pair not in global_filters:
                            gg.append(pair)
            if gg:
                blocks.append(np.asarray(gg, dtype=np.int64))

        all_pairs = np.concatenate(blocks) if blocks else np.empty((0, 2), dtype=np.int64)
        model.shape_contact_pairs = wp.array(all_pairs, dtype=wp.vec2i, device=model.device)
        model.shape_contact_pair_count = all_pairs.shape[0]
