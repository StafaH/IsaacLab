# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import torch
import warp as wp
from newton import ModelBuilder, solvers
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd

from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sim.batched_model_builder import BatchedModelBuilder


def _build_newton_builder_from_mapping(
    stage: Usd.Stage,
    sources: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    up_axis: str = "Z",
    simplify_meshes: bool = True,
) -> tuple[ModelBuilder, object, dict]:
    """Build a Newton model builder from clone mapping inputs.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of the populated Newton model builder, stage metadata returned
        by ``add_usd``, and a site index map for
        :attr:`NewtonManager._cl_site_index_map`.
    """
    if positions is None:
        positions = torch.zeros((mapping.size(1), 3), device=mapping.device, dtype=torch.float32)
    if quaternions is None:
        quaternions = torch.zeros((mapping.size(1), 4), device=mapping.device, dtype=torch.float32)
        quaternions[:, 3] = 1.0

    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

    builder = NewtonManager.create_builder(up_axis=up_axis, batched=True)
    stage_info = builder.add_usd(
        stage,
        ignore_paths=["/World/envs", *sources],
        schema_resolvers=schema_resolvers,
    )

    # The prototype is built from env_0 in absolute world coordinates.
    # add_builder xforms are deltas from env_0 so positions don't get double-counted.
    env0_pos = positions[0]

    # Deformable prim paths are handled by per_world_builder_hooks, not add_usd.
    # Resolve the regex prim_path patterns to concrete env_0 paths so add_usd
    # can skip them via ignore_paths.
    _deformable_ignore_paths: list[str] = []
    if hasattr(NewtonManager, "_deformable_registry"):
        for entry in NewtonManager._deformable_registry:
            pat = re.compile(entry.prim_path.replace(".*", "[^/]*") + "$")
            for src_path in sources:
                # Check if any prim under this source matches the deformable pattern
                prim = stage.GetPrimAtPath(src_path)
                if prim.IsValid():
                    for child in Usd.PrimRange(prim):
                        child_path = str(child.GetPath())
                        if pat.match(child_path):
                            _deformable_ignore_paths.append(child_path)

    protos: dict[str, ModelBuilder] = {}
    for src_path in sources:
        p = NewtonManager.create_builder(up_axis=up_axis)
        solvers.SolverMuJoCo.register_custom_attributes(p)
        p.add_usd(
            stage,
            root_path=src_path,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            schema_resolvers=schema_resolvers,
            ignore_paths=_deformable_ignore_paths if _deformable_ignore_paths else None,
        )
        if simplify_meshes:
            p.approximate_meshes("convex_hull", keep_visual_shapes=True)
        protos[src_path] = p

    # Inject registered sites into prototypes (and global sites into main builder)
<<<<<<< Updated upstream
    global_sites, proto_sites, world_sites = NewtonManager._cl_inject_sites(builder, protos)

    # Global sites: (int, None)
=======
    global_sites, proto_sites = NewtonManager._cl_inject_sites(builder, protos)
>>>>>>> Stashed changes
    global_site_map: dict[str, tuple[int, None]] = {label: (idx, None) for label, idx in global_sites.items()}

    local_site_map = _replicate_into_builder(
        builder, sources, protos, env_ids, mapping, positions, quaternions, env0_pos, proto_sites, up_axis
    )

    site_index_map = {
        **global_site_map,
        **{label: (None, per_world) for label, per_world in local_site_map.items()},
    }
    return builder, stage_info, site_index_map


def _per_world_signatures(mapping: torch.Tensor) -> list[tuple[int, ...]]:
    """Per-column tuple of active source rows (the world's "source signature")."""
    return [tuple(torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist()) for col in range(mapping.size(1))]


def _hooks_active() -> bool:
    """Whether deformable per-world / post-replicate hooks are registered.

    These mutate worlds individually and break the "identical worlds" assumption the batched
    path relies on, so their presence forces the sequential replication path.
    """
    return bool(getattr(NewtonManager, "_per_world_builder_hooks", None)) or bool(
        getattr(NewtonManager, "_post_replicate_hooks", None)
    )


def _replicate_into_builder(
    builder: ModelBuilder,
    sources: Sequence[str],
    protos: dict[str, ModelBuilder],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    env0_pos: torch.Tensor,
    proto_sites: dict[int, dict[str, list[int]]],
    up_axis: str,
) -> dict[str, list[list[int]]]:
    """Replicate prototypes into ``builder`` (batched fast path or sequential fallback).

    Returns the per-world local site-index map (``{label: [[shape_idx, ...] per world]}``).
    """
    signatures = _per_world_signatures(mapping)

    # Build one combined prototype per unique source signature. A single-source signature
    # reuses its prototype directly (no copy); multi-source signatures concatenate their
    # prototypes once. ``shape_offsets`` records each source's shape offset within the
    # combined prototype so site indices can be resolved without a per-world pattern match.
    sig_ids: dict[tuple[int, ...], int] = {}
    combined_protos: list[ModelBuilder] = []
    shape_offsets: list[dict[int, int]] = []
    for sig in signatures:
        if sig in sig_ids:
            continue
        sig_ids[sig] = len(combined_protos)
        if len(sig) == 1:
            combined_protos.append(protos[sources[sig[0]]])
            shape_offsets.append({sig[0]: 0})
        else:
            combined = NewtonManager.create_builder(up_axis=up_axis)
            solvers.SolverMuJoCo.register_custom_attributes(combined)
            offsets: dict[int, int] = {}
            for row in sig:
                offsets[row] = combined.shape_count
                combined.add_builder(protos[sources[row]])
            combined_protos.append(combined)
            shape_offsets.append(offsets)

    signature_ids = [sig_ids[sig] for sig in signatures]

    if (
        isinstance(builder, BatchedModelBuilder)
        and not _hooks_active()
        and BatchedModelBuilder.can_replicate(combined_protos)
    ):
        front_shapes = builder.shape_count
        # Per-world rigid transforms ([N, 7]: delta position from env_0 + orientation).
        deltas = (positions - env0_pos).cpu().numpy().astype(np.float32)
        quats = quaternions.cpu().numpy().astype(np.float32)
        xforms = np.concatenate([deltas, quats], axis=1)
        builder.replicate_grouped(combined_protos, signature_ids, xforms)
        return _batched_site_map(
            front_shapes, signatures, signature_ids, combined_protos, shape_offsets, protos, sources, proto_sites
        )

    return _replicate_sequential(
        builder, sources, protos, env_ids, mapping, positions, quaternions, env0_pos, proto_sites
    )


def _batched_site_map(
    front_shapes: int,
    signatures: list[tuple[int, ...]],
    signature_ids: list[int],
    combined_protos: list[ModelBuilder],
    shape_offsets: list[dict[int, int]],
    protos: dict[str, ModelBuilder],
    sources: Sequence[str],
    proto_sites: dict[int, dict[str, list[int]]],
) -> dict[str, list[list[int]]]:
    """Resolve per-world site shape indices for the batched path (vectorized over worlds)."""
    num_worlds = len(signatures)
    shape_counts = np.array([combined_protos[sid].shape_count for sid in signature_ids], dtype=np.int64)
    # Exclusive prefix sum -> start shape index (within the worlds region) of each world.
    shape_prefix = np.concatenate([[0], np.cumsum(shape_counts)[:-1]])

    local_site_map: dict[str, list[list[int]]] = {}
    for col, sig in enumerate(signatures):
        base = front_shapes + int(shape_prefix[col])
        offsets = shape_offsets[signature_ids[col]]
        for row in sig:
            proto = protos[sources[row]]
            row_offset = base + offsets[row]
            for label, proto_shape_indices in proto_sites.get(id(proto), {}).items():
                if label not in local_site_map:
                    local_site_map[label] = [[] for _ in range(num_worlds)]
                local_site_map[label][col].extend(row_offset + psi for psi in proto_shape_indices)
    return local_site_map


def _replicate_sequential(
    builder: ModelBuilder,
    sources: Sequence[str],
    protos: dict[str, ModelBuilder],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    env0_pos: torch.Tensor,
    proto_sites: dict[int, dict[str, list[int]]],
) -> dict[str, list[list[int]]]:
    """Sequential per-world replication via :meth:`~newton.ModelBuilder.add_builder`.

    The guaranteed-correct fallback used when batched replication is unavailable (deformable
    hooks registered, particles in a prototype, or unsupported custom attributes).
    """
    num_worlds = mapping.size(1)
    local_site_map: dict[str, list[list[int]]] = {}

    # create a separate world for each environment (heterogeneous spawning)
    # Newton assigns sequential world IDs (0, 1, 2, ...), so we need to track the mapping
    for col, _ in enumerate(env_ids.tolist()):
        # begin a new world context (Newton assigns world ID = col)
        builder.begin_world()
        # add all active sources for this world
        delta_pos = (positions[col] - env0_pos).tolist()
        env_xform = wp.transform(positions[col].tolist(), quaternions[col].tolist())
        for label, xform in world_sites.items():
            if label not in local_site_map:
                local_site_map[label] = [[] for _ in range(num_worlds)]
            site_idx = builder.add_site(body=-1, xform=wp.transform_multiply(env_xform, xform), label=label)
            local_site_map[label][col].append(site_idx)
        for row in torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist():
            proto = protos[sources[row]]
            offset = builder.shape_count
            builder.add_builder(
                proto,
                xform=wp.transform(delta_pos, quaternions[col].tolist()),
            )
            # Compute final shape indices for sites in this proto
            for label, proto_shape_indices in proto_sites.get(id(proto), {}).items():
                if label not in local_site_map:
                    local_site_map[label] = [[] for _ in range(num_worlds)]
                for proto_shape_idx in proto_shape_indices:
                    local_site_map[label][col].append(offset + proto_shape_idx)

        # Run per-world builder hooks (e.g. deformable body registration).
        if hasattr(NewtonManager, "_per_world_builder_hooks"):
            for hook in NewtonManager._per_world_builder_hooks:
                hook(builder, col, positions[col].tolist(), quaternions[col].tolist())

        # end the world context
        builder.end_world()

    # Run post-replicate hooks (e.g. builder.color() for deformable coloring).
    if hasattr(NewtonManager, "_post_replicate_hooks"):
        for hook in NewtonManager._post_replicate_hooks:
            hook(builder)

    return local_site_map


# Built-in label arrays that ``_rename_builder_labels`` rewrites in Pass 1.
# Each type ``t`` has a paired ``<t>_label`` (or ``<t>_key``) string column
# and a ``<t>_world`` int column on Newton's ``ModelBuilder``. Exposed as a
# module-level constant so tests can import it instead of duplicating.
_BUILTIN_LABEL_TYPES: tuple[str, ...] = (
    "body",
    "joint",
    "shape",
    "articulation",
    "constraint_mimic",
    "equality_constraint",
)


def _rename_builder_labels(
    builder: ModelBuilder,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
) -> None:
    """Rename builder labels/keys from source roots to destination roots.

    Walks both built-in label arrays (see :data:`_BUILTIN_LABEL_TYPES`) and any
    string-typed custom-attribute column whose frequency declares a sibling
    world column (``references="world"``).
    The boundary-safe match (exact source root, or source root followed by ``/``)
    makes the rewrite a no-op for strings that are not paths under the source.
    Non-path custom string columns are passed through untouched and any future
    solver-registered string column is handled automatically without changes here.

    Args:
        builder: Newton model builder to update in-place.
        sources: Source prim root paths.
        destinations: Destination prim path templates.
        env_ids: Environment ids corresponding to mapping columns.
        mapping: Boolean source-to-environment mapping matrix.
    """
    # per-source, per-world renaming (strict prefix swap), compact style preserved
    for i, src_path in enumerate(sources):
        # Canonicalize the source root (drop any trailing ``/``) so the
        # boundary-safe match logic in ``_rename_pair`` is unambiguous.
        src_root = src_path.rstrip("/")
        world_cols = torch.nonzero(mapping[i], as_tuple=True)[0].tolist()
        # Map Newton world IDs (sequential) to destination paths using env_ids
        world_roots = {int(env_ids[c]): destinations[i].format(int(env_ids[c])) for c in world_cols}

        def _rename_pair(values, worlds):
            if len(values) != len(worlds):
                raise ValueError(f"label/world column length mismatch: {len(values)} vs {len(worlds)}")
            for k in range(len(values)):
                v = values[k]
                if not isinstance(v, str):
                    continue
                world_id = int(worlds[k])
                if world_id not in world_roots:
                    continue
                # Gate on an explicit prefix test before slicing. ``str.removeprefix``
                # is tempting but conflates "match with empty suffix" and "no match"
                # (both return a string starting with "/"), so a label already
                # rewritten in an earlier source-iteration would be re-prepended to
                # the next iteration's dst root.
                if not v.startswith(src_root):
                    continue
                suffix = v[len(src_root) :]
                # ``suffix == ""``     -> exact source-root match (rewrite to dst root).
                # ``suffix[0] == "/"`` -> child path under source.
                # otherwise           -> boundary-bleed sibling like "/Sources/protoAB/x"
                #                        when src_root is "/Sources/protoA" -> skip.
                if suffix and not suffix.startswith("/"):
                    continue
                values[k] = world_roots[world_id] + suffix

        # Pass 1: built-in label arrays. Each has a paired ``*_world`` int column.
        # Use ``is None`` (not ``or``) so an empty-but-defined ``*_label`` column
        # is recognized — falling through to ``*_key`` would over-match a
        # builder that legitimately exposes both attributes.
        for t in _BUILTIN_LABEL_TYPES:
            labels = getattr(builder, f"{t}_label", None)
            if labels is None:
                labels = getattr(builder, f"{t}_key", None)
            worlds_arr = getattr(builder, f"{t}_world", None)
            if labels is None or worlds_arr is None:
                continue
            _rename_pair(labels, worlds_arr)

        # Pass 2: string-typed custom-attribute columns (e.g. ``mujoco:tendon_label``)
        # paired with a world companion declared via ``references="world"``. Index
        # world companions by frequency for O(1) lookup, then walk the str columns.
        custom = builder.custom_attributes
        world_by_freq: dict[str, ModelBuilder.CustomAttribute] = {}
        for attr in custom.values():
            if getattr(attr, "references", None) == "world":
                world_by_freq[attr.frequency] = attr
        for attr in custom.values():
            if attr.dtype is not str:
                continue
            world_attr = world_by_freq.get(attr.frequency)
            if world_attr is None:
                continue
            values = attr.values
            worlds = world_attr.values
            if not values or not worlds:
                continue
            _rename_pair(values, worlds)


def newton_physics_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    up_axis: str = "Z",
    simplify_meshes: bool = True,
):
    """Replicate prims into a Newton ``ModelBuilder`` using a per-source mapping.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        device: Device used by the finalized Newton model builder.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of the populated Newton model builder and stage metadata.
    """
    if positions is None:
        positions = torch.zeros((mapping.size(1), 3), device=mapping.device, dtype=torch.float32)
    if quaternions is None:
        quaternions = torch.zeros((mapping.size(1), 4), device=mapping.device, dtype=torch.float32)
        quaternions[:, 3] = 1.0

    builder, stage_info, site_index_map = _build_newton_builder_from_mapping(
        stage=stage,
        sources=sources,
        env_ids=env_ids,
        mapping=mapping,
        positions=positions,
        quaternions=quaternions,
        up_axis=up_axis,
        simplify_meshes=simplify_meshes,
    )
    _rename_builder_labels(builder, sources, destinations, env_ids, mapping)
    NewtonManager._cl_site_index_map = site_index_map
    NewtonManager._world_xforms = [
        wp.transform(positions[col].tolist(), quaternions[col].tolist()) for col in range(mapping.size(1))
    ]
    NewtonManager.set_builder(builder)
    NewtonManager._num_envs = mapping.size(1)
    return builder, stage_info


def newton_visualizer_prebuild(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    up_axis: str = "Z",
    simplify_meshes: bool = True,
):
    """Replicate a clone plan into a finalized Newton model/state for visualization.

    Unlike :func:`newton_physics_replicate`, this path does not mutate ``NewtonManager`` and is intended
    for prebuilding visualizer-only artifacts that can be consumed by scene data providers.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        device: Device used by the finalized Newton model.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of finalized Newton model and state.
    """
    builder, _, _site_index_map = _build_newton_builder_from_mapping(
        stage=stage,
        sources=sources,
        env_ids=env_ids,
        mapping=mapping,
        positions=positions,
        quaternions=quaternions,
        up_axis=up_axis,
        simplify_meshes=simplify_meshes,
    )
    _rename_builder_labels(builder, sources, destinations, env_ids, mapping)
    model = builder.finalize(device=device)
    state = model.state()
    return model, state
