# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRT/X-native rendering for Newton-family visualization markers.

The Newton marker backends registered in the simulation's marker registry own
all marker logic (prototype spec inference, mesh generation, environment
slicing). They render through the small Newton ``ViewerBase`` logging surface
(``log_mesh`` / ``log_instances`` / ``log_lines``); this module adapts that
surface onto the OVRT/X runtime stage, so markers are path-traced together
with the rest of the scene (correct occlusion and lighting, no browser-side
overlay geometry).

Marker prototype meshes are authored into the runtime stage once, instances
are cloned from them on demand into per-batch pools, and every rendered frame
updates instance transforms (scale baked into the matrix). Hiding uses
"parking" (a degenerate far-away transform) rather than the ``visibility``
attribute: OVRT/X ignores visibility writes on runtime-cloned prims, and
prototypes authored invisible poison their clones. Line batches (e.g. frame
axes) are rendered as thin oriented boxes.
"""

from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp
from isaac_viser.transforms import quaternion_wxyz_to_matrix

logger = logging.getLogger(__name__)

_PROTO_ROOT = "/IsaacViserMarkerProtos"
_INSTANCE_ROOT = "/IsaacViserMarkerInstances"

_UNIT_BOX_VERTICES = np.array(
    [
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)
_UNIT_BOX_FACES = np.array(
    [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ],
    dtype=np.int32,
)


@dataclass
class _InstancePool:
    """Clones of one prototype, grown on demand; ``active`` are visible."""

    prototype: str
    paths: list[str] = field(default_factory=list)
    active: int = 0


class OvrtxMarkerOverlay:
    """Adapts the Newton marker logging interface onto the OVRT/X runtime stage."""

    def __init__(self, renderer: Any, num_envs: int):
        """Create the overlay.

        Args:
            renderer: OVRT/X renderer owning the runtime stage.
            num_envs: Number of vectorized environments.
        """
        self._renderer = renderer
        # Marker poses arrive in world frame; the Newton renderer adds these.
        self.world_offsets = wp.zeros(max(1, num_envs), dtype=wp.vec3)
        self._meshes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
        self._prototypes: dict[tuple[str, tuple[float, float, float]], str] = {}
        self._pools: dict[str, _InstancePool] = {}
        self._instance_count = 0
        self._prototype_count = 0

    def render(self, visible_env_ids: list[int] | None, num_envs: int) -> None:
        """Render all registered Newton-family marker groups into the stage."""
        from isaaclab_visualizers.newton.newton_visualization_markers import (
            render_newton_visualization_markers,
        )

        render_newton_visualization_markers(self, visible_env_ids, num_envs=num_envs)

    def close(self) -> None:
        """Hide all marker instances."""
        for pool in self._pools.values():
            with contextlib.suppress(Exception):
                self._deactivate(pool)

    # -- Newton ``ViewerBase`` logging surface used by NewtonVisualizationMarkers --

    def log_mesh(
        self,
        name: str,
        points: Any,
        indices: Any,
        normals: Any = None,
        uvs: Any = None,
        texture: Any = None,
        hidden: bool = False,
    ) -> None:
        """Register a marker prototype mesh (textures are ignored)."""
        vertices = np.asarray(points.numpy(), dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(indices.numpy(), dtype=np.int32).reshape(-1, 3)
        normal_array = None
        if normals is not None:
            normal_array = np.asarray(normals.numpy(), dtype=np.float32).reshape(-1, 3)
            if len(normal_array) != len(vertices):
                normal_array = None
        self._meshes[name] = (vertices, faces, normal_array)

    def log_instances(
        self,
        name: str,
        mesh_name: str,
        xforms: Any,
        scales: Any,
        colors: Any,
        materials: Any,
        hidden: bool = False,
    ) -> None:
        """Update one marker batch from packed instance transforms."""
        pool_key = _sanitize(name)
        if hidden or xforms is None:
            self._deactivate_key(pool_key)
            return
        mesh = self._meshes.get(mesh_name)
        if mesh is None:
            return

        # wp.transform rows are (px, py, pz, qx, qy, qz, qw).
        transform_array = np.asarray(xforms.numpy(), dtype=np.float64).reshape(-1, 7)
        positions = transform_array[:, :3]
        quats_wxyz = transform_array[:, [6, 3, 4, 5]]
        scale_array = (
            np.ones((len(transform_array), 3), dtype=np.float64)
            if scales is None
            else np.asarray(scales.numpy(), dtype=np.float64).reshape(-1, 3)
        )
        # Marker colors are constant per prototype; bake them into the prototype mesh.
        color = (0.7, 0.7, 0.7)
        if colors is not None:
            color_array = np.asarray(colors.numpy(), dtype=np.float64).reshape(-1, 3)
            if len(color_array):
                color = tuple(float(v) for v in color_array[0])

        prototype = self._ensure_prototype(mesh_name, mesh, color)
        self._write_pool(pool_key, prototype, positions, quats_wxyz, scale_array)

    def log_lines(
        self,
        name: str,
        starts: Any,
        ends: Any,
        colors: Any,
        width: float = 0.01,
        hidden: bool = False,
    ) -> None:
        """Render line segments (e.g. frame axes) as thin oriented boxes."""
        base_key = _sanitize(name)
        if hidden or starts is None:
            for pool_key in list(self._pools):
                if pool_key.startswith(f"{base_key}__c"):
                    self._deactivate_key(pool_key)
            return

        start_array = np.asarray(starts.numpy(), dtype=np.float64).reshape(-1, 3)
        end_array = np.asarray(ends.numpy(), dtype=np.float64).reshape(-1, 3)
        color_array = np.asarray(colors.numpy(), dtype=np.float64).reshape(-1, 3)
        width = max(float(width), 1e-4)

        directions = end_array - start_array
        lengths = np.linalg.norm(directions, axis=1)
        valid = lengths > 1e-9
        midpoints = (start_array + end_array) / 2.0
        quats = _quats_z_to(directions, lengths)
        scale_array = np.column_stack([np.full_like(lengths, width), np.full_like(lengths, width), lengths])

        # Group segments by color so each pool's prototype has one baked color.
        color_keys = [tuple(np.round(row, 3)) for row in color_array]
        seen: dict[tuple[float, float, float], list[int]] = {}
        for index, key in enumerate(color_keys):
            if valid[index]:
                seen.setdefault(key, []).append(index)
        for group_index, (color, segment_indices) in enumerate(sorted(seen.items())):
            pool_key = f"{base_key}__c{group_index}"
            prototype = self._ensure_prototype(
                "__unit_box__", (_UNIT_BOX_VERTICES, _UNIT_BOX_FACES, None), color
            )
            selection = np.asarray(segment_indices, dtype=np.int64)
            self._write_pool(
                pool_key, prototype, midpoints[selection], quats[selection], scale_array[selection]
            )

    # -- Internals --

    def _ensure_prototype(
        self,
        mesh_key: str,
        mesh: tuple[np.ndarray, np.ndarray, np.ndarray | None],
        color: tuple[float, float, float],
    ) -> str:
        """Author one prototype mesh on the runtime stage, once per color.

        The prototype is parked (moved far away with a degenerate scale)
        immediately after authoring so only its clones are ever visible.
        """
        key = (mesh_key, tuple(round(float(v), 3) for v in color))
        prototype = self._prototypes.get(key)
        if prototype is not None:
            return prototype
        prototype = f"{_PROTO_ROOT}/proto_{self._prototype_count}"
        self._prototype_count += 1
        vertices, faces, normals = mesh
        self._renderer.add_usd_reference_from_string(_mesh_usda(vertices, faces, normals, color), prototype)
        self._park([prototype])
        self._prototypes[key] = prototype
        return prototype

    def _park(self, paths: list[str]) -> None:
        """Hide prims by moving them far away with a degenerate scale.

        OVRT/X ignores ``visibility`` writes on runtime-cloned prims, so
        parking through the (proven) transform path is used for hiding.
        """
        from ovrtx import Semantic

        matrices = np.zeros((len(paths), 4, 4), dtype=np.float64)
        matrices[:, 0, 0] = matrices[:, 1, 1] = matrices[:, 2, 2] = 1.0e-4
        matrices[:, 3, 2] = -1.0e6
        matrices[:, 3, 3] = 1.0
        self._renderer.write_attribute(paths, "omni:resetXformStack", np.ones(len(paths), dtype=bool))
        self._renderer.write_attribute(paths, "omni:xform", matrices, semantic=Semantic.XFORM_MAT4x4)

    def _write_pool(
        self,
        pool_key: str,
        prototype: str,
        positions: np.ndarray,
        quats_wxyz: np.ndarray,
        scales: np.ndarray,
    ) -> None:
        """Size one instance pool to the batch and write transforms/visibility."""
        count = len(positions)
        pool = self._pools.get(pool_key)
        if pool is None:
            pool = self._pools[pool_key] = _InstancePool(prototype=prototype)
        if count == 0:
            self._deactivate(pool)
            return

        if count > len(pool.paths):
            new_paths = [
                f"{_INSTANCE_ROOT}/i_{self._instance_count + offset}"
                for offset in range(count - len(pool.paths))
            ]
            self._instance_count += len(new_paths)
            self._renderer.clone_usd(pool.prototype, new_paths)
            # World poses must not re-compose with ancestor transforms.
            self._renderer.write_attribute(
                new_paths, "omni:resetXformStack", np.ones(len(new_paths), dtype=bool)
            )
            pool.paths.extend(new_paths)

        if count < pool.active:
            self._park(pool.paths[count : pool.active])
        pool.active = count

        matrices = np.zeros((count, 4, 4), dtype=np.float64)
        rotations = quaternion_wxyz_to_matrix(quats_wxyz)
        # Row-vector convention with local scale: M[:3,:3] = diag(s) @ Rᵀ.
        matrices[:, :3, :3] = np.transpose(rotations, (0, 2, 1)) * scales[:, :, None]
        matrices[:, 3, :3] = positions
        matrices[:, 3, 3] = 1.0
        from ovrtx import Semantic

        self._renderer.write_attribute(pool.paths[:count], "omni:xform", matrices, semantic=Semantic.XFORM_MAT4x4)

    def _deactivate_key(self, pool_key: str) -> None:
        pool = self._pools.get(pool_key)
        if pool is not None:
            self._deactivate(pool)

    def _deactivate(self, pool: _InstancePool) -> None:
        if pool.active == 0 or not pool.paths:
            return
        self._park(pool.paths[: pool.active])
        pool.active = 0


def _sanitize(name: str) -> str:
    """Map a marker batch name to a stable pool key."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def _quats_z_to(directions: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Quaternions (wxyz) rotating the local +Z axis onto each direction."""
    quats = np.zeros((len(directions), 4), dtype=np.float64)
    quats[:, 0] = 1.0
    safe = lengths > 1e-9
    unit = np.zeros_like(directions)
    unit[safe] = directions[safe] / lengths[safe, None]
    z_axis = np.array([0.0, 0.0, 1.0])
    dots = unit @ z_axis
    # General case: half-angle construction between +Z and the direction.
    general = safe & (dots > -1.0 + 1e-9)
    axes = np.cross(np.broadcast_to(z_axis, unit.shape), unit)
    quats[general, 0] = 1.0 + dots[general]
    quats[general, 1:] = axes[general]
    # Antiparallel: 180-degree rotation about X.
    flipped = safe & ~general
    quats[flipped] = (0.0, 1.0, 0.0, 0.0)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    return quats / np.maximum(norms, 1e-12)


def _mesh_usda(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None,
    color: tuple[float, float, float],
) -> str:
    """Author one single-color triangle mesh as an inline USDA layer."""

    def _vec3_list(array: np.ndarray) -> str:
        return ", ".join(f"({row[0]:.6g}, {row[1]:.6g}, {row[2]:.6g})" for row in array)

    normal_block = ""
    if normals is not None:
        normal_block = (
            f"    normal3f[] normals = [{_vec3_list(normals)}] (\n"
            '        interpolation = "vertex"\n'
            "    )\n"
        )
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    indices = ", ".join(str(index) for index in faces.reshape(-1))
    counts = ", ".join("3" for _ in range(len(faces)))
    red, green, blue = (float(channel) for channel in color)
    return f"""#usda 1.0
(
    defaultPrim = "Marker"
)

def Mesh "Marker"
{{
    point3f[] points = [{_vec3_list(vertices)}]
    float3[] extent = [({low[0]:.6g}, {low[1]:.6g}, {low[2]:.6g}), ({high[0]:.6g}, {high[1]:.6g}, {high[2]:.6g})]
    int[] faceVertexIndices = [{indices}]
    int[] faceVertexCounts = [{counts}]
{normal_block}    color3f[] primvars:displayColor = [({red:.6g}, {green:.6g}, {blue:.6g})]
}}
"""
