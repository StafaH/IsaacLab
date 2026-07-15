# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viser-based visualizer backed by isaac_viser (OVRT/X RTX rendering).

The browser viewport is an OVRT/X-rendered image streamed through a Viser
server; no geometry is sent to Viser's native WebGL renderer — markers are
injected into the OVRT/X runtime stage and path-traced with the rest of the
scene. The composed USD stage is exported once during
:meth:`ViserVisualizer.initialize`, then rigid-body world transforms from the
:class:`~isaaclab.scene_data.SceneDataProvider` are written into the OVRT/X
runtime stage every rendered frame — GPU-to-GPU through a persistent attribute
binding when the simulation runs on CUDA, with a NumPy fallback otherwise.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from isaac_viser import Viewer, ViewerConfig
from isaac_viser.transforms import poses_wxyz_to_matrices

from isaaclab.scene_data.scene_data_backend import SceneDataFormat
from isaaclab.visualizers.base_visualizer import BaseVisualizer

from .viser_visualizer_cfg import ViserVisualizerCfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.scene_data import SceneDataProvider


def _kit_app_is_running() -> bool:
    """Return whether an Isaac Sim (Kit) app is running in this process.

    Kitless flows may still import ``omni.kit.app`` without starting an app, so
    the module's presence alone is not a signal.
    """
    app_module = sys.modules.get("omni.kit.app")
    if app_module is None:
        return False
    try:
        app = app_module.get_app()
        return app is not None and bool(app.is_running())
    except Exception:
        return False


@wp.kernel(enable_backward=False)
def _transforms_to_usd_matrices(
    src: wp.array(dtype=wp.transformf),
    source_indices: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.mat44d),
):
    """Write world transforms as USD row-vector matrices (Rᵀ, translation in row 3)."""
    tid = wp.tid()
    transform = src[source_indices[tid]]
    p = wp.transform_get_translation(transform)
    q = wp.transform_get_rotation(transform)
    if q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] < 1.0e-12:
        # Uninitialized transform: keep the position, use an identity rotation.
        q = wp.quatf(0.0, 0.0, 0.0, 1.0)
    r = wp.quat_to_matrix(q)
    z = wp.float64(0.0)
    out[tid] = wp.mat44d(
        wp.float64(r[0, 0]), wp.float64(r[1, 0]), wp.float64(r[2, 0]), z,
        wp.float64(r[0, 1]), wp.float64(r[1, 1]), wp.float64(r[2, 1]), z,
        wp.float64(r[0, 2]), wp.float64(r[1, 2]), wp.float64(r[2, 2]), z,
        wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]), wp.float64(1.0),
    )


class ViserVisualizer(BaseVisualizer):
    """Web visualizer streaming OVRT/X path-traced frames through Viser."""

    def __init__(self, cfg: ViserVisualizerCfg):
        """Initialize Viser visualizer state.

        Args:
            cfg: Viser visualizer configuration.
        """
        super().__init__(cfg)
        self.cfg: ViserVisualizerCfg = cfg
        self._viewer: Viewer | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._transform_paths: list[str] = []
        self._transform_output: Any | None = None
        self._binding: Any | None = None
        self._binding_source_indices: list[int] = []
        self._binding_indices_wp: Any | None = None
        self._env_prim_paths: list[str] = []
        self._num_envs = 0
        self._resolved_env_ids: list[int] | None = None
        self._applied_env_flags: tuple[bool, ...] | None = None
        self._marker_overlay: Any | None = None
        self._warned_marker_render_failure = False
        self._sim_time = 0.0
        self._step_count = 0

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        """Export the composed stage and start the OVRT/X-backed web viewer.

        Args:
            scene_data_provider: Scene data provider used to fetch stage and transforms.

        Raises:
            RuntimeError: If no USD stage is available or the stage export fails.
        """
        if self._is_initialized:
            logger.debug("[ViserVisualizer] initialize() called while already initialized.")
            return

        if _kit_app_is_running():
            # OVRT/X's plugins clash with Kit's in-process USD build (same renamed
            # symbol namespace, different symbol set), mirroring the launcher's
            # existing '--visualizer kit' + ovrtx incompatibility.
            raise RuntimeError(
                "ViserVisualizer renders with OVRT/X, which cannot load inside an Isaac Sim (Kit)"
                " process. Launch with a kitless physics backend instead, e.g."
                " 'env.sim.physics=newton_mjwarp' or 'env.sim.physics=ovphysx'."
            )

        provider = self._set_scene_data_provider(scene_data_provider)
        stage = provider.usd_stage
        if stage is None:
            raise RuntimeError(
                "ViserVisualizer requires a USD stage to export for OVRT/X rendering, but none is available."
            )
        self._env_ids = self._compute_visualized_env_ids()

        self._temporary_directory = tempfile.TemporaryDirectory(prefix="isaac_viser_")
        stage_path = Path(self._temporary_directory.name) / "scene.usda"
        if not self._export_render_stage(stage, stage_path):
            raise RuntimeError(f"failed to export the composed stage to {stage_path}")

        # Isaac Lab pins usd-core to the same USD major.minor that ovrtx bundles, which
        # ovrtx rejects by default. In this kitless flow usd-core's pxr is fully loaded
        # before ovrtx (the stage export above), which co-exists; opt out of the guard
        # unless the user has set it explicitly.
        os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")

        eye, lookat = self._resolve_cfg_camera_pose("ViserVisualizer")
        config = ViewerConfig(
            host=self.cfg.bind_address,
            port=self.cfg.port,
            label=self.cfg.label or "Isaac Viser",
            width=self.cfg.width,
            height=self.cfg.height,
            target_fps=self.cfg.target_fps,
            jpeg_quality=self.cfg.jpeg_quality,
            render_mode=self.cfg.render_mode,
            camera_position=eye,
            camera_look_at=lookat,
            vertical_fov_degrees=self._focal_length_to_vertical_fov_degrees(),
        )
        self._viewer = self._create_viewer(stage_path, config)

        stage_paths = self._query_stage_paths()
        # Newton's kitless cloning replicates environments inside the physics model
        # only; the USD stage (and therefore the render) may hold fewer env prims
        # than the simulation. Offer only environments that exist on the stage.
        self._num_envs = provider.num_envs
        self._env_prim_paths = [f"/World/envs/env_{index}" for index in range(self._num_envs)]
        if stage_paths is not None:
            self._env_prim_paths = [path for path in self._env_prim_paths if path in stage_paths]
        if self._num_envs > 0:
            self._viewer.set_num_environments(max(1, len(self._env_prim_paths)))
        from isaaclab_visualizers.newton_adapter import resolve_visible_env_indices

        self._resolved_env_ids = resolve_visible_env_indices(
            self._env_ids, self.cfg.max_visible_envs, self._num_envs
        )

        self._transform_paths = self._resolve_transform_paths(provider, stage_paths)
        self._transform_output = SceneDataFormat.Transform()
        try:
            self._setup_transform_binding()
        except Exception as exc:
            logger.warning(
                "[ViserVisualizer] Zero-copy transform binding unavailable (%s); using CPU transform writes.",
                exc,
            )
            self._binding = None

        if self.cfg.enable_markers:
            self._marker_overlay = self._create_marker_overlay()

        # Compile shaders and stream textures during startup instead of at the
        # first browser connect; rendering starts once warmup completes.
        self._viewer.warmup()

        viewer_url = f"http://{self.cfg.display_address}:{int(self.cfg.port)}"
        if self.cfg.verbose:
            self._log_viewer_url("ViserVisualizer", viewer_url)
        if self.cfg.open_browser and not webbrowser.open_new_tab(viewer_url):
            logger.info("[ViserVisualizer] Could not auto-open browser tab. Open manually: %s", viewer_url)
        if self.cfg.share:
            threading.Thread(target=self._request_share_url, name="viser-share-url", daemon=True).start()

        self._log_initialization_table(
            logger=logger,
            title="ViserVisualizer Configuration",
            rows=[
                ("eye", eye),
                ("lookat", lookat),
                ("resolution", f"{self.cfg.width}x{self.cfg.height}"),
                ("render_mode", self.cfg.render_mode),
                ("target_fps", self.cfg.target_fps),
                ("num_envs", self._num_envs),
                ("visible_envs", "all" if self._resolved_env_ids is None else len(self._resolved_env_ids)),
                ("tracked_transforms", len(self._transform_paths)),
                ("transform_path", "gpu zero-copy" if self._binding is not None else "cpu"),
                ("markers", self._marker_overlay is not None),
                ("bind_address", self.cfg.bind_address),
                ("port", self.cfg.port),
            ],
        )
        self._is_initialized = True

    def step(self, dt: float) -> None:
        """Push current world transforms into OVRT/X and render browser clients.

        Args:
            dt: Simulation time-step in seconds. ``0.0`` keeps clients responsive
                while training is paused.
        """
        if not self._is_initialized or self._viewer is None:
            return

        if dt > 0.0:
            self._sim_time += dt
            self._step_count += 1

        if not self._viewer.has_clients:
            # Nothing to render; avoid GPU work and pause-loop busy spinning.
            if dt == 0.0:
                time.sleep(0.01)
            return

        self._apply_environment_selection()
        self._viewer.update_metrics(step_count=self._step_count, sim_time=self._sim_time)
        # Transform writes run inside the render pipeline (after the previous
        # frame is consumed, before the next step is kicked) so GPU mappings
        # never overlap an in-flight OVRT/X render.
        rendered = self._viewer.render(delta_time=dt or None, before_kick=self._write_transforms)
        if rendered and self._marker_overlay is not None:
            self._render_markers()
        if dt == 0.0 and not rendered:
            # Paused and frame-rate limited: yield so the pause loop does not spin hot.
            time.sleep(0.005)

    def close(self) -> None:
        """Close the web viewer and remove the exported temporary stage."""
        if not self._is_initialized:
            return
        if self._binding is not None:
            with contextlib.suppress(Exception):
                self._binding.unbind()
            self._binding = None
        if self._marker_overlay is not None:
            with contextlib.suppress(Exception):
                self._marker_overlay.close()
            self._marker_overlay = None
        try:
            if self._viewer is not None:
                self._viewer.close()
        except Exception as exc:
            logger.warning("[ViserVisualizer] Error during close: %s", exc)
        self._viewer = None
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
        self._is_initialized = False
        self._is_closed = True

    def is_running(self) -> bool:
        """Return whether the visualizer should continue stepping."""
        return self._is_initialized and not self._is_closed and self._viewer is not None

    def is_training_paused(self) -> bool:
        """Return whether training is paused from the browser GUI.

        A queued single-step request resumes training for exactly one step
        (consumed through :meth:`isaac_viser.Viewer.should_step`).
        """
        if self._viewer is None:
            return False
        return not self._viewer.should_step()

    def supports_markers(self) -> bool:
        """Markers are path-traced by OVRT/X together with the rest of the scene."""
        if not self.cfg.enable_markers:
            return False
        return self._marker_overlay is not None or not self._is_initialized

    def supports_live_plots(self) -> bool:
        """Live-plot widgets are not exposed by this backend."""
        return False

    def set_camera_view(self, eye: tuple, target: tuple) -> None:
        """Point connected browser cameras at a new pose.

        Args:
            eye: Camera eye position.
            target: Camera target position.
        """
        if self._viewer is None:
            return
        self._viewer._initial_camera_position = tuple(float(v) for v in eye)
        self._viewer._initial_camera_look_at = tuple(float(v) for v in target)
        for client in self._viewer.server.get_clients().values():
            client.camera.position = tuple(float(v) for v in eye)
            client.camera.look_at = tuple(float(v) for v in target)

    def _create_viewer(self, stage_path: Path, config: ViewerConfig) -> Viewer:
        """Create the isaac_viser viewer. Split out for tests."""
        return Viewer(stage_path, config=config)

    def _create_marker_overlay(self) -> Any | None:
        """Create the OVRT/X marker overlay; ``None`` when unavailable."""
        try:
            from .ovrtx_marker_overlay import OvrtxMarkerOverlay

            return OvrtxMarkerOverlay(self._viewer.renderer, self._num_envs)
        except Exception as exc:
            logger.warning("[ViserVisualizer] Marker overlay unavailable: %s", exc)
            return None

    def _request_share_url(self) -> None:
        """Request a public Viser share URL (network round-trip; runs in background)."""
        try:
            url = self._viewer.server.request_share_url()
            print(f"[ViserVisualizer] Share URL: {url}")
        except Exception as exc:
            logger.warning("[ViserVisualizer] Could not create a share URL: %s", exc)

    @staticmethod
    def _export_render_stage(stage: Any, stage_path: Path) -> bool:
        """Flatten ``stage`` to ``stage_path`` in a fully renderable, writable form.

        Two transformations are applied to the flattened copy:

        * Instancing is expanded. Cloned environments are typically
          instanceable, which turns their link prims into unwritable instance
          proxies; disabling instancing makes every environment's links real
          prims so per-link world transforms can be written into OVRT/X.
        * Environments replicated only inside the physics model (Newton's
          kitless fast cloning leaves ``env_1..N`` as empty Xforms on the USD
          stage) receive a copy of ``env_0``'s children so they render too.
        """
        from pxr import Sdf, Usd

        flattened = Usd.Stage.Open(stage.Flatten())
        instance_paths = [prim.GetPath() for prim in flattened.Traverse() if prim.IsInstance()]
        for path in instance_paths:
            flattened.GetPrimAtPath(path).SetInstanceable(False)

        layer = flattened.GetRootLayer()
        envs_root = flattened.GetPrimAtPath("/World/envs")
        source = flattened.GetPrimAtPath("/World/envs/env_0")
        if envs_root and source:
            source_children = [child.GetName() for child in source.GetChildren()]
            for env_prim in envs_root.GetChildren():
                if env_prim.GetPath() == source.GetPath():
                    continue
                existing = {child.GetName() for child in env_prim.GetChildren()}
                for name in source_children:
                    if name in existing:
                        continue
                    Sdf.CopySpec(
                        layer,
                        source.GetPath().AppendChild(name),
                        layer,
                        env_prim.GetPath().AppendChild(name),
                    )
        return bool(layer.Export(str(stage_path)))

    def _query_stage_paths(self) -> set[str] | None:
        """Return the prim paths on the OVRT/X runtime stage, or ``None`` on failure."""
        try:
            return set(self._viewer.renderer.query_prims())
        except Exception as exc:
            logger.warning("[ViserVisualizer] Could not query OVRT/X stage prims: %s", exc)
            return None

    def _resolve_transform_paths(self, provider: SceneDataProvider, stage_paths: set[str] | None) -> list[str]:
        """Match backend transform paths against prims on the OVRT/X runtime stage.

        Transforms whose path is unknown to the runtime stage (for example
        environments replicated only inside the physics model) are dropped;
        their slots keep ``None`` so indices stay aligned with the backend
        transform order.
        """
        backend_paths = list(provider.backend.transform_paths)
        if not backend_paths:
            logger.warning("[ViserVisualizer] Scene data backend exposes no transform paths.")
            return []
        if stage_paths is None:
            return backend_paths
        resolved = [path if path in stage_paths else None for path in backend_paths]
        missing = [path for path, kept in zip(backend_paths, resolved) if kept is None]
        if missing:
            logger.info(
                "[ViserVisualizer] %d of %d transforms have no prim on the OVRT/X stage and are skipped"
                " (expected for environments replicated only inside the physics model; e.g. %s).",
                len(missing),
                len(backend_paths),
                ", ".join(missing[:4]),
            )
        return resolved

    def _setup_transform_binding(self) -> None:
        """Create a persistent OVRT/X attribute binding for per-frame transform writes."""
        bound = [(index, path) for index, path in enumerate(self._transform_paths) if path is not None]
        if not bound:
            return
        from ovrtx import BindingFlag

        renderer = self._viewer.renderer
        prim_paths = [path for _, path in bound]
        # World poses must not re-compose with ancestor transforms.
        renderer.write_attribute(prim_paths, "omni:resetXformStack", np.ones(len(prim_paths), dtype=bool))
        self._binding = renderer.bind_attribute(
            prim_paths=prim_paths,
            attribute_name="omni:xform",
            dtype="float64",
            shape=(4, 4),
            flags=BindingFlag.OPTIMIZE,
        )
        self._binding_source_indices = [index for index, _ in bound]

    def _write_transforms(self) -> None:
        """Write current backend world transforms into the OVRT/X runtime stage."""
        provider = self._scene_data_provider
        if provider is None or not self._transform_paths or provider.transform_count == 0:
            return
        if not provider.get_transforms(self._transform_output):
            logger.warning("[ViserVisualizer] Scene data provider could not convert transforms.")
            return

        src = self._transform_output.transforms
        if self._binding is not None and src.device.is_cuda:
            self._write_transforms_bound(src)
            return

        # Fallback: convert on the CPU and write through the viewer.
        # wp.transformf rows are (x, y, z, qx, qy, qz, qw); reorder to (x, y, z, qw, qx, qy, qz).
        transforms = np.asarray(src.numpy(), dtype=np.float64)
        count = min(len(transforms), len(self._transform_paths))
        poses = np.empty((count, 7), dtype=np.float64)
        poses[:, :3] = transforms[:count, :3]
        poses[:, 3] = transforms[:count, 6]
        poses[:, 4:7] = transforms[:count, 3:6]

        # Skip unmatched paths and uninitialized (zero-quaternion) transforms.
        mask = np.linalg.norm(poses[:, 3:7], axis=1) > 1e-6
        mask &= np.array([path is not None for path in self._transform_paths[:count]], dtype=bool)
        if not mask.any():
            return
        prim_paths = [self._transform_paths[index] for index in np.flatnonzero(mask)]
        self._viewer.write_world_poses(prim_paths, poses_wxyz_to_matrices(poses[mask]))

    def _write_transforms_bound(self, src: Any) -> None:
        """Convert transforms into OVRT/X-owned GPU memory with a Warp kernel (zero host copies)."""
        from ovrtx import Device as OvrtxDevice

        if self._binding_indices_wp is None or str(self._binding_indices_wp.device) != str(src.device):
            self._binding_indices_wp = wp.array(self._binding_source_indices, dtype=wp.int32, device=src.device)
        with self._binding.map(device=OvrtxDevice.CUDA, device_id=src.device.ordinal) as mapping:
            out = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)
            wp.launch(
                _transforms_to_usd_matrices,
                dim=len(self._binding_source_indices),
                inputs=[src, self._binding_indices_wp],
                outputs=[out],
                device=src.device,
            )
            # Hand Warp's stream to OVRT/X instead of a host-blocking device
            # sync: ordering is resolved on the GPU timeline, and the CPU never
            # waits for the (unrelated) simulation kernels queued on the device.
            mapping.unmap(stream=wp.get_stream(src.device).cuda_stream)

    def _render_markers(self) -> None:
        """Render marker overlays without letting them interrupt frame streaming."""
        try:
            self._marker_overlay.render(self._current_visible_env_ids(), num_envs=self._num_envs)
        except Exception as exc:
            if not self._warned_marker_render_failure:
                logger.warning("[ViserVisualizer] Marker rendering failed; continuing frame updates: %s", exc)
                self._warned_marker_render_failure = True
            else:
                logger.debug("[ViserVisualizer] Marker rendering failed: %s", exc)

    def _visible_env_flags(self) -> tuple[bool, ...]:
        """Per-environment visibility from config limits and the GUI isolation toggle."""
        isolated = self._viewer.isolated_environment
        if isolated is not None:
            isolated = min(max(0, isolated), max(0, self._num_envs - 1))
            return tuple(index == isolated for index in range(self._num_envs))
        if self._resolved_env_ids is None:
            return tuple(True for _ in range(self._num_envs))
        visible = set(self._resolved_env_ids)
        return tuple(index in visible for index in range(self._num_envs))

    def _current_visible_env_ids(self) -> list[int] | None:
        """Visible environment ids for marker slicing; ``None`` when all are visible."""
        flags = self._applied_env_flags
        if flags is None or all(flags):
            return None
        return [index for index, flag in enumerate(flags) if flag]

    def _apply_environment_selection(self) -> None:
        """Apply config visibility limits and the GUI's environment isolation choice.

        Environments allowed by ``visible_env_indices`` / ``max_visible_envs``
        are visible by default; enabling "Isolate environment" in the GUI hides
        every environment except the selected one.
        """
        if self._viewer is None or self._num_envs == 0 or self._viewer.warming_up:
            return
        flags = self._visible_env_flags()
        if flags == self._applied_env_flags:
            return
        if self._applied_env_flags is None and all(flags):
            # Everything is already visible on a freshly loaded stage.
            self._applied_env_flags = flags
            return
        stage_count = len(self._env_prim_paths)
        self._viewer.set_environment_visibility(self._env_prim_paths, list(flags[:stage_count]))
        self._applied_env_flags = flags
