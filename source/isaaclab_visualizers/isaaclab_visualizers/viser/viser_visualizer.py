# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viser-based visualizer backed by isaac_viser (OVRT/X RTX rendering).

The browser viewport is an OVRT/X-rendered image streamed through a Viser
server; no scene geometry is sent to Viser's native WebGL renderer. The
composed USD stage is exported once during :meth:`ViserVisualizer.initialize`,
then rigid-body world transforms from the :class:`~isaaclab.scene_data.SceneDataProvider`
are pushed into the OVRT/X runtime stage every step.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
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
        self._env_prim_paths: list[str] = []
        self._visible_environment: int | None = None
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
        if not self._export_deinstanced_stage(stage, stage_path):
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
            width=self.cfg.width,
            height=self.cfg.height,
            target_fps=self.cfg.target_fps,
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
        num_envs = provider.num_envs
        self._env_prim_paths = [f"/World/envs/env_{index}" for index in range(num_envs)]
        if stage_paths is not None:
            self._env_prim_paths = [path for path in self._env_prim_paths if path in stage_paths]
        if num_envs > 0:
            self._viewer.set_num_environments(max(1, len(self._env_prim_paths)))

        self._transform_paths = self._resolve_transform_paths(provider, stage_paths)
        self._transform_output = SceneDataFormat.Transform()

        viewer_url = f"http://{self.cfg.display_address}:{int(self.cfg.port)}"
        if self.cfg.verbose:
            self._log_viewer_url("ViserVisualizer", viewer_url)
        if self.cfg.open_browser and not webbrowser.open_new_tab(viewer_url):
            logger.info("[ViserVisualizer] Could not auto-open browser tab. Open manually: %s", viewer_url)

        self._log_initialization_table(
            logger=logger,
            title="ViserVisualizer Configuration",
            rows=[
                ("eye", eye),
                ("lookat", lookat),
                ("resolution", f"{self.cfg.width}x{self.cfg.height}"),
                ("render_mode", self.cfg.render_mode),
                ("target_fps", self.cfg.target_fps),
                ("num_envs", num_envs),
                ("tracked_transforms", len(self._transform_paths)),
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
        self._write_transforms()
        self._viewer.update_metrics(step_count=self._step_count, sim_time=self._sim_time)
        rendered = self._viewer.render(delta_time=dt or None)
        if dt == 0.0 and not rendered:
            # Paused and frame-rate limited: yield so the pause loop does not spin hot.
            time.sleep(0.005)

    def close(self) -> None:
        """Close the web viewer and remove the exported temporary stage."""
        if not self._is_initialized:
            return
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
        """OVRT/X renders the USD stage only; Viser marker overlays are not drawn."""
        return False

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

    @staticmethod
    def _export_deinstanced_stage(stage: Any, stage_path: Path) -> bool:
        """Flatten ``stage`` to ``stage_path`` with instancing expanded.

        Cloned environments are typically instanceable, which turns their link
        prims into unwritable instance proxies. Disabling instancing on the
        flattened copy makes every environment's links real prims so per-link
        world transforms can be written into OVRT/X.
        """
        from pxr import Usd

        flattened = Usd.Stage.Open(stage.Flatten())
        instance_paths = [prim.GetPath() for prim in flattened.Traverse() if prim.IsInstance()]
        for path in instance_paths:
            flattened.GetPrimAtPath(path).SetInstanceable(False)
        return bool(flattened.GetRootLayer().Export(str(stage_path)))

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

    def _write_transforms(self) -> None:
        """Copy current backend world transforms into the OVRT/X runtime stage."""
        provider = self._scene_data_provider
        if provider is None or not self._transform_paths or provider.transform_count == 0:
            return
        if not provider.get_transforms(self._transform_output):
            logger.warning("[ViserVisualizer] Scene data provider could not convert transforms.")
            return

        # wp.transformf rows are (x, y, z, qx, qy, qz, qw); reorder to (x, y, z, qw, qx, qy, qz).
        transforms = np.asarray(self._transform_output.transforms.numpy(), dtype=np.float64)
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

    def _apply_environment_selection(self) -> None:
        """Show the environment selected in the browser GUI and hide the others."""
        if len(self._env_prim_paths) <= 1 or self._viewer is None:
            return
        selected = min(max(0, self._viewer.selected_environment), len(self._env_prim_paths) - 1)
        if selected == self._visible_environment:
            return
        self._viewer.show_environment(self._env_prim_paths, selected)
        self._visible_environment = selected
