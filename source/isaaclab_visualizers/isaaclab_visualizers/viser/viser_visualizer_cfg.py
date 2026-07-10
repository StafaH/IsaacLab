# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Viser visualizer."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg


@configclass
class ViserVisualizerCfg(VisualizerCfg):
    """Configuration for the Viser visualizer (OVRT/X frames streamed to the browser)."""

    visualizer_type: str = "viser"
    """Type identifier for Viser visualizer."""

    port: int = 8080
    """Port of the local viser web server."""

    bind_address: str = "0.0.0.0"
    """Host/interface for the Viser server to bind.

    Use ``"0.0.0.0"`` to listen on all interfaces for remote access.
    """

    display_address: str = "localhost"
    """Host name or IP address shown in the printed browser URL.

    For remote access, set this to the hostname/IP reachable from your browser.
    """

    open_browser: bool = False
    """Whether to attempt opening the viser web viewer URL in a browser.

    The viewer URL is always logged during initialization. Set this to ``True`` to auto-launch it.
    """

    verbose: bool = True
    """Whether to print viewer server startup information."""

    width: int = 1280
    """OVRT/X render width in pixels."""

    height: int = 720
    """OVRT/X render height in pixels."""

    target_fps: float = 30.0
    """Maximum browser stream update frequency."""

    render_mode: str = "RealTimePathTracing"
    """OVRT/X render mode: ``"RealTimePathTracing"``, ``"PathTracing"``, or ``"Minimal"``."""
