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

    label: str | None = "Isaac Lab Simulation"
    """Optional label shown in the viewer page title."""

    share: bool = False
    """Whether to request a public share URL from viser."""

    width: int = 2560
    """OVRT/X render width in pixels.

    The browser decodes one JPEG per streamed frame on its main thread, so
    ``width * height * target_fps`` bounds client responsiveness: 2560x1440 at
    60 FPS suits desktop browsers; 3840x2160 renders at full rate server-side
    but can saturate client decoding and make interaction laggy.
    """

    height: int = 1440
    """OVRT/X render height in pixels. See :attr:`width` for the client-side trade-off."""

    target_fps: float = 60.0
    """Maximum browser stream update frequency.

    Lower this (e.g. to 30) if interaction feels laggy on remote or low-power
    clients; see :attr:`width`.
    """

    jpeg_quality: int = 90
    """JPEG quality of the browser stream (1-100)."""

    render_mode: str = "RealTimePathTracing"
    """OVRT/X render mode: ``"RealTimePathTracing"``, ``"PathTracing"``, or ``"Minimal"``."""
