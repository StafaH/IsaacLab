"""Shared helpers for rendering MDP2 policies with OVRTX (RTX) on Newton physics, for the deck clips."""

import os
import subprocess
import sys

REPO = os.environ["DECK_REPO"]
sys.path.insert(0, os.path.join(REPO, "scripts/reinforcement_learning/mdp2"))

import ovrtx  # noqa: E402

# What isaaclab_ov.app.OvrtxLauncher does: register the OVRTX USD schemas before any stage is opened.
ovrtx.register_schema_paths()

import torch  # noqa: E402
from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402

MEDIA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")


def dress_scene(cfg, dome=400.0, key=2500.0, key_angle=(0.25, -0.35), ground=(0.06, 0.065, 0.07)):
    """Dark ground, dim dome light plus a warm-white key light from the side, for depth and shadows."""
    if ground is not None and getattr(cfg.scene, "terrain", None) is not None:
        cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=ground, roughness=0.5)
    elif ground is not None and getattr(cfg.scene, "plane", None) is not None:
        cfg.scene.plane.spawn = sim_utils.GroundPlaneCfg(color=ground)
    cfg.scene.sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=dome, color=(0.75, 0.8, 0.9))
    )
    # Distant light rotated about x and z; quaternion (w, x, y, z) composed by hand below.
    import math

    ax, az = key_angle
    qx = (math.cos(ax / 2), math.sin(ax / 2), 0.0, 0.0)
    qz = (math.cos(az / 2), 0.0, 0.0, math.sin(az / 2))
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qx
    q = (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
    cfg.scene.key_light = AssetBaseCfg(
        prim_path="/World/keyLight",
        spawn=sim_utils.DistantLightCfg(intensity=key, color=(1.0, 0.96, 0.9), angle=1.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(q[1], q[2], q[3], q[0])),
    )


def add_camera(
    cfg,
    width=1920,
    height=1080,
    focal=24.0,
    per_env=True,
    prim="/World/DeckCamera",
    settings=None,
    background=(0.015, 0.016, 0.02),
):
    """Add an OVRTX camera: one per world (``per_env``) or a single one at ``prim`` that sees the whole scene."""
    cfg.scene.video_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/VideoCamera" if per_env else prim,
        offset=CameraCfg.OffsetCfg(pos=(3.0, -3.0, 1.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=20.955, clipping_range=(0.05, 200.0)),
        width=width,
        height=height,
        renderer_cfg=OVRTXRendererCfg(),
        render_settings=settings or {},
        background_color=background,
    )
    cfg.capture = False


class Writer:
    """ffmpeg H.264 writer fed with RGB frames; resamples from ``fps_in`` to 30 fps."""

    def __init__(self, path, width, height, fps_in, crf=22, fps_out=30):
        command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"]
        command += ["-s", f"{width}x{height}", "-r", str(fps_in), "-i", "-"]
        command += ["-vf", f"fps={fps_out}", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf)]
        command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame):
        self.process.stdin.write(frame.to(torch.uint8).contiguous().cpu().numpy().tobytes())

    def close(self):
        self.process.stdin.close()
        self.process.wait()


def rgb(camera):
    image = camera.data.output["rgb"]
    return (image.torch if hasattr(image, "torch") else image)[..., :3]
