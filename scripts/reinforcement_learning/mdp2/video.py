# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a policy in an MDP2 environment with the Newton Warp renderer, one tile per world, as an MP4."""

import subprocess

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

EYE = (2.6, -3.0, 0.5)
"""Camera position relative to the root of the world's robot [m]. The camera looks at the root."""


def add_camera(cfg, width: int = 480, height: int = 480, eye: tuple[float, float, float] = EYE) -> None:
    """Add a camera to every world of an environment config, and step it without capture, which rendering needs."""
    from isaaclab_newton.renderers import NewtonWarpRendererCfg  # noqa: PLC0415

    cfg.scene.video_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/VideoCamera",
        offset=CameraCfg.OffsetCfg(pos=eye, rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.1, 100.0)),
        width=width,
        height=height,
        renderer_cfg=NewtonWarpRendererCfg(enable_shadows=True),
    )
    cfg.capture = False


def record(
    env,
    policy,
    path: str,
    steps: int,
    columns: int,
    eye: tuple[float, float, float] = EYE,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    roots=None,
) -> None:
    """Step ``env`` with ``policy(observations)`` and write its worlds as a grid of ``columns`` tiles per row.

    The camera looks at ``center``, relative to the root of the robot [m], or to the positions that ``roots()``
    returns for the worlds, for scenes with other robots.
    """
    camera = env.scene["video_camera"]
    if roots is None:
        roots = lambda: env.scene["robot"].data.root_link_pose_w.torch[:, :3]  # noqa: E731
    offset = torch.tensor(eye, device=env.device)
    obs = env.reset()[0]["policy"]
    center = torch.tensor(center, device=env.device)
    target = roots() + center
    writer = None
    with torch.no_grad():
        for _ in range(steps):
            # The camera follows the root smoothly, so that its bobbing stays visible.
            target += 0.2 * (roots() + center - target)
            camera.set_world_poses_from_view(target + offset, target)
            obs = env.step(policy(obs))[0]["policy"]
            image = camera.data.output["rgb"]
            image = (image.torch if hasattr(image, "torch") else image)[..., :3]
            rows = image.reshape(-1, columns, *image.shape[1:]).permute(0, 2, 1, 3, 4)
            frame = rows.reshape(rows.shape[0] * image.shape[1], columns * image.shape[2], 3)
            if writer is None:
                size = f"{frame.shape[1]}x{frame.shape[0]}"
                command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size]
                command += ["-r", str(round(1.0 / env.step_dt)), "-i", "-", "-pix_fmt", "yuv420p", "-crf", "26", path]
                writer = subprocess.Popen(command, stdin=subprocess.PIPE)
            writer.stdin.write(frame.to(torch.uint8).cpu().numpy().tobytes())
    writer.stdin.close()
    writer.wait()
