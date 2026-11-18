"""Drop-in replacement for ``scripts/reinforcement_learning/mdp2/video.py`` that renders with OVRTX (RTX).

Physics stays Newton (MJWarp); only the video camera's renderer changes. Each world gets its own follow camera,
rendered as one panel of a 1920x1080 frame. Deck options come from environment variables:

    DECK_W, DECK_H       frame size (1920x1080)
    DECK_COLS            panels per row (default: up to 3)
    DECK_EYE             "x,y,z" camera offset from the followed point [m] (default: the script's eye)
    DECK_CENTER          "x,y,z" followed point relative to the root [m] (default: the script's center)
    DECK_FOCAL           focal length [mm] (24)
    DECK_STEPS           control steps to record (default: the script's)
    DECK_SKIP            first recorded steps to drop (renderer warm-up, 8)
    DECK_LABELS          comma-separated panel labels
    DECK_GROUND          "r,g,b" ground tint (dark grey)
    DECK_DOME, DECK_KEY  dome and key light intensities
    DECK_CRF             H.264 CRF (22)
    DECK_SMOOTH          follow-camera smoothing per step (0.08)
"""

import math
import os
import subprocess

import deckcam  # noqa: F401  (registers the OVRTX schemas first)
import torch
from PIL import Image, ImageDraw, ImageFont

EYE = (2.6, -3.0, 0.5)
ON_STEP = []
SELECT = None
"""Optional ``SELECT(env) -> (worlds, labels)`` run after the reset: the worlds to show as panels."""
"""Callables run after every recorded step, e.g. to move goal markers."""


def _vec(name, default):
    value = os.environ.get(name)
    return tuple(float(v) for v in value.split(",")) if value else default


def _layout(n):
    n = int(os.environ.get("DECK_SHOW", n))
    columns = int(os.environ.get("DECK_COLS", min(n, 3)))
    rows = math.ceil(n / columns)
    width, height = int(os.environ.get("DECK_W", 1920)), int(os.environ.get("DECK_H", 1080))
    return columns, rows, width // columns, height // rows


def add_camera(cfg, width=None, height=None, eye=EYE):
    """Add an OVRTX follow camera to every world, sized as one panel of the deck frame."""
    _, _, panel_w, panel_h = _layout(cfg.scene.num_envs)
    deckcam.dress_scene(
        cfg,
        dome=float(os.environ.get("DECK_DOME", 400.0)),
        key=float(os.environ.get("DECK_KEY", 2500.0)),
        ground=_vec("DECK_GROUND", (0.03, 0.032, 0.036)),
    )
    if os.environ.get("DECK_HERO"):
        # One camera outside the worlds, which sees all of them, for a wide shot.
        if os.environ.get("DECK_SPACING"):
            cfg.scene.env_spacing = float(os.environ["DECK_SPACING"])
        width, height = int(os.environ.get("DECK_W", 1920)), int(os.environ.get("DECK_H", 1080))
        deckcam.add_camera(cfg, width, height, focal=float(os.environ.get("DECK_FOCAL", 24.0)), per_env=False)
        return
    deckcam.add_camera(
        cfg,
        panel_w,
        panel_h,
        focal=float(os.environ.get("DECK_FOCAL", 24.0)),
        background=_vec("DECK_BG", (0.015, 0.016, 0.02)),
    )


def _font(size):
    for path in (
        "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _label(frame, labels, columns, panel_w, panel_h):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = _font(max(22, panel_h // 26))
    for k, text in enumerate(labels):
        x, y = (k % columns) * panel_w + 28, (k // columns) * panel_h + panel_h - 28 - font.size
        draw.rectangle((x, y - 2, x + 5, y + font.size + 4), fill=(118, 185, 0))
        draw.text((x + 16, y - font.size // 8), text[:1].upper() + text[1:], font=font, fill=(235, 235, 235))
    # Thin separators between panels.
    for c in range(1, columns):
        draw.rectangle((c * panel_w - 1, 0, c * panel_w, image.height), fill=(10, 10, 12))
    for r in range(1, image.height // panel_h):
        draw.rectangle((0, r * panel_h - 1, image.width, r * panel_h), fill=(10, 10, 12))
    return image


def record(env, policy, path, steps, columns=None, eye=EYE, center=(0.0, 0.0, 0.0), roots=None):
    """Step ``env`` with ``policy`` and write every world's follow camera as panels of one 1080p frame."""
    if os.environ.get("DECK_HERO"):
        return record_hero(env, policy, path, steps, roots)
    camera = env.scene["video_camera"]
    columns, rows, panel_w, panel_h = _layout(env.num_envs)
    n = min(env.num_envs, rows * columns)
    eye = _vec("DECK_EYE", tuple(eye))
    center = _vec("DECK_CENTER", tuple(center))
    steps = int(os.environ.get("DECK_STEPS", steps))
    skip = int(os.environ.get("DECK_SKIP", 8))
    smooth = float(os.environ.get("DECK_SMOOTH", 0.08))
    labels = [s for s in os.environ.get("DECK_LABELS", "").split(",") if s]
    if roots is None:
        roots = lambda: env.scene["robot"].data.root_link_pose_w.torch[:, :3]  # noqa: E731
    offset = torch.tensor(eye, device=env.device)
    center = torch.tensor(center, device=env.device)
    obs = env.reset()[0]["policy"]
    worlds = torch.arange(n, device=env.device)
    if SELECT is not None:
        chosen, labels = SELECT(env)
        worlds = torch.tensor(chosen[:n], device=env.device)
    target = roots() + center
    fps = 1.0 / env.step_dt
    width, height = columns * panel_w, rows * panel_h
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}"]
    command += ["-r", f"{fps:.6f}", "-i", "-", "-vf", "fps=30", "-c:v", "libx264", "-preset", "slow"]
    command += ["-crf", os.environ.get("DECK_CRF", "22"), "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    poster_step = skip + (steps - skip) // 2
    with torch.no_grad():
        for step in range(steps):
            target += smooth * (roots() + center - target)
            camera.set_world_poses_from_view(target + offset, target)
            obs = env.step(policy(obs))[0]["policy"]
            for hook in ON_STEP:
                hook()
            if step < skip:
                continue
            image = deckcam.rgb(camera)
            canvas = torch.zeros(rows * columns, panel_h, panel_w, 3, dtype=torch.uint8, device=image.device)
            canvas[: len(worlds)] = image[worlds]
            grid = canvas.reshape(rows, columns, panel_h, panel_w, 3).permute(0, 2, 1, 3, 4)
            frame = grid.reshape(height, width, 3).cpu().numpy()
            if labels or columns > 1 or rows > 1:
                frame = _label(frame, labels() if callable(labels) else labels, columns, panel_w, panel_h)
            else:
                frame = Image.fromarray(frame)
            if step == poster_step:
                frame.save(os.path.splitext(path)[0] + ".jpg", quality=92)
            writer.stdin.write(frame.tobytes())
    writer.stdin.close()
    writer.wait()
    print(f"DECK wrote {path} ({(steps - skip) * env.step_dt:.1f} s, {width}x{height})", flush=True)


def record_hero(env, policy, path, steps, roots=None):
    """Wide shot of every world from one camera on a slow orbit around the crowd.

    DECK_ORBIT "radius,height,start_deg,sweep_deg" (default 14,4.5,-60,35); DECK_LOOK_Z height looked at [m].
    """
    camera = env.scene["video_camera"]
    radius, height, start, sweep = _vec("DECK_ORBIT", (14.0, 4.5, -60.0, 35.0))
    look_z = float(os.environ.get("DECK_LOOK_Z", 0.6))
    steps = int(os.environ.get("DECK_STEPS", steps))
    skip = int(os.environ.get("DECK_SKIP", 8))
    if roots is None:
        roots = lambda: env.scene["robot"].data.root_link_pose_w.torch[:, :3]  # noqa: E731
    obs = env.reset()[0]["policy"]
    center = roots().mean(0)
    center[2] = look_z
    width, height_px = int(os.environ.get("DECK_W", 1920)), int(os.environ.get("DECK_H", 1080))
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height_px}",
    ]
    command += ["-r", f"{1.0 / env.step_dt:.6f}", "-i", "-", "-vf", "fps=30", "-c:v", "libx264", "-preset", "slow"]
    command += ["-crf", os.environ.get("DECK_CRF", "22"), "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    poster_step = skip + (steps - skip) // 2
    with torch.no_grad():
        for step in range(steps):
            # Ease in and out along the arc.
            u = 0.5 - 0.5 * math.cos(math.pi * step / max(1, steps - 1))
            angle = math.radians(start + sweep * u)
            eye = center + torch.tensor(
                [radius * math.cos(angle), radius * math.sin(angle), height - look_z], device=env.device
            )
            camera.set_world_poses_from_view(eye[None], center[None])
            obs = env.step(policy(obs))[0]["policy"]
            for hook in ON_STEP:
                hook()
            if step < skip:
                continue
            frame = Image.fromarray(deckcam.rgb(camera)[0].cpu().numpy())
            if step == poster_step:
                frame.save(os.path.splitext(path)[0] + ".jpg", quality=92)
            writer.stdin.write(frame.tobytes())
    writer.stdin.close()
    writer.wait()
    print(f"DECK wrote {path} ({(steps - skip) * env.step_dt:.1f} s, hero {env.num_envs} worlds)", flush=True)
