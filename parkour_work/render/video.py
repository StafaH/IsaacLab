"""Stand-in for the video module of the training scripts that renders with OVRTX: the evaluation worlds only."""

import os

import deck_video
from deck_video import add_camera, record  # noqa: F401


def _select(env):
    first = int(os.environ["DECK_FIRST"])
    worlds = [int(world) + first for world in os.environ.get("DECK_PICK", "0").split(",")]
    return worlds, [label for label in os.environ.get("DECK_LABELS", "").split(",") if label]


deck_video.SELECT = _select
