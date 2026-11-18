# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain of the parkour task: the boxes that a clip was recorded on."""

from __future__ import annotations

import json
from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass


def clip_boxes_terrain(difficulty: float, cfg: ClipBoxesTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Return a ground plane with the boxes of a level of a clip around its centre, and the centre."""
    level = min(int(difficulty * len(cfg.files)), len(cfg.files) - 1)
    with open(cfg.files[level]) as file:
        boxes = json.load(file)["boxes"]
    centre = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])
    meshes = [make_plane(cfg.size, 0.0, center_zero=False)]
    for box in boxes:
        transform = trimesh.transformations.rotation_matrix(box["yaw"], [0.0, 0.0, 1.0])
        transform[:3, 3] = centre + np.array(box["centre"])
        meshes.append(trimesh.creation.box(box["size"], transform))
    return meshes, centre


@configclass
class ClipBoxesTerrainCfg(SubTerrainBaseCfg):
    """A ground plane with the boxes of a clip, which a row of the terrain scales in height."""

    function = clip_boxes_terrain
    files: tuple[str, ...] = MISSING
    """Files with the boxes of the clip, one per level, as ``convert_omniretarget.py`` writes them."""
