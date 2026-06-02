# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .newton_ray_caster import NewtonRayCaster


@configclass
class NewtonRayCasterCfg(RayCasterCfg):
    """Configuration for the Newton ray-cast sensor.

    Casts rays via BVH-accelerated queries against the Newton model's shape BVH, which must be
    refit after each physics update.
    """

    class_type: type[NewtonRayCaster] | str = "{DIR}.newton_ray_caster:NewtonRayCaster"

    mesh_prim_paths: list[str] = []
    """Unused by the Newton ray-caster."""

    enable_global_world: bool = True
    """Whether to also cast against the global Newton world (index ``-1``). Defaults to True."""
