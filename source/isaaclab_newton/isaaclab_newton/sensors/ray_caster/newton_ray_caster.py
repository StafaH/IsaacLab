# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# pyright: reportInvalidTypeForm=none, reportPrivateUsage=none
from typing import TYPE_CHECKING

import newton
import numpy as np
import warp as wp

from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.kernels import apply_z_drift_kernel

from isaaclab_newton.physics import NewtonManager

from .ray_caster import _NewtonRayCasterMixin

if TYPE_CHECKING:
    from .newton_ray_caster_cfg import NewtonRayCasterCfg


@wp.kernel(enable_backward=False)
def _resolve_ray_hits_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    ray_starts_w: wp.array2d(dtype=wp.vec3f),
    ray_directions_w: wp.array2d(dtype=wp.vec3f),
    ray_dist: wp.array(dtype=wp.float32),
    num_rays: int,
    max_dist: wp.float32,
    # output
    ray_hits_w: wp.array2d(dtype=wp.vec3f),
):
    """Convert per-ray hit distances into world-frame hit positions.

    Args:
        env_mask: Mask of environments to update. Shape is (num_envs,).
        ray_starts_w: World-frame ray starts [m]. Shape is (num_envs, num_rays).
        ray_directions_w: World-frame ray directions (unit). Shape is (num_envs, num_rays).
        ray_dist: Flat hit distances [m]. Shape is (num_envs * num_rays,).
        num_rays: Rays per sensor, used to flatten the (env, ray) index.
        max_dist: Maximum hit distance [m]; beyond this counts as a miss.
        ray_hits_w: Output hit positions [m]; ``inf`` on miss. Shape is (num_envs, num_rays).
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    dist = ray_dist[env * num_rays + ray]
    if dist < 0.0 or dist > max_dist:
        ray_hits_w[env, ray] = wp.vec3f(wp.inf, wp.inf, wp.inf)
    else:
        ray_hits_w[env, ray] = ray_starts_w[env, ray] + dist * ray_directions_w[env, ray]


class NewtonRayCaster(_NewtonRayCasterMixin, BaseRayCaster):
    """Newton ray-caster that casts against the live collision model.

    Rays are placed in the world frame via the mesh ray-caster's site-based pose tracking, then
    resolved with :func:`newton.intersect_ray` against each environment's collision shapes (and,
    optionally, the global world) through the Newton model's BVH. This handles dynamic geometry
    without parsing or tracking individual meshes.
    """

    cfg: NewtonRayCasterCfg
    """The configuration parameters."""

    def _initialize_warp_meshes(self):
        """Skip USD mesh parsing; the Newton collision model is queried directly."""
        pass

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        num_envs, num_rays = self._view_count, self.num_rays
        # Flat (zero-copy) ray views for newton.intersect_ray.
        self._ray_starts_w_flat = self._ray_starts_w.flatten()
        self._ray_directions_w_flat = self._ray_directions_w.flatten()
        self._ray_dist = wp.empty(num_envs * num_rays, dtype=wp.float32, device=self._device)
        # Each environment maps to its own Newton world (env index == world index).
        ray_worlds = np.repeat(np.arange(num_envs, dtype=np.int32), num_rays)
        self._ray_worlds = wp.array(ray_worlds, dtype=wp.int32, device=self._device)

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        self._update_ray_infos(env_mask)

        model = NewtonManager._model
        if model is None:
            raise RuntimeError("Newton model is not initialized.")

        # Cast rays against the Newton collision model; misses are encoded as -1 distance.
        newton.intersect_ray(
            model,
            ray_origins=self._ray_starts_w_flat,
            ray_directions=self._ray_directions_w_flat,
            ray_worlds=self._ray_worlds,
            enable_global_world=self.cfg.enable_global_world,
            out_dist=self._ray_dist,
        )

        wp.launch(
            _resolve_ray_hits_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                env_mask,
                self._ray_starts_w,
                self._ray_directions_w,
                self._ray_dist,
                int(self.num_rays),
                float(self.cfg.max_distance),
            ],
            outputs=[self._data._ray_hits_w],
            device=self._device,
        )

        # Apply vertical drift to ray hits.
        wp.launch(
            apply_z_drift_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, self.ray_cast_drift.warp, self._data._ray_hits_w],
            device=self._device,
        )
