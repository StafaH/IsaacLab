# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental MDP runtime with Torch and Warp backends and a fixed, graph-capturable step.

An :class:`MdpCfg` declares registered terms and their parameters. :func:`compile_plan` validates it
against a :class:`PhysicsBinding` and a backend (``"warp"`` or ``"torch"``), and the resulting
:class:`ExecutionPlan` is bound to explicit input, state, and output buffers as an :class:`MdpProgram`.
The runtime is independent of the stable :mod:`isaaclab.managers` API.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
