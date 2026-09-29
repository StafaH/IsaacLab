# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A heterogeneous point-mass population for the experimental MDP runtime.

Two agent types share one population and its per-environment reward and done buffers:

========== ====== =========== ======= ============================================================
Type       Joints Observation Action  Additional terms
========== ====== =========== ======= ============================================================
``slider`` x      2           1       --
``gantry`` x,y,z  6           3       ``action_rate_l2`` reward, interval velocity push, bound on z
========== ====== =========== ======= ============================================================
"""

from __future__ import annotations

from isaaclab_experimental.mdp_runtime import (
    ActionTermCfg,
    EventTermCfg,
    HeterogeneousProgram,
    MdpCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    PointMassPhysics,
    RewardTermCfg,
    TerminationTermCfg,
)

from isaaclab.utils import configclass

PHYSICS_DT = 0.01
"""Physics step [s]."""

DECIMATION = 2
"""Physics steps per control step."""


@configclass
class SliderMdpCfg(MdpCfg):
    """Drive a point mass on one axis to the origin."""

    episode_length_s: float = 2.0
    actions: dict[str, ActionTermCfg] = {"effort": ActionTermCfg(term="joint_effort", scale=5.0, clip=(-5.0, 5.0))}
    observations: dict[str, ObservationGroupCfg] = {
        "policy": ObservationGroupCfg(
            terms={
                "pos": ObservationTermCfg(term="joint_pos_rel"),
                "vel": ObservationTermCfg(term="joint_vel_rel", scale=0.5),
            }
        )
    }
    rewards: dict[str, RewardTermCfg] = {
        "alive": RewardTermCfg(term="is_alive", weight=1.0),
        "position": RewardTermCfg(term="joint_pos_target_l2", weight=-1.0, params={"target": 0.0}),
    }
    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(term="time_out", time_out=True),
        "out_of_bounds": TerminationTermCfg(term="joint_pos_out_of_manual_limit", params={"bounds": (-2.0, 2.0)}),
    }
    events: dict[str, EventTermCfg] = {
        "reset": EventTermCfg(
            term="reset_joints_by_offset", params={"position_range": (-1.0, 1.0), "velocity_range": (-0.5, 0.5)}
        )
    }


@configclass
class GantryMdpCfg(MdpCfg):
    """Hold a three-axis gantry at the origin under random velocity pushes."""

    episode_length_s: float = 3.0
    actions: dict[str, ActionTermCfg] = {"effort": ActionTermCfg(term="joint_effort", scale=5.0, clip=(-5.0, 5.0))}
    observations: dict[str, ObservationGroupCfg] = {
        "policy": ObservationGroupCfg(
            terms={
                "pos": ObservationTermCfg(term="joint_pos_rel"),
                "vel": ObservationTermCfg(term="joint_vel_rel", scale=0.5, clip=(-4.0, 4.0)),
            }
        )
    }
    rewards: dict[str, RewardTermCfg] = {
        "alive": RewardTermCfg(term="is_alive", weight=1.0),
        "position": RewardTermCfg(term="joint_pos_target_l2", weight=-1.0, params={"target": 0.0}),
        "velocity": RewardTermCfg(term="joint_vel_l2", weight=-0.05),
        "action_rate": RewardTermCfg(term="action_rate_l2", weight=-0.01),
    }
    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(term="time_out", time_out=True),
        "out_of_bounds": TerminationTermCfg(term="joint_pos_out_of_manual_limit", params={"bounds": (-2.0, 2.0)}),
        "height": TerminationTermCfg(
            term="joint_pos_out_of_manual_limit", params={"bounds": (-0.5, 0.5), "joints": ["z"]}
        ),
    }
    events: dict[str, EventTermCfg] = {
        "reset": EventTermCfg(
            term="reset_joints_by_offset", params={"position_range": (-0.5, 0.5), "velocity_range": (-0.2, 0.2)}
        ),
        "push": EventTermCfg(
            term="push_joints_by_velocity",
            mode="interval",
            interval_range_s=(0.5, 1.0),
            params={"velocity_range": (-0.5, 0.5)},
        ),
    }


def make_point_mass_population(
    num_sliders: int, num_gantries: int, backend: str = "warp", device: str = "cuda:0"
) -> HeterogeneousProgram:
    """Build the two-type population: sliders occupy rows ``[0, num_sliders)``, gantries the rest."""
    return HeterogeneousProgram(
        {
            "slider": (SliderMdpCfg(seed=1), PointMassPhysics(num_sliders, ["x"], PHYSICS_DT, DECIMATION, device)),
            "gantry": (
                GantryMdpCfg(seed=2),
                PointMassPhysics(num_gantries, ["x", "y", "z"], PHYSICS_DT, DECIMATION, device),
            ),
        },
        backend=backend,
    )
