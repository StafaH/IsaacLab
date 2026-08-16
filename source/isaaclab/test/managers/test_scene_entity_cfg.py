# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp import (
    illegal_contact,
    joint_acc_l2,
    joint_deviation_l1,
    joint_pos_limits,
    joint_torques_l2,
    undesired_contacts,
)
from isaaclab.managers import SceneEntityCfg

pytestmark = pytest.mark.unit


class MockEntity:
    body_names = ["body_a", "body_b", "body_c", "body_d"]
    joint_names = ["joint_a", "joint_b", "joint_c", "joint_d", "joint_e"]
    num_bodies = len(body_names)
    num_joints = len(joint_names)

    def find_bodies(self, names, preserve_order=False):
        raise AssertionError("Explicit body IDs should not require name resolution.")


class MockScene(dict):
    def __init__(self, entities):
        super().__init__(entities)
        self.device = "cpu"
        self.sensors = {}


def test_resolve_caches_body_indices_without_changing_public_fields_or_serialization():
    """Resolved body selections have reusable device indices while config fields remain lists."""
    scene = MockScene({"robot": MockEntity()})
    cfg = SceneEntityCfg("robot", body_ids=[3, 1], preserve_order=True)
    cfg.resolve(scene)

    assert cfg.body_ids == [3, 1]
    assert cfg.body_ids_torch.dtype == torch.long
    assert cfg.body_ids_torch.device.type == "cpu"
    assert cfg.body_ids_torch.data_ptr() == cfg.body_ids_torch.data_ptr()
    torch.testing.assert_close(cfg.body_ids_torch, torch.tensor([3, 1]))
    serialized = cfg.to_dict()
    assert serialized["body_ids"] == [3, 1]
    assert "__body_ids_torch" not in serialized
    assert "__body_ids_source" not in serialized


def test_resolve_preserves_body_slice_indexing():
    """Full body selections retain basic slice indexing instead of becoming advanced gathers."""
    scene = MockScene({"robot": MockEntity()})
    cfg = SceneEntityCfg("robot")
    cfg.resolve(scene)
    assert cfg.body_ids_torch == slice(None)


def test_resolved_selector_cache_falls_back_after_public_field_mutation():
    """Mutating public selectors after resolution retains their existing runtime behavior."""
    scene = MockScene({"robot": MockEntity()})
    cfg = SceneEntityCfg("robot", body_ids=[3, 1], joint_ids=[4, 1])
    cfg.resolve(scene)

    cfg.body_ids[:] = [2, 0]
    cfg.joint_ids[:] = [3, 0]

    assert cfg.body_ids_torch == [2, 0]
    assert cfg.joint_mask_torch is None


def test_contact_hot_terms_match_public_list_indexing():
    """Termination and reward contact terms preserve results with cached body indices."""
    contact_forces = torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 6.0, 0.0]]],
        ]
    )
    sensor = MockEntity()
    sensor.data = SimpleNamespace(net_forces_w_history=SimpleNamespace(torch=contact_forces))
    scene = MockScene({"contact_forces": sensor})
    scene.sensors["contact_forces"] = sensor
    env = SimpleNamespace(scene=scene)
    sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[3, 1])
    sensor_cfg.resolve(scene)

    selected_forces = contact_forces[:, :, [3, 1]]
    expected_contacts = torch.max(torch.linalg.norm(selected_forces, dim=-1), dim=1)[0] > 4.0
    torch.testing.assert_close(
        illegal_contact(env, threshold=4.0, sensor_cfg=sensor_cfg), torch.any(expected_contacts, dim=1)
    )
    torch.testing.assert_close(
        undesired_contacts(env, threshold=4.0, sensor_cfg=sensor_cfg), torch.sum(expected_contacts, dim=1)
    )


def _make_joint_env() -> tuple[SimpleNamespace, MockScene]:
    asset = MockEntity()
    asset.data = SimpleNamespace(
        applied_torque=SimpleNamespace(torch=torch.tensor([[1.0, 2.0, float("nan"), 4.0, 5.0]])),
        joint_acc=SimpleNamespace(torch=torch.tensor([[0.5, -2.0, float("nan"), 3.0, -4.0]])),
        joint_pos=SimpleNamespace(torch=torch.tensor([[0.0, -2.0, float("nan"), 4.0, 8.0]])),
        default_joint_pos=SimpleNamespace(torch=torch.tensor([[0.0, -1.0, 0.0, 2.0, 5.0]])),
        soft_joint_pos_limits=SimpleNamespace(
            torch=torch.tensor([[[-1.0, 1.0], [-1.5, 1.5], [-2.0, 2.0], [-3.0, 3.0], [-4.0, 4.0]]])
        ),
    )
    scene = MockScene({"robot": asset})
    return SimpleNamespace(scene=scene), scene


def _joint_reward_values(env, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, ...]:
    return (
        joint_deviation_l1(env, asset_cfg),
        joint_pos_limits(env, asset_cfg),
        joint_torques_l2(env, asset_cfg),
        joint_acc_l2(env, asset_cfg),
    )


def test_joint_reduction_mask_preserves_noncontiguous_selection_and_excludes_nan():
    """A noncontiguous mask matches compact indexing and excludes an unselected NaN."""
    env, scene = _make_joint_env()
    cfg = SceneEntityCfg("robot", joint_ids=[4, 1, 3], preserve_order=True)
    cfg.resolve(scene)

    assert cfg.joint_ids == [4, 1, 3]
    torch.testing.assert_close(cfg.joint_mask_torch, torch.tensor([False, True, False, True, True]))
    assert "__joint_mask_torch" not in cfg.to_dict()
    assert "__joint_ids_source" not in cfg.to_dict()

    data = scene["robot"].data
    ids = cfg.joint_ids
    pos = data.joint_pos.torch[:, ids]
    limits = data.soft_joint_pos_limits.torch[:, ids]
    limit_error = -(pos - limits[..., 0]).clip(max=0.0) + (pos - limits[..., 1]).clip(min=0.0)
    expected = (
        torch.sum(torch.abs(pos - data.default_joint_pos.torch[:, ids]), dim=1),
        torch.sum(limit_error, dim=1),
        torch.sum(torch.square(data.applied_torque.torch[:, ids]), dim=1),
        torch.sum(torch.square(data.joint_acc.torch[:, ids]), dim=1),
    )
    actual = _joint_reward_values(env, cfg)
    for actual_value, expected_value in zip(actual, expected):
        assert torch.isfinite(actual_value).all()
        torch.testing.assert_close(actual_value, expected_value)

    reordered_cfg = SceneEntityCfg("robot", joint_ids=[1, 3, 4], preserve_order=True)
    reordered_cfg.resolve(scene)
    for reordered_value, actual_value in zip(_joint_reward_values(env, reordered_cfg), actual):
        torch.testing.assert_close(reordered_value, actual_value)


def test_joint_reduction_mask_preserves_slice_all():
    """The all-joint slice retains the existing full-shape reduction results."""
    env, scene = _make_joint_env()
    data = scene["robot"].data
    for value in (data.applied_torque.torch, data.joint_acc.torch, data.joint_pos.torch):
        value[:, 2] = 0.25
    cfg = SceneEntityCfg("robot")
    cfg.resolve(scene)

    assert torch.all(cfg.joint_mask_torch)
    pos = data.joint_pos.torch
    limits = data.soft_joint_pos_limits.torch
    limit_error = -(pos - limits[..., 0]).clip(max=0.0) + (pos - limits[..., 1]).clip(min=0.0)
    expected = (
        torch.sum(torch.abs(pos - data.default_joint_pos.torch), dim=1),
        torch.sum(limit_error, dim=1),
        torch.sum(torch.square(data.applied_torque.torch), dim=1),
        torch.sum(torch.square(data.joint_acc.torch), dim=1),
    )
    for actual_value, expected_value in zip(_joint_reward_values(env, cfg), expected):
        torch.testing.assert_close(actual_value, expected_value)


@pytest.mark.parametrize("joint_ids", ([4, 1, 4], [4, 1, -1]))
def test_joint_reduction_falls_back_for_duplicate_selector(joint_ids):
    """Duplicate explicit IDs retain advanced-index multiplicity instead of collapsing to a mask."""
    env, scene = _make_joint_env()
    cfg = SceneEntityCfg("robot", joint_ids=joint_ids, preserve_order=True)
    cfg.resolve(scene)

    assert cfg.joint_mask_torch is None
    data = scene["robot"].data
    ids = cfg.joint_ids
    pos = data.joint_pos.torch[:, ids]
    limits = data.soft_joint_pos_limits.torch[:, ids]
    limit_error = -(pos - limits[..., 0]).clip(max=0.0) + (pos - limits[..., 1]).clip(min=0.0)
    expected = (
        torch.sum(torch.abs(pos - data.default_joint_pos.torch[:, ids]), dim=1),
        torch.sum(limit_error, dim=1),
        torch.sum(torch.square(data.applied_torque.torch[:, ids]), dim=1),
        torch.sum(torch.square(data.joint_acc.torch[:, ids]), dim=1),
    )
    for actual_value, expected_value in zip(_joint_reward_values(env, cfg), expected):
        torch.testing.assert_close(actual_value, expected_value)
