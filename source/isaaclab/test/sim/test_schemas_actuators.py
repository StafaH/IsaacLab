# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton actuator-schema checkpoint preparation."""

import io
import json
import os
from pathlib import Path

import pytest
import torch

from isaaclab.sim.schemas import schemas_actuators

pytestmark = pytest.mark.unit


def _checkpoint_bytes(checkpoint_kind: str) -> bytes:
    """Create a small Torch checkpoint payload."""
    buffer = io.BytesIO()
    model = torch.nn.Linear(2, 1)
    if checkpoint_kind == "torchscript":
        torch.jit.save(
            torch.jit.script(model),
            buffer,
            _extra_files={"metadata.json": json.dumps({"existing": 1, "overridden": "old"})},
        )
    else:
        torch.save({"model": model, "metadata": {"existing": 1, "overridden": "old"}}, buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("path_kind", ["local", "https", "omniverse"])
@pytest.mark.parametrize("checkpoint_kind", ["torchscript", "dict"])
def test_resave_checkpoint_uses_asset_resolver(
    path_kind: str,
    checkpoint_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resolve local and remote checkpoints through the shared asset reader."""
    payload = _checkpoint_bytes(checkpoint_kind)
    if path_kind == "local":
        checkpoint_path = str(tmp_path / "checkpoint.pt")
        (tmp_path / "checkpoint.pt").write_bytes(payload)
    elif path_kind == "https":
        checkpoint_path = "https://assets.example.com/checkpoint.pt"
    else:
        checkpoint_path = "omniverse://assets.example.com/checkpoint.pt"

    asset_reader = schemas_actuators.read_file
    resolved_paths: list[str] = []

    def fake_read_file(path: str) -> io.BytesIO:
        resolved_paths.append(path)
        if path_kind == "local":
            return asset_reader(path)
        return io.BytesIO(payload)

    monkeypatch.setattr(schemas_actuators, "read_file", fake_read_file)

    patched_path = schemas_actuators._resave_checkpoint_with_metadata(
        checkpoint_path,
        {"added": 2, "overridden": "new"},
    )
    try:
        assert resolved_paths == [checkpoint_path]
        if checkpoint_kind == "torchscript":
            extra_files = {"metadata.json": ""}
            torch.jit.load(patched_path, _extra_files=extra_files)
            metadata = json.loads(extra_files["metadata.json"])
        else:
            metadata = torch.load(patched_path, weights_only=False)["metadata"]
        assert metadata == {"existing": 1, "added": 2, "overridden": "new"}
    finally:
        os.unlink(patched_path)
