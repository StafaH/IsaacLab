# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ASV adapter for Isaac Lab's supported runtime benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

_CONFIG_PATH = Path(__file__).parents[1] / "tasks.json"
_REPO_ROOT = Path(__file__).parents[3]
_CONFIG = json.loads(_CONFIG_PATH.read_text())


def _load_case() -> dict:
    """Load the benchmark case selected by the CI matrix."""
    key = os.environ.get("PERF_SMOKE_CASE")
    if not key:
        raise RuntimeError("PERF_SMOKE_CASE must name a case in tasks.json")

    try:
        return next(case for case in _CONFIG["cases"] if case["key"] == key)
    except StopIteration as exc:
        raise RuntimeError(f"unknown performance smoke case: {key}") from exc


def _benchmark_version() -> str:
    """Hash the selected workload so unrelated case changes retain history."""
    key = os.environ.get("PERF_SMOKE_CASE")
    workloads = [case for case in _CONFIG["cases"] if key is None or case["key"] == key]
    payload = json.dumps(workloads, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def track_runtime_seconds_per_frame() -> float:
    """Measure total runtime in seconds per simulated frame."""
    case = _load_case()
    with tempfile.TemporaryDirectory(prefix="isaaclab-perf-") as output_path:
        args = [
            str(_REPO_ROOT / "isaaclab.sh"),
            "-p",
            str(_REPO_ROOT / "scripts/benchmarks/runtime.py"),
            "--task",
            case["task"],
            "--num_envs",
            str(case["num_envs"]),
            "--num_steps",
            str(case["num_steps"]),
            "--warmup_steps",
            str(case["warmup_steps"]),
            "--seed",
            "42",
            "--benchmark_formatter",
            "schema",
            "--output_path",
            output_path,
            *case["overrides"],
        ]
        completed = subprocess.run(args, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=1700, check=False)
        if completed.returncode != 0:
            output = (completed.stdout + completed.stderr)[-4000:]
            raise RuntimeError(f"runtime benchmark failed with code {completed.returncode}:\n{output}")
        output_files = list(Path(output_path).glob("*.json"))
        if len(output_files) != 1:
            raise RuntimeError(f"runtime benchmark wrote {len(output_files)} schema results")
        result = json.loads(output_files[0].read_text())
        fps = float(result["runtime"]["total_fps"]["mean"])

    if not math.isfinite(fps) or fps <= 0.0:
        raise RuntimeError(f"runtime benchmark returned invalid throughput: {fps}")
    return 1.0 / fps


track_runtime_seconds_per_frame.unit = "seconds/frame"
track_runtime_seconds_per_frame.timeout = 1800
track_runtime_seconds_per_frame.number = 1
track_runtime_seconds_per_frame.repeat = 1
track_runtime_seconds_per_frame.rounds = 1
# ASV only compares results with matching benchmark versions. Changing the
# workload therefore starts a new baseline window instead of mixing samples.
track_runtime_seconds_per_frame.version = _benchmark_version()
