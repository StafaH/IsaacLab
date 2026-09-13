# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ASV result comparator."""

import json
from pathlib import Path

import pytest
from compare import BENCHMARK, Measurement, compare, load_measurements


@pytest.fixture
def config() -> dict:
    return {
        "regression_threshold_pct": 5.0,
        "baseline_window": 3,
        "min_baseline_samples": 2,
        "cases": [{"key": "case"}],
    }


def measurement(commit_hash: str, seconds: float, version: str = "v1") -> Measurement:
    return Measurement("perf-case-l40s", commit_hash, seconds, version)


def test_missing_current_result_fails(config):
    verdicts, failed = compare(config, [measurement("base", 0.01)], "head", ["base"])

    assert failed
    assert verdicts[0].status == "FAIL"
    assert verdicts[0].detail == "current result is missing"


def test_insufficient_baseline_warns_without_failing(config):
    verdicts, failed = compare(config, [measurement("head", 0.01), measurement("base", 0.01)], "head", ["base"])

    assert not failed
    assert verdicts[0].status == "WARN"
    assert verdicts[0].samples == 1


def test_regression_fails_against_recent_compatible_median(config):
    results = [
        measurement("head", 0.012),
        measurement("a", 0.010),
        measurement("b", 0.009),
        measurement("c", 0.011),
        measurement("old", 1.0),
        measurement("different-version", 1.0, "v0"),
    ]

    verdicts, failed = compare(config, results, "head", ["a", "different-version", "b", "c", "old"])

    assert failed
    assert verdicts[0].status == "FAIL"
    assert verdicts[0].samples == 3
    assert verdicts[0].baseline_fps == pytest.approx(100.0)
    assert verdicts[0].change_pct == pytest.approx(-100.0 / 6.0)


def test_five_percent_boundary_passes(config):
    results = [measurement("head", 1.0 / 95.0), measurement("a", 0.01), measurement("b", 0.01)]

    verdicts, failed = compare(config, results, "head", ["a", "b"])

    assert not failed
    assert verdicts[0].status == "PASS"
    assert verdicts[0].change_pct == pytest.approx(-5.0)


def test_loads_asv_v2_scalar_result(tmp_path: Path):
    machine_dir = tmp_path / "perf-case-l40s"
    machine_dir.mkdir()
    result = {
        "version": 2,
        "commit_hash": "head",
        "params": {"machine": "perf-case-l40s"},
        "result_columns": ["result", "params", "version", "started_at", "duration"],
        "results": {BENCHMARK: [[0.01], [], "v1", 123]},
    }
    (machine_dir / "head-existing.json").write_text(json.dumps(result))

    assert load_measurements(tmp_path) == [measurement("head", 0.01)]


def test_ignores_failed_asv_result(tmp_path: Path):
    machine_dir = tmp_path / "perf-case-l40s"
    machine_dir.mkdir()
    result = {
        "version": 2,
        "commit_hash": "head",
        "params": {"machine": "perf-case-l40s"},
        "result_columns": ["result", "params", "version"],
        "results": {BENCHMARK: [None, [], "v1"]},
    }
    (machine_dir / "head-existing.json").write_text(json.dumps(result))

    assert load_measurements(tmp_path) == []
