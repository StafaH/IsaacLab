# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare ASV results with recent compatible results from base ancestors."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

BENCHMARK = "benchmark_runtime.track_runtime_seconds_per_frame"


@dataclass(frozen=True)
class Measurement:
    """One ASV measurement."""

    machine: str
    commit_hash: str
    value: float
    version: str


@dataclass(frozen=True)
class Verdict:
    """Comparison result for one benchmark case."""

    case: str
    status: str
    current_fps: float | None
    baseline_fps: float | None
    change_pct: float | None
    samples: int
    detail: str


def load_config(path: Path) -> dict:
    """Load and validate the small performance-smoke configuration."""
    config = json.loads(path.read_text())
    required = {"regression_threshold_pct", "baseline_window", "min_baseline_samples", "cases"}
    if not required <= config.keys():
        raise ValueError(f"{path} is missing keys: {sorted(required - config.keys())}")
    if not 0.0 < float(config["regression_threshold_pct"]) < 100.0:
        raise ValueError("regression_threshold_pct must be between 0 and 100")
    if not 1 <= int(config["min_baseline_samples"]) <= int(config["baseline_window"]):
        raise ValueError("min_baseline_samples must be between 1 and baseline_window")

    keys = [case.get("key") for case in config["cases"]]
    if not keys or any(not isinstance(key, str) or not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("cases must have unique, non-empty keys")
    for case in config["cases"]:
        for field in ("task", "num_envs", "num_steps", "warmup_steps", "timeout_minutes", "overrides"):
            if field not in case:
                raise ValueError(f"case {case['key']} is missing {field}")
    return config


def load_measurements(results_dir: Path) -> list[Measurement]:
    """Read the selected scalar benchmark from ASV result files."""
    measurements = []
    for path in sorted(results_dir.glob("*/*.json")):
        if path.name == "machine.json":
            continue
        data = json.loads(path.read_text())
        if data.get("version") != 2 or BENCHMARK not in data.get("results", {}):
            continue

        columns = data.get("result_columns", [])
        row = data["results"][BENCHMARK]
        if len(row) > len(columns):
            raise ValueError(f"invalid ASV result columns in {path}")
        result = dict(zip(columns, row, strict=False))
        raw_value = result.get("result")
        version = result.get("version")
        if raw_value is None:
            continue
        if not isinstance(raw_value, list) or len(raw_value) != 1:
            raise ValueError(f"non-scalar benchmark result in {path}")
        value = raw_value[0]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"non-numeric benchmark result in {path}")
        if not math.isfinite(value) or value <= 0.0 or not isinstance(version, str) or not version:
            raise ValueError(f"invalid benchmark result in {path}")
        measurements.append(
            Measurement(
                machine=str(data.get("params", {}).get("machine", "")),
                commit_hash=str(data.get("commit_hash", "")),
                value=float(value),
                version=version,
            )
        )
    return measurements


def _measurement_for_commit(measurements: list[Measurement], machine: str, commit_hash: str) -> Measurement | None:
    matches = [item for item in measurements if item.machine == machine and item.commit_hash == commit_hash]
    if len(matches) > 1:
        raise ValueError(f"multiple {machine} results found for {commit_hash}")
    return matches[0] if matches else None


def compare(
    config: dict, measurements: list[Measurement], current_sha: str, ancestors: list[str]
) -> tuple[list[Verdict], bool]:
    """Compare current results against a median of compatible ancestors."""
    verdicts = []
    failed = False
    window = int(config["baseline_window"])
    minimum = int(config["min_baseline_samples"])
    threshold = float(config["regression_threshold_pct"])

    for case in config["cases"]:
        key = case["key"]
        machine = f"perf-{key}-l40s"
        current = _measurement_for_commit(measurements, machine, current_sha)
        if current is None:
            failed = True
            verdicts.append(Verdict(key, "FAIL", None, None, None, 0, "current result is missing"))
            continue

        compatible = []
        by_commit = {
            item.commit_hash: item
            for item in measurements
            if item.machine == machine and item.version == current.version and item.commit_hash != current_sha
        }
        for commit_hash in ancestors:
            if commit_hash in by_commit:
                compatible.append(by_commit[commit_hash])
            if len(compatible) == window:
                break

        current_fps = 1.0 / current.value
        if len(compatible) < minimum:
            verdicts.append(
                Verdict(
                    key,
                    "WARN",
                    current_fps,
                    None,
                    None,
                    len(compatible),
                    f"need {minimum} compatible baseline samples",
                )
            )
            continue

        baseline_seconds = statistics.median(item.value for item in compatible)
        baseline_fps = 1.0 / baseline_seconds
        change_pct = (current_fps / baseline_fps - 1.0) * 100.0
        regressed = change_pct < -threshold and not math.isclose(change_pct, -threshold, abs_tol=1e-9)
        failed |= regressed
        verdicts.append(
            Verdict(
                key,
                "FAIL" if regressed else "PASS",
                current_fps,
                baseline_fps,
                change_pct,
                len(compatible),
                f"threshold: -{threshold:g}%",
            )
        )
    return verdicts, failed


def render_summary(verdicts: list[Verdict]) -> str:
    """Render comparison verdicts as a GitHub-flavored Markdown table."""

    def number(value: float | None, suffix: str = "") -> str:
        return "—" if value is None else f"{value:,.1f}{suffix}"

    lines = [
        "## Performance smoke test",
        "",
        "| Case | Status | Current FPS | Baseline FPS | Change | Samples | Detail |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for verdict in verdicts:
        lines.append(
            f"| `{verdict.case}` | **{verdict.status}** | {number(verdict.current_fps)} | "
            f"{number(verdict.baseline_fps)} | {number(verdict.change_pct, '%')} | {verdict.samples} | "
            f"{verdict.detail} |"
        )
    return "\n".join(lines) + "\n"


def git_ancestors(base_sha: str, repo: Path) -> list[str]:
    """Return the base commit and its ancestors, newest first."""
    completed = subprocess.run(["git", "rev-list", base_sha], cwd=repo, check=True, capture_output=True, text=True)
    return completed.stdout.splitlines()


def main() -> int:
    """Run the comparator CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--current-sha", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--summary-file", type=Path)
    args = parser.parse_args()

    config = load_config(args.tasks)
    measurements = load_measurements(args.results_dir)
    verdicts, failed = compare(config, measurements, args.current_sha, git_ancestors(args.base_sha, args.repo))
    summary = render_summary(verdicts)
    print(summary, end="")
    if args.summary_file:
        args.summary_file.write_text(summary)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
