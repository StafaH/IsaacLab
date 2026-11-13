# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize the logs of ``train_interfaces.py`` as a Markdown table, one row per method.

    uv run python scripts/reinforcement_learning/mdp2/summarize_interfaces.py runs/research/gate1 --at 1000

A run is a file ``<method>_s<seed>.json``. The value of a run is the mean of its last ``--rows`` log rows, or of the
last rows up to ``--at`` million samples. A cell is the mean over the seeds, with their range. S is the held-out
success averaged over the arms whose specialist, the method named after the arm, reaches ``--learnable``.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("folders", nargs="+")
parser.add_argument("--at", type=float, default=None, help="Samples at which to read the runs, in millions")
parser.add_argument("--rows", type=int, default=10)
parser.add_argument("--split", default="test", choices=["train", "test"])
parser.add_argument("--metric", default="success", choices=["success", "error_mm", "length"])
parser.add_argument("--learnable", type=float, default=0.2)
args = parser.parse_args()


def value(rows, key):
    """Return the mean of ``key`` over the last rows of a run, up to ``--at`` million samples."""
    rows = [row for row in rows if args.at is None or row["samples_m"] <= args.at + 1.0][-args.rows :]
    return sum(row[key] for row in rows) / len(rows)


def reached(rows, key, level=0.5, window=5):
    """Return the samples [M] at which the mean of ``key`` over ``window`` rows first passes ``level``."""
    for k in range(window, len(rows) + 1):
        if sum(row[key] for row in rows[k - window : k]) / window >= level:
            return rows[k - 1]["samples_m"]
    return None


def cell(values, scale, digits):
    """Return the mean of ``values`` and, for several, their range."""
    text = f"{scale * sum(values) / len(values):.{digits}f}"
    if len(values) > 1:
        text += f" ({scale * min(values):.{digits}f} to {scale * max(values):.{digits}f})"
    return text


runs = defaultdict(dict)
for folder in args.folders:
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        method, seed = re.fullmatch(r"(.+)_s(\d+)", os.path.basename(path)[:-5]).groups()
        with open(path) as file:
            runs[method][int(seed)] = json.load(file)["history"]
first = next(iter(next(iter(runs.values())).values()))[0]
arms = [key.split("/")[0] for key in first if key.endswith(f"/{args.split}/success")]

success = {
    method: {arm: [value(rows, f"{arm}/{args.split}/success") for rows in seeds.values()] for arm in arms}
    for method, seeds in runs.items()
}
learnable = [arm for arm in arms if arm not in runs or sum(success[arm][arm]) / len(runs[arm]) >= args.learnable]
scale = 100.0 if args.metric == "success" else 1.0
last = learnable[-1]
print(f"Arms in S: {', '.join(learnable)}. Split: {args.split}. Read at: {args.at or 'the end'} M.\n")
print(f"| Method | Seeds | Samples [M] | {' | '.join(f'`{arm}`' for arm in arms)} | S | 50% on `{last}` at [M] |")
print(f"|---|---|---|{'---|' * len(arms)}---|---|")
for method, seeds in sorted(runs.items()):
    samples = min(max(rows[-1]["samples_m"] for rows in seeds.values()), args.at or float("inf"))
    cells = [
        cell([value(rows, f"{arm}/{args.split}/{args.metric}") for rows in seeds.values()], scale, 0) for arm in arms
    ]
    score = [sum(success[method][arm][k] for arm in learnable) / len(learnable) for k in range(len(seeds))]
    marks = [reached(rows, f"{last}/{args.split}/success") for rows in seeds.values()]
    marks = [mark for mark in marks if mark is not None]
    mark = f"{sum(marks) / len(marks):.0f} ({len(marks)} of {len(seeds)})" if marks else "never"
    print(f"| `{method}` | {len(seeds)} | {samples:.0f} | {' | '.join(cells)} | {cell(score, 100.0, 1)} | {mark} |")
