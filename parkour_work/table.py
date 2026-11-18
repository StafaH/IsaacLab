"""Print, per run, the samples at which success holds a level and its final success: usage table.py <run.json> ..."""

import json
import sys


def holds(history, rung, level):
    """Return the samples [M] from which the success of ``rung`` stays at ``level`` or above, or None."""
    reached = None
    for row in history:
        if row["eval/rungs"][rung] >= level:
            reached = row["samples_m"] if reached is None else reached
        else:
            reached = None
    return reached


print(
    f"{'run':28s} {'clip 0.9 from':>14s} {'away 0.8 from':>14s} {'final clip':>11s} {'final away':>11s} {'samples':>8s}"
)
for path in sys.argv[1:]:
    with open(path) as file:
        run = json.load(file)
    history = run["history"]
    last = history[-3:]
    clip = sum(row["eval/rungs"][0] for row in last) / len(last)
    away = sum(row["eval/rungs"][-1] for row in last) / len(last)
    name = path.split("/")[-1].removesuffix(".json")
    first, second = holds(history, 0, 0.9), holds(history, -1, 0.8)
    print(
        f"{name:28s} {'never' if first is None else f'{first:.0f}M':>14s}"
        f" {'never' if second is None else f'{second:.0f}M':>14s} {clip:11.2f} {away:11.2f}"
        f" {history[-1]['samples_m']:7.0f}M"
    )
