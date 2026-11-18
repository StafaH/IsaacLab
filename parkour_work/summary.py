"""Print the evaluation curves of runs: usage summary.py <run.json> ..."""

import json
import sys

for path in sys.argv[1:]:
    with open(path) as file:
        run = json.load(file)
    history = run["history"]
    print(path.split("/")[-1], "share", run["args"]["share"], "schedule", run["args"]["schedule"])
    for row in history[:: max(1, len(history) // 12)] + history[-1:]:
        rungs = " ".join(f"{value:.2f}" for value in row["eval/rungs"])
        print(
            f"  {row['samples_m']:6.0f}M {row['time']:6.0f}s share {row['share']:.2f} track steps {row['track/steps']:4.0f}"
            f" lost {row['track/lost']:.2f} goal {row['goal/success']:.2f} | eval {rungs}"
        )
