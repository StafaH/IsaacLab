"""Write the sentences of the paper that depend on the seeds of the policy on all clips: usage make_seeds.py <runs folder>."""

import glob
import json
import os
import sys

import numpy as np

RUNS = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}

with open(os.path.join(HERE, "figs", "numbers.json")) as file:
    numbers = json.load(file)
credit, even = numbers["full/credit"], numbers["full/fixed0.50"]
count = len(credit["clip_final"])
seeds = WORDS[count]


def spread(values, digits=2):
    """Return the range of the seeds as text, or the one value they share."""
    low, high = f"{min(values):.{digits}f}", f"{max(values):.{digits}f}"
    return f"{low} in each" if low == high else f"{low} to {high}"


def mean(values):
    return f"{np.mean(values):.2f}"


# Does every seed fail the tallest box of clip 28, the 1.14 m box?
fails = []
for path in sorted(glob.glob(os.path.join(RUNS, "full", "credit_s*.json"))):
    with open(path) as file:
        run = json.load(file)
    if run["history"][-1]["iteration"] == run["args"]["iterations"]:
        table = run["history"][-1]["eval/tasks"]
        fails.append(table[run["tasks"].index("climb_28 z1.2 up")][0] < 0.5)
assert len(fails) == count, (len(fails), count)
tallest = "Every seed fails the 1.14\\,m box." if all(fails) else "The first seed fails one task, the 1.14\\,m box."

several = count > 1
over = (
    f" and the {seeds} seeds (ranges {spread(credit['clip_final'])} and {spread(credit['away_final'])})"
    if several
    else ""
)
solved = (
    f"{spread(credit['clip_solved'], 0)} and {spread(credit['away_solved'], 0)}"
    if several
    else f"{credit['clip_solved'][0]:.0f} and {credit['away_solved'][0]:.0f}"
)
passes = spread(credit["clip_tracker"], 0) if several else f"{credit['clip_tracker'][0]:.0f}"
half = spread(credit["clip_half"], 0) if several else f"{credit['clip_half'][0]:.0f}"
beyond = spread(credit["beyond_teacher"], 0) if several else f"{credit['beyond_teacher'][0]:.0f}"
assert max(credit["teacher_only"]) == 0

sentences = {
    "seedsAll": seeds,
    "fullSentence": (
        f"After 590 million samples it succeeds in {mean(credit['clip_final'])} of episodes from the clip start and"
        f" {mean(credit['away_final'])} from displaced starts, averaged over the tasks{over}, and solves {solved} of the"
        " \\tasksall{} tasks at 0.5 or better. The tracker is at 0.52 and 0.17, and at 62 tasks and one. The policy passes the"
        f" final clip-start success of its teacher after {passes} million samples of its own, and the share of reference worlds"
        " falls from 0.84 to about 0.35 on the way. \\fullBaseline{}"
    ),
    "fullBaseline": (
        f"With an even share and tasks drawn evenly, the usual setting, the same budget ends at {even['clip_final'][0]:.2f} and"
        f" {even['away_final'][0]:.2f} with {even['clip_solved'][0]:.0f} and {even['away_solved'][0]:.0f} tasks solved in its one seed,"
        f" and the policy needs {even['clip_half'][0]:.0f} million samples to reach a mean of 0.5 from the clip start, against"
        f" {half} million with the rule."
    ),
    "fullTasksSentence": (
        f"The policy solves {beyond} tasks from the clip start that its teacher does not, and none the other way"
        + (" in any seed" if several else "")
        + ". Among them"
        + (", in the first seed," if several else "")
        + " are all heights of four clips on which the tracker never completes the climb (0.00 to 0.02 for the tracker, 0.87 to 0.99"
        f" for the policy) and most of the tallest boxes (0.33 against 0.86 at the largest scale). {tallest}"
    ),
    "abstractAll": (
        f"One policy trained this way on 105 tasks succeeds in {mean(credit['clip_final'])} of episodes from the starts of the clips"
        f" and {mean(credit['away_final'])} from displaced starts"
        + (f" (mean of {seeds} seeds)" if several else "")
        + f", where the tracker that teaches it reaches 0.52 and 0.17, and it solves {beyond} tasks that its teacher does not."
    ),
}

path = os.path.join(HERE, "numbers.tex")
with open(path) as file:
    lines = file.read().rstrip("\n").split("\n")
for name, text in sentences.items():
    line = f"\\newcommand{{\\{name}}}{{{text}}}"
    hits = [index for index, old in enumerate(lines) if old.startswith(f"\\newcommand{{\\{name}}}")]
    assert len(hits) <= 1, name
    if hits:
        lines[hits[0]] = line
    else:
        lines.append(line)
    print(line, "\n")
with open(path, "w") as file:
    file.write("\n".join(lines) + "\n")
