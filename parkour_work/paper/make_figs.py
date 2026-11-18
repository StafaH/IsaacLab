"""Draw the figures of the paper and write its numbers from the logs of the runs: usage make_figs.py <runs folder>."""

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)
# Colours that stay apart for readers with colour vision deficiencies (Okabe and Ito).
BLUE, ORANGE, GREEN, RED, PURPLE, GREY, SKY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#8c8c8c",
    "#56B4E9",
)
numbers = {}


def load(pattern, finished=True):
    """Return the logs of the runs that match ``pattern`` in the runs folder, by seed: those that ran to their end."""
    runs = []
    for path in sorted(glob.glob(os.path.join(RUNS, pattern))):
        with open(path) as file:
            run = json.load(file)
        if not finished or run["history"][-1]["iteration"] == run["args"]["iterations"]:
            runs.append(run)
    return runs


def curve(runs, rung):
    """Return the samples [M] and the success of ``rung`` of every run, cut to the shortest run."""
    rows = min(len(run["history"]) for run in runs)
    samples = np.array([row["samples_m"] for row in runs[0]["history"][:rows]])
    success = np.array([[row["eval/rungs"][rung] for row in run["history"][:rows]] for run in runs])
    return samples, success


def task_curve(runs, rung):
    """Return the samples [M] and the success of ``rung`` averaged over the tasks, each task counting the same."""
    rows = min(len(run["history"]) for run in runs)
    samples = np.array([row["samples_m"] for row in runs[0]["history"][:rows]])
    success = np.array(
        [[np.mean([task[rung] for task in row["eval/tasks"]]) for row in run["history"][:rows]] for run in runs]
    )
    return samples, success


def reaches(samples, success, level):
    """Return the samples [M] at which a curve first reaches ``level``, or infinity."""
    above = np.nonzero(success >= level)[0]
    return float(samples[above[0]]) if len(above) else np.inf


def holds(run, rung, level):
    """Return the samples [M] from which the success of ``rung`` stays at ``level`` or above, or infinity."""
    reached = np.inf
    for row in run["history"]:
        if row["eval/rungs"][rung] >= level:
            reached = min(reached, row["samples_m"])
        else:
            reached = np.inf
    return reached


def final(run, rung, rows=3):
    return float(np.mean([row["eval/rungs"][rung] for row in run["history"][-rows:]]))


def band(axis, samples, success, colour, label, style="-"):
    axis.plot(samples, success.mean(0), style, color=colour, label=label, linewidth=1.6)
    axis.fill_between(samples, success.min(0), success.max(0), color=colour, alpha=0.15, linewidth=0)


def schedules(folder="dev"):
    """Success away from the clip over samples, and the share of tracking worlds, per schedule."""
    methods = [
        ("gate", "Credit rule (ours)", BLUE, "-"),
        ("deficit", "Ours without gates", SKY, "-"),
        ("fixed0.50", "Fixed 0.50", GREY, "-"),
        ("progress", "Progress schedule", ORANGE, "--"),
        ("two_stage", "Two stages", RED, "--"),
        ("credit", "Trial learners", PURPLE, ":"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.25))
    for name, label, colour, style in methods:
        runs = load(f"{folder}/{name}_s*.json")
        if not runs:
            continue
        for axis, rung in zip(axes[:2], (0, -1)):
            band(axis, *curve(runs, rung), colour, label, style)
        rows = min(len(run["history"]) for run in runs)
        samples = [row["samples_m"] for row in runs[0]["history"][:rows]]
        share = np.array([[row["share"] for row in run["history"][:rows]] for run in runs])
        axes[2].plot(samples, share.mean(0), style, color=colour, linewidth=1.6)
        times = np.array([[holds(run, 0, 0.9), holds(run, -1, 0.8)] for run in runs])
        numbers[f"{folder}/{name}"] = {
            "clip_from": times[:, 0].tolist(),
            "away_from": times[:, 1].tolist(),
            "clip_final": [final(run, 0) for run in runs],
            "away_final": [final(run, -1) for run in runs],
        }
    axes[0].set_title("From the start of the clip", fontsize=8)
    axes[1].set_title("From displaced starts", fontsize=8)
    axes[2].set_title("Share of reference worlds", fontsize=8)
    for axis in axes:
        axis.set_xlabel("Samples (millions)")
        axis.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Success")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=6.8, columnspacing=1.6)
    figure.tight_layout(pad=0.4, rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(os.path.join(OUT, f"schedules_{folder}.pdf"))
    plt.close(figure)


def sweep(folder="dev"):
    """Final success from the clip start and away from it, per fixed share of tracking worlds."""
    shares = ["goal_only", "fixed0.10", "fixed0.25", "fixed0.50", "fixed0.75", "fixed1.00"]
    labels = ["0", "0.10", "0.25", "0.50", "0.75", "1.00"]
    clip, away = [], []
    for name in shares:
        runs = load(f"{folder}/{name}_s*.json")
        clip.append([final(run, 0) for run in runs] or [np.nan])
        away.append([final(run, -1) for run in runs] or [np.nan])
        if runs:
            times = np.array([[holds(run, 0, 0.9), holds(run, -1, 0.8)] for run in runs])
            numbers[f"{folder}/{name}"] = {
                "clip_from": times[:, 0].tolist(),
                "away_from": times[:, 1].tolist(),
                "clip_final": clip[-1],
                "away_final": away[-1],
            }
    figure, axis = plt.subplots(figsize=(3.4, 2.0))
    x = np.arange(len(shares))
    for offset, values, colour, label in (
        (-0.18, clip, GREY, "From the start of the clip"),
        (0.18, away, BLUE, "From displaced starts"),
    ):
        axis.bar(x + offset, [np.mean(v) for v in values], 0.34, color=colour, label=label)
        for k, v in enumerate(values):
            axis.plot(np.full(len(v), x[k] + offset), v, "o", color="black", markersize=2)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Fixed share of reference worlds")
    axis.set_ylabel("Success at the end of training")
    axis.set_ylim(0.0, 1.05)
    axis.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 1.22), ncol=2, columnspacing=1.0)
    figure.tight_layout(pad=0.4)
    figure.savefig(os.path.join(OUT, f"sweep_{folder}.pdf"))
    plt.close(figure)


def displaced():
    """Success from starts displaced by growing amounts: the policy and the tracker of the development clip."""
    amounts = ["0.25", "0.5", "0.75", "1.0"]
    figure, axis = plt.subplots(figsize=(3.4, 1.9))
    for name, label, colour in (("student", "Policy without the clip (ours)", BLUE), ("tracker", "Tracker", RED)):
        rows = [
            run["history"][-1]["eval/rungs"] for amount in amounts for run in load(f"displaced/{name}_{amount}.json")
        ]
        if len(rows) < len(amounts):
            continue
        clip = float(np.mean([row[0] for row in rows]))
        axis.plot(
            [0.0] + [float(a) for a in amounts],
            [clip] + [row[-1] for row in rows],
            "o-",
            color=colour,
            label=label,
            linewidth=1.6,
            markersize=3.5,
        )
        numbers[f"displaced/{name}"] = {"clip": [clip], "away": [row[-1] for row in rows]}
    axis.axvspan(0.0, 0.5, color=GREY, alpha=0.12, linewidth=0)
    axis.text(0.25, 0.08, "trained range", fontsize=6.5, color="#555555", ha="center")
    axis.set_xlabel("Largest displacement of the start (m and rad)")
    axis.set_ylabel("Success")
    axis.set_ylim(-0.02, 1.05)
    axis.legend(fontsize=6.5, loc="center right")
    figure.tight_layout(pad=0.4)
    figure.savefig(os.path.join(OUT, "displaced.pdf"))
    plt.close(figure)


def benchmark(pattern, name, rung, title):
    """Success of every task of a run, a row per clip and a column per box height."""
    runs = load(pattern)
    if not runs:
        return
    run = runs[0]
    tasks, table = run["tasks"], np.array(run["history"][-1]["eval/tasks"])
    clips = sorted({task.split()[0] for task in tasks})
    levels = sorted({task.split()[1] for task in tasks})
    grid = np.array([[table[tasks.index(f"{clip} {level} up")][rung] for level in levels] for clip in clips])
    figure, axis = plt.subplots(figsize=(1.7, 3.3))
    image = axis.imshow(grid, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(levels)), [level[1:] for level in levels], fontsize=6)
    axis.set_yticks(range(len(clips)), [clip[-2:] for clip in clips], fontsize=5.5)
    axis.set_xlabel("Box height scale", fontsize=7)
    axis.set_ylabel("Clip", fontsize=7)
    axis.set_title(title, fontsize=7)
    axis.grid(False)
    figure.colorbar(image, ax=axis, fraction=0.08, pad=0.04).ax.tick_params(labelsize=6)
    figure.tight_layout(pad=0.3)
    figure.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.close(figure)
    numbers[name] = {"mean": [float(grid.mean())], "solved": [float((grid >= 0.5).mean())]}


def grid_of(run, rung, rows=1):
    """Return the success of every task of a run from a rung, a row per clip and a column per box height."""
    tasks = run["tasks"]
    table = np.mean([np.array(row["eval/tasks"]) for row in run["history"][-rows:]], axis=0)
    clips = sorted({task.split()[0] for task in tasks})
    levels = sorted({task.split()[1] for task in tasks})
    grid = np.array([[table[tasks.index(f"{clip} {level} up")][rung] for level in levels] for clip in clips])
    return grid, clips, levels


def full():
    """All clips: success during training against the tracker, and the success of every task."""
    tracker = load("tracker_all.json")
    methods = [("credit", "Credit rule (ours)", BLUE, "-"), ("fixed0.50", "Fixed 0.50, even tasks", GREY, "-")]
    if not tracker or not load("full/credit_s0.json"):
        return
    figure, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    for name, label, colour, style in methods:
        runs = load(f"full/{name}_s*.json")
        if not runs:
            continue
        for axis, rung in zip(axes, (0, -1)):
            band(axis, *task_curve(runs, rung), colour, label, style)
        grids = [[grid_of(run, rung)[0] for rung in (0, -1)] for run in runs]
        clip_curve, away_curve = task_curve(runs, 0), task_curve(runs, -1)
        teach = grid_of(tracker[0], 0)[0] >= 0.5
        level = grid_of(tracker[0], 0)[0].mean()
        numbers[f"full/{name}"] = {
            "clip_final": [float(grid[0].mean()) for grid in grids],
            "away_final": [float(grid[1].mean()) for grid in grids],
            "clip_solved": [float((grid[0] >= 0.5).sum()) for grid in grids],
            "away_solved": [float((grid[1] >= 0.5).sum()) for grid in grids],
            "clip_teachable": [float(grid[0][teach].mean()) for grid in grids],
            "away_teachable": [float(grid[1][teach].mean()) for grid in grids],
            "beyond_teacher": [float(((grid[0] >= 0.5) & ~teach).sum()) for grid in grids],
            "teacher_only": [float(((grid[0] < 0.5) & teach).sum()) for grid in grids],
            "samples": [run["history"][-1]["samples_m"] for run in runs],
            "clip_half": [reaches(clip_curve[0], row, 0.5) for row in clip_curve[1]],
            "away_half": [reaches(away_curve[0], row, 0.5) for row in away_curve[1]],
            "clip_tracker": [reaches(clip_curve[0], row, level) for row in clip_curve[1]],
        }
    for axis, rung in zip(axes, (0, -1)):
        samples, success = task_curve(tracker, rung)
        axis.plot(samples, success[0], ":", color=RED, linewidth=1.4, label="Tracker (observes the clip)")
        axis.set_xlabel("Samples (millions)")
    grids = [grid_of(tracker[0], rung)[0] for rung in (0, -1)]
    numbers["full/tracker"] = {
        "clip_final": [float(grids[0].mean())],
        "away_final": [float(grids[1].mean())],
        "clip_solved": [float((grids[0] >= 0.5).sum())],
        "away_solved": [float((grids[1] >= 0.5).sum())],
        "clip_teachable": [float(grids[0][grids[0] >= 0.5].mean())],
        "away_teachable": [float(grids[1][grids[0] >= 0.5].mean())],
    }
    axes[0].set_title("From the start of the clip", fontsize=8)
    axes[1].set_title("From displaced starts", fontsize=8)
    axes[0].set_ylabel("Success, mean of 105 tasks")
    axes[0].set_ylim(-0.02, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, fontsize=6.3, columnspacing=1.0)
    figure.tight_layout(pad=0.4, rect=(0.0, 0.0, 1.0, 0.86))
    figure.savefig(os.path.join(OUT, "full_curves.pdf"))
    plt.close(figure)

    # The success of every task: the tracker and the policy from the clip start, and the policy from displaced starts.
    student = load("full/credit_s0.json")[0]
    panels = [
        (grid_of(tracker[0], 0), "Tracker, clip start"),
        (grid_of(student, 0), "Policy, clip start"),
        (grid_of(tracker[0], -1), "Tracker, displaced"),
        (grid_of(student, -1), "Policy, displaced"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(7.1, 3.0), sharey=True)
    for axis, ((grid, clips, levels), title) in zip(axes, panels):
        image = axis.imshow(grid, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_xticks(range(len(levels)), [level[1:] for level in levels], fontsize=6)
        axis.set_yticks(range(len(clips)), [clip[-2:] for clip in clips], fontsize=5.5)
        axis.set_xlabel("Scale of the box height", fontsize=7)
        axis.set_title(f"{title}: {grid.mean():.2f}", fontsize=7.5)
        axis.grid(False)
    axes[0].set_ylabel("Clip", fontsize=7)
    figure.tight_layout(pad=0.4, rect=(0.0, 0.0, 0.93, 1.0))
    bar = figure.add_axes((0.94, 0.16, 0.012, 0.7))
    figure.colorbar(image, cax=bar).ax.tick_params(labelsize=6)
    bar.set_ylabel("Success", fontsize=7)
    figure.savefig(os.path.join(OUT, "full_tasks.pdf"))
    plt.close(figure)


def heldout_table():
    """Write the table of the runs that hold tasks out of training: success on the tasks trained on and on the rest."""
    lines = [
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Tasks trained on} & \\multicolumn{2}{c}{Tasks held out} \\\\",
        "Held out of training & Tasks & clip & displaced & clip & displaced \\\\",
        "\\midrule",
    ]
    for label, name in (("Two of five box heights", "holdout_levels"), ("Five of 21 clips", "holdout_clips")):
        runs = load(f"full/{name}_s0.json", finished=False)
        if not runs:
            continue
        run = runs[0]
        seen = np.array(run["seen"])
        table = np.array(run["history"][-1]["eval/tasks"])
        values = [
            table[seen][:, 0].mean(),
            table[seen][:, -1].mean(),
            table[~seen][:, 0].mean(),
            table[~seen][:, -1].mean(),
        ]
        numbers[f"full/{name}"] = {
            "seen_clip": [float(values[0])],
            "seen_away": [float(values[1])],
            "unseen_clip": [float(values[2])],
            "unseen_away": [float(values[3])],
            "unseen_tasks": [float((~seen).sum())],
            "samples": [run["history"][-1]["samples_m"]],
        }
        lines.append(f"{label} & {int((~seen).sum())} & " + " & ".join(f"{value:.2f}" for value in values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "table_heldout.tex"), "w") as file:
        file.write("\n".join(lines) + "\n")


def cell(values, never=np.inf, digits=0):
    """Return the mean of the seeds and their range, or how many of them never got there."""
    values = np.array(values, dtype=float)
    reached = values[np.isfinite(values)]
    if len(reached) == 0:
        return "never"
    text = f"{reached.mean():.{digits}f}"
    if len(reached) > 1:
        text += f" \\,{{\\scriptsize({reached.min():.{digits}f}--{reached.max():.{digits}f})}}"
    if len(reached) < len(values):
        text += f" \\,{{\\scriptsize[{len(reached)}/{len(values)}]}}"
    return text


def schedule_table(folder="dev"):
    """Write the table of the schedules: samples to the two levels of success and the success at the end."""
    rows = [
        ("Goal worlds only", "goal_only"),
        "Reference worlds taught by a tracker",
        ("Fixed share 0.10", "fixed0.10"),
        ("Fixed share 0.25", "fixed0.25"),
        ("Fixed share 0.50", "fixed0.50"),
        ("Fixed share 0.75", "fixed0.75"),
        ("Fixed share 1.00", "fixed1.00"),
        ("Two stages~\\cite{hil}", "two_stage"),
        ("Tracking progress~\\cite{wang2026generalizing}", "progress"),
        ("Two trial learners", "credit"),
        ("Credit rule without gates", "deficit"),
        ("Credit rule (ours)", "gate"),
        "Reference worlds with the tracking reward only",
        ("Fixed share 0.50", "no_teacher0.50"),
        ("Fixed share 1.00", "no_teacher1.00"),
        ("Two stages~\\cite{hil}", "nt_two_stage"),
        ("Tracking progress~\\cite{wang2026generalizing}", "nt_progress"),
        ("Credit rule without gates", "nt_deficit"),
        ("Credit rule", "nt_gate"),
    ]
    lines = [
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Samples (M) until success holds} & \\multicolumn{2}{c}{Success at 59\\,M} \\\\",
        "Share of reference worlds & clip $\\geq$0.9 & displaced $\\geq$0.8 & clip & displaced \\\\",
        "\\midrule",
    ]
    for row in rows:
        if isinstance(row, str):
            lines += ["\\midrule", f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{row}}}}} \\\\"]
            continue
        label, name = row
        values = numbers.get(f"{folder}/{name}")
        if values is None:
            continue
        lines.append(
            f"{label} & {cell(values['clip_from'])} & {cell(values['away_from'])} & {cell(values['clip_final'], digits=2)}"
            f" & {cell(values['away_final'], digits=2)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, f"table_{folder}.tex"), "w") as file:
        file.write("\n".join(lines) + "\n")


def cause_table(folder="dev"):
    """Write the table of what each kind of world causes: success from the clip start and from displaced starts."""
    tracker = load(f"{folder}/tracker.json")
    if tracker:
        numbers[f"{folder}/tracker"] = {"clip_final": [final(tracker[0], 0)], "away_final": [final(tracker[0], -1)]}
    rows = [
        ("Goal worlds only", "goal_only", "--", "--"),
        ("Tracker (observes the clip)", "tracker", "reward", "--"),
        None,
        ("Reference worlds, tracking reward", "no_teacher1.00", "reward", "--"),
        ("Reference worlds, teacher", "fixed1.00", "teacher", "--"),
        ("Both, tracking reward", "no_teacher0.50", "reward", "Boolean"),
        ("Both, teacher", "fixed0.50", "teacher", "Boolean"),
    ]
    lines = [
        "\\begin{tabular}{@{}lcc@{}}",
        "\\toprule",
        "Worlds that train the policy & Clip start & Displaced start \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row is None:
            lines.append("\\midrule")
            continue
        label, name, _, _ = row
        values = numbers.get(f"{folder}/{name}")
        if values is None:
            continue
        lines.append(f"{label} & {cell(values['clip_final'], digits=2)} & {cell(values['away_final'], digits=2)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "table_cause.tex"), "w") as file:
        file.write("\n".join(lines) + "\n")


def more_runs(folder="dev"):
    """Read the runs of the schedules that the figures do not draw."""
    for name in (
        "credit",
        "gate",
        "no_teacher0.50",
        "no_teacher1.00",
        "nt_gate",
        "nt_two_stage",
        "nt_progress",
        "nt_deficit",
    ):
        runs = load(f"{folder}/{name}_s*.json")
        if runs:
            times = np.array([[holds(run, 0, 0.9), holds(run, -1, 0.8)] for run in runs])
            numbers[f"{folder}/{name}"] = {
                "clip_from": times[:, 0].tolist(),
                "away_from": times[:, 1].tolist(),
                "clip_final": [final(run, 0) for run in runs],
                "away_final": [final(run, -1) for run in runs],
            }


if __name__ == "__main__":
    schedules()
    sweep()
    displaced()
    more_runs()
    schedule_table()
    cause_table()
    full()
    heldout_table()
    with open(os.path.join(OUT, "numbers.json"), "w") as file:
        json.dump(numbers, file, indent=1)
    for name, values in numbers.items():
        print(name, {key: [round(v, 2) for v in value] for key, value in values.items()})
