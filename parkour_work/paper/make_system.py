"""Draw the diagram of the system: the kinds of worlds, what each gives the learner, and what sets their shares."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
BLUE, ORANGE, GREEN, GREY, INK = "#0072B2", "#E69F00", "#009E73", "#8c8c8c", "#222222"

figure, axis = plt.subplots(figsize=(7.1, 3.0))
axis.set_xlim(0.0, 100.0)
axis.set_ylim(0.0, 42.3)
axis.axis("off")


def box(x, y, width, height, colour, title, lines, fill="#ffffff"):
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            linewidth=1.1,
            edgecolor=colour,
            facecolor=fill,
        )
    )
    axis.text(x + 1.2, y + height - 2.6, title, fontsize=8, fontweight="bold", color=colour, va="center")
    for k, line in enumerate(lines):
        axis.text(x + 1.2, y + height - 5.9 - 3.0 * k, line, fontsize=6.6, color=INK, va="center")


def arrow(start, end, colour, label=None, offset=(0.0, 1.3), style="-|>", dashed=False):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=1.1,
            color=colour,
            linestyle="--" if dashed else "-",
        )
    )
    if label:
        axis.text(
            (start[0] + end[0]) / 2 + offset[0],
            (start[1] + end[1]) / 2 + offset[1],
            label,
            fontsize=6.3,
            color=colour,
            ha="center",
            va="center",
        )


# One simulation, three kinds of worlds.
axis.add_patch(
    FancyBboxPatch(
        (1.0, 1.0),
        46.0,
        40.3,
        boxstyle="round,pad=0.3,rounding_size=1.5",
        linewidth=0.8,
        edgecolor=GREY,
        facecolor="#f6f6f6",
    )
)
axis.text(3.0, 39.6, "One simulation: a tile per clip and box height", fontsize=6.8, color=GREY, va="center")
box(
    3.0, 26.5, 42.0, 11.0, BLUE, "Reference worlds", ["start on a frame of a clip", "imitate a tracker's joint targets"]
)
box(
    3.0,
    14.3,
    42.0,
    11.0,
    ORANGE,
    "Goal worlds",
    ["start on a clip frame, or displaced", "one Boolean at the end: stands at the goal"],
)
box(
    3.0,
    2.1,
    42.0,
    11.0,
    GREEN,
    "Evaluation worlds",
    ["every task, from a ladder of starts", "no noise; no data to the learner"],
)

# The learner, its teacher, and the rule.
box(57.0, 23.5, 18.5, 14.0, INK, "Policy", ["sees the robot, a", "height scan, and the", "goal; never a clip"])
box(82.5, 23.5, 16.0, 14.0, GREY, "Tracker", ["sees the clip;", "trained first, on", "reference worlds"])
box(
    58.0,
    2.1,
    40.5,
    17.0,
    GREEN,
    "Credit the cause",
    [
        "reference worlds where the tracker",
        "succeeds and the policy fails on the clip;",
        "goal worlds where the policy succeeds",
        "on the clip and fails away from it",
    ],
    fill="#f2faf7",
)

arrow((45.6, 30.5), (56.6, 31.5), BLUE, "samples", (0.6, 1.6))
arrow((45.6, 18.5), (56.6, 27.0), ORANGE, "samples", (1.6, -2.4))
arrow((45.6, 7.0), (57.6, 7.0), GREEN, "success per", (0.4, 1.6))
axis.text(52.0, 5.3, "task and start", fontsize=6.3, color=GREEN, ha="center", va="center")
arrow((82.1, 30.5), (75.9, 30.5), GREY)
axis.text(79.0, 32.3, "targets", fontsize=6.3, color=GREY, ha="center")
arrow((57.6, 14.5), (50.0, 14.5), GREEN, style="-", dashed=True)
arrow((50.0, 14.5), (50.0, 35.5), GREEN, style="-", dashed=True)
arrow((50.0, 35.5), (45.6, 35.5), GREEN, dashed=True)
arrow((50.0, 23.2), (45.6, 23.2), GREEN, dashed=True)
axis.text(53.6, 16.2, "shares", fontsize=6.3, color=GREEN, ha="center", va="center")

figure.savefig(os.path.join(OUT, "system.pdf"), bbox_inches="tight", pad_inches=0.02)
figure.savefig(os.path.join(OUT, "system.png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
