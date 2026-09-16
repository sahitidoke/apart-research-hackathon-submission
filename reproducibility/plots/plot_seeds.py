"""Two-seed versions of the three figures the paper leads with.

Same constraint as the other two scripts: reads `audit.json` only, so matplotlib
never enters `orchestrator/provenance/`'s dependency graph and no figure can
change a number.

    /tmp/plotenv/bin/python scripts/plot_seeds.py \
        --seed0 runs/prov-ind/audit-v2/audit.json \
        --seed1 runs/prov-ind-s1/audit/audit.json -o figures

Why two seeds belong in these three and not in the others. At `temperature 0`
a seed does not resample: it redraws *which questions* are asked and *which
distractor* is planted. So a second seed is a second question sample, and these
three figures are the ones whose numbers are conditional on that sample rather
than on the instrument. Putting both draws in the frame is the honest way to
show a rate that moved by 27 points between them.

Seed colour is fixed and never cycled, and every bar carries its value, so
identity is not carried by colour alone. (The palette validator from the dataviz
guidance needs node, which was unavailable here; blue/red is the conventional
CVD-safe pair and is already the pairing used in `plot_thesis.py`.)
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BLUE, RED = "#0F4D92", "#B64342"
GREY_FILL, GREY_MID, GREY_DARK, INK = "#CFCECE", "#767676", "#4D4D4D", "#272727"
SEEDS = ((0, BLUE), (1, RED))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 15,
    "axes.linewidth": 2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(figure, path, pad=2):
    figure.tight_layout(pad=pad)
    figure.savefig(path, dpi=300)
    figure.savefig(Path(path).with_suffix(".pdf"))
    plt.close(figure)
    return path


def _cells(audit):
    """Per-hop rows plus the pooled row, in plotting order."""
    return list(audit["deference"]) + [audit["deference_overall"]]


def figure_deference(audits, out):
    """Deference per hop count and pooled, both question draws.

    Two uncertainties, drawn differently and never pooled: the error bar is
    sampling error on the episodes that could be labelled, the grey band is the
    separate question of what the *unlabelled* wrong answers were. A third
    source of variation -- which questions were drawn at all -- is what the two
    colours show, and it is the largest of the three.
    """
    figure, axes = plt.subplots(figsize=(10.4, 5.8))
    offsets = (-0.17, 0.17)
    labels, positions = [], []
    for index, hops in enumerate([2, 3, 4, "all"]):
        positions.append(index)
        counts = []
        for (seed, colour), offset in zip(SEEDS, offsets):
            cells = _cells(audits[seed])
            row = (cells[-1] if hops == "all"
                   else next((r for r in cells[:-1] if r["hops"] == hops), None))
            if row is None:
                continue
            x = index + offset
            axes.plot([x, x], [row["attribution_low"], row["attribution_high"]],
                      color=GREY_FILL, linewidth=11, solid_capstyle="butt", zorder=1)
            axes.errorbar([x], [row["deference_rate"]],
                          yerr=[[row["deference_rate"] - row["ci_low"]],
                                [row["ci_high"] - row["deference_rate"]]],
                          fmt="o", markersize=13 if hops == "all" else 10,
                          color=colour, ecolor=colour, elinewidth=2.5,
                          capsize=7, capthick=2.5, zorder=3)
            # Beside the marker, not above it: directly above sits on the
            # whisker and the digits get a vertical line through them.
            axes.annotate(f"{row['deference_rate']:.2f}", (x, row["deference_rate"]),
                          textcoords="offset points",
                          # A rate of 0 sits on the baseline, where a label
                          # offset downward gets a rule struck through it.
                          xytext=((-13, -5) if offset < 0 else (13, -5))
                          if row["deference_rate"] > 0.05
                          else ((-13, 4) if offset < 0 else (13, 4)),
                          ha="right" if offset < 0 else "left",
                          fontsize=12, fontweight="bold", color=colour)
            counts.append(str(row["episodes"]))
        labels.append(f"{hops}-hop\nn={'/'.join(counts)}" if hops != "all"
                      else f"all hops\nn={'/'.join(counts)}")
    axes.axvline(2.5, color=GREY_MID, linewidth=1.5, zorder=0)
    axes.axhline(0, color=INK, linewidth=2)
    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_xlim(-0.55, 3.55)
    axes.set_ylim(-0.03, 1.12)
    axes.set_ylabel("deference rate")
    axes.set_title("Deference under a seeded planner error, two question draws",
                   fontsize=16, pad=14)
    axes.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color=BLUE, markersize=10,
               label="seed 0"),
        Line2D([], [], marker="o", linestyle="none", color=RED, markersize=10,
               label="seed 1"),
        Line2D([], [], color=GREY_FILL, linewidth=9, label="attribution bounds"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3, fontsize=12)
    return save(figure, out / "fig-deference-seeds.png")


def figure_worker_behaviour(audits, out):
    """What the workers did in the same episodes, pooled over hop count.

    These are not mutually exclusive outcomes and the figure does not stack
    them: a worker that contests the planted claim and a worker that repeats it
    can be the same worker in the same episode, which is the point.
    """
    fields = (("worker_pushed_back_rate", "contested\nthe planted claim"),
              ("worker_restated_evidence_rate", "restated\nthe true fact"),
              ("worker_repeated_error_rate", "repeated\nthe planted error"))
    figure, axes = plt.subplots(figsize=(9.6, 5.6))
    width = 0.34
    for (seed, colour), offset in zip(SEEDS, (-width / 2 - 0.012, width / 2 + 0.012)):
        rows = audits[seed]["deference"]
        total = sum(r["episodes"] for r in rows)
        for index, (key, _) in enumerate(fields):
            value = sum(r[key] * r["episodes"] for r in rows) / total
            axes.bar(index + offset, value, width=width, color=colour, zorder=2,
                     label=f"seed {seed} (n={total})" if index == 0 else None)
            axes.annotate(f"{value:.2f}", (index + offset, value), ha="center",
                          textcoords="offset points", xytext=(0, 7),
                          fontsize=13, fontweight="bold", color=colour)
    axes.set_xticks(range(len(fields)))
    axes.set_xticklabels([label for _, label in fields])
    axes.set_ylabel("rate over eligible seeded episodes")
    axes.set_ylim(0, 1.18)
    axes.axhline(0, color=INK, linewidth=2)
    axes.set_title("Worker response to the planted error, two question draws",
                   fontsize=16, pad=14)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=13)
    return save(figure, out / "fig-worker-behaviour-seeds.png")


def figure_per_agent(audits, out):
    """Each worker read through its own messages, not the collective's answer.

    One panel per role rather than one crowded axis, because the comparison a
    reader needs is seed against seed within a role. The bystander has no
    evidence bar in either draw -- the swap was never about its paragraphs --
    and that is drawn as an absent arm rather than as a zero, which would read
    as "insensitive to its own evidence".
    """
    fields = (("mean_planner_sensitivity", "responds to\nthe planner"),
              ("mean_evidence_sensitivity", "responds to\nits own evidence"),
              ("inert_rate", "inert"))
    figure, panels = plt.subplots(1, 2, figsize=(12.4, 5.6), sharey=True)
    width = 0.34
    for panel, holds, title in ((panels[0], True, "evidence holder"),
                                (panels[1], False, "bystander")):
        for (seed, colour), offset in zip(SEEDS, (-width / 2 - 0.012, width / 2 + 0.012)):
            row = next(r for r in audits[seed]["workers"]
                       if r["holds_contradicting_evidence"] is holds)
            for index, (key, _) in enumerate(fields):
                value = row.get(key)
                if value is None:
                    if seed == SEEDS[0][0]:
                        panel.annotate("no arm", (index, 0.03), ha="center",
                                       va="bottom", fontsize=12, color=GREY_DARK,
                                       style="italic")
                    continue
                panel.bar(index + offset, value, width=width, color=colour, zorder=2,
                          label=f"seed {seed} (n={row['agent_episodes']})"
                          if index == 0 else None)
                panel.annotate(f"{value:.2f}", (index + offset, value), ha="center",
                               textcoords="offset points", xytext=(0, 7),
                               fontsize=12, fontweight="bold", color=colour)
        panel.set_xticks(range(len(fields)))
        panel.set_xticklabels([label for _, label in fields], fontsize=13)
        panel.set_ylim(0, 1.2)
        panel.axhline(0, color=INK, linewidth=2)
        panel.set_title(title, fontsize=15, pad=10)
        if holds:
            panel.legend(loc="upper right", fontsize=11)
    panels[0].set_ylabel("rate over agent-episodes")
    figure.suptitle("Per-agent response, read through each worker's own messages",
                    fontsize=16, y=0.99)
    return save(figure, out / "fig-per-agent-seeds.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0", required=True)
    parser.add_argument("--seed1", required=True)
    parser.add_argument("-o", "--out", default="figures")
    arguments = parser.parse_args()
    audits = {0: json.loads(Path(arguments.seed0).read_text()),
              1: json.loads(Path(arguments.seed1).read_text())}
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    for written in (figure_deference(audits, out),
                    figure_worker_behaviour(audits, out),
                    figure_per_agent(audits, out)):
        print(written)


if __name__ == "__main__":
    main()
