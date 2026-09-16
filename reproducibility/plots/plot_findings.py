"""Figures for the answer-provenance findings.

Like `plot_roc.py`, this lives outside `orchestrator/provenance/` and reads
`audit.json` only, so matplotlib never enters that package's dependency graph
and nothing here can change a number.

    /tmp/plotenv/bin/python scripts/plot_findings.py \
        --after <run>/audit-v2/audit.json --before <run>/audit/audit.json -o figures

Every figure is an **observation**. Interpretations live in the prose, not in a
title: a chart titled with its conclusion is an argument wearing a chart's
clothes, and the reader cannot check it against the marks.

Conventions follow the publication guidance in ChenLiu-1996/figures4papers --
sans-serif, no top/right spines, frameless legends, 300 dpi, `svg.fonttype`
none so vector text stays editable.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Palette from the same guidance. Grey carries anything that is context rather
# than result -- the three competence screens, and the "before" state of a
# comparison -- so emphasis is spent only on what is being claimed.
BLUE, BLUE_LIGHT = "#0F4D92", "#3775BA"
RED, GREEN, GOLD = "#B64342", "#8BCF8B", "#FFD700"
GREY_FILL, GREY_MID, GREY_DARK, INK = "#CFCECE", "#767676", "#4D4D4D", "#272727"

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


def figure_deference(after, out):
    """Deference per hop count, with both kinds of uncertainty.

    The Wilson interval is sampling error on the episodes that could be
    labelled. The attribution band is the separate question of what the
    *unlabelled* wrong answers were -- there are as many of those as labelled
    ones, so a point estimate alone would hide a 30-point range. Two different
    uncertainties, drawn differently, never pooled into one bar.
    """
    rows = after["deference"]
    pooled = after["deference_overall"]
    # The pooled point carries the headline. Without it the leftmost cell reads
    # 0.00 on two episodes and a reader can leave the figure believing the
    # phenomenon is absent. It is set apart by a rule rather than sitting in the
    # sequence, because it is a different object from a hop-count stratum.
    cells = rows + [pooled]
    labels = [f"{r['hops']}-hop\nn={r['episodes']}" for r in rows]
    labels.append(f"all hops\nn={pooled['episodes']}")
    x = range(len(cells))
    figure, axes = plt.subplots(figsize=(8.0, 5.4))

    for index, row in zip(x, cells):
        axes.plot([index, index], [row["attribution_low"], row["attribution_high"]],
                  color=GREY_FILL, linewidth=13, solid_capstyle="butt", zorder=1)
    colors = [BLUE_LIGHT] * len(rows) + [BLUE]
    for index, row, color in zip(x, cells, colors):
        axes.errorbar([index], [row["deference_rate"]],
                      yerr=[[row["deference_rate"] - row["ci_low"]],
                            [row["ci_high"] - row["deference_rate"]]],
                      fmt="o", markersize=13 if color == BLUE else 11,
                      color=color, ecolor=color, elinewidth=2.5,
                      capsize=8, capthick=2.5, zorder=3)
    axes.axvline(len(rows) - 0.5, color=GREY_MID, linewidth=1.5, zorder=0)
    axes.annotate(f"{pooled['deference_rate']:.2f}",
                  (len(rows), pooled["deference_rate"]), textcoords="offset points",
                  xytext=(16, -6), fontsize=15, color=BLUE, fontweight="bold")
    axes.axhline(0, color=INK, linewidth=2)
    axes.set_xticks(list(x))
    # Counts ride with the tick label rather than floating above the interval,
    # where at 3 hops they land on top of the legend.
    axes.set_xticklabels(labels)
    axes.set_xlim(-0.6, len(cells) - 0.25)
    axes.set_ylim(-0.03, 1.05)
    axes.set_ylabel("deference rate")
    axes.set_title("Deference under a seeded planner error", fontsize=16, pad=14)
    axes.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color=BLUE, markersize=11,
               label="rate, 95% Wilson CI"),
        Line2D([], [], color=GREY_FILL, linewidth=11,
               label="attribution bounds (unlabelled errors)")],
        loc="upper left", fontsize=12)
    return save(figure, out)


def figure_accuracy(after, out):
    """Exact match by condition and hop count.

    The three screens are drawn in grey on purpose. They decide which questions
    may enter the analysis and are never read as a comparison -- putting them in
    the same visual register as the collectives would invite exactly the
    comparison the design forbids.
    """
    series = {
        "closed_book": ("closed book (screen)", GREY_FILL, ":", "s"),
        "hop_probe": ("hop probe (screen)", GREY_FILL, ":", "^"),
        "isolated": ("isolated worker (screen)", GREY_MID, ":", "v"),
        "solo": ("solo, all paragraphs", RED, "-", "D"),
        "flat": ("flat pair, no planner", GREEN, "-", "o"),
        "planner": ("planner-led", BLUE, "-", "o"),
    }
    table = {}
    for row in after["accuracy"]:
        table.setdefault(row["condition"], {})[row["hops"]] = row["exact_match"]
    hops = sorted({row["hops"] for row in after["accuracy"]})
    figure, axes = plt.subplots(figsize=(7.8, 5.4))
    for condition, (label, color, style, marker) in series.items():
        values = [table.get(condition, {}).get(h) for h in hops]
        if not any(v is not None for v in values):
            continue
        axes.plot(hops, values, style, color=color, marker=marker, markersize=9,
                  linewidth=3 if style == "-" else 2.5, label=label,
                  markeredgecolor="white", markeredgewidth=1.2,
                  zorder=3 if style == "-" else 2)
    axes.set_xticks(hops)
    axes.set_xlabel("reasoning hops")
    axes.set_ylabel("exact match")
    axes.set_ylim(-0.03, 1.0)
    axes.set_title("Task competence by condition", fontsize=16, pad=14)
    axes.legend(fontsize=11.5, loc="upper right", ncol=1)
    return save(figure, out)


def _auc_rows(result, names):
    found = {s["name"]: s for s in result.get("roc_strata") or []}
    return [found[name] for name in names if name in found]


def figure_detection(before, after, out):
    """AUC before and after the instrument fix, with intervals.

    Paired on one axis because the question is whether the *same* strata moved.
    The chance rule is the reference the eye needs; without it an AUC chart
    invites reading 0.48 as "nearly half right" rather than "a coin".
    """
    names = ["Overall", "3-hop", "4-hop", "Evidence holder, per-agent score"]
    labels = ["Overall", "3-hop", "4-hop", "Evidence holder\n(per agent)"]
    pairs = list(zip(_auc_rows(before, names), _auc_rows(after, names)))
    figure, axes = plt.subplots(figsize=(8.6, 4.9))
    offset = 0.19
    for index, (old, new) in enumerate(pairs):
        for row, shift, color, marker in ((old, -offset, GREY_MID, "o"),
                                          (new, +offset, BLUE, "o")):
            found = row["roc"]
            if found["auc"] is None:
                continue
            low, high = found.get("auc_low"), found.get("auc_high")
            degenerate = low == high
            if low is not None and not degenerate:
                axes.plot([low, high], [index + shift] * 2, color=color,
                          linewidth=2.5, alpha=0.45, solid_capstyle="round", zorder=2)
            axes.plot([found["auc"]], [index + shift], marker=marker, markersize=12,
                      color=color, markeredgecolor="white", markeredgewidth=1.4,
                      zorder=4)
            if degenerate:
                axes.annotate("degenerate", (found["auc"], index + shift),
                              textcoords="offset points", xytext=(14, -4),
                              fontsize=11, color=GREY_DARK)
    axes.axvline(0.5, color=INK, linewidth=2, zorder=1)
    axes.set_yticks(range(len(pairs)))
    axes.set_yticklabels(labels, fontsize=13)
    axes.set_ylim(len(pairs) - 0.45, -0.55)
    axes.set_xlim(0, 1)
    axes.set_xlabel("AUC  (0.5 = chance)")
    axes.annotate("chance", (0.5, -0.5), textcoords="offset points",
                  xytext=(7, -16), fontsize=12, color=INK, va="bottom")
    axes.set_title("Detection of deference, before and after the fix",
                   fontsize=16, pad=14)
    # Below the axes: every row already has marks on both sides, so any in-plot
    # placement sits on data.
    axes.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=GREY_MID,
                                markersize=11, label="before  (ablation-only score)"),
                         Line2D([], [], marker="o", linestyle="none", color=BLUE,
                                markersize=11, label="after  (assembled score)")],
                fontsize=12, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.17))
    return save(figure, out, pad=1.4)


def figure_coverage(before, after, out):
    """How often each perturbation could actually be run.

    The number the first run could not see. An arm that could not run is not a
    finding that its edit did not matter, and pooling the two is how an audit
    reports a vacuous null as robustness.
    """
    names = ["ablate_planner", "swap_evidence", "paraphrase_planner",
             "paraphrase_sentences_planner", "strip_framing_planner",
             "ablate_pushback", "ablate_worker_control"]
    pretty = {"ablate_planner": "ablate planner",
              "swap_evidence": "swap evidence",
              "paraphrase_planner": "paraphrase (whole message)",
              "paraphrase_sentences_planner": "paraphrase (per sentence)",
              "strip_framing_planner": "strip framing",
              "ablate_pushback": "drop pushback",
              "ablate_worker_control": "drop ordinary message (control)"}
    old = {r["perturbation"]: r for r in before.get("arm_coverage") or []}
    new = {r["perturbation"]: r for r in after.get("arm_coverage") or []}
    figure, axes = plt.subplots(figsize=(9.4, 5.8))
    height = 0.36
    for index, name in enumerate(names):
        for source, shift, color, tag in ((old, +height / 2, GREY_MID, "before"),
                                          (new, -height / 2, BLUE, "after")):
            row = source.get(name)
            value = 0.0 if row is None else (row["coverage"] or 0.0)
            axes.barh(index + shift, value, height=height, color=color,
                      edgecolor="black", linewidth=1.5, zorder=3)
            if row is not None and row["applicable"]:
                axes.annotate(f"{row['ran']}/{row['applicable']}",
                              (value, index + shift), textcoords="offset points",
                              xytext=(6, -4), fontsize=11, color=GREY_DARK)
            elif row is None:
                axes.annotate("arm did not exist", (0, index + shift),
                              textcoords="offset points", xytext=(6, -4),
                              fontsize=11, color=GREY_DARK, style="italic")
    axes.set_yticks(range(len(names)))
    axes.set_yticklabels([pretty[n] for n in names], fontsize=12.5)
    axes.invert_yaxis()
    axes.set_xlim(0, 1.18)
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.set_xlabel("share of episodes where the arm could be run")
    axes.set_title("Perturbation coverage, before and after the fix",
                   fontsize=16, pad=14)
    # Explicit patch handles: an empty `barh` draws nothing for matplotlib to
    # take a colour from, so the legend silently falls back to the default
    # cycle and both swatches come out the same.
    axes.legend(handles=[Patch(facecolor=BLUE, edgecolor="black", label="after"),
                         Patch(facecolor=GREY_MID, edgecolor="black", label="before")],
                fontsize=12, loc="lower right")
    return save(figure, out)


def figure_worker_behaviour(after, out):
    """Contesting and propagating, per hop count.

    Drawn as grouped bars over the same episodes on purpose: these are not
    exclusive outcomes, and the whole point is that the same collectives do both.
    """
    rows = after["deference"]
    hops = [r["hops"] for r in rows]
    bars = [("worker_pushed_back_rate", "contested the planner", BLUE),
            ("worker_restated_evidence_rate", "restated the true fact", BLUE_LIGHT),
            ("worker_repeated_error_rate", "repeated the planted error", RED)]
    figure, axes = plt.subplots(figsize=(8.4, 5.4))
    width = 0.26
    for offset, (key, label, color) in zip((-width, 0, width), bars):
        axes.bar([i + offset for i in range(len(rows))], [r[key] for r in rows],
                 width=width, color=color, edgecolor="black", linewidth=1.5,
                 label=label, zorder=3)
    axes.set_xticks(range(len(rows)))
    axes.set_xticklabels([f"{h}-hop\n(n={r['episodes']})" for h, r in zip(hops, rows)])
    axes.set_ylim(0, 1.18)
    axes.set_ylabel("share of eligible episodes")
    # Descriptive, not "workers contest and propagate at once". That reading is
    # the point of the figure, but a title that states it is an argument the
    # reader cannot check against the marks.
    axes.set_title("Worker response to the planted error", fontsize=16, pad=14)
    axes.legend(fontsize=12, loc="upper center", ncol=3,
                bbox_to_anchor=(0.5, -0.16))
    return save(figure, out, pad=1.4)


def figure_agents(after, out):
    """What each worker responds to, read through its own messages.

    Not the final answer -- the planner composes that. This is the closest the
    battery gets to "which agent behaved which way".
    """
    rows = {row["holds_contradicting_evidence"]: row for row in after["workers"]}
    holder, bystander = rows.get(True), rows.get(False)
    groups = ["responds to\nthe planner", "responds to\nits own evidence",
              "inert\n(nothing moved it)"]
    holder_values = [holder["mean_planner_sensitivity"],
                     holder["mean_evidence_sensitivity"], holder["inert_rate"]]
    bystander_values = [bystander["mean_planner_sensitivity"], None,
                        bystander["inert_rate"]]
    figure, axes = plt.subplots(figsize=(8.0, 5.4))
    width = 0.34
    for offset, values, color, label in ((-width / 2, holder_values, BLUE,
                                          "holds the contradicting evidence"),
                                         (+width / 2, bystander_values, GREY_MID,
                                          "bystander")):
        for index, value in enumerate(values):
            if value is None:
                axes.annotate("n/a", (index + offset, 0.02), ha="center",
                              fontsize=11.5, color=GREY_DARK, style="italic")
                continue
            axes.bar(index + offset, value, width=width, color=color,
                     edgecolor="black", linewidth=1.5, zorder=3,
                     label=label if index == 0 else None)
            axes.annotate(f"{value:.2f}", (index + offset, value),
                          textcoords="offset points", xytext=(0, 6), ha="center",
                          fontsize=12, color=GREY_DARK)
    axes.set_xticks(range(len(groups)))
    axes.set_xticklabels(groups, fontsize=12.5)
    axes.set_ylim(0, 1.12)
    axes.set_ylabel("rate over agent-episodes")
    axes.set_title("Per agent, read through its own messages", fontsize=16, pad=14)
    axes.legend(fontsize=12, loc="upper right")
    return save(figure, out, pad=1.6)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("-o", "--output-dir", default="figures")
    arguments = parser.parse_args(argv)
    after = json.loads(Path(arguments.after).read_text())
    before = json.loads(Path(arguments.before).read_text())
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written = [
        figure_deference(after, output / "fig1-deference.png"),
        figure_accuracy(after, output / "fig2-competence.png"),
        figure_worker_behaviour(after, output / "fig3-worker-behaviour.png"),
        figure_agents(after, output / "fig4-per-agent.png"),
        figure_detection(before, after, output / "fig5-detection.png"),
        figure_coverage(before, after, output / "fig6-coverage.png"),
    ]
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
