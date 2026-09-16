"""Figures that lead with what this run actually established.

Companion to `plot_findings.py`, same conventions and same constraint: reads
`audit.json` only, so matplotlib stays out of `orchestrator/provenance/` and no
figure can change a number.

    /tmp/plotenv/bin/python scripts/plot_thesis.py \
        --audit <run>/audit/audit.json -o figures

Why a second script rather than more panels in the first. `plot_findings.py`
is organised around the observations O1-O11 in the order they were discovered.
These four are organised around what a reader needs to be convinced of, which
is a different order -- and the AUC, which the first script gives a full figure,
gets a quarter of one here because it is the weakest number in the run rather
than the headline.

Every title states what was measured, never what it means. A chart titled with
its conclusion is an argument wearing a chart's clothes.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

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


def figure_screen_artifact(audit, out):
    """The competence screen against the labels it was meant to be blind to.

    A screen is supposed to be independent of the outcome. This one was not:
    it excluded questions a solo agent could not answer, and deference lives on
    exactly those. The two bars are the same screen applied to the two classes,
    which is the whole finding -- a differential exclusion rate is what makes a
    screen an artifact rather than a filter.
    """
    counts = audit["_screen"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.4, 5.2),
                                         gridspec_kw={"width_ratios": [1.15, 1]})

    classes = ["deferred", "derived"]
    excluded = [counts["deferred_excluded"], counts["derived_excluded"]]
    totals = [counts["deferred"], counts["derived"]]
    x = range(len(classes))
    left.bar(x, totals, color=GREY_FILL, width=0.62, zorder=1)
    left.bar(x, excluded, color=RED, width=0.62, zorder=2)
    for index, (cut, total) in enumerate(zip(excluded, totals)):
        left.annotate(f"{cut}/{total}", (index, cut), ha="center",
                      textcoords="offset points", xytext=(0, 8),
                      fontsize=15, fontweight="bold", color=RED)
    left.set_xticks(list(x))
    left.set_xticklabels([f"{name}\nn={total}" for name, total in zip(classes, totals)])
    left.set_ylabel("seeded episodes")
    left.set_ylim(0, max(totals) * 1.32)
    left.axhline(0, color=INK, linewidth=2)
    left.set_title("Episodes the solo screen removed", fontsize=16, pad=12)
    left.legend(handles=[Patch(facecolor=RED, label="excluded by the screen"),
                         Patch(facecolor=GREY_FILL, label="retained")],
                loc="upper right", fontsize=13)

    rates = [counts["rate_screened"], counts["rate_covariate"]]
    labels = [f"screen applied\n{counts['deferred_kept']}/{counts['kept']}",
              f"screen as covariate\n{counts['deferred']}/{counts['total']}"]
    bars = right.bar(range(2), rates, color=[GREY_MID, BLUE], width=0.55, zorder=2)
    for bar, rate in zip(bars, rates):
        right.annotate(f"{rate:.0%}", (bar.get_x() + bar.get_width() / 2, rate),
                       ha="center", textcoords="offset points", xytext=(0, 8),
                       fontsize=16, fontweight="bold",
                       color=BLUE if rate == rates[1] else GREY_DARK)
    right.set_xticks(range(2))
    right.set_xticklabels(labels)
    right.set_ylabel("measured deference rate")
    right.set_ylim(0, max(rates) * 1.45)
    right.axhline(0, color=INK, linewidth=2)
    right.set_title("The rate each choice reports", fontsize=16, pad=12)
    return save(figure, out / "fig-screen-artifact.png")


def figure_channel_influence(audit, out):
    """Flip rate per perturbed channel, on episodes carrying both arms.

    The matched restriction is load-bearing and is stated in the title, not
    left to a caption: pooled over every episode each arm ran on, the planner
    ablation scores 0.23 and the worker control 0.47, which reads as workers
    mattering twice as much. That gap is a difference between subsets -- the
    control only runs where a worker contested -- and it closes on the matched
    set.
    """
    found = audit["channel_influence"]
    rows = found["arms"]
    colour = {"planner": BLUE, "worker": RED, "evidence": GREY_DARK}
    names = {"ablate_planner": "ablate planner",
             "strip_framing_planner": "strip framing",
             "paraphrase_sentences_planner": "paraphrase (sentence)",
             "ablate_worker_control": "ablate worker (control)",
             "ablate_pushback": "ablate pushback",
             "swap_evidence": "swap evidence"}
    rows = sorted(rows, key=lambda r: r["flip_rate"])
    figure, axes = plt.subplots(figsize=(9.2, 5.6))
    y = range(len(rows))
    axes.barh(y, [r["flip_rate"] for r in rows],
              color=[colour[r["channel"]] for r in rows], height=0.6, zorder=2)
    for index, row in zip(y, rows):
        axes.annotate(f"{row['flip_rate']:.2f}  ({row['flipped']}/{row['episodes']})",
                      (row["flip_rate"], index), va="center",
                      textcoords="offset points", xytext=(8, 0), fontsize=13,
                      color=INK)
    axes.set_yticks(list(y))
    axes.set_yticklabels([names[r["perturbation"]] for r in rows])
    axes.set_xlabel("episodes where the answer moved")
    axes.set_xlim(0, 0.86)
    axes.axvline(0, color=INK, linewidth=2)
    axes.set_title("Answer sensitivity by perturbed channel\n"
                   f"matched episodes, n={found['matched_episodes']}",
                   fontsize=16, pad=12)
    # Below the axes, not inside them: at these bar lengths every in-axes
    # corner collides with either a bar or its value label.
    axes.legend(handles=[Patch(facecolor=BLUE, label="planner channel"),
                         Patch(facecolor=RED, label="worker channel"),
                         Patch(facecolor=GREY_DARK, label="evidence (system prompt)")],
                loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=12)
    return save(figure, out / "fig-channel-influence.png")


def figure_per_agent(audit, out):
    """Each worker read through its own messages rather than the final answer.

    The final answer is the planner's composition, so it cannot say which agent
    behaved which way. The bystander has no evidence bar -- the swap was never
    about its paragraphs -- and an absent bar is drawn as absent rather than as
    zero, which would read as "insensitive to its evidence".
    """
    rows = {row["holds_contradicting_evidence"]: row for row in audit["workers"]}
    holder, bystander = rows[True], rows[False]
    figure, axes = plt.subplots(figsize=(9.0, 5.4))
    groups = ["responds to\nthe planner", "responds to\nits own evidence", "inert"]
    holder_v = [holder["mean_planner_sensitivity"],
                holder["mean_evidence_sensitivity"], holder["inert_rate"]]
    bystander_v = [bystander["mean_planner_sensitivity"], None, bystander["inert_rate"]]
    # A 2px-equivalent gap between adjacent bars: touching fills read as one
    # stacked mark rather than two measurements.
    width = 0.34
    for offset, values, colour, label in ((-width / 2 - 0.012, holder_v, BLUE, "evidence holder"),
                                          (width / 2 + 0.012, bystander_v, GREY_MID, "bystander")):
        for index, value in enumerate(values):
            if value is None:
                axes.annotate("no arm", (index + offset, 0.03), ha="center",
                              va="bottom", fontsize=12, color=GREY_DARK, style="italic")
                continue
            axes.bar(index + offset, value, width=width, color=colour, zorder=2,
                     label=label if index == 0 else None)
            axes.annotate(f"{value:.2f}", (index + offset, value), ha="center",
                          textcoords="offset points", xytext=(0, 7),
                          fontsize=13, fontweight="bold", color=colour)
    axes.set_xticks(range(len(groups)))
    axes.set_xticklabels(groups)
    axes.set_ylabel("rate over agent-episodes")
    axes.set_ylim(0, 1.12)
    axes.axhline(0, color=INK, linewidth=2)
    axes.set_title(f"Per-agent response, n={holder['agent_episodes']} agent-episodes each",
                   fontsize=16, pad=12)
    axes.legend(loc="upper right", fontsize=13)
    return save(figure, out / "fig-per-agent-role.png")


def figure_detection(audit, out):
    """Every AUC in the run against chance, with the counts that produced it.

    Drawn as intervals rather than bars on purpose. A bar chart of AUCs invites
    reading the heights against each other; what matters here is that every
    non-degenerate interval covers 0.5, and that the widest ones rest on four
    episodes against six.
    """
    keep = [{"stratum": row["name"], **row["roc"]} for row in audit["roc_strata"]
            if row["roc"].get("auc") is not None]
    keep = sorted(keep, key=lambda r: r["auc"])
    figure, axes = plt.subplots(figsize=(9.6, 5.8))
    y = range(len(keep))
    axes.axvspan(0.0, 1.0, color="white")
    axes.axvline(0.5, color=RED, linewidth=2, linestyle="--", zorder=1)
    for index, row in zip(y, keep):
        low, high = row.get("auc_low"), row.get("auc_high")
        degenerate = low is not None and high is not None and high - low < 1e-9
        colour = GREY_MID if degenerate else BLUE
        if low is not None and high is not None:
            axes.plot([low, high], [index, index], color=colour, linewidth=3,
                      solid_capstyle="butt", zorder=2)
        axes.plot([row["auc"]], [index], "o", color=colour, markersize=11, zorder=3)
        axes.annotate(f"{row['positives']}v{row['negatives']}", (1.02, index),
                      va="center", fontsize=12, color=GREY_DARK)
    axes.set_yticks(list(y))
    axes.set_yticklabels([row["stratum"] for row in keep], fontsize=13)
    axes.set_xlabel("AUC (95% CI)")
    axes.set_xlim(-0.02, 1.14)
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.set_title("Detection, every stratum against chance", fontsize=16, pad=12)
    axes.legend(handles=[
        Line2D([], [], color=RED, linestyle="--", linewidth=2, label="chance (0.5)"),
        Line2D([], [], color=BLUE, marker="o", linewidth=3, markersize=9,
               label="AUC, 95% CI"),
        Line2D([], [], color=GREY_MID, marker="o", linewidth=3, markersize=9,
               label="degenerate interval")],
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12)
    return save(figure, out / "fig-detection-honest.png")


def screen_counts(audit):
    """The solo screen's differential effect, from the episode rows.

    Computed here rather than read from a table because `competence` reports
    the screen's effect on *questions* and the artifact is about its effect on
    *labels* -- which class it removed, not how many it removed.
    """
    rows = [row for row in audit["episodes"] if row.get("label") is not None]
    deferred = [row for row in rows if row["label"] == 1]
    derived = [row for row in rows if row["label"] == 0]

    def cut(subset):
        return sum(1 for row in subset if row.get("solo_solved") is False)

    kept_deferred = len(deferred) - cut(deferred)
    kept = kept_deferred + (len(derived) - cut(derived))
    return {"deferred": len(deferred), "derived": len(derived),
            "deferred_excluded": cut(deferred), "derived_excluded": cut(derived),
            "deferred_kept": kept_deferred, "kept": kept,
            "total": len(deferred) + len(derived),
            "rate_screened": kept_deferred / kept if kept else 0.0,
            "rate_covariate": len(deferred) / (len(deferred) + len(derived))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, help="Path to audit.json")
    parser.add_argument("-o", "--out", default="figures")
    arguments = parser.parse_args()
    audit = json.loads(Path(arguments.audit).read_text())
    audit["_screen"] = screen_counts(audit)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    for written in (figure_screen_artifact(audit, out),
                    figure_channel_influence(audit, out),
                    figure_per_agent(audit, out),
                    figure_detection(audit, out)):
        print(written)


if __name__ == "__main__":
    main()
