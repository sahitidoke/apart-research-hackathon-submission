"""Render a provenance audit's ROC curves with matplotlib.

**This is not part of `orchestrator/provenance/`.** That package is stdlib-only
outside its model wrapper, on purpose: a plumbing test that needs a scientific
stack stops being run, and `plots.py` hand-renders SVG so a run's results are
readable on the GPU box that produced them with no dependencies at all.

This script exists alongside it for the case that rule does not cover -- putting
a figure in a document -- and it earns its separation by reading **`audit.json`
and nothing else**. It never imports the package, so it cannot drag matplotlib
into the dependency graph, and it cannot change a number: everything it draws
was already computed by `audit.py`.

    uv venv /tmp/plotenv && /tmp/plotenv/bin/python -m pip install matplotlib
    /tmp/plotenv/bin/python scripts/plot_roc.py <run>/audit/audit.json -o figures/

Two figures, and the second is the one to read:

* **roc-curves** -- one panel per stratum with both classes present. Small
  multiples rather than five curves on one axes: at four positives against six
  negatives, overplotting invites a comparison between strata that the sample
  sizes cannot support.
* **roc-auc** -- every AUC with its 95% Hanley-McNeil interval against the 0.5
  chance rule. This is the honest headline. A point estimate of 0.33 quoted bare
  reads as "worse than chance"; with the interval on it, it reads as what it is.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

# Slots from the reference palette, validated for their own surfaces. The dark
# column is the same hue re-stepped for the dark band, never an automatic flip
# of the light one.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e",
              "series": "#2a78d6", "grid": "#e6e5e2", "rule": "#9a9892"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "muted": "#c3c2b7",
             "series": "#3987e5", "grid": "#33332f", "rule": "#6f6e68"},
}
# Strata worth a panel, in reading order. `directive / cooperative` is omitted
# deliberately: this run used exactly one prompt arm, so that curve is the
# Overall curve relabelled, and showing both would imply two measurements where
# there is one.
PANELS = ("Overall", "3-hop", "4-hop", "Evidence holder, per-agent score")
# The audit's stratum names are written to be unambiguous in a table, which
# makes them too long to sit over a 3-inch panel without colliding with the
# next one.
SHORT = {"Evidence holder, per-agent score": "Evidence holder (per agent)"}


def load(path):
    return json.loads(Path(path).read_text())


def curves(result):
    return {stratum["name"]: stratum for stratum in result.get("roc_strata") or []}


def _interval(found):
    if found.get("auc_low") is None:
        return ""
    return f"  [95% CI {found['auc_low']:.2f}–{found['auc_high']:.2f}]"


def _degenerate(found):
    """Perfect separation at tiny n collapses the Hanley-McNeil variance to
    zero, so the interval is a point and means nothing. It has to be labelled
    on the figure, not just in a caption nobody reads with it."""
    return (found.get("auc") in (0.0, 1.0)
            and found.get("auc_low") == found.get("auc_high"))


def roc_panel(axes, stratum, theme):
    found = stratum["roc"]
    points = found.get("points") or []
    axes.set_facecolor(theme["surface"])
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color(theme["grid"])
    axes.tick_params(colors=theme["muted"], labelsize=8, length=3)
    axes.grid(True, color=theme["grid"], linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    # Chance, solid rather than dashed: a dashed rule reads as a projection
    # rather than as a reference.
    axes.plot([0, 1], [0, 1], color=theme["rule"], linewidth=1, zorder=1)
    axes.set_xlim(-0.02, 1.02)
    axes.set_ylim(-0.02, 1.02)
    axes.set_xticks([0, 0.5, 1])
    axes.set_yticks([0, 0.5, 1])
    axes.set_aspect("equal")

    if not points:
        axes.text(0.5, 0.5, "not computable\n(one class only)", ha="center",
                  va="center", color=theme["muted"], fontsize=8.5)
    else:
        # Straight segments, not steps. The AUC here is Mann-Whitney with ties
        # counted as half, which *is* the trapezoidal area, so connecting the
        # operating points linearly is the shape the number was computed from.
        # Markers are on because at this sample size the count of distinct
        # operating points is itself part of the result.
        axes.plot([p["fpr"] for p in points], [p["tpr"] for p in points],
                  color=theme["series"], linewidth=2, marker="o", markersize=5,
                  markerfacecolor=theme["series"], markeredgecolor=theme["surface"],
                  markeredgewidth=1.5, zorder=3, solid_capstyle="round")

    # Title and caption are placed in axes coordinates rather than through
    # `set_title`, so the two caption lines stack instead of running into the
    # neighbouring panel -- which a single long line does at this panel width.
    axes.text(0, 1.28, SHORT.get(stratum["name"], stratum["name"]),
              transform=axes.transAxes, color=theme["ink"], fontsize=10.5,
              fontweight="medium", va="baseline")
    auc = found.get("auc")
    head = "AUC —" if auc is None else f"AUC {auc:.3f}"
    axes.text(0, 1.15, head, transform=axes.transAxes, color=theme["ink"],
              fontsize=9, va="baseline")
    tail = f"{found['positives']}v{found['negatives']} episodes"
    if found.get("auc_low") is not None:
        tail = f"95% CI {found['auc_low']:.2f}–{found['auc_high']:.2f}  ·  {tail}"
    axes.text(0, 1.04, tail, transform=axes.transAxes, color=theme["muted"],
              fontsize=8, va="baseline")


def figure_curves(result, theme_name, out):
    theme = THEMES[theme_name]
    found = curves(result)
    panels = [found[name] for name in PANELS if name in found]
    figure, axes = plt.subplots(1, len(panels), figsize=(2.75 * len(panels), 4.35))
    figure.patch.set_facecolor(theme["surface"])
    for axis, stratum in zip(axes, panels):
        roc_panel(axis, stratum, theme)
    axes[0].set_ylabel("true positive rate", color=theme["muted"], fontsize=8.5)
    for axis in axes:
        axis.set_xlabel("false positive rate", color=theme["muted"], fontsize=8.5)
    figure.text(0.012, 0.955, "Provenance score against seeded deference labels",
                color=theme["ink"], fontsize=13.5, ha="left", va="top",
                fontweight="semibold")
    figure.text(0.012, 0.895,
                "Each panel is one stratum. The diagonal is chance; markers are the "
                "actual operating points, and there are few.",
                color=theme["muted"], fontsize=9, ha="left", va="top")
    # Explicit margins rather than `tight_layout`: the captions live outside the
    # axes box, which tight_layout cannot see, so it collapses the space they
    # need.
    figure.subplots_adjust(left=0.075, right=0.988, top=0.70, bottom=0.115,
                           wspace=0.34)
    figure.savefig(out, dpi=200, facecolor=theme["surface"])
    plt.close(figure)
    return out


def figure_auc(result, theme_name, out):
    """Every AUC with its interval against the chance rule.

    A dot plot rather than bars: bars encode magnitude from a zero baseline, and
    an AUC's meaningful baseline is 0.5, not 0. Bars here would make 0.33 look
    like "a third of something" instead of "below chance".
    """
    theme = THEMES[theme_name]
    rows = [stratum for stratum in result.get("roc_strata") or []
            if stratum["name"] in PANELS]
    rows.reverse()
    figure, axes = plt.subplots(figsize=(8.6, 0.52 * len(rows) + 1.9))
    figure.patch.set_facecolor(theme["surface"])
    axes.set_facecolor(theme["surface"])
    for spine in ("top", "right", "left"):
        axes.spines[spine].set_visible(False)
    axes.spines["bottom"].set_color(theme["grid"])
    axes.tick_params(colors=theme["muted"], labelsize=9, length=3)
    axes.grid(True, axis="x", color=theme["grid"], linewidth=0.6)
    axes.set_axisbelow(True)
    axes.axvline(0.5, color=theme["rule"], linewidth=1.2, zorder=2)

    for index, stratum in enumerate(rows):
        found = stratum["roc"]
        auc = found.get("auc")
        if auc is None:
            axes.text(0.5, index, "  not computable — one class only",
                      va="center", ha="left", color=theme["muted"], fontsize=8.5)
            continue
        low, high = found.get("auc_low"), found.get("auc_high")
        if low is not None and not _degenerate(found):
            axes.plot([low, high], [index, index], color=theme["series"],
                      linewidth=2, solid_capstyle="round", alpha=0.42, zorder=3)
        axes.plot([auc], [index], marker="o", markersize=9,
                  color=theme["series"], markeredgecolor=theme["surface"],
                  markeredgewidth=1.6, zorder=4)
        note = "  (degenerate)" if _degenerate(found) else ""
        axes.text(1.035, index, f"{auc:.3f}{note}", va="center", ha="left",
                  color=theme["muted"] if note else theme["ink"], fontsize=9,
                  clip_on=False)

    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels(
        [f"{SHORT.get(s['name'], s['name'])}   "
         f"({s['roc']['positives']}v{s['roc']['negatives']})"
         for s in rows], color=theme["ink"], fontsize=9.5)
    axes.set_xlim(0, 1)
    axes.set_ylim(-0.55, len(rows) - 0.45)
    axes.set_xlabel("AUC  (0.5 = chance)", color=theme["muted"], fontsize=9)
    # Anchored to the figure, not the axes. `loc="left"` on a title is left of
    # the *plot box*, which long category labels push most of the way across the
    # canvas -- so a left-aligned title lands looking centred.
    # Descriptive rather than "no stratum separates from chance". That is the
    # reading, and it is the reason the figure exists, but a title that asserts
    # it is a conclusion the reader cannot check against the marks -- so it
    # belongs in the prose beside the figure.
    figure.text(0.012, 0.955, "AUC by stratum, against the chance rule",
                color=theme["ink"], fontsize=13.5, ha="left", va="top",
                fontweight="semibold")
    figure.text(0.012, 0.885,
                "Point estimate with its 95% Hanley–McNeil interval, over ten "
                "scored episodes.",
                color=theme["muted"], fontsize=9, ha="left", va="top")
    figure.text(0.012, 0.045,
                "Degenerate: at 2v2 with perfect inversion the Hanley–McNeil "
                "variance collapses to zero, so the interval is a point and "
                "carries no information.",
                color=theme["muted"], fontsize=8, ha="left", va="bottom")
    # The right margin holds the value column, including the longest note, so
    # it is sized for that text rather than left to `tight_layout`, which sizes
    # to the axes and clips anything drawn outside it.
    figure.subplots_adjust(left=0.29, right=0.785, top=0.76, bottom=0.245)
    figure.savefig(out, dpi=200, facecolor=theme["surface"])
    plt.close(figure)
    return out


def table(result):
    """The table view. A figure that cannot be read as text is not accessible,
    and these numbers are small enough that the text is often the better
    artifact anyway."""
    lines = ["| stratum | AUC | 95% CI | deferred | derived |",
             "|---|---:|---|---:|---:|"]
    for stratum in result.get("roc_strata") or []:
        found = stratum["roc"]
        auc = "—" if found.get("auc") is None else f"{found['auc']:.3f}"
        interval = ("—" if found.get("auc_low") is None
                    else f"{found['auc_low']:.2f}–{found['auc_high']:.2f}")
        lines.append(f"| {stratum['name']} | {auc} | {interval} | "
                     f"{found['positives']} | {found['negatives']} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_json")
    parser.add_argument("-o", "--output-dir", default="figures")
    parser.add_argument("--prefix", default="roc")
    arguments = parser.parse_args(argv)
    result = load(arguments.audit_json)
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for theme in THEMES:
        suffix = "" if theme == "light" else "-dark"
        written.append(figure_curves(
            result, theme, output / f"{arguments.prefix}-curves{suffix}.png"))
        written.append(figure_auc(
            result, theme, output / f"{arguments.prefix}-auc{suffix}.png"))
    (output / f"{arguments.prefix}-table.md").write_text(table(result) + "\n")
    for path in written:
        print(path)
    print(output / f"{arguments.prefix}-table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
