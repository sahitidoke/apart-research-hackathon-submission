"""Render an audit's ROC curves as one self-contained HTML page.

    python -m orchestrator.provenance.plots --audit runs/prov-01/audit/audit.json

Stdlib only, same contract as `orchestrator/view_run.py`: no matplotlib, no
network, no build step. The output is a single file you can scp off a GPU box
and open, which matters because the machine that produces these numbers is
usually not the machine anybody reads them on.

What it draws, and why each one is separate:

* a small multiple of ROC curves -- overall, per hop count, per ladder arm, and
  one for the *per-agent* score. Pooling these would average the arm the audit
  is predicted to see straight through with the arm it is predicted to go blind
  on, and report neither.
* the AUCs side by side, against the 0.5 chance line.
* every scored episode as a dot, by label. At the sample sizes this experiment
  realistically reaches, a histogram would smooth away the fact that an AUC
  rests on a dozen points; the dots show you what it rests on.

A stratum with only one class present is drawn as an explicit "not computable"
panel. An empty axis, or a curve fitted to one class, would both read as a
result; this is the same rule the rest of the package follows for a
perturbation that did not apply.
"""
import argparse
import html
import json
from pathlib import Path

# Slots 1 and 2 of the reference categorical palette, in their fixed order, for
# both modes. Two series is the most any chart here carries, and colour is never
# the only channel: the dot strip is row-separated and directly labelled, and
# every chart has a table under it.
PALETTE = """
.viz{color-scheme:light;
 --surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--line:#e6e5e1;--grid:#efeee9;
 --s1:#2a78d6;--s2:#eb6834;--chance:#c3c2b7}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])) .viz{
 color-scheme:dark;
 --surface:#1a1a19;--ink:#fff;--ink-2:#c3c2b7;--line:#333330;--grid:#262624;
 --s1:#3987e5;--s2:#d95926;--chance:#52514e}}
:root[data-theme=dark] .viz{color-scheme:dark;
 --surface:#1a1a19;--ink:#fff;--ink-2:#c3c2b7;--line:#333330;--grid:#262624;
 --s1:#3987e5;--s2:#d95926;--chance:#52514e}
"""
CSS = PALETTE + """
*{box-sizing:border-box}
body{margin:0;padding:32px 24px;background:var(--surface);color:var(--ink);
 font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px;font-weight:600}
h2{font-size:15px;font-weight:600;margin:36px 0 4px}
p.note{color:var(--ink-2);margin:0 0 18px;max-width:68ch}
.tiles{display:flex;flex-wrap:wrap;gap:28px;margin:20px 0 8px;
 padding:16px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.tile .label{color:var(--ink-2);font-size:12px}
.tile .value{font-size:26px;font-weight:600;line-height:1.2}
.tile .value.hero{font-size:48px}
.grid{display:grid;gap:20px;grid-template-columns:repeat(auto-fill,minmax(250px,1fr))}
figure{margin:0}
figcaption{font-size:13px;font-weight:600;margin-bottom:2px}
figcaption .sub{display:block;font-weight:400;color:var(--ink-2);font-size:12px}
svg{display:block;width:100%;height:auto;overflow:visible}
.axis{stroke:var(--line);stroke-width:1;fill:none}
.grid-line{stroke:var(--grid);stroke-width:1;fill:none}
.tick{fill:var(--ink-2);font-size:10px;font-variant-numeric:tabular-nums}
.curve{fill:none;stroke:var(--s1);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chance{stroke:var(--chance);stroke-width:1;fill:none}
.dot{fill:var(--s1);stroke:var(--surface);stroke-width:2}
.dot.b{fill:var(--s2)}
.bar{fill:var(--s1)}
.hit{fill:transparent;cursor:default}
.empty{fill:var(--ink-2);font-size:11px}
.legend{display:flex;gap:16px;align-items:center;color:var(--ink-2);font-size:12px;
 margin:2px 0 8px}
.key{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;
 vertical-align:-1px}
details{margin-top:8px}
summary{cursor:pointer;color:var(--ink-2);font-size:12px}
table{border-collapse:collapse;margin-top:8px;font-size:12px;
 font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:3px 10px 3px 0;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .08s;
 background:var(--ink);color:var(--surface);padding:5px 8px;border-radius:5px;
 font-size:12px;white-space:nowrap;z-index:9}
"""
SCRIPT = """
(function(){
 var tip=document.getElementById('tip');
 function show(e){var t=e.target.getAttribute('data-tip');if(!t)return;
  tip.textContent=t;tip.style.opacity=1;
  var r=e.target.getBoundingClientRect();
  tip.style.left=Math.min(window.innerWidth-tip.offsetWidth-8,
    Math.max(8,r.left+r.width/2-tip.offsetWidth/2))+'px';
  tip.style.top=Math.max(8,r.top-tip.offsetHeight-8)+'px';}
 function hide(){tip.style.opacity=0;}
 document.addEventListener('mouseover',show);
 document.addEventListener('mouseout',hide);
 document.addEventListener('focusin',show);
 document.addEventListener('focusout',hide);
})();
"""
# Panel geometry. The box includes the axis bands, so the caption and ticks are
# never clipped by a fixed height.
W, H, PAD_L, PAD_B, PAD_T, PAD_R = 250, 250, 34, 30, 10, 10


def _e(text):
    return html.escape(str(text), quote=True)


def _fmt(value, places=3):
    return "—" if value is None else f"{value:.{places}f}".rstrip("0").rstrip(".")


def roc_panel(stratum):
    """One ROC curve, or an honest blank when only one class is present."""
    found = stratum["roc"]
    plot_w, plot_h = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def x(value):
        return PAD_L + value * plot_w

    def y(value):
        return PAD_T + (1 - value) * plot_h

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="ROC curve for {_e(stratum["name"])}">']
    for step in (0.25, 0.5, 0.75):
        parts.append(f'<line class="grid-line" x1="{x(step):.1f}" y1="{PAD_T}" '
                     f'x2="{x(step):.1f}" y2="{y(0):.1f}"/>')
        parts.append(f'<line class="grid-line" x1="{PAD_L}" y1="{y(step):.1f}" '
                     f'x2="{x(1):.1f}" y2="{y(step):.1f}"/>')
    parts.append(f'<line class="axis" x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{y(0):.1f}"/>')
    parts.append(f'<line class="axis" x1="{PAD_L}" y1="{y(0):.1f}" '
                 f'x2="{x(1):.1f}" y2="{y(0):.1f}"/>')
    # Chance, drawn under the curve as a hairline: solid, because a dashed rule
    # reads as a projection rather than a reference.
    parts.append(f'<line class="chance" x1="{PAD_L}" y1="{y(0):.1f}" '
                 f'x2="{x(1):.1f}" y2="{y(1):.1f}"/>')
    for value in (0, 0.5, 1):
        parts.append(f'<text class="tick" x="{x(value):.1f}" y="{y(0) + 13:.1f}" '
                     f'text-anchor="middle">{value}</text>')
        parts.append(f'<text class="tick" x="{PAD_L - 6}" y="{y(value) + 3:.1f}" '
                     f'text-anchor="end">{value}</text>')
    parts.append(f'<text class="tick" x="{x(0.5):.1f}" y="{H - 4}" '
                 f'text-anchor="middle">false positive rate</text>')
    parts.append(f'<text class="tick" transform="translate(10,{y(0.5):.1f}) rotate(-90)" '
                 f'text-anchor="middle">true positive rate</text>')

    if found["auc"] is None:
        parts.append(f'<text class="empty" x="{x(0.5):.1f}" y="{y(0.55):.1f}" '
                     'text-anchor="middle">not computable</text>')
        parts.append(f'<text class="empty" x="{x(0.5):.1f}" y="{y(0.42):.1f}" '
                     f'text-anchor="middle">{found["positives"]} deferred, '
                     f'{found["negatives"]} derived</text>')
        parts.append("</svg>")
        return "".join(parts)

    points = found["points"]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(p['fpr']):.1f},{y(p['tpr']):.1f}"
                    for i, p in enumerate(points))
    parts.append(f'<path class="curve" d="{path}"/>')
    for point in points:
        threshold = "" if point["threshold"] is None else \
            f"score ≥ {_fmt(point['threshold'], 2)} · "
        tip = (f"{threshold}TPR {_fmt(point['tpr'], 2)} · FPR {_fmt(point['fpr'], 2)}")
        # Hit area far larger than the mark, so a point is not a pinpoint target.
        parts.append(f'<circle class="hit" cx="{x(point["fpr"]):.1f}" '
                     f'cy="{y(point["tpr"]):.1f}" r="12" tabindex="0" '
                     f'data-tip="{_e(tip)}"/>')
        parts.append(f'<circle class="dot" cx="{x(point["fpr"]):.1f}" '
                     f'cy="{y(point["tpr"]):.1f}" r="4"/>')
    parts.append("</svg>")
    return "".join(parts)


def roc_table(stratum):
    rows = "".join(
        f"<tr><td>{_fmt(p['threshold'], 3) if p['threshold'] is not None else '—'}</td>"
        f"<td>{_fmt(p['fpr'], 3)}</td><td>{_fmt(p['tpr'], 3)}</td></tr>"
        for p in stratum["roc"]["points"])
    if not rows:
        # A stratum with only one class still gets a table, and it says so.
        # The panel beside it is deliberately blank, and a blank panel with no
        # table underneath is the one case where a reader who cannot use the
        # chart gets nothing at all -- which is the rule this file is built on.
        found = stratum["roc"]
        rows = ('<tr><td colspan="3">No curve: '
                f'{found["positives"]} deferred, {found["negatives"]} derived. '
                'A ROC needs both classes.</td></tr>')
    return ("<details><summary>Table</summary><table><thead><tr><th>threshold</th>"
            "<th>FPR</th><th>TPR</th></tr></thead><tbody>" + rows +
            "</tbody></table></details>")


def roc_figures(strata):
    figures = []
    for stratum in strata:
        found = stratum["roc"]
        # The interval rides with the point estimate, always. At four positives
        # against six negatives an AUC of 0.33 carries a 95% interval covering
        # two thirds of the range, and the bare number reads as a finding.
        interval = ""
        if found.get("auc_low") is not None:
            interval = (f" (95% CI {_fmt(found['auc_low'], 2)}–"
                        f"{_fmt(found['auc_high'], 2)})")
        subtitle = (f"AUC {_fmt(found['auc'], 3)}{interval} · {found['positives']} deferred, "
                    f"{found['negatives']} derived")
        figures.append(
            f'<figure><figcaption>{_e(stratum["name"])}'
            f'<span class="sub">{_e(subtitle)}</span></figcaption>'
            f'{roc_panel(stratum)}{roc_table(stratum)}</figure>')
    return f'<div class="grid">{"".join(figures)}</div>'


def auc_bars(strata):
    """Every AUC on one axis against the 0.5 chance rule.

    One series, one colour: the bars are the same quantity measured on
    different slices, so colouring them by value would burn the only free
    channel on what the bar length already says.
    """
    usable = [s for s in strata if s["roc"]["auc"] is not None]
    if not usable:
        return ('<p class="note">No stratum has both classes present, so there is '
                'no AUC to compare.</p>')
    row_h, gap, label_w, bar_max = 26, 8, 210, 420
    height = len(usable) * (row_h + gap) + 34
    width = label_w + bar_max + 60
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             'aria-label="AUC by stratum">']
    chance_x = label_w + 0.5 * bar_max
    for index, stratum in enumerate(usable):
        auc = stratum["roc"]["auc"]
        top = index * (row_h + gap)
        # Capped bar thickness, 4px rounded data-end, square at the baseline.
        bar_w = max(auc * bar_max, 0.001)
        parts.append(f'<text class="tick" x="{label_w - 10}" y="{top + 16}" '
                     f'text-anchor="end">{_e(stratum["name"])}</text>')
        parts.append(f'<rect class="bar" x="{label_w}" y="{top}" width="{bar_w:.1f}" '
                     f'height="18" rx="4"/>')
        parts.append(f'<rect class="bar" x="{label_w}" y="{top}" '
                     f'width="{min(4, bar_w):.1f}" height="18"/>')
        parts.append(f'<rect class="hit" x="{label_w}" y="{top - 4}" '
                     f'width="{bar_max}" height="26" tabindex="0" '
                     f'data-tip="{_e(stratum["name"])}: AUC {_fmt(auc, 3)} '
                     f'({stratum["roc"]["positives"]} deferred, '
                     f'{stratum["roc"]["negatives"]} derived)"/>')
        # The interval as a whisker through the bar end. A bar alone says the
        # difference between two strata is real; the overlap says whether it is.
        low, high = stratum["roc"].get("auc_low"), stratum["roc"].get("auc_high")
        if low is not None:
            x0, x1 = label_w + low * bar_max, label_w + high * bar_max
            parts.append(f'<line class="chance" x1="{x0:.1f}" y1="{top + 9}" '
                         f'x2="{x1:.1f}" y2="{top + 9}"/>')
            for edge in (x0, x1):
                parts.append(f'<line class="chance" x1="{edge:.1f}" y1="{top + 4}" '
                             f'x2="{edge:.1f}" y2="{top + 14}"/>')
        parts.append(f'<text class="tick" x="{label_w + bar_max + 8:.1f}" y="{top + 13}">'
                     f'{_fmt(auc, 3)}</text>')
    baseline = len(usable) * (row_h + gap)
    parts.append(f'<line class="chance" x1="{chance_x:.1f}" y1="-4" '
                 f'x2="{chance_x:.1f}" y2="{baseline}"/>')
    parts.append(f'<text class="tick" x="{chance_x:.1f}" y="{baseline + 14}" '
                 'text-anchor="middle">0.5 = chance</text>')
    parts.append(f'<line class="axis" x1="{label_w}" y1="{baseline}" '
                 f'x2="{label_w + bar_max}" y2="{baseline}"/>')
    parts.append("</svg>")
    rows = "".join(f"<tr><td>{_e(s['name'])}</td><td>{_fmt(s['roc']['auc'], 4)}</td>"
                   f"<td>{s['roc']['positives']}</td><td>{s['roc']['negatives']}</td></tr>"
                   for s in strata)
    table = ("<details><summary>Table</summary><table><thead><tr><th>stratum</th>"
             "<th>AUC</th><th>deferred</th><th>derived</th></tr></thead><tbody>"
             + rows + "</tbody></table></details>")
    return "".join(parts) + table


def score_strip(episodes):
    """Every scored episode as one dot, split by its label.

    A histogram at this sample size would smooth away how few points an AUC
    rests on. Two series, so the legend is always present -- and the rows are
    separated and labelled as well, because colour is never the only channel.
    """
    scored = [row for row in episodes
              if row.get("eligible") and row.get("label") is not None
              and row.get("score") is not None]
    if not scored:
        return '<p class="note">No labelled, eligible episode carries a score yet.</p>'
    width, pad_l, pad_r, row_h = 640, 96, 16, 46
    span = width - pad_l - pad_r
    parts = [f'<svg viewBox="0 0 {width} {row_h * 2 + 34}" role="img" '
             'aria-label="Provenance score by label">']
    for step in (0, 0.25, 0.5, 0.75, 1):
        position = pad_l + step * span
        parts.append(f'<line class="grid-line" x1="{position:.1f}" y1="6" '
                     f'x2="{position:.1f}" y2="{row_h * 2}"/>')
        parts.append(f'<text class="tick" x="{position:.1f}" y="{row_h * 2 + 18}" '
                     f'text-anchor="middle">{step}</text>')
    for index, (label, name, css) in enumerate(
            ((1, "Deferred", "dot"), (0, "Derived", "dot b"))):
        centre = 20 + index * row_h
        parts.append(f'<text class="tick" x="{pad_l - 12}" y="{centre + 4}" '
                     f'text-anchor="end">{name}</text>')
        for offset, row in enumerate(r for r in scored if r["label"] == label):
            # A gentle vertical fan so identical scores stay countable.
            jitter = (offset % 3 - 1) * 7
            position = pad_l + row["score"] * span
            tip = (f'{row["record_id"]} · score {_fmt(row["score"], 2)} · '
                   f'{row["hops"]}-hop · {name.lower()}')
            parts.append(f'<circle class="hit" cx="{position:.1f}" '
                         f'cy="{centre + jitter}" r="12" tabindex="0" '
                         f'data-tip="{_e(tip)}"/>')
            parts.append(f'<circle class="{css}" cx="{position:.1f}" '
                         f'cy="{centre + jitter}" r="5"/>')
    parts.append(f'<text class="tick" x="{pad_l + span / 2:.1f}" y="{row_h * 2 + 32}" '
                 'text-anchor="middle">provenance score · 0 = evidence-driven, '
                 '1 = planner-dictated</text>')
    parts.append("</svg>")
    legend = ('<div class="legend">'
              '<span><span class="key" style="background:var(--s1)"></span>'
              'Deferred (followed the planted error)</span>'
              '<span><span class="key" style="background:var(--s2)"></span>'
              'Derived (answered correctly anyway)</span></div>')
    rows = "".join(f"<tr><td>{_e(r['record_id'])}</td><td>{_fmt(r['score'], 3)}</td>"
                   f"<td>{r['hops']}</td>"
                   f"<td>{'deferred' if r['label'] else 'derived'}</td></tr>"
                   for r in scored)
    table = ("<details><summary>Table</summary><table><thead><tr><th>question</th>"
             "<th>score</th><th>hops</th><th>label</th></tr></thead><tbody>" + rows +
             "</tbody></table></details>")
    return legend + "".join(parts) + table


def tiles(result):
    overall = result["roc"]
    competence, health = result["competence"], result["health"]
    items = [("Overall AUC", _fmt(overall["auc"], 3), True),
             ("Labelled episodes",
              f'{overall["positives"]} deferred / {overall["negatives"]} derived', False),
             ("Questions retained",
              f'{competence.get("retained", 0)} of {competence["questions"]}', False),
             ("Replay divergence", _fmt(health["diverged_rate"], 3), False)]
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="label">{_e(label)}</div>'
        f'<div class="value{" hero" if hero else ""}">{_e(value)}</div></div>'
        for label, value, hero in items) + "</div>"


def render(result, title="Answer provenance audit"):
    strata = result.get("roc_strata") or []
    ungated = result["competence"].get("screens_not_run")
    warning = ""
    if ungated:
        warning = (f'<p class="note"><strong>These numbers are not fully gated.</strong> '
                   f'This run did not include {_e(", ".join(ungated))}, so no question '
                   'could be excluded on that screen.</p>')
    return f"""<!doctype html>
<html lang="en" class="viz"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>{CSS}</style></head>
<body class="viz"><main>
<h1>{_e(title)}</h1>
<p class="note">How well the provenance score separates episodes the collective
deferred on from ones it derived past, using only inter-agent messages and final
answers. Every curve is over eligible, labelled episodes only.</p>
{warning}
{tiles(result)}
<h2>ROC by stratum</h2>
<p class="note">Hop count is the difficulty axis; the ladder arms are the
conditions the audit is <em>predicted</em> to differ between, so a pooled curve
would report neither. The last panel scores the per-agent number over the worker
holding the contradicting evidence.</p>
{roc_figures(strata)}
<h2>AUC side by side</h2>
<p class="note">The same quantity on different slices, against chance. A high
deference rate with a chance-level AUC is dictation this audit cannot see, which
is a result and not a bug.</p>
{auc_bars(strata)}
<h2>What the AUC rests on</h2>
<p class="note">One dot per scored episode. Read the count before the curve.</p>
{score_strip(result.get("episodes") or [])}
</main><div id="tip" role="status"></div>
<script>{SCRIPT}</script></body></html>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, help="Path to audit.json")
    parser.add_argument("--output", default=None,
                        help="Defaults to plots.html beside the audit.json")
    arguments = parser.parse_args(argv)
    source = Path(arguments.audit)
    result = json.loads(source.read_text())
    output = Path(arguments.output or source.parent / "plots.html")
    output.write_text(render(result))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
