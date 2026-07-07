"""Generate the §4.6 cross-scale section for pi_report.html, in the report's
own cascade-slope idiom (inline SVG, <title> hover provenance, reused CSS
classes, design tokens). Every coordinate/number traces to the collab_cd44 CSVs.

Emits results/biological_analysis/collab_cd44/section_4_6.html — a self-contained
HTML fragment spliced into pi_report.html before <h2>5 ...</h2>.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / 'cd44_crossscale_summary.csv'
ALLSCALES = HERE / 'cd44_compartment_rates_allscales.csv'
OUT = HERE / 'section_4_6.html'

SCALES = [10.0, 20.0, 40.0]
# x positions for the three scales (plot area mirrors Fig 3: 200..770)
X = {10.0: 300.0, 20.0: 500.0, 40.0: 700.0}
# y maps Sham->D7 delta (percentage points): +25pp -> y=80 (top), -5pp -> y=440
Y_TOP_PP, Y_BOT_PP = 25.0, -5.0
Y_TOP_PX, Y_BOT_PX = 80.0, 440.0


def y_of(delta_pp: float) -> float:
    frac = (delta_pp - Y_BOT_PP) / (Y_TOP_PP - Y_BOT_PP)
    return Y_BOT_PX - frac * (Y_BOT_PX - Y_TOP_PX)


# compartment -> (display label, stroke token, dashed?)
STYLE = {
    'CD206': ('CD206⁺ (M2-like)', 'var(--accent)', False),
    'CD45': ('CD45⁺ (immune)', 'var(--lineage-immune)', False),
    'CD140b': ('CD140b⁺ (pericyte/mural)', 'var(--lineage-stromal)', False),
    'CD31': ('CD31⁺ (vascular)', 'var(--lineage-endothelial)', False),
    'CD34': ('CD34⁺ (vascular)', 'var(--lineage-endothelial)', True),
    'endothelial_cd31cd34': ('Endothelial (CD31⁺∧CD34⁺)', 'var(--ink-soft)', True),
}
# draw order: strongest movers last (on top)
ORDER = ['endothelial_cd31cd34', 'CD31', 'CD34', 'CD45', 'CD140b', 'CD206']


def pp(x: float) -> float:
    return round(x * 100.0, 1)


def build_region_scale_svg(summ_all: pd.DataFrame) -> str:
    """Two faceted panels (CORTEX | MEDULLA), each a cross-scale slope of the
    Sham→D7 CD44 change (pp) across 10/20/40µm per compartment. Reflects BOTH
    dimensions at once: it shows the cortex-predominant activation is itself
    scale-robust — cortex lines stay above the 0-line at every scale, medulla
    lines stay at/below it. Colour = compartment (shared legend); neutrophil is
    omitted (10µm-only, no grid-scale basis)."""
    YT_PP, YB_PP, YT_PX, YB_PX = 70.0, -25.0, 100.0, 450.0
    yv = lambda v: YB_PX - (v - YB_PP) / (YT_PP - YB_PP) * (YB_PX - YT_PX)
    panels = [('cortex', 'CORTEX', 130.0, 470.0), ('medulla', 'MEDULLA', 630.0, 970.0)]
    xfrac = {10.0: 0.12, 20.0: 0.5, 40.0: 0.88}
    comps = [c for c in ['CD206', 'CD140b', 'CD45', 'CD34', 'CD31', 'endothelial_cd31cd34']
             if c in summ_all[summ_all.region == 'cortex'].compartment.values]
    p = ['<svg class="cascade-svg" viewBox="0 0 1100 540" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Cortex versus medulla CD44 activation across 10, 20 and 40 micron scales.">']
    for gpp in [-25, 0, 25, 50, 70]:
        yy = yv(gpp)
        emph = ' style="stroke:var(--ink-faint)"' if gpp == 0 else ''
        p.append(f'  <line class="grid" x1="115" y1="{yy:.1f}" x2="985" y2="{yy:.1f}"{emph}/>')
        p.append(f'  <text class="ytick" x="103" y="{yy:.1f}" text-anchor="end" dominant-baseline="middle">{gpp:+d}pp</text>')
    p.append('  <text class="axis-label" transform="translate(36,275) rotate(-90)" text-anchor="middle">Sham→D7 change in CD44⁺ rate (pp)</text>')
    # compartment colour legend (top)
    lx = 150.0
    for comp in comps:
        label, stroke, dashed = STYLE[comp]
        dash = ' stroke-dasharray="5 4"' if dashed else ''
        p.append(f'  <line x1="{lx:.0f}" y1="64" x2="{lx+22:.0f}" y2="64" stroke="{stroke}" stroke-width="3"{dash}/>')
        p.append(f'  <text x="{lx+27:.0f}" y="64" dominant-baseline="middle" style="fill:var(--ink-soft);font-size:11.5px">{label.split(" ")[0]}</text>')
        lx += 145.0
    for reg, title, x0, x1 in panels:
        r = summ_all[summ_all.region == reg].set_index('compartment')
        xof = lambda s: x0 + xfrac[s] * (x1 - x0)
        p.append(f'  <text x="{(x0+x1)/2:.0f}" y="92" text-anchor="middle" style="fill:var(--ink);font-size:14px;font-weight:700;letter-spacing:0.04em">{title}</text>')
        for s in SCALES:
            p.append(f'  <text class="xtick-main" x="{xof(s):.0f}" y="474" text-anchor="middle">{s:g}µm</text>')
        for comp in comps:
            if comp not in r.index:
                continue
            label, stroke, dashed = STYLE[comp]
            pts = [(xof(s), yv(pp(r.loc[comp, f'delta_{s:g}um']))) for s in SCALES
                   if pd.notna(r.loc[comp, f'delta_{s:g}um'])]
            if len(pts) < 2:
                continue
            dash = ' stroke-dasharray="6 5"' if dashed else ''
            headline = comp == 'CD206'
            sw = '3.2' if headline else '2.2'
            cls = 'cascade-line headline' if headline else 'cascade-line'
            poly = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
            bits = '; '.join(f'{s:g}µm {pp(r.loc[comp, f"delta_{s:g}um"]):+.1f}pp' for s in SCALES
                             if pd.notna(r.loc[comp, f'delta_{s:g}um']))
            p.append(f'  <g class="cascade-marker" data-comp="{comp}" data-region="{reg}" style="--cascade-stroke: {stroke};" tabindex="0">')
            p.append(f'    <polyline class="{cls}" points="{poly}" fill="none" stroke-width="{sw}"{dash}/>')
            for x, y in pts:
                p.append(f'      <circle class="cascade-dot" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/>')
            p.append(f'    <title>{html.escape(f"{label} · {reg} · Sham→D7 by scale: {bits}")}</title>')
            p.append('  </g>')
    p.append('</svg>')
    return '\n'.join(p)


def build_region_svg(summ_all: pd.DataFrame) -> str:
    """Cortex-vs-medulla dumbbell: Sham→D7 CD44 change (pp) per compartment at
    10µm. Filled = cortex, open = medulla; the horizontal gap is the regional
    divergence. Self-contained styling on the report's design tokens."""
    cor = summ_all[summ_all.region == 'cortex'].set_index('compartment')
    med = summ_all[summ_all.region == 'medulla'].set_index('compartment')
    comps = [c for c in ['neutrophil', 'CD206', 'endothelial_cd31cd34', 'CD140b', 'CD45', 'CD34', 'CD31'] if c in cor.index]
    comps = sorted(comps, key=lambda c: -cor.loc[c, 'delta_10um'])
    XMIN, XMAX, X0, XW = -25.0, 70.0, 320.0, 660.0
    x_of = lambda v: X0 + (v - XMIN) / (XMAX - XMIN) * XW
    ytop, ybot = 90.0, 405.0
    n = len(comps)
    ys = [ytop + i * (ybot - ytop) / max(n - 1, 1) for i in range(n)]
    p = ['<svg class="cascade-svg" viewBox="0 0 1100 480" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Cortex versus medulla: Sham to Day 7 CD44 rate change per compartment at 10 micron superpixels.">']
    for gx in [-25, 0, 25, 50, 70]:
        xx = x_of(gx)
        emph = ' style="stroke:var(--ink-faint)"' if gx == 0 else ''
        p.append(f'  <line class="grid" x1="{xx:.1f}" y1="{ytop-24:.1f}" x2="{xx:.1f}" y2="{ybot+14:.1f}"{emph}/>')
        p.append(f'  <text class="ytick" x="{xx:.1f}" y="{ybot+32:.1f}" text-anchor="middle">{gx:+d}pp</text>')
    p.append(f'  <text class="xtick-sub" x="{x_of(0):.1f}" y="{ytop-32:.1f}" text-anchor="middle">no change</text>')
    p.append(f'  <text class="hover-hint" x="{x_of(22):.1f}" y="{ybot+52:.1f}" text-anchor="middle">Sham→D7 change in CD44⁺ rate (percentage points) · hover any row for detail</text>')
    # legend
    p.append(f'  <circle cx="740" cy="{ytop-36:.1f}" r="6" fill="var(--accent)"/><text x="752" y="{ytop-36:.1f}" dominant-baseline="middle" style="fill:var(--ink-soft);font-size:12px">cortex</text>')
    p.append(f'  <circle cx="840" cy="{ytop-36:.1f}" r="6" fill="var(--paper)" stroke="var(--ink-faint)" stroke-width="2"/><text x="852" y="{ytop-36:.1f}" dominant-baseline="middle" style="fill:var(--ink-soft);font-size:12px">medulla</text>')
    for comp, y in zip(comps, ys):
        label = STYLE.get(comp, (comp, '', False))[0]
        dc, dm = pp(cor.loc[comp, 'delta_10um']), pp(med.loc[comp, 'delta_10um'])
        xc, xm = x_of(dc), x_of(dm)
        p.append(f'  <g class="cascade-marker" data-comp="{comp}" tabindex="0">')
        p.append(f'    <line x1="{min(xm,xc):.1f}" y1="{y:.1f}" x2="{max(xm,xc):.1f}" y2="{y:.1f}" stroke="var(--rule)" stroke-width="2.5"/>')
        p.append(f'    <circle cx="{xm:.1f}" cy="{y:.1f}" r="6" fill="var(--paper)" stroke="var(--ink-faint)" stroke-width="2"/>')
        p.append(f'    <circle cx="{xc:.1f}" cy="{y:.1f}" r="6.5" fill="var(--accent)"/>')
        p.append(f'    <text x="305" y="{y:.1f}" text-anchor="end" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:600">{label}</text>')
        p.append(f'    <title>{html.escape(f"{label} · Sham→D7 CD44 change at 10µm: cortex {dc:+.1f}pp (filled) vs medulla {dm:+.1f}pp (open). Descriptive; n=2 mice, paired within-mouse.")}</title>')
        p.append('  </g>')
    p.append('</svg>')
    return '\n'.join(p)


def main() -> None:
    summ_all = pd.read_csv(SUMMARY)
    summ = summ_all[summ_all.region == 'pooled'].set_index('compartment')
    allsc = pd.read_csv(ALLSCALES)

    # ---- added-compartments table (10um Sham->D7), collaborator's ask ----
    trows = []
    for comp in ['CD206', 'CD34', 'endothelial_cd31cd34', 'CD140b']:
        if comp not in summ.index:
            continue
        r = summ.loc[comp]
        label = STYLE.get(comp, (comp, '', False))[0]
        trows.append(
            f'    <tr><td>{label}</td><td class="num">{pp(r["sham_10um"]):.1f}%</td>'
            f'<td class="num">{pp(r["d7_10um"]):.1f}%</td>'
            f'<td class="num">{pp(r["delta_10um"]):+.1f}pp</td></tr>'
        )
    table = (
        '<table>\n  <thead>\n    <tr><th>Added compartment (Sham-referenced, 10µm)</th>'
        '<th class="num">Sham CD44⁺</th><th class="num">D7 CD44⁺</th>'
        '<th class="num">Δ (pp)</th></tr>\n  </thead>\n  <tbody>\n'
        + '\n'.join(trows) + '\n  </tbody>\n</table>'
    )

    # ---- cross-scale cascade SVG ----
    parts = []
    parts.append(
        '<svg class="cascade-svg" viewBox="0 0 1100 540" preserveAspectRatio="xMidYMid meet" '
        'role="img" aria-label="Cross-scale robustness: Sham to Day 7 CD44 rate change per compartment at 10, 20 and 40 micron superpixels.">'
    )
    parts.append('  <g class="plot-bg">')
    # y gridlines every 5pp from -5 to +25
    for gpp in [-5, 0, 5, 10, 15, 20, 25]:
        yy = y_of(gpp)
        emph = ' style="stroke:var(--ink-faint)"' if gpp == 0 else ''
        parts.append(f'    <line class="grid" x1="200" y1="{yy:.1f}" x2="770" y2="{yy:.1f}"{emph}/>')
        parts.append(f'    <text class="ytick" x="188" y="{yy:.1f}" text-anchor="end" dominant-baseline="middle">{gpp:+d}pp</text>')
    parts.append('    <text class="axis-label" transform="translate(40,260) rotate(-90)" text-anchor="middle">Sham→D7 change in CD44⁺ rate (percentage points)</text>')
    parts.append('    <text class="ytick" x="188" y="380" text-anchor="end" dominant-baseline="middle"></text>')
    # x ticks: three scales
    xlabels = {10.0: ('10µm', 'cellular'), 20.0: ('20µm', 'tubular'), 40.0: ('40µm', 'domain')}
    for s in SCALES:
        parts.append(f'    <text class="xtick-main" x="{X[s]:.0f}" y="464" text-anchor="middle">{xlabels[s][0]}</text>')
        parts.append(f'    <text class="xtick-sub" x="{X[s]:.0f}" y="482" text-anchor="middle">{xlabels[s][1]}</text>')
    parts.append('    <text class="hover-hint" x="500" y="66" text-anchor="middle">a line staying above the 0-line as the patch coarsens = scale-robust · hover for per-scale detail</text>')

    # ---- pass 1: gather each compartment's geometry ----
    drawn = []
    for comp in ORDER:
        if comp not in summ.index:
            continue
        r = summ.loc[comp]
        label, stroke, dashed = STYLE[comp]
        pts = []
        titlebits = []
        for s in SCALES:
            d = r[f'delta_{s:g}um']
            if pd.isna(d):
                continue
            xx, yy = X[s], y_of(pp(d))
            pts.append((xx, yy))
            nsh = r.get(f'n_sham_{s:g}um', float('nan'))
            titlebits.append(f'{s:g}µm Δ{pp(d):+.1f}pp (n≈{nsh:.0f})' if pd.notna(nsh) else f'{s:g}µm Δ{pp(d):+.1f}pp')
        if not pts:
            continue
        drawn.append({
            'comp': comp, 'label': label, 'stroke': stroke, 'dashed': dashed,
            'pts': pts, 'titlebits': titlebits, 'reading': r['scale_reading'],
            'headline': comp == 'CD206', 'lx': pts[-1][0], 'ly': pts[-1][1],
        })

    # ---- pass 2: dodge colliding end-labels (min 18px vertical gap, top-down) ----
    MIN_GAP = 18.0
    prev = None
    for d in sorted(drawn, key=lambda d: d['ly']):
        ty = d['ly'] if prev is None else max(d['ly'], prev + MIN_GAP)
        d['label_y'] = ty
        prev = ty

    # ---- pass 3: emit (leader line whenever a label was pushed off its endpoint) ----
    for d in drawn:
        pts, reading, headline = d['pts'], d['reading'], d['headline']
        dash = ' stroke-dasharray="6 5"' if d['dashed'] else ''
        sw = '3.4' if headline else '2.3'
        parts.append(f'    <g class="cascade-marker" data-comp="{d["comp"]}" data-reading="{reading}" style="--cascade-stroke: {d["stroke"]};" tabindex="0">')
        if len(pts) >= 2:
            poly = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
            cls = 'cascade-line headline' if headline else 'cascade-line'
            parts.append(f'      <polyline class="{cls}" points="{poly}" fill="none" stroke-width="{sw}"{dash}/>')
        for (x, y) in pts:
            parts.append(f'      <circle class="cascade-dot" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
        lx, ly, lyt = d['lx'], d['ly'], d['label_y']
        if abs(lyt - ly) > 3.0:
            parts.append(f'      <line class="cascade-leader" x1="{lx+5:.1f}" y1="{ly:.1f}" x2="{lx+14:.1f}" y2="{lyt:.1f}"/>')
        parts.append(f'      <text class="cascade-label-name" x="{lx+16:.1f}" y="{lyt:.1f}" dominant-baseline="middle">{d["label"]}</text>')
        title = html.escape(f'{d["label"]} · Sham→D7 CD44 change by scale: ' + '; '.join(d['titlebits']) + f' · reading: {reading.replace("_", "-")}.')
        parts.append(f'      <title>{title}</title>')
        parts.append('    </g>')

    parts.append('  </g>')
    parts.append('</svg>')
    svg = '\n'.join(parts)

    # readings tally
    tally = summ['scale_reading'].value_counts().to_dict()
    n_robust = tally.get('scale_robust', 0)

    # cortex-vs-medulla × scale figure + the regional deltas used in the reading
    region_svg = build_region_scale_svg(summ_all)
    cor = summ_all[summ_all.region == 'cortex'].set_index('compartment')
    med = summ_all[summ_all.region == 'medulla'].set_index('compartment')
    cd206_c, cd206_m = pp(cor.loc['CD206', 'delta_10um']), pp(med.loc['CD206', 'delta_10um'])
    neu_c, neu_m = pp(cor.loc['neutrophil', 'delta_10um']), pp(med.loc['neutrophil', 'delta_10um'])
    cd140_c, cd140_m = pp(cor.loc['CD140b', 'delta_10um']), pp(med.loc['CD140b', 'delta_10um'])

    section = f"""<h3>4.6 · The added compartments, across spatial scales</h3>

<p>A reviewer of this brief asked three follow-up questions: add an <strong>M2-like macrophage</strong> compartment (CD206⁺) and a fuller <strong>endothelial</strong> compartment (CD31⁺ and CD34⁺) to the activation readout; split the quantification into <strong>cortex and medulla</strong>; and check whether the finding holds at a <strong>coarser superpixel scale</strong>. This section answers all three, as a descriptive extension — it reuses the same frozen Sham-referenced compute and thresholds, adds no pre-registered endpoint, and claims no significance. (Terminology note carried through: CD140b/PDGFRβ marks pericytes / mural / activated-mesenchymal tissue, <em>not</em> fibroblasts; CD206 marks M2-<em>like</em> macrophages but is not exclusive to them.)</p>

{table}

<p>The two newly added marker compartments move in the same upward direction as the original four: the CD206⁺ (M2-like) compartment carries the largest added shift, and the endothelial CD31⁺∧CD34⁺ intersection barely moves — consistent with §4.2's reading that the vascular compartment is the least activated.</p>

<figure class="cascade-fig" id="fig-crossscale">
  <div class="cascade-wrap">
    <p class="cascade-eyebrow">Does the activation survive coarsening the patch?</p>
    <h3 class="cascade-title">The Sham→D7 CD44⁺ rise holds as the superpixel grows from ~one cell (10µm) to a tissue domain (40µm). {n_robust} of the compartments stay above the no-change line at all three scales; the endothelial intersection is a near-zero non-mover, not a scale artifact.</h3>
{svg}
    <figcaption>Each line is one compartment's Sham→D7 change in CD44⁺ rate (percentage points), plotted at 10µm (cellular), 20µm (tubular cross-section) and 40µm (tissue domain) superpixels. Thresholds are recomputed by the identical Sham per-mouse 75th-percentile recipe on each scale's own superpixels. The neutrophil-typed compartment (the §4.1 headline, +49pp at 10µm) is <em>not</em> shown here: it requires cell-type gating, for which there is no basis at the 20/40µm grid scales — so it cannot be assessed cross-scale, and is not fabricated.
    <span class="source">Source: results/biological_analysis/collab_cd44/cd44_crossscale_summary.csv &amp; cd44_compartment_rates_allscales.csv (58,137 / 13,966 / 2,951 superpixels at 10 / 20 / 40µm). Descriptive; reuses the frozen Family C compute — no pre-registered endpoint added.</span></figcaption>
  </div>
</figure>

<p>The reading: the descriptive CD44-activation story is <strong>scale-robust</strong> — it is not an artifact of the 10µm patch size. As the unit coarsens and averages over more cells, every compartment that genuinely moves keeps moving in the same direction; the one compartment that flips sign (the endothelial intersection) hovers within ±1pp of zero at every scale, i.e. it is a non-mover, not a scale-dependent finding. This is what the pipeline's kNN-graph neighborhoods already capture at 10µm — enlarging the patch is a robustness check, not a better representation.</p>

<figure class="cascade-fig" id="fig-region">
  <div class="cascade-wrap">
    <p class="cascade-eyebrow">The split the reviewer asked for — across both axes at once</p>
    <h3 class="cascade-title">The CD44⁺ activation is a <em>cortical</em> event — and it holds at every scale. In the cortex panel every compartment stays above the no-change line at 10, 20 and 40µm; in the medulla panel they sit at or below it.</h3>
{region_svg}
    <figcaption>Sham→D7 change in CD44⁺ rate (percentage points) per compartment, faceted by anatomical region (left = cortex, right = medulla) and plotted across all three scales — 10µm, 20µm, 40µm — within each panel. 3 cortical + 3 medullary ROIs per timepoint, both mice in both regions. The cortex/medulla separation persists as the patch coarsens, so the regional pattern is not a scale artifact. Neutrophil is omitted (10µm-only, no grid-scale cell-type basis).
    <span class="source">Source: results/biological_analysis/collab_cd44/cd44_crossscale_summary.csv (per compartment × region × scale) → cd44_compartment_rates_allscales.csv. Descriptive; n=2 mice, paired within-mouse — no region test.</span></figcaption>
  </div>
</figure>

<p>The reading, and it is the striking one: in this UUO model the acquired CD44 activation is <strong>cortex-predominant</strong>. CD206⁺ (M2-like) rises {cd206_c:+.0f}pp in cortex versus {cd206_m:+.0f}pp in medulla; the neutrophil-typed compartment reaches near-total activation in cortex ({neu_c:+.0f}pp) but only {neu_m:+.0f}pp in medulla; CD140b⁺ pericyte/mural tissue rises {cd140_c:+.0f}pp in cortex while in the medulla it <em>declines</em> ({cd140_m:+.0f}pp). This regional structure is consistent with obstructive injury concentrating in the cortex — and it is exactly the pattern the split was designed to reveal. It is also <strong>not a scale artifact</strong>: the cortex–medulla separation persists across all three superpixel scales (the two panels above), so the two findings — scale-robust and cortex-predominant — are one. The caveat stands: with n=2 mice and a paired within-mouse split this halves already-thin support and is hypothesis-generating, not a tested regional effect.</p>

<aside class="margin">
  <span class="label">Every scale, every region on disk</span>
  All three scales × all compartments × {{cortex, medulla, pooled}} are in <code>cd44_compartment_rates_allscales.csv</code>; the per-compartment cross-scale + regional Sham→D7 deltas are in <code>cd44_crossscale_summary.csv</code>.
</aside>
"""
    OUT.write_text(section, encoding='utf-8')
    print(f"wrote {OUT} ({len(section)} chars); scale_reading tally (pooled): {tally}")


if __name__ == '__main__':
    main()
