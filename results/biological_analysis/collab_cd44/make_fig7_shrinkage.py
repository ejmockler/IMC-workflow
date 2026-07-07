"""Rebuild Fig 7 (§6 shrinkage) as an INTERACTIVE prior-switching figure.

The static fig-7 in pi_report.html draws every Sham→D7 headline endpoint as a
horizontal candlestick that shows all three Bayesian priors at once (bracket =
skeptical→optimistic, diamond = neutral, hollow circle = raw g). This rebuild
turns the prior into the *interactive* axis: one shrunk-g dot per endpoint on a
fixed Hedges' g axis, and a segmented Prior control {skeptical · neutral ·
optimistic} (default neutral). On a switch, only each dot's x-position tweens —
the g-axis, the row anchors (y), the family colour, the label and the faint
0→raw "shrink track" are all held fixed, so every endpoint reads as ONE object
that slid (object constancy). The per-row n-required-for-80%-power readout in the
right gutter, and the aggregate readout above the plot, update in lockstep.

Why the PRIOR is the natural lens here (and scale / region are not): these are
the 10µm pre-registered temporal-interface endpoints (Family A/B/C). They have
no superpixel-scale sweep and no cortex/medulla split — the only experiment-
adjacent dimension along which the *same* effect estimate moves is the analyst's
prior on effect size. That is exactly what the shrinkage figure is about.

Idiom matches make_report_explore.py / make_fig2_neutrophil.py: inline JSON data,
460 ms cubic-ease rAF morph, prefers-reduced-motion jump-cut, a segmented
radiogroup cloning the report's token styles, honest n=2 labels (no CIs), and a
source line. Descriptive — no significance claims.

Numbers come from endpoint_summary.csv (the analysis source of truth). Row
order, display labels, family colours and lineage glyphs are baked into the
LABELS table below (snapshotted from the report's original static fig-7 metadata,
window.FIG7_DATA) and joined to the CSV on the raw Hedges' g — unique across the
56 headline rows — so the labels stay drop-in faithful while every plotted number
comes from the CSV. Baking the labels (rather than scraping the live report)
keeps this generator idempotent: it does not read pi_report.html at all, so it
still runs after the interactive section has replaced the static fig-7 in place.

Emits fig7_shrinkage.html. Does NOT read or modify pi_report.html.
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / '..' / 'temporal_interfaces' / 'endpoint_summary.csv'
OUT = HERE / 'fig7_shrinkage.html'

PRIORS = ['skeptical', 'neutral', 'optimistic']          # display order
PRIOR_SD = {'skeptical': 0.5, 'neutral': 1.0, 'optimistic': 2.0}   # N(0, sd^2)
DEFAULT = 'neutral'
FAM_COLOR = {'A': '#3d4f6f', 'B': '#7a3322', 'C': '#4e6e3a'}

# Row order + display labels + lineage glyphs + threshold flag, snapshotted from
# the report's original static fig-7 (window.FIG7_DATA). Joined to the CSV on the
# unique raw Hedges' g so every plotted number stays sourced from the CSV. Baked
# inline (not scraped) so the generator is idempotent and self-contained — it
# still runs after the interactive section has replaced the static fig-7 in place.
LABELS = [
    {'g_raw': 3.919668861, 'id': 'A_interface_clr-0', 'fam': 'A', 'label': 'endothelial (interface CLR)', 'glyph': '#2a5d8f', 'flag': True},
    {'g_raw': 3.397313506, 'id': 'A_interface_clr-1', 'fam': 'A', 'label': 'endothelial + immune + stromal (interface CLR)', 'glyph': None, 'flag': False},
    {'g_raw': 1.874860202, 'id': 'A_interface_clr-2', 'fam': 'A', 'label': 'neutrophil (interface CLR)', 'glyph': None, 'flag': False},
    {'g_raw': 1.168753488, 'id': 'A_interface_clr-3', 'fam': 'A', 'label': 'endothelial + immune (interface CLR)', 'glyph': None, 'flag': False},
    {'g_raw': 1.069328738, 'id': 'A_interface_clr-4', 'fam': 'A', 'label': 'immune + stromal (interface CLR)', 'glyph': None, 'flag': False},
    {'g_raw': -0.520639081, 'id': 'A_interface_clr-5', 'fam': 'A', 'label': 'other_rare (interface CLR)', 'glyph': None, 'flag': False},
    {'g_raw': -0.802794271, 'id': 'A_interface_clr-6', 'fam': 'A', 'label': 'immune (interface CLR)', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': -2.579080916, 'id': 'A_interface_clr-7', 'fam': 'A', 'label': 'fibroblast (interface CLR)', 'glyph': None, 'flag': False},
    {'g_raw': -4.419877438, 'id': 'A_interface_clr-8', 'fam': 'A', 'label': 'unassigned (interface CLR)', 'glyph': None, 'flag': False},
    {'g_raw': 3.978074923, 'id': 'B_continuous_neighborhood-0', 'fam': 'B', 'label': 'Δ immune-neighborhood · around stromal', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 4.092562053, 'id': 'B_continuous_neighborhood-1', 'fam': 'B', 'label': 'Δ endothelial-neighborhood · around mixed', 'glyph': '#2a5d8f', 'flag': False},
    {'g_raw': 4.112890091, 'id': 'B_continuous_neighborhood-2', 'fam': 'B', 'label': 'Δ immune-neighborhood · around activated endothelial · CD44', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 4.116236042, 'id': 'B_continuous_neighborhood-3', 'fam': 'B', 'label': 'Δ immune-neighborhood · around activated stromal · CD140b CD44', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 4.450803148, 'id': 'B_continuous_neighborhood-4', 'fam': 'B', 'label': 'Δ endothelial-neighborhood · around activated stromal · CD140b CD44', 'glyph': '#2a5d8f', 'flag': False},
    {'g_raw': 3.566297907, 'id': 'B_continuous_neighborhood-5', 'fam': 'B', 'label': 'Δ immune-neighborhood · around non-myeloid immune', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 4.582800172, 'id': 'B_continuous_neighborhood-6', 'fam': 'B', 'label': 'Δ immune-neighborhood · around stromal', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 3.327886307, 'id': 'B_continuous_neighborhood-7', 'fam': 'B', 'label': 'Δ endothelial-neighborhood · around activated endothelial · CD44', 'glyph': '#2a5d8f', 'flag': False},
    {'g_raw': 3.079232195, 'id': 'B_continuous_neighborhood-8', 'fam': 'B', 'label': 'Δ immune-neighborhood · around activated endothelial · CD140b', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 2.946143189, 'id': 'B_continuous_neighborhood-9', 'fam': 'B', 'label': 'Δ immune-neighborhood · around mixed', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 2.864057386, 'id': 'B_continuous_neighborhood-10', 'fam': 'B', 'label': 'Δ immune-neighborhood · around unassigned', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 2.572058582, 'id': 'B_continuous_neighborhood-11', 'fam': 'B', 'label': 'Δ immune-neighborhood · around endothelial', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 2.559976615, 'id': 'B_continuous_neighborhood-12', 'fam': 'B', 'label': 'Δ immune-neighborhood · around neutrophil', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 6.259537995, 'id': 'B_continuous_neighborhood-13', 'fam': 'B', 'label': 'Δ immune-neighborhood · around activated stromal · CD140b CD44', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 2.522367902, 'id': 'B_continuous_neighborhood-14', 'fam': 'B', 'label': 'Δ immune-neighborhood · around activated endothelial · CD44', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 2.501442171, 'id': 'B_continuous_neighborhood-15', 'fam': 'B', 'label': 'Δ immune-neighborhood · around unassigned', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 2.440245823, 'id': 'B_continuous_neighborhood-16', 'fam': 'B', 'label': 'Δ immune-neighborhood · around non-myeloid immune', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 2.172363062, 'id': 'B_continuous_neighborhood-17', 'fam': 'B', 'label': 'Δ endothelial-neighborhood · around activated stromal · CD140b CD44', 'glyph': '#2a5d8f', 'flag': False},
    {'g_raw': 2.108782197, 'id': 'B_continuous_neighborhood-18', 'fam': 'B', 'label': 'Δ endothelial-neighborhood · around non-myeloid immune', 'glyph': '#2a5d8f', 'flag': False},
    {'g_raw': 2.039875343, 'id': 'B_continuous_neighborhood-19', 'fam': 'B', 'label': 'Δ stromal-neighborhood · around mixed', 'glyph': '#4e7a3e', 'flag': True},
    {'g_raw': 1.83965881, 'id': 'B_continuous_neighborhood-20', 'fam': 'B', 'label': 'Δ stromal-neighborhood · around activated endothelial · CD140b', 'glyph': '#4e7a3e', 'flag': True},
    {'g_raw': 1.811995547, 'id': 'B_continuous_neighborhood-21', 'fam': 'B', 'label': 'Δ stromal-neighborhood · around unassigned', 'glyph': '#4e7a3e', 'flag': True},
    {'g_raw': 1.738454692, 'id': 'B_continuous_neighborhood-22', 'fam': 'B', 'label': 'Δ endothelial-neighborhood · around activated endothelial · CD44', 'glyph': '#2a5d8f', 'flag': False},
    {'g_raw': 1.651672799, 'id': 'B_continuous_neighborhood-23', 'fam': 'B', 'label': 'Δ endothelial-neighborhood · around unassigned', 'glyph': '#2a5d8f', 'flag': False},
    {'g_raw': 1.63734987, 'id': 'B_continuous_neighborhood-24', 'fam': 'B', 'label': 'Δ endothelial-neighborhood · around activated endothelial · CD140b CD44', 'glyph': '#2a5d8f', 'flag': False},
    {'g_raw': 1.584462142, 'id': 'B_continuous_neighborhood-25', 'fam': 'B', 'label': 'Δ immune-neighborhood · around endothelial', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 1.562428801, 'id': 'B_continuous_neighborhood-26', 'fam': 'B', 'label': 'Δ stromal-neighborhood · around activated stromal · CD140b CD44', 'glyph': '#4e7a3e', 'flag': True},
    {'g_raw': 1.549517795, 'id': 'B_continuous_neighborhood-27', 'fam': 'B', 'label': 'Δ stromal-neighborhood · around activated stromal · CD140b CD44', 'glyph': '#4e7a3e', 'flag': True},
    {'g_raw': 1.512464083, 'id': 'B_continuous_neighborhood-28', 'fam': 'B', 'label': 'Δ endothelial-neighborhood · around unassigned', 'glyph': '#2a5d8f', 'flag': False},
    {'g_raw': 1.5035133, 'id': 'B_continuous_neighborhood-29', 'fam': 'B', 'label': 'Δ immune-neighborhood · around mixed', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 1.483431343, 'id': 'B_continuous_neighborhood-30', 'fam': 'B', 'label': 'Δ stromal-neighborhood · around mixed', 'glyph': '#4e7a3e', 'flag': True},
    {'g_raw': 10.848704107, 'id': 'B_continuous_neighborhood-31', 'fam': 'B', 'label': 'Δ stromal-neighborhood · around stromal', 'glyph': '#4e7a3e', 'flag': True},
    {'g_raw': 1.415602003, 'id': 'B_continuous_neighborhood-32', 'fam': 'B', 'label': 'Δ stromal-neighborhood · around activated endothelial · CD44', 'glyph': '#4e7a3e', 'flag': True},
    {'g_raw': 1.345024434, 'id': 'B_continuous_neighborhood-33', 'fam': 'B', 'label': 'Δ immune-neighborhood · around activated endothelial · CD140b CD44', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': 1.32682744, 'id': 'B_continuous_neighborhood-34', 'fam': 'B', 'label': 'Δ stromal-neighborhood · around activated endothelial · CD140b CD44', 'glyph': '#4e7a3e', 'flag': True},
    {'g_raw': 1.26499895, 'id': 'B_continuous_neighborhood-35', 'fam': 'B', 'label': 'Δ stromal-neighborhood · around non-myeloid immune', 'glyph': '#4e7a3e', 'flag': True},
    {'g_raw': 1.249240301, 'id': 'B_continuous_neighborhood-36', 'fam': 'B', 'label': 'Δ endothelial-neighborhood · around stromal', 'glyph': '#2a5d8f', 'flag': False},
    {'g_raw': 1.249014923, 'id': 'B_continuous_neighborhood-37', 'fam': 'B', 'label': 'Δ endothelial-neighborhood · around endothelial', 'glyph': '#2a5d8f', 'flag': False},
    {'g_raw': 1.16007645, 'id': 'B_continuous_neighborhood-38', 'fam': 'B', 'label': 'Δ stromal-neighborhood · around activated endothelial · CD140b', 'glyph': '#4e7a3e', 'flag': True},
    {'g_raw': 1.130673808, 'id': 'B_continuous_neighborhood-39', 'fam': 'B', 'label': 'Δ immune-neighborhood · around activated endothelial · CD140b CD44', 'glyph': '#b8412c', 'flag': False},
    {'g_raw': -1.236577103, 'id': 'B_continuous_neighborhood-40', 'fam': 'B', 'label': 'Δ endothelial-neighborhood · around endothelial', 'glyph': '#2a5d8f', 'flag': False},
    {'g_raw': -1.52573219, 'id': 'B_continuous_neighborhood-41', 'fam': 'B', 'label': 'Δ endothelial-neighborhood · around neutrophil', 'glyph': '#2a5d8f', 'flag': False},
    {'g_raw': -1.743222413, 'id': 'B_continuous_neighborhood-42', 'fam': 'B', 'label': 'Δ stromal-neighborhood · around endothelial', 'glyph': '#4e7a3e', 'flag': True},
    {'g_raw': 4.22219617, 'id': 'C_compartment_activation-0', 'fam': 'C', 'label': 'CD44⁺ rate within neutrophil compartment', 'glyph': None, 'flag': False},
    {'g_raw': 3.305828428, 'id': 'C_compartment_activation-1', 'fam': 'C', 'label': 'triple-positive lineage overlap fraction', 'glyph': None, 'flag': False},
    {'g_raw': 2.880193109, 'id': 'C_compartment_activation-2', 'fam': 'C', 'label': 'CD44⁺ rate within background compartment', 'glyph': None, 'flag': False},
    {'g_raw': 1.461163395, 'id': 'C_compartment_activation-3', 'fam': 'C', 'label': 'CD44⁺ rate within CD140b compartment', 'glyph': None, 'flag': False},
]

# ---- geometry (mirrored verbatim in the JS) --------------------------------
X0 = 830.0                       # g = 0 anchor
SC = (1170.0 - 490.0) / 6.0      # px per unit g  (−3→490 … +3→1170)
GLO, GHI = -3.0, 3.0             # on-axis clamp
ROW_H = 19.0
GUTTER_X = 1230.0                # n-required column centre
DIV_X = 1176.0                   # gutter divider


def xg(g: float) -> float:
    g = max(GLO, min(GHI, g))
    return X0 + SC * g


# ---- data ------------------------------------------------------------------
def build_data() -> dict:
    df = pd.read_csv(CSV)
    h = df[(df.is_headline == True) & (df.contrast == 'Sham_vs_D7')].copy()  # noqa: E712
    if len(h) != len(LABELS):
        raise RuntimeError(f'expected {len(LABELS)} headline Sham_vs_D7 rows, got {len(h)}')
    # join key: raw Hedges' g, unique across the headline rows
    if h.hedges_g.round(9).nunique() != len(h):
        raise RuntimeError('hedges_g is not unique — cannot join labels safely')
    by_g = {round(float(r.hedges_g), 9): r for r in h.itertuples(index=False)}

    rows = []
    for meta in LABELS:
        key = round(float(meta['g_raw']), 9)
        c = by_g.get(key)
        if c is None:
            raise RuntimeError(f"no CSV match for {meta['id']} (g_raw={meta['g_raw']})")
        rows.append({
            'id': meta['id'],
            'fam': meta['fam'],
            'label': meta['label'],
            'glyph': meta['glyph'],
            'flag': 'threshold-sensitive' if meta['flag'] else None,
            'raw': round(float(c.hedges_g), 4),
            # shrunk g under each prior  [skeptical, neutral, optimistic]
            'g': [round(float(c.g_shrunk_skeptical), 4),
                  round(float(c.g_shrunk_neutral), 4),
                  round(float(c.g_shrunk_optimistic), 4)],
            # n / group for 80% power under each prior
            'n': [int(c.n_required_skeptical),
                  int(c.n_required_neutral),
                  int(c.n_required_optimistic)],
        })

    # aggregate n-required stats per prior (for the live readout)
    stats = {}
    for i, p in enumerate(PRIORS):
        ns = sorted(r['n'][i] for r in rows)
        mid = ns[len(ns) // 2] if len(ns) % 2 else (ns[len(ns) // 2 - 1] + ns[len(ns) // 2]) // 2
        stats[p] = {'median': mid, 'min': ns[0], 'max': ns[-1]}

    counts = {f: sum(1 for r in rows if r['fam'] == f) for f in ('A', 'B', 'C')}
    return {'priors': PRIORS, 'default': DEFAULT, 'sd': PRIOR_SD,
            'rows': rows, 'stats': stats, 'counts': counts}


# ---- static SVG (pre-rendered at the default = neutral prior) ---------------
FAM_TITLE = {
    'A': 'Family A · interface composition (CLR)',
    'B': 'Family B · continuous neighborhood enrichment',
    'C': 'Family C · compartment activation rate',
}
DI = PRIORS.index(DEFAULT)   # default prior index


def _row_svg(r: dict, yr: float) -> str:
    cy = yr + 9.5
    ly = yr + 13.5
    col = FAM_COLOR[r['fam']]
    gd = r['g'][DI]
    xd = xg(gd)
    raw = r['raw']
    xraw = xg(raw)
    off = abs(raw) > GHI
    p = [f'<g class="fig7-row" data-id="{r["id"]}" data-fam="{r["fam"]}">',
         f'  <rect class="fig7-hit" x="0" y="{yr:.1f}" width="1280" height="{ROW_H:.0f}"/>']
    # left label: optional glyph · pretty label · optional flag
    lab = _html.escape(r['label'])
    spans = ''
    if r['glyph']:
        spans += f'<tspan class="fig7-glyph" fill="{r["glyph"]}">●</tspan><tspan dx="5"> </tspan>'
    spans += f'<tspan>{lab}</tspan>'
    if r['flag']:
        spans += f'<tspan class="fig7-flag" dx="6">{r["flag"]}</tspan>'
    p.append(f'  <text class="fig7-rlabel" x="458" y="{ly:.1f}" text-anchor="end">{spans}</text>')
    # faint 0 -> raw shrink track (static); raw anchor or off-scale label
    p.append(f'  <line class="fig7-track" x1="{X0:.1f}" y1="{cy:.1f}" x2="{xraw:.1f}" y2="{cy:.1f}"/>')
    if off:
        if raw > 0:
            p.append(f'  <text class="fig7-off" x="1166" y="{ly:.1f}" text-anchor="end" fill="{col}">→{raw:.1f}</text>')
        else:
            p.append(f'  <text class="fig7-off" x="494" y="{ly:.1f}" text-anchor="start" fill="{col}">←{abs(raw):.1f}</text>')
    else:
        p.append(f'  <circle class="fig7-raw" cx="{xraw:.1f}" cy="{cy:.1f}" r="3.1" stroke="{col}"/>')
    # moving stem (0 -> shrunk) + dot
    p.append(f'  <line class="fig7-stem" id="stem-{r["id"]}" x1="{X0:.1f}" y1="{cy:.1f}" x2="{xd:.1f}" y2="{cy:.1f}" stroke="{col}"/>')
    p.append(f'  <circle class="fig7-dot" id="dot-{r["id"]}" cx="{xd:.1f}" cy="{cy:.1f}" r="4.2" fill="{col}"/>')
    # n-required readout (gutter)
    p.append(f'  <text class="fig7-nreq" id="nreq-{r["id"]}" x="{GUTTER_X:.0f}" y="{ly:.1f}" text-anchor="middle" fill="{col}">{r["n"][DI]}</text>')
    # hover title: full per-prior numbers, honest n=2
    ti = (f'{r["label"]} · Sham→D7 (n=2 vs 2) · raw g={raw:+.2f} · '
          f'shrunk g skeptical {r["g"][0]:+.2f} / neutral {r["g"][1]:+.2f} / optimistic {r["g"][2]:+.2f} · '
          f'n/group for 80% power {r["n"][0]} / {r["n"][1]} / {r["n"][2]}')
    p.append(f'  <title>{_html.escape(ti)}</title>')
    p.append('</g>')
    return '\n'.join('  ' + line for line in p)


def prerender(data: dict) -> tuple[str, float]:
    parts = []
    # x grid + ticks + zero + axis titles
    for gi in range(-3, 4):
        x = xg(gi)
        cls = 'fig7-xgrid-zero' if gi == 0 else 'fig7-xgrid'
        parts.append(f'<line class="{cls}" x1="{x:.1f}" y1="60" x2="{x:.1f}" y2="__H2__"/>')
    parts.append('<text class="fig7-xtitle" x="490" y="32">Hedges’ g · effect size (shrunk under the selected prior)</text>')
    parts.append('<text class="fig7-xanno" x="1170" y="32" text-anchor="end">← favors Sham &#160;·&#160; favors D7 →</text>')
    for gi in range(-3, 4):
        x = xg(gi)
        s = ('−' + str(-gi)) if gi < 0 else ('+' + str(gi) if gi > 0 else '0')
        parts.append(f'<text class="fig7-xtick" x="{x:.1f}" y="56" text-anchor="middle">{s}</text>')
    # skeptical ceiling band |g|≈0.32
    bx1, bx2 = xg(-0.32), xg(0.32)
    parts.append(f'<rect class="fig7-band" x="{bx1:.1f}" y="60" width="{bx2 - bx1:.1f}" height="__BH__"/>')
    parts.append('<text class="fig7-bandlab" x="830" y="14" text-anchor="middle">skeptical-prior collapses every effect inside |g| ≈ 0.32</text>')
    # gutter header + divider
    parts.append(f'<line class="fig7-div" x1="{DIV_X:.0f}" y1="60" x2="{DIV_X:.0f}" y2="__H2__"/>')
    parts.append(f'<text class="fig7-guthead" x="{GUTTER_X:.0f}" y="40" text-anchor="middle">n / group</text>')
    parts.append(f'<text class="fig7-guthead2" x="{GUTTER_X:.0f}" y="52" text-anchor="middle">80% power</text>')

    # panels
    yt = {'A': 88.0}
    order = ['A', 'B', 'C']
    bottom = 0.0
    yr = 106.0
    prev_bottom = None
    for fam in order:
        title_y = 88.0 if fam == 'A' else prev_bottom + 40.0
        parts.append(f'<text class="fig7-ptitle" x="20" y="{title_y:.0f}" fill="{FAM_COLOR[fam]}">{FAM_TITLE[fam]}</text>')
        parts.append(f'<text class="fig7-pcount" x="1160" y="{title_y:.0f}" text-anchor="end" fill="{FAM_COLOR[fam]}">{data["counts"][fam]} headline endpoints</text>')
        yr = title_y + 18.0
        for r in [x for x in data['rows'] if x['fam'] == fam]:
            parts.append(_row_svg(r, yr))
            yr += ROW_H
        prev_bottom = yr
        bottom = yr
    H = bottom + 34.0
    body = '\n'.join(parts)
    body = body.replace('__H2__', f'{bottom + 6:.0f}').replace('__BH__', f'{bottom + 6 - 60:.0f}')
    return body, H


# ---- styles (clone the report's segmented-toggle + shrinkage tokens) --------
CSS = """<style>
  #fig7-shrink-explore { margin: 2.4rem 0; }
  #fig7-shrink-explore .fig7-eyebrow { font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--ink-faint); margin:0 0 4px; }
  #fig7-shrink-explore h3.fig7-title { font-size:20px; line-height:1.28; font-weight:600; color:var(--ink); margin:0 0 4px; max-width:60ch; }
  #fig7-shrink-explore h3.fig7-title em { font-style:italic; color:var(--accent); }
  #fig7-shrink-explore .fig7-lens-note { font-size:12.5px; color:var(--ink-soft); line-height:1.5; max-width:70ch; margin:6px 0 12px; }
  #fig7-shrink-explore .fig7-controls { display:flex; flex-wrap:wrap; align-items:center; gap:8px 12px; margin:6px 0 12px; }
  #fig7-shrink-explore .fig7-ctl-label { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-faint); }
  #fig7-shrink-explore .fig7-toggle { display:inline-flex; border:1px solid var(--rule); border-radius:3px; overflow:hidden; }
  #fig7-shrink-explore .fig7-toggle button { appearance:none; border:0; background:var(--paper); color:var(--ink-soft); font:600 11px/1 ui-monospace,Menlo,monospace; letter-spacing:.02em; padding:6px 12px; cursor:pointer; border-left:1px solid var(--rule); transition:background .12s,color .12s; }
  #fig7-shrink-explore .fig7-toggle button:first-child { border-left:0; }
  #fig7-shrink-explore .fig7-toggle button[aria-checked="true"] { background:var(--ink); color:var(--paper); }
  #fig7-shrink-explore .fig7-toggle button:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
  #fig7-shrink-explore .fig7-readout { font-size:12px; color:var(--ink-faint); font-variant-numeric:tabular-nums; }
  #fig7-shrink-explore .fig7-readout b { color:var(--ink-soft); }
  #fig7-shrink-explore .shrinkage-svg { width:100%; height:auto; display:block; margin:2px 0 4px; overflow:visible; }
  #fig7-svg .fig7-xgrid { stroke:var(--rule-soft); stroke-width:1; opacity:.45; }
  #fig7-svg .fig7-xgrid-zero { stroke:var(--ink-faint); stroke-width:1; opacity:.7; stroke-dasharray:2 2; }
  #fig7-svg .fig7-div { stroke:var(--rule); stroke-width:1; opacity:.6; }
  #fig7-svg .fig7-xtick { fill:var(--ink-faint); font-size:11px; font-variant-numeric:tabular-nums; }
  #fig7-svg .fig7-xtitle { fill:var(--ink-soft); font-size:11px; font-style:italic; letter-spacing:.02em; }
  #fig7-svg .fig7-xanno { fill:var(--ink-faint); font-size:10.5px; font-style:italic; letter-spacing:.04em; }
  #fig7-svg .fig7-band { fill:var(--accent); fill-opacity:.05; }
  #fig7-svg .fig7-bandlab { fill:var(--ink-faint); font-size:10px; font-style:italic; letter-spacing:.04em; }
  #fig7-svg .fig7-guthead, #fig7-svg .fig7-guthead2 { fill:var(--ink-faint); font-size:9.5px; letter-spacing:.03em; text-transform:uppercase; }
  #fig7-svg .fig7-ptitle { font-size:13px; font-weight:600; letter-spacing:.01em; }
  #fig7-svg .fig7-pcount { font-size:10.5px; font-weight:500; font-variant-numeric:tabular-nums; opacity:.75; }
  #fig7-svg .fig7-rlabel { fill:var(--ink-soft); font-size:11.5px; }
  #fig7-svg .fig7-glyph { font-size:11px; }
  #fig7-svg .fig7-flag { fill:var(--ink-faint); font-size:9px; font-style:italic; letter-spacing:.02em; }
  #fig7-svg .fig7-hit { fill:transparent; pointer-events:all; }
  #fig7-svg .fig7-track { stroke:var(--ink-faint); stroke-width:1.4; opacity:.16; }
  #fig7-svg .fig7-raw { fill:var(--paper); stroke-width:1.3; opacity:.7; }
  #fig7-svg .fig7-stem { stroke-width:2.2; opacity:.5; stroke-linecap:round; }
  #fig7-svg .fig7-dot { stroke:var(--paper); stroke-width:1.3; }
  #fig7-svg .fig7-off { font-size:10px; font-variant-numeric:tabular-nums; opacity:.8; }
  #fig7-svg .fig7-nreq { font-size:11px; font-weight:600; font-variant-numeric:tabular-nums; }
  #fig7-svg .fig7-row .fig7-dot { transition:none; }
  #fig7-svg:hover .fig7-row:not(:hover) { opacity:.34; }
  #fig7-svg .fig7-row { transition:opacity .18s; }
  #fig7-svg .fig7-row:hover .fig7-dot { stroke:var(--ink); stroke-width:1.6; }
  #fig7-svg .fig7-row:hover .fig7-rlabel { fill:var(--ink); font-weight:600; }
  @media (prefers-reduced-motion: reduce) { #fig7-svg * { transition:none !important; } }
</style>"""


# ---- interaction script -----------------------------------------------------
def build_script(data: dict) -> str:
    payload = json.dumps(data, separators=(',', ':'))
    return (
        '<script>\n(function(){\n'
        'var D = ' + payload + ';\n'
        'var fig = document.getElementById("fig7-shrink-explore");\n'
        'if(!fig) return;\n'
        'var PR = D.priors, SD = D.sd;\n'
        'var X0=' + f'{X0}' + ', SC=' + f'{SC}' + ', GLO=' + f'{GLO}' + ', GHI=' + f'{GHI}' + ';\n'
        'function xg(g){g=Math.max(GLO,Math.min(GHI,g));return X0+SC*g;}\n'
        'function ease(t){return t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;}\n'
        'function lerp(a,b,t){return a+(b-a)*t;}\n'
        'var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;\n'
        'function idx(p){return PR.indexOf(p);}\n'
        # current per-row state {g, n} at the active prior
        'function stateFor(p){var i=idx(p),st={};D.rows.forEach(function(r){st[r.id]={g:r.g[i],n:r.n[i]};});return st;}\n'
        'var cur = stateFor(fig.getAttribute("data-prior"));\n'
        'function draw(st){D.rows.forEach(function(r){var s=st[r.id];\n'
        '  var dot=document.getElementById("dot-"+r.id); if(dot)dot.setAttribute("cx",xg(s.g).toFixed(1));\n'
        '  var stem=document.getElementById("stem-"+r.id); if(stem)stem.setAttribute("x2",xg(s.g).toFixed(1));\n'
        '  var nq=document.getElementById("nreq-"+r.id); if(nq)nq.textContent=Math.round(s.n);\n'
        '});}\n'
        # aggregate readout for a prior
        'function readout(p){var s=D.stats[p];\n'
        '  fig.querySelector(".fig7-readout").innerHTML="<b>"+p+"</b> \\u00b7 N(0, "+SD[p].toFixed(1)+"\\u00b2) prior \\u00b7 median <b>n\\u2248"+s.median+"</b> / group for 80% power \\u00b7 per-endpoint "+s.min+"\\u2013"+s.max;}\n'
        'var raf=null;\n'
        'function apply(p){fig.setAttribute("data-prior",p);\n'
        '  fig.querySelectorAll("[data-set-prior]").forEach(function(b){b.setAttribute("aria-checked",b.getAttribute("data-set-prior")===p?"true":"false");});\n'
        '  readout(p);\n'
        '  var target=stateFor(p);\n'
        '  if(reduce){cur=target;draw(cur);return;}\n'
        '  if(raf)cancelAnimationFrame(raf);\n'
        '  var from={}; D.rows.forEach(function(r){from[r.id]={g:cur[r.id].g,n:cur[r.id].n};});\n'
        '  var t0=null,DUR=460;\n'
        '  function step(ts){if(t0===null)t0=ts;var t=Math.min(1,(ts-t0)/DUR),e=ease(t);\n'
        '    D.rows.forEach(function(r){var a=from[r.id],b=target[r.id],s=cur[r.id];\n'
        '      s.g=lerp(a.g,b.g,e); s.n=lerp(a.n,b.n,e);});\n'
        '    draw(cur);\n'
        '    if(t<1){raf=requestAnimationFrame(step);}else{D.rows.forEach(function(r){cur[r.id]={g:target[r.id].g,n:target[r.id].n};});draw(cur);raf=null;}}\n'
        '  raf=requestAnimationFrame(step);}\n'
        # controls: click + arrow keys
        'fig.addEventListener("click",function(e){var t=e.target.closest("[data-set-prior]");if(!t)return;apply(t.getAttribute("data-set-prior"));});\n'
        'fig.addEventListener("keydown",function(e){if(/^(INPUT|TEXTAREA)$/.test(e.target.tagName))return;\n'
        '  if(e.key==="ArrowLeft"||e.key==="ArrowRight"){var j=idx(fig.getAttribute("data-prior"));j=(j+(e.key==="ArrowRight"?1:PR.length-1))%PR.length;apply(PR[j]);e.preventDefault();}});\n'
        'readout(fig.getAttribute("data-prior"));\n'
        'draw(cur);\n'
        '})();\n</script>'
    )


def main():
    data = build_data()
    body, H = prerender(data)
    script = build_script(data)
    dstats = data['stats']
    section = f"""{CSS}
<figure class="shrinkage-fig" id="fig7-shrink-explore" data-prior="{DEFAULT}">
  <div class="shrinkage-wrap">
    <p class="fig7-eyebrow">Effect-size shrinkage, made switchable · §6</p>
    <h3 class="fig7-title">Turn the prior and watch all {len(data['rows'])} headline effects <em>slide</em>: skeptical collapses them into |g|≈{'0.32'}, optimistic lets them fan back out toward their raw values — and the sample size you’d need <em>moves with them</em>.</h3>
    <p class="fig7-lens-note">These are the 10µm pre-registered Family A/B/C temporal-interface endpoints — there is no superpixel-scale sweep and no cortex/medulla split to switch, so the natural experiment-adjacent lens is the <strong>prior on effect size</strong>. Each endpoint is one dot on a fixed Hedges’ g axis at its Bayesian-shrunk estimate; the faint line behind it runs from 0 to the raw (unshrunk) g, so the dot rides between “all variance” (0) and “all signal” (raw). Switch the prior and only the dots move — axis, rows, colour and labels stay put.</p>
    <div class="fig7-controls">
      <span class="fig7-ctl-label">Prior</span>
      <div class="fig7-toggle" role="radiogroup" aria-label="Bayesian prior on effect size">
        <button role="radio" data-set-prior="skeptical" aria-checked="false">Skeptical</button>
        <button role="radio" data-set-prior="neutral" aria-checked="true">Neutral</button>
        <button role="radio" data-set-prior="optimistic" aria-checked="false">Optimistic</button>
      </div>
      <span class="fig7-readout"><b>{DEFAULT}</b> · N(0, {PRIOR_SD[DEFAULT]:.1f}²) prior · median <b>n≈{dstats[DEFAULT]['median']}</b> / group for 80% power · per-endpoint {dstats[DEFAULT]['min']}–{dstats[DEFAULT]['max']}</span>
    </div>
    <svg class="shrinkage-svg" id="fig7-svg" viewBox="0 0 1280 {H:.0f}" preserveAspectRatio="xMidYMin meet" role="img" aria-label="Interactive effect-size shrinkage; switch the Bayesian prior and each endpoint's shrunk Hedges' g and required sample size morph in place.">
{body}
    </svg>
    <figcaption>
      Every Sham → D7 headline endpoint (n = {len(data['rows'])}) as one shrunk-g dot on a shared Hedges’ g axis; the faint track behind each runs 0 → raw g. The <strong>Prior</strong> control ({' · '.join(p.capitalize() for p in PRIORS)}, default neutral — N(0, {PRIOR_SD['skeptical']:.1f}²)/{PRIOR_SD['neutral']:.1f}²/{PRIOR_SD['optimistic']:.1f}²) tweens each dot to its estimate under that prior (460 ms, object constancy — only x moves); the right gutter shows the n per group needed for 80% power, which updates with it. The shaded band is the skeptical-prior ceiling |g| ≈ 0.32. Hover a row for its full per-prior numbers. <strong>No endpoint survives BH-FDR q &lt; 0.05</strong> at this sample size — this figure is descriptive (n = 2 mice/group, no CIs): it shows how much of each effect is prior versus data, and what confirming it would cost. Rows sort by neutral g within family.
      <span class="source">Source: results/biological_analysis/temporal_interfaces/endpoint_summary.csv · is_headline=True · contrast=Sham_vs_D7 · columns g_shrunk_{{skeptical,neutral,optimistic}}, n_required_{{skeptical,neutral,optimistic}}, hedges_g. Descriptive; n=2 mice/group.</span>
    </figcaption>
  </div>
</figure>
{script}
"""
    OUT.write_text(section, encoding='utf-8')
    print(f"wrote {OUT} ({len(section)} chars); rows={len(data['rows'])}; "
          f"H={H:.0f}; stats={data['stats']}")


if __name__ == '__main__':
    main()
