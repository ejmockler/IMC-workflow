"""Fig 4 region lens — LINEAGE-grain spatial neighborhood enrichment (descriptive).

Rebuilds ``review_packet/pi_report.html`` Fig 4 (``id="fig-4"``) as an interactive
REGION lens (Cortex / Medulla / Pooled) at the DEFENSIBLE lineage grain
(immune / endothelial / stromal), matching the report's existing interactive-lens
idiom (inline SVG + ``data-*`` + a vanilla-JS segmented ``role="radiogroup"`` toggle,
object-constancy y-tween morph, ``prefers-reduced-motion`` jump-cut, ArrowLeft/Right
nav — cloned from ``collab_cd44/make_fig6_region.py`` and the §4.4 Fig 5 lens).

The figure is a LINEAGE-grain 3×3 small-multiples: rows = focal lineage, columns =
neighbouring lineage, each panel a Sham→D7 neighborhood-enrichment trajectory. A
region switch tweens ONLY the y-values across all 9 panels; panel positions, axes,
labels and the fixed lineage colours (``--lineage-immune/-endothelial/-stromal``)
never move (object constancy). Per panel/timepoint: the per-mouse points (n=2, no
error bars), the observed per-mouse spread as a whisker, and the focal-cell count.

HONESTY (DESIGN Decision 0 + hypergraph invariants):
  * Sourced verbatim from ``lineage_neighborhood_by_region_timepoint.csv`` (incl. its
    new ``region='Pooled'`` marginal, real pooled-across-region lineage data — never a
    fabricated mean of the Cortex/Medulla rows).
  * MOUSE-of-mouse only (values are ``mouse_values`` / ``log2_enrichment`` from
    ``aggregate_strata``); no ROI pseudoreplication.
  * NO significance / null anywhere: no p/q, no ``fraction_significant``, no ring-weight,
    no >0.5 / >1.5 binarization. Uncertainty is the OBSERVED per-mouse spread only.
  * EMIT-NOT-DROP support gating: any panel/timepoint flagged ``below_min_support``
    (``n_focal < MIN``) or ``insufficient_support`` (``<2`` effective mice) renders
    DIMMED with a visible ``insufficient support (n<MIN)`` note — never silently dropped,
    never imputed. (No current lineage panel trips it — the grain was chosen for exactly
    this reason — but the path is live so a future sparser input cannot hide.)
  * The fine 15-type detail STAYS POOLED: the verbatim "unresolvable finer at
    region×timepoint" note is rendered in the figure, pointing to RESULTS.md §4. This
    lens is never a per-region fine-type trajectory.

Self-contained: inline ``<style>`` + ``<svg>`` + a ``<script>`` IIFE scoped to
``#fig-4`` with a collision-free ``f4-`` id namespace, so the Fig 5 / Fig 6 lens
scripts (which key off their own ``getElementById`` ids) are unaffected. ``id="fig-4"``
is preserved. Touches only the (non-frozen) Fig 4 block; ``verify_frozen_prereg.py``
stays PASS.

Deterministic: re-running regenerates the identical spliced block (idempotent between
the ``<!-- FIG4_LENS_START -->`` / ``<!-- FIG4_LENS_END -->`` sentinels).

Run:  .venv/bin/python results/biological_analysis/spatial_neighborhoods/make_fig4_region_lens.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

# Verbatim DESIGN Decision 0 note + config-driven support floor — reused, not re-typed.
from run_lineage_neighborhood_lens import UNRESOLVABLE_FINER_NOTE  # noqa: E402
from src.analysis.temporal_interface_analysis import DEFAULT_MIN_SUPPORT  # noqa: E402

CSV = HERE / 'lineage_neighborhood_by_region_timepoint.csv'
REPORT = REPO / 'review_packet' / 'pi_report.html'
OUT = HERE / 'fig4_region_lens.html'

TPS = ['Sham', 'D1', 'D3', 'D7']
LIN = ['endothelial', 'immune', 'stromal']          # row/col order (fixed)
LINCOL = {
    'endothelial': 'var(--lineage-endothelial)',
    'immune': 'var(--lineage-immune)',
    'stromal': 'var(--lineage-stromal)',
}
REGIONS = ['Cortex', 'Medulla', 'Pooled']
MIN = int(DEFAULT_MIN_SUPPORT)

# ── fixed geometry (shared by Python prerender + JS morph) ────────────────────
YMIN, YMAX = -2.0, 1.4                               # log2 enrichment domain (fixed)
YTICKS = [(-2.0, '0.25×'), (-1.0, '0.5×'), (0.0, '1×'), (1.0, '2×')]
GX0, COL_W, COL_GAP = 96.0, 300.0, 40.0
GY0, ROW_H, ROW_GAP = 70.0, 168.0, 58.0
XPAD_L, XPAD_R = 44.0, 20.0
COL_LEFT = [GX0 + c * (COL_W + COL_GAP) for c in range(3)]
ROW_TOP = [GY0 + r * (ROW_H + ROW_GAP) for r in range(3)]
ROW_BOT = [rt + ROW_H for rt in ROW_TOP]
VBW = 1200.0
VBH = ROW_BOT[2] + 46.0


def tpx(col: int, i: int) -> float:
    inner = COL_W - XPAD_L - XPAD_R
    return COL_LEFT[col] + XPAD_L + i * (inner / 3.0)


def yv(v: float, row: int) -> float:
    t = (v - YMIN) / (YMAX - YMIN)
    return ROW_BOT[row] - t * (ROW_BOT[row] - ROW_TOP[row])


# ── data ──────────────────────────────────────────────────────────────────────
def build_data() -> dict:
    """Load the lineage lens CSV into the {region -> [9 panels]} payload the JS reads.

    Panel index p = focal_idx*3 + neighbor_idx over the fixed LIN order. Every rendered
    number (mean, per-mouse points, spread, n, flag) is copied straight from a CSV row —
    the JS never recomputes an estimate, so all marks trace to the CSV by construction.
    """
    df = pd.read_csv(CSV)
    got = sorted(df['region'].unique())
    if got != sorted(REGIONS):
        raise SystemExit(f'CSV regions {got} != expected {sorted(REGIONS)} '
                         f'(run run_lineage_neighborhood_lens.py to emit the Pooled marginal)')
    # Guard the significance-free invariant at the source.
    banned = [c for c in df.columns
              if any(k in c.lower() for k in ('p_value', 'q_value', 'fraction_significant'))]
    if banned:
        raise SystemExit(f'significance columns present in source CSV: {banned}')

    def cell(v):
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    regions: dict = {}
    for region in REGIONS:
        panels = []
        for fi, focal in enumerate(LIN):
            for ni, neighbor in enumerate(LIN):
                mean, mv, n, flag = [], [], [], []
                for tp in TPS:
                    row = df[(df.region == region) & (df.timepoint == tp)
                             & (df.focal_cell_type == focal)
                             & (df.neighbor_cell_type == neighbor)]
                    if len(row) != 1:
                        raise SystemExit(
                            f'expected 1 row for {region}/{tp}/{focal}/{neighbor}, got {len(row)}')
                    r = row.iloc[0]
                    is_flag = bool(r['below_min_support']) or bool(r['insufficient_support'])
                    flag.append(is_flag)
                    mean.append(cell(r['log2_enrichment']))
                    n.append(int(r['n_focal_cells']))
                    pts = [float(x) for x in ast.literal_eval(r['mouse_values'])]
                    pts = pts[:2] + [None] * (2 - len(pts))       # pad to 2 (emit-not-drop)
                    mv.append(pts)
                panels.append({'focal': focal, 'neighbor': neighbor,
                               'mean': mean, 'mv': mv, 'n': n, 'flag': flag})
        regions[region] = panels
    return {'tps': TPS, 'lineages': LIN, 'min_support': MIN, 'regions': regions}


# ── static scaffold (identical for every region; JS never touches it) ─────────
def scaffold() -> str:
    s = []
    # super axis-label
    s.append(f'    <text class="f4-superlab" x="{GX0-70:.0f}" y="26" text-anchor="start">'
             f'Rows = focal lineage · columns = neighbouring lineage · '
             f'each panel: Sham → D7 neighborhood enrichment (log₂), 1× = random expectation</text>')
    for c, neighbor in enumerate(LIN):
        cx = COL_LEFT[c] + COL_W / 2.0
        s.append(f'    <text class="f4-collab" x="{cx:.1f}" y="{GY0-16:.0f}" text-anchor="middle" '
                 f'style="fill:{LINCOL[neighbor]}">next to {neighbor}</text>')
    for r, focal in enumerate(LIN):
        cy = ROW_TOP[r] + ROW_H / 2.0
        s.append(f'    <text class="f4-rowlab" transform="translate(26,{cy:.1f}) rotate(-90)" '
                 f'text-anchor="middle" style="fill:{LINCOL[focal]}">focal {focal}</text>')
    for r in range(3):
        for c in range(3):
            xl, xr = COL_LEFT[c], COL_LEFT[c] + COL_W
            yt, yb = ROW_TOP[r], ROW_BOT[r]
            diagonal = (r == c)
            s.append(f'    <rect class="f4-panelbg{" f4-self" if diagonal else ""}" '
                     f'x="{xl:.1f}" y="{yt:.1f}" width="{COL_W:.1f}" height="{ROW_H:.1f}" rx="4"/>')
            for v, lab in YTICKS:
                gy = yv(v, r)
                cls = 'f4-ref' if v == 0.0 else 'f4-grid'
                s.append(f'    <line class="{cls}" x1="{xl:.1f}" y1="{gy:.1f}" x2="{xr:.1f}" y2="{gy:.1f}"/>')
                if c == 0:
                    s.append(f'    <text class="f4-ytick" x="{xl-6:.1f}" y="{gy:.1f}" '
                             f'text-anchor="end" dominant-baseline="middle">{lab}</text>')
            s.append(f'    <line class="f4-spine" x1="{xl:.1f}" y1="{yt:.1f}" x2="{xl:.1f}" y2="{yb:.1f}"/>')
            s.append(f'    <line class="f4-spine" x1="{xl:.1f}" y1="{yb:.1f}" x2="{xr:.1f}" y2="{yb:.1f}"/>')
            for i, tp in enumerate(TPS):
                x = tpx(c, i)
                s.append(f'    <line class="f4-xtick" x1="{x:.1f}" y1="{yb:.1f}" x2="{x:.1f}" y2="{yb+3:.1f}"/>')
                if r == 2:
                    s.append(f'    <text class="f4-xlab" x="{x:.1f}" y="{yb+16:.0f}" '
                             f'text-anchor="middle">{tp}</text>')
            if diagonal:
                s.append(f'    <text class="f4-selftag" x="{xr-8:.1f}" y="{yt+14:.0f}" '
                         f'text-anchor="end">self-enrichment</text>')
    return '\n'.join(s)


# ── dynamic marks (morph target; prerendered for the default region) ──────────
def dynamic(region_panels: list) -> str:
    s = []
    for p, d in enumerate(region_panels):
        r, c = p // 3, p % 3
        focal, neighbor = d['focal'], d['neighbor']
        color = LINCOL[focal]
        # range whiskers (observed per-mouse spread) + per-mouse dots
        for i in range(4):
            x = tpx(c, i)
            mv = [v for v in d['mv'][i] if v is not None]
            if mv:
                lo, hi = min(mv), max(mv)
                s.append(f'    <line id="f4-wk-{p}-{i}" class="f4-whisk" x1="{x:.1f}" '
                         f'y1="{yv(hi, r):.1f}" x2="{x:.1f}" y2="{yv(lo, r):.1f}" '
                         f'style="stroke:{color}"/>')
            else:
                s.append(f'    <line id="f4-wk-{p}-{i}" class="f4-whisk" x1="{x:.1f}" '
                         f'y1="{yv(0, r):.1f}" x2="{x:.1f}" y2="{yv(0, r):.1f}" '
                         f'style="stroke:{color};opacity:0"/>')
            for m in range(2):
                val = d['mv'][i][m]
                if val is None:
                    s.append(f'    <circle id="f4-md-{p}-{i}-{m}" class="f4-mdot" cx="{x:.1f}" '
                             f'cy="{yv(0, r):.1f}" r="2.4" data-mv="" style="opacity:0;stroke:{color}"/>')
                else:
                    s.append(f'    <circle id="f4-md-{p}-{i}-{m}" class="f4-mdot" cx="{x:.1f}" '
                             f'cy="{yv(val, r):.1f}" r="2.4" data-mv="{val:.4f}" style="stroke:{color}"/>')
        # trajectory polyline (skips flagged / NaN vertices)
        pts = ' '.join(f'{tpx(c, i):.1f},{yv(d["mean"][i], r):.1f}'
                       for i in range(4) if d['mean'][i] is not None)
        s.append(f'    <polyline id="f4-line-{p}" class="f4-line" points="{pts}" '
                 f'style="stroke:{color}"/>')
        # mean dots (carry the data-* the CSV-trace check reads)
        for i, tp in enumerate(TPS):
            x = tpx(c, i)
            if d['mean'][i] is None:
                s.append(f'    <g class="f4-mean" data-panel="{p}" data-tp="{tp}">'
                         f'<circle id="f4-mean-{p}-{i}" class="f4-mean-dot" cx="{x:.1f}" '
                         f'cy="{yv(0, r):.1f}" r="4.2" data-focal="{focal}" data-neighbor="{neighbor}" '
                         f'data-tp="{tp}" data-log2="" data-n="{d["n"][i]}" data-flag="1" '
                         f'style="fill:{color};opacity:0"/><title id="f4-ti-{p}-{i}"></title></g>')
            else:
                s.append(f'    <g class="f4-mean" data-panel="{p}" data-tp="{tp}">'
                         f'<circle id="f4-mean-{p}-{i}" class="f4-mean-dot" cx="{x:.1f}" '
                         f'cy="{yv(d["mean"][i], r):.1f}" r="4.2" data-focal="{focal}" '
                         f'data-neighbor="{neighbor}" data-tp="{tp}" data-log2="{d["mean"][i]:.4f}" '
                         f'data-n="{d["n"][i]}" data-flag="0" style="fill:{color}"/>'
                         f'<title id="f4-ti-{p}-{i}"></title></g>')
        # per-panel n range + emit-not-drop flag note (empty unless a tp is flagged)
        ncells = d['n']
        nlab = f'n {min(ncells)}–{max(ncells)}'
        s.append(f'    <text id="f4-nrange-{p}" class="f4-nrange" x="{COL_LEFT[c]+COL_W-8:.1f}" '
                 f'y="{ROW_BOT[r]-8:.0f}" text-anchor="end">{nlab}</text>')
        flagged = any(d['flag'])
        flagtxt = f'insufficient support (n<{MIN})' if flagged else ''
        s.append(f'    <text id="f4-flag-{p}" class="f4-flagnote" x="{COL_LEFT[c]+COL_W/2:.1f}" '
                 f'y="{ROW_TOP[r]+ROW_H/2:.0f}" text-anchor="middle">{flagtxt}</text>')
    return '\n'.join(s)


CSS = """<style>
  #fig-4 .f4-controls { display:flex; flex-wrap:wrap; align-items:center; gap:8px 12px; margin:2px 0 8px; }
  #fig-4 .f4-ctl-label { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-faint); font-family:var(--font-sans); }
  #fig-4 .f4-toggle { display:inline-flex; border:1px solid var(--rule); border-radius:3px; overflow:hidden; }
  #fig-4 .f4-toggle button { appearance:none; border:0; background:var(--paper); color:var(--ink-soft); font:600 11px/1 ui-monospace,Menlo,monospace; letter-spacing:.02em; padding:6px 11px; cursor:pointer; border-left:1px solid var(--rule); transition:background .12s,color .12s; }
  #fig-4 .f4-toggle button:first-child { border-left:0; }
  #fig-4 .f4-toggle button[aria-checked="true"] { background:var(--ink); color:var(--paper); }
  #fig-4 .f4-toggle button:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
  #fig-4 .f4-readout { font-size:12px; color:var(--ink-faint); font-variant-numeric:tabular-nums; font-family:var(--font-sans); }
  #fig-4 .f4-readout b { color:var(--ink-soft); }
  #fig-4 .f4-note { font-size:12px; line-height:1.45; color:var(--ink-soft); font-family:var(--font-sans); background:var(--rule-soft); border-left:3px solid var(--accent); padding:7px 11px; margin:2px 0 12px; max-width:96ch; }
  #fig-4 .f4-note b { color:var(--ink); }
  #fig-4 .f4-svg { width:100%; height:auto; display:block; font-family:var(--font-sans); }
  #fig-4 .f4-superlab { fill:var(--ink-faint); font-size:11px; font-style:italic; letter-spacing:.02em; }
  #fig-4 .f4-collab { font-size:12.5px; font-weight:600; letter-spacing:.005em; }
  #fig-4 .f4-rowlab { font-size:12.5px; font-weight:600; letter-spacing:.005em; }
  #fig-4 .f4-panelbg { fill:transparent; stroke:var(--rule); stroke-width:1; opacity:.7; }
  #fig-4 .f4-panelbg.f4-self { fill:rgba(122,51,34,0.035); }
  #fig-4 .f4-grid { stroke:var(--rule-soft); stroke-width:1; opacity:.6; }
  #fig-4 .f4-ref { stroke:var(--ink-faint); stroke-width:1; stroke-dasharray:3 3; opacity:.6; }
  #fig-4 .f4-spine { stroke:var(--rule); stroke-width:1; opacity:.7; }
  #fig-4 .f4-ytick, #fig-4 .f4-xtick { stroke:var(--ink-faint); stroke-width:1; }
  #fig-4 text.f4-ytick { fill:var(--ink-faint); stroke:none; font-size:9.5px; font-variant-numeric:tabular-nums; }
  #fig-4 .f4-xlab { fill:var(--ink-faint); font-size:10px; font-variant-numeric:tabular-nums; }
  #fig-4 .f4-selftag { fill:var(--ink-faint); font-size:9px; font-style:italic; letter-spacing:.02em; }
  #fig-4 .f4-nrange { fill:var(--ink-faint); font-size:9.5px; font-variant-numeric:tabular-nums; }
  #fig-4 .f4-flagnote { fill:var(--accent); font-size:11px; font-weight:600; }
  #fig-4 .f4-line { fill:none; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; opacity:.9; }
  #fig-4 .f4-whisk { stroke-width:1.4; opacity:.4; }
  #fig-4 .f4-mdot { fill:var(--paper); stroke-width:1.4; opacity:.9; }
  #fig-4 .f4-mean-dot { stroke:var(--paper); stroke-width:1; }
  #fig-4 .f4-panel-dim { opacity:.28; }
  @media (prefers-reduced-motion: reduce) { #fig-4 .f4-svg * { transition:none !important; } }
</style>"""


def build_script(data: dict) -> str:
    payload = json.dumps(data, separators=(',', ':'))
    geo = json.dumps({
        'RT': ROW_TOP, 'RB': ROW_BOT,
        'TPX': [[tpx(c, i) for i in range(4)] for c in range(3)],
        'YMIN': YMIN, 'YMAX': YMAX,
    }, separators=(',', ':'))
    js = r'''<script>
(function(){
var F4 = __PAYLOAD__;
var G = __GEO__;
var fig = document.getElementById("fig-4");
if(!fig) return;
var TPS = F4.tps, LIN = F4.lineages, MIN = F4.min_support;
var REG = ["Cortex","Medulla","Pooled"];
function yv(v,row){var t=(v-G.YMIN)/(G.YMAX-G.YMIN);return G.RB[row]-t*(G.RB[row]-G.RT[row]);}
function ease(t){return t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;}
function lerp(a,b,t){return a+(b-a)*t;}
function gid(id){return document.getElementById(id);}
var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function geom(region){return F4.regions[region].map(function(d){
  return {mean:d.mean.slice(),mv:d.mv.map(function(r){return r.slice();})};});}
var cur = geom(fig.getAttribute("data-region"));
function tweenPanel(a,b,e){
  return {mean:b.mean.map(function(v,i){return (v==null||a.mean[i]==null)?v:lerp(a.mean[i],v,e);}),
    mv:b.mv.map(function(row,i){return row.map(function(v,m){
      var av=a.mv[i]?a.mv[i][m]:null;return (v==null)?null:((av==null)?v:lerp(av,v,e));});})};}
// data-* carry the CSV target values, set SYNCHRONOUSLY (rAF-independent value truth).
function setData(region){var P=F4.regions[region];
  for(var p=0;p<9;p++){var d=P[p];var flagged=false;
    for(var i=0;i<4;i++){
      var c=gid("f4-mean-"+p+"-"+i);
      if(c){if(d.flag[i]){c.setAttribute("data-log2","");c.setAttribute("data-flag","1");}
            else{c.setAttribute("data-log2",d.mean[i].toFixed(4));c.setAttribute("data-flag","0");}
            c.setAttribute("data-n",d.n[i]);}
      if(d.flag[i])flagged=true;
      for(var m=0;m<2;m++){var md=gid("f4-md-"+p+"-"+i+"-"+m);
        if(md)md.setAttribute("data-mv",(d.mv[i][m]==null)?"":d.mv[i][m].toFixed(4));}}
    var fn=gid("f4-flag-"+p);if(fn)fn.textContent=flagged?("insufficient support (n<"+MIN+")"):"";
    // emit-not-drop: flagged timepoints render DIMMED (never dropped, never imputed).
    for(var i2=0;i2<4;i2++){var mc=gid("f4-mean-"+p+"-"+i2);
      if(mc)mc.classList.toggle("f4-panel-dim",!!d.flag[i2]);
      for(var mm=0;mm<2;mm++){var mdd=gid("f4-md-"+p+"-"+i2+"-"+mm);if(mdd)mdd.classList.toggle("f4-panel-dim",!!d.flag[i2]);}
      var wk=gid("f4-wk-"+p+"-"+i2);if(wk)wk.classList.toggle("f4-panel-dim",!!d.flag[i2]);}
  }}
function draw(st,region){var P=F4.regions[region];
  for(var p=0;p<9;p++){var row=Math.floor(p/3),col=p%3;var s=st[p],d=P[p];
    var pts=[];
    for(var i=0;i<4;i++){if(s.mean[i]!=null)pts.push(G.TPX[col][i].toFixed(1)+","+yv(s.mean[i],row).toFixed(1));}
    var ln=gid("f4-line-"+p);if(ln)ln.setAttribute("points",pts.join(" "));
    for(var i=0;i<4;i++){
      var mc=gid("f4-mean-"+p+"-"+i);
      if(mc){if(s.mean[i]==null){mc.style.opacity="0";}
        else{mc.style.opacity="";mc.setAttribute("cy",yv(s.mean[i],row).toFixed(1));}}
      var mv=s.mv[i],present=[];
      for(var m=0;m<2;m++){var md=gid("f4-md-"+p+"-"+i+"-"+m);
        if(md){if(mv[m]==null){md.style.opacity="0";}else{md.style.opacity="";md.setAttribute("cy",yv(mv[m],row).toFixed(1));present.push(mv[m]);}}}
      var wk=gid("f4-wk-"+p+"-"+i);
      if(wk){if(present.length){var lo=Math.min.apply(null,present),hi=Math.max.apply(null,present);
        wk.style.opacity="";wk.setAttribute("y1",yv(hi,row).toFixed(1));wk.setAttribute("y2",yv(lo,row).toFixed(1));}
        else{wk.style.opacity="0";}}
      var ti=gid("f4-ti-"+p+"-"+i);
      if(ti){if(d.mean[i]==null){ti.textContent=d.focal+" ↔ "+d.neighbor+" · "+region+" · "+TPS[i]+" · insufficient support (n="+d.n[i]+", <"+MIN+" or <2 mice)";}
        else{var mvs=d.mv[i].filter(function(x){return x!=null;}).map(function(x){return Math.pow(2,x).toFixed(2)+"×";}).join(", ");
          ti.textContent=d.focal+" ↔ "+d.neighbor+" · "+region+" · "+TPS[i]+" · "+Math.pow(2,d.mean[i]).toFixed(2)+"× (per-mouse "+mvs+"; n="+d.n[i]+", mouse-of-mouse)";}}
    }
    var nr=gid("f4-nrange-"+p);if(nr){var lo2=Math.min.apply(null,d.n),hi2=Math.max.apply(null,d.n);nr.textContent="n "+lo2+"–"+hi2;}
  }}
function readout(region){var rd=fig.querySelector(".f4-readout");if(!rd)return;
  var P=F4.regions[region],lo=1e12,hi=0;
  for(var p=0;p<9;p++){for(var i=0;i<4;i++){lo=Math.min(lo,P[p].n[i]);hi=Math.max(hi,P[p].n[i]);}}
  rd.innerHTML="<b>"+region+"</b> · 10µm · 3 lineages · per-mouse points (n=2 mice) + observed spread · focal cells "+lo+"–"+hi;}
var raf=null;
function apply(region){fig.setAttribute("data-region",region);
  fig.querySelectorAll("[data-set-region]").forEach(function(b){b.setAttribute("aria-checked",b.getAttribute("data-set-region")===region?"true":"false");});
  setData(region);readout(region);
  var target=geom(region);
  if(reduce){cur=target;draw(cur,region);return;}
  if(raf)cancelAnimationFrame(raf);
  // tween from the currently displayed frame (object constancy: same marks move)
  var from=cur.map(function(o){return {mean:o.mean.slice(),mv:o.mv.map(function(r){return r.slice();})};});
  var t0=null,D=460;
  function step(ts){if(t0===null)t0=ts;var t=Math.min(1,(ts-t0)/D),e=ease(t);
    var frame=[];for(var p=0;p<9;p++)frame.push(tweenPanel(from[p],target[p],e));
    cur=frame;draw(frame,region);
    if(t<1){raf=requestAnimationFrame(step);}else{cur=target;draw(cur,region);raf=null;}}
  raf=requestAnimationFrame(step);}
fig.addEventListener("click",function(e){var b=e.target.closest("[data-set-region]");if(!b)return;apply(b.getAttribute("data-set-region"));});
fig.addEventListener("keydown",function(e){if(/^(INPUT|TEXTAREA)$/.test(e.target.tagName))return;
  if(e.key==="ArrowLeft"||e.key==="ArrowRight"){var j=REG.indexOf(fig.getAttribute("data-region"));j=(j+(e.key==="ArrowRight"?1:2))%3;apply(REG[j]);e.preventDefault();}});
setData(fig.getAttribute("data-region"));readout(fig.getAttribute("data-region"));draw(cur,fig.getAttribute("data-region"));
})();
</script>'''
    return js.replace('__PAYLOAD__', payload).replace('__GEO__', geo)


def build_block(data: dict) -> str:
    default = 'Pooled'
    marks = scaffold() + '\n' + dynamic(data['regions'][default])
    script = build_script(data)
    note = (UNRESOLVABLE_FINER_NOTE +
            ' The fine 15-type detail stays pooled and lives at the pooled grain in '
            'RESULTS.md §4 — never a per-region fine-type trajectory.')
    block = f'''<!-- FIG4_LENS_START -->
{CSS}
<figure class="multiples-fig" id="fig-4" data-region="{default}">
  <div class="multiples-wrap">
    <p class="multiples-eyebrow">Spatial self- and cross-association, by region · §4.3</p>
    <h3 class="multiples-title">Where do the three lineages sit relative to each other, Sham→D7 — and does the pattern hold in <em>cortex vs medulla</em>? Switch region; every panel re-aggregates to that region's mouse-of-mouse enrichment while the grid, axes and lineage colours hold fixed.</h3>
    <div class="f4-controls">
      <span class="f4-ctl-label">Region</span>
      <div class="f4-toggle" role="radiogroup" aria-label="Kidney region">
        <button role="radio" data-set-region="Cortex" aria-checked="false">Cortex</button>
        <button role="radio" data-set-region="Medulla" aria-checked="false">Medulla</button>
        <button role="radio" data-set-region="Pooled" aria-checked="true">Pooled</button>
      </div>
      <span class="f4-readout"></span>
    </div>
    <p class="f4-note"><b>Grain note.</b> {note}</p>
    <svg class="multiples-svg f4-svg" viewBox="0 0 {VBW:.0f} {VBH:.0f}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Interactive lineage neighborhood-enrichment small multiples by kidney region; switch cortex, medulla, or pooled.">
    <g class="f4-plot">
{marks}
    </g>
    </svg>
  </div>
  <figcaption>
    <span class="num">Fig. 4</span> Lineage-grain spatial neighborhood enrichment across Sham→D1→D3→D7, as a 3×3 matrix of focal lineage (row) × neighbouring lineage (column). Each panel's trajectory is the k=10 nearest-neighbor enrichment: for each focal patch of the row lineage, the fraction of its 10 nearest neighbours that are the column lineage, divided by the fraction expected under a shuffled-label baseline (1× = random expectation; log₂ y-axis, fixed and shared). The diagonal panels are self-enrichment (how strongly a lineage clusters with its own kind). <strong>Switch region</strong> (cortex / medulla / pooled) and every panel's line, per-mouse points and spread morph in place while panels, axes and lineage colours stay fixed. The two dots per timepoint are the per-mouse means (n=2 mice, no error bars); the whisker is the observed per-mouse spread (min–max of the two mice); the corner label is the focal-cell count range. Aggregation is mouse-of-mouse (per-ROI → per-mouse → per-stratum, each mouse once); region is a whole-ROI label, so a region is a re-aggregation over that region's ROIs, n=2 mice — <strong>no region test</strong>. Uncertainty here is the observed per-mouse spread only. Sparser and finer-than-lineage detail is not resolvable at region×timepoint and stays pooled in RESULTS.md §4 (see grain note above).
    <span class="source">Source: results/biological_analysis/spatial_neighborhoods/lineage_neighborhood_by_region_timepoint.csv (incl. the region=Pooled marginal; mouse-of-mouse via aggregate_strata; descriptive, n=2 mice).</span>
  </figcaption>
</figure>
{script}
<!-- FIG4_LENS_END -->'''
    return block


def splice(block: str) -> None:
    html = REPORT.read_text(encoding='utf-8')
    START, END = '<!-- FIG4_LENS_START -->', '<!-- FIG4_LENS_END -->'
    if START in html and END in html:
        new = re.sub(re.escape(START) + r'.*?' + re.escape(END), lambda m: block, html, count=1, flags=re.S)
    else:
        # First splice: replace the original static <figure id="fig-4"> ... </figure>.
        pat = r'<figure[^>]*id="fig-4".*?</figure>'
        if not re.search(pat, html, flags=re.S):
            raise SystemExit('could not locate the fig-4 figure block in pi_report.html')
        new = re.sub(pat, lambda m: block, html, count=1, flags=re.S)
    if new == html:
        # Idempotent no-op: the freshly-built block already equals what's in the
        # report. This is success on a re-run, not a failure — return cleanly.
        print('fig-4 lens already up to date (idempotent no-op)')
        return
    REPORT.write_text(new, encoding='utf-8')


def main() -> int:
    data = build_data()
    block = build_block(data)
    OUT.write_text(block, encoding='utf-8')
    splice(block)
    print(f'wrote {OUT} ({len(block)} chars) and spliced into {REPORT}')
    for region in REGIONS:
        diag = [data['regions'][region][i * 3 + i] for i in range(3)]
        print(f'  {region:8s} self-enrichment (Sham→D7, ×):')
        for d in diag:
            traj = ' → '.join(f'{2 ** m:.2f}' if m is not None else 'NA' for m in d['mean'])
            print(f'    {d["focal"]:12s}: {traj}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
