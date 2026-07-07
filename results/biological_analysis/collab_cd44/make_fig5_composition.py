"""Rebuild Fig 5 (§4.4 compositional dumbbell) as an INTERACTIVE region-switching
cell-type composition figure. Cell-type annotation exists only at 10µm, so the
lens is REGION (cortex / medulla / pooled), not scale. Same object-constancy morph
as the §4.1/§4.2 figures: rows are held in a FIXED pooled-Δ order (each cell type is
one row at a fixed Y, fixed colour, fixed label, fixed axis) and on a region switch
only the value dimension moves — the dumbbells slide horizontally to that region's
Sham→D7 fractions, per-mouse points, sparklines, numbers, Δ and the unassigned
budget all morph in place (460ms cubic-ease rAF; prefers-reduced-motion jump-cut).

Descriptive, n=2 mice, mouse-of-mouse means, no CIs, no region test.
Emits fig5_composition.html (spliced to REPLACE the static <figure id="fig-5">).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / '..' / 'differential_abundance' / 'roi_abundances.csv'
OUT = HERE / 'fig5_composition.html'

TPS = ['Sham', 'D1', 'D3', 'D7']
MICE = ['MS1', 'MS2']
REGIONS = ['cortex', 'medulla', 'pooled']
REGMAP = {'pooled': None, 'cortex': 'Cortex', 'medulla': 'Medulla'}

# lineage → (badge fill, row-colour token, badge letter)
LIN = {
    'immune':      ('#b8412c', 'var(--lineage-immune)', 'I'),
    'endothelial': ('#2a5d8f', 'var(--lineage-endothelial)', 'E'),
    'stromal':     ('#4e7a3e', 'var(--lineage-stromal)', 'S'),
}

# Row order is FIXED (pooled Δpp descending — the current fig-5 order) so each row
# stays one object across region switches. (id, label, lineage, callout)
CELLS = [
    ('endothelial',                 'endothelial',          'endothelial', True),
    ('neutrophil',                  'neutrophil',           'immune',      True),
    ('activated_endothelial_cd140b','endothelial · CD140b⁺', 'endothelial', True),
    ('activated_m2_cd140b',         'M2 · CD140b⁺',    'immune',      False),
    ('m2_macrophage',               'M2 macrophage',        'immune',      False),
    ('activated_fibroblast_cd44',   'fibroblast · CD44⁺', 'stromal',   False),
    ('activated_myeloid_cd140b',    'myeloid · CD140b⁺','immune',      False),
    ('myeloid',                     'myeloid',              'immune',      False),
    ('activated_immune',            'immune · activated',   'immune',      False),
    ('activated_endothelial_cd44',  'endothelial · CD44⁺', 'endothelial', False),
    ('activated_m2_cd44',           'M2 · CD44⁺',      'immune',      False),
    ('activated_myeloid_cd44',      'myeloid · CD44⁺', 'immune',      False),
    ('immune_cells',                'immune (other)',       'immune',      False),
    ('fibroblast',                  'fibroblast',           'stromal',     False),
    ('activated_fibroblast_cd140b', 'fibroblast · CD140b⁺','stromal',    True),
]

# ---- geometry (mirrors the JS so the pre-rendered pooled slice is a valid chart) ----
NX0, NX1 = 230.0, 800.0          # named log-axis pixel span (0.01% … 10%)
UX0, UX1 = 230.0, 800.0          # unassigned linear-axis span (80% … 90%)
SPX = [1020.0, 1075.0, 1130.0, 1185.0]   # sparkline x-anchors (Sham·D1·D3·D7)
ROW_CY0, ROW_DY = 99.0, 26.0     # first row dumbbell y; row pitch
UNASS_CY = 549.0


def logx(vpct: float) -> float:
    v = max(vpct, 0.01)
    x = 230.0 + (math.log10(v) + 2.0) * 190.0
    return min(max(x, NX0), NX1)


def linx(vpct: float) -> float:
    x = 230.0 + (vpct - 80.0) * 57.0
    return min(max(x, UX0), UX1)


def rad(n: int) -> float:
    return round(max(1.5, min(3.4, 1.0 + 1.9 * math.sqrt(n / 7400.0))), 2)


def _mom(sub: pd.DataFrame, col: str) -> tuple[float, list]:
    """mouse-of-mouse mean of a *_prop column (pct) + per-mouse [pct, n_support]."""
    per_mouse = []
    for m in MICE:
        g = sub[sub.mouse == m]
        if len(g):
            per_mouse.append([round(float(g[col].mean()) * 100.0, 4),
                              int(g['n_total'].sum())])
    mean = round(float(np.mean([p[0] for p in per_mouse])), 4)
    return mean, per_mouse


def build_region(df: pd.DataFrame, region: str) -> dict:
    r = REGMAP[region]
    rdf = df if r is None else df[df.region == r]
    cells = {}
    for cid, _lab, _lin, _cal in CELLS:
        col = cid + '_prop'
        mean, sham_mice, d7_mice = [], [], []
        for tp in TPS:
            sub = rdf[rdf.timepoint == tp]
            mu, pm = _mom(sub, col)
            mean.append(mu)
            if tp == 'Sham':
                sham_mice = pm
            elif tp == 'D7':
                d7_mice = pm
        cells[cid] = {'mean': mean, 'sham_mice': sham_mice, 'd7_mice': d7_mice,
                      'delta': round(mean[3] - mean[0], 4)}
    # unassigned budget
    umean, usham, ud7 = [], [], []
    for tp in TPS:
        sub = rdf[rdf.timepoint == tp]
        mu, pm = _mom(sub, 'unassigned_prop')
        umean.append(mu)
        if tp == 'Sham':
            usham = pm
        elif tp == 'D7':
            ud7 = pm
    unassigned = {'mean': umean, 'sham_mice': usham, 'd7_mice': ud7,
                  'delta': round(umean[3] - umean[0], 4)}
    # support range across mouse × tp (patches per mouse)
    supp = []
    for tp in TPS:
        for m in MICE:
            g = rdf[(rdf.timepoint == tp) & (rdf.mouse == m)]
            if len(g):
                supp.append(int(g['n_total'].sum()))
    return {'cells': cells, 'unassigned': unassigned,
            'support_lo': min(supp), 'support_hi': max(supp)}


def build_data() -> dict:
    df = pd.read_csv(CSV)
    regions = {reg: build_region(df, reg) for reg in REGIONS}
    cells_meta = []
    for cid, lab, lin, cal in CELLS:
        fill, tok, letter = LIN[lin]
        cells_meta.append({'id': cid, 'label': lab, 'lineage': lin,
                           'fill': fill, 'color': tok, 'letter': letter, 'callout': cal})
    return {'tps': TPS, 'cells': cells_meta, 'regions': regions}


# ---------------------------------------------------------------------------
def _spark_ys(mean: list, cy: float) -> list:
    lo, hi = min(mean), max(mean)
    if hi <= lo:
        return [cy] * 4
    return [round(cy + 9.0 - 18.0 * (m - lo) / (hi - lo), 1) for m in mean]


def prerender(data: dict) -> str:
    """Static pooled slice so the figure reads correctly with JS off."""
    sl = data['regions']['pooled']
    p = []
    for i, cm in enumerate(data['cells']):
        cid = cm['id']
        s = sl['cells'][cid]
        cy = ROW_CY0 + ROW_DY * i
        callout = cm['callout']
        sx, dx = logx(s['mean'][0]), logx(s['mean'][3])
        d7r = 5.5 if callout else 4.5
        rowcls = 'row row-callout' if callout else 'row'
        namecls = 'row-name row-name-callout' if callout else 'row-name'
        numcls = 'row-num row-num-callout' if callout else 'row-num'
        dsign = s['delta'] >= 0
        dcls = 'delta-up' if dsign else 'delta-down'
        dcallout = ' row-delta-callout' if callout else ''
        p.append(f'        <g class="{rowcls}" id="c-{cid}" data-key="{cid}" data-lineage="{cm["lineage"]}" style="--row-color: {cm["color"]};">')
        p.append(f'          <rect class="row-hit" x="0" y="{cy-13:.1f}" width="1200" height="26" fill="transparent"/>')
        p.append(f'          <rect class="lineage-badge" x="17.0" y="{cy-7:.1f}" width="14" height="14" rx="2" fill="{cm["fill"]}"/>')
        p.append(f'          <text class="lineage-badge-letter" x="24.0" y="{cy+4:.1f}" text-anchor="middle">{cm["letter"]}</text>')
        p.append(f'          <text class="{namecls}" x="50.0" y="{cy+4:.1f}">{cm["label"]}</text>')
        p.append(f'          <line class="dumb-line" id="ln-{cid}" x1="{sx:.1f}" y1="{cy:.1f}" x2="{dx:.1f}" y2="{cy:.1f}"/>')
        for k, mv in enumerate(s['sham_mice']):
            p.append(f'          <circle class="mouse-pt" id="sm{k}-{cid}" cx="{logx(mv[0]):.1f}" cy="{cy-5:.1f}" r="{rad(mv[1]):.2f}"/>')
        for k, mv in enumerate(s['d7_mice']):
            p.append(f'          <circle class="mouse-pt" id="dm{k}-{cid}" cx="{logx(mv[0]):.1f}" cy="{cy+5:.1f}" r="{rad(mv[1]):.2f}"/>')
        p.append(f'          <circle class="dumb-sham" id="sd-{cid}" cx="{sx:.1f}" cy="{cy:.1f}" r="3.5"/>')
        p.append(f'          <circle class="dumb-d7" id="dd-{cid}" cx="{dx:.1f}" cy="{cy:.1f}" r="{d7r}"/>')
        p.append(f'          <text class="{numcls}" id="rn-{cid}" x="815.0" y="{cy+4:.1f}">{s["mean"][0]:.2f}% → {s["mean"][3]:.2f}%</text>')
        p.append(f'          <text class="row-delta {dcls}{dcallout}" id="rd-{cid}" x="940.0" y="{cy+4:.1f}">{"+" if dsign else "−"}{abs(s["delta"]):.2f}pp</text>')
        sy = _spark_ys(s['mean'], cy)
        pts = ' '.join(f'{SPX[j]:.1f},{sy[j]:.1f}' for j in range(4))
        p.append(f'          <polyline class="spark-line" id="sp-{cid}" points="{pts}"/>')
        p.append(f'          <circle class="spark-dot spark-dot-sham" id="spS-{cid}" cx="{SPX[0]:.1f}" cy="{sy[0]:.1f}" r="2"/>')
        p.append(f'          <circle class="spark-dot spark-dot-d7" id="spD-{cid}" cx="{SPX[3]:.1f}" cy="{sy[3]:.1f}" r="2.4"/>')
        p.append(f'          <title id="f5ti-{cid}"></title>')
        p.append('        </g>')
    return '\n'.join(p)


def prerender_unass(data: dict) -> str:
    s = data['regions']['pooled']['unassigned']
    cy = UNASS_CY
    sx, dx = linx(s['mean'][0]), linx(s['mean'][3])
    p = ['      <g class="row row-unass" id="c-unassigned" data-key="unassigned" data-lineage="unassigned" style="--row-color: var(--ink-soft);">']
    p.append(f'        <rect class="row-hit" x="0" y="{cy-17:.1f}" width="1200" height="34" fill="transparent"/>')
    p.append(f'        <rect class="lineage-badge" x="17.0" y="{cy-7:.1f}" width="14" height="14" rx="2" fill="#7a7a7a"/>')
    p.append(f'        <text class="lineage-badge-letter" x="24.0" y="{cy+4:.1f}" text-anchor="middle">U</text>')
    p.append(f'        <text class="row-name row-name-callout" x="50.0" y="{cy+4:.1f}">unassigned</text>')
    p.append(f'        <line class="dumb-line dumb-line-unass" id="ln-unassigned" x1="{sx:.1f}" y1="{cy:.1f}" x2="{dx:.1f}" y2="{cy:.1f}"/>')
    for k, mv in enumerate(s['sham_mice']):
        p.append(f'        <circle class="mouse-pt mouse-pt-unass" id="sm{k}-unassigned" cx="{linx(mv[0]):.1f}" cy="{cy-5:.1f}" r="{rad(mv[1]):.2f}"/>')
    for k, mv in enumerate(s['d7_mice']):
        p.append(f'        <circle class="mouse-pt mouse-pt-unass" id="dm{k}-unassigned" cx="{linx(mv[0]):.1f}" cy="{cy+5:.1f}" r="{rad(mv[1]):.2f}"/>')
    p.append(f'        <circle class="dumb-sham dumb-sham-unass" id="sd-unassigned" cx="{sx:.1f}" cy="{cy:.1f}" r="4"/>')
    p.append(f'        <circle class="dumb-d7 dumb-d7-unass" id="dd-unassigned" cx="{dx:.1f}" cy="{cy:.1f}" r="6"/>')
    p.append(f'        <text class="row-num row-num-callout" id="rn-unassigned" x="815.0" y="{cy+4:.1f}">{s["mean"][0]:.2f}% → {s["mean"][3]:.2f}%</text>')
    dsign = s['delta'] >= 0
    p.append(f'        <text class="row-delta {"delta-up" if dsign else "delta-down"} row-delta-callout" id="rd-unassigned" x="940.0" y="{cy+4:.1f}">{"+" if dsign else "−"}{abs(s["delta"]):.2f}pp</text>')
    sy = _spark_ys(s['mean'], cy)
    pts = ' '.join(f'{SPX[j]:.1f},{sy[j]:.1f}' for j in range(4))
    p.append(f'        <polyline class="spark-line" id="sp-unassigned" points="{pts}"/>')
    p.append(f'        <circle class="spark-dot spark-dot-sham" id="spS-unassigned" cx="{SPX[0]:.1f}" cy="{sy[0]:.1f}" r="2"/>')
    p.append(f'        <circle class="spark-dot spark-dot-d7" id="spD-unassigned" cx="{SPX[3]:.1f}" cy="{sy[3]:.1f}" r="2.4"/>')
    p.append('      </g>')
    return '\n'.join(p)


CSS = """<style>
  #fig-5 .r5-controls { display:flex; flex-wrap:wrap; align-items:center; gap:8px 12px; margin:2px 0 14px; }
  #fig-5 .r5-ctl-label { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-faint); font-family:var(--font-sans); }
  #fig-5 .r5-toggle { display:inline-flex; border:1px solid var(--rule); border-radius:3px; overflow:hidden; }
  #fig-5 .r5-toggle button { appearance:none; border:0; background:var(--paper); color:var(--ink-soft); font:600 11px/1 ui-monospace,Menlo,monospace; letter-spacing:.02em; padding:6px 11px; cursor:pointer; border-left:1px solid var(--rule); transition:background .12s,color .12s; }
  #fig-5 .r5-toggle button:first-child { border-left:0; }
  #fig-5 .r5-toggle button[aria-checked="true"] { background:var(--ink); color:var(--paper); }
  #fig-5 .r5-toggle button:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
  #fig-5 .r5-readout { font-size:12px; color:var(--ink-faint); font-variant-numeric:tabular-nums; font-family:var(--font-sans); }
  #fig-5 .r5-readout b { color:var(--ink-soft); }
  @media (prefers-reduced-motion: reduce) { #fig-5 .composition-svg * { transition:none !important; } }
</style>"""


def build_script(data: dict) -> str:
    payload = json.dumps(data, separators=(',', ':'))
    return (
        '<script>\n(function(){\n'
        'var F5 = ' + payload + ';\n'
        'var fig = document.getElementById("fig-5");\n'
        'if(!fig) return;\n'
        'var TPS=["Sham","D1","D3","D7"], SPX=[1020,1075,1130,1185];\n'
        'var CY0=99, DY=26, UCY=549;\n'
        'function logx(v){v=Math.max(v,0.01);var x=230+(Math.log(v)/Math.LN10+2)*190;return Math.min(Math.max(x,230),800);}\n'
        'function linx(v){var x=230+(v-80)*57;return Math.min(Math.max(x,230),800);}\n'
        'function rad(n){return Math.max(1.5,Math.min(3.4,1+1.9*Math.sqrt(n/7400)));}\n'
        'function ease(t){return t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;}\n'
        'function fmt(v){return v.toFixed(2)+"%";}\n'
        'var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;\n'
        'function clone(o){return JSON.parse(JSON.stringify(o));}\n'
        'function stateOf(region){return clone(F5.regions[region]);}\n'
        'var cur = stateOf(fig.getAttribute("data-region"));\n'
        # ---- draw a row (named=log / unass=linear) from tweened values ----
        'function sparkY(mean,cy){var lo=Math.min.apply(null,mean),hi=Math.max.apply(null,mean);\n'
        '  if(hi<=lo)return [cy,cy,cy,cy];\n'
        '  return mean.map(function(m){return cy+9-18*(m-lo)/(hi-lo);});}\n'
        'function drawRow(id,s,cy,mapx,region,label){\n'
        '  var sx=mapx(s.mean[0]),dx=mapx(s.mean[3]);\n'
        '  var ln=document.getElementById("ln-"+id);ln.setAttribute("x1",sx.toFixed(1));ln.setAttribute("x2",dx.toFixed(1));\n'
        '  document.getElementById("sd-"+id).setAttribute("cx",sx.toFixed(1));\n'
        '  document.getElementById("dd-"+id).setAttribute("cx",dx.toFixed(1));\n'
        '  s.sham_mice.forEach(function(mv,k){var d=document.getElementById("sm"+k+"-"+id);if(d){d.setAttribute("cx",mapx(mv[0]).toFixed(1));d.setAttribute("r",rad(mv[1]).toFixed(2));}});\n'
        '  s.d7_mice.forEach(function(mv,k){var d=document.getElementById("dm"+k+"-"+id);if(d){d.setAttribute("cx",mapx(mv[0]).toFixed(1));d.setAttribute("r",rad(mv[1]).toFixed(2));}});\n'
        '  var rn=document.getElementById("rn-"+id);rn.textContent=fmt(s.mean[0])+" \\u2192 "+fmt(s.mean[3]);\n'
        '  var delta=s.mean[3]-s.mean[0],up=delta>=0;var rd=document.getElementById("rd-"+id);\n'
        '  rd.textContent=(up?"+":"\\u2212")+Math.abs(delta).toFixed(2)+"pp";\n'
        '  rd.classList.toggle("delta-up",up);rd.classList.toggle("delta-down",!up);\n'
        '  var sy=sparkY(s.mean,cy);\n'
        '  document.getElementById("sp-"+id).setAttribute("points",SPX.map(function(x,j){return x+","+sy[j].toFixed(1);}).join(" "));\n'
        '  document.getElementById("spS-"+id).setAttribute("cy",sy[0].toFixed(1));\n'
        '  document.getElementById("spD-"+id).setAttribute("cy",sy[3].toFixed(1));\n'
        '  var ti=document.getElementById("f5ti-"+id);if(ti)ti.textContent=label+" \\u00b7 "+region+" \\u00b7 Sham "+fmt(s.mean[0])+" \\u2192 D7 "+fmt(s.mean[3])+" \\u00b7 \\u0394"+(up?"+":"\\u2212")+Math.abs(delta).toFixed(2)+"pp (mouse-of-mouse, n=2)";\n'
        '}\n'
        'function draw(st,region){\n'
        '  F5.cells.forEach(function(cm,i){drawRow(cm.id,st.cells[cm.id],CY0+DY*i,logx,region,cm.label);});\n'
        '  drawRow("unassigned",st.unassigned,UCY,linx,region,"unassigned");\n'
        '  var u=st.unassigned,tag=document.getElementById("cons-tag");\n'
        '  if(tag)tag.textContent="\\u2193 The named expansion above is funded by a "+Math.abs(u.mean[3]-u.mean[0]).toFixed(2)+"pp drop in unassigned \\u2193";\n'
        '}\n'
        # ---- value-space tween (object constancy: only positions move) ----
        'var raf=null;\n'
        'function lerp(a,b,t){return a+(b-a)*t;}\n'
        'function tweenRow(a,b,e){var s={mean:b.mean.map(function(v,i){return lerp(a.mean[i],v,e);}),\n'
        '  sham_mice:b.sham_mice.map(function(mv,k){return [lerp(a.sham_mice[k]?a.sham_mice[k][0]:mv[0],mv[0],e),mv[1]];}),\n'
        '  d7_mice:b.d7_mice.map(function(mv,k){return [lerp(a.d7_mice[k]?a.d7_mice[k][0]:mv[0],mv[0],e),mv[1]];})};return s;}\n'
        'function apply(region){fig.setAttribute("data-region",region);\n'
        '  fig.querySelectorAll("[data-set-region]").forEach(function(bt){bt.setAttribute("aria-checked",bt.getAttribute("data-set-region")===region?"true":"false");});\n'
        '  var target=stateOf(region);updateChrome(region);\n'
        '  if(reduce){cur=target;draw(cur,region);return;}\n'
        '  if(raf)cancelAnimationFrame(raf);var from=clone(cur),t0=null,D=460;\n'
        '  function step(ts){if(t0===null)t0=ts;var t=Math.min(1,(ts-t0)/D),e=ease(t);\n'
        '    var frame={cells:{},unassigned:tweenRow(from.unassigned,target.unassigned,e)};\n'
        '    F5.cells.forEach(function(cm){frame.cells[cm.id]=tweenRow(from.cells[cm.id],target.cells[cm.id],e);});\n'
        '    draw(frame,region);\n'
        '    if(t<1){raf=requestAnimationFrame(step);}else{cur=clone(target);draw(cur,region);raf=null;}}\n'
        '  raf=requestAnimationFrame(step);}\n'
        'function updateChrome(region){var r=F5.regions[region];\n'
        '  var rd=fig.querySelector(".r5-readout");\n'
        '  if(rd)rd.innerHTML="<b>"+region+"</b> \\u00b7 10\\u00b5m \\u00b7 15 discrete types + unassigned \\u00b7 support n\\u2248"+r.support_lo+"\\u2013"+r.support_hi+" patches/mouse";}\n'
        # ---- controls ----
        'fig.addEventListener("click",function(e){var t=e.target.closest("[data-set-region]");if(!t)return;apply(t.getAttribute("data-set-region"));});\n'
        'fig.addEventListener("keydown",function(e){if(/^(INPUT|TEXTAREA)$/.test(e.target.tagName))return;\n'
        '  if(e.key==="ArrowLeft"||e.key==="ArrowRight"){var r=["cortex","medulla","pooled"],j=r.indexOf(fig.getAttribute("data-region"));j=(j+(e.key==="ArrowRight"?1:2))%3;apply(r[j]);e.preventDefault();}});\n'
        'updateChrome(fig.getAttribute("data-region"));\n'
        'draw(cur,fig.getAttribute("data-region"));\n'
        '})();\n</script>'
    )


def main():
    data = build_data()
    rows = prerender(data)
    unass = prerender_unass(data)
    script = build_script(data)
    pooled = data['regions']['pooled']['unassigned']
    cons0 = abs(pooled['mean'][3] - pooled['mean'][0])
    slo, shi = data['regions']['pooled']['support_lo'], data['regions']['pooled']['support_hi']

    section = f"""{CSS}
<figure class="composition-fig" id="fig-5" data-region="pooled">
  <div class="composition-wrap">
    <p class="composition-eyebrow">Compositional motion · §4.4 · switch region</p>
    <h3 class="composition-title">The same compartments, <em>re-weighted by region</em> — flip Cortex / Medulla / Pooled and every dumbbell slides to that region's Sham→Day 7 fraction while the row order, colour and axis hold fixed.</h3>
    <div class="r5-controls">
      <span class="r5-ctl-label">Region</span>
      <div class="r5-toggle" role="radiogroup" aria-label="Kidney region">
        <button role="radio" data-set-region="cortex" aria-checked="false">Cortex</button>
        <button role="radio" data-set-region="medulla" aria-checked="false">Medulla</button>
        <button role="radio" data-set-region="pooled" aria-checked="true">Pooled</button>
      </div>
      <span class="r5-readout"><b>pooled</b> · 10µm · 15 discrete types + unassigned · support n≈{slo}–{shi} patches/mouse</span>
    </div>
    <svg class="composition-svg" viewBox="0 0 1200 620" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Interactive region-switching dumbbell: cell-type compositional shift from sham to day 7, with unassigned as the budget source; switch region.">
      <g class="header">
        <text class="panel-title" x="18.0" y="34">Inside the named fraction — risers, fallers, and the unassigned budget</text>
        <text class="panel-sub" x="18.0" y="52">Sham (hollow dot) → Day 7 (filled lineage-colored dot) · log x-axis of tissue fraction · rows held in pooled-Δ order</text>
        <g class="lineage-legend" transform="translate(886.0,22)">
          <text class="legend-title" x="300" y="6" text-anchor="end">Hover a lineage to spotlight it</text>
          <g class="legend-chip" data-lineage="immune">
            <rect class="chip-bg" x="0" y="14" width="96" height="22" rx="3"/>
            <circle class="chip-swatch" cx="14" cy="25" r="5" fill="var(--lineage-immune)"/>
            <text class="chip-label" x="26" y="29">immune</text>
          </g>
          <g class="legend-chip" data-lineage="endothelial">
            <rect class="chip-bg" x="102" y="14" width="96" height="22" rx="3"/>
            <circle class="chip-swatch" cx="116" cy="25" r="5" fill="var(--lineage-endothelial)"/>
            <text class="chip-label" x="128" y="29">endothelial</text>
          </g>
          <g class="legend-chip" data-lineage="stromal">
            <rect class="chip-bg" x="204" y="14" width="96" height="22" rx="3"/>
            <circle class="chip-swatch" cx="218" cy="25" r="5" fill="var(--lineage-stromal)"/>
            <text class="chip-label" x="230" y="29">stromal</text>
          </g>
        </g>
      </g>
      <g class="x-axis-named">
        <line class="xgrid" x1="230.0" y1="80.0" x2="230.0" y2="476.0"/>
        <text class="xtick" x="230.0" y="76.0" text-anchor="middle">0.01%</text>
        <line class="xgrid" x1="420.0" y1="80.0" x2="420.0" y2="476.0"/>
        <text class="xtick" x="420.0" y="76.0" text-anchor="middle">0.1%</text>
        <line class="xgrid" x1="610.0" y1="80.0" x2="610.0" y2="476.0"/>
        <text class="xtick" x="610.0" y="76.0" text-anchor="middle">1%</text>
        <line class="xgrid" x1="800.0" y1="80.0" x2="800.0" y2="476.0"/>
        <text class="xtick" x="800.0" y="76.0" text-anchor="middle">10%</text>
        <text class="x-axis-title" x="515.0" y="60.0" text-anchor="middle">tissue fraction (log)</text>
      </g>
      <g class="rows">
{rows}
      </g>
      <line class="row-sep" x1="18.0" y1="488.0" x2="1186.0" y2="488.0"/>
      <text class="conservation-tag" id="cons-tag" x="515.0" y="506.0" text-anchor="middle">↓ The named expansion above is funded by a {cons0:.2f}pp drop in unassigned ↓</text>
      <g class="x-axis-unass">
        <line class="xgrid xgrid-unass" x1="230.0" y1="532.0" x2="230.0" y2="566.0"/>
        <text class="xtick xtick-unass" x="230.0" y="527.0" text-anchor="middle">80%</text>
        <line class="xgrid xgrid-unass" x1="458.0" y1="532.0" x2="458.0" y2="566.0"/>
        <text class="xtick xtick-unass" x="458.0" y="527.0" text-anchor="middle">84%</text>
        <line class="xgrid xgrid-unass" x1="686.0" y1="532.0" x2="686.0" y2="566.0"/>
        <text class="xtick xtick-unass" x="686.0" y="527.0" text-anchor="middle">88%</text>
      </g>
{unass}
      <text class="scale-note" x="515.0" y="584.0" text-anchor="middle">↑ unassigned uses its own linear scale (80–90%); named cells use a log scale (0.01–10%) ↑</text>
    </svg>
  </div>
  <figcaption>
    <span class="num">Fig. 5</span> One row per cell type, held in <strong>pooled-Δ order</strong> so each row stays one object as you switch region — flip <em>Cortex / Medulla / Pooled</em> and the dumbbells slide horizontally to that region's Sham (hollow) → Day 7 (filled) fraction on the log x-axis, with the per-mouse points, sparklines (Day 1 / Day 3 intermediates), numbers, Δ and the unassigned budget morphing in place (object constancy; the axis, colour and row order are held fixed). Per-mouse points (n=2, hover a row) carry area ∝ √n_support, so the single-region slices — fewer patches per mouse — visibly shrink. Cell-type annotation exists <strong>only at 10µm</strong>, so the lens here is <strong>region</strong>, not scale. Descriptive; no significance is claimed.
    <span class="source">Source: results/biological_analysis/differential_abundance/roi_abundances.csv · mouse-of-mouse means, 4 timepoints × 2 mice × 3 ROIs/mouse (cortex/medulla split within mouse). n=2 mice — no region test.</span>
  </figcaption>
</figure>
{script}"""
    OUT.write_text(section, encoding='utf-8')
    print(f"wrote {OUT} ({len(section)} chars); regions={list(data['regions'])}; "
          f"cells={len(data['cells'])}; pooled unass Δ={cons0:.2f}pp")


if __name__ == '__main__':
    main()
