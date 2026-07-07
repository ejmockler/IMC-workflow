"""Rebuild Fig 2 (§4.1 neutrophil headline) as an INTERACTIVE region-switching
trajectory — no static headline. Neutrophil is 10µm-only (no grid-scale basis),
so the lens is Region (cortex / medulla / pooled), not Scale. Same object-
constancy morph as the §4.2 figure: on switch only Y moves; the headline number,
trajectory, per-mouse dots, Sham-mean baseline and per-timepoint labels update.

Emits fig2_neutrophil.html (spliced to REPLACE the static <figure id="fig-2">).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV10 = HERE / 'cd44_compartment_rates_10um.csv'
OUT = HERE / 'fig2_neutrophil.html'

TPS = ['Sham', 'D1', 'D3', 'D7']
MICE = ['MS1', 'MS2']
REGIONS = ['cortex', 'medulla', 'pooled']
XT = [248.8, 466.2, 683.8, 901.2]           # timepoint x-anchors (match original)
def yv(r): return 440.0 - 360.0 * r          # 0%→440, 100%→80
def radius(n): return max(2.0, min(6.5, 1.5 + 1.8 * (n / 300.0) ** 0.5))


def build_data() -> dict:
    df = pd.read_csv(CSV10)
    g = df[df.compartment == 'neutrophil']
    regions = {}
    for reg in REGIONS:
        mean, mice = [], {}
        for tp in TPS:
            sub = g[(g.region == reg) & (g.timepoint == tp)]
            mean.append(round(float(sub['cd44_rate'].mean()), 4))
            mice[tp] = [[round(float(sub[sub.mouse == m]['cd44_rate'].iloc[0]), 4),
                         int(sub[sub.mouse == m]['n_support'].iloc[0])]
                        for m in MICE if len(sub[sub.mouse == m])]
        regions[reg] = {'mean': mean, 'mice': mice}
    return {'tps': TPS, 'regions': regions}


def prerender(data: dict, region='pooled') -> str:
    r = data['regions'][region]
    p = []
    # Sham-mean baseline
    p.append(f'    <line id="n-baseline" class="ref-baseline" x1="140" y1="{yv(r["mean"][0]):.1f}" x2="1010" y2="{yv(r["mean"][0]):.1f}"/>')
    p.append(f'    <text id="n-baseline-lab" class="ref-label" x="1016" y="{yv(r["mean"][0]):.1f}" dominant-baseline="middle">Sham mean</text>')
    # trajectory polyline
    pts = ' '.join(f'{XT[i]:.1f},{yv(r["mean"][i]):.1f}' for i in range(4))
    p.append(f'    <polyline id="n-line" class="trajectory" points="{pts}"/>')
    # per-mouse dots
    for i, tp in enumerate(TPS):
        for mi, mv in enumerate(r['mice'][tp]):
            cx = XT[i] + (mi * 2 - 1) * 9
            p.append(f'    <circle id="n-m{mi}-{i}" class="mouse-dot" cx="{cx:.1f}" cy="{yv(mv[0]):.1f}" r="{radius(mv[1]):.1f}"/>')
    # mean dots + titles + per-tp summary pct
    for i, tp in enumerate(TPS):
        p.append(f'    <g class="mean-marker" data-tp="{tp}"><circle id="n-dot-{i}" class="mean-dot" cx="{XT[i]:.1f}" cy="{yv(r["mean"][i]):.1f}" r="7"/><title id="n-ti-{i}"></title></g>')
        p.append(f'    <text id="n-pct-{i}" class="summary-pct" x="{XT[i]:.1f}" y="464" text-anchor="middle">{round(r["mean"][i]*100)}%</text>')
    return '\n'.join(p)


def build_script(data: dict) -> str:
    payload = json.dumps(data, separators=(',', ':'))
    return (
        '<script>\n(function(){\n'
        'var F2 = ' + payload + ';\n'
        'var fig = document.getElementById("fig-2");\n'
        'if(!fig) return;\n'
        'var XT=[248.8,466.2,683.8,901.2], TPS=["Sham","D1","D3","D7"];\n'
        'function y(r){return 440-360*r;}\n'
        'function rad(n){return Math.max(2,Math.min(6.5,1.5+1.8*Math.sqrt(n/300)));}\n'
        'function ease(t){return t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;}\n'
        'var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;\n'
        'var cur = F2.regions[fig.getAttribute("data-region")].mean.slice();\n'
        'function draw(mean,region){\n'
        '  document.getElementById("n-line").setAttribute("points",mean.map(function(v,i){return XT[i]+","+y(v).toFixed(1);}).join(" "));\n'
        '  document.getElementById("n-baseline").setAttribute("y1",y(mean[0]).toFixed(1));\n'
        '  document.getElementById("n-baseline").setAttribute("y2",y(mean[0]).toFixed(1));\n'
        '  document.getElementById("n-baseline-lab").setAttribute("y",y(mean[0]).toFixed(1));\n'
        '  var reg=F2.regions[region];\n'
        '  for(var i=0;i<4;i++){\n'
        '    document.getElementById("n-dot-"+i).setAttribute("cy",y(mean[i]).toFixed(1));\n'
        '    document.getElementById("n-pct-"+i).textContent=Math.round(mean[i]*100)+"%";\n'
        '    document.getElementById("n-ti-"+i).textContent=TPS[i]+" \\u00b7 "+region+" \\u00b7 neutrophil-typed CD44\\u207a "+Math.round(mean[i]*100)+"% (mouse-of-mouse mean, n=2)";\n'
        '    var mm=reg.mice[TPS[i]];\n'
        '    for(var k=0;k<mm.length;k++){var d=document.getElementById("n-m"+k+"-"+i);if(d){d.setAttribute("cy",y(mm[k][0]).toFixed(1));d.setAttribute("r",rad(mm[k][1]).toFixed(1));}}\n'
        '  }\n'
        '  var hn=document.getElementById("n-headnum");if(hn)hn.textContent=Math.round(reg.mean[0]*100)+"% \\u2192 "+Math.round(reg.mean[3]*100)+"%";\n'
        '  var rd=fig.querySelector(".n-readout");if(rd)rd.innerHTML="<b>"+region+"</b> \\u00b7 10\\u00b5m \\u00b7 neutrophil-typed patches (no grid-scale basis \\u2014 region lens only)";\n'
        '}\n'
        'var raf=null;\n'
        'function apply(region){fig.setAttribute("data-region",region);var target=F2.regions[region].mean.slice();\n'
        '  fig.querySelectorAll("[data-set-region]").forEach(function(b){b.setAttribute("aria-checked",b.getAttribute("data-set-region")===region?"true":"false");});\n'
        '  if(reduce){cur=target;draw(cur,region);return;}\n'
        '  if(raf)cancelAnimationFrame(raf);var from=cur.slice(),t0=null;\n'
        '  function step(ts){if(t0===null)t0=ts;var t=Math.min(1,(ts-t0)/460),e=ease(t);\n'
        '    cur=target.map(function(v,i){return from[i]+(v-from[i])*e;});draw(cur,region);\n'
        '    if(t<1)raf=requestAnimationFrame(step);else{cur=target.slice();draw(cur,region);raf=null;}}\n'
        '  raf=requestAnimationFrame(step);}\n'
        'fig.addEventListener("click",function(e){var t=e.target.closest("[data-set-region]");if(!t)return;apply(t.getAttribute("data-set-region"));});\n'
        'fig.addEventListener("keydown",function(e){if(/^(INPUT|TEXTAREA)$/.test(e.target.tagName))return;if(e.key==="ArrowLeft"||e.key==="ArrowRight"){var r=["cortex","medulla","pooled"],j=r.indexOf(fig.getAttribute("data-region"));j=(j+(e.key==="ArrowRight"?1:2))%3;apply(r[j]);e.preventDefault();}});\n'
        'draw(cur,fig.getAttribute("data-region"));\n'
        '})();\n</script>'
    )


CSS = """<style>
  #fig-2 .n-controls { display:flex; flex-wrap:wrap; align-items:center; gap:8px 12px; margin:6px 0 10px; }
  #fig-2 .n-ctl-label { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-faint); }
  #fig-2 .n-toggle { display:inline-flex; border:1px solid var(--rule); border-radius:3px; overflow:hidden; }
  #fig-2 .n-toggle button { appearance:none; border:0; background:var(--paper); color:var(--ink-soft); font:600 11px/1 ui-monospace,Menlo,monospace; padding:6px 11px; cursor:pointer; border-left:1px solid var(--rule); }
  #fig-2 .n-toggle button:first-child { border-left:0; }
  #fig-2 .n-toggle button[aria-checked="true"] { background:var(--ink); color:var(--paper); }
  #fig-2 .n-toggle button:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
  #fig-2 .n-readout { font-size:12px; color:var(--ink-faint); }
  #fig-2 .n-readout b { color:var(--ink-soft); }
  #fig-2 .mouse-dot { fill:var(--accent); stroke:var(--paper); stroke-width:1.4; opacity:.55; }
  #fig-2 .trajectory, #fig-2 .mean-dot { transition:none; }
</style>"""


def main():
    data = build_data()
    marks = prerender(data, 'pooled')
    script = build_script(data)
    section = f"""<figure class="trajectory-fig" id="fig-2" data-region="pooled">
  <div class="trajectory-wrap">
    <p class="trajectory-eyebrow">Largest single effect in the cohort — switch region</p>
    <h3 class="trajectory-title">CD44⁺ activation rate within neutrophil-typed tissue rises <span class="head-num" id="n-headnum">32% → 81%</span> from sham to day 7.</h3>
    {CSS}
    <div class="n-controls">
      <span class="n-ctl-label">Region</span>
      <div class="n-toggle" role="radiogroup" aria-label="Kidney region">
        <button role="radio" data-set-region="cortex" aria-checked="false">Cortex</button>
        <button role="radio" data-set-region="medulla" aria-checked="false">Medulla</button>
        <button role="radio" data-set-region="pooled" aria-checked="true">Pooled</button>
      </div>
      <span class="n-readout"><b>pooled</b> · 10µm · neutrophil-typed patches (no grid-scale basis — region lens only)</span>
    </div>
<svg class="trajectory-svg" viewBox="0 0 1100 560" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Interactive neutrophil-typed CD44 activation trajectory; switch region.">
  <g class="plot-bg">
    <line class="grid" x1="140" y1="440.0" x2="1010" y2="440.0"/>
    <text class="ytick" x="128" y="440.0" text-anchor="end" dominant-baseline="middle">0%</text>
    <line class="grid" x1="140" y1="350.0" x2="1010" y2="350.0"/>
    <text class="ytick" x="128" y="350.0" text-anchor="end" dominant-baseline="middle">25%</text>
    <line class="grid" x1="140" y1="260.0" x2="1010" y2="260.0"/>
    <text class="ytick" x="128" y="260.0" text-anchor="end" dominant-baseline="middle">50%</text>
    <line class="grid" x1="140" y1="170.0" x2="1010" y2="170.0"/>
    <text class="ytick" x="128" y="170.0" text-anchor="end" dominant-baseline="middle">75%</text>
    <line class="grid" x1="140" y1="80.0" x2="1010" y2="80.0"/>
    <text class="ytick" x="128" y="80.0" text-anchor="end" dominant-baseline="middle">100%</text>
    <text class="axis-label" transform="translate(34,260.0) rotate(-90)" text-anchor="middle">CD44⁺ rate among neutrophil-typed patches</text>
    <text class="xtick-main" x="248.8" y="498" text-anchor="middle">Sham</text>
    <text class="xtick-sub"  x="248.8" y="518" text-anchor="middle">baseline</text>
    <text class="xtick-main" x="466.2" y="498" text-anchor="middle">Day 1</text>
    <text class="xtick-sub"  x="466.2" y="518" text-anchor="middle">acute inflammation</text>
    <text class="xtick-main" x="683.8" y="498" text-anchor="middle">Day 3</text>
    <text class="xtick-sub"  x="683.8" y="518" text-anchor="middle">myeloid expansion</text>
    <text class="xtick-main" x="901.2" y="498" text-anchor="middle">Day 7</text>
    <text class="xtick-sub"  x="901.2" y="518" text-anchor="middle">early fibrotic remodeling</text>
{marks}
  </g>
</svg>
    <figcaption>Neutrophil-typed CD44⁺ activation across Sham→D1→D3→D7. Switch <em>region</em> (cortex / medulla / pooled) and the trajectory, per-mouse points, Sham-mean baseline and the headline number morph in place. Neutrophil requires cell-type gating, which exists only at 10µm — so this figure has a region lens, not a scale lens. Per-mouse points (n=2), no error bars; dot area ∝ √n_support.
    <span class="source">Source: results/biological_analysis/collab_cd44/cd44_compartment_rates_10um.csv (neutrophil × region × timepoint × mouse). Descriptive; n=2 mice.</span></figcaption>
  </div>
</figure>
{script}"""
    OUT.write_text(section, encoding='utf-8')
    print(f"wrote {OUT} ({len(section)} chars)")


if __name__ == '__main__':
    main()
