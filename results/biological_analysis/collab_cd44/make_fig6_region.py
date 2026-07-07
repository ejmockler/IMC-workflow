"""Region lens for the §4.5 triple-positive interface (Fig 6).

Fig 6 is a ternary interface grid: 4 timepoints x 2 mice x 3 ROIs of pre-rendered
RGB ternary point clouds (R=immune, G=fibroblast, B=vessel), whose headline is the
cohort triple-positive trajectory 14.6% -> 19.3% -> 26.7% -> 32.6% (Sham->D7).

The honest interactive lens here is REGION, and it is clean for a specific reason:
region (Cortex / Medulla) is a WHOLE-ROI label in this study, not a within-ROI
split (results/biological_analysis/differential_abundance/roi_abundances.csv maps
each roi_id -> one region). The design is perfectly balanced -- exactly 3 Cortex + 3
Medulla ROIs at EVERY timepoint (12/12 overall). So a region lens does NOT need to
touch the ternary geometry (region does not partition a point cloud); it just selects
which ROIs aggregate into the triple+ trajectory. Cortex-only and Medulla-only
trajectories are each built from 3 ROIs/timepoint, n=2 mice -> honest, unconfounded
with time.

This reuses Fig 6's own authoritative per-ROI interface fractions (window.FIG6_DATA
embedded in pi_report.html; read-only -- the report is never modified) so the Pooled
trajectory reproduces the report headline bit-for-bit, and joins the per-ROI region
label from the pipeline's roi_abundances.csv.

Same object-constancy morph as the §4.1 / §4.2 figures: on a Region switch only the
Y-positions tween (x-anchors, colour, axis, labels held fixed; 460ms cubic-ease rAF;
prefers-reduced-motion jump-cut). Per-mouse points (n=2, no error bars); dot area
proportional to sqrt(n_support patches). Descriptive; adds no pre-registered endpoint.

Emits fig6_region.html (a standalone section; splice near <figure id="fig-6">).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REPORT = REPO / 'review_packet' / 'pi_report.html'
ABUNDANCES = REPO / 'results' / 'biological_analysis' / 'differential_abundance' / 'roi_abundances.csv'
OUT = HERE / 'fig6_region.html'

TPS = ['Sham', 'D1', 'D3', 'D7']
MICE = ['MS1', 'MS2']
REGIONS = ['Cortex', 'Medulla', 'Pooled']

# geometry (mirrors the JS; 0%->440, 40%->80 => 360px spans 40pp)
XT = [248.8, 466.2, 683.8, 901.2]
YMAX_PCT = 40.0
def yv(p): return 440.0 - (360.0 / YMAX_PCT) * p
def radius(n): return max(3.0, min(7.0, 1.4 + 1.6 * (n / 1000.0) ** 0.5))


def load_fig6_data() -> dict:
    """Authoritative per-ROI interface fractions from the report (read-only)."""
    html = REPORT.read_text(encoding='utf-8')
    m = re.search(r'window\.FIG6_DATA\s*=\s*(\{.*?\})\s*;?\s*</script>', html, re.S)
    if not m:
        raise SystemExit('FIG6_DATA not found in pi_report.html')
    return json.loads(m.group(1))


def load_region_map() -> dict:
    """roi_key (loader-style, unprefixed) -> 'Cortex' / 'Medulla'."""
    ab = pd.read_csv(ABUNDANCES)
    ab['roi_key'] = ab['roi_id'].str.replace(r'^roi_', '', regex=True)
    return dict(zip(ab['roi_key'], ab['region']))


def build_data() -> dict:
    fig6 = load_fig6_data()
    regmap = load_region_map()
    # tidy per-ROI frame: roi, tp, mouse, region, pct_triple, n
    recs = []
    for roi, v in fig6['rois'].items():
        reg = regmap.get(roi)
        if reg is None:
            raise SystemExit(f'region join incomplete: {roi} missing a region')
        recs.append(dict(roi=roi, tp=v['tp'], mouse=v['mouse'], region=reg,
                         pct=float(v['pct_triple']), n=int(v['n'])))
    df = pd.DataFrame(recs)

    def region_frame(sub: pd.DataFrame) -> dict:
        mean, mice = [], {}
        for tp in TPS:
            s = sub[sub.tp == tp]
            # trajectory line = mean over the region's ROIs (reproduces the report headline for Pooled)
            mean.append(round(float(s['pct'].mean()), 3))
            pts = []
            for m in MICE:
                sm = s[s.mouse == m]
                if len(sm):
                    pts.append([round(float(sm['pct'].mean()), 3), int(sm['n'].sum())])
            mice[tp] = pts
        return {'mean': mean, 'mice': mice, 'delta': round(mean[3] - mean[0], 1)}

    regions = {}
    for reg in ('Cortex', 'Medulla'):
        regions[reg] = region_frame(df[df.region == reg])
    regions['Pooled'] = region_frame(df)
    return {'tps': TPS, 'regions': regions}


def prerender(data: dict, region='Pooled') -> str:
    r = data['regions'][region]
    p = []
    # Sham-mean baseline (morphs with region)
    p.append(f'    <line id="t-baseline" class="ref-baseline" x1="140" y1="{yv(r["mean"][0]):.1f}" x2="1010" y2="{yv(r["mean"][0]):.1f}"/>')
    p.append(f'    <text id="t-baseline-lab" class="ref-label" x="1016" y="{yv(r["mean"][0]):.1f}" dominant-baseline="middle">Sham mean</text>')
    # trajectory polyline
    pts = ' '.join(f'{XT[i]:.1f},{yv(r["mean"][i]):.1f}' for i in range(4))
    p.append(f'    <polyline id="t-line" class="trajectory" points="{pts}"/>')
    # per-mouse dots (n=2), area ~ sqrt(n_support patches)
    for i, tp in enumerate(TPS):
        for mi, mv in enumerate(r['mice'][tp]):
            cx = XT[i] + (mi * 2 - 1) * 9
            p.append(f'    <g class="mouse-marker" data-tp="{tp}" data-mouse="{MICE[mi]}"><circle id="t-mo{mi}-{i}" class="mouse-dot" cx="{cx:.1f}" cy="{yv(mv[0]):.1f}" r="{radius(mv[1]):.1f}"/></g>')
    # mean dots + titles + per-tp summary pct
    for i, tp in enumerate(TPS):
        p.append(f'    <g class="mean-marker" data-tp="{tp}"><circle id="t-dot-{i}" class="mean-dot" cx="{XT[i]:.1f}" cy="{yv(r["mean"][i]):.1f}" r="7"/><title id="t-ti-{i}"></title></g>')
        p.append(f'    <text id="t-pct-{i}" class="summary-pct" x="{XT[i]:.1f}" y="464" text-anchor="middle">{r["mean"][i]:.1f}%</text>')
    return '\n'.join(p)


def build_script(data: dict) -> str:
    payload = json.dumps(data, separators=(',', ':'))
    return (
        '<script>\n(function(){\n'
        'var F6 = ' + payload + ';\n'
        'var fig = document.getElementById("fig-6r");\n'
        'if(!fig) return;\n'
        'var XT=[248.8,466.2,683.8,901.2], TPS=["Sham","D1","D3","D7"], MICE=["MS1","MS2"];\n'
        'function y(p){return 440-9*p;}\n'
        'function rad(n){return Math.max(3,Math.min(7,1.4+1.6*Math.sqrt(n/1000)));}\n'
        'function ease(t){return t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;}\n'
        'var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;\n'
        'var cur = F6.regions[fig.getAttribute("data-region")].mean.slice();\n'
        'function draw(mean,region){var reg=F6.regions[region];\n'
        '  document.getElementById("t-line").setAttribute("points",mean.map(function(v,i){return XT[i]+","+y(v).toFixed(1);}).join(" "));\n'
        '  document.getElementById("t-baseline").setAttribute("y1",y(mean[0]).toFixed(1));\n'
        '  document.getElementById("t-baseline").setAttribute("y2",y(mean[0]).toFixed(1));\n'
        '  document.getElementById("t-baseline-lab").setAttribute("y",y(mean[0]).toFixed(1));\n'
        '  for(var i=0;i<4;i++){\n'
        '    document.getElementById("t-dot-"+i).setAttribute("cy",y(mean[i]).toFixed(1));\n'
        '    document.getElementById("t-pct-"+i).textContent=mean[i].toFixed(1)+"%";\n'
        '    document.getElementById("t-ti-"+i).textContent=TPS[i]+" \\u00b7 "+region+" \\u00b7 triple+ interface "+mean[i].toFixed(1)+"% (mean of 3 ROIs; n=2 mice)";\n'
        '    var mm=reg.mice[TPS[i]];\n'
        '    for(var k=0;k<mm.length;k++){var d=document.getElementById("t-mo"+k+"-"+i);if(d){d.setAttribute("cy",y(mm[k][0]).toFixed(1));d.setAttribute("r",rad(mm[k][1]).toFixed(1));}}\n'
        '  }\n'
        '  var hn=document.getElementById("t-headnum");if(hn)hn.textContent=reg.mean[0].toFixed(1)+"% \\u2192 "+reg.mean[3].toFixed(1)+"%";\n'
        '  var dl=document.getElementById("t-delta");if(dl)dl.textContent=(reg.delta>=0?"+":"")+reg.delta.toFixed(1)+" pp Sham\\u2192D7";\n'
        '  var npatch=0;for(var t=0;t<4;t++){reg.mice[TPS[t]].forEach(function(mv){npatch+=mv[1];});}\n'
        '  var rd=fig.querySelector(".t-readout");if(rd)rd.innerHTML="<b>"+region+"</b> \\u00b7 10\\u00b5m \\u00b7 triple+ = a patch above threshold on all three lineages (immune \\u2227 fibroblast \\u2227 vessel)"+(region!=="Pooled"?" \\u00b7 3 ROIs/timepoint":"");\n'
        '}\n'
        'var raf=null;\n'
        'function apply(region){fig.setAttribute("data-region",region);var target=F6.regions[region].mean.slice();\n'
        '  fig.querySelectorAll("[data-set-region]").forEach(function(b){b.setAttribute("aria-checked",b.getAttribute("data-set-region")===region?"true":"false");});\n'
        '  if(reduce){cur=target;draw(cur,region);return;}\n'
        '  if(raf)cancelAnimationFrame(raf);var from=cur.slice(),t0=null;\n'
        '  function step(ts){if(t0===null)t0=ts;var t=Math.min(1,(ts-t0)/460),e=ease(t);\n'
        '    cur=target.map(function(v,i){return from[i]+(v-from[i])*e;});draw(cur,region);\n'
        '    if(t<1)raf=requestAnimationFrame(step);else{cur=target.slice();draw(cur,region);raf=null;}}\n'
        '  raf=requestAnimationFrame(step);}\n'
        'fig.addEventListener("click",function(e){var b=e.target.closest("[data-set-region]");if(!b)return;apply(b.getAttribute("data-set-region"));});\n'
        'fig.addEventListener("keydown",function(e){if(/^(INPUT|TEXTAREA)$/.test(e.target.tagName))return;if(e.key==="ArrowLeft"||e.key==="ArrowRight"){var r=["Cortex","Medulla","Pooled"],j=r.indexOf(fig.getAttribute("data-region"));j=(j+(e.key==="ArrowRight"?1:2))%3;apply(r[j]);e.preventDefault();}});\n'
        'draw(cur,fig.getAttribute("data-region"));\n'
        '})();\n</script>'
    )


CSS = """<style>
  #fig-6r .t-controls { display:flex; flex-wrap:wrap; align-items:center; gap:8px 12px; margin:6px 0 10px; }
  #fig-6r .t-ctl-label { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-faint); }
  #fig-6r .t-toggle { display:inline-flex; border:1px solid var(--rule); border-radius:3px; overflow:hidden; }
  #fig-6r .t-toggle button { appearance:none; border:0; background:var(--paper); color:var(--ink-soft); font:600 11px/1 ui-monospace,Menlo,monospace; padding:6px 11px; cursor:pointer; border-left:1px solid var(--rule); }
  #fig-6r .t-toggle button:first-child { border-left:0; }
  #fig-6r .t-toggle button[aria-checked="true"] { background:var(--ink); color:var(--paper); }
  #fig-6r .t-toggle button:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
  #fig-6r .t-readout { font-size:12px; color:var(--ink-faint); }
  #fig-6r .t-readout b { color:var(--ink-soft); }
  #fig-6r .t-delta { font-size:12px; color:var(--ink-soft); font-variant-numeric:tabular-nums; font-weight:600; }
  #fig-6r .mouse-dot { fill:var(--paper); stroke:var(--accent); stroke-width:2; opacity:.85; }
</style>"""


def main():
    data = build_data()
    marks = prerender(data, 'Pooled')
    script = build_script(data)
    pooled = data['regions']['Pooled']
    head0 = f'{pooled["mean"][0]:.1f}% → {pooled["mean"][3]:.1f}%'
    section = f"""<figure class="trajectory-fig" id="fig-6r" data-region="Pooled">
  <div class="trajectory-wrap">
    <p class="trajectory-eyebrow">The triple+ interface, by region · §4.5 companion</p>
    <h3 class="trajectory-title">The multi-lineage (triple-positive) interface rises <span class="head-num" id="t-headnum">{head0}</span> Sham→D7 — and the rise is <em>steeper in cortex than medulla</em>. Switch region.</h3>
    {CSS}
    <div class="t-controls">
      <span class="t-ctl-label">Region</span>
      <div class="t-toggle" role="radiogroup" aria-label="Kidney region">
        <button role="radio" data-set-region="Cortex" aria-checked="false">Cortex</button>
        <button role="radio" data-set-region="Medulla" aria-checked="false">Medulla</button>
        <button role="radio" data-set-region="Pooled" aria-checked="true">Pooled</button>
      </div>
      <span class="t-delta" id="t-delta">{"+" if pooled["delta"]>=0 else ""}{pooled["delta"]:.1f} pp Sham→D7</span>
      <span class="t-readout"><b>Pooled</b> · 10µm · triple+ = a patch above threshold on all three lineages (immune ∧ fibroblast ∧ vessel)</span>
    </div>
<svg class="trajectory-svg" viewBox="0 0 1100 560" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Interactive triple-positive interface trajectory by region; switch cortex / medulla / pooled.">
  <g class="plot-bg">
    <line class="grid" x1="140" y1="440.0" x2="1010" y2="440.0"/>
    <text class="ytick" x="128" y="440.0" text-anchor="end" dominant-baseline="middle">0%</text>
    <line class="grid" x1="140" y1="350.0" x2="1010" y2="350.0"/>
    <text class="ytick" x="128" y="350.0" text-anchor="end" dominant-baseline="middle">10%</text>
    <line class="grid" x1="140" y1="260.0" x2="1010" y2="260.0"/>
    <text class="ytick" x="128" y="260.0" text-anchor="end" dominant-baseline="middle">20%</text>
    <line class="grid" x1="140" y1="170.0" x2="1010" y2="170.0"/>
    <text class="ytick" x="128" y="170.0" text-anchor="end" dominant-baseline="middle">30%</text>
    <line class="grid" x1="140" y1="80.0" x2="1010" y2="80.0"/>
    <text class="ytick" x="128" y="80.0" text-anchor="end" dominant-baseline="middle">40%</text>
    <text class="axis-label" transform="translate(34,260.0) rotate(-90)" text-anchor="middle">triple+ interface (% of patches)</text>
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
    <figcaption>Triple-positive interface fraction across Sham→D1→D3→D7. Switch <em>region</em> (cortex / medulla / pooled) and the trajectory, per-mouse points, Sham-mean baseline, headline and Δ morph in place. Region is a whole-ROI label (each ROI is cortex or medulla), balanced <strong>3 cortex + 3 medulla ROIs at every timepoint</strong>, so the region split does not touch the ternary geometry of Fig 6 — it re-aggregates the same per-ROI triple+ fractions. The line is the mean across the region's 3 ROIs (Pooled reproduces the Fig 6 headline exactly); the two dots are per-mouse means (n=2, no error bars); dot area ∝ √n_support (patches contributing). Cortex rises the more steeply of the two — consistent with the cortex-predominant CD44 activation in §4.6. Descriptive; n=2 mice, paired within-mouse — no region test.
    <span class="source">Source: per-ROI triple+ fractions from Fig 6 (window.FIG6_DATA; results/biological_analysis/cell_type_annotations/, 10µm) joined to region from results/biological_analysis/differential_abundance/roi_abundances.csv. Descriptive; n=2 mice.</span></figcaption>
  </div>
</figure>
{script}"""
    OUT.write_text(section, encoding='utf-8')
    print(f"wrote {OUT} ({len(section)} chars)")
    for reg in REGIONS:
        r = data['regions'][reg]
        print(f"  {reg:8s} triple+: " + " -> ".join(f"{m:.1f}%" for m in r['mean']) + f"   (Δ {r['delta']:+.1f} pp)")


if __name__ == '__main__':
    main()
