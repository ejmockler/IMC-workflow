"""Generate the §4.6 INTERACTIVE dimension-switcher for pi_report.html.

Design = the "Same lines, new slice" panel (perceptual-engineering design panel
winner): ONE Sham→D7 CD44⁺ slopegraph on a fixed 0–100% axis; two segmented
controls (Scale 10/20/40µm · Region cortex/medulla/pooled) select the visible
slice; on a switch only the Y-positions morph (axis, x-anchors, colour, dash,
labels all held fixed) so each compartment reads as ONE object that moved
(object constancy). Region switch = motion/swing; Scale switch = stillness +
ghost. Self-contained, CSP-safe (inline data + vanilla JS, no libs). Emits
section_4_6.html (spliced into the report before §5).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ALLSCALES = HERE / 'cd44_compartment_rates_allscales.csv'
SUMMARY = HERE / 'cd44_crossscale_summary.csv'
OUT = HERE / 'section_4_2_insert.html'

TPS = ['Sham', 'D1', 'D3', 'D7']
MICE = ['MS1', 'MS2']
SCALES = [10.0, 20.0, 40.0]
REGIONS = ['cortex', 'medulla', 'pooled']

# compartment vocab: id, label, colour token, dash, stroke weight, headline
COMPS = [
    ('CD206', 'CD206⁺ (M2-like)', 'var(--accent)', '', 3.4, True),
    ('neutrophil', 'Neutrophil-typed', 'var(--cd44-neut)', '', 3.4, True),
    ('CD140b', 'CD140b⁺ (pericyte)', 'var(--lineage-stromal)', '', 2.4, False),
    ('CD45', 'CD45⁺ (immune)', 'var(--lineage-immune)', '', 2.4, False),
    ('CD34', 'CD34⁺ (vascular)', 'var(--lineage-endothelial)', '6 5', 2.2, False),
    ('CD31', 'CD31⁺ (vascular)', 'var(--lineage-endothelial)', '', 2.2, False),
    ('endothelial_cd31cd34', 'Endothelial (CD31∧CD34)', 'var(--ink-soft)', '5 4', 2.2, False),
]
COMP_IDS = [c[0] for c in COMPS]


def build_data() -> dict:
    df = pd.read_csv(ALLSCALES)
    slices = {}
    for scale in SCALES:
        for region in REGIONS:
            key = f'{scale:g}|{region}'
            sl = {}
            sub = df[(df.scale_um == scale) & (df.region == region)]
            for comp in COMP_IDS:
                g = sub[sub.compartment == comp]
                if g.empty:
                    continue  # neutrophil at grid scales — absent, not fabricated
                mean = []
                for tp in TPS:
                    v = g[g.timepoint == tp]['cd44_rate']
                    mean.append(round(float(v.mean()), 4) if len(v) else None)
                if any(m is None for m in mean):
                    continue
                def mice_at(tp):
                    out = []
                    for m in MICE:
                        r = g[(g.timepoint == tp) & (g.mouse == m)]
                        if len(r):
                            out.append([round(float(r['cd44_rate'].iloc[0]), 4),
                                        int(r['n_support'].iloc[0])])
                    return out
                sl[comp] = {'mean': mean, 'sham': mice_at('Sham'), 'd7': mice_at('D7'),
                            'delta': round((mean[3] - mean[0]) * 100, 1)}
            slices[key] = sl
    return {'tps': TPS, 'slices': slices,
            'comps': [{'id': c[0], 'label': c[1], 'color': c[2], 'dash': c[3],
                       'w': c[4], 'head': c[5]} for c in COMPS]}


# ---- geometry (mirrors the JS; used to pre-render the default 10|pooled slice) ----
XS = {0: 250.0, 1: 440.0, 2: 630.0, 3: 820.0}
def yv(r): return 450.0 - 360.0 * r
def radius(n): return max(2.0, min(7.5, 1.5 + 2.0 * (n / 300.0) ** 0.5))


def prerender_default(data: dict) -> str:
    """Static marks for 10|pooled so the figure is a valid chart with JS off."""
    sl = data['slices']['10|pooled']
    # greedy label de-collision on D7 means (top→down, min gap 16)
    order = sorted([c['id'] for c in data['comps'] if c['id'] in sl],
                   key=lambda cid: yv(sl[cid]['mean'][3]))
    lyy, prev = {}, None
    for cid in order:
        ty = yv(sl[cid]['mean'][3])
        if prev is not None and ty < prev + 16:
            ty = prev + 16
        lyy[cid] = ty
        prev = ty
    p = []
    for c in data['comps']:
        cid = c['id']
        if cid not in sl:
            continue
        s = sl[cid]
        pts = ' '.join(f'{XS[i]:.1f},{yv(s["mean"][i]):.1f}' for i in range(4))
        dash = f' stroke-dasharray="{c["dash"]}"' if c['dash'] else ''
        cls = 'cd44-line headline' if c['head'] else 'cd44-line'
        p.append(f'    <g class="cascade-marker cd44-comp" id="cd44-{cid}" data-comp="{cid}" style="--cascade-stroke: {c["color"]};" tabindex="0">')
        p.append(f'      <polyline id="pl-{cid}" class="{cls}" points="{pts}" fill="none" stroke-width="{c["w"]}"{dash}/>')
        for i in (1, 2):  # waypoints
            p.append(f'      <circle id="w{i}-{cid}" class="cd44-way" cx="{XS[i]:.1f}" cy="{yv(s["mean"][i]):.1f}" r="3"/>')
        for anc, key in ((0, 'sham'), (3, 'd7')):  # framing anchors: mean + per-mouse
            p.append(f'      <circle id="{"sm" if anc==0 else "dm"}-{cid}" class="cd44-mean" cx="{XS[anc]:.1f}" cy="{yv(s["mean"][anc]):.1f}" r="4.5"/>')
            for mi, mv in enumerate(s[key]):
                r = radius(mv[1]); hollow = ' data-hollow="1"' if mv[1] < 30 else ''
                cx = XS[anc] + (mi * 2 - 1) * 7
                p.append(f'      <circle id="{("s" if anc==0 else "d")}{mi}-{cid}" class="cd44-mouse" cx="{cx:.1f}" cy="{yv(mv[0]):.1f}" r="{r:.1f}"{hollow}/>')
        lab = f'{c["label"]}  Δ{s["delta"]:+.0f}pp'
        p.append(f'      <text id="lb-{cid}" class="cascade-label-name" x="838" y="{lyy[cid]:.1f}" dominant-baseline="middle">{lab}</text>')
        p.append(f'      <title id="ti-{cid}"></title>')
        p.append('    </g>')
    return '\n'.join(p)


CSS = """<style>
  :root { --cd44-neut: #7a3f9c; }
  #fig-cd44-explore .cd44-controls { display:flex; flex-wrap:wrap; align-items:center; gap:8px 12px; margin:6px 0 14px; }
  #fig-cd44-explore .cd44-ctl-label { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-faint); }
  #fig-cd44-explore .cd44-toggle { display:inline-flex; border:1px solid var(--rule); border-radius:3px; overflow:hidden; }
  #fig-cd44-explore .cd44-toggle button { appearance:none; border:0; background:var(--paper); color:var(--ink-soft); font:600 11px/1 ui-monospace,Menlo,monospace; letter-spacing:.02em; padding:6px 11px; cursor:pointer; border-left:1px solid var(--rule); transition:background .12s,color .12s; }
  #fig-cd44-explore .cd44-toggle button:first-child { border-left:0; }
  #fig-cd44-explore .cd44-toggle button[aria-checked="true"] { background:var(--ink); color:var(--paper); }
  #fig-cd44-explore .cd44-toggle button:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
  #fig-cd44-explore .cd44-readout { font-size:12px; color:var(--ink-faint); font-variant-numeric:tabular-nums; }
  #fig-cd44-explore .cd44-readout b { color:var(--ink-soft); }
  #cd44-svg .cd44-line { stroke:var(--cascade-stroke); fill:none; stroke-linecap:round; stroke-linejoin:round; opacity:.9; }
  #cd44-svg .cd44-line.headline { opacity:1; }
  #cd44-svg .cd44-mean { fill:var(--cascade-stroke); stroke:var(--paper); stroke-width:1.5; }
  #cd44-svg .cd44-way  { fill:var(--cascade-stroke); opacity:.55; }
  #cd44-svg .cd44-mouse { fill:var(--cascade-stroke); stroke:var(--paper); stroke-width:1.4; opacity:.85; }
  #cd44-svg .cd44-mouse[data-hollow="1"] { fill:var(--paper); stroke:var(--cascade-stroke); }
  #cd44-svg .cd44-ghost { fill:none; stroke:var(--cascade-stroke); stroke-width:1.4; opacity:.5; }
  #cd44-svg .cascade-label-name { fill:var(--cascade-stroke); font-size:13px; font-weight:600; }
  #cd44-svg:has(.cd44-comp:hover) .cd44-comp:not(:hover) { opacity:.22; transition:opacity .15s; }
  #cd44-svg .cd44-comp { transition:opacity .3s; }
  @media (prefers-reduced-motion: reduce) { #cd44-svg * { transition:none !important; } }
</style>"""


def build_script(data: dict) -> str:
    payload = json.dumps(data, separators=(',', ':'))
    # NB: braces are doubled for .format-free f-string safety — we build with %-join instead.
    return (
        '<script>\n(function(){\n'
        'var CD44 = ' + payload + ';\n'
        'var fig = document.getElementById("fig-cd44-explore");\n'
        'if(!fig) return;\n'
        'var XS=[250,440,630,820];\n'
        'function y(r){return 450-360*r;}\n'
        'function rad(n){return Math.max(2,Math.min(7.5,1.5+2*Math.sqrt(n/300)));}\n'
        'function ease(t){return t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;}\n'
        'var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;\n'
        # build current state from a slice
        'function sliceState(key){var sl=CD44.slices[key],st={};CD44.comps.forEach(function(c){var s=sl[c.id];\n'
        '  if(!s){st[c.id]={on:false};return;}\n'
        '  st[c.id]={on:true,mean:s.mean.slice(),sham:s.sham.map(function(m){return m.slice();}),d7:s.d7.map(function(m){return m.slice();}),delta:s.delta};});\n'
        '  // greedy label de-collision on D7 means\n'
        '  var ord=CD44.comps.filter(function(c){return st[c.id].on;}).sort(function(a,b){return y(st[a.id].mean[3])-y(st[b.id].mean[3]);});\n'
        '  var prev=null;ord.forEach(function(c){var ty=y(st[c.id].mean[3]);if(prev!==null&&ty<prev+16)ty=prev+16;st[c.id].ly=ty;prev=ty;});\n'
        '  return st;}\n'
        'var cur = sliceState(fig.getAttribute("data-scale")+"|"+fig.getAttribute("data-region"));\n'
        'function draw(st){CD44.comps.forEach(function(c){var g=document.getElementById("cd44-"+c.id);if(!g)return;var s=st[c.id];\n'
        '  if(!s.on){g.style.opacity=0;g.style.pointerEvents="none";return;} g.style.opacity="";g.style.pointerEvents="";\n'
        '  var pl=document.getElementById("pl-"+c.id);pl.setAttribute("points",[0,1,2,3].map(function(i){return XS[i]+","+y(s.mean[i]).toFixed(1);}).join(" "));\n'
        '  document.getElementById("w1-"+c.id).setAttribute("cy",y(s.mean[1]).toFixed(1));\n'
        '  document.getElementById("w2-"+c.id).setAttribute("cy",y(s.mean[2]).toFixed(1));\n'
        '  document.getElementById("sm-"+c.id).setAttribute("cy",y(s.mean[0]).toFixed(1));\n'
        '  document.getElementById("dm-"+c.id).setAttribute("cy",y(s.mean[3]).toFixed(1));\n'
        '  [["s",s.sham,0],["d",s.d7,3]].forEach(function(pr){pr[1].forEach(function(mv,mi){var d=document.getElementById(pr[0]+mi+"-"+c.id);if(!d)return;\n'
        '    d.setAttribute("cy",y(mv[0]).toFixed(1));d.setAttribute("r",rad(mv[1]).toFixed(1));if(mv[1]<30)d.setAttribute("data-hollow","1");else d.removeAttribute("data-hollow");});});\n'
        '  var lb=document.getElementById("lb-"+c.id);lb.setAttribute("y",s.ly.toFixed(1));lb.textContent=c.label+"  \\u0394"+(s.delta>=0?"+":"")+Math.round(s.delta)+"pp";\n'
        '  var sc=fig.getAttribute("data-scale"),rg=fig.getAttribute("data-region");\n'
        '  document.getElementById("ti-"+c.id).textContent=c.label+" \\u00b7 "+sc+"\\u00b5m \\u00b7 "+rg+" \\u00b7 Sham "+Math.round(s.mean[0]*100)+"% \\u2192 D7 "+Math.round(s.mean[3]*100)+"% \\u00b7 \\u0394"+(s.delta>=0?"+":"")+s.delta+"pp";\n'
        '});}\n'
        # tween cur -> target
        'var raf=null;\n'
        'function lerp(a,b,t){return a+(b-a)*t;}\n'
        'function tween(target,withGhost){if(raf)cancelAnimationFrame(raf);\n'
        '  var from=JSON.parse(JSON.stringify(cur));\n'
        '  if(withGhost)spawnGhosts(from);\n'
        '  if(reduce){cur=target;draw(cur);pulse();return;}\n'
        '  var t0=null,D=460;\n'
        '  function step(ts){if(t0===null)t0=ts;var t=Math.min(1,(ts-t0)/D),e=ease(t);\n'
        '    CD44.comps.forEach(function(c){var a=from[c.id],b=target[c.id],s=cur[c.id];\n'
        '      if(!b.on){s.on=false;return;} if(!a.on){cur[c.id]=JSON.parse(JSON.stringify(b));return;} s.on=true;\n'
        '      s.mean=b.mean.map(function(v,i){return lerp(a.mean[i],v,e);});\n'
        '      s.sham=b.sham.map(function(m,i){return [lerp(a.sham[i]?a.sham[i][0]:m[0],m[0],e),m[1]];});\n'
        '      s.d7=b.d7.map(function(m,i){return [lerp(a.d7[i]?a.d7[i][0]:m[0],m[0],e),m[1]];});\n'
        '      s.ly=lerp(a.ly!=null?a.ly:b.ly,b.ly,e);s.delta=b.delta;});\n'
        '    draw(cur);\n'
        '    if(t<1){raf=requestAnimationFrame(step);}else{cur=JSON.parse(JSON.stringify(target));draw(cur);raf=null;}}\n'
        '  raf=requestAnimationFrame(step);}\n'
        # ghost anchors on scale switches: faint outgoing D7 means
        'var glayer=null;\n'
        'function spawnGhosts(from){if(!glayer){glayer=document.createElementNS("http://www.w3.org/2000/svg","g");document.getElementById("cd44-svg").appendChild(glayer);}\n'
        '  glayer.innerHTML="";CD44.comps.forEach(function(c){var a=from[c.id];if(!a.on)return;var el=document.createElementNS("http://www.w3.org/2000/svg","circle");\n'
        '    el.setAttribute("class","cd44-ghost");el.setAttribute("cx",820);el.setAttribute("cy",y(a.mean[3]).toFixed(1));el.setAttribute("r",6);el.style.setProperty("--cascade-stroke",c.color);glayer.appendChild(el);});\n'
        '  setTimeout(function(){if(glayer){glayer.style.transition="opacity .3s";glayer.style.opacity=0;setTimeout(function(){if(glayer){glayer.innerHTML="";glayer.style.opacity="";glayer.style.transition="";}},320);}},500);}\n'
        'function pulse(){/* reduced-motion: brief endpoint emphasis handled by draw */}\n'
        # controls
        'function setActive(group,attr,val){group.querySelectorAll("button").forEach(function(b){b.setAttribute("aria-checked",b.getAttribute(attr)===val?"true":"false");});}\n'
        'function apply(scale,region,scaleChanged){fig.setAttribute("data-scale",scale);fig.setAttribute("data-region",region);\n'
        '  var target=sliceState(scale+"|"+region);tween(target,scaleChanged);\n'
        '  var key=scale+"|"+region,sl=CD44.slices[key];var ns=[];Object.keys(sl).forEach(function(k){sl[k].sham.forEach(function(m){ns.push(m[1]);});sl[k].d7.forEach(function(m){ns.push(m[1]);});});\n'
        '  var lo=Math.min.apply(null,ns),hi=Math.max.apply(null,ns);\n'
        '  fig.querySelector(".cd44-readout").innerHTML="<b>"+scale+"\\u00b5m</b> \\u00b7 <b>"+region+"</b> \\u00b7 support n\\u2248"+lo+"\\u2013"+hi+" px/mouse"+(scale!=="10"?" \\u00b7 neutrophil has no grid-scale basis":"");}\n'
        'fig.addEventListener("click",function(e){var t=e.target.closest("[data-set-scale],[data-set-region]");if(!t)return;\n'
        '  var scale=fig.getAttribute("data-scale"),region=fig.getAttribute("data-region"),scaleChanged=false;\n'
        '  if(t.hasAttribute("data-set-scale")){scale=t.getAttribute("data-set-scale");setActive(t.parentNode,"data-set-scale",scale);scaleChanged=true;}\n'
        '  else{region=t.getAttribute("data-set-region");setActive(t.parentNode,"data-set-region",region);}\n'
        '  apply(scale,region,scaleChanged);});\n'
        'fig.addEventListener("keydown",function(e){if(/^(INPUT|TEXTAREA)$/.test(e.target.tagName))return;var k=e.key;\n'
        '  var scale=fig.getAttribute("data-scale"),region=fig.getAttribute("data-region");\n'
        '  if(k==="["||k==="]"){var i=["10","20","40"].indexOf(scale);i=Math.max(0,Math.min(2,i+(k==="]"?1:-1)));scale=["10","20","40"][i];setActive(fig.querySelector("[aria-label=\\"Superpixel scale\\"]"),"data-set-scale",scale);apply(scale,region,true);e.preventDefault();}\n'
        '  else if(k==="ArrowLeft"||k==="ArrowRight"){var r=["cortex","medulla","pooled"],j=r.indexOf(region);j=(j+(k==="ArrowRight"?1:2))%3;region=r[j];setActive(fig.querySelector("[aria-label=\\"Kidney region\\"]"),"data-set-region",region);apply(scale,region,false);e.preventDefault();}});\n'
        'draw(cur);\n'
        '})();\n</script>'
    )


def main():
    data = build_data()
    summ = pd.read_csv(SUMMARY); summ = summ[summ.region == 'pooled'].set_index('compartment')
    # added-compartments table (10µm pooled Sham→D7)
    trows = []
    for cid, lab in [('CD206', 'CD206⁺ (M2-like)'), ('CD34', 'CD34⁺ (vascular)'),
                     ('endothelial_cd31cd34', 'Endothelial (CD31⁺∧CD34⁺)'), ('CD140b', 'CD140b⁺ (pericyte)')]:
        r = summ.loc[cid]
        trows.append(f'    <tr><td>{lab}</td><td class="num">{r["sham_10um"]*100:.1f}%</td>'
                     f'<td class="num">{r["d7_10um"]*100:.1f}%</td><td class="num">{r["delta_10um"]*100:+.1f}pp</td></tr>')
    table = ('<table>\n  <thead>\n    <tr><th>Added compartment (Sham-referenced, 10µm)</th>'
             '<th class="num">Sham CD44⁺</th><th class="num">D7 CD44⁺</th><th class="num">Δ (pp)</th></tr>\n'
             '  </thead>\n  <tbody>\n' + '\n'.join(trows) + '\n  </tbody>\n</table>')

    # static grid + axis + x-ticks
    axis = ['  <g class="plot-bg">']
    for r, lab in [(0.0, '0%'), (0.25, '25%'), (0.5, '50%'), (0.75, '75%'), (1.0, '100%')]:
        axis.append(f'    <line class="grid" x1="250" y1="{yv(r):.1f}" x2="820" y2="{yv(r):.1f}"/>')
        axis.append(f'    <text class="ytick" x="236" y="{yv(r):.1f}" text-anchor="end" dominant-baseline="middle">{lab}</text>')
    axis.append('    <text class="axis-label" transform="translate(56,270) rotate(-90)" text-anchor="middle">CD44⁺ rate (% of compartment)</text>')
    for i, tp in enumerate(TPS):
        cls = 'xtick-main' if tp in ('Sham', 'D7') else 'xtick-sub'
        axis.append(f'    <text class="{cls}" x="{XS[i]:.0f}" y="472" text-anchor="middle">{tp}</text>')
    axis.append('    <text class="hover-hint" x="535" y="66" text-anchor="middle">switch Scale or Region — the lines morph in place · hover a compartment for its numbers</text>')
    axis.append('  </g>')

    marks = prerender_default(data)
    script = build_script(data)

    section = f"""<p>That table is a snapshot — one superpixel scale, both regions pooled. The figure below makes the compartment activation <strong>explorable</strong> instead: it adds the reviewer-requested <strong>CD206⁺ (M2-like)</strong> and <strong>CD34⁺ / endothelial (CD31∧CD34)</strong> compartments, and lets you switch the two experiment dimensions directly — pick a <em>scale</em> (10 / 20 / 40µm) and a <em>region</em> (cortex / medulla / pooled) and the compartment lines morph in place. Descriptive; reuses the frozen Sham-referenced compute, adds no pre-registered endpoint. (CD140b/PDGFRβ marks pericytes/mural — not fibroblasts; CD206 marks M2-<em>like</em> macrophages, not exclusively.)</p>

{CSS}
<figure class="cascade-fig" id="fig-cd44-explore" data-scale="10" data-region="pooled">
  <div class="cascade-wrap">
    <p class="cascade-eyebrow">The compartment activation, explorable — switch scale and region</p>
    <h3 class="cascade-title">CD44⁺ activation rises Sham→D7 in every compartment — and it <em>holds at every scale</em> (flip Scale: the bundle barely moves) while it <em>separates by region</em> (flip to Medulla: the lines fall to the baseline). Two findings in one figure: scale-robust, and cortex-predominant.</h3>
    <div class="cd44-controls">
      <span class="cd44-ctl-label">Scale</span>
      <div class="cd44-toggle" role="radiogroup" aria-label="Superpixel scale">
        <button role="radio" data-set-scale="10" aria-checked="true">10µm</button>
        <button role="radio" data-set-scale="20" aria-checked="false">20µm</button>
        <button role="radio" data-set-scale="40" aria-checked="false">40µm</button>
      </div>
      <span class="cd44-ctl-label">Region</span>
      <div class="cd44-toggle" role="radiogroup" aria-label="Kidney region">
        <button role="radio" data-set-region="cortex" aria-checked="false">Cortex</button>
        <button role="radio" data-set-region="medulla" aria-checked="false">Medulla</button>
        <button role="radio" data-set-region="pooled" aria-checked="true">Pooled</button>
      </div>
      <span class="cd44-readout"><b>10µm</b> · <b>pooled</b> · support n≈337–4361 px/mouse</span>
    </div>
    <svg class="cascade-svg" id="cd44-svg" viewBox="0 0 1100 540" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Interactive CD44 activation slopegraph; switch scale and region.">
{chr(10).join(axis)}
{marks}
    </svg>
    <figcaption>One Sham→D1→D3→D7 slopegraph on a fixed 0–100% axis; the Scale and Region controls select the slice and the seven compartment lines morph only vertically (identity — colour, dash, label, x-anchors — held fixed). Framing anchors (Sham, D7) show both per-mouse points (n=2, no error bars); dot area ∝ √n_support, so coarser scales visibly shrink (hollow = n&lt;30). Neutrophil appears only at 10µm (no cell-type basis at grid scales). Δpp on each end-label + hover.
    <span class="source">Source: results/biological_analysis/collab_cd44/cd44_compartment_rates_allscales.csv (compartment × region × scale × timepoint × mouse). Descriptive; n=2 mice, paired within-mouse — no region test.</span></figcaption>
  </div>
</figure>
{script}

<p>The reading, in two moves. <strong>Flip Scale (10→20→40µm):</strong> the bundle holds its shape — the Sham→D7 rise survives coarsening the patch from ~one cell to a tissue domain, so it is not a 10µm artifact (the pipeline's kNN-graph already models neighbourhoods at 10µm). <strong>Flip Region (pooled→cortex→medulla):</strong> the lines fan apart and several invert — the activation is <strong>cortex-predominant</strong>: CD206⁺ rises +47pp in cortex vs −4pp in medulla, the neutrophil compartment reaches near-total activation in cortex but only partially in medulla, and CD140b⁺ rises in cortex while the medulla declines. And these two are one: the cortical predominance itself persists across all three scales. Consistent with obstructive injury concentrating in the cortex. Caveat: n=2 mice, the region split is paired within-mouse — descriptive, hypothesis-generating, not a tested effect.</p>
"""
    OUT.write_text(section, encoding='utf-8')
    print(f"wrote {OUT} ({len(section)} chars); slices={len(data['slices'])}; "
          f"comps={len(data['comps'])}")


if __name__ == '__main__':
    main()
