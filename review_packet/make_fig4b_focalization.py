#!/usr/bin/env python
"""Fig 4b — interactive composite-interface focalization (single large plot).

Mirrors the report's fig-cd44-explore idiom: Scale (10/20/40 um) + Region
(cortex/medulla/pooled) segmented toggles morph the four interface self-enrichment
trajectories (Sham->D7) in place; hovering a series reveals its per-mouse points
(n=2) and exact values (progressive disclosure). Shared y-axis with 1x random
baseline + 2x focal threshold + diffuse band. Flip Scale to watch the focal
interfaces dilute toward random as the grain coarsens; the triple stays diffuse.
Driven by results/biological_analysis/composite_focalization_scale_region.csv.
"""
import ast, json
import pandas as pd, numpy as np

CSV = "/Users/noot/Documents/IMC/results/biological_analysis/composite_focalization_scale_region.csv"
df = pd.read_csv(CSV)
TPS = ["Sham", "D1", "D3", "D7"]
CATS = [  # id, category, color
    ("E+S", "endothelial+stromal", "#4ea8b0"),
    ("I+S", "immune+stromal", "#c39421"),
    ("E+I", "endothelial+immune", "#9c3a7d"),
    ("Triple", "endothelial+immune+stromal", "#9a958c"),
]
LABELS = {  # spelled-out end-labels (single letters are cryptic)
    "E+S": "Endothelial + Stromal",
    "I+S": "Immune + Stromal",
    "E+I": "Endothelial + Immune",
    "Triple": "Triple (all three)",
}
VMIN, VMAX = 1.0, 4.4

def build_data():
    d = {}
    for sc in (10.0, 20.0, 40.0):
        for reg in ("Cortex", "Medulla", "Pooled"):
            key = f"{int(sc)}|{reg.lower()}"
            d[key] = {}
            for sid, cat, _ in CATS:
                mean, mice = [], []
                for tp in TPS:
                    r = df[(df.scale_um == sc) & (df.region == reg) & (df.category == cat) & (df.timepoint == tp)]
                    if not len(r):
                        mean.append(None); mice.append([]); continue
                    v = r.enrichment_score.iloc[0]
                    mean.append(None if pd.isna(v) else round(float(v), 3))
                    mv = r.mouse_values.iloc[0]
                    try:
                        pts = [round(2 ** float(x), 3) for x in ast.literal_eval(mv)] if isinstance(mv, str) else []
                    except Exception:
                        pts = []
                    mice.append(pts)
                d[key][sid] = {"mean": mean, "mice": mice}
    return d

FK = build_data()

# geometry
W, H = 1010, 430
XS = [96, 304, 512, 720]
YB, YT = 372, 28
ELX = 736
def yv(v): return YB - (min(max(v, VMIN), VMAX) - VMIN) / (VMAX - VMIN) * (YB - YT)

s = []
s.append('<figure class="focalization-fig" id="fig-4b" data-scale="10" data-region="pooled">')
s.append('''  <style>
  #fig-4b .fk-controls{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;margin:.2rem 0 .6rem;font-family:var(--font-sans)}
  #fig-4b .fk-ctl-label{font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-faint,#8b8378);font-weight:700}
  #fig-4b .fk-toggle{display:inline-flex;background:rgba(0,0,0,.04);border-radius:3px;padding:2px}
  #fig-4b .fk-toggle button{appearance:none;background:transparent;border:0;padding:3px 10px;font:600 .72rem var(--font-mono);color:var(--ink-soft);cursor:pointer;border-radius:2px}
  #fig-4b .fk-toggle button[aria-checked="true"]{background:var(--ink);color:var(--paper)}
  #fig-4b .fk-readout{font:.66rem var(--font-mono);color:#8b8378;margin-left:auto}
  #fig-4b .fk-eyebrow{font:.72rem var(--font-sans);color:#8b8378;margin:0 0 .1rem}
  #fig-4b svg{width:100%;height:auto}
  #fig-4b .fk-series{transition:opacity .12s}
  #fig-4b .fk-mouse,#fig-4b .fk-val{opacity:0;transition:opacity .12s;pointer-events:none}
  #fig-4b .fk-series.hi .fk-mouse,#fig-4b .fk-series.hi .fk-val{opacity:1}
  #fig-4b.dimmed .fk-series:not(.hi){opacity:.16}
  #fig-4b .fk-line{fill:none;stroke-linejoin:round}
  #fig-4b .fk-series.hi .fk-line{stroke-width:3.4}
  </style>''')
s.append('  <p class="fk-eyebrow">Interface focalization, explorable — switch scale &amp; region; hover a line for its per-mouse points (n=2).</p>')
# controls
s.append('  <div class="fk-controls">')
s.append('    <span class="fk-ctl-label">Scale</span><div class="fk-toggle" role="radiogroup" aria-label="Superpixel scale">')
for sc in ("10", "20", "40"):
    s.append(f'      <button role="radio" data-set-scale="{sc}" aria-checked="{"true" if sc=="10" else "false"}">{sc}µm</button>')
s.append('    </div>')
s.append('    <span class="fk-ctl-label">Region</span><div class="fk-toggle" role="radiogroup" aria-label="Kidney region">')
for rg in ("cortex", "medulla", "pooled"):
    s.append(f'      <button role="radio" data-set-region="{rg}" aria-checked="{"true" if rg=="pooled" else "false"}">{rg.title()}</button>')
s.append('    </div>')
s.append('    <span class="fk-readout"><b>10µm</b> · <b>pooled</b> · self-enrichment, mouse-of-mouse</span>')
s.append('  </div>')
# svg
s.append(f'  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" role="img" '
         f'aria-label="Interactive interface self-enrichment; switch scale and region; focal interfaces dilute toward 1x as grain coarsens, the triple-positive stays diffuse.">')
# refs + grid
s.append(f'    <rect x="{XS[0]}" y="{yv(2.0):.1f}" width="{ELX-XS[0]}" height="{YB-yv(2.0):.1f}" fill="#000" fill-opacity="0.03"/>')
for vv in (1.0, 2.0, 3.0, 4.0):
    yy = yv(vv); bold = vv == 1.0; thr = vv == 2.0
    dash = ' stroke-dasharray="5 4"' if thr else ''
    s.append(f'    <line x1="{XS[0]}" y1="{yy:.1f}" x2="{ELX}" y2="{yy:.1f}" '
             f'stroke="{"#1a1a1a" if bold else "#c9a24a" if thr else "#e6e1d8"}" '
             f'stroke-width="{1.6 if bold else 1.2 if thr else 1}"{dash}/>')
    lab = "1× random" if bold else ("2× focal" if thr else f"{int(vv)}×")
    s.append(f'    <text x="{XS[0]-10}" y="{yy+4:.1f}" text-anchor="end" font-family="var(--font-mono)" '
             f'font-size="11" fill="{"#1a1a1a" if bold else "#8a7320" if thr else "#b0a99c"}">{lab}</text>')
for i, tp in enumerate(TPS):
    s.append(f'    <text x="{XS[i]}" y="{YB+22}" text-anchor="middle" font-family="var(--font-mono)" font-size="12" fill="#4a4a4a">{tp}</text>')
# series skeletons (rendered at 10|pooled default; JS morphs)
init = FK["10|pooled"]
for sid, cat, col in CATS:
    focal = sid != "Triple"
    dash = '' if focal else ' stroke-dasharray="6 5"'
    m = init[sid]["mean"]; mice = init[sid]["mice"]
    s.append(f'    <g class="fk-series" id="fk-{sid}" data-cat="{sid}" style="cursor:pointer">')
    # path (with gaps at undefined)
    dcmd = ""; pen = False
    for i in range(4):
        if m[i] is None: pen = False; continue
        dcmd += (f"M{XS[i]},{yv(m[i]):.1f}" if not pen else f"L{XS[i]},{yv(m[i]):.1f}"); pen = True
    s.append(f'      <path class="fk-line" id="fk-pl-{sid}" d="{dcmd}" stroke="{col}" stroke-width="{2.4 if focal else 1.8}"{dash}/>')
    # hit target (wide invisible stroke) for easy hover
    s.append(f'      <path class="fk-hit" d="{dcmd}" stroke="transparent" stroke-width="16" fill="none"/>')
    # mean dots + per-mouse dots + value labels
    for i in range(4):
        vis = m[i] is not None
        cy = yv(m[i]) if vis else YB
        r0 = 3.2 if focal else 3.2
        dot = (f'<circle id="fk-m{i}-{sid}" class="fk-mean-dot" cx="{XS[i]}" cy="{cy:.1f}" r="{r0}" '
               + (f'fill="{col}"' if focal else f'fill="#fafaf7" stroke="{col}" stroke-width="1.5"')
               + f' style="opacity:{1 if vis else 0}"/>')
        s.append("      " + dot)
        for j in range(2):
            pv = mice[i][j] if j < len(mice[i]) else None
            pcy = yv(pv) if pv is not None else cy
            s.append(f'      <circle id="fk-p{i}-{j}-{sid}" class="fk-mouse" cx="{XS[i]}" cy="{pcy:.1f}" r="2.4" '
                     f'fill="#fafaf7" stroke="{col}" stroke-width="1.2"/>')
        vtxt = f"{m[i]:.1f}×" if vis else ""
        s.append(f'      <text id="fk-v{i}-{sid}" class="fk-val" x="{XS[i]}" y="{cy-9:.1f}" text-anchor="middle" '
                 f'font-family="var(--font-mono)" font-size="10" font-weight="700" fill="{col}">{vtxt}</text>')
    # end label
    lyv = yv(m[3]) if m[3] is not None else YB
    s.append(f'      <text id="fk-lb-{sid}" x="{ELX+8}" y="{lyv+4:.1f}" text-anchor="start" '
             f'font-family="var(--font-sans)" font-size="12.5" font-weight="{700 if focal else 600}" '
             f'fill="{col if focal else "#6b655c"}">{LABELS[sid]}</text>')
    s.append(f'      <title id="fk-ti-{sid}"></title>')
    s.append('    </g>')
s.append('  </svg>')
s.append('  <figcaption><span class="num">Fig. 4b</span> Interface focalization across scale × region (interactive). '
         '<span class="source">Source: results/biological_analysis/composite_focalization_scale_region.csv</span></figcaption>')
# JS: morph on toggle + hover reveal (mirrors fig-cd44-explore)
s.append('  <script>')
s.append('  (function(){')
s.append('  var FK=' + json.dumps(FK, separators=(',', ':')) + ';')
s.append('  var LBL=' + json.dumps(LABELS, separators=(',', ':')) + ';')
s.append(f'  var TPS={json.dumps(TPS)}, XS={json.dumps(XS)}, CATS={json.dumps([c[0] for c in CATS])};')
s.append(f'  var YB={YB},YT={YT},VMIN={VMIN},VMAX={VMAX};')
s.append('''  var fig=document.getElementById("fig-4b"); if(!fig) return;
  function yv(v){ if(v==null) return YB; v=Math.max(VMIN,Math.min(VMAX,v)); return YB-(v-VMIN)/(VMAX-VMIN)*(YB-YT); }
  var reduce=window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function state(key){ var s=FK[key]||FK["10|pooled"], o={}; CATS.forEach(function(c){o[c]={mean:s[c].mean.slice(),mice:s[c].mice.map(function(a){return a.slice();})};}); return o; }
  var cur=state("10|pooled");
  function pathd(mean){ var d="",pen=false; for(var i=0;i<4;i++){ if(mean[i]==null){pen=false;continue;} d+=(pen?"L":"M")+XS[i]+","+yv(mean[i]).toFixed(1); pen=true; } return d; }
  function draw(st){ var sc=fig.getAttribute("data-scale"),rg=fig.getAttribute("data-region");
    CATS.forEach(function(c){ var s=st[c], m=s.mean;
      document.getElementById("fk-pl-"+c).setAttribute("d",pathd(m));
      var hit=document.querySelector('#fk-'+c+' .fk-hit'); if(hit) hit.setAttribute("d",pathd(m));
      for(var i=0;i<4;i++){ var vis=m[i]!=null, cy=yv(m[i]);
        var md=document.getElementById("fk-m"+i+"-"+c); md.setAttribute("cy",cy.toFixed(1)); md.style.opacity=vis?1:0;
        for(var j=0;j<2;j++){ var pv=s.mice[i][j], pd=document.getElementById("fk-p"+i+"-"+j+"-"+c); if(!pd)continue;
          pd.setAttribute("cy",yv(pv==null?m[i]:pv).toFixed(1)); pd.style.display=(pv==null?"none":""); }
        var vt=document.getElementById("fk-v"+i+"-"+c); vt.setAttribute("y",(cy-9).toFixed(1)); vt.textContent=vis?m[i].toFixed(1)+"×":"";
      }
      document.getElementById("fk-ti-"+c).textContent=(LBL[c]||c)+" · "+sc+"µm · "+rg+" · "+TPS.map(function(t,i){return t+" "+(m[i]==null?"·":m[i].toFixed(2)+"×");}).join("  ");
    });
    // end-label de-collision (greedy, mirrors fig-cd44-explore): sort by y, push apart >=16px
    var labs=[]; CATS.forEach(function(c){ var m=st[c].mean,last=null; for(var k=3;k>=0;k--){if(m[k]!=null){last=m[k];break;}} if(last!=null)labs.push({c:c,y:yv(last)}); });
    labs.sort(function(a,b){return a.y-b.y;});
    for(var i=1;i<labs.length;i++){ if(labs[i].y<labs[i-1].y+16)labs[i].y=labs[i-1].y+16; }
    labs.forEach(function(L){ document.getElementById("fk-lb-"+L.c).setAttribute("y",(L.y+4).toFixed(1)); });
  }
  function ease(t){return t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;}
  var raf=null;
  function tween(target){ if(raf)cancelAnimationFrame(raf); var from=state(fig.getAttribute("data-scale")+"|"+fig.getAttribute("data-region")+"_prev")||JSON.parse(JSON.stringify(cur));
    from=JSON.parse(JSON.stringify(cur));
    if(reduce){cur=target;draw(cur);return;}
    var t0=null,D=440;
    function step(ts){ if(t0==null)t0=ts; var t=Math.min(1,(ts-t0)/D),e=ease(t);
      CATS.forEach(function(c){ var a=from[c],b=target[c];
        cur[c].mean=b.mean.map(function(v,i){ return (v==null||a.mean[i]==null)?v:(a.mean[i]+(v-a.mean[i])*e); });
        cur[c].mice=b.mice.map(function(arr,i){ return arr.map(function(v,j){ var av=(a.mice[i]&&a.mice[i][j]); return (v==null||av==null)?v:(av+(v-av)*e); }); });
      });
      draw(cur);
      if(t<1)raf=requestAnimationFrame(step); else {cur=JSON.parse(JSON.stringify(target));draw(cur);raf=null;}
    }
    raf=requestAnimationFrame(step);
  }
  function setActive(group,attr,val){ group.querySelectorAll("button").forEach(function(b){ b.setAttribute("aria-checked",b.getAttribute(attr)===val?"true":"false"); }); }
  function apply(scale,region){ fig.setAttribute("data-scale",scale); fig.setAttribute("data-region",region);
    tween(state(scale+"|"+region));
    var key=scale+"|"+region, sldef=0, tot=0; CATS.forEach(function(c){ FK[key][c].mean.forEach(function(v){ tot++; if(v!=null)sldef++; }); });
    fig.querySelector(".fk-readout").innerHTML="<b>"+scale+"µm</b> · <b>"+region+"</b> · "+sldef+"/"+tot+" strata defined (n=2)";
  }
  fig.addEventListener("click",function(e){ var t=e.target.closest("[data-set-scale],[data-set-region]"); if(!t)return;
    var scale=fig.getAttribute("data-scale"),region=fig.getAttribute("data-region");
    if(t.hasAttribute("data-set-scale")){scale=t.getAttribute("data-set-scale");setActive(t.parentNode,"data-set-scale",scale);}
    else{region=t.getAttribute("data-set-region");setActive(t.parentNode,"data-set-region",region);}
    apply(scale,region);
  });
  fig.addEventListener("keydown",function(e){ if(/^(INPUT|TEXTAREA)$/.test(e.target.tagName))return; var k=e.key;
    var scale=fig.getAttribute("data-scale"),region=fig.getAttribute("data-region");
    if(k==="["||k==="]"){var i=["10","20","40"].indexOf(scale);i=Math.max(0,Math.min(2,i+(k==="]"?1:-1)));scale=["10","20","40"][i];setActive(fig.querySelector('[aria-label="Superpixel scale"]'),"data-set-scale",scale);apply(scale,region);e.preventDefault();}
    else if(k==="ArrowLeft"||k==="ArrowRight"){var r=["cortex","medulla","pooled"],j=r.indexOf(region);j=(j+(k==="ArrowRight"?1:2))%3;region=r[j];setActive(fig.querySelector('[aria-label="Kidney region"]'),"data-set-region",region);apply(scale,region);e.preventDefault();}
  });
  // hover progressive disclosure
  fig.querySelectorAll(".fk-series").forEach(function(g){
    g.addEventListener("mouseenter",function(){ fig.classList.add("dimmed"); g.classList.add("hi"); });
    g.addEventListener("mouseleave",function(){ fig.classList.remove("dimmed"); g.classList.remove("hi"); });
  });
  draw(cur);
  })();
  </script>''')
s.append('</figure>')
print("\n".join(s))
