#!/usr/bin/env python
"""Per-scale composite composition for Fig 6 (sub/pure/iface/triple across 10/20/40um).

Fig 6 partitions every superpixel by its active-lineage count at threshold 0.5
(verified to reproduce the embedded 10um FIG6_DATA exactly):
  0 lineages -> 'sub' (sub-threshold), 1 -> 'pure', 2 -> 'iface', 3 -> 'triple'.
This is a purely COMPOSITE partition (continuous immune/endothelial/stromal scores),
so it extends to 20/40um. 10um uses the frozen annotation parquets (exact match);
20/40um uses annotate_roi_from_results -> membership lineage scores at that scale.
Aggregation matches Fig 6: per-ROI fractions; mice = mean over the mouse's ROIs;
tps = mean over the timepoint's 6 ROIs (simple ROI means, NOT mouse-of-mouse).

Emits results/biological_analysis/fig6_scale_composition.json:
  {"10": {"rois": {..}, "mice": {..}, "tps": {..}}, "20": {..}, "40": {..}}
Each row carries pct_sub/pct_pure/pct_iface/pct_triple (+ n, tp, mouse for rois).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/noot/Documents/IMC"); sys.path.insert(0, str(REPO))
from src.config import Config
from src.utils.canonical_loader import load_roi_results
from src.utils.metadata import parse_roi_metadata
from src.analysis.cell_type_annotation import annotate_roi_from_results
from batch_annotate_all_rois import deserialize_numpy_arrays

BIO = REPO / "results" / "biological_analysis"
THR = 0.5
SCALES = [10.0, 20.0, 40.0]

def partition(im, en, st):
    a = (np.asarray(im) > THR).astype(int) + (np.asarray(en) > THR).astype(int) + (np.asarray(st) > THR).astype(int)
    n = len(a)
    return {"pct_sub": 100*(a == 0).sum()/n, "pct_pure": 100*(a == 1).sum()/n,
            "pct_iface": 100*(a == 2).sum()/n, "pct_triple": 100*(a == 3).sum()/n, "n": int(n)}

def load_ref(sc):
    ref = json.load(open(BIO / f"sham_reference_{sc}um.json")).get("reference", {})
    return {m: {"threshold": float(v["threshold"]), "scale": float(v["scale"])}
            for m, v in ref.items() if isinstance(v, dict) and "threshold" in v}

def roi_rows_10um():
    out = {}
    for pf in sorted((BIO / "cell_type_annotations").glob("roi_*_cell_types.parquet")):
        roi = pf.stem.replace("_cell_types", "").replace("roi_", "", 1)
        df = pd.read_parquet(pf)
        out[roi] = partition(df.lineage_immune, df.lineage_endothelial, df.lineage_stromal)
    return out

def roi_rows_scale(sc, config, ref):
    out = {}
    for rf in sorted((REPO / "results" / "roi_results").glob("roi_*_results.json.gz")):
        roi = rf.stem.replace("_results.json", "").replace("roi_", "", 1)
        res = deserialize_numpy_arrays(load_roi_results(rf))
        if str(sc) not in res.get("multiscale_results", {}):
            continue
        ann = annotate_roi_from_results(res, config, scale=str(sc), reference_distribution=ref)
        lin = (ann.get("memberships") or {}).get("lineage_scores", {})
        if not lin:
            continue
        out[roi] = partition(lin["immune"], lin["endothelial"], lin["stromal"])
    return out

def key_of(roi):
    m = parse_roi_metadata(roi)
    return m["timepoint"], m["mouse"]

def aggregate(rois):
    # attach tp/mouse; mice = mean over mouse's ROIs; tps = mean over 6 ROIs
    for roi, r in rois.items():
        tp, mouse = key_of(roi); r["tp"] = tp; r["mouse"] = mouse
    cols = ["pct_sub", "pct_pure", "pct_iface", "pct_triple"]
    df = pd.DataFrame(rois).T
    df = df[df.tp != "Test"]
    mice = {}
    for (tp, mouse), g in df.groupby(["tp", "mouse"]):
        mice[f"{tp}/{mouse}"] = {**{c: float(g[c].astype(float).mean()) for c in cols},
                                 "n_rois": int(len(g)), "tp": tp, "mouse": mouse}
    tps = {}
    for tp, g in df.groupby("tp"):
        tps[tp] = {**{c: float(g[c].astype(float).mean()) for c in cols}, "n_rois": int(len(g))}
    return mice, tps

def main():
    config = Config(str(REPO / "config.json"))
    out = {}
    for sc in SCALES:
        rois = roi_rows_10um() if sc == 10.0 else roi_rows_scale(sc, config, load_ref(sc))
        mice, tps = aggregate({k: dict(v) for k, v in rois.items()})
        out[str(int(sc))] = {"rois": rois, "mice": mice, "tps": tps}
        print(f"[{int(sc)}µm] {len(rois)} rois; Sham triple {tps['Sham']['pct_triple']:.1f}% -> D7 {tps['D7']['pct_triple']:.1f}%", flush=True)
    outp = BIO / "fig6_scale_composition.json"
    json.dump(out, open(outp, "w"))
    print(f"[done] {outp}")

if __name__ == "__main__":
    main()
