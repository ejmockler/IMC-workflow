#!/usr/bin/env python
"""Composite-interface self-enrichment across SCALE x REGION x timepoint.

Generalizes run_lineage_neighborhood_lens.py from the 3-lineage grain to the 8
composite interface categories, and from 10um only to 10/20/40um. Mouse-of-mouse,
support-flagged (aggregate_strata VERBATIM). Reuses the frozen kNN primitives.

10um categories come from the annotation parquet lineage_* scores (thresh 0.3);
20/40um from annotate_roi_from_results (memberships.lineage_scores) at that scale.
Category string = '+'.join(sorted active lineages) or 'none' (matches Family A v1).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/Users/noot/Documents/IMC"); sys.path.insert(0, str(REPO))
from spatial_neighborhood_analysis import (
    compute_knn_neighborhoods, compute_neighborhood_composition,
    compute_expected_composition, aggregate_strata, _PATHS,
)
from src.config import Config
from src.utils.canonical_loader import load_roi_results, deserialize_array
from src.analysis.cell_type_annotation import annotate_roi_from_results
from batch_annotate_all_rois import deserialize_numpy_arrays

BIO = REPO / "results" / "biological_analysis"
SCALES = [10.0, 20.0, 40.0]
THR = 0.3
K = 10
LINS = ("immune", "endothelial", "stromal")
INTERFACES = ["endothelial+immune", "immune+stromal", "endothelial+stromal", "endothelial+immune+stromal"]

def category(im, en, st, thr=THR):
    a = []
    if im >= thr: a.append("immune")
    if en >= thr: a.append("endothelial")
    if st >= thr: a.append("stromal")
    return "+".join(sorted(a)) if a else "none"

def load_reference(scale_um):
    ref = json.load(open(BIO / f"sham_reference_{scale_um}um.json")).get("reference", {})
    return {m: {"threshold": float(v["threshold"]), "scale": float(v["scale"])}
            for m, v in ref.items() if isinstance(v, dict) and "threshold" in v}

def roi_labels_10um():
    """(roi_id -> (coords Nx2, categories N)) from the annotation parquets."""
    out = {}
    for pf in sorted(_PATHS.annotations_dir.glob("roi_*_cell_types.parquet")):
        roi_id = pf.stem.replace("_cell_types", "")
        df = pd.read_parquet(pf)
        cats = np.array([category(r.lineage_immune, r.lineage_endothelial, r.lineage_stromal)
                         for r in df.itertuples()], dtype=object)
        out[roi_id] = (df[["x", "y"]].to_numpy(), cats)
    return out

def roi_labels_scale(scale_um, config, ref):
    out = {}
    for rf in sorted((REPO / "results" / "roi_results").glob("roi_*_results.json.gz")):
        roi_id = rf.stem.replace("_results.json", "")
        res = deserialize_numpy_arrays(load_roi_results(rf))
        skey = str(scale_um)
        sd = res.get("multiscale_results", {}).get(skey)
        if sd is None:
            continue
        ann = annotate_roi_from_results(res, config, scale=skey, reference_distribution=ref)
        mem = ann.get("memberships") or {}
        lin = mem.get("lineage_scores", {})
        if not lin:
            continue
        im, en, st = lin["immune"], lin["endothelial"], lin["stromal"]
        cats = np.array([category(im[i], en[i], st[i]) for i in range(len(im))], dtype=object)
        coords = np.asarray(deserialize_array(sd["superpixel_coords"]))
        out[roi_id] = (coords, cats)
    return out

def roi_self_enrichment(labels):
    """Per-ROI kNN enrichment rows (all focal categories x neighbor categories)."""
    results = []
    for roi_id, (coords, cats) in labels.items():
        if len(coords) < K + 1:
            continue
        knn = compute_knn_neighborhoods(coords, k=K)
        exp = compute_expected_composition(cats)
        rows = []
        for focal in np.unique(cats):
            obs = compute_neighborhood_composition(cats, knn, focal)
            for neigh in np.unique(cats):
                op, ep = obs.get(neigh, 0.0), exp.get(neigh, 0.0)
                if ep == 0:
                    continue
                enr = op / ep
                rows.append({
                    "roi_id": roi_id, "focal_cell_type": focal, "neighbor_cell_type": neigh,
                    "observed_proportion": op, "expected_proportion": ep, "enrichment_score": enr,
                    "log2_enrichment": (np.clip(np.log2(enr), -10, 10) if enr > 0 else np.nan),
                    "n_focal_cells": int(np.sum(cats == focal)),
                })
        if rows:
            results.append(pd.DataFrame(rows))
    return results

def main():
    config = Config(str(REPO / "config.json"))
    frames = []
    for sc in SCALES:
        print(f"[scale {sc}] labeling ...", flush=True)
        labels = roi_labels_10um() if sc == 10.0 else roi_labels_scale(sc, config, load_reference(sc))
        rr = roi_self_enrichment(labels)
        prod = aggregate_strata(rr, ["region", "timepoint"])
        pooled = aggregate_strata(rr, ["timepoint"]); pooled.insert(0, "region", "Pooled")
        pooled = pooled[prod.columns]
        both = pd.concat([prod, pooled], ignore_index=True)
        both.insert(0, "scale_um", sc)
        frames.append(both)
        print(f"[scale {sc}] {both.shape[0]} rows", flush=True)
    allp = pd.concat(frames, ignore_index=True)
    # keep SELF rows only (focal == neighbor)
    self_rows = allp[allp.focal_cell_type == allp.neighbor_cell_type].copy()
    self_rows = self_rows.rename(columns={"focal_cell_type": "category"}).drop(columns=["neighbor_cell_type"])
    out = self_rows[["scale_um", "region", "timepoint", "category", "enrichment_score",
                     "log2_enrichment", "n_focal_cells", "n_rois", "n_mice_effective",
                     "below_min_support", "insufficient_support", "mouse_values"]]
    outp = str(BIO / "composite_focalization_scale_region.csv")
    out.to_csv(outp, index=False)
    print(f"\n[done] {outp}  ({len(out)} self rows)")
    # support snapshot for the interface categories
    iv = out[out.category.isin(INTERFACES)]
    print("\nSUPPORT — interface categories, fraction with a DEFINED estimate (>=2 mice, >=min support):")
    for sc in SCALES:
        for reg in ["Cortex", "Medulla", "Pooled"]:
            s = iv[(iv.scale_um == sc) & (iv.region == reg)]
            if len(s) == 0: continue
            defined = int(s.enrichment_score.notna().sum())
            print(f"  {sc:4} {reg:7} : {defined}/{len(s)} defined")

if __name__ == "__main__":
    main()
