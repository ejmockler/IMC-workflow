#!/usr/bin/env python
"""Sensitivity of the continuous lineage scores to the marker-role assignment.

The three continuous axes are configured as immune=[CD45], endothelial=[CD31,CD34] (mean),
stromal=[CD140a]. A coherence review found no recorded rationale for (a) using a single
marker per lineage when the panel carries more, (b) assigning CD140b (PDGFRb, a canonical
pericyte/mural marker) to the activation overlay rather than the stromal lineage, or (c)
averaging CD31/CD34 rather than taking their maximum. The implementation accepts arbitrary
marker lists and max/mean/min, so all three are configuration conventions.

This script asks whether those conventions change any conclusion. It recomputes the three
lineage scores under four marker-assignment variants, using the SAME frozen Sham-reference
sigmoid parameters and the SAME persisted arcsinh-transformed arrays, and reports the
per-timepoint trajectories. Nothing in the pipeline is modified; the primary configuration
(variant A) is unchanged.

Emits results/biological_analysis/marker_assignment_sensitivity.csv
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/Users/noot/Documents/IMC"); sys.path.insert(0, str(REPO))
from batch_annotate_all_rois import deserialize_numpy_arrays
from src.utils.canonical_loader import load_roi_results
from src.utils.metadata import parse_roi_metadata

SCALE = "10.0"
STEEPNESS = 10.0
BIO = REPO / "results" / "biological_analysis"

# variant -> {axis: (markers, aggregation)}
VARIANTS = {
    "A_current":   {"immune": (["CD45"], "max"),
                    "endothelial": (["CD31", "CD34"], "mean"),
                    "stromal": (["CD140a"], "max")},
    "B_multimarker": {"immune": (["CD45", "CD11b"], "max"),
                      "endothelial": (["CD31", "CD34"], "max"),
                      "stromal": (["CD140a", "CD140b"], "max")},
    "C_cd140b_stromal": {"immune": (["CD45"], "max"),
                         "endothelial": (["CD31", "CD34"], "mean"),
                         "stromal": (["CD140a", "CD140b"], "max")},
    "D_endothelial_max": {"immune": (["CD45"], "max"),
                          "endothelial": (["CD31", "CD34"], "max"),
                          "stromal": (["CD140a"], "max")},
}
TIMEPOINTS = ["Sham", "D1", "D3", "D7"]


def main() -> int:
    ref = json.load(open(BIO / f"sham_reference_{SCALE}um.json"))["reference"]

    def sigmoid(x, marker):
        r = ref[marker]
        return 1.0 / (1.0 + np.exp(-STEEPNESS * (np.asarray(x, float) - r["threshold"]) / r["scale"]))

    rows = []
    for f in sorted((REPO / "results" / "roi_results").glob("roi_*_results.json.gz")):
        roi = f.stem.replace("_results.json", "").replace("roi_", "", 1)
        meta = parse_roi_metadata(roi)
        res = deserialize_numpy_arrays(load_roi_results(f))
        arrays = (res.get("multiscale_results", {}).get(SCALE) or {}).get("transformed_arrays") or {}
        if not arrays:
            continue
        needed = {m for v in VARIANTS.values() for mk, _ in v.values() for m in mk}
        scores = {m: sigmoid(arrays[m], m) for m in needed if m in arrays}
        if len(scores) < len(needed):
            continue
        for vname, axes in VARIANTS.items():
            row = {"roi_id": roi, "timepoint": meta["timepoint"],
                   "mouse": meta["mouse"], "region": meta.get("region"), "variant": vname}
            for axis, (markers, agg) in axes.items():
                M = np.column_stack([scores[m] for m in markers])
                v = M.mean(axis=1) if agg == "mean" else M.max(axis=1)
                row[f"{axis}_mean"] = float(v.mean())
            rows.append(row)

    df = pd.DataFrame(rows)
    outp = BIO / "marker_assignment_sensitivity.csv"
    df.to_csv(outp, index=False)
    print(f"[done] {outp}  ({len(df)} rows)")

    print("\n" + "=" * 78)
    print("Mean lineage score by timepoint, per marker-assignment variant")
    print("(A_current is the shipped configuration; the rest are counterfactuals)")
    print("=" * 78)
    verdicts = []
    for axis in ("immune", "endothelial", "stromal"):
        p = df.pivot_table(index="variant", columns="timepoint", values=f"{axis}_mean")[TIMEPOINTS]
        p["Sham_to_D7"] = p["D7"] - p["Sham"]
        print(f"\n--- {axis} ---")
        print(p.round(3).to_string())
        signs = set(np.sign(p["Sham_to_D7"].values))
        mono = all((p.loc[v, TIMEPOINTS].diff().dropna() >= 0).all() for v in p.index)
        verdicts.append((axis, len(signs) == 1, mono))

    print("\n" + "-" * 78)
    for axis, same_sign, mono in verdicts:
        print(f"  {axis:12} direction identical across variants: {same_sign} | monotone rise in every variant: {mono}")
    print("-" * 78)
    print("Reading: if direction is identical across all variants, the reported trajectory")
    print("does not depend on the marker-role assignment. Absolute levels DO shift (taking a")
    print("max over more markers raises every score, including the Sham baseline), so levels")
    print("are only comparable within a variant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
