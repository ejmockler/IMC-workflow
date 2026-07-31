#!/usr/bin/env python
"""kNN neighborhood-size (k) sensitivity sweep for composite focalization.

Pre-registered in analysis_plans/k_sensitivity_precommitment.md (committed 3873c01,
BEFORE this script produced any output). k=10 remains primary; this is a disclosure
sweep, not a re-selection.

Reuses run_composite_focalization_scale_region.py VERBATIM (imports its labelling and
enrichment functions, and aggregate_strata) and varies ONLY the module-level K.
Labels are computed once per scale and reused across k, since category assignment does
not depend on k -- only the kNN graph does.

Basis note: statuses here are computed on the scale x region product at 10um Pooled.
The published composite_focalization.csv used a different (temporal composite
neighborhood) aggregation basis, so absolute values differ slightly; what this sweep
tests is whether the focal/diffuse STATUS is stable under k on a fixed basis.

Emits results/biological_analysis/k_sensitivity_focalization.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/Users/noot/Documents/IMC"); sys.path.insert(0, str(REPO))
import run_composite_focalization_scale_region as M
from src.config import Config
from classify_composite_focalization import _read_threshold

KS = [5, 10, 20]
CUTOFF = _read_threshold()
BIO = REPO / "results" / "biological_analysis"


def main():
    config = Config(str(REPO / "config.json"))
    print(f"[precommit] cutoff = {CUTOFF}x ; k in {KS} ; primary remains k=10", flush=True)

    # label once per scale (k-independent), then sweep k over the kNN step only
    labels_by_scale = {}
    for sc in M.SCALES:
        print(f"[label] scale {sc} ...", flush=True)
        labels_by_scale[sc] = (M.roi_labels_10um() if sc == 10.0
                               else M.roi_labels_scale(sc, config, M.load_reference(sc)))

    frames = []
    for k in KS:
        M.K = k  # the only thing that varies
        for sc in M.SCALES:
            rr = M.roi_self_enrichment(labels_by_scale[sc])
            prod = M.aggregate_strata(rr, ["region", "timepoint"])
            pooled = M.aggregate_strata(rr, ["timepoint"])
            pooled.insert(0, "region", "Pooled")
            pooled = pooled[prod.columns]
            both = pd.concat([prod, pooled], ignore_index=True)
            both.insert(0, "scale_um", sc)
            both.insert(0, "k", k)
            frames.append(both)
            print(f"[k={k}] scale {sc}: {both.shape[0]} rows", flush=True)

    allp = pd.concat(frames, ignore_index=True)
    self_rows = allp[allp.focal_cell_type == allp.neighbor_cell_type].copy()
    self_rows = self_rows.rename(columns={"focal_cell_type": "category"}).drop(columns=["neighbor_cell_type"])
    keep = ["k", "scale_um", "region", "timepoint", "category", "enrichment_score",
            "log2_enrichment", "n_focal_cells", "n_rois", "n_mice_effective",
            "below_min_support", "insufficient_support", "mouse_values"]
    out = self_rows[[c for c in keep if c in self_rows.columns]]
    outp = BIO / "k_sensitivity_focalization.csv"
    out.to_csv(outp, index=False)
    print(f"\n[done] {outp}  ({len(out)} rows)")

    # ---- the pre-registered endpoint: D7 status at 10um Pooled, per k ----
    print("\n" + "=" * 78)
    print("PRE-REGISTERED ENDPOINT — D7 self-enrichment, 10um Pooled, cutoff "
          f"{CUTOFF}x (>= cutoff => focal)")
    print("=" * 78)
    d7 = out[(out.scale_um == 10.0) & (out.region == "Pooled") & (out.timepoint == "D7")]
    table = {}
    for cat in M.INTERFACES:
        row = {}
        for k in KS:
            r = d7[(d7.k == k) & (d7.category == cat)]
            v = float(r.enrichment_score.iloc[0]) if len(r) and pd.notna(r.enrichment_score.iloc[0]) else np.nan
            row[k] = v
        table[cat] = row
    print(f"{'category':<32} " + "".join(f"k={k:<14}" for k in KS) + "status stable?")
    flips = []
    for cat, row in table.items():
        statuses = []
        cells = ""
        for k in KS:
            v = row[k]
            st = ("focal" if v >= CUTOFF else "diffuse") if v == v else "undef"
            statuses.append(st)
            cells += f"{v:.3f} ({st[:4]}) ".ljust(16) if v == v else "undef".ljust(16)
        stable = len(set(statuses)) == 1
        if not stable:
            flips.append((cat, dict(zip(KS, statuses))))
        print(f"{cat:<32} {cells}{'YES' if stable else '*** NO — FLIPS ***'}")

    print("\n" + "-" * 78)
    if not flips:
        print("OUTCOME 1 (pre-registered): all four interface statuses STABLE across k in "
              f"{KS}. The focalization result is k-robust; report as a disclosed envelope.")
    else:
        print("OUTCOME 2/3 (pre-registered): STATUS FLIP DETECTED — the headline is k-dependent "
              "and must be rewritten per the pre-commitment:")
        for cat, st in flips:
            print(f"   - {cat}: {st}")
    print("-" * 78)


if __name__ == "__main__":
    main()
