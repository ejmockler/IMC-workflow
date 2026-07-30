#!/usr/bin/env python
"""Regenerate Fig 6's 24 ternary RGB tiles at 10/20/40um (for the scale toggle).

Reuses the Fig 1 lineage painter (review_packet/make_fig1_multiscale_rasters.py):
per-superpixel continuous lineage RGB (R=immune, G=stromal, B=endothelial) — the
exact mapping Fig 6's caption states — painted from each scale's superpixel_labels.
All three scales use one painter, so switching grain is a clean grain change, not a
rendering-style jump. Keyed by roi_id (FIG6 tile key). Emits
results/biological_analysis/fig6_scale_tiles.json = {"10":{roi:datauri}, "20":.., "40":..}.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path("/Users/noot/Documents/IMC"); sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("mf1", str(REPO / "review_packet/make_fig1_multiscale_rasters.py"))
mf1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(mf1)
from src.config import Config
from src.utils.canonical_loader import load_roi_results
from src.analysis.cell_type_annotation import annotate_roi_from_results
from batch_annotate_all_rois import deserialize_numpy_arrays

TILE = 176
config = Config(str(REPO / "config.json"))
out = {}
for sc in (10.0, 20.0, 40.0):
    ref = mf1.load_reference(sc)
    d = {}
    for rf in sorted((REPO / "results" / "roi_results").glob("roi_*_results.json.gz")):
        roi = rf.stem.replace("_results.json", "").replace("roi_", "", 1)
        res = deserialize_numpy_arrays(load_roi_results(rf))
        sd = res.get("multiscale_results", {}).get(str(sc))
        if sd is None:
            continue
        seg = mf1.deser(sd["superpixel_labels"]).astype(np.int64)
        ann = annotate_roi_from_results(res, config, scale=str(sc), reference_distribution=ref)
        lin = (ann.get("memberships") or {}).get("lineage_scores", {})
        if not lin:
            continue
        im = mf1.paint_from_labels(seg, mf1.lineage_rgba(lin)).resize((TILE, TILE), Image.NEAREST)
        d[roi] = mf1.encode_ternary(im)
    out[str(int(sc))] = d
    print(f"[{int(sc)}µm] {len(d)} tiles", flush=True)

outp = REPO / "results" / "biological_analysis" / "fig6_scale_tiles.json"
json.dump(out, open(outp, "w"))
tot = sum(len(v) for k in out for v in [out[k].values()] for _ in [0])
sz = sum(len(u) for k in out for u in out[k].values())
print(f"[done] {outp}  ({sz//1024} KB of datauris)")
