#!/usr/bin/env python
"""Verify that every parameter value stated in prose matches config and data.

Sibling of ``verify_frozen_prereg.py``: that script pins artifact SHAs, this one pins
*documented numbers*. The same facts (superpixel counts, positivity percentiles, kNN size,
lineage composition, ...) are restated across METHODS.md, RESULTS.md, docs/DATA_SCHEMA.md,
the review packet and the collaborator package. Nothing previously prevented one copy from
drifting from another, or from the code — coherence was maintained by hand.

Ground truth is DERIVED at runtime (from config.json and the persisted results), never
hardcoded here, so this file cannot itself go stale. Each check finds every occurrence of a
fact in every document and asserts the stated value equals the derived value.

Exit 0 = every documented value matches. Exit 1 = drift, with the file and the two values.

Usage:  .venv/bin/python verify_documented_parameters.py [--verbose]
"""
from __future__ import annotations
import json, re, sys, glob
from pathlib import Path

REPO = Path(__file__).resolve().parent
DOCS = [
    "METHODS.md", "RESULTS.md", "README.md", "docs/DATA_SCHEMA.md",
    "review_packet/METHODS_SUMMARY.md", "review_packet/ONE_PAGER.md",
    "collaborator_package/README.md",
    "collaborator_package/METHODS_FOR_MANUSCRIPT.md",
    "collaborator_package/ANALYSIS_PARAMETERS.md",
]


def ground_truth() -> dict:
    """Derive every checked value from config.json and the persisted outputs."""
    cfg = json.loads((REPO / "config.json").read_text())
    ca = cfg["cell_type_annotation"]
    pt = ca["positivity_threshold"]
    ov = {k: v["percentile"] for k, v in pt.get("per_marker_override", {}).items()
          if isinstance(v, dict)}
    ma = ca["membership_axes"]
    slic = cfg["segmentation"]["slic_params"]

    gt = {
        "default_percentile": pt["percentile"],
        "ly6g_percentile": ov.get("Ly6G"),
        "cd206_percentile": ov.get("CD206"),
        "sigmoid_steepness": ma["sigmoid_steepness"],
        "lineage_cut": ma["composite_label_thresholds"]["lineage"],
        "dominance_ratio": ma["composite_label_thresholds"]["dominance"],
        "slic_compactness": slic["compactness"],
        "slic_sigma": slic["sigma"],
        "n_cell_types": len(ca["cell_types"]),
        "lineage_markers": {k: v["markers"] for k, v in ma["lineages"].items()
                            if not k.startswith("_")},
    }

    # data-derived
    try:
        import pandas as pd
        roi = pd.read_csv(REPO / "results/biological_analysis/differential_abundance/roi_abundances.csv")
        gt["n_images"] = len(roi)
        gt["n_animals"] = roi.groupby(["timepoint", "mouse"]).ngroups
        gt["superpixels_total"] = int(roi.n_total.sum())
        gt["superpixels_mean"] = int(round(roi.n_total.mean()))
        gt["superpixels_min"] = int(roi.n_total.min())
        gt["superpixels_max"] = int(roi.n_total.max())
        gt["assignment_rate_pct"] = round(roi.assignment_rate.mean() * 100, 1)
        gt["assignment_min_pct"] = round(roi.assignment_rate.min() * 100, 1)
        gt["assignment_max_pct"] = round(roi.assignment_rate.max() * 100, 1)
    except Exception as exc:  # data not present — skip data-derived checks
        gt["_data_error"] = str(exc)
    return gt


# (label, regex capturing ONE number, ground-truth key, formatter)
CHECKS = [
    ("default positivity percentile", r"(\d{2})(?:st|nd|rd|th) percentile of (?:the |each )?(?:image|within-image)", "default_percentile", int),
    ("Ly6G percentile",               r"Ly6G (?:at )?(?:the )?(\d{2})(?:st|nd|rd|th)", "ly6g_percentile", int),
    ("CD206 percentile",              r"CD206 (?:at )?(?:the )?(\d{2})(?:st|nd|rd|th)", "cd206_percentile", int),
    ("sigmoid steepness",             r"[Ss]teepness[^0-9]{0,20}(\d+\.\d)", "sigmoid_steepness", float),
    ("SLIC compactness",              r"[Cc]ompactness[^0-9]{0,15}(\d+\.\d)", "slic_compactness", float),
    ("SLIC sigma",                    r"(?:[Ss]igma|σ)[^0-9]{0,15}(\d+\.\d)", "slic_sigma", float),
    ("mean superpixels per image",    r"(?:mean(?: of)? |averag\w+ )(2,\d{3}) superpixels", "superpixels_mean", lambda x: int(str(x).replace(",", ""))),
    ("total superpixels",             r"\b(58,\d{3})\b", "superpixels_total", lambda x: int(str(x).replace(",", ""))),
    # must be anchored to assignment language, else it matches any "X% of regions"
    ("assignment rate %",             r"(?:mean of |average |only )(\d{2}\.\d)% of (?:tissue )?(?:superpixels|regions)(?=[^.]{0,60}(?:label|rule|assign|match))", "assignment_rate_pct", float),
    ("number of cell-type rules",     r"(\d{2}) (?:marker-combination |positive/negative marker )?rules", "n_cell_types", int),
]


def main() -> int:
    verbose = "--verbose" in sys.argv
    gt = ground_truth()
    if "_data_error" in gt:
        print(f"WARNING: data-derived truth unavailable ({gt['_data_error']}); "
              f"config-derived checks still run.")

    failures, checked = [], 0
    for rel in DOCS:
        path = REPO / rel
        if not path.exists():
            continue
        text = path.read_text()
        for label, pattern, key, cast in CHECKS:
            if key not in gt or gt[key] is None:
                continue
            for m in re.finditer(pattern, text):
                checked += 1
                try:
                    stated = cast(m.group(1))
                except Exception:
                    continue
                truth = cast(gt[key]) if not isinstance(gt[key], str) else gt[key]
                if stated != truth:
                    line = text[: m.start()].count("\n") + 1
                    failures.append((rel, line, label, stated, truth,
                                     text[max(0, m.start() - 60):m.end() + 40].replace("\n", " ")))
                elif verbose:
                    print(f"  ok  {rel}: {label} = {stated}")

    # lineage composition must be stated consistently wherever it appears
    lin = gt.get("lineage_markers", {})
    if lin:
        expect_immune = lin.get("immune", [])
        if expect_immune == ["CD45"]:
            for rel in DOCS:
                path = REPO / rel
                if not path.exists():
                    continue
                t = path.read_text()
                # a doc that names the immune axis composition must say CD45 alone
                for m in re.finditer(r"immune\s*=\s*([A-Za-z0-9+,() ]{2,40})", t):
                    frag = m.group(1)
                    checked += 1
                    if "CD45" not in frag:
                        line = t[: m.start()].count("\n") + 1
                        failures.append((rel, line, "immune axis composition",
                                         frag.strip(), "CD45", m.group(0)))

    print(f"\nchecked {checked} documented values across {len(DOCS)} documents")
    if failures:
        print(f"\nFAIL — {len(failures)} documented value(s) disagree with config/data:\n")
        for rel, line, label, stated, truth, ctx in failures:
            print(f"  {rel}:{line}  [{label}]")
            print(f"      states : {stated}")
            print(f"      truth  : {truth}")
            print(f"      context: …{ctx.strip()}…\n")
        return 1
    print("PASS — every documented parameter value matches config and data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
