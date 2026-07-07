"""
Lineage Neighborhood Lens (region x timepoint, mouse-of-mouse, descriptive)
===========================================================================

Emits ONE product,
``results/biological_analysis/spatial_neighborhoods/lineage_neighborhood_by_region_timepoint.csv``,
the region x timepoint neighborhood-enrichment LENS at the DEFENSIBLE lineage grain
(DESIGN Decision 0 RANK-2: immune / endothelial / stromal). This is the well-supported
Fig-4 lens data: N3's stratified product built only the fine-15 support-audit surface
(10/119 region x timepoint x celltype strata rest on a single mouse); the lens needs the
coarser lineage grain, which the DESIGN verified is 0 single-mouse / min 63 / median 213
focal cells per region x timepoint x lineage stratum.

REUSE-FIRST (no new primitive where one exists):
  - per-ROI kNN + composition + expected primitives reused verbatim from
    spatial_neighborhood_analysis.py (compute_knn_neighborhoods :43,
    compute_neighborhood_composition :57, compute_expected_composition :93,
    load_roi_annotations :34); the per-ROI relabel-and-enrich loop mirrors
    analyze_roi_neighborhoods (:165) minus the permutation null.
  - mouse-of-mouse aggregation reused VERBATIM via aggregate_strata (:323) with
    strata=['region','timepoint'] -- same support ledger, emit-not-drop flags,
    n_mice_effective, observed per-mouse spread. NO significance is computed
    (aggregate_strata never reads p_value), so the lens is significance-free by
    construction: no permutation p, no p_value/q_value/fraction_significant, no
    binarized >0.5/>1.5 headline. Uncertainty is the OBSERVED per-mouse spread only.

The cell_type -> lineage map is CONFIG-DRIVEN from config.json
``cell_type_annotation.cell_types[*].family`` matched by ``family.startswith(lineage)``
over ``membership_axes.lineages`` keys {immune, endothelial, stromal}, with
'unassigned' -> 'unassigned'. This mirrors the canonical get_lineage() family-prefix
logic at build_indra_evidence_table.py:791-821 (replicated here as ~6 lines rather than
imported, to avoid that module's INDRA / module-level loads). In practice the
confidence>0.0 filter (mirrored from spatial_neighborhood_analysis.py:192) drops every
'unassigned' superpixel (unassigned confidence == 0.0), so the focal AND neighbor axes
resolve to the 3 real lineages.

SCOPE (DESIGN Decision 0 + hypergraph invariants). This node writes ONLY the lineage
region x timepoint lens. It does NOT emit a fine-15 region x timepoint per-stratum
trajectory: the fine grain STAYS POOLED with the verbatim "unresolvable finer" note
(printed at run time), never relabeled under a lineage bucket to manufacture support.
The RANK-1 CONTINUOUS Family B lineage-kNN figure lens
(compute_knn_neighbor_lineage_scores, temporal_interface_analysis.py:508) is a downstream
node's scope and is deliberately NOT built here.

Touches NO existing/frozen code (reuse-only imports); writes ONLY to
``spatial_neighborhoods/`` (a NON-pinned path -- only ``spatial_neighborhoods_composite/``
is SHA-locked in review_packet/FROZEN_PREREG.md). ``verify_frozen_prereg.py`` stays PASS
and the existing temporal/regional/stratified CSVs are byte-identical.

Run:  .venv/bin/python run_lineage_neighborhood_lens.py   (~30s, 24 ROIs, no permutation)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from spatial_neighborhood_analysis import (
    load_roi_annotations,
    compute_knn_neighborhoods,
    compute_neighborhood_composition,
    compute_expected_composition,
    aggregate_strata,
    _PATHS,
)

# Verbatim DESIGN Decision 0 honesty note -- travels with the product so the pooled
# fine-15 caveat is never lost. Kept BYTE-FOR-BYTE from DESIGN.md.
UNRESOLVABLE_FINER_NOTE = (
    "Finer than lineage is unresolvable at region×timepoint: 10 of 119 "
    "region×timepoint×cell-type strata rest on a single mouse and 27 on ≤10 cells "
    "(the sparse activated_* subtypes). These are shown pooled, never as a per-stratum "
    "trajectory."
)


def build_lineage_map(config_path: str = 'config.json') -> Dict[str, str]:
    """Config-driven cell_type -> lineage map (immune / endothelial / stromal).

    Mirrors the canonical get_lineage() family-prefix logic
    (build_indra_evidence_table.py:791-821): read each cell type's ``family`` from
    config.json ``cell_type_annotation.cell_types`` and assign the lineage whose name
    is a prefix of that family, over the ``membership_axes.lineages`` keys. 'unassigned'
    maps to 'unassigned' (and is filtered out upstream by confidence>0.0 anyway).
    """
    with open(config_path) as fh:
        cfg = json.load(fh)
    cta = cfg['cell_type_annotation']
    lineage_names = list(cta['membership_axes']['lineages'].keys())  # immune/endothelial/stromal
    cell_types = cta['cell_types']

    mapping: Dict[str, str] = {}
    for name, defn in cell_types.items():
        family = defn.get('family', '')
        lineage = 'unassigned'
        for ln in lineage_names:
            if family.startswith(ln):
                lineage = ln
                break
        mapping[name] = lineage
    mapping['unassigned'] = 'unassigned'
    return mapping


def compute_lineage_roi_results(
    lineage_map: Dict[str, str],
    k_neighbors: int = 10,
) -> List[pd.DataFrame]:
    """Per-ROI kNN neighborhood enrichment on the 3-lineage relabeling.

    Mirrors analyze_roi_neighborhoods (spatial_neighborhood_analysis.py:165) exactly on
    filtering (confidence>0.0, :192), the len>=k+1 guard (:196), and the log2_enrichment
    formula (clip(log2(obs/exp), -10, 10), :246) -- but relabels cell_type -> lineage and
    computes NO permutation/p_value (the lens is descriptive; aggregate_strata never reads
    it). Reuses compute_knn_neighborhoods / compute_expected_composition /
    compute_neighborhood_composition verbatim. One row per
    (focal_lineage != 'unassigned') x (neighbor_lineage), carrying the columns
    aggregate_strata consumes: roi_id, focal_cell_type, neighbor_cell_type,
    log2_enrichment, n_focal_cells. Deterministic (no permutation randomness enters).
    """
    annotation_dir = _PATHS.annotations_dir
    annotation_files = sorted(annotation_dir.glob('roi_*_cell_types.parquet'))
    print(f"\n✓ Found {len(annotation_files)} ROI annotations")

    roi_results: List[pd.DataFrame] = []
    print("\nAnalyzing ROI lineage neighborhoods...")
    for i, annotation_file in enumerate(annotation_files, 1):
        roi_id = annotation_file.stem.replace('_cell_types', '')
        print(f"  [{i}/{len(annotation_files)}] {roi_id}...", end='', flush=True)

        annotations = load_roi_annotations(roi_id)
        if annotations is None or len(annotations) == 0:
            print(" (skipped - no annotations)")
            continue

        coords = annotations[['x', 'y']].values
        cell_types = annotations['cell_type'].values

        # Mirror the per-ROI primitive's confidence>0.0 filter (:192) exactly.
        valid_mask = annotations['confidence'] > 0.0
        coords = coords[valid_mask]
        cell_types = cell_types[valid_mask]

        # Mirror the len>=k+1 guard (:196).
        if len(coords) < k_neighbors + 1:
            print(" (skipped - insufficient data)")
            continue

        # Relabel cell_type -> lineage.
        lineages = np.array([lineage_map.get(ct, 'unassigned') for ct in cell_types])

        knn_indices = compute_knn_neighborhoods(coords, k=k_neighbors)
        expected_comp = compute_expected_composition(lineages)

        focal_lineages = [l for l in np.unique(lineages) if l != 'unassigned']
        neighbor_lineages = np.unique(lineages)

        rows = []
        for focal in focal_lineages:
            observed_comp = compute_neighborhood_composition(lineages, knn_indices, focal)
            for neighbor in neighbor_lineages:
                observed_prop = observed_comp.get(neighbor, 0.0)
                expected_prop = expected_comp.get(neighbor, 0.0)
                if expected_prop == 0:
                    continue
                enrichment = observed_prop / expected_prop
                rows.append({
                    'roi_id': roi_id,
                    'focal_cell_type': focal,
                    'neighbor_cell_type': neighbor,
                    'observed_proportion': observed_prop,
                    'expected_proportion': expected_prop,
                    'enrichment_score': enrichment,
                    'log2_enrichment': (np.clip(np.log2(enrichment), -10, 10)
                                        if enrichment > 0 else np.nan),
                    'n_focal_cells': int(np.sum(lineages == focal)),
                })

        if rows:
            roi_results.append(pd.DataFrame(rows))
            print(f" ✓ ({len(rows)} lineage interactions)")
        else:
            print(" (skipped - no lineage focal cells)")

    print(f"\n✓ Analyzed {len(roi_results)} ROIs successfully")
    return roi_results


def main() -> int:
    print('=' * 80)
    print('Lineage Neighborhood Lens  (region × timepoint, mouse-of-mouse, descriptive)')
    print('=' * 80)

    # Step 1 -- config-driven lineage map.
    lineage_map = build_lineage_map('config.json')
    print("\nCell_type → lineage map (config-driven, family-prefix; "
          "mirrors get_lineage build_indra_evidence_table.py:791-821):")
    for ct, ln in lineage_map.items():
        print(f"  {ct:32s} -> {ln}")

    # Step 2 -- per-ROI lineage kNN enrichment (reused primitives, no permutation).
    roi_results = compute_lineage_roi_results(lineage_map, k_neighbors=10)

    # Step 3 -- mouse-of-mouse region × timepoint aggregation (aggregate_strata VERBATIM).
    product = aggregate_strata(roi_results, ['region', 'timepoint'])

    # Step 3b -- pooled-across-region lineage marginal (aggregate_strata VERBATIM at the
    # ['timepoint'] grain), emitted as region='Pooled' rows so the Fig-4 lens Pooled toggle
    # rides REAL pooled-across-region lineage data (mouse-of-mouse), never a fabricated
    # average of the Cortex/Medulla rows. Same columns, same support ledger, same
    # emit-not-drop flags, NO significance (aggregate_strata never reads p_value).
    pooled = aggregate_strata(roi_results, ['timepoint'])
    pooled.insert(0, 'region', 'Pooled')
    pooled = pooled[product.columns]  # align column order exactly
    product = pd.concat([product, pooled], ignore_index=True)

    # Support summary.
    n_strata = product.groupby(['region', 'timepoint']).ngroups
    frac_supported = float((~product['insufficient_support'].astype(bool)).mean())
    print('\n' + '=' * 80)
    print('SUPPORT SUMMARY  (region × timepoint lineage lens)')
    print('=' * 80)
    print(f"  rows                     : {product.shape[0]}")
    print(f"  regions                  : {sorted(product['region'].unique())}")
    print(f"  focal lineages (nunique) : {product['focal_cell_type'].nunique()} "
          f"({sorted(product['focal_cell_type'].unique())})")
    print(f"  region × timepoint strata: {n_strata}")
    print(f"  supported (not insuff.)  : {frac_supported:.0%}")
    print(f"  below_min_support rows   : {int(product['below_min_support'].sum())}")
    print(f"  insufficient_support rows: {int(product['insufficient_support'].sum())}")
    print(f"  min n_focal_cells        : {int(product['n_focal_cells'].min())}")
    print(f"  min n_mice_effective     : {int(product['n_mice_effective'].min())}")

    # Verbatim DESIGN Decision 0 honesty note -- the fine-15 detail STAYS POOLED and is
    # never emitted as a per-stratum trajectory (invariant "Defensible grain only").
    print('\n' + '-' * 80)
    print('DESIGN Decision 0 (fine-15 stays pooled — verbatim):')
    print(f"  {UNRESOLVABLE_FINER_NOTE}")
    print('-' * 80)

    # Assert region × timepoint MOSTLY supported (defensible grain delivers). Never fake
    # a pass: on failure exit non-zero.
    if not (frac_supported > 0.5):
        print(f"\nGATE_FAIL: region × timepoint lineage lens mostly insufficient "
              f"({frac_supported:.2f} supported) — grain not defensible.")
        return 1

    # Confirm the significance-free invariant on the emitted surface.
    banned = [c for c in product.columns
              if any(k in c.lower() for k in ('p_value', 'q_value', 'fraction_significant'))]
    if banned:
        print(f"\nGATE_FAIL: significance columns present at the lens: {banned}")
        return 1

    # Write ONLY the lineage lens; the reference/composite CSVs are never touched.
    out_dir = _PATHS.spatial_neighborhoods_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'lineage_neighborhood_by_region_timepoint.csv'
    product.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({product.shape[0]} rows × {product.shape[1]} cols)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
