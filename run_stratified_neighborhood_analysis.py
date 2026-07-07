"""
Stratified Neighborhood Enrichment Orchestrator (mouse-of-mouse)
================================================================

Emits ONE tidy long product,
``results/biological_analysis/spatial_neighborhoods/stratified_neighborhood_enrichments.csv``,
covering four strata lenses off a SINGLE per-ROI compute:

    strata=[]                      -> pooled              (strata_level='pooled')
    strata=['timepoint']           -> temporal marginal   (strata_level='timepoint')
    strata=['region']              -> regional marginal    (strata_level='region')
    strata=['region','timepoint']  -> region x timepoint   (strata_level='region_x_timepoint')

On every row BOTH ``region`` and ``timepoint`` columns exist; the collapsed axis
carries the sentinel ``'pooled'`` (e.g. a temporal-marginal row has region='pooled').

Aggregation is MOUSE-OF-MOUSE mean-of-logs (``aggregate_strata``; DESIGN Decision 1+4),
NOT the ``n_focal``-weighted ROI mean of ``aggregate_across_rois`` -- retiring the ROI
pseudoreplication defect (one cell-dense ROI can no longer dominate a stratum). Every
stratum x focal x neighbor is EMITTED with a support ledger (``n_focal_cells``, ``n_rois``,
``n_mice``, ``n_mice_effective``, ``n_clipped_extremes``) and emit-not-drop flags
(``below_min_support``, ``insufficient_support``); a sparse region x timepoint row is
FLAGGED with its estimate NaN'd but its counts + per-mouse points (``mouse_values``,
``range_min``/``range_max``) retained. NO significance is surfaced (no ``p_value`` /
``fraction_significant``): the region(xtimepoint) view is descriptive, uncertainty is the
OBSERVED per-mouse spread only -- no bootstrap, no CI.

SCOPE (DESIGN Decision 0 + hypergraph invariants). This is the DISCRETE-15-type
stratified product -- the grain ``aggregate_strata`` natively produces and the only grain
with existing marginals to reproduce. The region x timepoint rows are an explicitly
FLAGGED exploratory support-audit surface (fine-15 flagged/pooled), NOT a headline
fine-type trajectory (10/119 region x timepoint x celltype strata rest on a single mouse;
27 on <=10 cells -- the sparse activated_* subtypes). The RANK-1 CONTINUOUS Family B
lineage-kNN figure lens (``compute_knn_neighbor_lineage_scores``,
``temporal_interface_analysis.py:508``) is owned by a downstream node and is deliberately
NOT built here (no marginal to reproduce; building it here would be scope creep against
the bound invariant "the LINEAGE/CONTINUOUS figure lens remains N6's scope").

The ['timepoint'] / ['region'] marginals REPRODUCE the existing ROI-weighted
temporal/regional CSVs on SIGN + ORDERING (strong-positive Spearman rho on the
both-log2-defined intersection; high sign agreement on the |log2|>0.1 subset). The
numeric MAGNITUDE delta is expected and documented in-script -- it is the intended
consequence of retiring ``n_focal``-weighted ROI averaging in favour of unweighted
mouse-of-mouse averaging, NOT a regression. Determinism (seeded per-ROI RNG) makes the
marginals reproducible.

Writes ``stratified_neighborhood_enrichments.csv`` plus two derived canonical
mouse-of-mouse products sliced from it -- ``temporal_neighborhood_enrichments_mouse.csv``
(strata_level=='timepoint') and ``regional_neighborhood_enrichments_mouse.csv``
(strata_level=='region') -- via ``_PATHS.spatial_neighborhoods_dir``. The OLD ROI-level
temporal/regional reference CSVs and the frozen ``spatial_neighborhoods_composite/`` CSV
are never touched. ``verify_frozen_prereg.py`` stays PASS.

Run:  .venv/bin/python run_stratified_neighborhood_analysis.py   (~5 min, 24 ROIs x 1000 perms)
"""

import sys
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from spatial_neighborhood_analysis import (
    compute_all_roi_results,
    aggregate_strata,
    _PATHS,
)

# --- Strata lenses: (strata list, strata_level label). One per aggregate_strata call. ---
STRATA_LENSES = [
    ([], 'pooled'),
    (['timepoint'], 'timepoint'),
    (['region'], 'region'),
    (['region', 'timepoint'], 'region_x_timepoint'),
]

# Sentinel marking a collapsed (pooled-over) axis so BOTH region and timepoint
# columns exist on every emitted row.
POOLED = 'pooled'

# Tidy long column order for the single emitted CSV.
OUTPUT_COLUMNS = [
    'region', 'timepoint', 'strata_level',
    'focal_cell_type', 'neighbor_cell_type',
    'log2_enrichment', 'enrichment_score',
    'range_min', 'range_max', 'mouse_values',
    'n_focal_cells', 'n_rois', 'n_mice', 'n_mice_effective', 'n_clipped_extremes',
    'below_min_support', 'insufficient_support',
]

# --- Marginal-reproduction thresholds. CALIBRATED to the OBSERVED distribution (24 ROIs
# x 1000 perms, deterministic), documented here so they are auditable and not force-fit.
# Observed on the both-log2-defined intersection:
#   timepoint: pooled rho=0.9702, per-stratum min=0.9353 median=0.9775, sign agree=0.9845
#   region   : pooled rho=0.9697, per-stratum min=0.9601 median=0.9682, sign agree=1.0000
#   |delta log2|: median~0.043-0.045, max~0.40-0.53 (the ROI-weight -> mouse-of-mouse shift)
# Each floor sits comfortably BELOW the observed value (>=~0.06-0.10 headroom) so the gate
# catches a genuine sign/ordering breakdown without failing on the honest magnitude shift.
# The claim is SIGN + ORDERING, never magnitude equality: mouse-of-mouse re-weighting
# shifts magnitudes on purpose. ---
MIN_POOLED_SPEARMAN = 0.90        # pooled rank correlation (observed 0.970)
MIN_PERSTRATUM_MEDIAN_SPEARMAN = 0.90   # median per-stratum-value rank corr (observed 0.968-0.978)
MIN_PERSTRATUM_MIN_SPEARMAN = 0.85      # weakest per-stratum-value rank corr (observed 0.935-0.960)
MIN_SIGN_AGREEMENT = 0.90         # sign agreement on |old log2|>0.1 subset (observed 0.985-1.000)
SIGN_SUBSET_ABS_THRESHOLD = 0.1


def build_stratified_product(roi_results: List[pd.DataFrame]) -> pd.DataFrame:
    """Run aggregate_strata over the four lenses and stack into one tidy long frame.

    For each lens the collapsed axis is filled with the ``'pooled'`` sentinel so both
    ``region`` and ``timepoint`` columns exist on every row, and a ``strata_level``
    label is attached. Columns are reindexed to OUTPUT_COLUMNS.
    """
    parts = []
    for strata, level in STRATA_LENSES:
        frame = aggregate_strata(roi_results, strata)
        # Fill collapsed axes with the sentinel so region + timepoint always present.
        if 'region' not in frame.columns:
            frame['region'] = POOLED
        if 'timepoint' not in frame.columns:
            frame['timepoint'] = POOLED
        frame['strata_level'] = level
        parts.append(frame)

    product = pd.concat(parts, ignore_index=True)
    # Reindex to the tidy column order (all columns are guaranteed present).
    return product[OUTPUT_COLUMNS]


# --- Canonical mouse-of-mouse temporal/regional products. Drop-in mouse-of-mouse
# equivalents of the OLD ROI-level temporal/regional CSVs (old schema: focal/neighbor
# + timepoint|region + enrichment_score/log2_enrichment) PLUS the support ledger flags.
# PURE SLICES of the already-built stratified product -- no recompute, no per-ROI pass. ---
CANONICAL_MOUSE_COLUMNS = [
    'focal_cell_type', 'neighbor_cell_type',
    'enrichment_score', 'log2_enrichment',
    'n_focal_cells', 'n_rois', 'n_mice_effective',
    'below_min_support', 'insufficient_support',
]


def emit_canonical_mouse_products(product: pd.DataFrame, out_dir) -> None:
    """Slice the in-memory stratified ``product`` into the two canonical mouse-of-mouse
    products and write them to ``out_dir``. PURE slice -- no recompute.

    ``temporal_neighborhood_enrichments_mouse.csv`` is the strata_level=='timepoint'
    slice (axis column ``timepoint``); ``regional_neighborhood_enrichments_mouse.csv``
    is the strata_level=='region' slice (axis column ``region``). Both carry the old
    temporal/regional schema (focal/neighbor + axis + enrichment_score + log2_enrichment)
    plus the emit-not-drop support flags (n_focal_cells/n_rois/n_mice_effective/
    below_min_support/insufficient_support). The OLD ROI-level temporal/regional CSVs are
    NEVER touched -- these are additive drop-in mouse-of-mouse equivalents.
    """
    for level, axis, name in (
        ('timepoint', 'timepoint', 'temporal_neighborhood_enrichments_mouse.csv'),
        ('region', 'region', 'regional_neighborhood_enrichments_mouse.csv'),
    ):
        cols = ['focal_cell_type', 'neighbor_cell_type', axis] + CANONICAL_MOUSE_COLUMNS[2:]
        slice_ = product[product['strata_level'] == level][cols].copy()
        out_path = out_dir / name
        slice_.to_csv(out_path, index=False)
        print(f"Wrote {out_path}  ({slice_.shape[0]} rows x {slice_.shape[1]} cols)")


def _reference_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(_PATHS.spatial_neighborhoods_dir / name)


def _compare_marginal(product: pd.DataFrame, level: str, key: str, ref_csv: str) -> dict:
    """Join one marginal of the product to its existing ROI-weighted reference CSV and
    return sign/ordering-reproduction metrics + the documented magnitude delta.

    ``key`` is the stratum axis ('timepoint' or 'region'); the collapsed axis is 'pooled'
    on these product rows. Metrics are computed on the BOTH-log2-defined intersection.
    """
    new = product[product['strata_level'] == level][
        ['focal_cell_type', 'neighbor_cell_type', key, 'log2_enrichment']
    ].copy()
    old = _reference_csv(ref_csv)[
        ['focal_cell_type', 'neighbor_cell_type', key, 'log2_enrichment']
    ].copy()
    merged = new.merge(old, on=['focal_cell_type', 'neighbor_cell_type', key],
                       suffixes=('_new', '_old'))
    both = merged[merged['log2_enrichment_new'].notna()
                  & merged['log2_enrichment_old'].notna()].copy()

    old_l = both['log2_enrichment_old'].to_numpy()
    new_l = both['log2_enrichment_new'].to_numpy()

    pooled_rho, _ = stats.spearmanr(old_l, new_l)

    # Per-stratum-value Spearman (ordering within each timepoint / region).
    per = []
    for val, g in both.groupby(key):
        if len(g) >= 3 and g['log2_enrichment_old'].nunique() > 1 \
                and g['log2_enrichment_new'].nunique() > 1:
            r, _ = stats.spearmanr(g['log2_enrichment_old'], g['log2_enrichment_new'])
            if not np.isnan(r):
                per.append((val, len(g), r))
    per_rhos = [r for _, _, r in per]

    # Sign agreement on the meaningful-magnitude subset.
    sub = both[both['log2_enrichment_old'].abs() > SIGN_SUBSET_ABS_THRESHOLD]
    sign_agree = float((np.sign(sub['log2_enrichment_old'])
                        == np.sign(sub['log2_enrichment_new'])).mean()) if len(sub) else float('nan')

    delta = np.abs(new_l - old_l)

    return {
        'level': level, 'key': key, 'ref_csv': ref_csv,
        'n_new': len(new), 'n_old': len(old),
        'n_both_defined': len(both),
        'pooled_rho': pooled_rho,
        'per_stratum': per,
        'per_rho_min': (min(per_rhos) if per_rhos else float('nan')),
        'per_rho_median': (float(np.median(per_rhos)) if per_rhos else float('nan')),
        'sign_n': len(sub), 'sign_agree': sign_agree,
        'delta_median': float(np.median(delta)) if len(delta) else float('nan'),
        'delta_max': float(np.max(delta)) if len(delta) else float('nan'),
        'delta_mean': float(np.mean(delta)) if len(delta) else float('nan'),
    }


def check_marginal_reproduction(product: pd.DataFrame) -> None:
    """Assert the ['timepoint'] / ['region'] marginals reproduce the existing
    temporal/regional CSVs on SIGN + ORDERING, and document the magnitude delta.

    Raises AssertionError (-> non-zero exit) if any calibrated threshold fails. The
    reproduction claim is deliberately SIGN + ORDERING, never magnitude equality: the
    mouse-of-mouse re-weighting shifts magnitudes on purpose.
    """
    print('\n' + '=' * 80)
    print('MARGINAL-REPRODUCTION CHECK  (mouse-of-mouse marginals vs ROI-weighted CSVs)')
    print('=' * 80)
    print('Claim: SIGN + ORDERING preserved. The magnitude delta is EXPECTED and is the')
    print('intended effect of retiring n_focal-weighted ROI averaging for unweighted')
    print('mouse-of-mouse averaging (N2-verified exemplar: Cortex/D1 log2 = 0.4639')
    print('mouse-of-mouse vs 0.1209 n_focal-weighted; max n_mice = 2).')

    results = [
        _compare_marginal(product, 'timepoint', 'timepoint',
                          'temporal_neighborhood_enrichments.csv'),
        _compare_marginal(product, 'region', 'region',
                          'regional_neighborhood_enrichments.csv'),
    ]

    failures = []
    for m in results:
        print('\n' + '-' * 80)
        print(f"[{m['level']}] marginal vs {m['ref_csv']}")
        print(f"  new rows={m['n_new']}  old rows={m['n_old']}  "
              f"both-log2-defined intersection={m['n_both_defined']}")
        print(f"  POOLED Spearman rho          = {m['pooled_rho']:.4f}   "
              f"(threshold >= {MIN_POOLED_SPEARMAN})")
        print(f"  per-stratum-value Spearman   : "
              + ', '.join(f"{v}(n={n}) rho={r:.4f}" for v, n, r in m['per_stratum']))
        print(f"     per-stratum rho  min={m['per_rho_min']:.4f} "
              f"median={m['per_rho_median']:.4f}   "
              f"(thresholds min >= {MIN_PERSTRATUM_MIN_SPEARMAN}, "
              f"median >= {MIN_PERSTRATUM_MEDIAN_SPEARMAN})")
        print(f"  sign agreement |old log2|>{SIGN_SUBSET_ABS_THRESHOLD} "
              f"(n={m['sign_n']}) = {m['sign_agree']:.4f}   "
              f"(threshold >= {MIN_SIGN_AGREEMENT})")
        print(f"  |delta log2|  median={m['delta_median']:.4f}  "
              f"max={m['delta_max']:.4f}  mean={m['delta_mean']:.4f}   "
              f"<- ROI n_focal-weighting -> unweighted mouse-of-mouse")

        if not (m['pooled_rho'] >= MIN_POOLED_SPEARMAN):
            failures.append(f"{m['level']}: pooled Spearman {m['pooled_rho']:.4f} "
                            f"< {MIN_POOLED_SPEARMAN}")
        if not (m['per_rho_median'] >= MIN_PERSTRATUM_MEDIAN_SPEARMAN):
            failures.append(f"{m['level']}: per-stratum median Spearman "
                            f"{m['per_rho_median']:.4f} < {MIN_PERSTRATUM_MEDIAN_SPEARMAN}")
        if not (m['per_rho_min'] >= MIN_PERSTRATUM_MIN_SPEARMAN):
            failures.append(f"{m['level']}: per-stratum min Spearman "
                            f"{m['per_rho_min']:.4f} < {MIN_PERSTRATUM_MIN_SPEARMAN}")
        if not (m['sign_agree'] >= MIN_SIGN_AGREEMENT):
            failures.append(f"{m['level']}: sign agreement {m['sign_agree']:.4f} "
                            f"< {MIN_SIGN_AGREEMENT}")

    print('\n' + '=' * 80)
    if failures:
        print('MARGINAL-REPRODUCTION CHECK: FAIL')
        for f in failures:
            print(f'  - {f}')
        raise AssertionError('marginal reproduction failed: ' + '; '.join(failures))
    print('MARGINAL-REPRODUCTION CHECK: PASS (sign + ordering reproduced; '
          'magnitude delta documented above)')
    print('=' * 80)


def main(roi_results: Optional[List[pd.DataFrame]] = None) -> None:
    print('=' * 80)
    print('Stratified Neighborhood Enrichment (mouse-of-mouse)')
    print('=' * 80)

    # Step 1 -- single per-ROI compute (reused per-ROI primitive, unchanged).
    if roi_results is None:
        roi_results = compute_all_roi_results(k_neighbors=10, n_permutations=1000)

    # Step 2+3 -- four lenses -> one tidy long product.
    product = build_stratified_product(roi_results)

    rt = product[(product['strata_level'] == 'region_x_timepoint')]
    print(f"\nStratified product: {product.shape}")
    for _, level in STRATA_LENSES:
        n = int((product['strata_level'] == level).sum())
        print(f"  strata_level={level:18s}: {n} rows")
    print(f"  region x timepoint rows: {len(rt)}  "
          f"(below_min_support={int(rt['below_min_support'].sum())}, "
          f"insufficient_support={int(rt['insufficient_support'].sum())}, "
          f"estimate defined={int(rt['log2_enrichment'].notna().sum())})")
    print("  NOTE: region x timepoint is a FLAGGED exploratory support-audit surface "
          "(fine-15 flagged/pooled), not a headline fine-type trajectory.")

    # Step 4 -- marginal reproduction (asserts; non-zero exit on failure).
    check_marginal_reproduction(product)

    # Write ONLY the new stratified product; never the reference/composite CSVs.
    out_dir = _PATHS.spatial_neighborhoods_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'stratified_neighborhood_enrichments.csv'
    product.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({product.shape[0]} rows x {product.shape[1]} cols)")

    # Derived canonical mouse-of-mouse temporal/regional products (pure slices of the
    # product just written; no recompute). The OLD ROI-level CSVs stay untouched.
    emit_canonical_mouse_products(product, out_dir)


if __name__ == '__main__':
    sys.exit(main())
