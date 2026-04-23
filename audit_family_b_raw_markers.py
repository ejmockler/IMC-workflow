"""
Family B parallel raw-marker audit (Phase 1.5 follow-up).

Family B's primary neighbor-minus-self delta is computed on the sigmoid
Sham-referenced continuous lineage scores (``lineage_immune`` etc. in
the annotation parquets). Those scores inherit Family A's Sham-reference
sigmoid, so a Family B endpoint agreeing with Family A at the continuous
path is not independent evidence.

This script computes a parallel Family B using **raw arcsinh marker
expression** as the lineage quantity, with no sigmoid intermediary:

  lineage_immune_raw       = CD45
  lineage_endothelial_raw  = mean(CD31, CD34)
  lineage_stromal_raw      = CD140a

The neighbor-minus-self operation is differential (neighbor mean minus
self), so any Sham-reference additive offset cancels: the raw-marker
path is genuinely sigmoid-independent. Output written to
``results/biological_analysis/temporal_interfaces/family_b_raw_marker_audit.parquet``
and a stability summary at
``family_b_raw_marker_comparison.csv``.

Does not alter persistent pipeline artifacts. The primary Family B
continues to use the sigmoid Sham-ref path.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.analysis import temporal_interface_analysis as tia
from src.utils.paths import get_paths


REPO_ROOT = Path(__file__).resolve().parent
SCALE_UM = 10.0

# Raw-marker analogs of the sigmoid lineage columns. These columns are
# already present on the merged annotation DataFrame returned by
# ``load_annotations_with_markers`` because the canonical superpixel
# dataframe carries the arcsinh markers, which we stack here into
# lineage-level composites.
RAW_LINEAGE_DEFINITIONS = {
    'lineage_immune': ['CD45'],
    'lineage_endothelial': ['CD31', 'CD34'],
    'lineage_stromal': ['CD140a'],
}


def _build_raw_lineage_columns(annotations: pd.DataFrame) -> pd.DataFrame:
    """Replace the sigmoid lineage_* columns with raw arcsinh composites.

    Creates a copy so the primary sigmoid scoring remains intact.
    Endpoint tables emit the same schema — only the values differ.
    """
    out = annotations.copy()
    for col, markers in RAW_LINEAGE_DEFINITIONS.items():
        missing = [m for m in markers if m not in out.columns]
        if missing:
            raise ValueError(
                f'audit_family_b_raw_markers: raw markers {missing} missing '
                f'from annotations; cannot build {col}'
            )
        vals = out[markers].values
        out[col] = vals.mean(axis=1) if vals.shape[1] > 1 else vals[:, 0]
    return out


def _run_family_b_on(
    annotations: pd.DataFrame, min_support: int,
) -> pd.DataFrame:
    """Compute Family B endpoints on whatever ``lineage_*`` columns are in
    the dataframe. Parallels run_temporal_interface_analysis._run_family_b_at_support
    but returns only the endpoint table (no parquet writes)."""
    per_roi_frames: List[pd.DataFrame] = []
    for roi_id, group in annotations.groupby('roi_id'):
        per_roi = tia.compute_neighbor_minus_self_per_roi(
            group, roi_id=roi_id, k=tia.DEFAULT_K_NEIGHBORS,
            min_support=min_support,
        )
        if not per_roi.empty:
            per_roi_frames.append(per_roi)
    if not per_roi_frames:
        return pd.DataFrame()
    per_roi_all = pd.concat(per_roi_frames, ignore_index=True)
    sufficient = per_roi_all[~per_roi_all['below_min_support']]
    mouse = (
        tia.aggregate_neighbor_delta_to_mouse(sufficient)
        if not sufficient.empty else pd.DataFrame()
    )
    filtered, _ = tia.apply_trajectory_filter(mouse, per_roi_raw=per_roi_all)
    if filtered.empty:
        return pd.DataFrame()
    delta_vs_sham = tia.compute_delta_vs_sham(filtered)
    delta_cols = [
        c for c in delta_vs_sham.columns if c.startswith('vs_sham_mean_delta_')
    ]
    endpoints = tia.pairwise_endpoint_table(
        delta_vs_sham, value_cols=delta_cols,
        family='B_continuous_neighborhood',
        extra_index_cols=['composite_label'],
    )
    return endpoints


def main() -> int:
    print('=' * 80)
    print('Family B parallel raw-marker audit')
    print('=' * 80)

    # Import lazily — load_annotations_with_markers is defined in the
    # orchestrator script at repo root.
    sys.path.insert(0, str(REPO_ROOT))
    from run_temporal_interface_analysis import load_annotations_with_markers  # noqa: E402

    annotations = load_annotations_with_markers(scale_um=SCALE_UM)
    print(
        f'\n  Loaded {len(annotations):,} superpixels across '
        f'{annotations["roi_id"].nunique()} ROIs'
    )

    # --- Primary (sigmoid Sham-ref continuous lineage scores) ---
    print('\n  Running Family B on primary (sigmoid Sham-ref continuous) ...')
    primary = _run_family_b_on(annotations, min_support=tia.DEFAULT_MIN_SUPPORT)
    primary['lineage_source'] = 'sham_reference_v2_continuous'
    print(f'    {len(primary)} endpoints')

    # --- Raw-marker parallel ---
    print('\n  Running Family B on raw markers (sigmoid-independent) ...')
    raw_annotations = _build_raw_lineage_columns(annotations)
    raw = _run_family_b_on(raw_annotations, min_support=tia.DEFAULT_MIN_SUPPORT)
    raw['lineage_source'] = 'raw_marker_arcsinh'
    print(f'    {len(raw)} endpoints')

    # --- Compare ---
    merge_keys = ['endpoint', 'contrast', 'composite_label']
    cmp = primary[merge_keys + ['hedges_g']].merge(
        raw[merge_keys + ['hedges_g']],
        on=merge_keys, suffixes=('_sigmoid', '_raw'),
        how='outer', indicator=True,
    )
    both = cmp[cmp['_merge'] == 'both'].copy()
    only_sigmoid = cmp[cmp['_merge'] == 'left_only']
    only_raw = cmp[cmp['_merge'] == 'right_only']

    both_finite = (
        both['hedges_g_sigmoid'].notna() & both['hedges_g_raw'].notna()
    )
    sign_reverse = (
        np.sign(both['hedges_g_sigmoid']) != np.sign(both['hedges_g_raw'])
    ) & both_finite
    abs_max = np.maximum(
        both['hedges_g_sigmoid'].abs(), both['hedges_g_raw'].abs(),
    )
    rel_diff = (both['hedges_g_sigmoid'] - both['hedges_g_raw']).abs() / abs_max.where(abs_max > 0, 1)
    magnitude_disagree = ((abs_max > 0.5) & (rel_diff > 0.5)) & both_finite

    both['family_b_lineage_source_sign_reverse'] = sign_reverse.values
    both['family_b_lineage_source_magnitude_disagree'] = magnitude_disagree.values

    paths = get_paths()
    out_dir = paths.biological_analysis_dir / 'temporal_interfaces'
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([primary, raw], ignore_index=True)
    combined.to_parquet(out_dir / 'family_b_raw_marker_audit.parquet')

    both.drop(columns=['_merge']).to_csv(
        out_dir / 'family_b_raw_marker_comparison.csv', index=False,
    )

    print(f'\n✓ Wrote {out_dir}/family_b_raw_marker_audit.parquet '
          f'({len(combined)} rows, primary + raw)')
    print(f'✓ Wrote {out_dir}/family_b_raw_marker_comparison.csv '
          f'({len(both)} comparable endpoints)')

    n_sr = int(sign_reverse.sum())
    n_md = int(magnitude_disagree.sum())
    n_finite = int(both_finite.sum())
    nontrivial = both_finite & (
        (both['hedges_g_sigmoid'].abs() > 0.5) | (both['hedges_g_raw'].abs() > 0.5)
    )
    n_sr_nt = int((sign_reverse & nontrivial).sum())
    n_md_nt = int((magnitude_disagree & nontrivial).sum())
    n_nt = int(nontrivial.sum())

    print('\n  Stability between sigmoid and raw-marker Family B paths:')
    print(f'    comparable endpoints: {n_finite}')
    print(f'    sign reversals (all |g|): {n_sr}')
    print(f'    sign reversals at |g|>0.5: {n_sr_nt}/{n_nt}')
    print(f'    magnitude disagreements (≥2× symmetric, |g|>0.5): {n_md_nt}/{n_nt}')

    if len(only_sigmoid):
        print(f'\n  Endpoints unique to sigmoid path (trajectory-filter artifact): '
              f'{len(only_sigmoid)}')
    if len(only_raw):
        print(f'  Endpoints unique to raw-marker path: {len(only_raw)}')

    # Top raw-marker headlines at Sham→D7 to see if the "no Family B
    # headlines" null result replicates on the sigmoid-independent path.
    d7 = raw[(raw['contrast'] == 'Sham_vs_D7')].copy()
    d7 = d7.reindex(d7['hedges_g'].abs().sort_values(ascending=False).index)
    nontrivial_d7 = d7[d7['hedges_g'].abs() > 0.5]
    print(f'\n  Raw-marker Family B Sham→D7 headlines at |g|>0.5: {len(nontrivial_d7)}')
    for _, r in nontrivial_d7.head(5).iterrows():
        print(
            f'    {r["endpoint"]:<40s} label={r["composite_label"]:<30s} '
            f'g={r["hedges_g"]:+.2f}'
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
