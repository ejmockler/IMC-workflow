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


def load_lineage_definitions_from_config(config: 'Config') -> Dict[str, Dict[str, object]]:
    """Pull raw-marker lineage definitions from config so the panel-portability
    promise of Phase 5.2 is concrete: a different cohort with a different
    panel updates `config.cell_type_annotation.membership_axes.lineages` and
    the raw-marker basis re-runs unchanged.

    Returns a dict {lineage_name: {'markers': [...], 'aggregation': 'max'|'mean'}}.
    Raises if the config is missing the section. Reads from `config.raw`
    because `Config` doesn't expose `cell_type_annotation` as a typed
    attribute.
    """
    annotation_cfg = config.raw.get('cell_type_annotation', {}) if hasattr(config, 'raw') else {}
    membership = annotation_cfg.get('membership_axes', {})
    lineages = membership.get('lineages')
    if not lineages:
        raise ValueError(
            'config.cell_type_annotation.membership_axes.lineages is missing; '
            'Phase 5.2 raw-marker basis cannot be built without it.'
        )
    out: Dict[str, Dict[str, object]] = {}
    for name, spec in lineages.items():
        if name.startswith('_'):  # skip _comment keys
            continue
        markers = spec.get('markers')
        agg = spec.get('aggregation', 'mean')
        if not markers:
            raise ValueError(
                f'lineage {name!r} in config has no markers; cannot build raw composite'
            )
        if agg not in ('max', 'mean'):
            raise ValueError(
                f'lineage {name!r} aggregation={agg!r} not supported '
                '(must be max or mean)'
            )
        out[f'lineage_{name}'] = {'markers': list(markers), 'aggregation': agg}
    return out


def build_raw_lineage_columns(
    annotations: pd.DataFrame,
    lineage_definitions: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    """Replace sigmoid lineage_* columns with raw arcsinh composites built
    from config-defined markers + aggregation per lineage.

    Creates a copy so the primary sigmoid scoring remains intact. Endpoint
    tables emit the same schema — only the values differ.
    """
    out = annotations.copy()
    for col, spec in lineage_definitions.items():
        markers = spec['markers']
        agg = spec['aggregation']
        missing = [m for m in markers if m not in out.columns]
        if missing:
            raise ValueError(
                f'raw markers {missing} missing from annotations; '
                f'cannot build {col}'
            )
        vals = out[markers].values
        if agg == 'max':
            out[col] = vals.max(axis=1)
        else:  # mean
            out[col] = vals.mean(axis=1)
    return out


def run_family_b_on_basis(
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


HEADLINE_GATE_G = 0.5


def compute_basis_divergence(
    sigmoid_endpoints: pd.DataFrame,
    raw_endpoints: pd.DataFrame,
) -> pd.DataFrame:
    """Phase 5.2 amendment: classify every (endpoint × composite_label ×
    contrast) by headline-overlap status across the two lineage-source bases.

    `headline_overlap_status` ∈ {both_above, sigmoid_only, raw_only, both_below}.
    Headlines clear |g_shrunk_neutral| > 0.5 AND not g_pathological AND
    not support_sensitive.
    """
    keys = ['endpoint', 'composite_label', 'contrast']
    cols_to_pull = [
        'hedges_g', 'g_shrunk_neutral', 'g_pathological', 'support_sensitive',
    ]
    sig = sigmoid_endpoints[keys + [c for c in cols_to_pull if c in sigmoid_endpoints.columns]].copy()
    raw = raw_endpoints[keys + [c for c in cols_to_pull if c in raw_endpoints.columns]].copy()
    merged = sig.merge(
        raw, on=keys, suffixes=('_sigmoid', '_raw'), how='outer', indicator=True,
    )

    def _is_headline_on(suffix: str):
        def _check(row: pd.Series) -> bool:
            g = row.get(f'g_shrunk_neutral{suffix}')
            if pd.isna(g):
                return False
            if abs(g) <= HEADLINE_GATE_G:
                return False
            if bool(row.get(f'g_pathological{suffix}', False)):
                return False
            # support_sensitive may not be present on the raw-marker basis
            # (it is only computed in the sigmoid path's sweep); treat
            # missing as False.
            if bool(row.get(f'support_sensitive{suffix}', False)):
                return False
            return True
        return _check

    merged['sigmoid_above'] = merged.apply(_is_headline_on('_sigmoid'), axis=1)
    merged['raw_above'] = merged.apply(_is_headline_on('_raw'), axis=1)

    def _status(row: pd.Series) -> str:
        sa, ra = bool(row['sigmoid_above']), bool(row['raw_above'])
        if sa and ra:
            return 'both_above'
        if sa:
            return 'sigmoid_only'
        if ra:
            return 'raw_only'
        return 'both_below'

    merged['headline_overlap_status'] = merged.apply(_status, axis=1)

    # Sign-agreement on the shrunk g values where both bases finite
    def _sign_agree(row: pd.Series) -> bool:
        gs = row.get('g_shrunk_neutral_sigmoid')
        gr = row.get('g_shrunk_neutral_raw')
        if pd.isna(gs) or pd.isna(gr) or gs == 0 or gr == 0:
            return True  # vacuously; conflict only meaningful at non-zero magnitude
        return (gs > 0) == (gr > 0)

    merged['sign_agree'] = merged.apply(_sign_agree, axis=1)
    return merged.drop(columns=['_merge'])


def compute_basis_conflict(divergence: pd.DataFrame) -> pd.DataFrame:
    """Phase 5.2 amendment: opposite-sign-same-endpoint subset. If both
    bases clear |g_neut|>0.5 but signs disagree, neither basis enters the
    headline set; the row is moved to this conflict table for inspection."""
    if divergence.empty:
        return divergence
    mask = (
        divergence['headline_overlap_status'].eq('both_above')
        & ~divergence['sign_agree']
    )
    return divergence[mask].copy()


# Backward-compat aliases for older callers
_build_raw_lineage_columns = build_raw_lineage_columns  # noqa: E305
_run_family_b_on = run_family_b_on_basis  # noqa: E305
RAW_LINEAGE_DEFINITIONS = {
    'lineage_immune': ['CD45'],
    'lineage_endothelial': ['CD31', 'CD34'],
    'lineage_stromal': ['CD140a'],
}


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

    # Phase 5.2 amendment: divergence + conflict tables
    divergence = compute_basis_divergence(primary, raw)
    divergence.to_csv(out_dir / 'family_b_basis_divergence.csv', index=False)
    conflict = compute_basis_conflict(divergence)
    conflict.to_csv(out_dir / 'family_b_basis_conflict.csv', index=False)

    print(f'\n✓ Wrote {out_dir}/family_b_raw_marker_audit.parquet '
          f'({len(combined)} rows, primary + raw)')
    print(f'✓ Wrote {out_dir}/family_b_raw_marker_comparison.csv '
          f'({len(both)} comparable endpoints)')
    print(f'✓ Wrote {out_dir}/family_b_basis_divergence.csv '
          f'({len(divergence)} rows, headline-overlap status per endpoint)')
    print(f'✓ Wrote {out_dir}/family_b_basis_conflict.csv '
          f'({len(conflict)} opposite-sign-same-endpoint rows)')

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

    # Sham→D7 headline counts on BOTH paths, raw |g| and shrunk neutral g.
    # The earlier version printed only the raw-marker count; the sigmoid
    # count was assumed (incorrectly) to be zero. Always print both so the
    # claim-vs-data delta is visible at audit time.
    sham_d7_raw = raw[raw['contrast'] == 'Sham_vs_D7']
    sham_d7_sig = primary[primary['contrast'] == 'Sham_vs_D7']
    print('\n  Sham→D7 headline counts on both bases (cross-check):')
    for label, df in [('sigmoid (primary)', sham_d7_sig),
                      ('raw-marker', sham_d7_raw)]:
        n_raw = int((df['hedges_g'].abs() > 0.5).sum())
        n_neut = int((df['g_shrunk_neutral'].abs() > 0.5).sum())
        n_path = int(df['g_pathological'].sum()) if 'g_pathological' in df.columns else 0
        print(f'    {label:<22s}  |g|>0.5: {n_raw:>2d}   '
              f'|g_neut|>0.5: {n_neut:>2d}   pathological: {n_path}')

    # Top raw-marker rows by raw |g| (kept from earlier output for spot-check).
    d7 = raw[raw['contrast'] == 'Sham_vs_D7'].copy()
    d7 = d7.reindex(d7['hedges_g'].abs().sort_values(ascending=False).index)
    nontrivial_d7 = d7[d7['hedges_g'].abs() > 0.5]
    print(f'\n  Top raw-marker Sham→D7 rows by |hedges_g|:')
    for _, r in nontrivial_d7.head(5).iterrows():
        print(
            f'    {r["endpoint"]:<40s} label={r["composite_label"]:<30s} '
            f'g={r["hedges_g"]:+.2f}  g_neut={r["g_shrunk_neutral"]:+.2f}'
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
