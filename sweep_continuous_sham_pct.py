"""
Continuous Sham-percentile sensitivity sweep (Phase 1.5 follow-up).

Companion to the existing raw-marker Sham-reference sweep at
{65, 75, 85} pct. This script tests whether Family A CLR endpoints
are stable when the *continuous* Sham-reference percentile — the value
that centers compute_continuous_memberships' sigmoid — varies.

Approach: reference generation + annotation run in-memory for each
percentile (no parquet round-trip, no cascade re-run), then Family A
endpoints are computed per sweep point. Output written to
``results/biological_analysis/temporal_interfaces/continuous_sham_pct_sweep.csv``.

Does not touch persistent artifacts — the primary sham_reference_10.0um.json
at the config percentile (60) remains the source of truth for the
pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import Config
from src.analysis import temporal_interface_analysis as tia
from src.analysis.cell_type_annotation import compute_continuous_memberships
from src.analysis.sham_reference import build_reference_distribution
from src.constants import LINEAGE_COLS
from src.utils.canonical_loader import build_superpixel_dataframe, load_all_rois
from src.utils.paths import get_paths


REPO_ROOT = Path(__file__).resolve().parent
SCALE_UM = 10.0
SWEEP_PERCENTILES = (50.0, 60.0, 70.0)


def _load_config() -> Config:
    return Config(str(REPO_ROOT / 'config.json'))


def _base_dataframe() -> pd.DataFrame:
    paths = get_paths()
    results = load_all_rois(str(paths.roi_results_dir))
    return build_superpixel_dataframe(results, scale=SCALE_UM)


def _membership_config(config: Config) -> Dict:
    return config.raw['cell_type_annotation']['membership_axes']


def _threshold_config(config: Config) -> Dict:
    return config.raw['cell_type_annotation']['positivity_threshold']


def _protein_channels(config: Config) -> List[str]:
    return config.channels.get('protein_channels', [])


def _score_lineages(
    df: pd.DataFrame,
    protein_channels: List[str],
    membership_cfg: Dict,
    threshold_cfg: Dict,
    reference: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Apply compute_continuous_memberships to each ROI; attach lineage
    scores as columns. Returns the input DataFrame with `lineage_*` columns
    populated for downstream Family A classification.
    """
    out = df.copy()
    # Pre-allocate lineage columns
    for col in LINEAGE_COLS:
        out[col] = np.nan

    for roi, group in df.groupby('roi'):
        expression = group[protein_channels].values
        mem = compute_continuous_memberships(
            expression, protein_channels, membership_cfg, threshold_cfg,
            reference_distribution=reference,
        )
        for ln in ('immune', 'endothelial', 'stromal'):
            col = f'lineage_{ln}'
            out.loc[group.index, col] = mem['lineage_scores'][ln]
    return out


def _family_a_endpoints(annotated: pd.DataFrame) -> pd.DataFrame:
    """Mirror run_family_a's Family A pipeline, without persistence."""
    per_roi = tia.compute_interface_fractions_per_roi(
        annotated.rename(columns={'roi': 'roi_id'}),
        threshold=tia.DEFAULT_LINEAGE_THRESHOLD,
    )
    annotated_roi = tia.attach_roi_metadata(per_roi)
    mouse = tia.aggregate_fractions_to_mouse_level(annotated_roi)
    filtered, _ = tia.apply_min_prevalence_filter(mouse)
    clr_with = tia.compute_interface_clr_table(filtered, exclude_none=False)
    cols = [c for c in clr_with.columns if c.endswith('_clr')]
    endpoints = tia.pairwise_endpoint_table(
        clr_with, value_cols=cols, family='A_interface_clr',
    )
    return endpoints


def main() -> int:
    print('=' * 80)
    print(f'Continuous Sham-pct sensitivity sweep at {SWEEP_PERCENTILES}')
    print('=' * 80)

    config = _load_config()
    membership_cfg = _membership_config(config)
    threshold_cfg = _threshold_config(config)
    protein = _protein_channels(config)
    per_marker_overrides = threshold_cfg.get('per_marker_override', {}) or {}
    # Drop comment keys
    per_marker_overrides = {
        k: v for k, v in per_marker_overrides.items() if not k.startswith('_')
    }

    df = _base_dataframe()
    print(f'\nLoaded {len(df):,} superpixels across {df["roi"].nunique()} ROIs')

    all_endpoints: List[pd.DataFrame] = []
    for pct in SWEEP_PERCENTILES:
        print(f'\n  pct={pct:.1f}: building reference + scoring...')
        reference = build_reference_distribution(
            df, markers=protein, percentile=pct,
            per_marker_overrides=per_marker_overrides,
            aggregation='per_mouse',
        )
        annotated = _score_lineages(df, protein, membership_cfg, threshold_cfg, reference)
        endpoints = _family_a_endpoints(annotated)
        endpoints['continuous_sham_percentile'] = pct
        all_endpoints.append(endpoints)
        n_nontrivial = int((endpoints['hedges_g'].abs() > 0.5).sum())
        print(
            f'    {len(endpoints)} Family A endpoints; '
            f'{n_nontrivial} with |g|>0.5'
        )

    sweep = pd.concat(all_endpoints, ignore_index=True)

    # Stability analysis: for each (endpoint, contrast), compare signs +
    # shrunken magnitudes across the three percentiles.
    stability_rows = []
    for (endpoint, contrast), group in sweep.groupby(['endpoint', 'contrast']):
        gs = group.set_index('continuous_sham_percentile')['hedges_g']
        shrunks = group.set_index('continuous_sham_percentile')['g_shrunk_neutral']
        signs = np.sign(gs.dropna().values)
        finite_signs = signs[np.isfinite(signs) & (signs != 0)]
        mixed = len(np.unique(finite_signs)) > 1 if len(finite_signs) else False
        max_abs_disagree = (
            (gs.max() - gs.min()) / max(abs(gs.max()), abs(gs.min()), 1e-9)
            if gs.notna().all() and max(abs(gs.max()), abs(gs.min())) > 0.5
            else np.nan
        )
        stability_rows.append({
            'endpoint': endpoint,
            'contrast': contrast,
            'g_at_pct50': float(gs.get(50.0, np.nan)),
            'g_at_pct60': float(gs.get(60.0, np.nan)),
            'g_at_pct70': float(gs.get(70.0, np.nan)),
            'g_neut_at_pct50': float(shrunks.get(50.0, np.nan)),
            'g_neut_at_pct60': float(shrunks.get(60.0, np.nan)),
            'g_neut_at_pct70': float(shrunks.get(70.0, np.nan)),
            'continuous_sham_sign_mixed': bool(mixed),
            'continuous_sham_rel_range': float(max_abs_disagree),
        })
    stability = pd.DataFrame(stability_rows)

    paths = get_paths()
    out_dir = paths.biological_analysis_dir / 'temporal_interfaces'
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_parquet(
        out_dir / 'family_a_continuous_sham_pct_sweep.parquet'
    )
    stability.to_csv(
        out_dir / 'continuous_sham_pct_sweep.csv', index=False,
    )

    print(f'\n✓ Wrote {out_dir}/family_a_continuous_sham_pct_sweep.parquet '
          f'({len(sweep)} rows × {len(SWEEP_PERCENTILES)} pcts)')
    print(f'✓ Wrote {out_dir}/continuous_sham_pct_sweep.csv (stability per endpoint)')

    # Summary
    n_total = len(stability)
    n_mixed = int(stability['continuous_sham_sign_mixed'].sum())
    nontrivial = stability[
        stability[['g_at_pct50', 'g_at_pct60', 'g_at_pct70']].abs().max(axis=1) > 0.5
    ]
    n_mixed_nontrivial = int(nontrivial['continuous_sham_sign_mixed'].sum())
    n_nontrivial = len(nontrivial)
    print(f'\n  Stability across continuous Sham pct ∈ {SWEEP_PERCENTILES}:')
    print(f'    {n_mixed}/{n_total} endpoints sign-mix across the sweep (all |g|)')
    print(f'    {n_mixed_nontrivial}/{n_nontrivial} endpoints sign-mix at |g|>0.5')

    # Top-5 Sham_vs_D7 at each pct
    print('\n  Top-5 |g| Sham→D7 at each percentile:')
    for pct in SWEEP_PERCENTILES:
        subset = sweep[
            (sweep['continuous_sham_percentile'] == pct)
            & (sweep['tp1'] == 'Sham') & (sweep['tp2'] == 'D7')
        ]
        top = subset.reindex(subset['hedges_g'].abs().sort_values(ascending=False).index).head(5)
        print(f'    pct={pct:.1f}:')
        for _, row in top.iterrows():
            print(f'      {row["endpoint"]:<35s} g={row["hedges_g"]:+.2f}  g_neut={row["g_shrunk_neutral"]:+.3f}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
