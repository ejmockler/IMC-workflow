"""Composite-lineage DA + SN.

Parallel Phase 1 track operating on the 8-category interface decomposition
(none, immune, endothelial, stromal, three two-way interfaces, triple-positive)
derived per superpixel from continuous lineage_immune/endothelial/stromal
scores. Provides a named interface state for much of the tissue left
`unassigned` by the strict-discrete cell-type Phase 1 track.

Inputs (read-only):
  - results/biological_analysis/cell_type_annotations/*.parquet
    (continuous lineage scores per superpixel)

Outputs:
  - results/biological_analysis/differential_abundance_composite/
      temporal_differential_abundance.csv         (8 cats x 6 contrasts x 2 metrics)
      temporal_top_ranked_by_effect.csv           (top-5 per contrast per metric)
      roi_abundances.csv                          (24 rows x 8 cats)
  - results/biological_analysis/spatial_neighborhoods_composite/
      temporal_neighborhood_enrichments.csv       (8 x 8 x 4 timepoints)

The 8 interface categories are derived per superpixel via
`classify_interface_per_superpixel` from src/analysis/temporal_interface_analysis.py
to guarantee equivalence with the pre-registered Family A v1 categorization.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Dict
import json
from collections import Counter

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis import temporal_interface_analysis as tia
from src.utils.metadata import parse_roi_metadata
from src.utils.paths import get_paths
from run_temporal_interface_analysis import load_annotations_with_markers


SCALE_UM = 10.0
K_NEIGHBORS = 10
N_PERMUTATIONS = 1_000
LINEAGE_THRESHOLD = tia.DEFAULT_LINEAGE_THRESHOLD  # 0.3
INTERFACE_CATEGORIES = tia.INTERFACE_CATEGORIES  # 8 categories
PAIRWISE_CONTRASTS = tia.PAIRWISE_CONTRASTS

RNG_SEED = 42


def load_annotations() -> pd.DataFrame:
    """Load annotations + derive 8-interface category column."""
    df = load_annotations_with_markers(scale_um=SCALE_UM)
    df['interface'] = tia.classify_interface_per_superpixel(
        df, threshold=LINEAGE_THRESHOLD
    )
    return df


def verify_against_family_a_v1(df: pd.DataFrame) -> None:
    """Verify mouse-timepoint proportions against Family A v1 exactly."""
    paths = get_paths()
    ifrac_path = paths.biological_analysis_dir / 'temporal_interfaces' / 'interface_fractions.parquet'
    if not ifrac_path.exists():
        print('  (interface_fractions.parquet not present — skipping equivalence check)')
        return
    fa1 = pd.read_parquet(ifrac_path)
    if 'normalization_mode' in fa1.columns:
        fa1 = fa1[fa1['normalization_mode'] == 'per_roi_sigmoid']
    if 'threshold' in fa1.columns:
        fa1 = fa1[np.isclose(fa1['threshold'], LINEAGE_THRESHOLD)]

    key_cols = ['timepoint', 'mouse']
    required = set(key_cols) | set(INTERFACE_CATEGORIES)
    missing = sorted(required - set(fa1.columns))
    if missing:
        raise AssertionError(
            f'Family A v1 interface fractions are missing columns: {missing}'
        )

    verification_df = df.copy()
    canonical_mouse = {
        roi_id: parse_roi_metadata(roi_id)['mouse']
        for roi_id in verification_df['roi_id'].unique()
    }
    verification_df['mouse'] = verification_df['roi_id'].map(canonical_mouse)
    ours, _ = per_mouse_interface_proportions(verification_df)
    our_idx = ours.set_index(key_cols)[list(INTERFACE_CATEGORIES)].sort_index()
    fa1_idx = fa1.set_index(key_cols)[list(INTERFACE_CATEGORIES)].sort_index()
    if not our_idx.index.equals(fa1_idx.index):
        raise AssertionError(
            'Composite and Family A v1 mouse-timepoint keys do not match: '
            f'{our_idx.index.tolist()} != {fa1_idx.index.tolist()}'
        )

    differences = (our_idx - fa1_idx).abs()
    if differences.isna().any().any():
        raise AssertionError('Family A v1 equivalence comparison produced NaN')
    max_diff = float(differences.to_numpy().max())
    print(f'  Equivalence check: max abs(diff) = {max_diff:.2e}')
    if max_diff > 1e-6:
        raise AssertionError(
            f'Composite proportions differ from Family A v1 by {max_diff:.3e}'
        )


# ============================================================================
# Differential abundance over interface categories
# ============================================================================

def per_mouse_interface_proportions(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """For each (mouse, timepoint), proportion of superpixels in each interface
    category. ROI-level proportions averaged within mouse to prevent pseudoreplication."""
    # Per-ROI fractions
    roi_counts = (
        df.groupby(['roi_id', 'mouse', 'timepoint', 'interface'])
        .size().unstack(fill_value=0)
        .reindex(columns=list(INTERFACE_CATEGORIES), fill_value=0)
    )
    roi_totals = roi_counts.sum(axis=1)
    roi_frac = roi_counts.div(roi_totals, axis=0).reset_index()
    # Mouse-level: average across ROIs within mouse
    mouse_means = (
        roi_frac
        .drop(columns=['roi_id'])
        .groupby(['mouse', 'timepoint'])
        .mean()
        .reset_index()
    )
    return mouse_means, roi_frac


def hedges_g(g1: np.ndarray, g2: np.ndarray) -> Tuple[float, float, float, float]:
    """Hedges' g (small-sample-corrected); also returns pooled std and means."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan, np.nan
    m1, m2 = g1.mean(), g2.mean()
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled <= 0:
        return np.nan, pooled, m1, m2
    cohen = (m1 - m2) / pooled
    # Hedges correction (small-sample bias)
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    return cohen * j, pooled, m1, m2


def clr_transform(props: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Centered log-ratio transform on the interface proportion columns."""
    cols = [c for c in props.columns if c in INTERFACE_CATEGORIES]
    P = props[cols].values + eps
    geom_mean = np.exp(np.log(P).mean(axis=1, keepdims=True))
    return pd.DataFrame(np.log(P / geom_mean), columns=cols, index=props.index)


def bootstrap_g_range(g1: np.ndarray, g2: np.ndarray, n_iter: int = 10_000) -> Tuple[float, float]:
    """Min and max Hedges' g under bootstrap resampling. At n=2 there are only
    a handful of unique resamples so range is degenerate but informative."""
    rng = np.random.default_rng(RNG_SEED)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    gs = []
    for _ in range(n_iter):
        idx1 = rng.integers(0, n1, size=n1)
        idx2 = rng.integers(0, n2, size=n2)
        g, _, _, _ = hedges_g(g1[idx1], g2[idx2])
        if not np.isnan(g):
            gs.append(g)
    if not gs:
        return np.nan, np.nan
    return min(gs), max(gs)


def run_differential_abundance(mouse_means: pd.DataFrame) -> pd.DataFrame:
    """Per-category, per-pairwise-contrast Hedges' g + bootstrap + CLR."""
    rows = []
    cat_cols = [c for c in mouse_means.columns if c in INTERFACE_CATEGORIES]

    # Raw proportions g
    for cat in cat_cols:
        for tp1, tp2 in PAIRWISE_CONTRASTS:
            v1 = mouse_means[mouse_means['timepoint'] == tp1][cat].values
            v2 = mouse_means[mouse_means['timepoint'] == tp2][cat].values
            g, std, m1, m2 = hedges_g(v1, v2)
            br_min, br_max = bootstrap_g_range(v1, v2, n_iter=200) if not np.isnan(g) else (np.nan, np.nan)
            pathological = (not np.isnan(g) and abs(g) > 3 and std < 0.01)
            rows.append({
                'interface_category': cat, 'comparison': f'{tp1}_vs_{tp2}',
                'tp1': tp1, 'tp2': tp2,
                'n_mice_1': len(v1), 'n_mice_2': len(v2),
                'mean_1': m1, 'mean_2': m2, 'pooled_std': std,
                'hedges_g': g,
                'bootstrap_range_min': br_min, 'bootstrap_range_max': br_max,
                'g_pathological': pathological,
                'metric': 'raw_proportion',
            })

    # CLR-transformed g
    clr_df = clr_transform(mouse_means)
    clr_df['mouse'] = mouse_means['mouse'].values
    clr_df['timepoint'] = mouse_means['timepoint'].values
    for cat in cat_cols:
        for tp1, tp2 in PAIRWISE_CONTRASTS:
            v1 = clr_df[clr_df['timepoint'] == tp1][cat].values
            v2 = clr_df[clr_df['timepoint'] == tp2][cat].values
            g, std, m1, m2 = hedges_g(v1, v2)
            br_min, br_max = bootstrap_g_range(v1, v2, n_iter=200) if not np.isnan(g) else (np.nan, np.nan)
            pathological = (not np.isnan(g) and abs(g) > 3 and std < 0.01)
            rows.append({
                'interface_category': cat, 'comparison': f'{tp1}_vs_{tp2}',
                'tp1': tp1, 'tp2': tp2,
                'n_mice_1': len(v1), 'n_mice_2': len(v2),
                'mean_1': m1, 'mean_2': m2, 'pooled_std': std,
                'hedges_g': g,
                'bootstrap_range_min': br_min, 'bootstrap_range_max': br_max,
                'g_pathological': pathological,
                'metric': 'clr',
            })

    return pd.DataFrame(rows)


# ============================================================================
# Spatial neighborhood enrichment over 8 interface categories
# ============================================================================

def knn_indices(coords: np.ndarray, k: int = K_NEIGHBORS) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    n = len(coords)
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    return idx[:, 1:]  # exclude self


def permutation_enrichment_one_roi(
    roi_id: str, df_roi: pd.DataFrame, rng: np.random.Generator,
    n_perms: int = N_PERMUTATIONS,
) -> pd.DataFrame:
    """For each (focal, neighbor) pair, compute observed proportion of focal
    cells' k-NN that are neighbor type, vs permutation null. Phipson-Smyth p."""
    coords = df_roi[['x', 'y']].values
    types = df_roi['interface'].values
    n_total = len(types)
    if n_total < K_NEIGHBORS + 1:
        return pd.DataFrame()
    knn = knn_indices(coords, k=K_NEIGHBORS)

    # Expected from global frequencies
    cnts = Counter(types)
    expected = {t: c / n_total for t, c in cnts.items()}

    rows = []
    focals = [c for c in INTERFACE_CATEGORIES if c in cnts]
    neighbors = focals

    for focal_ct in focals:
        focal_mask = (types == focal_ct)
        n_focal = focal_mask.sum()
        if n_focal == 0:
            continue
        neigh_idx = knn[focal_mask]
        for neighbor_ct in neighbors:
            neighbor_count_obs = (types[neigh_idx] == neighbor_ct).sum()
            denom = neigh_idx.size
            obs_prop = neighbor_count_obs / denom if denom > 0 else 0.0
            exp_prop = expected.get(neighbor_ct, 0.0)
            enrichment = obs_prop / exp_prop if exp_prop > 0 else np.nan
            log2_e = np.log2(enrichment) if enrichment and enrichment > 0 else np.nan
            # Permutation
            n_extreme = 0
            for _ in range(n_perms):
                perm = rng.permutation(types)
                p_neigh = perm[neigh_idx]
                p_obs = (p_neigh == neighbor_ct).sum() / denom if denom > 0 else 0.0
                if (obs_prop > exp_prop and p_obs >= obs_prop) or (obs_prop < exp_prop and p_obs <= obs_prop):
                    n_extreme += 1
            p_value = (n_extreme + 1) / (n_perms + 1)  # Phipson-Smyth
            rows.append({
                'roi_id': roi_id, 'focal_cell_type': focal_ct, 'neighbor_cell_type': neighbor_ct,
                'observed_proportion': obs_prop, 'expected_proportion': exp_prop,
                'enrichment_score': enrichment, 'log2_enrichment': log2_e,
                'p_value_raw': p_value, 'n_focal_cells': int(n_focal),
            })
    return pd.DataFrame(rows)


def run_spatial_neighborhoods(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ROI permutation tests, then aggregate to timepoint."""
    rng = np.random.default_rng(RNG_SEED)
    per_roi = []
    for roi_id, df_roi in df.groupby('roi_id'):
        per_roi_df = permutation_enrichment_one_roi(
            roi_id, df_roi.reset_index(drop=True), rng,
        )
        if not per_roi_df.empty:
            tp = df_roi['timepoint'].iloc[0]
            per_roi_df['timepoint'] = tp
            per_roi.append(per_roi_df)
    if not per_roi:
        return pd.DataFrame()
    all_per_roi = pd.concat(per_roi, ignore_index=True)

    # Aggregate to timepoint
    agg = (
        all_per_roi.groupby(['focal_cell_type', 'neighbor_cell_type', 'timepoint'])
        .agg(
            observed_proportion=('observed_proportion', 'mean'),
            expected_proportion=('expected_proportion', 'mean'),
            enrichment_score=('enrichment_score', 'mean'),
            log2_enrichment=('log2_enrichment', 'mean'),
            fraction_significant_raw=('p_value_raw', lambda s: (s < 0.05).mean()),
            n_focal_cells=('n_focal_cells', 'sum'),
            n_rois=('roi_id', 'nunique'),
        )
        .reset_index()
    )
    return agg


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    paths = get_paths()
    da_dir = paths.biological_analysis_dir / 'differential_abundance_composite'
    sn_dir = paths.biological_analysis_dir / 'spatial_neighborhoods_composite'
    da_dir.mkdir(parents=True, exist_ok=True)
    sn_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print('Composite-Lineage Analysis (DA + SN over 8 interface categories)')
    print('=' * 80)

    print('\n[Load] Loading annotations + deriving 8-interface column...')
    df = load_annotations()
    print(f'  Loaded {len(df):,} superpixels across {df["roi_id"].nunique()} ROIs')
    counts = df['interface'].value_counts()
    print(f'  Interface category counts:')
    for cat in INTERFACE_CATEGORIES:
        n = counts.get(cat, 0)
        print(f'    {cat:<30s} {n:>6,d}  ({100*n/len(df):.2f}%)')

    print('\n[Verify] Equivalence with Family A v1 interface_fractions...')
    verify_against_family_a_v1(df)

    print('\n[DA] Computing per-mouse interface proportions + Hedges\' g + CLR...')
    mouse_means, roi_frac = per_mouse_interface_proportions(df)
    print(f'  {len(mouse_means)} mouse-timepoint rows; {len(roi_frac)} ROI rows')

    roi_frac.to_csv(da_dir / 'roi_abundances.csv', index=False)
    print(f'  ✓ Wrote {da_dir / "roi_abundances.csv"} ({len(roi_frac)} rows)')

    da_df = run_differential_abundance(mouse_means)
    da_df.to_csv(da_dir / 'temporal_differential_abundance.csv', index=False)
    print(f'  ✓ Wrote {da_dir / "temporal_differential_abundance.csv"} ({len(da_df)} rows)')

    # Top-ranked per contrast (across both metrics)
    top_rows = []
    for metric in ['raw_proportion', 'clr']:
        sub = da_df[da_df['metric'] == metric].copy()
        for contrast in sub['comparison'].unique():
            cs = sub[sub['comparison'] == contrast].copy()
            cs = cs.reindex(cs['hedges_g'].abs().sort_values(ascending=False).index)
            top5 = cs.head(5)
            top_rows.append(top5)
    top_df = pd.concat(top_rows, ignore_index=True)
    top_df.to_csv(da_dir / 'temporal_top_ranked_by_effect.csv', index=False)
    print(f'  ✓ Wrote {da_dir / "temporal_top_ranked_by_effect.csv"} ({len(top_df)} rows)')

    print('\n[SN] Computing per-ROI 8x8 spatial neighborhood enrichment...')
    print(f'  ({N_PERMUTATIONS} permutations per ROI x {df["roi_id"].nunique()} ROIs)')
    sn_df = run_spatial_neighborhoods(df)
    sn_df.to_csv(sn_dir / 'temporal_neighborhood_enrichments.csv', index=False)
    print(f'  ✓ Wrote {sn_dir / "temporal_neighborhood_enrichments.csv"} ({len(sn_df)} rows)')

    print('\nDone.')


if __name__ == '__main__':
    main()
