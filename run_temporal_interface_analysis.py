"""Orchestration script for the temporal interface analysis pipeline.

Implements Phase 2 of the pre-registered plan
(`analysis_plans/temporal_interfaces_plan.md`):
  Family A: interface fractions + CLR + threshold sensitivity (T21)
  Spatial: join-count + Moran's I on continuous lineage scores (T22)
  Family B: continuous neighborhood + neighbor-minus-self + trajectory filter (T23)
  Family C: Sham-reference compartment activation (T24)
  Endpoint summary table (T25)

Reads per-ROI annotation parquets (lineage scores) and the canonical superpixel
DataFrame (raw markers); joins them on (roi, superpixel_id). Writes typed
parquet outputs to results/biological_analysis/temporal_interfaces/.

Pure-function module is in src/analysis/temporal_interface_analysis.py — this
script is just a thin orchestrator.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.analysis import temporal_interface_analysis as tia
from src.utils.canonical_loader import build_superpixel_dataframe, load_all_rois


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
ANNOTATION_DIR = REPO_ROOT / 'results' / 'biological_analysis' / 'cell_type_annotations'
ROI_RESULTS_DIR = REPO_ROOT / 'results' / 'roi_results'
OUTPUT_DIR = REPO_ROOT / 'results' / 'biological_analysis' / 'temporal_interfaces'
CONFIG_PATH = REPO_ROOT / 'config.json'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_annotations_with_markers(scale_um: float = 10.0) -> pd.DataFrame:
    """Join per-ROI annotation parquets (lineage scores) with canonical
    superpixel DataFrame (raw markers, metadata).

    Returns a single DataFrame with columns: roi_id, x, y, superpixel_id,
    timepoint, mouse, lineage_*, composite_label, raw markers.
    """
    canonical_results = load_all_rois(str(ROI_RESULTS_DIR))
    superpixels = build_superpixel_dataframe(canonical_results, scale=scale_um)
    superpixels = superpixels.rename(columns={'roi': 'roi_short'})

    annotation_frames: List[pd.DataFrame] = []
    for parquet in sorted(ANNOTATION_DIR.glob('roi_*_cell_types.parquet')):
        roi_short = parquet.stem.replace('roi_', '').replace('_cell_types', '')
        df = pd.read_parquet(parquet)
        df['roi_short'] = roi_short
        annotation_frames.append(df)
    annotations = pd.concat(annotation_frames, ignore_index=True)

    drop_cols = [c for c in ('x', 'y', 'cluster') if c in annotations.columns]
    annotations = annotations.drop(columns=drop_cols, errors='ignore')

    merged = superpixels.merge(
        annotations, on=['roi_short', 'superpixel_id'], how='inner', validate='one_to_one',
    )
    merged['roi_id'] = merged['roi_short']
    return merged


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def write_provenance(annotations: pd.DataFrame, scale_um: float) -> Dict[str, object]:
    import datetime
    import platform
    import scipy
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT
    ).decode().strip()
    git_dirty = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=REPO_ROOT
    ).decode().strip().split('\n')
    untracked_files = [ln[3:] for ln in git_dirty if ln.startswith('??')]
    modified_files = [ln[3:] for ln in git_dirty if ln.startswith(' M') or ln.startswith('M ')]

    config_hash = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()

    # Hash the analysis files themselves so provenance is reproducible even if
    # they're not yet committed to git
    analysis_files = {
        'src/analysis/temporal_interface_analysis.py': REPO_ROOT / 'src' / 'analysis' / 'temporal_interface_analysis.py',
        'run_temporal_interface_analysis.py': REPO_ROOT / 'run_temporal_interface_analysis.py',
        'analysis_plans/temporal_interfaces_plan.md': REPO_ROOT / 'analysis_plans' / 'temporal_interfaces_plan.md',
        'analysis_plans/deprecation_manifest.md': REPO_ROOT / 'analysis_plans' / 'deprecation_manifest.md',
    }
    file_hashes = {
        rel: hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        for rel, path in analysis_files.items()
    }

    rois = sorted(annotations['roi_id'].unique().tolist())
    mouse_table = (
        annotations[['roi_id', 'timepoint', 'mouse']]
        .drop_duplicates()
        .sort_values(['timepoint', 'mouse', 'roi_id'])
        .to_dict(orient='records')
    )

    provenance = {
        'run_datetime_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'git_commit': git_hash,
        'git_dirty': len(git_dirty) > 0 and git_dirty != [''],
        'git_untracked_critical_files': [
            f for f in untracked_files
            if any(f.startswith(p) for p in ('src/analysis/temporal_', 'run_temporal_', 'analysis_plans/', 'tests/test_temporal_'))
        ],
        'git_modified_critical_files': [
            f for f in modified_files
            if 'differential_abundance' in f or 'batch_annotate' in f
        ],
        'analysis_file_sha256': file_hashes,
        'config_hash_sha256': config_hash,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'package_versions': {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scipy': scipy.__version__,
        },
        'random_seeds': {
            'bootstrap_seed': 42,
            'join_count_seed': 42,
        },
        'pipeline_parameters': {
            'k_neighbors': tia.DEFAULT_K_NEIGHBORS,
            'permutations': tia.DEFAULT_PERMUTATIONS,
            'bootstrap_iter': tia.DEFAULT_BOOTSTRAP_ITER,
            'lineage_threshold_default': tia.DEFAULT_LINEAGE_THRESHOLD,
            'lineage_threshold_sensitivity_sweep': [0.2, 0.3, 0.4],
            'min_support_default': tia.DEFAULT_MIN_SUPPORT,
            'min_support_sensitivity_sweep': [10, 20, 40],
            'sham_reference_percentile_default': tia.DEFAULT_SHAM_PERCENTILE,
            'sham_reference_percentile_sensitivity_sweep': [65.0, 75.0, 85.0],
            'min_prevalence_for_collapse': tia.DEFAULT_MIN_PREVALENCE,
            'pathology_g_threshold': tia.PATHOLOGY_G_THRESHOLD,
            'pathology_std_threshold': tia.PATHOLOGY_STD_THRESHOLD,
            'shrinkage_prior_sd_skeptical': tia.SHRINKAGE_PRIOR_SD_SKEPTICAL,
            'shrinkage_prior_sd_neutral': tia.SHRINKAGE_PRIOR_SD_NEUTRAL,
            'shrinkage_prior_sd_optimistic': tia.SHRINKAGE_PRIOR_SD_OPTIMISTIC,
            'pairwise_contrasts': [list(c) for c in tia.PAIRWISE_CONTRASTS],
        },
        'scale_um': scale_um,
        'n_rois_analyzed': len(rois),
        'rois_analyzed': rois,
        'roi_to_mouse_mapping': mouse_table,
        'excluded_rois': [
            'Test/calibration acquisitions (filtered before annotation pipeline; not present in roi_results/)',
        ],
        'n_total_superpixels': int(len(annotations)),
    }
    return provenance


# ---------------------------------------------------------------------------
# Family A — Interface fractions + CLR + threshold sensitivity
# ---------------------------------------------------------------------------

def run_family_a(annotations: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Returns dict: interface_fractions (mouse-level), interface_clr,
    interface_clr_no_none, sensitivity_thresholds, family_a_endpoints.
    """
    out: Dict[str, pd.DataFrame] = {}

    # Primary threshold (0.3)
    per_roi = tia.compute_interface_fractions_per_roi(
        annotations, threshold=tia.DEFAULT_LINEAGE_THRESHOLD,
    )
    annotated = tia.attach_roi_metadata(per_roi)
    mouse = tia.aggregate_fractions_to_mouse_level(annotated)
    filtered, collapsed = tia.apply_min_prevalence_filter(mouse)
    out['interface_fractions'] = filtered.assign(threshold=tia.DEFAULT_LINEAGE_THRESHOLD)

    # CLR with and without "none"
    clr_with = tia.compute_interface_clr_table(filtered, exclude_none=False)
    clr_without = tia.compute_interface_clr_table(filtered, exclude_none=True)
    out['interface_clr'] = clr_with
    out['interface_clr_no_none'] = clr_without

    # Threshold sensitivity sweep
    sensitivity_rows = []
    for thr in (0.2, 0.3, 0.4):
        s_per_roi = tia.compute_interface_fractions_per_roi(annotations, threshold=thr)
        s_annotated = tia.attach_roi_metadata(s_per_roi)
        s_mouse = tia.aggregate_fractions_to_mouse_level(s_annotated)
        s_filtered, _ = tia.apply_min_prevalence_filter(s_mouse)
        s_filtered['threshold'] = thr
        sensitivity_rows.append(s_filtered)
    out['sensitivity_thresholds'] = pd.concat(sensitivity_rows, ignore_index=True)

    # Endpoint table on CLR coordinates (primary)
    clr_value_cols = [c for c in clr_with.columns if c.endswith('_clr')]
    endpoints = tia.pairwise_endpoint_table(
        clr_with, value_cols=clr_value_cols, family='A_interface_clr',
    )
    out['family_a_endpoints'] = endpoints
    out['family_a_collapsed_categories'] = pd.DataFrame({'collapsed': collapsed})

    return out


# ---------------------------------------------------------------------------
# Spatial — join-count + Moran's I per ROI per indicator/lineage
# ---------------------------------------------------------------------------

def run_spatial_coherence(annotations: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    join_count_rows = []
    morans_rows = []

    for roi_id, group in annotations.groupby('roi_id'):
        coords = group[['x', 'y']].values
        labels = tia.classify_interface_per_superpixel(group, threshold=tia.DEFAULT_LINEAGE_THRESHOLD)
        meta_row = {'roi_id': roi_id, 'timepoint': group['timepoint'].iloc[0], 'mouse': group['mouse'].iloc[0]}

        for category in tia.INTERFACE_CATEGORIES:
            binary = (labels.values == category).astype(int)
            jc = tia.compute_join_count_bb(binary, coords, k=tia.DEFAULT_K_NEIGHBORS, n_perm=tia.DEFAULT_PERMUTATIONS, seed=42)
            row = {**meta_row, 'category': category}
            row.update(jc)
            join_count_rows.append(row)

        for col in tia.LINEAGE_COLS:
            mi = tia.compute_morans_i_continuous(group[col].values, coords, k=tia.DEFAULT_K_NEIGHBORS)
            morans_rows.append({**meta_row, 'lineage': col, 'morans_i': mi})

    out['join_counts'] = pd.DataFrame(join_count_rows)
    out['lineage_morans_i'] = pd.DataFrame(morans_rows)
    return out


# ---------------------------------------------------------------------------
# Family B — Continuous neighborhood + neighbor-minus-self + trajectory filter
# ---------------------------------------------------------------------------

def _run_family_b_at_support(
    annotations: pd.DataFrame, min_support: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (delta_vs_sham filtered, missingness, family_b_endpoints) for a given min_support."""
    per_roi_frames: List[pd.DataFrame] = []
    for roi_id, group in annotations.groupby('roi_id'):
        per_roi = tia.compute_neighbor_minus_self_per_roi(
            group, roi_id=roi_id, k=tia.DEFAULT_K_NEIGHBORS, min_support=min_support,
        )
        if not per_roi.empty:
            per_roi_frames.append(per_roi)
    if not per_roi_frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    per_roi_all = pd.concat(per_roi_frames, ignore_index=True)
    # Aggregate uses only sufficient (non-below-min-support) rows for delta means
    sufficient = per_roi_all[~per_roi_all['below_min_support']]
    mouse = tia.aggregate_neighbor_delta_to_mouse(sufficient) if not sufficient.empty else pd.DataFrame()
    filtered, missingness = tia.apply_trajectory_filter(mouse, per_roi_raw=per_roi_all)
    if filtered.empty:
        return filtered, missingness, pd.DataFrame()
    delta_vs_sham = tia.compute_delta_vs_sham(filtered)
    delta_cols = [c for c in delta_vs_sham.columns if c.startswith('vs_sham_mean_delta_')]
    family_b_endpoints = tia.pairwise_endpoint_table(
        delta_vs_sham, value_cols=delta_cols,
        family='B_continuous_neighborhood', extra_index_cols=['composite_label'],
    )
    return delta_vs_sham, missingness, family_b_endpoints


def run_family_b(annotations: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    primary_dvs, primary_miss, primary_endpoints = _run_family_b_at_support(
        annotations, min_support=tia.DEFAULT_MIN_SUPPORT,
    )
    out['continuous_neighborhood_temporal'] = primary_dvs
    out['continuous_neighborhood_missingness'] = primary_miss
    out['family_b_endpoints'] = primary_endpoints

    # Sensitivity sweep across min_support (10/20/40)
    sensitivity_rows: List[pd.DataFrame] = []
    for ms in (10, 20, 40):
        _, _, ep = _run_family_b_at_support(annotations, min_support=ms)
        if not ep.empty:
            ep['min_support'] = ms
            sensitivity_rows.append(ep)
    out['family_b_sensitivity'] = pd.concat(sensitivity_rows, ignore_index=True) if sensitivity_rows else pd.DataFrame()
    return out


# ---------------------------------------------------------------------------
# Family C — Sham-reference compartment activation
# ---------------------------------------------------------------------------

def _run_family_c_at_percentile(
    annotations: pd.DataFrame, percentile: float,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    compartment_markers = ('CD45', 'CD31', 'CD140b')
    activation_marker = 'CD44'
    sham_thresholds = tia.compute_sham_reference_thresholds(
        annotations, list(compartment_markers) + [activation_marker], percentile=percentile,
    )
    per_roi = tia.compute_compartment_activation_per_roi(
        annotations, sham_thresholds=sham_thresholds,
        compartment_markers=compartment_markers, activation_marker=activation_marker,
    )
    mouse = tia.aggregate_compartment_activation_to_mouse(per_roi)
    rate_cols = [c for c in mouse.columns if c.endswith('_cd44_rate') or c == 'triple_overlap_fraction']
    endpoints = tia.pairwise_endpoint_table(
        mouse, value_cols=rate_cols, family='C_compartment_activation',
    )
    return sham_thresholds, mouse, endpoints


def run_family_c(annotations: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    sham_thresholds, mouse, endpoints = _run_family_c_at_percentile(
        annotations, percentile=tia.DEFAULT_SHAM_PERCENTILE,
    )
    out['sham_reference_thresholds'] = pd.DataFrame([sham_thresholds])
    out['compartment_activation_temporal'] = mouse
    out['family_c_endpoints'] = endpoints

    # Sensitivity sweep across Sham-reference percentile (65/75/85)
    sensitivity_rows: List[pd.DataFrame] = []
    for pct in (65.0, 75.0, 85.0):
        _, _, ep = _run_family_c_at_percentile(annotations, percentile=pct)
        if not ep.empty:
            ep['sham_percentile'] = pct
            sensitivity_rows.append(ep)
    out['family_c_sensitivity'] = pd.concat(sensitivity_rows, ignore_index=True) if sensitivity_rows else pd.DataFrame()
    return out


# ---------------------------------------------------------------------------
# Endpoint summary
# ---------------------------------------------------------------------------

def assemble_endpoint_summary(
    family_outputs: List[pd.DataFrame],
    sensitivity_outputs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build the reviewer-facing summary table.

    Pathological rows and insufficient-support rows are EXCLUDED from BH-FDR
    computations (q-proxy values set to NaN for them) so they don't anchor
    the BH step-up procedure. The threshold_sensitive flag fires when a
    finding's Hedges' g sign reverses across the sensitivity sweep.
    """
    non_empty = [df for df in family_outputs if df is not None and not df.empty]
    if not non_empty:
        return pd.DataFrame()
    summary = pd.concat(non_empty, ignore_index=True)

    # observed_range column = max(mouse_range_1, mouse_range_2)
    summary['observed_range'] = summary[['mouse_range_1', 'mouse_range_2']].max(axis=1)

    # threshold_sensitive flag: sign of g reverses across the relevant sensitivity sweep
    summary['threshold_sensitive'] = False
    for sens_name, sens_df in sensitivity_outputs.items():
        if sens_df.empty:
            continue
        sweep_var = next((c for c in ('threshold', 'min_support', 'sham_percentile') if c in sens_df.columns), None)
        if sweep_var is None:
            continue
        for keys, group in sens_df.groupby(['family', 'endpoint', 'contrast']):
            signs = np.sign(group['hedges_g'].dropna().values)
            if len(np.unique(signs)) > 1:
                mask = (
                    (summary['family'] == keys[0]) &
                    (summary['endpoint'] == keys[1]) &
                    (summary['contrast'] == keys[2])
                )
                summary.loc[mask, 'threshold_sensitive'] = True

    # p_proxy + BH-FDR — exclude pathological and insufficient_support from the calculation
    z = summary['hedges_g'].abs().fillna(0).values
    from scipy import stats as scipy_stats
    summary['p_proxy_from_g'] = 2 * (1 - scipy_stats.norm.cdf(z))
    eligible = ~(summary['g_pathological'] | summary['insufficient_support'])
    summary['q_proxy_within_family'] = np.nan
    for fam, idx in summary[eligible].groupby('family').groups.items():
        sub_p = summary.loc[idx, 'p_proxy_from_g'].values
        summary.loc[idx, 'q_proxy_within_family'] = tia.benjamini_hochberg(sub_p)
    summary['q_proxy_pooled'] = np.nan
    eligible_idx = summary.index[eligible]
    summary.loc[eligible_idx, 'q_proxy_pooled'] = tia.benjamini_hochberg(
        summary.loc[eligible_idx, 'p_proxy_from_g'].values
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(scale_um: float = 10.0) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading annotations + raw markers at {scale_um}μm scale...")
    annotations = load_annotations_with_markers(scale_um=scale_um)
    print(f"  {len(annotations)} superpixels across {annotations['roi_id'].nunique()} ROIs")

    provenance = write_provenance(annotations, scale_um=scale_um)
    (OUTPUT_DIR / 'run_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f"  Wrote provenance: {OUTPUT_DIR / 'run_provenance.json'}")

    print("\n[Family A] Interface fractions + CLR + threshold sensitivity")
    fa = run_family_a(annotations)
    fa['interface_fractions'].to_parquet(OUTPUT_DIR / 'interface_fractions.parquet')
    fa['interface_clr'].to_parquet(OUTPUT_DIR / 'interface_clr.parquet')
    fa['interface_clr_no_none'].to_parquet(OUTPUT_DIR / 'interface_clr_no_none.parquet')
    fa['sensitivity_thresholds'].to_parquet(OUTPUT_DIR / 'sensitivity_thresholds.parquet')
    print(f"  Mouse-level fractions: {len(fa['interface_fractions'])} rows")
    print(f"  CLR endpoints: {len(fa['family_a_endpoints'])} rows")
    if not fa['family_a_collapsed_categories'].empty:
        print(f"  Collapsed rare categories: {fa['family_a_collapsed_categories']['collapsed'].tolist()}")

    print("\n[Spatial] Join-count statistics + Moran's I")
    sc = run_spatial_coherence(annotations)
    sc['join_counts'].to_parquet(OUTPUT_DIR / 'join_counts.parquet')
    sc['lineage_morans_i'].to_parquet(OUTPUT_DIR / 'lineage_morans_i.parquet')
    print(f"  Join-count rows: {len(sc['join_counts'])}")
    print(f"  Moran's I rows: {len(sc['lineage_morans_i'])}")

    print("\n[Family B] Continuous neighborhood + neighbor-minus-self")
    fb = run_family_b(annotations)
    if 'continuous_neighborhood_temporal' in fb:
        fb['continuous_neighborhood_temporal'].to_parquet(OUTPUT_DIR / 'continuous_neighborhood_temporal.parquet')
    if 'continuous_neighborhood_missingness' in fb:
        fb['continuous_neighborhood_missingness'].to_parquet(OUTPUT_DIR / 'continuous_neighborhood_missingness.parquet')
    print(f"  Mouse-level delta rows: {len(fb.get('continuous_neighborhood_temporal', []))}")

    print("\n[Family C] Cross-compartment activation (Sham-reference threshold)")
    fc = run_family_c(annotations)
    fc['compartment_activation_temporal'].to_parquet(OUTPUT_DIR / 'compartment_activation_temporal.parquet')
    fc['sham_reference_thresholds'].to_parquet(OUTPUT_DIR / 'sham_reference_thresholds.parquet')
    print(f"  Mouse-level activation rows: {len(fc['compartment_activation_temporal'])}")

    print("\n[Summary] Endpoint summary across families")
    family_endpoints = [
        fa.get('family_a_endpoints'),
        fb.get('family_b_endpoints'),
        fc.get('family_c_endpoints'),
    ]
    sensitivity_outputs = {
        'family_a': fa.get('sensitivity_thresholds', pd.DataFrame()),
        'family_b': fb.get('family_b_sensitivity', pd.DataFrame()),
        'family_c': fc.get('family_c_sensitivity', pd.DataFrame()),
    }
    # Family A sensitivity needs to be transformed into endpoint format for the threshold check
    # Use the Family A primary endpoints rerun across thresholds
    fa_sens_endpoints = []
    for thr_label, thr_subset in fa['sensitivity_thresholds'].groupby('threshold'):
        fraction_cols = [c for c in tia.INTERFACE_CATEGORIES if c in thr_subset.columns]
        # Build CLR for this threshold's fractions
        clr_thr = tia.compute_interface_clr_table(thr_subset, exclude_none=False)
        clr_value_cols = [c for c in clr_thr.columns if c.endswith('_clr')]
        ep_thr = tia.pairwise_endpoint_table(clr_thr, value_cols=clr_value_cols, family='A_interface_clr')
        ep_thr['threshold'] = thr_label
        fa_sens_endpoints.append(ep_thr)
    sensitivity_outputs['family_a'] = pd.concat(fa_sens_endpoints, ignore_index=True) if fa_sens_endpoints else pd.DataFrame()

    # Write sensitivity endpoint outputs for inspection
    if not sensitivity_outputs['family_a'].empty:
        sensitivity_outputs['family_a'].to_parquet(OUTPUT_DIR / 'family_a_sensitivity_endpoints.parquet')
    if not sensitivity_outputs['family_b'].empty:
        sensitivity_outputs['family_b'].to_parquet(OUTPUT_DIR / 'family_b_sensitivity_endpoints.parquet')
    if not sensitivity_outputs['family_c'].empty:
        sensitivity_outputs['family_c'].to_parquet(OUTPUT_DIR / 'family_c_sensitivity_endpoints.parquet')

    summary = assemble_endpoint_summary(family_endpoints, sensitivity_outputs)
    summary.to_csv(OUTPUT_DIR / 'endpoint_summary.csv', index=False)
    print(f"  Total endpoints: {len(summary)}")
    if not summary.empty:
        flagged = int(summary['g_pathological'].sum())
        insufficient = int(summary.get('insufficient_support', pd.Series([])).sum() or 0)
        thr_sens = int(summary['threshold_sensitive'].sum())
        n_with_q = int(summary['q_proxy_pooled'].notna().sum())
        print(f"  Pathological-flag rows: {flagged} (excluded from BH)")
        print(f"  Insufficient-support rows: {insufficient} (excluded from BH)")
        print(f"  Threshold-sensitive rows (sign reversal in sweep): {thr_sens}")
        print(f"  Eligible-for-BH rows: {n_with_q}")

    print(f"\n✓ Outputs at {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
