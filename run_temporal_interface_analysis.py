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

    # Load Sham-reference artifact metadata so the continuous-path
    # normalization knob is recorded alongside the raw-marker sweep
    # percentiles. Critical for reviewer reproducibility (Codex #5).
    sham_ref_path = (
        REPO_ROOT / 'results' / 'biological_analysis' /
        f'sham_reference_{scale_um}um.json'
    )
    if sham_ref_path.exists():
        sham_ref_sha = hashlib.sha256(sham_ref_path.read_bytes()).hexdigest()
        with open(sham_ref_path) as f:
            sham_ref_meta = json.load(f).get('_metadata', {})
    else:
        sham_ref_sha = None
        sham_ref_meta = {}

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
            if (
                'differential_abundance' in f
                or 'batch_annotate' in f
                or 'temporal_interface' in f
                or 'spatial_neighborhood' in f
                or f.startswith('analysis_plans/')
                or f.startswith('src/utils/metadata.py')
                or f.startswith('run_temporal_interface_analysis.py')
            )
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
            'continuous_sham_reference': {
                'artifact_path': str(sham_ref_path.relative_to(REPO_ROOT))
                    if sham_ref_path.exists() else None,
                'artifact_sha256': sham_ref_sha,
                'percentile': sham_ref_meta.get('percentile'),
                'aggregation': sham_ref_meta.get('aggregation'),
                'n_sham_mice': sham_ref_meta.get('n_sham_mice'),
                'n_sham_rois': sham_ref_meta.get('n_sham_rois'),
                'note': 'Drives compute_continuous_memberships sigmoid centers '
                        'via batch_annotate_all_rois. Deferred sensitivity '
                        'sweep at 50/60/70 pct is a Phase-1.5 follow-up.',
            },
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

def _load_continuous_sham_percentile(scale_um: float) -> float:
    """Read the continuous-Sham percentile from the artifact so we can
    stamp it onto per_roi_sigmoid endpoint rows (reviewers need to see it
    without cross-referencing a separate file). Returns np.nan if the
    artifact is missing (e.g., a diagnostic-only run without memberships).
    """
    path = (
        REPO_ROOT / 'results' / 'biological_analysis' /
        f'sham_reference_{scale_um}um.json'
    )
    if not path.exists():
        return float('nan')
    with open(path) as f:
        meta = json.load(f).get('_metadata', {})
    pct = meta.get('percentile')
    return float(pct) if pct is not None else float('nan')


def run_family_a(
    annotations: pd.DataFrame,
    continuous_sham_percentile: float = float('nan'),
) -> Dict[str, pd.DataFrame]:
    """Returns dict: interface_fractions (mouse-level), interface_clr,
    interface_clr_no_none, sensitivity_thresholds, family_a_endpoints,
    normalization_sensitivity.

    ``continuous_sham_percentile`` is stamped onto every per_roi_sigmoid
    endpoint row so reviewers can see which Sham percentile drove the
    sigmoid centers.
    """
    out: Dict[str, pd.DataFrame] = {}

    # Primary threshold (0.3) — uses per-ROI sigmoid-normalized lineage scores
    per_roi = tia.compute_interface_fractions_per_roi(
        annotations, threshold=tia.DEFAULT_LINEAGE_THRESHOLD,
    )
    annotated = tia.attach_roi_metadata(per_roi)
    mouse = tia.aggregate_fractions_to_mouse_level(annotated)
    filtered, collapsed = tia.apply_min_prevalence_filter(mouse)
    out['interface_fractions'] = filtered.assign(
        threshold=tia.DEFAULT_LINEAGE_THRESHOLD,
        normalization_mode='per_roi_sigmoid',
    )

    # CLR with and without "none"
    clr_with = tia.compute_interface_clr_table(filtered, exclude_none=False)
    clr_without = tia.compute_interface_clr_table(filtered, exclude_none=True)
    out['interface_clr'] = clr_with
    out['interface_clr_no_none'] = clr_without

    # Threshold sensitivity sweep (per-ROI normalization with varying lineage threshold)
    sensitivity_rows = []
    for thr in (0.2, 0.3, 0.4):
        s_per_roi = tia.compute_interface_fractions_per_roi(annotations, threshold=thr)
        s_annotated = tia.attach_roi_metadata(s_per_roi)
        s_mouse = tia.aggregate_fractions_to_mouse_level(s_annotated)
        s_filtered, _ = tia.apply_min_prevalence_filter(s_mouse)
        s_filtered['threshold'] = thr
        sensitivity_rows.append(s_filtered)
    out['sensitivity_thresholds'] = pd.concat(sensitivity_rows, ignore_index=True)

    # Normalization sensitivity: raw-marker Sham-reference threshold classification.
    # The per-ROI sigmoid normalization is the longest-standing unresolved
    # confound. Use Sham-only superpixels to define the threshold (matches
    # Family C philosophy; avoids outcome contamination by D1/D3/D7 elevated
    # markers driving the threshold, which would partially obscure the
    # injury-driven effects we want to detect).
    sweep_percentiles = (65.0, 75.0, 85.0)
    sensitivity_per_pct: Dict[float, pd.DataFrame] = {}
    sensitivity_endpoint_frames = []
    for pct in sweep_percentiles:
        thresholds_pct = tia.compute_global_marker_thresholds(
            annotations, percentile=pct, sham_only=True,
        )
        sham_per_roi = tia.compute_interface_fractions_global_per_roi(
            annotations, global_thresholds=thresholds_pct,
        )
        sham_annotated = tia.attach_roi_metadata(sham_per_roi)
        sham_mouse = tia.aggregate_fractions_to_mouse_level(sham_annotated)
        sham_filtered, _ = tia.apply_min_prevalence_filter(sham_mouse)
        sham_filtered = sham_filtered.assign(
            sham_percentile=pct,
            normalization_mode='sham_reference_raw_marker',
        )
        sensitivity_per_pct[pct] = sham_filtered
        sham_clr_pct = tia.compute_interface_clr_table(sham_filtered, exclude_none=False)
        endpoints_pct = tia.pairwise_endpoint_table(
            sham_clr_pct, value_cols=[c for c in sham_clr_pct.columns if c.endswith('_clr')],
            family='A_interface_clr',
        )
        endpoints_pct['normalization_mode'] = 'sham_reference_raw_marker'
        endpoints_pct['sham_percentile'] = pct
        sensitivity_endpoint_frames.append(endpoints_pct)

    # Primary normalization comparison: per-ROI vs Sham-ref @ 75th percentile
    sham_ref_primary = sensitivity_per_pct[75.0]
    out['normalization_sensitivity'] = pd.concat(
        [out['interface_fractions'], sham_ref_primary], ignore_index=True,
    )

    clr_value_cols = [c for c in clr_with.columns if c.endswith('_clr')]
    endpoints = tia.pairwise_endpoint_table(
        clr_with, value_cols=clr_value_cols, family='A_interface_clr',
    )
    endpoints['normalization_mode'] = 'per_roi_sigmoid'

    # Pre-registration obligation §5 (CLR sensitivity): compare endpoints
    # with and without the 'none' category. A qualitative change (sign
    # reversal of Hedges' g on the same category across the two CLR
    # transforms) is flagged via ``clr_none_sensitivity`` — 'none'
    # participates in the CLR geometric mean, so excluding it shifts every
    # other category's log-ratio non-uniformly.
    clr_without_cols = [c for c in clr_without.columns if c.endswith('_clr')]
    endpoints_no_none = tia.pairwise_endpoint_table(
        clr_without, value_cols=clr_without_cols, family='A_interface_clr',
    ) if clr_without_cols else pd.DataFrame()
    if not endpoints_no_none.empty:
        merged = endpoints[['endpoint', 'contrast', 'hedges_g']].merge(
            endpoints_no_none[['endpoint', 'contrast', 'hedges_g']],
            on=['endpoint', 'contrast'], suffixes=('', '_no_none'),
            how='left',
        )
        both_finite = (
            merged['hedges_g'].notna() & merged['hedges_g_no_none'].notna()
        )
        sign_flip = (
            np.sign(merged['hedges_g']) != np.sign(merged['hedges_g_no_none'])
        ) & both_finite
        endpoints = endpoints.merge(
            merged[['endpoint', 'contrast', 'hedges_g_no_none']].assign(
                clr_none_sensitivity=sign_flip.values,
            ),
            on=['endpoint', 'contrast'], how='left',
        )
    else:
        endpoints['hedges_g_no_none'] = np.nan
        endpoints['clr_none_sensitivity'] = False

    # The continuous-Sham percentile that drove compute_continuous_memberships.
    # Populated here so reviewers can see the value on every Family A per_roi
    # row without cross-referencing sham_reference_10.0um.json.
    endpoints['sham_percentile'] = continuous_sham_percentile

    primary_global = sensitivity_endpoint_frames[1]  # 75th percentile
    out['family_a_endpoints'] = endpoints
    out['family_a_endpoints_global_norm'] = primary_global
    out['family_a_endpoints_norm_sweep'] = pd.concat(sensitivity_endpoint_frames, ignore_index=True)
    out['family_a_collapsed_categories'] = pd.DataFrame({'collapsed': collapsed})
    out['family_a_global_thresholds'] = pd.DataFrame([
        {**tia.compute_global_marker_thresholds(annotations, percentile=p, sham_only=True), 'sham_percentile': p}
        for p in sweep_percentiles
    ])

    # Normalization audit merged into the per-ROI endpoint table so every
    # comparison carries its disagreement flags. Symmetric magnitude check
    # added alongside the historical asymmetric one (Claude brutalist: the
    # asymmetric metric flags only "per_roi inflates, sham_ref deflates"
    # direction and understands disagreement by ~13×).
    rev = endpoints[['endpoint', 'contrast', 'hedges_g']].merge(
        primary_global[['endpoint', 'contrast', 'hedges_g']],
        on=['endpoint', 'contrast'], suffixes=('_per_roi', '_sham_ref'),
    )
    per_roi_g = rev['hedges_g_per_roi']
    sham_ref_g = rev['hedges_g_sham_ref']
    both_finite = per_roi_g.notna() & sham_ref_g.notna()
    rev['normalization_sign_reverse'] = (
        np.sign(per_roi_g) != np.sign(sham_ref_g)
    ) & both_finite
    rev['normalization_g_collapse'] = (
        per_roi_g.abs() > 0.5
    ) & (sham_ref_g.abs() < 0.2 * per_roi_g.abs())
    # Symmetric magnitude disagreement: either path has |g|>0.5 AND the
    # two paths disagree by >= 2x in the larger magnitude.
    abs_max = np.maximum(per_roi_g.abs(), sham_ref_g.abs())
    abs_diff = (per_roi_g - sham_ref_g).abs()
    rev['normalization_magnitude_disagree'] = (
        (abs_max > 0.5) & (abs_diff / abs_max.where(abs_max > 0, 1) > 0.5)
    ) & both_finite
    out['family_a_endpoints'] = endpoints.merge(
        rev[['endpoint', 'contrast',
             'normalization_sign_reverse',
             'normalization_g_collapse',
             'normalization_magnitude_disagree',
             'hedges_g_sham_ref']],
        on=['endpoint', 'contrast'], how='left',
    )
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
    # Family B neighbor-minus-self is computed on the Sham-referenced
    # continuous lineage scores (lineage_immune/endothelial/stromal cols in
    # the parquet), so it inherits the Seam-1 Sham-reference sigmoid. Stamp
    # so every row carries its normalization provenance; downstream
    # reviewers no longer need to chase Family A to infer it.
    if not family_b_endpoints.empty:
        family_b_endpoints['normalization_mode'] = 'sham_reference_v2_continuous'
    return delta_vs_sham, missingness, family_b_endpoints


def run_family_b(annotations: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    primary_dvs, primary_miss, primary_endpoints = _run_family_b_at_support(
        annotations, min_support=tia.DEFAULT_MIN_SUPPORT,
    )
    out['continuous_neighborhood_temporal'] = primary_dvs
    out['continuous_neighborhood_missingness'] = primary_miss

    # Sensitivity sweep across min_support (10/20/40)
    sensitivity_rows: List[pd.DataFrame] = []
    presence_by_ms: Dict[int, set] = {}
    for ms in (10, 20, 40):
        _, _, ep = _run_family_b_at_support(annotations, min_support=ms)
        if not ep.empty:
            ep['min_support'] = ms
            sensitivity_rows.append(ep)
            presence_by_ms[ms] = set(
                zip(ep['endpoint'], ep['contrast'], ep['composite_label'])
            )
        else:
            presence_by_ms[ms] = set()
    out['family_b_sensitivity'] = pd.concat(sensitivity_rows, ignore_index=True) if sensitivity_rows else pd.DataFrame()

    # Pre-registration obligation §Family B: flag findings that appear at
    # one support threshold but not another (filter-fragile). A row in the
    # primary sweep (ms=20) is support_sensitive if the same
    # (endpoint, contrast, composite_label) key is NOT present at every
    # min_support in the sweep. The threshold_sensitive flag built in
    # assemble_endpoint_summary catches sign reversals across the sweep;
    # this catches the presence-based fragility pre-reg §87 specifies.
    if not primary_endpoints.empty:
        all_ms = set(presence_by_ms.keys())

        def _support_sensitive(row: pd.Series) -> bool:
            key = (row['endpoint'], row['contrast'], row['composite_label'])
            return any(key not in presence_by_ms[ms] for ms in all_ms)

        primary_endpoints = primary_endpoints.copy()
        primary_endpoints['support_sensitive'] = primary_endpoints.apply(
            _support_sensitive, axis=1,
        )
    out['family_b_endpoints'] = primary_endpoints

    # Phase 5.2 amendment: co-primary raw-marker basis. Build raw-arcsinh
    # composites for each lineage from the config-defined channel-set, run
    # the same neighbor-minus-self pipeline, stamp the rows so they ride
    # alongside sigmoid in endpoint_summary.csv. Emits the basis-divergence
    # CSV (per-endpoint headline-overlap status) and basis-conflict CSV
    # (opposite-sign-same-endpoint subset). Audit script remains runnable
    # standalone; this duplicates the work so a normal pipeline run produces
    # the artifacts without the user having to remember the audit script.
    sys.path.insert(0, str(REPO_ROOT))
    from audit_family_b_raw_markers import (  # noqa: E402
        load_lineage_definitions_from_config,
        build_raw_lineage_columns,
        run_family_b_on_basis,
    )
    from src.config import Config  # noqa: E402

    config = Config(str(CONFIG_PATH))
    lineage_defs = load_lineage_definitions_from_config(config)
    raw_annotations = build_raw_lineage_columns(annotations, lineage_defs)
    raw_endpoints = run_family_b_on_basis(
        raw_annotations, min_support=tia.DEFAULT_MIN_SUPPORT,
    )
    if not raw_endpoints.empty:
        raw_endpoints = raw_endpoints.copy()
        raw_endpoints['normalization_mode'] = 'sham_reference_raw_marker_per_mouse'
    out['family_b_endpoints_raw_marker'] = raw_endpoints
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
    # Family C operates directly on raw markers with Sham-only per-mouse
    # percentile thresholds — independent of compute_continuous_memberships.
    # Stamp so every row carries its normalization provenance.
    if not endpoints.empty:
        endpoints['normalization_mode'] = 'sham_reference_raw_marker_per_mouse'
        endpoints['sham_percentile'] = percentile
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

    # Gate 6 seam-closure: removed p_proxy_from_g and q_proxy_* columns. At n=2
    # per group no real p-value exists, so FDR-adjusted proxies were cognitive
    # anchoring risk regardless of disclaimers. The audit they supported
    # (within-family vs pooled rank comparison) can be recovered by sorting
    # endpoint_summary by |hedges_g| directly.
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

    # Continuous-Sham percentile that drove compute_continuous_memberships,
    # stamped onto per_roi_sigmoid Family A rows so it appears in
    # endpoint_summary.csv alongside the raw-marker sweep percentile.
    continuous_sham_pct = _load_continuous_sham_percentile(scale_um)

    print("\n[Family A] Interface fractions + CLR + threshold sensitivity + normalization sensitivity")
    fa = run_family_a(annotations, continuous_sham_percentile=continuous_sham_pct)
    fa['interface_fractions'].to_parquet(OUTPUT_DIR / 'interface_fractions.parquet')
    fa['interface_clr'].to_parquet(OUTPUT_DIR / 'interface_clr.parquet')
    fa['interface_clr_no_none'].to_parquet(OUTPUT_DIR / 'interface_clr_no_none.parquet')
    fa['sensitivity_thresholds'].to_parquet(OUTPUT_DIR / 'sensitivity_thresholds.parquet')
    fa['normalization_sensitivity'].to_parquet(OUTPUT_DIR / 'interface_fractions_normalization_sensitivity.parquet')
    fa['family_a_endpoints_global_norm'].to_parquet(OUTPUT_DIR / 'family_a_endpoints_global_norm.parquet')
    fa['family_a_endpoints_norm_sweep'].to_parquet(OUTPUT_DIR / 'family_a_endpoints_norm_sweep.parquet')
    fa['family_a_global_thresholds'].to_parquet(OUTPUT_DIR / 'family_a_global_thresholds.parquet')
    # Magnitude-stratified normalization sign-reversal counts (Gate 6 critic feedback:
    # raw 18/48 overcounts compositionally-coupled near-zero flips)
    fa_eps = fa['family_a_endpoints']
    has_norm_data = fa_eps['hedges_g'].notna() & fa_eps['hedges_g_sham_ref'].notna()
    n_total_cmp = int(has_norm_data.sum())
    n_rev_all = int(fa_eps.loc[has_norm_data, 'normalization_sign_reverse'].sum())
    nontrivial = has_norm_data & (fa_eps['hedges_g'].abs() > 0.5)
    n_rev_nontrivial = int(fa_eps.loc[nontrivial, 'normalization_sign_reverse'].sum())
    n_total_nontrivial = int(nontrivial.sum())
    n_collapse = int(fa_eps.loc[has_norm_data, 'normalization_g_collapse'].sum())
    print(f"  Mouse-level fractions: {len(fa['interface_fractions'])} rows")
    print(f"  CLR endpoints (per-ROI norm): {len(fa['family_a_endpoints'])} rows")
    print(f"  Normalization sign-reverse (all): {n_rev_all}/{n_total_cmp}")
    print(f"  Normalization sign-reverse (|g|>0.5 only): {n_rev_nontrivial}/{n_total_nontrivial}")
    print(f"  Magnitude collapse (|g|>0.5 per-ROI -> |g|<20% under Sham-ref): {n_collapse}/{n_total_cmp}")
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
        fb.get('family_b_endpoints_raw_marker'),  # Phase 5.2: co-primary basis
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
        print(f"  Pathological-flag rows: {flagged} (NaN shrunk values)")
        print(f"  Insufficient-support rows: {insufficient} (NaN derived stats)")
        print(f"  Threshold-sensitive rows (sign reversal in sweep): {thr_sens}")

    # Phase 5.2 amendment artifacts: family_b_basis_divergence + conflict.
    # Computed against the two Family B basis tables we already produced; the
    # standalone audit script duplicates this for users who only run that
    # script, but the orchestrator emits them on every pipeline run so the
    # rule is "happens automatically" rather than "remember to invoke."
    sigmoid_eps = fb.get('family_b_endpoints', pd.DataFrame())
    raw_eps = fb.get('family_b_endpoints_raw_marker', pd.DataFrame())
    if not sigmoid_eps.empty and not raw_eps.empty:
        sys.path.insert(0, str(REPO_ROOT))
        from audit_family_b_raw_markers import (  # noqa: E402
            compute_basis_divergence, compute_basis_conflict,
        )
        divergence = compute_basis_divergence(sigmoid_eps, raw_eps)
        divergence.to_csv(OUTPUT_DIR / 'family_b_basis_divergence.csv', index=False)
        conflict = compute_basis_conflict(divergence)
        conflict.to_csv(OUTPUT_DIR / 'family_b_basis_conflict.csv', index=False)
        # Headline-overlap counts for terminal output
        if 'headline_overlap_status' in divergence.columns:
            counts = divergence['headline_overlap_status'].value_counts().to_dict()
            print(f"  Family B basis-divergence: {counts} ({len(conflict)} conflict rows)")

    # Combined Family B parquet across both bases (mirrors what the audit
    # script produces; useful for downstream consumers that prefer parquet
    # over CSV).
    if not sigmoid_eps.empty or not raw_eps.empty:
        sigmoid_tagged = sigmoid_eps.copy()
        raw_tagged = raw_eps.copy()
        if not sigmoid_tagged.empty:
            sigmoid_tagged['lineage_source'] = 'sham_reference_v2_continuous'
        if not raw_tagged.empty:
            raw_tagged['lineage_source'] = 'raw_marker_arcsinh'
        combined = pd.concat(
            [df for df in (sigmoid_tagged, raw_tagged) if not df.empty],
            ignore_index=True,
        )
        combined.to_parquet(OUTPUT_DIR / 'family_b_raw_marker_audit.parquet')

    print(f"\n✓ Outputs at {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
