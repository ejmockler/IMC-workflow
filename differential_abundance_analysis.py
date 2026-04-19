"""
Module 2: Differential Abundance Analysis

Compares cell type abundances across kidney injury time course:
- Timepoints: Sham, D1, D3, D7 post-ischemia
- Regions: Cortex vs Medulla (from metadata, NOT from M1/M2 filename convention)
- Unit of analysis: MOUSE (biological replicate), not ROI
  ROI-level proportions are averaged within each mouse before testing.
- Statistical tests: Mann-Whitney U on mouse-level means (n=2 per group)
- Effect sizes: Hedges' g (small-sample corrected Cohen's d) with bootstrap range (NOT a coverage-bearing CI at n=2; only ~9 unique resampled values exist per group)
- Multiple testing: Benjamini-Hochberg FDR across all pairwise comparisons

Input: Batch annotation results from results/biological_analysis/cell_type_annotations/
Output: Differential abundance statistics and summary tables
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple

from scipy.stats.mstats import gmean

from src.utils.metadata import parse_roi_metadata as _parse_canonical
from src.utils.paths import get_paths

_PATHS = get_paths()


def load_metadata() -> pd.DataFrame:
    """Load experimental metadata with actual anatomical regions."""
    metadata = pd.read_csv(_PATHS.metadata_csv)
    metadata.columns = metadata.columns.str.strip()
    return metadata

def load_batch_summary() -> dict:
    """Load batch annotation summary."""
    summary_file = _PATHS.annotations_dir / 'batch_annotation_summary.json'
    with open(summary_file, 'r') as f:
        return json.load(f)

def parse_roi_metadata(roi_id: str, metadata_df: pd.DataFrame = None) -> dict:
    """Delegate to canonical metadata parser in src.utils.metadata."""
    return _parse_canonical(roi_id)

def compute_abundances(batch_summary: dict, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cell type abundances for each ROI.

    Returns DataFrame with columns:
      - roi_id, timepoint, region, replicate, mouse
      - n_total, n_assigned, assignment_rate
      - One column per cell type (counts)
      - One column per cell type (proportions)
    """
    rows = []

    for roi_id, summary in batch_summary['roi_summaries'].items():
        metadata = parse_roi_metadata(roi_id, metadata_df)

        # Basic counts
        row = {
            'roi_id': roi_id,
            'timepoint': metadata['timepoint'],
            'region': metadata['region'],
            'replicate': metadata['replicate'],
            'mouse': metadata.get('mouse', None),
            'n_total': summary['n_superpixels'],
            'n_assigned': summary['n_assigned'],
            'assignment_rate': summary['assignment_rate']
        }

        # Cell type counts and proportions (discrete gating)
        cell_type_counts = summary['cell_type_counts']
        n_total = summary['n_superpixels']

        for cell_type, count in cell_type_counts.items():
            row[f'{cell_type}_count'] = count
            row[f'{cell_type}_prop'] = count / n_total if n_total > 0 else 0.0

        # Continuous lineage membership scores (if available)
        membership = summary.get('membership', {})
        lineage_means = membership.get('lineage_means', {})
        for lineage_name, mean_score in lineage_means.items():
            row[f'lineage_{lineage_name}_mean'] = mean_score

        # Subtype proportions from continuous decomposition
        subtype_counts = membership.get('subtype_counts', {})
        for subtype_name, count in subtype_counts.items():
            row[f'subtype_{subtype_name}_count'] = count
            row[f'subtype_{subtype_name}_prop'] = count / n_total if n_total > 0 else 0.0

        # Multi-lineage interface fraction
        n_mixed = membership.get('n_mixed', 0)
        row['mixed_fraction'] = n_mixed / n_total if n_total > 0 else 0.0

        rows.append(row)

    df = pd.DataFrame(rows)

    # Fill NaN for cell types absent in some ROIs (zero count, not missing data)
    count_cols = [c for c in df.columns if c.endswith('_count')]
    prop_cols = [c for c in df.columns if c.endswith('_prop')]
    df[count_cols] = df[count_cols].fillna(0)
    df[prop_cols] = df[prop_cols].fillna(0.0)

    # Filter out test ROIs
    df = df[df['timepoint'] != 'Test']

    return df


def add_clr_columns(df: pd.DataFrame, cell_types: List[str]) -> pd.DataFrame:
    """
    Add centered log-ratio (CLR) transformed columns for compositional awareness.

    Cell type proportions are compositional data (they share the denominator,
    and ~79% is unassigned). CLR transforms break the spurious negative
    correlation induced by the shared denominator.

    CLR(x_i) = log(x_i / geometric_mean(x))

    Uses a pseudocount of 1e-6 to handle zero proportions.
    Includes 'unassigned' in the composition vector since it's a real component.
    """
    prop_cols = [f'{ct}_prop' for ct in cell_types if f'{ct}_prop' in df.columns]
    # Include unassigned as part of the composition
    if 'unassigned_prop' in df.columns and 'unassigned_prop' not in prop_cols:
        prop_cols.append('unassigned_prop')

    if not prop_cols:
        return df

    eps = 1e-6
    props = df[prop_cols].values + eps  # Pseudocount for zeros
    geo_mean = gmean(props, axis=1, keepdims=True)
    clr_values = np.log(props / geo_mean)

    for i, col in enumerate(prop_cols):
        clr_col = col.replace('_prop', '_clr')
        df[clr_col] = clr_values[:, i]

    return df

def mann_whitney_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """
    Perform Mann-Whitney U test.

    Returns (U-statistic, p-value)
    """
    if len(group1) < 2 or len(group2) < 2:
        return np.nan, np.nan

    try:
        statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        return statistic, pvalue
    except Exception as e:
        return np.nan, np.nan

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Cohen's d = (mean1 - mean2) / pooled_std
    """
    if len(group1) < 2 or len(group2) < 2:
        return np.nan

    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (mean1 - mean2) / pooled_std


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Hedges' g: small-sample corrected Cohen's d.

    Applies correction factor J = 1 - 3/(4*(n1+n2-2) - 1)
    Critical for n=2 where Cohen's d overestimates effect size.
    """
    d = cohens_d(group1, group2)
    if np.isnan(d):
        return np.nan
    n = len(group1) + len(group2)
    df = n - 2
    if df <= 0:
        return np.nan
    correction = 1 - (3 / (4 * df - 1))
    return d * correction


def bootstrap_effect_size_ci(group1: np.ndarray, group2: np.ndarray,
                              n_bootstrap: int = 10000,
                              ci: float = 0.95,
                              random_state: int = 42) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for Hedges' g.

    Uses percentile method (BCa requires larger samples than n=2).
    With n=2, CIs will be wide — that's the honest answer.
    """
    if len(group1) < 2 or len(group2) < 2:
        return np.nan, np.nan

    rng = np.random.RandomState(random_state)
    boot_gs = []

    for _ in range(n_bootstrap):
        b1 = rng.choice(group1, size=len(group1), replace=True)
        b2 = rng.choice(group2, size=len(group2), replace=True)
        g = hedges_g(b1, b2)
        if not np.isnan(g):
            boot_gs.append(g)

    if len(boot_gs) < 100:
        return np.nan, np.nan

    alpha = (1 - ci) / 2
    return np.percentile(boot_gs, 100 * alpha), np.percentile(boot_gs, 100 * (1 - alpha))

def aggregate_to_mouse_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ROI-level proportions to mouse-level means.

    The biological replicate is the mouse (n=2 per timepoint), not the ROI
    (n=3 per mouse). Averaging ROIs within each mouse prevents pseudoreplication.
    """
    prop_cols = [c for c in df.columns if c.endswith('_prop')]
    count_cols = [c for c in df.columns if c.endswith('_count')]
    lineage_mean_cols = [c for c in df.columns if c.startswith('lineage_') and c.endswith('_mean')]

    agg_dict = {col: 'mean' for col in prop_cols}
    agg_dict.update({col: 'sum' for col in count_cols})
    agg_dict.update({col: 'mean' for col in lineage_mean_cols})
    agg_dict['n_total'] = 'sum'
    agg_dict['n_assigned'] = 'sum'
    agg_dict['assignment_rate'] = 'mean'
    if 'mixed_fraction' in df.columns:
        agg_dict['mixed_fraction'] = 'mean'
    agg_dict['roi_id'] = 'count'  # n_rois per mouse

    group_cols = ['timepoint', 'mouse']
    # Note: region is NOT included here — temporal analysis averages across regions
    # within each mouse. Regional analysis uses its own groupby in perform_regional_analysis().

    mouse_df = df.groupby(group_cols).agg(agg_dict).reset_index()
    mouse_df = mouse_df.rename(columns={'roi_id': 'n_rois'})

    return mouse_df


def perform_differential_abundance(df: pd.DataFrame, cell_types: List[str]) -> pd.DataFrame:
    """
    Perform pairwise differential abundance tests across timepoints.

    CRITICAL: Operates on mouse-level means (n=2 per timepoint), not ROI-level data.
    ROI proportions are averaged within each mouse to avoid pseudoreplication.

    Tests each cell type for differences between:
      - Sham vs D1, Sham vs D3, Sham vs D7, D1 vs D3, D3 vs D7

    Reports: Hedges' g (small-sample corrected), bootstrap range (NOT 95% CI at n=2),
    Mann-Whitney p-values (n=2 vs n=2, mostly non-significant — that's honest),
    and BH FDR-corrected p-values across all comparisons.
    """
    # Aggregate to mouse level first, then apply CLR
    # CLR must be applied AFTER aggregation because mean(CLR(x)) ≠ CLR(mean(x))
    mouse_df = aggregate_to_mouse_level(df)
    mouse_df = add_clr_columns(mouse_df, cell_types)

    results = []

    timepoint_pairs = [
        ('Sham', 'D1'),
        ('Sham', 'D3'),
        ('Sham', 'D7'),
        ('D1', 'D3'),
        ('D1', 'D7'),
        ('D3', 'D7'),
    ]

    for cell_type in cell_types:
        prop_col = f'{cell_type}_prop'
        clr_col = f'{cell_type}_clr'

        if prop_col not in mouse_df.columns:
            continue

        # Skip cell types with zero total counts across all mice
        if mouse_df[prop_col].sum() == 0:
            continue

        has_clr = clr_col in mouse_df.columns

        for tp1, tp2 in timepoint_pairs:
            group1 = mouse_df[mouse_df['timepoint'] == tp1][prop_col].values
            group2 = mouse_df[mouse_df['timepoint'] == tp2][prop_col].values

            if len(group1) == 0 or len(group2) == 0:
                continue

            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1 = np.std(group1, ddof=1) if len(group1) > 1 else 0
            std2 = np.std(group2, ddof=1) if len(group2) > 1 else 0

            u_stat, p_value = mann_whitney_test(group1, group2)
            g = hedges_g(group1, group2)
            ci_lower, ci_upper = bootstrap_effect_size_ci(group1, group2)

            # Fold change with pseudocount for stability
            eps = 1e-6
            fold_change = (mean2 + eps) / (mean1 + eps)
            log2_fc = np.log2(fold_change) if fold_change > 0 else np.nan

            row = {
                'cell_type': cell_type,
                'comparison': f'{tp1}_vs_{tp2}',
                'timepoint_1': tp1,
                'timepoint_2': tp2,
                'n_mice_1': len(group1),
                'n_mice_2': len(group2),
                'mean_1': mean1,
                'mean_2': mean2,
                'std_1': std1,
                'std_2': std2,
                'fold_change': fold_change,
                'log2_fc': log2_fc,
                'u_statistic': u_stat,
                'p_value_raw': p_value,
                'hedges_g': g,
                'ci_lower_95': ci_lower,
                'ci_upper_95': ci_upper,
            }

            # CLR-transformed effect size (compositional-aware)
            if has_clr:
                clr1 = mouse_df[mouse_df['timepoint'] == tp1][clr_col].values
                clr2 = mouse_df[mouse_df['timepoint'] == tp2][clr_col].values
                row['hedges_g_clr'] = hedges_g(clr1, clr2)

            results.append(row)

    results_df = pd.DataFrame(results)

    # Apply BH FDR correction across all comparisons
    if len(results_df) > 0:
        valid_mask = results_df['p_value_raw'].notna()
        if valid_mask.sum() > 0:
            reject, pvals_corrected, _, _ = multipletests(
                results_df.loc[valid_mask, 'p_value_raw'],
                method='fdr_bh'
            )
            results_df.loc[valid_mask, 'p_value_fdr'] = pvals_corrected
            results_df.loc[valid_mask, 'significant_fdr'] = reject
        else:
            results_df['p_value_fdr'] = np.nan
            results_df['significant_fdr'] = False

    return results_df

def perform_differential_abundance_continuous(
    df: pd.DataFrame, score_cols: List[str], feature_names: List[str]
) -> pd.DataFrame:
    """
    Perform pairwise DA tests on continuous membership scores (e.g., lineage means).

    Same statistical framework as discrete DA (Hedges' g, Mann-Whitney, FDR),
    but operates on continuous mean scores per ROI rather than discrete proportions.
    """
    mouse_df = aggregate_to_mouse_level(df)

    results = []
    timepoint_pairs = [
        ('Sham', 'D1'), ('Sham', 'D3'), ('Sham', 'D7'),
        ('D1', 'D3'), ('D1', 'D7'), ('D3', 'D7'),
    ]

    for score_col, feature_name in zip(score_cols, feature_names):
        if score_col not in mouse_df.columns:
            continue
        if mouse_df[score_col].sum() == 0:
            continue

        for tp1, tp2 in timepoint_pairs:
            group1 = mouse_df[mouse_df['timepoint'] == tp1][score_col].values
            group2 = mouse_df[mouse_df['timepoint'] == tp2][score_col].values

            if len(group1) == 0 or len(group2) == 0:
                continue

            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1 = np.std(group1, ddof=1) if len(group1) > 1 else 0
            std2 = np.std(group2, ddof=1) if len(group2) > 1 else 0
            u_stat, p_value = mann_whitney_test(group1, group2)
            g = hedges_g(group1, group2)
            ci_lower, ci_upper = bootstrap_effect_size_ci(group1, group2)

            results.append({
                'cell_type': f'lineage:{feature_name}',
                'comparison': f'{tp1}_vs_{tp2}',
                'timepoint_1': tp1,
                'timepoint_2': tp2,
                'n_mice_1': len(group1),
                'n_mice_2': len(group2),
                'mean_1': mean1,
                'mean_2': mean2,
                'std_1': std1,
                'std_2': std2,
                'fold_change': (mean2 + 1e-6) / (mean1 + 1e-6),
                'log2_fc': np.log2((mean2 + 1e-6) / (mean1 + 1e-6)),
                'u_statistic': u_stat,
                'p_value_raw': p_value,
                'hedges_g': g,
                'ci_lower_95': ci_lower,
                'ci_upper_95': ci_upper,
            })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        valid_mask = results_df['p_value_raw'].notna()
        if valid_mask.sum() > 0:
            reject, pvals_corrected, _, _ = multipletests(
                results_df.loc[valid_mask, 'p_value_raw'], method='fdr_bh'
            )
            results_df.loc[valid_mask, 'p_value_fdr'] = pvals_corrected
            results_df.loc[valid_mask, 'significant_fdr'] = reject

    return results_df


def perform_regional_analysis(df: pd.DataFrame, cell_types: List[str]) -> pd.DataFrame:
    """
    Compare cell type abundances between Cortex and Medulla (actual anatomical regions).

    Stratified by timepoint (D1, D3, D7 only).
    Aggregates to mouse-level means per region.

    Note: Cortex and Medulla from the same mouse are paired observations.
    Mann-Whitney treats them as independent, which is technically incorrect.
    With n=2 mice per timepoint, neither Mann-Whitney nor Wilcoxon signed-rank
    can reach significance (minimum p > 0.05). Effect sizes (Hedges' g) remain
    the primary inferential quantities. The paired structure means that
    between-mouse variance is conflated with between-region variance.
    """
    results = []

    df_injury = df[df['timepoint'].isin(['D1', 'D3', 'D7'])].copy()

    if 'region' not in df_injury.columns:
        return pd.DataFrame()

    # Aggregate ROIs to mouse-level means within each region
    prop_cols = [c for c in df_injury.columns if c.endswith('_prop')]
    agg_dict = {col: 'mean' for col in prop_cols}
    agg_dict['roi_id'] = 'count'

    mouse_region_df = df_injury.groupby(
        ['timepoint', 'mouse', 'region']
    ).agg(agg_dict).reset_index()
    mouse_region_df = mouse_region_df.rename(columns={'roi_id': 'n_rois'})

    for timepoint in ['D1', 'D3', 'D7']:
        df_tp = mouse_region_df[mouse_region_df['timepoint'] == timepoint]

        for cell_type in cell_types:
            prop_col = f'{cell_type}_prop'

            if prop_col not in df_tp.columns:
                continue

            cortex = df_tp[df_tp['region'] == 'Cortex'][prop_col].values
            medulla = df_tp[df_tp['region'] == 'Medulla'][prop_col].values

            if len(cortex) == 0 or len(medulla) == 0:
                continue

            mean_cortex, mean_medulla = np.mean(cortex), np.mean(medulla)
            std_cortex = np.std(cortex, ddof=1) if len(cortex) > 1 else 0
            std_medulla = np.std(medulla, ddof=1) if len(medulla) > 1 else 0

            u_stat, p_value = mann_whitney_test(cortex, medulla)
            g = hedges_g(cortex, medulla)
            ci_lower, ci_upper = bootstrap_effect_size_ci(cortex, medulla)

            eps = 1e-6
            fold_change = (mean_medulla + eps) / (mean_cortex + eps)

            results.append({
                'cell_type': cell_type,
                'timepoint': timepoint,
                'n_mice_cortex': len(cortex),
                'n_mice_medulla': len(medulla),
                'mean_cortex': mean_cortex,
                'mean_medulla': mean_medulla,
                'std_cortex': std_cortex,
                'std_medulla': std_medulla,
                'fold_change': fold_change,
                'u_statistic': u_stat,
                'p_value_raw': p_value,
                'hedges_g': g,
                'ci_lower_95': ci_lower,
                'ci_upper_95': ci_upper,
            })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        valid_mask = results_df['p_value_raw'].notna()
        if valid_mask.sum() > 0:
            reject, pvals_corrected, _, _ = multipletests(
                results_df.loc[valid_mask, 'p_value_raw'],
                method='fdr_bh'
            )
            results_df.loc[valid_mask, 'p_value_fdr'] = pvals_corrected
            results_df.loc[valid_mask, 'significant_fdr'] = reject

    return results_df

def main():
    print("="*80)
    print("Differential Abundance Analysis - Kidney Injury Time Course")
    print("="*80)

    # Load metadata with actual anatomical regions
    print("\nLoading experimental metadata...")
    metadata_df = load_metadata()
    print(f"✓ Loaded metadata for {len(metadata_df)} ROIs")

    # Load batch annotation results
    print("\nLoading batch annotation summary...")
    batch_summary = load_batch_summary()

    print(f"✓ Loaded {batch_summary['n_rois_processed']} ROI annotations")

    # Compute abundances
    print("\nComputing cell type abundances per ROI...")
    abundance_df = compute_abundances(batch_summary, metadata_df)

    print(f"✓ Computed abundances for {len(abundance_df)} ROIs")
    print(f"  Timepoints: {sorted(abundance_df['timepoint'].unique())}")
    print(f"  Regions: {sorted([r for r in abundance_df['region'].unique() if r is not None])}")
    print(f"  Mice: {sorted([m for m in abundance_df['mouse'].unique() if m is not None])}")

    # Get all cell types (excluding unassigned)
    cell_type_cols = [c for c in abundance_df.columns if c.endswith('_count') and not c.startswith('unassigned')]
    cell_types = [c.replace('_count', '') for c in cell_type_cols]

    # Note: CLR transform applied inside perform_differential_abundance()
    # AFTER mouse-level aggregation, because mean(CLR(x)) ≠ CLR(mean(x))

    print(f"  Cell types: {len(cell_types)}")
    for ct in cell_types:
        total_count = abundance_df[f'{ct}_count'].sum()
        print(f"    - {ct}: {total_count} total")

    # Lineage-level continuous analysis (if available)
    lineage_cols = [c for c in abundance_df.columns if c.startswith('lineage_') and c.endswith('_mean')]
    lineage_names = [c.replace('lineage_', '').replace('_mean', '') for c in lineage_cols]
    if lineage_names:
        print(f"  Lineage scores: {lineage_names}")

    # Perform differential abundance analysis (mouse-level)
    print("\n" + "-"*80)
    print("Temporal Differential Abundance (Mouse-Level, FDR-Corrected)")
    print("-"*80)
    print("  Unit of analysis: mouse (ROI proportions averaged within each mouse)")

    temporal_results = perform_differential_abundance(abundance_df, cell_types)

    # Also run DA on continuous lineage scores
    if lineage_names:
        lineage_temporal = perform_differential_abundance_continuous(
            abundance_df, lineage_cols, lineage_names
        )
        if len(lineage_temporal) > 0:
            temporal_results = pd.concat([temporal_results, lineage_temporal], ignore_index=True)

    print(f"\n  {len(temporal_results)} comparisons, BH FDR applied")
    n_sig_fdr = temporal_results['significant_fdr'].sum() if 'significant_fdr' in temporal_results.columns else 0
    print(f"  Significant after FDR: {n_sig_fdr}")

    print("\nTop Temporal Changes (by |Hedges' g|):")
    temporal_sorted = temporal_results.sort_values('hedges_g', key=abs, ascending=False)

    for idx, row in temporal_sorted.head(10).iterrows():
        direction = "+" if row['mean_2'] > row['mean_1'] else "-"
        ci_str = f"[{row['ci_lower_95']:.2f}, {row['ci_upper_95']:.2f}]" if not np.isnan(row.get('ci_lower_95', np.nan)) else "[NA]"  # bootstrap range, NOT a coverage-bearing 95% CI at n=2
        fdr_str = f"q={row.get('p_value_fdr', np.nan):.3f}" if not np.isnan(row.get('p_value_fdr', np.nan)) else "q=NA"
        print(f"\n  {row['cell_type']}")
        print(f"    {row['comparison']:20s} | {direction}{row['fold_change']:.2f}x | g={row['hedges_g']:.2f} bootstrap-range {ci_str} | {fdr_str}")
        print(f"    {row['timepoint_1']:10s}: {row['mean_1']:.3%} +/- {row['std_1']:.3%} (n={row['n_mice_1']} mice)")
        print(f"    {row['timepoint_2']:10s}: {row['mean_2']:.3%} +/- {row['std_2']:.3%} (n={row['n_mice_2']} mice)")

    # Perform regional analysis
    print("\n" + "-"*80)
    print("Regional Differential Abundance (Cortex vs Medulla)")
    print("-"*80)

    regional_results = perform_regional_analysis(abundance_df, cell_types)

    if len(regional_results) > 0:
        print("\nRegional Differences (by |Hedges' g|):")
        regional_sorted = regional_results.sort_values('hedges_g', key=abs, ascending=False)

        for idx, row in regional_sorted.head(10).iterrows():
            direction = "Med+" if row['mean_medulla'] > row['mean_cortex'] else "Ctx+"
            ci_str = f"[{row['ci_lower_95']:.2f}, {row['ci_upper_95']:.2f}]" if not np.isnan(row.get('ci_lower_95', np.nan)) else "[NA]"  # bootstrap range, NOT a coverage-bearing 95% CI at n=2
            fdr_str = f"q={row.get('p_value_fdr', np.nan):.3f}" if not np.isnan(row.get('p_value_fdr', np.nan)) else "q=NA"
            print(f"\n  {row['cell_type']} @ {row['timepoint']}")
            print(f"    {direction} {row['fold_change']:.2f}x | g={row['hedges_g']:.2f} bootstrap-range {ci_str} | {fdr_str}")
            print(f"    Cortex:  {row['mean_cortex']:.3%} +/- {row['std_cortex']:.3%} (n={row['n_mice_cortex']} mice)")
            print(f"    Medulla: {row['mean_medulla']:.3%} +/- {row['std_medulla']:.3%} (n={row['n_mice_medulla']} mice)")
    else:
        print("  No regional comparisons available")

    # Save results
    output_dir = _PATHS.differential_abundance_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    abundance_df.to_csv(output_dir / 'roi_abundances.csv', index=False)
    temporal_results.to_csv(output_dir / 'temporal_differential_abundance.csv', index=False)

    if len(regional_results) > 0:
        regional_results.to_csv(output_dir / 'regional_differential_abundance.csv', index=False)

    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}")
    print(f"\n✓ ROI abundances: {output_dir / 'roi_abundances.csv'}")
    print(f"✓ Temporal results: {output_dir / 'temporal_differential_abundance.csv'}")
    if len(regional_results) > 0:
        print(f"✓ Regional results: {output_dir / 'regional_differential_abundance.csv'}")
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()
