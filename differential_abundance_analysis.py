"""
Module 2: Differential Abundance Analysis

Compares cell type abundances across kidney injury time course:
- Timepoints: Sham, D1, D3, D7 post-ischemia
- Regions: Cortex vs Medulla (from metadata, NOT from M1/M2 filename convention)
- Unit of analysis: MOUSE (biological replicate), not ROI
  ROI-level proportions are averaged within each mouse before testing.
- Statistical tests: Mann-Whitney U on mouse-level means (n=2 per group)
- Effect sizes: Hedges' g (small-sample corrected Cohen's d) with bootstrap CIs
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

def load_metadata() -> pd.DataFrame:
    """Load experimental metadata with actual anatomical regions."""
    metadata_file = Path('data/241218_IMC_Alun/Metadata-Table 1.csv')
    metadata = pd.read_csv(metadata_file)
    metadata.columns = metadata.columns.str.strip()
    return metadata

def load_batch_summary() -> dict:
    """Load batch annotation summary."""
    summary_file = Path('results/biological_analysis/cell_type_annotations/batch_annotation_summary.json')
    with open(summary_file, 'r') as f:
        return json.load(f)

def parse_roi_metadata(roi_id: str, metadata_df: pd.DataFrame) -> dict:
    """
    Extract metadata from ROI ID using actual metadata CSV.

    Format: IMC_241218_Alun_ROI_{timepoint}_{mouse}_{replicate}_{index}
    Examples:
      - IMC_241218_Alun_ROI_D1_M1_01_9 (Mouse 1, could be Cortex OR Medulla)
      - IMC_241218_Alun_ROI_Sam1_01_2 (Sham timepoint)

    Note: M1/M2 in filename = Mouse 1/Mouse 2 (biological replicates), NOT regions!
    Actual anatomical region comes from metadata "Details" column.
    """
    # Remove 'roi_' prefix if present
    file_name = roi_id.replace('roi_', '')

    # Look up actual anatomical region from metadata
    metadata_row = metadata_df[metadata_df['File Name'] == file_name]

    if len(metadata_row) == 0:
        # Fallback parsing for ROIs not in metadata
        parts = roi_id.split('_')
        if 'Sam' in roi_id:
            timepoint = 'Sham'
            sam_part = [p for p in parts if 'Sam' in p][0]
            replicate = sam_part
            region = None
        elif 'Test' in roi_id:
            timepoint = 'Test'
            replicate = 'Test01'
            region = None
        else:
            timepoint = [p for p in parts if p.startswith('D')][0]
            mouse = [p for p in parts if p.startswith('M')][0]
            replicate_idx = [i for i, p in enumerate(parts) if p.startswith('M')][0] + 1
            replicate = f"{timepoint}_{mouse}_{parts[replicate_idx]}"
            region = None  # Unknown without metadata

        return {
            'roi_id': roi_id,
            'timepoint': timepoint,
            'region': region,
            'replicate': replicate
        }

    # Extract from metadata
    row = metadata_row.iloc[0]
    timepoint = 'Sham' if row['Injury Day'] == 0 else f"D{int(row['Injury Day'])}"
    region = row['Details'].strip() if pd.notna(row['Details']) else None
    mouse = row['Mouse']

    # Create replicate ID
    parts = roi_id.split('_')
    if 'Sam' in roi_id:
        replicate = mouse.replace('MS', 'Sam')  # MS1 -> Sam1, MS2 -> Sam2
    else:
        # Extract replicate number from filename
        replicate_num = [p for p in parts if p.isdigit() and len(p) == 2]
        if replicate_num:
            replicate = f"{timepoint}_{mouse}_{replicate_num[0]}"
        else:
            replicate = f"{timepoint}_{mouse}"

    return {
        'roi_id': roi_id,
        'timepoint': timepoint,
        'region': region,
        'replicate': replicate,
        'mouse': mouse
    }

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

        # Cell type counts and proportions
        cell_type_counts = summary['cell_type_counts']
        n_total = summary['n_superpixels']

        for cell_type, count in cell_type_counts.items():
            row[f'{cell_type}_count'] = count
            row[f'{cell_type}_prop'] = count / n_total if n_total > 0 else 0.0

        rows.append(row)

    df = pd.DataFrame(rows)

    # Filter out test ROIs
    df = df[df['timepoint'] != 'Test']

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

    agg_dict = {col: 'mean' for col in prop_cols}
    agg_dict.update({col: 'sum' for col in count_cols})
    agg_dict['n_total'] = 'sum'
    agg_dict['n_assigned'] = 'sum'
    agg_dict['assignment_rate'] = 'mean'
    agg_dict['roi_id'] = 'count'  # n_rois per mouse

    group_cols = ['timepoint', 'mouse']
    # Keep region if present (for regional analysis)
    if 'region' in df.columns:
        agg_dict['region'] = 'first'

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

    Reports: Hedges' g (small-sample corrected), bootstrap 95% CIs,
    Mann-Whitney p-values (n=2 vs n=2, mostly non-significant — that's honest),
    and BH FDR-corrected p-values across all comparisons.
    """
    # Aggregate to mouse level first
    mouse_df = aggregate_to_mouse_level(df)

    results = []

    timepoint_pairs = [
        ('Sham', 'D1'),
        ('Sham', 'D3'),
        ('Sham', 'D7'),
        ('D1', 'D3'),
        ('D3', 'D7'),
    ]

    for cell_type in cell_types:
        prop_col = f'{cell_type}_prop'

        if prop_col not in mouse_df.columns:
            continue

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

            results.append({
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
            })

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

def perform_regional_analysis(df: pd.DataFrame, cell_types: List[str]) -> pd.DataFrame:
    """
    Compare cell type abundances between Cortex and Medulla (actual anatomical regions).

    Stratified by timepoint (D1, D3, D7 only).
    Uses ROI-level data since region is nested within mouse×timepoint
    (each mouse contributes ROIs to both regions).

    Note: Regional comparisons are within-mouse, so pseudoreplication is less severe
    than temporal comparisons. Still, n is small (typically 1-2 ROIs per region per mouse).
    """
    results = []

    df_injury = df[df['timepoint'].isin(['D1', 'D3', 'D7'])].copy()

    for timepoint in ['D1', 'D3', 'D7']:
        df_tp = df_injury[df_injury['timepoint'] == timepoint]

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
                'n_cortex': len(cortex),
                'n_medulla': len(medulla),
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

    print(f"  Cell types: {len(cell_types)}")
    for ct in cell_types:
        total_count = abundance_df[f'{ct}_count'].sum()
        print(f"    - {ct}: {total_count} total")

    # Perform differential abundance analysis (mouse-level)
    print("\n" + "-"*80)
    print("Temporal Differential Abundance (Mouse-Level, FDR-Corrected)")
    print("-"*80)
    print("  Unit of analysis: mouse (ROI proportions averaged within each mouse)")

    temporal_results = perform_differential_abundance(abundance_df, cell_types)

    print(f"\n  {len(temporal_results)} comparisons, BH FDR applied")
    n_sig_fdr = temporal_results['significant_fdr'].sum() if 'significant_fdr' in temporal_results.columns else 0
    print(f"  Significant after FDR: {n_sig_fdr}")

    print("\nTop Temporal Changes (by |Hedges' g|):")
    temporal_sorted = temporal_results.sort_values('hedges_g', key=abs, ascending=False)

    for idx, row in temporal_sorted.head(10).iterrows():
        direction = "+" if row['mean_2'] > row['mean_1'] else "-"
        ci_str = f"[{row['ci_lower_95']:.2f}, {row['ci_upper_95']:.2f}]" if not np.isnan(row.get('ci_lower_95', np.nan)) else "[NA]"
        fdr_str = f"q={row.get('p_value_fdr', np.nan):.3f}" if not np.isnan(row.get('p_value_fdr', np.nan)) else "q=NA"
        print(f"\n  {row['cell_type']}")
        print(f"    {row['comparison']:20s} | {direction}{row['fold_change']:.2f}x | g={row['hedges_g']:.2f} 95%CI {ci_str} | {fdr_str}")
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
            ci_str = f"[{row['ci_lower_95']:.2f}, {row['ci_upper_95']:.2f}]" if not np.isnan(row.get('ci_lower_95', np.nan)) else "[NA]"
            fdr_str = f"q={row.get('p_value_fdr', np.nan):.3f}" if not np.isnan(row.get('p_value_fdr', np.nan)) else "q=NA"
            print(f"\n  {row['cell_type']} @ {row['timepoint']}")
            print(f"    {direction} {row['fold_change']:.2f}x | g={row['hedges_g']:.2f} 95%CI {ci_str} | {fdr_str}")
            print(f"    Cortex:  {row['mean_cortex']:.3%} +/- {row['std_cortex']:.3%} (n={row['n_cortex']})")
            print(f"    Medulla: {row['mean_medulla']:.3%} +/- {row['std_medulla']:.3%} (n={row['n_medulla']})")
    else:
        print("  No regional comparisons available")

    # Save results
    output_dir = Path('results/biological_analysis/differential_abundance')
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
