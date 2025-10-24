"""
Module 2: Differential Abundance Analysis

Compares cell type abundances across kidney injury time course:
- Timepoints: Sham, D1, D3, D7 post-ischemia
- Regions: Cortex (M1) vs Medulla (M2)
- Statistical tests: Mann-Whitney U (n=2-3 per group)
- Effect sizes: Cohen's d

Input: Batch annotation results from results/biological_analysis/cell_type_annotations/
Output: Differential abundance statistics and summary tables
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

def load_batch_summary() -> dict:
    """Load batch annotation summary."""
    summary_file = Path('results/biological_analysis/cell_type_annotations/batch_annotation_summary.json')
    with open(summary_file, 'r') as f:
        return json.load(f)

def parse_roi_metadata(roi_id: str) -> dict:
    """
    Extract metadata from ROI ID.

    Format: IMC_241218_Alun_ROI_{timepoint}_{region}_{replicate}_{index}
    Examples:
      - IMC_241218_Alun_ROI_D1_M1_01_9
      - IMC_241218_Alun_ROI_Sam1_01_2 (Sham timepoint)
    """
    parts = roi_id.split('_')

    # Extract timepoint
    if 'Sam' in roi_id:
        timepoint = 'Sham'
        # Sam1, Sam2 format
        sam_part = [p for p in parts if 'Sam' in p][0]
        replicate = sam_part  # "Sam1" or "Sam2"
        region = None  # Sham samples don't have cortex/medulla designation
    elif 'Test' in roi_id:
        timepoint = 'Test'
        replicate = 'Test01'
        region = None
    else:
        # D1, D3, D7 format
        timepoint = [p for p in parts if p.startswith('D')][0]
        region = [p for p in parts if p.startswith('M')][0]
        replicate_idx = [i for i, p in enumerate(parts) if p.startswith('M')][0] + 1
        replicate = f"{timepoint}_{region}_{parts[replicate_idx]}"

    return {
        'roi_id': roi_id,
        'timepoint': timepoint,
        'region': region,
        'replicate': replicate
    }

def compute_abundances(batch_summary: dict) -> pd.DataFrame:
    """
    Compute cell type abundances for each ROI.

    Returns DataFrame with columns:
      - roi_id, timepoint, region, replicate
      - n_total, n_assigned, assignment_rate
      - One column per cell type (counts)
      - One column per cell type (proportions)
    """
    rows = []

    for roi_id, summary in batch_summary['roi_summaries'].items():
        metadata = parse_roi_metadata(roi_id)

        # Basic counts
        row = {
            'roi_id': roi_id,
            'timepoint': metadata['timepoint'],
            'region': metadata['region'],
            'replicate': metadata['replicate'],
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

def perform_differential_abundance(df: pd.DataFrame, cell_types: List[str]) -> pd.DataFrame:
    """
    Perform pairwise differential abundance tests across timepoints.

    Tests each cell type for differences between:
      - Sham vs D1
      - Sham vs D3
      - Sham vs D7
      - D1 vs D3
      - D3 vs D7

    Returns DataFrame with test results.
    """
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

        # Skip if column doesn't exist (e.g., unassigned)
        if prop_col not in df.columns:
            continue

        for tp1, tp2 in timepoint_pairs:
            # Get proportions for each timepoint
            group1 = df[df['timepoint'] == tp1][prop_col].values
            group2 = df[df['timepoint'] == tp2][prop_col].values

            # Skip if insufficient data
            if len(group1) == 0 or len(group2) == 0:
                continue

            # Compute statistics
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1) if len(group1) > 1 else 0, np.std(group2, ddof=1) if len(group2) > 1 else 0

            u_stat, p_value = mann_whitney_test(group1, group2)
            effect_size = cohens_d(group1, group2)

            # Fold change (mean2 / mean1)
            fold_change = mean2 / mean1 if mean1 > 0 else np.nan
            log2_fc = np.log2(fold_change) if not np.isnan(fold_change) and fold_change > 0 else np.nan

            results.append({
                'cell_type': cell_type,
                'comparison': f'{tp1}_vs_{tp2}',
                'timepoint_1': tp1,
                'timepoint_2': tp2,
                'n_1': len(group1),
                'n_2': len(group2),
                'mean_1': mean1,
                'mean_2': mean2,
                'std_1': std1,
                'std_2': std2,
                'fold_change': fold_change,
                'log2_fc': log2_fc,
                'u_statistic': u_stat,
                'p_value': p_value,
                'cohens_d': effect_size
            })

    return pd.DataFrame(results)

def perform_regional_analysis(df: pd.DataFrame, cell_types: List[str]) -> pd.DataFrame:
    """
    Compare cell type abundances between cortex (M1) and medulla (M2).

    Stratified by timepoint (D1, D3, D7 only - Sham has no region info).
    """
    results = []

    # Filter to injury timepoints only (D1, D3, D7)
    df_injury = df[df['timepoint'].isin(['D1', 'D3', 'D7'])].copy()

    for timepoint in ['D1', 'D3', 'D7']:
        df_tp = df_injury[df_injury['timepoint'] == timepoint]

        for cell_type in cell_types:
            prop_col = f'{cell_type}_prop'

            if prop_col not in df_tp.columns:
                continue

            # Get proportions for each region
            cortex = df_tp[df_tp['region'] == 'M1'][prop_col].values
            medulla = df_tp[df_tp['region'] == 'M2'][prop_col].values

            if len(cortex) == 0 or len(medulla) == 0:
                continue

            # Compute statistics
            mean_cortex, mean_medulla = np.mean(cortex), np.mean(medulla)
            std_cortex, std_medulla = np.std(cortex, ddof=1) if len(cortex) > 1 else 0, np.std(medulla, ddof=1) if len(medulla) > 1 else 0

            u_stat, p_value = mann_whitney_test(cortex, medulla)
            effect_size = cohens_d(cortex, medulla)

            fold_change = mean_medulla / mean_cortex if mean_cortex > 0 else np.nan

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
                'p_value': p_value,
                'cohens_d': effect_size
            })

    return pd.DataFrame(results)

def main():
    print("="*80)
    print("Differential Abundance Analysis - Kidney Injury Time Course")
    print("="*80)

    # Load batch annotation results
    print("\nLoading batch annotation summary...")
    batch_summary = load_batch_summary()

    print(f"✓ Loaded {batch_summary['n_rois_processed']} ROI annotations")

    # Compute abundances
    print("\nComputing cell type abundances per ROI...")
    abundance_df = compute_abundances(batch_summary)

    print(f"✓ Computed abundances for {len(abundance_df)} ROIs")
    print(f"  Timepoints: {sorted(abundance_df['timepoint'].unique())}")
    print(f"  Regions: {sorted([r for r in abundance_df['region'].unique() if r is not None])}")

    # Get all cell types (excluding unassigned)
    cell_type_cols = [c for c in abundance_df.columns if c.endswith('_count') and not c.startswith('unassigned')]
    cell_types = [c.replace('_count', '') for c in cell_type_cols]

    print(f"  Cell types: {len(cell_types)}")
    for ct in cell_types:
        total_count = abundance_df[f'{ct}_count'].sum()
        print(f"    - {ct}: {total_count} total")

    # Perform differential abundance analysis
    print("\n" + "─"*80)
    print("Temporal Differential Abundance (Timepoint Comparisons)")
    print("─"*80)

    temporal_results = perform_differential_abundance(abundance_df, cell_types)

    # Display significant findings
    print("\nTop Temporal Changes (by |Cohen's d|):")
    temporal_sorted = temporal_results.sort_values('cohens_d', key=abs, ascending=False)

    for idx, row in temporal_sorted.head(10).iterrows():
        direction = "↑" if row['mean_2'] > row['mean_1'] else "↓"
        print(f"\n  {row['cell_type']}")
        print(f"    {row['comparison']:20s} | {direction} {row['fold_change']:.2f}x | d={row['cohens_d']:.2f} | p={row['p_value']:.3f}")
        print(f"    {row['timepoint_1']:10s}: {row['mean_1']:.3%} ± {row['std_1']:.3%} (n={row['n_1']})")
        print(f"    {row['timepoint_2']:10s}: {row['mean_2']:.3%} ± {row['std_2']:.3%} (n={row['n_2']})")

    # Perform regional analysis
    print("\n" + "─"*80)
    print("Regional Differential Abundance (Cortex vs Medulla)")
    print("─"*80)

    regional_results = perform_regional_analysis(abundance_df, cell_types)

    if len(regional_results) > 0:
        print("\nRegional Differences (by |Cohen's d|):")
        regional_sorted = regional_results.sort_values('cohens_d', key=abs, ascending=False)

        for idx, row in regional_sorted.head(10).iterrows():
            direction = "Med↑" if row['mean_medulla'] > row['mean_cortex'] else "Ctx↑"
            print(f"\n  {row['cell_type']} @ {row['timepoint']}")
            print(f"    {direction} {row['fold_change']:.2f}x | d={row['cohens_d']:.2f} | p={row['p_value']:.3f}")
            print(f"    Cortex:  {row['mean_cortex']:.3%} ± {row['std_cortex']:.3%} (n={row['n_cortex']})")
            print(f"    Medulla: {row['mean_medulla']:.3%} ± {row['std_medulla']:.3%} (n={row['n_medulla']})")
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
