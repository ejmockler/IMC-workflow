"""
Module 3: Spatial Neighborhood Analysis

Analyzes spatial relationships between cell types:
- For each cell type, identifies enriched/depleted neighbor cell types
- Computes neighborhood composition using k-nearest neighbors
- Statistical significance via permutation tests
- Stratified by timepoint and region

Input: Cell type annotations from results/biological_analysis/cell_type_annotations/
Output: Neighborhood enrichment matrices and statistical tests
"""

import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
from scipy import stats
from statsmodels.stats.multitest import multipletests
from collections import Counter

from src.utils.metadata import parse_roi_metadata as _parse_roi_metadata_canonical
from src.utils.paths import get_paths

_PATHS = get_paths()


def load_roi_annotations(roi_id: str) -> pd.DataFrame:
    """Load cell type annotations for a single ROI."""
    annotation_file = _PATHS.annotations_dir / f'{roi_id}_cell_types.parquet'

    if not annotation_file.exists():
        return None

    return pd.read_parquet(annotation_file)

def compute_knn_neighborhoods(coords: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Compute k-nearest neighbors for each point.

    Returns: (n_points, k) array of neighbor indices
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    # k+1 because query includes self as nearest neighbor
    _, indices = tree.query(coords, k=k + 1)
    # Exclude self (column 0)
    return indices[:, 1:]

def compute_neighborhood_composition(
    cell_types: np.ndarray,
    knn_indices: np.ndarray,
    focal_cell_type: str
) -> Dict[str, float]:
    """
    Compute neighborhood composition for a focal cell type.

    For all cells of the focal type, what fraction of their neighbors
    are of each other cell type?

    Returns: dict mapping neighbor_cell_type -> proportion
    """
    # Find all cells of the focal type
    focal_mask = cell_types == focal_cell_type
    focal_indices = np.where(focal_mask)[0]

    if len(focal_indices) == 0:
        return {}

    # Get all neighbors of focal cells
    neighbor_indices = knn_indices[focal_indices].flatten()
    neighbor_cell_types = cell_types[neighbor_indices]

    # Count neighbor cell types
    neighbor_counts = Counter(neighbor_cell_types)
    total_neighbors = len(neighbor_indices)

    # Compute proportions
    neighbor_props = {
        ct: count / total_neighbors
        for ct, count in neighbor_counts.items()
    }

    return neighbor_props

def compute_expected_composition(cell_types: np.ndarray) -> Dict[str, float]:
    """
    Compute expected (random) composition based on global cell type frequencies.
    """
    cell_type_counts = Counter(cell_types)
    total = len(cell_types)

    return {ct: count / total for ct, count in cell_type_counts.items()}

def permutation_test(
    observed_prop: float,
    cell_types: np.ndarray,
    knn_indices: np.ndarray,
    focal_cell_type: str,
    neighbor_cell_type: str,
    n_permutations: int = 1000,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Test if observed neighbor proportion is significantly different from expected.

    Uses global label permutation (standard null for neighborhood enrichment —
    same approach as histoCAT/squidpy). Tests whether co-localization exceeds
    chance given cell type frequencies.

    Returns: (enrichment_score, p_value)
    enrichment_score = observed / expected
    p_value = fraction of permutations >= observed (Phipson & Smyth corrected)
    """
    # Get expected proportion
    expected_props = compute_expected_composition(cell_types)
    expected_prop = expected_props.get(neighbor_cell_type, 0.0)

    if expected_prop == 0:
        return np.nan, np.nan

    enrichment = observed_prop / expected_prop

    # Permutation test
    focal_mask = cell_types == focal_cell_type
    focal_indices = np.where(focal_mask)[0]

    if len(focal_indices) == 0:
        return enrichment, np.nan

    # Deterministic RNG for reproducibility
    rng = np.random.default_rng(random_state)

    # Generate null distribution
    null_props = []

    for _ in range(n_permutations):
        # Shuffle cell type labels (preserves total counts, randomizes positions)
        permuted_types = rng.permutation(cell_types)

        # Compute neighbor composition under permutation
        neighbor_indices = knn_indices[focal_indices].flatten()
        neighbor_types = permuted_types[neighbor_indices]

        neighbor_counts = Counter(neighbor_types)
        neighbor_prop = neighbor_counts.get(neighbor_cell_type, 0) / len(neighbor_types)

        null_props.append(neighbor_prop)

    # Two-sided p-value with Phipson & Smyth (2010) pseudocount correction
    # Prevents p=0 which implies infinite evidence against the null
    null_props = np.array(null_props)
    n_extreme = np.sum(np.abs(null_props - expected_prop) >= np.abs(observed_prop - expected_prop))
    p_value = (n_extreme + 1) / (len(null_props) + 1)

    return enrichment, p_value

def analyze_roi_neighborhoods(
    roi_id: str,
    k_neighbors: int = 10,
    n_permutations: int = 1000
) -> pd.DataFrame:
    """
    Analyze spatial neighborhoods for a single ROI.

    Returns DataFrame with:
      - focal_cell_type
      - neighbor_cell_type
      - observed_proportion
      - expected_proportion
      - enrichment_score
      - p_value
    """
    # Load annotations
    annotations = load_roi_annotations(roi_id)

    if annotations is None or len(annotations) == 0:
        return None

    # Extract data
    coords = annotations[['x', 'y']].values
    cell_types = annotations['cell_type'].values

    # Filter out low-quality cells
    valid_mask = annotations['confidence'] > 0.0
    coords = coords[valid_mask]
    cell_types = cell_types[valid_mask]

    if len(coords) < k_neighbors + 1:
        return None

    # Compute k-nearest neighbors
    knn_indices = compute_knn_neighborhoods(coords, k=k_neighbors)

    # Get expected composition
    expected_comp = compute_expected_composition(cell_types)

    # Get unique cell types (excluding unassigned for focal type)
    focal_cell_types = [ct for ct in np.unique(cell_types) if ct != 'unassigned']
    neighbor_cell_types = np.unique(cell_types)

    results = []

    # Deterministic seed per ROI (hashlib is session-independent, unlike hash())
    roi_seed = int(hashlib.sha256(roi_id.encode()).hexdigest(), 16) % (2**31)

    for fi, focal_ct in enumerate(focal_cell_types):
        # Get observed neighborhood composition
        observed_comp = compute_neighborhood_composition(cell_types, knn_indices, focal_ct)

        for ni, neighbor_ct in enumerate(neighbor_cell_types):
            observed_prop = observed_comp.get(neighbor_ct, 0.0)
            expected_prop = expected_comp.get(neighbor_ct, 0.0)

            if expected_prop == 0:
                continue

            # Unique seed per ROI × focal × neighbor combination
            pair_seed = roi_seed + fi * 1000 + ni

            # Compute enrichment and significance
            enrichment, p_value = permutation_test(
                observed_prop,
                cell_types,
                knn_indices,
                focal_ct,
                neighbor_ct,
                n_permutations=n_permutations,
                random_state=pair_seed
            )

            results.append({
                'roi_id': roi_id,
                'focal_cell_type': focal_ct,
                'neighbor_cell_type': neighbor_ct,
                'observed_proportion': observed_prop,
                'expected_proportion': expected_prop,
                'enrichment_score': enrichment,
                'log2_enrichment': np.clip(np.log2(enrichment), -10, 10) if enrichment > 0 else np.nan,
                'p_value': p_value,
                'n_focal_cells': int(np.sum(cell_types == focal_ct))
            })

    results_df = pd.DataFrame(results)

    # Apply BH FDR within this ROI across all focal×neighbor pairs
    if len(results_df) > 0:
        valid_mask = results_df['p_value'].notna()
        if valid_mask.sum() > 0:
            reject, pvals_corrected, _, _ = multipletests(
                results_df.loc[valid_mask, 'p_value'],
                method='fdr_bh'
            )
            results_df.loc[valid_mask, 'p_value_fdr'] = pvals_corrected
            results_df.loc[valid_mask, 'significant_fdr'] = reject
        else:
            results_df['p_value_fdr'] = np.nan
            results_df['significant_fdr'] = False

    return results_df

def analyze_roi_neighborhoods_continuous(
    roi_id: str,
    k_neighbors: int = 10,
) -> Dict[str, Any]:
    """
    Compute continuous neighborhood composition using lineage membership scores.

    Instead of discrete neighbor counts, computes the mean lineage score of
    each superpixel's k-nearest neighbors. This captures interface biology:
    a superpixel surrounded by immune-endothelial mixed neighbors gets high
    mean scores for both lineages.

    Returns dict with:
    - neighborhood_lineage_means: DataFrame with per-superpixel mean neighbor lineage scores
    - roi_summary: aggregate statistics
    """
    annotations = load_roi_annotations(roi_id)
    if annotations is None or len(annotations) == 0:
        return None

    coords = annotations[['x', 'y']].values
    lineage_cols = [c for c in annotations.columns if c.startswith('lineage_')]
    activation_cols = [c for c in annotations.columns if c.startswith('activation_')]

    if not lineage_cols:
        return None

    if len(coords) < k_neighbors + 1:
        return None

    knn_indices = compute_knn_neighborhoods(coords, k=k_neighbors)

    # For each superpixel, compute mean lineage scores of its neighbors
    result_cols = {}
    for col in lineage_cols + activation_cols:
        scores = annotations[col].values
        # Mean neighbor score for each superpixel
        neighbor_scores = scores[knn_indices]  # (n_superpixels, k)
        result_cols[f'neighbor_{col}'] = neighbor_scores.mean(axis=1)

    neighborhood_df = pd.DataFrame(result_cols)
    neighborhood_df['superpixel_id'] = annotations['superpixel_id'].values
    neighborhood_df['x'] = coords[:, 0]
    neighborhood_df['y'] = coords[:, 1]

    # Add the superpixel's own scores for comparison
    for col in lineage_cols + activation_cols:
        neighborhood_df[f'self_{col}'] = annotations[col].values

    if 'composite_label' in annotations.columns:
        neighborhood_df['composite_label'] = annotations['composite_label'].values

    # ROI-level summary: mean neighbor lineage scores stratified by composite label
    summary = {}
    if 'composite_label' in annotations.columns:
        for label in annotations['composite_label'].unique():
            mask = annotations['composite_label'].values == label
            if mask.sum() == 0:
                continue
            label_summary = {}
            for col in lineage_cols:
                neighbor_col = f'neighbor_{col}'
                label_summary[f'mean_{neighbor_col}'] = float(neighborhood_df.loc[mask, neighbor_col].mean())
                label_summary[f'mean_self_{col}'] = float(neighborhood_df.loc[mask, f'self_{col}'].mean())
            label_summary['n_superpixels'] = int(mask.sum())
            summary[label] = label_summary

    return {
        'neighborhood_df': neighborhood_df,
        'roi_summary': summary,
        'roi_id': roi_id,
    }


def parse_roi_metadata(roi_id: str) -> dict:
    """
    Extract metadata from ROI ID using the canonical metadata parser.

    Delegates to src.utils.metadata.parse_roi_metadata.
    Returns dict with keys: timepoint, region (and others).
    """
    return _parse_roi_metadata_canonical(roi_id)

def aggregate_across_rois(
    roi_results: List[pd.DataFrame],
    groupby: str = 'timepoint'
) -> pd.DataFrame:
    """
    Aggregate neighborhood enrichment across ROIs.

    Groups by timepoint or region and computes mean enrichment scores.
    """
    if not roi_results:
        return pd.DataFrame()

    # Combine all ROI results
    combined = pd.concat(roi_results, ignore_index=True)

    # Add metadata
    combined['timepoint'] = combined['roi_id'].apply(lambda x: parse_roi_metadata(x)['timepoint'])
    combined['region'] = combined['roi_id'].apply(lambda x: parse_roi_metadata(x)['region'])

    # Filter out test ROIs
    combined = combined[combined['timepoint'] != 'Test']

    # Group and aggregate
    # Use weighted means for continuous columns, weighted by n_focal_cells,
    # so ROIs with more focal cells contribute proportionally more
    group_cols = ['focal_cell_type', 'neighbor_cell_type', groupby]

    weighted_cols = ['observed_proportion', 'expected_proportion', 'enrichment_score', 'log2_enrichment']

    def _weighted_agg(group_df):
        weights = group_df['n_focal_cells']
        result = {}
        for col in weighted_cols:
            if weights.sum() > 0:
                result[col] = np.average(group_df[col], weights=weights)
            else:
                result[col] = group_df[col].mean()
        result['p_value'] = np.mean(group_df['p_value'] < 0.05)  # Fraction raw significant (for comparison)
        result['n_focal_cells'] = group_df['n_focal_cells'].sum()
        result['roi_id'] = len(group_df)
        if 'significant_fdr' in group_df.columns:
            result['significant_fdr'] = np.mean(group_df['significant_fdr'])  # Fraction FDR-significant
        return pd.Series(result)

    aggregated = combined.groupby(group_cols).apply(_weighted_agg, include_groups=False).reset_index()

    aggregated.rename(columns={
        'p_value': 'fraction_significant_raw',
        'significant_fdr': 'fraction_significant_fdr',
        'roi_id': 'n_rois'
    }, inplace=True)

    return aggregated

def main():
    print("="*80)
    print("Spatial Neighborhood Analysis - Cell Type Interactions")
    print("="*80)

    # Parameters
    k_neighbors = 10
    n_permutations = 1000

    print(f"\nParameters:")
    print(f"  k-nearest neighbors: {k_neighbors}")
    print(f"  Permutations: {n_permutations}")

    # Get all ROI annotation files
    annotation_dir = _PATHS.annotations_dir
    annotation_files = sorted(annotation_dir.glob('roi_*_cell_types.parquet'))

    print(f"\n✓ Found {len(annotation_files)} ROI annotations")

    # Analyze each ROI
    roi_results = []

    print("\nAnalyzing ROI neighborhoods...")
    for i, annotation_file in enumerate(annotation_files, 1):
        roi_id = annotation_file.stem.replace('_cell_types', '')

        print(f"  [{i}/{len(annotation_files)}] {roi_id}...", end='', flush=True)

        try:
            results = analyze_roi_neighborhoods(roi_id, k_neighbors, n_permutations)

            if results is not None and len(results) > 0:
                roi_results.append(results)
                print(f" ✓ ({len(results)} interactions)")
            else:
                print(" (skipped - insufficient data)")

        except Exception as e:
            print(f" ❌ Error: {e}")

    print(f"\n✓ Analyzed {len(roi_results)} ROIs successfully")

    # Aggregate by timepoint
    print("\n" + "─"*80)
    print("Temporal Neighborhood Patterns")
    print("─"*80)

    temporal_agg = aggregate_across_rois(roi_results, groupby='timepoint')

    # Show top enrichments by timepoint
    for timepoint in ['Sham', 'D1', 'D3', 'D7']:
        tp_data = temporal_agg[temporal_agg['timepoint'] == timepoint]

        if len(tp_data) == 0:
            continue

        print(f"\n{timepoint} - Top Spatial Enrichments:")

        # Filter for strong enrichments (>1.5x, FDR-significant in >50% of ROIs)
        fdr_col = 'fraction_significant_fdr' if 'fraction_significant_fdr' in tp_data.columns else 'fraction_significant_raw'
        enriched = tp_data[
            (tp_data['enrichment_score'] > 1.5) &
            (tp_data[fdr_col] > 0.5) &
            (tp_data['neighbor_cell_type'] != 'unassigned')
        ].sort_values('enrichment_score', ascending=False)

        for idx, row in enriched.head(5).iterrows():
            fdr_frac = row.get('fraction_significant_fdr', row.get('fraction_significant_raw', 0))
            print(f"  {row['focal_cell_type']:35s} <- {row['neighbor_cell_type']:35s}")
            print(f"    {row['enrichment_score']:.2f}x enriched | {fdr_frac:.0%} ROIs FDR-sig | {row['n_rois']} ROIs")

        if len(enriched) == 0:
            print("  (No strong enrichments detected)")

    # Aggregate by region
    print("\n" + "─"*80)
    print("Regional Neighborhood Patterns (Cortex vs Medulla)")
    print("─"*80)

    regional_agg = aggregate_across_rois(roi_results, groupby='region')

    for region in ['Cortex', 'Medulla']:
        reg_data = regional_agg[regional_agg['region'] == region]

        if len(reg_data) == 0:
            continue

        print(f"\n{region} - Top Spatial Enrichments:")

        fdr_col = 'fraction_significant_fdr' if 'fraction_significant_fdr' in reg_data.columns else 'fraction_significant_raw'
        enriched = reg_data[
            (reg_data['enrichment_score'] > 1.5) &
            (reg_data[fdr_col] > 0.5) &
            (reg_data['neighbor_cell_type'] != 'unassigned')
        ].sort_values('enrichment_score', ascending=False)

        for idx, row in enriched.head(5).iterrows():
            fdr_frac = row.get('fraction_significant_fdr', row.get('fraction_significant_raw', 0))
            print(f"  {row['focal_cell_type']:35s} <- {row['neighbor_cell_type']:35s}")
            print(f"    {row['enrichment_score']:.2f}x enriched | {fdr_frac:.0%} ROIs FDR-sig | {row['n_rois']} ROIs")

        if len(enriched) == 0:
            print("  (No strong enrichments detected)")

    # --- Continuous neighborhood analysis ---
    print("\n" + "─"*80)
    print("Continuous Neighborhood Analysis (Lineage Scores)")
    print("─"*80)

    continuous_summaries = []
    for i, annotation_file in enumerate(annotation_files, 1):
        roi_id = annotation_file.stem.replace('_cell_types', '')
        try:
            cont_result = analyze_roi_neighborhoods_continuous(roi_id, k_neighbors)
            if cont_result is not None:
                # Add metadata to summary
                metadata = parse_roi_metadata(roi_id)
                for label, stats in cont_result['roi_summary'].items():
                    stats['roi_id'] = roi_id
                    stats['composite_label'] = label
                    stats['timepoint'] = metadata['timepoint']
                    stats['region'] = metadata['region']
                    continuous_summaries.append(stats)
        except Exception as e:
            pass  # Already reported in discrete analysis

    continuous_df = pd.DataFrame(continuous_summaries) if continuous_summaries else pd.DataFrame()
    if len(continuous_df) > 0:
        continuous_df = continuous_df[continuous_df['timepoint'] != 'Test']
        n_labels = continuous_df['composite_label'].nunique()
        print(f"\n✓ {len(continuous_df)} composite_label × ROI combinations analyzed")
        print(f"  {n_labels} unique composite labels")

    # Save results
    output_dir = _PATHS.spatial_neighborhoods_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-ROI results
    if roi_results:
        all_roi_results = pd.concat(roi_results, ignore_index=True)
        all_roi_results.to_csv(output_dir / 'roi_neighborhood_enrichments.csv', index=False)

    # Save aggregated results
    temporal_agg.to_csv(output_dir / 'temporal_neighborhood_enrichments.csv', index=False)
    regional_agg.to_csv(output_dir / 'regional_neighborhood_enrichments.csv', index=False)

    if len(continuous_df) > 0:
        continuous_df.to_csv(output_dir / 'continuous_neighborhood_summary.csv', index=False)

    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}")
    print(f"\n✓ Per-ROI results: {output_dir / 'roi_neighborhood_enrichments.csv'}")
    print(f"✓ Temporal aggregates: {output_dir / 'temporal_neighborhood_enrichments.csv'}")
    print(f"✓ Regional aggregates: {output_dir / 'regional_neighborhood_enrichments.csv'}")
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()
