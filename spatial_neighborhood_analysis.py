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
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial import distance_matrix
from scipy import stats
from collections import Counter

def load_roi_annotations(roi_id: str) -> pd.DataFrame:
    """Load cell type annotations for a single ROI."""
    annotation_file = Path('results/biological_analysis/cell_type_annotations') / f'{roi_id}_cell_types.parquet'

    if not annotation_file.exists():
        return None

    return pd.read_parquet(annotation_file)

def compute_knn_neighborhoods(coords: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Compute k-nearest neighbors for each point.

    Returns: (n_points, k) array of neighbor indices
    """
    # Compute pairwise distances
    dists = distance_matrix(coords, coords)

    # For each point, get k+1 nearest neighbors (including self)
    # Then exclude self (first neighbor)
    knn_indices = np.argsort(dists, axis=1)[:, 1:k+1]

    return knn_indices

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
    n_permutations: int = 1000
) -> Tuple[float, float]:
    """
    Test if observed neighbor proportion is significantly different from expected.

    Returns: (enrichment_score, p_value)
    enrichment_score = observed / expected
    p_value = fraction of permutations >= observed
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

    # Generate null distribution
    null_props = []

    for _ in range(n_permutations):
        # Shuffle cell type labels
        permuted_types = np.random.permutation(cell_types)

        # Compute neighbor composition under permutation
        neighbor_indices = knn_indices[focal_indices].flatten()
        neighbor_types = permuted_types[neighbor_indices]

        neighbor_counts = Counter(neighbor_types)
        neighbor_prop = neighbor_counts.get(neighbor_cell_type, 0) / len(neighbor_types)

        null_props.append(neighbor_prop)

    # Two-sided p-value
    null_props = np.array(null_props)
    p_value = np.mean(np.abs(null_props - expected_prop) >= np.abs(observed_prop - expected_prop))

    return enrichment, p_value

def analyze_roi_neighborhoods(
    roi_id: str,
    k_neighbors: int = 10,
    n_permutations: int = 500
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

    for focal_ct in focal_cell_types:
        # Get observed neighborhood composition
        observed_comp = compute_neighborhood_composition(cell_types, knn_indices, focal_ct)

        for neighbor_ct in neighbor_cell_types:
            observed_prop = observed_comp.get(neighbor_ct, 0.0)
            expected_prop = expected_comp.get(neighbor_ct, 0.0)

            if expected_prop == 0:
                continue

            # Compute enrichment and significance
            enrichment, p_value = permutation_test(
                observed_prop,
                cell_types,
                knn_indices,
                focal_ct,
                neighbor_ct,
                n_permutations=n_permutations
            )

            results.append({
                'roi_id': roi_id,
                'focal_cell_type': focal_ct,
                'neighbor_cell_type': neighbor_ct,
                'observed_proportion': observed_prop,
                'expected_proportion': expected_prop,
                'enrichment_score': enrichment,
                'log2_enrichment': np.log2(enrichment) if enrichment > 0 else np.nan,
                'p_value': p_value,
                'n_focal_cells': int(np.sum(cell_types == focal_ct))
            })

    return pd.DataFrame(results)

def parse_roi_metadata(roi_id: str) -> dict:
    """Extract metadata from ROI ID."""
    parts = roi_id.split('_')

    if 'Sam' in roi_id:
        timepoint = 'Sham'
        region = None
    elif 'Test' in roi_id:
        timepoint = 'Test'
        region = None
    else:
        timepoint = [p for p in parts if p.startswith('D')][0]
        region = [p for p in parts if p.startswith('M')][0]

    return {'timepoint': timepoint, 'region': region}

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
    group_cols = ['focal_cell_type', 'neighbor_cell_type', groupby]

    aggregated = combined.groupby(group_cols).agg({
        'observed_proportion': 'mean',
        'expected_proportion': 'mean',
        'enrichment_score': 'mean',
        'log2_enrichment': 'mean',
        'p_value': lambda x: np.mean(x < 0.05),  # Fraction significant
        'n_focal_cells': 'sum',
        'roi_id': 'count'
    }).reset_index()

    aggregated.rename(columns={
        'p_value': 'fraction_significant',
        'roi_id': 'n_rois'
    }, inplace=True)

    return aggregated

def main():
    print("="*80)
    print("Spatial Neighborhood Analysis - Cell Type Interactions")
    print("="*80)

    # Parameters
    k_neighbors = 10
    n_permutations = 500

    print(f"\nParameters:")
    print(f"  k-nearest neighbors: {k_neighbors}")
    print(f"  Permutations: {n_permutations}")

    # Get all ROI annotation files
    annotation_dir = Path('results/biological_analysis/cell_type_annotations')
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

        # Filter for strong enrichments (>1.5x, fraction_significant >0.5)
        enriched = tp_data[
            (tp_data['enrichment_score'] > 1.5) &
            (tp_data['fraction_significant'] > 0.5) &
            (tp_data['neighbor_cell_type'] != 'unassigned')
        ].sort_values('enrichment_score', ascending=False)

        for idx, row in enriched.head(5).iterrows():
            print(f"  {row['focal_cell_type']:35s} ← {row['neighbor_cell_type']:35s}")
            print(f"    {row['enrichment_score']:.2f}x enriched | {row['fraction_significant']:.0%} ROIs significant | {row['n_rois']} ROIs")

        if len(enriched) == 0:
            print("  (No strong enrichments detected)")

    # Aggregate by region
    print("\n" + "─"*80)
    print("Regional Neighborhood Patterns")
    print("─"*80)

    regional_agg = aggregate_across_rois(roi_results, groupby='region')

    for region in ['M1', 'M2']:
        reg_data = regional_agg[regional_agg['region'] == region]

        if len(reg_data) == 0:
            continue

        region_name = 'Cortex (M1)' if region == 'M1' else 'Medulla (M2)'
        print(f"\n{region_name} - Top Spatial Enrichments:")

        enriched = reg_data[
            (reg_data['enrichment_score'] > 1.5) &
            (reg_data['fraction_significant'] > 0.5) &
            (reg_data['neighbor_cell_type'] != 'unassigned')
        ].sort_values('enrichment_score', ascending=False)

        for idx, row in enriched.head(5).iterrows():
            print(f"  {row['focal_cell_type']:35s} ← {row['neighbor_cell_type']:35s}")
            print(f"    {row['enrichment_score']:.2f}x enriched | {row['fraction_significant']:.0%} ROIs significant | {row['n_rois']} ROIs")

        if len(enriched) == 0:
            print("  (No strong enrichments detected)")

    # Save results
    output_dir = Path('results/biological_analysis/spatial_neighborhoods')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-ROI results
    if roi_results:
        all_roi_results = pd.concat(roi_results, ignore_index=True)
        all_roi_results.to_csv(output_dir / 'roi_neighborhood_enrichments.csv', index=False)

    # Save aggregated results
    temporal_agg.to_csv(output_dir / 'temporal_neighborhood_enrichments.csv', index=False)
    regional_agg.to_csv(output_dir / 'regional_neighborhood_enrichments.csv', index=False)

    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}")
    print(f"\n✓ Per-ROI results: {output_dir / 'roi_neighborhood_enrichments.csv'}")
    print(f"✓ Temporal aggregates: {output_dir / 'temporal_neighborhood_enrichments.csv'}")
    print(f"✓ Regional aggregates: {output_dir / 'regional_neighborhood_enrichments.csv'}")
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()
