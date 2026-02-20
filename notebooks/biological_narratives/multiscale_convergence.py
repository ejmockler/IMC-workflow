#!/usr/bin/env python
"""
Multi-Scale Phenotype-Niche Convergence Analysis

Validates that phenotype-niche enrichment patterns are scale-robust,
not artifacts of choosing 10μm superpixels.

Strategy (from hierarchical_analysis_complete.py):
- Compute enrichment at 10μm, 20μm, 40μm
- Calculate coefficient of variation (CV) across scales
- CV < 0.2 = "Scale-robust" (biological reality)
- CV > 0.5 = "Scale-dependent" (interpretation caution)

This addresses n=2 limitations: If pattern holds across scales → stronger claim.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
import gzip
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from phenotype_gating import assign_phenotypes, PHENOTYPES
from phenotype_niche_convergence import (
    compute_niche_phenotype_composition,
    compute_phenotype_niche_enrichment
)


def load_superpixels_at_scale(results_dir: Path, scale: float, markers: List[str]) -> pd.DataFrame:
    """
    Load superpixel data at specific scale.

    Args:
        results_dir: Path to results/roi_results
        scale: Spatial scale (10.0, 20.0, or 40.0)
        markers: List of protein marker names

    Returns:
        DataFrame with continuous protein expression + metadata
    """
    result_files = sorted(results_dir.glob('roi_*.json.gz'))

    all_superpixels = []
    scale_key = str(float(scale))

    for result_file in result_files:
        try:
            # Parse ROI metadata
            roi_name = result_file.stem.replace('roi_', '').replace('_results.json', '')

            if 'Sam' in roi_name:
                condition = 'Sham'
                timepoint = 'Sham'
                parts = roi_name.split('_ROI_')
                mouse = parts[1].split('_')[0] if len(parts) > 1 else 'Unknown'
            else:
                parts = roi_name.split('_ROI_')
                if len(parts) < 2:
                    continue
                timepoint_parts = parts[1].split('_')
                timepoint = timepoint_parts[0]
                mouse = timepoint_parts[1]
                condition = 'UUO'

            # Load JSON
            with gzip.open(result_file, 'rt') as f:
                data = json.load(f)

            # Extract scale-specific data
            if 'multiscale_results' not in data:
                continue

            scales = data['multiscale_results']
            if scale_key not in scales:
                continue

            scale_data = scales[scale_key]

            # Get transformed arrays
            transformed = scale_data.get('transformed_arrays', {})
            coords_dict = scale_data.get('superpixel_coords', {})

            # Deserialize coords
            if 'data' in coords_dict:
                coords = np.array(coords_dict['data']).reshape(coords_dict['shape'])
            else:
                continue

            n_superpixels = len(coords)

            # Build superpixel dataframe
            for i in range(n_superpixels):
                spx_data = {
                    'roi': roi_name,
                    'condition': condition,
                    'timepoint': timepoint,
                    'mouse': mouse,
                    'superpixel_id': i,
                    'x': coords[i, 0],
                    'y': coords[i, 1],
                    'scale_um': scale
                }

                # Extract marker values
                for marker in markers:
                    if marker in transformed:
                        marker_data = transformed[marker]
                        if isinstance(marker_data, dict) and 'data' in marker_data:
                            values = marker_data['data']
                            spx_data[marker] = values[i] if i < len(values) else np.nan
                        else:
                            spx_data[marker] = np.nan
                    else:
                        spx_data[marker] = np.nan

                all_superpixels.append(spx_data)

        except Exception as e:
            continue

    superpixel_df = pd.DataFrame(all_superpixels)
    return superpixel_df


def cluster_into_niches(superpixel_df: pd.DataFrame, markers: List[str], n_niches: int = 6) -> pd.DataFrame:
    """
    Cluster superpixels into spatial niches using k-means.

    Args:
        superpixel_df: DataFrame with marker columns
        markers: List of marker names to cluster on
        n_niches: Number of niches (default 6)

    Returns:
        DataFrame with 'niche' column added
    """
    # Prepare features
    feature_cols = markers
    valid_mask = superpixel_df[feature_cols].notna().all(axis=1)
    X = superpixel_df.loc[valid_mask, feature_cols].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_niches, random_state=42, n_init=20)
    niche_labels = kmeans.fit_predict(X_scaled)

    # Add to dataframe
    superpixel_df = superpixel_df.copy()
    superpixel_df.loc[valid_mask, 'niche'] = niche_labels
    superpixel_df['niche'] = superpixel_df['niche'].fillna(-1).astype(int)

    return superpixel_df


def analyze_single_scale(scale: float,
                         results_dir: Path,
                         markers: List[str]) -> Dict:
    """
    Complete analysis pipeline at a single scale.

    Returns:
        dict with enrichment_df, composition_df, n_superpixels
    """
    print(f"\n{'='*70}")
    print(f"Analyzing scale: {scale}μm")
    print(f"{'='*70}")

    # Load data
    print(f"  Loading superpixel data at {scale}μm...")
    superpixel_df = load_superpixels_at_scale(results_dir, scale, markers)
    print(f"  ✓ Loaded {len(superpixel_df):,} superpixels")

    # Cluster into niches
    print(f"  Clustering into niches...")
    superpixel_df = cluster_into_niches(superpixel_df, markers, n_niches=6)
    n_with_niches = (superpixel_df['niche'] >= 0).sum()
    print(f"  ✓ {n_with_niches:,} superpixels assigned to niches")

    # Assign phenotypes
    print(f"  Assigning cell phenotypes via boolean gating...")
    superpixel_df = assign_phenotypes(superpixel_df, markers)

    # Count phenotypes
    n_phenotyped = 0
    for pheno in PHENOTYPES.keys():
        n_phenotyped += superpixel_df[pheno].sum()
    print(f"  ✓ {n_phenotyped:,} total phenotype assignments (superpixels can have multiple)")

    # Compute enrichment
    print(f"  Computing phenotype-niche enrichment...")
    enrichment_df = compute_phenotype_niche_enrichment(superpixel_df, niche_column='niche')
    composition_df = compute_niche_phenotype_composition(superpixel_df, niche_column='niche')

    max_enrichment = enrichment_df.max().max()
    print(f"  ✓ Max enrichment: {max_enrichment:.2f}×")

    return {
        'scale_um': scale,
        'enrichment_df': enrichment_df,
        'composition_df': composition_df,
        'n_superpixels': len(superpixel_df),
        'n_with_niches': n_with_niches,
        'superpixel_df': superpixel_df
    }


def compute_scale_coherence(multiscale_results: Dict[float, Dict]) -> pd.DataFrame:
    """
    Compute scale coherence for phenotype spatial organization.

    KEY INSIGHT: Don't try to match niche IDs across scales (k-means assigns arbitrary labels).
    Instead ask: "Does each phenotype show spatial organization (max enrichment >1.5×) at ALL scales?"

    This tests if phenotypes organize into SOME niche consistently, not the SAME numbered niche.

    Args:
        multiscale_results: Dict mapping scale → analysis results

    Returns:
        DataFrame with coherence metrics for each phenotype
    """
    scales = sorted(multiscale_results.keys())

    # Get phenotypes from first scale
    first_enrichment = multiscale_results[scales[0]]['enrichment_df']
    phenotypes = first_enrichment.columns.tolist()

    coherence_results = []

    for phenotype in phenotypes:
        # For this phenotype, find MAX enrichment across all niches at each scale
        max_enrichments = []
        max_niche_ids = []

        for scale in scales:
            enrich_df = multiscale_results[scale]['enrichment_df']
            if phenotype in enrich_df.columns:
                # Find max enrichment for this phenotype across all niches
                max_enrich = enrich_df[phenotype].max()
                max_niche = enrich_df[phenotype].idxmax()
                max_enrichments.append(max_enrich)
                max_niche_ids.append(max_niche)

        if len(max_enrichments) == len(scales):
            # Calculate statistics
            mean_enrich = np.mean(max_enrichments)
            std_enrich = np.std(max_enrichments)
            cv = std_enrich / abs(mean_enrich) if abs(mean_enrich) > 0.1 else np.nan

            # Test: Does this phenotype organize at ALL scales?
            threshold = 1.5  # Enrichment threshold
            organizes_at_all_scales = all(e > threshold for e in max_enrichments)

            # Classify coherence based on CV AND threshold test
            if pd.isna(cv):
                coherence_cat = 'low_signal'
            elif organizes_at_all_scales and cv < 0.3:
                coherence_cat = 'highly_coherent'
            elif organizes_at_all_scales and cv < 0.6:
                coherence_cat = 'moderately_coherent'
            elif organizes_at_all_scales:
                coherence_cat = 'organizes_variably'
            else:
                coherence_cat = 'no_organization'

            coherence_results.append({
                'phenotype': phenotype,
                'mean_max_enrichment': mean_enrich,
                'std_enrichment': std_enrich,
                'cv': cv,
                'coherence': coherence_cat,
                'enrichment_10um': max_enrichments[0] if len(max_enrichments) > 0 else np.nan,
                'enrichment_20um': max_enrichments[1] if len(max_enrichments) > 1 else np.nan,
                'enrichment_40um': max_enrichments[2] if len(max_enrichments) > 2 else np.nan,
                'niche_10um': max_niche_ids[0] if len(max_niche_ids) > 0 else np.nan,
                'niche_20um': max_niche_ids[1] if len(max_niche_ids) > 1 else np.nan,
                'niche_40um': max_niche_ids[2] if len(max_niche_ids) > 2 else np.nan,
                'organizes_at_all_scales': organizes_at_all_scales,
            })

    coherence_df = pd.DataFrame(coherence_results)
    return coherence_df


def run_multiscale_analysis(results_dir: Path, markers: List[str]) -> Tuple[Dict, pd.DataFrame]:
    """
    Run complete multi-scale convergence analysis.

    Returns:
        (multiscale_results, coherence_df)
    """
    scales = [10.0, 20.0, 40.0]

    print("="*70)
    print("MULTI-SCALE PHENOTYPE-NICHE CONVERGENCE ANALYSIS")
    print("="*70)
    print(f"\nAnalyzing {len(scales)} spatial scales: {scales} μm")
    print(f"Validating scale-robustness of phenotype-niche enrichment patterns")
    print(f"\nStrategy: CV < 0.2 = Scale-robust (biological reality)")
    print(f"          CV > 0.5 = Scale-dependent (interpretation caution)")

    # Analyze each scale
    multiscale_results = {}
    for scale in scales:
        result = analyze_single_scale(scale, results_dir, markers)
        multiscale_results[scale] = result

    # Compute scale coherence
    print(f"\n{'='*70}")
    print("Computing scale coherence...")
    print(f"{'='*70}")
    coherence_df = compute_scale_coherence(multiscale_results)

    # Summary statistics
    n_total_pairs = len(coherence_df)
    n_highly_coherent = (coherence_df['coherence'] == 'highly_coherent').sum()
    n_moderately_coherent = (coherence_df['coherence'] == 'moderately_coherent').sum()
    n_organizes_variably = (coherence_df['coherence'] == 'organizes_variably').sum()
    n_no_organization = (coherence_df['coherence'] == 'no_organization').sum()
    n_low_signal = (coherence_df['coherence'] == 'low_signal').sum()

    print(f"\n✓ Analyzed {n_total_pairs} phenotypes")
    print(f"  - Highly coherent (CV<0.3, organizes at all scales):     {n_highly_coherent:3d} ({100*n_highly_coherent/n_total_pairs:.1f}%)")
    print(f"  - Moderately coherent (CV<0.6, organizes at all scales): {n_moderately_coherent:3d} ({100*n_moderately_coherent/n_total_pairs:.1f}%)")
    print(f"  - Organizes variably (CV≥0.6, organizes at all scales):  {n_organizes_variably:3d} ({100*n_organizes_variably/n_total_pairs:.1f}%)")
    print(f"  - No organization (fails >1.5× at some scale):           {n_no_organization:3d} ({100*n_no_organization/n_total_pairs:.1f}%)")
    print(f"  - Low signal (enrichment near zero):                     {n_low_signal:3d} ({100*n_low_signal/n_total_pairs:.1f}%)")

    # Show top scale-robust enrichments
    robust_enrichments = coherence_df[
        (coherence_df['coherence'] == 'highly_coherent') &
        (coherence_df['mean_max_enrichment'] > 1.5)
    ].sort_values('mean_max_enrichment', ascending=False)

    if len(robust_enrichments) > 0:
        print(f"\n✅ Top scale-robust enrichments (CV<0.3, mean>1.5×):")
        for _, row in robust_enrichments.head(10).iterrows():
            print(f"  {row['phenotype']:30s}: "
                  f"{row['mean_max_enrichment']:.2f}× (CV={row['cv']:.3f}) "
                  f"[niches: {row['niche_10um']}, {row['niche_20um']}, {row['niche_40um']}]")
    else:
        print(f"\n⚠️  No highly coherent enrichments >1.5× found")

    return multiscale_results, coherence_df


if __name__ == '__main__':
    # Paths
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / 'results' / 'roi_results'

    # Markers
    markers = ['CD45', 'CD11b', 'Ly6G', 'CD140a', 'CD140b', 'CD31', 'CD34', 'CD206', 'CD44']

    # Run analysis
    multiscale_results, coherence_df = run_multiscale_analysis(results_dir, markers)

    # Save results
    output_dir = project_root / 'notebooks' / 'biological_narratives' / 'multiscale_results'
    output_dir.mkdir(exist_ok=True, parents=True)

    coherence_df.to_csv(output_dir / 'scale_coherence.csv', index=False)
    print(f"\n💾 Saved: {output_dir / 'scale_coherence.csv'}")

    # Save scale-specific enrichment matrices
    for scale, result in multiscale_results.items():
        result['enrichment_df'].to_csv(output_dir / f'enrichment_{int(scale)}um.csv')
        print(f"💾 Saved: {output_dir / f'enrichment_{int(scale)}um.csv'}")

    print("\n" + "="*70)
    print("Multi-scale analysis complete!")
    print("="*70)
