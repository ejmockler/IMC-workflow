"""
Analyze spatial autocorrelation of markers to identify best segmentation channels.

Tests hypothesis: Markers with high spatial autocorrelation create more stable
superpixels, leading to more stable clustering.
"""

import json
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr


def load_roi_results(roi_file: Path) -> dict:
    """Load ROI results from gzipped JSON."""
    with gzip.open(roi_file, 'rt') as f:
        return json.load(f)


def deserialize_numpy_arrays(data):
    """Recursively deserialize numpy arrays from JSON representation."""
    if isinstance(data, dict):
        if data.get('__numpy_array__'):
            arr = np.array(data['data'], dtype=data['dtype'])
            return arr.reshape(data['shape'])
        else:
            return {k: deserialize_numpy_arrays(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deserialize_numpy_arrays(item) for item in data]
    else:
        return data


def compute_morans_i(values: np.ndarray, coords: np.ndarray, distance_threshold: float = 50.0) -> float:
    """
    Compute Moran's I spatial autocorrelation statistic.

    Moran's I ranges from -1 (perfect dispersion) to +1 (perfect clustering).
    Values near 0 indicate random spatial pattern.

    Args:
        values: Expression values [n_points]
        coords: Spatial coordinates [n_points, 2]
        distance_threshold: Distance threshold for neighbors (microns)

    Returns:
        Moran's I statistic
    """
    n = len(values)
    if n < 10:
        return np.nan

    # Compute pairwise distances
    distances = squareform(pdist(coords))

    # Define spatial weights (binary: neighbor if within threshold)
    W = (distances <= distance_threshold) & (distances > 0)
    W = W.astype(float)

    # Sum of weights
    W_sum = W.sum()
    if W_sum == 0:
        return np.nan

    # Standardize values
    mean_val = values.mean()
    deviations = values - mean_val
    variance = (deviations ** 2).sum() / n

    if variance == 0:
        return np.nan

    # Compute Moran's I
    numerator = 0.0
    for i in range(n):
        for j in range(n):
            numerator += W[i, j] * deviations[i] * deviations[j]

    morans_i = (n / W_sum) * (numerator / (variance * n))

    return morans_i


def analyze_marker_autocorrelation(roi_file: Path, scale: str = '10.0') -> pd.DataFrame:
    """
    Analyze spatial autocorrelation for all markers in an ROI.

    Args:
        roi_file: Path to ROI results file
        scale: Spatial scale to analyze

    Returns:
        DataFrame with Moran's I statistics for each marker
    """
    print(f"\nAnalyzing: {roi_file.name}")

    # Load results
    roi_results = load_roi_results(roi_file)
    roi_results = deserialize_numpy_arrays(roi_results)

    # Extract scale data
    scale_results = roi_results['multiscale_results'][scale]
    features = scale_results['features']  # [n_superpixels, n_markers]
    coords = scale_results['superpixel_coords']  # [n_superpixels, 2]

    # Get marker names
    marker_names = roi_results['configuration_used']['channels']['protein_channels']

    # Compute Moran's I for each marker
    results = []

    for i, marker in enumerate(marker_names):
        marker_values = features[:, i]

        # Skip if all zeros
        if marker_values.std() == 0:
            morans_i = np.nan
            print(f"  {marker:15s}: No variation")
        else:
            morans_i = compute_morans_i(marker_values, coords, distance_threshold=50.0)
            print(f"  {marker:15s}: Moran's I = {morans_i:+.3f}")

        results.append({
            'marker': marker,
            'morans_i': morans_i,
            'mean_expression': marker_values.mean(),
            'std_expression': marker_values.std(),
            'zero_fraction': (marker_values == 0).mean()
        })

    return pd.DataFrame(results)


def compare_roi_stability_vs_autocorrelation(results_dir: Path) -> pd.DataFrame:
    """
    Compare clustering stability against marker spatial autocorrelation.

    Hypothesis: ROIs where structural markers have high autocorrelation
    should have more stable clustering.
    """
    summary_path = Path('results/analysis_summary.json')

    if not summary_path.exists():
        print(f"Summary file not found: {summary_path}")
        return None

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    roi_summaries = summary.get('roi_summaries', {})

    all_results = []

    for roi_id, roi_summary in roi_summaries.items():
        roi_file = results_dir / f"roi_{roi_id}_results.json.gz"

        if not roi_file.exists():
            continue

        # Get clustering stability
        stability = roi_summary.get('consistency_metrics', {}).get('cluster_count_stability', np.nan)

        # Analyze marker autocorrelation
        try:
            marker_stats = analyze_marker_autocorrelation(roi_file, scale='10.0')

            # Get structural markers' autocorrelation
            cd31_auto = marker_stats[marker_stats['marker'] == 'CD31']['morans_i'].values[0]
            cd140a_auto = marker_stats[marker_stats['marker'] == 'CD140a']['morans_i'].values[0]
            cd140b_auto = marker_stats[marker_stats['marker'] == 'CD140b']['morans_i'].values[0]

            # Mean autocorrelation of structural markers
            structural_auto = np.nanmean([cd31_auto, cd140a_auto, cd140b_auto])

            all_results.append({
                'roi_id': roi_id,
                'stability': stability,
                'cd31_morans_i': cd31_auto,
                'cd140a_morans_i': cd140a_auto,
                'cd140b_morans_i': cd140b_auto,
                'structural_mean_morans_i': structural_auto
            })

        except Exception as e:
            print(f"  Error processing {roi_id}: {e}")
            continue

    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        # Compute correlation
        valid_mask = ~(results_df['stability'].isna() | results_df['structural_mean_morans_i'].isna())
        valid_data = results_df[valid_mask]

        if len(valid_data) > 3:
            corr, pval = pearsonr(valid_data['stability'], valid_data['structural_mean_morans_i'])
            print(f"\n{'='*80}")
            print(f"Correlation: Clustering Stability vs Structural Marker Autocorrelation")
            print(f"{'='*80}")
            print(f"Pearson r = {corr:+.3f}, p = {pval:.4f}")
            print(f"n = {len(valid_data)} ROIs")

            if pval < 0.05:
                print(f"\n✓ SIGNIFICANT: High structural autocorrelation → stable clustering")
            else:
                print(f"\n✗ Not significant (but may be underpowered with n={len(valid_data)})")

    return results_df


def main():
    print("="*80)
    print("Spatial Autocorrelation Analysis - Segmentation Channel Selection")
    print("="*80)

    results_dir = Path('results/roi_results')

    # Test on representative ROIs
    test_rois = {
        'High Stability (D7_M1_01_21, s=0.81)': 'roi_IMC_241218_Alun_ROI_D7_M1_01_21_results.json.gz',
        'Low Stability (D1_M1_01_9, s=0.08)': 'roi_IMC_241218_Alun_ROI_D1_M1_01_9_results.json.gz',
        'Medium Stability (D3_M1_01_15, s=0.60)': 'roi_IMC_241218_Alun_ROI_D3_M1_01_15_results.json.gz'
    }

    print("\n" + "─"*80)
    print("Representative ROI Analysis")
    print("─"*80)

    for label, filename in test_rois.items():
        roi_file = results_dir / filename
        if roi_file.exists():
            print(f"\n{label}:")
            marker_stats = analyze_marker_autocorrelation(roi_file, scale='10.0')

            # Identify best segmentation channels
            marker_stats_sorted = marker_stats.sort_values('morans_i', ascending=False)
            print(f"\n  Top 3 markers by spatial autocorrelation:")
            for idx, row in marker_stats_sorted.head(3).iterrows():
                print(f"    {row['marker']:15s}: Moran's I = {row['morans_i']:+.3f}")

    # Full dataset analysis
    print(f"\n{'─'*80}")
    print("Full Dataset Analysis - All ROIs")
    print("─"*80)

    results_df = compare_roi_stability_vs_autocorrelation(results_dir)

    if results_df is not None and len(results_df) > 0:
        # Save results
        output_file = 'autocorrelation_vs_stability.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")

        # Recommendations
        print(f"\n{'='*80}")
        print("Recommendations for Segmentation Channel Selection")
        print("="*80)

        # Aggregate marker statistics
        all_marker_stats = []
        for roi_file in results_dir.glob('roi_*.json.gz'):
            try:
                stats = analyze_marker_autocorrelation(roi_file, scale='10.0')
                all_marker_stats.append(stats)
            except:
                continue

        if all_marker_stats:
            combined = pd.concat(all_marker_stats)
            marker_summary = combined.groupby('marker')['morans_i'].agg(['mean', 'std', 'count'])
            marker_summary = marker_summary.sort_values('mean', ascending=False)

            print("\nMean Moran's I across all ROIs:")
            for marker, row in marker_summary.iterrows():
                print(f"  {marker:15s}: {row['mean']:+.3f} ± {row['std']:.3f} (n={int(row['count'])})")

            print(f"\nRecommended SLIC input channels (top 3 by autocorrelation):")
            for marker in marker_summary.head(3).index:
                print(f"  - {marker}")

            print(f"\nCurrent SLIC channels: DNA1, DNA2")
            print(f"Proposed: Replace with structural markers showing high autocorrelation")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
