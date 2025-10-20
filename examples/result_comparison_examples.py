"""
Result Comparison Examples

Demonstrates usage of the result comparison utilities for IMC analysis.
Shows various comparison scenarios and tolerance settings.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.result_comparison import (
    ResultComparer, ToleranceProfile, compare_results, 
    quick_cluster_comparison, ComparisonSeverity
)


def create_sample_analysis_result(roi_id: str, add_noise: bool = False) -> dict:
    """Create a sample analysis result for demonstration."""
    np.random.seed(42 if not add_noise else 43)  # Different seeds for comparison
    
    n_points = 1000
    n_proteins = 5
    n_clusters = 4
    
    # Generate synthetic data
    coords = np.random.rand(n_points, 2) * 100
    feature_matrix = np.random.exponential(2.0, (n_points, n_proteins))
    
    # Add noise if requested
    if add_noise:
        feature_matrix += np.random.normal(0, 0.05, feature_matrix.shape)
        coords += np.random.normal(0, 0.1, coords.shape)
    
    # Generate cluster labels
    cluster_labels = np.random.randint(0, n_clusters, n_points)
    
    # Calculate cluster centroids
    cluster_centroids = {}
    protein_names = [f"Protein_{i}" for i in range(n_proteins)]
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if np.sum(mask) > 0:
            cluster_centroids[str(cluster_id)] = {
                protein: float(np.mean(feature_matrix[mask, i]))
                for i, protein in enumerate(protein_names)
            }
    
    # Create full result structure
    result = {
        'coords': coords,
        'feature_matrix': feature_matrix,
        'protein_names': protein_names,
        'cluster_labels': cluster_labels,
        'cluster_centroids': cluster_centroids,
        'multiscale_results': {
            'scale_results': {
                '20.0': {
                    'features': feature_matrix,
                    'spatial_coords': coords,
                    'cluster_labels': cluster_labels,
                    'spatial_coherence': 0.75 + (0.05 if add_noise else 0.0),
                    'scale_um': 20.0
                }
            },
            'consistency_results': {
                'scale_consistency': {
                    'consistency_score': 0.85 + (0.02 if add_noise else 0.0),
                    'cross_scale_correlation': 0.92 + (0.03 if add_noise else 0.0)
                }
            }
        },
        'metadata': {
            'n_measurements': n_points,
            'n_clusters': n_clusters,
            'silhouette_score': 0.65 + (0.05 if add_noise else 0.0),
            'method': 'leiden'
        },
        'configuration_used': {
            'scales_um': [10.0, 20.0, 40.0],
            'method': 'leiden',
            'use_slic': True
        }
    }
    
    return result


def example_1_identical_results():
    """Example 1: Compare identical results."""
    print("=" * 60)
    print("EXAMPLE 1: Comparing Identical Results")
    print("=" * 60)
    
    # Create identical results
    result1 = create_sample_analysis_result("ROI_001", add_noise=False)
    result2 = create_sample_analysis_result("ROI_001", add_noise=False)
    
    # Package as ROI dictionaries
    results1 = {"ROI_001": result1}
    results2 = {"ROI_001": result2}
    
    # Compare
    comparer = ResultComparer()
    diff_report = comparer.compare_analysis_results(results1, results2, "standard")
    
    print(f"Total comparisons: {diff_report.total_comparisons}")
    print(f"Scientific equivalence: {diff_report.is_scientifically_equivalent}")
    print(f"Summary: {diff_report.summary_stats}")
    
    if diff_report.get_critical_differences():
        print("Critical differences found:")
        for diff in diff_report.get_critical_differences():
            print(f"  - {diff.field_path}: {diff.message}")
    else:
        print("No critical differences found ✓")
    
    print()


def example_2_equivalent_with_noise():
    """Example 2: Compare results with small numerical differences."""
    print("=" * 60)
    print("EXAMPLE 2: Comparing Results with Small Noise")
    print("=" * 60)
    
    # Create results with small differences
    result1 = create_sample_analysis_result("ROI_001", add_noise=False)
    result2 = create_sample_analysis_result("ROI_001", add_noise=True)
    
    results1 = {"ROI_001": result1}
    results2 = {"ROI_001": result2}
    
    # Compare with standard tolerance
    diff_report = compare_results(results1, results2, "standard")
    
    print(f"Total comparisons: {diff_report.total_comparisons}")
    print(f"Scientific equivalence: {diff_report.is_scientifically_equivalent}")
    print(f"Equivalent results: {diff_report.equivalent_count}")
    print(f"Different results: {diff_report.different_count}")
    
    # Show some specific comparisons
    print("\nSample comparison results:")
    for result in diff_report.results[:5]:
        print(f"  {result.field_path}: {result.severity.value} - {result.message}")
    
    print()


def example_3_strict_vs_permissive():
    """Example 3: Compare different tolerance profiles."""
    print("=" * 60)
    print("EXAMPLE 3: Strict vs Permissive Tolerance Profiles")
    print("=" * 60)
    
    # Create results with moderate differences
    result1 = create_sample_analysis_result("ROI_001", add_noise=False)
    result2 = create_sample_analysis_result("ROI_001", add_noise=True)
    
    # Modify result2 to have larger differences
    result2['metadata']['silhouette_score'] = result1['metadata']['silhouette_score'] + 0.08
    result2['multiscale_results']['consistency_results']['scale_consistency']['consistency_score'] += 0.12
    
    results1 = {"ROI_001": result1}
    results2 = {"ROI_001": result2}
    
    # Compare with strict tolerance
    strict_report = compare_results(results1, results2, "strict")
    print("STRICT TOLERANCE:")
    print(f"  Scientific equivalence: {strict_report.is_scientifically_equivalent}")
    print(f"  Different: {strict_report.different_count}, Equivalent: {strict_report.equivalent_count}")
    
    # Compare with permissive tolerance
    permissive_report = compare_results(results1, results2, "permissive")
    print("\nPERMISSIVE TOLERANCE:")
    print(f"  Scientific equivalence: {permissive_report.is_scientifically_equivalent}")
    print(f"  Different: {permissive_report.different_count}, Equivalent: {permissive_report.equivalent_count}")
    
    # Show critical differences in strict mode
    critical_diffs = strict_report.get_critical_differences()
    if critical_diffs:
        print(f"\nCritical differences in strict mode ({len(critical_diffs)}):")
        for diff in critical_diffs[:3]:  # Show first 3
            print(f"  - {diff.field_path}: {diff.message}")
    
    print()


def example_4_cluster_comparison():
    """Example 4: Focus on clustering result comparison."""
    print("=" * 60)
    print("EXAMPLE 4: Clustering Result Comparison")
    print("=" * 60)
    
    # Create clustering results with permuted labels
    np.random.seed(42)
    n_points = 500
    labels1 = np.random.randint(0, 4, n_points)
    
    # Create permuted version (same clusters, different labels)
    label_mapping = {0: 2, 1: 0, 2: 3, 3: 1}
    labels2 = np.array([label_mapping[label] for label in labels1])
    
    # Add some noise (5% different assignments)
    noise_indices = np.random.choice(n_points, int(0.05 * n_points), replace=False)
    labels2[noise_indices] = np.random.randint(0, 4, len(noise_indices))
    
    # Quick comparison
    cluster_comp = quick_cluster_comparison(labels1, labels2, tolerance=0.10)
    print("Quick cluster comparison:")
    print(f"  Similarity: {cluster_comp['similarity']:.3f}")
    print(f"  Equivalent (10% tolerance): {cluster_comp['equivalent']}")
    print(f"  Clusters in result 1: {cluster_comp['n_clusters_1']}")
    print(f"  Clusters in result 2: {cluster_comp['n_clusters_2']}")
    
    # Full comparison
    result1 = {'cluster_labels': labels1}
    result2 = {'cluster_labels': labels2}
    
    comparer = ResultComparer()
    comparison_results = comparer.compare_clustering_results(result1, result2)
    
    print("\nDetailed clustering comparison:")
    for comp in comparison_results:
        print(f"  {comp.field_path}: {comp.severity.value}")
        print(f"    {comp.message}")
        if comp.difference_metric is not None:
            print(f"    Difference metric: {comp.difference_metric:.4f}")
    
    print()


def example_5_multiple_rois():
    """Example 5: Compare multiple ROIs."""
    print("=" * 60)
    print("EXAMPLE 5: Multiple ROI Comparison")
    print("=" * 60)
    
    # Create multiple ROI results
    roi_ids = ["ROI_001", "ROI_002", "ROI_003"]
    
    results1 = {}
    results2 = {}
    
    for roi_id in roi_ids:
        results1[roi_id] = create_sample_analysis_result(roi_id, add_noise=False)
        results2[roi_id] = create_sample_analysis_result(roi_id, add_noise=True)
    
    # Add a missing ROI in results2
    results1["ROI_004"] = create_sample_analysis_result("ROI_004", add_noise=False)
    
    # Compare all ROIs
    diff_report = compare_results(results1, results2, "standard")
    
    print(f"ROIs in result set 1: {len(results1)}")
    print(f"ROIs in result set 2: {len(results2)}")
    print(f"Total field comparisons: {diff_report.total_comparisons}")
    print(f"Overall scientific equivalence: {diff_report.is_scientifically_equivalent}")
    
    # Breakdown by severity
    severity_counts = {}
    for result in diff_report.results:
        severity = result.severity
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    print("\nComparison breakdown:")
    for severity, count in severity_counts.items():
        print(f"  {severity.value}: {count}")
    
    # Show incomparable results (missing ROIs)
    incomparable = [r for r in diff_report.results if r.severity == ComparisonSeverity.INCOMPARABLE]
    if incomparable:
        print(f"\nIncomparable results ({len(incomparable)}):")
        for result in incomparable:
            print(f"  - {result.field_path}: {result.message}")
    
    print()


def example_6_custom_tolerance_profile():
    """Example 6: Custom tolerance profile."""
    print("=" * 60)
    print("EXAMPLE 6: Custom Tolerance Profile")
    print("=" * 60)
    
    # Create custom tolerance profile for development use
    custom_profile = ToleranceProfile(
        cluster_assignment_tolerance=0.20,  # Allow 20% cluster reassignment
        expression_rtol=0.08,               # 8% tolerance for protein expression
        spatial_stats_rtol=0.10,            # 10% tolerance for spatial stats
        quality_score_tolerance=0.15        # 15% tolerance for quality scores
    )
    
    # Create results with moderate differences
    result1 = create_sample_analysis_result("ROI_001", add_noise=False)
    result2 = create_sample_analysis_result("ROI_001", add_noise=True)
    
    # Add larger differences
    result2['metadata']['silhouette_score'] = result1['metadata']['silhouette_score'] + 0.10
    
    results1 = {"ROI_001": result1}
    results2 = {"ROI_001": result2}
    
    # Compare with custom profile
    comparer = ResultComparer(custom_profile)
    diff_report = comparer.compare_analysis_results(results1, results2)
    
    print("Custom tolerance profile settings:")
    print(f"  Cluster assignment tolerance: {custom_profile.cluster_assignment_tolerance}")
    print(f"  Expression relative tolerance: {custom_profile.expression_rtol}")
    print(f"  Quality score tolerance: {custom_profile.quality_score_tolerance}")
    
    print(f"\nComparison results:")
    print(f"  Scientific equivalence: {diff_report.is_scientifically_equivalent}")
    print(f"  Equivalent: {diff_report.equivalent_count}")
    print(f"  Different: {diff_report.different_count}")
    
    print()


def example_7_save_and_load_report():
    """Example 7: Save and analyze diff report."""
    print("=" * 60)
    print("EXAMPLE 7: Save and Load Diff Report")
    print("=" * 60)
    
    # Create and compare results
    result1 = create_sample_analysis_result("ROI_001", add_noise=False)
    result2 = create_sample_analysis_result("ROI_001", add_noise=True)
    
    results1 = {"ROI_001": result1}
    results2 = {"ROI_001": result2}
    
    # Generate diff report and save
    output_path = Path("result_comparison_report.json")
    diff_report = compare_results(results1, results2, "standard", output_path)
    
    print(f"Diff report saved to: {output_path}")
    print(f"Report comparison ID: {diff_report.comparison_id}")
    print(f"Report timestamp: {diff_report.timestamp}")
    
    # Show report structure
    report_dict = diff_report.to_dict()
    print(f"\nReport structure:")
    for key in report_dict.keys():
        if key == 'all_results':
            print(f"  {key}: {len(report_dict[key])} comparison results")
        elif key == 'critical_differences':
            print(f"  {key}: {len(report_dict[key])} critical differences")
        else:
            print(f"  {key}: {type(report_dict[key])}")
    
    # Clean up
    if output_path.exists():
        output_path.unlink()
        print(f"\nCleaned up: {output_path}")
    
    print()


if __name__ == "__main__":
    print("IMC Result Comparison Examples")
    print("==============================")
    print()
    
    # Run all examples
    example_1_identical_results()
    example_2_equivalent_with_noise()
    example_3_strict_vs_permissive()
    example_4_cluster_comparison()
    example_5_multiple_rois()
    example_6_custom_tolerance_profile()
    example_7_save_and_load_report()
    
    print("All examples completed successfully! ✓")
    print("\nKey takeaways:")
    print("1. Use 'standard' tolerance for most comparisons")
    print("2. Use 'strict' for validation, 'permissive' for development")
    print("3. Focus on scientific equivalence rather than exact matches")
    print("4. Cluster assignments are compared with permutation tolerance")
    print("5. Save diff reports for documentation and debugging")