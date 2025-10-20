"""
Demo of Result Comparison Utilities (no external dependencies)

Shows basic functionality with simple mock data.
"""

import sys
from pathlib import Path
import importlib.util

# Direct import to avoid numpy dependencies
spec = importlib.util.spec_from_file_location(
    "result_comparison", 
    str(Path(__file__).parent / "src" / "analysis" / "result_comparison.py")
)
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

# Extract classes
ResultComparer = rc.ResultComparer
ToleranceProfile = rc.ToleranceProfile
ComparisonSeverity = rc.ComparisonSeverity
compare_results = rc.compare_results
quick_cluster_comparison = rc.quick_cluster_comparison


def demo_basic_comparison():
    """Demo basic result comparison."""
    print("=" * 60)
    print("DEMO: Basic Result Comparison")
    print("=" * 60)
    
    # Create mock analysis results
    result1 = {
        'cluster_labels': [0, 0, 1, 1, 2, 2],
        'metadata': {
            'n_clusters': 3,
            'silhouette_score': 0.75,
            'method': 'leiden'
        },
        'spatial_coords': [[1.0, 1.0], [1.1, 1.1], [2.0, 2.0], [2.1, 2.1], [3.0, 3.0], [3.1, 3.1]]
    }
    
    # Create similar result with small differences
    result2 = {
        'cluster_labels': [0, 0, 1, 1, 2, 2],  # Same clustering
        'metadata': {
            'n_clusters': 3,
            'silhouette_score': 0.77,  # Slightly different score
            'method': 'leiden'
        },
        'spatial_coords': [[1.0, 1.0], [1.1, 1.1], [2.0, 2.0], [2.1, 2.1], [3.0, 3.0], [3.1, 3.1]]
    }
    
    # Package as ROI results
    results1 = {"ROI_001": result1}
    results2 = {"ROI_001": result2}
    
    # Compare
    diff_report = compare_results(results1, results2, "standard")
    
    print(f"Total comparisons: {diff_report.total_comparisons}")
    print(f"Scientific equivalence: {diff_report.is_scientifically_equivalent}")
    print(f"Summary: {diff_report.summary_stats}")
    print()


def demo_cluster_comparison():
    """Demo cluster assignment comparison."""
    print("=" * 60)
    print("DEMO: Cluster Assignment Comparison")
    print("=" * 60)
    
    # Original cluster labels
    labels1 = [0, 0, 1, 1, 2, 2, 3, 3]
    
    # Permuted labels (same clusters, different numbers)
    labels2 = [2, 2, 0, 0, 1, 1, 3, 3]
    
    # Quick comparison
    comparison = quick_cluster_comparison(labels1, labels2, tolerance=0.1)
    
    print(f"Original labels: {labels1}")
    print(f"Permuted labels: {labels2}")
    print(f"Similarity: {comparison['similarity']:.3f}")
    print(f"Equivalent (10% tolerance): {comparison['equivalent']}")
    print(f"Clusters in set 1: {comparison['n_clusters_1']}")
    print(f"Clusters in set 2: {comparison['n_clusters_2']}")
    print()


def demo_tolerance_profiles():
    """Demo different tolerance profiles."""
    print("=" * 60)
    print("DEMO: Tolerance Profiles")
    print("=" * 60)
    
    # Create results with moderate differences
    result1 = {
        'metadata': {'silhouette_score': 0.70}
    }
    
    result2 = {
        'metadata': {'silhouette_score': 0.75}  # 5% difference
    }
    
    results1 = {"ROI_001": result1}
    results2 = {"ROI_001": result2}
    
    # Test with strict tolerance
    strict_report = compare_results(results1, results2, "strict")
    print("STRICT TOLERANCE (2% threshold):")
    print(f"  Scientific equivalence: {strict_report.is_scientifically_equivalent}")
    print(f"  Different count: {strict_report.different_count}")
    
    # Test with standard tolerance  
    standard_report = compare_results(results1, results2, "standard")
    print("\nSTANDARD TOLERANCE (5% threshold):")
    print(f"  Scientific equivalence: {standard_report.is_scientifically_equivalent}")
    print(f"  Different count: {standard_report.different_count}")
    
    # Test with permissive tolerance
    permissive_report = compare_results(results1, results2, "permissive")
    print("\nPERMISSIVE TOLERANCE (10% threshold):")
    print(f"  Scientific equivalence: {permissive_report.is_scientifically_equivalent}")
    print(f"  Different count: {permissive_report.different_count}")
    print()


def demo_missing_data():
    """Demo handling of missing data."""
    print("=" * 60)
    print("DEMO: Missing Data Handling")
    print("=" * 60)
    
    # Results with missing ROI
    results1 = {
        "ROI_001": {'metadata': {'n_clusters': 3}},
        "ROI_002": {'metadata': {'n_clusters': 4}},
        "ROI_003": {'metadata': {'n_clusters': 2}}
    }
    
    results2 = {
        "ROI_001": {'metadata': {'n_clusters': 3}},
        "ROI_002": {'metadata': {'n_clusters': 4}}
        # ROI_003 missing
    }
    
    diff_report = compare_results(results1, results2, "standard")
    
    print(f"ROIs in set 1: {len(results1)}")
    print(f"ROIs in set 2: {len(results2)}")
    print(f"Incomparable results: {diff_report.incomparable_count}")
    
    # Show specific incomparable results
    incomparable = [r for r in diff_report.results if r.severity == ComparisonSeverity.INCOMPARABLE]
    for result in incomparable:
        print(f"  - {result.field_path}: {result.message}")
    print()


def demo_custom_tolerances():
    """Demo custom tolerance profile."""
    print("=" * 60)
    print("DEMO: Custom Tolerance Profile")
    print("=" * 60)
    
    # Create custom tolerance for development
    custom_profile = ToleranceProfile(
        expression_rtol=0.20,  # Very permissive for protein expression
        quality_score_tolerance=0.15,  # 15% for quality scores
        cluster_assignment_tolerance=0.30  # 30% cluster reassignment OK
    )
    
    print("Custom tolerance settings:")
    print(f"  Expression tolerance: {custom_profile.expression_rtol * 100}%")
    print(f"  Quality score tolerance: {custom_profile.quality_score_tolerance * 100}%")
    print(f"  Cluster assignment tolerance: {custom_profile.cluster_assignment_tolerance * 100}%")
    
    # Create comparer with custom profile
    comparer = ResultComparer(custom_profile)
    
    print(f"\nCustom comparer created with {len(comparer.comparison_methods)} comparison methods")
    print()


if __name__ == "__main__":
    print("IMC Result Comparison Demo")
    print("=========================")
    print("Demonstrating result comparison without external dependencies")
    print()
    
    demo_basic_comparison()
    demo_cluster_comparison() 
    demo_tolerance_profiles()
    demo_missing_data()
    demo_custom_tolerances()
    
    print("=" * 60)
    print("✓ Demo completed successfully!")
    print()
    print("Key features demonstrated:")
    print("1. ✓ Scientific equivalence assessment")
    print("2. ✓ Cluster assignment similarity (with permutation tolerance)")
    print("3. ✓ Configurable tolerance profiles") 
    print("4. ✓ Missing data handling")
    print("5. ✓ Structured diff reporting")
    print()
    print("Ready for production use with numpy/pandas for full functionality!")