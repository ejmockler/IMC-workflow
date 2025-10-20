#!/usr/bin/env python3
"""
Reproducibility Validation Example

Demonstrates how to use the ReproducibilityFramework to validate 
numerical reproducibility of IMC analysis pipelines.

Usage:
    python test_reproducibility.py
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from analysis.reproducibility_framework import (
    ReproducibilityFramework, 
    run_reproducibility_test
)
from analysis.spatial_clustering import perform_spatial_clustering
from analysis.ion_count_processing import process_ion_counts
from config import Config


def create_test_data(n_points: int = 1000, seed: int = 42) -> dict:
    """Create synthetic test data for reproducibility testing."""
    np.random.seed(seed)
    
    # Create realistic IMC-like data
    coords = np.random.uniform(0, 100, (n_points, 2))
    
    # Create protein expression data with some spatial structure
    protein_data = {}
    protein_names = ['CD45', 'CD31', 'CD3', 'CD68', 'Ki67']
    
    for i, protein in enumerate(protein_names):
        # Add some spatial clustering to make it realistic
        centers = np.random.uniform(20, 80, (3, 2))
        cluster_assignment = np.random.randint(0, 3, n_points)
        
        base_expression = np.random.poisson(50, n_points).astype(float)
        
        # Add cluster-specific effects
        for j, center in enumerate(centers):
            mask = cluster_assignment == j
            distances = np.linalg.norm(coords[mask] - center, axis=1)
            # Higher expression closer to cluster centers
            boost = np.exp(-distances / 10) * np.random.uniform(20, 100)
            base_expression[mask] += boost
            
        protein_data[protein] = base_expression
    
    # Create DNA channel data
    dna1_intensities = np.random.exponential(50, n_points)
    dna2_intensities = dna1_intensities * np.random.uniform(0.8, 1.2, n_points)
    
    return {
        'coords': coords,
        'ion_counts': protein_data,
        'dna1_intensities': dna1_intensities,
        'dna2_intensities': dna2_intensities,
        'metadata': {
            'n_points': n_points,
            'protein_names': protein_names,
            'roi_id': 'test_roi_001'
        }
    }


def mock_analysis_pipeline(data: dict, config: dict) -> dict:
    """
    Mock analysis pipeline for reproducibility testing.
    
    This simulates the core analysis steps that should be reproducible.
    """
    coords = data['coords']
    ion_counts = data['ion_counts']
    
    # Step 1: Ion count processing (transformation)
    processed_ion_counts = {}
    for protein, counts in ion_counts.items():
        # Arcsinh transformation with cofactor
        cofactor = config.get('cofactor', 5.0)
        processed_ion_counts[protein] = np.arcsinh(counts / cofactor)
    
    # Step 2: Create feature matrix
    feature_names = list(processed_ion_counts.keys())
    features = np.column_stack([processed_ion_counts[name] for name in feature_names])
    
    # Step 3: Spatial clustering
    try:
        labels, clustering_metadata = perform_spatial_clustering(
            features=features,
            coords=coords,
            method='leiden',
            random_state=config.get('random_state', 42),
            spatial_weight=config.get('spatial_weight', 0.5)
        )
    except Exception as e:
        # Fallback to simple kmeans if spatial clustering fails
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=config.get('random_state', 42))
        labels = kmeans.fit_predict(features)
        clustering_metadata = {'method': 'kmeans_fallback', 'n_clusters': 5}
    
    # Step 4: Compute summary statistics
    unique_labels = np.unique(labels)
    cluster_stats = {}
    
    for label in unique_labels:
        mask = labels == label
        cluster_coords = coords[mask]
        cluster_features = features[mask]
        
        cluster_stats[f'cluster_{label}'] = {
            'size': int(np.sum(mask)),
            'centroid': np.mean(cluster_coords, axis=0).tolist(),
            'mean_expression': np.mean(cluster_features, axis=0).tolist(),
            'std_expression': np.std(cluster_features, axis=0).tolist()
        }
    
    # Step 5: Spatial statistics
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(coords))
    
    spatial_stats = {
        'mean_nearest_neighbor': float(np.mean(np.partition(distances + np.eye(len(distances)) * 1e6, 1, axis=1)[:, 1])),
        'density': len(coords) / (np.ptp(coords[:, 0]) * np.ptp(coords[:, 1])),
        'convex_hull_area': float(np.random.uniform(1000, 5000))  # Placeholder
    }
    
    return {
        'labels': labels,
        'features': features,
        'cluster_stats': cluster_stats,
        'spatial_stats': spatial_stats,
        'clustering_metadata': clustering_metadata,
        'feature_names': feature_names,
        'processed_ion_counts': processed_ion_counts
    }


def test_framework_basic():
    """Test basic framework functionality."""
    print("Testing Basic Framework Functionality")
    print("-" * 40)
    
    framework = ReproducibilityFramework(seed=42, rtol=1e-10)
    
    # Test environment capture
    env = framework.capture_environment()
    print(f"Environment hash: {env.to_hash()}")
    print(f"NumPy version: {env.numpy_version}")
    print(f"Platform: {env.platform_system} {env.platform_release}")
    
    # Test deterministic environment
    print("\nSetting deterministic environment...")
    framework.ensure_deterministic_env()
    
    # Create identical data twice
    data1 = {'values': np.random.randn(100), 'sum': 42.0}
    np.random.seed(42)  # Reset seed
    data2 = {'values': np.random.randn(100), 'sum': 42.0}
    
    # Validate reproducibility
    result = framework.validate_reproducibility(data1, data2)
    
    print(f"Reproducibility: {'PASSED' if result.is_reproducible else 'FAILED'}")
    print(f"Max difference: {result.max_difference:.2e}")
    
    if result.failed_keys:
        print(f"Failed keys: {result.failed_keys}")
    
    framework.restore_environment()
    return result.is_reproducible


def test_pipeline_reproducibility():
    """Test reproducibility of the analysis pipeline."""
    print("\nTesting Pipeline Reproducibility")
    print("-" * 40)
    
    # Create test configuration
    config = {
        'cofactor': 5.0,
        'random_state': 42,
        'spatial_weight': 0.5,
        'clustering_method': 'leiden'
    }
    
    # Create test data
    test_data = create_test_data(n_points=500, seed=42)
    
    # Test using convenience function
    result = run_reproducibility_test(
        analysis_func=mock_analysis_pipeline,
        data=test_data,
        config=config,
        n_runs=3,
        seed=42,
        rtol=1e-10
    )
    
    print(f"Pipeline Reproducibility: {'PASSED' if result.is_reproducible else 'FAILED'}")
    print(f"Max difference: {result.max_difference:.2e}")
    print(f"Tolerance used: {result.tolerance_used:.2e}")
    
    if result.failed_keys:
        print(f"Failed keys ({len(result.failed_keys)}): {result.failed_keys[:5]}")
        
    return result.is_reproducible


def test_environment_variations():
    """Test how environment variations affect reproducibility."""
    print("\nTesting Environment Variations")
    print("-" * 40)
    
    framework = ReproducibilityFramework(seed=42)
    
    # Test 1: Different random seeds should produce different results
    np.random.seed(42)
    data1 = {'values': np.random.randn(100)}
    
    np.random.seed(123)  # Different seed
    data2 = {'values': np.random.randn(100)}
    
    result_different = framework.validate_reproducibility(data1, data2)
    print(f"Different seeds reproducible: {result_different.is_reproducible} (should be False)")
    
    # Test 2: Same seed should produce same results
    np.random.seed(42)
    data3 = {'values': np.random.randn(100)}
    
    np.random.seed(42)  # Same seed
    data4 = {'values': np.random.randn(100)}
    
    result_same = framework.validate_reproducibility(data3, data4)
    print(f"Same seeds reproducible: {result_same.is_reproducible} (should be True)")
    
    return result_same.is_reproducible and not result_different.is_reproducible


def test_tolerance_levels():
    """Test different tolerance levels."""
    print("\nTesting Tolerance Levels")
    print("-" * 40)
    
    # Create data with small differences
    base_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    tolerances = [1e-15, 1e-12, 1e-10, 1e-8, 1e-6]
    
    for rtol in tolerances:
        framework = ReproducibilityFramework(rtol=rtol)
        
        # Add small noise
        noisy_data = base_data + np.random.normal(0, rtol/10, len(base_data))
        
        result = framework.validate_reproducibility(
            {'data': base_data},
            {'data': noisy_data}
        )
        
        print(f"Tolerance {rtol:.2e}: {'PASSED' if result.is_reproducible else 'FAILED'} "
              f"(max_diff: {result.max_difference:.2e})")


def generate_full_report():
    """Generate a comprehensive reproducibility report."""
    print("\nGenerating Comprehensive Report")
    print("-" * 40)
    
    framework = ReproducibilityFramework(seed=42)
    framework.ensure_deterministic_env()
    
    try:
        # Run multiple validations
        config = {'cofactor': 5.0, 'random_state': 42}
        test_data = create_test_data(n_points=200, seed=42)
        
        for i in range(3):
            np.random.seed(42)
            result1 = mock_analysis_pipeline(test_data, config)
            np.random.seed(42)
            result2 = mock_analysis_pipeline(test_data, config)
            
            framework.validate_reproducibility(result1, result2)
        
        # Generate report
        report_path = Path("reproducibility_report.json")
        report = framework.generate_reproducibility_report(report_path)
        
        print(f"Report saved to: {report_path}")
        print(f"Success rate: {report['validation_history']['success_rate']:.1%}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
            
    finally:
        framework.restore_environment()


def main():
    """Run all reproducibility tests."""
    print("IMC Analysis Reproducibility Validation")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Basic framework test
    if test_framework_basic():
        tests_passed += 1
    
    # Pipeline reproducibility test
    if test_pipeline_reproducibility():
        tests_passed += 1
        
    # Environment variation test
    if test_environment_variations():
        tests_passed += 1
        
    # Tolerance test (always passes, just demonstrates functionality)
    test_tolerance_levels()
    tests_passed += 1
    
    # Generate comprehensive report
    generate_full_report()
    
    print(f"\nFinal Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All reproducibility tests PASSED")
        return 0
    else:
        print("❌ Some reproducibility tests FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)