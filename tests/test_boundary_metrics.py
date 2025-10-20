#!/usr/bin/env python3
"""
Test script for the Boundary Metrics Framework

Demonstrates integration with existing IMC pipeline components and validates
the comprehensive boundary quality assessment functionality.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.analysis.boundary_metrics import (
    BoundaryQualityEvaluator, SegmentationMethod, BoundaryMetricType,
    BoundaryQualityValidator, create_boundary_evaluator, evaluate_method_comparison
)
from src.analysis.synthetic_data_generator import create_example_datasets
from src.analysis.spatial_clustering import perform_spatial_clustering
from src.analysis.slic_segmentation import slic_pipeline
from src.validation.framework import ValidationSuite, ValidationSuiteConfig


def test_boundary_metrics_basic():
    """Test basic boundary metrics functionality."""
    print("Testing Basic Boundary Metrics...")
    
    # Create synthetic data
    datasets = create_example_datasets()
    simple_dataset = datasets['simple']
    
    print(f"Using dataset with {len(simple_dataset['coordinates'])} cells")
    
    # Create evaluator
    evaluator = create_boundary_evaluator(random_state=42)
    
    # Test with ground truth clustering
    ground_truth_labels = simple_dataset['ground_truth_clusters']
    coordinates = simple_dataset['coordinates']
    ion_counts = simple_dataset['ion_counts']
    
    # Create feature matrix for clustering
    feature_matrix = np.column_stack([counts for counts in ion_counts.values()])
    
    # Generate predicted labels using spatial clustering
    predicted_labels, _ = perform_spatial_clustering(
        feature_matrix, coordinates, method='leiden', resolution=1.0
    )
    
    print(f"Ground truth clusters: {len(np.unique(ground_truth_labels[ground_truth_labels >= 0]))}")
    print(f"Predicted clusters: {len(np.unique(predicted_labels[predicted_labels >= 0]))}")
    
    # Test clustering quality evaluation
    clustering_metrics = evaluator.evaluate_clustering_quality(
        predicted_labels, ground_truth_labels, feature_matrix, coordinates
    )
    
    print(f"\nClustering Quality Metrics ({len(clustering_metrics)}):")
    for metric in clustering_metrics:
        print(f"  {metric.metric_name}: {metric.value:.3f} (quality: {metric.quality_score:.3f})")
    
    # Test biological relevance evaluation
    biological_metrics = evaluator.evaluate_biological_relevance(
        predicted_labels, ion_counts, coordinates
    )
    
    print(f"\nBiological Relevance Metrics ({len(biological_metrics)}):")
    for metric in biological_metrics:
        print(f"  {metric.metric_name}: {metric.value:.3f} (quality: {metric.quality_score:.3f})")
    
    # Test boundary precision (using ground truth as reference)
    boundary_metrics = evaluator.evaluate_boundary_precision(
        predicted_labels, ground_truth_labels, coordinates
    )
    
    print(f"\nBoundary Precision Metrics ({len(boundary_metrics)}):")
    for metric in boundary_metrics:
        print(f"  {metric.metric_name}: {metric.value:.3f} (quality: {metric.quality_score:.3f})")
    
    return True


def test_comprehensive_evaluation():
    """Test comprehensive evaluation with SLIC integration."""
    print("\nTesting Comprehensive Evaluation with SLIC...")
    
    # Create synthetic data
    datasets = create_example_datasets()
    complex_dataset = datasets['complex']
    
    coordinates = complex_dataset['coordinates']
    ion_counts = complex_dataset['ion_counts']
    dna1 = complex_dataset.get('dna1_intensities', np.ones(len(coordinates)))
    dna2 = complex_dataset.get('dna2_intensities', dna1 * 0.8)
    
    # Run SLIC segmentation
    try:
        slic_results = slic_pipeline(
            coords=coordinates,
            ion_counts=ion_counts,
            dna1_intensities=dna1,
            dna2_intensities=dna2,
            target_scale_um=20.0
        )
        
        print(f"SLIC generated {slic_results.get('n_segments_used', 0)} segments")
        
        # Prepare results for evaluation
        segmentation_results = {
            'labels': slic_results.get('superpixel_labels', np.array([])),
            'coordinates': slic_results.get('superpixel_coords', np.array([])),
            'ion_counts': slic_results.get('superpixel_counts', {}),
            'segmentation_mask': slic_results.get('superpixel_labels')
        }
        
        # Comprehensive evaluation
        evaluator = create_boundary_evaluator()
        all_metrics = evaluator.evaluate_comprehensive(
            segmentation_results, complex_dataset
        )
        
        print(f"\nComprehensive Evaluation Results ({len(all_metrics)} metrics):")
        
        # Group by metric type
        metrics_by_type = {}
        for metric in all_metrics:
            metric_type = metric.metric_type.value
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
            metrics_by_type[metric_type].append(metric)
        
        for metric_type, metrics in metrics_by_type.items():
            print(f"\n{metric_type.upper()}:")
            for metric in metrics:
                print(f"  {metric.metric_name}: {metric.value:.3f} (quality: {metric.quality_score:.3f})")
        
        # Generate evaluation report
        report = evaluator.generate_evaluation_report(all_metrics, "SLIC")
        
        print(f"\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        print(f"Method: {report['method_name']}")
        print(f"Overall Quality: {report['quality_assessment']['rating']}")
        print(f"Quality Score: {report['summary']['overall_quality_score']:.3f}")
        print(f"Assessment: {report['quality_assessment']['interpretation']}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
        
        return True
        
    except Exception as e:
        print(f"SLIC integration test failed: {e}")
        return False


def test_method_comparison():
    """Test comparison between multiple segmentation methods."""
    print("\nTesting Method Comparison...")
    
    # Create synthetic data
    datasets = create_example_datasets()
    dataset = datasets['simple']
    
    coordinates = dataset['coordinates']
    ion_counts = dataset['ion_counts']
    
    # Create feature matrix
    feature_matrix = np.column_stack([counts for counts in ion_counts.values()])
    
    # Test multiple clustering methods
    method_results = {}
    
    # Method 1: Leiden clustering (high resolution)
    try:
        labels1, info1 = perform_spatial_clustering(
            feature_matrix, coordinates, method='leiden', resolution=1.5
        )
        method_results['leiden_high'] = {
            'labels': labels1,
            'coordinates': coordinates,
            'ion_counts': ion_counts,
            'feature_matrix': feature_matrix
        }
        print(f"Leiden (high res): {info1['n_clusters']} clusters")
    except Exception as e:
        print(f"Leiden high res failed: {e}")
    
    # Method 2: Leiden clustering (low resolution)
    try:
        labels2, info2 = perform_spatial_clustering(
            feature_matrix, coordinates, method='leiden', resolution=0.8
        )
        method_results['leiden_low'] = {
            'labels': labels2,
            'coordinates': coordinates,
            'ion_counts': ion_counts,
            'feature_matrix': feature_matrix
        }
        print(f"Leiden (low res): {info2['n_clusters']} clusters")
    except Exception as e:
        print(f"Leiden low res failed: {e}")
    
    # Method 3: HDBSCAN clustering
    try:
        labels3, info3 = perform_spatial_clustering(
            feature_matrix, coordinates, method='hdbscan', min_cluster_size=50
        )
        method_results['hdbscan'] = {
            'labels': labels3,
            'coordinates': coordinates,
            'ion_counts': ion_counts,
            'feature_matrix': feature_matrix
        }
        print(f"HDBSCAN: {info3['n_clusters']} clusters")
    except Exception as e:
        print(f"HDBSCAN failed: {e}")
    
    if len(method_results) >= 2:
        # Compare methods
        comparison_result = evaluate_method_comparison(
            method_results, dataset
        )
        
        print(f"\n" + "="*50)
        print("METHOD COMPARISON RESULTS")
        print("="*50)
        
        print("\nMethod Rankings:")
        for i, (method, score) in enumerate(comparison_result.ranking, 1):
            print(f"  {i}. {method} (score: {score:.3f})")
        
        print("\nStatistical Tests:")
        for test_name, results in comparison_result.statistical_tests.items():
            if results.get('significant', False):
                print(f"  {test_name}: F={results['f_statistic']:.3f}, p={results['p_value']:.3f} *")
            else:
                print(f"  {test_name}: F={results['f_statistic']:.3f}, p={results['p_value']:.3f}")
        
        print("\nRecommendations:")
        for rec in comparison_result.recommendations:
            print(f"  - {rec}")
        
        return True
    else:
        print("Not enough methods succeeded for comparison")
        return False


def test_validation_integration():
    """Test integration with validation framework."""
    print("\nTesting Validation Framework Integration...")
    
    # Create synthetic data
    datasets = create_example_datasets()
    dataset = datasets['simple']
    
    # Create mock segmentation results
    n_cells = len(dataset['coordinates'])
    mock_results = {
        'segmentation_results': {
            'labels': np.random.randint(0, 4, n_cells),
            'coordinates': dataset['coordinates'],
            'ion_counts': dataset['ion_counts']
        },
        'ground_truth_data': dataset
    }
    
    # Create validation suite
    config = ValidationSuiteConfig(
        name="boundary_quality_test",
        enabled_categories=[ValidationCategory.SCIENTIFIC_QUALITY]
    )
    suite = ValidationSuite(config)
    
    # Add boundary quality validator
    boundary_validator = BoundaryQualityValidator(quality_threshold=0.6)
    suite.add_rule(boundary_validator)
    
    # Run validation
    validation_result = suite.validate(mock_results)
    
    print(f"Validation Status: {validation_result.summary_stats['status']}")
    print(f"Overall Quality Score: {validation_result.summary_stats.get('overall_quality_score', 'N/A')}")
    
    # Print validation results
    for result in validation_result.results:
        print(f"\nRule: {result.rule_name}")
        print(f"Severity: {result.severity.value}")
        print(f"Message: {result.message}")
        print(f"Quality Score: {result.quality_score}")
        
        if result.recommendations:
            print("Recommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")
    
    return validation_result.summary_stats['status'] != 'critical'


def main():
    """Run all tests."""
    print("Boundary Metrics Framework Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Boundary Metrics", test_boundary_metrics_basic),
        ("Comprehensive Evaluation", test_comprehensive_evaluation),
        ("Method Comparison", test_method_comparison),
        ("Validation Integration", test_validation_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            results[test_name] = "ERROR"
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASS" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("\n🎉 All tests passed! Boundary metrics framework is ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above for details.")


if __name__ == "__main__":
    main()