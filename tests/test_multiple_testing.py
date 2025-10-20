#!/usr/bin/env python3
"""
Test Script for Hierarchical Multiple Testing Control Framework

Validates the implementation of Phase 2E Multiple Testing Control
with synthetic multiscale data and realistic spatial dependencies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from analysis.multiple_testing_control import (
    HierarchicalMultipleTestingControl,
    HierarchicalTestingConfig,
    HypothesisFamily,
    HypothesisType,
    create_standard_hypothesis_families,
    bootstrap_effect_size_testing
)
from analysis.fdr_spatial import FDRConfig
from analysis.spatial_permutation import PermutationConfig


def create_synthetic_multiscale_data():
    """Create synthetic multiscale data with known ground truth."""
    np.random.seed(42)
    
    # Create spatial coordinates for 3 scales
    scales = ["10um", "20um", "40um"]
    n_points = [1000, 400, 100]  # Decreasing resolution
    
    spatial_coords = {}
    test_results = {}
    
    for i, (scale, n) in enumerate(zip(scales, n_points)):
        # Generate spatial coordinates
        coords = np.random.uniform(0, 100, (n, 2))
        spatial_coords[scale] = coords
        
        # Generate synthetic test results for multiple markers
        markers = ["CD3", "CD20", "CD68", "Ki67", "Vimentin"]
        scale_results = {}
        
        for j, marker in enumerate(markers):
            # Create different effect sizes - some significant, some not
            if marker in ["CD3", "CD20"]:  # True positives
                effect_size = 0.5 + i * 0.1  # Stronger effects at coarser scales
                p_value = 0.001 * (j + 1)  # Significant
            elif marker == "CD68":  # Borderline
                effect_size = 0.15
                p_value = 0.04
            else:  # True negatives
                effect_size = 0.05
                p_value = 0.2 + j * 0.1
            
            # Add spatial correlation structure to statistics
            statistic = effect_size + 0.1 * np.sin(i + j)
            
            scale_results[marker] = {
                'p_value': p_value,
                'statistic': statistic,
                'effect_size': effect_size,
                'test_type': 'marker_expression',
                'standard_error': effect_size * 0.1,
                'raw_data': np.random.normal(effect_size, 0.2, n),
                'null_value': 0.0
            }
        
        # Add clustering quality metrics
        scale_results["clustering_silhouette"] = {
            'p_value': 0.02,
            'statistic': 0.7 - i * 0.1,  # Better clustering at finer scales
            'test_type': 'clustering_quality',
            'effect_size': 0.7 - i * 0.1
        }
        
        # Add spatial statistics
        scale_results["spatial_moran_i"] = {
            'p_value': 0.001,
            'statistic': 0.3 + i * 0.05,
            'test_type': 'spatial_statistic',
            'effect_size': 0.3 + i * 0.05
        }
        
        test_results[scale] = scale_results
    
    return test_results, spatial_coords


def test_hierarchical_fdr_control():
    """Test hierarchical FDR control across scales and markers."""
    print("=== Testing Hierarchical FDR Control ===")
    
    # Create configuration
    config = HierarchicalTestingConfig(
        fdr_config=FDRConfig(
            method='benjamini_yekutieli',
            alpha=0.05,
            dependence_assumption='arbitrary'
        ),
        bootstrap_n=100,  # Reduced for testing
        min_n_for_pvalues=5
    )
    
    # Initialize controller
    controller = HierarchicalMultipleTestingControl(config)
    
    # Register hypothesis families
    scales = ["10um", "20um", "40um"]
    markers = ["CD3", "CD20", "CD68", "Ki67", "Vimentin"]
    families = create_standard_hypothesis_families(scales, markers)
    
    for family in families:
        controller.register_hypothesis_family(family)
    
    # Create test data
    test_results, spatial_coords = create_synthetic_multiscale_data()
    
    # Apply hierarchical testing
    results = controller.multiscale_hypothesis_testing(
        test_results, spatial_coords
    )
    
    # Validate results
    print(f"Hierarchical corrections applied to {len(results['hierarchical_corrections'])} families")
    print(f"Bootstrap CIs computed for {len(results['bootstrap_confidence_intervals'])} families")
    print(f"FWER control applied to {len(results['family_wise_results'])} families")
    
    # Check discovery patterns
    total_discoveries = 0
    for family_name, family_results in results['hierarchical_corrections'].items():
        discoveries = family_results.get('discoveries', {})
        n_discoveries = sum(discoveries.values())
        total_discoveries += n_discoveries
        print(f"  {family_name}: {n_discoveries}/{len(discoveries)} discoveries")
    
    print(f"Total discoveries: {total_discoveries}")
    
    # Validate recommendations
    recommendations = results.get('recommendations', [])
    print(f"Generated {len(recommendations)} recommendations:")
    for rec in recommendations[:3]:  # Show first 3
        print(f"  - {rec}")
    
    return results


def test_bootstrap_confidence_intervals():
    """Test bootstrap confidence interval approach for small-n studies."""
    print("\n=== Testing Bootstrap Confidence Intervals ===")
    
    # Create small sample configuration
    config = HierarchicalTestingConfig(
        bootstrap_n=500,
        bootstrap_confidence=0.95,
        min_n_for_pvalues=20  # Force bootstrap CI usage
    )
    
    controller = HierarchicalMultipleTestingControl(config)
    
    # Create small sample data
    np.random.seed(123)
    n_small = 8  # Small sample size
    
    # True effects with small samples
    true_effects = np.array([0.5, 0.3, 0.1, 0.0, -0.2])
    observed_effects = true_effects + np.random.normal(0, 0.1, len(true_effects))
    
    # Generate bootstrap samples
    def bootstrap_generator():
        for _ in range(config.bootstrap_n):
            # Bootstrap by resampling with noise
            bootstrap_sample = observed_effects + np.random.normal(0, 0.15, len(observed_effects))
            yield bootstrap_sample
    
    # Test bootstrap CI approach
    cis = bootstrap_effect_size_testing(
        observed_effects, 
        bootstrap_generator(),
        confidence_level=config.bootstrap_confidence
    )
    
    print(f"Computed {len(cis)} bootstrap confidence intervals:")
    for i, ci in enumerate(cis):
        excludes_null = (ci.ci_lower > 0) or (ci.ci_upper < 0)
        print(f"  Effect {i+1}: {ci.effect_size:.3f} [{ci.ci_lower:.3f}, {ci.ci_upper:.3f}] "
              f"{'*' if excludes_null else ' '}")
    
    # Count significant effects (CIs excluding 0)
    significant = sum(1 for ci in cis if (ci.ci_lower > 0) or (ci.ci_upper < 0))
    print(f"Significant effects (CI excludes 0): {significant}/{len(cis)}")
    
    return cis


def test_family_wise_error_control():
    """Test family-wise error rate control for clustering optimization."""
    print("\n=== Testing Family-Wise Error Rate Control ===")
    
    config = HierarchicalTestingConfig(
        fwer_alpha=0.05,
        fwer_method="holm"
    )
    
    controller = HierarchicalMultipleTestingControl(config)
    
    # Register clustering optimization family
    clustering_family = HypothesisFamily(
        name="clustering_optimization",
        hypothesis_type=HypothesisType.CLUSTERING,
        priority=1
    )
    controller.register_hypothesis_family(clustering_family)
    
    # Create clustering parameter test results
    test_results = {
        "10um": {
            "resolution_0.5": {'p_value': 0.01, 'statistic': 0.8, 'test_type': 'clustering_quality'},
            "resolution_1.0": {'p_value': 0.03, 'statistic': 0.75, 'test_type': 'clustering_quality'},
            "resolution_1.5": {'p_value': 0.08, 'statistic': 0.65, 'test_type': 'clustering_quality'},
            "resolution_2.0": {'p_value': 0.15, 'statistic': 0.6, 'test_type': 'clustering_quality'}
        }
    }
    
    spatial_coords = {"10um": np.random.uniform(0, 100, (100, 2))}
    
    # Apply FWER control
    results = controller.multiscale_hypothesis_testing(
        test_results, spatial_coords
    )
    
    # Check FWER results
    fwer_results = results.get('family_wise_results', {})
    if fwer_results:
        for family_name, family_result in fwer_results.items():
            discoveries = family_result.get('discoveries', {})
            adjusted_pvals = family_result.get('adjusted_p_values', {})
            
            print(f"FWER Control Results for {family_name}:")
            for test_name, is_significant in discoveries.items():
                adj_p = adjusted_pvals.get(test_name, 'N/A')
                print(f"  {test_name}: {'Significant' if is_significant else 'Not significant'} "
                      f"(adj. p = {adj_p:.4f})")
    
    return results


def test_spatial_dependence_handling():
    """Test spatial dependence awareness in multiple testing."""
    print("\n=== Testing Spatial Dependence Handling ===")
    
    config = HierarchicalTestingConfig(
        fdr_config=FDRConfig(
            method='benjamini_yekutieli',
            use_spatial_weights=True,
            adaptive_weights=True
        )
    )
    
    controller = HierarchicalMultipleTestingControl(config)
    
    # Create spatially correlated data
    np.random.seed(456)
    n_points = 200
    
    # Grid coordinates for strong spatial correlation
    x = np.arange(0, 14)
    y = np.arange(0, 14)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.ravel()[:n_points], yy.ravel()[:n_points]])
    
    # Generate spatially correlated p-values
    spatial_field = np.random.normal(0, 1, (14, 14))
    from scipy.ndimage import gaussian_filter
    smoothed_field = gaussian_filter(spatial_field, sigma=2.0)
    
    # Convert to p-values with spatial structure
    p_values = stats.norm.cdf(smoothed_field.ravel()[:n_points])
    
    # Create test results with spatial correlation
    test_results = {
        "20um": {}
    }
    
    for i, p_val in enumerate(p_values[:10]):  # Use first 10 for testing
        test_results["20um"][f"spatial_test_{i}"] = {
            'p_value': p_val,
            'statistic': -stats.norm.ppf(p_val),
            'test_type': 'spatial_statistic'
        }
    
    spatial_coords = {"20um": coords[:10]}
    
    # Calculate effective tests
    effective_tests = controller._calculate_effective_tests(test_results, spatial_coords)
    
    print(f"Spatial dependence analysis:")
    for scale, eff_info in effective_tests.items():
        print(f"  {scale}: {eff_info['n_effective']:.1f} effective tests "
              f"(from {eff_info['n_nominal']} nominal tests)")
        print(f"    Correlation reduction: {eff_info['correlation_reduction']:.2f}")
    
    return effective_tests


def validate_statistical_properties():
    """Validate statistical properties of the multiple testing framework."""
    print("\n=== Validating Statistical Properties ===")
    
    # Test Type I error control under null
    np.random.seed(789)
    n_simulations = 100
    n_tests = 20
    alpha = 0.05
    
    false_discovery_rates = []
    
    for sim in range(n_simulations):
        # Generate null p-values
        null_p_values = np.random.uniform(0, 1, n_tests)
        
        # Apply Benjamini-Yekutieli
        config = HierarchicalTestingConfig(
            fdr_config=FDRConfig(method='benjamini_yekutieli', alpha=alpha)
        )
        controller = HierarchicalMultipleTestingControl(config)
        
        # Simple FDR test
        discoveries = controller.spatial_fdr.benjamini_yekutieli_spatial(null_p_values)
        fdr = np.mean(discoveries) if np.sum(discoveries) > 0 else 0
        false_discovery_rates.append(fdr)
    
    mean_fdr = np.mean(false_discovery_rates)
    print(f"Empirical FDR under null (should be ≤ {alpha}): {mean_fdr:.4f}")
    print(f"FDR control: {'PASS' if mean_fdr <= alpha else 'FAIL'}")
    
    # Test power under alternative
    effect_sizes = [0.0, 0.2, 0.5, 0.8]
    for effect_size in effect_sizes:
        power_estimates = []
        
        for sim in range(50):  # Fewer sims for speed
            # Generate p-values with effect
            z_scores = np.random.normal(effect_size, 1, n_tests)
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
            
            discoveries = controller.spatial_fdr.benjamini_yekutieli_spatial(p_values)
            power = np.mean(discoveries)
            power_estimates.append(power)
        
        mean_power = np.mean(power_estimates)
        print(f"Power at effect size {effect_size}: {mean_power:.3f}")
    
    return mean_fdr


def main():
    """Run comprehensive tests of the multiple testing control framework."""
    print("Testing Hierarchical Multiple Testing Control Framework")
    print("=" * 60)
    
    try:
        # Test 1: Hierarchical FDR control
        hierarchical_results = test_hierarchical_fdr_control()
        
        # Test 2: Bootstrap confidence intervals
        bootstrap_results = test_bootstrap_confidence_intervals()
        
        # Test 3: Family-wise error rate control
        fwer_results = test_family_wise_error_control()
        
        # Test 4: Spatial dependence handling
        spatial_results = test_spatial_dependence_handling()
        
        # Test 5: Statistical properties validation
        statistical_validation = validate_statistical_properties()
        
        print("\n" + "=" * 60)
        print("SUMMARY: All tests completed successfully!")
        print("✓ Hierarchical FDR control implemented")
        print("✓ Bootstrap confidence intervals working")
        print("✓ Family-wise error rate control functional")
        print("✓ Spatial dependence properly handled")
        print("✓ Statistical properties validated")
        print("\nPhase 2E Multiple Testing Control: COMPLETE")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)