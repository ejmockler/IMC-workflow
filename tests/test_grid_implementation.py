#!/usr/bin/env python3
"""
Test script to validate the grid segmentation implementation.

This script creates synthetic IMC data and tests the grid segmentation
pipeline to ensure it works correctly with the existing codebase.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.analysis.grid_segmentation import grid_pipeline, compare_grid_vs_slic
    from src.analysis.segmentation_benchmark import SegmentationBenchmark
    from src.analysis.multiscale_analysis import perform_multiscale_analysis
    from src.analysis.segmentation_comparison_examples import example_basic_comparison
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def create_synthetic_imc_data(n_points: int = 1000, spatial_extent: float = 100.0):
    """Create synthetic IMC data for testing."""
    print(f"Creating synthetic IMC data with {n_points} points over {spatial_extent}μm")
    
    # Create random spatial coordinates
    coords = np.random.uniform(0, spatial_extent, (n_points, 2))
    
    # Create synthetic protein channels
    proteins = ['CD3', 'CD4', 'CD8', 'CD20', 'DAPI1', 'DAPI2']
    ion_counts = {}
    
    # Simulate protein expression with spatial structure
    for i, protein in enumerate(proteins[:4]):  # First 4 are proteins
        # Create spatial gradient with noise
        base_expr = np.sin(coords[:, 0] / 20) * np.cos(coords[:, 1] / 20) + 1
        noise = np.random.exponential(scale=0.5, size=n_points)
        ion_counts[protein] = np.maximum(0, base_expr * (i + 1) * 10 + noise)
    
    # DNA channels (DAPI1, DAPI2) with tissue structure
    tissue_mask = (coords[:, 0] > 10) & (coords[:, 0] < 90) & (coords[:, 1] > 10) & (coords[:, 1] < 90)
    
    dna1_intensities = np.where(tissue_mask, 
                               np.random.gamma(2, 5, n_points), 
                               np.random.gamma(0.5, 1, n_points))
    
    dna2_intensities = np.where(tissue_mask,
                               np.random.gamma(2, 4, n_points),
                               np.random.gamma(0.5, 1, n_points))
    
    return coords, ion_counts, dna1_intensities, dna2_intensities


def test_grid_pipeline():
    """Test grid segmentation pipeline."""
    print("\n=== Testing Grid Pipeline ===")
    
    coords, ion_counts, dna1, dna2 = create_synthetic_imc_data()
    
    try:
        result = grid_pipeline(
            coords=coords,
            ion_counts=ion_counts,
            dna1_intensities=dna1,
            dna2_intensities=dna2,
            target_scale_um=20.0
        )
        
        print(f"✓ Grid pipeline completed successfully")
        print(f"  - Generated {result['n_segments_used']} grid cells")
        print(f"  - Method: {result['method']}")
        print(f"  - Has performance metrics: {'metrics' in result}")
        print(f"  - Has boundary quality: {'boundary_quality' in result}")
        
        # Check result structure
        required_keys = ['superpixel_counts', 'superpixel_coords', 'superpixel_labels', 
                        'composite_dna', 'bounds', 'transformed_arrays']
        missing_keys = [k for k in required_keys if k not in result]
        if missing_keys:
            print(f"⚠ Missing keys: {missing_keys}")
        else:
            print("✓ All required keys present")
        
        return True
        
    except Exception as e:
        print(f"✗ Grid pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison():
    """Test SLIC vs Grid comparison."""
    print("\n=== Testing SLIC vs Grid Comparison ===")
    
    coords, ion_counts, dna1, dna2 = create_synthetic_imc_data()
    
    try:
        comparison = compare_grid_vs_slic(
            coords=coords,
            ion_counts=ion_counts,
            dna1_intensities=dna1,
            dna2_intensities=dna2,
            target_scale_um=20.0
        )
        
        print("✓ Comparison completed successfully")
        
        perf = comparison['performance_comparison']
        print(f"  - Grid time: {perf['grid_time']:.3f}s")
        print(f"  - SLIC time: {perf['slic_time']:.3f}s") 
        print(f"  - Speedup: {perf['speedup_factor']:.2f}x")
        print(f"  - Grid segments: {perf['grid_segments']}")
        print(f"  - SLIC segments: {perf['slic_segments']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiscale_integration():
    """Test multiscale analysis integration."""
    print("\n=== Testing Multiscale Integration ===")
    
    coords, ion_counts, dna1, dna2 = create_synthetic_imc_data()
    
    # Test grid method
    try:
        grid_results = perform_multiscale_analysis(
            coords=coords,
            ion_counts=ion_counts,
            dna1_intensities=dna1,
            dna2_intensities=dna2,
            scales_um=[10.0, 20.0],  # Smaller test
            segmentation_method='grid'
        )
        
        print("✓ Grid multiscale analysis completed")
        scales_analyzed = [k for k in grid_results.keys() if isinstance(k, (int, float))]
        print(f"  - Analyzed scales: {scales_analyzed}")
        
        for scale in scales_analyzed:
            scale_result = grid_results[scale]
            print(f"  - {scale}μm: {scale_result.get('segmentation_method')} method")
            if 'grid_metrics' in scale_result:
                metrics = scale_result['grid_metrics']
                print(f"    - {metrics.n_grid_cells} cells, {metrics.boundary_coherence:.3f} coherence")
        
        return True
        
    except Exception as e:
        print(f"✗ Multiscale integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark():
    """Test benchmarking framework."""
    print("\n=== Testing Benchmark Framework ===")
    
    coords, ion_counts, dna1, dna2 = create_synthetic_imc_data(n_points=500)  # Smaller for speed
    
    try:
        benchmark = SegmentationBenchmark()
        
        # Test single method benchmark
        grid_result = benchmark.benchmark_single_method(
            'grid', coords, ion_counts, dna1, dna2, 20.0
        )
        
        print("✓ Single method benchmark completed")
        print(f"  - Method: {grid_result.method}")
        print(f"  - Time: {grid_result.total_time:.3f}s")
        print(f"  - Segments: {grid_result.n_segments}")
        print(f"  - Memory: {grid_result.memory_usage_mb:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_examples():
    """Test example functions."""
    print("\n=== Testing Example Functions ===")
    
    coords, ion_counts, dna1, dna2 = create_synthetic_imc_data(n_points=500)
    
    try:
        # Test basic comparison example
        result = example_basic_comparison(
            coords, ion_counts, dna1, dna2
        )
        
        print("✓ Basic comparison example completed")
        perf = result['performance_comparison']
        print(f"  - Speedup factor: {perf['speedup_factor']:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"✗ Examples failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Grid Segmentation Implementation Test")
    print("=" * 40)
    
    tests = [
        ("Grid Pipeline", test_grid_pipeline),
        ("SLIC vs Grid Comparison", test_comparison),
        ("Multiscale Integration", test_multiscale_integration),
        ("Benchmark Framework", test_benchmark),
        ("Example Functions", test_examples)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        else:
            print(f"⚠ {test_name} test failed")
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Grid segmentation implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())