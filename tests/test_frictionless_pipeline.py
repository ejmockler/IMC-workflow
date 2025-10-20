#!/usr/bin/env python3
"""
Test script for the frictionless IMC pipeline.

Tests the one-button pipeline with mock data to ensure it works correctly.
"""

import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import json
import sys
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.analysis.frictionless_pipeline import (
    run_frictionless_analysis,
    run_frictionless_analysis_fast,
    FrictionlessConfig,
    FrictionlessPipeline
)


def create_mock_imc_data(output_dir: Path, n_rois: int = 3, n_cells_per_roi: int = 1000):
    """Create mock IMC data files for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Common protein markers
    proteins = ['CD45', 'CD31', 'CD11b', 'CD206', 'CD3', 'CD4', 'CD8', 'CD20', 'PanCK']
    
    created_files = []
    
    for roi_idx in range(n_rois):
        # Generate mock spatial coordinates
        roi_width, roi_height = 800, 600
        x_coords = np.random.uniform(0, roi_width, n_cells_per_roi)
        y_coords = np.random.uniform(0, roi_height, n_cells_per_roi)
        
        # Generate mock protein expression data
        data = {
            'X': x_coords,
            'Y': y_coords
        }
        
        # Add protein channels with realistic distributions
        for protein in proteins:
            # Simulate different expression patterns
            if protein in ['CD45', 'CD31']:  # Highly expressed
                base_expression = np.random.exponential(200, n_cells_per_roi)
            elif protein in ['CD3', 'CD4', 'CD8']:  # T cell markers - subset
                base_expression = np.random.exponential(50, n_cells_per_roi)
                # Make some cells highly positive
                high_positive = np.random.choice(n_cells_per_roi, n_cells_per_roi // 4, replace=False)
                base_expression[high_positive] *= 5
            else:  # Other markers
                base_expression = np.random.exponential(100, n_cells_per_roi)
            
            # Add measurement noise
            noise = np.random.normal(0, 5, n_cells_per_roi)
            expression = np.maximum(0, base_expression + noise)
            
            data[f'{protein}(143Nd)'] = expression  # Add mass tag
        
        # Add DNA channels
        data['DNA1(191Ir)'] = np.random.exponential(300, n_cells_per_roi)
        data['DNA2(193Ir)'] = np.random.exponential(280, n_cells_per_roi)
        
        # Add calibration channels
        data['130Ba'] = np.random.exponential(50, n_cells_per_roi)
        data['131Xe'] = np.random.exponential(45, n_cells_per_roi)
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        roi_file = output_dir / f"ROI_{roi_idx:03d}_mock.txt"
        df.to_csv(roi_file, sep='\t', index=False)
        created_files.append(roi_file)
        
        print(f"Created mock ROI file: {roi_file} ({n_cells_per_roi} cells)")
    
    return created_files


def test_basic_pipeline():
    """Test basic frictionless pipeline functionality."""
    print("=== Testing Basic Frictionless Pipeline ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock data
        print("Creating mock IMC data...")
        mock_files = create_mock_imc_data(temp_path / "mock_imc_data", n_rois=2, n_cells_per_roi=500)
        
        # Test pipeline
        print("Running frictionless analysis...")
        start_time = time.time()
        
        try:
            results = run_frictionless_analysis_fast(temp_path / "mock_imc_data")
            
            processing_time = time.time() - start_time
            print(f"✅ Pipeline completed in {processing_time:.2f} seconds")
            
            # Validate results
            assert 'frictionless_pipeline_metadata' in results
            assert 'analysis_results' in results
            assert 'qc_results' in results
            assert 'performance_metrics' in results
            
            # Check performance
            under_time_limit = results['frictionless_pipeline_metadata']['under_time_limit']
            print(f"✅ Under time limit: {under_time_limit}")
            
            # Check analysis results
            analysis_results = results['analysis_results']['analysis_results']
            print(f"✅ Analyzed {len(analysis_results)} ROIs")
            
            # Check QC results
            qc_pass_rate = results['qc_results']['overall_pass_rate']
            print(f"✅ QC pass rate: {qc_pass_rate:.1%}")
            
            print("✅ Basic pipeline test PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Basic pipeline test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_configuration_options():
    """Test different configuration options."""
    print("\n=== Testing Configuration Options ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock data
        mock_files = create_mock_imc_data(temp_path / "mock_imc_data", n_rois=2, n_cells_per_roi=300)
        
        # Test custom configuration
        config = FrictionlessConfig(
            max_processing_time_minutes=3.0,
            parallel_processing=False,  # Sequential for testing
            generate_plots=False,  # Faster
            continue_on_errors=True
        )
        
        pipeline = FrictionlessPipeline(config)
        
        try:
            results = pipeline.run_analysis(temp_path / "mock_imc_data")
            
            # Validate custom config worked
            assert not results['frictionless_pipeline_metadata']['config_used']['generate_plots']
            assert not results['frictionless_pipeline_metadata']['config_used']['parallel_processing']
            
            print("✅ Custom configuration test PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Configuration test FAILED: {e}")
            return False


def test_error_handling():
    """Test error handling with invalid data."""
    print("\n=== Testing Error Handling ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create invalid data directory (empty)
        invalid_dir = temp_path / "empty_data"
        invalid_dir.mkdir()
        
        config = FrictionlessConfig(continue_on_errors=True)
        pipeline = FrictionlessPipeline(config)
        
        try:
            results = pipeline.run_analysis(invalid_dir)
            
            # Should return error results, not crash
            assert 'error' in results or not results.get('success', True)
            print("✅ Error handling test PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Error handling test FAILED: {e}")
            return False


def test_performance_optimization():
    """Test performance optimization features."""
    print("\n=== Testing Performance Optimization ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create larger dataset
        mock_files = create_mock_imc_data(temp_path / "large_imc_data", n_rois=5, n_cells_per_roi=800)
        
        # Test with performance monitoring
        config = FrictionlessConfig(
            max_processing_time_minutes=10.0,
            parallel_processing=True,
            enable_chunked_processing=True
        )
        
        pipeline = FrictionlessPipeline(config)
        
        try:
            start_time = time.time()
            results = pipeline.run_analysis(temp_path / "large_imc_data")
            total_time = time.time() - start_time
            
            # Check performance metrics
            perf_metrics = results['performance_metrics']
            assert 'total_time' in perf_metrics
            assert 'roi_count' in perf_metrics
            
            print(f"✅ Processed {perf_metrics['roi_count']} ROIs in {total_time:.2f} seconds")
            print("✅ Performance optimization test PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Performance test FAILED: {e}")
            return False


def main():
    """Run all tests."""
    print("🧪 FRICTIONLESS PIPELINE TEST SUITE 🧪\n")
    
    tests = [
        test_basic_pipeline,
        test_configuration_options,
        test_error_handling,
        test_performance_optimization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests
    
    print("=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Frictionless pipeline is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)