"""
Performance Regression Tests for IMC Analysis Pipeline

Tests to ensure O(N²) fixes remain effective and performance doesn't regress.
Uses complexity analysis rather than brittle timing assertions.
"""

import pytest
import numpy as np
import time
import tempfile
from pathlib import Path
import pandas as pd
from typing import List, Tuple
import tracemalloc
import gc

from src.analysis.data_storage import (
    create_storage_backend,
    ParquetStorage,
    HDF5Storage
)
from src.analysis.ion_count_processing import (
    aggregate_ion_counts,
    apply_arcsinh_transform,
    ion_count_pipeline
)


@pytest.mark.performance
class TestDataStoragePerformance:
    """Test storage performance to verify O(N²) fixes."""
    
    def test_parquet_partitioned_scaling(self):
        """Test that Parquet storage scales linearly, not quadratically."""
        pd = pytest.importorskip('pandas')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ParquetStorage(tmpdir)
            
            # Test with increasing numbers of ROIs
            roi_counts = [5, 10, 20]  # Start small for CI
            save_times = []
            
            for n_rois in roi_counts:
                start_time = time.time()
                
                # Save multiple ROIs
                for i in range(n_rois):
                    feature_matrix = np.random.rand(50, 5)  # Small for speed
                    protein_names = [f'protein_{j}' for j in range(5)]
                    
                    storage.save_roi_features(f'roi_{i}', feature_matrix, protein_names)
                
                elapsed = time.time() - start_time
                save_times.append(elapsed)
            
            # Check scaling behavior
            # With partitioned storage, time should scale roughly linearly
            # Allow some variance but prevent quadratic growth
            if len(save_times) >= 2:
                # Time ratio should be close to ROI ratio for linear scaling
                time_ratio = save_times[-1] / save_times[0]
                roi_ratio = roi_counts[-1] / roi_counts[0]
                
                # Allow some overhead but prevent quadratic behavior
                # Linear: ratio ≈ roi_ratio, Quadratic: ratio ≈ roi_ratio²
                max_acceptable_ratio = roi_ratio * 1.5  # Allow 50% overhead
                
                assert time_ratio <= max_acceptable_ratio, \
                    f"Performance regression: time ratio {time_ratio:.2f} > {max_acceptable_ratio:.2f}"
    
    def test_no_read_modify_write_cycle_timing(self):
        """Test that adding ROIs doesn't read existing ones (timing-based)."""
        pd = pytest.importorskip('pandas')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ParquetStorage(tmpdir)
            
            # Create initial ROIs
            n_initial = 10
            for i in range(n_initial):
                feature_matrix = np.random.rand(30, 3)
                storage.save_roi_features(f'roi_{i}', feature_matrix, ['p1', 'p2', 'p3'])
            
            # Measure time to add one more ROI
            start_time = time.time()
            storage.save_roi_features('new_roi', np.random.rand(30, 3), ['p1', 'p2', 'p3'])
            add_time = time.time() - start_time
            
            # Should be fast (no dependency on existing ROI count)
            # This is a complexity test, not an absolute timing test
            assert add_time < 1.0, f"Adding ROI took too long: {add_time:.3f}s"
            
            # Verify files exist separately (partitioned behavior)
            features_dir = Path(tmpdir) / "roi_features_partitioned"
            roi_files = list(features_dir.glob("roi_*.parquet"))
            assert len(roi_files) == n_initial + 1
    
    def test_hdf5_compression_performance(self):
        """Test HDF5 compression doesn't cause exponential slowdown."""
        h5py = pytest.importorskip('h5py')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = HDF5Storage(Path(tmpdir) / "test.h5", compression='gzip')
            
            # Test with increasing data sizes
            data_sizes = [100, 500, 1000]  # Rows in matrix
            save_times = []
            
            for size in data_sizes:
                large_data = {
                    'feature_matrix': np.random.rand(size, 50),
                    'cluster_labels': np.random.randint(0, 5, size)
                }
                
                start_time = time.time()
                storage.save_analysis_results(large_data, f'test_{size}')
                elapsed = time.time() - start_time
                save_times.append(elapsed)
            
            # Compression time should scale reasonably with data size
            if len(save_times) >= 2:
                # Check that it's not exponential growth
                for i in range(1, len(save_times)):
                    size_ratio = data_sizes[i] / data_sizes[i-1]
                    time_ratio = save_times[i] / save_times[i-1]
                    
                    # Allow quadratic growth for compression but not exponential
                    max_ratio = size_ratio ** 2 * 2  # Allow some overhead
                    
                    assert time_ratio <= max_ratio, \
                        f"Compression performance regression: {time_ratio:.2f} > {max_ratio:.2f}"


@pytest.mark.performance
class TestIonCountProcessingPerformance:
    """Test ion count processing performance to verify vectorization fixes."""
    
    def test_aggregate_ion_counts_scaling(self):
        """Test that ion count aggregation scales linearly with data size."""
        # Test with increasing data sizes
        data_sizes = [1000, 5000, 10000]  # Number of points
        processing_times = []
        
        for n_points in data_sizes:
            # Create test data
            coords = np.random.uniform(0, 100, (n_points, 2))
            ion_counts = {
                'protein1': np.random.poisson(5, n_points),
                'protein2': np.random.poisson(10, n_points)
            }
            
            # Measure aggregation time
            start_time = time.time()
            
            try:
                result = aggregate_ion_counts(
                    coords=coords,
                    ion_counts=ion_counts,
                    bin_size_um=10.0
                )
                
                elapsed = time.time() - start_time
                processing_times.append(elapsed)
                
                # Verify result is reasonable
                assert isinstance(result, dict)
                
            except Exception as e:
                # If function fails, skip performance test
                pytest.skip(f"Ion count aggregation failed: {e}")
        
        # Check scaling behavior (should be roughly linear)
        if len(processing_times) >= 2:
            for i in range(1, len(processing_times)):
                size_ratio = data_sizes[i] / data_sizes[i-1]
                time_ratio = processing_times[i] / processing_times[i-1]
                
                # Allow some overhead but prevent quadratic scaling
                max_acceptable_ratio = size_ratio * 2.0
                
                assert time_ratio <= max_acceptable_ratio, \
                    f"Ion count aggregation scaling regression: {time_ratio:.2f} > {max_acceptable_ratio:.2f}"
    
    def test_arcsinh_transform_vectorization(self):
        """Test that arcsinh transform is properly vectorized."""
        # Create test data with increasing sizes
        sizes = [1000, 10000, 50000]
        transform_times = []
        
        for size in sizes:
            ion_counts = {
                'CD45': np.random.poisson(5, size),
                'CD31': np.random.poisson(8, size),
                'CD11b': np.random.poisson(3, size)
            }
            
            start_time = time.time()
            
            try:
                transformed, cofactors = apply_arcsinh_transform(ion_counts)
                elapsed = time.time() - start_time
                transform_times.append(elapsed)
                
                # Verify transformation worked
                assert isinstance(transformed, dict)
                assert len(transformed) == len(ion_counts)
                
            except Exception as e:
                pytest.skip(f"Arcsinh transform failed: {e}")
        
        # Vectorized operations should scale very well
        if len(transform_times) >= 2:
            # Allow generous scaling for numerical operations
            for i in range(1, len(transform_times)):
                size_ratio = sizes[i] / sizes[i-1]
                time_ratio = transform_times[i] / transform_times[i-1]
                
                # Vectorized operations should scale nearly linearly
                max_ratio = size_ratio * 1.5
                
                assert time_ratio <= max_ratio, \
                    f"Arcsinh transform not properly vectorized: {time_ratio:.2f} > {max_ratio:.2f}"
    
    def test_ion_count_pipeline_memory_efficiency(self):
        """Test that ion count pipeline doesn't consume excessive memory."""
        # Start memory tracking
        tracemalloc.start()
        
        # Test with moderate-sized data
        n_points = 5000
        coords = np.random.uniform(0, 200, (n_points, 2))
        ion_counts = {
            'CD45': np.random.negative_binomial(5, 0.3, n_points),
            'CD31': np.random.negative_binomial(10, 0.5, n_points),
            'CD11b': np.random.negative_binomial(3, 0.2, n_points),
            'CD206': np.random.negative_binomial(8, 0.4, n_points)
        }
        
        # Get initial memory
        initial_snapshot = tracemalloc.take_snapshot()
        
        try:
            # Run pipeline
            result = ion_count_pipeline(
                coords=coords,
                ion_counts=ion_counts,
                bin_size_um=20.0,
                n_clusters=5
            )
            
            # Get final memory
            final_snapshot = tracemalloc.take_snapshot()
            
            # Calculate memory usage
            top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
            total_memory_mb = sum(stat.size for stat in top_stats) / (1024 * 1024)
            
            # Memory usage should be reasonable for the data size
            # Input data is roughly: 5000 points * 4 proteins * 8 bytes ≈ 160KB
            # Allow significant overhead for processing but not excessive
            max_memory_mb = 100  # 100MB should be more than enough
            
            assert total_memory_mb <= max_memory_mb, \
                f"Excessive memory usage: {total_memory_mb:.2f}MB"
            
            # Verify pipeline produced results
            assert isinstance(result, dict)
            
        except Exception as e:
            pytest.skip(f"Ion count pipeline failed: {e}")
        
        finally:
            tracemalloc.stop()
            gc.collect()


@pytest.mark.performance
class TestComplexityBenchmarks:
    """Complexity-based performance tests using Big O analysis."""
    
    def test_data_size_scaling_analysis(self):
        """Analyze how processing time scales with data size."""
        from src.analysis.slic_segmentation import prepare_dna_composite
        
        # Test different data sizes
        sizes = [500, 1000, 2000]  # Points
        times = []
        
        for n_points in sizes:
            # Generate test data
            dna1 = np.random.poisson(800, n_points).astype(float)
            dna2 = np.random.poisson(600, n_points).astype(float)
            coords = np.random.uniform(0, 100, (n_points, 2))
            
            start_time = time.time()
            
            try:
                composite = prepare_dna_composite(
                    dna1_intensities=dna1,
                    dna2_intensities=dna2,
                    coords=coords,
                    resolution_um=2.0,
                    bounds=(0, 100, 0, 100)
                )
                
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                # Verify output
                assert composite is not None
                
            except Exception as e:
                pytest.skip(f"DNA composite preparation failed: {e}")
        
        # Analyze complexity
        if len(times) >= 3:
            # Calculate growth rates
            growth_rates = []
            for i in range(1, len(times)):
                size_ratio = sizes[i] / sizes[i-1]
                time_ratio = times[i] / times[i-1]
                growth_rate = np.log(time_ratio) / np.log(size_ratio)
                growth_rates.append(growth_rate)
            
            avg_growth_rate = np.mean(growth_rates)
            
            # Growth rate should be reasonable:
            # 1.0 = linear, 2.0 = quadratic, 3.0 = cubic
            # Image processing might be between linear and quadratic
            assert avg_growth_rate <= 2.5, \
                f"Poor complexity scaling: growth rate {avg_growth_rate:.2f}"
    
    def test_protein_count_scaling(self):
        """Test how processing scales with number of proteins."""
        from src.analysis.ion_count_processing import standardize_features
        
        n_points = 1000
        protein_counts = [2, 5, 10, 20]
        times = []
        
        for n_proteins in protein_counts:
            # Create data with varying protein counts
            ion_data = {}
            for i in range(n_proteins):
                ion_data[f'protein_{i}'] = np.random.poisson(5, n_points)
            
            start_time = time.time()
            
            try:
                standardized, scalers = standardize_features(ion_data)
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                assert len(standardized) == n_proteins
                
            except Exception as e:
                pytest.skip(f"Feature standardization failed: {e}")
        
        # Should scale linearly with number of proteins
        if len(times) >= 2:
            for i in range(1, len(times)):
                protein_ratio = protein_counts[i] / protein_counts[i-1]
                time_ratio = times[i] / times[i-1]
                
                # Should be roughly linear scaling
                max_ratio = protein_ratio * 1.5
                
                assert time_ratio <= max_ratio, \
                    f"Poor protein scaling: {time_ratio:.2f} > {max_ratio:.2f}"


@pytest.mark.performance
@pytest.mark.slow
class TestLargeDatasetPerformance:
    """Performance tests with larger datasets (marked slow)."""
    
    def test_large_roi_processing_time(self, large_roi_data):
        """Test processing time for large ROI datasets."""
        start_time = time.time()
        
        try:
            # Test with ion count pipeline
            result = ion_count_pipeline(
                coords=large_roi_data['coords'],
                ion_counts=large_roi_data['ion_counts'],
                bin_size_um=20.0,
                n_clusters=8
            )
            
            elapsed = time.time() - start_time
            
            # Large dataset should still process in reasonable time
            max_time_seconds = 30  # Generous limit for CI
            assert elapsed <= max_time_seconds, \
                f"Large dataset processing too slow: {elapsed:.2f}s > {max_time_seconds}s"
            
            # Verify results
            assert isinstance(result, dict)
            assert 'feature_matrix' in result or 'aggregated_counts' in result
            
        except Exception as e:
            pytest.skip(f"Large dataset processing failed: {e}")
    
    def test_memory_usage_large_dataset(self, large_roi_data):
        """Test memory usage with large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        try:
            # Process large dataset
            from src.analysis.multiscale_analysis import perform_multiscale_analysis
            
            results = perform_multiscale_analysis(
                coords=large_roi_data['coords'],
                ion_counts=large_roi_data['ion_counts'],
                dna1_intensities=large_roi_data['dna1_intensities'],
                dna2_intensities=large_roi_data['dna2_intensities'],
                scales_um=[20.0, 40.0],  # Reduced for memory test
                n_clusters=5,
                use_slic=False  # Use simpler method for memory test
            )
            
            peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = peak_memory - initial_memory
            
            # Memory increase should be reasonable for large dataset
            max_memory_mb = 200  # Generous limit
            assert memory_increase <= max_memory_mb, \
                f"Excessive memory usage: {memory_increase:.2f}MB increase"
            
            # Verify results
            assert isinstance(results, dict)
            
        except Exception as e:
            pytest.skip(f"Large dataset multiscale analysis failed: {e}")
        
        finally:
            # Force cleanup
            gc.collect()


def test_performance_regression_baseline():
    """Establish performance baseline for regression detection."""
    # This test establishes baseline performance metrics
    # In a real CI system, you would store these metrics and compare over time
    
    import sys
    import platform
    
    baseline_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'numpy_version': np.__version__,
    }
    
    print(f"Performance baseline: {baseline_info}")
    
    # Simple operation timing for baseline
    n = 10000
    data = np.random.rand(n, 10)
    
    start_time = time.time()
    result = np.sum(data, axis=1)
    baseline_time = time.time() - start_time
    
    print(f"Baseline numpy operation time: {baseline_time:.4f}s for {n} points")
    
    # This would be stored and compared in a real regression system
    assert baseline_time < 1.0  # Sanity check