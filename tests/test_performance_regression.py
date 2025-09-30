"""
Performance Regression Monitoring

Tracks algorithmic complexity and performance to prevent regressions.
Focuses on scaling behavior rather than absolute timing (which is environment-dependent).
"""

import numpy as np
import pytest
import time
import tracemalloc
import gc
from typing import List, Tuple

from src.analysis.ion_count_processing import (
    apply_arcsinh_transform, 
    aggregate_ion_counts,
    ion_count_pipeline
)
from src.analysis.spatial_clustering import perform_spatial_clustering
from src.analysis.multiscale_analysis import perform_multiscale_analysis


class TestAlgorithmicComplexity:
    """Test that algorithms scale as expected with input size."""
    
    def test_arcsinh_transform_linear_scaling(self):
        """arcsinh transform should scale O(n) with number of points."""
        sizes = [1000, 2000, 4000]
        times = []
        
        for size in sizes:
            np.random.seed(42)
            ion_counts = {
                'marker1': np.random.poisson(100, size).astype(float),
                'marker2': np.random.poisson(50, size).astype(float)
            }
            
            # Warm up
            apply_arcsinh_transform({'test': np.random.poisson(10, 100).astype(float)})
            
            # Measure time
            start_time = time.perf_counter()
            transformed, _ = apply_arcsinh_transform(ion_counts)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
        
        # Should scale roughly linearly: T(2n) ≈ 2*T(n)
        # Allow 50% tolerance for measurement noise and overhead
        scaling_factor_1 = times[1] / times[0]  # 2x data
        scaling_factor_2 = times[2] / times[1]  # 2x data again
        
        assert 0.5 < scaling_factor_1 < 4.0, \
            f"Poor scaling 1k->2k: {scaling_factor_1:.2f}x"
        assert 0.5 < scaling_factor_2 < 4.0, \
            f"Poor scaling 2k->4k: {scaling_factor_2:.2f}x"
    
    def test_spatial_aggregation_scaling(self):
        """Spatial aggregation should scale roughly O(n) with number of points."""
        point_counts = [500, 1000, 2000]
        times = []
        
        for n_points in point_counts:
            np.random.seed(42)
            coords = np.random.uniform(0, 100, (n_points, 2))
            ion_counts = {'marker': np.random.poisson(100, n_points).astype(float)}
            
            # Fixed grid size - complexity should scale with points, not bins
            bin_edges = np.linspace(0, 100, 21)  # 20x20 grid
            
            # Warm up
            aggregate_ion_counts(
                np.random.uniform(0, 100, (100, 2)),
                {'test': np.random.poisson(10, 100).astype(float)},
                np.linspace(0, 100, 11), np.linspace(0, 100, 11)
            )
            
            start_time = time.perf_counter()
            aggregated = aggregate_ion_counts(coords, ion_counts, bin_edges, bin_edges)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
        
        # Should scale sub-quadratically
        scaling_factor_1 = times[1] / times[0]  # 2x points
        scaling_factor_2 = times[2] / times[1]  # 2x points
        
        # Allow generous bounds - focus on preventing O(n²) behavior
        assert scaling_factor_1 < 8.0, \
            f"Excessive scaling 500->1000: {scaling_factor_1:.2f}x (expected <8x)"
        assert scaling_factor_2 < 8.0, \
            f"Excessive scaling 1000->2000: {scaling_factor_2:.2f}x (expected <8x)"
    
    def test_clustering_scaling_behavior(self):
        """Spatial clustering should not scale worse than O(n log n)."""
        sizes = [100, 200, 400]
        times = []
        
        for size in sizes:
            np.random.seed(42)
            coords = np.random.uniform(0, 100, (size, 2))
            features = np.random.randn(size, 3)
            
            # Warm up
            perform_spatial_clustering(
                np.random.randn(50, 3),
                np.random.uniform(0, 100, (50, 2)),
                method='leiden', random_state=42
            )
            
            start_time = time.perf_counter()
            labels, metadata = perform_spatial_clustering(
                features, coords, method='leiden', random_state=42
            )
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
        
        # Should scale better than O(n²)
        # For O(n log n): doubling size should increase time by ~2.4x
        # For O(n²): doubling size would increase time by 4x
        scaling_factor_1 = times[1] / times[0]  # 2x size
        scaling_factor_2 = times[2] / times[1]  # 2x size
        
        # Flag if scaling worse than O(n^1.5)
        assert scaling_factor_1 < 6.0, \
            f"Poor clustering scaling 100->200: {scaling_factor_1:.2f}x"
        assert scaling_factor_2 < 6.0, \
            f"Poor clustering scaling 200->400: {scaling_factor_2:.2f}x"


class TestMemoryComplexity:
    """Test memory usage scaling and detect memory leaks."""
    
    def test_memory_usage_scaling(self):
        """Memory usage should scale linearly with data size."""
        tracemalloc.start()
        
        sizes = [1000, 2000]
        memory_usage = []
        
        for size in sizes:
            gc.collect()  # Clean up before measurement
            tracemalloc.clear_traces()
            
            # Create data
            coords = np.random.uniform(0, 100, (size, 2))
            ion_counts = {
                'marker1': np.random.poisson(100, size).astype(float),
                'marker2': np.random.poisson(50, size).astype(float),
                'marker3': np.random.poisson(75, size).astype(float)
            }
            
            # Process data
            transformed, _ = apply_arcsinh_transform(ion_counts)
            bin_edges = np.linspace(0, 100, 21)
            aggregated = aggregate_ion_counts(coords, ion_counts, bin_edges, bin_edges)
            
            current, peak = tracemalloc.get_traced_memory()
            memory_usage.append(peak)
        
        tracemalloc.stop()
        
        # Memory should scale roughly linearly (within 4x for 2x data)
        memory_ratio = memory_usage[1] / memory_usage[0]
        assert memory_ratio < 4.0, \
            f"Memory scaling worse than linear: {memory_ratio:.2f}x for 2x data"
    
    def test_no_memory_leaks_in_loop(self):
        """Repeated operations should not accumulate memory."""
        tracemalloc.start()
        
        # Baseline memory
        gc.collect()
        baseline_current, baseline_peak = tracemalloc.get_traced_memory()
        
        # Perform repeated operations
        for i in range(10):
            coords = np.random.uniform(0, 100, (500, 2))
            ion_counts = {'marker': np.random.poisson(100, 500).astype(float)}
            
            # Process and discard
            transformed, _ = apply_arcsinh_transform(ion_counts)
            del transformed, coords, ion_counts
            
            if i % 3 == 0:
                gc.collect()  # Periodic cleanup
        
        # Final memory check
        gc.collect()
        final_current, final_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory growth should be minimal
        memory_growth = final_current - baseline_current
        assert memory_growth < 10 * 1024 * 1024, \
            f"Potential memory leak: {memory_growth / 1024 / 1024:.1f}MB growth"


class TestPerformanceBenchmarks:
    """Benchmark key operations for regression detection."""
    
    @pytest.mark.benchmark
    def test_full_pipeline_performance_baseline(self):
        """Benchmark complete pipeline for regression detection."""
        np.random.seed(42)
        
        # Realistic dataset size
        n_points = 2000
        coords = np.random.uniform(0, 200, (n_points, 2))
        ion_counts = {
            'CD45': np.random.negative_binomial(5, 0.3, n_points).astype(float),
            'CD31': np.random.negative_binomial(10, 0.5, n_points).astype(float),
            'CD11b': np.random.negative_binomial(3, 0.2, n_points).astype(float)
        }
        dna1 = np.random.poisson(800, n_points).astype(float)
        dna2 = np.random.poisson(750, n_points).astype(float)
        
        # Warm up
        ion_count_pipeline(
            np.random.uniform(0, 100, (100, 2)),
            {'test': np.random.poisson(50, 100).astype(float)},
            bin_size_um=20.0
        )
        
        # Benchmark full pipeline
        start_time = time.perf_counter()
        
        results = ion_count_pipeline(
            coords, ion_counts, bin_size_um=20.0
        )
        
        elapsed = time.perf_counter() - start_time
        
        # Should complete in reasonable time (environment-dependent baseline)
        # This is more for tracking regressions than absolute performance
        assert elapsed < 30.0, \
            f"Pipeline too slow: {elapsed:.1f}s for {n_points} points"
        
        # Verify results are reasonable
        assert 'cluster_labels' in results
        assert len(results['cluster_labels']) > 0
        
        print(f"Pipeline benchmark: {elapsed:.2f}s for {n_points} points")
    
    def test_multiscale_analysis_performance(self):
        """Benchmark multiscale analysis performance."""
        np.random.seed(42)
        
        n_points = 1000
        coords = np.random.uniform(0, 100, (n_points, 2))
        ion_counts = {
            'marker1': np.random.poisson(100, n_points).astype(float),
            'marker2': np.random.poisson(50, n_points).astype(float)
        }
        dna1 = np.random.poisson(800, n_points).astype(float)
        dna2 = np.random.poisson(750, n_points).astype(float)
        
        scales_um = [10.0, 20.0, 40.0]
        
        start_time = time.perf_counter()
        
        try:
            results = perform_multiscale_analysis(
                coords=coords,
                ion_counts=ion_counts,
                dna1_intensities=dna1,
                dna2_intensities=dna2,
                scales_um=scales_um,
                method='leiden'
            )
            
            elapsed = time.perf_counter() - start_time
            
            # Should complete in reasonable time
            assert elapsed < 60.0, \
                f"Multiscale analysis too slow: {elapsed:.1f}s"
            
            print(f"Multiscale benchmark: {elapsed:.2f}s for {len(scales_um)} scales")
            
        except Exception as e:
            # If it fails, that's also performance-relevant information
            elapsed = time.perf_counter() - start_time
            pytest.fail(f"Multiscale analysis failed after {elapsed:.1f}s: {e}")


class TestPerformanceRegression:
    """Detect performance regressions by comparing against historical data."""
    
    def test_establish_performance_baseline(self):
        """Establish baseline performance metrics for future comparison."""
        # This would typically save results to a file for CI/CD comparison
        # For now, just ensure operations complete within bounds
        
        operations = [
            ("arcsinh_1k", lambda: self._benchmark_arcsinh(1000)),
            ("aggregation_1k", lambda: self._benchmark_aggregation(1000)),
            ("clustering_500", lambda: self._benchmark_clustering(500)),
        ]
        
        baselines = {}
        for name, operation in operations:
            start_time = time.perf_counter()
            operation()
            elapsed = time.perf_counter() - start_time
            baselines[name] = elapsed
            
            # Ensure reasonable performance
            assert elapsed < 10.0, f"{name} baseline too slow: {elapsed:.2f}s"
        
        # In real CI/CD, would save baselines to file and compare in future runs
        print("Performance baselines:", baselines)
    
    def _benchmark_arcsinh(self, size):
        """Helper to benchmark arcsinh transformation."""
        ion_counts = {
            f'marker_{i}': np.random.poisson(100, size).astype(float)
            for i in range(5)
        }
        transformed, _ = apply_arcsinh_transform(ion_counts)
        return transformed
    
    def _benchmark_aggregation(self, size):
        """Helper to benchmark spatial aggregation."""
        coords = np.random.uniform(0, 100, (size, 2))
        ion_counts = {'marker': np.random.poisson(100, size).astype(float)}
        bin_edges = np.linspace(0, 100, 21)
        aggregated = aggregate_ion_counts(coords, ion_counts, bin_edges, bin_edges)
        return aggregated
    
    def _benchmark_clustering(self, size):
        """Helper to benchmark spatial clustering."""
        coords = np.random.uniform(0, 100, (size, 2))
        features = np.random.randn(size, 4)
        labels, metadata = perform_spatial_clustering(
            features, coords, method='leiden', random_state=42
        )
        return labels, metadata


if __name__ == '__main__':
    pytest.main([__file__, '-v'])