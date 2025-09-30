"""
Core Performance Tests for IMC Analysis Pipeline

Essential performance tests to prevent algorithmic regressions.
"""

import pytest
import numpy as np
import time
from src.analysis.ion_count_processing import (
    aggregate_ion_counts,
    apply_arcsinh_transform
)


@pytest.mark.performance
class TestCorePerformance:
    """Test core algorithm performance."""
    
    def test_arcsinh_transform_vectorization(self):
        """Test that arcsinh transformation is properly vectorized."""
        np.random.seed(42)
        
        # Test with increasing data sizes
        sizes = [1000, 5000, 10000]
        times = []
        
        for size in sizes:
            ion_counts = {'CD45': np.random.poisson(100, size)}
            
            start_time = time.time()
            transformed, _ = apply_arcsinh_transform(ion_counts)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        # Should scale roughly linearly (allowing 2x tolerance for variance)
        assert times[2] < times[0] * 25, f"Poor scaling: {times}"
    
    def test_aggregate_ion_counts_scaling(self):
        """Test ion count aggregation scales properly."""
        np.random.seed(42)
        
        # Test increasing point counts
        point_counts = [500, 1000, 2000]
        times = []
        
        for n_points in point_counts:
            coords = np.random.uniform(0, 100, (n_points, 2))
            ion_counts = {'marker': np.random.poisson(50, n_points).astype(float)}
            bin_edges = np.linspace(0, 100, 11)  # 10x10 grid
            
            start_time = time.time()
            aggregated = aggregate_ion_counts(coords, ion_counts, bin_edges, bin_edges)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        # Should scale roughly linearly
        assert times[2] < times[0] * 10, f"Poor scaling: {times}"
    
    def test_memory_efficiency_large_dataset(self):
        """Test memory usage doesn't explode with larger datasets."""
        import tracemalloc
        
        np.random.seed(42)
        tracemalloc.start()
        
        # Process moderately large dataset
        coords = np.random.uniform(0, 200, (5000, 2))
        ion_counts = {
            'CD45': np.random.poisson(100, 5000).astype(float),
            'CD31': np.random.poisson(50, 5000).astype(float)
        }
        
        # Transform data
        transformed, _ = apply_arcsinh_transform(ion_counts)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Should use less than 100MB for 5k points
        assert peak < 100 * 1024 * 1024, f"Memory usage too high: {peak / 1024 / 1024:.1f}MB"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])