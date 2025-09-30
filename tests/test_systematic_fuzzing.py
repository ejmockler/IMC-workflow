"""
Systematic Fuzzing Tests for Pipeline Robustness

Tests pipeline behavior under malformed, extreme, and edge case inputs.
Validates graceful failure and error handling rather than correctness.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import json
from pathlib import Path
from io import StringIO

from src.analysis.main_pipeline import IMCAnalysisPipeline
from src.analysis.ion_count_processing import apply_arcsinh_transform, aggregate_ion_counts
from src.config import Config


class TestMalformedInputRobustness:
    """Test pipeline robustness against malformed input data."""
    
    def test_corrupted_file_headers(self):
        """Test handling of files with corrupted headers."""
        malformed_files = [
            # Missing required columns
            "X\tY\tCD45\n1\t2\t100\n",
            # Duplicate column names  
            "X\tY\tX\tCD45\n1\t2\t3\t100\n",
            # Empty header
            "\n1\t2\t100\n",
            # Header with special characters
            "X(bad)\tY@#$\tCD45\n1\t2\t100\n",
            # Mixed separators
            "X,Y\tCD45\n1,2\t100\n"
        ]
        
        for i, content in enumerate(malformed_files):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                temp_path = f.name
            
            try:
                # Create minimal config
                config_data = {
                    "data": {"raw_data_dir": "tests/data"},
                    "channels": {"protein_channels": ["CD45"]},
                    "analysis": {"clustering": {"method": "leiden"}},
                    "output": {"results_dir": "/tmp"}
                }
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as cf:
                    json.dump(config_data, cf)
                    config_path = cf.name
                
                config = Config(config_path)
                pipeline = IMCAnalysisPipeline(config)
                
                # Should either handle gracefully or raise clear error
                try:
                    roi_data = pipeline.load_roi_data(temp_path, ["CD45"])
                    # If it succeeds, data should be valid
                    assert 'coords' in roi_data
                    assert 'ion_counts' in roi_data
                except Exception as e:
                    # Should be a clear, informative error, not a crash
                    assert len(str(e)) > 10, f"Error message too vague: {e}"
                    
            finally:
                Path(temp_path).unlink(missing_ok=True)
                Path(config_path).unlink(missing_ok=True)
    
    def test_extreme_coordinate_values(self):
        """Test handling of extreme coordinate values."""
        extreme_coords_cases = [
            # Very large coordinates
            np.array([[1e10, 1e10], [1e10 + 1, 1e10 + 1]]),
            # Very small coordinates  
            np.array([[1e-10, 1e-10], [2e-10, 2e-10]]),
            # Mixed extreme ranges
            np.array([[-1e6, 1e6], [1e6, -1e6]]),
            # Zero coordinates
            np.array([[0, 0], [0, 0]]),
        ]
        
        for coords in extreme_coords_cases:
            ion_counts = {'CD45': np.array([100.0, 200.0])}
            
            # Should handle without crashing
            try:
                # Test arcsinh transform
                transformed, _ = apply_arcsinh_transform(ion_counts)
                assert np.all(np.isfinite(transformed['CD45']))
                
                # Test aggregation with extreme coords
                bin_edges_x = np.linspace(np.min(coords[:, 0]) - 1, 
                                        np.max(coords[:, 0]) + 1, 5)
                bin_edges_y = np.linspace(np.min(coords[:, 1]) - 1, 
                                        np.max(coords[:, 1]) + 1, 5)
                
                aggregated = aggregate_ion_counts(coords, ion_counts, bin_edges_x, bin_edges_y)
                assert np.all(np.isfinite(aggregated['CD45']))
                
            except Exception as e:
                # Should be informative error, not silent failure
                assert "coordinate" in str(e).lower() or "range" in str(e).lower(), \
                    f"Unclear error for extreme coordinates: {e}"
    
    def test_pathological_ion_count_distributions(self):
        """Test handling of pathological ion count distributions."""
        pathological_cases = [
            # All zeros
            np.zeros(100),
            # All same value
            np.full(100, 42.0),
            # Extreme outliers
            np.concatenate([np.ones(99), [1e6]]),
            # Very large values
            np.full(10, 1e8),
            # Negative values (should not occur but test robustness)
            np.array([-1, -100, -1000, 0, 1]),
        ]
        
        for ion_values in pathological_cases:
            ion_counts = {'CD45': ion_values}
            
            try:
                # Transform should handle gracefully
                transformed, cofactors = apply_arcsinh_transform(ion_counts)
                
                # Results should be finite
                assert np.all(np.isfinite(transformed['CD45'])), \
                    f"Non-finite results for pathological input"
                
                # Cofactor should be positive and finite
                # For pathological data, cofactor may be large but should be reasonable relative to data
                cofactor = cofactors['CD45']
                max_data_value = np.max(np.abs(ion_values))
                assert 0 < cofactor <= max_data_value * 10, \
                    f"Cofactor {cofactor} unreasonable for data max {max_data_value}"
                    
            except Exception as e:
                # Should be clear error about data issues
                assert any(word in str(e).lower() for word in ['data', 'value', 'range']), \
                    f"Unclear error for pathological data: {e}"


class TestMemoryPressureFuzzing:
    """Test behavior under memory pressure and large datasets."""
    
    @pytest.mark.slow
    def test_large_dataset_handling(self):
        """Test pipeline behavior with large datasets."""
        # Create progressively larger datasets until memory constraints
        sizes = [1000, 5000, 10000]
        
        for size in sizes:
            try:
                # Generate large synthetic dataset
                coords = np.random.uniform(0, 1000, (size, 2))
                ion_counts = {
                    'CD45': np.random.poisson(100, size).astype(float),
                    'CD31': np.random.poisson(50, size).astype(float),
                    'CD11b': np.random.poisson(75, size).astype(float)
                }
                
                # Test memory-intensive operations
                transformed, _ = apply_arcsinh_transform(ion_counts)
                
                # Should complete without memory explosion
                assert len(transformed['CD45']) == size
                
                # Test aggregation (memory intensive for large datasets)
                bin_edges = np.linspace(0, 1000, 51)  # 50x50 grid
                aggregated = aggregate_ion_counts(coords, ion_counts, bin_edges, bin_edges)
                
                # Should complete and conserve counts
                original_total = sum(np.sum(counts) for counts in ion_counts.values())
                aggregated_total = sum(np.sum(counts) for counts in aggregated.values())
                
                assert abs(original_total - aggregated_total) < 1e-6
                
            except MemoryError:
                # Expected for very large datasets - should be graceful
                pytest.skip(f"Memory limit reached at size {size}")
            except Exception as e:
                # Should not crash with other errors for large data
                pytest.fail(f"Unexpected error for size {size}: {e}")
    
    def test_memory_cleanup_after_errors(self):
        """Test that memory is cleaned up after processing errors."""
        import gc
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Intentionally cause errors with large datasets
        for _ in range(5):
            try:
                # Create large array that should cause issues
                large_coords = np.random.uniform(0, 100, (50000, 2))
                large_ion_counts = {
                    f'protein_{i}': np.random.poisson(100, 50000).astype(float)
                    for i in range(20)  # Many proteins
                }
                
                # This might fail due to memory or other issues
                transformed, _ = apply_arcsinh_transform(large_ion_counts)
                
            except Exception:
                # Expected - just testing cleanup
                pass
            finally:
                # Force garbage collection
                gc.collect()
        
        # Memory should not have grown excessively
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Allow some growth but not excessive (less than 500MB)
        assert memory_growth < 500, \
            f"Excessive memory growth: {memory_growth:.1f}MB"


class TestConcurrencyRobustness:
    """Test pipeline behavior under concurrent access (if applicable)."""
    
    def test_deterministic_results_same_seed(self):
        """Test that same random seed produces identical results."""
        # Create test data
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        ion_counts = {'CD45': np.random.poisson(100, 100).astype(float)}
        
        # Run transformation multiple times with same seed
        results = []
        for _ in range(3):
            np.random.seed(42)  # Reset seed
            transformed, cofactors = apply_arcsinh_transform(ion_counts.copy())
            results.append((transformed['CD45'].copy(), cofactors['CD45']))
        
        # All results should be identical
        for i in range(1, len(results)):
            assert np.allclose(results[0][0], results[i][0]), \
                "Non-deterministic results with same seed"
            assert abs(results[0][1] - results[i][1]) < 1e-10, \
                "Non-deterministic cofactors with same seed"


class TestConfigurationFuzzing:
    """Test robustness against malformed configuration."""
    
    def test_invalid_config_structures(self):
        """Test handling of invalid configuration structures."""
        invalid_configs = [
            # Missing required sections
            {"data": {}},
            # Invalid data types
            {"channels": {"protein_channels": "not_a_list"}},
            # Negative numbers where inappropriate
            {"analysis": {"clustering": {"resolution_range": [-1, -2]}}},
            # Empty config
            {},
            # Circular references (if any)
            {"analysis": {"method": "analysis"}},
        ]
        
        for config_data in invalid_configs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                config_path = f.name
            
            try:
                # Should either handle gracefully or give clear error
                config = Config(config_path)
                pipeline = IMCAnalysisPipeline(config)
                
                # If it initializes, basic operations should work
                assert hasattr(pipeline, 'analysis_config')
                
            except Exception as e:
                # Should be informative error about configuration
                assert any(word in str(e).lower() for word in ['config', 'missing', 'invalid']), \
                    f"Unclear config error: {e}"
            finally:
                Path(config_path).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])