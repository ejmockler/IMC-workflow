"""
Tests for ion count processing functions.
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, Tuple
from unittest.mock import Mock, patch

# Import functions to test
from src.analysis.ion_count_processing import (
    aggregate_ion_counts,
    apply_arcsinh_transform,
    standardize_features,
    create_feature_matrix,
    ion_count_pipeline
)


class TestArcSinhTransform:
    """Test arcsinh transformation for ion count data."""
    
    @pytest.fixture
    def sample_ion_counts(self):
        """Create sample ion count data."""
        np.random.seed(42)
        return {
            'low_counts': np.random.poisson(5, 100),
            'medium_counts': np.random.poisson(50, 100),
            'high_counts': np.random.poisson(500, 100),
            'zero_inflated': np.concatenate([np.zeros(30), np.random.poisson(100, 70)])
        }
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.low_counts = np.random.poisson(5, 100)
        self.medium_counts = np.random.poisson(50, 100)
        self.high_counts = np.random.poisson(500, 100)
        self.zero_inflated = np.concatenate([np.zeros(30), np.random.poisson(100, 70)])
    
    def test_arcsinh_basic_properties(self):
        """Test basic properties of arcsinh transformation."""
        # Create ion count dictionary format expected by the function
        ion_count_dict = {'CD45': self.medium_counts}
        
        # Transform data with auto-optimization
        transformed_dict, used_cofactors = apply_arcsinh_transform(ion_count_dict)
        transformed = transformed_dict['CD45']
        
        # Check that transformation is monotonic
        sorted_original = np.sort(self.medium_counts)
        sorted_transformed_dict, _ = apply_arcsinh_transform({'CD45': sorted_original})
        sorted_transformed = sorted_transformed_dict['CD45']
        
        # Should be monotonically increasing
        assert np.all(np.diff(sorted_transformed) >= 0)
        
        # Zero should map to zero
        zero_transformed_dict, _ = apply_arcsinh_transform({'CD45': np.array([0])})
        assert zero_transformed_dict['CD45'][0] == 0
        
        # Should handle array inputs
        assert len(transformed) == len(self.medium_counts)
    
    def test_cofactor_optimization(self):
        """Test automatic cofactor optimization."""
        test_data_dict = {'CD45': self.high_counts}
        
        # Test percentile method
        transformed_percentile, cofactors_percentile = apply_arcsinh_transform(
            test_data_dict, optimization_method='percentile'
        )
        
        # Test MAD method
        transformed_mad, cofactors_mad = apply_arcsinh_transform(
            test_data_dict, optimization_method='mad'
        )
        
        # Both methods should produce valid cofactors
        assert cofactors_percentile['CD45'] > 0
        assert cofactors_mad['CD45'] > 0
        
        # Transformed values should be finite
        assert np.all(np.isfinite(transformed_percentile['CD45']))
        assert np.all(np.isfinite(transformed_mad['CD45']))
        
        # Cofactors should be different for different methods
        assert cofactors_percentile['CD45'] != cofactors_mad['CD45']
    
    def test_zero_inflated_handling(self):
        """Test handling of zero-inflated data common in IMC."""
        zero_inflated_dict = {'CD45': self.zero_inflated}
        
        # Auto-optimize with zero-inflated data
        transformed_dict, cofactors = apply_arcsinh_transform(zero_inflated_dict)
        
        transformed = transformed_dict['CD45']
        
        # Check that zeros remain zeros
        zero_indices = np.where(self.zero_inflated == 0)[0]
        assert np.all(transformed[zero_indices] == 0)
        
        # Non-zero values should be transformed
        non_zero_indices = np.where(self.zero_inflated > 0)[0]
        assert np.all(transformed[non_zero_indices] > 0)
    
    def test_variance_stabilization(self):
        """Test that arcsinh transformation stabilizes Poisson variance."""
        np.random.seed(42)
        
        # Generate Poisson data with different means
        low_poisson = np.random.poisson(lam=5, size=1000)
        high_poisson = np.random.poisson(lam=500, size=1000)
        
        # For Poisson, variance equals mean
        assert np.abs(np.var(low_poisson) - np.mean(low_poisson)) < 5
        assert np.abs(np.var(high_poisson) - np.mean(high_poisson)) < 50
        
        # Apply with auto-optimization
        low_transformed_dict, low_cofactors = apply_arcsinh_transform({'low': low_poisson})
        high_transformed_dict, high_cofactors = apply_arcsinh_transform({'high': high_poisson})
        
        low_transformed = low_transformed_dict['low']
        high_transformed = high_transformed_dict['high']
        
        # After transformation, variance should be more similar
        var_ratio_original = np.var(high_poisson) / np.var(low_poisson)
        var_ratio_transformed = np.var(high_transformed) / np.var(low_transformed)
        
        # Variance ratio should be reduced after transformation
        assert var_ratio_transformed < var_ratio_original
        
        # The higher-expression data should use a larger cofactor
        assert high_cofactors['high'] > low_cofactors['low']


class TestStandardization:
    """Test feature standardization."""
    
    def test_standardize_features(self):
        """Test StandardScaler on transformed features."""
        np.random.seed(42)
        
        # Create protein data with different scales (post-arcsinh)
        transformed_arrays = {
            'CD45': np.random.normal(2, 0.5, (10, 10)),
            'CD31': np.random.normal(4, 1.0, (10, 10)),
            'CD11b': np.random.normal(1, 0.2, (10, 10))
        }
        
        standardized, scalers = standardize_features(transformed_arrays)
        
        # Check that each protein is standardized
        for protein_name, standardized_array in standardized.items():
            flat_values = standardized_array.ravel()
            
            # Mean should be approximately 0
            assert np.abs(np.mean(flat_values)) < 0.01
            
            # Standard deviation should be approximately 1
            assert np.abs(np.std(flat_values) - 1.0) < 0.01
        
        # Check that scalers are stored
        assert len(scalers) == len(transformed_arrays)
    
    def test_standardize_with_mask(self):
        """Test standardization with spatial mask."""
        np.random.seed(42)
        
        # Create data and mask
        transformed_arrays = {
            'CD45': np.random.normal(2, 0.5, (10, 10))
        }
        
        # Create mask - only use center region
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        
        standardized, scalers = standardize_features(transformed_arrays, mask)
        
        # Only masked values should be standardized
        masked_values = standardized['CD45'][mask]
        
        # Mean of masked values should be approximately 0
        assert np.abs(np.mean(masked_values)) < 0.01
        
        # Std of masked values should be approximately 1
        assert np.abs(np.std(masked_values) - 1.0) < 0.01


class TestFeatureMatrix:
    """Test feature matrix creation."""
    
    def test_create_feature_matrix(self):
        """Test creation of feature matrix for clustering."""
        np.random.seed(42)
        
        # Create standardized arrays
        standardized_arrays = {
            'CD45': np.random.normal(0, 1, (5, 5)),
            'CD31': np.random.normal(0, 1, (5, 5)),
            'CD11b': np.random.normal(0, 1, (5, 5))
        }
        
        feature_matrix, protein_names, valid_indices = create_feature_matrix(standardized_arrays)
        
        # Check dimensions
        assert feature_matrix.shape[0] == 25  # 5x5 flattened
        assert feature_matrix.shape[1] == 3   # 3 proteins
        
        # Check protein names preserved
        assert protein_names == ['CD45', 'CD31', 'CD11b']
        
        # All indices should be valid without mask
        assert len(valid_indices) == 25
    
    def test_create_feature_matrix_with_mask(self):
        """Test feature matrix creation with mask."""
        np.random.seed(42)
        
        standardized_arrays = {
            'CD45': np.random.normal(0, 1, (5, 5)),
            'CD31': np.random.normal(0, 1, (5, 5))
        }
        
        # Create mask - only use some pixels
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True  # 9 valid pixels
        
        feature_matrix, protein_names, valid_indices = create_feature_matrix(
            standardized_arrays, mask
        )
        
        # Should only include masked pixels
        assert feature_matrix.shape[0] == 9
        assert feature_matrix.shape[1] == 2
        
        # Valid indices should match mask
        assert len(valid_indices) == 9


class TestIonCountPipeline:
    """Test complete ion count processing pipeline."""
    
    def test_pipeline_integration(self):
        """Test full pipeline with realistic data."""
        np.random.seed(42)
        
        # Create realistic IMC-like data
        n_points = 1000
        coords = np.random.uniform(0, 100, (n_points, 2))
        
        # Create ion counts with different expression patterns
        ion_counts = {
            'CD45': np.random.negative_binomial(5, 0.3, n_points),  # Immune marker
            'CD31': np.random.negative_binomial(10, 0.5, n_points),  # Vascular marker
            'CD11b': np.random.negative_binomial(3, 0.2, n_points)   # Myeloid marker
        }
        
        # Run pipeline
        results = ion_count_pipeline(
            coords,
            ion_counts,
            bin_size_um=20.0,
            n_clusters=5
        )
        
        # Check all expected outputs
        assert 'aggregated_counts' in results
        assert 'transformed_arrays' in results
        assert 'cofactors_used' in results
        assert 'standardized_arrays' in results
        assert 'feature_matrix' in results
        assert 'cluster_labels' in results
        assert 'cluster_centroids' in results
        
        # Check dimensions consistency
        n_bins = len(results['bin_edges_x']) - 1
        n_bins_y = len(results['bin_edges_y']) - 1
        
        # Feature matrix should have correct shape
        assert results['feature_matrix'].shape[1] == 3  # 3 proteins
        
        # Cluster labels should match feature matrix
        assert len(results['cluster_labels']) == results['feature_matrix'].shape[0]
        
        # Should have 5 clusters as requested
        assert len(np.unique(results['cluster_labels'])) == 5
        
        # Cofactors should be optimized for each protein
        assert len(results['cofactors_used']) == 3
        for protein in ion_counts:
            assert protein in results['cofactors_used']
            assert results['cofactors_used'][protein] > 0
    
    def test_pipeline_with_optimization(self):
        """Test pipeline with automatic cluster optimization."""
        np.random.seed(42)
        
        coords = np.random.uniform(0, 50, (500, 2))
        ion_counts = {
            'Marker1': np.random.poisson(100, 500)
        }
        
        # Run without specifying n_clusters
        results = ion_count_pipeline(
            coords,
            ion_counts,
            bin_size_um=10.0,
            n_clusters=None  # Should trigger optimization
        )
        
        # Should have optimization results
        assert 'optimization_results' in results
        assert 'optimal_k' in results['optimization_results']
        
        # Optimal k should be reasonable
        optimal_k = results['optimization_results']['optimal_k']
        assert 2 <= optimal_k <= 15


if __name__ == '__main__':
    pytest.main([__file__, '-v'])