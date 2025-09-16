"""
Test suite for ion count processing module.

Tests the core ion count statistics pipeline that forms the foundation
of the production IMC analysis system.
"""

import numpy as np
import pytest
from typing import Dict

from src.analysis.ion_count_processing import (
    ion_count_pipeline,
    apply_arcsinh_transform,
    standardize_features,
    aggregate_ion_counts
)


class TestArcsinhTransform:
    """Test arcsinh transformation for ion count data."""
    
    def setup_method(self):
        """Setup test data with realistic IMC ion count characteristics."""
        np.random.seed(42)
        
        # Typical IMC ion count distributions (Poisson-like, low counts)
        self.low_counts = np.random.poisson(2, 1000)  # CD45 in non-immune areas
        self.medium_counts = np.random.poisson(10, 1000)  # CD45 in immune areas
        self.high_counts = np.random.poisson(50, 1000)  # High expression marker
        self.zero_inflated = np.concatenate([
            np.zeros(500),  # Many zeros (common in IMC)
            np.random.poisson(20, 500)
        ])
    
    def test_arcsinh_basic_properties(self):
        """Test basic properties of arcsinh transformation."""
        # Create ion count dictionary format expected by the function
        ion_count_dict = {'CD45': self.medium_counts}
        cofactors = {'CD45': 1.0}
        
        # Transform data
        transformed_dict, used_cofactors = apply_arcsinh_transform(ion_count_dict, cofactors)
        transformed = transformed_dict['CD45']
        
        # Check that transformation is monotonic
        sorted_original = np.sort(self.medium_counts)
        sorted_transformed_dict, _ = apply_arcsinh_transform({'CD45': sorted_original}, cofactors)
        sorted_transformed = sorted_transformed_dict['CD45']
        
        # Should be monotonically increasing
        assert np.all(np.diff(sorted_transformed) >= 0)
        
        # Zero should map to zero
        zero_transformed_dict, _ = apply_arcsinh_transform({'CD45': np.array([0])}, cofactors)
        assert zero_transformed_dict['CD45'][0] == 0
        
        # Should handle array inputs
        assert len(transformed) == len(self.medium_counts)
    
    def test_cofactor_effects(self):
        """Test effect of different cofactor values."""
        test_data_dict = {'CD45': self.high_counts}
        
        cofactor_low = {'CD45': 0.1}
        cofactor_high = {'CD45': 10.0}
        
        transformed_low_dict, _ = apply_arcsinh_transform(test_data_dict, cofactor_low)
        transformed_high_dict, _ = apply_arcsinh_transform(test_data_dict, cofactor_high)
        
        transformed_low = transformed_low_dict['CD45']
        transformed_high = transformed_high_dict['CD45']
        
        # Lower cofactor should give larger transformed values for same input
        assert np.mean(transformed_low) > np.mean(transformed_high)
        
        # Both should preserve order
        assert np.all(np.diff(np.argsort(self.high_counts)) == np.diff(np.argsort(transformed_low)))
        assert np.all(np.diff(np.argsort(self.high_counts)) == np.diff(np.argsort(transformed_high)))
    
    def test_zero_inflated_handling(self):
        """Test handling of zero-inflated data common in IMC."""
        zero_inflated_dict = {'CD45': self.zero_inflated}
        cofactor = {'CD45': 1.0}
        
        transformed_dict, _ = apply_arcsinh_transform(zero_inflated_dict, cofactor)
        transformed = transformed_dict['CD45']
        
        # Zeros should remain zeros
        zero_mask = self.zero_inflated == 0
        assert np.all(transformed[zero_mask] == 0)
        
        # Non-zero values should be transformed
        nonzero_mask = self.zero_inflated > 0
        assert np.all(transformed[nonzero_mask] > 0)
    
    def test_noise_variance_stabilization(self):
        """Test that arcsinh transformation stabilizes Poisson variance."""
        # Create Poisson data with different means
        np.random.seed(42)  # For reproducible results
        low_poisson = np.random.poisson(5, 10000)
        high_poisson = np.random.poisson(50, 10000)
        
        # Raw data should have variance ≈ mean (Poisson property)
        assert abs(np.var(low_poisson) - np.mean(low_poisson)) < 2
        assert abs(np.var(high_poisson) - np.mean(high_poisson)) < 10
        
        # After transformation, variance should be more similar
        cofactor = {'low': 1.0, 'high': 1.0}
        
        low_transformed_dict, _ = apply_arcsinh_transform({'low': low_poisson}, {'low': 1.0})
        high_transformed_dict, _ = apply_arcsinh_transform({'high': high_poisson}, {'high': 1.0})
        
        low_transformed = low_transformed_dict['low']
        high_transformed = high_transformed_dict['high']
        
        # Variance should be more similar after transformation
        var_ratio_raw = np.var(high_poisson) / np.var(low_poisson)
        var_ratio_transformed = np.var(high_transformed) / np.var(low_transformed)
        
        assert var_ratio_transformed < var_ratio_raw


class TestStandardization:
    """Test standardization of ion count data."""
    
    def setup_method(self):
        """Setup test data for standardization."""
        np.random.seed(42)
        
        # Create protein data with different scales (post-arcsinh)
        self.protein_data = {
            'CD45': np.random.normal(2.0, 1.0, 1000),  # Medium expression
            'CD11b': np.random.normal(0.5, 0.3, 1000),  # Low expression
            'CD206': np.random.normal(3.0, 1.5, 1000)   # High expression
        }
    
    def test_standardization_properties(self):
        """Test basic properties of standardization."""
        # standardize_features expects a dictionary format
        standardized_dict, scalers = standardize_features(self.protein_data)
        
        # Each protein should have ~zero mean and ~unit variance
        for protein_name, standardized_array in standardized_dict.items():
            mean = np.mean(standardized_array)
            std = np.std(standardized_array)
            
            assert abs(mean) < 0.1  # Close to zero mean
            assert abs(std - 1.0) < 0.1  # Close to unit standard deviation
            
            # Should have a corresponding scaler
            assert protein_name in scalers
    
    def test_standardization_preserves_relationships(self):
        """Test that standardization preserves relative relationships."""
        standardized_dict, scalers = standardize_features(self.protein_data)
        
        # Correlation between proteins should be preserved
        original_corr = np.corrcoef(
            self.protein_data['CD45'], 
            self.protein_data['CD11b']
        )[0, 1]
        
        standardized_corr = np.corrcoef(
            standardized_dict['CD45'],
            standardized_dict['CD11b']
        )[0, 1]
        
        assert abs(original_corr - standardized_corr) < 0.01


class TestSpatialBinning:
    """Test spatial binning functionality via ion_count_pipeline."""
    
    def setup_method(self):
        """Setup spatial data for binning tests."""
        np.random.seed(42)
        
        # Create 100x100 μm tissue region with 1μm pixels
        self.coords = np.random.uniform(0, 100, (5000, 2))  # 5k pixels for faster tests
        
        # Create protein data with spatial structure
        # CD45 high in upper-left quadrant
        cd45_values = np.random.poisson(5, 5000)
        upper_left = (self.coords[:, 0] < 50) & (self.coords[:, 1] < 50)
        cd45_values[upper_left] += np.random.poisson(15, np.sum(upper_left))
        
        self.ion_counts = {'CD45': cd45_values}
    
    def test_spatial_binning_basic(self):
        """Test basic spatial binning functionality using ion_count_pipeline."""
        bin_size_um = 10.0
        
        pipeline_result = ion_count_pipeline(
            self.coords, 
            self.ion_counts, 
            bin_size_um=bin_size_um,
            n_clusters=3  # Small number for test speed
        )
        
        # Check structure
        assert 'aggregated_counts' in pipeline_result
        assert 'bin_edges_x' in pipeline_result
        assert 'bin_edges_y' in pipeline_result
        
        # Should preserve total counts (ion counts are summed in aggregation)
        original_total = np.sum(self.ion_counts['CD45'])
        binned_total = np.sum(pipeline_result['aggregated_counts']['CD45'])
        
        # Should be equal (ion counts are summed, not averaged)
        assert abs(original_total - binned_total) < 1  # Allow small rounding errors
    
    def test_different_bin_sizes(self):
        """Test spatial binning with different bin sizes."""
        small_bins = ion_count_pipeline(self.coords, self.ion_counts, bin_size_um=5.0, n_clusters=3)
        large_bins = ion_count_pipeline(self.coords, self.ion_counts, bin_size_um=20.0, n_clusters=3)
        
        # Smaller bins should create more bins
        small_bin_count = len(small_bins['bin_edges_x']) - 1
        large_bin_count = len(large_bins['bin_edges_x']) - 1
        assert small_bin_count > large_bin_count
        
        # Total counts should be preserved in both cases
        original_total = np.sum(self.ion_counts['CD45'])
        small_total = np.sum(small_bins['aggregated_counts']['CD45'])
        large_total = np.sum(large_bins['aggregated_counts']['CD45'])
        
        assert abs(original_total - small_total) < 1
        assert abs(original_total - large_total) < 1
    
    def test_spatial_structure_preservation(self):
        """Test that spatial binning preserves spatial structure."""
        bin_size_um = 10.0
        
        pipeline_result = ion_count_pipeline(
            self.coords, 
            self.ion_counts, 
            bin_size_um=bin_size_um,
            n_clusters=3
        )
        
        # Check if spatial structure is preserved in aggregated data
        aggregated_cd45 = pipeline_result['aggregated_counts']['CD45']
        
        # Create coordinate grids for bin centers
        bin_edges_x = pipeline_result['bin_edges_x']
        bin_edges_y = pipeline_result['bin_edges_y']
        
        x_centers = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2
        y_centers = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2
        
        # Find upper-left vs lower-right quadrants in aggregated data
        upper_left_total = 0
        lower_right_total = 0
        ul_count = 0
        lr_count = 0
        
        for i, y_center in enumerate(y_centers):
            for j, x_center in enumerate(x_centers):
                if x_center < 50 and y_center < 50:  # Upper-left
                    upper_left_total += aggregated_cd45[i, j]
                    ul_count += 1
                elif x_center >= 50 and y_center >= 50:  # Lower-right  
                    lower_right_total += aggregated_cd45[i, j]
                    lr_count += 1
        
        # Upper-left should have higher average CD45 due to our spatial structure
        if ul_count > 0 and lr_count > 0:
            ul_avg = upper_left_total / ul_count
            lr_avg = lower_right_total / lr_count
            assert ul_avg > lr_avg


class TestIonCountPipelineIntegration:
    """Test the complete ion count pipeline integration."""
    
    def setup_method(self):
        """Setup realistic IMC data for pipeline testing."""
        np.random.seed(42)
        
        # Create realistic spatial coordinates (200x200 μm region)
        n_pixels = 5000
        self.coords = np.random.uniform(0, 200, (n_pixels, 2))
        
        # Create realistic ion count data for multiple proteins
        self.ion_counts = {
            'CD45': np.random.poisson(8, n_pixels),    # Pan-leukocyte
            'CD11b': np.random.poisson(3, n_pixels),   # Myeloid
            'CD206': np.random.poisson(5, n_pixels),   # M2 macrophages
            'CD31': np.random.poisson(12, n_pixels)    # Endothelial
        }
    
    def test_complete_pipeline_execution(self):
        """Test that the complete pipeline runs without errors."""
        result = ion_count_pipeline(
            self.coords,
            self.ion_counts,
            bin_size_um=20.0,
            n_clusters=5
        )
        
        # Check that all expected components are present
        expected_keys = [
            'aggregated_counts', 'transformed_arrays', 'standardized_arrays',
            'cluster_labels', 'cluster_centroids'
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_pipeline_consistency(self):
        """Test pipeline consistency with repeated runs."""
        # Run pipeline twice with same random seed
        np.random.seed(42)
        result1 = ion_count_pipeline(
            self.coords, self.ion_counts, 
            bin_size_um=20.0, n_clusters=5
        )
        
        np.random.seed(42)
        result2 = ion_count_pipeline(
            self.coords, self.ion_counts, 
            bin_size_um=20.0, n_clusters=5
        )
        
        # Results should be identical (deterministic)
        np.testing.assert_array_equal(result1['cluster_labels'], result2['cluster_labels'])
        
        # Check that optimization results are consistent if present
        if 'optimization_results' in result1 and 'optimization_results' in result2:
            opt1 = result1['optimization_results']
            opt2 = result2['optimization_results']
            if 'silhouette_score' in opt1 and 'silhouette_score' in opt2:
                assert abs(opt1['silhouette_score'] - opt2['silhouette_score']) < 1e-10
    
    def test_pipeline_parameter_validation(self):
        """Test pipeline parameter validation and error handling."""
        # Test with reasonable parameters first to ensure baseline works
        result = ion_count_pipeline(
            self.coords, self.ion_counts,
            bin_size_um=20.0,
            n_clusters=3
        )
        assert 'aggregated_counts' in result
        
        # Test edge cases that should be handled gracefully
        # Very small dataset
        tiny_coords = self.coords[:10]  # Just 10 points
        tiny_ion_counts = {k: v[:10] for k, v in self.ion_counts.items()}
        
        small_result = ion_count_pipeline(
            tiny_coords, tiny_ion_counts,
            bin_size_um=20.0,
            n_clusters=2  # Small number appropriate for tiny dataset
        )
        assert 'aggregated_counts' in small_result
        
        # Empty data should return empty result structure
        empty_result = ion_count_pipeline(
            np.array([]).reshape(0, 2),  # Empty coordinates
            {},  # Empty ion counts
            bin_size_um=20.0,
            n_clusters=5
        )
        assert 'processing_method' in empty_result
        assert empty_result['processing_method'] == 'empty_input'


if __name__ == "__main__":
    pytest.main([__file__])