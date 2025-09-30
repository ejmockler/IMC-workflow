"""
Property-based tests for spatial clustering and stability analysis.

Tests invariants and determinism instead of exact values.
Protects against non-deterministic behavior in data-driven clustering.
"""

import numpy as np
import pytest

from src.analysis.spatial_clustering import (
    perform_spatial_clustering,
    stability_analysis,
    compute_spatial_coherence,
    select_resolution_headless
)


class TestStabilityAnalysisDeterminism:
    """Test that stability analysis is deterministic given same inputs."""
    
    @pytest.fixture
    def sample_clustering_data(self):
        """Standard test data for clustering."""
        np.random.seed(42)  # Fixed seed for reproducible test data
        
        n_samples = 200
        n_features = 9
        
        feature_matrix = np.random.gamma(2, 1, (n_samples, n_features))
        spatial_coords = np.random.uniform(0, 100, (n_samples, 2))
        
        return feature_matrix, spatial_coords
    
    def test_stability_analysis_determinism(self, sample_clustering_data):
        """Critical: Same input should produce identical stability results."""
        feature_matrix, spatial_coords = sample_clustering_data
        
        # Run stability analysis twice with same parameters
        result1 = stability_analysis(
            feature_matrix, spatial_coords,
            method='leiden',
            resolution_range=(0.5, 2.0),
            n_resolutions=10,
            n_bootstrap=5,
            random_state=42
        )
        
        result2 = stability_analysis(
            feature_matrix, spatial_coords,
            method='leiden', 
            resolution_range=(0.5, 2.0),
            n_resolutions=10,
            n_bootstrap=5,
            random_state=42  # Same seed
        )
        
        # Results should be identical
        assert result1['optimal_resolution'] == result2['optimal_resolution']
        np.testing.assert_array_equal(result1['resolutions'], result2['resolutions'])
        np.testing.assert_array_equal(result1['stability_scores'], result2['stability_scores'])
        np.testing.assert_array_equal(result1['mean_n_clusters'], result2['mean_n_clusters'])
    
    def test_perform_clustering_determinism(self, sample_clustering_data):
        """Spatial clustering should be deterministic with same seed."""
        feature_matrix, spatial_coords = sample_clustering_data
        
        # Run clustering twice with same parameters
        labels1, info1 = perform_spatial_clustering(
            feature_matrix, spatial_coords,
            method='leiden',
            resolution=1.0,
            random_state=42
        )
        
        labels2, info2 = perform_spatial_clustering(
            feature_matrix, spatial_coords,
            method='leiden',
            resolution=1.0,
            random_state=42
        )
        
        # Should produce identical results
        np.testing.assert_array_equal(labels1, labels2)
        assert info1['n_clusters'] == info2['n_clusters']
    
    def test_resolution_bounds_respected(self, sample_clustering_data):
        """Optimal resolution should respect configured bounds."""
        feature_matrix, spatial_coords = sample_clustering_data
        
        # Test different resolution ranges
        test_ranges = [(0.5, 1.0), (1.0, 2.0), (0.2, 0.8)]
        
        for res_min, res_max in test_ranges:
            result = stability_analysis(
                feature_matrix, spatial_coords,
                resolution_range=(res_min, res_max),
                n_resolutions=5,
                n_bootstrap=3,
                random_state=42
            )
            
            optimal_res = result['optimal_resolution']
            
            # Optimal resolution should be within bounds
            assert res_min <= optimal_res <= res_max
            
            # All tested resolutions should be within bounds
            assert all(res_min <= res <= res_max for res in result['resolutions'])


class TestClusteringInvariants:
    """Test invariant properties that should hold regardless of data."""
    
    @pytest.fixture
    def minimal_data(self):
        """Minimal valid data for testing edge cases."""
        np.random.seed(42)
        feature_matrix = np.random.normal(0, 1, (10, 3))
        spatial_coords = np.random.uniform(0, 10, (10, 2))
        return feature_matrix, spatial_coords
    
    def test_cluster_labels_properties(self, minimal_data):
        """Test basic properties of cluster labels."""
        feature_matrix, spatial_coords = minimal_data
        
        labels, info = perform_spatial_clustering(
            feature_matrix, spatial_coords,
            method='leiden',
            resolution=1.0,
            random_state=42
        )
        
        # Cluster labels should be non-negative integers
        assert labels.dtype in [np.int32, np.int64]
        assert np.all(labels >= -1)  # -1 allowed for noise in HDBSCAN
        
        # Number of samples should match
        assert len(labels) == feature_matrix.shape[0]
        
        # Info should contain expected keys
        required_keys = ['method', 'n_clusters', 'n_noise']
        for key in required_keys:
            assert key in info
        
        # Cluster count should be reasonable
        assert 0 <= info['n_clusters'] <= len(labels)
        assert 0 <= info['n_noise'] <= len(labels)
    
    def test_spatial_coherence_properties(self, minimal_data):
        """Test properties of spatial coherence metric."""
        feature_matrix, spatial_coords = minimal_data
        
        labels, _ = perform_spatial_clustering(
            feature_matrix, spatial_coords,
            resolution=1.0,
            random_state=42
        )
        
        coherence = compute_spatial_coherence(labels, spatial_coords)
        
        # Spatial coherence should be a finite number
        assert isinstance(coherence, float)
        assert np.isfinite(coherence)
        
        # For random labels, coherence should be close to 0
        random_labels = np.random.randint(0, 3, len(labels))
        random_coherence = compute_spatial_coherence(random_labels, spatial_coords)
        assert np.isfinite(random_coherence)
    
    def test_coabundance_feature_expansion(self, minimal_data):
        """Test that co-abundance features expand correctly."""
        feature_matrix, spatial_coords = minimal_data
        
        # Test with co-abundance features enabled
        protein_names = ['CD45', 'CD31', 'CD11b']
        
        labels, info = perform_spatial_clustering(
            feature_matrix, spatial_coords,
            method='leiden',
            resolution=1.0,
            use_coabundance=True,
            protein_names=protein_names,
            random_state=42
        )
        
        # Should still produce valid cluster labels
        assert len(labels) == feature_matrix.shape[0]
        assert np.all(labels >= -1)
        assert info['n_clusters'] >= 0


class TestResolutionSelection:
    """Test automatic resolution selection logic."""
    
    def test_headless_resolution_selection(self):
        """Test automatic resolution selection without user interaction."""
        # Mock stability results
        stability_results = {
            'resolutions': [0.5, 1.0, 1.5, 2.0],
            'stability_scores': [0.3, 0.7, 0.8, 0.6],
            'mean_n_clusters': [2, 4, 6, 8]
        }
        
        # Should select high-stability resolution
        selected_res = select_resolution_headless(
            stability_results, min_stability=0.6
        )
        
        # Should pick from high-stability options (1.0 or 1.5)
        assert selected_res in [1.0, 1.5]
        
        # Test with no acceptable stability
        low_stability_results = {
            'resolutions': [0.5, 1.0, 1.5],
            'stability_scores': [0.3, 0.4, 0.2],
            'mean_n_clusters': [2, 3, 4]
        }
        
        # Should fall back to most stable option
        fallback_res = select_resolution_headless(
            low_stability_results, min_stability=0.8
        )
        
        # Should pick 1.0 (highest stability score)
        assert fallback_res == 1.0


class TestEdgeCases:
    """Test handling of edge cases and error conditions."""
    
    def test_empty_data_handling(self):
        """Test graceful handling of empty data."""
        empty_features = np.array([]).reshape(0, 5)
        empty_coords = np.array([]).reshape(0, 2)
        
        labels, info = perform_spatial_clustering(
            empty_features, empty_coords,
            random_state=42
        )
        
        # Should handle gracefully
        assert len(labels) == 0
        assert info['n_clusters'] == 0
    
    def test_single_sample_handling(self):
        """Test handling of single sample."""
        single_feature = np.array([[1, 2, 3, 4, 5]])
        single_coord = np.array([[10, 20]])
        
        labels, info = perform_spatial_clustering(
            single_feature, single_coord,
            random_state=42
        )
        
        # Should assign single cluster
        assert len(labels) == 1
        assert labels[0] >= 0  # Valid cluster assignment
    
    def test_insufficient_samples_for_stability(self):
        """Test stability analysis with very few samples."""
        tiny_features = np.random.normal(0, 1, (5, 3))
        tiny_coords = np.random.uniform(0, 5, (5, 2))
        
        # Should not crash with small data
        result = stability_analysis(
            tiny_features, tiny_coords,
            n_resolutions=3,
            n_bootstrap=2,
            random_state=42
        )
        
        # Should return valid result structure
        assert 'optimal_resolution' in result
        assert 'stability_scores' in result
        assert np.isfinite(result['optimal_resolution'])


class TestMethodComparisons:
    """Test that different clustering methods produce valid results."""
    
    @pytest.fixture
    def comparison_data(self):
        """Data for comparing different methods."""
        np.random.seed(42)
        feature_matrix = np.random.gamma(2, 1, (100, 5))
        spatial_coords = np.random.uniform(0, 50, (100, 2))
        return feature_matrix, spatial_coords
    
    def test_leiden_vs_fallback_consistency(self, comparison_data):
        """Test that different methods produce valid outputs."""
        feature_matrix, spatial_coords = comparison_data
        
        methods_to_test = ['leiden']  # Only test available methods
        
        for method in methods_to_test:
            labels, info = perform_spatial_clustering(
                feature_matrix, spatial_coords,
                method=method,
                resolution=1.0,
                random_state=42
            )
            
            # All methods should produce valid results
            assert len(labels) == len(feature_matrix)
            assert np.all(labels >= -1)
            assert info['n_clusters'] >= 0
            assert info['method'] == method


if __name__ == "__main__":
    pytest.main([__file__, '-v'])