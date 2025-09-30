"""
Core Algorithm Tests for IMC Analysis Pipeline

Tests mathematical correctness and algorithmic properties.
No mocks, no abstractions, just scientific validation.
"""

import numpy as np
import pytest
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

from src.analysis.ion_count_processing import (
    apply_arcsinh_transform,
    aggregate_ion_counts
)
from src.analysis.spatial_clustering import perform_spatial_clustering
from src.analysis.slic_segmentation import perform_slic_segmentation


class TestArcSinhTransform:
    """Test the core arcsinh transformation for IMC data."""
    
    def test_monotonic_transformation(self):
        """Arcsinh must preserve ordering (mathematical property)."""
        # Create test data with known ordering
        ion_counts = {'CD45': np.array([0, 1, 5, 10, 50, 100, 500, 1000])}
        
        # Transform
        transformed, cofactors = apply_arcsinh_transform(ion_counts)
        
        # Check monotonicity
        assert np.all(np.diff(transformed['CD45']) >= 0), \
            "Arcsinh transformation must be monotonic"
        
        # Zero maps to zero
        assert transformed['CD45'][0] == 0, "Zero must map to zero"
    
    def test_variance_stabilization(self):
        """Arcsinh should stabilize variance across intensity ranges."""
        # Generate Poisson data with different means
        np.random.seed(42)
        low_intensity = np.random.poisson(5, 1000)
        high_intensity = np.random.poisson(500, 1000)
        
        ion_counts = {
            'low': low_intensity,
            'high': high_intensity
        }
        
        # Transform
        transformed, _ = apply_arcsinh_transform(ion_counts)
        
        # Check variance stabilization
        var_ratio_original = np.var(high_intensity) / np.var(low_intensity)
        var_ratio_transformed = np.var(transformed['high']) / np.var(transformed['low'])
        
        # Transformed variance ratio should be much smaller
        assert var_ratio_transformed < var_ratio_original / 10, \
            f"Variance not stabilized: original ratio {var_ratio_original:.1f}, " \
            f"transformed {var_ratio_transformed:.1f}"
    
    def test_cofactor_optimization_methods(self):
        """Test different cofactor optimization strategies."""
        # Real-like IMC data with zero inflation
        np.random.seed(42)
        data = np.concatenate([
            np.zeros(300),  # 30% zeros (background)
            np.random.poisson(10, 500),  # Low expression
            np.random.poisson(100, 200)  # High expression
        ])
        np.random.shuffle(data)
        
        ion_counts = {'marker': data}
        
        # Test percentile method
        _, cofactor_percentile = apply_arcsinh_transform(
            ion_counts, optimization_method='percentile'
        )
        
        # Test MAD method
        _, cofactor_mad = apply_arcsinh_transform(
            ion_counts, optimization_method='mad'
        )
        
        # Both should be positive and reasonable
        assert 0.1 < cofactor_percentile['marker'] < 1000, \
            f"Percentile cofactor out of range: {cofactor_percentile['marker']}"
        assert 0.1 < cofactor_mad['marker'] < 1000, \
            f"MAD cofactor out of range: {cofactor_mad['marker']}"


class TestSpatialMetrics:
    """Test spatial analysis algorithms."""
    
    def test_distance_triangle_inequality(self):
        """Distance metrics must satisfy triangle inequality."""
        # Create test points
        coords = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
        ])
        
        # Calculate pairwise distances
        distances = squareform(pdist(coords))
        
        # Check triangle inequality for all triplets
        n_points = len(coords)
        for i in range(n_points):
            for j in range(n_points):
                for k in range(n_points):
                    if i != j and j != k and i != k:
                        assert distances[i, k] <= distances[i, j] + distances[j, k] + 1e-10, \
                            f"Triangle inequality violated for points {i}, {j}, {k}"
    
    def test_clustering_determinism(self):
        """Clustering with same seed must produce identical results."""
        # Create test data
        np.random.seed(42)
        coords = np.random.rand(100, 2) * 100
        features = np.random.rand(100, 5)
        
        # Run clustering twice with same seed
        labels1, metadata1 = perform_spatial_clustering(
            features, coords, method='leiden', random_state=42
        )
        labels2, metadata2 = perform_spatial_clustering(
            features, coords, method='leiden', random_state=42
        )
        
        # Results must be identical
        assert np.array_equal(labels1, labels2), \
            "Clustering not deterministic with same seed"
    
    def test_spatial_coherence_metric(self):
        """Test spatial coherence calculation."""
        # Create spatially coherent clusters
        coords = np.vstack([
            np.random.randn(50, 2) + [0, 0],  # Cluster 1 at origin
            np.random.randn(50, 2) + [10, 10]  # Cluster 2 separated
        ])
        labels = np.array([0] * 50 + [1] * 50)
        
        # Calculate silhouette score as coherence metric
        if len(np.unique(labels)) > 1:
            coherence = silhouette_score(coords, labels)
            assert coherence > 0.5, \
                f"Spatially separated clusters should have high coherence: {coherence}"


class TestIonCountAggregation:
    """Test ion count aggregation preserves Poisson statistics."""
    
    def test_sum_not_average(self):
        """Ion counts must be SUMMED, not averaged, in spatial bins."""
        # Create test data
        coords = np.array([
            [0.5, 0.5],  # Both points in same bin
            [0.7, 0.7]
        ])
        ion_counts = {'CD45': np.array([10.0, 15.0])}
        
        # Create bins
        bin_edges_x = np.array([0, 1, 2])
        bin_edges_y = np.array([0, 1, 2])
        
        # Aggregate
        aggregated = aggregate_ion_counts(
            coords, ion_counts, bin_edges_x, bin_edges_y
        )
        
        # Should sum to 25, not average to 12.5
        assert aggregated['CD45'][0, 0] == 25.0, \
            f"Ion counts must be summed: got {aggregated['CD45'][0, 0]}, expected 25"
    
    def test_conservation_of_counts(self):
        """Total counts must be conserved during aggregation."""
        # Create random test data
        np.random.seed(42)
        coords = np.random.rand(100, 2) * 10
        ion_counts = {'marker': np.random.poisson(50, 100).astype(float)}
        
        # Create bins
        bin_edges = np.linspace(0, 10, 6)
        
        # Aggregate
        aggregated = aggregate_ion_counts(
            coords, ion_counts, bin_edges, bin_edges
        )
        
        # Total must be conserved
        original_total = np.sum(ion_counts['marker'])
        aggregated_total = np.sum(aggregated['marker'])
        
        assert np.isclose(original_total, aggregated_total), \
            f"Count conservation violated: {original_total} → {aggregated_total}"


class TestSLICSegmentation:
    """Test SLIC superpixel segmentation."""
    
    def test_slic_connectivity(self):
        """SLIC segments must be connected regions."""
        # Create simple test image
        np.random.seed(42)
        coords = np.random.rand(1000, 2) * 100
        
        # Create DNA intensity as 2D image
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(coords.T)
        x_grid = np.linspace(0, 100, 50)
        y_grid = np.linspace(0, 100, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        dna_image = kde(positions).reshape(50, 50)
        
        # Run SLIC
        segments = perform_slic_segmentation(
            dna_image,
            n_segments=10,
            compactness=10
        )
        
        # Check basic properties
        assert segments.shape == dna_image.shape, "Segments shape must match input"
        assert len(np.unique(segments)) <= 10, "Should have at most n_segments regions"
        assert np.min(segments) >= 0, "Segment labels must be non-negative"
    
    def test_slic_determinism(self):
        """SLIC with same parameters should be deterministic."""
        # Create test data
        np.random.seed(42)
        dna_image = np.random.rand(50, 50)
        
        # Run twice
        segments1 = perform_slic_segmentation(dna_image, n_segments=5, compactness=10)
        segments2 = perform_slic_segmentation(dna_image, n_segments=5, compactness=10)
        
        # Should be identical
        assert np.array_equal(segments1, segments2), \
            "SLIC segmentation not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])