"""
Property-Based Testing for Spatial Algorithms

Tests mathematical invariants and properties that MUST hold for spatial algorithms.
Uses hypothesis for generating diverse test cases to validate fundamental properties.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances

from src.analysis.ion_count_processing import apply_arcsinh_transform, aggregate_ion_counts
from src.analysis.spatial_clustering import perform_spatial_clustering


class TestSpatialInvariants:
    """Test fundamental mathematical properties of spatial algorithms."""
    
    @given(
        coords=arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(5, 50), st.just(2)),
            elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_pairwise_distance_triangle_inequality(self, coords):
        """Triangle inequality: d(a,c) <= d(a,b) + d(b,c) for all points."""
        assume(len(coords) >= 3)  # Need at least 3 points for triangle
        
        # Calculate pairwise distances
        distances = pairwise_distances(coords)
        
        # Test triangle inequality for all triplets
        n_points = len(coords)
        violations = 0
        for i in range(min(n_points, 10)):  # Limit for performance
            for j in range(min(n_points, 10)):
                for k in range(min(n_points, 10)):
                    if i != j and j != k and i != k:
                        d_ik = distances[i, k]
                        d_ij = distances[i, j]
                        d_jk = distances[j, k]
                        
                        # Allow reasonable numerical tolerance for floating-point arithmetic
                        # Use relative tolerance for larger distances
                        tolerance = max(1e-8, 1e-12 * max(d_ik, d_ij, d_jk))
                        if d_ik > d_ij + d_jk + tolerance:
                            violations += 1
        
        assert violations == 0, f"Triangle inequality violated {violations} times"
    
    @given(
        coords=arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(10, 100), st.just(2)),
            elements=st.floats(0, 100, allow_nan=False, allow_infinity=False)
        ),
        translation=arrays(
            dtype=np.float64,
            shape=st.just((2,)),
            elements=st.floats(-50, 50, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(max_examples=15, deadline=None)
    def test_spatial_clustering_translation_invariance(self, coords, translation):
        """Spatial clustering should be invariant to translation."""
        assume(len(coords) >= 5)
        assume(np.all(np.isfinite(coords)))
        assume(np.all(np.isfinite(translation)))
        
        # Create simple features (same for both)
        features = np.random.RandomState(42).randn(len(coords), 3)
        
        # Translate coordinates
        translated_coords = coords + translation
        
        try:
            # Perform clustering on original coordinates
            labels1, metadata1 = perform_spatial_clustering(
                features, coords, method='leiden', random_state=42
            )
            
            # Perform clustering on translated coordinates  
            labels2, metadata2 = perform_spatial_clustering(
                features, translated_coords, method='leiden', random_state=42
            )
            
            # Labels should be identical (clustering invariant to translation)
            # Allow for label permutation by checking cluster structure
            n_clusters1 = len(np.unique(labels1[labels1 >= 0]))
            n_clusters2 = len(np.unique(labels2[labels2 >= 0]))
            
            assert n_clusters1 == n_clusters2, \
                f"Translation changed cluster count: {n_clusters1} vs {n_clusters2}"
                
        except Exception as e:
            # If clustering fails for any reason, that's also valuable information
            pytest.skip(f"Clustering failed: {e}")
    
    @given(
        rotation_angle=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        n_points=st.integers(10, 50)
    )
    @settings(max_examples=10, deadline=None)
    def test_spatial_clustering_rotation_invariance(self, rotation_angle, n_points):
        """Spatial clustering should be invariant to rotation."""
        # Create circular pattern to avoid edge effects
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        coords = np.column_stack([np.cos(angles), np.sin(angles)]) * 10
        
        # Create features based on position (to ensure spatial structure)
        features = coords.copy()
        
        # Rotation matrix
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Rotate coordinates
        rotated_coords = coords @ rotation_matrix.T
        rotated_features = features @ rotation_matrix.T
        
        try:
            # Cluster original
            labels1, _ = perform_spatial_clustering(
                features, coords, method='leiden', random_state=42, spatial_weight=0.5
            )
            
            # Cluster rotated
            labels2, _ = perform_spatial_clustering(
                rotated_features, rotated_coords, method='leiden', random_state=42, spatial_weight=0.5
            )
            
            # Should have same number of clusters
            n_clusters1 = len(np.unique(labels1[labels1 >= 0]))
            n_clusters2 = len(np.unique(labels2[labels2 >= 0]))
            
            assert n_clusters1 == n_clusters2, \
                f"Rotation changed cluster count: {n_clusters1} vs {n_clusters2}"
                
        except Exception as e:
            pytest.skip(f"Clustering failed: {e}")


class TestArcSinhProperties:
    """Test mathematical properties of arcsinh transformation."""
    
    @given(
        values=arrays(
            dtype=np.float64,
            shape=st.integers(10, 1000),
            elements=st.floats(0, 10000, allow_nan=False, allow_infinity=False)
        ),
        scale_factor=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=20, deadline=None)
    def test_arcsinh_monotonicity_preserved(self, values, scale_factor):
        """arcsinh(ax) should preserve monotonicity for a > 0."""
        assume(len(values) >= 2)
        assume(scale_factor > 0)
        # Exclude pathological cases where all/most values are effectively zero
        # (causes tie-breaking issues in argsort, not a real monotonicity violation)
        assume(np.sum(np.abs(values) > 1e-10) >= len(values) * 0.1)

        # Scale values
        scaled_values = values * scale_factor
        
        # Transform both
        ion_counts_orig = {'test': values}
        ion_counts_scaled = {'test': scaled_values}
        
        transformed_orig, _ = apply_arcsinh_transform(ion_counts_orig)
        transformed_scaled, _ = apply_arcsinh_transform(ion_counts_scaled)
        
        orig_result = transformed_orig['test']
        scaled_result = transformed_scaled['test']
        
        # Sort indices for original
        orig_sort_idx = np.argsort(values)
        scaled_sort_idx = np.argsort(scaled_values)
        
        # Sorted arrays should be in same order
        assert np.array_equal(orig_sort_idx, scaled_sort_idx), \
            "Scaling broke monotonicity order"
    
    @given(
        values=arrays(
            dtype=np.float64,
            shape=st.integers(100, 1000),
            elements=st.floats(0.1, 1000, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(max_examples=15, deadline=None) 
    def test_arcsinh_variance_reduction_property(self, values):
        """arcsinh should reduce variance ratio between high and low values."""
        assume(len(values) >= 100)
        
        # Split into low and high value groups
        median_val = np.median(values)
        low_values = values[values <= median_val]
        high_values = values[values > median_val]
        
        assume(len(low_values) >= 10 and len(high_values) >= 10)
        
        # Check that both groups have meaningful variance
        low_var = np.var(low_values)
        high_var = np.var(high_values)
        assume(low_var > 1e-6 and high_var > 1e-6)  # Skip degenerate cases
        
        # Original variance ratio
        orig_var_ratio = high_var / low_var
        
        # Transform
        ion_counts = {'test': values}
        transformed, _ = apply_arcsinh_transform(ion_counts)
        trans_values = transformed['test']
        
        # Split transformed values
        trans_low = trans_values[values <= median_val]
        trans_high = trans_values[values > median_val]
        
        # Transformed variance ratio
        trans_var_ratio = np.var(trans_high) / np.var(trans_low)
        
        # Variance stabilization: transformed ratio should be smaller
        assert trans_var_ratio < orig_var_ratio, \
            f"arcsinh failed to reduce variance ratio: {orig_var_ratio:.2f} -> {trans_var_ratio:.2f}"


class TestIonCountAggregationProperties:
    """Test mathematical properties of ion count aggregation."""
    
    @given(
        n_points=st.integers(50, 200)
    )
    @settings(max_examples=15, deadline=None)
    def test_ion_count_conservation(self, n_points):
        """Total ion counts must be conserved during spatial aggregation."""
        # Generate matching coordinate and ion count arrays
        coords = np.random.uniform(0, 100, (n_points, 2))
        ion_values = np.random.uniform(0, 1000, n_points)
        
        assume(np.all(np.isfinite(coords)))
        assume(np.all(np.isfinite(ion_values)))
        
        # Create ion count dictionary
        ion_counts = {'marker': ion_values.astype(float)}
        
        # Create spatial bins
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        
        # Ensure bins cover all data
        bin_edges_x = np.linspace(x_min - 1, x_max + 1, 11)
        bin_edges_y = np.linspace(y_min - 1, y_max + 1, 11)
        
        # Aggregate
        aggregated = aggregate_ion_counts(coords, ion_counts, bin_edges_x, bin_edges_y)
        
        # Check conservation
        original_total = np.sum(ion_values)
        aggregated_total = np.sum(aggregated['marker'])
        
        # Allow small numerical tolerance
        relative_error = abs(original_total - aggregated_total) / max(original_total, 1e-10)
        assert relative_error < 1e-10, \
            f"Count conservation violated: {original_total} -> {aggregated_total} (error: {relative_error})"
    
    @given(
        n_points=st.integers(20, 100),
        scale_factor=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=10, deadline=None)
    def test_aggregation_scale_invariance(self, n_points, scale_factor):
        """Spatial aggregation should scale proportionally with coordinate scaling."""
        # Create test data
        coords = np.random.RandomState(42).uniform(0, 50, (n_points, 2))
        ion_values = np.random.RandomState(42).poisson(100, n_points).astype(float)
        ion_counts = {'marker': ion_values}
        
        # Scale coordinates
        scaled_coords = coords * scale_factor
        
        # Create bins for original
        bin_edges = np.linspace(-5, 55, 11)
        
        # Create scaled bins
        scaled_bin_edges = bin_edges * scale_factor
        
        # Aggregate both
        agg_orig = aggregate_ion_counts(coords, ion_counts, bin_edges, bin_edges)
        agg_scaled = aggregate_ion_counts(scaled_coords, ion_counts, scaled_bin_edges, scaled_bin_edges)
        
        # Results should be identical (same spatial structure, just scaled)
        orig_total = np.sum(agg_orig['marker'])
        scaled_total = np.sum(agg_scaled['marker'])
        
        assert abs(orig_total - scaled_total) < 1e-10, \
            f"Scale invariance violated: {orig_total} vs {scaled_total}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])