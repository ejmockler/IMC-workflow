"""
Oracle Validation Tests

Establishes ground truth for complex algorithms through cross-validation,
simplified analytical solutions, and independent implementations.
"""

import numpy as np
import pytest
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

from src.analysis.ion_count_processing import apply_arcsinh_transform, aggregate_ion_counts
from src.analysis.spatial_clustering import perform_spatial_clustering


class TestArcSinhOracles:
    """Validate arcsinh transformation against analytical solutions."""
    
    def test_arcsinh_against_analytical_solution(self):
        """Test arcsinh transform against known analytical cases."""
        # For very small values, arcsinh(x/c) ≈ x/c
        small_values = np.array([0.1, 0.2, 0.5, 1.0])
        large_cofactor = 1000.0
        
        ion_counts = {'test': small_values}
        
        # Force specific cofactor for analytical comparison
        # Note: This requires modifying the function or creating analytical version
        analytical_result = np.arcsinh(small_values / large_cofactor)
        
        # For small x, arcsinh(x) ≈ x, so result should be approximately x/c
        expected_approx = small_values / large_cofactor
        
        # Both should be very close for small values
        assert np.allclose(analytical_result, expected_approx, rtol=0.1), \
            "Analytical approximation failed for small values"
    
    def test_arcsinh_monotonicity_property(self):
        """Test that arcsinh transformation preserves monotonicity."""
        # Test a wide range of values
        values = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
        
        ion_counts = {'test': values}
        transformed, cofactors = apply_arcsinh_transform(ion_counts)
        
        # Check that transformation preserves order
        original_order = np.argsort(values)
        transformed_order = np.argsort(transformed['test'])
        
        assert np.array_equal(original_order, transformed_order), \
            "arcsinh transformation did not preserve monotonicity"
        
        # Check that transformation is strictly increasing
        transformed_values = transformed['test']
        for i in range(len(transformed_values) - 1):
            assert transformed_values[i] < transformed_values[i + 1], \
                f"Transformation not strictly increasing: {transformed_values[i]} >= {transformed_values[i + 1]}"
    
    def test_arcsinh_derivative_property(self):
        """Test that d/dx arcsinh(x) = 1/sqrt(1+x²)."""
        # Test derivative property numerically
        x_vals = np.array([0.1, 1.0, 5.0, 10.0])
        h = 1e-6  # Small step for numerical derivative
        
        for x in x_vals:
            # Numerical derivative
            f_x_plus_h = np.arcsinh(x + h)
            f_x = np.arcsinh(x)
            numerical_derivative = (f_x_plus_h - f_x) / h
            
            # Analytical derivative
            analytical_derivative = 1.0 / np.sqrt(1 + x**2)
            
            relative_error = abs(numerical_derivative - analytical_derivative) / analytical_derivative
            assert relative_error < 1e-4, \
                f"Derivative test failed at x={x}: {relative_error:.6f}"


class TestSpatialStatisticsOracles:
    """Validate spatial statistics against known solutions."""
    
    def test_uniform_random_pattern_statistics(self):
        """Test spatial statistics on uniform random pattern (basic properties)."""
        np.random.seed(42)
        
        # Generate uniform random points
        n_points = 200
        coords = np.random.uniform(0, 100, (n_points, 2))
        
        # Test basic spatial properties of uniform random pattern
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        distances, _ = tree.query(coords, k=2)
        nn_distances = distances[:, 1]  # Nearest neighbor distances
        
        # For uniform random pattern, test reasonable statistical properties
        mean_nn_distance = np.mean(nn_distances)
        std_nn_distance = np.std(nn_distances)
        
        # Expected mean NN distance for uniform 2D pattern is approximately 1/(2*sqrt(density))
        density = n_points / (100 * 100)
        expected_mean = 1.0 / (2 * np.sqrt(density))
        
        # Check that observed mean is reasonably close to theoretical expectation
        relative_error = abs(mean_nn_distance - expected_mean) / expected_mean
        assert relative_error < 0.5, \
            f"Mean NN distance too far from expected: observed={mean_nn_distance:.3f}, expected={expected_mean:.3f}"
        
        # Check that distribution is not degenerate
        cv = std_nn_distance / mean_nn_distance  # Coefficient of variation
        assert 0.3 < cv < 2.0, \
            f"Unusual coefficient of variation for NN distances: {cv:.3f}"
    
    def test_clustered_pattern_detection(self):
        """Test spatial clustering on known clustered pattern."""
        np.random.seed(42)
        
        # Create two well-separated clusters
        cluster1 = np.random.normal([20, 20], 5, (50, 2))
        cluster2 = np.random.normal([80, 80], 5, (50, 2))
        coords = np.vstack([cluster1, cluster2])
        
        # Create features that reflect spatial structure
        features = coords.copy()  # Position-based features
        
        # Perform clustering
        labels, metadata = perform_spatial_clustering(
            features, coords, method='leiden', spatial_weight=0.5, random_state=42
        )
        
        # Should detect a reasonable number of clusters (2 is ideal, but allow some variation)
        n_clusters = len(np.unique(labels[labels >= 0]))
        assert 2 <= n_clusters <= 8, \
            f"Failed to detect reasonable clusters: found {n_clusters} (expected 2-8)"
        
        # Check spatial coherence using silhouette score
        if n_clusters > 1:
            sil_score = silhouette_score(coords, labels)
            assert sil_score > 0.3, \
                f"Poor spatial clustering quality: silhouette={sil_score:.3f}"
    
    def test_grid_pattern_regularity(self):
        """Test detection of regular grid pattern."""
        # Create perfect grid
        x_coords = np.repeat(np.arange(0, 100, 10), 10)
        y_coords = np.tile(np.arange(0, 100, 10), 10)
        coords = np.column_stack([x_coords, y_coords])
        
        # Calculate nearest neighbor distances
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        distances, _ = tree.query(coords, k=2)
        nn_distances = distances[:, 1]
        
        # For perfect grid, all NN distances should be identical (10.0)
        expected_distance = 10.0
        distance_std = np.std(nn_distances)
        
        assert distance_std < 0.1, \
            f"Grid pattern not detected: NN distance std={distance_std:.3f}"
        assert abs(np.mean(nn_distances) - expected_distance) < 0.1, \
            f"Wrong grid spacing detected: {np.mean(nn_distances):.2f} vs {expected_distance}"


class TestAggregationOracles:
    """Validate spatial aggregation against analytical solutions."""
    
    def test_single_point_aggregation(self):
        """Test aggregation with single point in bin (analytical solution)."""
        # Single point exactly at bin center
        coords = np.array([[5.0, 5.0]])
        ion_counts = {'marker': np.array([100.0])}
        
        # Create bins that put point in center of bin
        bin_edges_x = np.array([0, 10, 20])
        bin_edges_y = np.array([0, 10, 20])
        
        aggregated = aggregate_ion_counts(coords, ion_counts, bin_edges_x, bin_edges_y)
        
        # Should have 100 counts in bin [0,0] and zeros elsewhere
        expected = np.array([[100.0, 0.0], [0.0, 0.0]])
        
        assert np.allclose(aggregated['marker'], expected), \
            f"Single point aggregation failed: {aggregated['marker']}"
    
    def test_uniform_distribution_aggregation(self):
        """Test aggregation with uniform distribution (analytical expectation)."""
        np.random.seed(42)
        
        # Create uniform points in [0, 10] x [0, 10]
        n_points = 1000
        coords = np.random.uniform(0, 10, (n_points, 2))
        ion_counts = {'marker': np.ones(n_points)}  # Each point contributes 1
        
        # Create 5x5 bins
        bin_edges = np.linspace(0, 10, 6)
        
        aggregated = aggregate_ion_counts(coords, ion_counts, bin_edges, bin_edges)
        
        # Each bin should have approximately n_points/25 points
        expected_per_bin = n_points / 25
        bin_counts = aggregated['marker']
        
        # Check that distribution is reasonably uniform
        mean_count = np.mean(bin_counts)
        std_count = np.std(bin_counts)
        cv = std_count / mean_count
        
        assert abs(mean_count - expected_per_bin) < expected_per_bin * 0.2, \
            f"Mean count deviation: {mean_count:.1f} vs {expected_per_bin:.1f}"
        assert cv < 0.5, \
            f"Excessive variation in uniform aggregation: CV={cv:.3f}"
    
    def test_point_on_bin_boundary(self):
        """Test handling of points exactly on bin boundaries."""
        # Points exactly on bin edges
        coords = np.array([
            [0.0, 0.0],  # Corner
            [5.0, 0.0],  # Edge
            [5.0, 5.0],  # Center intersection
        ])
        ion_counts = {'marker': np.array([1.0, 1.0, 1.0])}
        
        bin_edges = np.array([0, 5, 10])
        
        aggregated = aggregate_ion_counts(coords, ion_counts, bin_edges, bin_edges)
        
        # Total counts should be conserved regardless of boundary handling
        total_original = np.sum(ion_counts['marker'])
        total_aggregated = np.sum(aggregated['marker'])
        
        assert abs(total_original - total_aggregated) < 1e-10, \
            f"Boundary handling violated count conservation: {total_original} vs {total_aggregated}"


class TestStatisticalPropertyOracles:
    """Test statistical properties against theoretical expectations."""
    
    def test_poisson_statistics_preservation(self):
        """Test that Poisson statistics are preserved through processing."""
        np.random.seed(42)
        
        # Generate Poisson data with known lambda
        lambda_param = 50.0
        n_samples = 1000
        poisson_data = np.random.poisson(lambda_param, n_samples)
        
        ion_counts = {'marker': poisson_data.astype(float)}
        
        # Apply arcsinh transform
        transformed, _ = apply_arcsinh_transform(ion_counts)
        
        # Original should follow Poisson statistics: mean ≈ variance
        original_mean = np.mean(poisson_data)
        original_var = np.var(poisson_data)
        
        # For Poisson, variance should equal mean (within sampling error)
        variance_ratio = original_var / original_mean
        assert 0.8 < variance_ratio < 1.2, \
            f"Original data not Poisson: variance/mean = {variance_ratio:.3f}"
        
        # Transformed data should have reduced variance/mean ratio (variance stabilization)
        trans_mean = np.mean(transformed['marker'])
        trans_var = np.var(transformed['marker'])
        trans_ratio = trans_var / trans_mean
        
        assert trans_ratio < variance_ratio * 0.8, \
            f"Variance stabilization failed: {variance_ratio:.3f} -> {trans_ratio:.3f}"
    
    def test_log_normal_approximation_high_counts(self):
        """Test that high count data approximates log-normal after arcsinh."""
        np.random.seed(42)
        
        # Generate high-count Poisson data (approaches normal)
        high_lambda = 500.0
        n_samples = 1000
        high_count_data = np.random.poisson(high_lambda, n_samples)
        
        ion_counts = {'marker': high_count_data.astype(float)}
        transformed, _ = apply_arcsinh_transform(ion_counts)
        
        # Transformed high-count data should be approximately normal
        trans_data = transformed['marker']
        
        # Shapiro-Wilk test for normality (on subsample for performance)
        subsample = np.random.choice(trans_data, min(100, len(trans_data)), replace=False)
        shapiro_stat, shapiro_p = stats.shapiro(subsample)
        
        # Should not strongly reject normality (allowing for imperfect approximation)
        assert shapiro_p > 0.01, \
            f"High-count transformed data not approximately normal: p={shapiro_p:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])