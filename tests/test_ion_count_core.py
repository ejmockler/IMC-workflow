"""
Core Tests for Ion Count Processing (Priority 1)

Tests the foundational ion count pipeline against the ACTUAL implementation:
1. estimate_optimal_cofactor() - percentile-based cofactor optimization
2. apply_arcsinh_transform() - Dict-based arcsinh with auto-cofactor
3. aggregate_ion_counts() - spatial binning of ion counts
4. Edge cases: zeros, sparse data, extreme values
5. Cofactor scaling with dynamic range
6. Preservation of biological signal order

These tests validate core preprocessing that all downstream analysis depends on.
"""

import pytest
import numpy as np

from src.analysis.ion_count_processing import (
    estimate_optimal_cofactor,
    apply_arcsinh_transform,
    aggregate_ion_counts
)


class TestCofactorEstimation:
    """Test automatic cofactor estimation."""

    def test_estimate_cofactor_percentile_method(self):
        """Test percentile-based cofactor estimation."""
        # Create data with known distribution
        np.random.seed(42)
        ion_counts = np.random.exponential(scale=100, size=1000)

        cofactor = estimate_optimal_cofactor(
            ion_counts,
            method="percentile",
            percentile_threshold=5.0
        )

        # Cofactor should be positive
        assert cofactor > 0

        # Should be close to 5th percentile of positive values
        p5 = np.percentile(ion_counts[ion_counts > 0], 5.0)
        assert 0.5 * p5 < cofactor <= p5 * 1.5  # Within reasonable range

    def test_cofactor_handles_all_zeros(self):
        """Test that all-zero data returns minimum cofactor."""
        ion_counts = np.zeros(100)

        cofactor = estimate_optimal_cofactor(ion_counts, method="percentile")

        # Should return minimum cofactor (0.1)
        assert cofactor == 0.1

    def test_cofactor_scales_with_intensity(self):
        """Test that cofactor adapts to data scale."""
        np.random.seed(42)

        # Low intensity data
        low_intensity = np.random.exponential(scale=10, size=1000)
        cofactor_low = estimate_optimal_cofactor(low_intensity, method="percentile")

        # High intensity data
        high_intensity = np.random.exponential(scale=1000, size=1000)
        cofactor_high = estimate_optimal_cofactor(high_intensity, method="percentile")

        # Higher intensity → higher cofactor
        assert cofactor_high > cofactor_low
        assert cofactor_high / cofactor_low > 5

    def test_cofactor_robust_to_sparse_data(self):
        """Test cofactor handles data with many zeros."""
        np.random.seed(42)

        # 90% zeros, 10% exponential signal
        ion_counts = np.zeros(1000)
        non_zero_indices = np.random.choice(1000, size=100, replace=False)
        ion_counts[non_zero_indices] = np.random.exponential(scale=100, size=100)

        cofactor = estimate_optimal_cofactor(ion_counts, method="percentile")

        # Should still get reasonable cofactor
        assert 0.1 < cofactor < 1000
        assert cofactor > 0

    def test_cofactor_mad_method(self):
        """Test MAD-based cofactor estimation."""
        np.random.seed(42)
        ion_counts = np.random.exponential(scale=100, size=1000)

        cofactor = estimate_optimal_cofactor(ion_counts, method="mad")

        assert cofactor > 0
        assert np.isfinite(cofactor)


class TestArcsinhTransformation:
    """Test arcsinh transformation with Dict API."""

    def test_arcsinh_transform_dict_api(self):
        """Test basic Dict-based arcsinh transformation."""
        # Create dict of ion count arrays
        ion_count_arrays = {
            "CD31": np.array([[10, 20], [30, 40]]),
            "CD34": np.array([[100, 200], [300, 400]]),
            "CD45": np.array([[1000, 2000], [3000, 4000]])
        }

        transformed, cofactors = apply_arcsinh_transform(
            ion_count_arrays,
            optimization_method="percentile",
            percentile_threshold=5.0
        )

        # Should return dict with same keys
        assert set(transformed.keys()) == set(ion_count_arrays.keys())

        # Should return cofactor for each protein
        assert set(cofactors.keys()) == set(ion_count_arrays.keys())

        # All cofactors should be positive
        assert all(cf > 0 for cf in cofactors.values())

        # Transformed arrays should have same shape
        for protein in ion_count_arrays:
            assert transformed[protein].shape == ion_count_arrays[protein].shape

    def test_arcsinh_per_protein_cofactor_optimization(self):
        """Test that per-protein cofactor optimization works correctly."""
        np.random.seed(42)

        # Create data where CD31 >> CD34 >> CD45 in raw scale
        ion_count_arrays = {
            "CD31": np.random.exponential(scale=1000, size=(200, 200)),
            "CD34": np.random.exponential(scale=100, size=(200, 200)),
            "CD45": np.random.exponential(scale=10, size=(200, 200))
        }

        transformed, cofactors = apply_arcsinh_transform(
            ion_count_arrays,
            optimization_method="percentile"
        )

        # CORRECT BEHAVIOR: Cofactors should scale with expression level
        # This allows low-abundance markers to be analyzed effectively
        assert cofactors["CD31"] > cofactors["CD34"] > cofactors["CD45"]

        # CORRECT BEHAVIOR: Variance stabilization means transformed ranges are similar
        # This is the GOAL - allows fair clustering/analysis across all markers
        std_cd31 = np.std(transformed["CD31"])
        std_cd34 = np.std(transformed["CD34"])
        std_cd45 = np.std(transformed["CD45"])

        # Stds should be in similar range (within 2x of each other)
        # This proves variance stabilization is working
        max_std = max(std_cd31, std_cd34, std_cd45)
        min_std = min(std_cd31, std_cd34, std_cd45)
        assert max_std / min_std < 2.0  # Well-stabilized

    def test_arcsinh_handles_zeros(self):
        """Test that zero values transform correctly."""
        ion_count_arrays = {
            "protein1": np.array([[0.0, 0.0], [0.0, 0.0]])
        }

        transformed, cofactors = apply_arcsinh_transform(
            ion_count_arrays,
            optimization_method="percentile"
        )

        # arcsinh(0) = 0
        assert np.all(transformed["protein1"] == 0.0)

    def test_arcsinh_cached_cofactors(self):
        """Test using cached cofactors (optimization='cached')."""
        ion_count_arrays = {
            "CD31": np.array([[10, 20], [30, 40]]),
            "CD34": np.array([[100, 200], [300, 400]])
        }

        # Pre-computed cofactors
        cached_cofactors = {
            "CD31": 5.0,
            "CD34": 50.0
        }

        transformed, cofactors_used = apply_arcsinh_transform(
            ion_count_arrays,
            optimization_method="cached",
            cached_cofactors=cached_cofactors
        )

        # Should use cached cofactors
        assert cofactors_used == cached_cofactors

    def test_arcsinh_output_finite(self):
        """Test that arcsinh output is always finite."""
        np.random.seed(42)

        ion_count_arrays = {
            "protein1": np.random.exponential(scale=100, size=(50, 50)),
            "protein2": np.random.exponential(scale=1000, size=(50, 50))
        }

        transformed, cofactors = apply_arcsinh_transform(
            ion_count_arrays,
            optimization_method="percentile"
        )

        # All outputs should be finite
        for protein, data in transformed.items():
            assert np.all(np.isfinite(data))


class TestIonCountAggregation:
    """Test spatial aggregation of ion counts."""

    def test_aggregate_basic(self):
        """Test basic spatial binning."""
        # Create simple coordinate grid
        coords = np.array([
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5]
        ])

        # Ion counts for single protein
        ion_counts = {
            "protein1": np.array([10, 20, 30, 40])
        }

        # Bin edges for 2x2 grid (0-2 in x and y)
        bin_edges_x = np.array([0, 1, 2])
        bin_edges_y = np.array([0, 1, 2])

        aggregated = aggregate_ion_counts(
            coords,
            ion_counts,
            bin_edges_x,
            bin_edges_y
        )

        # Should have 2x2 array
        assert aggregated["protein1"].shape == (2, 2)

        # Check aggregation: each pixel goes to correct bin
        expected = np.array([
            [10, 20],  # Bottom row
            [30, 40]   # Top row
        ])
        np.testing.assert_array_equal(aggregated["protein1"], expected)

    def test_aggregate_sums_not_averages(self):
        """Test that aggregation SUMS counts (preserves Poisson stats)."""
        # Multiple pixels in same bin
        coords = np.array([
            [0.5, 0.5],
            [0.6, 0.5],
            [0.7, 0.5]
        ])

        ion_counts = {
            "protein1": np.array([10, 20, 30])
        }

        bin_edges_x = np.array([0, 1])
        bin_edges_y = np.array([0, 1])

        aggregated = aggregate_ion_counts(
            coords,
            ion_counts,
            bin_edges_x,
            bin_edges_y
        )

        # Should sum to 60 (not average to 20)
        assert aggregated["protein1"][0, 0] == 60

    def test_aggregate_handles_empty_bins(self):
        """Test that empty bins get zero counts."""
        coords = np.array([[0.5, 0.5]])  # Only one pixel
        ion_counts = {"protein1": np.array([100])}

        bin_edges_x = np.array([0, 1, 2, 3])  # 3 bins
        bin_edges_y = np.array([0, 1, 2, 3])

        aggregated = aggregate_ion_counts(
            coords,
            ion_counts,
            bin_edges_x,
            bin_edges_y
        )

        # Should be 3x3 grid
        assert aggregated["protein1"].shape == (3, 3)

        # Only one bin should have counts
        assert np.sum(aggregated["protein1"] > 0) == 1
        assert aggregated["protein1"][0, 0] == 100

    def test_aggregate_multiple_proteins(self):
        """Test aggregation with multiple proteins."""
        coords = np.array([[0.5, 0.5], [1.5, 1.5]])

        ion_counts = {
            "CD31": np.array([10, 20]),
            "CD34": np.array([30, 40]),
            "CD45": np.array([50, 60])
        }

        bin_edges_x = np.array([0, 1, 2])
        bin_edges_y = np.array([0, 1, 2])

        aggregated = aggregate_ion_counts(
            coords,
            ion_counts,
            bin_edges_x,
            bin_edges_y
        )

        # Should aggregate all proteins independently
        assert set(aggregated.keys()) == {"CD31", "CD34", "CD45"}
        assert aggregated["CD31"][0, 0] == 10
        assert aggregated["CD34"][0, 0] == 30
        assert aggregated["CD45"][0, 0] == 50

    def test_aggregate_empty_input(self):
        """Test that empty input returns empty dict."""
        coords = np.array([]).reshape(0, 2)
        ion_counts = {}

        bin_edges_x = np.array([0, 1])
        bin_edges_y = np.array([0, 1])

        aggregated = aggregate_ion_counts(
            coords,
            ion_counts,
            bin_edges_x,
            bin_edges_y
        )

        assert aggregated == {}


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_protein_single_pixel(self):
        """Test minimal input doesn't crash."""
        ion_count_arrays = {
            "protein1": np.array([[100]])
        }

        transformed, cofactors = apply_arcsinh_transform(
            ion_count_arrays,
            optimization_method="percentile"
        )

        assert transformed["protein1"].shape == (1, 1)
        assert np.isfinite(transformed["protein1"][0, 0])

    def test_extreme_values_handled(self):
        """Test that very large values don't overflow."""
        ion_count_arrays = {
            "protein1": np.array([[1e6, 1e7], [1e8, 1e9]])
        }

        transformed, cofactors = apply_arcsinh_transform(
            ion_count_arrays,
            optimization_method="percentile"
        )

        # Should not overflow
        assert np.all(np.isfinite(transformed["protein1"]))

    def test_cofactor_minimum_enforced(self):
        """Test that cofactor has minimum value."""
        # All very small values
        ion_counts = np.array([0.001, 0.002, 0.003])

        cofactor = estimate_optimal_cofactor(ion_counts, method="percentile")

        # Should enforce minimum of 0.1
        assert cofactor >= 0.1


class TestBiologicalSignalPreservation:
    """Test that transformation preserves biologically relevant patterns."""

    def test_variance_stabilization_works(self):
        """Test that per-protein cofactor achieves variance stabilization."""
        np.random.seed(42)

        # Simulate typical IMC marker expression patterns with 100x range
        ion_count_arrays = {
            "High_Marker": np.random.exponential(scale=2000, size=(200, 200)),  # Abundant
            "Medium_Marker": np.random.exponential(scale=200, size=(200, 200)),  # Moderate
            "Low_Marker": np.random.exponential(scale=20, size=(200, 200))  # Rare
        }

        transformed, cofactors = apply_arcsinh_transform(
            ion_count_arrays,
            optimization_method="percentile"
        )

        # CORRECT BEHAVIOR: Cofactors adapt to each marker's scale
        assert cofactors["High_Marker"] > cofactors["Medium_Marker"] > cofactors["Low_Marker"]

        # CRITICAL: Variance stabilization working?
        # Raw data has 100x scale difference → transformed should be much closer
        std_high = np.std(transformed["High_Marker"])
        std_med = np.std(transformed["Medium_Marker"])
        std_low = np.std(transformed["Low_Marker"])

        # All stds should be in similar range (successful stabilization)
        stds = [std_high, std_med, std_low]
        max_std = max(stds)
        min_std = min(stds)

        # Stabilization means stds within 3x of each other (down from 100x in raw data)
        assert max_std / min_std < 3.0

    def test_spatial_patterns_preserved(self):
        """Test that spatial gradients are preserved."""
        # Create gradient pattern: high on left, low on right
        x_coords = np.linspace(0, 10, 100)
        gradient_pattern = 1000 - 90 * x_coords  # Linear decay

        ion_count_arrays = {
            "protein1": gradient_pattern.reshape(10, 10)
        }

        transformed, cofactors = apply_arcsinh_transform(
            ion_count_arrays,
            optimization_method="percentile"
        )

        # Check that left > right still holds
        left_mean = np.mean(transformed["protein1"][:, :5])
        right_mean = np.mean(transformed["protein1"][:, 5:])

        assert left_mean > right_mean
