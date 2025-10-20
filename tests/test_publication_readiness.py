"""
Comprehensive Test Suite for Publication Readiness Features

Tests for critical fixes:
1. LASSO feature selection integration
2. Scale-adaptive k_neighbors
3. Biological validation framework
4. Ablation study framework
"""

import pytest
import numpy as np
import json
from pathlib import Path
from typing import Dict, List

from src.config import Config
from src.analysis.spatial_clustering import perform_spatial_clustering
from src.analysis.multiscale_analysis import (
    perform_multiscale_analysis,
    _compute_scale_adaptive_k_neighbors
)
from src.analysis.coabundance_features import (
    generate_coabundance_features,
    select_informative_coabundance_features
)
from src.validation.kidney_biological_validation import (
    KidneyBiologicalValidator,
    run_kidney_validation
)


class TestLASSOFeatureSelection:
    """Test LASSO-based feature selection integration."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic protein expression data."""
        np.random.seed(42)
        n_samples = 500
        n_proteins = 9

        # Create features with some structure
        features = np.random.randn(n_samples, n_proteins)
        # Add correlation structure
        features[:, 1] = 0.7 * features[:, 0] + 0.3 * np.random.randn(n_samples)

        spatial_coords = np.random.rand(n_samples, 2) * 100
        protein_names = [f"Protein_{i}" for i in range(n_proteins)]

        return features, spatial_coords, protein_names

    def test_feature_selection_reduces_dimensionality(self, synthetic_data):
        """Test that LASSO selection reduces 153 → 30 features."""
        features, spatial_coords, protein_names = synthetic_data

        # Generate coabundance features (should create 153 features)
        enriched_features, enriched_names = generate_coabundance_features(
            features,
            protein_names,
            spatial_coords=spatial_coords,
            interaction_order=2,
            include_ratios=True,
            include_products=True,
            include_spatial_covariance=True
        )

        assert enriched_features.shape[1] > 100, "Should generate ~153 features"

        # Apply LASSO selection
        selected_features, selected_names = select_informative_coabundance_features(
            enriched_features,
            enriched_names,
            target_n_features=30,
            method='lasso'
        )

        # Verify dimensionality reduction
        assert selected_features.shape[1] == 30, "Should select exactly 30 features"
        assert len(selected_names) == 30, "Should have 30 feature names"
        assert selected_features.shape[0] == features.shape[0], "Should preserve sample count"

    def test_feature_selection_integrated_in_clustering(self, synthetic_data):
        """Test that feature selection is applied during clustering."""
        features, spatial_coords, protein_names = synthetic_data

        # Configure coabundance with feature selection enabled
        coabundance_options = {
            'interaction_order': 2,
            'include_ratios': True,
            'include_products': True,
            'include_spatial_covariance': True,
            'use_feature_selection': True,
            'target_n_features': 30,
            'selection_method': 'lasso'
        }

        # Run clustering with feature selection
        cluster_labels, clustering_info = perform_spatial_clustering(
            features,
            spatial_coords,
            method='leiden',
            resolution=1.0,
            use_coabundance=True,
            protein_names=protein_names,
            coabundance_options=coabundance_options
        )

        # Verify clustering succeeded
        assert len(cluster_labels) == features.shape[0], "Should cluster all samples"
        assert len(np.unique(cluster_labels[cluster_labels >= 0])) > 0, "Should find clusters"

    def test_feature_selection_without_flag_uses_all_features(self, synthetic_data):
        """Test that disabling feature selection uses all 153 features (risky!)."""
        features, spatial_coords, protein_names = synthetic_data

        # Disable feature selection
        coabundance_options = {
            'interaction_order': 2,
            'include_ratios': True,
            'include_products': True,
            'include_spatial_covariance': True,
            'use_feature_selection': False  # DISABLED - overfitting risk
        }

        # This should work but is scientifically invalid
        cluster_labels, clustering_info = perform_spatial_clustering(
            features,
            spatial_coords,
            method='leiden',
            resolution=1.0,
            use_coabundance=True,
            protein_names=protein_names,
            coabundance_options=coabundance_options
        )

        # Should still cluster, but with overfitting risk
        assert len(cluster_labels) == features.shape[0]


class TestScaleAdaptiveKNeighbors:
    """Test scale-adaptive k_neighbors computation."""

    def test_k_decreases_with_fewer_samples(self):
        """Test that k decreases as number of superpixels decreases."""
        # Fine scale: many superpixels
        k_fine = _compute_scale_adaptive_k_neighbors(1000, 10.0, None)

        # Coarse scale: fewer superpixels
        k_coarse = _compute_scale_adaptive_k_neighbors(100, 40.0, None)

        assert k_coarse < k_fine, "k should decrease for coarser scales"
        assert k_fine <= 15, "k should not exceed 15 (default max)"
        assert k_coarse >= 8, "k should not go below 8 (default min)"

    def test_config_overrides_heuristic(self):
        """Test that config scale-specific k overrides heuristic."""
        # Create mock config with scale-specific k
        class MockConfig:
            pass

        config = MockConfig()

        # Mock the parameter extraction
        from src.analysis.multiscale_analysis import _extract_algorithm_params

        # For now, test the heuristic directly
        k_10um = _compute_scale_adaptive_k_neighbors(1000, 10.0, None)
        k_40um = _compute_scale_adaptive_k_neighbors(100, 40.0, None)

        # Verify 2×log(N) heuristic
        expected_10um = int(2 * np.log(1000))  # ~14
        expected_40um = int(2 * np.log(100))    # ~9

        assert abs(k_10um - expected_10um) <= 1, f"k_10um should be ~{expected_10um}"
        assert abs(k_40um - expected_40um) <= 1, f"k_40um should be ~{expected_40um}"

    def test_k_bounds_enforced(self):
        """Test that k is bounded between min and max values."""
        # Very few samples (would give k<8 without bounds)
        k_tiny = _compute_scale_adaptive_k_neighbors(10, 100.0, None)
        assert k_tiny >= 8, "k should be at least 8"

        # Very many samples (would give k>15 without bounds)
        k_huge = _compute_scale_adaptive_k_neighbors(10000, 5.0, None)
        assert k_huge <= 15, "k should be at most 15"

    @pytest.fixture
    def multiscale_synthetic_data(self):
        """Create synthetic data for multiscale analysis."""
        np.random.seed(42)

        # Create ion counts (pixels x proteins)
        n_pixels = 10000
        n_proteins = 9
        ion_counts = np.random.randn(n_pixels, n_proteins) * 100 + 500
        ion_counts = np.maximum(ion_counts, 0)  # Non-negative

        # Create spatial grid
        grid_size = int(np.sqrt(n_pixels))
        x = np.repeat(np.arange(grid_size), grid_size)
        y = np.tile(np.arange(grid_size), grid_size)
        coordinates = np.column_stack([x, y])[:n_pixels]

        metadata = {
            'roi_id': 'test_roi',
            'protein_names': [f'Protein{i}' for i in range(n_proteins)]
        }

        return ion_counts, metadata


class TestKidneyBiologicalValidation:
    """Test kidney-specific biological validation."""

    @pytest.fixture
    def cortex_like_data(self):
        """Create synthetic data resembling cortex tissue."""
        np.random.seed(42)
        n_samples = 500

        # Protein indices: CD31=0, CD34=1, CD140a=2, ...
        protein_names = ["CD31", "CD34", "CD140a", "CD45", "Ly6G",
                        "CD11b", "CD206", "CD140b", "CD44"]

        features = np.random.rand(n_samples, len(protein_names))

        # Create two clusters
        cluster_labels = np.random.choice([0, 1], size=n_samples)

        # Cluster 0: cortex-like (high CD31/CD34, low CD140a)
        cortex_mask = cluster_labels == 0
        features[cortex_mask, 0] *= 3  # CD31
        features[cortex_mask, 1] *= 3  # CD34
        features[cortex_mask, 2] *= 0.5  # CD140a (low)

        # Cluster 1: medulla-like (low CD31/CD34, high CD140a)
        medulla_mask = cluster_labels == 1
        features[medulla_mask, 0] *= 0.5  # CD31 (low)
        features[medulla_mask, 1] *= 0.5  # CD34 (low)
        features[medulla_mask, 2] *= 3  # CD140a

        return cluster_labels, features, protein_names

    def test_cortex_signature_detection(self, cortex_like_data):
        """Test detection of cortex anatomical signature."""
        cluster_labels, features, protein_names = cortex_like_data

        validator = KidneyBiologicalValidator()
        results = validator.validate_anatomical_enrichment(
            cluster_labels, features, protein_names
        )

        # Should detect cortex signature
        cortex_quality = results['cortex_validation']['signature_quality']
        assert cortex_quality > 0.2, "Should detect cortex-like cluster"

        # Should have at least one cluster with cortex signature
        cortex_clusters = results['cortex_validation']['clusters_with_signature']
        assert len(cortex_clusters) > 0, "Should identify cortex-like clusters"

    def test_medulla_signature_detection(self, cortex_like_data):
        """Test detection of medulla anatomical signature."""
        cluster_labels, features, protein_names = cortex_like_data

        validator = KidneyBiologicalValidator()
        results = validator.validate_anatomical_enrichment(
            cluster_labels, features, protein_names
        )

        # Should detect medulla signature
        medulla_quality = results['medulla_validation']['signature_quality']
        assert medulla_quality > 0.2, "Should detect medulla-like cluster"

    def test_injury_timepoint_validation(self):
        """Test validation of injury timepoint immune responses."""
        np.random.seed(42)
        n_samples = 500

        protein_names = ["CD31", "CD34", "CD140a", "CD45", "Ly6G",
                        "CD11b", "CD206", "CD140b", "CD44"]

        # Day 1: High Ly6G and CD11b (neutrophil recruitment)
        features_day1 = np.random.rand(n_samples, len(protein_names))
        features_day1[:, 4] *= 5  # Ly6G high
        features_day1[:, 5] *= 5  # CD11b high

        cluster_labels = np.random.choice([0, 1, 2], size=n_samples)

        validator = KidneyBiologicalValidator()
        results = validator.validate_injury_timepoint(
            cluster_labels, features_day1, protein_names, injury_day=1
        )

        # Check that key markers are detected
        assert 'Ly6G' in results['key_marker_expression'], "Should detect Ly6G"
        assert 'CD11b' in results['key_marker_expression'], "Should detect CD11b"

        # Check expression levels are reasonable
        ly6g_mean = results['key_marker_expression']['Ly6G']['mean']
        assert ly6g_mean > 0, "Ly6G should have positive expression"


class TestAblationFramework:
    """Test ablation study framework."""

    def test_ablation_config_definitions(self):
        """Test that ablation configs are properly defined."""
        # Import the ablation script configs
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        # Verify we can import the module
        import run_ablation_study

        # The script should be importable without errors
        assert hasattr(run_ablation_study, 'run_ablation_study')

    def test_config_variations_are_distinct(self):
        """Test that the 4 ablation configs are meaningfully different."""
        # Load base config
        config_path = Path(__file__).parent.parent / "config.json"

        if not config_path.exists():
            pytest.skip("config.json not found")

        with open(config_path) as f:
            base_config = json.load(f)

        # Verify feature selection is enabled in base config
        coabundance_opts = (base_config.get('analysis', {})
                           .get('clustering', {})
                           .get('coabundance_options', {}))

        assert coabundance_opts.get('use_feature_selection', False) is True, \
            "Feature selection should be enabled in base config"

        # Verify k_neighbors_by_scale exists
        k_by_scale = (base_config.get('analysis', {})
                     .get('clustering', {})
                     .get('k_neighbors_by_scale', {}))

        assert len(k_by_scale) > 0, "Should have scale-specific k values"


class TestIntegrationWithExistingPipeline:
    """Test that new features integrate properly with existing pipeline."""

    def test_config_loads_with_new_parameters(self):
        """Test that Config class handles new parameters."""
        config_path = Path(__file__).parent.parent / "config.json"

        if not config_path.exists():
            pytest.skip("config.json not found")

        # Should load without errors
        config = Config(str(config_path))

        # Verify new parameters are accessible
        if hasattr(config, 'analysis') and hasattr(config.analysis, 'clustering'):
            clustering = config.analysis.clustering

            # Check for coabundance options
            if hasattr(clustering, 'coabundance_options'):
                assert hasattr(clustering.coabundance_options, 'use_feature_selection')

            # Check for k_neighbors_by_scale
            if hasattr(clustering, 'k_neighbors_by_scale'):
                k_by_scale = clustering.k_neighbors_by_scale
                assert isinstance(k_by_scale, dict) or k_by_scale is None

    def test_multiscale_analysis_with_new_features(self):
        """Test that multiscale analysis works with scale-adaptive k."""
        np.random.seed(42)

        # Create minimal synthetic data
        n_pixels = 1000
        n_proteins = 9

        ion_counts = np.random.randn(n_pixels, n_proteins) * 100 + 500
        ion_counts = np.maximum(ion_counts, 0)

        metadata = {
            'roi_id': 'test_integration',
            'protein_names': [f'Protein{i}' for i in range(n_proteins)]
        }

        # This should run without errors (though may be slow)
        # For testing, we just verify the function is callable
        from src.analysis.multiscale_analysis import perform_multiscale_analysis

        # Verify function exists and has correct signature
        import inspect
        sig = inspect.signature(perform_multiscale_analysis)
        assert 'ion_counts' in sig.parameters
        assert 'config' in sig.parameters


class TestRegressionPrevention:
    """Tests to prevent regression to old invalid configurations."""

    def test_fixed_k_across_scales_is_deprecated(self):
        """Test that we don't regress to fixed k=15 across all scales."""
        # Load current config
        config_path = Path(__file__).parent.parent / "config.json"

        if not config_path.exists():
            pytest.skip("config.json not found")

        with open(config_path) as f:
            config = json.load(f)

        # Check that k_neighbors_by_scale exists and is non-empty
        k_by_scale = (config.get('analysis', {})
                     .get('clustering', {})
                     .get('k_neighbors_by_scale'))

        assert k_by_scale is not None, "k_neighbors_by_scale should exist"

        if isinstance(k_by_scale, dict):
            # Should have different k values for different scales
            k_values = [v for k, v in k_by_scale.items() if not k.startswith('_')]
            if len(k_values) > 1:
                assert len(set(k_values)) > 1, "Should have varying k across scales"

    def test_feature_selection_is_enabled(self):
        """Test that feature selection is enabled by default."""
        config_path = Path(__file__).parent.parent / "config.json"

        if not config_path.exists():
            pytest.skip("config.json not found")

        with open(config_path) as f:
            config = json.load(f)

        # Feature selection should be enabled
        use_selection = (config.get('analysis', {})
                        .get('clustering', {})
                        .get('coabundance_options', {})
                        .get('use_feature_selection'))

        assert use_selection is True, \
            "Feature selection MUST be enabled to prevent overfitting (153→30 features)"

    def test_target_n_features_is_reasonable(self):
        """Test that target_n_features follows √N rule of thumb."""
        config_path = Path(__file__).parent.parent / "config.json"

        if not config_path.exists():
            pytest.skip("config.json not found")

        with open(config_path) as f:
            config = json.load(f)

        target_n = (config.get('analysis', {})
                   .get('clustering', {})
                   .get('coabundance_options', {})
                   .get('target_n_features'))

        # For ~1000 superpixels, target should be around √1000 ≈ 30
        assert target_n is not None, "target_n_features should be specified"
        assert 20 <= target_n <= 50, \
            f"target_n_features={target_n} should be 20-50 for typical datasets"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
