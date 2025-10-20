"""
Regression guardrails for optimization work.

These tests ensure that computational complexity improvements do not alter
scientific results by checking determinism and cluster quality.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from src.analysis.spatial_clustering import perform_spatial_clustering, stability_analysis


@pytest.fixture(scope="module")
def synthetic_data():
    """Provide deterministic synthetic dataset used across regression tests."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 40
    feature_matrix = rng.standard_normal((n_samples, n_features))
    spatial_coords = rng.uniform(0, 100, (n_samples, 2))
    return feature_matrix, spatial_coords


def test_stability_analysis_deterministic(synthetic_data):
    """Repeated runs with the same seed should produce identical outputs."""
    feature_matrix, spatial_coords = synthetic_data

    result1 = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_resolutions=6,
        n_bootstrap=4,
        random_state=123,
        parallel=False,
    )
    result2 = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_resolutions=6,
        n_bootstrap=4,
        random_state=123,
        parallel=False,
    )

    assert result1["optimal_resolution"] == result2["optimal_resolution"]
    assert np.allclose(result1["stability_scores"], result2["stability_scores"])


def test_clustering_consistency(synthetic_data):
    """Clustering invocation should remain deterministic."""
    feature_matrix, spatial_coords = synthetic_data

    labels1, _ = perform_spatial_clustering(
        feature_matrix,
        spatial_coords,
        method="leiden",
        resolution=1.0,
        random_state=123,
    )
    labels2, _ = perform_spatial_clustering(
        feature_matrix,
        spatial_coords,
        method="leiden",
        resolution=1.0,
        random_state=123,
    )

    assert adjusted_rand_score(labels1, labels2) == 1.0


def test_stability_scores_reasonable(synthetic_data):
    """Baseline stability scores should remain within expected ranges."""
    feature_matrix, spatial_coords = synthetic_data

    result = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_resolutions=6,
        n_bootstrap=4,
        random_state=123,
        parallel=False,
    )

    max_score = max(result["stability_scores"])
    assert max_score >= 0.4, f"Unexpectedly low stability score: {max_score:.3f}"
    assert result["optimal_resolution"] > 0.0, "Optimal resolution should be positive"
