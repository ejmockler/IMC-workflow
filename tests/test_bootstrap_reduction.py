"""
Validate reduced bootstrap iterations maintain stability accuracy.
"""

import numpy as np

from src.analysis.spatial_clustering import stability_analysis


def test_bootstrap_convergence():
    """Optimal resolution should converge by five bootstrap iterations."""
    np.random.seed(42)
    feature_matrix = np.random.randn(200, 40)
    spatial_coords = np.random.uniform(0, 100, (200, 2))

    optimal_by_bootstrap = {}
    for n_bootstrap in [3, 5, 7, 10, 15]:
        result = stability_analysis(
            feature_matrix,
            spatial_coords,
            n_bootstrap=n_bootstrap,
            random_state=123,
            n_resolutions=8,
            parallel=False
        )
        optimal_by_bootstrap[n_bootstrap] = result["optimal_resolution"]

    assert abs(optimal_by_bootstrap[5] - optimal_by_bootstrap[10]) < 0.05
    assert abs(optimal_by_bootstrap[5] - optimal_by_bootstrap[15]) < 0.05


def test_bootstrap_quality_threshold():
    """Even with five iterations, stability scores remain high."""
    np.random.seed(24)
    feature_matrix = np.random.randn(200, 40)
    spatial_coords = np.random.uniform(0, 100, (200, 2))

    result = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_bootstrap=5,
        random_state=99,
        n_resolutions=8,
        parallel=False
    )

    assert max(result["stability_scores"]) >= 0.45
