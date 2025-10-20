"""
Adaptive resolution search tests.
"""

import numpy as np

from src.analysis.spatial_clustering import stability_analysis
from src.analysis.multiscale_analysis import perform_multiscale_analysis


def _create_dataset(seed: int = 123, n_samples: int = 300, n_features: int = 35):
    rng = np.random.default_rng(seed)
    feature_matrix = rng.standard_normal((n_samples, n_features))
    spatial_coords = rng.uniform(0, 150, (n_samples, 2))
    return feature_matrix, spatial_coords


def test_adaptive_matches_grid_with_fewer_evaluations():
    feature_matrix, spatial_coords = _create_dataset()

    adaptive = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_resolutions=15,
        n_bootstrap=5,
        adaptive_search=True,
        adaptive_target_stability=0.55,
        random_state=42,
        parallel=False,
    )

    grid = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_resolutions=15,
        n_bootstrap=5,
        adaptive_search=False,
        random_state=42,
        parallel=False,
    )

    diff = abs(adaptive["optimal_resolution"] - grid["optimal_resolution"])
    assert diff < 0.2
    assert adaptive["n_evaluations"] < len(grid["resolutions"])
    assert adaptive.get("adaptive_search_used", False) is True


def test_adaptive_deterministic():
    feature_matrix, spatial_coords = _create_dataset(seed=77)

    result1 = stability_analysis(
        feature_matrix,
        spatial_coords,
        adaptive_search=True,
        random_state=21,
        parallel=False,
    )
    result2 = stability_analysis(
        feature_matrix,
        spatial_coords,
        adaptive_search=True,
        random_state=21,
        parallel=False,
    )

    assert result1["resolutions"] == result2["resolutions"]
    assert np.allclose(result1["stability_scores"], result2["stability_scores"])


def test_multiscale_uses_config_adaptive():
    rng = np.random.default_rng(2024)
    n_cells = 250
    coords = rng.uniform(0, 200, (n_cells, 2))
    ion_counts = {
        "MarkerA": rng.poisson(120, n_cells).astype(float),
        "MarkerB": rng.poisson(80, n_cells).astype(float),
        "MarkerC": rng.poisson(60, n_cells).astype(float),
    }
    dna1 = rng.poisson(500, n_cells).astype(float)
    dna2 = rng.poisson(520, n_cells).astype(float)

    stability_settings = {
        "n_bootstrap_iterations": 3,
        "n_resolutions": 12,
        "use_graph_caching": True,
        "parallel_execution": False,
        "adaptive_search": True,
        "adaptive_target_stability": 0.4,
        "adaptive_tolerance": 0.05,
        "adaptive_max_evaluations": 6,
    }

    class _StubConfig:
        def __init__(self, optimization):
            self.optimization = optimization

    config = _StubConfig({"stability_analysis": stability_settings})

    results = perform_multiscale_analysis(
        coords=coords,
        ion_counts=ion_counts,
        dna1_intensities=dna1,
        dna2_intensities=dna2,
        scales_um=[10.0],
        method="leiden",
        segmentation_method="grid",
        config=config,
    )

    stability = results[10.0]["stability_analysis"]
    assert stability.get("adaptive_search_used", False)
    assert stability["n_evaluations"] <= stability_settings["adaptive_max_evaluations"]
