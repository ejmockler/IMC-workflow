"""
Tests ensuring graph caching matches baseline behaviour and improves speed.
"""

import time

import numpy as np

from src.analysis.spatial_clustering import stability_analysis


def _generate_data(seed: int = 42, n_samples: int = 250):
    rng = np.random.default_rng(seed)
    feature_matrix = rng.standard_normal((n_samples, 30))
    spatial_coords = rng.uniform(0, 100, (n_samples, 2))
    return feature_matrix, spatial_coords


def test_caching_equivalence():
    """Graph caching should closely match baseline stability results."""
    feature_matrix, spatial_coords = _generate_data()

    no_cache = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_resolutions=6,
        n_bootstrap=4,
        random_state=321,
        use_graph_caching=False,
        parallel=False,
    )
    cached = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_resolutions=6,
        n_bootstrap=4,
        random_state=321,
        use_graph_caching=True,
        parallel=False,
    )

    assert abs(no_cache["optimal_resolution"] - cached["optimal_resolution"]) < 0.15
    correlation = np.corrcoef(no_cache["stability_scores"], cached["stability_scores"])[0, 1]
    assert correlation > 0.95
    assert cached["graph_caching_used"] is True


def test_caching_speedup():
    """Caching should provide a measurable speedup for repeated runs."""
    feature_matrix, spatial_coords = _generate_data(seed=7, n_samples=400)

    def _time_run(use_cache: bool) -> float:
        timings = []
        for _ in range(2):
            start = time.perf_counter()
            stability_analysis(
                feature_matrix,
                spatial_coords,
                n_resolutions=10,
                n_bootstrap=6,
                random_state=11,
                use_graph_caching=use_cache,
                parallel=False,
            )
            timings.append(time.perf_counter() - start)
        return min(timings)

    baseline_time = _time_run(use_cache=False)
    cached_time = _time_run(use_cache=True)

    speedup = baseline_time / cached_time if cached_time > 0 else 0
    assert speedup >= 0.9
