"""
Parallel execution tests for stability analysis.
"""

import multiprocessing
import time

import numpy as np
import pytest

from src.analysis.spatial_clustering import stability_analysis


@pytest.fixture
def large_dataset():
    rng = np.random.default_rng(7)
    feature_matrix = rng.standard_normal((800, 30))
    spatial_coords = rng.uniform(0, 200, (800, 2))
    return feature_matrix, spatial_coords


def test_parallel_determinism(large_dataset):
    feature_matrix, spatial_coords = large_dataset

    result1 = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_resolutions=10,
        n_bootstrap=4,
        random_state=123,
        parallel=True,
        n_workers=2,
    )
    result2 = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_resolutions=10,
        n_bootstrap=4,
        random_state=123,
        parallel=True,
        n_workers=2,
    )

    assert result1["parallel_used"] is True
    assert result2["parallel_used"] is True
    assert result1["optimal_resolution"] == result2["optimal_resolution"]
    assert np.allclose(result1["stability_scores"], result2["stability_scores"])


def test_parallel_vs_serial_agreement(large_dataset):
    feature_matrix, spatial_coords = large_dataset

    serial = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_resolutions=10,
        n_bootstrap=4,
        random_state=987,
        parallel=False,
    )
    parallel = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_resolutions=10,
        n_bootstrap=4,
        random_state=987,
        parallel=True,
        n_workers=2,
    )

    assert parallel["parallel_used"] is True
    assert abs(serial["optimal_resolution"] - parallel["optimal_resolution"]) < 0.05
    correlation = np.corrcoef(serial["stability_scores"], parallel["stability_scores"])[0, 1]
    assert correlation > 0.98


@pytest.mark.skipif(multiprocessing.cpu_count() < 2, reason="Parallel speedup needs >=2 cores")
def test_parallel_speed_reasonable(large_dataset):
    feature_matrix, spatial_coords = large_dataset

    start = time.perf_counter()
    serial = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_resolutions=12,
        n_bootstrap=4,
        random_state=222,
        parallel=False,
    )
    serial_time = time.perf_counter() - start

    start = time.perf_counter()
    parallel = stability_analysis(
        feature_matrix,
        spatial_coords,
        n_resolutions=12,
        n_bootstrap=4,
        random_state=222,
        parallel=True,
        n_workers=min(4, multiprocessing.cpu_count()),
    )
    parallel_time = time.perf_counter() - start

    assert parallel["parallel_used"] is True
    speedup = serial_time / parallel_time
    assert speedup >= 0.5
