"""
Shared test fixtures for IMC Analysis Pipeline tests.

Centralizes common test data generation and configuration to eliminate
duplication and improve maintainability.
"""

import numpy as np
import pytest
from types import SimpleNamespace
from typing import Dict, Any


@pytest.fixture
def random_seed():
    """Ensure deterministic tests."""
    np.random.seed(42)


@pytest.fixture
def mock_config():
    """Create a standard test configuration using SimpleNamespace."""
    return SimpleNamespace(
        multiscale=SimpleNamespace(
            scales_um=[10.0, 20.0, 40.0],
            enable_scale_analysis=True
        ),
        slic=SimpleNamespace(
            use_slic=True,
            compactness=10.0,
            sigma=2.0
        ),
        clustering=SimpleNamespace(
            method="leiden",
            resolution=1.0
        ),
        storage=SimpleNamespace(
            format="hdf5",
            compression=True
        ),
        normalization=SimpleNamespace(
            method="arcsinh",
            cofactor=1.0
        ),
        to_dict=lambda: {
            "multiscale": {"scales_um": [10.0, 20.0, 40.0], "enable_scale_analysis": True},
            "slic": {"use_slic": True, "compactness": 10.0, "sigma": 2.0},
            "clustering": {"method": "leiden", "resolution": 1.0},
            "storage": {"format": "hdf5", "compression": True},
            "normalization": {"method": "arcsinh", "cofactor": 1.0}
        }
    )


@pytest.fixture
def small_roi_data(random_seed):
    """Generate small ROI dataset for unit tests."""
    n_points = 100
    
    return {
        'coords': np.random.uniform(0, 50, (n_points, 2)),
        'ion_counts': {
            'CD45': np.random.negative_binomial(5, 0.3, n_points).astype(float),
            'CD31': np.random.negative_binomial(10, 0.5, n_points).astype(float),
        },
        'dna1_intensities': np.random.poisson(800, n_points).astype(float),
        'dna2_intensities': np.random.poisson(600, n_points).astype(float),
        'protein_names': ['CD45', 'CD31'],
        'n_measurements': n_points
    }


@pytest.fixture
def medium_roi_data(random_seed):
    """Generate medium ROI dataset for integration tests."""
    n_points = 1000
    
    return {
        'coords': np.random.uniform(0, 100, (n_points, 2)),
        'ion_counts': {
            'CD45': np.random.negative_binomial(5, 0.3, n_points).astype(float),
            'CD31': np.random.negative_binomial(10, 0.5, n_points).astype(float),
            'CD11b': np.random.negative_binomial(3, 0.2, n_points).astype(float)
        },
        'dna1_intensities': np.random.poisson(800, n_points).astype(float),
        'dna2_intensities': np.random.poisson(600, n_points).astype(float),
        'protein_names': ['CD45', 'CD31', 'CD11b'],
        'n_measurements': n_points
    }


@pytest.fixture
def large_roi_data(random_seed):
    """Generate large ROI dataset for performance tests."""
    n_points = 5000
    
    return {
        'coords': np.random.uniform(0, 200, (n_points, 2)),
        'ion_counts': {
            'CD45': np.random.negative_binomial(5, 0.3, n_points).astype(float),
            'CD31': np.random.negative_binomial(10, 0.5, n_points).astype(float),
            'CD11b': np.random.negative_binomial(3, 0.2, n_points).astype(float),
            'CD206': np.random.negative_binomial(8, 0.4, n_points).astype(float),
            'CD68': np.random.negative_binomial(6, 0.3, n_points).astype(float)
        },
        'dna1_intensities': np.random.poisson(800, n_points).astype(float),
        'dna2_intensities': np.random.poisson(600, n_points).astype(float),
        'protein_names': ['CD45', 'CD31', 'CD11b', 'CD206', 'CD68'],
        'n_measurements': n_points
    }


