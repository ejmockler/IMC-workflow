"""
Shared test fixtures for IMC Analysis Pipeline tests.

Centralizes common test data generation and configuration to eliminate
duplication and improve maintainability.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
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
            optimization_method="comprehensive",
            k_range=[2, 8]
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
            "clustering": {"optimization_method": "comprehensive", "k_range": [2, 8]},
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


@pytest.fixture
def empty_roi_data():
    """Generate empty ROI dataset for edge case testing."""
    return {
        'coords': np.array([]).reshape(0, 2),
        'ion_counts': {},
        'dna1_intensities': np.array([]),
        'dna2_intensities': np.array([]),
        'protein_names': [],
        'n_measurements': 0
    }


@pytest.fixture
def temp_roi_file():
    """Create a temporary ROI file with realistic IMC data."""
    import pandas as pd
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        # Create realistic IMC data
        df = pd.DataFrame({
            'X': [10.5, 20.3, 30.1, 15.7, 25.9],
            'Y': [15.2, 25.8, 35.4, 18.9, 28.3],
            'CD45(Sm149Di)': [100, 200, 150, 180, 120],
            'CD31(Nd145Di)': [50, 300, 100, 250, 80],
            'DNA1(Ir191Di)': [800, 900, 850, 920, 780],
            'DNA2(Ir193Di)': [600, 650, 620, 680, 580]
        })
        df.to_csv(f.name, sep='\t', index=False)
        
        yield f.name
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def malicious_pickle_file(temp_directory):
    """Create a pickle file for security testing."""
    import pickle
    
    # Create a benign pickle file (we won't actually test RCE execution)
    pickle_path = Path(temp_directory) / "test_data.pkl"
    test_data = {"test": "data", "numbers": [1, 2, 3]}
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(test_data, f)
    
    return pickle_path