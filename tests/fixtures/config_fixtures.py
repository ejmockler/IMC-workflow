"""
Standard configuration fixtures for IMC analysis testing.

Provides consistent configuration objects to prevent test duplication
and ensure standardized test parameters across the test suite.
"""

import pytest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class TestConfigProfile:
    """Configuration profile for different test scenarios."""
    name: str
    description: str
    scales_um: list
    clustering_method: str
    clustering_resolution: float
    use_slic: bool
    slic_compactness: float
    normalization_method: str
    normalization_cofactor: float
    storage_format: str
    storage_compression: bool
    enable_coabundance: bool = True
    enable_multiscale: bool = True
    quality_gates_enabled: bool = False


class ConfigProfiles:
    """Predefined configuration profiles for different testing scenarios."""
    
    # Fast configuration for unit tests
    UNIT_TEST = TestConfigProfile(
        name="unit_test",
        description="Minimal config for fast unit tests",
        scales_um=[20.0],  # Single scale for speed
        clustering_method="leiden",
        clustering_resolution=1.0,
        use_slic=False,  # Skip expensive SLIC
        slic_compactness=10.0,
        normalization_method="arcsinh",
        normalization_cofactor=1.0,
        storage_format="json",
        storage_compression=False,
        enable_coabundance=True,
        enable_multiscale=False,  # Single scale only
        quality_gates_enabled=False
    )
    
    # Standard configuration for integration tests
    INTEGRATION_TEST = TestConfigProfile(
        name="integration_test", 
        description="Standard config for integration tests",
        scales_um=[10.0, 20.0, 40.0],
        clustering_method="leiden",
        clustering_resolution=1.0,
        use_slic=True,
        slic_compactness=10.0,
        normalization_method="arcsinh",
        normalization_cofactor=1.0,
        storage_format="hdf5",
        storage_compression=True,
        enable_coabundance=True,
        enable_multiscale=True,
        quality_gates_enabled=True
    )
    
    # Performance testing configuration
    PERFORMANCE_TEST = TestConfigProfile(
        name="performance_test",
        description="Optimized config for performance testing",
        scales_um=[20.0, 40.0],  # Limited scales
        clustering_method="leiden",
        clustering_resolution=1.0,
        use_slic=False,  # Skip expensive operations
        slic_compactness=10.0,
        normalization_method="arcsinh", 
        normalization_cofactor=1.0,
        storage_format="parquet",  # Fast format
        storage_compression=False,  # Skip compression for speed
        enable_coabundance=True,
        enable_multiscale=True,
        quality_gates_enabled=False
    )
    
    # Comprehensive configuration for regression tests
    REGRESSION_TEST = TestConfigProfile(
        name="regression_test",
        description="Full-featured config for regression testing",
        scales_um=[5.0, 10.0, 20.0, 40.0],
        clustering_method="leiden",
        clustering_resolution=1.0,
        use_slic=True,
        slic_compactness=10.0,
        normalization_method="arcsinh",
        normalization_cofactor=1.0,
        storage_format="hdf5",
        storage_compression=True,
        enable_coabundance=True,
        enable_multiscale=True,
        quality_gates_enabled=True
    )
    
    # Security testing configuration
    SECURITY_TEST = TestConfigProfile(
        name="security_test",
        description="Safe config for security testing",
        scales_um=[20.0],
        clustering_method="leiden",
        clustering_resolution=1.0,
        use_slic=False,
        slic_compactness=10.0,
        normalization_method="arcsinh",
        normalization_cofactor=1.0,
        storage_format="json",  # Safe format, no pickle
        storage_compression=False,
        enable_coabundance=True,
        enable_multiscale=False,
        quality_gates_enabled=False
    )


def create_config_namespace(profile: TestConfigProfile, 
                          output_dir: Optional[str] = None) -> SimpleNamespace:
    """Create a configuration SimpleNamespace from a profile."""
    config = SimpleNamespace(
        # Multiscale analysis
        multiscale=SimpleNamespace(
            scales_um=profile.scales_um.copy(),
            enable_scale_analysis=profile.enable_multiscale
        ),
        
        # SLIC segmentation
        slic=SimpleNamespace(
            use_slic=profile.use_slic,
            compactness=profile.slic_compactness,
            sigma=2.0,  # Fixed value
            resolution_um=2.0
        ),
        
        # Clustering
        clustering=SimpleNamespace(
            method=profile.clustering_method,
            resolution=profile.clustering_resolution,
            resolution_range=[0.5, 2.0] if profile.clustering_method == "leiden" else None,
            n_resolutions=5
        ),
        
        # Data storage  
        storage=SimpleNamespace(
            format=profile.storage_format,
            compression=profile.storage_compression
        ),
        
        # Normalization
        normalization=SimpleNamespace(
            method=profile.normalization_method,
            cofactor=profile.normalization_cofactor
        ),
        
        # Analysis features
        coabundance=SimpleNamespace(
            enable=profile.enable_coabundance,
            neighborhood_size=10
        ),
        
        # Quality gates
        quality=SimpleNamespace(
            enable_gates=profile.quality_gates_enabled,
            min_pixels=100,
            min_proteins=3,
            min_protein_completeness=0.8
        ),
        
        # Output configuration
        output=SimpleNamespace(
            results_dir=output_dir or "/tmp/test_results",
            save_intermediate=True,
            create_figures=False  # Skip in tests
        ),
        
        # Compatibility method
        to_dict=lambda: convert_namespace_to_dict(config)
    )
    
    return config


def convert_namespace_to_dict(namespace_obj: SimpleNamespace) -> Dict[str, Any]:
    """Convert SimpleNamespace to dictionary for compatibility."""
    result = {}
    
    for key, value in namespace_obj.__dict__.items():
        if key == 'to_dict':  # Skip the method itself
            continue
        elif isinstance(value, SimpleNamespace):
            result[key] = convert_namespace_to_dict(value)
        else:
            result[key] = value
    
    return result


# Pytest fixtures
@pytest.fixture
def unit_test_config(tmp_path):
    """Fast configuration for unit tests."""
    return create_config_namespace(ConfigProfiles.UNIT_TEST, str(tmp_path))


@pytest.fixture  
def integration_test_config(tmp_path):
    """Standard configuration for integration tests."""
    return create_config_namespace(ConfigProfiles.INTEGRATION_TEST, str(tmp_path))


@pytest.fixture
def performance_test_config(tmp_path):
    """Optimized configuration for performance tests."""
    return create_config_namespace(ConfigProfiles.PERFORMANCE_TEST, str(tmp_path))


@pytest.fixture
def regression_test_config(tmp_path):
    """Comprehensive configuration for regression tests."""
    return create_config_namespace(ConfigProfiles.REGRESSION_TEST, str(tmp_path))


@pytest.fixture
def security_test_config(tmp_path):
    """Safe configuration for security tests."""
    return create_config_namespace(ConfigProfiles.SECURITY_TEST, str(tmp_path))


@pytest.fixture
def mock_config(tmp_path):
    """Default test configuration - alias for unit_test_config."""
    return create_config_namespace(ConfigProfiles.UNIT_TEST, str(tmp_path))


@pytest.fixture
def custom_test_config():
    """Factory for creating custom test configurations."""
    def _create_config(profile_name: str = "unit_test", 
                      modifications: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
        # Get base profile
        profile_map = {
            "unit_test": ConfigProfiles.UNIT_TEST,
            "integration_test": ConfigProfiles.INTEGRATION_TEST,
            "performance_test": ConfigProfiles.PERFORMANCE_TEST,
            "regression_test": ConfigProfiles.REGRESSION_TEST,
            "security_test": ConfigProfiles.SECURITY_TEST
        }
        
        base_profile = profile_map.get(profile_name, ConfigProfiles.UNIT_TEST)
        
        # Apply modifications if provided
        if modifications:
            # Create modified profile
            profile_dict = asdict(base_profile)
            profile_dict.update(modifications)
            modified_profile = TestConfigProfile(**profile_dict)
            return create_config_namespace(modified_profile)
        
        return create_config_namespace(base_profile)
    
    return _create_config


# Configuration validation utilities
def validate_config_structure(config: SimpleNamespace) -> Dict[str, bool]:
    """Validate that configuration has expected structure."""
    checks = {
        'has_multiscale': hasattr(config, 'multiscale'),
        'has_slic': hasattr(config, 'slic'),
        'has_clustering': hasattr(config, 'clustering'),
        'has_storage': hasattr(config, 'storage'),
        'has_normalization': hasattr(config, 'normalization'),
        'has_to_dict': hasattr(config, 'to_dict') and callable(config.to_dict),
    }
    
    # Check nested attributes
    if hasattr(config, 'multiscale'):
        checks['multiscale_has_scales'] = hasattr(config.multiscale, 'scales_um')
        
    if hasattr(config, 'clustering'):
        checks['clustering_has_method'] = hasattr(config.clustering, 'method')
        
    if hasattr(config, 'storage'):
        checks['storage_has_format'] = hasattr(config.storage, 'format')
    
    return checks


def create_minimal_config() -> SimpleNamespace:
    """Create minimal configuration for basic testing."""
    return SimpleNamespace(
        multiscale=SimpleNamespace(
            scales_um=[20.0],
            enable_scale_analysis=False
        ),
        clustering=SimpleNamespace(
            method="leiden",
            resolution=1.0
        ),
        to_dict=lambda: {"multiscale": {"scales_um": [20.0]}, "clustering": {"method": "leiden"}}
    )


def create_config_with_quality_gates(enable_gates: bool = True) -> SimpleNamespace:
    """Create configuration with quality gates enabled/disabled."""
    profile = ConfigProfiles.INTEGRATION_TEST
    profile.quality_gates_enabled = enable_gates
    return create_config_namespace(profile)


def create_config_json_file(profile: TestConfigProfile, filepath: Path) -> Path:
    """Create a JSON configuration file from a profile."""
    config_data = {
        "analysis": {
            "scales_um": profile.scales_um,
            "clustering": {
                "method": profile.clustering_method,
                "resolution": profile.clustering_resolution
            },
            "coabundance": {
                "enable": profile.enable_coabundance
            }
        },
        "slic": {
            "use_slic": profile.use_slic,
            "compactness": profile.slic_compactness
        },
        "storage": {
            "format": profile.storage_format,
            "compression": profile.storage_compression
        },
        "normalization": {
            "method": profile.normalization_method,
            "cofactor": profile.normalization_cofactor
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    return filepath


# Error configuration for testing error handling
@pytest.fixture
def invalid_configs():
    """Create invalid configurations for error testing."""
    return {
        'missing_scales': SimpleNamespace(
            multiscale=SimpleNamespace(),  # Missing scales_um
            clustering=SimpleNamespace(method="leiden")
        ),
        
        'invalid_clustering_method': SimpleNamespace(
            multiscale=SimpleNamespace(scales_um=[20.0]),
            clustering=SimpleNamespace(method="invalid_method")
        ),
        
        'negative_scales': SimpleNamespace(
            multiscale=SimpleNamespace(scales_um=[-10.0, 0, 20.0]),
            clustering=SimpleNamespace(method="leiden")
        ),
        
        'empty_scales': SimpleNamespace(
            multiscale=SimpleNamespace(scales_um=[]),
            clustering=SimpleNamespace(method="leiden")
        ),
        
        'missing_storage_format': SimpleNamespace(
            multiscale=SimpleNamespace(scales_um=[20.0]),
            storage=SimpleNamespace(compression=True)  # Missing format
        )
    }


if __name__ == "__main__":
    # Example usage and validation
    print("Testing configuration fixtures...")
    
    # Test all profiles
    profiles = [
        ConfigProfiles.UNIT_TEST,
        ConfigProfiles.INTEGRATION_TEST, 
        ConfigProfiles.PERFORMANCE_TEST,
        ConfigProfiles.REGRESSION_TEST,
        ConfigProfiles.SECURITY_TEST
    ]
    
    for profile in profiles:
        config = create_config_namespace(profile)
        checks = validate_config_structure(config)
        
        failed_checks = [check for check, passed in checks.items() if not passed]
        if failed_checks:
            print(f"❌ {profile.name}: Failed checks: {failed_checks}")
        else:
            print(f"✅ {profile.name}: All validation checks passed")
        
        # Test to_dict conversion
        try:
            config_dict = config.to_dict()
            print(f"   to_dict() conversion successful - {len(config_dict)} keys")
        except Exception as e:
            print(f"   to_dict() conversion failed: {e}")
    
    print("Configuration fixture testing complete!")