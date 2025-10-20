#!/usr/bin/env python3
"""
Simple tests for parameter profiles functionality.

Verifies that parameter profiles work correctly and integrate with Config.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from analysis.parameter_profiles import (
    get_tissue_profile,
    convert_um_to_pixels,
    adapt_slic_params_for_resolution,
    estimate_data_characteristics,
    adapt_qc_thresholds,
    apply_profile_to_config,
    create_adaptive_config,
    get_available_profiles
)


def test_basic_profiles():
    """Test basic profile functionality."""
    print("Testing basic profiles...")
    
    # Test profile retrieval
    kidney_profile = get_tissue_profile('kidney')
    assert 'scales_um' in kidney_profile
    assert 'slic_params' in kidney_profile
    assert isinstance(kidney_profile['scales_um'], list)
    
    # Test default fallback
    unknown_profile = get_tissue_profile('unknown_tissue')
    default_profile = get_tissue_profile('default')
    assert unknown_profile == default_profile
    
    # Test available profiles
    profiles = get_available_profiles()
    assert 'kidney' in profiles
    assert 'brain' in profiles
    assert 'tumor' in profiles
    assert 'default' in profiles
    
    print("✓ Basic profile tests passed")


def test_resolution_conversion():
    """Test resolution conversion functions."""
    print("Testing resolution conversion...")
    
    # Test micrometer to pixel conversion
    assert convert_um_to_pixels(10.0, 1.0) == 10
    assert convert_um_to_pixels(5.0, 0.5) == 10
    assert convert_um_to_pixels(3.0, 2.0) == 2  # Rounded up from 1.5
    assert convert_um_to_pixels(0.5, 1.0) == 1  # Minimum 1 pixel
    
    # Test SLIC parameter adaptation
    base_params = {'compactness': 10.0, 'sigma': 1.5}
    
    # High resolution (sub-cellular)
    high_res_params = adapt_slic_params_for_resolution(base_params, 0.3)
    assert high_res_params['sigma'] < base_params['sigma']
    
    # Low resolution
    low_res_params = adapt_slic_params_for_resolution(base_params, 3.0)
    assert low_res_params['sigma'] > base_params['sigma']
    
    print("✓ Resolution conversion tests passed")


def test_data_characteristics():
    """Test data characteristic estimation."""
    print("Testing data characteristics...")
    
    # Test with mock data
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # 4 points in 1x1 area
    dna = np.array([1.0, 2.0, 1.5, 2.5])
    
    chars = estimate_data_characteristics(coords, dna)
    assert 'density' in chars
    assert 'signal_quality' in chars  
    assert 'sparsity' in chars
    assert chars['density'] == 4.0  # 4 points per μm²
    assert chars['sparsity'] == 0.0  # No zeros
    
    # Test with sparse data
    sparse_dna = np.array([0.0, 0.0, 1.0, 0.0])
    sparse_chars = estimate_data_characteristics(coords, sparse_dna)
    assert sparse_chars['sparsity'] == 0.75  # 3 zeros out of 4
    
    print("✓ Data characteristics tests passed")


def test_qc_adaptation():
    """Test QC threshold adaptation."""
    print("Testing QC adaptation...")
    
    base_thresholds = {
        'min_dna_signal': 1.0,
        'tissue_threshold': 0.1,
        'min_tissue_coverage_percent': 10
    }
    
    # Test adaptation for sparse data
    sparse_characteristics = {'sparsity': 0.8, 'signal_quality': 0.5}
    adapted = adapt_qc_thresholds(base_thresholds, sparse_characteristics)
    
    # Thresholds should be lowered for sparse data
    assert adapted['min_dna_signal'] < base_thresholds['min_dna_signal']
    assert adapted['tissue_threshold'] < base_thresholds['tissue_threshold']
    
    print("✓ QC adaptation tests passed")


def test_config_integration():
    """Test integration with Config (using mock config)."""
    print("Testing config integration...")
    
    # Create mock config object
    class MockConfig:
        def __init__(self):
            self.segmentation = {'scales_um': [10, 20, 40]}
            self.analysis = {'clustering': {}}
            
        def to_dict(self):
            return {'mock': True}
    
    mock_config = MockConfig()
    
    # Test profile application
    overrides = apply_profile_to_config(mock_config, 'kidney', 1.0)
    assert 'scales_um' in overrides
    assert 'slic_params' in overrides
    assert 'clustering' in overrides
    
    # Test adaptive config creation
    coords = np.random.rand(100, 2) * 50
    dna = np.random.exponential(2.0, 100)
    
    adaptive = create_adaptive_config(
        config=mock_config,
        coords=coords,
        dna_intensities=dna,
        tissue_type='brain'
    )
    
    assert '_data_characteristics' in adaptive
    assert '_profile_used' in adaptive
    assert adaptive['_profile_used'] == 'brain'
    
    print("✓ Config integration tests passed")


def run_all_tests():
    """Run all tests."""
    print("Running parameter profiles tests...\n")
    
    try:
        test_basic_profiles()
        test_resolution_conversion()
        test_data_characteristics()
        test_qc_adaptation()
        test_config_integration()
        
        print("\n✓ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)