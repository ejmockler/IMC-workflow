#!/usr/bin/env python3
"""
Example: Using Parameter Profiles with IMC Pipeline

Shows how to integrate tissue-specific parameter profiles with the existing 
Config system for adaptive IMC analysis.
"""

import numpy as np
from pathlib import Path
from src.config import Config
from src.analysis.parameter_profiles import (
    create_adaptive_config, 
    get_available_profiles,
    describe_profile,
    apply_profile_to_config
)
from src.analysis.main_pipeline import IMCAnalysisPipeline


def example_basic_usage():
    """Basic example: Using predefined tissue profiles."""
    print("=== Basic Parameter Profiles Usage ===")
    
    # Load standard config
    config = Config('config.json')
    
    # See available profiles
    profiles = get_available_profiles()
    print(f"Available profiles: {profiles}")
    
    for profile in profiles:
        print(f"- {profile}: {describe_profile(profile)}")
    
    # Get kidney-specific parameters
    kidney_params = apply_profile_to_config(
        config=config,
        tissue_type='kidney',
        resolution_um=1.0
    )
    
    print(f"\nKidney profile scales: {kidney_params['scales_um']}")
    print(f"Kidney SLIC params: {kidney_params['slic_params']}")
    print(f"Kidney QC thresholds: {kidney_params['qc_thresholds']}")


def example_adaptive_usage():
    """Advanced example: Adaptive parameters based on actual data."""
    print("\n=== Adaptive Parameter Selection ===")
    
    config = Config('config.json')
    
    # Simulate different data characteristics
    test_cases = [
        {
            'name': 'High-density kidney',
            'coords': np.random.rand(5000, 2) * 200,  # Dense data
            'dna': np.random.exponential(5.0, 5000),   # Good signal
            'tissue': 'kidney'
        },
        {
            'name': 'Sparse brain tissue', 
            'coords': np.random.rand(800, 2) * 500,    # Sparse data
            'dna': np.random.exponential(1.0, 800),    # Weak signal
            'tissue': 'brain'
        },
        {
            'name': 'Heterogeneous tumor',
            'coords': np.random.rand(3000, 2) * 300,  # Medium density
            'dna': np.random.exponential(3.0, 3000),   # Variable signal
            'tissue': 'tumor'
        }
    ]
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        
        adaptive_params = create_adaptive_config(
            config=config,
            coords=case['coords'],
            dna_intensities=case['dna'], 
            tissue_type=case['tissue'],
            resolution_um=1.0
        )
        
        data_chars = adaptive_params['_data_characteristics']
        print(f"Data density: {data_chars['density']:.1f} pixels/μm²")
        print(f"Signal quality: {data_chars['signal_quality']:.2f}")
        print(f"Sparsity: {data_chars['sparsity']:.2f}")
        print(f"Adapted scales: {adaptive_params['scales_um']}")
        print(f"Clustering range: {adaptive_params['clustering']['resolution_range']}")


def example_pipeline_integration():
    """Example: Using profiles with the actual analysis pipeline."""
    print("\n=== Pipeline Integration Example ===")
    
    # This would be your real analysis workflow
    config = Config('config.json')
    
    # Initialize pipeline
    pipeline = IMCAnalysisPipeline(config)
    
    # Mock data (in real use, this comes from load_roi_data)
    mock_roi_data = {
        'coords': np.random.rand(2000, 2) * 400,
        'ion_counts': {
            'CD45': np.random.exponential(2.0, 2000),
            'CD31': np.random.exponential(1.5, 2000),
            'CD11b': np.random.exponential(1.8, 2000)
        },
        'dna1_intensities': np.random.exponential(3.0, 2000),
        'dna2_intensities': np.random.exponential(2.8, 2000),
        'protein_names': ['CD45', 'CD31', 'CD11b'],
        'n_measurements': 2000
    }
    
    # Get adaptive parameters for kidney tissue
    adaptive_params = create_adaptive_config(
        config=config,
        coords=mock_roi_data['coords'],
        dna_intensities=mock_roi_data['dna1_intensities'],
        tissue_type='kidney',
        resolution_um=1.0
    )
    
    print(f"Using {adaptive_params['_profile_used']} profile")
    print(f"Scales: {adaptive_params['scales_um']}")
    
    # Run analysis with profile-based parameters
    # Note: This would pass adaptive_params as override_config
    print("Running analysis with adaptive parameters...")
    
    # In real usage:
    # result = pipeline.analyze_single_roi(
    #     roi_data=mock_roi_data,
    #     override_config=adaptive_params
    # )
    
    print("Analysis would be run with tissue-specific, data-adaptive parameters")


def example_config_creation():
    """Example: Creating a config file with profile-based defaults."""
    print("\n=== Profile-Based Config Creation ===")
    
    # Load base config
    base_config = Config('config.json')
    
    # Get brain profile parameters
    brain_params = apply_profile_to_config(
        config=base_config,
        tissue_type='brain',
        resolution_um=0.8  # High-resolution brain data
    )
    
    # Show how you could update config sections
    print("Brain profile would set:")
    print(f"  segmentation.scales_um = {brain_params['scales_um']}")
    print(f"  segmentation.slic_params = {brain_params['slic_params']}")
    print(f"  analysis.clustering.resolution_range = {brain_params['clustering']['resolution_range']}")
    print(f"  quality_control.thresholds.min_dna_signal = {brain_params['qc_thresholds']['min_dna_signal']}")
    
    # In practice, you could:
    # 1. Update your config.json with these values
    # 2. Use override_config in pipeline calls
    # 3. Create tissue-specific config files


if __name__ == "__main__":
    # Run examples if this file is executed directly
    try:
        example_basic_usage()
        example_adaptive_usage() 
        example_pipeline_integration()
        example_config_creation()
        print("\n✓ All examples completed successfully!")
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("Make sure config.json exists and src/ is in your Python path")