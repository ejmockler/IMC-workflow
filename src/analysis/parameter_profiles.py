"""
Parameter Profiles for IMC Analysis

Simple, pragmatic parameter sets for different tissues and data characteristics.
Integrates with existing Config system without replacement.

Key features:
- Tissue-specific parameter sets (kidney, brain, tumor)
- Data-driven SLIC scaling (μm → pixels based on resolution)
- QC threshold adaptation based on data characteristics
- Direct integration with existing Config
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np


# Tissue-specific parameter profiles
TISSUE_PROFILES = {
    'kidney': {
        'description': 'Kidney tissue with tubular and glomerular structures',
        'scales_um': [10.0, 20.0, 75.0],  # Capillary, tubular, architectural
        'slic_params': {
            'compactness': 15.0,  # Higher for elongated tubules
            'sigma': 1.2
        },
        'clustering': {
            'resolution_range': [0.3, 1.5],  # Conservative for clear structures
            'optimization_method': 'stability'
        },
        'qc_thresholds': {
            'min_dna_signal': 1.2,  # Higher for dense nuclei
            'tissue_threshold': 0.15,
            'min_tissue_coverage_percent': 15
        },
        'dna_processing': {
            'noise_floor_percentile': 5,  # Lower for high-quality kidney data
            'cofactor_multiplier': 2.5
        }
    },
    
    'brain': {
        'description': 'Brain tissue with neuronal and glial structures',
        'scales_um': [8.0, 25.0, 100.0],  # Cellular, circuit, regional
        'slic_params': {
            'compactness': 8.0,  # Lower for irregular cell shapes
            'sigma': 2.0
        },
        'clustering': {
            'resolution_range': [0.5, 2.0],  # Wider range for cellular diversity
            'optimization_method': 'stability'
        },
        'qc_thresholds': {
            'min_dna_signal': 0.8,  # Lower for sparse nuclei
            'tissue_threshold': 0.08,
            'min_tissue_coverage_percent': 8
        },
        'dna_processing': {
            'noise_floor_percentile': 15,  # Higher for sparse tissue
            'cofactor_multiplier': 4.0
        }
    },
    
    'tumor': {
        'description': 'Tumor tissue with mixed cellular environments',
        'scales_um': [12.0, 30.0, 80.0],  # Cellular, microenvironment, architecture
        'slic_params': {
            'compactness': 12.0,  # Balanced for heterogeneous structures
            'sigma': 1.8
        },
        'clustering': {
            'resolution_range': [0.4, 2.5],  # Wide range for heterogeneity
            'optimization_method': 'modularity'  # Better for mixed populations
        },
        'qc_thresholds': {
            'min_dna_signal': 1.0,
            'tissue_threshold': 0.12,
            'min_tissue_coverage_percent': 12
        },
        'dna_processing': {
            'noise_floor_percentile': 10,
            'cofactor_multiplier': 3.0
        }
    },
    
    'default': {
        'description': 'General-purpose parameters for unknown tissue types',
        'scales_um': [10.0, 20.0, 40.0],
        'slic_params': {
            'compactness': 10.0,
            'sigma': 1.5
        },
        'clustering': {
            'resolution_range': [0.5, 2.0],
            'optimization_method': 'stability'
        },
        'qc_thresholds': {
            'min_dna_signal': 1.0,
            'tissue_threshold': 0.1,
            'min_tissue_coverage_percent': 10
        },
        'dna_processing': {
            'noise_floor_percentile': 10,
            'cofactor_multiplier': 3.0
        }
    }
}


def get_tissue_profile(tissue_type: str) -> Dict[str, Any]:
    """
    Get parameter profile for specific tissue type.
    
    Args:
        tissue_type: Tissue type ('kidney', 'brain', 'tumor', or 'default')
        
    Returns:
        Dictionary with tissue-specific parameters
    """
    if tissue_type.lower() in TISSUE_PROFILES:
        return TISSUE_PROFILES[tissue_type.lower()].copy()
    else:
        return TISSUE_PROFILES['default'].copy()


def convert_um_to_pixels(scale_um: float, resolution_um: float = 1.0) -> int:
    """
    Convert micrometer scale to pixel scale based on resolution.
    
    Args:
        scale_um: Target scale in micrometers
        resolution_um: Image resolution in micrometers per pixel
        
    Returns:
        Scale in pixels
    """
    return max(1, int(round(scale_um / resolution_um)))


def adapt_slic_params_for_resolution(
    base_params: Dict[str, float], 
    resolution_um: float
) -> Dict[str, Any]:
    """
    Adapt SLIC parameters based on actual image resolution.
    
    Args:
        base_params: Base SLIC parameters (compactness, sigma)
        resolution_um: Image resolution in micrometers per pixel
        
    Returns:
        Resolution-adapted SLIC parameters
    """
    adapted = base_params.copy()
    
    # Adjust sigma for resolution (higher resolution needs less smoothing)
    if resolution_um < 0.5:  # Sub-cellular resolution
        adapted['sigma'] *= 0.7
    elif resolution_um > 2.0:  # Low resolution
        adapted['sigma'] *= 1.5
    
    # Compactness usually doesn't need adjustment for resolution
    return adapted


def estimate_data_characteristics(coords: np.ndarray, dna_intensities: np.ndarray) -> Dict[str, float]:
    """
    Estimate data characteristics for adaptive parameter selection.
    
    Args:
        coords: Coordinate array (N x 2)
        dna_intensities: DNA signal intensities
        
    Returns:
        Dictionary with data characteristics
    """
    if len(coords) == 0 or len(dna_intensities) == 0:
        return {'density': 0.0, 'signal_quality': 0.0, 'sparsity': 1.0}
    
    # Estimate spatial density
    x_range = coords[:, 0].max() - coords[:, 0].min() if coords.shape[0] > 1 else 1.0
    y_range = coords[:, 1].max() - coords[:, 1].min() if coords.shape[0] > 1 else 1.0
    area = x_range * y_range
    density = len(coords) / max(area, 1.0)  # pixels per μm²
    
    # Estimate signal quality
    non_zero_dna = dna_intensities[dna_intensities > 0]
    if len(non_zero_dna) > 0:
        signal_quality = np.median(non_zero_dna) / (np.std(non_zero_dna) + 1e-6)
    else:
        signal_quality = 0.0
    
    # Estimate sparsity
    sparsity = 1.0 - (len(non_zero_dna) / len(dna_intensities))
    
    return {
        'density': float(density),
        'signal_quality': float(signal_quality), 
        'sparsity': float(sparsity)
    }


def adapt_qc_thresholds(
    base_thresholds: Dict[str, float], 
    data_characteristics: Dict[str, float]
) -> Dict[str, float]:
    """
    Adapt QC thresholds based on data characteristics.
    
    Args:
        base_thresholds: Base QC threshold values
        data_characteristics: Data characteristics from estimate_data_characteristics
        
    Returns:
        Adapted QC thresholds
    """
    adapted = base_thresholds.copy()
    
    # Lower thresholds for sparse data
    if data_characteristics['sparsity'] > 0.7:
        adapted['min_dna_signal'] *= 0.7
        adapted['tissue_threshold'] *= 0.8
        adapted['min_tissue_coverage_percent'] *= 0.6
    
    # Adjust for signal quality
    if data_characteristics['signal_quality'] < 1.0:
        adapted['min_dna_signal'] *= 0.8
    elif data_characteristics['signal_quality'] > 3.0:
        adapted['min_dna_signal'] *= 1.2
    
    return adapted


def apply_profile_to_config(config, tissue_type: str, resolution_um: float = 1.0) -> Dict[str, Any]:
    """
    Apply parameter profile to existing config without modifying it.
    
    Args:
        config: Existing Config object
        tissue_type: Tissue type for profile selection
        resolution_um: Image resolution in micrometers per pixel
        
    Returns:
        Dictionary with profile-based parameter overrides
    """
    profile = get_tissue_profile(tissue_type)
    
    # Create parameter overrides (don't modify original config)
    overrides = {}
    
    # Segmentation parameters
    overrides['scales_um'] = profile['scales_um'].copy()
    overrides['slic_params'] = adapt_slic_params_for_resolution(
        profile['slic_params'], resolution_um
    )
    
    # Clustering parameters
    overrides['clustering'] = profile['clustering'].copy()
    
    # QC thresholds 
    overrides['qc_thresholds'] = profile['qc_thresholds'].copy()
    
    # DNA processing
    overrides['dna_processing'] = profile['dna_processing'].copy()
    
    return overrides


def create_adaptive_config(
    config, 
    coords: np.ndarray, 
    dna_intensities: np.ndarray,
    tissue_type: str = 'default',
    resolution_um: float = 1.0
) -> Dict[str, Any]:
    """
    Create adaptive configuration based on data characteristics.
    
    Args:
        config: Base Config object
        coords: Coordinate array for data analysis
        dna_intensities: DNA intensities for quality estimation
        tissue_type: Tissue type for profile selection
        resolution_um: Image resolution in micrometers per pixel
        
    Returns:
        Dictionary with adaptive parameter overrides
    """
    # Get base profile
    base_overrides = apply_profile_to_config(config, tissue_type, resolution_um)
    
    # Estimate data characteristics
    data_chars = estimate_data_characteristics(coords, dna_intensities)
    
    # Adapt QC thresholds based on data
    base_overrides['qc_thresholds'] = adapt_qc_thresholds(
        base_overrides['qc_thresholds'], data_chars
    )
    
    # Adapt clustering resolution based on density
    if data_chars['density'] > 100:  # High density
        base_overrides['clustering']['resolution_range'] = [0.3, 1.2]
    elif data_chars['density'] < 10:  # Low density  
        base_overrides['clustering']['resolution_range'] = [0.8, 3.0]
    
    # Store data characteristics for reference
    base_overrides['_data_characteristics'] = data_chars
    base_overrides['_profile_used'] = tissue_type
    
    return base_overrides


def get_available_profiles() -> List[str]:
    """Get list of available tissue profiles."""
    return list(TISSUE_PROFILES.keys())


def describe_profile(tissue_type: str) -> str:
    """Get description of tissue profile."""
    profile = get_tissue_profile(tissue_type)
    return profile.get('description', 'No description available')


# Simple usage example
def example_usage():
    """Example of how to use parameter profiles."""
    from ..config import Config
    
    # Load your normal config
    config = Config('config.json')
    
    # Example data (would come from your actual data)
    coords = np.random.rand(1000, 2) * 500  # Mock coordinates
    dna = np.random.exponential(2.0, 1000)  # Mock DNA intensities
    
    # Get kidney-specific parameters adapted to your data
    kidney_params = create_adaptive_config(
        config=config,
        coords=coords, 
        dna_intensities=dna,
        tissue_type='kidney',
        resolution_um=1.0
    )
    
    print(f"Using kidney profile with scales: {kidney_params['scales_um']}")
    print(f"Adapted QC thresholds: {kidney_params['qc_thresholds']}")
    
    return kidney_params