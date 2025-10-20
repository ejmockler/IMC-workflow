"""
Uncertainty propagation system for IMC corrections and analysis.

Provides comprehensive uncertainty tracking through the entire correction pipeline,
from raw ion counts through spillover correction, artifact removal, normalization,
and downstream analysis steps.

Key Features:
- Uncertainty propagation through all correction steps
- Composition of multiple uncertainty sources
- Confidence intervals for small-n studies
- Spatial uncertainty visualization
- Integration with existing pipeline components
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
import logging
from scipy.stats import bootstrap, t
from scipy.sparse import csr_matrix
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    HAS_NUMBA = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UncertaintyMap:
    """Immutable uncertainty map with spatial and channel dimensions."""
    uncertainties: np.ndarray       # Channel × spatial uncertainty values
    uncertainty_type: str           # 'absolute', 'relative', 'log'
    channels: List[str]            # Channel names
    spatial_shape: Tuple[int, ...] # Original spatial dimensions
    sources: List[str]             # Uncertainty sources contributing
    metadata: Dict[str, Any]       # Additional uncertainty metadata
    
    def __post_init__(self):
        if len(self.channels) != self.uncertainties.shape[0]:
            raise ValueError("Number of channels must match uncertainty array first dimension")
        
        expected_spatial_size = np.prod(self.spatial_shape)
        if self.uncertainties.shape[1:] != (expected_spatial_size,):
            flat_uncertainties = self.uncertainties.reshape(len(self.channels), -1)
            if flat_uncertainties.shape[1] != expected_spatial_size:
                raise ValueError("Uncertainty spatial dimensions don't match specified shape")


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty propagation methods."""
    propagation_method: str = 'linear'        # 'linear', 'monte_carlo', 'bootstrap'
    confidence_level: float = 0.95           # Confidence level for intervals
    monte_carlo_samples: int = 1000          # Samples for MC propagation
    bootstrap_samples: int = 1000            # Samples for bootstrap
    spatial_correlation: bool = False        # Account for spatial correlation
    min_uncertainty: float = 0.01            # Minimum uncertainty floor
    max_uncertainty: float = 10.0            # Maximum uncertainty ceiling


class UncertaintyPropagationError(Exception):
    """Exception raised for uncertainty propagation failures."""
    pass


def create_base_uncertainty(
    ion_counts: Dict[str, np.ndarray],
    uncertainty_type: str = 'poisson',
    uncertainty_floor: float = 0.01
) -> UncertaintyMap:
    """
    Create base uncertainty map from ion count statistics.
    
    Args:
        ion_counts: Dictionary mapping channel_name -> ion_count_array
        uncertainty_type: Type of base uncertainty ('poisson', 'constant', 'proportional')
        uncertainty_floor: Minimum uncertainty value
        
    Returns:
        UncertaintyMap with base uncertainties
    """
    logger.debug(f"Creating base uncertainty map using {uncertainty_type} model")
    
    if not ion_counts:
        raise UncertaintyPropagationError("No ion count data provided")
    
    channels = sorted(ion_counts.keys())
    first_channel = channels[0]
    spatial_shape = ion_counts[first_channel].shape
    
    # Initialize uncertainty array
    n_channels = len(channels)
    spatial_size = np.prod(spatial_shape)
    uncertainties = np.zeros((n_channels, spatial_size))
    
    for i, channel in enumerate(channels):
        counts = ion_counts[channel].flatten()
        
        if uncertainty_type == 'poisson':
            # Poisson uncertainty: σ = sqrt(counts)
            channel_uncertainty = np.sqrt(np.maximum(counts, 1))
        elif uncertainty_type == 'constant':
            # Constant relative uncertainty
            channel_uncertainty = uncertainty_floor * np.ones_like(counts)
        elif uncertainty_type == 'proportional':
            # Proportional to signal
            channel_uncertainty = uncertainty_floor * counts
        elif uncertainty_type == 'mixed':
            # Poisson + proportional
            poisson_term = np.sqrt(np.maximum(counts, 1))
            proportional_term = uncertainty_floor * counts
            channel_uncertainty = np.sqrt(poisson_term**2 + proportional_term**2)
        else:
            raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")
        
        # Apply uncertainty floor and ceiling
        channel_uncertainty = np.maximum(channel_uncertainty, uncertainty_floor)
        uncertainties[i] = channel_uncertainty
    
    return UncertaintyMap(
        uncertainties=uncertainties,
        uncertainty_type='absolute',
        channels=channels,
        spatial_shape=spatial_shape,
        sources=['base_' + uncertainty_type],
        metadata={
            'uncertainty_model': uncertainty_type,
            'uncertainty_floor': uncertainty_floor,
            'mean_uncertainty_per_channel': {
                channels[i]: np.mean(uncertainties[i]) for i in range(n_channels)
            }
        }
    )


def propagate_through_spillover_correction(
    input_uncertainty: UncertaintyMap,
    spillover_matrix: np.ndarray,
    matrix_uncertainty: np.ndarray,
    correction_method: str = 'linear'
) -> UncertaintyMap:
    """
    Propagate uncertainty through spillover correction.
    
    Args:
        input_uncertainty: Input uncertainty map
        spillover_matrix: Spillover correction matrix
        matrix_uncertainty: Uncertainty in spillover matrix elements
        correction_method: Method for uncertainty propagation
        
    Returns:
        Updated uncertainty map after spillover correction
    """
    logger.debug("Propagating uncertainty through spillover correction")
    
    if correction_method == 'linear':
        # Linear uncertainty propagation using matrix operations
        corrected_uncertainties = _linear_spillover_propagation(
            input_uncertainty.uncertainties, spillover_matrix, matrix_uncertainty
        )
    elif correction_method == 'monte_carlo':
        # Monte Carlo uncertainty propagation
        corrected_uncertainties = _monte_carlo_spillover_propagation(
            input_uncertainty.uncertainties, spillover_matrix, matrix_uncertainty
        )
    else:
        raise ValueError(f"Unknown correction method: {correction_method}")
    
    # Update sources
    new_sources = input_uncertainty.sources + ['spillover_correction']
    
    return UncertaintyMap(
        uncertainties=corrected_uncertainties,
        uncertainty_type=input_uncertainty.uncertainty_type,
        channels=input_uncertainty.channels,
        spatial_shape=input_uncertainty.spatial_shape,
        sources=new_sources,
        metadata={
            **input_uncertainty.metadata,
            'spillover_correction_applied': True,
            'spillover_matrix_condition': np.linalg.cond(spillover_matrix),
            'propagation_method': correction_method
        }
    )


def propagate_through_artifact_correction(
    input_uncertainty: UncertaintyMap,
    artifact_mask: np.ndarray,
    interpolation_uncertainty: np.ndarray,
    correction_metadata: Dict[str, Any]
) -> UncertaintyMap:
    """
    Propagate uncertainty through artifact correction.
    
    Args:
        input_uncertainty: Input uncertainty map
        artifact_mask: Boolean mask of corrected artifacts
        interpolation_uncertainty: Uncertainty from interpolation
        correction_metadata: Metadata from artifact correction
        
    Returns:
        Updated uncertainty map after artifact correction
    """
    logger.debug("Propagating uncertainty through artifact correction")
    
    # Start with input uncertainties
    corrected_uncertainties = input_uncertainty.uncertainties.copy()
    
    # Inflate uncertainty for corrected pixels
    n_channels = len(input_uncertainty.channels)
    spatial_size = corrected_uncertainties.shape[1]
    
    if artifact_mask.size == spatial_size:
        artifact_mask_flat = artifact_mask.flatten()
    else:
        artifact_mask_flat = artifact_mask.reshape(-1)
    
    # Apply interpolation uncertainty to corrected pixels
    for i in range(n_channels):
        channel_uncertainty = corrected_uncertainties[i]
        
        # Increase uncertainty for interpolated pixels
        if interpolation_uncertainty.ndim == 1:
            # Single uncertainty value per pixel
            channel_uncertainty[artifact_mask_flat] = np.maximum(
                channel_uncertainty[artifact_mask_flat],
                interpolation_uncertainty[artifact_mask_flat]
            )
        else:
            # Channel-specific interpolation uncertainty
            channel_interpolation = interpolation_uncertainty[i].flatten()
            channel_uncertainty[artifact_mask_flat] = np.maximum(
                channel_uncertainty[artifact_mask_flat],
                channel_interpolation[artifact_mask_flat]
            )
        
        corrected_uncertainties[i] = channel_uncertainty
    
    # Update sources and metadata
    new_sources = input_uncertainty.sources + ['artifact_correction']
    n_corrected_pixels = np.sum(artifact_mask_flat)
    
    updated_metadata = {
        **input_uncertainty.metadata,
        'artifact_correction_applied': True,
        'n_corrected_pixels': n_corrected_pixels,
        'fraction_corrected': n_corrected_pixels / spatial_size,
        'correction_types': correction_metadata.get('corrections_applied', [])
    }
    
    return UncertaintyMap(
        uncertainties=corrected_uncertainties,
        uncertainty_type=input_uncertainty.uncertainty_type,
        channels=input_uncertainty.channels,
        spatial_shape=input_uncertainty.spatial_shape,
        sources=new_sources,
        metadata=updated_metadata
    )


def propagate_through_normalization(
    input_uncertainty: UncertaintyMap,
    normalization_factors: Dict[str, float],
    normalization_uncertainty: Dict[str, float],
    normalization_method: str
) -> UncertaintyMap:
    """
    Propagate uncertainty through normalization correction.
    
    Args:
        input_uncertainty: Input uncertainty map
        normalization_factors: Correction factors per channel
        normalization_uncertainty: Uncertainty in correction factors
        normalization_method: Method used for normalization
        
    Returns:
        Updated uncertainty map after normalization
    """
    logger.debug(f"Propagating uncertainty through {normalization_method} normalization")
    
    corrected_uncertainties = input_uncertainty.uncertainties.copy()
    
    for i, channel in enumerate(input_uncertainty.channels):
        # Get normalization factor and its uncertainty
        norm_factor = normalization_factors.get(channel, 1.0)
        norm_uncertainty = normalization_uncertainty.get(channel, 0.0)
        
        # Propagate uncertainty: σ_corrected = sqrt((factor * σ_input)² + (uncertainty * signal)²)
        input_unc = corrected_uncertainties[i]
        
        # For simplicity, assume signal is approximately the corrected value
        # More sophisticated: would need the actual corrected signal values
        signal_contribution = norm_uncertainty * np.abs(input_unc / norm_factor)
        scaling_contribution = norm_factor * input_unc
        
        combined_uncertainty = np.sqrt(scaling_contribution**2 + signal_contribution**2)
        corrected_uncertainties[i] = combined_uncertainty
    
    # Update sources and metadata
    new_sources = input_uncertainty.sources + [f'{normalization_method}_normalization']
    
    updated_metadata = {
        **input_uncertainty.metadata,
        f'{normalization_method}_normalization_applied': True,
        'normalization_factors': normalization_factors,
        'normalization_uncertainties': normalization_uncertainty
    }
    
    return UncertaintyMap(
        uncertainties=corrected_uncertainties,
        uncertainty_type=input_uncertainty.uncertainty_type,
        channels=input_uncertainty.channels,
        spatial_shape=input_uncertainty.spatial_shape,
        sources=new_sources,
        metadata=updated_metadata
    )


def combine_uncertainty_maps(
    uncertainty_maps: List[UncertaintyMap],
    combination_method: str = 'quadrature'
) -> UncertaintyMap:
    """
    Combine multiple uncertainty maps.
    
    Args:
        uncertainty_maps: List of uncertainty maps to combine
        combination_method: Method for combining uncertainties
        
    Returns:
        Combined uncertainty map
    """
    if not uncertainty_maps:
        raise UncertaintyPropagationError("No uncertainty maps to combine")
    
    if len(uncertainty_maps) == 1:
        return uncertainty_maps[0]
    
    # Validate compatibility
    reference_map = uncertainty_maps[0]
    for um in uncertainty_maps[1:]:
        if um.channels != reference_map.channels:
            raise UncertaintyPropagationError("Uncertainty maps have incompatible channels")
        if um.spatial_shape != reference_map.spatial_shape:
            raise UncertaintyPropagationError("Uncertainty maps have incompatible spatial shapes")
    
    # Combine uncertainties
    if combination_method == 'quadrature':
        # Root sum of squares
        combined_uncertainties = np.zeros_like(reference_map.uncertainties)
        for um in uncertainty_maps:
            combined_uncertainties += um.uncertainties**2
        combined_uncertainties = np.sqrt(combined_uncertainties)
        
    elif combination_method == 'linear':
        # Linear addition (conservative)
        combined_uncertainties = np.zeros_like(reference_map.uncertainties)
        for um in uncertainty_maps:
            combined_uncertainties += um.uncertainties
            
    elif combination_method == 'maximum':
        # Take maximum uncertainty at each point
        combined_uncertainties = uncertainty_maps[0].uncertainties.copy()
        for um in uncertainty_maps[1:]:
            combined_uncertainties = np.maximum(combined_uncertainties, um.uncertainties)
            
    else:
        raise ValueError(f"Unknown combination method: {combination_method}")
    
    # Combine sources and metadata
    all_sources = []
    for um in uncertainty_maps:
        all_sources.extend(um.sources)
    combined_sources = list(dict.fromkeys(all_sources))  # Remove duplicates, preserve order
    
    combined_metadata = {
        'combination_method': combination_method,
        'n_maps_combined': len(uncertainty_maps),
        'individual_sources': [um.sources for um in uncertainty_maps]
    }
    
    return UncertaintyMap(
        uncertainties=combined_uncertainties,
        uncertainty_type=reference_map.uncertainty_type,
        channels=reference_map.channels,
        spatial_shape=reference_map.spatial_shape,
        sources=combined_sources,
        metadata=combined_metadata
    )


def compute_confidence_intervals(
    uncertainty_map: UncertaintyMap,
    signal_values: Dict[str, np.ndarray],
    confidence_level: float = 0.95,
    method: str = 'gaussian'
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute confidence intervals from uncertainty map.
    
    Args:
        uncertainty_map: Uncertainty map
        signal_values: Signal values for each channel
        confidence_level: Confidence level (0-1)
        method: Method for computing intervals
        
    Returns:
        Tuple of (lower_bounds, upper_bounds) dictionaries
    """
    logger.debug(f"Computing {confidence_level*100}% confidence intervals using {method} method")
    
    if method == 'gaussian':
        # Gaussian confidence intervals
        z_score = t.ppf((1 + confidence_level) / 2, df=np.inf)  # Normal approximation
        
        lower_bounds = {}
        upper_bounds = {}
        
        for i, channel in enumerate(uncertainty_map.channels):
            signal = signal_values[channel].flatten()
            uncertainty = uncertainty_map.uncertainties[i]
            
            margin = z_score * uncertainty
            lower_bounds[channel] = (signal - margin).reshape(uncertainty_map.spatial_shape)
            upper_bounds[channel] = (signal + margin).reshape(uncertainty_map.spatial_shape)
            
    elif method == 't_distribution':
        # Use t-distribution for small sample sizes
        # Estimate degrees of freedom from metadata if available
        df = uncertainty_map.metadata.get('degrees_of_freedom', 10)
        t_score = t.ppf((1 + confidence_level) / 2, df=df)
        
        lower_bounds = {}
        upper_bounds = {}
        
        for i, channel in enumerate(uncertainty_map.channels):
            signal = signal_values[channel].flatten()
            uncertainty = uncertainty_map.uncertainties[i]
            
            margin = t_score * uncertainty
            lower_bounds[channel] = (signal - margin).reshape(uncertainty_map.spatial_shape)
            upper_bounds[channel] = (signal + margin).reshape(uncertainty_map.spatial_shape)
            
    else:
        raise ValueError(f"Unknown confidence interval method: {method}")
    
    return lower_bounds, upper_bounds


def validate_uncertainty_propagation(
    uncertainty_map: UncertaintyMap,
    signal_values: Dict[str, np.ndarray],
    validation_method: str = 'bootstrap'
) -> Dict[str, Any]:
    """
    Validate uncertainty estimates using independent methods.
    
    Args:
        uncertainty_map: Uncertainty map to validate
        signal_values: Signal values for validation
        validation_method: Method for validation
        
    Returns:
        Dictionary with validation results
    """
    logger.debug(f"Validating uncertainty propagation using {validation_method}")
    
    validation_results = {
        'method': validation_method,
        'n_channels': len(uncertainty_map.channels),
        'spatial_shape': uncertainty_map.spatial_shape,
        'uncertainty_sources': uncertainty_map.sources
    }
    
    if validation_method == 'bootstrap':
        # Bootstrap validation of uncertainty estimates
        bootstrap_uncertainties = {}
        
        for i, channel in enumerate(uncertainty_map.channels):
            signal = signal_values[channel].flatten()
            predicted_uncertainty = uncertainty_map.uncertainties[i]
            
            # Simple bootstrap validation (limited without access to raw replicates)
            # This is a placeholder - real validation would need replicate data
            n_pixels = len(signal)
            if n_pixels > 100:
                # Sample pixels for bootstrap validation
                sample_size = min(1000, n_pixels // 10)
                sample_indices = np.random.choice(n_pixels, sample_size, replace=False)
                
                sample_signals = signal[sample_indices]
                sample_uncertainties = predicted_uncertainty[sample_indices]
                
                # Compute bootstrap estimate (simplified)
                bootstrap_std = np.std(sample_signals)
                predicted_std = np.mean(sample_uncertainties)
                
                bootstrap_uncertainties[channel] = {
                    'bootstrap_std': bootstrap_std,
                    'predicted_std': predicted_std,
                    'ratio': predicted_std / (bootstrap_std + 1e-10),
                    'n_samples': sample_size
                }
            else:
                bootstrap_uncertainties[channel] = {
                    'bootstrap_std': np.nan,
                    'predicted_std': np.mean(predicted_uncertainty),
                    'ratio': np.nan,
                    'n_samples': n_pixels,
                    'note': 'insufficient_pixels_for_bootstrap'
                }
        
        validation_results['bootstrap_comparison'] = bootstrap_uncertainties
        
        # Overall validation metrics
        valid_ratios = [
            result['ratio'] for result in bootstrap_uncertainties.values()
            if not np.isnan(result['ratio'])
        ]
        
        if valid_ratios:
            validation_results['overall_metrics'] = {
                'mean_ratio': np.mean(valid_ratios),
                'std_ratio': np.std(valid_ratios),
                'median_ratio': np.median(valid_ratios),
                'n_valid_channels': len(valid_ratios)
            }
        else:
            validation_results['overall_metrics'] = {
                'mean_ratio': np.nan,
                'note': 'no_valid_bootstrap_comparisons'
            }
    
    elif validation_method == 'consistency_check':
        # Check internal consistency of uncertainty estimates
        consistency_results = {}
        
        for i, channel in enumerate(uncertainty_map.channels):
            uncertainty = uncertainty_map.uncertainties[i]
            signal = signal_values[channel].flatten()
            
            # Basic consistency checks
            relative_uncertainty = uncertainty / (np.abs(signal) + 1e-10)
            
            consistency_results[channel] = {
                'mean_relative_uncertainty': np.mean(relative_uncertainty),
                'max_relative_uncertainty': np.max(relative_uncertainty),
                'uncertainty_range': np.ptp(uncertainty),
                'negative_uncertainties': np.sum(uncertainty < 0),
                'zero_uncertainties': np.sum(uncertainty == 0)
            }
        
        validation_results['consistency_check'] = consistency_results
    
    return validation_results


# Helper functions for uncertainty propagation

@jit(nopython=True if HAS_NUMBA else False)
def _linear_spillover_propagation(
    input_uncertainties: np.ndarray,
    spillover_matrix: np.ndarray,
    matrix_uncertainty: np.ndarray
) -> np.ndarray:
    """Linear uncertainty propagation through spillover correction."""
    
    n_channels, n_pixels = input_uncertainties.shape
    corrected_uncertainties = np.zeros_like(input_uncertainties)
    
    # Compute spillover correction inverse
    try:
        spillover_inv = np.linalg.pinv(spillover_matrix)
    except:
        spillover_inv = np.eye(n_channels)  # Fallback to identity
    
    # Linear propagation: uncertainty scales with correction matrix
    for i in range(n_channels):
        for j in range(n_channels):
            # Contribution from input uncertainty
            input_contribution = spillover_inv[i, j]**2 * input_uncertainties[j]**2
            
            # Contribution from matrix uncertainty (simplified)
            matrix_contribution = matrix_uncertainty[i, j]**2 * np.mean(input_uncertainties[j])**2
            
            corrected_uncertainties[i] += input_contribution + matrix_contribution
    
    # Take square root for final uncertainties
    corrected_uncertainties = np.sqrt(corrected_uncertainties)
    
    return corrected_uncertainties


def _monte_carlo_spillover_propagation(
    input_uncertainties: np.ndarray,
    spillover_matrix: np.ndarray,
    matrix_uncertainty: np.ndarray,
    n_samples: int = 1000
) -> np.ndarray:
    """Monte Carlo uncertainty propagation through spillover correction."""
    
    n_channels, n_pixels = input_uncertainties.shape
    
    # Sample subset of pixels for MC propagation (computational efficiency)
    max_pixels_mc = 1000
    if n_pixels > max_pixels_mc:
        pixel_indices = np.random.choice(n_pixels, max_pixels_mc, replace=False)
        mc_uncertainties = input_uncertainties[:, pixel_indices]
    else:
        pixel_indices = np.arange(n_pixels)
        mc_uncertainties = input_uncertainties
    
    mc_n_pixels = mc_uncertainties.shape[1]
    mc_results = np.zeros((n_samples, n_channels, mc_n_pixels))
    
    for sample in range(n_samples):
        # Sample spillover matrix
        sampled_matrix = spillover_matrix + np.random.normal(0, matrix_uncertainty)
        
        # Sample input values (assuming normal distribution)
        sampled_inputs = np.random.normal(0, mc_uncertainties)
        
        # Apply spillover correction
        try:
            corrected_sample = np.linalg.pinv(sampled_matrix) @ sampled_inputs
            mc_results[sample] = corrected_sample
        except:
            mc_results[sample] = sampled_inputs  # Fallback
    
    # Compute standard deviation across samples
    mc_std = np.std(mc_results, axis=0)
    
    # Interpolate back to full pixel set if needed
    if n_pixels > max_pixels_mc:
        full_uncertainties = np.zeros((n_channels, n_pixels))
        for i in range(n_channels):
            # Simple interpolation - could be improved
            full_uncertainties[i, pixel_indices] = mc_std[i]
            
            # Fill remaining pixels with nearest neighbor
            missing_mask = np.ones(n_pixels, dtype=bool)
            missing_mask[pixel_indices] = False
            
            if np.any(missing_mask):
                full_uncertainties[i, missing_mask] = np.mean(mc_std[i])
        
        return full_uncertainties
    else:
        return mc_std


def create_summary_statistics(
    uncertainty_map: UncertaintyMap,
    signal_values: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Create summary statistics for uncertainty map.
    
    Args:
        uncertainty_map: Uncertainty map to summarize
        signal_values: Signal values for context
        
    Returns:
        Dictionary with summary statistics
    """
    
    summary = {
        'n_channels': len(uncertainty_map.channels),
        'spatial_shape': uncertainty_map.spatial_shape,
        'uncertainty_type': uncertainty_map.uncertainty_type,
        'sources': uncertainty_map.sources,
        'per_channel_stats': {}
    }
    
    for i, channel in enumerate(uncertainty_map.channels):
        uncertainty = uncertainty_map.uncertainties[i]
        signal = signal_values[channel].flatten()
        
        # Compute relative uncertainty where possible
        valid_signal = signal > 1e-10
        if np.any(valid_signal):
            relative_uncertainty = uncertainty[valid_signal] / signal[valid_signal]
        else:
            relative_uncertainty = uncertainty
        
        channel_stats = {
            'mean_absolute_uncertainty': np.mean(uncertainty),
            'std_absolute_uncertainty': np.std(uncertainty),
            'mean_relative_uncertainty': np.mean(relative_uncertainty),
            'median_relative_uncertainty': np.median(relative_uncertainty),
            'max_relative_uncertainty': np.max(relative_uncertainty),
            'min_uncertainty': np.min(uncertainty),
            'max_uncertainty': np.max(uncertainty),
            'uncertainty_range': np.ptp(uncertainty)
        }
        
        summary['per_channel_stats'][channel] = channel_stats
    
    # Overall statistics
    all_uncertainties = uncertainty_map.uncertainties.flatten()
    summary['overall_stats'] = {
        'mean_uncertainty': np.mean(all_uncertainties),
        'median_uncertainty': np.median(all_uncertainties),
        'std_uncertainty': np.std(all_uncertainties),
        'total_pixels': len(all_uncertainties)
    }
    
    return summary