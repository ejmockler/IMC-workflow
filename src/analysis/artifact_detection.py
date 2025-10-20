"""
Artifact detection and correction for IMC data.

Handles hot pixels, oxidation state interference, deadtime/saturation effects,
and other detector-related artifacts with full uncertainty propagation.

Key Features:
- Hot pixel detection using spatial outlier statistics
- Oxidation state modeling via mass adjacency graphs  
- Deadtime and saturation correction with detector physics
- Spatial interpolation for artifact correction
- Vectorized operations for large datasets
- Graceful degradation for edge cases
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
import logging
from scipy.ndimage import median_filter, binary_dilation, label
from scipy.spatial.distance import pdist, squareform
from scipy.stats import median_abs_deviation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    HAS_NUMBA = False

try:
    from skimage.restoration import inpaint
    HAS_SKIMAGE_INPAINT = True
except ImportError:
    HAS_SKIMAGE_INPAINT = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for detector physics parameters."""
    deadtime_ns: float = 50.0           # Detector deadtime in nanoseconds
    saturation_level: int = 65535       # Maximum detector counts (16-bit)
    nonlinearity_coeff: float = 1e-6   # Nonlinearity coefficient
    dark_current: float = 0.1           # Dark current level
    
    def __post_init__(self):
        if self.deadtime_ns < 0:
            raise ValueError("Deadtime must be non-negative")
        if self.saturation_level <= 0:
            raise ValueError("Saturation level must be positive")


@dataclass(frozen=True)
class ArtifactDetectionResult:
    """Results from artifact detection and correction."""
    corrected_counts: np.ndarray        # Corrected ion counts
    artifact_mask: np.ndarray           # Boolean mask of detected artifacts
    uncertainty_map: np.ndarray         # Uncertainty introduced by correction
    correction_metadata: Dict[str, Any] # Correction statistics and parameters


class ArtifactDetectionError(Exception):
    """Exception raised for artifact detection failures."""
    pass


def detect_hot_pixels(
    ion_counts: np.ndarray,
    threshold_sigma: float = 5.0,
    kernel_size: int = 3,
    min_neighbors: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect hot pixels using spatial outlier statistics.
    
    Hot pixels are identified as pixels that are statistical outliers
    compared to their spatial neighborhood.
    
    Args:
        ion_counts: 2D array of ion counts
        threshold_sigma: Standard deviations above local statistics for detection
        kernel_size: Size of neighborhood for local statistics (odd number)
        min_neighbors: Minimum valid neighbors required for detection
        
    Returns:
        Tuple of (corrected_counts, uncertainty_map)
    """
    logger.debug(f"Detecting hot pixels with threshold {threshold_sigma}σ")
    
    if ion_counts.ndim != 2:
        raise ArtifactDetectionError("Ion counts must be 2D array")
    
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    
    # Compute local statistics using robust estimators
    local_median = median_filter(ion_counts, size=kernel_size)
    
    # Compute local MAD (Median Absolute Deviation) for robust scale estimation
    local_mad = _compute_local_mad(ion_counts, kernel_size)
    
    # Identify outliers using robust z-score
    robust_z_scores = (ion_counts - local_median) / (local_mad + 1e-10)
    hot_pixel_mask = robust_z_scores > threshold_sigma
    
    # Filter out edge pixels and pixels with insufficient neighbors
    valid_mask = _get_valid_pixel_mask(ion_counts.shape, kernel_size, min_neighbors)
    hot_pixel_mask = hot_pixel_mask & valid_mask
    
    # Correct hot pixels using spatial interpolation
    corrected_counts = ion_counts.copy()
    uncertainty_map = np.ones_like(ion_counts)
    
    if np.any(hot_pixel_mask):
        corrected_counts, interpolation_uncertainty = _interpolate_artifacts(
            ion_counts, hot_pixel_mask, method='median'
        )
        uncertainty_map[hot_pixel_mask] = interpolation_uncertainty[hot_pixel_mask]
        
        n_hot_pixels = np.sum(hot_pixel_mask)
        logger.info(f"Detected and corrected {n_hot_pixels} hot pixels ({100*n_hot_pixels/ion_counts.size:.2f}%)")
    
    return corrected_counts, uncertainty_map


def correct_oxidation_states(
    ion_counts: Dict[str, np.ndarray],
    oxidation_graph: Dict[str, List[str]],
    correction_method: str = 'proportional'
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Correct for oxidation state interference using mass adjacency relationships.
    
    Oxidation can cause signal to appear in adjacent mass channels.
    This function models and corrects these systematic interferences.
    
    Args:
        ion_counts: Dictionary mapping channel_name -> ion_count_array
        oxidation_graph: Dict mapping primary_channel -> [interfering_channels]
        correction_method: Method for estimating oxidation contribution
        
    Returns:
        Tuple of (corrected_counts, uncertainty_maps)
    """
    logger.debug(f"Correcting oxidation interference for {len(oxidation_graph)} channels")
    
    corrected_counts = {}
    uncertainty_maps = {}
    
    for primary_channel, interfering_channels in oxidation_graph.items():
        
        if primary_channel not in ion_counts:
            logger.warning(f"Primary channel {primary_channel} not found in data")
            continue
        
        observed_signal = ion_counts[primary_channel].copy()
        original_signal = observed_signal.copy()
        
        # Estimate and subtract oxidation contributions
        total_oxidation_signal = np.zeros_like(observed_signal)
        oxidation_uncertainty = np.zeros_like(observed_signal)
        
        for interfering_channel in interfering_channels:
            if interfering_channel not in ion_counts:
                continue
            
            interfering_signal = ion_counts[interfering_channel]
            
            # Estimate oxidation transfer coefficient
            if correction_method == 'proportional':
                transfer_coeff, transfer_uncertainty = _estimate_proportional_transfer(
                    observed_signal, interfering_signal
                )
            elif correction_method == 'regression':
                transfer_coeff, transfer_uncertainty = _estimate_regression_transfer(
                    observed_signal, interfering_signal
                )
            else:
                raise ValueError(f"Unknown correction method: {correction_method}")
            
            # Compute oxidation contribution
            oxidation_contribution = transfer_coeff * interfering_signal
            total_oxidation_signal += oxidation_contribution
            
            # Propagate uncertainty
            contrib_uncertainty = transfer_uncertainty * interfering_signal
            oxidation_uncertainty += contrib_uncertainty**2
        
        # Apply correction with positivity constraint
        corrected_signal = observed_signal - total_oxidation_signal
        
        # Keep minimum 5% of original signal to avoid complete suppression
        min_signal = 0.05 * original_signal
        corrected_signal = np.maximum(corrected_signal, min_signal)
        
        # Final uncertainty combines oxidation and suppression uncertainty
        final_uncertainty = np.sqrt(oxidation_uncertainty)
        suppression_mask = corrected_signal <= min_signal
        final_uncertainty[suppression_mask] *= 3.0  # Higher uncertainty for suppressed pixels
        
        corrected_counts[primary_channel] = corrected_signal
        uncertainty_maps[primary_channel] = final_uncertainty
    
    # Copy non-corrected channels
    for channel, counts in ion_counts.items():
        if channel not in corrected_counts:
            corrected_counts[channel] = counts.copy()
            uncertainty_maps[channel] = np.ones_like(counts)
    
    logger.info(f"Oxidation correction applied to {len(oxidation_graph)} channels")
    
    return corrected_counts, uncertainty_maps


def correct_detector_nonlinearity(
    ion_counts: np.ndarray,
    detector_config: DetectorConfig,
    acquisition_time_ms: float = 1000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correct for detector deadtime and saturation nonlinearity.
    
    At high count rates, detectors suffer from deadtime losses and 
    saturation effects that cause nonlinear response.
    
    Args:
        ion_counts: Raw ion count array
        detector_config: Detector physics parameters
        acquisition_time_ms: Pixel acquisition time in milliseconds
        
    Returns:
        Tuple of (corrected_counts, uncertainty_map)
    """
    logger.debug("Applying detector nonlinearity correction")
    
    # Convert acquisition time to seconds
    acquisition_time_s = acquisition_time_ms / 1000.0
    
    # Deadtime correction: true_rate = observed_rate / (1 - observed_rate * deadtime)
    observed_rate = ion_counts / acquisition_time_s  # counts per second
    deadtime_s = detector_config.deadtime_ns * 1e-9
    
    # Avoid division by zero and numerical issues
    deadtime_factor = observed_rate * deadtime_s
    correction_factor = 1.0 / (1.0 - np.clip(deadtime_factor, 0, 0.99))
    
    deadtime_corrected = ion_counts * correction_factor
    
    # Saturation correction for high count rates
    saturation_corrected = _correct_saturation(
        deadtime_corrected, detector_config.saturation_level
    )
    
    # Dark current subtraction
    dark_corrected = np.maximum(
        saturation_corrected - detector_config.dark_current * acquisition_time_s,
        0.1 * saturation_corrected
    )
    
    # Estimate uncertainty from correction
    correction_magnitude = np.abs(dark_corrected - ion_counts) / (ion_counts + 1)
    uncertainty_map = 1.0 + 0.5 * correction_magnitude  # Base uncertainty plus correction uncertainty
    
    # Higher uncertainty for heavily corrected pixels
    heavy_correction_mask = correction_magnitude > 0.5
    uncertainty_map[heavy_correction_mask] *= 2.0
    
    logger.debug(f"Detector correction: mean factor = {np.mean(correction_factor):.3f}")
    
    return dark_corrected, uncertainty_map


def detect_and_correct_artifacts(
    ion_counts: Dict[str, np.ndarray],
    detector_config: Optional[DetectorConfig] = None,
    oxidation_graph: Optional[Dict[str, List[str]]] = None,
    hot_pixel_threshold: float = 5.0,
    acquisition_time_ms: float = 1000.0
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Comprehensive artifact detection and correction pipeline.
    
    Applies all artifact corrections in the optimal order:
    1. Detector nonlinearity (deadtime, saturation)
    2. Hot pixel detection and correction
    3. Oxidation state interference correction
    
    Args:
        ion_counts: Dictionary mapping channel_name -> ion_count_array
        detector_config: Detector physics parameters
        oxidation_graph: Oxidation interference relationships
        hot_pixel_threshold: Standard deviations for hot pixel detection
        acquisition_time_ms: Pixel acquisition time
        
    Returns:
        Tuple of (corrected_counts, uncertainty_maps, correction_metadata)
    """
    logger.info("Starting comprehensive artifact correction pipeline")
    
    if detector_config is None:
        detector_config = DetectorConfig()
    
    corrected_counts = {}
    uncertainty_maps = {}
    correction_metadata = {
        'corrections_applied': [],
        'detector_config': detector_config,
        'hot_pixel_threshold': hot_pixel_threshold
    }
    
    # Step 1: Detector nonlinearity correction (applied per channel)
    for channel, counts in ion_counts.items():
        corrected_channel, detector_uncertainty = correct_detector_nonlinearity(
            counts, detector_config, acquisition_time_ms
        )
        corrected_counts[channel] = corrected_channel
        uncertainty_maps[channel] = detector_uncertainty
    
    correction_metadata['corrections_applied'].append('detector_nonlinearity')
    
    # Step 2: Hot pixel correction (applied per channel)
    hot_pixel_stats = {}
    for channel in corrected_counts:
        corrected_channel, hot_pixel_uncertainty = detect_hot_pixels(
            corrected_counts[channel], hot_pixel_threshold
        )
        
        # Combine uncertainties
        combined_uncertainty = np.sqrt(
            uncertainty_maps[channel]**2 + hot_pixel_uncertainty**2
        )
        
        corrected_counts[channel] = corrected_channel
        uncertainty_maps[channel] = combined_uncertainty
        
        # Track hot pixel statistics
        n_hot_pixels = np.sum(hot_pixel_uncertainty > 1.1)  # Pixels with elevated uncertainty
        hot_pixel_stats[channel] = {
            'n_hot_pixels': n_hot_pixels,
            'fraction_corrected': n_hot_pixels / corrected_channel.size
        }
    
    correction_metadata['corrections_applied'].append('hot_pixels')
    correction_metadata['hot_pixel_stats'] = hot_pixel_stats
    
    # Step 3: Oxidation correction (applied across channels)
    if oxidation_graph:
        oxidation_corrected, oxidation_uncertainty = correct_oxidation_states(
            corrected_counts, oxidation_graph
        )
        
        # Combine uncertainties for corrected channels
        for channel in oxidation_corrected:
            if channel in oxidation_uncertainty:
                combined_uncertainty = np.sqrt(
                    uncertainty_maps[channel]**2 + oxidation_uncertainty[channel]**2
                )
                uncertainty_maps[channel] = combined_uncertainty
        
        corrected_counts = oxidation_corrected
        correction_metadata['corrections_applied'].append('oxidation_states')
        correction_metadata['oxidation_channels'] = list(oxidation_graph.keys())
    
    logger.info(f"Artifact correction completed. Applied: {correction_metadata['corrections_applied']}")
    
    return corrected_counts, uncertainty_maps, correction_metadata


# Helper functions

@jit(nopython=True if HAS_NUMBA else False)
def _compute_local_mad(data: np.ndarray, kernel_size: int) -> np.ndarray:
    """Compute local Median Absolute Deviation for robust outlier detection."""
    
    padded_data = np.pad(data, kernel_size//2, mode='reflect')
    mad_array = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Extract local neighborhood
            neighborhood = padded_data[i:i+kernel_size, j:j+kernel_size]
            
            # Compute local MAD
            local_median = np.median(neighborhood)
            deviations = np.abs(neighborhood - local_median)
            local_mad = np.median(deviations)
            
            mad_array[i, j] = local_mad
    
    return mad_array


def _get_valid_pixel_mask(
    shape: Tuple[int, int], 
    kernel_size: int, 
    min_neighbors: int
) -> np.ndarray:
    """Create mask for pixels with sufficient neighbors for reliable statistics."""
    
    mask = np.ones(shape, dtype=bool)
    border = kernel_size // 2
    
    # Exclude border pixels
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False
    
    return mask


def _interpolate_artifacts(
    ion_counts: np.ndarray,
    artifact_mask: np.ndarray,
    method: str = 'median'
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate artifact pixels using spatial information."""
    
    corrected = ion_counts.copy()
    uncertainty = np.ones_like(ion_counts)
    
    if method == 'median':
        # Simple median filter interpolation
        interpolated_values = median_filter(ion_counts, size=3)
        corrected[artifact_mask] = interpolated_values[artifact_mask]
        uncertainty[artifact_mask] = 3.0  # Higher uncertainty for interpolated pixels
        
    elif method == 'inpainting' and HAS_SKIMAGE_INPAINT:
        # Advanced inpainting if available
        try:
            corrected = inpaint.inpaint_biharmonic(ion_counts, artifact_mask)
            uncertainty[artifact_mask] = 2.0
        except:
            # Fallback to median
            interpolated_values = median_filter(ion_counts, size=3)
            corrected[artifact_mask] = interpolated_values[artifact_mask]
            uncertainty[artifact_mask] = 3.0
    else:
        # Fallback median interpolation
        interpolated_values = median_filter(ion_counts, size=3)
        corrected[artifact_mask] = interpolated_values[artifact_mask]
        uncertainty[artifact_mask] = 3.0
    
    return corrected, uncertainty


def _estimate_proportional_transfer(
    primary_signal: np.ndarray,
    interfering_signal: np.ndarray
) -> Tuple[float, float]:
    """Estimate oxidation transfer coefficient using proportional method."""
    
    # Use pixels where both signals are present
    valid_mask = (primary_signal > 0) & (interfering_signal > 0)
    
    if np.sum(valid_mask) < 10:
        return 0.0, 1.0  # No reliable estimate
    
    # Estimate transfer coefficient as median ratio
    ratios = primary_signal[valid_mask] / interfering_signal[valid_mask]
    
    # Use robust statistics
    transfer_coeff = np.median(ratios)
    transfer_uncertainty = median_abs_deviation(ratios)
    
    # Bound the coefficient to reasonable values
    transfer_coeff = np.clip(transfer_coeff, 0, 1.0)
    
    return transfer_coeff, transfer_uncertainty


def _estimate_regression_transfer(
    primary_signal: np.ndarray,
    interfering_signal: np.ndarray
) -> Tuple[float, float]:
    """Estimate oxidation transfer coefficient using regression."""
    
    # Flatten arrays for regression
    primary_flat = primary_signal.flatten()
    interfering_flat = interfering_signal.flatten()
    
    # Use only positive signals
    valid_mask = (primary_flat > 0) & (interfering_flat > 0)
    
    if np.sum(valid_mask) < 10:
        return 0.0, 1.0
    
    x = interfering_flat[valid_mask].reshape(-1, 1)
    y = primary_flat[valid_mask]
    
    # Simple linear regression: y = coeff * x
    try:
        coeff = np.sum(x.flatten() * y) / np.sum(x.flatten()**2)
        
        # Estimate uncertainty from residuals
        predicted = coeff * x.flatten()
        residuals = y - predicted
        uncertainty = np.std(residuals) / np.sqrt(len(y))
        
        coeff = np.clip(coeff, 0, 1.0)
        
        return coeff, uncertainty
        
    except:
        return 0.0, 1.0


def _correct_saturation(
    counts: np.ndarray,
    saturation_level: int
) -> np.ndarray:
    """Apply saturation correction for counts near detector limit."""
    
    # Identify saturated pixels
    saturated_mask = counts >= 0.95 * saturation_level
    
    if not np.any(saturated_mask):
        return counts  # No saturation
    
    corrected = counts.copy()
    
    # Simple saturation correction: extrapolate from nearby pixels
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if saturated_mask[i, j]:
                # Find nearby non-saturated pixels
                local_region = counts[max(0, i-2):min(counts.shape[0], i+3),
                                    max(0, j-2):min(counts.shape[1], j+3)]
                local_mask = saturated_mask[max(0, i-2):min(counts.shape[0], i+3),
                                          max(0, j-2):min(counts.shape[1], j+3)]
                
                non_saturated = local_region[~local_mask]
                if len(non_saturated) > 0:
                    # Extrapolate based on local maximum
                    corrected[i, j] = min(1.5 * np.max(non_saturated), 2 * saturation_level)
    
    return corrected


def create_default_oxidation_graph() -> Dict[str, List[str]]:
    """Create default oxidation interference graph for common IMC markers."""
    
    # Common oxidation interferences in IMC
    oxidation_graph = {
        # Lanthanides commonly affected by oxidation
        '139La': ['140Ce'],     # La -> Ce oxidation  
        '140Ce': ['141Pr'],     # Ce -> Pr oxidation
        '141Pr': ['142Nd'],     # Pr -> Nd oxidation
        '151Eu': ['152Sm'],     # Eu -> Sm oxidation
        '153Eu': ['154Sm'],     # Eu -> Sm oxidation
        '159Tb': ['160Gd'],     # Tb -> Gd oxidation
        '165Ho': ['166Er'],     # Ho -> Er oxidation
        '169Tm': ['170Er'],     # Tm -> Er oxidation
        '175Lu': ['176Yb'],     # Lu -> Yb oxidation
        
        # Metal oxidation states
        '56Fe': ['57Fe'],       # Fe2+ -> Fe3+ 
        '63Cu': ['65Cu'],       # Cu isotope oxidation
        '66Zn': ['67Zn'],       # Zn isotope effects
    }
    
    return oxidation_graph