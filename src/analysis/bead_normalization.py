"""
Bead Normalization Protocol for IMC Pipeline

Cross-run instrument standardization using calibration beads with comprehensive
quality control and validation. Builds on existing batch correction and artifact
detection infrastructure for seamless pipeline integration.

Key Features:
- Automated bead detection using artifact_detection patterns
- Cross-run normalization factor computation
- Integration with existing batch_correction infrastructure  
- Quality control metrics and validation framework
- Storage compatibility with data_storage systems
- Pipeline integration through main_pipeline architecture
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import warnings
import logging
from datetime import datetime
from pathlib import Path
import json

# Import existing infrastructure patterns
from .artifact_detection import (
    DetectorConfig, ArtifactDetectionError, detect_hot_pixels,
    _get_valid_pixel_mask, _interpolate_artifacts
)
from .batch_correction import (
    BatchCorrectionConfig, validate_batch_structure, 
    detect_batch_effects, _compute_improvement_metrics
)
from .ion_count_processing import estimate_optimal_cofactor, apply_arcsinh_transform
from .quality_control import (
    monitor_calibration_channels, detect_spatial_artifacts,
    track_calibration_drift, validate_batch_consistency
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BeadDetectionConfig:
    """Configuration for bead detection parameters."""
    bead_channels: List[str]                    # Standard bead channels
    signal_threshold_percentile: float = 95.0   # Signal threshold as percentile
    spatial_uniformity_threshold: float = 0.3   # CV threshold for spatial uniformity
    intensity_stability_threshold: float = 0.2  # CV threshold for intensity stability
    min_bead_pixels: int = 50                   # Minimum pixels required for valid bead
    hot_pixel_sigma: float = 3.0                # Sigma for hot pixel rejection
    edge_exclusion_fraction: float = 0.1        # Fraction of ROI edge to exclude
    
    def __post_init__(self):
        if not self.bead_channels:
            raise ValueError("Bead channels list cannot be empty")
        if self.signal_threshold_percentile <= 0 or self.signal_threshold_percentile >= 100:
            raise ValueError("Signal threshold percentile must be between 0 and 100")


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration for normalization protocol parameters."""
    reference_selection_method: str = 'median'  # 'median', 'first_run', 'external'
    normalization_method: str = 'multiplicative' # 'multiplicative', 'additive', 'robust'
    outlier_detection_method: str = 'iqr'       # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold: float = 1.5              # Threshold for outlier detection
    minimum_signal_ratio: float = 0.1           # Minimum signal relative to reference
    maximum_signal_ratio: float = 10.0          # Maximum signal relative to reference
    confidence_level: float = 0.95              # Confidence level for intervals
    bootstrap_samples: int = 1000               # Bootstrap samples for uncertainty
    
    def __post_init__(self):
        if self.reference_selection_method not in ['median', 'first_run', 'external']:
            raise ValueError("Invalid reference selection method")
        if self.normalization_method not in ['multiplicative', 'additive', 'robust']:
            raise ValueError("Invalid normalization method")


@dataclass
class BeadDetectionResult:
    """Results from bead detection analysis."""
    detected_beads: Dict[str, np.ndarray]       # Channel -> bead mask
    bead_intensities: Dict[str, float]          # Channel -> median intensity
    quality_metrics: Dict[str, Any]             # Quality assessment metrics
    spatial_uniformity: Dict[str, float]        # Channel -> spatial CV
    detection_metadata: Dict[str, Any]          # Detection parameters and stats
    
    def is_valid_for_normalization(self) -> bool:
        """Check if bead detection is suitable for normalization."""
        return (
            len(self.detected_beads) > 0 and
            all(self.quality_metrics.get(f'{ch}_valid', False) for ch in self.detected_beads.keys()) and
            all(intensity > 0 for intensity in self.bead_intensities.values())
        )


@dataclass
class NormalizationResult:
    """Results from bead normalization protocol."""
    normalization_factors: Dict[str, float]     # Channel -> normalization factor
    normalized_data: Dict[str, np.ndarray]      # Channel -> normalized intensities
    reference_values: Dict[str, float]          # Channel -> reference values used
    quality_assessment: Dict[str, Any]          # Normalization quality metrics
    uncertainty_estimates: Dict[str, float]     # Channel -> uncertainty estimates
    correction_metadata: Dict[str, Any]         # Correction parameters and stats


class BeadNormalizationError(Exception):
    """Exception raised for bead normalization failures."""
    pass


def detect_calibration_beads(
    ion_counts: Dict[str, np.ndarray],
    coords: np.ndarray,
    config: BeadDetectionConfig,
    roi_metadata: Optional[Dict[str, Any]] = None
) -> BeadDetectionResult:
    """
    Detect calibration beads using robust spatial and intensity analysis.
    
    Uses artifact_detection patterns for spatial outlier identification
    combined with intensity-based bead characterization.
    
    Args:
        ion_counts: Dictionary mapping channel_name -> ion_count_array
        coords: Nx2 array of spatial coordinates
        config: Bead detection configuration
        roi_metadata: Optional ROI metadata for context
        
    Returns:
        BeadDetectionResult with detected beads and quality metrics
    """
    logger.info(f"Detecting calibration beads in {len(config.bead_channels)} channels")
    
    detected_beads = {}
    bead_intensities = {}
    quality_metrics = {}
    spatial_uniformity = {}
    detection_stats = {
        'total_pixels': len(coords),
        'detection_method': 'spatial_intensity_hybrid',
        'channels_processed': [],
        'channels_detected': []
    }
    
    # Create spatial exclusion mask (exclude ROI edges)
    if len(coords) > 0:
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        x_margin = (x_max - x_min) * config.edge_exclusion_fraction
        y_margin = (y_max - y_min) * config.edge_exclusion_fraction
        
        edge_mask = (
            (coords[:, 0] >= x_min + x_margin) &
            (coords[:, 0] <= x_max - x_margin) &
            (coords[:, 1] >= y_min + y_margin) &
            (coords[:, 1] <= y_max - y_margin)
        )
    else:
        edge_mask = np.array([], dtype=bool)
    
    for channel in config.bead_channels:
        detection_stats['channels_processed'].append(channel)
        
        if channel not in ion_counts:
            logger.warning(f"Bead channel {channel} not found in data")
            quality_metrics[f'{channel}_valid'] = False
            continue
        
        channel_data = ion_counts[channel]
        
        try:
            # Step 1: Hot pixel rejection using artifact_detection patterns
            corrected_data, _ = detect_hot_pixels(
                channel_data.reshape(-1, 1),  # Make 2D for hot pixel detection
                threshold_sigma=config.hot_pixel_sigma,
                kernel_size=3,
                min_neighbors=4
            )
            corrected_data = corrected_data.flatten()
            
            # Step 2: Intensity-based bead identification
            signal_threshold = np.percentile(
                corrected_data[corrected_data > 0], 
                config.signal_threshold_percentile
            )
            
            high_intensity_mask = corrected_data >= signal_threshold
            
            # Step 3: Spatial filtering (exclude edges)
            if len(edge_mask) == len(high_intensity_mask):
                bead_candidates = high_intensity_mask & edge_mask
            else:
                bead_candidates = high_intensity_mask
            
            # Step 4: Quality assessment
            n_bead_pixels = np.sum(bead_candidates)
            
            if n_bead_pixels < config.min_bead_pixels:
                logger.warning(f"Insufficient bead pixels in {channel}: {n_bead_pixels} < {config.min_bead_pixels}")
                quality_metrics[f'{channel}_valid'] = False
                continue
            
            # Calculate bead intensity statistics
            bead_signals = corrected_data[bead_candidates]
            median_intensity = np.median(bead_signals)
            
            # Spatial uniformity assessment
            if len(coords) == len(bead_candidates):
                bead_coords = coords[bead_candidates]
                if len(bead_coords) > 1:
                    # Simple spatial uniformity: CV of bead intensities
                    spatial_cv = np.std(bead_signals) / (np.mean(bead_signals) + 1e-10)
                else:
                    spatial_cv = 0.0
            else:
                spatial_cv = np.nan
            
            # Quality checks
            intensity_stable = (
                np.std(bead_signals) / (np.mean(bead_signals) + 1e-10) 
                <= config.intensity_stability_threshold
            )
            spatially_uniform = (
                np.isnan(spatial_cv) or 
                spatial_cv <= config.spatial_uniformity_threshold
            )
            
            is_valid = intensity_stable and spatially_uniform and n_bead_pixels >= config.min_bead_pixels
            
            # Store results
            detected_beads[channel] = bead_candidates
            bead_intensities[channel] = median_intensity
            spatial_uniformity[channel] = spatial_cv if not np.isnan(spatial_cv) else 0.0
            quality_metrics[f'{channel}_valid'] = is_valid
            quality_metrics[f'{channel}_n_pixels'] = n_bead_pixels
            quality_metrics[f'{channel}_intensity_cv'] = np.std(bead_signals) / (np.mean(bead_signals) + 1e-10)
            quality_metrics[f'{channel}_median_intensity'] = median_intensity
            
            if is_valid:
                detection_stats['channels_detected'].append(channel)
                logger.info(f"Valid beads detected in {channel}: {n_bead_pixels} pixels, "
                           f"intensity={median_intensity:.1f}, spatial_cv={spatial_cv:.3f}")
            else:
                logger.warning(f"Poor quality beads in {channel}: "
                             f"stable={intensity_stable}, uniform={spatially_uniform}")
        
        except Exception as e:
            logger.error(f"Bead detection failed for {channel}: {e}")
            quality_metrics[f'{channel}_valid'] = False
            quality_metrics[f'{channel}_error'] = str(e)
    
    # Overall detection statistics
    detection_stats.update({
        'n_channels_processed': len(detection_stats['channels_processed']),
        'n_channels_detected': len(detection_stats['channels_detected']),
        'detection_success_rate': len(detection_stats['channels_detected']) / max(1, len(detection_stats['channels_processed'])),
        'config_used': asdict(config)
    })
    
    result = BeadDetectionResult(
        detected_beads=detected_beads,
        bead_intensities=bead_intensities,
        quality_metrics=quality_metrics,
        spatial_uniformity=spatial_uniformity,
        detection_metadata=detection_stats
    )
    
    logger.info(f"Bead detection completed: {len(detected_beads)} channels, "
               f"{result.is_valid_for_normalization()} valid for normalization")
    
    return result


def compute_normalization_factors(
    run_bead_data: Dict[str, BeadDetectionResult],
    config: NormalizationConfig,
    external_reference: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Compute cross-run normalization factors from bead detection results.
    
    Args:
        run_bead_data: Dictionary mapping run_id -> BeadDetectionResult
        config: Normalization configuration
        external_reference: Optional external reference values
        
    Returns:
        Dictionary with normalization factors and quality metrics
    """
    logger.info(f"Computing normalization factors for {len(run_bead_data)} runs")
    
    if not run_bead_data:
        raise BeadNormalizationError("No bead data provided for normalization")
    
    # Extract bead intensities by channel across runs
    channel_intensities = {}
    valid_runs = []
    
    for run_id, bead_result in run_bead_data.items():
        if not bead_result.is_valid_for_normalization():
            logger.warning(f"Skipping run {run_id}: invalid bead detection")
            continue
        
        valid_runs.append(run_id)
        
        for channel, intensity in bead_result.bead_intensities.items():
            if channel not in channel_intensities:
                channel_intensities[channel] = {}
            channel_intensities[channel][run_id] = intensity
    
    if not valid_runs:
        raise BeadNormalizationError("No valid runs for normalization")
    
    logger.info(f"Using {len(valid_runs)} valid runs for normalization")
    
    # Compute reference values
    reference_values = {}
    
    if config.reference_selection_method == 'external' and external_reference:
        reference_values = external_reference.copy()
        logger.info("Using external reference values")
    else:
        for channel, run_intensities in channel_intensities.items():
            intensities = list(run_intensities.values())
            
            if config.reference_selection_method == 'median':
                reference_values[channel] = np.median(intensities)
            elif config.reference_selection_method == 'first_run':
                # Use first valid run as reference
                first_run = min(run_intensities.keys())
                reference_values[channel] = run_intensities[first_run]
            
        logger.info(f"Computed reference values using {config.reference_selection_method} method")
    
    # Detect outliers across runs
    outlier_runs = set()
    
    for channel, run_intensities in channel_intensities.items():
        if channel not in reference_values:
            continue
            
        intensities = np.array(list(run_intensities.values()))
        run_ids = list(run_intensities.keys())
        
        # Outlier detection
        if config.outlier_detection_method == 'iqr':
            q1, q3 = np.percentile(intensities, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - config.outlier_threshold * iqr
            upper_bound = q3 + config.outlier_threshold * iqr
            outliers = (intensities < lower_bound) | (intensities > upper_bound)
        elif config.outlier_detection_method == 'zscore':
            z_scores = np.abs((intensities - np.mean(intensities)) / (np.std(intensities) + 1e-10))
            outliers = z_scores > config.outlier_threshold
        else:
            outliers = np.zeros(len(intensities), dtype=bool)
        
        for i, is_outlier in enumerate(outliers):
            if is_outlier:
                outlier_runs.add(run_ids[i])
                logger.warning(f"Outlier detected: run {run_ids[i]}, channel {channel}, "
                             f"intensity {intensities[i]:.1f}")
    
    # Remove outlier runs
    filtered_valid_runs = [run_id for run_id in valid_runs if run_id not in outlier_runs]
    
    if len(filtered_valid_runs) < 2:
        logger.warning("Insufficient runs after outlier removal, proceeding with all valid runs")
        filtered_valid_runs = valid_runs
    
    # Compute normalization factors
    normalization_factors = {}
    uncertainty_estimates = {}
    
    for run_id in filtered_valid_runs:
        run_factors = {}
        run_uncertainties = {}
        
        bead_result = run_bead_data[run_id]
        
        for channel, observed_intensity in bead_result.bead_intensities.items():
            if channel not in reference_values:
                continue
                
            reference_intensity = reference_values[channel]
            
            if config.normalization_method == 'multiplicative':
                factor = reference_intensity / (observed_intensity + 1e-10)
            elif config.normalization_method == 'additive':
                factor = reference_intensity - observed_intensity
            elif config.normalization_method == 'robust':
                # Robust M-estimator approach
                factor = reference_intensity / (observed_intensity + 1e-10)
                # Could add Huber loss or other robust estimates here
            else:
                factor = 1.0
            
            # Apply signal ratio constraints
            factor = np.clip(factor, config.minimum_signal_ratio, config.maximum_signal_ratio)
            
            # Estimate uncertainty (simplified approach)
            # In practice, this could use bootstrap or propagation of measurement errors
            relative_uncertainty = 0.05  # 5% base uncertainty
            if channel in bead_result.spatial_uniformity:
                spatial_contrib = bead_result.spatial_uniformity[channel] * 0.5
                relative_uncertainty += spatial_contrib
            
            uncertainty = relative_uncertainty * abs(factor)
            
            run_factors[channel] = factor
            run_uncertainties[channel] = uncertainty
        
        normalization_factors[run_id] = run_factors
        uncertainty_estimates[run_id] = run_uncertainties
    
    # Quality assessment
    quality_metrics = _assess_normalization_quality(
        channel_intensities, reference_values, normalization_factors, config
    )
    
    result = {
        'normalization_factors': normalization_factors,
        'reference_values': reference_values,
        'uncertainty_estimates': uncertainty_estimates,
        'quality_metrics': quality_metrics,
        'valid_runs': filtered_valid_runs,
        'outlier_runs': list(outlier_runs),
        'config_used': asdict(config),
        'method': config.normalization_method
    }
    
    logger.info(f"Normalization factors computed for {len(filtered_valid_runs)} runs, "
               f"{len(outlier_runs)} outliers removed")
    
    return result


def apply_bead_normalization(
    ion_counts: Dict[str, np.ndarray],
    normalization_factors: Dict[str, float],
    method: str = 'multiplicative'
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Apply bead-based normalization factors to ion count data.
    
    Args:
        ion_counts: Dictionary mapping channel_name -> ion_count_array
        normalization_factors: Dictionary mapping channel_name -> factor
        method: Normalization method ('multiplicative' or 'additive')
        
    Returns:
        Tuple of (normalized_data, application_stats)
    """
    normalized_data = {}
    application_stats = {
        'method': method,
        'channels_normalized': [],
        'channels_unchanged': [],
        'normalization_summary': {}
    }
    
    for channel, counts in ion_counts.items():
        if channel in normalization_factors:
            factor = normalization_factors[channel]
            
            if method == 'multiplicative':
                normalized_counts = counts * factor
            elif method == 'additive':
                normalized_counts = counts + factor
            else:
                normalized_counts = counts.copy()
                factor = 1.0
            
            # Track statistics
            original_mean = np.mean(counts)
            normalized_mean = np.mean(normalized_counts)
            
            application_stats['channels_normalized'].append(channel)
            application_stats['normalization_summary'][channel] = {
                'factor_applied': factor,
                'original_mean': original_mean,
                'normalized_mean': normalized_mean,
                'correction_magnitude': abs(factor - 1.0) if method == 'multiplicative' else abs(factor)
            }
            
            normalized_data[channel] = normalized_counts
            
        else:
            # Channel not in normalization factors - keep original
            normalized_data[channel] = counts.copy()
            application_stats['channels_unchanged'].append(channel)
    
    logger.info(f"Applied normalization to {len(application_stats['channels_normalized'])} channels, "
               f"{len(application_stats['channels_unchanged'])} unchanged")
    
    return normalized_data, application_stats


def validate_bead_normalization(
    batch_data_before: Dict[str, Dict[str, np.ndarray]],
    batch_data_after: Dict[str, Dict[str, np.ndarray]],
    normalization_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate bead normalization effectiveness using batch correction patterns.
    
    Uses existing batch_correction validation infrastructure to assess
    normalization quality and improvement in cross-run consistency.
    
    Args:
        batch_data_before: Data before normalization
        batch_data_after: Data after normalization  
        normalization_metadata: Normalization metadata
        
    Returns:
        Comprehensive validation results
    """
    logger.info("Validating bead normalization effectiveness")
    
    validation_results = {
        'validation_method': 'batch_effect_analysis',
        'timestamp': datetime.now().isoformat(),
        'normalization_metadata': normalization_metadata
    }
    
    try:
        # Detect batch effects before normalization
        batch_effects_before = detect_batch_effects(batch_data_before)
        
        # Detect batch effects after normalization  
        batch_effects_after = detect_batch_effects(batch_data_after)
        
        # Compute improvement metrics
        improvement_metrics = _compute_improvement_metrics(
            batch_effects_before, batch_effects_after
        )
        
        # Validate batch consistency
        consistency_before = validate_batch_consistency(
            batch_data_before, 
            {run_id: {} for run_id in batch_data_before.keys()}
        )
        
        consistency_after = validate_batch_consistency(
            batch_data_after,
            {run_id: {} for run_id in batch_data_after.keys()}
        )
        
        # Channel-specific validation
        channel_validation = {}
        bead_channels = normalization_metadata.get('config_used', {}).get('bead_channels', [])
        
        for channel in bead_channels:
            if channel in batch_effects_before.get('protein_effects', {}):
                before_cv = batch_effects_before['protein_effects'][channel].get('effect_size', np.nan)
                after_cv = batch_effects_after['protein_effects'][channel].get('effect_size', np.nan)
                
                channel_validation[channel] = {
                    'cv_before_normalization': before_cv,
                    'cv_after_normalization': after_cv,
                    'improvement_achieved': before_cv > after_cv if not np.isnan(before_cv) and not np.isnan(after_cv) else False,
                    'relative_improvement': (before_cv - after_cv) / before_cv if before_cv > 0 else 0.0
                }
        
        # Overall validation assessment
        overall_severity_improved = (
            batch_effects_before['overall_severity'] > batch_effects_after['overall_severity']
        )
        
        normalization_effective = (
            overall_severity_improved and
            improvement_metrics['improvement_ratio'] > 0.1  # At least 10% improvement
        )
        
        validation_results.update({
            'batch_effects_before': batch_effects_before,
            'batch_effects_after': batch_effects_after,
            'improvement_metrics': improvement_metrics,
            'consistency_before': consistency_before,
            'consistency_after': consistency_after,
            'channel_validation': channel_validation,
            'overall_assessment': {
                'normalization_effective': normalization_effective,
                'severity_improved': overall_severity_improved,
                'improvement_ratio': improvement_metrics['improvement_ratio'],
                'recommendation': 'apply_normalization' if normalization_effective else 'review_parameters'
            }
        })
        
        logger.info(f"Validation completed: effective={normalization_effective}, "
                   f"improvement={improvement_metrics['improvement_ratio']:.2f}")
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        validation_results.update({
            'validation_failed': True,
            'error': str(e),
            'overall_assessment': {
                'normalization_effective': False,
                'recommendation': 'validation_failed'
            }
        })
    
    return validation_results


def _assess_normalization_quality(
    channel_intensities: Dict[str, Dict[str, float]],
    reference_values: Dict[str, float],
    normalization_factors: Dict[str, Dict[str, float]],
    config: NormalizationConfig
) -> Dict[str, Any]:
    """Assess quality of computed normalization factors."""
    
    quality_metrics = {
        'reference_consistency': {},
        'factor_stability': {},
        'overall_quality': 'good'
    }
    
    # Check reference consistency across channels
    for channel, run_intensities in channel_intensities.items():
        if channel not in reference_values:
            continue
            
        intensities = list(run_intensities.values())
        reference = reference_values[channel]
        
        # CV of observed intensities (lower is better)
        cv_observed = np.std(intensities) / (np.mean(intensities) + 1e-10)
        
        # Distance from reference
        deviations = [abs(intensity - reference) / reference for intensity in intensities]
        mean_deviation = np.mean(deviations)
        
        quality_metrics['reference_consistency'][channel] = {
            'cv_observed_intensities': cv_observed,
            'mean_deviation_from_reference': mean_deviation,
            'consistent': cv_observed < 0.3 and mean_deviation < 0.5
        }
    
    # Check factor stability across runs
    for channel in reference_values.keys():
        factors = []
        for run_id, run_factors in normalization_factors.items():
            if channel in run_factors:
                factors.append(run_factors[channel])
        
        if factors:
            cv_factors = np.std(factors) / (np.mean(factors) + 1e-10)
            extreme_factors = sum(1 for f in factors if f < 0.5 or f > 2.0)
            
            quality_metrics['factor_stability'][channel] = {
                'cv_normalization_factors': cv_factors,
                'n_extreme_factors': extreme_factors,
                'stable': cv_factors < 0.5 and extreme_factors == 0
            }
    
    # Overall quality assessment
    consistent_channels = sum(
        1 for metrics in quality_metrics['reference_consistency'].values()
        if metrics.get('consistent', False)
    )
    stable_channels = sum(
        1 for metrics in quality_metrics['factor_stability'].values()
        if metrics.get('stable', False)
    )
    
    total_channels = len(reference_values)
    
    if total_channels > 0:
        consistency_rate = consistent_channels / total_channels
        stability_rate = stable_channels / total_channels
        
        if consistency_rate >= 0.8 and stability_rate >= 0.8:
            quality_metrics['overall_quality'] = 'excellent'
        elif consistency_rate >= 0.6 and stability_rate >= 0.6:
            quality_metrics['overall_quality'] = 'good'
        elif consistency_rate >= 0.4 and stability_rate >= 0.4:
            quality_metrics['overall_quality'] = 'fair'
        else:
            quality_metrics['overall_quality'] = 'poor'
    
    quality_metrics.update({
        'consistency_rate': consistency_rate if total_channels > 0 else 0,
        'stability_rate': stability_rate if total_channels > 0 else 0,
        'total_channels_assessed': total_channels
    })
    
    return quality_metrics


def create_bead_normalization_report(
    detection_results: Dict[str, BeadDetectionResult],
    normalization_results: Dict[str, Any],
    validation_results: Dict[str, Any],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create comprehensive bead normalization report.
    
    Args:
        detection_results: Bead detection results by run
        normalization_results: Normalization computation results
        validation_results: Validation assessment results
        output_path: Optional path to save report
        
    Returns:
        Complete normalization report
    """
    report = {
        'report_metadata': {
            'creation_date': datetime.now().isoformat(),
            'report_type': 'bead_normalization_analysis',
            'n_runs_analyzed': len(detection_results)
        },
        'detection_summary': {},
        'normalization_summary': {},
        'validation_summary': {},
        'recommendations': []
    }
    
    # Detection summary
    valid_detections = sum(1 for result in detection_results.values() if result.is_valid_for_normalization())
    detection_success_rate = valid_detections / len(detection_results) if detection_results else 0
    
    report['detection_summary'] = {
        'total_runs': len(detection_results),
        'valid_detections': valid_detections,
        'detection_success_rate': detection_success_rate,
        'per_run_summary': {
            run_id: {
                'valid': result.is_valid_for_normalization(),
                'channels_detected': len(result.detected_beads),
                'median_spatial_uniformity': np.median(list(result.spatial_uniformity.values())) if result.spatial_uniformity else np.nan
            }
            for run_id, result in detection_results.items()
        }
    }
    
    # Normalization summary
    report['normalization_summary'] = {
        'method': normalization_results.get('method', 'unknown'),
        'valid_runs_used': len(normalization_results.get('valid_runs', [])),
        'outlier_runs_removed': len(normalization_results.get('outlier_runs', [])),
        'quality_assessment': normalization_results.get('quality_metrics', {}),
        'reference_values': normalization_results.get('reference_values', {})
    }
    
    # Validation summary
    overall_assessment = validation_results.get('overall_assessment', {})
    report['validation_summary'] = {
        'normalization_effective': overall_assessment.get('normalization_effective', False),
        'improvement_ratio': overall_assessment.get('improvement_ratio', 0.0),
        'recommendation': overall_assessment.get('recommendation', 'unknown'),
        'batch_effect_reduction': validation_results.get('improvement_metrics', {})
    }
    
    # Generate recommendations
    if detection_success_rate < 0.7:
        report['recommendations'].append(
            "Low bead detection success rate. Review bead signal thresholds and spatial uniformity requirements."
        )
    
    if normalization_results.get('quality_metrics', {}).get('overall_quality') in ['fair', 'poor']:
        report['recommendations'].append(
            "Normalization quality is suboptimal. Consider reviewing outlier detection parameters or reference selection method."
        )
    
    if not overall_assessment.get('normalization_effective', False):
        report['recommendations'].append(
            "Bead normalization did not effectively reduce batch effects. Consider alternative normalization strategies."
        )
    
    if len(normalization_results.get('outlier_runs', [])) > 0:
        outlier_runs = normalization_results['outlier_runs']
        report['recommendations'].append(
            f"Outlier runs detected ({', '.join(outlier_runs)}). Investigate potential instrumental issues."
        )
    
    if not report['recommendations']:
        report['recommendations'].append("Bead normalization completed successfully. Results are suitable for downstream analysis.")
    
    # Save report if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Bead normalization report saved to {output_path}")
    
    return report


def integrate_with_batch_correction(
    batch_data: Dict[str, Dict[str, np.ndarray]],
    batch_metadata: Dict[str, Dict[str, Any]],
    bead_config: Optional[BeadDetectionConfig] = None,
    norm_config: Optional[NormalizationConfig] = None,
    coords_data: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    """
    Integrate bead normalization with existing batch correction pipeline.
    
    This function provides seamless integration with the existing batch_correction
    infrastructure, allowing bead normalization to be used as a preprocessing
    step before sham-anchored or other batch correction methods.
    
    Args:
        batch_data: Dictionary mapping batch_id -> protein_name -> ion_counts
        batch_metadata: Dictionary mapping batch_id -> metadata_dict
        bead_config: Bead detection configuration (uses defaults if None)
        norm_config: Normalization configuration (uses defaults if None)
        coords_data: Optional spatial coordinates for each batch
        
    Returns:
        Tuple of (normalized_batch_data, normalization_metadata)
    """
    logger.info("Integrating bead normalization with batch correction pipeline")
    
    # Use default configurations if not provided
    if bead_config is None:
        bead_config = BeadDetectionConfig(
            bead_channels=['130Ba', '131Xe'],  # Common bead channels - should be configured from data
            signal_threshold_percentile=95.0,
            spatial_uniformity_threshold=0.3,
            min_bead_pixels=50
        )
    
    if norm_config is None:
        norm_config = NormalizationConfig(
            reference_selection_method='median',
            normalization_method='multiplicative',
            outlier_detection_method='iqr'
        )
    
    # Validate batch structure using existing infrastructure
    validate_batch_structure(batch_data, batch_metadata)
    
    # Step 1: Detect beads in each batch
    detection_results = {}
    
    for batch_id, protein_data in batch_data.items():
        try:
            # Get coordinates if available
            batch_coords = coords_data.get(batch_id, np.array([[0, 0]]))  # Fallback coordinates
            
            # Detect beads in this batch
            detection_result = detect_calibration_beads(
                ion_counts=protein_data,
                coords=batch_coords,
                config=bead_config,
                roi_metadata=batch_metadata.get(batch_id, {})
            )
            
            detection_results[batch_id] = detection_result
            
        except Exception as e:
            logger.warning(f"Bead detection failed for batch {batch_id}: {e}")
            # Create empty result for failed detection
            detection_results[batch_id] = BeadDetectionResult(
                detected_beads={},
                bead_intensities={},
                quality_metrics={},
                spatial_uniformity={},
                detection_metadata={'error': str(e)}
            )
    
    # Step 2: Compute normalization factors
    try:
        normalization_computation = compute_normalization_factors(
            run_bead_data=detection_results,
            config=norm_config
        )
    except BeadNormalizationError as e:
        logger.warning(f"Normalization factor computation failed: {e}")
        # Return original data if normalization fails
        return batch_data, {
            'normalization_applied': False,
            'error': str(e),
            'detection_results': detection_results
        }
    
    # Step 3: Apply normalization to each batch
    normalized_batch_data = {}
    application_stats = {}
    
    for batch_id, protein_data in batch_data.items():
        if batch_id in normalization_computation['normalization_factors']:
            factors = normalization_computation['normalization_factors'][batch_id]
            
            normalized_data, stats = apply_bead_normalization(
                ion_counts=protein_data,
                normalization_factors=factors,
                method=norm_config.normalization_method
            )
            
            normalized_batch_data[batch_id] = normalized_data
            application_stats[batch_id] = stats
        else:
            # No normalization factors available - keep original data
            normalized_batch_data[batch_id] = {k: v.copy() for k, v in protein_data.items()}
            application_stats[batch_id] = {'normalization_applied': False}
    
    # Step 4: Validate normalization effectiveness
    validation_results = validate_bead_normalization(
        batch_data_before=batch_data,
        batch_data_after=normalized_batch_data,
        normalization_metadata=normalization_computation
    )
    
    # Compile comprehensive metadata
    integration_metadata = {
        'normalization_applied': True,
        'method': 'bead_normalization',
        'detection_results': detection_results,
        'normalization_computation': normalization_computation,
        'application_statistics': application_stats,
        'validation_results': validation_results,
        'config_used': {
            'bead_detection': asdict(bead_config),
            'normalization': asdict(norm_config)
        },
        'integration_timestamp': datetime.now().isoformat()
    }
    
    logger.info("Bead normalization integration completed successfully")
    
    return normalized_batch_data, integration_metadata


# Integration with quality_control framework
def create_bead_normalization_qc_metrics(
    detection_results: Dict[str, BeadDetectionResult],
    normalization_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create QC metrics compatible with quality_control framework.
    
    Generates QC metrics that integrate with the existing quality_control
    module's monitor_calibration_channels and track_calibration_drift functions.
    """
    qc_metrics = {
        'bead_detection_qc': {},
        'normalization_qc': {},
        'drift_analysis': {},
        'overall_status': 'pass'
    }
    
    # Detection QC metrics
    valid_detections = 0
    total_detections = len(detection_results)
    
    for run_id, result in detection_results.items():
        is_valid = result.is_valid_for_normalization()
        if is_valid:
            valid_detections += 1
        
        qc_metrics['bead_detection_qc'][run_id] = {
            'valid': is_valid,
            'n_channels_detected': len(result.detected_beads),
            'mean_spatial_uniformity': np.mean(list(result.spatial_uniformity.values())) if result.spatial_uniformity else np.nan,
            'detection_quality_score': len(result.detected_beads) / len(result.detection_metadata.get('channels_processed', [1])) if result.detection_metadata.get('channels_processed') else 0
        }
    
    detection_success_rate = valid_detections / total_detections if total_detections > 0 else 0
    
    # Normalization QC metrics
    quality_assessment = normalization_results.get('quality_metrics', {})
    qc_metrics['normalization_qc'] = {
        'overall_quality': quality_assessment.get('overall_quality', 'unknown'),
        'consistency_rate': quality_assessment.get('consistency_rate', 0),
        'stability_rate': quality_assessment.get('stability_rate', 0),
        'n_outlier_runs': len(normalization_results.get('outlier_runs', [])),
        'normalization_method': normalization_results.get('method', 'unknown')
    }
    
    # Overall status determination
    if detection_success_rate < 0.5:
        qc_metrics['overall_status'] = 'fail'
    elif quality_assessment.get('overall_quality') in ['poor', 'fair']:
        qc_metrics['overall_status'] = 'warning'
    elif len(normalization_results.get('outlier_runs', [])) > 0.3 * total_detections:
        qc_metrics['overall_status'] = 'warning'
    
    qc_metrics.update({
        'detection_success_rate': detection_success_rate,
        'total_runs_analyzed': total_detections,
        'qc_timestamp': datetime.now().isoformat()
    })
    
    return qc_metrics