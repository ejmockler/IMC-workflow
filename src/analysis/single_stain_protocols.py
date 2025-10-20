"""
Single-Stain Reference Protocols for IMC Pipeline

Provides standardized protocols for single-antibody controls and spillover matrix
estimation from single-stain measurements. Integrates with existing spillover 
correction, uncertainty propagation, and quality control frameworks.

Key Features:
- Standardized single-antibody staining protocols
- Spillover coefficient estimation algorithms
- Integration with spillover_correction.py infrastructure
- Quality validation for spillover matrix assessment
- Physics-aware correction workflows
- Uncertainty propagation for error handling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Literal
from dataclasses import dataclass
import warnings
import logging
from pathlib import Path
import json
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, normaltest
from scipy.optimize import minimize
from scipy.linalg import lstsq, LinAlgError

# Import existing pipeline components
from .spillover_correction import (
    SpilloverMatrix, SpilloverCorrectionError, 
    estimate_spillover_matrix, validate_spillover_correction
)
from .uncertainty_propagation import (
    UncertaintyMap, UncertaintyConfig, UncertaintyPropagationError,
    create_base_uncertainty, propagate_through_spillover_correction
)
from .artifact_detection import (
    DetectorConfig, detect_and_correct_artifacts,
    correct_detector_nonlinearity
)
from .ion_count_processing import apply_arcsinh_transform
from .quality_control import (
    monitor_calibration_channels, check_background_levels,
    detect_spatial_artifacts
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingleStainProtocol:
    """Standardized protocol for single-antibody staining."""
    antibody_name: str                  # Primary antibody target
    metal_tag: str                      # Metal isotope tag (e.g., '141Pr')
    expected_channels: List[str]        # Expected spillover channels
    staining_conditions: Dict[str, Any] # Staining protocol parameters
    acquisition_settings: Dict[str, Any] # Instrument acquisition settings
    quality_thresholds: Dict[str, float] # QC thresholds for this antibody
    
    def __post_init__(self):
        if not self.antibody_name:
            raise ValueError("Antibody name cannot be empty")
        if not self.metal_tag:
            raise ValueError("Metal tag cannot be empty")
        if not self.expected_channels:
            raise ValueError("Expected channels list cannot be empty")


@dataclass(frozen=True)
class SpilloverEstimationResult:
    """Results from single-stain spillover estimation."""
    spillover_matrix: SpilloverMatrix       # Estimated spillover matrix
    single_stain_data: Dict[str, Any]      # Processed single-stain measurements
    estimation_quality: Dict[str, Any]     # Quality metrics for estimation
    protocol_compliance: Dict[str, bool]   # Protocol adherence check results
    uncertainty_assessment: UncertaintyMap # Uncertainty in spillover estimates
    recommendations: List[str]             # Recommendations for improvement


class SingleStainProtocolError(Exception):
    """Exception raised for single-stain protocol failures."""
    pass


def create_standard_protocols() -> Dict[str, SingleStainProtocol]:
    """
    Create standard single-stain protocols for common IMC antibodies.
    
    Returns:
        Dictionary mapping antibody names to their protocols
    """
    protocols = {}
    
    # Common IMC antibodies with their expected spillover patterns
    standard_antibodies = [
        {
            'antibody': 'CD45', 'metal': '89Y',
            'spillover_channels': ['88Sr', '90Zr'],
            'staining': {'concentration_ug_ml': 1.0, 'incubation_time_min': 30, 'temperature_c': 4},
            'acquisition': {'laser_power_percent': 100, 'acquisition_time_ms': 1000},
            'quality': {'min_signal_intensity': 50, 'max_cv_percent': 20, 'min_specificity': 0.8}
        },
        {
            'antibody': 'CD3', 'metal': '170Er',
            'spillover_channels': ['169Tm', '171Yb'],
            'staining': {'concentration_ug_ml': 0.5, 'incubation_time_min': 30, 'temperature_c': 4},
            'acquisition': {'laser_power_percent': 100, 'acquisition_time_ms': 1000},
            'quality': {'min_signal_intensity': 30, 'max_cv_percent': 25, 'min_specificity': 0.75}
        },
        {
            'antibody': 'CD20', 'metal': '147Sm',
            'spillover_channels': ['146Nd', '148Nd'],
            'staining': {'concentration_ug_ml': 1.0, 'incubation_time_min': 30, 'temperature_c': 4},
            'acquisition': {'laser_power_percent': 100, 'acquisition_time_ms': 1000},
            'quality': {'min_signal_intensity': 40, 'max_cv_percent': 20, 'min_specificity': 0.8}
        },
        {
            'antibody': 'CD68', 'metal': '168Er',
            'spillover_channels': ['167Er', '169Tm'],
            'staining': {'concentration_ug_ml': 1.5, 'incubation_time_min': 45, 'temperature_c': 4},
            'acquisition': {'laser_power_percent': 100, 'acquisition_time_ms': 1000},
            'quality': {'min_signal_intensity': 35, 'max_cv_percent': 25, 'min_specificity': 0.75}
        },
        {
            'antibody': 'Vimentin', 'metal': '142Nd',
            'spillover_channels': ['141Pr', '143Nd'],
            'staining': {'concentration_ug_ml': 2.0, 'incubation_time_min': 60, 'temperature_c': 4},
            'acquisition': {'laser_power_percent': 100, 'acquisition_time_ms': 1000},
            'quality': {'min_signal_intensity': 60, 'max_cv_percent': 15, 'min_specificity': 0.85}
        },
        {
            'antibody': 'PanCK', 'metal': '159Tb',
            'spillover_channels': ['158Gd', '160Gd'],
            'staining': {'concentration_ug_ml': 1.0, 'incubation_time_min': 30, 'temperature_c': 4},
            'acquisition': {'laser_power_percent': 100, 'acquisition_time_ms': 1000},
            'quality': {'min_signal_intensity': 80, 'max_cv_percent': 15, 'min_specificity': 0.9}
        }
    ]
    
    for antibody_data in standard_antibodies:
        protocol = SingleStainProtocol(
            antibody_name=antibody_data['antibody'],
            metal_tag=antibody_data['metal'],
            expected_channels=antibody_data['spillover_channels'],
            staining_conditions=antibody_data['staining'],
            acquisition_settings=antibody_data['acquisition'],
            quality_thresholds=antibody_data['quality']
        )
        protocols[antibody_data['antibody']] = protocol
    
    logger.info(f"Created {len(protocols)} standard single-stain protocols")
    return protocols


def validate_single_stain_data(
    single_stain_measurements: Dict[str, Dict[str, np.ndarray]],
    protocols: Dict[str, SingleStainProtocol],
    detector_config: Optional[DetectorConfig] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Validate single-stain measurement data against protocols.
    
    Args:
        single_stain_measurements: Dict mapping antibody -> channel -> measurements
        protocols: Single-stain protocols for validation
        detector_config: Detector configuration for physics validation
        
    Returns:
        Dictionary with validation results for each antibody
    """
    logger.debug("Validating single-stain measurement data")
    
    validation_results = {}
    
    for antibody_name, measurements in single_stain_measurements.items():
        
        validation = {
            'protocol_found': antibody_name in protocols,
            'data_quality': {},
            'protocol_compliance': {},
            'physics_validation': {},
            'recommendations': []
        }
        
        if antibody_name not in protocols:
            validation['recommendations'].append(f"No standard protocol found for {antibody_name}")
            validation_results[antibody_name] = validation
            continue
        
        protocol = protocols[antibody_name]
        
        # Validate data quality
        primary_channel = protocol.metal_tag
        if primary_channel in measurements:
            primary_signal = measurements[primary_channel]
            
            # Signal intensity validation
            mean_intensity = np.mean(primary_signal)
            cv_percent = (np.std(primary_signal) / mean_intensity) * 100 if mean_intensity > 0 else np.inf
            
            min_intensity = protocol.quality_thresholds['min_signal_intensity']
            max_cv = protocol.quality_thresholds['max_cv_percent']
            
            validation['data_quality'] = {
                'mean_intensity': mean_intensity,
                'cv_percent': cv_percent,
                'meets_intensity_threshold': mean_intensity >= min_intensity,
                'meets_cv_threshold': cv_percent <= max_cv,
                'n_measurements': len(primary_signal)
            }
            
            # Protocol compliance
            validation['protocol_compliance'] = {
                'primary_channel_present': True,
                'expected_channels_present': all(
                    ch in measurements for ch in protocol.expected_channels
                ),
                'signal_adequacy': mean_intensity >= min_intensity,
                'signal_consistency': cv_percent <= max_cv
            }
            
            # Physics validation using detector config
            if detector_config:
                # Check for saturation
                saturation_level = detector_config.saturation_level
                max_signal = np.max(primary_signal)
                saturation_fraction = np.sum(primary_signal >= 0.95 * saturation_level) / len(primary_signal)
                
                # Check for deadtime effects
                deadtime_ns = detector_config.deadtime_ns
                count_rate = mean_intensity / 1000.0  # Assume 1s acquisition time
                deadtime_loss = count_rate * deadtime_ns * 1e-9
                
                validation['physics_validation'] = {
                    'max_signal': max_signal,
                    'saturation_fraction': saturation_fraction,
                    'potential_deadtime_loss': deadtime_loss,
                    'excessive_saturation': saturation_fraction > 0.01,
                    'significant_deadtime': deadtime_loss > 0.05
                }
                
                if validation['physics_validation']['excessive_saturation']:
                    validation['recommendations'].append("Reduce antibody concentration - saturation detected")
                if validation['physics_validation']['significant_deadtime']:
                    validation['recommendations'].append("Consider shorter acquisition time - deadtime losses detected")
            
            # Generate recommendations
            if not validation['data_quality']['meets_intensity_threshold']:
                validation['recommendations'].append("Increase antibody concentration or acquisition time")
            if not validation['data_quality']['meets_cv_threshold']:
                validation['recommendations'].append("Improve staining uniformity - high CV detected")
            if not validation['protocol_compliance']['expected_channels_present']:
                missing_channels = [ch for ch in protocol.expected_channels if ch not in measurements]
                validation['recommendations'].append(f"Missing expected spillover channels: {missing_channels}")
                
        else:
            validation['data_quality'] = {'error': f"Primary channel {primary_channel} not found"}
            validation['protocol_compliance'] = {'primary_channel_present': False}
            validation['recommendations'].append(f"Primary channel {primary_channel} missing from measurements")
        
        validation_results[antibody_name] = validation
    
    return validation_results


def estimate_spillover_from_single_stains(
    single_stain_measurements: Dict[str, Dict[str, np.ndarray]],
    protocols: Optional[Dict[str, SingleStainProtocol]] = None,
    estimation_method: Literal['nnls', 'admm', 'lstsq'] = 'nnls',
    uncertainty_config: Optional[UncertaintyConfig] = None,
    quality_filters: Optional[Dict[str, Any]] = None
) -> SpilloverEstimationResult:
    """
    Estimate spillover matrix from single-stain control measurements.
    
    Args:
        single_stain_measurements: Dict mapping antibody -> channel -> measurements
        protocols: Optional single-stain protocols for validation
        estimation_method: Method for spillover matrix estimation
        uncertainty_config: Configuration for uncertainty estimation
        quality_filters: Quality filtering parameters
        
    Returns:
        SpilloverEstimationResult with complete analysis
    """
    logger.info(f"Estimating spillover matrix from {len(single_stain_measurements)} single-stain controls")
    
    if not single_stain_measurements:
        raise SingleStainProtocolError("No single-stain measurements provided")
    
    # Initialize protocols if not provided
    if protocols is None:
        protocols = create_standard_protocols()
    
    # Initialize quality filters
    if quality_filters is None:
        quality_filters = {
            'min_signal_threshold': 10.0,
            'max_cv_threshold': 0.5,
            'min_correlation_threshold': 0.3
        }
    
    # Initialize uncertainty config
    if uncertainty_config is None:
        uncertainty_config = UncertaintyConfig()
    
    # Step 1: Validate single-stain data
    validation_results = validate_single_stain_data(
        single_stain_measurements, protocols
    )
    
    # Step 2: Apply quality filters and preprocessing
    filtered_measurements, preprocessing_report = _preprocess_single_stain_data(
        single_stain_measurements, quality_filters
    )
    
    # Step 3: Estimate spillover matrix using existing infrastructure
    try:
        spillover_matrix = estimate_spillover_matrix(
            filtered_measurements,
            method=estimation_method,
            bootstrap_samples=uncertainty_config.bootstrap_samples,
            min_signal_threshold=quality_filters['min_signal_threshold']
        )
    except SpilloverCorrectionError as e:
        raise SingleStainProtocolError(f"Spillover matrix estimation failed: {e}")
    
    # Step 4: Create uncertainty assessment
    uncertainty_map = _create_spillover_uncertainty_map(
        spillover_matrix, filtered_measurements, uncertainty_config
    )
    
    # Step 5: Assess estimation quality
    estimation_quality = _assess_spillover_estimation_quality(
        spillover_matrix, filtered_measurements, protocols
    )
    
    # Step 6: Check protocol compliance
    protocol_compliance = _check_protocol_compliance(
        validation_results, spillover_matrix
    )
    
    # Step 7: Generate recommendations
    recommendations = _generate_spillover_recommendations(
        estimation_quality, protocol_compliance, validation_results
    )
    
    logger.info(f"Spillover matrix estimation completed with {len(recommendations)} recommendations")
    
    return SpilloverEstimationResult(
        spillover_matrix=spillover_matrix,
        single_stain_data={
            'raw_measurements': single_stain_measurements,
            'filtered_measurements': filtered_measurements,
            'preprocessing_report': preprocessing_report,
            'validation_results': validation_results
        },
        estimation_quality=estimation_quality,
        protocol_compliance=protocol_compliance,
        uncertainty_assessment=uncertainty_map,
        recommendations=recommendations
    )


def optimize_single_stain_protocols(
    measurement_history: Dict[str, List[Dict[str, np.ndarray]]],
    performance_metrics: Dict[str, Dict[str, float]],
    optimization_target: str = 'spillover_accuracy'
) -> Dict[str, SingleStainProtocol]:
    """
    Optimize single-stain protocols based on historical measurement performance.
    
    Args:
        measurement_history: Historical single-stain measurements by antibody
        performance_metrics: Performance metrics for each antibody
        optimization_target: Target metric for optimization
        
    Returns:
        Dictionary of optimized protocols
    """
    logger.info(f"Optimizing single-stain protocols for {len(measurement_history)} antibodies")
    
    # Start with standard protocols
    optimized_protocols = create_standard_protocols()
    
    for antibody_name, measurements_list in measurement_history.items():
        
        if antibody_name not in optimized_protocols:
            logger.warning(f"No standard protocol found for {antibody_name}, skipping optimization")
            continue
        
        current_protocol = optimized_protocols[antibody_name]
        performance = performance_metrics.get(antibody_name, {})
        
        # Analyze measurement consistency
        cv_values = []
        intensity_values = []
        
        for measurements in measurements_list:
            primary_channel = current_protocol.metal_tag
            if primary_channel in measurements:
                signal = measurements[primary_channel]
                mean_intensity = np.mean(signal)
                cv = np.std(signal) / mean_intensity if mean_intensity > 0 else np.inf
                
                cv_values.append(cv)
                intensity_values.append(mean_intensity)
        
        if cv_values and intensity_values:
            # Calculate optimization metrics
            mean_cv = np.mean(cv_values)
            mean_intensity = np.mean(intensity_values)
            
            # Optimize thresholds based on observed performance
            new_quality_thresholds = current_protocol.quality_thresholds.copy()
            
            # Adjust CV threshold based on observed variability
            observed_cv_95th = np.percentile(cv_values, 95) * 100  # Convert to percentage
            new_quality_thresholds['max_cv_percent'] = min(
                observed_cv_95th * 1.2,  # 20% margin above 95th percentile
                current_protocol.quality_thresholds['max_cv_percent'] * 1.5  # Don't exceed 50% increase
            )
            
            # Adjust intensity threshold based on observed levels
            observed_intensity_5th = np.percentile(intensity_values, 5)
            new_quality_thresholds['min_signal_intensity'] = max(
                observed_intensity_5th * 0.8,  # 20% below 5th percentile
                current_protocol.quality_thresholds['min_signal_intensity'] * 0.5  # Don't go below 50% of original
            )
            
            # Update specificity if performance data available
            if 'specificity' in performance:
                new_quality_thresholds['min_specificity'] = max(
                    performance['specificity'] * 0.9,  # 10% below observed specificity
                    0.5  # Minimum acceptable specificity
                )
            
            # Create optimized protocol
            optimized_protocol = SingleStainProtocol(
                antibody_name=current_protocol.antibody_name,
                metal_tag=current_protocol.metal_tag,
                expected_channels=current_protocol.expected_channels,
                staining_conditions=current_protocol.staining_conditions,
                acquisition_settings=current_protocol.acquisition_settings,
                quality_thresholds=new_quality_thresholds
            )
            
            optimized_protocols[antibody_name] = optimized_protocol
            
            logger.debug(f"Optimized protocol for {antibody_name}: "
                        f"CV threshold {current_protocol.quality_thresholds['max_cv_percent']:.1f} → "
                        f"{new_quality_thresholds['max_cv_percent']:.1f}%, "
                        f"Intensity threshold {current_protocol.quality_thresholds['min_signal_intensity']:.1f} → "
                        f"{new_quality_thresholds['min_signal_intensity']:.1f}")
    
    return optimized_protocols


def create_spillover_correction_pipeline(
    single_stain_result: SpilloverEstimationResult,
    ion_count_data: Dict[str, np.ndarray],
    apply_artifact_correction: bool = True,
    apply_uncertainty_propagation: bool = True
) -> Dict[str, Any]:
    """
    Create complete spillover correction pipeline using estimated matrix.
    
    Args:
        single_stain_result: Results from spillover estimation
        ion_count_data: Ion count data to correct
        apply_artifact_correction: Whether to apply artifact correction
        apply_uncertainty_propagation: Whether to propagate uncertainties
        
    Returns:
        Dictionary with corrected data and quality metrics
    """
    logger.info("Creating spillover correction pipeline from single-stain data")
    
    pipeline_results = {
        'original_data': ion_count_data.copy(),
        'processing_steps': [],
        'quality_metrics': {},
        'final_uncertainty': None
    }
    
    # Step 1: Apply artifact correction if requested
    corrected_data = ion_count_data.copy()
    artifact_uncertainty = {}
    
    if apply_artifact_correction:
        logger.debug("Applying artifact correction")
        
        try:
            corrected_data, artifact_uncertainties, artifact_metadata = detect_and_correct_artifacts(
                corrected_data,
                detector_config=DetectorConfig(),  # Use default detector config
                hot_pixel_threshold=5.0
            )
            
            artifact_uncertainty = artifact_uncertainties
            pipeline_results['processing_steps'].append('artifact_correction')
            pipeline_results['quality_metrics']['artifact_correction'] = artifact_metadata
            
        except Exception as e:
            logger.warning(f"Artifact correction failed: {e}")
            pipeline_results['processing_steps'].append('artifact_correction_failed')
    
    # Step 2: Apply spillover correction
    logger.debug("Applying spillover correction")
    
    try:
        from .spillover_correction import correct_spillover
        
        spillover_corrected, spillover_uncertainty = correct_spillover(
            corrected_data,
            single_stain_result.spillover_matrix,
            apply_positivity_constraint=True
        )
        
        pipeline_results['spillover_corrected'] = spillover_corrected
        pipeline_results['processing_steps'].append('spillover_correction')
        
        # Validate spillover correction
        spillover_validation = validate_spillover_correction(
            single_stain_result.spillover_matrix
        )
        pipeline_results['quality_metrics']['spillover_validation'] = spillover_validation
        
    except SpilloverCorrectionError as e:
        raise SingleStainProtocolError(f"Spillover correction failed: {e}")
    
    # Step 3: Propagate uncertainties if requested
    if apply_uncertainty_propagation:
        logger.debug("Propagating uncertainties through correction pipeline")
        
        try:
            # Create base uncertainty from original data
            base_uncertainty = create_base_uncertainty(
                ion_count_data,
                uncertainty_type='poisson'
            )
            
            # Propagate through artifact correction if applied
            if apply_artifact_correction and artifact_uncertainty:
                # Convert artifact uncertainties to proper format
                artifact_interpolation_uncertainty = np.ones_like(
                    next(iter(artifact_uncertainty.values()))
                )
                
                artifact_corrected_uncertainty = propagate_through_artifact_correction(
                    base_uncertainty,
                    artifact_mask=np.zeros_like(artifact_interpolation_uncertainty, dtype=bool),  # No mask available
                    interpolation_uncertainty=artifact_interpolation_uncertainty,
                    correction_metadata={'corrections_applied': ['hot_pixels']}
                )
            else:
                artifact_corrected_uncertainty = base_uncertainty
            
            # Propagate through spillover correction
            final_uncertainty = propagate_through_spillover_correction(
                artifact_corrected_uncertainty,
                single_stain_result.spillover_matrix.matrix,
                single_stain_result.spillover_matrix.uncertainty
            )
            
            pipeline_results['final_uncertainty'] = final_uncertainty
            pipeline_results['processing_steps'].append('uncertainty_propagation')
            
        except UncertaintyPropagationError as e:
            logger.warning(f"Uncertainty propagation failed: {e}")
            pipeline_results['processing_steps'].append('uncertainty_propagation_failed')
    
    # Step 4: Final quality assessment
    final_quality = _assess_correction_quality(
        pipeline_results['original_data'],
        spillover_corrected,
        single_stain_result
    )
    pipeline_results['quality_metrics']['final_assessment'] = final_quality
    
    logger.info(f"Spillover correction pipeline completed with {len(pipeline_results['processing_steps'])} steps")
    
    return pipeline_results


def save_single_stain_analysis(
    result: SpilloverEstimationResult,
    output_path: Union[str, Path],
    include_raw_data: bool = False
) -> str:
    """
    Save single-stain analysis results to file.
    
    Args:
        result: SpilloverEstimationResult to save
        output_path: Output file path
        include_raw_data: Whether to include raw measurement arrays
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for serialization
    save_data = {
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'estimation_method': result.spillover_matrix.method,
            'n_antibodies': len(result.single_stain_data['raw_measurements']),
            'channels': result.spillover_matrix.channels
        },
        'spillover_matrix': {
            'matrix': result.spillover_matrix.matrix.tolist(),
            'uncertainty': result.spillover_matrix.uncertainty.tolist(),
            'method': result.spillover_matrix.method,
            'channels': result.spillover_matrix.channels,
            'metadata': result.spillover_matrix.metadata
        },
        'estimation_quality': result.estimation_quality,
        'protocol_compliance': result.protocol_compliance,
        'recommendations': result.recommendations,
        'uncertainty_summary': {
            'uncertainty_type': result.uncertainty_assessment.uncertainty_type,
            'sources': result.uncertainty_assessment.sources,
            'spatial_shape': result.uncertainty_assessment.spatial_shape,
            'metadata': result.uncertainty_assessment.metadata
        }
    }
    
    # Include raw data if requested
    if include_raw_data:
        save_data['single_stain_data'] = {}
        for antibody, measurements in result.single_stain_data['raw_measurements'].items():
            save_data['single_stain_data'][antibody] = {}
            for channel, data in measurements.items():
                save_data['single_stain_data'][antibody][channel] = data.tolist()
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    logger.info(f"Single-stain analysis results saved to {output_path}")
    return str(output_path)


def load_single_stain_analysis(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load single-stain analysis results from file.
    
    Args:
        file_path: Path to saved analysis file
        
    Returns:
        Dictionary with loaded analysis results
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Single-stain analysis file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        save_data = json.load(f)
    
    # Reconstruct spillover matrix
    matrix_data = save_data['spillover_matrix']
    spillover_matrix = SpilloverMatrix(
        matrix=np.array(matrix_data['matrix']),
        uncertainty=np.array(matrix_data['uncertainty']),
        method=matrix_data['method'],
        channels=matrix_data['channels'],
        metadata=matrix_data['metadata']
    )
    
    result = {
        'spillover_matrix': spillover_matrix,
        'estimation_quality': save_data['estimation_quality'],
        'protocol_compliance': save_data['protocol_compliance'],
        'recommendations': save_data['recommendations'],
        'analysis_metadata': save_data['analysis_metadata']
    }
    
    # Load raw data if available
    if 'single_stain_data' in save_data:
        result['single_stain_data'] = {}
        for antibody, measurements in save_data['single_stain_data'].items():
            result['single_stain_data'][antibody] = {}
            for channel, data in measurements.items():
                result['single_stain_data'][antibody][channel] = np.array(data)
    
    logger.info(f"Single-stain analysis results loaded from {file_path}")
    return result


# Helper functions

def _preprocess_single_stain_data(
    measurements: Dict[str, Dict[str, np.ndarray]],
    quality_filters: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    """Preprocess single-stain data with quality filtering."""
    
    filtered_measurements = {}
    preprocessing_report = {
        'original_antibodies': len(measurements),
        'filtered_antibodies': 0,
        'excluded_antibodies': [],
        'channel_statistics': {}
    }
    
    min_signal = quality_filters['min_signal_threshold']
    max_cv = quality_filters['max_cv_threshold']
    
    for antibody, channels in measurements.items():
        antibody_passed = True
        antibody_stats = {}
        
        for channel, data in channels.items():
            # Calculate basic statistics
            mean_signal = np.mean(data)
            cv = np.std(data) / mean_signal if mean_signal > 0 else np.inf
            
            antibody_stats[channel] = {
                'mean': mean_signal,
                'cv': cv,
                'n_points': len(data)
            }
            
            # Check quality filters
            if mean_signal < min_signal or cv > max_cv:
                antibody_passed = False
        
        if antibody_passed:
            filtered_measurements[antibody] = channels
            preprocessing_report['filtered_antibodies'] += 1
        else:
            preprocessing_report['excluded_antibodies'].append(antibody)
        
        preprocessing_report['channel_statistics'][antibody] = antibody_stats
    
    return filtered_measurements, preprocessing_report


def _create_spillover_uncertainty_map(
    spillover_matrix: SpilloverMatrix,
    measurements: Dict[str, Dict[str, np.ndarray]],
    uncertainty_config: UncertaintyConfig
) -> UncertaintyMap:
    """Create uncertainty map for spillover estimates."""
    
    # Use the uncertainty from the spillover matrix
    n_channels = len(spillover_matrix.channels)
    
    # Create spatial uncertainty map (simplified - assume single spatial location)
    spatial_shape = (1, 1)
    uncertainties = spillover_matrix.uncertainty.reshape(n_channels, -1)
    
    return UncertaintyMap(
        uncertainties=uncertainties,
        uncertainty_type='absolute',
        channels=spillover_matrix.channels,
        spatial_shape=spatial_shape,
        sources=['single_stain_estimation', 'bootstrap_sampling'],
        metadata={
            'estimation_method': spillover_matrix.method,
            'n_bootstrap_samples': spillover_matrix.metadata.get('n_bootstrap_samples', 0),
            'condition_number': spillover_matrix.metadata.get('condition_number', np.inf)
        }
    )


def _assess_spillover_estimation_quality(
    spillover_matrix: SpilloverMatrix,
    measurements: Dict[str, Dict[str, np.ndarray]],
    protocols: Dict[str, SingleStainProtocol]
) -> Dict[str, Any]:
    """Assess quality of spillover matrix estimation."""
    
    quality_assessment = {
        'matrix_properties': {},
        'estimation_reliability': {},
        'protocol_agreement': {},
        'overall_quality': 'unknown'
    }
    
    # Matrix properties
    matrix = spillover_matrix.matrix
    condition_number = spillover_matrix.metadata['condition_number']
    
    quality_assessment['matrix_properties'] = {
        'condition_number': condition_number,
        'is_well_conditioned': condition_number < 1e10,
        'diagonal_dominance': np.all(np.diag(matrix) >= np.sum(matrix, axis=1) - np.diag(matrix)),
        'max_off_diagonal': np.max(matrix - np.diag(np.diag(matrix))),
        'symmetry_score': np.mean(np.abs(matrix - matrix.T))
    }
    
    # Estimation reliability
    n_samples = spillover_matrix.metadata.get('n_single_stains', 0)
    fit_residuals = spillover_matrix.metadata.get('fit_residuals', [])
    
    quality_assessment['estimation_reliability'] = {
        'n_single_stains': n_samples,
        'sufficient_data': n_samples >= len(spillover_matrix.channels),
        'mean_fit_residual': np.mean(fit_residuals) if fit_residuals else np.inf,
        'max_fit_residual': np.max(fit_residuals) if fit_residuals else np.inf
    }
    
    # Protocol agreement
    protocol_agreement_scores = []
    for antibody, channels in measurements.items():
        if antibody in protocols:
            protocol = protocols[antibody]
            expected_channels = set(protocol.expected_channels + [protocol.metal_tag])
            measured_channels = set(channels.keys())
            
            agreement = len(expected_channels & measured_channels) / len(expected_channels)
            protocol_agreement_scores.append(agreement)
    
    quality_assessment['protocol_agreement'] = {
        'mean_agreement': np.mean(protocol_agreement_scores) if protocol_agreement_scores else 0,
        'min_agreement': np.min(protocol_agreement_scores) if protocol_agreement_scores else 0,
        'n_protocols_checked': len(protocol_agreement_scores)
    }
    
    # Overall quality assessment
    quality_score = 0
    
    if quality_assessment['matrix_properties']['is_well_conditioned']:
        quality_score += 25
    if quality_assessment['estimation_reliability']['sufficient_data']:
        quality_score += 25
    if quality_assessment['estimation_reliability']['mean_fit_residual'] < 0.1:
        quality_score += 25
    if quality_assessment['protocol_agreement']['mean_agreement'] > 0.8:
        quality_score += 25
    
    if quality_score >= 75:
        quality_assessment['overall_quality'] = 'excellent'
    elif quality_score >= 50:
        quality_assessment['overall_quality'] = 'good'
    elif quality_score >= 25:
        quality_assessment['overall_quality'] = 'acceptable'
    else:
        quality_assessment['overall_quality'] = 'poor'
    
    return quality_assessment


def _check_protocol_compliance(
    validation_results: Dict[str, Dict[str, Any]],
    spillover_matrix: SpilloverMatrix
) -> Dict[str, bool]:
    """Check protocol compliance across all antibodies."""
    
    compliance = {
        'all_protocols_found': True,
        'all_data_quality_met': True,
        'all_physics_validated': True,
        'sufficient_coverage': False
    }
    
    n_compliant = 0
    n_total = len(validation_results)
    
    for antibody, validation in validation_results.items():
        if not validation.get('protocol_found', False):
            compliance['all_protocols_found'] = False
        
        data_quality = validation.get('data_quality', {})
        if not (data_quality.get('meets_intensity_threshold', False) and 
                data_quality.get('meets_cv_threshold', False)):
            compliance['all_data_quality_met'] = False
        else:
            n_compliant += 1
        
        physics = validation.get('physics_validation', {})
        if (physics.get('excessive_saturation', False) or 
            physics.get('significant_deadtime', False)):
            compliance['all_physics_validated'] = False
    
    # Check if we have sufficient channel coverage
    compliance['sufficient_coverage'] = n_compliant >= len(spillover_matrix.channels) * 0.8
    
    return compliance


def _generate_spillover_recommendations(
    estimation_quality: Dict[str, Any],
    protocol_compliance: Dict[str, bool],
    validation_results: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Generate recommendations for improving spillover estimation."""
    
    recommendations = []
    
    # Matrix quality recommendations
    matrix_props = estimation_quality.get('matrix_properties', {})
    if not matrix_props.get('is_well_conditioned', True):
        recommendations.append("Matrix is ill-conditioned - consider additional single-stain controls")
    
    if matrix_props.get('max_off_diagonal', 0) > 0.5:
        recommendations.append("High spillover detected - validate antibody specificity")
    
    # Data quality recommendations
    if not protocol_compliance.get('all_data_quality_met', True):
        recommendations.append("Some antibodies failed quality thresholds - review concentrations and staining conditions")
    
    # Coverage recommendations
    if not protocol_compliance.get('sufficient_coverage', True):
        recommendations.append("Insufficient channel coverage - add more single-stain controls")
    
    # Physics recommendations
    if not protocol_compliance.get('all_physics_validated', True):
        recommendations.append("Physics validation failed - check for saturation or deadtime effects")
    
    # Protocol-specific recommendations
    for antibody, validation in validation_results.items():
        antibody_recommendations = validation.get('recommendations', [])
        for rec in antibody_recommendations:
            full_rec = f"{antibody}: {rec}"
            if full_rec not in recommendations:
                recommendations.append(full_rec)
    
    # Overall quality recommendations
    overall_quality = estimation_quality.get('overall_quality', 'unknown')
    if overall_quality == 'poor':
        recommendations.append("Overall estimation quality is poor - consider repeating single-stain experiments")
    elif overall_quality == 'acceptable':
        recommendations.append("Estimation quality is acceptable but could be improved")
    
    return recommendations


def _assess_correction_quality(
    original_data: Dict[str, np.ndarray],
    corrected_data: Dict[str, np.ndarray],
    single_stain_result: SpilloverEstimationResult
) -> Dict[str, Any]:
    """Assess quality of spillover correction."""
    
    quality_metrics = {
        'correction_magnitude': {},
        'residual_correlation': {},
        'signal_preservation': {}
    }
    
    for channel in original_data:
        if channel in corrected_data:
            original = original_data[channel]
            corrected = corrected_data[channel]
            
            # Correction magnitude
            correction_factor = np.mean(corrected) / np.mean(original) if np.mean(original) > 0 else 1.0
            max_change = np.max(np.abs(corrected - original))
            
            quality_metrics['correction_magnitude'][channel] = {
                'mean_correction_factor': correction_factor,
                'max_absolute_change': max_change,
                'significant_correction': abs(correction_factor - 1.0) > 0.1
            }
            
            # Signal preservation
            original_positive = np.sum(original > 0)
            corrected_positive = np.sum(corrected > 0)
            signal_preservation = corrected_positive / original_positive if original_positive > 0 else 0
            
            quality_metrics['signal_preservation'][channel] = {
                'fraction_preserved': signal_preservation,
                'adequate_preservation': signal_preservation > 0.9
            }
    
    return quality_metrics