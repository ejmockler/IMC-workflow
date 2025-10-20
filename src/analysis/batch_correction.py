"""
Batch Correction Module for IMC Analysis Pipeline.

Production-quality batch correction methods for addressing systematic differences
between batches while preserving biological signal in ion count data.

Key Features:
- Sham-anchored normalization preserving biological dynamics
- Scientific guardrails preventing invalid method combinations  
- Bootstrap confidence intervals for n=2 replicate analysis  
- Comprehensive batch effect detection and validation
- Memory-efficient implementation for large IMC datasets
- Robust error handling and configuration validation

WARNING: Quantile normalization is SCIENTIFICALLY INVALID for cross-sectional 
time-course studies as it destroys biological signal by forcing all timepoint 
distributions to be identical. Use sham_anchored_normalize() instead.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats
from scipy.interpolate import interp1d
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for scientifically invalid method combinations."""
    pass


def _validate_experimental_design(batch_metadata: Dict[str, Dict[str, any]], 
                                   method: str = 'quantile') -> None:
    """
    Scientific guardrails: Validate method appropriateness for experimental design.
    
    Args:
        batch_metadata: Dictionary mapping batch_id -> metadata_dict
        method: Normalization method being applied
        
    Raises:
        ValidationError: If method is inappropriate for experimental design
    """
    # Check if this is a cross-sectional time-course study
    timepoints = set()
    for batch_id, metadata in batch_metadata.items():
        if 'timepoint' in metadata or 'Injury Day' in metadata:
            timepoint = metadata.get('timepoint', metadata.get('Injury Day'))
            timepoints.add(timepoint)
    
    # If multiple timepoints detected and quantile normalization requested
    if len(timepoints) > 1 and method == 'quantile':
        raise ValidationError(
            f"CRITICAL ERROR: Quantile normalization is scientifically invalid for "
            f"cross-sectional time-course studies with {len(timepoints)} timepoints. "
            f"This method forces all timepoint distributions to be identical, thereby "
            f"DESTROYING the biological signal of temporal dynamics. "
            f"Use 'sham_anchored' normalization instead to preserve biological variation "
            f"while correcting for technical batch effects."
        )


def sham_anchored_normalize(
    batch_data: Dict[str, Dict[str, np.ndarray]],
    batch_metadata: Dict[str, Dict[str, any]],
    sham_condition: str = 'Sham',
    sham_timepoint: Union[int, str] = 0
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, any]]:
    """
    Perform sham-anchored normalization preserving biological dynamics.
    
    This method normalizes all data relative to sham control statistics,
    preserving biological changes across timepoints while correcting for
    technical batch effects.
    
    Args:
        batch_data: Dictionary mapping batch_id -> protein_name -> ion_counts
        batch_metadata: Dictionary mapping batch_id -> metadata_dict  
        sham_condition: Name of sham/control condition
        sham_timepoint: Timepoint value for sham controls
        
    Returns:
        Tuple of (normalized_data, normalization_statistics)
    """
    # Validate inputs
    validate_batch_structure(batch_data, batch_metadata)
    
    logger.info(f"Starting sham-anchored normalization for {len(batch_data)} batches")
    
    # Find sham control batches
    sham_batches = []
    for batch_id, metadata in batch_metadata.items():
        condition = metadata.get('condition', metadata.get('Condition', 'Unknown'))
        timepoint = metadata.get('timepoint', metadata.get('Injury Day', None))
        
        if (condition == sham_condition and 
            (timepoint == sham_timepoint or str(timepoint) == str(sham_timepoint))):
            sham_batches.append(batch_id)
    
    if not sham_batches:
        raise ValueError(f"No sham control batches found with condition='{sham_condition}' "
                        f"and timepoint='{sham_timepoint}'")
    
    logger.info(f"Found {len(sham_batches)} sham control batches: {sham_batches}")
    
    # Compute reference statistics from sham controls
    sham_stats = _compute_sham_reference_stats(batch_data, sham_batches)
    
    # Apply normalization to all batches
    normalized_data = {}
    normalization_stats = {
        'sham_batches': sham_batches,
        'reference_stats': sham_stats,
        'per_batch_stats': {}
    }
    
    for batch_id, protein_data in batch_data.items():
        normalized_batch = {}
        batch_stats = {}
        
        for protein_name, ion_counts in protein_data.items():
            if protein_name in sham_stats:
                # Normalize: (data - sham_mean) / sham_std
                sham_mean = sham_stats[protein_name]['mean']
                sham_std = sham_stats[protein_name]['std']
                
                # Handle zero standard deviation case properly
                if sham_std > 1e-10:  # Use small epsilon for numerical stability
                    normalized_counts = (ion_counts - sham_mean) / sham_std
                else:
                    # When sham controls have no variation, check if data is constant
                    data_std = np.std(ion_counts)
                    if data_std < 1e-10:
                        # Both sham and data are constant - set to 0 (no signal)
                        logger.info(f"Protein {protein_name}: Both sham and sample data are constant. "
                                   f"Setting normalized values to 0 (no biological signal).")
                        normalized_counts = np.zeros_like(ion_counts)
                    else:
                        # Sham is constant but data varies - preserve as z-score relative to data
                        data_mean = np.mean(ion_counts)
                        logger.warning(f"Protein {protein_name}: Sham controls are constant (std={sham_std:.2e}) "
                                      f"but sample data varies (std={data_std:.3f}). "
                                      f"Using sample-based z-score normalization.")
                        normalized_counts = (ion_counts - data_mean) / data_std
                
                normalized_batch[protein_name] = normalized_counts
                
                # Track statistics
                batch_stats[protein_name] = {
                    'original_mean': np.mean(ion_counts),
                    'original_std': np.std(ion_counts),
                    'normalized_mean': np.mean(normalized_counts),
                    'normalized_std': np.std(normalized_counts),
                    'sham_reference_mean': sham_mean,
                    'sham_reference_std': sham_std
                }
            else:
                logger.warning(f"No sham reference for protein {protein_name}. Keeping original data.")
                normalized_batch[protein_name] = ion_counts.copy()
        
        normalized_data[batch_id] = normalized_batch
        normalization_stats['per_batch_stats'][batch_id] = batch_stats
    
    logger.info("Sham-anchored normalization completed successfully")
    return normalized_data, normalization_stats


def _compute_sham_reference_stats(batch_data: Dict[str, Dict[str, np.ndarray]], 
                                  sham_batches: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute reference statistics from sham control batches.
    
    Args:
        batch_data: Dictionary mapping batch_id -> protein_name -> ion_counts
        sham_batches: List of batch IDs containing sham controls
        
    Returns:
        Dictionary mapping protein_name -> {'mean': float, 'std': float}
    """
    sham_stats = {}
    
    # Get all protein names
    all_proteins = set()
    for batch_id in sham_batches:
        all_proteins.update(batch_data[batch_id].keys())
    
    # Compute pooled statistics across sham batches
    for protein_name in all_proteins:
        all_sham_counts = []
        
        for batch_id in sham_batches:
            if protein_name in batch_data[batch_id]:
                all_sham_counts.extend(batch_data[batch_id][protein_name])
        
        if all_sham_counts:
            all_sham_counts = np.array(all_sham_counts)
            sham_stats[protein_name] = {
                'mean': np.mean(all_sham_counts),
                'std': np.std(all_sham_counts, ddof=1),  # Sample std
                'n_observations': len(all_sham_counts)
            }
        else:
            logger.warning(f"No sham control data found for protein {protein_name}")
    
    return sham_stats


@dataclass
class BatchCorrectionConfig:
    """Configuration for batch normalization parameters."""
    sham_condition: str = 'Sham'
    sham_timepoint: Union[int, str] = 0
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    min_samples_per_batch: int = 100
    
    # Bead normalization parameters
    bead_channels: List[str] = None
    bead_signal_threshold: float = 100.0
    drift_correction_method: str = 'linear'  # 'linear', 'spline', 'robust'

    def __post_init__(self):
        # Bead channels must be provided from config - no hardcoded defaults
        # Config should specify channels in analysis.batch_correction.bead_normalization.bead_channels
        if self.bead_channels is None:
            self.bead_channels = []  # Empty list if not configured


# All deprecated quantile normalization methods have been removed.
# Use sham_anchored_normalize() for scientifically valid batch correction.


def detect_batch_effects(
    batch_data: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, any]:
    """
    Detect and quantify batch effects across datasets.
    
    Args:
        batch_data: Dictionary mapping batch_id -> protein_name -> ion_counts
        
    Returns:
        Dictionary containing batch effect analysis results
    """
    if len(batch_data) < 2:
        return {
            'protein_effects': {},
            'overall_severity': 0.0,
            'warning': 'Less than 2 batches provided'
        }
    
    # Get common proteins across all batches
    all_proteins = set()
    for batch_proteins in batch_data.values():
        all_proteins.update(batch_proteins.keys())
    
    common_proteins = all_proteins.copy()
    for batch_proteins in batch_data.values():
        common_proteins &= set(batch_proteins.keys())
    
    if not common_proteins:
        return {
            'protein_effects': {},
            'overall_severity': 0.0,
            'error': 'No common proteins across batches'
        }
    
    protein_effects = {}
    batch_ids = list(batch_data.keys())
    
    for protein in common_proteins:
        # Collect data from all batches
        protein_batch_data = []
        batch_labels = []
        
        for batch_id in batch_ids:
            protein_counts = batch_data[batch_id][protein]
            protein_batch_data.append(protein_counts)
            batch_labels.extend([batch_id] * len(protein_counts))
        
        # Combine all data
        combined_data = np.concatenate(protein_batch_data)
        
        # Statistical tests for batch effects
        if len(batch_ids) == 2:
            # Two-sample test
            stat, p_value = stats.mannwhitneyu(
                protein_batch_data[0], 
                protein_batch_data[1],
                alternative='two-sided'
            )
            test_name = 'mann_whitney_u'
        else:
            # Multi-batch test
            stat, p_value = stats.kruskal(*protein_batch_data)
            test_name = 'kruskal_wallis'
        
        # Compute effect size (coefficient of variation of batch medians)
        batch_medians = [np.median(data) for data in protein_batch_data]
        cv_medians = np.std(batch_medians) / (np.mean(batch_medians) + 1e-10)
        
        protein_effects[protein] = {
            f'{test_name}_statistic': float(stat),
            f'{test_name}_pvalue': float(p_value),
            'batch_medians': [float(m) for m in batch_medians],
            'coefficient_variation_medians': float(cv_medians),
            'effect_size': float(cv_medians)  # Use CV as effect size measure
        }
    
    # Overall severity score
    if protein_effects:
        effect_sizes = [effects['effect_size'] for effects in protein_effects.values()]
        overall_severity = float(np.mean(effect_sizes))
    else:
        overall_severity = 0.0
    
    return {
        'protein_effects': protein_effects,
        'overall_severity': overall_severity,
        'n_batches': len(batch_ids),
        'n_proteins': len(common_proteins)
    }


def validate_batch_structure(
    batch_data: Dict[str, Dict[str, np.ndarray]],
    batch_metadata: Dict[str, Dict[str, any]]
) -> None:
    """
    Validate batch data structure and consistency.
    
    Args:
        batch_data: Dictionary mapping batch_id -> protein_name -> ion_counts
        batch_metadata: Dictionary mapping batch_id -> metadata_dict
    
    Raises:
        ValueError: If structure is invalid
    """
    if not batch_data:
        raise ValueError("Empty batch data provided")
    
    if not batch_metadata:
        raise ValueError("Missing metadata for batches")
    
    # Check that all batches have metadata
    missing_metadata = set(batch_data.keys()) - set(batch_metadata.keys())
    if missing_metadata:
        raise ValueError(f"Missing metadata for batches: {missing_metadata}")
    
    # Check protein consistency across batches
    all_protein_sets = [set(proteins.keys()) for proteins in batch_data.values()]
    
    if len(set(frozenset(s) for s in all_protein_sets)) > 1:
        # Find common proteins
        common_proteins = set.intersection(*all_protein_sets)
        if len(common_proteins) == 0:
            raise ValueError("No common proteins found across batches")
        
        warnings.warn(
            f"Inconsistent protein sets across batches. "
            f"Will proceed with {len(common_proteins)} common proteins."
        )
    
    # Validate ion count data
    for batch_id, protein_data in batch_data.items():
        for protein_name, ion_counts in protein_data.items():
            if not isinstance(ion_counts, np.ndarray):
                raise TypeError(
                    f"Ion counts for {batch_id}:{protein_name} must be numpy array"
                )
            
            if not np.issubdtype(ion_counts.dtype, np.number):
                raise TypeError(
                    f"Ion counts for {batch_id}:{protein_name} must be numeric"
                )


def compute_bootstrap_fold_change(
    condition1_replicates: List[np.ndarray],
    condition2_replicates: List[np.ndarray],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Union[float, Tuple[float, float]]]:
    """
    Compute bootstrap confidence intervals for fold change with n=2 replicates.
    
    Args:
        condition1_replicates: List of ion count arrays for condition 1
        condition2_replicates: List of ion count arrays for condition 2
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary with point estimate, confidence interval, and p-value
    """
    if len(condition1_replicates) < 2 or len(condition2_replicates) < 2:
        warnings.warn("Bootstrap with n<2 per condition may be unreliable")
    
    # Compute point estimate (median fold change)
    median1 = np.median(np.concatenate(condition1_replicates))
    median2 = np.median(np.concatenate(condition2_replicates))
    point_estimate = median2 / (median1 + 1e-10)  # Avoid division by zero
    
    # Bootstrap sampling
    np.random.seed(42)  # Reproducible results
    bootstrap_fold_changes = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement within each condition
        boot1_data = []
        for replicate in condition1_replicates:
            if len(replicate) > 0:
                boot_sample = np.random.choice(replicate, size=len(replicate), replace=True)
                boot1_data.append(boot_sample)
        
        boot2_data = []
        for replicate in condition2_replicates:
            if len(replicate) > 0:
                boot_sample = np.random.choice(replicate, size=len(replicate), replace=True)
                boot2_data.append(boot_sample)
        
        if boot1_data and boot2_data:
            boot_median1 = np.median(np.concatenate(boot1_data))
            boot_median2 = np.median(np.concatenate(boot2_data))
            boot_fc = boot_median2 / (boot_median1 + 1e-10)
            bootstrap_fold_changes.append(boot_fc)
    
    if not bootstrap_fold_changes:
        return {
            'point_estimate': point_estimate,
            'confidence_interval': (np.nan, np.nan),
            'p_value_bootstrap': np.nan
        }
    
    bootstrap_fold_changes = np.array(bootstrap_fold_changes)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_fold_changes, 100 * alpha/2)
    ci_upper = np.percentile(bootstrap_fold_changes, 100 * (1 - alpha/2))
    
    # Bootstrap p-value (proportion of bootstrap samples where FC = 1.0)
    p_value_bootstrap = 2 * min(
        np.mean(bootstrap_fold_changes >= 1.0),
        np.mean(bootstrap_fold_changes <= 1.0)
    )
    
    return {
        'point_estimate': float(point_estimate),
        'confidence_interval': (float(ci_lower), float(ci_upper)),
        'p_value_bootstrap': float(p_value_bootstrap),
        'bootstrap_distribution': bootstrap_fold_changes
    }


def _validate_config(config: BatchCorrectionConfig) -> None:
    """Validate sham-anchored normalization configuration parameters."""
    if config.confidence_level <= 0 or config.confidence_level >= 1:
        raise ValueError("Confidence level must be in range (0, 1)")
    
    if config.bootstrap_samples < 100:
        warnings.warn("Bootstrap samples < 100 may give unreliable results")
        
    if not config.sham_condition:
        raise ValueError("Sham condition name cannot be empty")


# Legacy quantile correction quality metrics removed.


def _compute_improvement_metrics(
    batch_effects_before: Dict[str, any],
    batch_effects_after: Dict[str, any]
) -> Dict[str, float]:
    """Compute batch correction improvement metrics."""
    before_severity = batch_effects_before.get('overall_severity', 0)
    after_severity = batch_effects_after.get('overall_severity', 0)
    
    improvement_ratio = (before_severity - after_severity) / (before_severity + 1e-10)
    
    return {
        'severity_before': float(before_severity),
        'severity_after': float(after_severity),
        'improvement_ratio': float(improvement_ratio)
    }


def bead_anchored_normalize(
    batch_data: Dict[str, Dict[str, np.ndarray]],
    batch_metadata: Dict[str, Dict[str, any]],
    config: BatchCorrectionConfig = None
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, any]]:
    """
    Perform bead-anchored normalization for cross-run instrument standardization.
    
    Uses bead channels (e.g., 130Ba, 131Xe) to correct for instrument drift,
    detector sensitivity changes, and cross-platform differences.
    
    Args:
        batch_data: Dictionary mapping batch_id -> protein_name -> ion_counts
        batch_metadata: Dictionary mapping batch_id -> metadata_dict
        config: Batch correction configuration
        
    Returns:
        Tuple of (normalized_data, normalization_statistics)
    """
    if config is None:
        config = BatchCorrectionConfig()
    
    logger.info(f"Starting bead-anchored normalization with channels: {config.bead_channels}")
    
    # Validate inputs
    validate_batch_structure(batch_data, batch_metadata)
    
    # Extract bead signals across all batches
    bead_signals, valid_batches = _extract_bead_signals(batch_data, config)

    if len(valid_batches) < 2:
        error_msg = (
            f"Bead normalization is enabled but insufficient bead signals were found "
            f"({len(valid_batches)}/{len(batch_data)} batches valid). "
            f"This is a critical data integrity failure. "
            f"Check that bead channels {config.bead_channels} are: "
            f"(1) present in the input data, "
            f"(2) loaded by load_roi_data(), and "
            f"(3) have median signal >= {config.bead_signal_threshold}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Fit temporal drift model
    drift_model = _fit_temporal_drift_model(bead_signals, batch_metadata, config)
    
    # Apply normalization to all batches
    normalized_data = {}
    normalization_stats = {
        'method': 'bead_anchored',
        'drift_correction_method': config.drift_correction_method,
        'bead_channels': config.bead_channels,
        'drift_model': drift_model,
        'valid_batches': valid_batches,
        'per_batch_stats': {}
    }
    
    for batch_id, protein_data in batch_data.items():
        if batch_id in valid_batches:
            # Apply drift correction
            normalized_batch, batch_stats = _apply_bead_normalization(
                protein_data, drift_model[batch_id], config
            )
        else:
            # Keep original data for batches without valid beads
            normalized_batch = {k: v.copy() for k, v in protein_data.items()}
            batch_stats = {'correction_applied': False, 'reason': 'insufficient_bead_signal'}
        
        normalized_data[batch_id] = normalized_batch
        normalization_stats['per_batch_stats'][batch_id] = batch_stats
    
    logger.info(f"Bead normalization completed for {len(valid_batches)}/{len(batch_data)} batches")
    
    return normalized_data, normalization_stats


def _extract_bead_signals(
    batch_data: Dict[str, Dict[str, np.ndarray]],
    config: BatchCorrectionConfig
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Extract bead channel signals from each batch."""

    bead_signals = {}
    valid_batches = []

    # Collect statistics for efficient reporting
    missing_channels = {}  # bead_channel -> count
    low_signal_channels = {}  # bead_channel -> count
    failed_batches = []

    for batch_id, protein_data in batch_data.items():
        batch_bead_signals = {}
        has_valid_beads = False

        for bead_channel in config.bead_channels:
            if bead_channel in protein_data:
                # Use median signal as robust estimator
                median_signal = np.median(protein_data[bead_channel])

                if median_signal >= config.bead_signal_threshold:
                    batch_bead_signals[bead_channel] = median_signal
                    has_valid_beads = True
                else:
                    low_signal_channels[bead_channel] = low_signal_channels.get(bead_channel, 0) + 1
            else:
                missing_channels[bead_channel] = missing_channels.get(bead_channel, 0) + 1

        if has_valid_beads:
            bead_signals[batch_id] = batch_bead_signals
            valid_batches.append(batch_id)
        else:
            failed_batches.append(batch_id)

    # Log concise summary instead of per-batch messages
    total_batches = len(batch_data)
    logger.info(f"Bead Signal Extraction: {len(valid_batches)}/{total_batches} batches have valid signals")

    if failed_batches:
        logger.warning(f"{len(failed_batches)} batches failed bead extraction:")
        if missing_channels:
            for ch, count in missing_channels.items():
                logger.warning(f"  - {count} batches missing channel '{ch}'")
        if low_signal_channels:
            for ch, count in low_signal_channels.items():
                logger.warning(f"  - {count} batches have low signal for '{ch}' (threshold={config.bead_signal_threshold})")

    return bead_signals, valid_batches


def _fit_temporal_drift_model(
    bead_signals: Dict[str, Dict[str, float]],
    batch_metadata: Dict[str, Dict[str, any]],
    config: BatchCorrectionConfig
) -> Dict[str, Dict[str, float]]:
    """Fit temporal drift correction model using bead signals."""
    
    # Extract temporal information if available
    batch_times = []
    batch_ids = []
    
    for batch_id in bead_signals.keys():
        metadata = batch_metadata.get(batch_id, {})
        
        # Try to extract acquisition time/order
        if 'acquisition_time' in metadata:
            batch_times.append(metadata['acquisition_time'])
        elif 'batch_order' in metadata:
            batch_times.append(metadata['batch_order'])
        else:
            # Use batch ID order as fallback
            batch_times.append(len(batch_times))
        
        batch_ids.append(batch_id)
    
    # Sort by temporal order
    sorted_indices = np.argsort(batch_times)
    sorted_batch_ids = [batch_ids[i] for i in sorted_indices]
    
    # Compute reference bead intensities (first batch or median)
    reference_signals = {}
    if config.drift_correction_method == 'median_reference':
        # Use median across all batches as reference
        for bead_channel in config.bead_channels:
            channel_signals = []
            for batch_id in sorted_batch_ids:
                if bead_channel in bead_signals[batch_id]:
                    channel_signals.append(bead_signals[batch_id][bead_channel])
            
            if channel_signals:
                reference_signals[bead_channel] = np.median(channel_signals)
    else:
        # Use first batch as reference
        first_batch = sorted_batch_ids[0]
        reference_signals = bead_signals[first_batch].copy()
    
    # Compute correction factors for each batch
    drift_model = {}
    
    for batch_id in bead_signals.keys():
        batch_corrections = {}
        
        for bead_channel, reference_signal in reference_signals.items():
            if bead_channel in bead_signals[batch_id]:
                observed_signal = bead_signals[batch_id][bead_channel]
                
                # Correction factor: reference / observed
                correction_factor = reference_signal / (observed_signal + 1e-10)
                
                # Bound correction factors to reasonable range
                correction_factor = np.clip(correction_factor, 0.1, 10.0)
                
                batch_corrections[bead_channel] = correction_factor
            else:
                batch_corrections[bead_channel] = 1.0  # No correction
        
        drift_model[batch_id] = batch_corrections
    
    return drift_model


def _apply_bead_normalization(
    protein_data: Dict[str, np.ndarray],
    batch_corrections: Dict[str, float],
    config: BatchCorrectionConfig
) -> Tuple[Dict[str, np.ndarray], Dict[str, any]]:
    """Apply bead-based correction factors to protein data."""
    
    normalized_data = {}
    batch_stats = {'correction_applied': True, 'correction_factors': {}}
    
    # Compute overall correction factor (median of bead corrections)
    valid_corrections = [
        factor for channel, factor in batch_corrections.items()
        if channel in config.bead_channels and factor != 1.0
    ]
    
    if valid_corrections:
        overall_correction = np.median(valid_corrections)
    else:
        overall_correction = 1.0
    
    # Apply correction to all protein channels
    for protein_name, ion_counts in protein_data.items():
        
        # Use channel-specific correction if available, otherwise use overall correction
        if protein_name in batch_corrections:
            correction_factor = batch_corrections[protein_name]
        else:
            correction_factor = overall_correction
        
        # Apply correction
        normalized_counts = ion_counts * correction_factor
        normalized_data[protein_name] = normalized_counts
        
        batch_stats['correction_factors'][protein_name] = correction_factor
    
    batch_stats['overall_correction_factor'] = overall_correction
    batch_stats['n_bead_channels_used'] = len([
        ch for ch in config.bead_channels if ch in batch_corrections and batch_corrections[ch] != 1.0
    ])
    
    return normalized_data, batch_stats


def detect_instrumental_drift(
    batch_data: Dict[str, Dict[str, np.ndarray]],
    batch_metadata: Dict[str, Dict[str, any]],
    config: BatchCorrectionConfig = None
) -> Dict[str, any]:
    """
    Detect instrumental drift using bead channel analysis.
    
    Args:
        batch_data: Dictionary mapping batch_id -> protein_name -> ion_counts
        batch_metadata: Dictionary mapping batch_id -> metadata_dict
        config: Batch correction configuration
        
    Returns:
        Dictionary with drift analysis results
    """
    if config is None:
        config = BatchCorrectionConfig()
    
    # Extract bead signals
    bead_signals, valid_batches = _extract_bead_signals(batch_data, config)
    
    if len(valid_batches) < 3:
        return {
            'drift_detected': False,
            'reason': 'insufficient_data',
            'n_valid_batches': len(valid_batches)
        }
    
    # Analyze drift for each bead channel
    drift_analysis = {}
    overall_drift_score = 0
    
    for bead_channel in config.bead_channels:
        channel_signals = []
        batch_order = []
        
        # Extract signals in temporal order
        for i, batch_id in enumerate(valid_batches):
            if bead_channel in bead_signals[batch_id]:
                channel_signals.append(bead_signals[batch_id][bead_channel])
                batch_order.append(i)
        
        if len(channel_signals) < 3:
            continue
        
        channel_signals = np.array(channel_signals)
        batch_order = np.array(batch_order)
        
        # Compute drift metrics
        # 1. Linear trend
        if len(batch_order) > 1:
            slope, intercept = np.polyfit(batch_order, channel_signals, 1)
            relative_slope = slope / (np.mean(channel_signals) + 1e-10)
        else:
            slope, relative_slope = 0, 0
        
        # 2. Coefficient of variation
        cv = np.std(channel_signals) / (np.mean(channel_signals) + 1e-10)
        
        # 3. Range relative to mean
        signal_range = np.max(channel_signals) - np.min(channel_signals)
        relative_range = signal_range / (np.mean(channel_signals) + 1e-10)
        
        # Drift score (higher = more drift)
        drift_score = abs(relative_slope) + cv + relative_range
        overall_drift_score += drift_score
        
        drift_analysis[bead_channel] = {
            'n_batches': len(channel_signals),
            'mean_signal': np.mean(channel_signals),
            'cv': cv,
            'linear_slope': slope,
            'relative_slope': relative_slope,
            'relative_range': relative_range,
            'drift_score': drift_score,
            'signals': channel_signals.tolist()
        }
    
    # Overall drift assessment
    n_channels = len(drift_analysis)
    if n_channels > 0:
        avg_drift_score = overall_drift_score / n_channels
        drift_detected = avg_drift_score > 0.2  # Empirical threshold
    else:
        avg_drift_score = 0
        drift_detected = False
    
    return {
        'drift_detected': drift_detected,
        'overall_drift_score': avg_drift_score,
        'n_valid_batches': len(valid_batches),
        'n_bead_channels': n_channels,
        'per_channel_analysis': drift_analysis,
        'recommendation': 'apply_bead_normalization' if drift_detected else 'no_correction_needed'
    }