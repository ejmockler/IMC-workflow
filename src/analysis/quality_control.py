"""
Quality Control Module for IMC Data

Monitors calibration channels, carrier gas signals, and batch consistency
to identify technical issues and ensure data quality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy import stats


def monitor_calibration_channels(
    roi_data: pd.DataFrame,
    calibration_channels: List[str],
    cv_threshold: float = 0.2
) -> Dict[str, Any]:
    """
    Monitor calibration channel stability within an ROI.
    
    Calibration channels (e.g., 130Ba, 131Xe) should have stable signals.
    High CV indicates instrumental drift or calibration issues.
    
    Args:
        roi_data: Raw ROI data with all channels
        calibration_channels: List of calibration channel names
        cv_threshold: Coefficient of variation threshold for warning
        
    Returns:
        Dictionary with calibration QC metrics
    """
    qc_results = {
        'calibration_stable': True,
        'channels': {}
    }
    
    for cal_channel in calibration_channels:
        # Find matching column
        cal_cols = [col for col in roi_data.columns if cal_channel in col]
        if not cal_cols:
            continue
            
        signal = roi_data[cal_cols[0]].values
        
        # Calculate statistics
        mean_signal = np.mean(signal)
        std_signal = np.std(signal)
        cv = std_signal / mean_signal if mean_signal > 0 else np.inf
        
        # Check stability
        is_stable = cv < cv_threshold
        if not is_stable:
            qc_results['calibration_stable'] = False
        
        qc_results['channels'][cal_channel] = {
            'mean': mean_signal,
            'std': std_signal,
            'cv': cv,
            'stable': is_stable,
            'n_pixels': len(signal),
            'percentiles': {
                '25': np.percentile(signal, 25),
                '50': np.percentile(signal, 50),
                '75': np.percentile(signal, 75)
            }
        }
    
    return qc_results


def monitor_carrier_gas(
    roi_data: pd.DataFrame,
    carrier_gas_channel: str,
    min_signal: float = 100
) -> Dict[str, Any]:
    """
    Monitor carrier gas (80ArAr) signal quality.
    
    Low carrier gas signal indicates plasma instability or
    sample introduction issues.
    
    Args:
        roi_data: Raw ROI data
        carrier_gas_channel: Name of carrier gas channel (e.g., '80ArAr')
        min_signal: Minimum acceptable signal level
        
    Returns:
        Dictionary with carrier gas QC metrics
    """
    # Find carrier gas column
    gas_cols = [col for col in roi_data.columns if carrier_gas_channel in col]
    if not gas_cols:
        return {
            'carrier_gas_present': False,
            'adequate_signal': False,
            'message': f'Carrier gas channel {carrier_gas_channel} not found'
        }
    
    signal = roi_data[gas_cols[0]].values
    
    # Calculate metrics
    mean_signal = np.mean(signal)
    median_signal = np.median(signal)
    min_observed = np.min(signal)
    max_observed = np.max(signal)
    
    # Check adequacy
    adequate = median_signal >= min_signal
    
    return {
        'carrier_gas_present': True,
        'adequate_signal': adequate,
        'mean': mean_signal,
        'median': median_signal,
        'min': min_observed,
        'max': max_observed,
        'threshold': min_signal,
        'percent_below_threshold': np.sum(signal < min_signal) / len(signal) * 100
    }


def check_background_levels(
    roi_data: pd.DataFrame,
    background_channel: str,
    protein_channels: List[str]
) -> Dict[str, Any]:
    """
    Assess background signal levels and signal-to-background ratios.
    
    Args:
        roi_data: Raw ROI data
        background_channel: Name of background channel (e.g., '190BCKG')
        protein_channels: List of protein channel names
        
    Returns:
        Dictionary with background QC metrics
    """
    # Find background column
    bg_cols = [col for col in roi_data.columns if background_channel in col]
    if not bg_cols:
        return {
            'background_present': False,
            'message': f'Background channel {background_channel} not found'
        }
    
    background = roi_data[bg_cols[0]].values
    
    results = {
        'background_present': True,
        'background_stats': {
            'mean': np.mean(background),
            'median': np.median(background),
            'std': np.std(background),
            'max': np.max(background)
        },
        'signal_to_background': {}
    }
    
    # Calculate signal-to-background for each protein
    for protein in protein_channels:
        protein_cols = [col for col in roi_data.columns if protein in col]
        if protein_cols:
            protein_signal = roi_data[protein_cols[0]].values
            
            # Calculate SNR
            mean_signal = np.mean(protein_signal)
            mean_background = np.mean(background)
            
            if mean_background > 0:
                snr = mean_signal / mean_background
            else:
                snr = np.inf if mean_signal > 0 else 0
            
            results['signal_to_background'][protein] = {
                'snr': snr,
                'mean_signal': mean_signal,
                'adequate': snr > 3.0  # Common SNR threshold
            }
    
    return results


def detect_spatial_artifacts(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    artifact_threshold: float = 5.0
) -> Dict[str, Any]:
    """
    Detect spatial artifacts like edge effects or striping patterns.
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of ion count arrays
        artifact_threshold: Z-score threshold for artifact detection
        
    Returns:
        Dictionary with spatial artifact detection results
    """
    results = {
        'edge_effects': {},
        'striping': {}
    }
    
    # Check for edge effects
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Define edge regions (10% of range from each edge)
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    
    edge_distance_x = 0.1 * x_range
    edge_distance_y = 0.1 * y_range
    
    edge_mask = (
        (x_coords < x_coords.min() + edge_distance_x) |
        (x_coords > x_coords.max() - edge_distance_x) |
        (y_coords < y_coords.min() + edge_distance_y) |
        (y_coords > y_coords.max() - edge_distance_y)
    )
    
    center_mask = ~edge_mask
    
    for protein, counts in ion_counts.items():
        if np.sum(center_mask) > 0 and np.sum(edge_mask) > 0:
            edge_mean = np.mean(counts[edge_mask])
            center_mean = np.mean(counts[center_mask])
            
            # Test for significant difference
            _, p_value = stats.ttest_ind(counts[edge_mask], counts[center_mask])
            
            edge_ratio = edge_mean / center_mean if center_mean > 0 else np.inf
            has_edge_effect = (
                p_value < 0.01 and 
                abs(np.log2(edge_ratio)) > 1  # >2-fold difference
            )
            
            results['edge_effects'][protein] = {
                'edge_mean': edge_mean,
                'center_mean': center_mean,
                'ratio': edge_ratio,
                'p_value': p_value,
                'has_artifact': has_edge_effect
            }
    
    return results


def generate_qc_report(
    batch_data: Dict[str, Dict[str, Any]],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate comprehensive QC report for all ROIs.
    
    Args:
        batch_data: Dictionary of batch data with ROI information
        config: Configuration dictionary with QC parameters
        
    Returns:
        Comprehensive QC report dictionary
    """
    logger = logging.getLogger('QualityControl')
    
    qc_report = {
        'summary': {
            'total_rois': 0,
            'passed_qc': 0,
            'failed_calibration': [],
            'low_carrier_gas': [],
            'low_snr': []
        },
        'roi_qc': {}
    }
    
    # Get QC parameters from config
    qc_config = config.get('quality_control', {})
    cv_threshold = qc_config.get('calibration_cv_threshold', 0.2)
    min_carrier_gas = qc_config.get('min_carrier_gas_signal', 100)
    
    # Process each ROI
    for batch_id, roi_dict in batch_data.items():
        for roi_name, roi_info in roi_dict.items():
            qc_report['summary']['total_rois'] += 1
            
            roi_qc = {
                'batch_id': batch_id,
                'roi_name': roi_name,
                'passed': True,
                'issues': []
            }
            
            # Run QC checks if raw data available
            if 'raw_data' in roi_info:
                roi_data = roi_info['raw_data']
                
                # Check calibration
                if 'calibration_channels' in config.get('channels', {}):
                    cal_qc = monitor_calibration_channels(
                        roi_data,
                        config['channels']['calibration_channels'],
                        cv_threshold
                    )
                    roi_qc['calibration'] = cal_qc
                    
                    if not cal_qc['calibration_stable']:
                        roi_qc['passed'] = False
                        roi_qc['issues'].append('Calibration unstable')
                        qc_report['summary']['failed_calibration'].append(roi_name)
                
                # Check carrier gas
                if 'carrier_gas_channel' in config.get('channels', {}):
                    gas_qc = monitor_carrier_gas(
                        roi_data,
                        config['channels']['carrier_gas_channel'],
                        min_carrier_gas
                    )
                    roi_qc['carrier_gas'] = gas_qc
                    
                    if not gas_qc.get('adequate_signal', False):
                        roi_qc['passed'] = False
                        roi_qc['issues'].append('Low carrier gas signal')
                        qc_report['summary']['low_carrier_gas'].append(roi_name)
                
                # Check background
                if 'background_channel' in config.get('channels', {}):
                    bg_qc = check_background_levels(
                        roi_data,
                        config['channels']['background_channel'],
                        config['channels'].get('protein_channels', [])
                    )
                    roi_qc['background'] = bg_qc
                    
                    # Check for low SNR proteins
                    low_snr = [
                        protein for protein, stats in bg_qc.get('signal_to_background', {}).items()
                        if not stats.get('adequate', True)
                    ]
                    if low_snr:
                        roi_qc['issues'].append(f'Low SNR: {", ".join(low_snr)}')
                        qc_report['summary']['low_snr'].append(roi_name)
            
            # Check for spatial artifacts
            if 'coords' in roi_info and 'ion_counts' in roi_info:
                artifact_qc = detect_spatial_artifacts(
                    roi_info['coords'],
                    roi_info['ion_counts']
                )
                roi_qc['spatial_artifacts'] = artifact_qc
                
                # Check for edge effects
                has_edge_effects = any(
                    stats.get('has_artifact', False)
                    for stats in artifact_qc.get('edge_effects', {}).values()
                )
                if has_edge_effects:
                    roi_qc['issues'].append('Edge effects detected')
            
            if roi_qc['passed']:
                qc_report['summary']['passed_qc'] += 1
            
            qc_report['roi_qc'][roi_name] = roi_qc
    
    # Calculate summary statistics
    qc_report['summary']['pass_rate'] = (
        qc_report['summary']['passed_qc'] / qc_report['summary']['total_rois'] * 100
        if qc_report['summary']['total_rois'] > 0 else 0
    )
    
    logger.info(f"QC Summary: {qc_report['summary']['passed_qc']}/{qc_report['summary']['total_rois']} "
                f"ROIs passed ({qc_report['summary']['pass_rate']:.1f}%)")
    
    return qc_report


def validate_batch_consistency(
    batch_ion_counts: Dict[str, Dict[str, np.ndarray]],
    batch_metadata: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate consistency across batches.
    
    Args:
        batch_ion_counts: Ion counts organized by batch
        batch_metadata: Metadata for each batch
        
    Returns:
        Batch consistency validation results
    """
    results = {
        'batch_effects_detected': False,
        'protein_consistency': {},
        'recommendations': []
    }
    
    # Get all protein names
    all_proteins = set()
    for batch_counts in batch_ion_counts.values():
        all_proteins.update(batch_counts.keys())
    
    # Check each protein for batch effects
    for protein in all_proteins:
        batch_means = []
        batch_names = []
        
        for batch_id, counts in batch_ion_counts.items():
            if protein in counts:
                batch_means.append(np.mean(counts[protein]))
                batch_names.append(batch_id)
        
        if len(batch_means) > 1:
            # ANOVA to test for batch effects
            batch_data_lists = [
                batch_ion_counts[batch][protein][:1000]  # Sample for efficiency
                for batch in batch_names
                if protein in batch_ion_counts[batch]
            ]
            
            if len(batch_data_lists) > 1:
                f_stat, p_value = stats.f_oneway(*batch_data_lists)
                
                # Calculate coefficient of variation across batches
                cv = np.std(batch_means) / np.mean(batch_means) if np.mean(batch_means) > 0 else 0
                
                has_batch_effect = p_value < 0.01 and cv > 0.3
                
                results['protein_consistency'][protein] = {
                    'batch_means': dict(zip(batch_names, batch_means)),
                    'cv_across_batches': cv,
                    'anova_p_value': p_value,
                    'has_batch_effect': has_batch_effect
                }
                
                if has_batch_effect:
                    results['batch_effects_detected'] = True
    
    # Generate recommendations
    if results['batch_effects_detected']:
        results['recommendations'].append('Apply batch correction before analysis')
        
        # Identify most affected proteins
        affected_proteins = [
            protein for protein, stats in results['protein_consistency'].items()
            if stats.get('has_batch_effect', False)
        ]
        if affected_proteins:
            results['recommendations'].append(
                f'Proteins with significant batch effects: {", ".join(affected_proteins[:5])}'
            )
    
    return results


def monitor_total_ion_counts(
    roi_data: pd.DataFrame,
    protein_channels: List[str],
    qc_thresholds: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Monitor Total Ion Counts (TIC) per ROI to detect acquisition issues.
    
    Low TIC indicates laser power loss or movement off tissue.
    
    Args:
        roi_data: Raw ROI data
        protein_channels: List of protein channel names
        qc_thresholds: QC threshold configuration dictionary
        
    Returns:
        Dictionary with TIC QC metrics
    """
    # Get thresholds from config with defaults
    if qc_thresholds is None:
        qc_thresholds = {}
    tic_thresholds = qc_thresholds.get('total_ion_counts', {})
    min_tic_percentile = tic_thresholds.get('min_tic_percentile', 10)
    max_low_tic_pixels_percent = tic_thresholds.get('max_low_tic_pixels_percent', 20)
    
    # Calculate TIC for each pixel
    total_counts = np.zeros(len(roi_data))
    
    for protein in protein_channels:
        protein_cols = [col for col in roi_data.columns if protein in col]
        if protein_cols:
            total_counts += roi_data[protein_cols[0]].values
    
    # Calculate metrics
    median_tic = np.median(total_counts)
    mean_tic = np.mean(total_counts)
    std_tic = np.std(total_counts)
    min_tic = np.min(total_counts)
    max_tic = np.max(total_counts)
    
    # Threshold based on percentile
    threshold = np.percentile(total_counts, min_tic_percentile)
    low_tic_pixels = np.sum(total_counts < threshold)
    
    # Check for acquisition issues using configurable thresholds
    has_acquisition_issues = (
        median_tic < threshold or
        low_tic_pixels > len(total_counts) * (max_low_tic_pixels_percent / 100.0)
    )
    
    return {
        'median_tic': median_tic,
        'mean_tic': mean_tic,
        'std_tic': std_tic,
        'min_tic': min_tic,
        'max_tic': max_tic,
        'threshold': threshold,
        'low_tic_pixels': int(low_tic_pixels),
        'percent_low_tic': low_tic_pixels / len(total_counts) * 100,
        'has_acquisition_issues': has_acquisition_issues
    }


def track_calibration_drift(
    batch_data: Dict[str, Dict[str, Any]],
    calibration_channels: List[str],
    qc_thresholds: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Track calibration channel stability across entire acquisition run.
    
    Args:
        batch_data: All batch data 
        calibration_channels: List of calibration channel names
        qc_thresholds: QC threshold configuration dictionary
        
    Returns:
        Dictionary with drift analysis results
    """
    # Get thresholds from config with defaults
    if qc_thresholds is None:
        qc_thresholds = {}
    drift_thresholds = qc_thresholds.get('calibration_drift', {})
    max_drift_percent = drift_thresholds.get('max_drift_percent', 5)
    max_cv_across_rois = drift_thresholds.get('max_cv_across_rois', 0.3)
    
    drift_results = {}
    
    for cal_channel in calibration_channels:
        roi_medians = []
        roi_names = []
        
        # Collect median values across all ROIs
        for batch_id, roi_dict in batch_data.items():
            for roi_name, roi_info in roi_dict.items():
                # Look for raw data if available
                if 'raw_data' in roi_info:
                    roi_data = roi_info['raw_data']
                    cal_cols = [col for col in roi_data.columns if cal_channel in col]
                    if cal_cols:
                        median_val = np.median(roi_data[cal_cols[0]].values)
                        roi_medians.append(median_val)
                        roi_names.append(roi_name)
        
        if len(roi_medians) > 0:
            roi_medians = np.array(roi_medians)
            
            # Calculate drift metrics
            linear_trend = np.polyfit(range(len(roi_medians)), roi_medians, 1)[0]
            cv_across_rois = np.std(roi_medians) / np.mean(roi_medians) if np.mean(roi_medians) > 0 else 0
            
            # Detect significant drift using configurable threshold
            if len(roi_medians) > 1:
                percent_change = (roi_medians[-1] - roi_medians[0]) / roi_medians[0] * 100
                has_drift = abs(percent_change) > max_drift_percent or cv_across_rois > max_cv_across_rois
            else:
                percent_change = 0
                has_drift = False
            
            drift_results[cal_channel] = {
                'n_rois': len(roi_medians),
                'mean_signal': np.mean(roi_medians),
                'cv_across_rois': cv_across_rois,
                'linear_trend': linear_trend,
                'percent_change': percent_change,
                'has_significant_drift': has_drift,
                'roi_values': roi_medians.tolist(),
                'roi_names': roi_names
            }
    
    return drift_results


def assess_segmentation_quality(
    coords: np.ndarray,
    dna_intensities: Dict[str, np.ndarray],
    qc_thresholds: Dict[str, Any] = None,
    target_segments: int = None
) -> Dict[str, Any]:
    """
    Assess quality of DNA-based segmentation.
    
    Args:
        coords: Pixel coordinates
        dna_intensities: DNA channel data
        qc_thresholds: QC threshold configuration dictionary
        target_segments: Expected number of segments
        
    Returns:
        Dictionary with segmentation quality metrics
    """
    # Get thresholds from config with defaults
    if qc_thresholds is None:
        qc_thresholds = {}
    seg_thresholds = qc_thresholds.get('segmentation_quality', {})
    min_dna_signal = seg_thresholds.get('min_dna_signal', 1.0)
    min_tissue_coverage = seg_thresholds.get('min_tissue_coverage_percent', 10)
    std_multiplier = seg_thresholds.get('dna_threshold_std_multiplier', 2.0)
    
    # Combine DNA channels
    dna1 = dna_intensities.get('DNA1', np.zeros(len(coords)))
    dna2 = dna_intensities.get('DNA2', np.zeros(len(coords)))
    total_dna = dna1 + dna2
    
    # Calculate DNA signal statistics
    dna_mean = np.mean(total_dna)
    dna_std = np.std(total_dna)
    dna_max = np.max(total_dna)
    
    # Count pixels with significant DNA signal using configurable threshold
    dna_threshold = dna_mean + std_multiplier * dna_std
    high_dna_pixels = np.sum(total_dna > dna_threshold)
    
    # Estimate tissue coverage
    tissue_coverage = high_dna_pixels / len(coords) * 100
    
    # Quality checks using configurable thresholds
    has_weak_dna = dna_mean < min_dna_signal
    has_poor_coverage = tissue_coverage < min_tissue_coverage
    
    quality_issues = []
    if has_weak_dna:
        quality_issues.append("Weak DNA signal")
    if has_poor_coverage:
        quality_issues.append("Poor tissue coverage")
    
    return {
        'dna_mean': dna_mean,
        'dna_std': dna_std,
        'dna_max': dna_max,
        'dna_threshold': dna_threshold,
        'high_dna_pixels': int(high_dna_pixels),
        'tissue_coverage_percent': tissue_coverage,
        'has_weak_dna': has_weak_dna,
        'has_poor_coverage': has_poor_coverage,
        'quality_issues': quality_issues,
        'segmentation_feasible': not (has_weak_dna or has_poor_coverage)
    }


def create_batch_effect_plots(
    batch_ion_counts: Dict[str, Dict[str, np.ndarray]],
    protein_names: List[str],
    output_dir: str = "qc_plots"
) -> Dict[str, str]:
    """
    Create before/after plots for batch correction visualization.
    
    Args:
        batch_ion_counts: Ion counts by batch
        protein_names: List of protein names
        output_dir: Directory for output plots
        
    Returns:
        Dictionary with plot filenames
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    plot_files = {}
    
    # Create box plots for each protein across batches
    for protein in protein_names[:6]:  # Limit to first 6 proteins
        fig, ax = plt.subplots(figsize=(10, 6))
        
        batch_data = []
        batch_labels = []
        
        for batch_id, ion_counts in batch_ion_counts.items():
            if protein in ion_counts:
                # Subsample for visualization
                data = ion_counts[protein]
                if len(data) > 10000:
                    data = np.random.choice(data, 10000, replace=False)
                batch_data.append(data)
                batch_labels.append(batch_id)
        
        if len(batch_data) > 1:
            ax.boxplot(batch_data, labels=batch_labels)
            ax.set_title(f'{protein} Distribution Across Batches')
            ax.set_ylabel('Ion Counts')
            ax.set_xlabel('Batch')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            filename = output_path / f'{protein}_batch_comparison.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            plot_files[protein] = str(filename)
    
    return plot_files