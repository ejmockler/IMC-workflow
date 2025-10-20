"""
Automatic Pass/Fail Quality Control System for IMC Analysis

Provides comprehensive automated QC with tissue coverage thresholds, signal quality gates,
batch effect detection, statistical monitoring, and automated reporting.

Builds on existing quality_control.py, quality_gates.py, statistical_monitoring.py,
and validation framework patterns for consistent QC architecture.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime
import json
import warnings

# Import existing QC infrastructure
from ..quality_control.quality_gates import QualityGateEngine, GateDecision, QualityThresholds
from ..quality_control.statistical_monitoring import QualityMonitor, QualityMetrics, QualityLimits
from ..validation.framework import ValidationSuite, ValidationRule, ValidationResult, ValidationSeverity, ValidationCategory
from .quality_control import (
    monitor_calibration_channels, monitor_carrier_gas, check_background_levels,
    assess_segmentation_quality, monitor_total_ion_counts, track_calibration_drift,
    validate_batch_consistency
)


@dataclass
class TissueCoverageThresholds:
    """Thresholds for automated tissue coverage assessment."""
    min_tissue_coverage_percent: float = 10.0
    min_dna_signal_intensity: float = 1.0
    min_high_dna_pixels: int = 1000
    dna_threshold_multiplier: float = 2.0
    edge_exclusion_percent: float = 10.0
    
    def validate(self) -> None:
        """Validate threshold parameters."""
        if not (0 < self.min_tissue_coverage_percent <= 100):
            raise ValueError("min_tissue_coverage_percent must be between 0 and 100")
        if self.min_dna_signal_intensity < 0:
            raise ValueError("min_dna_signal_intensity must be non-negative")


@dataclass 
class SignalQualityThresholds:
    """Thresholds for automated signal quality assessment."""
    min_snr: float = 3.0
    min_carrier_gas_signal: float = 100.0
    max_calibration_cv: float = 0.2
    min_tic_percentile: float = 10.0
    max_low_tic_pixels_percent: float = 20.0
    max_background_level: float = 10.0
    min_protein_detection_rate: float = 0.8
    
    def validate(self) -> None:
        """Validate threshold parameters."""
        if self.min_snr <= 0:
            raise ValueError("min_snr must be positive")
        if not (0 <= self.min_tic_percentile <= 100):
            raise ValueError("min_tic_percentile must be between 0 and 100")


@dataclass
class BatchEffectThresholds:
    """Thresholds for automated batch effect detection."""
    max_batch_cv: float = 0.3
    max_drift_percent: float = 5.0
    min_rois_per_batch: int = 3
    anova_p_threshold: float = 0.01
    effect_size_threshold: float = 0.5
    temporal_window_hours: float = 24.0
    
    def validate(self) -> None:
        """Validate threshold parameters."""
        if self.max_batch_cv <= 0:
            raise ValueError("max_batch_cv must be positive")
        if self.min_rois_per_batch < 1:
            raise ValueError("min_rois_per_batch must be at least 1")


@dataclass
class AutomaticQCConfig:
    """Configuration for automatic QC system."""
    tissue_coverage: TissueCoverageThresholds = field(default_factory=TissueCoverageThresholds)
    signal_quality: SignalQualityThresholds = field(default_factory=SignalQualityThresholds)
    batch_effects: BatchEffectThresholds = field(default_factory=BatchEffectThresholds)
    
    # Overall QC behavior
    fail_on_tissue_coverage: bool = True
    fail_on_signal_quality: bool = True
    warn_on_batch_effects: bool = True
    enable_statistical_monitoring: bool = True
    
    # Reporting configuration
    generate_detailed_reports: bool = True
    save_qc_plots: bool = True
    report_format: str = "json"  # "json", "html", "both"
    
    def validate(self) -> None:
        """Validate all threshold configurations."""
        self.tissue_coverage.validate()
        self.signal_quality.validate()
        self.batch_effects.validate()


class TissueCoverageAssessment:
    """Automated tissue coverage assessment with pass/fail decisions."""
    
    def __init__(self, thresholds: TissueCoverageThresholds):
        self.thresholds = thresholds
        self.logger = logging.getLogger('TissueCoverageAssessment')
    
    def assess_coverage(
        self, 
        coords: np.ndarray,
        dna_intensities: Dict[str, np.ndarray],
        roi_id: str
    ) -> Dict[str, Any]:
        """
        Assess tissue coverage with automated pass/fail decision.
        
        Args:
            coords: Pixel coordinates
            dna_intensities: DNA channel data
            roi_id: ROI identifier
            
        Returns:
            Dictionary with coverage assessment and pass/fail decision
        """
        try:
            # Use existing segmentation quality assessment as foundation
            segmentation_qc = assess_segmentation_quality(
                coords=coords,
                dna_intensities=dna_intensities,
                qc_thresholds={
                    'segmentation_quality': {
                        'min_dna_signal': self.thresholds.min_dna_signal_intensity,
                        'min_tissue_coverage_percent': self.thresholds.min_tissue_coverage_percent,
                        'dna_threshold_std_multiplier': self.thresholds.dna_threshold_multiplier
                    }
                }
            )
            
            # Extract key metrics
            tissue_coverage = segmentation_qc['tissue_coverage_percent']
            dna_mean = segmentation_qc['dna_mean']
            high_dna_pixels = segmentation_qc['high_dna_pixels']
            
            # Additional edge effect analysis
            edge_analysis = self._analyze_edge_effects(coords, dna_intensities)
            
            # Make pass/fail decision
            coverage_adequate = tissue_coverage >= self.thresholds.min_tissue_coverage_percent
            signal_adequate = dna_mean >= self.thresholds.min_dna_signal_intensity
            pixel_count_adequate = high_dna_pixels >= self.thresholds.min_high_dna_pixels
            edge_effects_acceptable = edge_analysis['edge_effect_severity'] < 0.5
            
            passed = coverage_adequate and signal_adequate and pixel_count_adequate and edge_effects_acceptable
            
            # Generate detailed assessment
            assessment = {
                'roi_id': roi_id,
                'passed': passed,
                'tissue_coverage_percent': tissue_coverage,
                'dna_mean_intensity': dna_mean,
                'high_dna_pixels': high_dna_pixels,
                'edge_analysis': edge_analysis,
                'thresholds_used': {
                    'min_coverage_percent': self.thresholds.min_tissue_coverage_percent,
                    'min_dna_intensity': self.thresholds.min_dna_signal_intensity,
                    'min_high_dna_pixels': self.thresholds.min_high_dna_pixels
                },
                'pass_criteria': {
                    'coverage_adequate': coverage_adequate,
                    'signal_adequate': signal_adequate,
                    'pixel_count_adequate': pixel_count_adequate,
                    'edge_effects_acceptable': edge_effects_acceptable
                },
                'quality_score': self._calculate_coverage_quality_score(
                    tissue_coverage, dna_mean, high_dna_pixels, edge_analysis
                ),
                'recommendations': self._generate_coverage_recommendations(
                    coverage_adequate, signal_adequate, pixel_count_adequate, edge_effects_acceptable
                )
            }
            
            # Log assessment
            status = "PASS" if passed else "FAIL"
            self.logger.info(f"Tissue coverage assessment for {roi_id}: {status} "
                           f"(coverage: {tissue_coverage:.1f}%, DNA signal: {dna_mean:.2f})")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Tissue coverage assessment failed for {roi_id}: {e}")
            return {
                'roi_id': roi_id,
                'passed': False,
                'error': str(e),
                'quality_score': 0.0,
                'recommendations': ['Repeat tissue coverage assessment after resolving technical issues']
            }
    
    def _analyze_edge_effects(
        self, 
        coords: np.ndarray, 
        dna_intensities: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze edge effects in tissue coverage."""
        # Calculate edge exclusion regions
        x_coords, y_coords = coords[:, 0], coords[:, 1]
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        
        edge_distance_x = (self.thresholds.edge_exclusion_percent / 100) * x_range
        edge_distance_y = (self.thresholds.edge_exclusion_percent / 100) * y_range
        
        # Define edge and center regions
        edge_mask = (
            (x_coords < x_coords.min() + edge_distance_x) |
            (x_coords > x_coords.max() - edge_distance_x) |
            (y_coords < y_coords.min() + edge_distance_y) |
            (y_coords > y_coords.max() - edge_distance_y)
        )
        center_mask = ~edge_mask
        
        # Analyze DNA signal in edge vs center regions
        dna1 = dna_intensities.get('DNA1', np.zeros(len(coords)))
        dna2 = dna_intensities.get('DNA2', np.zeros(len(coords)))
        total_dna = dna1 + dna2
        
        if np.sum(edge_mask) > 0 and np.sum(center_mask) > 0:
            edge_mean = np.mean(total_dna[edge_mask])
            center_mean = np.mean(total_dna[center_mask])
            
            # Calculate edge effect severity
            if center_mean > 0:
                edge_ratio = edge_mean / center_mean
                edge_effect_severity = abs(1.0 - edge_ratio)
            else:
                edge_effect_severity = 1.0  # Maximum severity if no center signal
            
            # Statistical test for edge effects
            from scipy import stats
            if len(total_dna[edge_mask]) > 5 and len(total_dna[center_mask]) > 5:
                _, p_value = stats.ttest_ind(total_dna[edge_mask], total_dna[center_mask])
                statistically_significant = p_value < 0.01
            else:
                p_value = 1.0
                statistically_significant = False
        else:
            edge_mean = center_mean = 0.0
            edge_effect_severity = 0.0
            p_value = 1.0
            statistically_significant = False
        
        return {
            'edge_mean_dna': edge_mean,
            'center_mean_dna': center_mean,
            'edge_effect_severity': edge_effect_severity,
            'p_value': p_value,
            'statistically_significant': statistically_significant,
            'edge_pixels': int(np.sum(edge_mask)),
            'center_pixels': int(np.sum(center_mask))
        }
    
    def _calculate_coverage_quality_score(
        self, 
        coverage_percent: float,
        dna_mean: float,
        high_dna_pixels: int,
        edge_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall coverage quality score (0-1)."""
        # Coverage score (0-1)
        coverage_score = min(1.0, coverage_percent / self.thresholds.min_tissue_coverage_percent)
        
        # DNA signal score (0-1)
        signal_score = min(1.0, dna_mean / self.thresholds.min_dna_signal_intensity)
        
        # Pixel count score (0-1)
        pixel_score = min(1.0, high_dna_pixels / self.thresholds.min_high_dna_pixels)
        
        # Edge effect penalty (0-1, where 1 is no penalty)
        edge_penalty = 1.0 - edge_analysis['edge_effect_severity']
        
        # Weighted average
        weights = [0.4, 0.3, 0.2, 0.1]  # coverage, signal, pixels, edge
        scores = [coverage_score, signal_score, pixel_score, edge_penalty]
        
        return float(np.average(scores, weights=weights))
    
    def _generate_coverage_recommendations(
        self,
        coverage_adequate: bool,
        signal_adequate: bool, 
        pixel_count_adequate: bool,
        edge_effects_acceptable: bool
    ) -> List[str]:
        """Generate recommendations based on coverage assessment."""
        recommendations = []
        
        if not coverage_adequate:
            recommendations.append("Tissue coverage below threshold - consider re-sectioning or different ROI selection")
        
        if not signal_adequate:
            recommendations.append("DNA signal intensity low - check staining protocol and imaging parameters")
        
        if not pixel_count_adequate:
            recommendations.append("Insufficient high-DNA pixels - may need larger ROI or better tissue preparation")
        
        if not edge_effects_acceptable:
            recommendations.append("Significant edge effects detected - review imaging setup and tissue mounting")
        
        if not recommendations:
            recommendations.append("Tissue coverage assessment passed - proceed with analysis")
        
        return recommendations


class SignalQualityGates:
    """Automated signal quality gates with pass/fail decisions."""
    
    def __init__(self, thresholds: SignalQualityThresholds):
        self.thresholds = thresholds
        self.logger = logging.getLogger('SignalQualityGates')
    
    def assess_signal_quality(
        self,
        roi_data: pd.DataFrame,
        protein_channels: List[str],
        calibration_channels: List[str],
        carrier_gas_channel: str,
        background_channel: str,
        roi_id: str
    ) -> Dict[str, Any]:
        """
        Comprehensive signal quality assessment with automated gates.
        
        Args:
            roi_data: Raw ROI data
            protein_channels: List of protein channel names
            calibration_channels: List of calibration channels
            carrier_gas_channel: Carrier gas channel name
            background_channel: Background channel name
            roi_id: ROI identifier
            
        Returns:
            Dictionary with signal quality assessment and pass/fail decision
        """
        try:
            assessment = {
                'roi_id': roi_id,
                'quality_checks': {},
                'overall_passed': True,
                'failed_checks': [],
                'warnings': [],
                'quality_score': 0.0,
                'recommendations': []
            }
            
            # 1. Calibration channel stability
            cal_qc = monitor_calibration_channels(
                roi_data, calibration_channels, self.thresholds.max_calibration_cv
            )
            cal_passed = cal_qc['calibration_stable']
            assessment['quality_checks']['calibration'] = {
                'passed': cal_passed,
                'details': cal_qc,
                'threshold': self.thresholds.max_calibration_cv
            }
            if not cal_passed:
                assessment['failed_checks'].append('calibration_unstable')
                assessment['overall_passed'] = False
            
            # 2. Carrier gas signal adequacy
            gas_qc = monitor_carrier_gas(
                roi_data, carrier_gas_channel, self.thresholds.min_carrier_gas_signal
            )
            gas_passed = gas_qc.get('adequate_signal', False)
            assessment['quality_checks']['carrier_gas'] = {
                'passed': gas_passed,
                'details': gas_qc,
                'threshold': self.thresholds.min_carrier_gas_signal
            }
            if not gas_passed:
                assessment['failed_checks'].append('low_carrier_gas')
                assessment['overall_passed'] = False
            
            # 3. Background levels and SNR
            bg_qc = check_background_levels(
                roi_data, background_channel, protein_channels
            )
            snr_results = bg_qc.get('signal_to_background', {})
            low_snr_proteins = [
                protein for protein, stats in snr_results.items()
                if stats.get('snr', 0) < self.thresholds.min_snr
            ]
            snr_passed = len(low_snr_proteins) == 0
            assessment['quality_checks']['signal_to_noise'] = {
                'passed': snr_passed,
                'details': bg_qc,
                'low_snr_proteins': low_snr_proteins,
                'threshold': self.thresholds.min_snr
            }
            if not snr_passed:
                assessment['failed_checks'].append(f'low_snr_{len(low_snr_proteins)}_proteins')
                if len(low_snr_proteins) > len(protein_channels) * 0.5:
                    assessment['overall_passed'] = False
                else:
                    assessment['warnings'].append(f'Low SNR in {len(low_snr_proteins)} proteins: {low_snr_proteins}')
            
            # 4. Total Ion Count (TIC) assessment
            tic_qc = monitor_total_ion_counts(
                roi_data, 
                protein_channels,
                {
                    'total_ion_counts': {
                        'min_tic_percentile': self.thresholds.min_tic_percentile,
                        'max_low_tic_pixels_percent': self.thresholds.max_low_tic_pixels_percent
                    }
                }
            )
            tic_passed = not tic_qc['has_acquisition_issues']
            assessment['quality_checks']['total_ion_counts'] = {
                'passed': tic_passed,
                'details': tic_qc,
                'thresholds': {
                    'min_tic_percentile': self.thresholds.min_tic_percentile,
                    'max_low_tic_percent': self.thresholds.max_low_tic_pixels_percent
                }
            }
            if not tic_passed:
                assessment['failed_checks'].append('tic_acquisition_issues')
                assessment['overall_passed'] = False
            
            # 5. Protein detection rate
            detection_rate = self._calculate_protein_detection_rate(roi_data, protein_channels)
            detection_passed = detection_rate >= self.thresholds.min_protein_detection_rate
            assessment['quality_checks']['protein_detection'] = {
                'passed': detection_passed,
                'detection_rate': detection_rate,
                'threshold': self.thresholds.min_protein_detection_rate
            }
            if not detection_passed:
                assessment['failed_checks'].append('low_protein_detection')
                assessment['overall_passed'] = False
            
            # Calculate overall quality score
            assessment['quality_score'] = self._calculate_signal_quality_score(assessment['quality_checks'])
            
            # Generate recommendations
            assessment['recommendations'] = self._generate_signal_recommendations(assessment)
            
            # Log assessment
            status = "PASS" if assessment['overall_passed'] else "FAIL"
            self.logger.info(f"Signal quality assessment for {roi_id}: {status} "
                           f"(score: {assessment['quality_score']:.3f}, "
                           f"failed checks: {len(assessment['failed_checks'])})")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Signal quality assessment failed for {roi_id}: {e}")
            return {
                'roi_id': roi_id,
                'overall_passed': False,
                'error': str(e),
                'quality_score': 0.0,
                'recommendations': ['Repeat signal quality assessment after resolving technical issues']
            }
    
    def _calculate_protein_detection_rate(
        self, 
        roi_data: pd.DataFrame, 
        protein_channels: List[str]
    ) -> float:
        """Calculate fraction of protein channels with detectable signal."""
        detected_proteins = 0
        
        for protein in protein_channels:
            protein_cols = [col for col in roi_data.columns if protein in col]
            if protein_cols:
                signal = roi_data[protein_cols[0]].values
                # Consider protein detected if >5% of pixels have signal above background
                signal_threshold = np.percentile(signal, 95)  # Use 95th percentile as threshold
                detected_pixels = np.sum(signal > signal_threshold * 0.1)  # 10% of max signal
                if detected_pixels > len(signal) * 0.05:  # >5% of pixels
                    detected_proteins += 1
        
        return detected_proteins / len(protein_channels) if protein_channels else 0.0
    
    def _calculate_signal_quality_score(self, quality_checks: Dict[str, Any]) -> float:
        """Calculate overall signal quality score (0-1)."""
        scores = []
        weights = []
        
        # Calibration score
        if 'calibration' in quality_checks:
            cal_details = quality_checks['calibration']['details']
            if cal_details.get('calibration_stable', False):
                # Calculate score based on CV values
                cv_scores = []
                for channel_data in cal_details.get('channels', {}).values():
                    cv = channel_data.get('cv', 1.0)
                    cv_score = max(0.0, 1.0 - cv / self.thresholds.max_calibration_cv)
                    cv_scores.append(cv_score)
                cal_score = np.mean(cv_scores) if cv_scores else 0.0
            else:
                cal_score = 0.0
            scores.append(cal_score)
            weights.append(0.2)
        
        # Carrier gas score
        if 'carrier_gas' in quality_checks:
            gas_details = quality_checks['carrier_gas']['details']
            if gas_details.get('adequate_signal', False):
                median_signal = gas_details.get('median', 0)
                gas_score = min(1.0, median_signal / self.thresholds.min_carrier_gas_signal)
            else:
                gas_score = 0.0
            scores.append(gas_score)
            weights.append(0.2)
        
        # SNR score
        if 'signal_to_noise' in quality_checks:
            snr_details = quality_checks['signal_to_noise']['details']
            snr_data = snr_details.get('signal_to_background', {})
            if snr_data:
                snr_scores = []
                for stats in snr_data.values():
                    snr = stats.get('snr', 0)
                    snr_score = min(1.0, snr / self.thresholds.min_snr)
                    snr_scores.append(snr_score)
                snr_score = np.mean(snr_scores)
            else:
                snr_score = 0.0
            scores.append(snr_score)
            weights.append(0.3)
        
        # TIC score
        if 'total_ion_counts' in quality_checks:
            tic_passed = quality_checks['total_ion_counts']['passed']
            tic_score = 1.0 if tic_passed else 0.0
            scores.append(tic_score)
            weights.append(0.2)
        
        # Detection rate score
        if 'protein_detection' in quality_checks:
            detection_rate = quality_checks['protein_detection']['detection_rate']
            detection_score = detection_rate  # Already 0-1
            scores.append(detection_score)
            weights.append(0.1)
        
        # Calculate weighted average
        if scores and weights:
            return float(np.average(scores, weights=weights))
        else:
            return 0.0
    
    def _generate_signal_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on signal quality assessment."""
        recommendations = []
        
        for failed_check in assessment['failed_checks']:
            if 'calibration' in failed_check:
                recommendations.append("Calibration unstable - check instrument drift and recalibrate")
            elif 'carrier_gas' in failed_check:
                recommendations.append("Low carrier gas signal - check plasma stability and gas flow")
            elif 'low_snr' in failed_check:
                recommendations.append("Low signal-to-noise ratio - optimize staining concentrations and imaging parameters")
            elif 'tic' in failed_check:
                recommendations.append("Total ion count issues - check for acquisition problems or tissue loss")
            elif 'protein_detection' in failed_check:
                recommendations.append("Low protein detection rate - verify antibody panel and staining protocol")
        
        for warning in assessment['warnings']:
            if 'Low SNR' in warning:
                recommendations.append("Consider optimizing staining for low-SNR proteins before analysis")
        
        if not recommendations:
            recommendations.append("Signal quality assessment passed - proceed with analysis")
        
        return recommendations


class BatchEffectDetector:
    """Automated batch effect detection with statistical monitoring."""
    
    def __init__(self, thresholds: BatchEffectThresholds):
        self.thresholds = thresholds
        self.logger = logging.getLogger('BatchEffectDetector')
    
    def detect_batch_effects(
        self,
        batch_data: Dict[str, Dict[str, Any]],
        batch_metadata: Dict[str, Dict[str, Any]],
        calibration_channels: List[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive batch effect detection with automated decisions.
        
        Args:
            batch_data: Ion counts organized by batch
            batch_metadata: Metadata for each batch
            calibration_channels: List of calibration channel names
            
        Returns:
            Dictionary with batch effect assessment and recommendations
        """
        try:
            assessment = {
                'batch_effects_detected': False,
                'affected_batches': [],
                'effect_severity': 'none',
                'recommendations': [],
                'statistical_results': {},
                'temporal_analysis': {},
                'quality_score': 1.0
            }
            
            # 1. Cross-batch consistency analysis
            consistency_results = validate_batch_consistency(batch_data, batch_metadata)
            
            # 2. Calibration drift analysis
            if calibration_channels:
                drift_results = track_calibration_drift(
                    batch_data, 
                    calibration_channels,
                    {
                        'calibration_drift': {
                            'max_drift_percent': self.thresholds.max_drift_percent,
                            'max_cv_across_rois': self.thresholds.max_batch_cv
                        }
                    }
                )
                assessment['calibration_drift'] = drift_results
                
                # Check for significant drift
                significant_drift = any(
                    channel_data.get('has_significant_drift', False)
                    for channel_data in drift_results.values()
                )
                if significant_drift:
                    assessment['batch_effects_detected'] = True
                    assessment['affected_batches'].extend(['drift_detected'])
            
            # 3. Temporal effect analysis
            temporal_analysis = self._analyze_temporal_effects(batch_metadata)
            assessment['temporal_analysis'] = temporal_analysis
            
            if temporal_analysis['significant_temporal_trend']:
                assessment['batch_effects_detected'] = True
                assessment['affected_batches'].extend(temporal_analysis['affected_time_periods'])
            
            # 4. Statistical significance testing
            stat_results = self._statistical_batch_analysis(batch_data)
            assessment['statistical_results'] = stat_results
            
            if stat_results['significant_batch_effects']:
                assessment['batch_effects_detected'] = True
                assessment['affected_batches'].extend(stat_results['affected_batches'])
            
            # 5. Determine effect severity
            assessment['effect_severity'] = self._determine_effect_severity(
                consistency_results, drift_results if calibration_channels else {}, 
                temporal_analysis, stat_results
            )
            
            # 6. Calculate quality score
            assessment['quality_score'] = self._calculate_batch_quality_score(assessment)
            
            # 7. Generate recommendations
            assessment['recommendations'] = self._generate_batch_recommendations(assessment)
            
            # Remove duplicates from affected batches
            assessment['affected_batches'] = list(set(assessment['affected_batches']))
            
            # Log assessment
            status = "DETECTED" if assessment['batch_effects_detected'] else "NONE"
            self.logger.info(f"Batch effect detection: {status} "
                           f"(severity: {assessment['effect_severity']}, "
                           f"quality score: {assessment['quality_score']:.3f})")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Batch effect detection failed: {e}")
            return {
                'batch_effects_detected': True,  # Fail-safe: assume effects present
                'error': str(e),
                'effect_severity': 'unknown',
                'quality_score': 0.0,
                'recommendations': ['Repeat batch effect analysis after resolving technical issues']
            }
    
    def _analyze_temporal_effects(self, batch_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal trends in batch processing."""
        temporal_data = []
        
        for batch_id, metadata in batch_metadata.items():
            if 'acquisition_time' in metadata:
                temporal_data.append({
                    'batch_id': batch_id,
                    'timestamp': metadata['acquisition_time'],
                    'order': len(temporal_data)
                })
        
        if len(temporal_data) < 3:
            return {
                'sufficient_data': False,
                'significant_temporal_trend': False,
                'affected_time_periods': []
            }
        
        # Sort by timestamp if available
        try:
            temporal_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
        except Exception:
            # If timestamp parsing fails, use order
            temporal_data.sort(key=lambda x: x['order'])
        
        # Look for clustering of batches within time windows
        time_periods = []
        current_period = [temporal_data[0]]
        
        for i in range(1, len(temporal_data)):
            # Check if within temporal window
            try:
                curr_time = pd.to_datetime(temporal_data[i]['timestamp'])
                prev_time = pd.to_datetime(current_period[-1]['timestamp'])
                time_diff = (curr_time - prev_time).total_seconds() / 3600  # hours
                
                if time_diff <= self.thresholds.temporal_window_hours:
                    current_period.append(temporal_data[i])
                else:
                    time_periods.append(current_period)
                    current_period = [temporal_data[i]]
            except Exception:
                # If time parsing fails, group by order
                if temporal_data[i]['order'] - current_period[-1]['order'] <= 3:
                    current_period.append(temporal_data[i])
                else:
                    time_periods.append(current_period)
                    current_period = [temporal_data[i]]
        
        time_periods.append(current_period)
        
        # Check for significant clustering
        period_sizes = [len(period) for period in time_periods]
        significant_temporal_trend = (
            len(time_periods) > 1 and 
            max(period_sizes) >= self.thresholds.min_rois_per_batch and
            len([p for p in period_sizes if p >= self.thresholds.min_rois_per_batch]) > 1
        )
        
        affected_periods = []
        if significant_temporal_trend:
            for i, period in enumerate(time_periods):
                if len(period) >= self.thresholds.min_rois_per_batch:
                    affected_periods.append(f"time_period_{i}")
        
        return {
            'sufficient_data': True,
            'n_time_periods': len(time_periods),
            'period_sizes': period_sizes,
            'significant_temporal_trend': significant_temporal_trend,
            'affected_time_periods': affected_periods
        }
    
    def _statistical_batch_analysis(self, batch_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis for batch effects."""
        from scipy import stats
        
        # Extract protein data by batch
        batch_proteins = {}
        all_proteins = set()
        
        for batch_id, data in batch_data.items():
            if isinstance(data, dict):
                # Handle nested structure (batch -> roi -> data)
                batch_protein_data = {}
                for roi_data in data.values():
                    if 'ion_counts' in roi_data:
                        for protein, counts in roi_data['ion_counts'].items():
                            if protein not in batch_protein_data:
                                batch_protein_data[protein] = []
                            batch_protein_data[protein].extend(counts.flatten())
                            all_proteins.add(protein)
                batch_proteins[batch_id] = batch_protein_data
            else:
                # Direct protein data
                batch_proteins[batch_id] = data
                all_proteins.update(data.keys())
        
        if len(batch_proteins) < 2:
            return {
                'sufficient_batches': False,
                'significant_batch_effects': False,
                'affected_batches': []
            }
        
        # Test each protein for batch effects
        significant_proteins = []
        batch_effect_pvalues = {}
        effect_sizes = {}
        
        for protein in all_proteins:
            protein_data_by_batch = []
            batch_ids_for_protein = []
            
            for batch_id, protein_data in batch_proteins.items():
                if protein in protein_data and len(protein_data[protein]) > 0:
                    # Sample data for efficiency
                    data = np.array(protein_data[protein])
                    if len(data) > 1000:
                        data = np.random.choice(data, 1000, replace=False)
                    protein_data_by_batch.append(data)
                    batch_ids_for_protein.append(batch_id)
            
            if len(protein_data_by_batch) >= 2:
                try:
                    # ANOVA test for batch effects
                    f_stat, p_value = stats.f_oneway(*protein_data_by_batch)
                    batch_effect_pvalues[protein] = p_value
                    
                    # Calculate effect size (eta-squared)
                    group_means = [np.mean(group) for group in protein_data_by_batch]
                    overall_mean = np.mean(np.concatenate(protein_data_by_batch))
                    
                    ss_between = sum(len(group) * (mean - overall_mean)**2 
                                   for group, mean in zip(protein_data_by_batch, group_means))
                    ss_total = sum(np.sum((group - overall_mean)**2) 
                                 for group in protein_data_by_batch)
                    
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    effect_sizes[protein] = eta_squared
                    
                    # Check significance
                    if (p_value < self.thresholds.anova_p_threshold and 
                        eta_squared > self.thresholds.effect_size_threshold):
                        significant_proteins.append(protein)
                        
                except Exception:
                    # Skip proteins that cause statistical errors
                    continue
        
        # Determine affected batches
        affected_batches = []
        if significant_proteins:
            # Find batches with most extreme values for significant proteins
            for protein in significant_proteins[:5]:  # Limit to top 5 proteins
                batch_means = {}
                for batch_id, protein_data in batch_proteins.items():
                    if protein in protein_data:
                        batch_means[batch_id] = np.mean(protein_data[protein])
                
                if batch_means:
                    # Find batches with extreme values (outliers)
                    means = list(batch_means.values())
                    batch_ids = list(batch_means.keys())
                    mean_value = np.mean(means)
                    std_value = np.std(means)
                    
                    for batch_id, batch_mean in batch_means.items():
                        if abs(batch_mean - mean_value) > 2 * std_value:  # 2-sigma outlier
                            affected_batches.append(batch_id)
        
        return {
            'sufficient_batches': True,
            'n_proteins_tested': len(all_proteins),
            'n_significant_proteins': len(significant_proteins),
            'significant_proteins': significant_proteins,
            'significant_batch_effects': len(significant_proteins) > 0,
            'batch_effect_pvalues': batch_effect_pvalues,
            'effect_sizes': effect_sizes,
            'affected_batches': list(set(affected_batches))
        }
    
    def _determine_effect_severity(
        self,
        consistency_results: Dict[str, Any],
        drift_results: Dict[str, Any],
        temporal_analysis: Dict[str, Any],
        stat_results: Dict[str, Any]
    ) -> str:
        """Determine overall batch effect severity."""
        # Count significant effects
        effects_detected = 0
        
        if consistency_results.get('batch_effects_detected', False):
            effects_detected += 1
        
        if any(data.get('has_significant_drift', False) for data in drift_results.values()):
            effects_detected += 1
        
        if temporal_analysis.get('significant_temporal_trend', False):
            effects_detected += 1
        
        if stat_results.get('significant_batch_effects', False):
            # Weight statistical effects by number of affected proteins
            n_significant = stat_results.get('n_significant_proteins', 0)
            n_total = stat_results.get('n_proteins_tested', 1)
            if n_significant / n_total > 0.5:
                effects_detected += 2  # Major statistical effects
            elif n_significant > 0:
                effects_detected += 1  # Minor statistical effects
        
        # Determine severity
        if effects_detected == 0:
            return 'none'
        elif effects_detected == 1:
            return 'minor'
        elif effects_detected <= 3:
            return 'moderate'
        else:
            return 'severe'
    
    def _calculate_batch_quality_score(self, assessment: Dict[str, Any]) -> float:
        """Calculate batch quality score (0-1, where 1 is no batch effects)."""
        severity = assessment['effect_severity']
        
        severity_scores = {
            'none': 1.0,
            'minor': 0.8,
            'moderate': 0.5,
            'severe': 0.2
        }
        
        base_score = severity_scores.get(severity, 0.0)
        
        # Adjust based on statistical significance
        stat_results = assessment.get('statistical_results', {})
        if stat_results.get('significant_batch_effects', False):
            n_significant = stat_results.get('n_significant_proteins', 0)
            n_total = stat_results.get('n_proteins_tested', 1)
            significance_penalty = (n_significant / n_total) * 0.3  # Up to 30% penalty
            base_score = max(0.0, base_score - significance_penalty)
        
        return float(base_score)
    
    def _generate_batch_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on batch effect assessment."""
        recommendations = []
        
        severity = assessment['effect_severity']
        
        if severity == 'none':
            recommendations.append("No significant batch effects detected - proceed with analysis")
        elif severity == 'minor':
            recommendations.append("Minor batch effects detected - consider noting in analysis limitations")
        elif severity == 'moderate':
            recommendations.append("Moderate batch effects detected - apply batch correction before analysis")
            recommendations.append("Review experimental protocol for sources of variation")
        elif severity == 'severe':
            recommendations.append("Severe batch effects detected - batch correction strongly recommended")
            recommendations.append("Consider re-processing samples with improved protocol standardization")
            recommendations.append("Evaluate whether analysis should proceed with current data")
        
        # Specific recommendations based on detected effects
        if 'calibration_drift' in assessment and assessment['calibration_drift']:
            drift_channels = [
                ch for ch, data in assessment['calibration_drift'].items()
                if data.get('has_significant_drift', False)
            ]
            if drift_channels:
                recommendations.append(f"Calibration drift detected in {', '.join(drift_channels)} - recalibrate instrument")
        
        temporal_analysis = assessment.get('temporal_analysis', {})
        if temporal_analysis.get('significant_temporal_trend', False):
            recommendations.append("Temporal batch clustering detected - randomize sample processing order")
        
        stat_results = assessment.get('statistical_results', {})
        if stat_results.get('significant_batch_effects', False):
            significant_proteins = stat_results.get('significant_proteins', [])
            if significant_proteins:
                recommendations.append(f"Statistical batch effects in proteins: {', '.join(significant_proteins[:5])}")
        
        return recommendations


class AutomaticQCSystem:
    """Main automatic QC system coordinating all quality assessments."""
    
    def __init__(self, config: AutomaticQCConfig, output_dir: str = "qc_results"):
        """Initialize automatic QC system."""
        config.validate()
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize QC components
        self.tissue_coverage = TissueCoverageAssessment(config.tissue_coverage)
        self.signal_quality = SignalQualityGates(config.signal_quality)
        self.batch_effects = BatchEffectDetector(config.batch_effects)
        
        # Initialize quality monitoring
        if config.enable_statistical_monitoring:
            self.quality_monitor = QualityMonitor()
            
            # Create quality gate engine with appropriate thresholds
            self.gate_engine = QualityGateEngine()
        else:
            self.quality_monitor = None
            self.gate_engine = None
        
        self.logger = logging.getLogger('AutomaticQCSystem')
        
        # Track QC results
        self.qc_history = []
        self.batch_qc_results = {}
    
    def run_comprehensive_qc(
        self,
        roi_data: Dict[str, Any],
        roi_id: str,
        batch_id: str = "default",
        channel_config: Dict[str, List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive QC assessment on single ROI.
        
        Args:
            roi_data: ROI data dictionary (coords, ion_counts, dna_intensities, raw_data)
            roi_id: ROI identifier
            batch_id: Batch identifier
            channel_config: Channel configuration (protein_channels, calibration_channels, etc.)
            
        Returns:
            Comprehensive QC assessment with pass/fail decisions
        """
        self.logger.info(f"Starting comprehensive QC for ROI {roi_id}")
        
        # Initialize result structure
        qc_result = {
            'roi_id': roi_id,
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'overall_passed': True,
            'critical_failures': [],
            'warnings': [],
            'assessments': {},
            'quality_scores': {},
            'recommendations': [],
            'config_used': self.config.__dict__
        }
        
        # Set default channel configuration
        if channel_config is None:
            channel_config = {
                'protein_channels': ['CD45', 'CD31', 'CD11b', 'CD206'],
                'calibration_channels': ['130Ba', '131Xe'],
                'carrier_gas_channel': '80ArAr',
                'background_channel': '190BCKG'
            }
        
        try:
            # 1. Tissue Coverage Assessment
            if self.config.fail_on_tissue_coverage:
                self.logger.debug(f"Running tissue coverage assessment for {roi_id}")
                
                coords = roi_data.get('coords', np.array([]))
                dna_intensities = {
                    'DNA1': roi_data.get('dna1_intensities', np.array([])),
                    'DNA2': roi_data.get('dna2_intensities', np.array([]))
                }
                
                coverage_assessment = self.tissue_coverage.assess_coverage(
                    coords, dna_intensities, roi_id
                )
                qc_result['assessments']['tissue_coverage'] = coverage_assessment
                qc_result['quality_scores']['tissue_coverage'] = coverage_assessment.get('quality_score', 0.0)
                
                if not coverage_assessment['passed']:
                    qc_result['critical_failures'].append('tissue_coverage_inadequate')
                    qc_result['overall_passed'] = False
                
                qc_result['recommendations'].extend(coverage_assessment.get('recommendations', []))
            
            # 2. Signal Quality Assessment
            if self.config.fail_on_signal_quality:
                self.logger.debug(f"Running signal quality assessment for {roi_id}")
                
                raw_data = roi_data.get('raw_data')
                if raw_data is not None and isinstance(raw_data, pd.DataFrame):
                    signal_assessment = self.signal_quality.assess_signal_quality(
                        roi_data=raw_data,
                        protein_channels=channel_config['protein_channels'],
                        calibration_channels=channel_config['calibration_channels'],
                        carrier_gas_channel=channel_config['carrier_gas_channel'],
                        background_channel=channel_config['background_channel'],
                        roi_id=roi_id
                    )
                    qc_result['assessments']['signal_quality'] = signal_assessment
                    qc_result['quality_scores']['signal_quality'] = signal_assessment.get('quality_score', 0.0)
                    
                    if not signal_assessment['overall_passed']:
                        qc_result['critical_failures'].extend(signal_assessment.get('failed_checks', []))
                        qc_result['overall_passed'] = False
                    
                    qc_result['warnings'].extend(signal_assessment.get('warnings', []))
                    qc_result['recommendations'].extend(signal_assessment.get('recommendations', []))
                else:
                    qc_result['warnings'].append('Raw data not available for signal quality assessment')
            
            # 3. Statistical Quality Monitoring (if enabled)
            if self.config.enable_statistical_monitoring and self.quality_monitor is not None:
                self.logger.debug(f"Running statistical quality monitoring for {roi_id}")
                
                # Create quality metrics from assessments
                coordinate_quality = qc_result['quality_scores'].get('tissue_coverage', 0.5)
                ion_count_quality = qc_result['quality_scores'].get('signal_quality', 0.5)
                biological_quality = 0.8  # Placeholder - would come from biological validation
                
                quality_metrics = QualityMetrics(
                    roi_id=roi_id,
                    batch_id=batch_id,
                    timestamp=qc_result['timestamp'],
                    coordinate_quality=coordinate_quality,
                    ion_count_quality=ion_count_quality,
                    biological_quality=biological_quality,
                    n_pixels=len(roi_data.get('coords', [])),
                    n_proteins=len(channel_config['protein_channels']),
                    protein_completeness=1.0  # Would calculate from actual data
                )
                
                # Add to quality monitor
                self.quality_monitor.add_roi_quality(quality_metrics)
                
                # Check against control limits
                if len(self.quality_monitor.quality_history) >= 10:
                    self.quality_monitor.update_control_limits()
                    quality_status = self.quality_monitor.check_quality_status(quality_metrics)
                    qc_result['assessments']['statistical_monitoring'] = quality_status
                    
                    if quality_status['overall_status'] == 'out_of_control':
                        qc_result['warnings'].append('Quality metrics out of statistical control')
            
            # 4. Calculate overall quality score
            quality_scores = list(qc_result['quality_scores'].values())
            if quality_scores:
                qc_result['overall_quality_score'] = float(np.mean(quality_scores))
            else:
                qc_result['overall_quality_score'] = 0.0
            
            # 5. Generate final recommendations
            if qc_result['overall_passed']:
                qc_result['recommendations'].insert(0, "All QC checks passed - ROI suitable for analysis")
            else:
                qc_result['recommendations'].insert(0, "QC failures detected - address issues before analysis")
            
            # Store in history
            self.qc_history.append(qc_result)
            
            # Store batch-level information
            if batch_id not in self.batch_qc_results:
                self.batch_qc_results[batch_id] = []
            self.batch_qc_results[batch_id].append({
                'roi_id': roi_id,
                'passed': qc_result['overall_passed'],
                'quality_score': qc_result['overall_quality_score']
            })
            
            self.logger.info(f"QC completed for {roi_id}: {'PASS' if qc_result['overall_passed'] else 'FAIL'} "
                           f"(score: {qc_result['overall_quality_score']:.3f})")
            
            return qc_result
            
        except Exception as e:
            self.logger.error(f"QC assessment failed for {roi_id}: {e}")
            error_result = {
                'roi_id': roi_id,
                'batch_id': batch_id,
                'timestamp': datetime.now().isoformat(),
                'overall_passed': False,
                'error': str(e),
                'overall_quality_score': 0.0,
                'recommendations': ['Repeat QC assessment after resolving technical issues']
            }
            self.qc_history.append(error_result)
            return error_result
    
    def run_batch_qc_analysis(
        self,
        batch_data: Dict[str, Dict[str, Any]],
        batch_metadata: Dict[str, Dict[str, Any]] = None,
        channel_config: Dict[str, List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run batch-level QC analysis including batch effect detection.
        
        Args:
            batch_data: Data organized by batch_id -> roi_id -> data
            batch_metadata: Metadata for each batch
            channel_config: Channel configuration
            
        Returns:
            Batch-level QC assessment
        """
        self.logger.info(f"Starting batch QC analysis for {len(batch_data)} batches")
        
        if batch_metadata is None:
            batch_metadata = {batch_id: {} for batch_id in batch_data.keys()}
        
        if channel_config is None:
            channel_config = {
                'calibration_channels': ['130Ba', '131Xe']
            }
        
        # Run batch effect detection
        batch_assessment = self.batch_effects.detect_batch_effects(
            batch_data=batch_data,
            batch_metadata=batch_metadata,
            calibration_channels=channel_config.get('calibration_channels', [])
        )
        
        # Add batch-level statistics
        batch_stats = {}
        for batch_id, rois in self.batch_qc_results.items():
            if rois:
                pass_rate = sum(1 for roi in rois if roi['passed']) / len(rois)
                avg_quality = np.mean([roi['quality_score'] for roi in rois])
                batch_stats[batch_id] = {
                    'n_rois': len(rois),
                    'pass_rate': pass_rate,
                    'average_quality_score': avg_quality,
                    'passed_batch_qc': pass_rate >= 0.8  # 80% pass rate threshold
                }
        
        batch_assessment['batch_statistics'] = batch_stats
        batch_assessment['overall_batch_quality'] = self._calculate_overall_batch_quality(batch_stats)
        
        self.logger.info(f"Batch QC completed: effects {'detected' if batch_assessment['batch_effects_detected'] else 'not detected'}")
        
        return batch_assessment
    
    def _calculate_overall_batch_quality(self, batch_stats: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall batch quality score."""
        if not batch_stats:
            return 0.0
        
        # Weight by number of ROIs in each batch
        total_rois = sum(stats['n_rois'] for stats in batch_stats.values())
        if total_rois == 0:
            return 0.0
        
        weighted_quality = sum(
            stats['average_quality_score'] * stats['n_rois']
            for stats in batch_stats.values()
        )
        
        return weighted_quality / total_rois
    
    def generate_qc_report(
        self,
        include_plots: bool = None,
        report_format: str = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive QC report with recommendations.
        
        Args:
            include_plots: Whether to include QC plots
            report_format: Report format ("json", "html", "both")
            
        Returns:
            Comprehensive QC report
        """
        if include_plots is None:
            include_plots = self.config.save_qc_plots
        
        if report_format is None:
            report_format = self.config.report_format
        
        self.logger.info("Generating comprehensive QC report")
        
        # Aggregate QC statistics
        total_rois = len(self.qc_history)
        passed_rois = sum(1 for qc in self.qc_history if qc.get('overall_passed', False))
        pass_rate = passed_rois / total_rois if total_rois > 0 else 0.0
        
        quality_scores = [qc.get('overall_quality_score', 0.0) for qc in self.qc_history]
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # Failure analysis
        failure_types = {}
        for qc in self.qc_history:
            for failure in qc.get('critical_failures', []):
                failure_types[failure] = failure_types.get(failure, 0) + 1
        
        # Warning analysis  
        warning_types = {}
        for qc in self.qc_history:
            for warning in qc.get('warnings', []):
                warning_types[warning] = warning_types.get(warning, 0) + 1
        
        # Generate report
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'qc_system_version': '1.0',
                'config_used': self.config.__dict__,
                'total_rois_assessed': total_rois
            },
            'overall_statistics': {
                'pass_rate': pass_rate,
                'average_quality_score': avg_quality,
                'quality_score_distribution': {
                    'min': float(np.min(quality_scores)) if quality_scores else 0.0,
                    'max': float(np.max(quality_scores)) if quality_scores else 0.0,
                    'std': float(np.std(quality_scores)) if quality_scores else 0.0,
                    'percentiles': {
                        '25': float(np.percentile(quality_scores, 25)) if quality_scores else 0.0,
                        '50': float(np.percentile(quality_scores, 50)) if quality_scores else 0.0,
                        '75': float(np.percentile(quality_scores, 75)) if quality_scores else 0.0
                    }
                }
            },
            'failure_analysis': {
                'common_failure_types': failure_types,
                'failure_rate_by_type': {
                    failure_type: count / total_rois for failure_type, count in failure_types.items()
                } if total_rois > 0 else {}
            },
            'warning_analysis': {
                'common_warning_types': warning_types,
                'warning_rate_by_type': {
                    warning_type: count / total_rois for warning_type, count in warning_types.items()
                } if total_rois > 0 else {}
            },
            'batch_analysis': self.batch_qc_results,
            'recommendations': self._generate_overall_recommendations(pass_rate, failure_types, warning_types),
            'detailed_roi_results': self.qc_history if self.config.generate_detailed_reports else []
        }
        
        # Save report
        report_path = self.output_dir / f"qc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if report_format in ["json", "both"]:
            json_path = report_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"QC report saved to {json_path}")
        
        if report_format in ["html", "both"]:
            html_path = report_path.with_suffix('.html')
            self._generate_html_report(report, html_path)
            self.logger.info(f"HTML QC report saved to {html_path}")
        
        return report
    
    def _generate_overall_recommendations(
        self,
        pass_rate: float,
        failure_types: Dict[str, int],
        warning_types: Dict[str, int]
    ) -> List[str]:
        """Generate overall QC recommendations."""
        recommendations = []
        
        # Pass rate recommendations
        if pass_rate >= 0.9:
            recommendations.append("Excellent QC performance - current protocols are working well")
        elif pass_rate >= 0.7:
            recommendations.append("Good QC performance - minor protocol improvements may be beneficial")
        elif pass_rate >= 0.5:
            recommendations.append("Moderate QC performance - review protocols and address common failure modes")
        else:
            recommendations.append("Poor QC performance - comprehensive protocol review and optimization needed")
        
        # Specific failure mode recommendations
        if 'tissue_coverage_inadequate' in failure_types:
            recommendations.append("High tissue coverage failure rate - review sectioning and mounting protocols")
        
        if any('signal_quality' in failure for failure in failure_types):
            recommendations.append("Signal quality issues detected - review staining and imaging protocols")
        
        if any('calibration' in failure for failure in failure_types):
            recommendations.append("Calibration instability detected - implement more frequent calibration checks")
        
        if any('batch' in warning for warning in warning_types):
            recommendations.append("Batch effect warnings - consider implementing batch correction procedures")
        
        # Statistical monitoring recommendations
        if len(failure_types) > 5:
            recommendations.append("Multiple failure modes detected - systematic protocol review recommended")
        
        return recommendations
    
    def _generate_html_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Generate HTML version of QC report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>IMC QC Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }}
                .pass {{ background-color: #d4edda; }}
                .warn {{ background-color: #fff3cd; }}
                .fail {{ background-color: #f8d7da; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>IMC Quality Control Report</h1>
                <p>Generated: {report['report_metadata']['generated_at']}</p>
                <p>Total ROIs Assessed: {report['report_metadata']['total_rois_assessed']}</p>
            </div>
            
            <div class="section">
                <h2>Overall Statistics</h2>
                <div class="metric {'pass' if report['overall_statistics']['pass_rate'] >= 0.8 else 'warn' if report['overall_statistics']['pass_rate'] >= 0.6 else 'fail'}">
                    <strong>Pass Rate:</strong> {report['overall_statistics']['pass_rate']:.1%}
                </div>
                <div class="metric">
                    <strong>Average Quality Score:</strong> {report['overall_statistics']['average_quality_score']:.3f}
                </div>
            </div>
            
            <div class="section">
                <h2>Failure Analysis</h2>
                <table>
                    <tr><th>Failure Type</th><th>Count</th><th>Rate</th></tr>
        """
        
        for failure_type, count in report['failure_analysis']['common_failure_types'].items():
            rate = report['failure_analysis']['failure_rate_by_type'].get(failure_type, 0)
            html_content += f"<tr><td>{failure_type}</td><td>{count}</td><td>{rate:.1%}</td></tr>"
        
        html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        for rec in report['recommendations']:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)


def create_automatic_qc_system(
    config_dict: Dict[str, Any] = None,
    output_dir: str = "qc_results"
) -> AutomaticQCSystem:
    """
    Factory function to create automatic QC system.
    
    Args:
        config_dict: Configuration dictionary
        output_dir: Output directory for QC results
        
    Returns:
        Configured automatic QC system
    """
    if config_dict is None:
        config = AutomaticQCConfig()
    else:
        # Parse config dictionary
        config = AutomaticQCConfig(
            tissue_coverage=TissueCoverageThresholds(**config_dict.get('tissue_coverage', {})),
            signal_quality=SignalQualityThresholds(**config_dict.get('signal_quality', {})),
            batch_effects=BatchEffectThresholds(**config_dict.get('batch_effects', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['tissue_coverage', 'signal_quality', 'batch_effects']}
        )
    
    return AutomaticQCSystem(config, output_dir)


# Example usage and integration
def run_automatic_qc_pipeline(
    roi_data_dict: Dict[str, Dict[str, Any]],
    batch_assignments: Dict[str, str] = None,
    channel_config: Dict[str, List[str]] = None,
    qc_config: Dict[str, Any] = None,
    output_dir: str = "qc_results"
) -> Dict[str, Any]:
    """
    Run complete automatic QC pipeline on multiple ROIs.
    
    Args:
        roi_data_dict: Dictionary mapping roi_id -> roi_data
        batch_assignments: Dictionary mapping roi_id -> batch_id
        channel_config: Channel configuration
        qc_config: QC configuration dictionary
        output_dir: Output directory
        
    Returns:
        Complete QC pipeline results
    """
    # Create QC system
    qc_system = create_automatic_qc_system(qc_config, output_dir)
    
    # Default batch assignments
    if batch_assignments is None:
        batch_assignments = {roi_id: "default" for roi_id in roi_data_dict.keys()}
    
    # Run QC on each ROI
    roi_qc_results = {}
    for roi_id, roi_data in roi_data_dict.items():
        batch_id = batch_assignments.get(roi_id, "default")
        qc_result = qc_system.run_comprehensive_qc(
            roi_data=roi_data,
            roi_id=roi_id,
            batch_id=batch_id,
            channel_config=channel_config
        )
        roi_qc_results[roi_id] = qc_result
    
    # Organize data by batch for batch-level analysis
    batch_data = {}
    batch_metadata = {}
    for roi_id, roi_data in roi_data_dict.items():
        batch_id = batch_assignments.get(roi_id, "default")
        if batch_id not in batch_data:
            batch_data[batch_id] = {}
            batch_metadata[batch_id] = {}
        batch_data[batch_id][roi_id] = roi_data
    
    # Run batch-level QC
    batch_qc_results = qc_system.run_batch_qc_analysis(
        batch_data=batch_data,
        batch_metadata=batch_metadata,
        channel_config=channel_config
    )
    
    # Generate comprehensive report
    qc_report = qc_system.generate_qc_report()
    
    return {
        'roi_qc_results': roi_qc_results,
        'batch_qc_results': batch_qc_results,
        'comprehensive_report': qc_report,
        'overall_pass_rate': qc_report['overall_statistics']['pass_rate'],
        'recommendations': qc_report['recommendations']
    }