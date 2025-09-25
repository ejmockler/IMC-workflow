"""
Marker Presence Validation and Audit Trail System

Ensures all configured markers are properly tracked through the analysis pipeline
and provides comprehensive audit trails for scientific reproducibility.
"""

import re
import warnings
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from ..utils.column_matching import ColumnMatcher, MatchingConfig


@dataclass
class MarkerAuditResult:
    """Results of marker presence audit."""
    configured_markers: List[str]
    found_markers: List[str]
    missing_markers: List[str]
    ambiguous_matches: Dict[str, List[str]]
    column_mappings: Dict[str, str]
    warnings: List[str]
    errors: List[str]


class MarkerValidator:
    """Validates marker presence and provides audit trails."""
    
    def __init__(self, config: 'Config'):
        """Initialize with configuration."""
        self.config = config
        self.logger = logging.getLogger('MarkerValidator')
        
        # Initialize robust column matcher
        self.column_matcher = ColumnMatcher(MatchingConfig(
            case_sensitive=False,
            fuzzy_threshold=0.8,
            custom_aliases=config.channels.get('marker_aliases', {})
        ))
        
        # Build comprehensive marker lists
        self.protein_channels = config.channels.get('protein_channels', [])
        self.dna_channels = config.channels.get('dna_channels', [])
        self.all_required_channels = self.protein_channels + self.dna_channels
        
        # Get critical markers that must not be silently excluded
        viz_config = config.visualization.get('validation_plots', {})
        self.critical_markers = set()
        
        # Add primary markers
        primary_markers = viz_config.get('primary_markers', {})
        self.critical_markers.update(primary_markers.values())
        
        # Add always_include markers
        always_include = viz_config.get('always_include', [])
        self.critical_markers.update(always_include)
        
        # Add markers from channel groups
        channel_groups = config.channel_groups
        for group_name, group_data in channel_groups.items():
            if isinstance(group_data, dict):
                for subgroup_name, markers in group_data.items():
                    if isinstance(markers, list):
                        self.critical_markers.update(markers)
            elif isinstance(group_data, list):
                self.critical_markers.update(group_data)
        
        self.logger.info(f"Initialized validator for {len(self.protein_channels)} protein channels")
        self.logger.info(f"Critical markers: {sorted(self.critical_markers)}")
    
    def audit_roi_file(self, roi_file_path: str, roi_data: pd.DataFrame) -> MarkerAuditResult:
        """
        Perform comprehensive audit of marker presence in ROI file.
        
        Args:
            roi_file_path: Path to ROI file being audited
            roi_data: Loaded ROI DataFrame
            
        Returns:
            Detailed audit result
        """
        self.logger.debug(f"Auditing markers in {roi_file_path}")
        
        result = MarkerAuditResult(
            configured_markers=self.all_required_channels.copy(),
            found_markers=[],
            missing_markers=[],
            ambiguous_matches={},
            column_mappings={},
            warnings=[],
            errors=[]
        )
        
        available_columns = list(roi_data.columns)
        self.logger.debug(f"Available columns: {available_columns}")
        
        # Use robust column matching for all markers
        matching_results = self.column_matcher.match_multiple_markers(
            self.all_required_channels, available_columns
        )
        
        # Generate matching report
        matching_report = self.column_matcher.generate_matching_report(
            matching_results, available_columns
        )
        self.logger.info(f"Column matching report: {matching_report['success_rate']:.1%} success rate")
        
        # Process matching results
        for marker in self.all_required_channels:
            match = matching_results.get(marker)
            
            if match is None:
                result.missing_markers.append(marker)
                if marker in self.critical_markers:
                    error_msg = f"CRITICAL: Required marker '{marker}' not found in {roi_file_path}"
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
                else:
                    warning_msg = f"Marker '{marker}' not found in {roi_file_path}"
                    result.warnings.append(warning_msg)
                    self.logger.warning(warning_msg)
            
            else:
                result.found_markers.append(marker)
                result.column_mappings[marker] = match.matched_column
                
                # Log match details
                self.logger.debug(f"Marker '{marker}' → column '{match.matched_column}' "
                                f"(confidence: {match.confidence:.2f}, type: {match.match_type})")
                
                # Warn about low confidence matches
                if match.confidence < 0.9:
                    warning_msg = f"Low confidence match for '{marker}': '{match.matched_column}' " \
                                f"(confidence: {match.confidence:.2f})"
                    result.warnings.append(warning_msg)
                    self.logger.warning(warning_msg)
        
        return result
    
    
    def validate_ion_counts(self, ion_counts: Dict[str, np.ndarray], roi_id: str) -> MarkerAuditResult:
        """
        Validate processed ion counts after data loading.
        
        Args:
            ion_counts: Processed ion count data
            roi_id: ROI identifier for logging
            
        Returns:
            Validation result
        """
        result = MarkerAuditResult(
            configured_markers=self.protein_channels.copy(),
            found_markers=list(ion_counts.keys()),
            missing_markers=[],
            ambiguous_matches={},
            column_mappings={},
            warnings=[],
            errors=[]
        )
        
        # Check for missing markers
        found_set = set(ion_counts.keys())
        configured_set = set(self.protein_channels)
        
        result.missing_markers = list(configured_set - found_set)
        
        # Check critical markers
        for marker in self.critical_markers:
            if marker not in found_set:
                error_msg = f"CRITICAL: Marker '{marker}' missing from processed data for {roi_id}"
                result.errors.append(error_msg)
                self.logger.error(error_msg)
        
        # Check for zero-only data (potential processing issues)
        for marker, data in ion_counts.items():
            if isinstance(data, np.ndarray):
                non_zero_count = np.count_nonzero(data)
                if non_zero_count == 0:
                    warning_msg = f"Marker '{marker}' has all-zero data in {roi_id}"
                    result.warnings.append(warning_msg)
                    self.logger.warning(warning_msg)
                elif non_zero_count < len(data) * 0.01:  # Less than 1% non-zero
                    warning_msg = f"Marker '{marker}' has very sparse data ({non_zero_count}/{len(data)} non-zero) in {roi_id}"
                    result.warnings.append(warning_msg)
                    self.logger.warning(warning_msg)
        
        return result
    
    def validate_analysis_results(self, results: Dict[str, Any], roi_id: str) -> MarkerAuditResult:
        """
        Validate that all markers made it through the analysis pipeline.
        
        Args:
            results: Analysis results dictionary
            roi_id: ROI identifier
            
        Returns:
            Validation result
        """
        result = MarkerAuditResult(
            configured_markers=self.protein_channels.copy(),
            found_markers=[],
            missing_markers=[],
            ambiguous_matches={},
            column_mappings={},
            warnings=[],
            errors=[]
        )
        
        # Check multiscale results for marker presence
        multiscale_results = results.get('multiscale_results', {})
        
        for scale, scale_result in multiscale_results.items():
            transformed_arrays = scale_result.get('transformed_arrays', {})
            if transformed_arrays:
                found_markers = set(transformed_arrays.keys())
                result.found_markers = list(found_markers)
                
                # Check for missing critical markers
                missing_critical = self.critical_markers - found_markers
                for marker in missing_critical:
                    error_msg = f"CRITICAL: Marker '{marker}' missing from analysis results at scale {scale} for {roi_id}"
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
                
                break  # Only need to check one scale for marker presence
        
        return result
    
    def create_audit_summary(self, audit_results: List[MarkerAuditResult], output_path: str) -> Dict[str, Any]:
        """
        Create comprehensive audit summary across all ROIs.
        
        Args:
            audit_results: List of audit results from all ROIs
            output_path: Path to save audit summary
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            'total_rois_audited': len(audit_results),
            'configured_markers': self.all_required_channels,
            'critical_markers': list(self.critical_markers),
            'marker_success_rate': {},
            'total_errors': 0,
            'total_warnings': 0,
            'roi_with_errors': [],
            'roi_with_missing_critical': []
        }
        
        # Aggregate statistics
        marker_found_count = {marker: 0 for marker in self.all_required_channels}
        
        for i, result in enumerate(audit_results):
            summary['total_errors'] += len(result.errors)
            summary['total_warnings'] += len(result.warnings)
            
            if result.errors:
                summary['roi_with_errors'].append(i)
            
            # Check for missing critical markers
            missing_critical = set(self.critical_markers) & set(result.missing_markers)
            if missing_critical:
                summary['roi_with_missing_critical'].append(i)
            
            # Count successful markers
            for marker in result.found_markers:
                if marker in marker_found_count:
                    marker_found_count[marker] += 1
        
        # Calculate success rates
        total_rois = len(audit_results)
        for marker, found_count in marker_found_count.items():
            summary['marker_success_rate'][marker] = found_count / total_rois if total_rois > 0 else 0
        
        # Save summary to file
        import json
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Audit summary saved to {output_path}")
        return summary


def validate_marker_presence_assertion(ion_counts: Dict[str, np.ndarray], 
                                      required_markers: List[str], 
                                      roi_id: str) -> None:
    """
    Assert that all required markers are present. Fail loudly if not.
    
    Args:
        ion_counts: Processed ion count data
        required_markers: List of markers that must be present
        roi_id: ROI identifier for error messages
        
    Raises:
        AssertionError: If any required marker is missing
    """
    found_markers = set(ion_counts.keys())
    missing_markers = set(required_markers) - found_markers
    
    if missing_markers:
        raise AssertionError(
            f"Missing required markers {list(missing_markers)} in ROI {roi_id}. "
            f"Found: {list(found_markers)}"
        )


def log_marker_audit_result(result: MarkerAuditResult, roi_id: str) -> None:
    """Log audit result in a structured format."""
    logger = logging.getLogger('MarkerAudit')
    
    logger.info(f"=== Marker Audit for {roi_id} ===")
    logger.info(f"Configured: {len(result.configured_markers)} markers")
    logger.info(f"Found: {len(result.found_markers)} markers")
    logger.info(f"Missing: {len(result.missing_markers)} markers")
    
    if result.missing_markers:
        logger.warning(f"Missing markers: {result.missing_markers}")
    
    if result.ambiguous_matches:
        logger.warning(f"Ambiguous matches: {result.ambiguous_matches}")
    
    if result.errors:
        logger.error(f"Validation errors: {result.errors}")
    
    if result.warnings:
        logger.warning(f"Validation warnings: {result.warnings}")
    
    logger.info("=== End Marker Audit ===")