"""
Unified High-Performance Validation Pipeline

Combines the scientific validation framework with high-performance processing
to eliminate the fragmentation between validation systems.

Provides:
- Scientific validation rules from framework.py
- High-performance streaming and caching
- Single source of truth for validation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import json

from .framework import (
    create_validation_suite, 
    ValidationCategory, 
    ValidationSeverity,
    ValidationSuiteResult
)


class UnifiedValidationPipeline:
    """High-performance validation using the scientific validation framework."""
    
    def __init__(self, max_workers: int = None, cache_enabled: bool = True):
        """Initialize unified validation pipeline.
        
        Args:
            max_workers: Number of parallel workers (default: CPU count)
            cache_enabled: Whether to enable result caching
        """
        self.max_workers = max_workers or 4
        self.cache_enabled = cache_enabled
        self.cache = {}
        self.logger = logging.getLogger(__name__)
        
        # Create the scientific validation suite
        self.validation_suite = create_validation_suite()
        
    def validate_single_roi(self, roi_file: Path) -> Dict[str, Any]:
        """Validate a single ROI file.
        
        Args:
            roi_file: Path to ROI file
            
        Returns:
            Validation result dictionary
        """
        # Check cache first
        if self.cache_enabled:
            cache_key = f"{roi_file.name}_{roi_file.stat().st_mtime}"
            if cache_key in self.cache:
                self.logger.debug(f"Cache hit for {roi_file.name}")
                return self.cache[cache_key]
        
        try:
            # Load ROI data efficiently
            roi_data = pd.read_csv(roi_file, sep='\t')
            
            # Extract coordinates
            if 'X' not in roi_data.columns or 'Y' not in roi_data.columns:
                return {
                    'roi_file': roi_file.name,
                    'status': 'critical',
                    'severity': ValidationSeverity.CRITICAL,
                    'message': 'Missing coordinate columns (X, Y)',
                    'execution_time_ms': 0.0
                }
            
            coords = roi_data[['X', 'Y']].values
            
            # Check for protein channels (handle both simple and element notation)
            protein_channels = ['CD45', 'CD11b', 'CD31', 'CD140a', 'CD140b', 'CD206']
            available_proteins = []
            missing_proteins = []
            
            for protein in protein_channels:
                # Find column with this protein name (with or without element notation)
                matching_cols = [col for col in roi_data.columns if protein in col]
                if matching_cols:
                    available_proteins.append((protein, matching_cols[0]))
                else:
                    missing_proteins.append(protein)
            
            if missing_proteins:
                return {
                    'roi_file': roi_file.name,
                    'status': 'critical',
                    'severity': ValidationSeverity.CRITICAL,
                    'message': f'Missing protein channels: {missing_proteins}',
                    'execution_time_ms': 0.0
                }
            
            # Run scientific validation suite
            start_time = time.time()
            
            # Prepare context for validation
            protein_data = {}
            for protein_name, column_name in available_proteins:
                protein_data[protein_name] = roi_data[column_name].values
            
            validation_context = {
                'coordinates': coords,
                'protein_data': protein_data,
                'dna_data': {
                    'DNA1': roi_data.get('DNA1(Ir191Di)', roi_data.get('DNA1', np.zeros(len(roi_data)))).values,
                    'DNA2': roi_data.get('DNA2(Ir193Di)', roi_data.get('DNA2', np.zeros(len(roi_data)))).values
                },
                'metadata': {
                    'file_path': str(roi_file),
                    'n_measurements': len(roi_data),
                    'spatial_extent': {
                        'x_range': coords[:, 0].max() - coords[:, 0].min(),
                        'y_range': coords[:, 1].max() - coords[:, 1].min()
                    }
                }
            }
            
            # Execute validation suite
            suite_result = self.validation_suite.validate(validation_context)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Convert to standardized format
            result = {
                'roi_file': roi_file.name,
                'status': self._determine_status(suite_result),
                'severity': self._determine_severity(suite_result),
                'validation_results': suite_result.to_dict(),
                'execution_time_ms': execution_time,
                'critical_count': len([r for r in suite_result.results if r.severity == ValidationSeverity.CRITICAL]),
                'warning_count': len([r for r in suite_result.results if r.severity == ValidationSeverity.WARNING])
            }
            
            # Cache result
            if self.cache_enabled:
                self.cache[cache_key] = result
                
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed for {roi_file.name}: {e}")
            return {
                'roi_file': roi_file.name,
                'status': 'critical',
                'severity': ValidationSeverity.CRITICAL,
                'message': f'Validation error: {str(e)}',
                'execution_time_ms': 0.0
            }
    
    def _determine_status(self, suite_result: ValidationSuiteResult) -> str:
        """Determine overall status from suite result."""
        if any(r.severity == ValidationSeverity.CRITICAL for r in suite_result.results):
            return 'critical'
        elif any(r.severity == ValidationSeverity.WARNING for r in suite_result.results):
            return 'warning'
        else:
            return 'pass'
    
    def _determine_severity(self, suite_result: ValidationSuiteResult) -> ValidationSeverity:
        """Determine highest severity from suite result."""
        severities = [r.severity for r in suite_result.results]
        if ValidationSeverity.CRITICAL in severities:
            return ValidationSeverity.CRITICAL
        elif ValidationSeverity.WARNING in severities:
            return ValidationSeverity.WARNING
        else:
            return ValidationSeverity.PASS
    
    def validate_dataset(self, roi_files: List[Path]) -> Dict[str, Any]:
        """Validate entire dataset with high performance.
        
        Args:
            roi_files: List of ROI files to validate
            
        Returns:
            Comprehensive validation report
        """
        start_time = time.time()
        self.logger.info(f"Starting unified validation of {len(roi_files)} ROI files")
        
        # Process ROIs in parallel for speed
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.validate_single_roi, roi_file) for roi_file in roi_files]
            
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=60)  # 1 minute per ROI max
                    results.append(result)
                    
                    # Progress logging
                    if (i + 1) % 5 == 0 or i == len(futures) - 1:
                        self.logger.info(f"Validated {i + 1}/{len(roi_files)} ROIs")
                        
                except Exception as e:
                    self.logger.error(f"ROI validation failed: {e}")
                    results.append({
                        'roi_file': 'unknown',
                        'status': 'critical',
                        'severity': ValidationSeverity.CRITICAL,
                        'message': f'Processing error: {str(e)}',
                        'execution_time_ms': 0.0
                    })
        
        # Compile summary
        total_time = (time.time() - start_time) * 1000
        
        critical_count = sum(1 for r in results if r.get('status') == 'critical')
        warning_count = sum(1 for r in results if r.get('status') == 'warning')
        pass_count = sum(1 for r in results if r.get('status') == 'pass')
        
        summary = {
            'total_rois': len(roi_files),
            'critical_errors': critical_count,
            'warnings': warning_count,
            'passed': pass_count,
            'validation_passed': critical_count == 0,
            'execution_time_ms': total_time,
            'average_time_per_roi_ms': total_time / len(roi_files) if roi_files else 0,
            'cache_hits': len([k for k in self.cache.keys()]) if self.cache_enabled else 0
        }
        
        self.logger.info(f"Validation completed: {critical_count} critical, {warning_count} warnings, {pass_count} passed")
        
        return {
            'summary': summary,
            'roi_results': results,
            'performance': {
                'total_time_ms': total_time,
                'parallel_workers': self.max_workers,
                'cache_enabled': self.cache_enabled
            }
        }


def create_unified_validation_pipeline(max_workers: int = None, cache_enabled: bool = True) -> UnifiedValidationPipeline:
    """Factory function to create unified validation pipeline.
    
    Args:
        max_workers: Number of parallel workers
        cache_enabled: Whether to enable caching
        
    Returns:
        Configured unified validation pipeline
    """
    return UnifiedValidationPipeline(max_workers=max_workers, cache_enabled=cache_enabled)