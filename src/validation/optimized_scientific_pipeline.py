"""
Optimized Scientific Validation Pipeline for IMC Data

Implements the brutalist's tiered validation approach:
- Tier 1: Fast sanity checks (seconds)
- Tier 2: Automated QC metrics with scientific rigor (sub-minute) 
- Tier 3: Cross-ROI consistency analysis (sub-minute)
- Tier 4: Flagging for manual review

Maintains full scientific integrity while optimizing computational performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

from .framework import (
    create_validation_suite, 
    ValidationCategory, 
    ValidationSeverity,
    ValidationSuiteResult
)
from .data_integrity import CoordinateValidator
from .scientific_quality import BiologicalValidator


class OptimizedScientificPipeline:
    """Tiered validation pipeline maintaining scientific rigor with computational efficiency."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize optimized scientific validation pipeline."""
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # IMC-specific protein channels for this study
        self.expected_proteins = ['CD45', 'CD11b', 'Ly6G', 'CD140a', 'CD140b', 
                                 'CD31', 'CD34', 'CD206', 'CD44']
        self.dna_channels = ['DNA1', 'DNA2']
        
        # Scientific quality thresholds based on IMC best practices
        self.quality_thresholds = {
            'min_cells_per_roi': 50,
            'max_cells_per_roi': 50000,
            'min_protein_snr': 1.5,
            'min_dna_signal': 100,
            'max_zero_fraction': 0.95,
            'min_spatial_density': 0.01,  # cells per μm²
            'max_cv_across_rois': 0.5,    # Coefficient of variation
        }
        
        # Cross-timepoint consistency is critical for cross-sectional studies
        self.cross_roi_metrics = {}
        
    def tier1_sanity_check(self, roi_file: Path) -> Tuple[bool, str]:
        """Tier 1: Ultra-fast sanity checks (should complete in milliseconds)."""
        try:
            # File existence and readability
            if not roi_file.exists() or roi_file.stat().st_size == 0:
                return False, f"File missing or empty: {roi_file.name}"
            
            # Quick header check without reading full file
            with open(roi_file, 'r') as f:
                header = f.readline().strip().split('\t')
            
            # Check for required coordinate columns
            if 'X' not in header or 'Y' not in header:
                return False, "Missing X/Y coordinate columns"
            
            # Check for DNA channels (critical for IMC)
            dna_found = any(any(dna in col for col in header) for dna in self.dna_channels)
            if not dna_found:
                return False, "Missing DNA channels - invalid IMC data"
            
            return True, "Passed Tier 1"
            
        except Exception as e:
            return False, f"Tier 1 error: {str(e)}"
    
    def tier2_automated_qc(self, roi_file: Path) -> Dict[str, Any]:
        """Tier 2: Automated QC metrics with scientific rigor."""
        start_time = time.time()
        
        try:
            # Load data efficiently
            roi_data = pd.read_csv(roi_file, sep='\t')
            
            # Extract coordinates
            coords = roi_data[['X', 'Y']].values
            n_cells = len(coords)
            
            # Spatial extent and density
            x_range = coords[:, 0].max() - coords[:, 0].min()
            y_range = coords[:, 1].max() - coords[:, 1].min()
            area_um2 = x_range * y_range
            spatial_density = n_cells / area_um2 if area_um2 > 0 else 0
            
            # Protein channel analysis
            protein_metrics = {}
            for protein in self.expected_proteins:
                # Find matching column (handle element notation)
                matching_cols = [col for col in roi_data.columns if protein in col]
                if matching_cols:
                    intensities = roi_data[matching_cols[0]].values
                    
                    # Signal-to-noise ratio (robust)
                    signal = np.percentile(intensities, 95)
                    noise = np.percentile(intensities, 5)
                    snr = signal / noise if noise > 0 else 0
                    
                    # Distribution properties
                    zero_fraction = np.sum(intensities == 0) / len(intensities)
                    cv = stats.variation(intensities[intensities > 0]) if np.any(intensities > 0) else float('inf')
                    
                    protein_metrics[protein] = {
                        'snr': snr,
                        'mean_intensity': np.mean(intensities),
                        'median_intensity': np.median(intensities),
                        'zero_fraction': zero_fraction,
                        'cv': cv,
                        'dynamic_range': signal - noise
                    }
                else:
                    protein_metrics[protein] = None
            
            # DNA quality assessment
            dna_metrics = {}
            for dna in self.dna_channels:
                matching_cols = [col for col in roi_data.columns if dna in col]
                if matching_cols:
                    dna_signal = roi_data[matching_cols[0]].values
                    dna_metrics[dna] = {
                        'mean_signal': np.mean(dna_signal),
                        'median_signal': np.median(dna_signal),
                        'min_signal': np.min(dna_signal)
                    }
            
            # Spatial coherence (simplified for speed)
            # Sample subset for large ROIs to maintain performance
            if n_cells > 5000:
                sample_idx = np.random.choice(n_cells, 5000, replace=False)
                sample_coords = coords[sample_idx]
            else:
                sample_coords = coords
            
            # Measure spatial clustering via nearest neighbor distances
            if len(sample_coords) > 10:
                nn = NearestNeighbors(n_neighbors=2)
                nn.fit(sample_coords)
                distances, _ = nn.kneighbors(sample_coords)
                mean_nn_distance = np.mean(distances[:, 1])  # Exclude self
                spatial_entropy = np.std(distances[:, 1]) / mean_nn_distance if mean_nn_distance > 0 else 0
            else:
                mean_nn_distance = 0
                spatial_entropy = 0
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                'roi_file': roi_file.name,
                'tier2_status': 'completed',
                'spatial_metrics': {
                    'n_cells': n_cells,
                    'spatial_density': spatial_density,
                    'area_um2': area_um2,
                    'mean_nn_distance': mean_nn_distance,
                    'spatial_entropy': spatial_entropy
                },
                'protein_metrics': protein_metrics,
                'dna_metrics': dna_metrics,
                'execution_time_ms': execution_time,
                'quality_flags': self._assess_quality_flags(n_cells, spatial_density, protein_metrics, dna_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Tier 2 failed for {roi_file.name}: {e}")
            return {
                'roi_file': roi_file.name,
                'tier2_status': 'failed',
                'error': str(e),
                'execution_time_ms': (time.time() - start_time) * 1000
            }
    
    def _assess_quality_flags(self, n_cells: int, spatial_density: float, 
                            protein_metrics: Dict, dna_metrics: Dict) -> List[str]:
        """Assess quality flags based on IMC best practices."""
        flags = []
        
        # Cell count validation
        if n_cells < self.quality_thresholds['min_cells_per_roi']:
            flags.append(f"LOW_CELL_COUNT: {n_cells} < {self.quality_thresholds['min_cells_per_roi']}")
        elif n_cells > self.quality_thresholds['max_cells_per_roi']:
            flags.append(f"HIGH_CELL_COUNT: {n_cells} > {self.quality_thresholds['max_cells_per_roi']}")
        
        # Spatial density
        if spatial_density < self.quality_thresholds['min_spatial_density']:
            flags.append(f"LOW_SPATIAL_DENSITY: {spatial_density:.3f} cells/μm²")
        
        # Protein quality
        for protein, metrics in protein_metrics.items():
            if metrics is None:
                flags.append(f"MISSING_PROTEIN: {protein}")
            else:
                if metrics['snr'] < self.quality_thresholds['min_protein_snr']:
                    flags.append(f"LOW_SNR_{protein}: {metrics['snr']:.2f}")
                if metrics['zero_fraction'] > self.quality_thresholds['max_zero_fraction']:
                    flags.append(f"HIGH_ZERO_FRACTION_{protein}: {metrics['zero_fraction']:.2f}")
        
        # DNA quality
        for dna, metrics in dna_metrics.items():
            if metrics['min_signal'] < self.quality_thresholds['min_dna_signal']:
                flags.append(f"LOW_DNA_SIGNAL_{dna}: {metrics['min_signal']:.1f}")
        
        return flags
    
    def tier3_cross_roi_analysis(self, all_tier2_results: List[Dict]) -> Dict[str, Any]:
        """Tier 3: Cross-ROI consistency analysis for cross-sectional study validity."""
        
        # Extract metrics across all ROIs
        protein_intensities = {}
        spatial_densities = []
        cell_counts = []
        
        for result in all_tier2_results:
            if result.get('tier2_status') == 'completed':
                spatial_densities.append(result['spatial_metrics']['spatial_density'])
                cell_counts.append(result['spatial_metrics']['n_cells'])
                
                for protein, metrics in result['protein_metrics'].items():
                    if metrics is not None:
                        if protein not in protein_intensities:
                            protein_intensities[protein] = []
                        protein_intensities[protein].append(metrics['mean_intensity'])
        
        # Cross-ROI consistency analysis
        consistency_flags = []
        
        # Check spatial density consistency
        if spatial_densities:
            density_cv = stats.variation(spatial_densities)
            if density_cv > self.quality_thresholds['max_cv_across_rois']:
                consistency_flags.append(f"INCONSISTENT_SPATIAL_DENSITY: CV={density_cv:.3f}")
        
        # Check protein intensity consistency across ROIs
        protein_cv_summary = {}
        for protein, intensities in protein_intensities.items():
            if len(intensities) > 1:
                cv = stats.variation(intensities)
                protein_cv_summary[protein] = cv
                if cv > self.quality_thresholds['max_cv_across_rois']:
                    consistency_flags.append(f"INCONSISTENT_{protein}_INTENSITY: CV={cv:.3f}")
        
        # Detect outlier ROIs
        outlier_rois = []
        for i, result in enumerate(all_tier2_results):
            if result.get('tier2_status') == 'completed':
                # Check if this ROI is an outlier in multiple metrics
                outlier_count = 0
                for protein in self.expected_proteins:
                    if protein in protein_intensities and len(protein_intensities[protein]) > 3:
                        roi_intensity = result['protein_metrics'].get(protein, {}).get('mean_intensity', 0)
                        median_intensity = np.median(protein_intensities[protein])
                        mad = np.median(np.abs(np.array(protein_intensities[protein]) - median_intensity))
                        if mad > 0 and abs(roi_intensity - median_intensity) > 3 * mad:
                            outlier_count += 1
                
                if outlier_count >= 3:  # Outlier in 3+ proteins
                    outlier_rois.append(result['roi_file'])
        
        return {
            'consistency_flags': consistency_flags,
            'protein_cv_summary': protein_cv_summary,
            'outlier_rois': outlier_rois,
            'summary_stats': {
                'n_valid_rois': len([r for r in all_tier2_results if r.get('tier2_status') == 'completed']),
                'median_spatial_density': np.median(spatial_densities) if spatial_densities else 0,
                'median_cell_count': np.median(cell_counts) if cell_counts else 0
            }
        }
    
    def validate_dataset(self, roi_files: List[Path]) -> Dict[str, Any]:
        """Execute complete tiered validation pipeline."""
        start_time = time.time()
        self.logger.info(f"Starting tiered scientific validation of {len(roi_files)} ROI files")
        
        # Tier 1: Parallel sanity checks
        tier1_start = time.time()
        tier1_results = []
        valid_files = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.tier1_sanity_check, f): f for f in roi_files}
            
            for future in as_completed(future_to_file):
                roi_file = future_to_file[future]
                try:
                    passed, message = future.result()
                    tier1_results.append({'file': roi_file.name, 'passed': passed, 'message': message})
                    if passed:
                        valid_files.append(roi_file)
                except Exception as e:
                    tier1_results.append({'file': roi_file.name, 'passed': False, 'message': f"Exception: {e}"})
        
        tier1_time = (time.time() - tier1_start) * 1000
        self.logger.info(f"Tier 1: {len(valid_files)}/{len(roi_files)} files passed sanity check ({tier1_time:.1f}ms)")
        
        if not valid_files:
            return {
                'validation_passed': False,
                'critical_errors': len(roi_files),
                'tier1_results': tier1_results,
                'message': 'All files failed Tier 1 sanity checks'
            }
        
        # Tier 2: Parallel automated QC
        tier2_start = time.time()
        tier2_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.tier2_automated_qc, f): f for f in valid_files}
            
            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    tier2_results.append(result)
                except Exception as e:
                    self.logger.error(f"Tier 2 exception: {e}")
        
        tier2_time = (time.time() - tier2_start) * 1000
        self.logger.info(f"Tier 2: QC analysis completed ({tier2_time:.1f}ms)")
        
        # Tier 3: Cross-ROI consistency analysis
        tier3_start = time.time()
        tier3_results = self.tier3_cross_roi_analysis(tier2_results)
        tier3_time = (time.time() - tier3_start) * 1000
        self.logger.info(f"Tier 3: Cross-ROI analysis completed ({tier3_time:.1f}ms)")
        
        # Determine overall validation status
        critical_errors = len([r for r in tier2_results if r.get('tier2_status') == 'failed'])
        critical_errors += len(tier3_results.get('outlier_rois', []))
        critical_errors += len([f for f in tier1_results if not f['passed']])
        
        warnings = sum(len(r.get('quality_flags', [])) for r in tier2_results)
        warnings += len(tier3_results.get('consistency_flags', []))
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'validation_passed': critical_errors == 0,
            'critical_errors': critical_errors,
            'warnings': warnings,
            'summary': {
                'total_rois': len(roi_files),
                'tier1_passed': len(valid_files),
                'tier2_completed': len([r for r in tier2_results if r.get('tier2_status') == 'completed']),
                'outlier_rois': len(tier3_results.get('outlier_rois', [])),
                'execution_time_ms': total_time,
                'tier_times': {
                    'tier1_ms': tier1_time,
                    'tier2_ms': tier2_time,
                    'tier3_ms': tier3_time
                }
            },
            'tier1_results': tier1_results,
            'tier2_results': tier2_results,
            'tier3_results': tier3_results,
            'recommended_actions': self._generate_recommendations(tier2_results, tier3_results)
        }
    
    def _generate_recommendations(self, tier2_results: List[Dict], tier3_results: Dict) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Check for systematic issues
        failed_count = len([r for r in tier2_results if r.get('tier2_status') == 'failed'])
        if failed_count > 0:
            recommendations.append(f"Manual review required: {failed_count} ROIs failed automated QC")
        
        outlier_count = len(tier3_results.get('outlier_rois', []))
        if outlier_count > 0:
            recommendations.append(f"Investigate {outlier_count} outlier ROIs for acquisition issues")
        
        consistency_issues = len(tier3_results.get('consistency_flags', []))
        if consistency_issues > 0:
            recommendations.append("Cross-timepoint consistency issues detected - check acquisition parameters")
        
        return recommendations


def create_optimized_scientific_pipeline(max_workers: int = 4) -> OptimizedScientificPipeline:
    """Factory function for optimized scientific validation pipeline."""
    return OptimizedScientificPipeline(max_workers=max_workers)