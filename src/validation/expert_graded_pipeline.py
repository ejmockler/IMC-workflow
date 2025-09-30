"""
Expert-Grade Quality Assessment Pipeline for IMC Data

Implements sophisticated, multi-tier quality grading system that distinguishes
technical artifacts from biological variation. Provides detailed quality reports
with contextual information to enable informed scientific decision-making.

PHILOSOPHY: 
- Quality is a spectrum, not binary
- Biological variation ≠ technical failure
- Expert judgment enhanced by quantitative metrics
- Full provenance and reproducibility
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import json
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns


class QualityGrade(Enum):
    """Quality grade levels for ROI assessment."""
    GRADE_A = "A"  # Pass - No significant deviations
    GRADE_B = "B"  # Warning - Minor deviations, requires review
    GRADE_C = "C"  # Quarantine - Major deviations, exclude from initial analysis
    GRADE_F = "F"  # Fail - Catastrophic failure, unsalvageable


@dataclass
class QualityMetric:
    """Individual quality metric with context."""
    name: str
    value: float
    expected_range: Tuple[float, float]
    severity: str  # 'minor', 'moderate', 'severe'
    biological_context: str
    recommendation: str


@dataclass
class ROIQualityReport:
    """Comprehensive quality report for single ROI."""
    roi_name: str
    overall_grade: QualityGrade
    confidence_score: float  # 0-1, how confident we are in the grade
    quality_metrics: List[QualityMetric]
    biological_context: Dict[str, Any]
    diagnostic_plots: List[str]  # Paths to generated plots
    expert_review_required: bool
    recommendations: List[str]
    execution_time_ms: float


class ExpertGradedPipeline:
    """Expert-grade quality assessment pipeline for IMC data."""
    
    def __init__(self, study_context: Dict[str, Any] = None):
        """Initialize expert-graded pipeline with study context.
        
        Args:
            study_context: Study-specific parameters and expected patterns
        """
        self.logger = logging.getLogger(__name__)
        
        # Study context - critical for proper interpretation
        self.study_context = study_context or {
            'study_type': 'cross_sectional_injury',
            'timepoints': ['Day_1', 'Day_3', 'Day_7'],
            'tissue_type': 'kidney',
            'expected_heterogeneity': 'high',  # injury progression creates natural variation
            'biological_markers': {
                'immune': ['CD45', 'CD11b', 'Ly6G', 'CD206'],
                'vascular': ['CD31', 'CD34'],
                'stromal': ['CD140a', 'CD140b'],
                'activation': ['CD44']
            }
        }
        
        # Stratified thresholds - different standards for different contexts
        self.stratified_thresholds = {
            'Day_1': {  # Acute injury - expect high immune, disrupted vasculature
                'immune_intensity_range': (50, 500),
                'vascular_coherence_min': 0.3,  # Disrupted expected
                'spatial_density_range': (0.5, 5.0)
            },
            'Day_3': {  # Peak inflammation - highest variation expected
                'immune_intensity_range': (100, 1000),
                'vascular_coherence_min': 0.2,  # Maximum disruption
                'spatial_density_range': (0.3, 8.0)
            },
            'Day_7': {  # Resolution/fibrosis - organized repair
                'immune_intensity_range': (20, 300),
                'vascular_coherence_min': 0.6,  # Repair expected
                'spatial_density_range': (0.4, 6.0)
            }
        }
        
        # Universal quality standards - apply regardless of biology
        self.universal_standards = {
            'min_cells_absolute': 20,      # Below this = technical failure
            'max_noise_ratio': 0.95,      # Above this = pure noise
            'min_dna_signal': 10,          # Below this = segmentation failure
            'coordinate_integrity': True   # Must have valid coordinates
        }
        
    def assess_roi_quality(self, roi_file: Path, timepoint_hint: str = None) -> ROIQualityReport:
        """Comprehensive quality assessment for single ROI."""
        start_time = time.time()
        
        try:
            # Load and parse ROI data
            roi_data = pd.read_csv(roi_file, sep='\t')
            coords = roi_data[['X', 'Y']].values
            n_cells = len(coords)
            
            # Determine timepoint context from filename if not provided
            if timepoint_hint is None:
                timepoint_hint = self._extract_timepoint_from_filename(roi_file.name)
            
            # Get appropriate thresholds for this timepoint
            timepoint_thresholds = self.stratified_thresholds.get(
                timepoint_hint, 
                self.stratified_thresholds['Day_3']  # Default to most permissive
            )
            
            # Initialize quality assessment
            quality_metrics = []
            issues = []
            grade_points = 100  # Start with perfect score, deduct for issues
            
            # TIER 1: Universal Standards (Hard Failures)
            universal_pass, universal_issues = self._check_universal_standards(
                roi_data, coords, n_cells
            )
            
            if not universal_pass:
                return ROIQualityReport(
                    roi_name=roi_file.name,
                    overall_grade=QualityGrade.GRADE_F,
                    confidence_score=1.0,
                    quality_metrics=[],
                    biological_context={'failure_reason': universal_issues},
                    diagnostic_plots=[],
                    expert_review_required=False,
                    recommendations=[f"FAIL: {issue}" for issue in universal_issues],
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # TIER 2: Spatial Quality Assessment
            spatial_metrics = self._assess_spatial_quality(
                coords, timepoint_thresholds
            )
            quality_metrics.extend(spatial_metrics)
            
            # TIER 3: Protein Expression Assessment
            protein_metrics = self._assess_protein_quality(
                roi_data, timepoint_hint, timepoint_thresholds
            )
            quality_metrics.extend(protein_metrics)
            
            # TIER 4: Biological Coherence Assessment
            biological_metrics = self._assess_biological_coherence(
                roi_data, timepoint_hint
            )
            quality_metrics.extend(biological_metrics)
            
            # Calculate overall grade based on severity of issues
            overall_grade, confidence = self._calculate_overall_grade(quality_metrics)
            
            # Generate expert review requirements
            expert_review_required = self._requires_expert_review(quality_metrics, overall_grade)
            
            # Generate actionable recommendations
            recommendations = self._generate_recommendations(
                quality_metrics, timepoint_hint, overall_grade
            )
            
            # Log detailed metrics for transparency
            self.logger.info(f"ROI {roi_file.name} Assessment:")
            self.logger.info(f"  Grade: {overall_grade.value} (confidence: {confidence:.2f})")
            self.logger.info(f"  Timepoint: {timepoint_hint}, Cells: {n_cells}")
            
            # Log concerning metrics
            concerning_metrics = [m for m in quality_metrics if m.severity in ['moderate', 'severe']]
            if concerning_metrics:
                self.logger.warning(f"  Quality Issues ({len(concerning_metrics)}):")
                for metric in concerning_metrics:
                    self.logger.warning(f"    {metric.name}: {metric.value:.3f} (expected: {metric.expected_range}) - {metric.severity}")
            else:
                self.logger.info(f"  All metrics within acceptable ranges")
            
            # Create biological context summary
            biological_context = {
                'timepoint': timepoint_hint,
                'n_cells': n_cells,
                'tissue_characteristics': self._summarize_tissue_characteristics(roi_data),
                'expected_patterns': self._get_expected_patterns(timepoint_hint)
            }
            
            execution_time = (time.time() - start_time) * 1000
            
            return ROIQualityReport(
                roi_name=roi_file.name,
                overall_grade=overall_grade,
                confidence_score=confidence,
                quality_metrics=quality_metrics,
                biological_context=biological_context,
                diagnostic_plots=[],  # TODO: Generate diagnostic plots
                expert_review_required=expert_review_required,
                recommendations=recommendations,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed for {roi_file.name}: {e}")
            return ROIQualityReport(
                roi_name=roi_file.name,
                overall_grade=QualityGrade.GRADE_F,
                confidence_score=1.0,
                quality_metrics=[],
                biological_context={'error': str(e)},
                diagnostic_plots=[],
                expert_review_required=False,
                recommendations=[f"TECHNICAL ERROR: {str(e)}"],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _extract_timepoint_from_filename(self, filename: str) -> str:
        """Extract timepoint from ROI filename."""
        if '_D1_' in filename or 'Day1' in filename:
            return 'Day_1'
        elif '_D3_' in filename or 'Day3' in filename:
            return 'Day_3'
        elif '_D7_' in filename or 'Day7' in filename:
            return 'Day_7'
        else:
            return 'Day_3'  # Default to most permissive
    
    def _check_universal_standards(self, roi_data: pd.DataFrame, 
                                 coords: np.ndarray, n_cells: int) -> Tuple[bool, List[str]]:
        """Check universal quality standards that apply regardless of biology."""
        issues = []
        
        # Absolute minimum cell count
        if n_cells < self.universal_standards['min_cells_absolute']:
            issues.append(f"Insufficient cells: {n_cells} < {self.universal_standards['min_cells_absolute']}")
        
        # Coordinate integrity
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            issues.append("Invalid coordinates (NaN or Inf detected)")
        
        # Spatial extent check
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        if x_range == 0 or y_range == 0:
            issues.append("Degenerate spatial extent (zero range in X or Y)")
        
        # DNA signal presence (critical for IMC)
        dna_cols = [col for col in roi_data.columns if 'DNA' in col]
        if not dna_cols:
            issues.append("No DNA channels found - invalid IMC data")
        else:
            for dna_col in dna_cols:
                if roi_data[dna_col].max() < self.universal_standards['min_dna_signal']:
                    issues.append(f"DNA signal too low: {dna_col} max = {roi_data[dna_col].max()}")
        
        return len(issues) == 0, issues
    
    def _assess_spatial_quality(self, coords: np.ndarray, 
                              timepoint_thresholds: Dict) -> List[QualityMetric]:
        """Assess spatial organization and density."""
        metrics = []
        
        # Spatial density
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        area_um2 = x_range * y_range
        density = len(coords) / area_um2 if area_um2 > 0 else 0
        
        density_range = timepoint_thresholds['spatial_density_range']
        if density < density_range[0]:
            severity = 'moderate'
            context = "Low spatial density may indicate tissue sparsity or edge effects"
            recommendation = "Review for tissue edge artifacts or sectioning issues"
        elif density > density_range[1]:
            severity = 'minor'
            context = "High spatial density indicates dense cellular region"
            recommendation = "Verify segmentation quality in dense regions"
        else:
            severity = 'minor'
            context = "Spatial density within expected range for timepoint"
            recommendation = "No action required"
        
        metrics.append(QualityMetric(
            name="spatial_density",
            value=density,
            expected_range=density_range,
            severity=severity,
            biological_context=context,
            recommendation=recommendation
        ))
        
        # Spatial organization (simplified)
        if len(coords) > 50:
            # Sample for performance
            sample_size = min(1000, len(coords))
            sample_idx = np.random.choice(len(coords), sample_size, replace=False)
            sample_coords = coords[sample_idx]
            
            # Nearest neighbor analysis
            nn = NearestNeighbors(n_neighbors=min(6, len(sample_coords)))
            nn.fit(sample_coords)
            distances, _ = nn.kneighbors(sample_coords)
            mean_nn_dist = np.mean(distances[:, 1:])  # Exclude self
            
            # Simple spatial organization metric
            spatial_cv = np.std(distances[:, 1]) / mean_nn_dist if mean_nn_dist > 0 else 0
            
            if spatial_cv > 2.0:
                severity = 'minor'
                context = "High spatial variation indicates heterogeneous tissue organization"
                recommendation = "Normal for injury/repair tissue - proceed with analysis"
            else:
                severity = 'minor'
                context = "Organized spatial structure detected"
                recommendation = "Good tissue organization"
            
            metrics.append(QualityMetric(
                name="spatial_organization",
                value=spatial_cv,
                expected_range=(0.5, 2.0),
                severity=severity,
                biological_context=context,
                recommendation=recommendation
            ))
        
        return metrics
    
    def _assess_protein_quality(self, roi_data: pd.DataFrame, 
                              timepoint: str, timepoint_thresholds: Dict) -> List[QualityMetric]:
        """Assess protein expression quality with biological context."""
        metrics = []
        
        # Expected proteins for this study
        expected_proteins = ['CD45', 'CD11b', 'Ly6G', 'CD140a', 'CD140b', 
                           'CD31', 'CD34', 'CD206', 'CD44']
        
        for protein in expected_proteins:
            # Find matching column
            matching_cols = [col for col in roi_data.columns if protein in col]
            if not matching_cols:
                metrics.append(QualityMetric(
                    name=f"{protein}_presence",
                    value=0.0,
                    expected_range=(1.0, 1.0),
                    severity='moderate',
                    biological_context=f"Missing {protein} channel - analysis will be incomplete",
                    recommendation=f"Verify {protein} antibody was included in panel"
                ))
                continue
            
            # Analyze protein expression
            intensities = roi_data[matching_cols[0]].values
            
            # Signal-to-noise ratio
            signal_95 = np.percentile(intensities, 95)
            noise_5 = np.percentile(intensities, 5)
            snr = signal_95 / noise_5 if noise_5 > 0 else float('inf')
            
            # Biological context for SNR assessment
            if protein in ['CD45', 'CD11b']:  # Immune markers
                if timepoint == 'Day_3' and snr < 2.0:
                    severity = 'minor'
                    context = f"Low {protein} SNR at peak inflammation timepoint - may indicate acquisition issue"
                    recommendation = "Check antibody binding efficiency"
                elif snr < 1.5:
                    severity = 'moderate'
                    context = f"Very low {protein} SNR - may affect immune analysis"
                    recommendation = "Consider excluding from immune clustering analysis"
                else:
                    severity = 'minor'
                    context = f"{protein} shows adequate signal quality"
                    recommendation = "Proceed with analysis"
            elif protein in ['CD31', 'CD34']:  # Vascular markers
                if timepoint == 'Day_1' and snr < 1.8:
                    severity = 'minor'
                    context = f"Low {protein} SNR expected in acute injury (vascular disruption)"
                    recommendation = "Biological - proceed with analysis"
                elif snr < 1.5:
                    severity = 'moderate'
                    context = f"Low {protein} SNR may affect vascular analysis"
                    recommendation = "Review vascular network analysis parameters"
                else:
                    severity = 'minor'
                    context = f"{protein} shows adequate vascular signal"
                    recommendation = "Proceed with analysis"
            else:  # Other markers
                if snr < 1.5:
                    severity = 'minor'
                    context = f"Low {protein} SNR - common in tissue heterogeneity"
                    recommendation = "Consider in context of other markers"
                else:
                    severity = 'minor'
                    context = f"{protein} shows adequate signal quality"
                    recommendation = "Proceed with analysis"
            
            metrics.append(QualityMetric(
                name=f"{protein}_snr",
                value=snr,
                expected_range=(1.5, 10.0),
                severity=severity,
                biological_context=context,
                recommendation=recommendation
            ))
        
        return metrics
    
    def _assess_biological_coherence(self, roi_data: pd.DataFrame, 
                                   timepoint: str) -> List[QualityMetric]:
        """Assess biological coherence and expected patterns."""
        metrics = []
        
        # DNA correlation check (should be high)
        dna_cols = [col for col in roi_data.columns if 'DNA' in col]
        if len(dna_cols) >= 2:
            dna1 = roi_data[dna_cols[0]].values
            dna2 = roi_data[dna_cols[1]].values
            dna_corr = np.corrcoef(dna1, dna2)[0, 1] if len(dna1) > 1 else 0
            
            if dna_corr < 0.6:
                severity = 'moderate'
                context = "Low DNA channel correlation may indicate acquisition issues"
                recommendation = "Review segmentation quality and acquisition parameters"
            else:
                severity = 'minor'
                context = "DNA channels show good correlation"
                recommendation = "Good data quality indicator"
            
            metrics.append(QualityMetric(
                name="dna_correlation",
                value=dna_corr,
                expected_range=(0.6, 1.0),
                severity=severity,
                biological_context=context,
                recommendation=recommendation
            ))
        
        # Immune-vascular relationship check (context-dependent)
        cd45_cols = [col for col in roi_data.columns if 'CD45' in col]
        cd31_cols = [col for col in roi_data.columns if 'CD31' in col]
        
        if cd45_cols and cd31_cols:
            cd45 = roi_data[cd45_cols[0]].values
            cd31 = roi_data[cd31_cols[0]].values
            immune_vasc_corr = np.corrcoef(cd45, cd31)[0, 1] if len(cd45) > 1 else 0
            
            # Biological context interpretation
            if timepoint == 'Day_1':
                expected_corr = (-0.3, 0.1)  # May be negatively correlated (immune infiltration)
                context = "Immune-vascular relationship in acute injury"
            elif timepoint == 'Day_3':
                expected_corr = (-0.2, 0.3)  # Variable relationship at peak inflammation
                context = "Complex immune-vascular dynamics at peak inflammation"
            else:  # Day 7
                expected_corr = (-0.1, 0.4)  # May be positively correlated (repair)
                context = "Immune-vascular relationship during repair phase"
            
            severity = 'minor'  # Usually not critical
            recommendation = "Biological pattern - proceed with spatial analysis"
            
            metrics.append(QualityMetric(
                name="immune_vascular_relationship",
                value=immune_vasc_corr,
                expected_range=expected_corr,
                severity=severity,
                biological_context=context,
                recommendation=recommendation
            ))
        
        return metrics
    
    def _calculate_overall_grade(self, quality_metrics: List[QualityMetric]) -> Tuple[QualityGrade, float]:
        """Calculate overall quality grade based on metric severities."""
        severe_count = sum(1 for m in quality_metrics if m.severity == 'severe')
        moderate_count = sum(1 for m in quality_metrics if m.severity == 'moderate')
        minor_count = sum(1 for m in quality_metrics if m.severity == 'minor')
        
        total_metrics = len(quality_metrics) if quality_metrics else 1
        
        # More stringent grade assignment logic
        if severe_count > 0:
            grade = QualityGrade.GRADE_C
            confidence = 0.9
        elif moderate_count >= 3:  # 3+ moderate issues = Grade C
            grade = QualityGrade.GRADE_C
            confidence = 0.8
        elif moderate_count >= 2:  # 2 moderate issues = Grade B
            grade = QualityGrade.GRADE_B
            confidence = 0.8
        elif moderate_count == 1:  # 1 moderate issue = Grade B
            grade = QualityGrade.GRADE_B
            confidence = 0.7
        elif minor_count > total_metrics * 0.5:  # >50% minor issues = Grade B
            grade = QualityGrade.GRADE_B
            confidence = 0.6
        else:
            grade = QualityGrade.GRADE_A
            confidence = 0.9
        
        return grade, confidence
    
    def _requires_expert_review(self, quality_metrics: List[QualityMetric], 
                              grade: QualityGrade) -> bool:
        """Determine if expert review is required."""
        if grade in [QualityGrade.GRADE_C, QualityGrade.GRADE_F]:
            return True
        
        # Require review for specific concerning patterns
        concerning_patterns = [
            'dna_correlation',
            'spatial_density'
        ]
        
        for metric in quality_metrics:
            if any(pattern in metric.name for pattern in concerning_patterns):
                if metric.severity in ['moderate', 'severe']:
                    return True
        
        return False
    
    def _generate_recommendations(self, quality_metrics: List[QualityMetric], 
                                timepoint: str, grade: QualityGrade) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Grade-specific recommendations
        if grade == QualityGrade.GRADE_A:
            recommendations.append("EXCELLENT: Proceed with all analyses")
        elif grade == QualityGrade.GRADE_B:
            recommendations.append("GOOD: Proceed with analysis, monitor specific metrics")
        elif grade == QualityGrade.GRADE_C:
            recommendations.append("CAUTION: Include in analysis but flag for interpretation")
        else:
            recommendations.append("EXCLUDE: Technical issues prevent reliable analysis")
        
        # Specific metric recommendations
        for metric in quality_metrics:
            if metric.severity in ['moderate', 'severe']:
                recommendations.append(f"{metric.name}: {metric.recommendation}")
        
        return recommendations
    
    def _summarize_tissue_characteristics(self, roi_data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize key tissue characteristics."""
        characteristics = {}
        
        # Cell density
        characteristics['cell_count'] = len(roi_data)
        
        # Immune infiltration
        cd45_cols = [col for col in roi_data.columns if 'CD45' in col]
        if cd45_cols:
            cd45_positive = np.sum(roi_data[cd45_cols[0]] > np.percentile(roi_data[cd45_cols[0]], 75))
            characteristics['immune_fraction'] = cd45_positive / len(roi_data)
        
        # Vascular density
        cd31_cols = [col for col in roi_data.columns if 'CD31' in col]
        if cd31_cols:
            cd31_positive = np.sum(roi_data[cd31_cols[0]] > np.percentile(roi_data[cd31_cols[0]], 75))
            characteristics['vascular_fraction'] = cd31_positive / len(roi_data)
        
        return characteristics
    
    def _get_expected_patterns(self, timepoint: str) -> Dict[str, str]:
        """Get expected biological patterns for timepoint."""
        patterns = {
            'Day_1': {
                'immune': 'Neutrophil infiltration (Ly6G+)',
                'vascular': 'Endothelial disruption',
                'spatial': 'Focal injury pattern'
            },
            'Day_3': {
                'immune': 'Peak macrophage activation (CD11b+, CD206+)',
                'vascular': 'Angiogenic response',
                'spatial': 'Expanded inflammatory zones'
            },
            'Day_7': {
                'immune': 'Resolution (M2 macrophages)',
                'vascular': 'Vascular repair',
                'spatial': 'Organized repair tissue'
            }
        }
        
        return patterns.get(timepoint, patterns['Day_3'])
    
    def assess_dataset(self, roi_files: List[Path]) -> Dict[str, Any]:
        """Assess entire dataset with expert-grade quality evaluation."""
        start_time = time.time()
        self.logger.info(f"Starting expert-grade quality assessment of {len(roi_files)} ROI files")
        
        # Process ROIs in parallel
        roi_reports = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(self.assess_roi_quality, roi_file): roi_file 
                for roi_file in roi_files
            }
            
            for future in as_completed(future_to_file):
                try:
                    report = future.result()
                    roi_reports.append(report)
                except Exception as e:
                    self.logger.error(f"Assessment failed: {e}")
        
        # Compile dataset summary
        grades = [report.overall_grade for report in roi_reports]
        grade_counts = {grade: grades.count(grade) for grade in QualityGrade}
        
        expert_review_count = sum(1 for report in roi_reports if report.expert_review_required)
        
        # Generate dataset recommendations
        total_usable = grade_counts[QualityGrade.GRADE_A] + grade_counts[QualityGrade.GRADE_B]
        analysis_ready = total_usable >= len(roi_files) * 0.8  # 80% threshold
        
        dataset_recommendations = []
        if analysis_ready:
            dataset_recommendations.append("PROCEED: Dataset quality sufficient for analysis")
        else:
            dataset_recommendations.append("REVIEW: Dataset may need expert evaluation before analysis")
        
        if expert_review_count > 0:
            dataset_recommendations.append(f"EXPERT REVIEW: {expert_review_count} ROIs flagged for review")
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'dataset_quality': 'PASS' if analysis_ready else 'REVIEW_REQUIRED',
            'analysis_ready': analysis_ready,
            'summary': {
                'total_rois': len(roi_files),
                'grade_distribution': {grade.value: count for grade, count in grade_counts.items()},
                'expert_review_required': expert_review_count,
                'execution_time_ms': total_time
            },
            'roi_reports': [
                {
                    'roi_name': report.roi_name,
                    'grade': report.overall_grade.value,
                    'confidence': report.confidence_score,
                    'expert_review': report.expert_review_required,
                    'recommendations': report.recommendations,
                    'biological_context': report.biological_context
                }
                for report in roi_reports
            ],
            'dataset_recommendations': dataset_recommendations
        }


def create_expert_graded_pipeline(study_context: Dict[str, Any] = None) -> ExpertGradedPipeline:
    """Factory function for expert-graded quality assessment pipeline."""
    return ExpertGradedPipeline(study_context=study_context)