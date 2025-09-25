"""
Data Integrity Validation Rules for IMC Analysis

Validates fundamental data properties, coordinate systems, ion count ranges,
and statistical assumptions to ensure robust analysis foundation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import logging

from .framework import (
    ValidationRule, ValidationResult, ValidationSeverity, 
    ValidationCategory, ValidationMetric, QualityMetrics
)


class CoordinateValidator(ValidationRule):
    """Validates spatial coordinate data quality and properties."""
    
    def __init__(self):
        super().__init__("coordinate_validation", ValidationCategory.DATA_INTEGRITY)
        
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Validate coordinate data integrity."""
        coords = data.get('coords')
        if coords is None:
            return self._create_result(
                ValidationSeverity.CRITICAL,
                "No coordinate data provided",
                quality_score=0.0,
                recommendations=["Provide coordinate data as 'coords' key in data dictionary"]
            )
        
        metrics = {}
        recommendations = []
        issues = []
        
        # Basic shape validation
        if coords.ndim != 2 or coords.shape[1] != 2:
            return self._create_result(
                ValidationSeverity.CRITICAL,
                f"Invalid coordinate shape: {coords.shape}, expected (N, 2)",
                quality_score=0.0
            )
        
        n_points = len(coords)
        metrics['n_points'] = ValidationMetric(
            name='n_points', 
            value=n_points, 
            description="Total number of coordinate points"
        )
        
        # Check for minimum points
        if n_points < 100:
            issues.append(f"Very few coordinate points: {n_points}")
            recommendations.append("Verify data loading - minimum 100 points recommended")
        
        # Spatial bounds analysis
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        area = x_range * y_range
        
        metrics.update({
            'x_range_um': ValidationMetric('x_range_um', x_range, units='μm', description="X-axis spatial range"),
            'y_range_um': ValidationMetric('y_range_um', y_range, units='μm', description="Y-axis spatial range"),
            'area_um2': ValidationMetric('area_um2', area, units='μm²', description="Total tissue area"),
            'density_points_per_um2': ValidationMetric(
                'density_points_per_um2', 
                n_points / area if area > 0 else 0,
                expected_range=(0.5, 2.0),
                units='points/μm²',
                description="Spatial density of measurement points"
            )
        })
        
        # Check for degenerate cases
        if x_range == 0 or y_range == 0:
            return self._create_result(
                ValidationSeverity.CRITICAL,
                "Degenerate coordinates: zero range in X or Y dimension",
                quality_score=0.0
            )
        
        # Density validation
        density = n_points / area
        if density < 0.1:
            issues.append(f"Very low spatial density: {density:.3f} points/μm²")
            recommendations.append("Check coordinate units - may need scaling")
        elif density > 10.0:
            issues.append(f"Very high spatial density: {density:.3f} points/μm²")
            recommendations.append("Consider subsampling for computational efficiency")
        
        # Outlier detection
        outlier_fraction = self._detect_spatial_outliers(coords)
        metrics['outlier_fraction'] = ValidationMetric(
            'outlier_fraction',
            outlier_fraction,
            expected_range=(0.0, 0.05),
            description="Fraction of points considered spatial outliers"
        )
        
        if outlier_fraction > 0.1:
            issues.append(f"High outlier fraction: {outlier_fraction:.2%}")
            recommendations.append("Review data preprocessing and filtering")
        
        # Missing/invalid values
        finite_mask = np.isfinite(coords).all(axis=1)
        invalid_fraction = 1 - np.mean(finite_mask)
        
        metrics['invalid_fraction'] = ValidationMetric(
            'invalid_fraction',
            invalid_fraction,
            expected_range=(0.0, 0.01),
            description="Fraction of coordinates with NaN/Inf values"
        )
        
        if invalid_fraction > 0:
            issues.append(f"Invalid coordinates: {invalid_fraction:.2%}")
            recommendations.append("Remove or interpolate invalid coordinate values")
        
        # Spatial regularity assessment
        regularity_score = self._assess_spatial_regularity(coords[finite_mask])
        metrics['spatial_regularity'] = ValidationMetric(
            'spatial_regularity',
            regularity_score,
            expected_range=(0.3, 1.0),
            description="Spatial regularity score (0=random, 1=perfectly regular)"
        )
        
        # Calculate overall quality score
        quality_components = [
            min(1.0, density / 1.0),  # Density component
            max(0.0, 1.0 - outlier_fraction * 10),  # Outlier component  
            max(0.0, 1.0 - invalid_fraction * 100),  # Validity component
            regularity_score  # Regularity component
        ]
        quality_score = np.mean(quality_components)
        
        # Determine severity
        if invalid_fraction > 0.1 or outlier_fraction > 0.2:
            severity = ValidationSeverity.CRITICAL
        elif len(issues) > 0:
            severity = ValidationSeverity.WARNING
        else:
            severity = ValidationSeverity.PASS
        
        message = f"Coordinate validation: {len(issues)} issues found" if issues else "Coordinates pass validation"
        if issues:
            message += f" - {'; '.join(issues[:3])}"
        
        return self._create_result(
            severity=severity,
            message=message,
            quality_score=quality_score,
            metrics=metrics,
            recommendations=recommendations,
            context={'issues': issues, 'n_points': n_points, 'area': area}
        )
    
    def _detect_spatial_outliers(self, coords: np.ndarray) -> float:
        """Detect spatial outliers using IQR method."""
        q1 = np.percentile(coords, 25, axis=0)
        q3 = np.percentile(coords, 75, axis=0)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = np.any((coords < lower_bound) | (coords > upper_bound), axis=1)
        return np.mean(outliers)
    
    def _assess_spatial_regularity(self, coords: np.ndarray, random_seed: int = 42) -> float:
        """Assess spatial regularity using deterministic sampling and nearest neighbor analysis."""
        if len(coords) < 10:
            return 0.5  # Insufficient data
        
        # Deterministic sampling for reproducibility
        np.random.seed(random_seed)
        sample_size = min(1000, len(coords))
        
        if len(coords) <= sample_size:
            sample_coords = coords
        else:
            # Use deterministic sampling based on spatial distribution
            # Divide space into grid and sample from each cell
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            
            # Create spatial grid for uniform sampling
            grid_size = int(np.ceil(np.sqrt(sample_size)))
            x_bins = np.linspace(x_min, x_max, grid_size + 1)
            y_bins = np.linspace(y_min, y_max, grid_size + 1)
            
            sampled_indices = []
            points_per_cell = max(1, sample_size // (grid_size ** 2))
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # Find points in this grid cell
                    x_mask = (coords[:, 0] >= x_bins[i]) & (coords[:, 0] < x_bins[i+1])
                    y_mask = (coords[:, 1] >= y_bins[j]) & (coords[:, 1] < y_bins[j+1])
                    cell_mask = x_mask & y_mask
                    
                    cell_indices = np.where(cell_mask)[0]
                    
                    if len(cell_indices) > 0:
                        # Deterministic selection within cell
                        n_select = min(points_per_cell, len(cell_indices))
                        selected = cell_indices[::len(cell_indices)//n_select][:n_select]
                        sampled_indices.extend(selected)
                        
                        if len(sampled_indices) >= sample_size:
                            break
                
                if len(sampled_indices) >= sample_size:
                    break
            
            # Truncate to exact sample size
            sampled_indices = sampled_indices[:sample_size]
            sample_coords = coords[sampled_indices]
        
        try:
            # Find nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=min(3, len(sample_coords))).fit(sample_coords)
            distances, _ = nbrs.kneighbors(sample_coords)
            nn_distances = distances[:, 1]  # Distance to nearest neighbor
            
            if len(nn_distances) == 0:
                return 0.5
            
            # Calculate coefficient of variation
            mean_dist = np.mean(nn_distances)
            cv = np.std(nn_distances) / mean_dist if mean_dist > 0 else 10
            
            # Convert to regularity score (lower CV = more regular)
            regularity = np.exp(-cv)
            return float(np.clip(regularity, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Error in spatial regularity calculation: {e}")
            return 0.5


class IonCountValidator(ValidationRule):
    """Validates ion count data properties and statistical assumptions."""
    
    def __init__(self):
        super().__init__("ion_count_validation", ValidationCategory.DATA_INTEGRITY)
    
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Validate ion count data integrity."""
        ion_counts = data.get('ion_counts')
        if not ion_counts:
            return self._create_result(
                ValidationSeverity.CRITICAL,
                "No ion count data provided",
                quality_score=0.0
            )
        
        metrics = {}
        recommendations = []
        issues = []
        protein_quality_scores = {}
        
        # Global statistics
        n_proteins = len(ion_counts)
        total_measurements = sum(len(counts) for counts in ion_counts.values())
        
        metrics.update({
            'n_proteins': ValidationMetric('n_proteins', n_proteins, description="Number of protein markers"),
            'total_measurements': ValidationMetric('total_measurements', total_measurements, description="Total ion count measurements")
        })
        
        # Vectorized per-protein validation
        protein_metrics = self._validate_protein_counts_vectorized(ion_counts)
        metrics.update(protein_metrics)
        
        # Vectorized quality calculation
        protein_quality_scores = self._calculate_protein_quality_vectorized(ion_counts, protein_metrics)
        
        # Check for critical issues
        for protein, quality_score in protein_quality_scores.items():
            if quality_score < 0.3:
                issues.append(f"{protein}: very low quality (score: {quality_score:.2f})")
                recommendations.append(f"Review {protein} data preprocessing")
            elif quality_score < 0.6:
                issues.append(f"{protein}: moderate quality issues (score: {quality_score:.2f})")
        
        # Cross-protein validation
        cross_validation = self._validate_cross_protein_properties(ion_counts)
        metrics.update(cross_validation['metrics'])
        issues.extend(cross_validation['issues'])
        recommendations.extend(cross_validation['recommendations'])
        
        # Overall quality assessment
        overall_quality = np.mean(list(protein_quality_scores.values())) if protein_quality_scores else 0.0
        
        metrics['overall_ion_count_quality'] = ValidationMetric(
            'overall_ion_count_quality',
            overall_quality,
            expected_range=(0.7, 1.0),
            description="Overall ion count data quality score"
        )
        
        # Determine severity
        critical_proteins = [p for p, q in protein_quality_scores.items() if q < 0.3]
        if critical_proteins:
            severity = ValidationSeverity.CRITICAL
            message = f"Critical ion count issues in {len(critical_proteins)} proteins: {', '.join(critical_proteins[:3])}"
        elif issues:
            severity = ValidationSeverity.WARNING
            message = f"Ion count validation: {len(issues)} issues detected"
        else:
            severity = ValidationSeverity.PASS
            message = "Ion count data passes validation"
        
        return self._create_result(
            severity=severity,
            message=message,
            quality_score=overall_quality,
            metrics=metrics,
            recommendations=recommendations,
            context={
                'protein_quality_scores': protein_quality_scores,
                'issues': issues,
                'n_proteins': n_proteins
            }
        )
    
    def _validate_protein_counts_vectorized(self, ion_counts: Dict[str, np.ndarray]) -> Dict[str, ValidationMetric]:
        """Vectorized validation for all proteins simultaneously."""
        
        proteins = list(ion_counts.keys())
        if not proteins:
            return {}
        
        # Stack all protein data into a single matrix (n_points, n_proteins)
        count_matrix = np.column_stack([ion_counts[protein] for protein in proteins])
        n_points, n_proteins = count_matrix.shape
        
        # Vectorized basic statistics
        n_positive = np.sum(count_matrix > 0, axis=0)
        n_zero = np.sum(count_matrix == 0, axis=0)
        mean_counts = np.mean(count_matrix, axis=0)
        std_counts = np.std(count_matrix, axis=0)
        min_counts = np.min(count_matrix, axis=0)
        max_counts = np.max(count_matrix, axis=0)
        
        # Vectorized fractions
        positive_fractions = n_positive / n_points
        zero_fractions = n_zero / n_points
        dynamic_ranges = max_counts - min_counts
        
        # Vectorized variance-to-mean ratios
        vmr_ratios = np.zeros(n_proteins)
        valid_means = mean_counts > 0
        vmr_ratios[valid_means] = (std_counts[valid_means] ** 2) / mean_counts[valid_means]
        
        # Vectorized outlier detection
        outlier_fractions = np.zeros(n_proteins)
        for i, (protein, counts) in enumerate(ion_counts.items()):
            if n_positive[i] > 10:
                positive_counts = counts[counts > 0]
                if len(positive_counts) > 0:
                    q99 = np.percentile(positive_counts, 99)
                    outliers = np.sum(counts > q99)
                    outlier_fractions[i] = outliers / n_points
        
        # Build metrics dictionary
        metrics = {}
        for i, protein in enumerate(proteins):
            prefix = f"{protein}_"
            metrics.update({
                f'{prefix}n_points': ValidationMetric('n_points', n_points, description=f"Number of measurements for {protein}"),
                f'{prefix}positive_fraction': ValidationMetric(
                    'positive_fraction', 
                    positive_fractions[i],
                    expected_range=(0.1, 0.9),
                    description=f"Fraction of positive measurements for {protein}"
                ),
                f'{prefix}zero_fraction': ValidationMetric(
                    'zero_fraction',
                    zero_fractions[i],
                    expected_range=(0.1, 0.8),
                    description=f"Fraction of zero measurements for {protein}"
                ),
                f'{prefix}mean_count': ValidationMetric('mean_count', mean_counts[i], units='counts', description=f"Mean ion count for {protein}"),
                f'{prefix}std_count': ValidationMetric('std_count', std_counts[i], units='counts', description=f"Standard deviation for {protein}"),
                f'{prefix}min_count': ValidationMetric('min_count', min_counts[i], expected_range=(0, np.inf), units='counts'),
                f'{prefix}max_count': ValidationMetric('max_count', max_counts[i], units='counts'),
                f'{prefix}dynamic_range': ValidationMetric('dynamic_range', dynamic_ranges[i], description=f"Dynamic range for {protein}"),
                f'{prefix}outlier_fraction': ValidationMetric(
                    'outlier_fraction',
                    outlier_fractions[i],
                    expected_range=(0.0, 0.02),
                    description=f"Fraction of outlier measurements for {protein}"
                )
            })
            
            # Add VMR if valid
            if valid_means[i]:
                metrics[f'{prefix}variance_to_mean_ratio'] = ValidationMetric(
                    'variance_to_mean_ratio',
                    vmr_ratios[i],
                    expected_range=(0.8, 1.5),
                    description=f"Variance-to-mean ratio for {protein}"
                )
        
        return metrics
    
    def _calculate_protein_quality_vectorized(self, ion_counts: Dict[str, np.ndarray], metrics: Dict[str, ValidationMetric]) -> Dict[str, float]:
        """Calculate quality scores for all proteins using vectorized operations."""
        
        proteins = list(ion_counts.keys())
        quality_scores = {}
        
        for protein in proteins:
            prefix = f"{protein}_"
            quality_components = []
            
            # Positive fraction component
            pos_frac = metrics.get(f'{prefix}positive_fraction')
            if pos_frac and pos_frac.expected_range:
                min_val, max_val = pos_frac.expected_range
                if min_val <= pos_frac.value <= max_val:
                    quality_components.append(1.0)
                else:
                    deviation = min(abs(pos_frac.value - min_val), abs(pos_frac.value - max_val))
                    quality_components.append(max(0.0, 1.0 - deviation))
            
            # VMR component
            vmr = metrics.get(f'{prefix}variance_to_mean_ratio')
            if vmr:
                deviation = abs(vmr.value - 1.0)
                quality_components.append(max(0.0, 1.0 - deviation))
            
            # Outlier component
            outlier_frac = metrics.get(f'{prefix}outlier_fraction')
            if outlier_frac:
                quality_components.append(max(0.0, 1.0 - outlier_frac.value * 50))
            
            # Calculate quality score
            quality_scores[protein] = np.mean(quality_components) if quality_components else 0.5
        
        return quality_scores
    
    def _validate_cross_protein_properties(self, ion_counts: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate properties across multiple proteins."""
        metrics = {}
        issues = []
        recommendations = []
        
        # Check measurement count consistency
        measurement_counts = {protein: len(counts) for protein, counts in ion_counts.items()}
        unique_counts = set(measurement_counts.values())
        
        if len(unique_counts) > 1:
            issues.append(f"Inconsistent measurement counts across proteins: {dict(measurement_counts)}")
            recommendations.append("Ensure all proteins have same number of measurements")
        
        # Dynamic range comparison
        dynamic_ranges = {}
        for protein, counts in ion_counts.items():
            if len(counts) > 0:
                dynamic_ranges[protein] = np.max(counts) - np.min(counts)
        
        if dynamic_ranges:
            range_cv = np.std(list(dynamic_ranges.values())) / np.mean(list(dynamic_ranges.values()))
            metrics['dynamic_range_cv'] = ValidationMetric(
                'dynamic_range_cv',
                range_cv,
                expected_range=(0.0, 2.0),
                description="Coefficient of variation in dynamic ranges across proteins"
            )
            
            if range_cv > 5.0:
                issues.append(f"Very different dynamic ranges across proteins (CV: {range_cv:.2f})")
                recommendations.append("Consider protein-specific normalization")
        
        return {
            'metrics': metrics,
            'issues': issues,
            'recommendations': recommendations
        }


class TransformationValidator(ValidationRule):
    """Validates data transformation quality and properties."""
    
    def __init__(self):
        super().__init__("transformation_validation", ValidationCategory.DATA_INTEGRITY)
    
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Validate data transformation integrity."""
        # Expect both original and transformed data
        original_data = data.get('original_ion_counts')
        transformed_data = data.get('transformed_arrays')
        
        if not original_data or not transformed_data:
            return self._create_result(
                ValidationSeverity.WARNING,
                "Insufficient data for transformation validation",
                quality_score=0.5,
                recommendations=["Provide both 'original_ion_counts' and 'transformed_arrays' for full validation"]
            )
        
        metrics = {}
        recommendations = []
        issues = []
        
        # Check protein alignment
        original_proteins = set(original_data.keys())
        transformed_proteins = set(transformed_data.keys())
        
        missing_in_transformed = original_proteins - transformed_proteins
        extra_in_transformed = transformed_proteins - original_proteins
        
        if missing_in_transformed:
            issues.append(f"Proteins lost in transformation: {list(missing_in_transformed)}")
            recommendations.append("Investigate why proteins were dropped during transformation")
        
        if extra_in_transformed:
            issues.append(f"Extra proteins in transformed data: {list(extra_in_transformed)}")
        
        # Validate transformations for common proteins
        common_proteins = original_proteins & transformed_proteins
        transformation_quality_scores = {}
        
        for protein in common_proteins:
            original = original_data[protein]
            transformed = transformed_data[protein]
            
            # Validate transformation properties
            protein_metrics = self._validate_protein_transformation(protein, original, transformed)
            
            # Add to global metrics
            for metric_name, metric in protein_metrics.items():
                metrics[f"{protein}_{metric_name}"] = metric
            
            # Calculate protein-specific quality
            quality = self._calculate_transformation_quality(protein_metrics)
            transformation_quality_scores[protein] = quality
            
            if quality < 0.5:
                issues.append(f"{protein}: poor transformation quality (score: {quality:.2f})")
        
        # Overall assessment
        overall_quality = np.mean(list(transformation_quality_scores.values())) if transformation_quality_scores else 0.5
        
        metrics['transformation_quality'] = ValidationMetric(
            'transformation_quality',
            overall_quality,
            expected_range=(0.7, 1.0),
            description="Overall transformation quality score"
        )
        
        # Determine severity - aligned with rebalanced thresholds
        if overall_quality < 0.15:
            severity = ValidationSeverity.CRITICAL
        elif issues:
            severity = ValidationSeverity.WARNING
        else:
            severity = ValidationSeverity.PASS
        
        message = f"Transformation validation: {len(issues)} issues found" if issues else "Transformations pass validation"
        
        return self._create_result(
            severity=severity,
            message=message,
            quality_score=overall_quality,
            metrics=metrics,
            recommendations=recommendations,
            context={
                'transformation_quality_scores': transformation_quality_scores,
                'common_proteins': list(common_proteins),
                'issues': issues
            }
        )
    
    def _validate_protein_transformation(self, protein: str, original: np.ndarray, transformed: np.ndarray) -> Dict[str, ValidationMetric]:
        """Validate transformation for individual protein."""
        metrics = {}
        
        # Shape consistency
        if original.shape != transformed.shape:
            metrics['shape_preserved'] = ValidationMetric(
                'shape_preserved', 
                False, 
                description="Whether array shape is preserved"
            )
            return metrics
        
        metrics['shape_preserved'] = ValidationMetric('shape_preserved', True, description="Shape preservation check")
        
        # Range properties
        original_range = np.max(original) - np.min(original)
        transformed_range = np.max(transformed) - np.min(transformed)
        
        metrics.update({
            'original_range': ValidationMetric('original_range', original_range, units='counts'),
            'transformed_range': ValidationMetric('transformed_range', transformed_range),
            'range_compression': ValidationMetric(
                'range_compression',
                transformed_range / original_range if original_range > 0 else 0,
                expected_range=(0.1, 0.8),
                description="Range compression factor (should be < 1 for variance stabilization)"
            )
        })
        
        # Monotonicity check (transformation should be monotonic)
        if len(original) > 1:
            # Sample points for monotonicity test
            sample_size = min(1000, len(original))
            indices = np.random.choice(len(original), sample_size, replace=False)
            
            orig_sample = original[indices]
            trans_sample = transformed[indices]
            
            # Sort by original values and check if transformed values are also increasing
            sort_indices = np.argsort(orig_sample)
            sorted_orig = orig_sample[sort_indices]
            sorted_trans = trans_sample[sort_indices]
            
            # Count monotonic violations
            violations = np.sum(np.diff(sorted_trans) < 0)
            monotonicity_score = 1.0 - (violations / len(sorted_trans))
            
            metrics['monotonicity_score'] = ValidationMetric(
                'monotonicity_score',
                monotonicity_score,
                expected_range=(0.95, 1.0),
                description="Fraction of monotonic ordering preserved"
            )
        
        # Zero preservation (zeros should remain zeros for most transformations)
        original_zeros = np.sum(original == 0)
        transformed_zeros = np.sum(np.abs(transformed) < 1e-10)  # Near-zero for floating point
        
        if original_zeros > 0:
            zero_preservation = transformed_zeros / original_zeros
            metrics['zero_preservation'] = ValidationMetric(
                'zero_preservation',
                zero_preservation,
                expected_range=(0.9, 1.0),
                description="Fraction of original zeros preserved after transformation"
            )
        
        # Finite values check
        finite_fraction = np.mean(np.isfinite(transformed))
        metrics['finite_fraction'] = ValidationMetric(
            'finite_fraction',
            finite_fraction,
            expected_range=(0.99, 1.0),
            description="Fraction of finite values after transformation"
        )
        
        return metrics
    
    def _calculate_transformation_quality(self, metrics: Dict[str, ValidationMetric]) -> float:
        """Calculate transformation quality score."""
        quality_components = []
        
        # Shape preservation (critical)
        if not metrics.get('shape_preserved', ValidationMetric('', False)).value:
            return 0.0
        
        # Range compression (should be reasonable)
        range_comp = metrics.get('range_compression')
        if range_comp and range_comp.expected_range:
            min_val, max_val = range_comp.expected_range
            if min_val <= range_comp.value <= max_val:
                quality_components.append(1.0)
            else:
                quality_components.append(0.5)
        
        # Monotonicity
        monotonicity = metrics.get('monotonicity_score')
        if monotonicity:
            quality_components.append(monotonicity.value)
        
        # Zero preservation
        zero_pres = metrics.get('zero_preservation')
        if zero_pres:
            quality_components.append(zero_pres.value)
        
        # Finite values
        finite_frac = metrics.get('finite_fraction')
        if finite_frac:
            quality_components.append(finite_frac.value)
        
        return np.mean(quality_components) if quality_components else 0.5