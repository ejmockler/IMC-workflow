"""
Scientific Quality Validation Rules for IMC Analysis

Validates biological assumptions, marker expression patterns, spatial relationships,
and scientific validity of analysis parameters for robust hypothesis generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import logging

from .framework import (
    ValidationRule, ValidationResult, ValidationSeverity,
    ValidationCategory, ValidationMetric, QualityMetrics
)


class BiologicalValidator(ValidationRule):
    """Validates biological assumptions and marker expression patterns."""
    
    def __init__(self):
        super().__init__("biological_validation", ValidationCategory.SCIENTIFIC_QUALITY)
        
        # Known biological marker relationships
        self.positive_correlations = {
            ('CD68', 'CD206'): 0.3,  # Macrophage markers should correlate
            ('DNA1', 'DNA2'): 0.7,   # DNA markers should strongly correlate
            ('αSMA', 'Collagen'): 0.4,  # Fibroblast/ECM relationship
        }
        
        self.negative_correlations = {
            ('CD68', 'αSMA'): -0.1,  # Macrophages vs fibroblasts
        }
        
        # Expected expression ranges (percentile bounds)
        self.expression_ranges = {
            'CD206': (5, 40),   # M2 macrophages: moderate expression
            'CD68': (10, 50),   # Pan-macrophage: higher expression  
            'CD44': (15, 60),   # Cell adhesion: variable expression
            'αSMA': (5, 30),    # Smooth muscle: specific expression
            'DNA1': (60, 95),   # DNA: should be high in most cells
            'DNA2': (60, 95),   # DNA: should be high in most cells
        }
    
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Validate biological properties of the data."""
        ion_counts = data.get('ion_counts')
        if not ion_counts:
            return self._create_result(
                ValidationSeverity.WARNING,
                "No ion count data for biological validation",
                quality_score=0.5
            )
        
        metrics = {}
        recommendations = []
        issues = []
        
        # Available proteins
        available_proteins = set(ion_counts.keys())
        metrics['available_markers'] = ValidationMetric(
            'available_markers',
            len(available_proteins),
            description="Number of available protein markers"
        )
        
        # Expression range validation
        expression_scores = self._validate_expression_ranges(ion_counts, available_proteins)
        metrics.update(expression_scores['metrics'])
        issues.extend(expression_scores['issues'])
        recommendations.extend(expression_scores['recommendations'])
        
        # Correlation validation
        correlation_scores = self._validate_marker_correlations(ion_counts, available_proteins)
        metrics.update(correlation_scores['metrics'])
        issues.extend(correlation_scores['issues'])
        recommendations.extend(correlation_scores['recommendations'])
        
        # Co-expression patterns
        coexpression_scores = self._validate_coexpression_patterns(ion_counts)
        metrics.update(coexpression_scores['metrics'])
        issues.extend(coexpression_scores['issues'])
        
        # Cell type inference validation
        if len(available_proteins) >= 3:
            celltype_validation = self._validate_cell_type_patterns(ion_counts)
            metrics.update(celltype_validation['metrics'])
            issues.extend(celltype_validation['issues'])
            recommendations.extend(celltype_validation['recommendations'])
        
        # Overall biological quality score
        quality_components = []
        if 'expression_quality' in metrics:
            quality_components.append(metrics['expression_quality'].value)
        if 'correlation_quality' in metrics:
            quality_components.append(metrics['correlation_quality'].value)
        if 'coexpression_quality' in metrics:
            quality_components.append(metrics['coexpression_quality'].value)
        
        overall_quality = np.mean(quality_components) if quality_components else 0.5
        
        metrics['biological_quality'] = ValidationMetric(
            'biological_quality',
            overall_quality,
            expected_range=(0.6, 1.0),
            description="Overall biological quality score"
        )
        
        # Determine severity - rebalanced for scientific rigor without over-strictness
        if overall_quality < 0.15:
            severity = ValidationSeverity.CRITICAL
            message = f"Severely compromised biological quality (score: {overall_quality:.2f})"
        elif overall_quality < 0.35:
            severity = ValidationSeverity.WARNING
            message = f"Marginal biological quality - proceed with caution (score: {overall_quality:.2f})"
        else:
            severity = ValidationSeverity.PASS
            message = f"Biological validation passed (score: {overall_quality:.2f})"
        
        return self._create_result(
            severity=severity,
            message=message,
            quality_score=overall_quality,
            metrics=metrics,
            recommendations=recommendations,
            context={'issues': issues, 'available_proteins': list(available_proteins)}
        )
    
    def _validate_expression_ranges(self, ion_counts: Dict[str, np.ndarray], available_proteins: set) -> Dict[str, Any]:
        """Validate expression ranges against biological expectations."""
        metrics = {}
        issues = []
        recommendations = []
        
        expression_quality_scores = []
        
        for protein, expected_range in self.expression_ranges.items():
            if protein in available_proteins:
                counts = ion_counts[protein]
                positive_counts = counts[counts > 0]
                
                if len(positive_counts) > 10:
                    # Calculate actual expression percentiles
                    actual_percentiles = np.percentile(positive_counts, [25, 75])
                    expected_low, expected_high = expected_range
                    
                    # Compare with expected range
                    range_score = self._calculate_range_overlap(actual_percentiles, expected_range)
                    expression_quality_scores.append(range_score)
                    
                    metrics[f'{protein}_expression_score'] = ValidationMetric(
                        f'{protein}_expression_score',
                        range_score,
                        expected_range=(0.7, 1.0),
                        description=f"Expression range quality for {protein}"
                    )
                    
                    if range_score < 0.5:
                        issues.append(f"{protein}: expression range deviates from expected biological pattern")
                        recommendations.append(f"Review {protein} staining protocol and antibody validation")
                
                # Check for all-zero or all-high patterns
                zero_fraction = np.sum(counts == 0) / len(counts)
                if zero_fraction > 0.95:
                    issues.append(f"{protein}: {zero_fraction:.1%} zero values - possible staining failure")
                    recommendations.append(f"Verify {protein} antibody staining and imaging parameters")
                elif zero_fraction < 0.05:
                    issues.append(f"{protein}: very few zero values - possible saturation or contamination")
        
        # Overall expression quality
        expression_quality = np.mean(expression_quality_scores) if expression_quality_scores else 0.5
        metrics['expression_quality'] = ValidationMetric(
            'expression_quality',
            expression_quality,
            expected_range=(0.7, 1.0),
            description="Overall expression range quality"
        )
        
        return {
            'metrics': metrics,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _validate_marker_correlations(self, ion_counts: Dict[str, np.ndarray], available_proteins: set) -> Dict[str, Any]:
        """Validate expected marker correlations."""
        metrics = {}
        issues = []
        recommendations = []
        
        correlation_scores = []
        
        # Check positive correlations
        for (marker1, marker2), expected_corr in self.positive_correlations.items():
            if marker1 in available_proteins and marker2 in available_proteins:
                counts1 = ion_counts[marker1]
                counts2 = ion_counts[marker2]
                
                # Calculate correlation for positive values
                positive_mask = (counts1 > 0) & (counts2 > 0)
                if np.sum(positive_mask) > 20:
                    actual_corr = np.corrcoef(counts1[positive_mask], counts2[positive_mask])[0, 1]
                    
                    # Score based on how close to expected
                    if actual_corr >= expected_corr:
                        score = 1.0
                    else:
                        score = max(0.0, actual_corr / expected_corr)
                    
                    correlation_scores.append(score)
                    
                    metrics[f'{marker1}_{marker2}_correlation'] = ValidationMetric(
                        f'{marker1}_{marker2}_correlation',
                        actual_corr,
                        expected_range=(expected_corr, 1.0),
                        description=f"Correlation between {marker1} and {marker2}"
                    )
                    
                    if score < 0.5:
                        issues.append(f"{marker1}-{marker2}: weak correlation ({actual_corr:.2f}, expected ≥{expected_corr:.2f})")
                        recommendations.append(f"Verify {marker1} and {marker2} antibody specificity")
        
        # Check negative correlations
        for (marker1, marker2), expected_corr in self.negative_correlations.items():
            if marker1 in available_proteins and marker2 in available_proteins:
                counts1 = ion_counts[marker1]
                counts2 = ion_counts[marker2]
                
                positive_mask = (counts1 > 0) | (counts2 > 0)
                if np.sum(positive_mask) > 20:
                    actual_corr = np.corrcoef(counts1[positive_mask], counts2[positive_mask])[0, 1]
                    
                    # Score based on how close to expected (negative)
                    if actual_corr <= expected_corr:
                        score = 1.0
                    else:
                        score = max(0.0, 1.0 - (actual_corr - expected_corr))
                    
                    correlation_scores.append(score)
                    
                    if score < 0.5:
                        issues.append(f"{marker1}-{marker2}: unexpected positive correlation ({actual_corr:.2f})")
        
        # Overall correlation quality
        correlation_quality = np.mean(correlation_scores) if correlation_scores else 0.5
        metrics['correlation_quality'] = ValidationMetric(
            'correlation_quality',
            correlation_quality,
            expected_range=(0.7, 1.0),
            description="Overall marker correlation quality"
        )
        
        return {
            'metrics': metrics,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _validate_coexpression_patterns(self, ion_counts: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate co-expression patterns for biological plausibility."""
        metrics = {}
        issues = []
        
        if len(ion_counts) < 2:
            return {'metrics': metrics, 'issues': issues}
        
        # Create expression matrix
        proteins = list(ion_counts.keys())
        n_points = len(next(iter(ion_counts.values())))
        
        expression_matrix = np.zeros((n_points, len(proteins)))
        for i, protein in enumerate(proteins):
            expression_matrix[:, i] = ion_counts[protein]
        
        # Identify co-expressing cells
        positive_threshold = np.percentile(expression_matrix, 75, axis=0)
        high_expressing = expression_matrix > positive_threshold
        
        # Calculate co-expression frequencies
        coexpression_frequencies = []
        for i in range(len(proteins)):
            for j in range(i + 1, len(proteins)):
                both_positive = np.sum(high_expressing[:, i] & high_expressing[:, j])
                either_positive = np.sum(high_expressing[:, i] | high_expressing[:, j])
                
                if either_positive > 0:
                    coexp_freq = both_positive / either_positive
                    coexpression_frequencies.append(coexp_freq)
        
        # Assess co-expression diversity
        if coexpression_frequencies:
            coexp_diversity = np.std(coexpression_frequencies)
            metrics['coexpression_diversity'] = ValidationMetric(
                'coexpression_diversity',
                coexp_diversity,
                expected_range=(0.1, 0.4),
                description="Diversity of co-expression patterns"
            )
            
            # Quality based on reasonable diversity
            if 0.1 <= coexp_diversity <= 0.4:
                coexp_quality = 1.0
            else:
                coexp_quality = max(0.0, 1.0 - abs(coexp_diversity - 0.25) * 2)
            
            metrics['coexpression_quality'] = ValidationMetric(
                'coexpression_quality',
                coexp_quality,
                expected_range=(0.7, 1.0),
                description="Co-expression pattern quality"
            )
            
            if coexp_quality < 0.5:
                if coexp_diversity < 0.1:
                    issues.append("Very uniform co-expression - possible technical artifact")
                else:
                    issues.append("Very diverse co-expression - possible noise or contamination")
        
        return {'metrics': metrics, 'issues': issues}
    
    def _validate_cell_type_patterns(self, ion_counts: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate expected cell type expression patterns."""
        metrics = {}
        issues = []
        recommendations = []
        
        proteins = list(ion_counts.keys())
        n_points = len(next(iter(ion_counts.values())))
        
        # Create normalized expression matrix
        expression_matrix = np.zeros((n_points, len(proteins)))
        for i, protein in enumerate(proteins):
            counts = ion_counts[protein]
            # Simple min-max normalization
            if np.max(counts) > np.min(counts):
                expression_matrix[:, i] = (counts - np.min(counts)) / (np.max(counts) - np.min(counts))
        
        # Attempt clustering to identify cell populations
        try:
            n_clusters = min(8, max(2, n_points // 100))  # Reasonable cluster number
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(expression_matrix)
            
            # Assess clustering quality
            silhouette_avg = silhouette_score(expression_matrix, cluster_labels)
            metrics['cell_clustering_quality'] = ValidationMetric(
                'cell_clustering_quality',
                silhouette_avg,
                expected_range=(0.2, 1.0),
                description="Quality of cell type clustering"
            )
            
            # Check for biologically reasonable cluster sizes
            cluster_sizes = np.bincount(cluster_labels)
            size_cv = np.std(cluster_sizes) / np.mean(cluster_sizes)
            
            metrics['cluster_size_diversity'] = ValidationMetric(
                'cluster_size_diversity',
                size_cv,
                expected_range=(0.3, 2.0),
                description="Diversity of cluster sizes"
            )
            
            if silhouette_avg < 0.1:
                issues.append("Poor cell type separation - possible technical noise")
                recommendations.append("Review data quality and consider noise reduction")
            
            if size_cv > 3.0:
                issues.append("Very uneven cluster sizes - possible batch effects")
                recommendations.append("Check for technical artifacts or batch effects")
        
        except Exception as e:
            self.logger.warning(f"Cell type clustering failed: {str(e)}")
            metrics['cell_clustering_quality'] = ValidationMetric(
                'cell_clustering_quality',
                0.0,
                description="Cell clustering failed"
            )
        
        return {
            'metrics': metrics,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _calculate_range_overlap(self, actual_range: Tuple[float, float], expected_range: Tuple[float, float]) -> float:
        """Calculate overlap between actual and expected ranges."""
        actual_low, actual_high = actual_range
        expected_low, expected_high = expected_range
        
        # Calculate intersection
        intersection_low = max(actual_low, expected_low)
        intersection_high = min(actual_high, expected_high)
        
        if intersection_high <= intersection_low:
            return 0.0  # No overlap
        
        # Calculate union
        union_low = min(actual_low, expected_low)
        union_high = max(actual_high, expected_high)
        
        intersection_size = intersection_high - intersection_low
        union_size = union_high - union_low
        
        return intersection_size / union_size if union_size > 0 else 0.0


class SpatialValidator(ValidationRule):
    """Validates spatial relationships and morphological coherence."""
    
    def __init__(self):
        super().__init__("spatial_validation", ValidationCategory.SCIENTIFIC_QUALITY)
    
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Validate spatial properties of the data."""
        coords = data.get('coords')
        ion_counts = data.get('ion_counts')
        
        if coords is None:
            return self._create_result(
                ValidationSeverity.WARNING,
                "No coordinate data for spatial validation",
                quality_score=0.5
            )
        
        metrics = {}
        recommendations = []
        issues = []
        
        # Spatial distribution analysis
        spatial_metrics = self._analyze_spatial_distribution(coords)
        metrics.update(spatial_metrics['metrics'])
        issues.extend(spatial_metrics['issues'])
        
        # If ion counts available, analyze spatial patterns
        if ion_counts:
            spatial_pattern_metrics = self._analyze_spatial_patterns(coords, ion_counts)
            metrics.update(spatial_pattern_metrics['metrics'])
            issues.extend(spatial_pattern_metrics['issues'])
            recommendations.extend(spatial_pattern_metrics['recommendations'])
        
        # Morphological coherence
        morphology_metrics = self._analyze_morphological_coherence(coords)
        metrics.update(morphology_metrics['metrics'])
        issues.extend(morphology_metrics['issues'])
        
        # Overall spatial quality
        quality_components = []
        for metric_name in ['spatial_uniformity', 'spatial_autocorr_quality', 'morphological_coherence']:
            if metric_name in metrics:
                quality_components.append(metrics[metric_name].value)
        
        overall_quality = np.mean(quality_components) if quality_components else 0.5
        
        metrics['spatial_quality'] = ValidationMetric(
            'spatial_quality',
            overall_quality,
            expected_range=(0.6, 1.0),
            description="Overall spatial quality score"
        )
        
        # Determine severity
        if overall_quality < 0.4:
            severity = ValidationSeverity.WARNING
            message = f"Spatial quality concerns (score: {overall_quality:.2f})"
        else:
            severity = ValidationSeverity.PASS
            message = f"Spatial validation passed (score: {overall_quality:.2f})"
        
        return self._create_result(
            severity=severity,
            message=message,
            quality_score=overall_quality,
            metrics=metrics,
            recommendations=recommendations,
            context={'issues': issues}
        )
    
    def _analyze_spatial_distribution(self, coords: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial distribution properties."""
        metrics = {}
        issues = []
        
        if len(coords) < 50:
            return {'metrics': metrics, 'issues': ["Too few points for spatial analysis"]}
        
        # Calculate spatial bounds
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        area = x_range * y_range
        
        # Spatial uniformity using quadrat analysis
        n_quadrats = 16  # 4x4 grid
        quadrat_size_x = x_range / 4
        quadrat_size_y = y_range / 4
        
        quadrat_counts = []
        for i in range(4):
            for j in range(4):
                x_min = coords[:, 0].min() + i * quadrat_size_x
                x_max = x_min + quadrat_size_x
                y_min = coords[:, 1].min() + j * quadrat_size_y
                y_max = y_min + quadrat_size_y
                
                in_quadrat = ((coords[:, 0] >= x_min) & (coords[:, 0] < x_max) &
                             (coords[:, 1] >= y_min) & (coords[:, 1] < y_max))
                quadrat_counts.append(np.sum(in_quadrat))
        
        # Calculate uniformity (lower CV = more uniform)
        mean_count = np.mean(quadrat_counts)
        cv_counts = np.std(quadrat_counts) / mean_count if mean_count > 0 else 10
        
        # Convert to uniformity score (0 = completely non-uniform, 1 = perfectly uniform)
        uniformity_score = np.exp(-cv_counts)
        
        metrics['spatial_uniformity'] = ValidationMetric(
            'spatial_uniformity',
            uniformity_score,
            expected_range=(0.3, 0.8),  # Perfect uniformity might indicate gridding artifacts
            description="Spatial uniformity of measurement points"
        )
        
        if uniformity_score < 0.2:
            issues.append("Very non-uniform spatial distribution")
        elif uniformity_score > 0.9:
            issues.append("Suspiciously uniform distribution - possible gridding artifact")
        
        return {'metrics': metrics, 'issues': issues}
    
    def _analyze_spatial_patterns(self, coords: np.ndarray, ion_counts: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze spatial patterns in protein expression."""
        metrics = {}
        issues = []
        recommendations = []
        
        if len(coords) < 100:
            return {'metrics': metrics, 'issues': [], 'recommendations': []}
        
        # Calculate spatial autocorrelation for each protein
        autocorr_scores = []
        
        for protein, counts in ion_counts.items():
            if len(counts) == len(coords):
                # Sample subset for performance
                sample_size = min(500, len(coords))
                sample_indices = np.random.choice(len(coords), sample_size, replace=False)
                
                sample_coords = coords[sample_indices]
                sample_counts = counts[sample_indices]
                
                # Calculate Moran's I (spatial autocorrelation)
                autocorr = self._calculate_morans_i(sample_coords, sample_counts)
                
                if not np.isnan(autocorr):
                    autocorr_scores.append(abs(autocorr))  # Absolute value for clustering strength
                    
                    metrics[f'{protein}_spatial_autocorr'] = ValidationMetric(
                        f'{protein}_spatial_autocorr',
                        autocorr,
                        expected_range=(-1.0, 1.0),
                        description=f"Spatial autocorrelation for {protein}"
                    )
        
        # Overall autocorrelation quality
        if autocorr_scores:
            # Moderate autocorrelation is expected for biological data
            mean_autocorr = np.mean(autocorr_scores)
            
            # Score based on reasonable autocorrelation (not too high, not too low)
            if 0.1 <= mean_autocorr <= 0.5:
                autocorr_quality = 1.0
            elif mean_autocorr < 0.1:
                autocorr_quality = mean_autocorr / 0.1  # Scale up to 1.0
            else:
                autocorr_quality = max(0.0, 1.0 - (mean_autocorr - 0.5) * 2)
            
            metrics['spatial_autocorr_quality'] = ValidationMetric(
                'spatial_autocorr_quality',
                autocorr_quality,
                expected_range=(0.7, 1.0),
                description="Quality of spatial autocorrelation patterns"
            )
            
            if mean_autocorr < 0.05:
                issues.append("Very low spatial autocorrelation - possible noise or poor resolution")
                recommendations.append("Check imaging resolution and signal-to-noise ratio")
            elif mean_autocorr > 0.7:
                issues.append("Very high spatial autocorrelation - possible technical artifacts")
                recommendations.append("Check for systematic spatial biases or artifacts")
        
        return {
            'metrics': metrics,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _analyze_morphological_coherence(self, coords: np.ndarray) -> Dict[str, Any]:
        """Analyze morphological coherence of tissue structure."""
        metrics = {}
        issues = []
        
        if len(coords) < 100:
            return {'metrics': metrics, 'issues': []}
        
        # Sample for performance
        sample_size = min(1000, len(coords))
        sample_indices = np.random.choice(len(coords), sample_size, replace=False)
        sample_coords = coords[sample_indices]
        
        # Calculate convex hull to tissue area ratio
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(sample_coords)
            hull_area = hull.volume  # 2D volume is area
            
            # Estimate actual tissue area using alpha shapes or similar
            # For simplicity, use bounding box area
            bbox_area = ((sample_coords[:, 0].max() - sample_coords[:, 0].min()) *
                        (sample_coords[:, 1].max() - sample_coords[:, 1].min()))
            
            if bbox_area > 0:
                compactness = hull_area / bbox_area
                metrics['tissue_compactness'] = ValidationMetric(
                    'tissue_compactness',
                    compactness,
                    expected_range=(0.3, 1.0),
                    description="Compactness of tissue structure"
                )
        
        except Exception as e:
            self.logger.warning(f"Morphological analysis failed: {str(e)}")
        
        # Edge density analysis
        edge_score = self._calculate_edge_density(sample_coords)
        metrics['morphological_coherence'] = ValidationMetric(
            'morphological_coherence',
            edge_score,
            expected_range=(0.5, 1.0),
            description="Morphological coherence score"
        )
        
        return {'metrics': metrics, 'issues': issues}
    
    def _calculate_morans_i(self, coords: np.ndarray, values: np.ndarray) -> float:
        """Calculate Moran's I spatial autocorrelation coefficient."""
        try:
            n = len(coords)
            if n < 10:
                return np.nan
            
            # Create spatial weights matrix (inverse distance)
            distances = squareform(pdist(coords))
            np.fill_diagonal(distances, np.inf)  # Avoid self-weighting
            
            # Use inverse distance weights with cutoff
            max_distance = np.percentile(distances[distances < np.inf], 90)
            weights = np.where(distances <= max_distance, 1.0 / distances, 0.0)
            np.fill_diagonal(weights, 0.0)
            
            # Normalize weights
            row_sums = np.sum(weights, axis=1)
            weights = weights / row_sums[:, np.newaxis]
            weights[np.isnan(weights)] = 0
            
            # Calculate Moran's I
            mean_val = np.mean(values)
            deviations = values - mean_val
            
            numerator = 0
            denominator = np.sum(deviations**2)
            
            for i in range(n):
                for j in range(n):
                    numerator += weights[i, j] * deviations[i] * deviations[j]
            
            if denominator > 0:
                morans_i = (n / np.sum(weights)) * (numerator / denominator)
                return morans_i
            else:
                return np.nan
                
        except Exception:
            return np.nan
    
    def _calculate_edge_density(self, coords: np.ndarray) -> float:
        """Calculate edge density as measure of morphological coherence."""
        try:
            from scipy.spatial import Delaunay
            
            # Create Delaunay triangulation
            tri = Delaunay(coords)
            
            # Calculate edge lengths
            edge_lengths = []
            for simplex in tri.simplices:
                for i in range(3):
                    p1 = coords[simplex[i]]
                    p2 = coords[simplex[(i + 1) % 3]]
                    edge_length = np.linalg.norm(p1 - p2)
                    edge_lengths.append(edge_length)
            
            # Coefficient of variation of edge lengths (lower = more coherent)
            cv_edges = np.std(edge_lengths) / np.mean(edge_lengths)
            
            # Convert to coherence score
            coherence = np.exp(-cv_edges)
            return coherence
            
        except Exception:
            return 0.5  # Default moderate score