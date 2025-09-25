"""
Pipeline State Validation Rules for IMC Analysis

Validates pipeline transformations, processing steps, and intermediate states
to ensure data integrity and proper analysis workflow execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.metrics import silhouette_score
import logging

from .framework import (
    ValidationRule, ValidationResult, ValidationSeverity,
    ValidationCategory, ValidationMetric, QualityMetrics
)


class PreprocessingValidator(ValidationRule):
    """Validates preprocessing steps and their effects."""
    
    def __init__(self):
        super().__init__("preprocessing_validation", ValidationCategory.PIPELINE_STATE)
    
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Validate preprocessing quality and effects."""
        # Expect original and processed data
        original_data = data.get('original_ion_counts')
        processed_data = data.get('processed_ion_counts', data.get('ion_counts'))
        
        if not processed_data:
            return self._create_result(
                ValidationSeverity.CRITICAL,
                "No processed data available for validation",
                quality_score=0.0
            )
        
        metrics = {}
        recommendations = []
        issues = []
        
        # Basic preprocessing validation
        basic_validation = self._validate_basic_preprocessing(processed_data)
        metrics.update(basic_validation['metrics'])
        issues.extend(basic_validation['issues'])
        
        # If we have original data, validate transformation effects
        if original_data:
            transformation_validation = self._validate_preprocessing_effects(original_data, processed_data)
            metrics.update(transformation_validation['metrics'])
            issues.extend(transformation_validation['issues'])
            recommendations.extend(transformation_validation['recommendations'])
        
        # Background correction validation
        background_validation = self._validate_background_correction(processed_data, context)
        metrics.update(background_validation['metrics'])
        issues.extend(background_validation['issues'])
        recommendations.extend(background_validation['recommendations'])
        
        # Calculate overall preprocessing quality
        quality_components = []
        for metric_name in ['data_completeness', 'transformation_quality', 'background_quality']:
            if metric_name in metrics:
                quality_components.append(metrics[metric_name].value)
        
        overall_quality = np.mean(quality_components) if quality_components else 0.5
        
        metrics['preprocessing_quality'] = ValidationMetric(
            'preprocessing_quality',
            overall_quality,
            expected_range=(0.7, 1.0),
            description="Overall preprocessing quality score"
        )
        
        # Determine severity - rebalanced for scientific rigor
        if overall_quality < 0.15:
            severity = ValidationSeverity.CRITICAL
            message = f"Severe preprocessing issues (score: {overall_quality:.2f})"
        elif overall_quality < 0.35:
            severity = ValidationSeverity.WARNING
            message = f"Preprocessing quality concerns (score: {overall_quality:.2f})"
        else:
            severity = ValidationSeverity.PASS
            message = f"Preprocessing validation passed (score: {overall_quality:.2f})"
        
        return self._create_result(
            severity=severity,
            message=message,
            quality_score=overall_quality,
            metrics=metrics,
            recommendations=recommendations,
            context={'issues': issues}
        )
    
    def _validate_basic_preprocessing(self, processed_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate basic preprocessing properties."""
        metrics = {}
        issues = []
        
        # Data completeness
        n_proteins = len(processed_data)
        empty_proteins = [protein for protein, counts in processed_data.items() if len(counts) == 0]
        
        completeness = (n_proteins - len(empty_proteins)) / n_proteins if n_proteins > 0 else 0
        metrics['data_completeness'] = ValidationMetric(
            'data_completeness',
            completeness,
            expected_range=(0.9, 1.0),
            description="Fraction of proteins with valid data"
        )
        
        if empty_proteins:
            issues.append(f"Empty data for proteins: {empty_proteins}")
        
        # Value range validation
        for protein, counts in processed_data.items():
            if len(counts) > 0:
                # Check for negative values (unusual after background correction)
                negative_fraction = np.sum(counts < 0) / len(counts)
                if negative_fraction > 0.1:
                    issues.append(f"{protein}: {negative_fraction:.1%} negative values after preprocessing")
                
                # Check for extreme values
                if np.any(np.isinf(counts)) or np.any(np.isnan(counts)):
                    issues.append(f"{protein}: contains Inf/NaN values after preprocessing")
        
        return {'metrics': metrics, 'issues': issues}
    
    def _validate_preprocessing_effects(self, original_data: Dict[str, np.ndarray], processed_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate effects of preprocessing transformations."""
        metrics = {}
        issues = []
        recommendations = []
        
        transformation_scores = []
        
        for protein in original_data.keys():
            if protein in processed_data:
                original = original_data[protein]
                processed = processed_data[protein]
                
                if len(original) == len(processed) and len(original) > 0:
                    # Signal preservation
                    signal_preservation = self._calculate_signal_preservation(original, processed)
                    transformation_scores.append(signal_preservation)
                    
                    metrics[f'{protein}_signal_preservation'] = ValidationMetric(
                        f'{protein}_signal_preservation',
                        signal_preservation,
                        expected_range=(0.7, 1.0),
                        description=f"Signal preservation for {protein}"
                    )
                    
                    # Dynamic range preservation
                    original_range = np.max(original) - np.min(original)
                    processed_range = np.max(processed) - np.min(processed)
                    
                    if original_range > 0:
                        range_preservation = processed_range / original_range
                        metrics[f'{protein}_range_preservation'] = ValidationMetric(
                            f'{protein}_range_preservation',
                            range_preservation,
                            expected_range=(0.5, 1.2),
                            description=f"Dynamic range preservation for {protein}"
                        )
                    
                    # Check for over-processing
                    if signal_preservation < 0.5:
                        issues.append(f"{protein}: significant signal loss during preprocessing")
                        recommendations.append(f"Review preprocessing parameters for {protein}")
        
        # Overall transformation quality
        if transformation_scores:
            transformation_quality = np.mean(transformation_scores)
            metrics['transformation_quality'] = ValidationMetric(
                'transformation_quality',
                transformation_quality,
                expected_range=(0.7, 1.0),
                description="Overall transformation quality"
            )
        
        return {
            'metrics': metrics,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _validate_background_correction(self, processed_data: Dict[str, np.ndarray], context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate background correction effects."""
        metrics = {}
        issues = []
        recommendations = []
        
        # Look for background correction artifacts
        background_scores = []
        
        for protein, counts in processed_data.items():
            if len(counts) > 100:
                # Check for unnatural distribution shape after background correction
                bg_quality = self._assess_background_correction_quality(counts)
                background_scores.append(bg_quality)
                
                metrics[f'{protein}_background_quality'] = ValidationMetric(
                    f'{protein}_background_quality',
                    bg_quality,
                    expected_range=(0.6, 1.0),
                    description=f"Background correction quality for {protein}"
                )
                
                # Check for excessive zero-clipping
                zero_fraction = np.sum(counts == 0) / len(counts)
                if zero_fraction > 0.9:
                    issues.append(f"{protein}: {zero_fraction:.1%} zero values - possible over-correction")
                    recommendations.append(f"Consider gentler background correction for {protein}")
        
        # Overall background correction quality
        if background_scores:
            background_quality = np.mean(background_scores)
            metrics['background_quality'] = ValidationMetric(
                'background_quality',
                background_quality,
                expected_range=(0.7, 1.0),
                description="Overall background correction quality"
            )
        
        return {
            'metrics': metrics,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _calculate_signal_preservation(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate signal preservation score."""
        if len(original) == 0 or len(processed) == 0:
            return 0.0
        
        # Use correlation as signal preservation measure
        if np.std(original) > 0 and np.std(processed) > 0:
            correlation = np.corrcoef(original, processed)[0, 1]
            return max(0.0, correlation)
        else:
            return 0.0
    
    def _assess_background_correction_quality(self, counts: np.ndarray) -> float:
        """Assess quality of background correction."""
        # Look for natural distribution properties
        positive_counts = counts[counts > 0]
        
        if len(positive_counts) < 10:
            return 0.5  # Insufficient data
        
        # Check for reasonable distribution shape
        # Background-corrected data should still follow roughly Poisson-like distribution
        try:
            # Test for reasonable distribution properties
            skewness = stats.skew(positive_counts)
            kurtosis = stats.kurtosis(positive_counts)
            
            # Reasonable skewness and kurtosis for biological data
            skew_score = max(0.0, 1.0 - abs(skewness - 1.0) / 3.0)  # Expect positive skew ~1
            kurt_score = max(0.0, 1.0 - abs(kurtosis) / 5.0)  # Not too extreme kurtosis
            
            return (skew_score + kurt_score) / 2
            
        except Exception:
            return 0.5


class SegmentationValidator(ValidationRule):
    """Validates segmentation quality and properties."""
    
    def __init__(self):
        super().__init__("segmentation_validation", ValidationCategory.PIPELINE_STATE)
    
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Validate segmentation quality."""
        # Look for segmentation results
        superpixel_labels = data.get('superpixel_labels')
        superpixel_coords = data.get('superpixel_coords')
        coords = data.get('coords')
        
        if superpixel_labels is None:
            return self._create_result(
                ValidationSeverity.INFO,
                "No segmentation data available for validation",
                quality_score=0.5
            )
        
        metrics = {}
        recommendations = []
        issues = []
        
        # Basic segmentation properties
        basic_validation = self._validate_basic_segmentation(superpixel_labels, coords)
        metrics.update(basic_validation['metrics'])
        issues.extend(basic_validation['issues'])
        
        # Superpixel quality metrics
        if superpixel_coords is not None:
            quality_validation = self._validate_superpixel_quality(superpixel_labels, superpixel_coords, coords)
            metrics.update(quality_validation['metrics'])
            issues.extend(quality_validation['issues'])
            recommendations.extend(quality_validation['recommendations'])
        
        # Segmentation coherence
        coherence_validation = self._validate_segmentation_coherence(superpixel_labels)
        metrics.update(coherence_validation['metrics'])
        issues.extend(coherence_validation['issues'])
        
        # Calculate overall segmentation quality
        quality_components = []
        for metric_name in ['superpixel_size_consistency', 'superpixel_compactness', 'segmentation_coverage']:
            if metric_name in metrics:
                quality_components.append(metrics[metric_name].value)
        
        overall_quality = np.mean(quality_components) if quality_components else 0.5
        
        metrics['segmentation_quality'] = ValidationMetric(
            'segmentation_quality',
            overall_quality,
            expected_range=(0.6, 1.0),
            description="Overall segmentation quality score"
        )
        
        # Determine severity
        if overall_quality < 0.4:
            severity = ValidationSeverity.WARNING
            message = f"Segmentation quality concerns (score: {overall_quality:.2f})"
        else:
            severity = ValidationSeverity.PASS
            message = f"Segmentation validation passed (score: {overall_quality:.2f})"
        
        return self._create_result(
            severity=severity,
            message=message,
            quality_score=overall_quality,
            metrics=metrics,
            recommendations=recommendations,
            context={'issues': issues}
        )
    
    def _validate_basic_segmentation(self, superpixel_labels: np.ndarray, coords: Optional[np.ndarray]) -> Dict[str, Any]:
        """Validate basic segmentation properties."""
        metrics = {}
        issues = []
        
        # Number of superpixels
        unique_labels = np.unique(superpixel_labels)
        valid_labels = unique_labels[unique_labels >= 0]  # Exclude background (-1)
        n_superpixels = len(valid_labels)
        
        metrics['n_superpixels'] = ValidationMetric(
            'n_superpixels',
            n_superpixels,
            description="Number of superpixels created"
        )
        
        # Coverage (fraction of pixels assigned to superpixels)
        if superpixel_labels.size > 0:
            assigned_pixels = np.sum(superpixel_labels >= 0)
            coverage = assigned_pixels / superpixel_labels.size
            
            metrics['segmentation_coverage'] = ValidationMetric(
                'segmentation_coverage',
                coverage,
                expected_range=(0.8, 1.0),
                description="Fraction of pixels assigned to superpixels"
            )
            
            if coverage < 0.7:
                issues.append(f"Low segmentation coverage: {coverage:.1%}")
        
        # Superpixel size distribution
        if n_superpixels > 0:
            superpixel_sizes = []
            for label in valid_labels:
                size = np.sum(superpixel_labels == label)
                superpixel_sizes.append(size)
            
            size_cv = np.std(superpixel_sizes) / np.mean(superpixel_sizes)
            metrics['superpixel_size_consistency'] = ValidationMetric(
                'superpixel_size_consistency',
                max(0.0, 1.0 - size_cv),  # Lower CV = higher consistency
                expected_range=(0.5, 1.0),
                description="Consistency of superpixel sizes"
            )
            
            if size_cv > 2.0:
                issues.append(f"Very inconsistent superpixel sizes (CV: {size_cv:.2f})")
        
        return {'metrics': metrics, 'issues': issues}
    
    def _validate_superpixel_quality(self, superpixel_labels: np.ndarray, superpixel_coords: np.ndarray, coords: Optional[np.ndarray]) -> Dict[str, Any]:
        """Validate superpixel quality metrics."""
        metrics = {}
        issues = []
        recommendations = []
        
        # Superpixel compactness
        if coords is not None and len(superpixel_coords) > 0:
            compactness_scores = []
            
            unique_labels = np.unique(superpixel_labels)
            valid_labels = unique_labels[unique_labels >= 0]
            
            for i, label in enumerate(valid_labels[:min(50, len(valid_labels))]):  # Sample for performance
                if i < len(superpixel_coords):
                    # Get pixels belonging to this superpixel
                    pixel_mask = superpixel_labels == label
                    
                    if superpixel_labels.ndim == 2:
                        # 2D label array - get coordinates of labeled pixels
                        y_coords, x_coords = np.where(pixel_mask)
                        pixel_coords = np.column_stack([x_coords, y_coords])
                    else:
                        # 1D case - use provided coordinates
                        pixel_coords = coords[pixel_mask] if len(coords) == len(superpixel_labels) else None
                    
                    if pixel_coords is not None and len(pixel_coords) > 2:
                        # Calculate compactness (area / perimeter^2)
                        compactness = self._calculate_superpixel_compactness(pixel_coords)
                        if not np.isnan(compactness):
                            compactness_scores.append(compactness)
            
            if compactness_scores:
                mean_compactness = np.mean(compactness_scores)
                metrics['superpixel_compactness'] = ValidationMetric(
                    'superpixel_compactness',
                    mean_compactness,
                    expected_range=(0.3, 1.0),
                    description="Average superpixel compactness"
                )
                
                if mean_compactness < 0.2:
                    issues.append("Low superpixel compactness - very irregular shapes")
                    recommendations.append("Consider increasing SLIC compactness parameter")
        
        return {
            'metrics': metrics,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _validate_segmentation_coherence(self, superpixel_labels: np.ndarray) -> Dict[str, Any]:
        """Validate segmentation coherence and connectivity."""
        metrics = {}
        issues = []
        
        if superpixel_labels.ndim == 2:
            # Check for fragmented superpixels (multiple disconnected components)
            unique_labels = np.unique(superpixel_labels)
            valid_labels = unique_labels[unique_labels >= 0]
            
            fragmented_count = 0
            
            for label in valid_labels[:min(20, len(valid_labels))]:  # Sample for performance
                # Check connectivity using 4-connectivity
                mask = superpixel_labels == label
                connected_components = self._count_connected_components(mask)
                
                if connected_components > 1:
                    fragmented_count += 1
            
            if len(valid_labels) > 0:
                fragmentation_rate = fragmented_count / min(20, len(valid_labels))
                metrics['superpixel_fragmentation'] = ValidationMetric(
                    'superpixel_fragmentation',
                    fragmentation_rate,
                    expected_range=(0.0, 0.1),
                    description="Fraction of fragmented superpixels"
                )
                
                if fragmentation_rate > 0.2:
                    issues.append(f"High superpixel fragmentation: {fragmentation_rate:.1%}")
        
        return {'metrics': metrics, 'issues': issues}
    
    def _calculate_superpixel_compactness(self, pixel_coords: np.ndarray) -> float:
        """Calculate compactness of a superpixel."""
        try:
            from scipy.spatial import ConvexHull
            
            if len(pixel_coords) < 3:
                return np.nan
            
            # Use convex hull as approximation
            hull = ConvexHull(pixel_coords)
            area = hull.volume  # 2D volume is area
            perimeter = 0
            
            # Calculate perimeter
            for simplex in hull.simplices:
                p1 = pixel_coords[simplex[0]]
                p2 = pixel_coords[simplex[1]]
                perimeter += np.linalg.norm(p1 - p2)
            
            if perimeter > 0:
                # Isoperimetric quotient: 4π * area / perimeter^2
                compactness = 4 * np.pi * area / (perimeter ** 2)
                return min(1.0, compactness)  # Cap at 1.0
            else:
                return np.nan
                
        except Exception:
            return np.nan
    
    def _count_connected_components(self, binary_mask: np.ndarray) -> int:
        """Count connected components in binary mask."""
        try:
            from scipy.ndimage import label
            labeled_array, num_features = label(binary_mask)
            return num_features
        except Exception:
            return 1  # Default to 1 if analysis fails


class ClusteringValidator(ValidationRule):
    """Validates clustering quality and biological relevance."""
    
    def __init__(self):
        super().__init__("clustering_validation", ValidationCategory.PIPELINE_STATE)
    
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Validate clustering quality."""
        feature_matrix = data.get('feature_matrix')
        cluster_labels = data.get('cluster_labels')
        protein_names = data.get('protein_names')
        
        if feature_matrix is None or cluster_labels is None:
            return self._create_result(
                ValidationSeverity.INFO,
                "No clustering data available for validation",
                quality_score=0.5
            )
        
        metrics = {}
        recommendations = []
        issues = []
        
        # Basic clustering validation
        basic_validation = self._validate_basic_clustering(feature_matrix, cluster_labels)
        metrics.update(basic_validation['metrics'])
        issues.extend(basic_validation['issues'])
        
        # Clustering quality metrics
        quality_validation = self._validate_clustering_quality(feature_matrix, cluster_labels)
        metrics.update(quality_validation['metrics'])
        issues.extend(quality_validation['issues'])
        recommendations.extend(quality_validation['recommendations'])
        
        # Biological relevance validation
        if protein_names:
            biological_validation = self._validate_biological_relevance(data, cluster_labels, protein_names)
            metrics.update(biological_validation['metrics'])
            issues.extend(biological_validation['issues'])
        
        # Calculate overall clustering quality
        quality_components = []
        for metric_name in ['silhouette_score', 'cluster_separation', 'cluster_size_balance']:
            if metric_name in metrics:
                quality_components.append(metrics[metric_name].value)
        
        overall_quality = np.mean(quality_components) if quality_components else 0.5
        
        metrics['clustering_quality'] = ValidationMetric(
            'clustering_quality',
            overall_quality,
            expected_range=(0.5, 1.0),
            description="Overall clustering quality score"
        )
        
        # Determine severity
        if overall_quality < 0.3:
            severity = ValidationSeverity.WARNING
            message = f"Clustering quality concerns (score: {overall_quality:.2f})"
        else:
            severity = ValidationSeverity.PASS
            message = f"Clustering validation passed (score: {overall_quality:.2f})"
        
        return self._create_result(
            severity=severity,
            message=message,
            quality_score=overall_quality,
            metrics=metrics,
            recommendations=recommendations,
            context={'issues': issues}
        )
    
    def _validate_basic_clustering(self, feature_matrix: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Validate basic clustering properties."""
        metrics = {}
        issues = []
        
        # Number of clusters
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        n_samples = len(cluster_labels)
        
        metrics['n_clusters'] = ValidationMetric(
            'n_clusters',
            n_clusters,
            description="Number of clusters found"
        )
        
        # Cluster size distribution
        cluster_sizes = [np.sum(cluster_labels == cluster) for cluster in unique_clusters]
        
        if n_clusters > 1:
            size_cv = np.std(cluster_sizes) / np.mean(cluster_sizes)
            balance_score = max(0.0, 1.0 - size_cv / 2.0)  # Penalize high CV
            
            metrics['cluster_size_balance'] = ValidationMetric(
                'cluster_size_balance',
                balance_score,
                expected_range=(0.5, 1.0),
                description="Balance of cluster sizes"
            )
            
            # Check for very small clusters
            min_size = np.min(cluster_sizes)
            min_fraction = min_size / n_samples
            
            if min_fraction < 0.01:
                issues.append(f"Very small cluster found: {min_size} samples ({min_fraction:.1%})")
            
            # Check for single dominant cluster
            max_fraction = np.max(cluster_sizes) / n_samples
            if max_fraction > 0.8:
                issues.append(f"Single dominant cluster: {max_fraction:.1%} of samples")
        
        return {'metrics': metrics, 'issues': issues}
    
    def _validate_clustering_quality(self, feature_matrix: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Validate clustering quality using standard metrics."""
        metrics = {}
        issues = []
        recommendations = []
        
        n_clusters = len(np.unique(cluster_labels))
        
        if n_clusters > 1 and len(feature_matrix) > n_clusters:
            try:
                # Silhouette score
                silhouette_avg = silhouette_score(feature_matrix, cluster_labels)
                metrics['silhouette_score'] = ValidationMetric(
                    'silhouette_score',
                    silhouette_avg,
                    expected_range=(0.3, 1.0),
                    description="Average silhouette score"
                )
                
                if silhouette_avg < 0.2:
                    issues.append(f"Poor cluster separation (silhouette: {silhouette_avg:.2f})")
                    recommendations.append("Consider different clustering parameters or preprocessing")
                
            except Exception as e:
                self.logger.warning(f"Silhouette score calculation failed: {str(e)}")
        
        # Intra-cluster vs inter-cluster distances
        try:
            separation_score = self._calculate_cluster_separation(feature_matrix, cluster_labels)
            metrics['cluster_separation'] = ValidationMetric(
                'cluster_separation',
                separation_score,
                expected_range=(0.5, 1.0),
                description="Cluster separation quality"
            )
            
        except Exception as e:
            self.logger.warning(f"Cluster separation calculation failed: {str(e)}")
        
        return {
            'metrics': metrics,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _validate_biological_relevance(self, data: Dict[str, Any], cluster_labels: np.ndarray, protein_names: List[str]) -> Dict[str, Any]:
        """Validate biological relevance of clusters."""
        metrics = {}
        issues = []
        
        cluster_centroids = data.get('cluster_centroids')
        if not cluster_centroids:
            return {'metrics': metrics, 'issues': issues}
        
        # Check for biologically meaningful patterns
        unique_clusters = np.unique(cluster_labels)
        
        # Look for marker-specific clusters
        marker_specificity_scores = []
        
        for cluster_id in unique_clusters:
            if cluster_id in cluster_centroids:
                centroid = cluster_centroids[cluster_id]
                
                # Calculate marker dominance
                if protein_names and len(protein_names) > 1:
                    values = [centroid.get(protein, 0.0) for protein in protein_names]
                    
                    if np.sum(values) > 0:
                        # Calculate specificity (how dominated by single marker)
                        normalized_values = np.array(values) / np.sum(values)
                        specificity = np.max(normalized_values)
                        marker_specificity_scores.append(specificity)
        
        if marker_specificity_scores:
            mean_specificity = np.mean(marker_specificity_scores)
            metrics['cluster_specificity'] = ValidationMetric(
                'cluster_specificity',
                mean_specificity,
                expected_range=(0.3, 0.8),  # Not too specific (artifactual) or too general
                description="Average cluster marker specificity"
            )
            
            if mean_specificity < 0.2:
                issues.append("Clusters lack marker specificity - possible noise or poor separation")
            elif mean_specificity > 0.9:
                issues.append("Clusters overly specific - possible over-segmentation")
        
        return {'metrics': metrics, 'issues': issues}
    
    def _calculate_cluster_separation(self, feature_matrix: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate cluster separation score."""
        unique_clusters = np.unique(cluster_labels)
        
        if len(unique_clusters) < 2:
            return 1.0
        
        # Calculate cluster centroids
        centroids = []
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            centroid = np.mean(feature_matrix[cluster_mask], axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Calculate inter-cluster distances
        inter_distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                inter_distances.append(dist)
        
        # Calculate intra-cluster distances
        intra_distances = []
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = feature_matrix[cluster_mask]
            
            if len(cluster_data) > 1:
                centroid = np.mean(cluster_data, axis=0)
                distances = [np.linalg.norm(point - centroid) for point in cluster_data]
                intra_distances.extend(distances)
        
        # Separation score: ratio of inter to intra distances
        if intra_distances and inter_distances:
            mean_inter = np.mean(inter_distances)
            mean_intra = np.mean(intra_distances)
            
            if mean_intra > 0:
                separation = mean_inter / mean_intra
                return min(1.0, separation / 2.0)  # Normalize roughly to 0-1
        
        return 0.5  # Default moderate score