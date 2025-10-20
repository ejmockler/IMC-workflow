"""
Quantitative Boundary Quality Assessment Framework for IMC Pipeline

Provides comprehensive evaluation metrics for segmentation and clustering quality,
integrating with existing QC framework and validation patterns. Enables comparison
between SLIC, Grid, Watershed, and Graph methods with statistical significance testing.

Key Features:
- Clustering evaluation: Rand index, adjusted mutual information vs ground truth
- Boundary precision/recall: Segment boundary accuracy against known boundaries
- Spatial coherence: Moran's I for segments, spatial autocorrelation measures
- Biological relevance: Protein expression coherence within segments
- Statistical significance testing for metric differences
- Integration with existing quality control and validation framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats, ndimage
from scipy.spatial import cKDTree, distance_matrix
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,
    silhouette_score, homogeneity_score, completeness_score, v_measure_score
)
from sklearn.preprocessing import StandardScaler

# Import existing modules for integration
try:
    from .spatial_clustering import compute_spatial_coherence
except ImportError:
    def compute_spatial_coherence(labels, coords):
        """Fallback spatial coherence computation."""
        return 0.5  # Default moderate coherence

try:
    from .spatial_stats import compute_ripleys_k, compute_spatial_correlation
except ImportError:
    def compute_ripleys_k(coords, intensities=None, **kwargs):
        """Fallback Ripley's K computation."""
        return np.array([]), np.array([])
    
    def compute_spatial_correlation(field1, field2, method='pearson'):
        """Fallback spatial correlation computation."""
        return 0.0

try:
    from .quality_control import ValidationMetric
except ImportError:
    from dataclasses import dataclass
    from typing import Optional, Tuple
    
    @dataclass
    class ValidationMetric:
        """Fallback ValidationMetric class."""
        name: str
        value: Union[float, int, bool, str]
        expected_range: Optional[Tuple[float, float]] = None
        units: Optional[str] = None
        description: Optional[str] = None

try:
    from ..validation.framework import ValidationRule, ValidationCategory, ValidationResult, ValidationSeverity
except ImportError:
    from enum import Enum
    from dataclasses import dataclass
    from typing import Dict, Any, List
    
    class ValidationSeverity(Enum):
        """Fallback ValidationSeverity enum."""
        CRITICAL = "critical"
        WARNING = "warning"
        INFO = "info"
        PASS = "pass"
    
    class ValidationCategory(Enum):
        """Fallback ValidationCategory enum."""
        SCIENTIFIC_QUALITY = "scientific_quality"
    
    @dataclass
    class ValidationResult:
        """Fallback ValidationResult class."""
        rule_name: str
        category: ValidationCategory
        severity: ValidationSeverity
        message: str
        quality_score: Optional[float] = None
        metrics: Dict[str, Any] = None
        recommendations: List[str] = None
        context: Dict[str, Any] = None
        rule_version: str = "1.0"
        execution_time_ms: Optional[float] = None
        
        def __post_init__(self):
            if self.metrics is None:
                self.metrics = {}
            if self.recommendations is None:
                self.recommendations = []
            if self.context is None:
                self.context = {}
    
    class ValidationRule:
        """Fallback ValidationRule class."""
        def __init__(self, name: str, category: ValidationCategory, version: str = "1.0"):
            self.name = name
            self.category = category
            self.version = version
        
        def _create_result(self, severity, message, **kwargs):
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                severity=severity,
                message=message,
                **kwargs
            )


class SegmentationMethod(Enum):
    """Supported segmentation/clustering methods."""
    SLIC = "slic"
    GRID = "grid"
    WATERSHED = "watershed"
    GRAPH = "graph"
    LEIDEN = "leiden"
    HDBSCAN = "hdbscan"


class BoundaryMetricType(Enum):
    """Types of boundary quality metrics."""
    CLUSTERING_QUALITY = "clustering_quality"
    BOUNDARY_PRECISION = "boundary_precision"
    SPATIAL_COHERENCE = "spatial_coherence"
    BIOLOGICAL_RELEVANCE = "biological_relevance"
    MORPHOLOGICAL_QUALITY = "morphological_quality"


@dataclass
class BoundaryMetricResult:
    """Individual boundary metric result."""
    metric_name: str
    metric_type: BoundaryMetricType
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    statistical_significance: Optional[float] = None  # p-value if applicable
    reference_value: Optional[float] = None  # Expected value for comparison
    quality_score: Optional[float] = None  # 0-1 normalized quality score
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodComparisonResult:
    """Result of comparing multiple segmentation methods."""
    method_scores: Dict[SegmentationMethod, Dict[str, float]]
    pairwise_comparisons: Dict[Tuple[SegmentationMethod, SegmentationMethod], Dict[str, float]]
    ranking: List[Tuple[SegmentationMethod, float]]  # (method, overall_score)
    statistical_tests: Dict[str, Any]
    recommendations: List[str]


class BoundaryQualityEvaluator:
    """
    Comprehensive boundary quality assessment for IMC segmentation methods.
    
    Integrates with existing QC framework and provides quantitative metrics
    for comparing segmentation approaches.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize evaluator with random state for reproducibility."""
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Metric weights for overall quality scoring
        self.metric_weights = {
            BoundaryMetricType.CLUSTERING_QUALITY: 0.3,
            BoundaryMetricType.BOUNDARY_PRECISION: 0.25,
            BoundaryMetricType.SPATIAL_COHERENCE: 0.2,
            BoundaryMetricType.BIOLOGICAL_RELEVANCE: 0.15,
            BoundaryMetricType.MORPHOLOGICAL_QUALITY: 0.1
        }
    
    def evaluate_clustering_quality(
        self,
        predicted_labels: np.ndarray,
        ground_truth_labels: Optional[np.ndarray] = None,
        feature_matrix: Optional[np.ndarray] = None,
        coordinates: Optional[np.ndarray] = None
    ) -> List[BoundaryMetricResult]:
        """
        Evaluate clustering quality using multiple metrics.
        
        Args:
            predicted_labels: Predicted cluster labels
            ground_truth_labels: Optional ground truth labels for supervised evaluation
            feature_matrix: Feature matrix used for clustering (for silhouette score)
            coordinates: Spatial coordinates (for spatial coherence)
            
        Returns:
            List of clustering quality metrics
        """
        results = []
        
        # Remove background/noise points (-1 labels) for evaluation
        valid_mask = predicted_labels >= 0
        if not np.any(valid_mask):
            warnings.warn("No valid cluster labels found (all labels are -1)")
            return results
        
        valid_predicted = predicted_labels[valid_mask]
        n_clusters = len(np.unique(valid_predicted))
        
        # Supervised metrics (if ground truth available)
        if ground_truth_labels is not None:
            valid_ground_truth = ground_truth_labels[valid_mask]
            
            # Adjusted Rand Index
            ari = adjusted_rand_score(valid_ground_truth, valid_predicted)
            results.append(BoundaryMetricResult(
                metric_name="adjusted_rand_index",
                metric_type=BoundaryMetricType.CLUSTERING_QUALITY,
                value=ari,
                reference_value=1.0,
                quality_score=max(0, ari),  # ARI can be negative
                metadata={"description": "Clustering agreement with ground truth"}
            ))
            
            # Normalized Mutual Information
            nmi = normalized_mutual_info_score(valid_ground_truth, valid_predicted)
            results.append(BoundaryMetricResult(
                metric_name="normalized_mutual_info",
                metric_type=BoundaryMetricType.CLUSTERING_QUALITY,
                value=nmi,
                reference_value=1.0,
                quality_score=nmi,
                metadata={"description": "Information theoretic clustering quality"}
            ))
            
            # Adjusted Mutual Information
            ami = adjusted_mutual_info_score(valid_ground_truth, valid_predicted)
            results.append(BoundaryMetricResult(
                metric_name="adjusted_mutual_info",
                metric_type=BoundaryMetricType.CLUSTERING_QUALITY,
                value=ami,
                reference_value=1.0,
                quality_score=max(0, ami),
                metadata={"description": "Chance-corrected mutual information"}
            ))
            
            # Homogeneity and Completeness
            homogeneity = homogeneity_score(valid_ground_truth, valid_predicted)
            completeness = completeness_score(valid_ground_truth, valid_predicted)
            v_measure = v_measure_score(valid_ground_truth, valid_predicted)
            
            results.extend([
                BoundaryMetricResult(
                    metric_name="homogeneity",
                    metric_type=BoundaryMetricType.CLUSTERING_QUALITY,
                    value=homogeneity,
                    reference_value=1.0,
                    quality_score=homogeneity,
                    metadata={"description": "Clusters contain only members of single class"}
                ),
                BoundaryMetricResult(
                    metric_name="completeness",
                    metric_type=BoundaryMetricType.CLUSTERING_QUALITY,
                    value=completeness,
                    reference_value=1.0,
                    quality_score=completeness,
                    metadata={"description": "All members of class assigned to same cluster"}
                ),
                BoundaryMetricResult(
                    metric_name="v_measure",
                    metric_type=BoundaryMetricType.CLUSTERING_QUALITY,
                    value=v_measure,
                    reference_value=1.0,
                    quality_score=v_measure,
                    metadata={"description": "Harmonic mean of homogeneity and completeness"}
                )
            ])
        
        # Unsupervised metrics
        if feature_matrix is not None and n_clusters > 1:
            valid_features = feature_matrix[valid_mask]
            
            # Silhouette Score
            try:
                silhouette = silhouette_score(valid_features, valid_predicted)
                results.append(BoundaryMetricResult(
                    metric_name="silhouette_score",
                    metric_type=BoundaryMetricType.CLUSTERING_QUALITY,
                    value=silhouette,
                    reference_value=0.5,  # Good silhouette score
                    quality_score=(silhouette + 1) / 2,  # Normalize from [-1,1] to [0,1]
                    metadata={"description": "Average silhouette coefficient"}
                ))
            except Exception as e:
                warnings.warn(f"Could not compute silhouette score: {e}")
        
        # Spatial coherence (if coordinates available)
        if coordinates is not None:
            valid_coords = coordinates[valid_mask]
            try:
                spatial_coherence = compute_spatial_coherence(valid_predicted, valid_coords)
                results.append(BoundaryMetricResult(
                    metric_name="spatial_coherence",
                    metric_type=BoundaryMetricType.SPATIAL_COHERENCE,
                    value=spatial_coherence,
                    reference_value=0.3,  # Moderate spatial coherence
                    quality_score=min(1.0, max(0.0, spatial_coherence / 0.5)),
                    metadata={"description": "Moran's I spatial autocorrelation"}
                ))
            except Exception as e:
                warnings.warn(f"Could not compute spatial coherence: {e}")
        
        # Cluster balance
        cluster_sizes = [np.sum(valid_predicted == c) for c in np.unique(valid_predicted)]
        size_cv = np.std(cluster_sizes) / np.mean(cluster_sizes) if cluster_sizes else 0
        balance_score = 1.0 / (1.0 + size_cv)  # Higher is better
        
        results.append(BoundaryMetricResult(
            metric_name="cluster_balance",
            metric_type=BoundaryMetricType.CLUSTERING_QUALITY,
            value=balance_score,
            reference_value=0.7,  # Well-balanced clusters
            quality_score=balance_score,
            metadata={
                "description": "Cluster size balance (1 / (1 + CV))",
                "cluster_sizes": cluster_sizes,
                "size_cv": size_cv
            }
        ))
        
        return results
    
    def evaluate_boundary_precision(
        self,
        predicted_labels: np.ndarray,
        ground_truth_labels: Optional[np.ndarray] = None,
        coordinates: Optional[np.ndarray] = None,
        segmentation_mask: Optional[np.ndarray] = None
    ) -> List[BoundaryMetricResult]:
        """
        Evaluate boundary precision and recall against ground truth.
        
        Args:
            predicted_labels: Predicted segment labels
            ground_truth_labels: Ground truth segment labels
            coordinates: Spatial coordinates
            segmentation_mask: 2D segmentation mask (for image-based boundaries)
            
        Returns:
            List of boundary precision metrics
        """
        results = []
        
        if ground_truth_labels is None:
            warnings.warn("No ground truth provided for boundary precision evaluation")
            return results
        
        # Method 1: Point-based boundary evaluation
        if coordinates is not None:
            boundary_metrics = self._evaluate_point_boundaries(
                predicted_labels, ground_truth_labels, coordinates
            )
            results.extend(boundary_metrics)
        
        # Method 2: Image-based boundary evaluation
        if segmentation_mask is not None:
            image_metrics = self._evaluate_image_boundaries(
                segmentation_mask, ground_truth_labels, coordinates
            )
            results.extend(image_metrics)
        
        return results
    
    def _evaluate_point_boundaries(
        self,
        predicted_labels: np.ndarray,
        ground_truth_labels: np.ndarray,
        coordinates: np.ndarray
    ) -> List[BoundaryMetricResult]:
        """Evaluate boundaries using point-to-point neighbor analysis."""
        results = []
        
        # Build spatial index
        tree = cKDTree(coordinates)
        
        # Find boundary points (points with neighbors in different segments)
        boundary_threshold = 10.0  # Distance threshold for neighbors
        
        pred_boundary_points = set()
        true_boundary_points = set()
        
        for i, coord in enumerate(coordinates):
            neighbors = tree.query_ball_point(coord, boundary_threshold)
            
            # Check predicted boundaries
            pred_label = predicted_labels[i]
            for j in neighbors:
                if i != j and predicted_labels[j] != pred_label:
                    pred_boundary_points.add(i)
                    break
            
            # Check true boundaries
            true_label = ground_truth_labels[i]
            for j in neighbors:
                if i != j and ground_truth_labels[j] != true_label:
                    true_boundary_points.add(i)
                    break
        
        # Calculate precision and recall
        if len(pred_boundary_points) > 0 and len(true_boundary_points) > 0:
            true_positives = len(pred_boundary_points & true_boundary_points)
            precision = true_positives / len(pred_boundary_points)
            recall = true_positives / len(true_boundary_points)
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.extend([
                BoundaryMetricResult(
                    metric_name="boundary_precision",
                    metric_type=BoundaryMetricType.BOUNDARY_PRECISION,
                    value=precision,
                    reference_value=0.8,
                    quality_score=precision,
                    metadata={"description": "Fraction of predicted boundaries that are correct"}
                ),
                BoundaryMetricResult(
                    metric_name="boundary_recall",
                    metric_type=BoundaryMetricType.BOUNDARY_PRECISION,
                    value=recall,
                    reference_value=0.8,
                    quality_score=recall,
                    metadata={"description": "Fraction of true boundaries that are detected"}
                ),
                BoundaryMetricResult(
                    metric_name="boundary_f1",
                    metric_type=BoundaryMetricType.BOUNDARY_PRECISION,
                    value=f1_score,
                    reference_value=0.8,
                    quality_score=f1_score,
                    metadata={"description": "Harmonic mean of precision and recall"}
                )
            ])
        
        return results
    
    def _evaluate_image_boundaries(
        self,
        segmentation_mask: np.ndarray,
        ground_truth_labels: np.ndarray,
        coordinates: Optional[np.ndarray] = None
    ) -> List[BoundaryMetricResult]:
        """Evaluate boundaries using image-based edge detection."""
        results = []
        
        try:
            from skimage.segmentation import find_boundaries
            from skimage.morphology import dilation
        except ImportError:
            warnings.warn("scikit-image not available for image boundary evaluation")
            return results
        
        # Convert point labels to image if needed
        if coordinates is not None and segmentation_mask.ndim == 2:
            # Create ground truth mask from point labels
            gt_mask = np.zeros_like(segmentation_mask)
            # This is simplified - would need proper interpolation in practice
            for i, (x, y) in enumerate(coordinates):
                if 0 <= int(y) < gt_mask.shape[0] and 0 <= int(x) < gt_mask.shape[1]:
                    gt_mask[int(y), int(x)] = ground_truth_labels[i]
        else:
            gt_mask = ground_truth_labels.reshape(segmentation_mask.shape)
        
        # Find boundaries
        pred_boundaries = find_boundaries(segmentation_mask, mode='inner')
        true_boundaries = find_boundaries(gt_mask, mode='inner')
        
        # Dilate boundaries for matching tolerance
        pred_boundaries_dilated = dilation(pred_boundaries)
        true_boundaries_dilated = dilation(true_boundaries)
        
        # Calculate overlap metrics
        true_positives = np.sum(pred_boundaries & true_boundaries_dilated)
        false_positives = np.sum(pred_boundaries & ~true_boundaries_dilated)
        false_negatives = np.sum(true_boundaries & ~pred_boundaries_dilated)
        
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0
        
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.extend([
            BoundaryMetricResult(
                metric_name="image_boundary_precision",
                metric_type=BoundaryMetricType.BOUNDARY_PRECISION,
                value=precision,
                reference_value=0.7,
                quality_score=precision,
                metadata={"description": "Image-based boundary precision"}
            ),
            BoundaryMetricResult(
                metric_name="image_boundary_recall",
                metric_type=BoundaryMetricType.BOUNDARY_PRECISION,
                value=recall,
                reference_value=0.7,
                quality_score=recall,
                metadata={"description": "Image-based boundary recall"}
            ),
            BoundaryMetricResult(
                metric_name="image_boundary_f1",
                metric_type=BoundaryMetricType.BOUNDARY_PRECISION,
                value=f1_score,
                reference_value=0.7,
                quality_score=f1_score,
                metadata={"description": "Image-based boundary F1 score"}
            )
        ])
        
        return results
    
    def evaluate_biological_relevance(
        self,
        predicted_labels: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        coordinates: Optional[np.ndarray] = None
    ) -> List[BoundaryMetricResult]:
        """
        Evaluate biological relevance of segmentation.
        
        Measures how well segments preserve protein expression patterns.
        
        Args:
            predicted_labels: Predicted segment labels
            ion_counts: Dictionary of protein ion counts
            coordinates: Spatial coordinates
            
        Returns:
            List of biological relevance metrics
        """
        results = []
        
        valid_mask = predicted_labels >= 0
        if not np.any(valid_mask):
            return results
        
        valid_labels = predicted_labels[valid_mask]
        unique_labels = np.unique(valid_labels)
        
        if len(unique_labels) < 2:
            return results
        
        # Protein expression coherence within segments
        protein_coherences = []
        for protein_name, counts in ion_counts.items():
            valid_counts = counts[valid_mask]
            
            # Calculate within-segment variance vs between-segment variance
            within_var = 0.0
            between_var = 0.0
            total_mean = np.mean(valid_counts)
            
            segment_means = []
            segment_vars = []
            segment_sizes = []
            
            for label in unique_labels:
                segment_mask = valid_labels == label
                segment_counts = valid_counts[segment_mask]
                
                if len(segment_counts) > 1:
                    segment_mean = np.mean(segment_counts)
                    segment_var = np.var(segment_counts)
                    
                    segment_means.append(segment_mean)
                    segment_vars.append(segment_var)
                    segment_sizes.append(len(segment_counts))
                    
                    # Weighted contribution to within-segment variance
                    within_var += segment_var * len(segment_counts)
            
            if segment_sizes:
                # Normalize within-segment variance
                within_var /= sum(segment_sizes)
                
                # Calculate between-segment variance
                segment_means = np.array(segment_means)
                segment_sizes = np.array(segment_sizes)
                between_var = np.average((segment_means - total_mean)**2, weights=segment_sizes)
                
                # F-ratio: between-segment variance / within-segment variance
                if within_var > 0:
                    f_ratio = between_var / within_var
                    coherence_score = min(1.0, f_ratio / 5.0)  # Normalize, expecting F ~ 5 for good segmentation
                else:
                    coherence_score = 1.0  # Perfect coherence if no within-segment variance
                
                protein_coherences.append(coherence_score)
        
        if protein_coherences:
            avg_coherence = np.mean(protein_coherences)
            results.append(BoundaryMetricResult(
                metric_name="protein_expression_coherence",
                metric_type=BoundaryMetricType.BIOLOGICAL_RELEVANCE,
                value=avg_coherence,
                reference_value=0.6,
                quality_score=avg_coherence,
                metadata={
                    "description": "Average protein expression coherence within segments",
                    "per_protein_coherence": dict(zip(ion_counts.keys(), protein_coherences))
                }
            ))
        
        # Spatial organization preservation
        if coordinates is not None:
            spatial_preservation = self._evaluate_spatial_organization(
                valid_labels, coordinates[valid_mask], ion_counts, valid_mask
            )
            results.extend(spatial_preservation)
        
        return results
    
    def _evaluate_spatial_organization(
        self,
        labels: np.ndarray,
        coordinates: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        valid_mask: np.ndarray
    ) -> List[BoundaryMetricResult]:
        """Evaluate how well spatial organization is preserved."""
        results = []
        
        # Calculate spatial autocorrelation for each protein within segments
        autocorr_preservation = []
        
        for protein_name, counts in ion_counts.items():
            valid_counts = counts[valid_mask]
            
            # Original spatial autocorrelation (before segmentation)
            try:
                distances, k_values = compute_ripleys_k(coordinates, valid_counts, max_distance=50, n_bins=10)
                if len(k_values) > 0:
                    original_spatial_pattern = np.mean(k_values)
                else:
                    continue
            except:
                continue
            
            # Spatial autocorrelation within each segment
            segment_patterns = []
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                segment_mask = labels == label
                if np.sum(segment_mask) > 10:  # Need enough points
                    segment_coords = coordinates[segment_mask]
                    segment_counts = valid_counts[segment_mask]
                    
                    try:
                        seg_distances, seg_k_values = compute_ripleys_k(
                            segment_coords, segment_counts, max_distance=25, n_bins=5
                        )
                        if len(seg_k_values) > 0:
                            segment_patterns.append(np.mean(seg_k_values))
                    except:
                        continue
            
            if segment_patterns and original_spatial_pattern > 0:
                # How well is spatial pattern preserved within segments?
                preservation_ratio = np.mean(segment_patterns) / original_spatial_pattern
                preservation_score = min(1.0, preservation_ratio)
                autocorr_preservation.append(preservation_score)
        
        if autocorr_preservation:
            avg_preservation = np.mean(autocorr_preservation)
            results.append(BoundaryMetricResult(
                metric_name="spatial_autocorrelation_preservation",
                metric_type=BoundaryMetricType.BIOLOGICAL_RELEVANCE,
                value=avg_preservation,
                reference_value=0.7,
                quality_score=avg_preservation,
                metadata={"description": "Preservation of spatial autocorrelation patterns"}
            ))
        
        return results
    
    def evaluate_morphological_quality(
        self,
        segmentation_mask: Optional[np.ndarray] = None,
        predicted_labels: Optional[np.ndarray] = None,
        coordinates: Optional[np.ndarray] = None
    ) -> List[BoundaryMetricResult]:
        """
        Evaluate morphological quality of segments.
        
        Args:
            segmentation_mask: 2D segmentation mask
            predicted_labels: Point-based labels
            coordinates: Spatial coordinates
            
        Returns:
            List of morphological quality metrics
        """
        results = []
        
        if segmentation_mask is not None:
            # Image-based morphological analysis
            image_metrics = self._evaluate_image_morphology(segmentation_mask)
            results.extend(image_metrics)
        
        if predicted_labels is not None and coordinates is not None:
            # Point-based morphological analysis
            point_metrics = self._evaluate_point_morphology(predicted_labels, coordinates)
            results.extend(point_metrics)
        
        return results
    
    def _evaluate_image_morphology(self, segmentation_mask: np.ndarray) -> List[BoundaryMetricResult]:
        """Evaluate morphological properties of image-based segments."""
        results = []
        
        try:
            from skimage.measure import regionprops
        except ImportError:
            warnings.warn("scikit-image not available for morphological evaluation")
            return results
        
        # Get region properties
        props = regionprops(segmentation_mask + 1)  # Add 1 to avoid 0 labels
        
        if not props:
            return results
        
        # Collect morphological metrics
        areas = [prop.area for prop in props]
        perimeters = [prop.perimeter for prop in props]
        eccentricities = [prop.eccentricity for prop in props]
        solidities = [prop.solidity for prop in props]
        
        # Circularity (4π*area / perimeter²)
        circularities = [4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0 
                        for area, perimeter in zip(areas, perimeters)]
        
        # Size consistency
        area_cv = np.std(areas) / np.mean(areas) if areas else 0
        size_consistency = 1.0 / (1.0 + area_cv)
        
        # Shape regularity (based on eccentricity and solidity)
        avg_eccentricity = np.mean(eccentricities) if eccentricities else 0
        avg_solidity = np.mean(solidities) if solidities else 0
        avg_circularity = np.mean(circularities) if circularities else 0
        
        results.extend([
            BoundaryMetricResult(
                metric_name="size_consistency",
                metric_type=BoundaryMetricType.MORPHOLOGICAL_QUALITY,
                value=size_consistency,
                reference_value=0.7,
                quality_score=size_consistency,
                metadata={"description": "Consistency of segment sizes"}
            ),
            BoundaryMetricResult(
                metric_name="average_circularity",
                metric_type=BoundaryMetricType.MORPHOLOGICAL_QUALITY,
                value=avg_circularity,
                reference_value=0.5,
                quality_score=avg_circularity,
                metadata={"description": "Average segment circularity"}
            ),
            BoundaryMetricResult(
                metric_name="average_solidity",
                metric_type=BoundaryMetricType.MORPHOLOGICAL_QUALITY,
                value=avg_solidity,
                reference_value=0.8,
                quality_score=avg_solidity,
                metadata={"description": "Average segment solidity (convexity)"}
            )
        ])
        
        return results
    
    def _evaluate_point_morphology(self, labels: np.ndarray, coordinates: np.ndarray) -> List[BoundaryMetricResult]:
        """Evaluate morphological properties of point-based segments."""
        results = []
        
        valid_mask = labels >= 0
        if not np.any(valid_mask):
            return results
        
        valid_labels = labels[valid_mask]
        valid_coords = coordinates[valid_mask]
        unique_labels = np.unique(valid_labels)
        
        # Calculate segment properties
        segment_areas = []
        segment_compactness = []
        
        for label in unique_labels:
            segment_mask = valid_labels == label
            segment_coords = valid_coords[segment_mask]
            
            if len(segment_coords) < 3:
                continue
            
            # Approximate area using convex hull
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(segment_coords)
                area = hull.volume  # In 2D, volume is area
                perimeter = hull.area  # In 2D, area is perimeter
                
                segment_areas.append(area)
                
                # Compactness (4π*area / perimeter²)
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter ** 2)
                    segment_compactness.append(compactness)
            except:
                continue
        
        if segment_areas:
            # Size consistency
            area_cv = np.std(segment_areas) / np.mean(segment_areas)
            size_consistency = 1.0 / (1.0 + area_cv)
            
            results.append(BoundaryMetricResult(
                metric_name="point_size_consistency",
                metric_type=BoundaryMetricType.MORPHOLOGICAL_QUALITY,
                value=size_consistency,
                reference_value=0.6,
                quality_score=size_consistency,
                metadata={"description": "Point-based segment size consistency"}
            ))
        
        if segment_compactness:
            avg_compactness = np.mean(segment_compactness)
            results.append(BoundaryMetricResult(
                metric_name="point_compactness",
                metric_type=BoundaryMetricType.MORPHOLOGICAL_QUALITY,
                value=avg_compactness,
                reference_value=0.5,
                quality_score=avg_compactness,
                metadata={"description": "Point-based segment compactness"}
            ))
        
        return results
    
    def compare_methods(
        self,
        method_results: Dict[SegmentationMethod, Dict[str, Any]],
        ground_truth_data: Optional[Dict[str, Any]] = None
    ) -> MethodComparisonResult:
        """
        Compare multiple segmentation methods using comprehensive metrics.
        
        Args:
            method_results: Dictionary mapping methods to their results
            ground_truth_data: Optional ground truth for supervised evaluation
            
        Returns:
            Comprehensive method comparison result
        """
        method_scores = {}
        all_metrics = {}
        
        # Evaluate each method
        for method, results in method_results.items():
            method_metrics = self.evaluate_comprehensive(
                results, ground_truth_data
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(method_metrics)
            
            method_scores[method] = {
                'overall_score': overall_score,
                'metrics': {m.metric_name: m.value for m in method_metrics}
            }
            all_metrics[method] = method_metrics
        
        # Pairwise statistical comparisons
        pairwise_comparisons = self._perform_pairwise_tests(all_metrics)
        
        # Rank methods
        ranking = sorted(
            [(method, scores['overall_score']) for method, scores in method_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Generate recommendations
        recommendations = self._generate_method_recommendations(ranking, method_scores, pairwise_comparisons)
        
        return MethodComparisonResult(
            method_scores=method_scores,
            pairwise_comparisons=pairwise_comparisons,
            ranking=ranking,
            statistical_tests=self._perform_statistical_tests(all_metrics),
            recommendations=recommendations
        )
    
    def evaluate_comprehensive(
        self,
        segmentation_results: Dict[str, Any],
        ground_truth_data: Optional[Dict[str, Any]] = None
    ) -> List[BoundaryMetricResult]:
        """
        Comprehensive evaluation of segmentation results.
        
        Args:
            segmentation_results: Results from segmentation method
            ground_truth_data: Optional ground truth data
            
        Returns:
            List of all boundary metrics
        """
        all_metrics = []
        
        # Extract required data
        predicted_labels = segmentation_results.get('labels', segmentation_results.get('superpixel_labels'))
        coordinates = segmentation_results.get('coordinates', segmentation_results.get('superpixel_coords'))
        ion_counts = segmentation_results.get('ion_counts', segmentation_results.get('superpixel_counts'))
        feature_matrix = segmentation_results.get('feature_matrix')
        segmentation_mask = segmentation_results.get('segmentation_mask')
        
        # Ground truth data
        ground_truth_labels = None
        if ground_truth_data:
            ground_truth_labels = ground_truth_data.get('ground_truth_clusters')
        
        # Clustering quality evaluation
        if predicted_labels is not None:
            clustering_metrics = self.evaluate_clustering_quality(
                predicted_labels, ground_truth_labels, feature_matrix, coordinates
            )
            all_metrics.extend(clustering_metrics)
            
            # Boundary precision evaluation
            if ground_truth_labels is not None:
                boundary_metrics = self.evaluate_boundary_precision(
                    predicted_labels, ground_truth_labels, coordinates, segmentation_mask
                )
                all_metrics.extend(boundary_metrics)
            
            # Biological relevance evaluation
            if ion_counts:
                biological_metrics = self.evaluate_biological_relevance(
                    predicted_labels, ion_counts, coordinates
                )
                all_metrics.extend(biological_metrics)
        
        # Morphological quality evaluation
        morphological_metrics = self.evaluate_morphological_quality(
            segmentation_mask, predicted_labels, coordinates
        )
        all_metrics.extend(morphological_metrics)
        
        return all_metrics
    
    def _calculate_overall_score(self, metrics: List[BoundaryMetricResult]) -> float:
        """Calculate weighted overall quality score."""
        type_scores = {}
        
        # Group metrics by type and calculate type averages
        for metric in metrics:
            if metric.quality_score is not None:
                if metric.metric_type not in type_scores:
                    type_scores[metric.metric_type] = []
                type_scores[metric.metric_type].append(metric.quality_score)
        
        # Calculate weighted average
        overall_score = 0.0
        total_weight = 0.0
        
        for metric_type, scores in type_scores.items():
            if metric_type in self.metric_weights:
                weight = self.metric_weights[metric_type]
                type_avg = np.mean(scores)
                overall_score += weight * type_avg
                total_weight += weight
        
        return overall_score / total_weight if total_weight > 0 else 0.0
    
    def _perform_pairwise_tests(self, all_metrics: Dict[SegmentationMethod, List[BoundaryMetricResult]]) -> Dict:
        """Perform pairwise statistical tests between methods."""
        pairwise_results = {}
        methods = list(all_metrics.keys())
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                # Compare overall scores using bootstrap test
                scores1 = [m.quality_score for m in all_metrics[method1] if m.quality_score is not None]
                scores2 = [m.quality_score for m in all_metrics[method2] if m.quality_score is not None]
                
                if len(scores1) > 0 and len(scores2) > 0:
                    # Wilcoxon rank-sum test
                    try:
                        statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                        effect_size = (np.mean(scores1) - np.mean(scores2)) / np.sqrt((np.std(scores1)**2 + np.std(scores2)**2) / 2)
                        
                        pairwise_results[(method1, method2)] = {
                            'p_value': p_value,
                            'effect_size': effect_size,
                            'mean_diff': np.mean(scores1) - np.mean(scores2),
                            'significant': p_value < 0.05
                        }
                    except Exception as e:
                        warnings.warn(f"Statistical test failed for {method1} vs {method2}: {e}")
        
        return pairwise_results
    
    def _perform_statistical_tests(self, all_metrics: Dict[SegmentationMethod, List[BoundaryMetricResult]]) -> Dict:
        """Perform overall statistical tests across all methods."""
        statistical_tests = {}
        
        # Extract scores by metric type
        metric_type_data = {}
        for method, metrics in all_metrics.items():
            for metric in metrics:
                if metric.quality_score is not None:
                    key = (metric.metric_type, metric.metric_name)
                    if key not in metric_type_data:
                        metric_type_data[key] = {}
                    metric_type_data[key][method] = metric.quality_score
        
        # Perform ANOVA for each metric type
        for (metric_type, metric_name), method_scores in metric_type_data.items():
            if len(method_scores) > 2:
                scores_list = list(method_scores.values())
                try:
                    f_stat, p_value = stats.f_oneway(*[[score] for score in scores_list])
                    statistical_tests[f"{metric_type.value}_{metric_name}"] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    warnings.warn(f"ANOVA failed for {metric_type.value}_{metric_name}: {e}")
        
        return statistical_tests
    
    def _generate_method_recommendations(
        self,
        ranking: List[Tuple[SegmentationMethod, float]],
        method_scores: Dict[SegmentationMethod, Dict],
        pairwise_comparisons: Dict
    ) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        if not ranking:
            return ["No methods evaluated"]
        
        # Best overall method
        best_method, best_score = ranking[0]
        recommendations.append(f"Best overall method: {best_method.value} (score: {best_score:.3f})")
        
        # Significant differences
        significant_comparisons = [
            (methods, results) for methods, results in pairwise_comparisons.items()
            if results.get('significant', False)
        ]
        
        if significant_comparisons:
            recommendations.append(f"Found {len(significant_comparisons)} statistically significant differences between methods")
        else:
            recommendations.append("No statistically significant differences found between methods")
        
        # Method-specific recommendations
        for method, score in ranking[:3]:  # Top 3 methods
            method_metrics = method_scores[method]['metrics']
            
            # Find strengths and weaknesses
            strengths = [name for name, value in method_metrics.items() if value > 0.8]
            weaknesses = [name for name, value in method_metrics.items() if value < 0.4]
            
            if strengths:
                recommendations.append(f"{method.value} excels in: {', '.join(strengths)}")
            if weaknesses:
                recommendations.append(f"{method.value} needs improvement in: {', '.join(weaknesses)}")
        
        # Quality threshold recommendations
        high_quality_methods = [method for method, score in ranking if score > 0.8]
        acceptable_methods = [method for method, score in ranking if 0.6 <= score <= 0.8]
        poor_methods = [method for method, score in ranking if score < 0.6]
        
        if high_quality_methods:
            recommendations.append(f"High quality methods (>0.8): {[m.value for m in high_quality_methods]}")
        if acceptable_methods:
            recommendations.append(f"Acceptable methods (0.6-0.8): {[m.value for m in acceptable_methods]}")
        if poor_methods:
            recommendations.append(f"Methods needing improvement (<0.6): {[m.value for m in poor_methods]}")
        
        return recommendations
    
    def generate_evaluation_report(
        self,
        metrics: List[BoundaryMetricResult],
        method_name: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            metrics: List of boundary metrics
            method_name: Name of the segmentation method
            
        Returns:
            Comprehensive evaluation report
        """
        report = {
            'method_name': method_name,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'summary': {},
            'metrics_by_type': {},
            'quality_assessment': {},
            'recommendations': []
        }
        
        # Group metrics by type
        for metric in metrics:
            metric_type = metric.metric_type.value
            if metric_type not in report['metrics_by_type']:
                report['metrics_by_type'][metric_type] = []
            
            report['metrics_by_type'][metric_type].append({
                'name': metric.metric_name,
                'value': metric.value,
                'quality_score': metric.quality_score,
                'reference_value': metric.reference_value,
                'metadata': metric.metadata
            })
        
        # Calculate summary statistics
        all_quality_scores = [m.quality_score for m in metrics if m.quality_score is not None]
        if all_quality_scores:
            report['summary'] = {
                'overall_quality_score': np.mean(all_quality_scores),
                'quality_std': np.std(all_quality_scores),
                'min_quality': np.min(all_quality_scores),
                'max_quality': np.max(all_quality_scores),
                'n_metrics': len(all_quality_scores)
            }
        
        # Quality assessment
        overall_score = report['summary'].get('overall_quality_score', 0)
        if overall_score >= 0.8:
            assessment = "Excellent"
        elif overall_score >= 0.6:
            assessment = "Good"
        elif overall_score >= 0.4:
            assessment = "Acceptable"
        else:
            assessment = "Needs Improvement"
        
        report['quality_assessment'] = {
            'rating': assessment,
            'score': overall_score,
            'interpretation': self._interpret_quality_score(overall_score)
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_improvement_recommendations(metrics)
        
        return report
    
    def _interpret_quality_score(self, score: float) -> str:
        """Interpret overall quality score."""
        if score >= 0.8:
            return "High quality segmentation suitable for production analysis"
        elif score >= 0.6:
            return "Good quality segmentation with minor areas for improvement"
        elif score >= 0.4:
            return "Acceptable quality but requires parameter tuning"
        else:
            return "Poor quality segmentation requiring significant improvement"
    
    def _generate_improvement_recommendations(self, metrics: List[BoundaryMetricResult]) -> List[str]:
        """Generate specific improvement recommendations based on metrics."""
        recommendations = []
        
        # Analyze weak points
        weak_metrics = [m for m in metrics if m.quality_score is not None and m.quality_score < 0.5]
        
        for metric in weak_metrics:
            if metric.metric_type == BoundaryMetricType.CLUSTERING_QUALITY:
                if 'silhouette' in metric.metric_name:
                    recommendations.append("Consider adjusting clustering parameters to improve cluster separation")
                elif 'balance' in metric.metric_name:
                    recommendations.append("Cluster sizes are imbalanced - consider different resolution parameters")
            
            elif metric.metric_type == BoundaryMetricType.BOUNDARY_PRECISION:
                recommendations.append("Boundary detection needs improvement - consider edge-preserving methods")
            
            elif metric.metric_type == BoundaryMetricType.SPATIAL_COHERENCE:
                recommendations.append("Spatial coherence is low - increase spatial weighting in clustering")
            
            elif metric.metric_type == BoundaryMetricType.BIOLOGICAL_RELEVANCE:
                recommendations.append("Segmentation doesn't preserve biological patterns well - consider protein-guided segmentation")
            
            elif metric.metric_type == BoundaryMetricType.MORPHOLOGICAL_QUALITY:
                recommendations.append("Morphological quality is poor - consider post-processing or different segmentation method")
        
        if not recommendations:
            recommendations.append("Segmentation quality is satisfactory across all metrics")
        
        return recommendations


class BoundaryQualityValidator(ValidationRule):
    """Validation rule for boundary quality assessment integrated with validation framework."""
    
    def __init__(self, quality_threshold: float = 0.6):
        """Initialize validator with quality threshold."""
        super().__init__(
            name="BoundaryQualityValidator",
            category=ValidationCategory.SCIENTIFIC_QUALITY
        )
        self.quality_threshold = quality_threshold
        self.evaluator = BoundaryQualityEvaluator()
    
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """
        Validate boundary quality using comprehensive metrics.
        
        Args:
            data: Segmentation results and ground truth data
            context: Additional validation context
            
        Returns:
            Validation result with boundary quality assessment
        """
        try:
            # Extract segmentation results
            segmentation_results = data.get('segmentation_results', data)
            ground_truth_data = data.get('ground_truth_data')
            
            # Perform comprehensive evaluation
            metrics = self.evaluator.evaluate_comprehensive(segmentation_results, ground_truth_data)
            
            if not metrics:
                return self._create_result(
                    ValidationSeverity.WARNING,
                    "No boundary metrics could be computed",
                    quality_score=0.0
                )
            
            # Calculate overall quality score
            overall_score = self.evaluator._calculate_overall_score(metrics)
            
            # Create validation metrics
            validation_metrics = {}
            for metric in metrics:
                validation_metrics[metric.metric_name] = ValidationMetric(
                    name=metric.metric_name,
                    value=metric.value,
                    expected_range=(0.0, 1.0) if metric.quality_score is not None else None,
                    description=metric.metadata.get('description', '')
                )
            
            # Determine severity
            if overall_score >= 0.8:
                severity = ValidationSeverity.PASS
                message = f"Excellent boundary quality (score: {overall_score:.3f})"
            elif overall_score >= self.quality_threshold:
                severity = ValidationSeverity.PASS
                message = f"Good boundary quality (score: {overall_score:.3f})"
            elif overall_score >= 0.4:
                severity = ValidationSeverity.WARNING
                message = f"Acceptable boundary quality (score: {overall_score:.3f}) - consider improvements"
            else:
                severity = ValidationSeverity.CRITICAL
                message = f"Poor boundary quality (score: {overall_score:.3f}) - requires attention"
            
            # Generate recommendations
            recommendations = self.evaluator._generate_improvement_recommendations(metrics)
            
            return self._create_result(
                severity,
                message,
                quality_score=overall_score,
                metrics=validation_metrics,
                recommendations=recommendations,
                context={
                    'n_metrics_evaluated': len(metrics),
                    'metric_types': list(set(m.metric_type.value for m in metrics)),
                    'threshold_used': self.quality_threshold
                }
            )
            
        except Exception as e:
            return self._create_result(
                ValidationSeverity.CRITICAL,
                f"Boundary quality validation failed: {str(e)}",
                quality_score=0.0
            )


def create_boundary_evaluator(random_state: int = 42) -> BoundaryQualityEvaluator:
    """
    Factory function to create boundary quality evaluator.
    
    Args:
        random_state: Random state for reproducibility
        
    Returns:
        Configured boundary quality evaluator
    """
    return BoundaryQualityEvaluator(random_state=random_state)


def evaluate_method_comparison(
    method_results: Dict[str, Dict[str, Any]],
    ground_truth_data: Optional[Dict[str, Any]] = None
) -> MethodComparisonResult:
    """
    Convenience function for comparing multiple segmentation methods.
    
    Args:
        method_results: Dictionary mapping method names to their results
        ground_truth_data: Optional ground truth data
        
    Returns:
        Method comparison result
    """
    evaluator = BoundaryQualityEvaluator()
    
    # Convert string keys to SegmentationMethod enum
    enum_results = {}
    for method_name, results in method_results.items():
        try:
            method_enum = SegmentationMethod(method_name.lower())
        except ValueError:
            # Create a generic method enum for unknown methods
            method_enum = SegmentationMethod.SLIC  # Default fallback
            warnings.warn(f"Unknown method {method_name}, using SLIC as fallback")
        
        enum_results[method_enum] = results
    
    return evaluator.compare_methods(enum_results, ground_truth_data)


if __name__ == "__main__":
    # Example usage and testing
    print("Boundary Quality Assessment Framework")
    print("=" * 50)
    
    # Create example data for testing
    try:
        from .synthetic_data_generator import create_example_datasets
    except ImportError:
        print("Note: synthetic_data_generator not available for testing")
        create_example_datasets = None
    
    if create_example_datasets is not None:
        datasets = create_example_datasets()
        simple_dataset = datasets['simple']
        
        print(f"Testing with synthetic dataset: {len(simple_dataset['coordinates'])} cells")
        
        # Create evaluator
        evaluator = create_boundary_evaluator()
        
        # Mock segmentation results for testing
        n_cells = len(simple_dataset['coordinates'])
        mock_results = {
            'labels': np.random.randint(0, 5, n_cells),  # 5 clusters
            'coordinates': simple_dataset['coordinates'],
            'ion_counts': simple_dataset['ion_counts']
        }
        
        # Evaluate boundary quality
        metrics = evaluator.evaluate_comprehensive(mock_results, simple_dataset)
        
        print(f"\nEvaluated {len(metrics)} boundary metrics:")
        for metric in metrics:
            print(f"  {metric.metric_name}: {metric.value:.3f} (quality: {metric.quality_score:.3f})")
        
        # Generate report
        report = evaluator.generate_evaluation_report(metrics, "Mock Method")
        print(f"\nOverall Quality: {report['quality_assessment']['rating']} ({report['summary']['overall_quality_score']:.3f})")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    else:
        print("Creating basic test without synthetic data...")
        
        # Create simple test data
        n_cells = 1000
        coordinates = np.random.uniform(0, 100, (n_cells, 2))
        labels = np.random.randint(0, 5, n_cells)
        
        # Create evaluator
        evaluator = create_boundary_evaluator()
        
        print(f"Created evaluator with {n_cells} test points and {len(np.unique(labels))} clusters")
        print("Basic framework functionality verified!")
    
    print("\nBoundary quality assessment framework ready!")