"""
Clustering Quality Metrics
Provides validation metrics for clustering quality assessment
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class ValidationResult:
    """Encapsulates validation metrics for a clustering result"""
    metric_name: str
    score: float
    details: Dict[str, Any]
    interpretation: str
    
    def is_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if validation score meets quality threshold"""
        return self.score >= threshold


class ClusterValidator(ABC):
    """Abstract base for clustering validation strategies"""
    
    @abstractmethod
    def validate(self, data: np.ndarray, labels: np.ndarray, 
                 **kwargs) -> ValidationResult:
        """
        Validate clustering quality
        
        Args:
            data: Original data matrix (n_samples, n_features)
            labels: Cluster assignments
            **kwargs: Additional parameters
            
        Returns:
            ValidationResult with metrics and interpretation
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return validator name"""
        pass


class SilhouetteValidator(ClusterValidator):
    """Silhouette analysis for cluster cohesion and separation"""
    
    def validate(self, data: np.ndarray, labels: np.ndarray, 
                 **kwargs) -> ValidationResult:
        """Compute silhouette score"""
        if len(np.unique(labels)) < 2:
            return ValidationResult(
                metric_name="silhouette",
                score=0.0,
                details={"n_clusters": len(np.unique(labels))},
                interpretation="Single cluster - silhouette not applicable"
            )
        
        score = silhouette_score(data, labels)
        
        # Interpretation based on standard thresholds
        if score > 0.7:
            interpretation = "Strong clustering structure"
        elif score > 0.5:
            interpretation = "Reasonable clustering structure"
        elif score > 0.25:
            interpretation = "Weak clustering structure"
        else:
            interpretation = "No meaningful clustering structure"
        
        return ValidationResult(
            metric_name="silhouette",
            score=score,
            details={
                "n_clusters": len(np.unique(labels)),
                "n_samples": len(data)
            },
            interpretation=interpretation
        )
    
    def name(self) -> str:
        return "Silhouette Analysis"


class SpatialCoherenceValidator(ClusterValidator):
    """Custom spatial coherence metric for tissue clustering"""
    
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = coordinates
    
    def validate(self, data: np.ndarray, labels: np.ndarray, 
                 **kwargs) -> ValidationResult:
        """Compute spatial coherence score"""
        from scipy.spatial.distance import pdist, squareform
        
        # Compute spatial distance matrix
        spatial_dist = squareform(pdist(self.coordinates))
        
        # For each cluster, compute intra-cluster spatial compactness
        coherence_scores = []
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_mask = labels == cluster_id
            if np.sum(cluster_mask) < 2:
                continue
            
            # Mean intra-cluster distance
            cluster_distances = spatial_dist[cluster_mask][:, cluster_mask]
            mean_intra_dist = np.mean(cluster_distances[np.triu_indices_from(cluster_distances, k=1)])
            
            # Mean distance to other clusters
            other_distances = spatial_dist[cluster_mask][:, ~cluster_mask]
            mean_inter_dist = np.mean(other_distances)
            
            # Coherence: inter/intra ratio (higher is better)
            if mean_intra_dist > 0:
                coherence = mean_inter_dist / mean_intra_dist
                coherence_scores.append(coherence)
        
        overall_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        # Normalize to 0-1 scale (approximate)
        normalized_score = min(1.0, overall_coherence / 3.0)
        
        return ValidationResult(
            metric_name="spatial_coherence",
            score=normalized_score,
            details={
                "raw_coherence": overall_coherence,
                "n_valid_clusters": len(coherence_scores)
            },
            interpretation=f"Spatial coherence: {normalized_score:.3f}"
        )
    
    def name(self) -> str:
        return "Spatial Coherence"


class ValidationSuite:
    """Validation for clustering results"""
    
    def __init__(self, coordinates: Optional[np.ndarray] = None):
        self.validators = [SilhouetteValidator()]
        if coordinates is not None:
            self.validators.append(SpatialCoherenceValidator(coordinates))
    
    def validate_clustering(self, data: np.ndarray, labels: np.ndarray) -> List[ValidationResult]:
        """Run all validation metrics"""
        results = []
        for validator in self.validators:
            try:
                result = validator.validate(data, labels)
                results.append(result)
            except Exception as e:
                # Create a failed result
                results.append(ValidationResult(
                    metric_name=validator.name().lower().replace(" ", "_"),
                    score=0.0,
                    details={"error": str(e)},
                    interpretation=f"Validation failed: {str(e)}"
                ))
        return results
    
    def summary_score(self, results: List[ValidationResult]) -> float:
        """Compute overall quality score"""
        scores = [r.score for r in results if r.score > 0]
        return np.mean(scores) if scores else 0.0