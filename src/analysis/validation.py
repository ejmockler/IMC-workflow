"""
Validation Framework for Spatial Clustering
Provides rigorous validation metrics for clustering quality assessment
Following best practices from OPTIMAL framework and recent IMC literature
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
        """Return validator name for reporting"""
        pass


class ConsistencyValidator(ClusterValidator):
    """
    Validates clustering stability across multiple runs
    Inspired by Pixie pipeline's consistency score
    """
    
    def __init__(self, n_runs: int = 5):
        self.n_runs = n_runs
    
    def name(self) -> str:
        return "consistency"
    
    def validate(self, data: np.ndarray, labels: np.ndarray, 
                 **kwargs) -> ValidationResult:
        """
        Run clustering multiple times and measure consistency
        Lower variance = higher consistency = better score
        """
        n_clusters = len(np.unique(labels))
        clustering_method = kwargs.get('clustering_method', KMeans)
        
        # Store cluster assignments from multiple runs
        all_labels = []
        for run in range(self.n_runs):
            if clustering_method == KMeans:
                clusterer = KMeans(n_clusters=n_clusters, 
                                 random_state=42 + run, 
                                 n_init=10)
                run_labels = clusterer.fit_predict(data)
            else:
                # For other methods, assume they have similar interface
                run_labels = clustering_method(data, n_clusters, seed=42+run)
            all_labels.append(run_labels)
        
        # Calculate pairwise consistency
        consistency_scores = []
        for i in range(self.n_runs):
            for j in range(i+1, self.n_runs):
                # Use Adjusted Rand Index to measure similarity
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                consistency_scores.append(ari)
        
        mean_consistency = np.mean(consistency_scores)
        std_consistency = np.std(consistency_scores)
        
        interpretation = self._interpret_score(mean_consistency)
        
        return ValidationResult(
            metric_name="Clustering Consistency",
            score=mean_consistency,
            details={
                'n_runs': self.n_runs,
                'mean_ari': mean_consistency,
                'std_ari': std_consistency,
                'all_scores': consistency_scores
            },
            interpretation=interpretation
        )
    
    def _interpret_score(self, score: float) -> str:
        if score >= 0.9:
            return "Excellent: Highly stable clustering"
        elif score >= 0.75:
            return "Good: Stable clustering with minor variations"
        elif score >= 0.6:
            return "Moderate: Some instability in cluster assignments"
        else:
            return "Poor: Unstable clustering, consider different parameters"


class SilhouetteValidator(ClusterValidator):
    """
    Validates cluster separation using silhouette coefficient
    Standard metric in IMC analysis
    """
    
    def name(self) -> str:
        return "silhouette"
    
    def validate(self, data: np.ndarray, labels: np.ndarray,
                 **kwargs) -> ValidationResult:
        """
        Calculate silhouette score for cluster quality
        Range: [-1, 1], higher is better
        """
        # Handle edge cases
        n_clusters = len(np.unique(labels))
        if n_clusters <= 1:
            return ValidationResult(
                metric_name="Silhouette Score",
                score=0.0,
                details={'n_clusters': n_clusters},
                interpretation="Cannot compute: Only one cluster found"
            )
        
        # Subsample for large datasets (performance)
        max_samples = kwargs.get('max_samples', 10000)
        if len(data) > max_samples:
            indices = np.random.choice(len(data), max_samples, replace=False)
            sample_data = data[indices]
            sample_labels = labels[indices]
        else:
            sample_data = data
            sample_labels = labels
        
        # Calculate silhouette score
        try:
            score = silhouette_score(sample_data, sample_labels)
        except ValueError as e:
            return ValidationResult(
                metric_name="Silhouette Score",
                score=0.0,
                details={'error': str(e)},
                interpretation="Error computing silhouette score"
            )
        
        # Calculate per-cluster scores for detailed analysis
        cluster_scores = {}
        for cluster_id in np.unique(sample_labels):
            mask = sample_labels == cluster_id
            if np.sum(mask) > 1:
                cluster_data = sample_data[mask]
                cluster_labels_binary = np.zeros(len(sample_data))
                cluster_labels_binary[mask] = 1
                if len(np.unique(cluster_labels_binary)) > 1:
                    cluster_score = silhouette_score(sample_data, 
                                                    cluster_labels_binary)
                    cluster_scores[int(cluster_id)] = float(cluster_score)
        
        interpretation = self._interpret_score(score)
        
        return ValidationResult(
            metric_name="Silhouette Score",
            score=float(score),
            details={
                'n_clusters': n_clusters,
                'n_samples': len(sample_labels),
                'per_cluster_scores': cluster_scores,
                'subsampled': len(data) > max_samples
            },
            interpretation=interpretation
        )
    
    def _interpret_score(self, score: float) -> str:
        if score >= 0.7:
            return "Excellent: Strong cluster separation"
        elif score >= 0.5:
            return "Good: Clear cluster structure"
        elif score >= 0.25:
            return "Moderate: Overlapping clusters"
        elif score >= 0:
            return "Weak: Poor cluster separation"
        else:
            return "Poor: Clusters not well-defined"


class SpatialCoherenceValidator(ClusterValidator):
    """
    Validates spatial coherence of clusters
    Ensures clusters form spatially contiguous regions
    """
    
    def name(self) -> str:
        return "spatial_coherence"
    
    def validate(self, data: np.ndarray, labels: np.ndarray,
                 **kwargs) -> ValidationResult:
        """
        Measure spatial autocorrelation of cluster assignments
        Requires coordinates to be passed in kwargs
        """
        coords = kwargs.get('coords')
        if coords is None:
            return ValidationResult(
                metric_name="Spatial Coherence",
                score=0.0,
                details={'error': 'No coordinates provided'},
                interpretation="Cannot compute without spatial coordinates"
            )
        
        # Calculate Moran's I for spatial autocorrelation
        morans_i = self._calculate_morans_i(coords, labels)
        
        # Calculate fragmentation (number of connected components per cluster)
        fragmentation = self._calculate_fragmentation(coords, labels)
        
        # Combined score (high Moran's I, low fragmentation)
        coherence_score = morans_i * (1 - fragmentation)
        
        interpretation = self._interpret_score(coherence_score)
        
        return ValidationResult(
            metric_name="Spatial Coherence",
            score=float(coherence_score),
            details={
                'morans_i': float(morans_i),
                'fragmentation': float(fragmentation),
                'n_clusters': len(np.unique(labels))
            },
            interpretation=interpretation
        )
    
    def _calculate_morans_i(self, coords: np.ndarray, 
                           labels: np.ndarray) -> float:
        """
        Simplified Moran's I calculation for spatial autocorrelation
        Range: [-1, 1], higher indicates stronger spatial clustering
        """
        from scipy.spatial import distance_matrix
        
        # Create spatial weights matrix (inverse distance)
        dist_matrix = distance_matrix(coords, coords)
        np.fill_diagonal(dist_matrix, np.inf)
        weights = 1.0 / (dist_matrix + 1e-10)
        weights[dist_matrix > 50] = 0  # Cutoff at 50μm
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Calculate Moran's I
        n = len(labels)
        label_mean = labels.mean()
        label_dev = labels - label_mean
        
        numerator = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    numerator += weights[i, j] * label_dev[i] * label_dev[j]
        
        denominator = np.sum(label_dev ** 2) / n
        
        if denominator == 0:
            return 0
        
        morans_i = numerator / denominator
        return min(1.0, max(-1.0, morans_i))  # Clamp to [-1, 1]
    
    def _calculate_fragmentation(self, coords: np.ndarray,
                                labels: np.ndarray) -> float:
        """
        Calculate fragmentation as ratio of components to ideal
        Lower is better (less fragmented)
        """
        from scipy.spatial import cKDTree
        
        n_clusters = len(np.unique(labels))
        total_components = 0
        
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_coords = coords[mask]
            
            if len(cluster_coords) < 2:
                total_components += 1
                continue
            
            # Build connectivity graph using KDTree
            tree = cKDTree(cluster_coords)
            # Connect points within 30μm (typical cell diameter)
            pairs = tree.query_pairs(r=30.0)
            
            # Count connected components using simple DFS
            n_points = len(cluster_coords)
            visited = [False] * n_points
            components = 0
            
            # Build adjacency list
            adj = {i: set() for i in range(n_points)}
            for i, j in pairs:
                adj[i].add(j)
                adj[j].add(i)
            
            # DFS to count components
            for i in range(n_points):
                if not visited[i]:
                    components += 1
                    stack = [i]
                    while stack:
                        node = stack.pop()
                        if not visited[node]:
                            visited[node] = True
                            stack.extend(adj[node])
            
            total_components += components
        
        # Ideal: one component per cluster
        fragmentation = (total_components - n_clusters) / max(n_clusters, 1)
        return min(1.0, fragmentation)  # Cap at 1.0
    
    def _interpret_score(self, score: float) -> str:
        if score >= 0.8:
            return "Excellent: Highly coherent spatial clusters"
        elif score >= 0.6:
            return "Good: Mostly coherent spatial regions"
        elif score >= 0.4:
            return "Moderate: Some spatial fragmentation"
        elif score >= 0.2:
            return "Weak: Significant spatial fragmentation"
        else:
            return "Poor: Clusters lack spatial coherence"


class ValidationSuite:
    """
    Orchestrates multiple validators for comprehensive assessment
    """
    
    def __init__(self, validators: Optional[List[ClusterValidator]] = None):
        if validators is None:
            # Default validator set
            self.validators = [
                ConsistencyValidator(n_runs=5),
                SilhouetteValidator(),
                SpatialCoherenceValidator()
            ]
        else:
            self.validators = validators
    
    def validate_all(self, data: np.ndarray, labels: np.ndarray,
                     coords: Optional[np.ndarray] = None,
                     **kwargs) -> Dict[str, ValidationResult]:
        """
        Run all validators and return comprehensive results
        """
        results = {}
        
        for validator in self.validators:
            # Add coords to kwargs if available
            val_kwargs = kwargs.copy()
            if coords is not None:
                val_kwargs['coords'] = coords
            
            try:
                result = validator.validate(data, labels, **val_kwargs)
                results[validator.name()] = result
            except Exception as e:
                # Graceful failure for individual validators
                results[validator.name()] = ValidationResult(
                    metric_name=validator.name(),
                    score=0.0,
                    details={'error': str(e)},
                    interpretation=f"Validation failed: {str(e)}"
                )
        
        return results
    
    def get_summary_score(self, results: Dict[str, ValidationResult]) -> float:
        """
        Calculate weighted average of all validation scores
        """
        # Weight different metrics based on importance
        weights = {
            'consistency': 0.3,
            'silhouette': 0.4,
            'spatial_coherence': 0.3
        }
        
        total_score = 0
        total_weight = 0
        
        for name, result in results.items():
            weight = weights.get(name, 0.2)  # Default weight
            total_score += result.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def format_report(self, results: Dict[str, ValidationResult]) -> str:
        """
        Generate human-readable validation report
        """
        report = ["=== Clustering Validation Report ===\n"]
        
        for name, result in results.items():
            report.append(f"\n{result.metric_name}:")
            report.append(f"  Score: {result.score:.3f}")
            report.append(f"  {result.interpretation}")
            
            # Add relevant details
            if 'n_clusters' in result.details:
                report.append(f"  Clusters: {result.details['n_clusters']}")
            if 'mean_ari' in result.details:
                report.append(f"  Mean ARI: {result.details['mean_ari']:.3f}")
            if 'morans_i' in result.details:
                report.append(f"  Moran's I: {result.details['morans_i']:.3f}")
        
        summary_score = self.get_summary_score(results)
        report.append(f"\n=== Overall Score: {summary_score:.3f} ===")
        
        if summary_score >= 0.7:
            report.append("Clustering quality: EXCELLENT")
        elif summary_score >= 0.5:
            report.append("Clustering quality: GOOD")
        elif summary_score >= 0.3:
            report.append("Clustering quality: MODERATE")
        else:
            report.append("Clustering quality: POOR - Consider adjusting parameters")
        
        return "\n".join(report)