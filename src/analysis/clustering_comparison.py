"""
Clustering Method Comparison Framework

Provides comprehensive comparison between different clustering approaches
including graph-based baselines, spatial clustering, and other methods.
Enables systematic evaluation and benchmarking of clustering algorithms.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
    homogeneity_score, completeness_score, v_measure_score
)
from scipy.stats import pearsonr, spearmanr
import warnings

# Import clustering modules
from .spatial_clustering import perform_spatial_clustering, stability_analysis, compute_spatial_coherence
from .graph_clustering import GraphClusteringBaseline, create_graph_clustering_baseline
from .coabundance_features import generate_coabundance_features


class ClusteringEvaluator:
    """
    Comprehensive evaluation of clustering methods for IMC data.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def compare_clustering_methods(
        self,
        feature_matrix: np.ndarray,
        protein_names: List[str],
        spatial_coords: Optional[np.ndarray] = None,
        methods: Optional[List[str]] = None,
        evaluation_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple clustering methods on same dataset.
        
        Args:
            feature_matrix: N x P feature matrix
            protein_names: List of protein names
            spatial_coords: Optional N x 2 spatial coordinates
            methods: List of clustering methods to compare
            evaluation_metrics: List of metrics to compute
            
        Returns:
            Comprehensive comparison results
        """
        if methods is None:
            methods = ['spatial_leiden', 'graph_leiden', 'graph_louvain', 'graph_spectral']
        
        if evaluation_metrics is None:
            evaluation_metrics = [
                'silhouette_score', 'spatial_coherence', 'modularity',
                'stability', 'cluster_separability'
            ]
        
        results = {}
        
        for method in methods:
            try:
                method_results = self._run_clustering_method(
                    method, feature_matrix, protein_names, spatial_coords
                )
                
                # Compute evaluation metrics
                evaluation_results = self._compute_evaluation_metrics(
                    feature_matrix, method_results['cluster_labels'],
                    spatial_coords, evaluation_metrics, method_results
                )
                
                results[method] = {
                    'clustering_results': method_results,
                    'evaluation_metrics': evaluation_results,
                    'method_info': {
                        'algorithm': method,
                        'n_clusters': len(np.unique(method_results['cluster_labels'][method_results['cluster_labels'] >= 0])),
                        'n_noise': np.sum(method_results['cluster_labels'] == -1) if -1 in method_results['cluster_labels'] else 0
                    }
                }
                
            except Exception as e:
                warnings.warn(f"Method {method} failed: {e}")
                results[method] = {
                    'clustering_results': None,
                    'evaluation_metrics': {},
                    'method_info': {'error': str(e)}
                }
        
        # Compute pairwise method comparisons
        pairwise_comparisons = self._compute_pairwise_comparisons(results)
        
        # Rank methods by performance
        method_ranking = self._rank_methods(results, evaluation_metrics)
        
        return {
            'method_results': results,
            'pairwise_comparisons': pairwise_comparisons,
            'method_ranking': method_ranking,
            'summary': self._create_comparison_summary(results)
        }
    
    def _run_clustering_method(
        self,
        method: str,
        feature_matrix: np.ndarray,
        protein_names: List[str],
        spatial_coords: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Run specified clustering method."""
        
        if method == 'spatial_leiden':
            # Existing spatial clustering with Leiden
            cluster_labels, clustering_info = perform_spatial_clustering(
                feature_matrix, spatial_coords,
                method='leiden', resolution=1.0, spatial_weight=0.3
            )
            return {
                'cluster_labels': cluster_labels,
                'clustering_info': clustering_info,
                'method': 'spatial_leiden'
            }
        
        elif method.startswith('graph_'):
            # Graph-based clustering methods
            algorithm = method.replace('graph_', '')
            baseline = GraphClusteringBaseline(self.random_state)
            
            results = baseline.cluster_protein_expression(
                feature_matrix, protein_names, spatial_coords,
                graph_method='knn', clustering_method=algorithm
            )
            
            return {
                'cluster_labels': results['cluster_labels'],
                'clustering_info': results['clustering_info'],
                'graph_metrics': results['graph_metrics'],
                'method': method
            }
        
        elif method == 'spatial_leiden_coabundance':
            # Spatial clustering with co-abundance features
            enriched_features, enriched_names = generate_coabundance_features(
                feature_matrix, protein_names, spatial_coords
            )
            
            cluster_labels, clustering_info = perform_spatial_clustering(
                enriched_features, spatial_coords,
                method='leiden', resolution=1.0, spatial_weight=0.3
            )
            
            return {
                'cluster_labels': cluster_labels,
                'clustering_info': clustering_info,
                'method': 'spatial_leiden_coabundance',
                'feature_enrichment': {
                    'original_features': len(protein_names),
                    'enriched_features': len(enriched_names)
                }
            }
        
        elif method == 'graph_leiden_coabundance':
            # Graph clustering with co-abundance features
            enriched_features, enriched_names = generate_coabundance_features(
                feature_matrix, protein_names, spatial_coords
            )
            
            baseline = GraphClusteringBaseline(self.random_state)
            results = baseline.cluster_protein_expression(
                enriched_features, enriched_names, spatial_coords,
                graph_method='knn', clustering_method='leiden'
            )
            
            return {
                'cluster_labels': results['cluster_labels'],
                'clustering_info': results['clustering_info'],
                'graph_metrics': results['graph_metrics'],
                'method': 'graph_leiden_coabundance',
                'feature_enrichment': {
                    'original_features': len(protein_names),
                    'enriched_features': len(enriched_names)
                }
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _compute_evaluation_metrics(
        self,
        feature_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        spatial_coords: Optional[np.ndarray],
        metric_names: List[str],
        method_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute evaluation metrics for clustering results."""
        metrics = {}
        
        # Basic cluster statistics
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        n_clusters = len(unique_labels)
        n_noise = np.sum(cluster_labels == -1) if -1 in cluster_labels else 0
        
        metrics['n_clusters'] = n_clusters
        metrics['n_noise'] = n_noise
        metrics['noise_ratio'] = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
        
        # Silhouette score
        if 'silhouette_score' in metric_names and n_clusters > 1:
            try:
                non_noise_mask = cluster_labels >= 0
                if np.sum(non_noise_mask) > 1:
                    silhouette = silhouette_score(
                        feature_matrix[non_noise_mask],
                        cluster_labels[non_noise_mask]
                    )
                    metrics['silhouette_score'] = silhouette
            except Exception:
                metrics['silhouette_score'] = np.nan
        
        # Spatial coherence
        if 'spatial_coherence' in metric_names and spatial_coords is not None:
            try:
                coherence = compute_spatial_coherence(cluster_labels, spatial_coords)
                metrics['spatial_coherence'] = coherence
            except Exception:
                metrics['spatial_coherence'] = np.nan
        
        # Modularity (from graph metrics if available)
        if 'modularity' in metric_names and 'graph_metrics' in method_results:
            modularity = method_results['graph_metrics'].get('modularity', np.nan)
            metrics['modularity'] = modularity
        
        # Cluster stability (using bootstrap resampling)
        if 'stability' in metric_names:
            try:
                stability = self._compute_cluster_stability(
                    feature_matrix, cluster_labels, spatial_coords
                )
                metrics['stability'] = stability
            except Exception:
                metrics['stability'] = np.nan
        
        # Cluster separability
        if 'cluster_separability' in metric_names and n_clusters > 1:
            try:
                separability = self._compute_cluster_separability(
                    feature_matrix, cluster_labels
                )
                metrics['separability'] = separability
            except Exception:
                metrics['separability'] = np.nan
        
        return metrics
    
    def _compute_cluster_stability(
        self,
        feature_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        spatial_coords: Optional[np.ndarray],
        n_bootstrap: int = 10,
        subsample_ratio: float = 0.8
    ) -> float:
        """Compute clustering stability using bootstrap resampling."""
        n_samples = len(cluster_labels)
        subsample_size = int(n_samples * subsample_ratio)
        
        stability_scores = []
        
        for _ in range(n_bootstrap):
            # Subsample data
            indices = np.random.choice(n_samples, subsample_size, replace=False)
            sub_features = feature_matrix[indices]
            sub_coords = spatial_coords[indices] if spatial_coords is not None else None
            
            # Re-cluster subsample
            try:
                sub_labels, _ = perform_spatial_clustering(
                    sub_features, sub_coords, method='leiden', resolution=1.0
                )
                
                # Compare with original clustering on same subset
                original_subset = cluster_labels[indices]
                
                # Only compare valid (non-noise) points
                valid_mask = (sub_labels >= 0) & (original_subset >= 0)
                if np.sum(valid_mask) > 1:
                    ari = adjusted_rand_score(
                        original_subset[valid_mask],
                        sub_labels[valid_mask]
                    )
                    stability_scores.append(ari)
                    
            except Exception:
                continue
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def _compute_cluster_separability(
        self,
        feature_matrix: np.ndarray,
        cluster_labels: np.ndarray
    ) -> float:
        """Compute average pairwise cluster separability."""
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        
        if len(unique_labels) < 2:
            return 0.0
        
        # Compute centroids
        centroids = []
        for label in unique_labels:
            mask = cluster_labels == label
            centroid = np.mean(feature_matrix[mask], axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Compute pairwise distances between centroids
        from scipy.spatial.distance import pdist
        centroid_distances = pdist(centroids, metric='euclidean')
        
        # Compute within-cluster variance
        within_cluster_vars = []
        for label in unique_labels:
            mask = cluster_labels == label
            cluster_data = feature_matrix[mask]
            if len(cluster_data) > 1:
                cluster_var = np.mean(np.var(cluster_data, axis=0))
                within_cluster_vars.append(cluster_var)
        
        mean_within_var = np.mean(within_cluster_vars) if within_cluster_vars else 1.0
        mean_between_dist = np.mean(centroid_distances)
        
        # Separability = between-cluster distance / within-cluster variance
        separability = mean_between_dist / (mean_within_var + 1e-10)
        
        return separability
    
    def _compute_pairwise_comparisons(
        self,
        method_results: Dict[str, Dict]
    ) -> Dict[str, Dict[str, float]]:
        """Compute pairwise comparisons between clustering methods."""
        methods = list(method_results.keys())
        comparisons = {}
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                
                if (method_results[method1]['clustering_results'] is None or
                    method_results[method2]['clustering_results'] is None):
                    continue
                
                labels1 = method_results[method1]['clustering_results']['cluster_labels']
                labels2 = method_results[method2]['clustering_results']['cluster_labels']
                
                if len(labels1) != len(labels2):
                    continue
                
                # Agreement metrics
                try:
                    # Only compare non-noise points
                    valid_mask = (labels1 >= 0) & (labels2 >= 0)
                    if np.sum(valid_mask) > 1:
                        ari = adjusted_rand_score(labels1[valid_mask], labels2[valid_mask])
                        nmi = normalized_mutual_info_score(labels1[valid_mask], labels2[valid_mask])
                        
                        pair_key = f"{method1}_vs_{method2}"
                        comparisons[pair_key] = {
                            'adjusted_rand_index': ari,
                            'normalized_mutual_info': nmi,
                            'n_compared_samples': np.sum(valid_mask),
                            'cluster_count_diff': abs(
                                len(np.unique(labels1[labels1 >= 0])) - 
                                len(np.unique(labels2[labels2 >= 0]))
                            )
                        }
                except Exception:
                    continue
        
        return comparisons
    
    def _rank_methods(
        self,
        method_results: Dict[str, Dict],
        evaluation_metrics: List[str]
    ) -> Dict[str, Any]:
        """Rank clustering methods by performance."""
        
        # Extract metrics for all methods
        method_scores = {}
        for method, results in method_results.items():
            if results['clustering_results'] is None:
                continue
                
            metrics = results['evaluation_metrics']
            method_scores[method] = metrics
        
        if not method_scores:
            return {'rankings': {}, 'composite_scores': {}}
        
        # Rank by individual metrics
        rankings = {}
        for metric in evaluation_metrics:
            metric_values = []
            methods = []
            
            for method, scores in method_scores.items():
                if metric in scores and not np.isnan(scores[metric]):
                    metric_values.append(scores[metric])
                    methods.append(method)
            
            if metric_values:
                # Higher is better for most metrics
                sorted_indices = np.argsort(metric_values)[::-1]
                rankings[metric] = [methods[i] for i in sorted_indices]
        
        # Compute composite ranking (average rank across metrics)
        composite_scores = {}
        for method in method_scores.keys():
            ranks = []
            for metric, ranking in rankings.items():
                if method in ranking:
                    rank = ranking.index(method) + 1  # 1-indexed rank
                    ranks.append(rank)
            
            if ranks:
                composite_scores[method] = np.mean(ranks)
        
        # Sort by composite score (lower is better)
        sorted_methods = sorted(composite_scores.keys(), key=lambda x: composite_scores[x])
        
        return {
            'rankings': rankings,
            'composite_scores': composite_scores,
            'overall_ranking': sorted_methods
        }
    
    def _create_comparison_summary(
        self,
        method_results: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Create summary of comparison results."""
        
        successful_methods = [
            method for method, results in method_results.items()
            if results['clustering_results'] is not None
        ]
        
        failed_methods = [
            method for method, results in method_results.items()
            if results['clustering_results'] is None
        ]
        
        # Extract key metrics
        cluster_counts = {}
        silhouette_scores = {}
        spatial_coherence_scores = {}
        
        for method in successful_methods:
            metrics = method_results[method]['evaluation_metrics']
            cluster_counts[method] = metrics.get('n_clusters', 0)
            silhouette_scores[method] = metrics.get('silhouette_score', np.nan)
            spatial_coherence_scores[method] = metrics.get('spatial_coherence', np.nan)
        
        return {
            'n_methods_tested': len(method_results),
            'n_successful_methods': len(successful_methods),
            'n_failed_methods': len(failed_methods),
            'successful_methods': successful_methods,
            'failed_methods': failed_methods,
            'cluster_count_range': (
                min(cluster_counts.values()) if cluster_counts else 0,
                max(cluster_counts.values()) if cluster_counts else 0
            ),
            'silhouette_score_range': (
                np.nanmin(list(silhouette_scores.values())) if silhouette_scores else np.nan,
                np.nanmax(list(silhouette_scores.values())) if silhouette_scores else np.nan
            ),
            'spatial_coherence_range': (
                np.nanmin(list(spatial_coherence_scores.values())) if spatial_coherence_scores else np.nan,
                np.nanmax(list(spatial_coherence_scores.values())) if spatial_coherence_scores else np.nan
            )
        }
    
    def evaluate_clustering_robustness(
        self,
        feature_matrix: np.ndarray,
        protein_names: List[str],
        spatial_coords: Optional[np.ndarray],
        method: str = 'spatial_leiden',
        perturbation_types: Optional[List[str]] = None,
        n_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate clustering robustness to various perturbations.
        
        Args:
            feature_matrix: N x P feature matrix
            protein_names: List of protein names
            spatial_coords: Optional spatial coordinates
            method: Clustering method to test
            perturbation_types: Types of perturbations to apply
            n_trials: Number of trials per perturbation
            
        Returns:
            Robustness evaluation results
        """
        if perturbation_types is None:
            perturbation_types = ['noise', 'subsampling', 'feature_dropout']
        
        # Get baseline clustering
        baseline_results = self._run_clustering_method(
            method, feature_matrix, protein_names, spatial_coords
        )
        baseline_labels = baseline_results['cluster_labels']
        
        robustness_results = {}
        
        for perturbation in perturbation_types:
            perturbation_scores = []
            
            for trial in range(n_trials):
                try:
                    # Apply perturbation
                    perturbed_features, perturbed_coords = self._apply_perturbation(
                        feature_matrix, spatial_coords, perturbation, trial
                    )
                    
                    # Re-cluster perturbed data
                    perturbed_results = self._run_clustering_method(
                        method, perturbed_features, protein_names, perturbed_coords
                    )
                    perturbed_labels = perturbed_results['cluster_labels']
                    
                    # Compute agreement with baseline
                    if len(baseline_labels) == len(perturbed_labels):
                        valid_mask = (baseline_labels >= 0) & (perturbed_labels >= 0)
                        if np.sum(valid_mask) > 1:
                            ari = adjusted_rand_score(
                                baseline_labels[valid_mask],
                                perturbed_labels[valid_mask]
                            )
                            perturbation_scores.append(ari)
                
                except Exception:
                    continue
            
            robustness_results[perturbation] = {
                'scores': perturbation_scores,
                'mean_robustness': np.mean(perturbation_scores) if perturbation_scores else 0.0,
                'std_robustness': np.std(perturbation_scores) if perturbation_scores else 0.0,
                'n_successful_trials': len(perturbation_scores)
            }
        
        return {
            'baseline_results': baseline_results,
            'perturbation_results': robustness_results,
            'overall_robustness': np.mean([
                results['mean_robustness'] 
                for results in robustness_results.values()
            ]) if robustness_results else 0.0
        }
    
    def _apply_perturbation(
        self,
        feature_matrix: np.ndarray,
        spatial_coords: Optional[np.ndarray],
        perturbation_type: str,
        trial: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply specified perturbation to data."""
        
        np.random.seed(self.random_state + trial)
        
        if perturbation_type == 'noise':
            # Add Gaussian noise
            noise_std = 0.1 * np.std(feature_matrix)
            noise = np.random.normal(0, noise_std, feature_matrix.shape)
            return feature_matrix + noise, spatial_coords
        
        elif perturbation_type == 'subsampling':
            # Subsample data points
            n_samples = feature_matrix.shape[0]
            subsample_size = int(0.8 * n_samples)
            indices = np.random.choice(n_samples, subsample_size, replace=False)
            
            subsampled_features = feature_matrix[indices]
            subsampled_coords = spatial_coords[indices] if spatial_coords is not None else None
            
            return subsampled_features, subsampled_coords
        
        elif perturbation_type == 'feature_dropout':
            # Randomly zero out some features
            perturbed_features = feature_matrix.copy()
            n_features = feature_matrix.shape[1]
            dropout_rate = 0.1
            n_dropout = int(dropout_rate * n_features)
            
            if n_dropout > 0:
                dropout_features = np.random.choice(n_features, n_dropout, replace=False)
                perturbed_features[:, dropout_features] = 0
            
            return perturbed_features, spatial_coords
        
        else:
            return feature_matrix, spatial_coords


def compare_graph_vs_spatial_clustering(
    feature_matrix: np.ndarray,
    protein_names: List[str],
    spatial_coords: np.ndarray,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Direct comparison between graph-based and spatial clustering approaches.
    
    Args:
        feature_matrix: N x P feature matrix
        protein_names: List of protein names
        spatial_coords: N x 2 spatial coordinates
        config: Optional configuration
        
    Returns:
        Detailed comparison results
    """
    evaluator = ClusteringEvaluator()
    
    # Run both clustering approaches
    methods = ['spatial_leiden', 'graph_leiden', 'graph_louvain']
    
    comparison_results = evaluator.compare_clustering_methods(
        feature_matrix, protein_names, spatial_coords, methods
    )
    
    # Add specific graph vs spatial analysis
    if ('spatial_leiden' in comparison_results['method_results'] and 
        'graph_leiden' in comparison_results['method_results']):
        
        spatial_results = comparison_results['method_results']['spatial_leiden']
        graph_results = comparison_results['method_results']['graph_leiden']
        
        if (spatial_results['clustering_results'] is not None and 
            graph_results['clustering_results'] is not None):
            
            # Direct comparison
            spatial_labels = spatial_results['clustering_results']['cluster_labels']
            graph_labels = graph_results['clustering_results']['cluster_labels']
            
            comparison_results['graph_vs_spatial'] = {
                'agreement': comparison_results['pairwise_comparisons'].get(
                    'spatial_leiden_vs_graph_leiden', {}
                ),
                'spatial_advantages': {
                    'spatial_coherence': spatial_results['evaluation_metrics'].get('spatial_coherence', 0),
                    'accounts_for_proximity': True
                },
                'graph_advantages': {
                    'modularity': graph_results['evaluation_metrics'].get('modularity', 0),
                    'protein_similarity_focused': True,
                    'scalability': 'Better for large datasets'
                },
                'recommendations': _generate_method_recommendations(
                    spatial_results, graph_results
                )
            }
    
    return comparison_results


def _generate_method_recommendations(
    spatial_results: Dict,
    graph_results: Dict
) -> List[str]:
    """Generate recommendations based on comparison results."""
    recommendations = []
    
    spatial_coherence = spatial_results['evaluation_metrics'].get('spatial_coherence', 0)
    graph_coherence = graph_results['evaluation_metrics'].get('spatial_coherence', 0)
    
    spatial_silhouette = spatial_results['evaluation_metrics'].get('silhouette_score', 0)
    graph_silhouette = graph_results['evaluation_metrics'].get('silhouette_score', 0)
    
    if spatial_coherence > graph_coherence + 0.1:
        recommendations.append("Spatial clustering better preserves tissue organization")
    
    if graph_silhouette > spatial_silhouette + 0.05:
        recommendations.append("Graph clustering achieves better protein expression separation")
    
    if abs(spatial_coherence - graph_coherence) < 0.05:
        recommendations.append("Both methods show similar spatial coherence - consider other factors")
    
    if len(recommendations) == 0:
        recommendations.append("Methods show comparable performance - use domain expertise to choose")
    
    return recommendations