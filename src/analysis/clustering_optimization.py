"""
Clustering Parameter Optimization

Replaces arbitrary n_clusters selection with systematic, data-driven approaches.
Addresses Gemini's critique about cherry-picking clustering parameters.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings


def elbow_method(
    feature_matrix: np.ndarray,
    k_range: Tuple[int, int] = (2, 15),
    random_state: int = 42
) -> Tuple[List[int], List[float], int]:
    """
    Use elbow method to find optimal number of clusters.
    
    Args:
        feature_matrix: N x P feature matrix
        k_range: Range of k values to test (min, max)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (k_values, inertias, optimal_k)
    """
    if feature_matrix.size == 0:
        return [], [], 2
    
    k_values = list(range(k_range[0], k_range[1] + 1))
    inertias = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(feature_matrix)
        inertias.append(kmeans.inertia_)
    
    # Find elbow using second derivative
    if len(inertias) >= 3:
        # Calculate second differences
        second_diffs = []
        for i in range(1, len(inertias) - 1):
            diff2 = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_diffs.append(diff2)
        
        # Find maximum second difference (elbow point)
        elbow_idx = np.argmax(second_diffs) + 1  # Offset for indexing
        optimal_k = k_values[elbow_idx]
    else:
        # Fallback if too few points
        optimal_k = k_values[len(k_values)//2]
    
    return k_values, inertias, optimal_k


def silhouette_analysis(
    feature_matrix: np.ndarray,
    k_range: Tuple[int, int] = (2, 15),
    random_state: int = 42
) -> Tuple[List[int], List[float], int]:
    """
    Use silhouette analysis to find optimal number of clusters.
    
    Args:
        feature_matrix: N x P feature matrix
        k_range: Range of k values to test
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (k_values, silhouette_scores, optimal_k)
    """
    if feature_matrix.size == 0:
        return [], [], 2
    
    k_values = list(range(k_range[0], k_range[1] + 1))
    silhouette_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        
        # Calculate silhouette score
        if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters
            score = silhouette_score(feature_matrix, cluster_labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1.0)  # Invalid clustering
    
    # Find k with highest silhouette score
    optimal_idx = np.argmax(silhouette_scores)
    optimal_k = k_values[optimal_idx]
    
    return k_values, silhouette_scores, optimal_k


def gap_statistic(
    feature_matrix: np.ndarray,
    k_range: Tuple[int, int] = (2, 15),
    n_references: int = 10,
    random_state: int = 42
) -> Tuple[List[int], List[float], int]:
    """
    Use gap statistic to find optimal number of clusters.
    
    Args:
        feature_matrix: N x P feature matrix
        k_range: Range of k values to test
        n_references: Number of reference datasets to generate
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (k_values, gap_values, optimal_k)
    """
    if feature_matrix.size == 0:
        return [], [], 2
    
    np.random.seed(random_state)
    
    k_values = list(range(k_range[0], k_range[1] + 1))
    gap_values = []
    
    # Get data range for reference generation
    mins = feature_matrix.min(axis=0)
    maxs = feature_matrix.max(axis=0)
    
    for k in k_values:
        # Compute inertia for actual data
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(feature_matrix)
        actual_inertia = np.log(kmeans.inertia_)
        
        # Generate reference datasets and compute average inertia
        reference_inertias = []
        for _ in range(n_references):
            # Generate reference data with same bounds
            reference_data = np.random.uniform(
                low=mins, high=maxs, 
                size=feature_matrix.shape
            )
            
            ref_kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            ref_kmeans.fit(reference_data)
            reference_inertias.append(np.log(ref_kmeans.inertia_))
        
        # Calculate gap statistic
        gap = np.mean(reference_inertias) - actual_inertia
        gap_values.append(gap)
    
    # Find first local maximum in gap statistic
    optimal_k = k_values[0]  # Default
    for i in range(1, len(gap_values)):
        if gap_values[i] > gap_values[i-1]:
            optimal_k = k_values[i]
        else:
            break  # First decrease found
    
    return k_values, gap_values, optimal_k


def multiple_validation_metrics(
    feature_matrix: np.ndarray,
    k_range: Tuple[int, int] = (2, 15),
    random_state: int = 42
) -> Dict[str, Tuple[List[int], List[float], int]]:
    """
    Evaluate clustering quality using multiple validation metrics.
    
    Args:
        feature_matrix: N x P feature matrix
        k_range: Range of k values to test
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with results for each metric
    """
    if feature_matrix.size == 0:
        return {}
    
    results = {}
    
    # Elbow method
    results['elbow'] = elbow_method(feature_matrix, k_range, random_state)
    
    # Silhouette analysis
    results['silhouette'] = silhouette_analysis(feature_matrix, k_range, random_state)
    
    # Gap statistic
    results['gap'] = gap_statistic(feature_matrix, k_range, random_state)
    
    # Additional validation metrics
    k_values = list(range(k_range[0], k_range[1] + 1))
    calinski_scores = []
    davies_bouldin_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)
        
        if len(np.unique(labels)) > 1:
            # Calinski-Harabasz Index (higher is better)
            ch_score = calinski_harabasz_score(feature_matrix, labels)
            calinski_scores.append(ch_score)
            
            # Davies-Bouldin Index (lower is better)
            db_score = davies_bouldin_score(feature_matrix, labels)
            davies_bouldin_scores.append(db_score)
        else:
            calinski_scores.append(0.0)
            davies_bouldin_scores.append(float('inf'))
    
    # Find optimal k for each metric
    calinski_optimal = k_values[np.argmax(calinski_scores)]
    davies_bouldin_optimal = k_values[np.argmin(davies_bouldin_scores)]
    
    results['calinski_harabasz'] = (k_values, calinski_scores, calinski_optimal)
    results['davies_bouldin'] = (k_values, davies_bouldin_scores, davies_bouldin_optimal)
    
    return results


def consensus_optimal_clusters(
    validation_results: Dict[str, Tuple[List[int], List[float], int]],
    weights: Optional[Dict[str, float]] = None
) -> int:
    """
    Determine consensus optimal number of clusters from multiple metrics.
    
    Args:
        validation_results: Results from multiple_validation_metrics
        weights: Optional weights for each metric
        
    Returns:
        Consensus optimal number of clusters
    """
    if not validation_results:
        return 8  # Fallback default
    
    if weights is None:
        # Default weights - prioritize silhouette and gap statistic
        weights = {
            'elbow': 0.15,
            'silhouette': 0.30,
            'gap': 0.30,
            'calinski_harabasz': 0.15,
            'davies_bouldin': 0.10
        }
    
    # Get optimal k from each method
    optimal_ks = []
    method_weights = []
    
    for method, (k_values, scores, optimal_k) in validation_results.items():
        if method in weights:
            optimal_ks.append(optimal_k)
            method_weights.append(weights[method])
    
    if not optimal_ks:
        return 8  # Fallback
    
    # Weighted average, rounded to nearest integer
    weighted_avg = np.average(optimal_ks, weights=method_weights)
    consensus_k = int(np.round(weighted_avg))
    
    # Ensure reasonable bounds
    consensus_k = max(2, min(consensus_k, 15))
    
    return consensus_k


def biological_validation_score(
    feature_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    protein_names: List[str],
    known_coexpression_pairs: Optional[List[Tuple[str, str]]] = None
) -> float:
    """
    Validate clustering based on known biological relationships.
    
    Args:
        feature_matrix: N x P feature matrix
        cluster_labels: Cluster assignments
        protein_names: Names of proteins (columns in feature matrix)
        known_coexpression_pairs: List of protein pairs expected to co-express
        
    Returns:
        Biological validation score (0-1, higher is better)
    """
    if known_coexpression_pairs is None:
        # Default pairs for IMC panel
        known_coexpression_pairs = [
            ('CD31', 'CD34'),      # Endothelial markers
            ('CD45', 'CD11b'),     # Immune markers
            ('CD206', 'CD44'),     # M2 macrophage/matrix
            ('CD140a', 'CD140b')   # Pericyte/mesenchymal
        ]
    
    if feature_matrix.size == 0 or len(np.unique(cluster_labels)) <= 1:
        return 0.0
    
    # Create protein name to index mapping
    protein_to_idx = {name: i for i, name in enumerate(protein_names)}
    
    validation_scores = []
    
    for protein1, protein2 in known_coexpression_pairs:
        if protein1 not in protein_to_idx or protein2 not in protein_to_idx:
            continue
        
        idx1 = protein_to_idx[protein1]
        idx2 = protein_to_idx[protein2]
        
        # Calculate within-cluster correlation for this pair
        cluster_correlations = []
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) < 3:  # Need at least 3 points
                continue
            
            cluster_feature1 = feature_matrix[cluster_mask, idx1]
            cluster_feature2 = feature_matrix[cluster_mask, idx2]
            
            # Calculate correlation within this cluster
            if np.std(cluster_feature1) > 0 and np.std(cluster_feature2) > 0:
                corr = np.corrcoef(cluster_feature1, cluster_feature2)[0, 1]
                if not np.isnan(corr):
                    cluster_correlations.append(abs(corr))  # Absolute correlation
        
        if cluster_correlations:
            # Average correlation across clusters
            avg_correlation = np.mean(cluster_correlations)
            validation_scores.append(avg_correlation)
    
    if not validation_scores:
        return 0.0
    
    # Return average biological validation score
    return np.mean(validation_scores)


def optimize_clustering_parameters(
    feature_matrix: np.ndarray,
    protein_names: List[str],
    k_range: Tuple[int, int] = (2, 15),
    validation_weights: Optional[Dict[str, float]] = None,
    use_biological_validation: bool = True,
    random_state: int = 42
) -> Dict:
    """
    Comprehensively optimize clustering parameters using multiple approaches.
    
    CRITICAL FIX: Replaces arbitrary n_clusters=8 with systematic selection.
    
    Args:
        feature_matrix: N x P feature matrix
        protein_names: Names of proteins (columns)
        k_range: Range of k values to test
        validation_weights: Weights for different validation metrics
        use_biological_validation: Whether to include biological validation
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with optimization results
    """
    if feature_matrix.size == 0:
        return {
            'optimal_k': 8,
            'validation_results': {},
            'biological_scores': {},
            'recommendation': 'fallback_default'
        }
    
    # Run multiple validation metrics
    validation_results = multiple_validation_metrics(
        feature_matrix, k_range, random_state
    )
    
    # Get consensus optimal k
    consensus_k = consensus_optimal_clusters(validation_results, validation_weights)
    
    # Biological validation for different k values
    biological_scores = {}
    if use_biological_validation:
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(feature_matrix)
            
            bio_score = biological_validation_score(
                feature_matrix, labels, protein_names
            )
            biological_scores[k] = bio_score
    
    # Final recommendation considering all factors
    if biological_scores:
        # Weight consensus k with biological validation
        best_bio_k = max(biological_scores, key=biological_scores.get)
        best_bio_score = biological_scores[best_bio_k]
        
        # If biological validation strongly favors a different k, consider it
        if best_bio_score > 0.5 and abs(best_bio_k - consensus_k) <= 2:
            optimal_k = best_bio_k
            recommendation = 'biological_validation_adjusted'
        else:
            optimal_k = consensus_k
            recommendation = 'consensus_statistical_metrics'
    else:
        optimal_k = consensus_k
        recommendation = 'consensus_statistical_metrics'
    
    return {
        'optimal_k': optimal_k,
        'consensus_k': consensus_k,
        'validation_results': validation_results,
        'biological_scores': biological_scores,
        'recommendation': recommendation,
        'k_range_tested': k_range
    }