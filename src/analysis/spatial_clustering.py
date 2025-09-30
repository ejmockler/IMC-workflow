"""
Spatial-Aware Clustering Module

Decoupled from spatial aggregation (SLIC/binning), this module provides
graph-based and density-based clustering that preserves spatial relationships.
Implements the Strategy pattern for different clustering approaches.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import warnings

try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    warnings.warn("Leiden algorithm not available. Install with: pip install leidenalg igraph")

try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN not available. Install with: pip install hdbscan")


def perform_spatial_clustering(
    feature_matrix: np.ndarray,
    spatial_coords: np.ndarray,
    method: str = 'leiden',
    resolution: float = 1.0,
    min_cluster_size: int = 10,
    spatial_weight: float = 0.3,
    k_neighbors: int = 15,
    random_state: int = 42,
    use_coabundance: bool = False,
    protein_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Perform spatial-aware clustering on features with coordinates.
    
    Args:
        feature_matrix: N x P feature matrix (e.g., protein expressions)
        spatial_coords: N x 2 spatial coordinates
        method: 'leiden', 'hdbscan', or 'kmeans' (fallback)
        resolution: Resolution parameter for Leiden (higher = more clusters)
        min_cluster_size: Minimum cluster size for HDBSCAN
        spatial_weight: Weight for spatial coordinates (0=ignore space, 1=only space)
        k_neighbors: Number of neighbors for graph construction
        random_state: Random seed for reproducibility
        use_coabundance: Whether to generate co-abundance features
        protein_names: Protein names (required if use_coabundance=True)
        
    Returns:
        Tuple of (cluster_labels, clustering_info)
    """
    if feature_matrix.shape[0] == 0:
        return np.array([]), {'method': method, 'n_clusters': 0}
    
    # Generate co-abundance features if requested
    if use_coabundance and protein_names is not None:
        from .coabundance_features import generate_coabundance_features
        enriched_features, enriched_names = generate_coabundance_features(
            feature_matrix, 
            protein_names,
            spatial_coords=spatial_coords,
            interaction_order=2,
            include_spatial_covariance=True
        )
        feature_matrix = enriched_features
    
    # Combine features with spatial information
    if spatial_weight > 0 and spatial_coords is not None:
        # Standardize both features and coordinates
        scaler_features = StandardScaler()
        scaler_coords = StandardScaler()
        
        features_scaled = scaler_features.fit_transform(feature_matrix)
        coords_scaled = scaler_coords.fit_transform(spatial_coords)
        
        # Weighted combination
        combined_features = np.hstack([
            features_scaled * (1 - spatial_weight),
            coords_scaled * spatial_weight
        ])
    else:
        combined_features = StandardScaler().fit_transform(feature_matrix)
    
    clustering_info = {'method': method}
    
    if method == 'leiden' and LEIDEN_AVAILABLE:
        labels = _leiden_clustering(
            combined_features, resolution, k_neighbors, random_state
        )
        clustering_info['resolution'] = resolution
        clustering_info['k_neighbors'] = k_neighbors
        
    elif method == 'hdbscan' and HDBSCAN_AVAILABLE:
        labels = _hdbscan_clustering(
            combined_features, min_cluster_size
        )
        clustering_info['min_cluster_size'] = min_cluster_size
        
    else:
        # No fallback - require proper dependencies
        available_methods = []
        if LEIDEN_AVAILABLE:
            available_methods.append('leiden')
        if HDBSCAN_AVAILABLE:
            available_methods.append('hdbscan')
        
        raise ValueError(
            f"Method '{method}' is not available. "
            f"Available methods: {available_methods}. "
            f"Install missing dependencies: pip install leidenalg igraph hdbscan"
        )
        
    clustering_info['n_clusters'] = len(np.unique(labels[labels >= 0]))
    clustering_info['n_noise'] = np.sum(labels == -1) if -1 in labels else 0
    
    return labels, clustering_info


def _leiden_clustering(
    features: np.ndarray,
    resolution: float,
    k_neighbors: int,
    random_state: int
) -> np.ndarray:
    """
    Perform Leiden community detection on kNN graph.
    """
    # Build kNN graph
    n_samples = features.shape[0]
    
    # Handle edge case: single sample can't have neighbors
    if n_samples == 1:
        return np.array([0])
    
    k_neighbors = min(k_neighbors, n_samples - 1)
    
    knn_graph = kneighbors_graph(
        features, n_neighbors=k_neighbors, mode='distance', include_self=False
    )
    
    # Convert to igraph
    sources, targets = knn_graph.nonzero()
    weights = knn_graph.data
    edges = list(zip(sources, targets))
    
    g = ig.Graph(edges=edges, directed=False)
    g.es['weight'] = weights
    
    # Run Leiden algorithm
    partition = leidenalg.find_partition(
        g, 
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_state
    )
    
    return np.array(partition.membership)


def _hdbscan_clustering(
    features: np.ndarray,
    min_cluster_size: int
) -> np.ndarray:
    """
    Perform HDBSCAN density-based clustering.
    """
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(1, min_cluster_size // 2),
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    return clusterer.fit_predict(features)


def stability_analysis(
    feature_matrix: np.ndarray,
    spatial_coords: np.ndarray,
    method: str = 'leiden',
    resolution_range: Tuple[float, float] = (0.5, 2.0),
    n_resolutions: int = 20,
    n_bootstrap: int = 10,
    subsample_ratio: float = 0.9,
    random_state: int = 42
) -> Dict:
    """
    Analyze clustering stability across resolution parameters.
    
    Args:
        feature_matrix: N x P feature matrix
        spatial_coords: N x 2 spatial coordinates
        method: Clustering method to use
        resolution_range: Range of resolutions to test
        n_resolutions: Number of resolutions to test
        n_bootstrap: Number of bootstrap iterations
        subsample_ratio: Fraction of data to subsample
        random_state: Random seed
        
    Returns:
        Dictionary with stability analysis results
    """
    np.random.seed(random_state)
    n_samples = feature_matrix.shape[0]
    subsample_size = int(n_samples * subsample_ratio)
    
    resolutions = np.linspace(resolution_range[0], resolution_range[1], n_resolutions)
    
    stability_scores = []
    mean_n_clusters = []
    
    for resolution in resolutions:
        bootstrap_labels = []
        
        for b in range(n_bootstrap):
            # Subsample data
            indices = np.random.choice(n_samples, subsample_size, replace=False)
            features_sub = feature_matrix[indices]
            coords_sub = spatial_coords[indices] if spatial_coords is not None else None
            
            # Cluster subsample
            labels, _ = perform_spatial_clustering(
                features_sub, coords_sub, method=method, resolution=resolution
            )
            bootstrap_labels.append(labels)
        
        # Calculate stability (average pairwise ARI between bootstrap runs)
        from sklearn.metrics import adjusted_rand_score
        ari_scores = []
        for i in range(n_bootstrap):
            for j in range(i + 1, n_bootstrap):
                if len(bootstrap_labels[i]) == len(bootstrap_labels[j]):
                    ari = adjusted_rand_score(bootstrap_labels[i], bootstrap_labels[j])
                    ari_scores.append(ari)
        
        stability = np.mean(ari_scores) if ari_scores else 0.0
        stability_scores.append(stability)
        
        # Track mean number of clusters
        n_clusters_list = [len(np.unique(labels[labels >= 0])) for labels in bootstrap_labels]
        mean_n_clusters.append(np.mean(n_clusters_list))
    
    # Find stable plateaus
    stable_resolutions = _find_stable_plateaus(stability_scores, resolutions)
    
    return {
        'resolutions': resolutions.tolist(),
        'stability_scores': stability_scores,
        'mean_n_clusters': mean_n_clusters,
        'stable_resolutions': stable_resolutions,
        'optimal_resolution': stable_resolutions[0] if stable_resolutions else resolutions[np.argmax(stability_scores)]
    }


def _find_stable_plateaus(
    stability_scores: List[float],
    resolutions: np.ndarray,
    min_stability: float = 0.6,
    plateau_threshold: float = 0.05
) -> List[float]:
    """
    Find resolutions where stability plateaus.
    """
    stable_resolutions = []
    
    for i in range(1, len(stability_scores) - 1):
        if stability_scores[i] >= min_stability:
            # Check if this is a plateau (small change in stability)
            left_diff = abs(stability_scores[i] - stability_scores[i-1])
            right_diff = abs(stability_scores[i] - stability_scores[i+1])
            
            if left_diff < plateau_threshold and right_diff < plateau_threshold:
                stable_resolutions.append(resolutions[i])
    
    return stable_resolutions


def select_resolution_headless(
    stability_results: Dict,
    min_stability: float = 0.6,
    prefer_higher_k: bool = False
) -> float:
    """
    Select resolution based on stability analysis without user interaction.
    
    Args:
        stability_results: Results from stability_analysis
        min_stability: Minimum acceptable stability score
        prefer_higher_k: If True, prefer higher cluster counts when stability is similar
        
    Returns:
        Selected resolution value
    """
    resolutions = stability_results['resolutions']
    stability_scores = stability_results['stability_scores']
    mean_n_clusters = stability_results['mean_n_clusters']
    
    # Find resolutions with acceptable stability
    acceptable_indices = [i for i, score in enumerate(stability_scores) 
                         if score >= min_stability]
    
    if not acceptable_indices:
        # If no resolution meets threshold, use the most stable
        best_idx = np.argmax(stability_scores)
        warnings.warn(f"No resolution achieved stability >= {min_stability}. "
                     f"Using most stable: {resolutions[best_idx]}")
        return resolutions[best_idx]
    
    # Among acceptable resolutions, find plateaus
    plateau_scores = []
    for idx in acceptable_indices:
        # Check local stability (small change in neighboring points)
        left_stable = (idx == 0 or 
                      abs(stability_scores[idx] - stability_scores[idx-1]) < 0.05)
        right_stable = (idx == len(stability_scores)-1 or 
                       abs(stability_scores[idx] - stability_scores[idx+1]) < 0.05)
        
        if left_stable and right_stable:
            plateau_scores.append((resolutions[idx], stability_scores[idx], 
                                  mean_n_clusters[idx]))
    
    if plateau_scores:
        # Choose from plateaus
        if prefer_higher_k:
            # Sort by cluster count (descending)
            plateau_scores.sort(key=lambda x: x[2], reverse=True)
        else:
            # Sort by stability (descending)
            plateau_scores.sort(key=lambda x: x[1], reverse=True)
        
        return plateau_scores[0][0]
    else:
        # No plateaus, choose best stability among acceptable
        best_acceptable = max(acceptable_indices, 
                             key=lambda i: stability_scores[i])
        return resolutions[best_acceptable]


def compute_spatial_coherence(
    cluster_labels: np.ndarray,
    spatial_coords: np.ndarray
) -> float:
    """
    Compute Moran's I for spatial autocorrelation of clusters.
    
    Higher values indicate spatially coherent clusters.
    """
    from scipy.spatial import distance_matrix
    
    if len(np.unique(cluster_labels)) <= 1:
        return 0.0
    
    # Create spatial weights matrix (inverse distance)
    dist_matrix = distance_matrix(spatial_coords, spatial_coords)
    np.fill_diagonal(dist_matrix, np.inf)  # Avoid self-connections
    
    # Use inverse distance as weights
    weights = 1.0 / (dist_matrix + 1e-10)
    weights[np.isinf(weights)] = 0
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Calculate Moran's I for cluster assignments
    n = len(cluster_labels)
    y = cluster_labels - cluster_labels.mean()
    
    numerator = np.sum(weights * np.outer(y, y))
    denominator = np.sum(y ** 2)
    
    morans_i = (n / weights.sum()) * (numerator / denominator)
    
    return float(morans_i)