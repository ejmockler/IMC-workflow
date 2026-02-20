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
from scipy import sparse
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

from .performance_profiling import PerformanceTimer, record_timing

_MIN_RESOLUTIONS_FOR_PARALLEL = 8
_MIN_SAMPLES_FOR_PARALLEL = 500

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
    protein_names: Optional[List[str]] = None,
    coabundance_options: Optional[Dict] = None
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
        coabundance_options: Dictionary of coabundance feature options
        
    Returns:
        Tuple of (cluster_labels, clustering_info)
    """
    if feature_matrix.shape[0] == 0:
        return np.array([]), {'method': method, 'n_clusters': 0}

    # Generate co-abundance features if requested
    enriched_feature_names = None
    if use_coabundance and protein_names is not None:
        from .coabundance_features import generate_coabundance_features, select_informative_coabundance_features

        # Use config options if provided, otherwise defaults
        if coabundance_options is None:
            coabundance_options = {}

        enriched_features, enriched_feature_names = generate_coabundance_features(
            feature_matrix,
            protein_names,
            spatial_coords=spatial_coords,
            interaction_order=coabundance_options.get('interaction_order', 2),
            include_ratios=coabundance_options.get('include_ratios', True),
            include_products=coabundance_options.get('include_products', True),
            include_spatial_covariance=coabundance_options.get('include_spatial_covariance', True),
            neighborhood_radius=coabundance_options.get('neighborhood_radius', 20.0),
            min_expression_percentile=coabundance_options.get('min_expression_percentile', 25.0)
        )

        # Apply feature selection to prevent overfitting (153 → ~30 features)
        if coabundance_options.get('use_feature_selection', True):
            feature_matrix, enriched_feature_names = select_informative_coabundance_features(
                enriched_features,
                enriched_feature_names,
                target_n_features=coabundance_options.get('target_n_features', 30),
                method=coabundance_options.get('selection_method', 'lasso'),
                options=coabundance_options  # Pass full options dict for method-specific params
            )
        else:
            feature_matrix = enriched_features
    
    combined_features, _, _ = _combine_features_and_spatial(
        feature_matrix,
        spatial_coords,
        spatial_weight=spatial_weight
    )
    
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

    # Include enriched features in clustering_info if coabundance was used
    if use_coabundance and protein_names is not None:
        clustering_info['enriched_features'] = feature_matrix
        clustering_info['enriched_feature_names'] = enriched_feature_names if enriched_feature_names is not None else protein_names

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


def _combine_features_and_spatial(
    feature_matrix: np.ndarray,
    spatial_coords: Optional[np.ndarray],
    spatial_weight: float,
    feature_scaler: Optional[StandardScaler] = None,
    coord_scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler, Optional[StandardScaler]]:
    """
    Standardize features and optional spatial coordinates, returning the combined array.
    """
    if feature_scaler is None:
        feature_scaler = StandardScaler().fit(feature_matrix)
    features_scaled = feature_scaler.transform(feature_matrix)

    if spatial_weight > 0 and spatial_coords is not None:
        if coord_scaler is None:
            coord_scaler = StandardScaler().fit(spatial_coords)
        coords_scaled = coord_scaler.transform(spatial_coords)
        combined = np.hstack([
            features_scaled * (1 - spatial_weight),
            coords_scaled * spatial_weight
        ])
    else:
        combined = features_scaled

    return combined, feature_scaler, coord_scaler


def _test_single_resolution_parallel(
    resolution: float,
    feature_matrix: np.ndarray,
    spatial_coords: Optional[np.ndarray],
    base_knn_graph: Optional[sparse.csr_matrix],
    n_bootstrap: int,
    subsample_ratio: float,
    random_state: int,
    method: str
) -> Tuple[float, float, float]:
    """
    Evaluate stability for a single resolution; safe for parallel execution.
    """
    n_samples = feature_matrix.shape[0]
    if n_samples == 0:
        return resolution, 0.0, 0.0

    subsample_size = max(1, int(n_samples * subsample_ratio))
    bootstrap_labels: List[np.ndarray] = []

    for b in range(n_bootstrap):
        seed = random_state + int(resolution * 1000) + b
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_samples, size=subsample_size, replace=False)

        labels: np.ndarray
        subgraph = _subsample_graph(base_knn_graph, indices) if base_knn_graph is not None else None

        if subgraph is not None and method == 'leiden':
            labels = _leiden_from_sparse_graph(
                subgraph,
                resolution=resolution,
                random_state=seed
            )
        else:
            features_sub = feature_matrix[indices]
            coords_sub = spatial_coords[indices] if spatial_coords is not None else None
            labels, _ = perform_spatial_clustering(
                features_sub,
                coords_sub,
                method=method,
                resolution=resolution,
                random_state=seed
            )

        bootstrap_labels.append(labels)

    from sklearn.metrics import adjusted_rand_score

    ari_scores: List[float] = []
    for i in range(n_bootstrap):
        for j in range(i + 1, n_bootstrap):
            if len(bootstrap_labels[i]) == len(bootstrap_labels[j]):
                ari = adjusted_rand_score(bootstrap_labels[i], bootstrap_labels[j])
                ari_scores.append(ari)

    stability = float(np.mean(ari_scores)) if ari_scores else 0.0
    n_clusters_list = [len(np.unique(labels[labels >= 0])) for labels in bootstrap_labels]
    mean_n_clusters = float(np.mean(n_clusters_list)) if n_clusters_list else 0.0

    return resolution, stability, mean_n_clusters


def _adaptive_resolution_search(
    feature_matrix: np.ndarray,
    spatial_coords: Optional[np.ndarray],
    base_knn_graph: Optional[sparse.csr_matrix],
    method: str,
    resolution_range: Tuple[float, float],
    n_bootstrap: int,
    subsample_ratio: float,
    random_state: int,
    target_stability: float,
    tolerance: float,
    max_evaluations: int
) -> Dict:
    """
    Adaptive resolution search that incrementally refines the most uncertain intervals.
    """
    low, high = resolution_range
    low = float(low)
    high = float(high)

    if high <= low:
        raise ValueError("resolution_range must have high > low")

    # Ensure minimum evaluations for initial sampling
    max_evaluations = max(3, int(max_evaluations))

    evaluations: Dict[float, Tuple[float, float]] = {}

    def evaluate(resolution: float) -> Tuple[float, float]:
        resolution = float(np.clip(resolution, low, high))
        key = round(resolution, 6)
        if key in evaluations:
            stored_resolution, stability, clusters = evaluations[key]
            return stability, clusters

        _, stability, clusters = _test_single_resolution_parallel(
            resolution,
            feature_matrix,
            spatial_coords,
            base_knn_graph,
            n_bootstrap,
            subsample_ratio,
            random_state,
            method
        )
        evaluations[key] = (resolution, stability, clusters)
        return stability, clusters

    # Initial sampling: low, midpoint, high
    initial_points = [low, (low + high) / 2.0, high]
    for point in initial_points:
        if len(evaluations) >= max_evaluations:
            break
        evaluate(point)

    def sorted_evaluations() -> List[Tuple[float, float, float]]:
        return sorted((res, stab, clusters) for res, (res, stab, clusters) in evaluations.items())

    # Iteratively refine largest gaps
    while len(evaluations) < max_evaluations:
        sorted_eval = sorted_evaluations()
        if len(sorted_eval) < 2:
            break

        # Determine gaps between successive evaluations
        gaps = []
        for left, right in zip(sorted_eval[:-1], sorted_eval[1:]):
            gap = right[0] - left[0]
            gaps.append((gap, left, right))

        if not gaps:
            break

        # Select the largest remaining gap
        gaps.sort(key=lambda item: item[0], reverse=True)
        largest_gap, left_eval, right_eval = gaps[0]

        if largest_gap <= tolerance:
            break

        candidate = (left_eval[0] + right_eval[0]) / 2.0
        evaluate(candidate)

    results_sorted = sorted_evaluations()
    resolutions = [res for res, _, _ in results_sorted]
    stability_scores = [stab for _, stab, _ in results_sorted]
    mean_n_clusters = [clusters for _, _, clusters in results_sorted]

    best_idx = int(np.argmax(stability_scores))
    optimal_resolution = resolutions[best_idx]

    stable_resolutions = [
        res for res, stab in zip(resolutions, stability_scores) if stab >= target_stability
    ]

    return {
        'resolutions': resolutions,
        'stability_scores': stability_scores,
        'mean_n_clusters': mean_n_clusters,
        'optimal_resolution': optimal_resolution,
        'stable_resolutions': stable_resolutions,
        'n_evaluations': len(resolutions),
        'adaptive_search_used': True
    }


def stability_analysis(
    feature_matrix: np.ndarray,
    spatial_coords: np.ndarray,
    method: str = 'leiden',
    resolution_range: Tuple[float, float] = (0.5, 2.0),
    n_resolutions: int = 20,
    n_bootstrap: int = 5,
    subsample_ratio: float = 0.9,
    random_state: int = 42,
    use_graph_caching: bool = True,
    parallel: bool = True,
    n_workers: Optional[int] = None,
    adaptive_search: bool = False,
    adaptive_target_stability: float = 0.6,
    adaptive_tolerance: float = 0.05,
    adaptive_max_evaluations: Optional[int] = None,
    spatial_weight: float = 0.3
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
        use_graph_caching: Cache kNN graph once per call (Leiden only)
        parallel: Whether to evaluate resolutions in parallel (process pool)
        n_workers: Number of worker processes (None = auto)
        adaptive_search: If True, use adaptive search instead of fixed grid
        adaptive_target_stability: Target stability for adaptive convergence
        adaptive_tolerance: Resolution tolerance for adaptive refinement
        adaptive_max_evaluations: Maximum evaluations for adaptive search
        
    Returns:
        Dictionary with stability analysis results
    """
    np.random.seed(random_state)
    n_samples = feature_matrix.shape[0]
    subsample_size = int(n_samples * subsample_ratio)
    
    resolutions = np.linspace(resolution_range[0], resolution_range[1], n_resolutions)
    
    stability_scores = []
    mean_n_clusters = []
    
    base_knn_graph = None
    cached_spatial_weight = spatial_weight

    if use_graph_caching and method == 'leiden' and n_samples > 1:
        with PerformanceTimer("stability_prepare_features", log_result=False):
            combined_features, _, _ = _combine_features_and_spatial(
                feature_matrix,
                spatial_coords,
                spatial_weight=cached_spatial_weight
            )
        with PerformanceTimer("stability_build_knn_graph", log_result=False) as build_timer:
            base_knn_graph = _build_knn_graph_cached(
                combined_features,
                k_neighbors=min(15, max(1, subsample_size - 1))
            )
        if build_timer and build_timer.elapsed is not None:
            record_timing("stability_graph_build", build_timer.elapsed)

    if adaptive_search:
        max_evaluations = adaptive_max_evaluations
        if max_evaluations is None:
            max_evaluations = min(max(6, n_resolutions), 10)
        result = _adaptive_resolution_search(
            feature_matrix,
            spatial_coords,
            base_knn_graph,
            method=method,
            resolution_range=resolution_range,
            n_bootstrap=n_bootstrap,
            subsample_ratio=subsample_ratio,
            random_state=random_state,
            target_stability=adaptive_target_stability,
            tolerance=adaptive_tolerance,
            max_evaluations=max_evaluations
        )
        result['graph_caching_used'] = base_knn_graph is not None
        result['parallel_used'] = False
        result['n_workers_used'] = 0
        return result

    # PRINCIPLED SOLUTION: Use threading for stability analysis to avoid nested process issues
    # Threading is appropriate here because:
    # 1. NumPy/scikit-learn release the GIL for numerical operations
    # 2. Threads can be spawned from daemon processes (unlike processes)
    # 3. Stability analysis is CPU-bound in compiled code (NumPy/Leiden)
    # 4. Allows nested parallelism when ROI batch processing runs in parallel
    import multiprocessing

    can_force_parallel = n_workers is not None and n_workers > 1
    use_parallel = (
        parallel
        and n_resolutions >= _MIN_RESOLUTIONS_FOR_PARALLEL
        and (feature_matrix.shape[0] >= _MIN_SAMPLES_FOR_PARALLEL or can_force_parallel)
    )

    with PerformanceTimer("stability_analysis_total"):
        if use_parallel:
            # Use ThreadPoolExecutor instead of ProcessPoolExecutor
            # This allows nested parallelism and works well with NumPy/scikit-learn
            from concurrent.futures import ThreadPoolExecutor

            if n_workers is None:
                available = max(1, multiprocessing.cpu_count() - 1)
                n_workers = min(available, n_resolutions)
            else:
                n_workers = max(1, min(n_workers, n_resolutions))

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(
                        _test_single_resolution_parallel,
                        float(resolution),
                        feature_matrix,
                        spatial_coords,
                        base_knn_graph,
                        n_bootstrap,
                        subsample_ratio,
                        random_state,
                        method
                    )
                    for resolution in resolutions
                ]

                results = [future.result() for future in futures]

            results.sort(key=lambda x: x[0])
            stability_scores = [res[1] for res in results]
            mean_n_clusters = [res[2] for res in results]
        else:
            for resolution in resolutions:
                with PerformanceTimer(f"stability_resolution_{resolution:.4f}", log_result=False) as resolution_timer:
                    _, stability, mean_clusters = _test_single_resolution_parallel(
                        float(resolution),
                        feature_matrix,
                        spatial_coords,
                        base_knn_graph,
                        n_bootstrap,
                        subsample_ratio,
                        random_state,
                        method
                    )
                if resolution_timer and resolution_timer.elapsed is not None:
                    record_timing("stability_resolution_iteration", resolution_timer.elapsed)

                stability_scores.append(stability)
                mean_n_clusters.append(mean_clusters)
    
    # Find stable plateaus
    stable_resolutions = _find_stable_plateaus(stability_scores, resolutions)
    
    return {
        'resolutions': resolutions.tolist(),
        'stability_scores': stability_scores,
        'mean_n_clusters': mean_n_clusters,
        'stable_resolutions': stable_resolutions,
        'optimal_resolution': stable_resolutions[0] if stable_resolutions else resolutions[np.argmax(stability_scores)],
        'graph_caching_used': base_knn_graph is not None,
        'parallel_used': use_parallel,
        'n_workers_used': n_workers if use_parallel else 0,
        'n_evaluations': len(resolutions)
    }


def _build_knn_graph_cached(
    combined_features: np.ndarray,
    k_neighbors: int
) -> Optional[sparse.csr_matrix]:
    """
    Build a cached kNN graph to reuse across bootstrap iterations.
    """
    if combined_features.shape[0] <= 1:
        return None

    k_neighbors = min(k_neighbors, combined_features.shape[0] - 1)
    graph = kneighbors_graph(
        combined_features,
        n_neighbors=k_neighbors,
        mode='distance',
        include_self=False
    )
    # Symmetrize to keep graph undirected
    graph = 0.5 * (graph + graph.T)
    return graph.tocsr()


def _subsample_graph(
    graph: Optional[sparse.csr_matrix],
    indices: np.ndarray
) -> Optional[sparse.csr_matrix]:
    """
    Extract a subgraph for the selected indices from the cached graph.
    """
    if graph is None or indices.size <= 1:
        return None
    return graph[indices][:, indices].tocsr()


def _leiden_from_sparse_graph(
    graph: Optional[sparse.csr_matrix],
    resolution: float,
    random_state: int
) -> np.ndarray:
    """
    Run Leiden clustering directly on a sparse graph.
    """
    if graph is None or graph.shape[0] == 0:
        return np.array([])
    if graph.shape[0] == 1:
        return np.array([0])

    sources, targets = graph.nonzero()
    weights = graph.data

    g = ig.Graph(n=graph.shape[0], edges=list(zip(sources.tolist(), targets.tolist())), directed=False)

    if len(weights) > 0:
        g.es['weight'] = weights
    else:
        g.es['weight'] = [1.0] * len(g.es)

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_state
    )

    return np.array(partition.membership)


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
    from scipy.spatial import cKDTree
    from scipy import sparse
    
    if len(np.unique(cluster_labels)) <= 1:
        return 0.0
    
    n = len(cluster_labels)
    if n < 2:
        return 0.0
    
    # Use KDTree for efficient spatial neighbor finding
    tree = cKDTree(spatial_coords)
    
    # Find neighbors within reasonable distance (avoid extremely long-range connections)
    max_distance = np.percentile(tree.query(spatial_coords, k=2)[0][:, 1], 95) * 3
    
    # Build sparse weights matrix using neighbor queries
    row_indices = []
    col_indices = []
    weights_data = []
    
    for i in range(n):
        # Find neighbors within max_distance
        neighbors = tree.query_ball_point(spatial_coords[i], max_distance)
        for j in neighbors:
            if i != j:  # Exclude self-connections
                dist = np.linalg.norm(spatial_coords[i] - spatial_coords[j])
                weight = 1.0 / (dist + 1e-10)
                row_indices.append(i)
                col_indices.append(j)
                weights_data.append(weight)
    
    # Create sparse weights matrix
    weights_sparse = sparse.csr_matrix(
        (weights_data, (row_indices, col_indices)), 
        shape=(n, n)
    )
    
    # Calculate Moran's I efficiently with sparse operations
    y = cluster_labels - cluster_labels.mean()
    W = weights_sparse.sum()
    
    # Compute weighted spatial autocovariance efficiently
    numerator = 0.0
    for i in range(n):
        for j_idx in range(weights_sparse.indptr[i], weights_sparse.indptr[i+1]):
            j = weights_sparse.indices[j_idx]
            weight = weights_sparse.data[j_idx]
            numerator += weight * y[i] * y[j]
    
    denominator = np.sum(y ** 2)
    
    morans_i = (n / W) * (numerator / denominator)
    
    return float(morans_i)
