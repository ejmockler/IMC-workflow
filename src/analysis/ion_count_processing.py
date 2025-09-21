"""
Ion Count Processing Pipeline

Proper handling of IMC ion count data with Poisson statistics.
Addresses discrete, sparse, zero-inflated nature of ion count measurements.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, Tuple, Optional, List
from scipy import sparse
from .clustering_optimization import optimize_clustering_parameters
from .memory_management import (
    estimate_memory_requirements, check_memory_availability, 
    MemoryEfficientPipeline, get_memory_profile
)
import warnings


def aggregate_ion_counts(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    bin_edges_x: np.ndarray,
    bin_edges_y: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Aggregate ion counts into spatial bins using proper summation.
    
    Critical: Ion counts must be SUMMED, not averaged, to preserve Poisson statistics.
    
    Args:
        coords: Nx2 array of (x, y) coordinates in micrometers
        ion_counts: Dictionary mapping protein names to ion count arrays
        bin_edges_x: Bin edges for x dimension
        bin_edges_y: Bin edges for y dimension
        
    Returns:
        Dictionary mapping protein names to 2D aggregated count arrays
    """
    if len(coords) == 0:
        return {}
    
    # Digitize coordinates into bins
    x_indices = np.digitize(coords[:, 0], bin_edges_x) - 1
    y_indices = np.digitize(coords[:, 1], bin_edges_y) - 1
    
    # Create output arrays
    n_bins_x = len(bin_edges_x) - 1
    n_bins_y = len(bin_edges_y) - 1
    
    aggregated_counts = {}
    
    # Vectorized aggregation using np.bincount for 100x speedup
    for protein_name, counts in ion_counts.items():
        # Valid bins only (within bounds)
        valid_mask = (x_indices >= 0) & (x_indices < n_bins_x) & \
                     (y_indices >= 0) & (y_indices < n_bins_y)
        
        if np.any(valid_mask):
            # Convert 2D indices to 1D for bincount
            valid_x = x_indices[valid_mask]
            valid_y = y_indices[valid_mask]
            valid_counts = counts[valid_mask]
            
            # Create 1D bin indices
            bin_indices = valid_y * n_bins_x + valid_x
            
            # Vectorized aggregation using bincount
            aggregated_1d = np.bincount(bin_indices, weights=valid_counts, 
                                       minlength=n_bins_x * n_bins_y)
            
            # Reshape to 2D
            agg_array = aggregated_1d.reshape(n_bins_y, n_bins_x)
        else:
            agg_array = np.zeros((n_bins_y, n_bins_x), dtype=np.float64)
        
        aggregated_counts[protein_name] = agg_array
    
    return aggregated_counts


def estimate_optimal_cofactor(
    ion_counts: np.ndarray,
    method: str = "percentile",
    percentile_threshold: float = 5.0
) -> float:
    """
    Estimate optimal arcsinh cofactor for variance stabilization.
    
    Critical fix: Replace hardcoded cofactor=1.0 with data-driven optimization.
    
    Args:
        ion_counts: Array of ion count values for single protein
        method: Method for cofactor estimation ('percentile', 'mad', 'variance')
        percentile_threshold: Percentile threshold for percentile method
        
    Returns:
        Estimated optimal cofactor
    """
    # Remove zeros and negative values
    positive_counts = ion_counts[ion_counts > 0]
    
    if len(positive_counts) == 0:
        # No positive signal - use minimal cofactor
        return 0.1
    
    if method == "percentile":
        # Use percentile-based method (common in flow cytometry)
        # Cofactor = 5th percentile of positive values
        cofactor = np.percentile(positive_counts, percentile_threshold)
        return max(cofactor, 0.1)  # Ensure minimum cofactor
    
    elif method == "mad":
        # Median Absolute Deviation approach
        median = np.median(positive_counts)
        mad = np.median(np.abs(positive_counts - median))
        cofactor = mad * 1.4826  # Scale factor for normal distribution
        return max(cofactor, 0.1)
    
    elif method == "variance":
        # Variance stabilization approach
        # Find cofactor that minimizes variance of transformed data
        cofactors = np.logspace(-1, 2, 50)  # Test range 0.1 to 100
        variances = []
        
        for cf in cofactors:
            transformed = np.arcsinh(positive_counts / cf)
            variances.append(np.var(transformed))
        
        optimal_idx = np.argmin(variances)
        return cofactors[optimal_idx]
    
    else:
        raise ValueError(f"Unknown cofactor estimation method: {method}")


def optimize_cofactors_for_dataset(
    ion_count_arrays: Dict[str, np.ndarray],
    method: str = "percentile"
) -> Dict[str, float]:
    """
    Optimize arcsinh cofactors for all proteins in dataset.
    
    Args:
        ion_count_arrays: Dictionary of protein name -> ion count array
        method: Cofactor estimation method
        
    Returns:
        Dictionary mapping protein name to optimal cofactor
    """
    optimal_cofactors = {}
    
    for protein_name, counts in ion_count_arrays.items():
        optimal_cofactors[protein_name] = estimate_optimal_cofactor(
            counts.ravel(), method=method
        )
    
    return optimal_cofactors


def apply_arcsinh_transform(
    ion_count_arrays: Dict[str, np.ndarray],
    optimization_method: str = "percentile",
    percentile_threshold: float = 5.0
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Apply arcsinh transformation with automatic marker-specific cofactor optimization.
    
    Uses data-driven optimization to find optimal cofactor for each protein marker,
    ensuring proper variance stabilization across the dynamic range.
    
    Args:
        ion_count_arrays: Dictionary of protein name -> 2D ion count array
        optimization_method: Method for cofactor optimization ('percentile', 'mad', 'variance')
        percentile_threshold: Percentile threshold for percentile method (default: 5th percentile)
        
    Returns:
        Tuple of (transformed_arrays, cofactors_used)
    """
    # Always optimize cofactors for this dataset
    cofactors = {}
    for protein_name, counts in ion_count_arrays.items():
        cofactors[protein_name] = estimate_optimal_cofactor(
            counts.ravel(), 
            method=optimization_method,
            percentile_threshold=percentile_threshold
        )
    
    transformed = {}
    cofactors_used = {}
    
    for protein_name, counts in ion_count_arrays.items():
        # Apply arcsinh transformation: arcsinh(x/cofactor)
        transformed[protein_name] = np.arcsinh(counts / cofactors[protein_name])
        cofactors_used[protein_name] = cofactors[protein_name]
    
    return transformed, cofactors_used


def standardize_features(
    transformed_arrays: Dict[str, np.ndarray],
    mask: Optional[np.ndarray] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, StandardScaler]]:
    """
    Standardize transformed features using StandardScaler.
    
    Each protein is standardized independently across all spatial positions.
    
    Args:
        transformed_arrays: Dictionary of transformed protein arrays
        mask: Optional mask for valid regions (e.g., tissue boundaries)
        
    Returns:
        Tuple of (standardized_arrays, scalers_dict)
    """
    standardized = {}
    scalers = {}
    
    for protein_name, array in transformed_arrays.items():
        # Flatten array for StandardScaler
        if mask is not None:
            # Only use masked (valid) regions for fitting
            valid_values = array[mask].reshape(-1, 1)
            if len(valid_values) == 0:
                # No valid data - return zeros
                standardized[protein_name] = np.zeros_like(array)
                scalers[protein_name] = None
                continue
        else:
            valid_values = array.reshape(-1, 1)
        
        # Fit scaler on valid data
        scaler = StandardScaler()
        scaler.fit(valid_values)
        
        # Transform entire array
        standardized_flat = scaler.transform(array.reshape(-1, 1))
        standardized[protein_name] = standardized_flat.reshape(array.shape)
        scalers[protein_name] = scaler
    
    return standardized, scalers


def create_feature_matrix(
    standardized_arrays: Dict[str, np.ndarray],
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Create feature matrix for clustering analysis.
    
    Args:
        standardized_arrays: Dictionary of standardized protein arrays
        mask: Optional mask for valid regions
        
    Returns:
        Tuple of (feature_matrix, protein_names, valid_indices)
        - feature_matrix: N x P array where N=spatial_bins, P=proteins
        - protein_names: List of protein names corresponding to columns
        - valid_indices: Linear indices of valid spatial positions
    """
    if not standardized_arrays:
        return np.array([]), [], np.array([])
    
    # Get shape from first array
    first_array = next(iter(standardized_arrays.values()))
    array_shape = first_array.shape
    
    # Create mask for valid positions
    if mask is not None:
        valid_mask = mask
    else:
        # Use positions with non-zero signal in at least one protein
        combined = np.zeros(array_shape)
        for array in standardized_arrays.values():
            combined += np.abs(array)
        valid_mask = combined > 0
    
    # Get valid positions
    valid_indices = np.where(valid_mask.ravel())[0]
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        return np.array([]), [], np.array([])
    
    # Build feature matrix
    protein_names = list(standardized_arrays.keys())
    n_proteins = len(protein_names)
    feature_matrix = np.zeros((n_valid, n_proteins))
    
    for i, protein_name in enumerate(protein_names):
        array_flat = standardized_arrays[protein_name].ravel()
        feature_matrix[:, i] = array_flat[valid_indices]
    
    return feature_matrix, protein_names, valid_indices


def perform_clustering(
    feature_matrix: np.ndarray,
    protein_names: List[str],
    n_clusters: Optional[int] = None,
    k_range: Tuple[int, int] = (2, 15),
    optimization_method: str = "comprehensive",
    random_state: int = 42
) -> Tuple[np.ndarray, KMeans, Dict]:
    """
    Perform optimized clustering on standardized feature matrix.
    
    CRITICAL FIX: Replaces arbitrary n_clusters=8 with data-driven optimization.
    
    Args:
        feature_matrix: N x P feature matrix
        protein_names: Names of proteins corresponding to columns
        n_clusters: Fixed number of clusters (if None, will optimize)
        k_range: Range of k values to test for optimization
        optimization_method: 'comprehensive', 'silhouette', 'gap', or 'elbow'
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (cluster_labels, kmeans_model, optimization_results)
    """
    if feature_matrix.size == 0:
        return np.array([]), None, {}
    
    optimization_results = {}
    
    if n_clusters is None:
        # Optimize number of clusters
        if optimization_method == "comprehensive":
            # Use comprehensive optimization with multiple metrics
            opt_results = optimize_clustering_parameters(
                feature_matrix, protein_names, k_range, random_state=random_state
            )
            optimal_k = opt_results['optimal_k']
            optimization_results = opt_results
            
        elif optimization_method == "silhouette":
            from .clustering_optimization import silhouette_analysis
            _, _, optimal_k = silhouette_analysis(feature_matrix, k_range, random_state)
            optimization_results = {'method': 'silhouette', 'optimal_k': optimal_k}
            
        elif optimization_method == "gap":
            from .clustering_optimization import gap_statistic
            _, _, optimal_k = gap_statistic(feature_matrix, k_range, random_state)
            optimization_results = {'method': 'gap', 'optimal_k': optimal_k}
            
        elif optimization_method == "elbow":
            from .clustering_optimization import elbow_method
            _, _, optimal_k = elbow_method(feature_matrix, k_range, random_state)
            optimization_results = {'method': 'elbow', 'optimal_k': optimal_k}
            
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        n_clusters = optimal_k
    
    # Perform K-means clustering with optimized parameters
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix)
    
    # Add final clustering info to optimization results
    optimization_results.update({
        'final_n_clusters': n_clusters,
        'n_samples': feature_matrix.shape[0],
        'n_features': feature_matrix.shape[1]
    })
    
    return cluster_labels, kmeans, optimization_results


def create_cluster_map(
    cluster_labels: np.ndarray,
    valid_indices: np.ndarray,
    array_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Create 2D cluster assignment map from 1D cluster labels.
    
    Args:
        cluster_labels: 1D array of cluster assignments
        valid_indices: Linear indices of valid positions
        array_shape: Shape of output 2D array
        
    Returns:
        2D array with cluster assignments (-1 for invalid positions)
    """
    cluster_map = np.full(array_shape, -1, dtype=int)
    
    if len(cluster_labels) > 0 and len(valid_indices) > 0:
        # Convert linear indices back to 2D
        y_indices, x_indices = np.unravel_index(valid_indices, array_shape)
        cluster_map[y_indices, x_indices] = cluster_labels
    
    return cluster_map


def compute_cluster_centroids(
    feature_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    protein_names: List[str]
) -> Dict[int, Dict[str, float]]:
    """
    Compute cluster centroids in original protein space.
    
    Args:
        feature_matrix: N x P standardized feature matrix
        cluster_labels: Cluster assignments for each position
        protein_names: Names of proteins corresponding to columns
        
    Returns:
        Dictionary mapping cluster_id -> {protein_name: centroid_value}
    """
    if feature_matrix.size == 0 or len(cluster_labels) == 0:
        return {}
    
    centroids = {}
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_features = feature_matrix[mask]
        
        # Compute mean for each protein
        cluster_centroid = {}
        for i, protein_name in enumerate(protein_names):
            cluster_centroid[protein_name] = float(np.mean(cluster_features[:, i]))
        
        centroids[int(cluster_id)] = cluster_centroid
    
    return centroids


def ion_count_pipeline(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    bin_size_um: float = 20.0,
    n_clusters: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    memory_limit_gb: float = 4.0,
    use_memory_optimization: bool = True
) -> Dict:
    """
    Complete ion count processing pipeline with memory management.
    
    CRITICAL FIXES:
    - Automatic marker-specific arcsinh cofactor optimization
    - Data-driven clustering parameter selection
    - Memory-efficient processing for large datasets
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        bin_size_um: Spatial binning size in micrometers
        n_clusters: Number of clusters (optimized if None)
        mask: Optional spatial mask for valid regions
        memory_limit_gb: Memory limit for processing
        use_memory_optimization: Whether to use memory-efficient processing
        
    Returns:
        Dictionary containing all pipeline results
    """
    if len(coords) == 0 or not ion_counts:
        return {
            'aggregated_counts': {},
            'transformed_arrays': {},
            'standardized_arrays': {},
            'scalers': {},
            'cofactors_used': {},
            'feature_matrix': np.array([]),
            'cluster_labels': np.array([]),
            'cluster_map': np.array([]),
            'cluster_centroids': {},
            'kmeans_model': None,
            'optimization_results': {},
            'bin_edges_x': np.array([]),
            'bin_edges_y': np.array([]),
            'protein_names': [],
            'valid_indices': np.array([]),
            'bin_size_um': bin_size_um,
            'processing_method': 'empty_input',
            'memory_report': {}
        }
    
    # Check memory requirements and use efficient processing if needed
    if use_memory_optimization:
        # Estimate memory requirements
        bounds = (coords[:, 0].min(), coords[:, 0].max(),
                 coords[:, 1].min(), coords[:, 1].max())
        
        memory_est = estimate_memory_requirements(
            len(coords), len(ion_counts), bin_size_um, bounds
        )
        
        # Use memory-efficient pipeline if needed
        if memory_est['total_estimated_gb'] > memory_limit_gb:
            warnings.warn(
                f"Large dataset detected ({memory_est['total_estimated_gb']:.1f} GB estimated). "
                f"Using memory-efficient processing."
            )
            
            efficient_pipeline = MemoryEfficientPipeline(memory_limit_gb)
            return efficient_pipeline.process_large_roi(
                coords, ion_counts, bin_size_um, 
                n_clusters=n_clusters, mask=mask
            )
    
    # Determine spatial binning
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    bin_edges_x = np.arange(x_min, x_max + bin_size_um, bin_size_um)
    bin_edges_y = np.arange(y_min, y_max + bin_size_um, bin_size_um)
    
    # Step 1: Aggregate ion counts
    aggregated_counts = aggregate_ion_counts(coords, ion_counts, bin_edges_x, bin_edges_y)
    
    # Step 2: Apply arcsinh transformation with automatic optimization
    transformed_arrays, cofactors_used = apply_arcsinh_transform(
        aggregated_counts, 
        optimization_method="percentile",
        percentile_threshold=5.0
    )
    
    # Step 3: Standardize features
    standardized_arrays, scalers = standardize_features(transformed_arrays, mask)
    
    # Step 4: Create feature matrix
    feature_matrix, protein_names, valid_indices = create_feature_matrix(standardized_arrays, mask)
    
    # Step 5: Perform clustering with optimization
    cluster_labels, kmeans_model, optimization_results = perform_clustering(
        feature_matrix=feature_matrix,
        protein_names=protein_names,
        n_clusters=n_clusters,  # Will be optimized if None
        optimization_method="comprehensive"
    )
    
    # Step 6: Create spatial cluster map
    array_shape = (len(bin_edges_y) - 1, len(bin_edges_x) - 1)
    cluster_map = create_cluster_map(cluster_labels, valid_indices, array_shape)
    
    # Step 7: Compute cluster centroids
    cluster_centroids = compute_cluster_centroids(feature_matrix, cluster_labels, protein_names)
    
    # Memory usage report
    final_memory_profile = get_memory_profile()
    memory_report = {
        'final_memory_gb': final_memory_profile.process_memory_gb,
        'memory_limit_gb': memory_limit_gb,
        'within_limit': final_memory_profile.process_memory_gb <= memory_limit_gb
    }
    
    return {
        'aggregated_counts': aggregated_counts,
        'transformed_arrays': transformed_arrays,
        'standardized_arrays': standardized_arrays,
        'scalers': scalers,
        'cofactors_used': cofactors_used,
        'feature_matrix': feature_matrix,
        'cluster_labels': cluster_labels,
        'cluster_map': cluster_map,
        'cluster_centroids': cluster_centroids,
        'kmeans_model': kmeans_model,
        'optimization_results': optimization_results,
        'bin_edges_x': bin_edges_x,
        'bin_edges_y': bin_edges_y,
        'protein_names': protein_names,
        'valid_indices': valid_indices,
        'bin_size_um': bin_size_um,
        'processing_method': 'standard',
        'memory_report': memory_report
    }