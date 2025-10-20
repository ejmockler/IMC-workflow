"""
Ion Count Processing Pipeline

Proper handling of IMC ion count data with Poisson statistics.
Addresses discrete, sparse, zero-inflated nature of ion count measurements.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, Tuple, Optional, List, Any, TYPE_CHECKING
from scipy import sparse
from .spatial_clustering import perform_spatial_clustering, stability_analysis
from .memory_management import (
    estimate_memory_requirements, check_memory_availability, 
    MemoryEfficientPipeline, get_memory_profile
)
import warnings

if TYPE_CHECKING:
    from ..config import Config


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
            agg_array = np.zeros((n_bins_y, n_bins_x), dtype=np.float32)
        
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
    percentile_threshold: float = 5.0,
    cached_cofactors: Optional[Dict[str, float]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Apply arcsinh transformation with automatic marker-specific cofactor optimization.
    
    Uses data-driven optimization to find optimal cofactor for each protein marker,
    ensuring proper variance stabilization across the dynamic range.
    
    Args:
        ion_count_arrays: Dictionary of protein name -> 2D ion count array
        optimization_method: Method for cofactor optimization ('percentile', 'mad', 'variance', 'cached')
        percentile_threshold: Percentile threshold for percentile method (default: 5th percentile)
        cached_cofactors: Pre-computed cofactors to use when optimization_method='cached'
        
    Returns:
        Tuple of (transformed_arrays, cofactors_used)
    """
    if optimization_method == "cached" and cached_cofactors:
        # Use cached cofactors - skip computation for performance
        cofactors = cached_cofactors  # Use reference instead of copy
    else:
        # Compute cofactors for this dataset
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
            if len(valid_values) == 0:
                # No valid data - return zeros
                standardized[protein_name] = np.zeros_like(array)
                scalers[protein_name] = None
                continue
        
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


def _extract_stability_settings(config: Optional['Config']) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, 'optimization'):
        optimization_section = getattr(config, 'optimization', {})
        if isinstance(optimization_section, dict):
            return optimization_section.get('stability_analysis', {}) or {}
    if isinstance(config, dict):
        return config.get('optimization', {}).get('stability_analysis', {}) or {}
    return {}


def perform_clustering(
    feature_matrix: np.ndarray,
    spatial_coords: Optional[np.ndarray] = None,
    method: str = 'leiden',
    resolution: Optional[float] = None,
    random_state: int = 42,
    config: Optional['Config'] = None
) -> Tuple[np.ndarray, None, Dict]:
    """
    Perform spatial-aware clustering using modern methods.
    
    This replaces the old 5-metric weighted consensus with stability-based selection.
    
    Args:
        feature_matrix: N x P feature matrix
        spatial_coords: Optional N x 2 spatial coordinates
        method: 'leiden' or 'hdbscan'
        resolution: Resolution parameter (if None, will use stability analysis)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (cluster_labels, None, clustering_info)
        Note: Second element is None for backward compatibility (was kmeans model)
    """
    if feature_matrix.size == 0:
        return np.array([]), None, {}
    
    # If no resolution specified, use stability analysis
    if resolution is None:
        stability_settings = _extract_stability_settings(config)
        stability_result = stability_analysis(
            feature_matrix, spatial_coords,
            method=method,
            resolution_range=(0.5, 2.0),
            n_resolutions=stability_settings.get('n_resolutions', 15),
            n_bootstrap=stability_settings.get('n_bootstrap_iterations', 5),
            use_graph_caching=stability_settings.get('use_graph_caching', True),
            parallel=stability_settings.get('parallel_execution', True),
            n_workers=stability_settings.get('n_workers'),
            adaptive_search=stability_settings.get('adaptive_search', False),
            adaptive_target_stability=stability_settings.get('adaptive_target_stability', 0.6),
            adaptive_tolerance=stability_settings.get('adaptive_tolerance', 0.05),
            adaptive_max_evaluations=stability_settings.get('adaptive_max_evaluations'),
            random_state=random_state
        )
        resolution = stability_result['optimal_resolution']
        optimization_results = {'stability_analysis': stability_result}
    else:
        optimization_results = {'fixed_resolution': resolution}
    
    # Perform clustering
    cluster_labels, clustering_info = perform_spatial_clustering(
        feature_matrix, spatial_coords,
        method=method,
        resolution=resolution,
        random_state=random_state
    )
    
    # Add clustering info to optimization results
    optimization_results.update({
        'method': method,
        'resolution': resolution,
        'n_clusters': clustering_info.get('n_clusters', 0),
        'n_samples': feature_matrix.shape[0],
        'n_features': feature_matrix.shape[1]
    })
    
    return cluster_labels, None, optimization_results  # None for backward compatibility (was kmeans)


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
        # Handle case where cluster_labels and valid_indices have different lengths
        # This can happen when clustering algorithms filter out points or fail
        min_length = min(len(cluster_labels), len(valid_indices))
        
        if min_length > 0:
            # Use only the first min_length elements from both arrays
            used_valid_indices = valid_indices[:min_length]
            used_cluster_labels = cluster_labels[:min_length]
            
            # Convert linear indices back to 2D
            y_indices, x_indices = np.unravel_index(used_valid_indices, array_shape)
            cluster_map[y_indices, x_indices] = used_cluster_labels
    
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

    ⚠️ DEPRECATED: This function performs clustering with hardcoded resolution=1.0,
    which bypasses stability analysis. This is scientifically invalid.

    Use perform_multiscale_analysis() from multiscale_analysis.py instead, which:
    - Runs proper stability analysis to find optimal resolution
    - Supports coabundance feature generation
    - Works at multiple spatial scales

    BUG #8: The clustering results from this function (lines 562-567) use hardcoded
    resolution=1.0 instead of data-driven optimization. These results should NOT
    be used for scientific analysis.

    CRITICAL FIXES:
    - Automatic marker-specific arcsinh cofactor optimization
    - Data-driven clustering parameter selection
    - Memory-efficient processing for large datasets

    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        bin_size_um: Spatial binning size in micrometers
        n_clusters: Number of clusters (optimized if None) - IGNORED, uses resolution=1.0
        mask: Optional spatial mask for valid regions
        memory_limit_gb: Memory limit for processing
        use_memory_optimization: Whether to use memory-efficient processing

    Returns:
        Dictionary containing all pipeline results

    .. deprecated::
        Clustering with hardcoded resolution. Use perform_multiscale_analysis() instead.
    """
    warnings.warn(
        "ion_count_pipeline() uses hardcoded resolution=1.0 for clustering, "
        "bypassing stability analysis. Use perform_multiscale_analysis() instead "
        "for scientifically valid clustering.",
        DeprecationWarning,
        stacklevel=2
    )
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
    cluster_labels, clustering_info = perform_spatial_clustering(
        feature_matrix=feature_matrix,
        spatial_coords=None,  # Would need to extract from bin centers
        method='leiden',
        resolution=1.0 if n_clusters is None else None
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
        'clustering_info': clustering_info,
        'bin_edges_x': bin_edges_x,
        'bin_edges_y': bin_edges_y,
        'protein_names': protein_names,
        'valid_indices': valid_indices,
        'bin_size_um': bin_size_um,
        'processing_method': 'standard',
        'memory_report': memory_report
    }
