"""
Multi-Scale Analysis Framework

Compare tissue microenvironments at multiple spatial scales to address heterogeneity.
Implements 10μm, 20μm, and 40μm scale analysis with consistency metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from .ion_count_processing import ion_count_pipeline
from .slic_segmentation import slic_pipeline
from .spatial_clustering import perform_spatial_clustering, stability_analysis, compute_spatial_coherence
from .hierarchical_multiscale import build_multiscale_hierarchy, add_neighbor_composition_features

if TYPE_CHECKING:
    from ..config import Config


def perform_multiscale_analysis(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    dna1_intensities: np.ndarray,
    dna2_intensities: np.ndarray,
    scales_um: List[float] = [10.0, 20.0, 40.0],
    method: str = 'leiden',
    use_slic: bool = True,
    config: Optional['Config'] = None,
    cached_cofactors: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Perform hierarchical multi-scale analysis with data-driven clustering.
    
    This new approach:
    1. Performs spatial aggregation at each scale
    2. Runs stability analysis to find optimal resolution
    3. Builds hierarchical relationships between scales
    4. Provides diagnostic visualizations for scientist review
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        dna1_intensities: DNA1 channel data
        dna2_intensities: DNA2 channel data
        scales_um: List of spatial scales to analyze
        method: Clustering method ('leiden', 'hdbscan')
        use_slic: Whether to use SLIC segmentation
        config: Configuration object
        cached_cofactors: Pre-computed arcsinh cofactors
        
    Returns:
        Dictionary with hierarchical analysis results
    """
    import logging
    logger = logging.getLogger('MultiscaleAnalysis')
    logger.info(f"Starting hierarchical multi-scale analysis with {len(coords)} pixels")
    
    results = {}
    
    # Process each scale
    for scale_idx, scale_um in enumerate(scales_um):
        logger.info(f"Processing scale {scale_um}μm...")
        
        # Step 1: Spatial aggregation
        if use_slic:
            aggregation_result = slic_pipeline(
                coords=coords,
                ion_counts=ion_counts,
                dna1_intensities=dna1_intensities,
                dna2_intensities=dna2_intensities,
                target_scale_um=scale_um,
                n_segments=None  # Data-driven
            )
            # Extract features from superpixels
            features = np.array([aggregation_result['superpixel_counts'][protein] 
                                for protein in ion_counts.keys()]).T
            spatial_coords = aggregation_result['superpixel_coords']
        else:
            aggregation_result = ion_count_pipeline(
                coords=coords,
                ion_counts=ion_counts,
                bin_size_um=scale_um
            )
            features = aggregation_result['feature_matrix']
            # Create spatial coordinates from bin centers
            bin_edges_x = aggregation_result['bin_edges_x']
            bin_edges_y = aggregation_result['bin_edges_y']
            valid_indices = aggregation_result['valid_indices']
            
            # Create bin center coordinates for valid bins
            if len(bin_edges_x) > 1 and len(bin_edges_y) > 1:
                x_centers = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2
                y_centers = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2
                
                # Convert valid indices to 2D coordinates  
                n_x_bins = len(bin_edges_x) - 1
                y_indices = valid_indices // n_x_bins
                x_indices = valid_indices % n_x_bins
                
                spatial_coords = np.column_stack([
                    x_centers[x_indices],
                    y_centers[y_indices]
                ])
            else:
                spatial_coords = np.zeros((len(features), 2))
        
        # Get protein names
        protein_names = list(ion_counts.keys())
        
        # Step 2: Add spatial features for coarse scales
        enhanced_protein_names = protein_names.copy()
        if scale_um >= 20 and features.size > 0 and len(features.shape) == 2:
            original_n_features = features.shape[1]
            features = add_neighbor_composition_features(
                features, spatial_coords, 
                np.zeros(len(features)),  # Placeholder labels
                radius_um=scale_um
            )
            # Add feature names for the composition features
            n_composition_features = features.shape[1] - original_n_features
            for i in range(n_composition_features):
                enhanced_protein_names.append(f"neighborhood_comp_{i}")
        
        # Step 3: Stability analysis for resolution selection
        stability_result = stability_analysis(
            features, spatial_coords,
            method=method,
            resolution_range=(0.5, 2.0) if scale_um <= 20 else (0.2, 1.0),
            n_resolutions=15,
            n_bootstrap=10
        )
        
        optimal_resolution = stability_result['optimal_resolution']
        
        # Step 4: Perform clustering at optimal resolution with co-abundance features
        cluster_labels, clustering_info = perform_spatial_clustering(
            features, spatial_coords,
            method=method,
            resolution=optimal_resolution,
            spatial_weight=0.2 if scale_um <= 20 else 0.4,  # More spatial weight at coarse scales
            use_coabundance=True,  # Enable co-abundance feature generation
            protein_names=enhanced_protein_names
        )
        
        # Step 5: Compute spatial coherence
        coherence = compute_spatial_coherence(cluster_labels, spatial_coords)
        
        results[scale_um] = {
            'features': features,
            'spatial_coords': spatial_coords,
            'cluster_labels': cluster_labels,
            'clustering_info': clustering_info,
            'stability_analysis': stability_result,
            'spatial_coherence': coherence,
            'scale_um': scale_um,
            'method': 'slic' if use_slic else 'square'
        }
    
    # Step 6: Build hierarchy if we have multiple scales
    if len(scales_um) > 1:
        # Use finest scale as base
        finest_scale = min(scales_um)
        hierarchy = build_multiscale_hierarchy(
            results[finest_scale]['features'],
            results[finest_scale]['spatial_coords'],
            results[finest_scale]['cluster_labels'],
            coarsening_factor=2.0,
            n_scales=len(scales_um)
        )
        results['hierarchy'] = hierarchy
    
    return results


# OLD perform_multiscale_analysis DELETED - Now using hierarchical approach only


def compute_scale_consistency(
    multiscale_results: Dict[float, Dict]
) -> Dict[str, Dict]:
    """
    Validate hierarchical relationships between scales.
    
    With the new hierarchical approach, we check parent-child coherence
    rather than arbitrary pairwise comparisons.
    
    Args:
        multiscale_results: Results from perform_multiscale_analysis
        
    Returns:
        Dictionary of hierarchical validation metrics
    """
    from .hierarchical_multiscale import validate_hierarchy
    
    if 'hierarchy' in multiscale_results:
        return validate_hierarchy(multiscale_results['hierarchy'])
    else:
        # Fallback for non-hierarchical results
        return {'error': 'No hierarchy found - run hierarchical analysis first'}


# Old pairwise comparison functions removed - using hierarchical validation instead

def compute_adjusted_rand_index(
    result1: Dict, 
    result2: Dict,
    overlap_threshold: float = 0.5
) -> float:
    """
    Compute Adjusted Rand Index between two scale clusterings.
    
    For different spatial scales, we need to handle overlapping regions.
    """
    if 'cluster_map' not in result1 or 'cluster_map' not in result2:
        return np.nan
    
    map1 = result1['cluster_map']
    map2 = result2['cluster_map']
    
    if map1.size == 0 or map2.size == 0:
        return np.nan
    
    # Resample to common resolution (use finer scale)
    if map1.shape != map2.shape:
        from scipy.ndimage import zoom
        
        # Determine target shape (finer resolution)
        if map1.size > map2.size:
            target_shape = map1.shape
            map2_resampled = zoom(map2.astype(float), 
                                  np.array(target_shape) / np.array(map2.shape), 
                                  order=0, mode='nearest').astype(int)
            map1_resampled = map1
        else:
            target_shape = map2.shape
            map1_resampled = zoom(map1.astype(float), 
                                  np.array(target_shape) / np.array(map1.shape), 
                                  order=0, mode='nearest').astype(int)
            map2_resampled = map2
    else:
        map1_resampled = map1
        map2_resampled = map2
    
    # Only compare valid regions (both scales have valid assignments)
    valid_mask = (map1_resampled >= 0) & (map2_resampled >= 0)
    
    if not np.any(valid_mask):
        return np.nan
    
    labels1 = map1_resampled[valid_mask]
    labels2 = map2_resampled[valid_mask]
    
    return adjusted_rand_score(labels1, labels2)


def compute_normalized_mutual_info(result1: Dict, result2: Dict) -> float:
    """
    Compute Normalized Mutual Information between scale clusterings.
    """
    if 'cluster_map' not in result1 or 'cluster_map' not in result2:
        return np.nan
    
    map1 = result1['cluster_map']
    map2 = result2['cluster_map']
    
    if map1.size == 0 or map2.size == 0:
        return np.nan
    
    # Resample to common resolution (similar to ARI computation)
    if map1.shape != map2.shape:
        from scipy.ndimage import zoom
        
        if map1.size > map2.size:
            target_shape = map1.shape
            map2_resampled = zoom(map2.astype(float), 
                                  np.array(target_shape) / np.array(map2.shape), 
                                  order=0, mode='nearest').astype(int)
            map1_resampled = map1
        else:
            target_shape = map2.shape
            map1_resampled = zoom(map1.astype(float), 
                                  np.array(target_shape) / np.array(map1.shape), 
                                  order=0, mode='nearest').astype(int)
            map2_resampled = map2
    else:
        map1_resampled = map1
        map2_resampled = map2
    
    # Only compare valid regions
    valid_mask = (map1_resampled >= 0) & (map2_resampled >= 0)
    
    if not np.any(valid_mask):
        return np.nan
    
    labels1 = map1_resampled[valid_mask]
    labels2 = map2_resampled[valid_mask]
    
    return normalized_mutual_info_score(labels1, labels2)


def compute_cluster_centroid_stability(result1: Dict, result2: Dict) -> Dict[str, float]:
    """
    Compare cluster centroids between scales using protein profiles.
    """
    if 'cluster_centroids' not in result1 or 'cluster_centroids' not in result2:
        return {}
    
    centroids1 = result1['cluster_centroids']
    centroids2 = result2['cluster_centroids']
    
    if not centroids1 or not centroids2:
        return {}
    
    # Get common proteins
    all_proteins1 = set()
    all_proteins2 = set()
    
    for cluster_data in centroids1.values():
        all_proteins1.update(cluster_data.keys())
    
    for cluster_data in centroids2.values():
        all_proteins2.update(cluster_data.keys())
    
    common_proteins = list(all_proteins1 & all_proteins2)
    
    if not common_proteins:
        return {}
    
    # Create centroid matrices
    n_clusters1 = len(centroids1)
    n_clusters2 = len(centroids2)
    n_proteins = len(common_proteins)
    
    matrix1 = np.zeros((n_clusters1, n_proteins))
    matrix2 = np.zeros((n_clusters2, n_proteins))
    
    # Fill matrices
    for i, (cluster_id, cluster_data) in enumerate(centroids1.items()):
        for j, protein in enumerate(common_proteins):
            matrix1[i, j] = cluster_data.get(protein, 0.0)
    
    for i, (cluster_id, cluster_data) in enumerate(centroids2.items()):
        for j, protein in enumerate(common_proteins):
            matrix2[i, j] = cluster_data.get(protein, 0.0)
    
    # Compute pairwise distances between all centroids
    from scipy.spatial.distance import cdist
    distances = cdist(matrix1, matrix2, metric='euclidean')
    
    # Find best matches (Hungarian algorithm would be better, but this is simpler)
    min_distances = []
    for i in range(n_clusters1):
        min_dist = np.min(distances[i])
        min_distances.append(min_dist)
    
    stability_metrics = {
        'mean_centroid_distance': float(np.mean(min_distances)),
        'max_centroid_distance': float(np.max(min_distances)),
        'std_centroid_distance': float(np.std(min_distances))
    }
    
    return stability_metrics


def compute_overall_consistency(consistency_results: Dict[str, Dict]) -> Dict[str, float]:
    """
    Compute overall consistency metrics across all scale pairs.
    """
    ari_scores = []
    nmi_scores = []
    centroid_distances = []
    
    for pair_key, metrics in consistency_results.items():
        if pair_key == 'overall':  # Skip self
            continue
        
        if 'adjusted_rand_index' in metrics and not np.isnan(metrics['adjusted_rand_index']):
            ari_scores.append(metrics['adjusted_rand_index'])
        
        if 'normalized_mutual_info' in metrics and not np.isnan(metrics['normalized_mutual_info']):
            nmi_scores.append(metrics['normalized_mutual_info'])
        
        if 'centroid_stability' in metrics and 'mean_centroid_distance' in metrics['centroid_stability']:
            centroid_distances.append(metrics['centroid_stability']['mean_centroid_distance'])
    
    overall = {}
    
    if ari_scores:
        overall['mean_ari'] = float(np.mean(ari_scores))
        overall['std_ari'] = float(np.std(ari_scores))
    
    if nmi_scores:
        overall['mean_nmi'] = float(np.mean(nmi_scores))
        overall['std_nmi'] = float(np.std(nmi_scores))
    
    if centroid_distances:
        overall['mean_centroid_distance'] = float(np.mean(centroid_distances))
        overall['std_centroid_distance'] = float(np.std(centroid_distances))
    
    return overall


def identify_scale_dependent_features(
    multiscale_results: Dict[float, Dict],
    protein_names: List[str]
) -> Dict[str, Dict[float, float]]:
    """
    Identify proteins that show scale-dependent spatial organization.
    
    Args:
        multiscale_results: Results from multiscale analysis
        protein_names: List of protein names to analyze
        
    Returns:
        Dictionary mapping protein -> scale -> spatial_metric
    """
    scale_features = {}
    
    for protein_name in protein_names:
        scale_features[protein_name] = {}
        
        for scale_um, result in multiscale_results.items():
            # Extract spatial coefficient of variation for this protein at this scale
            if ('standardized_arrays' in result and 
                protein_name in result['standardized_arrays']):
                
                protein_array = result['standardized_arrays'][protein_name]
                
                if protein_array.size > 0:
                    spatial_cv = float(np.std(protein_array) / (np.mean(np.abs(protein_array)) + 1e-10))
                    scale_features[protein_name][scale_um] = spatial_cv
    
    return scale_features


def summarize_multiscale_analysis(
    multiscale_results: Dict[float, Dict],
    consistency_results: Dict[str, Dict]
) -> Dict[str, any]:
    """
    Create comprehensive summary of multiscale analysis.
    """
    # Filter out non-numeric keys (like 'hierarchy')
    numeric_scales = [k for k in multiscale_results.keys() if isinstance(k, (int, float))]
    
    summary = {
        'scales_analyzed': sorted(numeric_scales),
        'consistency_summary': consistency_results.get('scale_progression', {}),
        'scale_specific_summaries': {}
    }
    
    # Summarize each scale
    for scale_um in numeric_scales:
        result = multiscale_results[scale_um]
        scale_summary = {
            'n_clusters_found': len(result.get('cluster_centroids', {})),
            'n_spatial_bins': len(result.get('cluster_labels', [])),
            'method_used': result.get('method', 'unknown')
        }
        
        # Add cluster size statistics
        if 'cluster_labels' in result and len(result['cluster_labels']) > 0:
            unique_labels, counts = np.unique(result['cluster_labels'], return_counts=True)
            scale_summary['cluster_size_stats'] = {
                'mean': float(np.mean(counts)),
                'std': float(np.std(counts)),
                'min': int(np.min(counts)),
                'max': int(np.max(counts))
            }
        
        summary['scale_specific_summaries'][scale_um] = scale_summary
    
    return summary