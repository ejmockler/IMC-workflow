"""
Multi-Scale Analysis Framework

Compare tissue microenvironments at multiple spatial scales to address heterogeneity.
Implements 10μm, 20μm, and 40μm scale analysis with consistency metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from .ion_count_processing import ion_count_pipeline
from .slic_segmentation import slic_pipeline


def perform_multiscale_analysis(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    dna1_intensities: np.ndarray,
    dna2_intensities: np.ndarray,
    scales_um: List[float] = [10.0, 20.0, 40.0],
    n_clusters: int = 8,
    use_slic: bool = True
) -> Dict[float, Dict]:
    """
    Perform analysis at multiple spatial scales.
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        dna1_intensities: DNA1 channel data
        dna2_intensities: DNA2 channel data
        scales_um: List of spatial scales to analyze
        n_clusters: Number of clusters for each scale
        use_slic: Whether to use SLIC segmentation (vs square binning)
        
    Returns:
        Dictionary mapping scale -> analysis results
    """
    results = {}
    
    for scale_um in scales_um:
        if use_slic:
            # Use SLIC-based morphology-aware analysis
            scale_result = slic_pipeline(
                coords, ion_counts, dna1_intensities, dna2_intensities,
                target_bin_size_um=scale_um
            )
            
            # Apply ion count processing to superpixel-aggregated data
            if scale_result['superpixel_counts']:
                ion_result = ion_count_pipeline(
                    scale_result['superpixel_coords'],
                    scale_result['superpixel_counts'],
                    bin_size_um=scale_um,  # This won't be used since data is pre-aggregated
                    n_clusters=n_clusters
                )
                
                # Merge results
                scale_result.update(ion_result)
        
        else:
            # Use square binning approach
            ion_result = ion_count_pipeline(
                coords, ion_counts,
                bin_size_um=scale_um,
                n_clusters=n_clusters
            )
            scale_result = ion_result
        
        # Add scale identifier
        scale_result['scale_um'] = scale_um
        scale_result['method'] = 'slic' if use_slic else 'square'
        
        results[scale_um] = scale_result
    
    return results


def compute_scale_consistency(
    multiscale_results: Dict[float, Dict],
    consistency_metrics: List[str] = ['ari', 'nmi', 'cluster_stability']
) -> Dict[str, Dict]:
    """
    Compute consistency metrics between different spatial scales.
    
    Args:
        multiscale_results: Results from perform_multiscale_analysis
        consistency_metrics: List of metrics to compute
        
    Returns:
        Dictionary of consistency measurements
    """
    scales = sorted(multiscale_results.keys())
    consistency_results = {}
    
    # Pairwise scale comparisons
    for i, scale1 in enumerate(scales):
        for scale2 in scales[i+1:]:
            pair_key = f"{scale1}um_vs_{scale2}um"
            consistency_results[pair_key] = {}
            
            result1 = multiscale_results[scale1]
            result2 = multiscale_results[scale2]
            
            if 'ari' in consistency_metrics:
                ari = compute_adjusted_rand_index(result1, result2)
                consistency_results[pair_key]['adjusted_rand_index'] = ari
            
            if 'nmi' in consistency_metrics:
                nmi = compute_normalized_mutual_info(result1, result2)
                consistency_results[pair_key]['normalized_mutual_info'] = nmi
            
            if 'cluster_stability' in consistency_metrics:
                stability = compute_cluster_centroid_stability(result1, result2)
                consistency_results[pair_key]['centroid_stability'] = stability
    
    # Overall consistency scores
    overall_consistency = compute_overall_consistency(consistency_results)
    consistency_results['overall'] = overall_consistency
    
    return consistency_results


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
    summary = {
        'scales_analyzed': sorted(multiscale_results.keys()),
        'consistency_summary': consistency_results.get('overall', {}),
        'scale_specific_summaries': {}
    }
    
    # Summarize each scale
    for scale_um, result in multiscale_results.items():
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