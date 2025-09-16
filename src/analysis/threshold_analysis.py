"""
Threshold-Based Analysis

Simple alternative to field-based analysis for n=2 pilot study.
Direct counting of positive pixels without continuous field generation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def extract_threshold_features(
    coords: np.ndarray,
    protein_data: Dict[str, np.ndarray],
    threshold_method: str = 'otsu',
    custom_thresholds: Optional[Dict[str, float]] = None,
    protein_pairs: Optional[List[Tuple[str, str]]] = None
) -> Dict[str, float]:
    """
    Extract features using simple thresholding approach.
    
    Alternative to continuous field generation for sparse IMC data.
    
    Args:
        coords: Nx2 array of (x, y) coordinates
        protein_data: Dictionary mapping protein names to intensity arrays
        threshold_method: Method for threshold selection ('otsu', 'percentile', 'custom')
        custom_thresholds: Pre-defined thresholds for each protein
        protein_pairs: Protein pairs for colocalization analysis
        
    Returns:
        Dictionary of feature_name -> value
    """
    features = {}
    
    # Determine thresholds for each protein
    thresholds = {}
    for protein_name, intensities in protein_data.items():
        if custom_thresholds and protein_name in custom_thresholds:
            thresholds[protein_name] = custom_thresholds[protein_name]
        elif threshold_method == 'otsu':
            thresholds[protein_name] = _compute_otsu_threshold(intensities)
        elif threshold_method == 'percentile':
            thresholds[protein_name] = np.percentile(intensities, 90)  # 90th percentile
        else:
            # Background + 2 standard deviations
            background = np.percentile(intensities, 10)
            noise_std = np.std(intensities[intensities <= background])
            thresholds[protein_name] = background + 2 * noise_std
    
    # Extract features for each protein
    for protein_name, intensities in protein_data.items():
        threshold = thresholds[protein_name]
        positive_mask = intensities > threshold
        
        # Basic counts
        features[f'positive_count_{protein_name}'] = float(np.sum(positive_mask))
        features[f'positive_fraction_{protein_name}'] = float(np.mean(positive_mask))
        features[f'threshold_{protein_name}'] = float(threshold)
        
        # Intensity statistics in positive pixels
        if np.any(positive_mask):
            positive_intensities = intensities[positive_mask]
            features[f'mean_positive_{protein_name}'] = float(np.mean(positive_intensities))
            features[f'std_positive_{protein_name}'] = float(np.std(positive_intensities))
            features[f'max_positive_{protein_name}'] = float(np.max(positive_intensities))
        else:
            features[f'mean_positive_{protein_name}'] = 0.0
            features[f'std_positive_{protein_name}'] = 0.0
            features[f'max_positive_{protein_name}'] = 0.0
    
    # Colocalization analysis
    if protein_pairs:
        for protein1, protein2 in protein_pairs:
            if protein1 in protein_data and protein2 in protein_data:
                mask1 = protein_data[protein1] > thresholds[protein1]
                mask2 = protein_data[protein2] > thresholds[protein2]
                
                # Overlap statistics
                overlap = mask1 & mask2
                union = mask1 | mask2
                
                features[f'colocalization_{protein1}_{protein2}'] = float(np.sum(overlap))
                features[f'jaccard_{protein1}_{protein2}'] = float(
                    np.sum(overlap) / np.sum(union) if np.sum(union) > 0 else 0
                )
                
                # Conditional probabilities
                features[f'p_{protein2}_given_{protein1}'] = float(
                    np.sum(overlap) / np.sum(mask1) if np.sum(mask1) > 0 else 0
                )
                features[f'p_{protein1}_given_{protein2}'] = float(
                    np.sum(overlap) / np.sum(mask2) if np.sum(mask2) > 0 else 0
                )
    
    return features


def _compute_otsu_threshold(intensities: np.ndarray) -> float:
    """
    Compute Otsu's threshold for binary classification.
    
    Args:
        intensities: Array of intensity values
        
    Returns:
        Optimal threshold value
    """
    if len(intensities) == 0:
        return 0.0
    
    # Create histogram
    hist, bin_edges = np.histogram(intensities, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize histogram
    hist = hist / np.sum(hist)
    
    # Compute cumulative statistics
    cumsum = np.cumsum(hist)
    cumsum_intensity = np.cumsum(hist * bin_centers)
    
    # Global mean
    global_mean = cumsum_intensity[-1]
    
    # Compute between-class variance for each threshold
    between_class_variance = np.zeros_like(cumsum)
    
    for i in range(len(cumsum)):
        if cumsum[i] == 0 or cumsum[i] == 1:
            continue
            
        # Class probabilities
        w0 = cumsum[i]
        w1 = 1 - w0
        
        # Class means
        mu0 = cumsum_intensity[i] / w0 if w0 > 0 else 0
        mu1 = (global_mean - cumsum_intensity[i]) / w1 if w1 > 0 else 0
        
        # Between-class variance
        between_class_variance[i] = w0 * w1 * (mu0 - mu1) ** 2
    
    # Find threshold that maximizes between-class variance
    optimal_idx = np.argmax(between_class_variance)
    optimal_threshold = bin_centers[optimal_idx]
    
    return float(optimal_threshold)


def compute_spatial_clustering(
    coords: np.ndarray,
    positive_mask: np.ndarray,
    max_distance_um: float = 50.0
) -> Dict[str, float]:
    """
    Compute spatial clustering metrics for positive pixels.
    
    Args:
        coords: Nx2 array of coordinates
        positive_mask: Boolean mask of positive pixels
        max_distance_um: Maximum distance for clustering analysis
        
    Returns:
        Dictionary with clustering metrics
    """
    if not np.any(positive_mask):
        return {
            'cluster_count': 0.0,
            'mean_cluster_size': 0.0,
            'largest_cluster_size': 0.0,
            'clustering_coefficient': 0.0
        }
    
    positive_coords = coords[positive_mask]
    
    if len(positive_coords) < 2:
        return {
            'cluster_count': 1.0 if len(positive_coords) == 1 else 0.0,
            'mean_cluster_size': float(len(positive_coords)),
            'largest_cluster_size': float(len(positive_coords)),
            'clustering_coefficient': 0.0
        }
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(positive_coords))
    
    # Create adjacency matrix
    adjacency = distances <= max_distance_um
    np.fill_diagonal(adjacency, False)
    
    # Find connected components (clusters)
    from scipy.sparse.csgraph import connected_components
    n_clusters, labels = connected_components(adjacency)
    
    # Compute cluster statistics
    cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
    
    # Clustering coefficient (fraction of possible edges that exist)
    n_edges = np.sum(adjacency) / 2  # Undirected graph
    n_possible = len(positive_coords) * (len(positive_coords) - 1) / 2
    clustering_coeff = n_edges / n_possible if n_possible > 0 else 0
    
    return {
        'cluster_count': float(n_clusters),
        'mean_cluster_size': float(np.mean(cluster_sizes)),
        'largest_cluster_size': float(np.max(cluster_sizes)),
        'clustering_coefficient': float(clustering_coeff)
    }