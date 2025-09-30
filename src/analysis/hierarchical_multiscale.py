"""
Hierarchical Multi-Scale Analysis

Creates true parent-child relationships across scales rather than
independent clustering at each scale. Implements graph coarsening
and hierarchical clustering for biologically meaningful multi-scale analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
import warnings


def build_multiscale_hierarchy(
    fine_features: np.ndarray,
    fine_coords: np.ndarray,
    fine_labels: np.ndarray,
    coarsening_factor: float = 2.0,
    n_scales: int = 3
) -> Dict:
    """
    Build hierarchical clustering from fine to coarse scales.
    
    Args:
        fine_features: N x P feature matrix at finest scale
        fine_coords: N x 2 spatial coordinates at finest scale  
        fine_labels: Cluster labels at finest scale
        coarsening_factor: How much to coarsen at each level
        n_scales: Number of hierarchical levels
        
    Returns:
        Dictionary with hierarchical clustering results
    """
    hierarchy = {
        'scale_0': {
            'features': fine_features,
            'coords': fine_coords,
            'labels': fine_labels,
            'scale_um': 10.0  # Finest scale
        }
    }
    
    current_features = fine_features
    current_coords = fine_coords
    current_labels = fine_labels
    
    for scale_idx in range(1, n_scales):
        # Coarsen the graph
        coarse_features, coarse_coords, coarse_labels, parent_map = coarsen_graph(
            current_features, current_coords, current_labels, coarsening_factor
        )
        
        hierarchy[f'scale_{scale_idx}'] = {
            'features': coarse_features,
            'coords': coarse_coords,
            'labels': coarse_labels,
            'parent_map': parent_map,
            'scale_um': 10.0 * (coarsening_factor ** scale_idx)
        }
        
        current_features = coarse_features
        current_coords = coarse_coords
        current_labels = coarse_labels
    
    return hierarchy


def coarsen_graph(
    features: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    coarsening_factor: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Coarsen spatial graph by merging nearby nodes.
    
    Args:
        features: N x P feature matrix
        coords: N x 2 spatial coordinates
        labels: Current cluster labels
        coarsening_factor: Factor by which to reduce nodes
        
    Returns:
        Tuple of (coarse_features, coarse_coords, coarse_labels, parent_mapping)
    """
    n_nodes = features.shape[0]
    
    # Handle empty input
    if n_nodes == 0:
        return np.array([]), np.array([]).reshape(0, 2), np.array([]), {}
    
    n_coarse = max(1, int(n_nodes / coarsening_factor))
    
    # Use spatial clustering to determine coarse nodes
    from sklearn.cluster import KMeans
    
    # Cluster based on spatial coordinates primarily
    spatial_clusterer = KMeans(n_clusters=n_coarse, random_state=42, n_init=10)
    coarse_assignments = spatial_clusterer.fit_predict(coords)
    
    # Aggregate features and coordinates for each coarse node
    coarse_features_list = []
    coarse_coords_list = []
    coarse_labels_list = []
    parent_map = {}
    
    for coarse_id in range(n_coarse):
        mask = coarse_assignments == coarse_id
        if not np.any(mask):
            continue
            
        # Average features
        coarse_feat = np.mean(features[mask], axis=0)
        coarse_features_list.append(coarse_feat)
        
        # Centroid of coordinates
        coarse_coord = np.mean(coords[mask], axis=0)
        coarse_coords_list.append(coarse_coord)
        
        # Majority vote for labels
        fine_labels_in_coarse = labels[mask]
        unique_labels, counts = np.unique(fine_labels_in_coarse, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        coarse_labels_list.append(majority_label)
        
        # Track parent-child relationships
        fine_indices = np.where(mask)[0]
        for fine_idx in fine_indices:
            parent_map[fine_idx] = coarse_id
    
    coarse_features = np.array(coarse_features_list)
    coarse_coords = np.array(coarse_coords_list)
    coarse_labels = np.array(coarse_labels_list)
    
    return coarse_features, coarse_coords, coarse_labels, parent_map


def add_neighbor_composition_features(
    features: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    radius_um: float = 20.0
) -> np.ndarray:
    """
    Add features based on neighborhood composition.
    
    Args:
        features: N x P original feature matrix
        coords: N x 2 spatial coordinates
        labels: Cluster labels
        radius_um: Radius for neighborhood definition
        
    Returns:
        Enhanced feature matrix with neighborhood composition
    """
    n_samples = features.shape[0]
    n_clusters = len(np.unique(labels[labels >= 0]))
    
    # Build spatial index
    tree = cKDTree(coords)
    
    # Calculate neighborhood composition for each point
    composition_features = np.zeros((n_samples, n_clusters))
    
    for i in range(n_samples):
        # Find neighbors within radius
        neighbor_indices = tree.query_ball_point(coords[i], radius_um)
        
        if len(neighbor_indices) > 1:  # Exclude self
            neighbor_labels = labels[neighbor_indices]
            neighbor_labels = neighbor_labels[neighbor_labels >= 0]  # Exclude noise
            
            # Count cluster occurrences
            for cluster_id in range(n_clusters):
                composition_features[i, cluster_id] = np.sum(neighbor_labels == cluster_id)
            
            # Normalize by total neighbors
            total = composition_features[i].sum()
            if total > 0:
                composition_features[i] /= total
    
    # Combine original features with composition
    enhanced_features = np.hstack([features, composition_features])
    
    return enhanced_features


def compute_texture_features(
    marker_intensities: Dict[str, np.ndarray],
    coords: np.ndarray,
    bin_size_um: float = 20.0
) -> np.ndarray:
    """
    Compute texture features for coarse-scale analysis.
    
    Args:
        marker_intensities: Dictionary of protein marker intensities
        coords: Spatial coordinates
        bin_size_um: Bin size for texture computation
        
    Returns:
        Texture feature matrix
    """
    from skimage.feature import graycomatrix, graycoprops
    
    # Create spatial grid
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    x_bins = np.arange(x_min, x_max + bin_size_um, bin_size_um)
    y_bins = np.arange(y_min, y_max + bin_size_um, bin_size_um)
    
    texture_features_all = []
    
    for protein_name, intensities in marker_intensities.items():
        # Bin intensities onto grid
        x_indices = np.digitize(coords[:, 0], x_bins) - 1
        y_indices = np.digitize(coords[:, 1], y_bins) - 1
        
        grid = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
        
        valid_mask = (x_indices >= 0) & (x_indices < len(x_bins) - 1) & \
                     (y_indices >= 0) & (y_indices < len(y_bins) - 1)
        
        for i in np.where(valid_mask)[0]:
            grid[y_indices[i], x_indices[i]] += intensities[i]
        
        # Normalize to uint8 for texture analysis
        if grid.max() > 0:
            grid_norm = (grid / grid.max() * 255).astype(np.uint8)
        else:
            grid_norm = grid.astype(np.uint8)
        
        # Compute GLCM texture features
        try:
            glcm = graycomatrix(grid_norm, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            
            # Extract Haralick features
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            
            texture_features_all.extend([contrast, homogeneity, energy, correlation])
        except:
            # If texture computation fails, use zeros
            texture_features_all.extend([0, 0, 0, 0])
    
    return np.array(texture_features_all)


def validate_hierarchy(
    hierarchy: Dict,
    min_coherence: float = 0.5
) -> Dict:
    """
    Validate that hierarchical clustering maintains biological coherence.
    
    Args:
        hierarchy: Hierarchical clustering results
        min_coherence: Minimum acceptable coherence score
        
    Returns:
        Validation results
    """
    validation_results = {}
    
    # Check parent-child coherence
    for scale_idx in range(1, len(hierarchy)):
        parent_scale = f'scale_{scale_idx - 1}'
        child_scale = f'scale_{scale_idx}'
        
        if child_scale in hierarchy and 'parent_map' in hierarchy[child_scale]:
            parent_labels = hierarchy[parent_scale]['labels']
            child_labels = hierarchy[child_scale]['labels']
            parent_map = hierarchy[child_scale]['parent_map']
            
            # Check label consistency
            coherence_scores = []
            for child_idx, parent_idx in parent_map.items():
                if child_idx < len(parent_labels) and parent_idx < len(child_labels):
                    # Check if parent and child have related labels
                    coherence = 1.0 if parent_labels[child_idx] == child_labels[parent_idx] else 0.0
                    coherence_scores.append(coherence)
            
            mean_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
            
            validation_results[f'{parent_scale}_to_{child_scale}'] = {
                'coherence': mean_coherence,
                'valid': mean_coherence >= min_coherence
            }
    
    # Check scale progression (clusters should generally decrease)
    n_clusters_per_scale = []
    for scale_key in sorted(hierarchy.keys()):
        labels = hierarchy[scale_key]['labels']
        n_clusters = len(np.unique(labels[labels >= 0]))
        n_clusters_per_scale.append(n_clusters)
    
    validation_results['scale_progression'] = {
        'n_clusters_per_scale': n_clusters_per_scale,
        'monotonic_decrease': all(n_clusters_per_scale[i] >= n_clusters_per_scale[i+1] 
                                  for i in range(len(n_clusters_per_scale)-1))
    }
    
    return validation_results