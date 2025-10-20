"""
Spatial Statistics Utilities

Descriptive spatial statistics for n=2 pilot study.
No hypothesis testing - only effect sizes and descriptive measures.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict


def compute_spatial_correlation(
    field1: np.ndarray,
    field2: np.ndarray,
    method: str = 'pearson'
) -> float:
    """
    Compute spatial correlation between two protein fields.
    
    Simple descriptive statistic - no hypothesis testing.
    
    Args:
        field1: First protein field
        field2: Second protein field  
        method: Correlation method ('pearson' or 'spearman')
        
    Returns:
        Correlation coefficient (no p-value)
    """
    if field1.shape != field2.shape:
        raise ValueError("Fields must have same shape")
    
    # Flatten and remove NaN/inf
    flat1 = field1.ravel()
    flat2 = field2.ravel()
    valid_mask = np.isfinite(flat1) & np.isfinite(flat2)
    flat1 = flat1[valid_mask]
    flat2 = flat2[valid_mask]
    
    if len(flat1) < 10:
        return np.nan
    
    if method == 'pearson':
        corr = np.corrcoef(flat1, flat2)[0, 1]
    elif method == 'spearman':
        from scipy.stats import spearmanr
        corr, _ = spearmanr(flat1, flat2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(corr)


def compute_region_difference(
    field: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compare protein levels between regions (e.g., region 1 vs region 2).
    
    Returns descriptive statistics only - no hypothesis testing.
    
    Args:
        field: Protein field
        labels: Binary mask or labels
        
    Returns:
        Dictionary with effect size and descriptive stats
    """
    if field.shape != labels.shape:
        raise ValueError("Field and labels must have same shape")
    
    # Extract regions
    group1 = field[labels == 1]
    group0 = field[labels == 0]
    
    if len(group1) == 0 or len(group0) == 0:
        return {
            'mean_difference': np.nan,
            'cohens_d': np.nan,
            'mean_group1': np.nan,
            'mean_group0': np.nan,
            'std_group1': np.nan,
            'std_group0': np.nan
        }
    
    # Calculate descriptive statistics
    mean1 = np.mean(group1)
    mean0 = np.mean(group0)
    std1 = np.std(group1)
    std0 = np.std(group0)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((std1**2 + std0**2) / 2)
    cohens_d = (mean1 - mean0) / pooled_std if pooled_std > 0 else 0
    
    return {
        'mean_difference': float(mean1 - mean0),
        'cohens_d': float(cohens_d),
        'mean_group1': float(mean1),
        'mean_group0': float(mean0),
        'std_group1': float(std1),
        'std_group0': float(std0)
    }


def compute_ripleys_k(
    coords: np.ndarray,
    intensities: Optional[np.ndarray] = None,
    max_distance: float = 100,
    n_bins: int = 20,
    mask: Optional[np.ndarray] = None,
    edge_correction: str = 'translation',
    inhomogeneous: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Ripley's K function with proper edge correction.
    
    Memory-efficient implementation using cKDTree instead of full distance matrix.
    Includes edge correction and inhomogeneous extensions.
    
    Args:
        coords: Nx2 array of point coordinates
        intensities: Optional weights/marks for each point
        max_distance: Maximum distance to compute
        n_bins: Number of distance bins
        mask: Binary mask defining observation window (if None, uses bounding box)
        edge_correction: 'none', 'translation', 'border', or 'isotropic'
        inhomogeneous: If True, compute inhomogeneous K function
        
    Returns:
        Tuple of (distances, k_values)
    """
    n_points = len(coords)
    if n_points < 2:
        return np.array([]), np.array([])
    
    from scipy.spatial import cKDTree
    
    # Build spatial index for efficient queries
    tree = cKDTree(coords)
    
    # Define distance bins
    distances = np.linspace(0, max_distance, n_bins + 1)
    k_values = []
    
    # Compute observation window area
    if mask is not None:
        # Use actual tissue area from mask
        if mask.ndim == 2:
            # Mask is a 2D image - count pixels
            area = np.sum(mask)
            # Convert to coordinate units if needed
            x_range = coords[:, 0].max() - coords[:, 0].min()
            y_range = coords[:, 1].max() - coords[:, 1].min()
            area = area * (x_range * y_range) / (mask.shape[0] * mask.shape[1])
        else:
            # Mask is a 1D indicator for each point - use convex hull
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            area = hull.volume  # In 2D, volume is area
    else:
        # Use bounding box area (less accurate)
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        area = (x_max - x_min) * (y_max - y_min)
    
    # Estimate intensity function for inhomogeneous K
    if inhomogeneous:
        intensity = _estimate_intensity_function(coords, mask)
    else:
        intensity = np.ones(n_points) * n_points / area
    
    # Compute edge correction weights
    edge_weights = _compute_edge_correction_weights(
        coords, mask, max_distance, edge_correction
    )
    
    # Process each distance bin
    for d in distances[1:]:  # Skip 0
        k = 0.0
        
        # Use tree for efficient neighbor queries
        for i in range(n_points):
            # Find neighbors within distance d
            neighbors = tree.query_ball_point(coords[i], d)
            
            for j in neighbors:
                if i != j:  # Exclude self
                    # Edge correction weight
                    if edge_correction != 'none':
                        weight = edge_weights[i, j] if edge_weights.ndim == 2 else edge_weights[i]
                    else:
                        weight = 1.0
                    
                    # Inhomogeneous correction
                    if inhomogeneous:
                        weight = weight / (intensity[i] * intensity[j])
                    
                    # Add contribution
                    if intensities is not None:
                        k += intensities[i] * intensities[j] * weight
                    else:
                        k += weight
        
        # Normalize
        if intensities is not None:
            k = k / np.sum(intensities) ** 2
        else:
            k = k / n_points
        
        # Scale by area
        k = k * area
        
        k_values.append(k)
    
    return distances[1:], np.array(k_values)


def compute_ripleys_l(coords: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute L function (variance-stabilized Ripley's K).
    
    L(r) = sqrt(K(r) / π) - r
    
    Under CSR, L(r) = 0 for all r.
    """
    distances, k_values = compute_ripleys_k(coords, **kwargs)
    l_values = np.sqrt(k_values / np.pi) - distances
    return distances, l_values


def _estimate_intensity_function(coords: np.ndarray, 
                                mask: Optional[np.ndarray] = None,
                                bandwidth: Optional[float] = None) -> np.ndarray:
    """
    Estimate spatially-varying intensity function for inhomogeneous K.
    
    Uses kernel density estimation.
    """
    from scipy.stats import gaussian_kde
    
    n_points = len(coords)
    
    if bandwidth is None:
        # Scott's rule for bandwidth
        bandwidth = n_points ** (-1.0 / 6.0)
    
    # Kernel density estimation
    kde = gaussian_kde(coords.T, bw_method=bandwidth)
    intensities = kde.evaluate(coords.T)
    
    # Normalize to sum to n_points
    intensities = intensities * n_points / np.sum(intensities)
    
    return intensities


def _compute_edge_correction_weights(coords: np.ndarray,
                                    mask: Optional[np.ndarray],
                                    max_distance: float,
                                    method: str) -> np.ndarray:
    """
    Compute edge correction weights for Ripley's K.
    
    Returns weights for each point or point pair.
    """
    n_points = len(coords)
    
    if method == 'none':
        return np.ones(n_points)
    
    elif method == 'border':
        # Border correction - only use points far from edge
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Distance to nearest edge
        dist_to_edge = np.minimum.reduce([
            coords[:, 0] - x_min,
            x_max - coords[:, 0],
            coords[:, 1] - y_min,
            y_max - coords[:, 1]
        ])
        
        # Weight = 1 if point is at least max_distance from edge, 0 otherwise
        weights = (dist_to_edge >= max_distance).astype(float)
        return weights
    
    elif method == 'translation':
        # Translation correction - weight by proportion of circle in window
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # For each point pair, compute proportion of circle in window
        weights = np.ones((n_points, n_points))
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                # Distance between points
                d = np.linalg.norm(coords[i] - coords[j])
                
                if d > 0:
                    # Proportion of circle with radius d centered at i that's in window
                    # Simplified: use ratio of feasible translations
                    dx_feasible = min(coords[i, 0] - x_min, x_max - coords[i, 0])
                    dy_feasible = min(coords[i, 1] - y_min, y_max - coords[i, 1])
                    
                    weight = min(1.0, (dx_feasible * dy_feasible) / (d * d))
                    weights[i, j] = weights[j, i] = 1.0 / max(weight, 0.1)
        
        return weights
    
    elif method == 'isotropic':
        # Isotropic correction - Ripley's original
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        weights = np.ones((n_points, n_points))
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                d = np.linalg.norm(coords[i] - coords[j])
                
                if d > 0:
                    # Proportion of circle perimeter in window
                    # Approximate by checking intersection with boundaries
                    perimeter_weight = 1.0
                    
                    # Check each boundary
                    for boundary in [x_min, x_max]:
                        if abs(coords[i, 0] - boundary) < d:
                            perimeter_weight *= 0.75  # Approximate correction
                    
                    for boundary in [y_min, y_max]:
                        if abs(coords[i, 1] - boundary) < d:
                            perimeter_weight *= 0.75
                    
                    weights[i, j] = weights[j, i] = 1.0 / max(perimeter_weight, 0.25)
        
        return weights
    
    else:
        raise ValueError(f"Unknown edge correction method: {method}")


def spatial_bootstrap(
    field: np.ndarray,
    statistic_func,
    n_bootstrap: int = 100,
    block_size: int = 50
) -> Tuple[float, float]:
    """
    Spatial bootstrap for uncertainty estimation.
    
    Uses block bootstrap to preserve spatial structure.
    
    Args:
        field: 2D protein field
        statistic_func: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        block_size: Size of blocks for resampling
        
    Returns:
        Tuple of (mean, std) of bootstrap distribution
    """
    bootstrap_values = []
    h, w = field.shape
    
    for _ in range(n_bootstrap):
        # Create bootstrap sample by resampling blocks
        bootstrap_field = np.zeros_like(field)
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                # Sample a random block
                src_i = np.random.randint(0, max(1, h - block_size))
                src_j = np.random.randint(0, max(1, w - block_size))
                
                # Copy block
                bootstrap_field[i:i+block_size, j:j+block_size] = \
                    field[src_i:src_i+block_size, src_j:src_j+block_size]
        
        # Compute statistic
        stat_value = statistic_func(bootstrap_field)
        bootstrap_values.append(stat_value)
    
    bootstrap_values = np.array(bootstrap_values)
    return float(np.mean(bootstrap_values)), float(np.std(bootstrap_values))