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
    n_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Ripley's K function for point pattern analysis.
    
    Args:
        coords: Nx2 array of point coordinates
        intensities: Optional weights for each point
        max_distance: Maximum distance to compute
        n_bins: Number of distance bins
        
    Returns:
        Tuple of (distances, k_values)
    """
    n_points = len(coords)
    if n_points < 2:
        return np.array([]), np.array([])
    
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    distances_matrix = cdist(coords, coords)
    
    # Define distance bins
    distances = np.linspace(0, max_distance, n_bins + 1)
    k_values = []
    
    # Compute area (bounding box)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    area = (x_max - x_min) * (y_max - y_min)
    
    # Weight by intensities if provided
    if intensities is not None:
        weights = intensities / np.sum(intensities)
    else:
        weights = np.ones(n_points) / n_points
    
    for d in distances[1:]:  # Skip 0
        # Count pairs within distance d
        within_d = distances_matrix <= d
        np.fill_diagonal(within_d, False)  # Exclude self
        
        # Weighted count
        if intensities is not None:
            k = np.sum(within_d * weights[:, np.newaxis])
        else:
            k = np.sum(within_d) / n_points
        
        # Normalize by density
        k = k * area / n_points
        k_values.append(k)
    
    return distances[1:], np.array(k_values)


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