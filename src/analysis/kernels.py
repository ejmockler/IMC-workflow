"""
Spatial Kernel System for Advanced Neighborhood Analysis
Implements BANKSY-style and other spatial kernels for IMC data
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class KernelResult:
    """Encapsulates kernel computation results"""
    weights: np.ndarray
    neighbor_indices: np.ndarray
    effective_radius: float
    n_neighbors: int
    
    def to_sparse(self, n_total: int) -> csr_matrix:
        """Convert to sparse matrix representation"""
        row_ind = np.zeros(self.n_neighbors)
        col_ind = self.neighbor_indices
        return csr_matrix((self.weights, (row_ind, col_ind)), 
                         shape=(1, n_total))


class SpatialKernel(ABC):
    """Abstract base for spatial kernel strategies"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @abstractmethod
    def compute_weights(self, 
                       coords: np.ndarray, 
                       center_idx: int,
                       tree: Optional[cKDTree] = None) -> KernelResult:
        """
        Compute kernel weights for neighbors around a center point
        
        Args:
            coords: All coordinates (n_points, 2)
            center_idx: Index of center point
            tree: Pre-built KDTree (optional, for efficiency)
            
        Returns:
            KernelResult with weights and neighbor indices
        """
        pass
    
    @abstractmethod
    def aggregate_features(self, 
                          features: np.ndarray,
                          kernel_result: KernelResult) -> np.ndarray:
        """
        Aggregate neighbor features using kernel weights
        
        Args:
            features: Feature matrix (n_points, n_features)
            kernel_result: Output from compute_weights
            
        Returns:
            Aggregated feature vector (n_features,)
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return kernel name for reporting"""
        pass
    
    def compute_all_weights(self, 
                           coords: np.ndarray,
                           n_jobs: int = 1) -> csr_matrix:
        """
        Compute weight matrix for all points (for batch processing)
        
        Returns sparse matrix (n_points, n_points) of weights
        """
        n_points = len(coords)
        tree = cKDTree(coords)
        
        # Collect all weights
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            result = self.compute_weights(coords, i, tree)
            rows.extend([i] * result.n_neighbors)
            cols.extend(result.neighbor_indices)
            data.extend(result.weights)
        
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))


class GaussianKernel(SpatialKernel):
    """
    BANKSY-style Gaussian kernel with radial decay
    Weight = exp(-d²/2σ²)
    """
    
    def __init__(self, sigma: float = 30.0, cutoff_radius: float = 100.0):
        """
        Args:
            sigma: Bandwidth parameter (μm)
            cutoff_radius: Maximum radius to consider (μm)
        """
        super().__init__()
        self.sigma = sigma
        self.cutoff_radius = cutoff_radius
        self.normalization = 1.0 / (2 * sigma * sigma)
    
    def name(self) -> str:
        return f"gaussian_σ{self.sigma}"
    
    def compute_weights(self, 
                       coords: np.ndarray, 
                       center_idx: int,
                       tree: Optional[cKDTree] = None) -> KernelResult:
        """Compute Gaussian weights for neighbors"""
        
        if tree is None:
            tree = cKDTree(coords)
        
        center = coords[center_idx]
        
        # Find neighbors within cutoff
        neighbor_indices = tree.query_ball_point(center, self.cutoff_radius)
        neighbor_indices = np.array([idx for idx in neighbor_indices 
                                    if idx != center_idx])
        
        if len(neighbor_indices) == 0:
            # No neighbors - return self with weight 1
            return KernelResult(
                weights=np.array([1.0]),
                neighbor_indices=np.array([center_idx]),
                effective_radius=0.0,
                n_neighbors=1
            )
        
        # Compute distances
        neighbor_coords = coords[neighbor_indices]
        distances = np.linalg.norm(neighbor_coords - center, axis=1)
        
        # Gaussian weights
        weights = np.exp(-distances * distances * self.normalization)
        
        # Filter out very small weights
        significant = weights > 0.01
        weights = weights[significant]
        neighbor_indices = neighbor_indices[significant]
        
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        
        return KernelResult(
            weights=weights,
            neighbor_indices=neighbor_indices,
            effective_radius=np.max(distances[significant]) if np.any(significant) else 0,
            n_neighbors=len(neighbor_indices)
        )
    
    def aggregate_features(self, 
                          features: np.ndarray,
                          kernel_result: KernelResult) -> np.ndarray:
        """Weighted average of neighbor features"""
        neighbor_features = features[kernel_result.neighbor_indices]
        return np.dot(kernel_result.weights, neighbor_features)


class AdaptiveKernel(SpatialKernel):
    """
    Data-driven adaptive kernel that adjusts to local density
    Uses k-NN in sparse regions, fixed radius in dense regions
    """
    
    def __init__(self, 
                min_neighbors: int = 10,
                max_radius: float = 150.0,
                density_threshold: float = 0.01):
        """
        Args:
            min_neighbors: Minimum neighbors to include
            max_radius: Maximum search radius
            density_threshold: Points/μm² to switch strategies
        """
        super().__init__()
        self.min_neighbors = min_neighbors
        self.max_radius = max_radius
        self.density_threshold = density_threshold
    
    def name(self) -> str:
        return "adaptive"
    
    def compute_weights(self, 
                       coords: np.ndarray, 
                       center_idx: int,
                       tree: Optional[cKDTree] = None) -> KernelResult:
        """Adaptively compute weights based on local density"""
        
        if tree is None:
            tree = cKDTree(coords)
        
        center = coords[center_idx]
        
        # Estimate local density
        k_for_density = min(20, len(coords) - 1)
        distances_k, indices_k = tree.query(center, k=k_for_density + 1)
        distances_k = distances_k[1:]  # Exclude self
        indices_k = indices_k[1:]
        
        # Local density estimate (points per unit area)
        if len(distances_k) > 0:
            radius_k = distances_k[-1]
            area_k = np.pi * radius_k * radius_k
            density = k_for_density / area_k if area_k > 0 else 0
        else:
            density = 0
        
        # Choose strategy based on density
        if density < self.density_threshold:
            # Sparse region: use k-NN
            k = min(self.min_neighbors, len(coords) - 1)
            distances, indices = tree.query(center, k=k + 1)
            neighbor_distances = distances[1:]
            neighbor_indices = indices[1:]
            
            # Adaptive radius based on k-th neighbor
            adaptive_radius = min(neighbor_distances[-1] * 1.2, self.max_radius)
        else:
            # Dense region: use fixed radius
            adaptive_radius = self.max_radius * (self.density_threshold / density) ** 0.5
            adaptive_radius = np.clip(adaptive_radius, 30, self.max_radius)
            
            indices = tree.query_ball_point(center, adaptive_radius)
            neighbor_indices = np.array([idx for idx in indices if idx != center_idx])
            
            if len(neighbor_indices) > 0:
                neighbor_coords = coords[neighbor_indices]
                neighbor_distances = np.linalg.norm(neighbor_coords - center, axis=1)
            else:
                neighbor_distances = np.array([])
        
        if len(neighbor_indices) == 0:
            return KernelResult(
                weights=np.array([1.0]),
                neighbor_indices=np.array([center_idx]),
                effective_radius=0.0,
                n_neighbors=1
            )
        
        # Compute adaptive weights (inverse distance with Gaussian modulation)
        sigma_adaptive = adaptive_radius / 3  # Rule of thumb
        weights = np.exp(-neighbor_distances**2 / (2 * sigma_adaptive**2))
        
        # Additional weight boost for very close neighbors
        close_mask = neighbor_distances < (adaptive_radius * 0.2)
        weights[close_mask] *= 2.0
        
        # Normalize
        weights = weights / weights.sum()
        
        return KernelResult(
            weights=weights,
            neighbor_indices=neighbor_indices,
            effective_radius=adaptive_radius,
            n_neighbors=len(neighbor_indices)
        )
    
    def aggregate_features(self, 
                          features: np.ndarray,
                          kernel_result: KernelResult) -> np.ndarray:
        """Weighted average with adaptive weights"""
        neighbor_features = features[kernel_result.neighbor_indices]
        return np.dot(kernel_result.weights, neighbor_features)


class LaplacianKernel(SpatialKernel):
    """
    Edge-aware kernel with exponential decay
    Weight = exp(-|d|/σ)
    Good for preserving tissue boundaries
    """
    
    def __init__(self, sigma: float = 40.0, cutoff_radius: float = 120.0):
        super().__init__()
        self.sigma = sigma
        self.cutoff_radius = cutoff_radius
    
    def name(self) -> str:
        return f"laplacian_σ{self.sigma}"
    
    def compute_weights(self, 
                       coords: np.ndarray, 
                       center_idx: int,
                       tree: Optional[cKDTree] = None) -> KernelResult:
        """Compute Laplacian weights"""
        
        if tree is None:
            tree = cKDTree(coords)
        
        center = coords[center_idx]
        
        # Find neighbors
        neighbor_indices = tree.query_ball_point(center, self.cutoff_radius)
        neighbor_indices = np.array([idx for idx in neighbor_indices 
                                    if idx != center_idx])
        
        if len(neighbor_indices) == 0:
            return KernelResult(
                weights=np.array([1.0]),
                neighbor_indices=np.array([center_idx]),
                effective_radius=0.0,
                n_neighbors=1
            )
        
        # Compute distances and weights
        neighbor_coords = coords[neighbor_indices]
        distances = np.linalg.norm(neighbor_coords - center, axis=1)
        
        # Laplacian kernel
        weights = np.exp(-distances / self.sigma)
        
        # Sharper cutoff than Gaussian
        significant = weights > 0.05
        weights = weights[significant]
        neighbor_indices = neighbor_indices[significant]
        
        # Normalize
        weights = weights / weights.sum()
        
        return KernelResult(
            weights=weights,
            neighbor_indices=neighbor_indices,
            effective_radius=self.sigma * 3,  # Effective radius
            n_neighbors=len(neighbor_indices)
        )
    
    def aggregate_features(self, 
                          features: np.ndarray,
                          kernel_result: KernelResult) -> np.ndarray:
        """Edge-preserving aggregation"""
        neighbor_features = features[kernel_result.neighbor_indices]
        return np.dot(kernel_result.weights, neighbor_features)


class KernelFactory:
    """Factory for creating spatial kernels from configuration"""
    
    @staticmethod
    def create_kernel(kernel_type: str, config: Dict[str, Any]) -> SpatialKernel:
        """
        Create kernel instance from type and config
        
        Args:
            kernel_type: 'gaussian', 'adaptive', or 'laplacian'
            config: Kernel-specific configuration
            
        Returns:
            SpatialKernel instance
        """
        if kernel_type == 'gaussian':
            return GaussianKernel(
                sigma=config.get('sigma', 30.0),
                cutoff_radius=config.get('cutoff_radius', 100.0)
            )
        elif kernel_type == 'adaptive':
            return AdaptiveKernel(
                min_neighbors=config.get('min_neighbors', 10),
                max_radius=config.get('max_radius', 150.0),
                density_threshold=config.get('density_threshold', 0.01)
            )
        elif kernel_type == 'laplacian':
            return LaplacianKernel(
                sigma=config.get('sigma', 40.0),
                cutoff_radius=config.get('cutoff_radius', 120.0)
            )
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")


def compute_augmented_features(coords: np.ndarray,
                              values: np.ndarray,
                              kernel: SpatialKernel,
                              lambda_param: float = 0.5) -> np.ndarray:
    """
    Augment features with neighborhood expression (BANKSY-style)
    
    Args:
        coords: Spatial coordinates (n_points, 2)
        values: Expression values (n_points, n_features)
        kernel: Spatial kernel to use
        lambda_param: Mixing parameter [0,1]
                     0 = only original features
                     1 = only neighborhood features
    
    Returns:
        Augmented features (n_points, n_features * 2) if lambda in (0,1)
        or (n_points, n_features) if lambda is 0 or 1
    """
    n_points = len(coords)
    n_features = values.shape[1]
    
    # Build tree once for efficiency
    tree = cKDTree(coords)
    
    # Compute neighborhood features for all points
    neighborhood_features = np.zeros((n_points, n_features))
    
    for i in range(n_points):
        kernel_result = kernel.compute_weights(coords, i, tree)
        neighborhood_features[i] = kernel.aggregate_features(values, kernel_result)
    
    # Mix features based on lambda
    if lambda_param == 0:
        return values
    elif lambda_param == 1:
        return neighborhood_features
    else:
        # Concatenate weighted features
        weighted_original = values * (1 - lambda_param)
        weighted_neighborhood = neighborhood_features * lambda_param
        return np.hstack([weighted_original, weighted_neighborhood])


def benchmark_kernels(coords: np.ndarray,
                     values: np.ndarray,
                     kernels: Dict[str, SpatialKernel]) -> Dict[str, Dict]:
    """
    Benchmark different kernels on the same data
    
    Returns:
        Dictionary with timing and quality metrics for each kernel
    """
    import time
    from src.analysis.validation import SilhouetteValidator
    from sklearn.cluster import KMeans
    
    results = {}
    
    for kernel_name, kernel in kernels.items():
        start_time = time.time()
        
        # Compute augmented features
        augmented = compute_augmented_features(coords, values, kernel, lambda_param=0.5)
        
        # Cluster
        n_clusters = min(30, len(coords) // 100)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        labels = kmeans.fit_predict(augmented)
        
        # Validate
        validator = SilhouetteValidator()
        val_result = validator.validate(augmented, labels)
        
        elapsed = time.time() - start_time
        
        results[kernel_name] = {
            'time_seconds': elapsed,
            'silhouette_score': val_result.score,
            'n_features': augmented.shape[1],
            'kernel_type': kernel.name()
        }
    
    return results