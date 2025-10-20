"""
Spatial Permutation Testing for IMC Data

Generates spatially-aware null distributions that preserve spatial autocorrelation
structure while breaking associations with outcomes. Implements multiple methods
based on data topology and tissue geometry.

Critical for valid hypothesis testing in the presence of spatial dependence.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Iterator, Union, Callable
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

try:
    import libpysal
    from libpysal import weights
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False
    warnings.warn("libpysal not available. Some spatial methods will be limited.")


@dataclass
class PermutationConfig:
    """Configuration for spatial permutation testing."""
    method: str = 'moran_spectral'  # 'toroidal', 'moran_spectral', 'graph_swap'
    n_permutations: int = 999
    preserve_marginals: bool = True
    random_state: int = 42
    cache_nulls: bool = True
    parallel: bool = True
    n_jobs: int = -1


class SpatialPermutation:
    """
    Generate spatially-aware null distributions for hypothesis testing.
    
    Preserves spatial autocorrelation while breaking associations with outcomes.
    """
    
    def __init__(self, config: PermutationConfig):
        self.config = config
        self.cached_nulls = {} if config.cache_nulls else None
        self.rng = np.random.Generator(np.random.PCG64(config.random_state))
        
    def generate_spatial_nulls(self,
                             observed: np.ndarray,
                             coords: np.ndarray,
                             mask: Optional[np.ndarray] = None,
                             method: Optional[str] = None,
                             n_perms: Optional[int] = None,
                             spatial_weights: Optional[np.ndarray] = None) -> Iterator[np.ndarray]:
        """
        Generate spatially-aware null distributions.
        
        Args:
            observed: Observed data (2D field or 1D values at coords)
            coords: Spatial coordinates (Nx2 for points, or implicit grid for 2D)
            mask: Tissue mask (True where tissue exists)
            method: Override config method ('toroidal', 'moran_spectral', 'graph_swap')
            n_perms: Override config n_permutations
            spatial_weights: Pre-computed spatial weights matrix (saves computation)
            
        Yields:
            Permuted null datasets preserving spatial structure
        """
        method = method or self.config.method
        n_perms = n_perms or self.config.n_permutations
        
        # Check cache
        cache_key = self._get_cache_key(observed, coords, mask, method)
        if self.cached_nulls and cache_key in self.cached_nulls:
            yield from self.cached_nulls[cache_key]
            return
        
        # Generate nulls based on method
        nulls = []
        
        if method == 'toroidal':
            null_generator = self._toroidal_shift(observed, mask, n_perms)
        elif method == 'moran_spectral':
            null_generator = self._moran_spectral_randomization(
                observed, coords, spatial_weights, n_perms
            )
        elif method == 'graph_swap':
            null_generator = self._graph_constrained_swaps(
                observed, coords, spatial_weights, n_perms
            )
        else:
            raise ValueError(f"Unknown permutation method: {method}")
        
        # Generate and optionally cache
        for null in null_generator:
            if self.config.cache_nulls:
                nulls.append(null.copy())
            yield null
        
        # Store in cache
        if self.config.cache_nulls:
            self.cached_nulls[cache_key] = nulls
    
    def _toroidal_shift(self,
                       field: np.ndarray,
                       mask: Optional[np.ndarray],
                       n_perms: int) -> Iterator[np.ndarray]:
        """
        Toroidal (circular) shifting for grid-like data.
        
        Preserves local autocorrelation up to wrap-around.
        Best for rectangular tissue with minimal holes.
        """
        if field.ndim != 2:
            raise ValueError("Toroidal shift requires 2D field")
        
        h, w = field.shape
        
        for _ in range(n_perms):
            # Random shifts in both dimensions
            shift_y = self.rng.integers(0, h)
            shift_x = self.rng.integers(0, w)
            
            # Apply toroidal shift
            shifted = np.roll(field, shift_y, axis=0)
            shifted = np.roll(shifted, shift_x, axis=1)
            
            # Re-apply mask to preserve tissue boundaries
            if mask is not None:
                # Preserve marginal intensity distribution within tissue
                if self.config.preserve_marginals:
                    tissue_values = field[mask]
                    shifted_tissue = shifted[mask]
                    
                    # Match marginal distribution via rank matching
                    ranks = stats.rankdata(shifted_tissue, method='ordinal')
                    sorted_original = np.sort(tissue_values)
                    # Use ranks as indices (ranks are 1-based, so subtract 1)
                    rank_indices = np.clip(ranks - 1, 0, len(sorted_original) - 1)
                    shifted[mask] = sorted_original[rank_indices]
                else:
                    shifted[~mask] = 0
            
            yield shifted
    
    def _moran_spectral_randomization(self,
                                     values: np.ndarray,
                                     coords: np.ndarray,
                                     spatial_weights: Optional[np.ndarray],
                                     n_perms: int) -> Iterator[np.ndarray]:
        """
        Moran Spectral Randomization (MSR) for irregular geometries.
        
        Preserves global autocorrelation spectrum via eigendecomposition
        of spatial weights matrix.
        """
        n_points = len(values) if values.ndim == 1 else values.size
        
        # Build spatial weights if not provided
        if spatial_weights is None:
            spatial_weights = self._build_spatial_weights(coords, values.shape)
        
        # Handle 2D fields by flattening
        original_shape = values.shape
        if values.ndim == 2:
            values_flat = values.ravel()
        else:
            values_flat = values
        
        # Compute Moran operator (I - 11'/n)W(I - 11'/n)
        n = len(values_flat)
        I = np.eye(n)
        ones = np.ones((n, 1))
        centering = I - ones @ ones.T / n
        moran_operator = centering @ spatial_weights @ centering
        
        # Eigendecomposition
        try:
            eigenvals, eigenvecs = np.linalg.eigh(moran_operator)
            
            # Sort by eigenvalue magnitude
            idx = np.abs(eigenvals).argsort()[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Keep significant eigenvectors (explaining 95% of variation)
            cum_var = np.cumsum(np.abs(eigenvals))
            n_keep = np.searchsorted(cum_var, 0.95 * cum_var[-1]) + 1
            eigenvecs_kept = eigenvecs[:, :n_keep]
            
        except np.linalg.LinAlgError:
            warnings.warn("Eigendecomposition failed. Falling back to simple permutation.")
            # Fallback to simple permutation
            for _ in range(n_perms):
                yield self.rng.permutation(values_flat).reshape(original_shape)
            return
        
        # Generate randomized fields
        for _ in range(n_perms):
            # Random coefficients with same marginal distribution
            random_coeffs = self.rng.standard_normal(n_keep)
            
            # Project to eigenbasis
            randomized = eigenvecs_kept @ random_coeffs
            
            # Match marginal distribution
            if self.config.preserve_marginals:
                ranks = stats.rankdata(randomized)
                sorted_original = np.sort(values_flat)
                randomized = sorted_original[np.searchsorted(
                    np.arange(len(sorted_original)), ranks - 1
                )]
            
            # Reshape if needed
            if values.ndim == 2:
                randomized = randomized.reshape(original_shape)
            
            yield randomized
    
    def _graph_constrained_swaps(self,
                                labels: np.ndarray,
                                coords: np.ndarray,
                                spatial_weights: Optional[np.ndarray],
                                n_perms: int) -> Iterator[np.ndarray]:
        """
        Graph-constrained label swapping for categorical data.
        
        Performs Markov chain of swaps within k-hop neighborhoods
        to preserve local structure while randomizing global patterns.
        """
        if spatial_weights is None:
            spatial_weights = self._build_spatial_weights(coords, labels.shape)
        
        n = len(labels) if labels.ndim == 1 else labels.size
        labels_flat = labels.ravel() if labels.ndim > 1 else labels.copy()
        
        # Build adjacency from weights
        adjacency = (spatial_weights > 0).astype(int)
        
        # Number of swaps per permutation (2n ensures good mixing)
        n_swaps = 2 * n
        
        for _ in range(n_perms):
            perm_labels = labels_flat.copy()
            
            # Markov chain of constrained swaps
            for _ in range(n_swaps):
                # Pick random node
                i = self.rng.integers(n)
                
                # Find neighbors within 2-hop distance
                neighbors_1hop = np.where(adjacency[i] > 0)[0]
                if len(neighbors_1hop) == 0:
                    continue
                
                neighbors_2hop = set(neighbors_1hop)
                for n1 in neighbors_1hop:
                    neighbors_2hop.update(np.where(adjacency[n1] > 0)[0])
                neighbors_2hop.discard(i)
                
                if len(neighbors_2hop) == 0:
                    continue
                
                # Swap with random neighbor
                j = self.rng.choice(list(neighbors_2hop))
                perm_labels[i], perm_labels[j] = perm_labels[j], perm_labels[i]
            
            # Reshape if needed
            if labels.ndim > 1:
                perm_labels = perm_labels.reshape(labels.shape)
            
            yield perm_labels
    
    def _build_spatial_weights(self,
                              coords: np.ndarray,
                              data_shape: Tuple) -> np.ndarray:
        """Build spatial weights matrix from coordinates."""
        
        if coords.ndim == 2 and coords.shape[1] == 2:
            # Point pattern - use k-nearest neighbors
            n_points = len(coords)
            k = min(15, n_points - 1)  # Adaptive k
            
            tree = cKDTree(coords)
            weights = np.zeros((n_points, n_points))
            
            for i in range(n_points):
                distances, indices = tree.query(coords[i], k=k+1)
                
                # Skip self (first neighbor)
                for dist, idx in zip(distances[1:], indices[1:]):
                    if dist > 0:
                        weights[i, idx] = np.exp(-dist / np.median(distances[1:]))
            
            # Symmetrize
            weights = (weights + weights.T) / 2
            
        else:
            # Grid data - use rook contiguity with vectorized construction
            if len(data_shape) != 2:
                raise ValueError("Expected 2D grid data or point coordinates")
            
            h, w = data_shape
            n = h * w
            
            # Vectorized construction of sparse adjacency matrix
            from scipy import sparse
            
            # Create grid indices
            i_indices, j_indices = np.meshgrid(range(h), range(w), indexing='ij')
            flat_indices = i_indices * w + j_indices
            
            row_indices = []
            col_indices = []
            
            # Vectorized neighbor finding
            # Up neighbors
            valid_up = i_indices > 0
            if np.any(valid_up):
                row_indices.extend(flat_indices[valid_up])
                col_indices.extend(flat_indices[valid_up] - w)
            
            # Down neighbors  
            valid_down = i_indices < h - 1
            if np.any(valid_down):
                row_indices.extend(flat_indices[valid_down])
                col_indices.extend(flat_indices[valid_down] + w)
            
            # Left neighbors
            valid_left = j_indices > 0
            if np.any(valid_left):
                row_indices.extend(flat_indices[valid_left])
                col_indices.extend(flat_indices[valid_left] - 1)
            
            # Right neighbors
            valid_right = j_indices < w - 1
            if np.any(valid_right):
                row_indices.extend(flat_indices[valid_right])
                col_indices.extend(flat_indices[valid_right] + 1)
            
            # Create sparse matrix with all connections
            weights_sparse = sparse.csr_matrix(
                (np.ones(len(row_indices)), (row_indices, col_indices)),
                shape=(n, n)
            )
            
            # Row-normalize efficiently
            weights = weights_sparse.toarray()
            row_sums = weights.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            weights = weights / row_sums
        
        return weights
    
    def _get_cache_key(self, observed, coords, mask, method):
        """Generate cache key for null distributions using content hashing."""
        import hashlib
        
        # Hash actual data content, not just shapes
        hasher = hashlib.sha256()
        hasher.update(method.encode())
        hasher.update(str(observed.shape).encode())
        hasher.update(str(observed.dtype).encode())
        
        # Hash sample of observed data (for large arrays)
        if observed.size > 1000:
            # Sample deterministically for consistent caching
            sample_idx = np.linspace(0, observed.size-1, 1000, dtype=int)
            sample_data = observed.flat[sample_idx]
        else:
            sample_data = observed.ravel()
        hasher.update(sample_data.tobytes())
        
        if coords is not None:
            hasher.update(str(coords.shape).encode())
            # Hash coordinate bounds for spatial structure
            hasher.update(coords.min(axis=0).tobytes())
            hasher.update(coords.max(axis=0).tobytes())
        
        if mask is not None:
            hasher.update(str(mask.sum()).encode())
            hasher.update(str(mask.shape).encode())
        
        return hasher.hexdigest()[:16]  # First 16 chars for shorter key


def compute_spatial_pvalue(observed_stat: float,
                          null_stats: np.ndarray,
                          alternative: str = 'two-sided') -> float:
    """
    Compute p-value from permutation null distribution.
    
    Args:
        observed_stat: Observed test statistic
        null_stats: Array of null test statistics from permutations
        alternative: 'two-sided', 'greater', or 'less'
        
    Returns:
        Permutation p-value
    """
    n_perms = len(null_stats)
    
    if alternative == 'greater':
        # P(null >= observed)
        n_extreme = np.sum(null_stats >= observed_stat)
    elif alternative == 'less':
        # P(null <= observed)
        n_extreme = np.sum(null_stats <= observed_stat)
    else:  # two-sided
        # P(|null| >= |observed|)
        n_extreme = np.sum(np.abs(null_stats) >= np.abs(observed_stat))
    
    # Add 1 to numerator and denominator for conservative estimate
    p_value = (n_extreme + 1) / (n_perms + 1)
    
    return float(p_value)


def freedman_lane_permutation(X: np.ndarray,
                             y: np.ndarray,
                             coords: np.ndarray,
                             permuter: SpatialPermutation,
                             n_perms: int = 999) -> Tuple[np.ndarray, np.ndarray]:
    """
    Freedman-Lane permutation for regression with spatial data.
    
    Permutes residuals under reduced model while preserving spatial structure.
    
    Args:
        X: Design matrix (n_samples x n_features)
        y: Response variable
        coords: Spatial coordinates
        permuter: SpatialPermutation instance
        n_perms: Number of permutations
        
    Returns:
        Tuple of (coefficients, p_values)
    """
    from sklearn.linear_model import LinearRegression
    
    n_samples, n_features = X.shape
    
    # Fit full model
    full_model = LinearRegression()
    full_model.fit(X, y)
    observed_coefs = full_model.coef_
    
    # Initialize null distribution
    null_coefs = np.zeros((n_perms, n_features))
    
    # Reduced model (intercept only)
    reduced_model = LinearRegression()
    reduced_model.fit(np.ones((n_samples, 1)), y)
    residuals = y - reduced_model.predict(np.ones((n_samples, 1)))
    
    # Generate spatially permuted residuals
    null_generator = permuter.generate_spatial_nulls(
        residuals, coords, n_perms=n_perms
    )
    
    for i, perm_residuals in enumerate(null_generator):
        # Add permuted residuals to reduced model predictions
        y_perm = reduced_model.predict(np.ones((n_samples, 1))) + perm_residuals
        
        # Fit full model to permuted data
        perm_model = LinearRegression()
        perm_model.fit(X, y_perm)
        null_coefs[i] = perm_model.coef_
    
    # Compute p-values
    p_values = np.zeros(n_features)
    for j in range(n_features):
        p_values[j] = compute_spatial_pvalue(
            observed_coefs[j], null_coefs[:, j], alternative='two-sided'
        )
    
    return observed_coefs, p_values


def parallel_permutation_test(test_func: Callable,
                            observed_data: np.ndarray,
                            permuter: SpatialPermutation,
                            coords: np.ndarray,
                            n_perms: int = 999,
                            n_jobs: int = -1) -> Tuple[float, float]:
    """
    Parallel computation of permutation test.
    
    Args:
        test_func: Function computing test statistic
        observed_data: Observed data
        permuter: SpatialPermutation instance
        coords: Spatial coordinates
        n_perms: Number of permutations
        n_jobs: Number of parallel jobs (-1 for all cores)
        
    Returns:
        Tuple of (observed_statistic, p_value)
    """
    from joblib import Parallel, delayed
    
    # Compute observed statistic
    observed_stat = test_func(observed_data)
    
    # Generate null statistics in parallel
    def compute_null_stat(null_data):
        return test_func(null_data)
    
    # Generate all nulls (for parallel processing)
    null_generator = permuter.generate_spatial_nulls(
        observed_data, coords, n_perms=n_perms
    )
    null_data_list = list(null_generator)
    
    # Parallel computation
    null_stats = Parallel(n_jobs=n_jobs)(
        delayed(compute_null_stat)(null_data) for null_data in null_data_list
    )
    
    # Compute p-value
    p_value = compute_spatial_pvalue(observed_stat, np.array(null_stats))
    
    return observed_stat, p_value