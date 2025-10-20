"""
Spatial Resampling Methods for IMC Data

Implements proper spatial bootstrap and resampling methods that preserve
spatial dependence structure. Replaces broken block bootstrap with
mask-aware, overlap-handling methods.

Critical for uncertainty quantification in spatial analyses.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Iterator, Callable
from dataclasses import dataclass
import warnings
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass
class ResamplingConfig:
    """Configuration for spatial resampling."""
    method: str = 'moving_block'  # 'moving_block', 'graph_block', 'spatial_subsample'
    block_size: Optional[float] = None  # Auto-estimate if None
    overlap: float = 0.5  # Block overlap fraction
    min_tissue_coverage: float = 0.5  # Minimum tissue in block to keep
    n_bootstrap: int = 100
    random_state: int = 42
    adaptive_block_size: bool = True
    blend_method: str = 'average'  # 'average', 'poisson', 'feather'


class SpatialResampling:
    """
    Spatial resampling methods preserving spatial dependence.
    
    Replaces broken block bootstrap with proper implementations.
    """
    
    def __init__(self, config: ResamplingConfig):
        self.config = config
        np.random.seed(config.random_state)
        self.estimated_block_sizes = {}
        
    def moving_block_bootstrap(self,
                              field: np.ndarray,
                              mask: Optional[np.ndarray] = None,
                              block_size: Optional[int] = None,
                              n_samples: Optional[int] = None) -> Iterator[np.ndarray]:
        """
        2D Moving Block Bootstrap with overlap and mask awareness.
        
        Args:
            field: 2D field to resample
            mask: Tissue mask (True where tissue exists)
            block_size: Size of blocks (estimated if None)
            n_samples: Number of bootstrap samples (default from config)
            
        Yields:
            Bootstrap samples preserving spatial structure
        """
        if field.ndim != 2:
            raise ValueError("Moving block bootstrap requires 2D field")
        
        h, w = field.shape
        n_samples = n_samples or self.config.n_bootstrap
        
        # Estimate block size if needed
        if block_size is None:
            if self.config.adaptive_block_size:
                block_size = self._estimate_block_size(field, mask)
            else:
                block_size = self.config.block_size or max(h, w) // 10
        
        # Ensure block size is reasonable
        block_size = min(block_size, min(h, w) // 2)
        block_size = max(block_size, 5)  # Minimum viable block
        
        # Calculate overlap
        overlap_pixels = int(block_size * self.config.overlap)
        step_size = block_size - overlap_pixels
        
        for _ in range(n_samples):
            # Initialize bootstrap sample
            bootstrap_field = np.zeros_like(field)
            weight_field = np.zeros_like(field)
            
            # Randomly sample block starting positions
            n_blocks_h = (h - overlap_pixels) // step_size + 1
            n_blocks_w = (w - overlap_pixels) // step_size + 1
            
            for _ in range(n_blocks_h * n_blocks_w):
                # Random block position
                start_i = np.random.randint(0, max(1, h - block_size + 1))
                start_j = np.random.randint(0, max(1, w - block_size + 1))
                
                # Extract block
                end_i = min(start_i + block_size, h)
                end_j = min(start_j + block_size, w)
                
                block = field[start_i:end_i, start_j:end_j]
                
                # Check tissue coverage if mask provided
                if mask is not None:
                    block_mask = mask[start_i:end_i, start_j:end_j]
                    tissue_coverage = np.mean(block_mask)
                    
                    if tissue_coverage < self.config.min_tissue_coverage:
                        continue  # Skip low-tissue blocks
                
                # Random placement in bootstrap sample
                place_i = np.random.randint(0, max(1, h - block.shape[0] + 1))
                place_j = np.random.randint(0, max(1, w - block.shape[1] + 1))
                
                place_end_i = place_i + block.shape[0]
                place_end_j = place_j + block.shape[1]
                
                # Add block with blending
                if self.config.blend_method == 'average':
                    # Simple averaging for overlaps
                    bootstrap_field[place_i:place_end_i, place_j:place_end_j] += block
                    weight_field[place_i:place_end_i, place_j:place_end_j] += 1
                
                elif self.config.blend_method == 'poisson':
                    # Poisson blending for seamless transitions
                    blended = self._poisson_blend(
                        bootstrap_field[place_i:place_end_i, place_j:place_end_j],
                        block
                    )
                    bootstrap_field[place_i:place_end_i, place_j:place_end_j] = blended
                    weight_field[place_i:place_end_i, place_j:place_end_j] = 1
                
                elif self.config.blend_method == 'feather':
                    # Feather edges for smooth transitions
                    feather_mask = self._create_feather_mask(block.shape)
                    bootstrap_field[place_i:place_end_i, place_j:place_end_j] += block * feather_mask
                    weight_field[place_i:place_end_i, place_j:place_end_j] += feather_mask
            
            # Normalize by weights
            weight_field[weight_field == 0] = 1
            bootstrap_field = bootstrap_field / weight_field
            
            # Re-apply mask if provided
            if mask is not None:
                bootstrap_field[~mask] = 0
            
            yield bootstrap_field
    
    def graph_block_bootstrap(self,
                            values: np.ndarray,
                            coords: np.ndarray,
                            spatial_weights: Optional[np.ndarray] = None,
                            target_block_size: int = 50,
                            n_samples: Optional[int] = None) -> Iterator[np.ndarray]:
        """
        Bootstrap via connected subgraphs for irregular point patterns.
        
        Args:
            values: Values at each point
            coords: Nx2 spatial coordinates
            spatial_weights: Pre-computed spatial weights/adjacency
            target_block_size: Target number of nodes per block
            n_samples: Number of bootstrap samples
            
        Yields:
            Bootstrap samples of values
        """
        n_points = len(values)
        n_samples = n_samples or self.config.n_bootstrap
        
        # Build spatial graph if needed
        if spatial_weights is None:
            spatial_weights = self._build_spatial_graph(coords)
        
        # Convert to adjacency
        adjacency = (spatial_weights > 0).astype(int)
        
        for _ in range(n_samples):
            bootstrap_values = np.zeros_like(values)
            sampled = np.zeros(n_points, dtype=bool)
            
            while np.sum(sampled) < n_points:
                # Pick random starting node from unsampled
                unsampled = np.where(~sampled)[0]
                if len(unsampled) == 0:
                    break
                
                start_node = np.random.choice(unsampled)
                
                # BFS to extract connected block
                block_nodes = self._extract_graph_block(
                    adjacency, start_node, target_block_size, sampled
                )
                
                # Sample block for resampling
                source_start = np.random.randint(n_points)
                source_block = self._extract_graph_block(
                    adjacency, source_start, len(block_nodes), np.zeros(n_points, dtype=bool)
                )
                
                # Map values
                if len(source_block) == len(block_nodes):
                    bootstrap_values[block_nodes] = values[source_block]
                else:
                    # Handle size mismatch by resampling with replacement
                    resampled_idx = np.random.choice(source_block, size=len(block_nodes), replace=True)
                    bootstrap_values[block_nodes] = values[resampled_idx]
                
                sampled[block_nodes] = True
            
            yield bootstrap_values
    
    def nested_spatial_bootstrap(self,
                                hierarchical_data: Dict[str, np.ndarray],
                                hierarchy_levels: List[str],
                                spatial_coords: Dict[str, np.ndarray],
                                n_samples: Optional[int] = None) -> Iterator[Dict[str, np.ndarray]]:
        """
        Nested bootstrap respecting hierarchical structure (Subject → ROI → Block).
        
        Args:
            hierarchical_data: Dict with data at each level
            hierarchy_levels: Ordered list of hierarchy levels
            spatial_coords: Spatial coordinates at each level
            n_samples: Number of bootstrap samples
            
        Yields:
            Bootstrap samples preserving hierarchical structure
        """
        n_samples = n_samples or self.config.n_bootstrap
        
        for _ in range(n_samples):
            bootstrap_sample = {}
            
            # Bootstrap at each hierarchical level
            for level in hierarchy_levels:
                if level == hierarchy_levels[0]:
                    # Top level - simple bootstrap
                    data = hierarchical_data[level]
                    n_units = len(data)
                    bootstrap_idx = np.random.choice(n_units, size=n_units, replace=True)
                    bootstrap_sample[level] = data[bootstrap_idx]
                    
                else:
                    # Lower levels - spatial bootstrap within units
                    parent_level = hierarchy_levels[hierarchy_levels.index(level) - 1]
                    parent_units = np.unique(bootstrap_sample[parent_level])
                    
                    level_bootstrap = []
                    for unit in parent_units:
                        unit_data = hierarchical_data[level][
                            hierarchical_data[parent_level] == unit
                        ]
                        unit_coords = spatial_coords[level][
                            hierarchical_data[parent_level] == unit
                        ]
                        
                        # Spatial bootstrap within unit
                        if len(unit_data) > 0:
                            unit_bootstrap = next(self.moving_block_bootstrap(
                                unit_data.reshape(-1, 1),
                                n_samples=1
                            )).ravel()
                            level_bootstrap.extend(unit_bootstrap)
                    
                    bootstrap_sample[level] = np.array(level_bootstrap)
            
            yield bootstrap_sample
    
    def adaptive_block_size_selection(self,
                                    field: np.ndarray,
                                    mask: Optional[np.ndarray] = None,
                                    method: str = 'variogram') -> int:
        """
        Automatically determine optimal block size from spatial correlation.
        
        Args:
            field: Spatial field
            mask: Tissue mask
            method: 'variogram', 'autocorrelation', or 'spectral'
            
        Returns:
            Estimated optimal block size
        """
        if method == 'variogram':
            return self._estimate_from_variogram(field, mask)
        elif method == 'autocorrelation':
            return self._estimate_from_autocorrelation(field, mask)
        elif method == 'spectral':
            return self._estimate_from_spectrum(field, mask)
        else:
            # Default fallback
            return max(field.shape) // 10
    
    def _estimate_block_size(self, field: np.ndarray, mask: Optional[np.ndarray]) -> int:
        """Estimate block size from spatial correlation structure."""
        
        # Check cache
        cache_key = (field.shape, field.dtype, str(mask.sum()) if mask is not None else 'None')
        if cache_key in self.estimated_block_sizes:
            return self.estimated_block_sizes[cache_key]
        
        # Estimate correlation range
        if mask is not None:
            field_masked = field.copy()
            field_masked[~mask] = np.nan
        else:
            field_masked = field
        
        # Compute autocorrelation
        autocorr = self._compute_2d_autocorrelation(field_masked)
        
        # Find correlation range (distance where autocorr drops to 1/e)
        center = np.array(autocorr.shape) // 2
        radial_profile = self._radial_profile(autocorr, center)
        
        # Find 1/e crossing
        threshold = radial_profile[0] / np.e
        correlation_range = np.where(radial_profile < threshold)[0]
        
        if len(correlation_range) > 0:
            block_size = int(2 * correlation_range[0])  # 2x correlation range
        else:
            block_size = max(field.shape) // 10  # Fallback
        
        # Cache result
        self.estimated_block_sizes[cache_key] = block_size
        
        return block_size
    
    def _compute_2d_autocorrelation(self, field: np.ndarray) -> np.ndarray:
        """Compute 2D autocorrelation function."""
        # Handle NaNs
        field_clean = np.nan_to_num(field, nan=0)
        
        # FFT-based autocorrelation
        fft = np.fft.fft2(field_clean)
        power_spectrum = np.abs(fft) ** 2
        autocorr = np.fft.ifft2(power_spectrum).real
        autocorr = np.fft.fftshift(autocorr)
        
        # Normalize
        autocorr = autocorr / np.nanmax(autocorr)
        
        return autocorr
    
    def _radial_profile(self, image: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Compute radial profile from center."""
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        max_r = min(center[0], center[1], 
                   image.shape[0] - center[0], 
                   image.shape[1] - center[1])
        
        profile = np.zeros(max_r)
        counts = np.zeros(max_r)
        
        for i in range(max_r):
            mask = (r >= i) & (r < i + 1)
            if np.any(mask):
                profile[i] = np.mean(image[mask])
                counts[i] = np.sum(mask)
        
        # Avoid division by zero
        counts[counts == 0] = 1
        profile = profile / counts
        
        return profile
    
    def _poisson_blend(self, target: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Poisson blending for seamless merging."""
        # Simplified Poisson blending - solve Laplace equation
        # ∇²f = ∇²g in Ω, f = g on ∂Ω
        
        if target.shape != source.shape:
            raise ValueError("Target and source must have same shape")
        
        # Compute Laplacian of source
        laplacian_source = ndimage.laplace(source)
        
        # Initialize with target at boundaries
        result = target.copy()
        mask = np.ones_like(source, dtype=bool)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
        
        # Simple iterative solver (Gauss-Seidel)
        for _ in range(10):  # Fixed iterations for speed
            result[mask] = 0.25 * (
                result[:-2, 1:-1][mask[1:-1, :]] +
                result[2:, 1:-1][mask[1:-1, :]] +
                result[1:-1, :-2][mask[:, 1:-1]] +
                result[1:-1, 2:][mask[:, 1:-1]] -
                laplacian_source[1:-1, 1:-1][mask[1:-1, 1:-1]]
            )
        
        return result
    
    def _create_feather_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create feathered edge mask for smooth blending."""
        h, w = shape
        mask = np.ones((h, w))
        
        # Feather width (10% of dimension)
        feather_h = max(1, h // 10)
        feather_w = max(1, w // 10)
        
        # Create distance field from edges
        for i in range(feather_h):
            weight = i / feather_h
            mask[i, :] *= weight
            mask[-(i+1), :] *= weight
        
        for j in range(feather_w):
            weight = j / feather_w
            mask[:, j] *= weight
            mask[:, -(j+1)] *= weight
        
        return mask
    
    def _build_spatial_graph(self, coords: np.ndarray) -> np.ndarray:
        """Build spatial adjacency graph from coordinates."""
        n_points = len(coords)
        
        # k-nearest neighbors graph
        k = min(15, n_points - 1)
        tree = cKDTree(coords)
        
        weights = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            distances, indices = tree.query(coords[i], k=k+1)
            
            # Skip self
            for dist, idx in zip(distances[1:], indices[1:]):
                weights[i, idx] = 1.0 / (1.0 + dist)
        
        # Symmetrize
        weights = (weights + weights.T) / 2
        
        return weights
    
    def _extract_graph_block(self,
                           adjacency: np.ndarray,
                           start_node: int,
                           target_size: int,
                           visited: np.ndarray) -> List[int]:
        """Extract connected block via BFS."""
        from collections import deque
        
        block = []
        queue = deque([start_node])
        block_visited = visited.copy()
        block_visited[start_node] = True
        
        while queue and len(block) < target_size:
            node = queue.popleft()
            block.append(node)
            
            # Add unvisited neighbors
            neighbors = np.where((adjacency[node] > 0) & ~block_visited)[0]
            for neighbor in neighbors:
                if len(block) >= target_size:
                    break
                queue.append(neighbor)
                block_visited[neighbor] = True
        
        return block
    
    def _estimate_from_variogram(self, field: np.ndarray, mask: Optional[np.ndarray]) -> int:
        """Estimate block size from empirical variogram."""
        # Sample points for variogram
        if mask is not None:
            y_coords, x_coords = np.where(mask)
        else:
            y_coords, x_coords = np.mgrid[0:field.shape[0], 0:field.shape[1]]
            y_coords = y_coords.ravel()
            x_coords = x_coords.ravel()
        
        # Subsample for efficiency
        n_samples = min(1000, len(y_coords))
        sample_idx = np.random.choice(len(y_coords), n_samples, replace=False)
        
        coords = np.column_stack([y_coords[sample_idx], x_coords[sample_idx]])
        values = field[y_coords[sample_idx], x_coords[sample_idx]]
        
        # Compute empirical variogram
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(coords))
        
        max_dist = np.percentile(distances[distances > 0], 50)
        bins = np.linspace(0, max_dist, 20)
        
        variogram = []
        for i in range(len(bins) - 1):
            mask_dist = (distances >= bins[i]) & (distances < bins[i+1])
            if np.any(mask_dist):
                pairs_i, pairs_j = np.where(mask_dist)
                semivariance = np.mean((values[pairs_i] - values[pairs_j])**2) / 2
                variogram.append(semivariance)
            else:
                variogram.append(np.nan)
        
        variogram = np.array(variogram)
        
        # Find range (distance where variogram plateaus)
        if np.any(~np.isnan(variogram)):
            sill = np.nanmax(variogram)
            range_idx = np.where(variogram >= 0.95 * sill)[0]
            if len(range_idx) > 0:
                correlation_range = bins[range_idx[0]]
                return int(2 * correlation_range)
        
        return max(field.shape) // 10
    
    def _estimate_from_autocorrelation(self, field: np.ndarray, mask: Optional[np.ndarray]) -> int:
        """Estimate from autocorrelation function."""
        return self._estimate_block_size(field, mask)
    
    def _estimate_from_spectrum(self, field: np.ndarray, mask: Optional[np.ndarray]) -> int:
        """Estimate from power spectrum."""
        # Compute power spectrum
        if mask is not None:
            field_masked = field.copy()
            field_masked[~mask] = np.mean(field[mask])
        else:
            field_masked = field
        
        # FFT and power spectrum
        fft = np.fft.fft2(field_masked)
        power_spectrum = np.abs(fft) ** 2
        
        # Radial average
        center = np.array(power_spectrum.shape) // 2
        radial_power = self._radial_profile(np.fft.fftshift(power_spectrum), center)
        
        # Find characteristic length scale (peak in spectrum)
        if len(radial_power) > 1:
            # Smooth to find peak
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(radial_power, sigma=2)
            
            peaks = np.where((smoothed[1:-1] > smoothed[:-2]) & 
                           (smoothed[1:-1] > smoothed[2:]))[0] + 1
            
            if len(peaks) > 0:
                characteristic_freq = peaks[0]
                characteristic_length = power_spectrum.shape[0] / (2 * characteristic_freq)
                return int(2 * characteristic_length)
        
        return max(field.shape) // 10