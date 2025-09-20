"""
Pure Geometric and Spatial Metrics

These metrics assess segmentation quality without any domain knowledge.
They work equally well for satellite images, medical scans, or microscopy.
"""

import numpy as np
from typing import Dict, Optional
from scipy import ndimage, stats
from scipy.spatial.distance import cdist
from .base import Metric


class GeometricCompactness(Metric):
    """
    Measures how compact/circular segments are.
    
    Perfect circle = 1.0, more irregular = lower values.
    No biological assumptions - just pure geometry.
    """
    
    @property
    def name(self) -> str:
        return "geometric_compactness"
    
    @property
    def description(self) -> str:
        return "Ratio of area to perimeter squared (circularity measure)"
    
    def compute(self, 
                segmentation: np.ndarray,
                context: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """Compute compactness for all segments."""
        
        unique_segments = np.unique(segmentation)
        unique_segments = unique_segments[unique_segments >= 0]
        
        if len(unique_segments) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        compactness_values = []
        
        for seg_id in unique_segments:
            mask = segmentation == seg_id
            
            # Calculate area
            area = np.sum(mask)
            if area == 0:
                continue
            
            # Calculate perimeter using morphological gradient
            gradient = ndimage.morphological_gradient(mask.astype(float), size=3)
            perimeter = np.sum(gradient > 0)
            
            if perimeter > 0:
                # Compactness = 4π * area / perimeter²
                compactness = (4 * np.pi * area) / (perimeter ** 2)
                compactness_values.append(min(compactness, 1.0))
        
        if compactness_values:
            return {
                'mean': float(np.mean(compactness_values)),
                'std': float(np.std(compactness_values)),
                'min': float(np.min(compactness_values)),
                'max': float(np.max(compactness_values))
            }
        
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}


class SizeUniformity(Metric):
    """
    Measures consistency of segment sizes.
    
    High uniformity = segments have similar sizes.
    Low uniformity = high size variation.
    """
    
    @property
    def name(self) -> str:
        return "size_uniformity"
    
    @property
    def description(self) -> str:
        return "Coefficient of variation of segment sizes"
    
    def compute(self,
                segmentation: np.ndarray,
                context: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """Compute size uniformity across segments."""
        
        unique_segments = np.unique(segmentation)
        unique_segments = unique_segments[unique_segments >= 0]
        
        if len(unique_segments) == 0:
            return {'uniformity': 0.0, 'cv': 0.0}
        
        sizes = []
        for seg_id in unique_segments:
            size = np.sum(segmentation == seg_id)
            if size > 0:
                sizes.append(size)
        
        if len(sizes) > 1:
            mean_size = np.mean(sizes)
            std_size = np.std(sizes)
            cv = std_size / mean_size if mean_size > 0 else 0.0
            uniformity = 1.0 / (1.0 + cv)  # Convert CV to uniformity score
            
            return {
                'uniformity': float(uniformity),
                'cv': float(cv),
                'mean_size': float(mean_size),
                'std_size': float(std_size)
            }
        
        return {'uniformity': 1.0, 'cv': 0.0}


class BoundaryRegularity(Metric):
    """
    Measures how smooth/regular segment boundaries are.
    
    Uses fractal dimension - lower = smoother boundaries.
    Works for any segmentation, no domain assumptions.
    """
    
    @property
    def name(self) -> str:
        return "boundary_regularity"
    
    @property
    def description(self) -> str:
        return "Smoothness of segment boundaries via fractal dimension"
    
    def compute(self,
                segmentation: np.ndarray,
                context: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """Compute boundary regularity."""
        
        # Find all boundaries
        gradient = ndimage.sobel(segmentation.astype(float))
        boundaries = gradient > 0
        
        if np.sum(boundaries) == 0:
            return {'regularity': 1.0, 'fractal_dim': 1.0}
        
        # Estimate fractal dimension using box-counting
        fractal_dim = self._box_counting_dimension(boundaries)
        
        # Convert to regularity score (lower fractal = more regular)
        regularity = 2.0 - fractal_dim  # Maps [1,2] to [1,0]
        
        return {
            'regularity': float(max(0, min(1, regularity))),
            'fractal_dim': float(fractal_dim),
            'boundary_pixels': int(np.sum(boundaries))
        }
    
    def _box_counting_dimension(self, binary_image: np.ndarray) -> float:
        """Estimate fractal dimension using box-counting method."""
        
        # Pad to square power of 2
        max_dim = max(binary_image.shape)
        size = 2 ** int(np.ceil(np.log2(max_dim)))
        
        padded = np.zeros((size, size))
        padded[:binary_image.shape[0], :binary_image.shape[1]] = binary_image
        
        # Count boxes at different scales
        scales = []
        counts = []
        
        for scale in [2**i for i in range(1, int(np.log2(size)))]:
            if scale > size / 4:
                break
                
            # Downsample and count non-empty boxes
            downsampled = padded.reshape(size//scale, scale, size//scale, scale).max(axis=(1, 3))
            count = np.sum(downsampled > 0)
            
            if count > 0:
                scales.append(scale)
                counts.append(count)
        
        if len(scales) > 1:
            # Fit log-log relationship
            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            return -coeffs[0]
        
        return 1.0


class SpatialAutocorrelation(Metric):
    """
    Measures spatial organization using Moran's I.
    
    High autocorrelation = similar segments cluster together.
    No autocorrelation = random spatial distribution.
    """
    
    @property
    def name(self) -> str:
        return "spatial_autocorrelation"
    
    @property  
    def description(self) -> str:
        return "Moran's I statistic for spatial clustering"
    
    def compute(self,
                segmentation: np.ndarray,
                context: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """Compute spatial autocorrelation of segments."""
        
        unique_segments = np.unique(segmentation)
        unique_segments = unique_segments[unique_segments >= 0]
        
        if len(unique_segments) < 2:
            return {'morans_i': 0.0, 'p_value': 1.0}
        
        # Create binary weights matrix (queen contiguity)
        weights = self._create_spatial_weights(segmentation)
        
        # Compute Moran's I for segment sizes
        segment_sizes = np.zeros_like(segmentation, dtype=float)
        for seg_id in unique_segments:
            mask = segmentation == seg_id
            segment_sizes[mask] = np.sum(mask)
        
        morans_i = self._morans_i(segment_sizes, weights)
        
        # Compute expected value and variance under null hypothesis
        n = len(unique_segments)
        expected_i = -1.0 / (n - 1)
        
        # Simplified variance calculation
        variance = 1.0 / (n - 1)
        
        # Z-score and p-value
        z_score = (morans_i - expected_i) / np.sqrt(variance)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'morans_i': float(morans_i),
            'expected_i': float(expected_i),
            'z_score': float(z_score),
            'p_value': float(p_value)
        }
    
    def _create_spatial_weights(self, segmentation: np.ndarray) -> np.ndarray:
        """Create spatial weights matrix for segments."""
        
        # Use structural element for queen contiguity (8-neighbors)
        struct = ndimage.generate_binary_structure(2, 2)
        
        # Dilate each segment and check overlaps
        unique_segments = np.unique(segmentation)
        unique_segments = unique_segments[unique_segments >= 0]
        
        n_segments = len(unique_segments)
        weights = np.zeros((n_segments, n_segments))
        
        seg_to_idx = {seg: i for i, seg in enumerate(unique_segments)}
        
        for i, seg_id in enumerate(unique_segments):
            mask = segmentation == seg_id
            dilated = ndimage.binary_dilation(mask, struct)
            
            # Find neighboring segments
            neighbors = np.unique(segmentation[dilated & ~mask])
            neighbors = neighbors[neighbors >= 0]
            
            for neighbor in neighbors:
                if neighbor in seg_to_idx:
                    j = seg_to_idx[neighbor]
                    weights[i, j] = 1
                    weights[j, i] = 1
        
        return weights
    
    def _morans_i(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Calculate Moran's I statistic."""
        
        # Flatten values
        flat_values = values.flatten()
        n = len(flat_values)
        
        # Mean center the values
        mean_val = np.mean(flat_values)
        centered = flat_values - mean_val
        
        # Compute numerator and denominator
        numerator = np.sum(weights * np.outer(centered, centered))
        denominator = np.sum(centered ** 2)
        
        # Sum of weights
        w_sum = np.sum(weights)
        
        if w_sum > 0 and denominator > 0:
            return (n / w_sum) * (numerator / denominator)
        
        return 0.0


class SignalAdherence(Metric):
    """
    Measures how well segments follow intensity gradients.
    
    Requires context (intensity image) but makes no assumptions
    about what the intensity represents.
    """
    
    @property
    def name(self) -> str:
        return "signal_adherence"
    
    @property
    def description(self) -> str:
        return "How well segment boundaries align with intensity gradients"
    
    def requires_context(self) -> bool:
        return True
    
    def compute(self,
                segmentation: np.ndarray,
                context: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """Compute signal adherence for any intensity image."""
        
        if context is None or len(context) == 0:
            return {'adherence': 0.0}
        
        # Use first context array (no assumptions about what it is)
        intensity = next(iter(context.values()))
        
        if intensity.shape != segmentation.shape:
            return {'adherence': 0.0, 'error': 'shape_mismatch'}
        
        # Compute intensity gradients
        grad_x = ndimage.sobel(intensity, axis=0)
        grad_y = ndimage.sobel(intensity, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find segment boundaries
        seg_boundaries = ndimage.sobel(segmentation.astype(float))
        boundary_mask = seg_boundaries > 0
        
        if np.sum(boundary_mask) == 0:
            return {'adherence': 0.0}
        
        # Measure gradient strength at boundaries
        boundary_gradients = gradient_magnitude[boundary_mask]
        mean_boundary_gradient = np.mean(boundary_gradients)
        
        # Compare to random positions
        n_boundary = np.sum(boundary_mask)
        random_positions = np.random.choice(gradient_magnitude.size, 
                                          min(n_boundary, 1000), 
                                          replace=False)
        random_gradients = gradient_magnitude.flat[random_positions]
        mean_random_gradient = np.mean(random_gradients)
        
        # Adherence score: ratio of boundary to random gradients
        if mean_random_gradient > 0:
            adherence = mean_boundary_gradient / mean_random_gradient
        else:
            adherence = 1.0
        
        return {
            'adherence': float(min(2.0, adherence)),  # Cap at 2x random
            'boundary_gradient': float(mean_boundary_gradient),
            'random_gradient': float(mean_random_gradient)
        }


# Registry of available metrics
REGISTERED_METRICS = {
    'compactness': GeometricCompactness,
    'size_uniformity': SizeUniformity,
    'boundary_regularity': BoundaryRegularity,
    'spatial_autocorrelation': SpatialAutocorrelation,
    'signal_adherence': SignalAdherence
}