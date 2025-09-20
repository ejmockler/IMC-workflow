"""
Superpixel-Based Methods for IMC Spatial Analysis

DEPRECATED: This module duplicates functionality in slic_segmentation.py
Please use src.analysis.slic_segmentation instead.

This file is kept for backwards compatibility but will be removed in future versions.
All functionality has been consolidated into slic_segmentation.py for maintainability.
"""

import warnings
warnings.warn(
    "superpixel_methods.py is deprecated. Use src.analysis.slic_segmentation instead.",
    DeprecationWarning,
    stacklevel=2
)

import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from skimage.segmentation import slic, watershed, quickshift
from skimage.segmentation import mark_boundaries
from scipy.spatial import cKDTree
from scipy import ndimage


@dataclass
class SuperpixelResult:
    """Encapsulates superpixel segmentation results for method comparison."""
    labels: np.ndarray  # Superpixel assignments for each pixel
    n_segments: int
    centroids: np.ndarray  # Centroid coordinates for each superpixel
    sizes: np.ndarray  # Number of pixels in each superpixel
    mean_expressions: np.ndarray  # Mean expression per superpixel
    adjacency: Dict[int, List[int]]  # Adjacency relationships
    method: str
    parameters: Dict[str, Any]
    validation_metrics: Optional[Dict[str, float]] = None


class TissueParcellator:
    """
    Parcellates tissue into superpixels for region-based analysis.
    
    Compares different superpixel methods for IMC spatial analysis,
    focusing on method validation and parameter optimization.
    """
    
    def __init__(self, method: str = 'slic', config: Optional[Dict] = None):
        """
        Initialize tissue parcellator.
        
        Args:
            method: Segmentation method ('slic', 'watershed', 'quickshift', 'grid')
            config: Method-specific configuration parameters
        """
        self.method = method
        self.config = config or {}
        
    def parcellate(self, 
                  coords: np.ndarray, 
                  values: np.ndarray,
                  image_shape: Optional[Tuple[int, int]] = None,
                  target_size_um: float = 50.0,
                  resolution_um: float = 1.0) -> SuperpixelResult:
        """
        Parcellate tissue into superpixels using specified method.
        
        Args:
            coords: Pixel coordinates (N, 2)
            values: Pixel values/intensities (N, channels)  
            image_shape: Optional image dimensions
            target_size_um: Target superpixel size in micrometers
            resolution_um: Image resolution (um/pixel)
            
        Returns:
            SuperpixelResult with segmentation and metadata
        """
        if image_shape is None:
            # Estimate from coordinates
            image_shape = (
                int(np.max(coords[:, 1]) - np.min(coords[:, 1])) + 1,
                int(np.max(coords[:, 0]) - np.min(coords[:, 0])) + 1
            )
        
        # Create dense image from sparse coordinates
        if values.ndim == 1:
            # Single channel
            dense_image = self._coords_to_dense(coords, values, image_shape)
        else:
            # Multi-channel - use first channel or create composite
            if values.shape[1] == 1:
                dense_image = self._coords_to_dense(coords, values[:, 0], image_shape)
            else:
                # Create composite for segmentation
                composite = np.mean(values, axis=1)
                dense_image = self._coords_to_dense(coords, composite, image_shape)
        
        # Apply superpixel method
        if self.method == 'slic':
            labels = self._slic_parcellation(
                dense_image, target_size_um, resolution_um
            )
        elif self.method == 'watershed':
            labels = self._watershed_parcellation(
                dense_image, target_size_um, resolution_um
            )
        elif self.method == 'quickshift':
            labels = self._quickshift_parcellation(
                dense_image, target_size_um, resolution_um
            )
        elif self.method == 'grid':
            labels = self._grid_parcellation(
                image_shape, target_size_um, resolution_um
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Compute superpixel properties
        result = self._compute_superpixel_properties(
            labels, coords, values, image_shape
        )
        
        return result
    
    def _coords_to_dense(self, 
                        coords: np.ndarray, 
                        values: np.ndarray, 
                        image_shape: Tuple[int, int]) -> np.ndarray:
        """Convert sparse coordinates to dense image."""
        
        dense_image = np.zeros(image_shape)
        
        # Offset coordinates to start from 0
        min_x, min_y = np.min(coords, axis=0)
        adj_coords = coords - np.array([min_x, min_y])
        
        # Ensure coordinates are within bounds
        valid_mask = ((adj_coords[:, 0] >= 0) & 
                     (adj_coords[:, 0] < image_shape[1]) &
                     (adj_coords[:, 1] >= 0) & 
                     (adj_coords[:, 1] < image_shape[0]))
        
        if np.any(valid_mask):
            valid_coords = adj_coords[valid_mask].astype(int)
            valid_values = values[valid_mask]
            dense_image[valid_coords[:, 1], valid_coords[:, 0]] = valid_values
        
        return dense_image
    
    def _slic_parcellation(self, 
                          image: np.ndarray, 
                          target_size_um: float,
                          resolution_um: float) -> np.ndarray:
        """Apply SLIC superpixel segmentation."""
        
        # Calculate number of segments
        image_area_pixels = image.size
        target_area_pixels = (target_size_um / resolution_um) ** 2
        n_segments = max(1, int(image_area_pixels / target_area_pixels))
        
        # SLIC parameters
        compactness = self.config.get('compactness', 10.0)
        sigma = self.config.get('sigma', 1.0)
        
        # Convert to RGB for SLIC (expects 3D input)
        if image.ndim == 2:
            rgb_image = np.stack([image, image, image], axis=2)
        else:
            rgb_image = image
        
        # Apply SLIC
        labels = slic(
            rgb_image,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=0
        )
        
        return labels
    
    def _watershed_parcellation(self, 
                               image: np.ndarray,
                               target_size_um: float,
                               resolution_um: float) -> np.ndarray:
        """Apply watershed superpixel segmentation."""
        
        from skimage.feature import peak_local_maxima
        from skimage.morphology import h_maxima
        
        # Find seed points
        h = self.config.get('h_maxima', 0.1)
        local_maxima = h_maxima(image, h)
        
        # Apply watershed
        labels = watershed(-image, markers=local_maxima, mask=image > 0)
        
        return labels
    
    def _quickshift_parcellation(self, 
                                image: np.ndarray,
                                target_size_um: float, 
                                resolution_um: float) -> np.ndarray:
        """Apply QuickShift superpixel segmentation."""
        
        kernel_size = self.config.get('kernel_size', 3)
        max_dist = self.config.get('max_dist', 10)
        
        if image.ndim == 2:
            rgb_image = np.stack([image, image, image], axis=2)
        else:
            rgb_image = image
            
        labels = quickshift(
            rgb_image,
            kernel_size=kernel_size,
            max_dist=max_dist
        )
        
        return labels
    
    def _grid_parcellation(self, 
                          image_shape: Tuple[int, int],
                          target_size_um: float,
                          resolution_um: float) -> np.ndarray:
        """Create regular grid parcellation."""
        
        # Grid size in pixels
        grid_size_pixels = int(target_size_um / resolution_um)
        
        labels = np.zeros(image_shape, dtype=int)
        
        label_id = 0
        for y in range(0, image_shape[0], grid_size_pixels):
            for x in range(0, image_shape[1], grid_size_pixels):
                y_end = min(y + grid_size_pixels, image_shape[0])
                x_end = min(x + grid_size_pixels, image_shape[1])
                labels[y:y_end, x:x_end] = label_id
                label_id += 1
        
        return labels
    
    def _compute_superpixel_properties(self,
                                     labels: np.ndarray,
                                     coords: np.ndarray,
                                     values: np.ndarray,
                                     image_shape: Tuple[int, int]) -> SuperpixelResult:
        """Compute properties of superpixels."""
        
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]
        n_segments = len(unique_labels)
        
        if n_segments == 0:
            return SuperpixelResult(
                labels=labels,
                n_segments=0,
                centroids=np.array([]),
                sizes=np.array([]),
                mean_expressions=np.array([]),
                adjacency={},
                method=self.method,
                parameters=self.config
            )
        
        # Compute centroids and sizes
        centroids = np.zeros((n_segments, 2))
        sizes = np.zeros(n_segments)
        
        for i, label_id in enumerate(unique_labels):
            mask = labels == label_id
            y_coords, x_coords = np.where(mask)
            
            if len(y_coords) > 0:
                centroids[i] = [np.mean(x_coords), np.mean(y_coords)]
                sizes[i] = len(y_coords)
        
        # Compute mean expressions per superpixel
        if values.ndim == 1:
            n_channels = 1
            mean_expressions = np.zeros((n_segments, 1))
        else:
            n_channels = values.shape[1]
            mean_expressions = np.zeros((n_segments, n_channels))
        
        # Map pixel coordinates to superpixel labels
        min_x, min_y = np.min(coords, axis=0)
        adj_coords = coords - np.array([min_x, min_y])
        
        for i, label_id in enumerate(unique_labels):
            # Find pixels belonging to this superpixel
            mask = labels == label_id
            y_coords, x_coords = np.where(mask)
            
            if len(y_coords) > 0:
                # Find corresponding data points
                pixel_coords = np.column_stack([x_coords, y_coords])
                
                # Match with input coordinates
                if len(coords) > 0:
                    tree = cKDTree(adj_coords)
                    distances, indices = tree.query(pixel_coords, distance_upper_bound=1.5)
                    
                    valid_matches = distances < 1.5
                    if np.any(valid_matches):
                        matched_indices = indices[valid_matches]
                        if values.ndim == 1:
                            mean_expressions[i, 0] = np.mean(values[matched_indices])
                        else:
                            mean_expressions[i] = np.mean(values[matched_indices], axis=0)
        
        # Compute adjacency (simplified)
        adjacency = self._compute_adjacency(labels, unique_labels)
        
        return SuperpixelResult(
            labels=labels,
            n_segments=n_segments,
            centroids=centroids,
            sizes=sizes,
            mean_expressions=mean_expressions,
            adjacency=adjacency,
            method=self.method,
            parameters=self.config.copy()
        )
    
    def _compute_adjacency(self, 
                          labels: np.ndarray, 
                          unique_labels: np.ndarray) -> Dict[int, List[int]]:
        """Compute adjacency relationships between superpixels."""
        
        adjacency = {int(label): [] for label in unique_labels}
        
        # Check 4-connectivity
        for y in range(labels.shape[0] - 1):
            for x in range(labels.shape[1] - 1):
                current_label = labels[y, x]
                
                # Check right neighbor
                if x < labels.shape[1] - 1:
                    right_label = labels[y, x + 1]
                    if current_label != right_label and current_label >= 0 and right_label >= 0:
                        if right_label not in adjacency[current_label]:
                            adjacency[current_label].append(right_label)
                        if current_label not in adjacency[right_label]:
                            adjacency[right_label].append(current_label)
                
                # Check bottom neighbor  
                if y < labels.shape[0] - 1:
                    bottom_label = labels[y + 1, x]
                    if current_label != bottom_label and current_label >= 0 and bottom_label >= 0:
                        if bottom_label not in adjacency[current_label]:
                            adjacency[current_label].append(bottom_label)
                        if current_label not in adjacency[bottom_label]:
                            adjacency[bottom_label].append(current_label)
        
        return adjacency


def compare_superpixel_methods(coords: np.ndarray,
                              values: np.ndarray,
                              image_shape: Tuple[int, int],
                              target_size_um: float = 50.0,
                              resolution_um: float = 1.0,
                              methods: List[str] = None) -> Dict[str, SuperpixelResult]:
    """
    Compare different superpixel methods on the same data.
    
    Args:
        coords: Pixel coordinates
        values: Pixel values/intensities
        image_shape: Image dimensions
        target_size_um: Target superpixel size
        resolution_um: Image resolution
        methods: List of methods to compare
        
    Returns:
        Dictionary of method name -> SuperpixelResult
    """
    if methods is None:
        methods = ['slic', 'grid', 'quickshift']
    
    results = {}
    
    for method in methods:
        try:
            parcellator = TissueParcellator(method=method)
            result = parcellator.parcellate(
                coords=coords,
                values=values,
                image_shape=image_shape,
                target_size_um=target_size_um,
                resolution_um=resolution_um
            )
            results[method] = result
            
        except Exception as e:
            print(f"Warning: Method {method} failed: {e}")
            continue
    
    return results