"""
Superpixel-Based Tissue Parcellation for IMC Analysis
Segments tissue into coherent regions without requiring cell boundaries
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from skimage.segmentation import slic, watershed, quickshift
from skimage.segmentation import mark_boundaries
from scipy.spatial import cKDTree
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SuperpixelResult:
    """Encapsulates superpixel segmentation results"""
    labels: np.ndarray  # Superpixel assignments for each pixel
    n_segments: int
    centroids: np.ndarray  # Centroid coordinates for each superpixel
    sizes: np.ndarray  # Number of pixels in each superpixel
    mean_expressions: np.ndarray  # Mean expression per superpixel
    adjacency: Dict[int, List[int]]  # Adjacency relationships
    method: str
    parameters: Dict[str, Any]


class TissueParcellator:
    """Parcellates tissue into superpixels for region-based analysis"""
    
    def __init__(self, method: str = 'slic', config: Optional[Dict] = None):
        """
        Initialize tissue parcellator
        
        Args:
            method: Segmentation method ('slic', 'watershed', 'quickshift', 'grid')
            config: Method-specific configuration
        """
        self.method = method
        self.config = config or {}
        
    def parcellate(self, coords: np.ndarray, 
                  values: np.ndarray,
                  image_shape: Optional[Tuple[int, int]] = None) -> SuperpixelResult:
        """
        Parcellate tissue into superpixels
        
        Args:
            coords: Spatial coordinates (n_pixels, 2)
            values: Expression values (n_pixels, n_proteins)
            image_shape: Shape of tissue image (height, width)
            
        Returns:
            SuperpixelResult with parcellation
        """
        # Create image representation
        if image_shape is None:
            image_shape = self._estimate_image_shape(coords)
        
        image = self._coords_to_image(coords, values, image_shape)
        
        # Apply segmentation method
        if self.method == 'slic':
            labels = self._slic_segmentation(image)
        elif self.method == 'watershed':
            labels = self._watershed_segmentation(image)
        elif self.method == 'quickshift':
            labels = self._quickshift_segmentation(image)
        elif self.method == 'grid':
            labels = self._grid_segmentation(image_shape)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Map back to pixel coordinates
        pixel_labels = self._image_to_pixel_labels(labels, coords, image_shape)
        
        # Compute superpixel properties
        result = self._compute_superpixel_properties(
            pixel_labels, coords, values
        )
        
        return result
    
    def _estimate_image_shape(self, coords: np.ndarray) -> Tuple[int, int]:
        """Estimate image dimensions from coordinates"""
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        
        # Estimate resolution (pixels per unit)
        resolution = self.config.get('resolution', 1.0)
        
        height = int(y_range * resolution) + 1
        width = int(x_range * resolution) + 1
        
        return (height, width)
    
    def _coords_to_image(self, coords: np.ndarray, 
                        values: np.ndarray,
                        image_shape: Tuple[int, int]) -> np.ndarray:
        """Convert scattered coordinates to image grid"""
        height, width = image_shape
        n_channels = values.shape[1]
        
        # Initialize image
        image = np.zeros((height, width, n_channels))
        
        # Normalize coordinates to image indices
        x_min, y_min = coords.min(axis=0)
        x_range = coords[:, 0].max() - x_min
        y_range = coords[:, 1].max() - y_min
        
        x_idx = ((coords[:, 0] - x_min) / x_range * (width - 1)).astype(int)
        y_idx = ((coords[:, 1] - y_min) / y_range * (height - 1)).astype(int)
        
        # Fill image with expression values
        for i, (x, y) in enumerate(zip(x_idx, y_idx)):
            image[y, x] = values[i]
        
        # Fill gaps with nearest neighbor interpolation
        for c in range(n_channels):
            mask = image[:, :, c] == 0
            if mask.any():
                indices = ndimage.distance_transform_edt(
                    mask, return_distances=False, return_indices=True
                )
                image[:, :, c] = image[:, :, c][tuple(indices)]
        
        return image
    
    def _slic_segmentation(self, image: np.ndarray) -> np.ndarray:
        """SLIC superpixel segmentation"""
        n_segments = self.config.get('n_segments', 500)
        compactness = self.config.get('compactness', 10)
        sigma = self.config.get('sigma', 1)
        
        # Use mean expression across proteins as main channel
        main_channel = image.mean(axis=2)
        
        # Add spatial coordinates as features
        height, width = main_channel.shape
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        
        # Stack features: mean expression + spatial coords
        features = np.dstack([main_channel, xx / width, yy / height])
        
        labels = slic(features, n_segments=n_segments, 
                     compactness=compactness, sigma=sigma,
                     start_label=0)
        
        return labels
    
    def _watershed_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Watershed segmentation on expression gradients"""
        from skimage.filters import sobel
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        from scipy import ndimage as ndi
        
        # Use mean expression
        main_channel = image.mean(axis=2)
        
        # Compute gradient
        edges = sobel(main_channel)
        
        # Find markers
        markers_distance = self.config.get('markers_distance', 10)
        
        distance = ndi.distance_transform_edt(main_channel > 0)
        coords = peak_local_max(distance, min_distance=markers_distance,
                               labels=main_channel > 0)
        
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        
        labels = watershed(edges, markers, mask=main_channel > 0)
        
        return labels
    
    def _quickshift_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Quickshift mode-seeking segmentation"""
        kernel_size = self.config.get('kernel_size', 5)
        max_dist = self.config.get('max_dist', 10)
        ratio = self.config.get('ratio', 0.5)
        
        # Normalize image for quickshift
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
        
        labels = quickshift(img_norm, kernel_size=kernel_size,
                          max_dist=max_dist, ratio=ratio)
        
        return labels
    
    def _grid_segmentation(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Simple grid-based segmentation"""
        tile_size = self.config.get('tile_size', 50)
        height, width = image_shape
        
        labels = np.zeros(image_shape, dtype=int)
        label_id = 0
        
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                labels[i:min(i+tile_size, height), 
                      j:min(j+tile_size, width)] = label_id
                label_id += 1
        
        return labels
    
    def _image_to_pixel_labels(self, labels: np.ndarray,
                               coords: np.ndarray,
                               image_shape: Tuple[int, int]) -> np.ndarray:
        """Map image labels back to pixel coordinates"""
        height, width = image_shape
        
        # Normalize coordinates to image indices
        x_min, y_min = coords.min(axis=0)
        x_range = coords[:, 0].max() - x_min
        y_range = coords[:, 1].max() - y_min
        
        x_idx = ((coords[:, 0] - x_min) / x_range * (width - 1)).astype(int)
        y_idx = ((coords[:, 1] - y_min) / y_range * (height - 1)).astype(int)
        
        # Get labels for each pixel
        pixel_labels = labels[y_idx, x_idx]
        
        return pixel_labels
    
    def _compute_superpixel_properties(self, labels: np.ndarray,
                                      coords: np.ndarray,
                                      values: np.ndarray) -> SuperpixelResult:
        """Compute properties for each superpixel"""
        n_segments = len(np.unique(labels))
        n_proteins = values.shape[1]
        
        centroids = np.zeros((n_segments, 2))
        sizes = np.zeros(n_segments, dtype=int)
        mean_expressions = np.zeros((n_segments, n_proteins))
        
        for seg_id in range(n_segments):
            mask = labels == seg_id
            sizes[seg_id] = mask.sum()
            
            if sizes[seg_id] > 0:
                centroids[seg_id] = coords[mask].mean(axis=0)
                mean_expressions[seg_id] = values[mask].mean(axis=0)
        
        # Compute adjacency
        adjacency = self._compute_adjacency(labels, coords)
        
        return SuperpixelResult(
            labels=labels,
            n_segments=n_segments,
            centroids=centroids,
            sizes=sizes,
            mean_expressions=mean_expressions,
            adjacency=adjacency,
            method=self.method,
            parameters=self.config
        )
    
    def _compute_adjacency(self, labels: np.ndarray,
                          coords: np.ndarray) -> Dict[int, List[int]]:
        """Compute adjacency relationships between superpixels"""
        from collections import defaultdict
        
        adjacency = defaultdict(set)
        
        # Build KDTree for efficient neighbor search
        tree = cKDTree(coords)
        
        # Find neighbors within reasonable distance
        radius = self.config.get('adjacency_radius', 10)
        
        for i, coord in enumerate(coords):
            label_i = labels[i]
            neighbors = tree.query_ball_point(coord, radius)
            
            for j in neighbors:
                label_j = labels[j]
                if label_i != label_j:
                    adjacency[label_i].add(label_j)
                    adjacency[label_j].add(label_i)
        
        # Convert to regular dict with lists
        return {k: list(v) for k, v in adjacency.items()}


class AdaptiveParcellator(TissueParcellator):
    """Adaptive superpixel segmentation that respects expression boundaries"""
    
    def parcellate(self, coords: np.ndarray,
                  values: np.ndarray,
                  image_shape: Optional[Tuple[int, int]] = None) -> SuperpixelResult:
        """
        Adaptive parcellation using hierarchical merging
        
        Args:
            coords: Spatial coordinates
            values: Expression values
            image_shape: Image dimensions
            
        Returns:
            Adaptively merged superpixels
        """
        # Start with fine segmentation
        self.config['n_segments'] = self.config.get('initial_segments', 1000)
        initial_result = super().parcellate(coords, values, image_shape)
        
        # Hierarchically merge similar adjacent superpixels
        merged_result = self._hierarchical_merge(initial_result, coords, values)
        
        return merged_result
    
    def _hierarchical_merge(self, result: SuperpixelResult, 
                           coords: np.ndarray,
                           values: np.ndarray) -> SuperpixelResult:
        """Merge similar adjacent superpixels"""
        similarity_threshold = self.config.get('similarity_threshold', 0.8)
        min_segments = self.config.get('min_segments', 100)
        
        # Build similarity graph
        import networkx as nx
        G = nx.Graph()
        
        for seg_id in range(result.n_segments):
            G.add_node(seg_id, expression=result.mean_expressions[seg_id])
        
        # Add edges between adjacent superpixels
        for seg_id, neighbors in result.adjacency.items():
            for neighbor_id in neighbors:
                if seg_id < neighbor_id and seg_id < result.n_segments and neighbor_id < result.n_segments:  # Avoid duplicates and bounds
                    # Compute similarity (correlation)
                    expr1 = result.mean_expressions[seg_id]
                    expr2 = result.mean_expressions[neighbor_id]
                    
                    similarity = np.corrcoef(expr1, expr2)[0, 1]
                    
                    if similarity > similarity_threshold:
                        G.add_edge(seg_id, neighbor_id, weight=similarity)
        
        # Find connected components (similar regions to merge)
        components = list(nx.connected_components(G))
        
        # Create new labels
        new_labels = result.labels.copy()
        new_id = 0
        
        for component in components:
            for seg_id in component:
                mask = result.labels == seg_id
                new_labels[mask] = new_id
            new_id += 1
        
        # Recompute properties with merged segments
        merged_result = self._compute_superpixel_properties(
            new_labels, coords, values
        )
        
        return merged_result


def create_tissue_parcellation(coords: np.ndarray,
                              values: np.ndarray,
                              method: str = 'slic',
                              config: Optional[Dict] = None) -> SuperpixelResult:
    """
    Convenience function to create tissue parcellation
    
    Args:
        coords: Spatial coordinates
        values: Expression values
        method: Parcellation method
        config: Method configuration
        
    Returns:
        SuperpixelResult
    """
    if config and config.get('adaptive', False):
        parcellator = AdaptiveParcellator(method, config)
    else:
        parcellator = TissueParcellator(method, config)
    
    return parcellator.parcellate(coords, values)