"""
Grid-Based Segmentation Baseline for IMC Data

Provides an honest comparison baseline to SLIC superpixel segmentation using
regular grid overlay. Maintains identical interface for drop-in replacement
and fair performance benchmarking.

This implementation:
1. Uses uniform grid instead of morphology-aware SLIC superpixels
2. Supports same scales (10μm, 20μm, 40μm) for direct comparison
3. Provides identical function signatures for seamless integration
4. Includes performance metrics and boundary quality assessment
5. Maintains compatibility with existing aggregation and analysis pipeline
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional, List, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..config import Config


@dataclass
class GridMetrics:
    """Performance and quality metrics for grid segmentation."""
    generation_time: float
    aggregation_time: float
    n_grid_cells: int
    grid_size_um: float
    coverage_ratio: float  # Fraction of tissue covered by grid
    boundary_coherence: float  # Measure of biological boundary alignment
    memory_usage_mb: float


def create_regular_grid(
    bounds: Tuple[float, float, float, float],
    target_scale_um: float,
    resolution_um: float = 1.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Create regular grid labels for spatial region.
    
    Unlike SLIC which creates irregular superpixels based on image content,
    this creates a uniform grid overlay with fixed spacing.
    
    Args:
        bounds: Spatial bounds (x_min, x_max, y_min, y_max)
        target_scale_um: Target grid cell size in micrometers
        resolution_um: Grid resolution in micrometers/pixel
        
    Returns:
        Tuple of (grid_labels, grid_info)
        - grid_labels: 2D array with grid cell assignments
        - grid_info: Dictionary with grid properties
    """
    start_time = time.time()
    
    x_min, x_max, y_min, y_max = bounds
    
    # Calculate grid dimensions
    width_um = x_max - x_min
    height_um = y_max - y_min
    
    # Number of grid cells in each dimension
    n_cells_x = max(1, int(np.ceil(width_um / target_scale_um)))
    n_cells_y = max(1, int(np.ceil(height_um / target_scale_um)))
    
    # Actual grid cell size (may differ slightly from target due to rounding)
    actual_cell_size_x = width_um / n_cells_x
    actual_cell_size_y = height_um / n_cells_y
    actual_cell_area = actual_cell_size_x * actual_cell_size_y
    
    # Create pixel grid matching the bounds and resolution
    n_pixels_x = int(np.ceil(width_um / resolution_um))
    n_pixels_y = int(np.ceil(height_um / resolution_um))
    
    # Create coordinate arrays for each pixel
    x_coords = np.linspace(x_min, x_max, n_pixels_x)
    y_coords = np.linspace(y_min, y_max, n_pixels_y)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='xy')
    
    # Assign each pixel to a grid cell
    # Calculate which grid cell each pixel belongs to
    cell_x_indices = np.floor((x_grid - x_min) / actual_cell_size_x).astype(int)
    cell_y_indices = np.floor((y_grid - y_min) / actual_cell_size_y).astype(int)
    
    # Ensure indices are within bounds
    cell_x_indices = np.clip(cell_x_indices, 0, n_cells_x - 1)
    cell_y_indices = np.clip(cell_y_indices, 0, n_cells_y - 1)
    
    # Create unique grid cell labels
    # Each grid cell gets a unique ID from 0 to (n_cells_x * n_cells_y - 1)
    grid_labels = cell_y_indices * n_cells_x + cell_x_indices
    
    generation_time = time.time() - start_time
    
    grid_info = {
        'n_cells_x': n_cells_x,
        'n_cells_y': n_cells_y,
        'total_cells': n_cells_x * n_cells_y,
        'actual_cell_size_x_um': actual_cell_size_x,
        'actual_cell_size_y_um': actual_cell_size_y,
        'actual_cell_area_um2': actual_cell_area,
        'target_scale_um': target_scale_um,
        'generation_time': generation_time,
        'coverage_ratio': 1.0,  # Grid always covers 100% of region
        'grid_spacing_regularity': 1.0  # Perfect regularity for grid
    }
    
    return grid_labels, grid_info


def prepare_dna_composite_grid(
    coords: np.ndarray,
    dna1_intensities: np.ndarray,
    dna2_intensities: np.ndarray,
    resolution_um: float = 1.0,
    config: Optional['Config'] = None
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Prepare DNA composite for grid segmentation.
    
    For grid segmentation, we don't actually need the DNA composite since
    the grid is geometry-based, not content-based. However, we maintain
    the same interface for compatibility and return a simple composite
    for visualization purposes.
    
    Args:
        coords: Nx2 coordinate array in micrometers
        dna1_intensities: DNA1 channel intensities
        dna2_intensities: DNA2 channel intensities
        resolution_um: Output resolution in micrometers
        config: Configuration object (maintained for compatibility)
        
    Returns:
        Tuple of (composite_dna_image, bounds)
    """
    if len(coords) == 0:
        return np.array([[]]), (0, 0, 0, 0)
    
    # Determine spatial bounds
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    # Create output grid
    x_grid = np.arange(x_min, x_max + resolution_um, resolution_um)
    y_grid = np.arange(y_min, y_max + resolution_um, resolution_um)
    
    # Bin DNA intensities onto grid (simplified version of SLIC approach)
    x_bins = np.digitize(coords[:, 0], x_grid) - 1
    y_bins = np.digitize(coords[:, 1], y_grid) - 1
    
    # Create DNA fields
    dna_composite = np.zeros((len(y_grid), len(x_grid)))
    
    valid_mask = (x_bins >= 0) & (x_bins < len(x_grid)) & \
                 (y_bins >= 0) & (y_bins < len(y_grid))
    
    # Vectorized accumulation
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 0:
        combined_intensities = dna1_intensities[valid_indices] + dna2_intensities[valid_indices]
        np.add.at(dna_composite, (y_bins[valid_indices], x_bins[valid_indices]), 
                  combined_intensities)
    
    return dna_composite, (x_min, x_max, y_min, y_max)


def perform_grid_segmentation(
    composite_image: np.ndarray,
    target_bin_size_um: float = 20.0,
    resolution_um: float = 1.0,
    compactness: float = 10.0,  # Maintained for compatibility, not used
    sigma: float = 1.5,  # Maintained for compatibility, not used
    n_segments: Optional[int] = None,
    config: Optional['Config'] = None
) -> np.ndarray:
    """
    Perform grid segmentation (uniform grid overlay).
    
    This is the core difference from SLIC: instead of using image content
    to guide superpixel boundaries, we create a regular geometric grid.
    
    Args:
        composite_image: DNA composite image (used only for bounds)
        target_bin_size_um: Target grid cell size in micrometers
        resolution_um: Image resolution in micrometers/pixel
        compactness: Not used for grid (maintained for compatibility)
        sigma: Not used for grid (maintained for compatibility)
        n_segments: If provided, override target_bin_size_um calculation
        
    Returns:
        2D array of grid cell labels
    """
    if composite_image.size == 0:
        return np.array([])
    
    # Infer spatial bounds from image dimensions
    height, width = composite_image.shape
    bounds = (0, width * resolution_um, 0, height * resolution_um)
    
    # Override target scale if n_segments is specified
    if n_segments is not None:
        total_area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
        target_bin_size_um = np.sqrt(total_area / n_segments)
    
    # Create regular grid
    grid_labels, grid_info = create_regular_grid(bounds, target_bin_size_um, resolution_um)
    
    # Ensure output matches composite_image shape
    if grid_labels.shape != composite_image.shape:
        # Resize grid_labels to match composite_image shape
        from scipy.ndimage import zoom
        zoom_factors = (composite_image.shape[0] / grid_labels.shape[0],
                       composite_image.shape[1] / grid_labels.shape[1])
        grid_labels = zoom(grid_labels.astype(float), zoom_factors, order=0).astype(int)
    
    return grid_labels


def aggregate_to_grid_cells(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    grid_labels: np.ndarray,
    bounds: Tuple[float, float, float, float],
    resolution_um: float = 1.0
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Aggregate ion counts to grid cells.
    
    This uses the same vectorized aggregation approach as SLIC but with
    regular grid cells instead of irregular superpixels.
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        grid_labels: 2D array of grid cell assignments
        bounds: Spatial bounds (x_min, x_max, y_min, y_max)
        resolution_um: Resolution used for grid generation
        
    Returns:
        Tuple of (grid_aggregated_counts, grid_coords)
    """
    if len(coords) == 0 or grid_labels.size == 0:
        return {}, np.array([])
    
    x_min, x_max, y_min, y_max = bounds
    
    # Convert coordinates to pixel indices
    x_indices = ((coords[:, 0] - x_min) / resolution_um).astype(int)
    y_indices = ((coords[:, 1] - y_min) / resolution_um).astype(int)
    
    # Bounds checking
    valid_mask = (x_indices >= 0) & (x_indices < grid_labels.shape[1]) & \
                 (y_indices >= 0) & (y_indices < grid_labels.shape[0])
    
    # Get grid cell assignments for each measurement
    grid_assignments = np.full(len(coords), -1, dtype=int)
    valid_indices = np.where(valid_mask)[0]
    grid_assignments[valid_indices] = grid_labels[
        y_indices[valid_indices], x_indices[valid_indices]
    ]
    
    # Find unique grid cells
    unique_cells = np.unique(grid_assignments)
    unique_cells = unique_cells[unique_cells >= 0]  # Remove invalid (-1)
    
    if len(unique_cells) == 0:
        return {}, np.array([])
    
    # Vectorized aggregation using bincount (same as SLIC)
    cell_id_to_idx = {cell_id: i for i, cell_id in enumerate(unique_cells)}
    
    # Create index array for bincount
    valid_assignments = grid_assignments >= 0
    assignment_indices = np.full_like(grid_assignments, -1)
    for cell_id, idx in cell_id_to_idx.items():
        assignment_indices[grid_assignments == cell_id] = idx
    
    # Aggregate ion counts for each grid cell
    grid_counts = {}
    for protein_name, counts in ion_counts.items():
        valid_mask = assignment_indices >= 0
        if np.any(valid_mask):
            binned_counts = np.bincount(
                assignment_indices[valid_mask], 
                weights=counts[valid_mask], 
                minlength=len(unique_cells)
            )
            grid_counts[protein_name] = binned_counts
        else:
            grid_counts[protein_name] = np.zeros(len(unique_cells))
    
    # Compute grid cell centroid coordinates
    grid_coords = np.zeros((len(unique_cells), 2))
    for coord_dim in range(2):
        valid_mask = assignment_indices >= 0
        if np.any(valid_mask):
            coord_sums = np.bincount(
                assignment_indices[valid_mask],
                weights=coords[valid_mask, coord_dim],
                minlength=len(unique_cells)
            )
            coord_counts = np.bincount(
                assignment_indices[valid_mask],
                minlength=len(unique_cells)
            )
            nonzero_mask = coord_counts > 0
            grid_coords[nonzero_mask, coord_dim] = (
                coord_sums[nonzero_mask] / coord_counts[nonzero_mask]
            )
    
    return grid_counts, grid_coords


def compute_grid_properties(
    grid_labels: np.ndarray,
    composite_image: np.ndarray,
    resolution_um: float = 1.0
) -> Dict[int, Dict[str, float]]:
    """
    Compute morphological properties of each grid cell.
    
    Grid cells have perfect geometric properties (rectangles), so we compute
    theoretical values based on grid geometry plus actual DNA intensity.
    
    Args:
        grid_labels: 2D array of grid cell assignments
        composite_image: DNA composite image for intensity measurements
        resolution_um: Image resolution
        
    Returns:
        Dictionary mapping grid_cell_id -> properties
    """
    if grid_labels.size == 0:
        return {}
    
    unique_cells = np.unique(grid_labels)
    unique_cells = unique_cells[unique_cells >= 0]  # Remove background (-1)
    
    grid_props = {}
    
    for cell_id in unique_cells:
        cell_mask = grid_labels == cell_id
        
        # Compute actual area and perimeter from mask
        area_pixels = np.sum(cell_mask)
        area_um2 = area_pixels * (resolution_um ** 2)
        
        # For grid cells, compute theoretical perimeter
        # Find bounding box of the cell
        y_coords, x_coords = np.where(cell_mask)
        if len(y_coords) > 0:
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()
            
            # Theoretical perimeter for rectangular grid cell
            width_pixels = x_max - x_min + 1
            height_pixels = y_max - y_min + 1
            perimeter_pixels = 2 * (width_pixels + height_pixels)
            perimeter_um = perimeter_pixels * resolution_um
            
            # Grid cells are always rectangular, so eccentricity depends on aspect ratio
            aspect_ratio = width_pixels / height_pixels if height_pixels > 0 else 1.0
            # Eccentricity of rectangle: sqrt(1 - (min/max)²)
            if aspect_ratio >= 1:
                eccentricity = np.sqrt(1 - (1 / aspect_ratio) ** 2)
            else:
                eccentricity = np.sqrt(1 - aspect_ratio ** 2)
            
            # Grid cells are always perfectly solid rectangles
            solidity = 1.0
            
            # Compute mean DNA intensity
            mean_dna_intensity = float(np.mean(composite_image[cell_mask]))
            
            # Centroid in micrometers
            centroid_y_um = float(np.mean(y_coords) * resolution_um)
            centroid_x_um = float(np.mean(x_coords) * resolution_um)
            
            grid_props[int(cell_id)] = {
                'area_um2': float(area_um2),
                'perimeter_um': float(perimeter_um),
                'eccentricity': float(eccentricity),
                'solidity': solidity,
                'mean_dna_intensity': mean_dna_intensity,
                'centroid_x_um': centroid_x_um,
                'centroid_y_um': centroid_y_um,
                'grid_width_um': float(width_pixels * resolution_um),
                'grid_height_um': float(height_pixels * resolution_um),
                'aspect_ratio': float(aspect_ratio)
            }
    
    return grid_props


def compute_boundary_quality_metrics(
    grid_labels: np.ndarray,
    composite_image: np.ndarray,
    resolution_um: float = 1.0
) -> Dict[str, float]:
    """
    Compute boundary quality metrics for grid segmentation.
    
    Grid boundaries are geometric and don't follow image content, so we
    measure how well they align with actual tissue boundaries.
    
    Args:
        grid_labels: 2D array of grid cell assignments
        composite_image: DNA composite image
        resolution_um: Image resolution
        
    Returns:
        Dictionary of boundary quality metrics
    """
    if grid_labels.size == 0 or composite_image.size == 0:
        return {}
    
    from scipy import ndimage
    
    # Find grid boundaries
    grid_boundaries = np.zeros_like(grid_labels, dtype=bool)
    
    # Detect edges by finding pixels adjacent to different grid cells
    shifted_right = np.roll(grid_labels, 1, axis=1)
    shifted_down = np.roll(grid_labels, 1, axis=0)
    
    # Grid boundaries occur where neighboring pixels have different labels
    grid_boundaries |= (grid_labels != shifted_right)
    grid_boundaries |= (grid_labels != shifted_down)
    
    # Remove boundaries at image edges
    grid_boundaries[0, :] = False
    grid_boundaries[-1, :] = False
    grid_boundaries[:, 0] = False
    grid_boundaries[:, -1] = False
    
    # Find tissue boundaries in DNA image using gradient
    # Compute image gradients
    grad_x = ndimage.sobel(composite_image, axis=1)
    grad_y = ndimage.sobel(composite_image, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Define tissue boundaries as high-gradient regions
    if gradient_magnitude.max() > 0:
        gradient_threshold = np.percentile(gradient_magnitude[gradient_magnitude > 0], 80)
        tissue_boundaries = gradient_magnitude > gradient_threshold
    else:
        tissue_boundaries = np.zeros_like(composite_image, dtype=bool)
    
    # Compute alignment metrics
    n_grid_boundary_pixels = np.sum(grid_boundaries)
    n_tissue_boundary_pixels = np.sum(tissue_boundaries)
    
    if n_grid_boundary_pixels > 0 and n_tissue_boundary_pixels > 0:
        # Overlap between grid boundaries and tissue boundaries
        overlap = np.sum(grid_boundaries & tissue_boundaries)
        
        # Boundary coherence: how well grid boundaries align with tissue boundaries
        boundary_coherence = overlap / n_grid_boundary_pixels
        
        # Precision: of all grid boundaries, how many align with tissue
        boundary_precision = overlap / n_grid_boundary_pixels
        
        # Recall: of all tissue boundaries, how many are captured by grid
        boundary_recall = overlap / n_tissue_boundary_pixels
        
        # F1 score
        if boundary_precision + boundary_recall > 0:
            boundary_f1 = 2 * (boundary_precision * boundary_recall) / (boundary_precision + boundary_recall)
        else:
            boundary_f1 = 0.0
    else:
        boundary_coherence = 0.0
        boundary_precision = 0.0
        boundary_recall = 0.0
        boundary_f1 = 0.0
    
    # Compute additional grid-specific metrics
    unique_cells = np.unique(grid_labels[grid_labels >= 0])
    n_cells = len(unique_cells)
    
    if n_cells > 0:
        # Grid regularity (should be 1.0 for perfect grid)
        cell_areas = []
        for cell_id in unique_cells:
            cell_area = np.sum(grid_labels == cell_id)
            cell_areas.append(cell_area)
        
        area_cv = np.std(cell_areas) / np.mean(cell_areas) if np.mean(cell_areas) > 0 else 0
        grid_regularity = 1.0 / (1.0 + area_cv)  # Closer to 1.0 = more regular
    else:
        grid_regularity = 0.0
    
    return {
        'boundary_coherence': float(boundary_coherence),
        'boundary_precision': float(boundary_precision),
        'boundary_recall': float(boundary_recall),
        'boundary_f1': float(boundary_f1),
        'grid_regularity': float(grid_regularity),
        'n_grid_boundary_pixels': int(n_grid_boundary_pixels),
        'n_tissue_boundary_pixels': int(n_tissue_boundary_pixels),
        'boundary_overlap_pixels': int(overlap) if 'overlap' in locals() else 0
    }


def grid_pipeline(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    dna1_intensities: np.ndarray,
    dna2_intensities: np.ndarray,
    target_scale_um: float = 20.0,
    resolution_um: float = 1.0,
    compactness: float = 10.0,  # Maintained for compatibility, not used
    n_segments: Optional[int] = None,
    config: Optional['Config'] = None,
    cached_cofactors: Optional[Dict[str, float]] = None,
    compute_quality_metrics: bool = True
) -> Dict:
    """
    Complete grid-based spatial aggregation pipeline.
    
    This provides a drop-in replacement for slic_pipeline() with identical
    interface but using regular grid segmentation instead of superpixels.
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        dna1_intensities: DNA1 channel data
        dna2_intensities: DNA2 channel data
        target_scale_um: Target grid cell size
        resolution_um: Working resolution (default 1.0μm for IMC)
        compactness: Not used for grid (maintained for compatibility)
        n_segments: Direct specification of number of grid cells
        config: Configuration object
        cached_cofactors: Pre-computed arcsinh cofactors
        compute_quality_metrics: Whether to compute boundary quality metrics
        
    Returns:
        Dictionary with all pipeline results (same structure as slic_pipeline)
    """
    start_time = time.time()
    
    if len(coords) == 0 or not ion_counts:
        return {
            'superpixel_counts': {},  # Named for compatibility
            'superpixel_coords': np.array([]),
            'superpixel_labels': np.array([]),
            'superpixel_props': {},
            'composite_dna': np.array([]),
            'bounds': (0, 0, 0, 0),
            'method': 'grid',
            'metrics': GridMetrics(0, 0, 0, 0, 0, 0, 0)
        }
    
    # Step 1: Create DNA composite (for visualization compatibility)
    composite_dna, bounds = prepare_dna_composite_grid(
        coords, dna1_intensities, dna2_intensities, resolution_um, config
    )
    
    # Step 2: Perform grid segmentation
    if n_segments is not None:
        effective_n_segments = n_segments
    else:
        effective_n_segments = None
    
    grid_labels = perform_grid_segmentation(
        composite_dna, target_scale_um, resolution_um, 
        n_segments=effective_n_segments, config=config
    )
    
    aggregation_start = time.time()
    
    # Step 3: Aggregate ion counts to grid cells
    grid_counts, grid_coords = aggregate_to_grid_cells(
        coords, ion_counts, grid_labels, bounds, resolution_um
    )
    
    aggregation_time = time.time() - aggregation_start
    
    # Step 4: Compute grid cell properties
    grid_props = compute_grid_properties(
        grid_labels, composite_dna, resolution_um
    )
    
    # Step 5: Apply arcsinh transformation (same as SLIC)
    from .ion_count_processing import apply_arcsinh_transform, estimate_optimal_cofactor
    transformed_arrays = {}
    cofactors_used = {}
    
    if grid_counts:
        if cached_cofactors:
            cofactors_used = cached_cofactors
            for protein_name, counts in grid_counts.items():
                cofactor = cofactors_used.get(protein_name, 1.0)
                transformed_arrays[protein_name] = np.arcsinh(counts / cofactor)
        else:
            for protein_name, counts in grid_counts.items():
                cofactor = estimate_optimal_cofactor(counts)
                cofactors_used[protein_name] = cofactor
                transformed_arrays[protein_name] = np.arcsinh(counts / cofactor)
    
    # Step 6: Compute quality metrics
    boundary_metrics = {}
    if compute_quality_metrics:
        boundary_metrics = compute_boundary_quality_metrics(
            grid_labels, composite_dna, resolution_um
        )
    
    # Step 7: Create performance metrics
    total_time = time.time() - start_time
    n_grid_cells = len(np.unique(grid_labels[grid_labels >= 0]))
    
    # Estimate memory usage (approximate)
    memory_usage_mb = (
        grid_labels.nbytes + 
        composite_dna.nbytes +
        sum(arr.nbytes for arr in grid_counts.values()) +
        sum(arr.nbytes for arr in transformed_arrays.values())
    ) / (1024**2)
    
    metrics = GridMetrics(
        generation_time=total_time - aggregation_time,
        aggregation_time=aggregation_time,
        n_grid_cells=n_grid_cells,
        grid_size_um=target_scale_um,
        coverage_ratio=1.0,  # Grid always covers 100%
        boundary_coherence=boundary_metrics.get('boundary_coherence', 0.0),
        memory_usage_mb=memory_usage_mb
    )
    
    # Return dictionary with same structure as slic_pipeline for compatibility
    return {
        'superpixel_counts': grid_counts,  # Named for compatibility
        'superpixel_coords': grid_coords,
        'superpixel_labels': grid_labels,
        'superpixel_props': grid_props,
        'composite_dna': composite_dna,
        'bounds': bounds,
        'resolution_um': resolution_um,
        'target_scale_um': target_scale_um,
        'n_segments_used': n_grid_cells,
        'transformed_arrays': transformed_arrays,
        'cofactors_used': cofactors_used,
        'method': 'grid',
        'metrics': metrics,
        'boundary_quality': boundary_metrics,
        'performance_comparison': {
            'grid_generation_time': metrics.generation_time,
            'aggregation_time': metrics.aggregation_time,
            'total_time': total_time,
            'memory_usage_mb': memory_usage_mb,
            'cells_per_second': n_grid_cells / total_time if total_time > 0 else 0
        }
    }


def compare_grid_vs_slic(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    dna1_intensities: np.ndarray,
    dna2_intensities: np.ndarray,
    target_scale_um: float = 20.0,
    config: Optional['Config'] = None
) -> Dict[str, Dict]:
    """
    Direct comparison between grid and SLIC segmentation.
    
    Runs both methods on identical data and compares performance and quality.
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        dna1_intensities: DNA1 channel data
        dna2_intensities: DNA2 channel data
        target_scale_um: Target scale for comparison
        config: Configuration object
        
    Returns:
        Dictionary with comparison results
    """
    from .slic_segmentation import slic_pipeline
    
    # Run grid segmentation
    grid_start = time.time()
    grid_results = grid_pipeline(
        coords, ion_counts, dna1_intensities, dna2_intensities,
        target_scale_um=target_scale_um, config=config
    )
    grid_time = time.time() - grid_start
    
    # Run SLIC segmentation
    slic_start = time.time()
    slic_results = slic_pipeline(
        coords, ion_counts, dna1_intensities, dna2_intensities,
        target_scale_um=target_scale_um, config=config
    )
    slic_time = time.time() - slic_start
    
    # Compare results
    comparison = {
        'grid_results': grid_results,
        'slic_results': slic_results,
        'performance_comparison': {
            'grid_time': grid_time,
            'slic_time': slic_time,
            'speedup_factor': slic_time / grid_time if grid_time > 0 else float('inf'),
            'grid_segments': grid_results['n_segments_used'],
            'slic_segments': slic_results['n_segments_used'],
            'segment_ratio': slic_results['n_segments_used'] / grid_results['n_segments_used'] 
                           if grid_results['n_segments_used'] > 0 else 0
        },
        'quality_comparison': {
            'grid_boundary_coherence': grid_results.get('boundary_quality', {}).get('boundary_coherence', 0),
            'grid_regularity': grid_results.get('boundary_quality', {}).get('grid_regularity', 0),
            'method_comparison': 'Grid provides regular geometry vs SLIC adapts to morphology'
        },
        'aggregation_comparison': {
            'identical_interface': True,
            'same_protein_processing': True,
            'same_arcsinh_transform': True,
            'difference': 'Spatial aggregation method only'
        }
    }
    
    return comparison


def benchmark_grid_performance(
    test_configs: List[Dict],
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    dna1_intensities: np.ndarray,
    dna2_intensities: np.ndarray
) -> Dict[str, List]:
    """
    Benchmark grid segmentation performance across different configurations.
    
    Args:
        test_configs: List of test configurations (scale, n_segments, etc.)
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        dna1_intensities: DNA1 channel data
        dna2_intensities: DNA2 channel data
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'scales': [],
        'times': [],
        'memory_usage': [],
        'n_segments': [],
        'boundary_coherence': [],
        'grid_regularity': []
    }
    
    for config_dict in test_configs:
        scale = config_dict.get('target_scale_um', 20.0)
        n_segments = config_dict.get('n_segments', None)
        
        # Run grid pipeline
        result = grid_pipeline(
            coords, ion_counts, dna1_intensities, dna2_intensities,
            target_scale_um=scale, n_segments=n_segments
        )
        
        # Collect metrics
        results['scales'].append(scale)
        results['times'].append(result['performance_comparison']['total_time'])
        results['memory_usage'].append(result['performance_comparison']['memory_usage_mb'])
        results['n_segments'].append(result['n_segments_used'])
        results['boundary_coherence'].append(
            result.get('boundary_quality', {}).get('boundary_coherence', 0)
        )
        results['grid_regularity'].append(
            result.get('boundary_quality', {}).get('grid_regularity', 0)
        )
    
    return results