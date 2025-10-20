"""
Watershed DNA Segmentation for IMC Data

Nucleus-based cell boundary detection using DNA channel information.
Provides drop-in compatibility with SLIC superpixel and grid-based methods.
Enables biologically-motivated spatial analysis of tissue organization.

Key Features:
- DNA1/DNA2 channel fusion for robust nucleus detection
- Adaptive threshold selection for variable tissue types
- Distance transform-based watershed for cell boundary inference
- Quality metrics for segmentation validation
- Compatible interface with existing aggregation pipeline
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, TYPE_CHECKING
import warnings

try:
    from skimage.segmentation import watershed
    from skimage.measure import regionprops, label
    from skimage.feature import peak_local_max
    from skimage.morphology import disk, remove_small_objects
    from skimage.filters import gaussian, threshold_otsu, threshold_local
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    watershed = None
    regionprops = None
    label = None
    peak_local_max = None
    disk = None
    remove_small_objects = None
    gaussian = None
    threshold_otsu = None
    threshold_local = None

try:
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ndimage = None
    cdist = None

if TYPE_CHECKING:
    from ..config import Config


def prepare_dna_for_nucleus_detection(
    coords: np.ndarray,
    dna1_intensities: np.ndarray,
    dna2_intensities: np.ndarray,
    resolution_um: float = 1.0,
    config: Optional['Config'] = None
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Prepare DNA composite optimized for nucleus detection.
    
    Unlike SLIC preparation, this focuses on enhancing nuclear structure
    while minimizing cytoplasmic background for watershed segmentation.
    
    Args:
        coords: Nx2 coordinate array in micrometers
        dna1_intensities: DNA1 channel intensities
        dna2_intensities: DNA2 channel intensities
        resolution_um: Output resolution in micrometers
        config: Configuration object
        
    Returns:
        Tuple of (nucleus_optimized_image, bounds)
    """
    if len(coords) == 0:
        return np.array([[]]), (0, 0, 0, 0)
    
    # Determine spatial bounds
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    # Create output grid
    x_grid = np.arange(x_min, x_max + resolution_um, resolution_um)
    y_grid = np.arange(y_min, y_max + resolution_um, resolution_um)
    
    # Bin DNA intensities onto grid
    x_bins = np.digitize(coords[:, 0], x_grid) - 1
    y_bins = np.digitize(coords[:, 1], y_grid) - 1
    
    # Create DNA fields
    dna1_field = np.zeros((len(y_grid), len(x_grid)))
    dna2_field = np.zeros((len(y_grid), len(x_grid)))
    
    valid_mask = (x_bins >= 0) & (x_bins < len(x_grid)) & \
                 (y_bins >= 0) & (y_bins < len(y_grid))
    
    # Vectorized accumulation
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 0:
        np.add.at(dna1_field, (y_bins[valid_indices], x_bins[valid_indices]), 
                  dna1_intensities[valid_indices])
        np.add.at(dna2_field, (y_bins[valid_indices], x_bins[valid_indices]), 
                  dna2_intensities[valid_indices])
    
    # Create nucleus-optimized composite
    # Use geometric mean to emphasize regions with both DNA1 and DNA2 signal
    # This reduces cytoplasmic artifacts that may have only one DNA channel
    composite_dna = np.sqrt(dna1_field * dna2_field + dna1_field + dna2_field)
    
    # Apply nucleus-specific preprocessing
    if composite_dna.size > 0 and composite_dna.max() > 0:
        # Get configuration for nucleus-specific processing
        if config is not None and hasattr(config, 'segmentation'):
            watershed_config = config.segmentation.get('watershed', {})
            dna_config = watershed_config.get('dna_processing', {})
        else:
            # Default nucleus detection parameters
            dna_config = {
                'noise_reduction': True,
                'noise_sigma': 0.5,
                'enhance_contrast': True,
                'arcsinh_transform': True,
                'cofactor_multiplier': 2.0
            }
        
        # Noise reduction for cleaner nucleus detection
        if dna_config.get('noise_reduction', True):
            sigma = dna_config.get('noise_sigma', 0.5)
            if SKIMAGE_AVAILABLE and gaussian is not None:
                composite_dna = gaussian(composite_dna, sigma=sigma, preserve_range=True)
        
        # Apply ArcSinh transformation optimized for nucleus detection
        if dna_config.get('arcsinh_transform', True):
            # Use more aggressive transformation for nuclear signals
            all_vals = composite_dna[composite_dna > 0]
            if len(all_vals) > 0:
                # Use median of positive values as cofactor base
                cofactor_base = np.median(all_vals)
                multiplier = dna_config.get('cofactor_multiplier', 2.0)
                cofactor = cofactor_base * multiplier
                
                # Apply transformation
                composite_dna = np.arcsinh(composite_dna / cofactor)
        
        # Contrast enhancement for nucleus detection
        if dna_config.get('enhance_contrast', True):
            # Apply histogram stretching to enhance nuclear contrast
            p1, p99 = np.percentile(composite_dna[composite_dna > 0], [1, 99])
            if p99 > p1:
                composite_dna = np.clip((composite_dna - p1) / (p99 - p1), 0, 1)
    
    return composite_dna, (x_min, x_max, y_min, y_max)


def detect_nucleus_seeds(
    dna_image: np.ndarray,
    min_nucleus_size_um2: float = 25.0,
    max_nucleus_size_um2: float = 400.0,
    resolution_um: float = 1.0,
    config: Optional['Config'] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Detect nucleus seeds for watershed segmentation.
    
    Uses adaptive thresholding and morphological filtering to identify
    high-confidence nucleus centers while excluding artifacts.
    
    Args:
        dna_image: DNA composite image
        min_nucleus_size_um2: Minimum nucleus area in square micrometers
        max_nucleus_size_um2: Maximum nucleus area in square micrometers
        resolution_um: Image resolution
        config: Configuration object
        
    Returns:
        Tuple of (nucleus_seeds_labeled, detection_stats)
    """
    if not SKIMAGE_AVAILABLE or not SCIPY_AVAILABLE:
        raise ImportError("Watershed segmentation requires scikit-image and scipy")
    
    if dna_image.size == 0:
        return np.array([]), {}
    
    # Get watershed configuration
    if config is not None and hasattr(config, 'segmentation'):
        watershed_config = config.segmentation.get('watershed', {})
        nucleus_config = watershed_config.get('nucleus_detection', {})
    else:
        nucleus_config = {}
    
    # Convert size constraints to pixels
    pixel_area = resolution_um ** 2
    min_nucleus_pixels = int(min_nucleus_size_um2 / pixel_area)
    max_nucleus_pixels = int(max_nucleus_size_um2 / pixel_area)
    
    # Adaptive threshold selection
    threshold_method = nucleus_config.get('threshold_method', 'adaptive_local')
    
    if threshold_method == 'otsu' and np.max(dna_image) > np.min(dna_image):
        # Global Otsu threshold
        threshold = threshold_otsu(dna_image)
    elif threshold_method == 'adaptive_local':
        # Local adaptive threshold - better for variable illumination
        block_size = nucleus_config.get('adaptive_block_size', 51)
        # Ensure odd block size
        if block_size % 2 == 0:
            block_size += 1
        threshold_image = threshold_local(dna_image, block_size=block_size, method='gaussian')
        binary_mask = dna_image > threshold_image
    else:
        # Percentile-based threshold
        threshold_percentile = nucleus_config.get('threshold_percentile', 75)
        threshold = np.percentile(dna_image[dna_image > 0], threshold_percentile)
    
    # Create binary mask
    if threshold_method != 'adaptive_local':
        binary_mask = dna_image > threshold
    
    # Morphological cleanup
    if SKIMAGE_AVAILABLE:
        # Remove small artifacts
        min_artifact_size = max(1, min_nucleus_pixels // 4)
        binary_mask = remove_small_objects(binary_mask, min_size=min_artifact_size)
        
        # Morphological opening to separate touching nuclei
        opening_radius = nucleus_config.get('opening_radius_um', 1.0)
        opening_pixels = max(1, int(opening_radius / resolution_um))
        if opening_pixels > 0:
            selem = disk(opening_pixels)
            binary_mask = ndimage.binary_opening(binary_mask, structure=selem)
    
    # Find nucleus seeds using distance transform + local maxima
    if np.any(binary_mask):
        # Distance transform to find nucleus centers
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Find local maxima as nucleus seeds
        min_distance = nucleus_config.get('min_seed_distance_um', 3.0)
        min_distance_pixels = max(1, int(min_distance / resolution_um))
        
        # Find peaks in distance transform
        coordinates = peak_local_max(
            distance, 
            min_distance=min_distance_pixels,
            threshold_abs=min_distance_pixels * 0.5,  # Minimum distance value
            exclude_border=True
        )
        
        # Create seed markers
        nucleus_seeds = np.zeros_like(dna_image, dtype=int)
        if len(coordinates) > 0:
            # Label each seed with unique ID
            for i, (y, x) in enumerate(coordinates):
                nucleus_seeds[y, x] = i + 1
    else:
        nucleus_seeds = np.zeros_like(dna_image, dtype=int)
        coordinates = []
    
    # Quality assessment
    n_seeds = len(coordinates) if coordinates is not None else 0
    tissue_area = np.sum(binary_mask) * pixel_area
    nucleus_density = n_seeds / tissue_area if tissue_area > 0 else 0
    
    detection_stats = {
        'n_nucleus_seeds': n_seeds,
        'tissue_area_um2': tissue_area,
        'nucleus_density_per_um2': nucleus_density,
        'min_nucleus_size_um2': min_nucleus_size_um2,
        'max_nucleus_size_um2': max_nucleus_size_um2,
        'threshold_method': threshold_method,
        'binary_mask_coverage': np.sum(binary_mask) / binary_mask.size
    }
    
    return nucleus_seeds, detection_stats


def perform_watershed_segmentation(
    dna_image: np.ndarray,
    nucleus_seeds: np.ndarray,
    max_cell_radius_um: float = 15.0,
    resolution_um: float = 1.0,
    config: Optional['Config'] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Perform watershed segmentation to define cell boundaries.
    
    Uses nucleus seeds to grow cell regions based on DNA signal gradients,
    creating biologically-motivated cell boundary segmentation.
    
    Args:
        dna_image: DNA composite image
        nucleus_seeds: Labeled nucleus seed points
        max_cell_radius_um: Maximum cell radius in micrometers
        resolution_um: Image resolution
        config: Configuration object
        
    Returns:
        Tuple of (cell_labels, segmentation_stats)
    """
    if not SKIMAGE_AVAILABLE or not SCIPY_AVAILABLE:
        raise ImportError("Watershed segmentation requires scikit-image and scipy")
    
    if dna_image.size == 0 or nucleus_seeds.size == 0:
        return np.array([]), {}
    
    # Get watershed configuration
    if config is not None and hasattr(config, 'segmentation'):
        watershed_config = config.segmentation.get('watershed', {})
        cell_config = watershed_config.get('cell_boundary', {})
    else:
        cell_config = {}
    
    # Create watershed mask - limit cell growth area
    max_radius_pixels = int(max_cell_radius_um / resolution_um)
    
    # Create distance-limited mask using erosion from tissue boundaries
    tissue_mask = dna_image > 0
    if np.any(tissue_mask):
        # Erode tissue mask to create watershed boundary
        boundary_erosion = cell_config.get('boundary_erosion_um', 2.0)
        erosion_pixels = max(1, int(boundary_erosion / resolution_um))
        watershed_mask = ndimage.binary_erosion(tissue_mask, iterations=erosion_pixels)
    else:
        watershed_mask = tissue_mask
    
    # Prepare elevation map for watershed
    # Use inverted DNA signal so watershed flows from nuclei outward
    elevation_mode = cell_config.get('elevation_mode', 'inverted_dna')
    
    if elevation_mode == 'inverted_dna':
        # Invert DNA signal - watersheds flow from low to high
        if dna_image.max() > 0:
            elevation = dna_image.max() - dna_image
        else:
            elevation = -dna_image
    elif elevation_mode == 'distance_transform':
        # Use distance from tissue boundaries
        elevation = ndimage.distance_transform_edt(~tissue_mask)
    else:
        # Combined approach: inverted DNA + distance transform
        if dna_image.max() > 0:
            inv_dna = dna_image.max() - dna_image
        else:
            inv_dna = -dna_image
        dist_transform = ndimage.distance_transform_edt(~tissue_mask)
        # Normalize and combine
        if inv_dna.max() > 0:
            inv_dna = inv_dna / inv_dna.max()
        if dist_transform.max() > 0:
            dist_transform = dist_transform / dist_transform.max()
        elevation = 0.7 * inv_dna + 0.3 * dist_transform
    
    # Apply smoothing to elevation map for better cell boundaries
    elevation_smoothing = cell_config.get('elevation_smoothing_um', 1.0)
    if elevation_smoothing > 0:
        smoothing_sigma = elevation_smoothing / resolution_um
        if SKIMAGE_AVAILABLE and gaussian is not None:
            elevation = gaussian(elevation, sigma=smoothing_sigma, preserve_range=True)
    
    # Perform watershed segmentation
    if np.any(nucleus_seeds > 0):
        cell_labels = watershed(
            elevation,
            markers=nucleus_seeds,
            mask=watershed_mask,
            compactness=cell_config.get('compactness', 0.1)
        )
    else:
        cell_labels = np.zeros_like(dna_image, dtype=int)
    
    # Post-process cell segmentation
    if np.any(cell_labels > 0):
        # Remove cells that are too small or too large
        cell_props = regionprops(cell_labels)
        pixel_area = resolution_um ** 2
        
        min_cell_area = cell_config.get('min_cell_area_um2', 10.0) / pixel_area
        max_cell_area = cell_config.get('max_cell_area_um2', 500.0) / pixel_area
        
        for prop in cell_props:
            if prop.area < min_cell_area or prop.area > max_cell_area:
                cell_labels[cell_labels == prop.label] = 0
        
        # Relabel to remove gaps
        cell_labels = label(cell_labels > 0)
    
    # Calculate segmentation statistics
    final_props = regionprops(cell_labels)
    if final_props:
        cell_areas = [prop.area * (resolution_um ** 2) for prop in final_props]
        cell_eccentricities = [prop.eccentricity for prop in final_props]
    else:
        cell_areas = []
        cell_eccentricities = []
    
    segmentation_stats = {
        'n_cells_detected': len(final_props),
        'mean_cell_area_um2': np.mean(cell_areas) if cell_areas else 0,
        'std_cell_area_um2': np.std(cell_areas) if cell_areas else 0,
        'mean_eccentricity': np.mean(cell_eccentricities) if cell_eccentricities else 0,
        'watershed_mask_coverage': np.sum(watershed_mask) / watershed_mask.size,
        'segmentation_coverage': np.sum(cell_labels > 0) / cell_labels.size,
        'elevation_mode': elevation_mode
    }
    
    return cell_labels, segmentation_stats


def aggregate_to_cells(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    cell_labels: np.ndarray,
    bounds: Tuple[float, float, float, float],
    resolution_um: float = 1.0
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Aggregate ion counts to cell regions.
    
    Compatible interface with superpixel aggregation for drop-in replacement.
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        cell_labels: 2D array of cell assignments
        bounds: Spatial bounds (x_min, x_max, y_min, y_max)
        resolution_um: Resolution used for cell segmentation
        
    Returns:
        Tuple of (cell_aggregated_counts, cell_coords)
    """
    if len(coords) == 0 or cell_labels.size == 0:
        return {}, np.array([])
    
    x_min, x_max, y_min, y_max = bounds
    
    # Convert coordinates to pixel indices
    x_indices = ((coords[:, 0] - x_min) / resolution_um).astype(int)
    y_indices = ((coords[:, 1] - y_min) / resolution_um).astype(int)
    
    # Bounds checking
    valid_mask = (x_indices >= 0) & (x_indices < cell_labels.shape[1]) & \
                 (y_indices >= 0) & (y_indices < cell_labels.shape[0])
    
    # Get cell assignments for each measurement
    cell_assignments = np.full(len(coords), -1, dtype=int)
    valid_indices = np.where(valid_mask)[0]
    cell_assignments[valid_indices] = cell_labels[
        y_indices[valid_indices], x_indices[valid_indices]
    ]
    
    # Find unique cells
    unique_cells = np.unique(cell_assignments)
    unique_cells = unique_cells[unique_cells > 0]  # Remove background (0) and invalid (-1)
    
    if len(unique_cells) == 0:
        return {}, np.array([])
    
    # Vectorized aggregation using bincount
    cell_id_to_idx = {cell_id: i for i, cell_id in enumerate(unique_cells)}
    max_cell_id = int(np.max(unique_cells))
    
    # Create index array for bincount
    assignment_indices = np.full_like(cell_assignments, -1)
    for cell_id, idx in cell_id_to_idx.items():
        assignment_indices[cell_assignments == cell_id] = idx
    
    # Aggregate ion counts for each cell
    cell_counts = {}
    for protein_name, counts in ion_counts.items():
        valid_mask = assignment_indices >= 0
        if np.any(valid_mask):
            binned_counts = np.bincount(
                assignment_indices[valid_mask], 
                weights=counts[valid_mask], 
                minlength=len(unique_cells)
            )
            cell_counts[protein_name] = binned_counts
        else:
            cell_counts[protein_name] = np.zeros(len(unique_cells))
    
    # Compute cell centroid coordinates
    cell_coords = np.zeros((len(unique_cells), 2))
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
            cell_coords[nonzero_mask, coord_dim] = (
                coord_sums[nonzero_mask] / coord_counts[nonzero_mask]
            )
    
    return cell_counts, cell_coords


def compute_cell_properties(
    cell_labels: np.ndarray,
    dna_image: np.ndarray,
    resolution_um: float = 1.0
) -> Dict[int, Dict[str, float]]:
    """
    Compute morphological properties of each detected cell.
    
    Args:
        cell_labels: 2D array of cell assignments
        dna_image: Original DNA composite image
        resolution_um: Image resolution
        
    Returns:
        Dictionary mapping cell_id -> properties
    """
    if cell_labels.size == 0:
        return {}
    
    # Use regionprops to compute properties
    props = regionprops(cell_labels, intensity_image=dna_image)
    
    cell_props = {}
    for prop in props:
        cell_id = prop.label
        
        # Convert pixel measurements to micrometers
        area_um2 = prop.area * (resolution_um ** 2)
        perimeter_um = prop.perimeter * resolution_um
        
        # Additional cell-specific properties
        major_axis_um = prop.major_axis_length * resolution_um
        minor_axis_um = prop.minor_axis_length * resolution_um
        
        cell_props[cell_id] = {
            'area_um2': float(area_um2),
            'perimeter_um': float(perimeter_um),
            'eccentricity': float(prop.eccentricity),
            'solidity': float(prop.solidity),
            'circularity': float(4 * np.pi * area_um2 / (perimeter_um ** 2)) if perimeter_um > 0 else 0,
            'aspect_ratio': float(major_axis_um / minor_axis_um) if minor_axis_um > 0 else 1,
            'major_axis_um': float(major_axis_um),
            'minor_axis_um': float(minor_axis_um),
            'mean_dna_intensity': float(prop.mean_intensity),
            'centroid_x_um': float(prop.centroid[1] * resolution_um),
            'centroid_y_um': float(prop.centroid[0] * resolution_um)
        }
    
    return cell_props


def assess_watershed_quality(
    cell_labels: np.ndarray,
    dna_image: np.ndarray,
    nucleus_seeds: np.ndarray,
    detection_stats: Dict,
    segmentation_stats: Dict,
    config: Optional['Config'] = None
) -> Dict[str, any]:
    """
    Assess quality of watershed segmentation for validation.
    
    Args:
        cell_labels: Final cell segmentation
        dna_image: DNA composite image
        nucleus_seeds: Nucleus seed points
        detection_stats: Nucleus detection statistics
        segmentation_stats: Segmentation statistics
        config: Configuration object
        
    Returns:
        Dictionary with quality assessment results
    """
    # Get quality thresholds from config
    if config is not None and hasattr(config, 'quality_control'):
        qc_config = config.quality_control.get('watershed_segmentation', {})
    else:
        qc_config = {}
    
    # Basic statistics
    n_seeds = detection_stats['n_nucleus_seeds']
    n_cells = segmentation_stats['n_cells_detected']
    
    # Assess seed-to-cell conversion rate
    seed_to_cell_ratio = n_cells / n_seeds if n_seeds > 0 else 0
    
    # Expected thresholds
    min_seed_to_cell_ratio = qc_config.get('min_seed_to_cell_ratio', 0.5)
    max_seed_to_cell_ratio = qc_config.get('max_seed_to_cell_ratio', 1.2)
    
    # Assess cell morphology distribution
    cell_props = regionprops(cell_labels)
    if cell_props:
        eccentricities = [prop.eccentricity for prop in cell_props]
        solidities = [prop.solidity for prop in cell_props]
        areas = [prop.area for prop in cell_props]
        
        mean_eccentricity = np.mean(eccentricities)
        mean_solidity = np.mean(solidities)
        cv_area = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
    else:
        mean_eccentricity = 0
        mean_solidity = 0
        cv_area = 0
    
    # Quality flags
    good_seed_conversion = min_seed_to_cell_ratio <= seed_to_cell_ratio <= max_seed_to_cell_ratio
    reasonable_morphology = mean_eccentricity < qc_config.get('max_mean_eccentricity', 0.8)
    good_solidity = mean_solidity > qc_config.get('min_mean_solidity', 0.6)
    reasonable_size_variation = cv_area < qc_config.get('max_area_cv', 2.0)
    
    # Overall quality assessment
    quality_score = sum([
        good_seed_conversion,
        reasonable_morphology,
        good_solidity,
        reasonable_size_variation
    ]) / 4.0
    
    quality_assessment = {
        'overall_quality_score': quality_score,
        'quality_flags': {
            'good_seed_conversion': good_seed_conversion,
            'reasonable_morphology': reasonable_morphology,
            'good_solidity': good_solidity,
            'reasonable_size_variation': reasonable_size_variation
        },
        'metrics': {
            'seed_to_cell_ratio': seed_to_cell_ratio,
            'mean_eccentricity': mean_eccentricity,
            'mean_solidity': mean_solidity,
            'area_cv': cv_area,
            'n_seeds': n_seeds,
            'n_cells': n_cells
        },
        'thresholds_used': {
            'min_seed_to_cell_ratio': min_seed_to_cell_ratio,
            'max_seed_to_cell_ratio': max_seed_to_cell_ratio,
            'max_mean_eccentricity': qc_config.get('max_mean_eccentricity', 0.8),
            'min_mean_solidity': qc_config.get('min_mean_solidity', 0.6),
            'max_area_cv': qc_config.get('max_area_cv', 2.0)
        },
        'detection_stats': detection_stats,
        'segmentation_stats': segmentation_stats
    }
    
    return quality_assessment


def watershed_pipeline(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    dna1_intensities: np.ndarray,
    dna2_intensities: np.ndarray,
    target_scale_um: float = 20.0,  # For compatibility with multiscale analysis
    resolution_um: float = 1.0,
    min_nucleus_size_um2: float = 25.0,
    max_nucleus_size_um2: float = 400.0,
    max_cell_radius_um: float = 15.0,
    config: Optional['Config'] = None,
    cached_cofactors: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Complete watershed-based cell segmentation pipeline.
    
    Provides drop-in compatibility with SLIC superpixel pipeline interface
    while using nucleus-based watershed segmentation for biologically-motivated
    spatial analysis.
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        dna1_intensities: DNA1 channel data
        dna2_intensities: DNA2 channel data
        target_scale_um: Target analysis scale (for compatibility)
        resolution_um: Working resolution (default 1.0μm for IMC)
        min_nucleus_size_um2: Minimum nucleus area
        max_nucleus_size_um2: Maximum nucleus area
        max_cell_radius_um: Maximum cell radius
        config: Configuration object
        cached_cofactors: Pre-computed cofactors for arcsinh transform
        
    Returns:
        Dictionary with all pipeline results (compatible with SLIC interface)
    """
    if not SKIMAGE_AVAILABLE or not SCIPY_AVAILABLE:
        raise ImportError(
            "Watershed segmentation requires scikit-image and scipy. "
            "Install with: pip install scikit-image scipy"
        )
    
    if len(coords) == 0 or not ion_counts:
        return {
            'cell_counts': {},
            'cell_coords': np.array([]),
            'cell_labels': np.array([]),
            'cell_props': {},
            'composite_dna': np.array([]),
            'bounds': (0, 0, 0, 0),
            'nucleus_seeds': np.array([]),
            'quality_assessment': {},
            'detection_stats': {},
            'segmentation_stats': {}
        }
    
    # Step 1: Prepare DNA composite optimized for nucleus detection
    composite_dna, bounds = prepare_dna_for_nucleus_detection(
        coords, dna1_intensities, dna2_intensities, resolution_um, config
    )
    
    # Step 2: Detect nucleus seeds
    nucleus_seeds, detection_stats = detect_nucleus_seeds(
        composite_dna, min_nucleus_size_um2, max_nucleus_size_um2, resolution_um, config
    )
    
    # Step 3: Perform watershed segmentation
    cell_labels, segmentation_stats = perform_watershed_segmentation(
        composite_dna, nucleus_seeds, max_cell_radius_um, resolution_um, config
    )
    
    # Step 4: Aggregate ion counts to cells
    cell_counts, cell_coords = aggregate_to_cells(
        coords, ion_counts, cell_labels, bounds, resolution_um
    )
    
    # Step 5: Compute cell morphological properties
    cell_props = compute_cell_properties(cell_labels, composite_dna, resolution_um)
    
    # Step 6: Quality assessment
    quality_assessment = assess_watershed_quality(
        cell_labels, composite_dna, nucleus_seeds, 
        detection_stats, segmentation_stats, config
    )
    
    # Step 7: Apply arcsinh transformation to aggregated counts
    from .ion_count_processing import apply_arcsinh_transform, estimate_optimal_cofactor
    transformed_arrays = {}
    cofactors_used = {}
    
    if cell_counts:
        if cached_cofactors:
            cofactors_used = cached_cofactors
            for protein_name, counts in cell_counts.items():
                cofactor = cofactors_used.get(protein_name, 1.0)
                transformed_arrays[protein_name] = np.arcsinh(counts / cofactor)
        else:
            for protein_name, counts in cell_counts.items():
                cofactor = estimate_optimal_cofactor(counts)
                cofactors_used[protein_name] = cofactor
                transformed_arrays[protein_name] = np.arcsinh(counts / cofactor)
    
    # Return results in compatible format
    return {
        # Main results (compatible with SLIC interface)
        'superpixel_counts': cell_counts,  # Aliased for compatibility
        'superpixel_coords': cell_coords,  # Aliased for compatibility
        'superpixel_labels': cell_labels,  # Aliased for compatibility
        'superpixel_props': cell_props,    # Aliased for compatibility
        
        # Watershed-specific results
        'cell_counts': cell_counts,
        'cell_coords': cell_coords,
        'cell_labels': cell_labels,
        'cell_props': cell_props,
        'nucleus_seeds': nucleus_seeds,
        
        # Shared results
        'composite_dna': composite_dna,
        'bounds': bounds,
        'resolution_um': resolution_um,
        'target_scale_um': target_scale_um,
        'transformed_arrays': transformed_arrays,
        'cofactors_used': cofactors_used,
        
        # Quality and statistics
        'quality_assessment': quality_assessment,
        'detection_stats': detection_stats,
        'segmentation_stats': segmentation_stats,
        
        # Pipeline metadata
        'method': 'watershed',
        'n_segments_used': len(np.unique(cell_labels[cell_labels > 0])) if cell_labels.size > 0 else 0,
        'processing_parameters': {
            'min_nucleus_size_um2': min_nucleus_size_um2,
            'max_nucleus_size_um2': max_nucleus_size_um2,
            'max_cell_radius_um': max_cell_radius_um,
            'resolution_um': resolution_um
        }
    }