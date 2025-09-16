"""
SLIC Superpixel Segmentation for IMC Data

Morphology-aware spatial binning using DNA channel information.
Replaces arbitrary square binning with tissue-structure-aware segmentation.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from skimage.segmentation import slic
from skimage.measure import regionprops
from scipy import ndimage


def prepare_dna_composite(
    coords: np.ndarray,
    dna1_intensities: np.ndarray,
    dna2_intensities: np.ndarray,
    resolution_um: float = 1.0,
    sigma_um: float = 2.0
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Create composite DNA image for SLIC segmentation.
    
    Uses both DNA channels to create high-resolution morphology map.
    
    Args:
        coords: Nx2 coordinate array in micrometers
        dna1_intensities: DNA1 channel intensities
        dna2_intensities: DNA2 channel intensities
        resolution_um: Output resolution in micrometers
        sigma_um: Gaussian smoothing sigma
        
    Returns:
        Tuple of (composite_dna_image, bounds)
    """
    if len(coords) == 0:
        return np.array([[]]), (0, 0, 0, 0)
    
    # Determine spatial bounds with buffer
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    buffer = 10 * resolution_um
    x_min -= buffer
    x_max += buffer
    y_min -= buffer
    y_max += buffer
    
    # Create output grid
    x_grid = np.arange(x_min, x_max + resolution_um, resolution_um)
    y_grid = np.arange(y_min, y_max + resolution_um, resolution_um)
    
    # Bin DNA intensities onto grid
    x_bins = np.digitize(coords[:, 0], x_grid) - 1
    y_bins = np.digitize(coords[:, 1], y_grid) - 1
    
    # Create DNA fields by accumulating intensities
    dna1_field = np.zeros((len(y_grid), len(x_grid)))
    dna2_field = np.zeros((len(y_grid), len(x_grid)))
    
    valid_mask = (x_bins >= 0) & (x_bins < len(x_grid)) & \
                 (y_bins >= 0) & (y_bins < len(y_grid))
    
    for i in np.where(valid_mask)[0]:
        dna1_field[y_bins[i], x_bins[i]] += dna1_intensities[i]
        dna2_field[y_bins[i], x_bins[i]] += dna2_intensities[i]
    
    # Apply Gaussian smoothing
    sigma_pixels = sigma_um / resolution_um
    dna1_smooth = ndimage.gaussian_filter(dna1_field, sigma=sigma_pixels)
    dna2_smooth = ndimage.gaussian_filter(dna2_field, sigma=sigma_pixels)
    
    # Create composite: sum of both channels
    composite_dna = dna1_smooth + dna2_smooth
    
    # Normalize to 0-1 range for SLIC
    if composite_dna.max() > 0:
        composite_dna = composite_dna / composite_dna.max()
    
    return composite_dna, (x_min, x_max, y_min, y_max)


def perform_slic_segmentation(
    composite_image: np.ndarray,
    target_bin_size_um: float = 20.0,
    resolution_um: float = 1.0,
    compactness: float = 10.0,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Perform SLIC superpixel segmentation on DNA composite image.
    
    Args:
        composite_image: Composite DNA image
        target_bin_size_um: Target superpixel size in micrometers
        resolution_um: Image resolution in micrometers/pixel
        compactness: SLIC compactness parameter (higher = more compact)
        sigma: Pre-smoothing parameter
        
    Returns:
        2D array of superpixel labels
    """
    if composite_image.size == 0:
        return np.array([])
    
    # Calculate number of superpixels based on target size
    image_area_um2 = composite_image.size * (resolution_um ** 2)
    target_superpixel_area_um2 = target_bin_size_um ** 2
    n_segments = max(1, int(image_area_um2 / target_superpixel_area_um2))
    
    # Perform SLIC segmentation
    # Convert to 3-channel for SLIC (grayscale -> RGB)
    if len(composite_image.shape) == 2:
        # Convert grayscale to 3-channel
        rgb_image = np.stack([composite_image] * 3, axis=-1)
    else:
        rgb_image = composite_image
    
    # Run SLIC
    superpixel_labels = slic(
        rgb_image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0
    )
    
    return superpixel_labels


def aggregate_to_superpixels(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    superpixel_labels: np.ndarray,
    bounds: Tuple[float, float, float, float],
    resolution_um: float = 1.0
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Aggregate ion counts to superpixel regions.
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        superpixel_labels: 2D array of superpixel assignments
        bounds: Spatial bounds (x_min, x_max, y_min, y_max)
        resolution_um: Resolution used for superpixel generation
        
    Returns:
        Tuple of (superpixel_aggregated_counts, superpixel_coords)
    """
    if len(coords) == 0 or superpixel_labels.size == 0:
        return {}, np.array([])
    
    x_min, x_max, y_min, y_max = bounds
    
    # Convert coordinates to pixel indices
    x_indices = ((coords[:, 0] - x_min) / resolution_um).astype(int)
    y_indices = ((coords[:, 1] - y_min) / resolution_um).astype(int)
    
    # Bounds checking
    valid_mask = (x_indices >= 0) & (x_indices < superpixel_labels.shape[1]) & \
                 (y_indices >= 0) & (y_indices < superpixel_labels.shape[0])
    
    # Get superpixel assignments for each measurement
    superpixel_assignments = np.full(len(coords), -1, dtype=int)
    valid_indices = np.where(valid_mask)[0]
    superpixel_assignments[valid_indices] = superpixel_labels[
        y_indices[valid_indices], x_indices[valid_indices]
    ]
    
    # Find unique superpixels
    unique_superpixels = np.unique(superpixel_assignments)
    unique_superpixels = unique_superpixels[unique_superpixels >= 0]  # Remove invalid (-1)
    
    if len(unique_superpixels) == 0:
        return {}, np.array([])
    
    # Aggregate ion counts for each superpixel
    superpixel_counts = {}
    for protein_name, counts in ion_counts.items():
        protein_superpixel_counts = np.zeros(len(unique_superpixels))
        
        for i, superpixel_id in enumerate(unique_superpixels):
            mask = superpixel_assignments == superpixel_id
            protein_superpixel_counts[i] = np.sum(counts[mask])
        
        superpixel_counts[protein_name] = protein_superpixel_counts
    
    # Compute superpixel centroid coordinates
    superpixel_coords = np.zeros((len(unique_superpixels), 2))
    for i, superpixel_id in enumerate(unique_superpixels):
        mask = superpixel_assignments == superpixel_id
        if np.any(mask):
            superpixel_coords[i] = np.mean(coords[mask], axis=0)
    
    return superpixel_counts, superpixel_coords


def compute_superpixel_properties(
    superpixel_labels: np.ndarray,
    composite_image: np.ndarray,
    resolution_um: float = 1.0
) -> Dict[int, Dict[str, float]]:
    """
    Compute morphological properties of each superpixel.
    
    Args:
        superpixel_labels: 2D array of superpixel assignments
        composite_image: Original composite DNA image
        resolution_um: Image resolution
        
    Returns:
        Dictionary mapping superpixel_id -> properties
    """
    if superpixel_labels.size == 0:
        return {}
    
    # Use regionprops to compute properties
    props = regionprops(superpixel_labels + 1, intensity_image=composite_image)
    
    superpixel_props = {}
    for prop in props:
        superpixel_id = prop.label - 1  # Convert back to 0-based
        
        # Convert pixel measurements to micrometers
        area_um2 = prop.area * (resolution_um ** 2)
        perimeter_um = prop.perimeter * resolution_um
        
        superpixel_props[superpixel_id] = {
            'area_um2': float(area_um2),
            'perimeter_um': float(perimeter_um),
            'eccentricity': float(prop.eccentricity),
            'solidity': float(prop.solidity),
            'mean_dna_intensity': float(prop.mean_intensity),
            'centroid_x_um': float(prop.centroid[1] * resolution_um),
            'centroid_y_um': float(prop.centroid[0] * resolution_um)
        }
    
    return superpixel_props


def slic_pipeline(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    dna1_intensities: np.ndarray,
    dna2_intensities: np.ndarray,
    target_bin_size_um: float = 20.0,
    resolution_um: float = 1.0,
    sigma_um: float = 2.0,
    compactness: float = 10.0
) -> Dict:
    """
    Complete SLIC-based morphology-aware aggregation pipeline.
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        dna1_intensities: DNA1 channel data
        dna2_intensities: DNA2 channel data
        target_bin_size_um: Target superpixel size
        resolution_um: Working resolution
        sigma_um: Smoothing for DNA composite
        compactness: SLIC compactness parameter
        
    Returns:
        Dictionary with all pipeline results
    """
    if len(coords) == 0 or not ion_counts:
        return {
            'superpixel_counts': {},
            'superpixel_coords': np.array([]),
            'superpixel_labels': np.array([]),
            'superpixel_props': {},
            'composite_dna': np.array([]),
            'bounds': (0, 0, 0, 0)
        }
    
    # Step 1: Create composite DNA image
    composite_dna, bounds = prepare_dna_composite(
        coords, dna1_intensities, dna2_intensities, resolution_um, sigma_um
    )
    
    # Step 2: Perform SLIC segmentation
    superpixel_labels = perform_slic_segmentation(
        composite_dna, target_bin_size_um, resolution_um, compactness
    )
    
    # Step 3: Aggregate ion counts to superpixels
    superpixel_counts, superpixel_coords = aggregate_to_superpixels(
        coords, ion_counts, superpixel_labels, bounds, resolution_um
    )
    
    # Step 4: Compute superpixel morphological properties
    superpixel_props = compute_superpixel_properties(
        superpixel_labels, composite_dna, resolution_um
    )
    
    return {
        'superpixel_counts': superpixel_counts,
        'superpixel_coords': superpixel_coords,
        'superpixel_labels': superpixel_labels,
        'superpixel_props': superpixel_props,
        'composite_dna': composite_dna,
        'bounds': bounds,
        'resolution_um': resolution_um,
        'target_bin_size_um': target_bin_size_um
    }