"""
SLIC Superpixel Segmentation for IMC Data

Morphology-aware spatial analysis using DNA channel information.
Creates superpixels for hypothesis generation in spatial proteomics.
Enables multi-scale analysis of tissue organization patterns.

Note: This is a methods development study for hypothesis generation.
Results should be interpreted with appropriate statistical caution given n=2 pilot data.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, TYPE_CHECKING
from skimage.segmentation import slic
from skimage.measure import regionprops
from scipy import ndimage

if TYPE_CHECKING:
    from ..config import Config


def prepare_dna_composite(
    coords: np.ndarray,
    dna1_intensities: np.ndarray,
    dna2_intensities: np.ndarray,
    resolution_um: float = 1.0,
    config: Optional['Config'] = None
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Create composite DNA image for SLIC segmentation.
    
    Uses both DNA channels to create morphology map at native IMC resolution.
    
    Args:
        coords: Nx2 coordinate array in micrometers
        dna1_intensities: DNA1 channel intensities
        dna2_intensities: DNA2 channel intensities
        resolution_um: Output resolution in micrometers
        
    Returns:
        Tuple of (composite_dna_image, bounds)
    """
    if len(coords) == 0:
        return np.array([[]]), (0, 0, 0, 0)
    
    
    # Determine spatial bounds (no artificial buffer)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
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
    
    # No smoothing for IMC data - preserve native resolution
    # IMC already provides cellular-level detail at ~1μm resolution
    # Smoothing only degrades the precise spatial information
    
    # Create composite: direct sum of both channels
    composite_dna = dna1_field + dna2_field
    
    # Apply simplified transformation for SLIC segmentation
    if composite_dna.size > 0 and composite_dna.max() > 0:
        # Get configuration or use defaults
        if config is not None and hasattr(config, 'dna_processing'):
            dna_config = config.dna_processing
            arcsinh_config = dna_config.get('arcsinh_transform', {})
        else:
            # Fallback defaults - more conservative values
            arcsinh_config = {'noise_floor_percentile': 10, 'cofactor_multiplier': 3}
        
        # Apply ArcSinh transformation for dynamic range handling
        if arcsinh_config.get('enabled', True):
            # Get ALL values including zeros for proper noise floor
            all_vals = composite_dna.flatten()
            
            if len(all_vals) > 0:
                # Calculate noise floor from ALL data (including zeros)
                # This gives us the true background level
                noise_percentile = arcsinh_config.get('noise_floor_percentile', 10)
                noise_floor = np.percentile(all_vals, noise_percentile)
                
                # Avoid division by zero
                if noise_floor <= 0:
                    # If noise floor is zero, use a small fraction of the median
                    non_zero = all_vals[all_vals > 0]
                    if len(non_zero) > 0:
                        noise_floor = np.median(non_zero) * 0.01
                    else:
                        noise_floor = 1.0  # Fallback
                
                # Conservative multiplier to preserve gradients
                multiplier = arcsinh_config.get('cofactor_multiplier', 3)
                cofactor = noise_floor * multiplier
                
                # Apply ArcSinh - this handles the full dynamic range
                # No additional scaling - let SLIC work with transformed values
                composite_dna = np.arcsinh(composite_dna / cofactor)
    
    return composite_dna, (x_min, x_max, y_min, y_max)


def prepare_dna_for_visualization(composite_dna: np.ndarray) -> np.ndarray:
    """
    Prepare DNA composite for visualization.
    
    Note: DNA composite from prepare_dna_composite() already has ArcSinh transformation
    applied. This function optimizes the display range for visualization.
    
    Args:
        composite_dna: DNA composite image (ArcSinh transformed)
        
    Returns:
        DNA image optimized for visualization
    """
    if composite_dna.size == 0:
        return composite_dna
    
    # The data is already ArcSinh transformed from prepare_dna_composite()
    # Just apply percentile clipping for optimal contrast
    # Use 2nd and 98th percentiles to handle outliers
    p2 = np.percentile(composite_dna, 2)
    p98 = np.percentile(composite_dna, 98)
    
    if p98 > p2:
        # Clip and scale to 0-1 for display
        viz_dna = np.clip((composite_dna - p2) / (p98 - p2), 0, 1)
    else:
        # If no variation, just normalize
        viz_dna = composite_dna / (np.max(composite_dna) + 1e-10)
    
    return viz_dna


def perform_slic_segmentation(
    composite_image: np.ndarray,
    target_bin_size_um: float = 20.0,
    resolution_um: float = 1.0,
    compactness: float = 10.0,
    sigma: float = 1.5,  # Restored to midpoint between original 2.0 and 1.0
    config: Optional['Config'] = None
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
    
    # Create tissue mask based on DNA signal presence
    # After ArcSinh transform, background should be near 0
    # Use a threshold to identify tissue regions
    if config is not None and hasattr(config, 'dna_processing'):
        tissue_threshold = config.dna_processing.get('tissue_threshold', 0.1)
    else:
        tissue_threshold = 0.1  # Default threshold for transformed values
    tissue_mask = composite_image > tissue_threshold
    
    # Apply morphological operations to clean up the mask
    from scipy.ndimage import binary_opening, binary_closing, binary_erosion
    tissue_mask = binary_closing(binary_opening(tissue_mask, iterations=2), iterations=2)
    
    # Create an eroded mask to avoid edge artifacts
    # Erode by a few pixels to exclude the very edge of tissue
    tissue_mask_eroded = binary_erosion(tissue_mask, iterations=3)
    
    # If erosion removed everything, use original mask
    if not np.any(tissue_mask_eroded):
        tissue_mask_eroded = tissue_mask
    
    # Calculate number of superpixels based on tissue area (not total image area)
    tissue_area_pixels = np.sum(tissue_mask_eroded)
    tissue_area_um2 = tissue_area_pixels * (resolution_um ** 2)
    target_superpixel_area_um2 = target_bin_size_um ** 2
    n_segments = max(1, int(tissue_area_um2 / target_superpixel_area_um2))
    
    # Perform SLIC segmentation
    # Convert to 3-channel for SLIC (grayscale -> RGB)
    if len(composite_image.shape) == 2:
        # Convert grayscale to 3-channel
        rgb_image = np.stack([composite_image] * 3, axis=-1)
    else:
        rgb_image = composite_image
    
    # Simplified SLIC parameters - no complex adjustments
    # Let the properly transformed DNA data guide the segmentation
    # Only apply minimal configuration-based adjustment if specified
    if config is not None and hasattr(config, 'dna_processing'):
        slic_config = config.dna_processing.get('slic_adjustments', {})
        # Default to no adjustment (multiplier = 1.0)
        compactness_mult = slic_config.get('compactness_multiplier', 1.0)
        sigma_mult = slic_config.get('sigma_multiplier', 1.0)
    else:
        compactness_mult = 1.0
        sigma_mult = 1.0
    
    # Run SLIC with simplified parameters
    # Don't pass mask to SLIC - let it segment naturally then post-process
    superpixel_labels = slic(
        rgb_image,
        n_segments=n_segments,
        compactness=compactness * compactness_mult,
        sigma=sigma * sigma_mult,
        start_label=0,
        channel_axis=-1 if len(rgb_image.shape) > 2 else None
    )
    
    # Post-process: only keep superpixels that have significant overlap with tissue
    # This avoids edge artifacts and border-following superpixels
    unique_labels = np.unique(superpixel_labels)
    for label in unique_labels:
        label_mask = superpixel_labels == label
        # Check overlap with eroded tissue mask (avoiding edges)
        tissue_overlap = np.sum(label_mask & tissue_mask_eroded)
        total_pixels = np.sum(label_mask)
        
        # If less than 50% overlap with tissue core, mark as background
        if tissue_overlap < 0.5 * total_pixels:
            superpixel_labels[label_mask] = -1
    
    # Also set regions completely outside tissue to -1
    superpixel_labels[~tissue_mask] = -1
    
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
    compactness: float = 10.0,
    config: Optional['Config'] = None
) -> Dict:
    """
    Complete SLIC-based morphology-aware aggregation pipeline.
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        dna1_intensities: DNA1 channel data
        dna2_intensities: DNA2 channel data
        target_bin_size_um: Target superpixel size
        resolution_um: Working resolution (default 1.0μm for IMC)
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
    
    # Step 1: Create composite DNA image at native resolution
    composite_dna, bounds = prepare_dna_composite(
        coords, dna1_intensities, dna2_intensities, resolution_um, config
    )
    
    # Step 2: Perform SLIC segmentation
    superpixel_labels = perform_slic_segmentation(
        composite_dna, target_bin_size_um, resolution_um, compactness, 1.0, config
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