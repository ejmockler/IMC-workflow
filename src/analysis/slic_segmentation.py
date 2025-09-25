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
    
    # Vectorized accumulation using np.add.at for massive performance improvement
    # This replaces the slow Python loop with efficient NumPy operations
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 0:
        np.add.at(dna1_field, (y_bins[valid_indices], x_bins[valid_indices]), 
                  dna1_intensities[valid_indices])
        np.add.at(dna2_field, (y_bins[valid_indices], x_bins[valid_indices]), 
                  dna2_intensities[valid_indices])
    
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
                    # For sparse grids, use percentile of non-zero values
                    non_zero = all_vals[all_vals > 0]
                    if len(non_zero) > 0:
                        # Use 25th percentile of non-zero values for robust cofactor
                        # This prevents over-compression of sparse data
                        noise_floor = np.percentile(non_zero, 25)
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
    
    # Create tissue mask using adaptive thresholding
    # For sparse IMC data, use a very low threshold to capture all tissue
    non_zero_vals = composite_image[composite_image > 0]
    if len(non_zero_vals) > 0:
        # For sparse data, any non-zero value likely represents tissue
        # Use a very low percentile or just above zero
        tissue_threshold = non_zero_vals.min() * 0.5  # Half of minimum non-zero value
        # This ensures we capture all pixels with actual signal
        tissue_threshold = max(tissue_threshold, 0.01)
    else:
        # If all zeros, use minimal threshold
        tissue_threshold = 0.01
    
    tissue_mask = composite_image > tissue_threshold
    
    # Apply conditional morphological operations based on tissue density
    from scipy.ndimage import binary_opening, binary_closing, binary_erosion, binary_dilation
    from skimage.morphology import remove_small_objects
    
    # Calculate tissue density to determine appropriate processing
    tissue_density = np.sum(tissue_mask) / tissue_mask.size
    
    if tissue_density > 0.3:  # Dense tissue
        # Can afford light cleanup
        tissue_mask = binary_closing(tissue_mask, iterations=1)
        tissue_mask_eroded = binary_erosion(tissue_mask, iterations=1)
    elif tissue_density > 0.1:  # Moderate density
        # Very light cleanup - just connect nearby pixels
        tissue_mask = binary_dilation(tissue_mask, iterations=1)
        tissue_mask_eroded = tissue_mask  # No erosion
    else:  # Sparse tissue (< 10% pixels)
        # No morphological operations - preserve all tissue pixels
        # Just remove very small isolated spots (noise)
        min_size = max(5, int(0.0005 * tissue_mask.size))  # 0.05% of image
        try:
            tissue_mask = remove_small_objects(tissue_mask, min_size=min_size)
        except:
            # If remove_small_objects fails, keep original mask
            pass
        tissue_mask_eroded = tissue_mask  # No erosion for sparse data
    
    # Ensure we have some tissue mask
    if not np.any(tissue_mask_eroded):
        tissue_mask_eroded = tissue_mask
    if not np.any(tissue_mask):
        # Fall back to including any non-zero pixels
        tissue_mask = composite_image > 0
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
    
    # Post-process: keep superpixels with adaptive overlap threshold
    # Adjust threshold based on tissue density for sparse data
    unique_labels = np.unique(superpixel_labels)
    
    # Use tissue density to determine appropriate overlap threshold
    tissue_density = np.sum(tissue_mask) / tissue_mask.size
    if tissue_density > 0.2:
        # Dense tissue: standard 50% threshold
        overlap_threshold = 0.5
    elif tissue_density > 0.05:
        # Moderate density: 20% threshold
        overlap_threshold = 0.2
    else:
        # Very sparse tissue: only 10% overlap required
        overlap_threshold = 0.1
    
    for label in unique_labels:
        label_mask = superpixel_labels == label
        # Check overlap with eroded tissue mask
        tissue_overlap = np.sum(label_mask & tissue_mask_eroded)
        total_pixels = np.sum(label_mask)
        
        # Apply adaptive threshold
        if total_pixels > 0 and tissue_overlap < overlap_threshold * total_pixels:
            superpixel_labels[label_mask] = -1
    
    # Also set regions completely outside tissue to -1
    superpixel_labels[~tissue_mask] = -1
    
    # Final safeguard: if we eliminated ALL superpixels, relax the threshold
    # This handles edge cases with very sparse or synthetic data
    if np.all(superpixel_labels == -1) and len(unique_labels) > 0:
        # Re-run the original SLIC labels and use more relaxed criteria
        original_labels = slic(
            rgb_image,
            n_segments=n_segments,
            compactness=compactness * compactness_mult,
            sigma=sigma * sigma_mult,
            start_label=0,
            channel_axis=-1 if len(rgb_image.shape) > 2 else None
        )
        
        # Keep superpixels that have ANY overlap with tissue (not eroded)
        for label in np.unique(original_labels):
            label_mask = original_labels == label
            # Check overlap with original tissue mask (not eroded)
            tissue_overlap = np.sum(label_mask & tissue_mask)
            
            # If any part overlaps with tissue, keep it
            if tissue_overlap > 0:
                superpixel_labels[label_mask] = label
        
        # Set background pixels to -1
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
    
    # Vectorized aggregation using bincount for O(N) performance
    # Create mapping from superpixel_id to index in results array
    superpixel_id_to_idx = {sp_id: i for i, sp_id in enumerate(unique_superpixels)}
    max_superpixel_id = int(np.max(unique_superpixels))
    
    # Create index array for bincount (superpixel_id -> result_index)
    # Use -1 for invalid assignments to exclude them
    valid_assignments = superpixel_assignments >= 0
    assignment_indices = np.full_like(superpixel_assignments, -1)
    for sp_id, idx in superpixel_id_to_idx.items():
        assignment_indices[superpixel_assignments == sp_id] = idx
    
    # Aggregate ion counts for each superpixel using vectorized operations
    superpixel_counts = {}
    for protein_name, counts in ion_counts.items():
        # Use bincount for vectorized summation - much faster than loops
        # Only include valid assignments (>= 0)
        valid_mask = assignment_indices >= 0
        if np.any(valid_mask):
            binned_counts = np.bincount(
                assignment_indices[valid_mask], 
                weights=counts[valid_mask], 
                minlength=len(unique_superpixels)
            )
            superpixel_counts[protein_name] = binned_counts
        else:
            superpixel_counts[protein_name] = np.zeros(len(unique_superpixels))
    
    # Vectorized computation of superpixel centroid coordinates
    superpixel_coords = np.zeros((len(unique_superpixels), 2))
    for coord_dim in range(2):  # x and y coordinates
        valid_mask = assignment_indices >= 0
        if np.any(valid_mask):
            # Use bincount to sum coordinates, then divide by counts
            coord_sums = np.bincount(
                assignment_indices[valid_mask],
                weights=coords[valid_mask, coord_dim],
                minlength=len(unique_superpixels)
            )
            coord_counts = np.bincount(
                assignment_indices[valid_mask],
                minlength=len(unique_superpixels)
            )
            # Avoid division by zero
            nonzero_mask = coord_counts > 0
            superpixel_coords[nonzero_mask, coord_dim] = (
                coord_sums[nonzero_mask] / coord_counts[nonzero_mask]
            )
    
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
    # Use default sigma from function signature (1.5) instead of hardcoding 1.0
    superpixel_labels = perform_slic_segmentation(
        composite_dna, target_bin_size_um, resolution_um, compactness, config=config
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