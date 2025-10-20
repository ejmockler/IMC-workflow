"""
Spatial data transformation utilities for IMC analysis.

Handles conversion between different spatial data representations:
- Raw coordinates → Spatial arrays  
- Superpixel aggregated data → Spatial arrays
- Multi-scale data handling and validation
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging


def validate_spatial_data_structure(
    transformed_arrays: Dict[str, np.ndarray],
    superpixel_coords: Optional[np.ndarray] = None,
    context: str = "validation"
) -> str:
    """
    Validate and determine the structure of transformed_arrays.
    
    Args:
        transformed_arrays: Dictionary of protein -> array data
        superpixel_coords: Optional superpixel coordinates (indicates 1D superpixel data)
        context: Context string for logging
        
    Returns:
        Data structure type: 'spatial_2d', 'superpixel_1d', or 'unknown'
    """
    logger = logging.getLogger('SpatialUtils')
    
    if not transformed_arrays:
        logger.warning(f"{context}: Empty transformed_arrays")
        return 'unknown'
    
    # Check first protein array to determine structure
    first_protein = next(iter(transformed_arrays.keys()))
    first_array = transformed_arrays[first_protein]
    
    logger.debug(f"{context}: First protein '{first_protein}' array shape: {first_array.shape}")
    
    if first_array.ndim == 2:
        # 2D array - this is already spatial data
        logger.info(f"{context}: Detected 2D spatial arrays ({first_array.shape})")
        return 'spatial_2d'
    elif first_array.ndim == 1 and superpixel_coords is not None:
        # 1D array with superpixel coordinates - needs spatial reconstruction
        logger.info(f"{context}: Detected 1D superpixel data ({first_array.shape[0]} superpixels)")
        return 'superpixel_1d'
    else:
        logger.error(f"{context}: Unknown data structure - {first_array.ndim}D array, superpixel_coords: {superpixel_coords is not None}")
        return 'unknown'


def create_spatial_arrays_from_superpixels(
    superpixel_labels: np.ndarray,
    transformed_arrays: Dict[str, np.ndarray],
    superpixel_coords: np.ndarray,
    bounds: Tuple[float, float, float, float]
) -> Dict[str, np.ndarray]:
    """
    Convert superpixel-aggregated data back to spatial arrays for visualization.
    
    This function handles the case where transformed_arrays contains 1D arrays
    of superpixel-aggregated values that need to be mapped back to 2D spatial arrays.
    
    Args:
        superpixel_labels: 2D array of superpixel assignments  
        transformed_arrays: Dict of protein_name -> 1D superpixel values
        superpixel_coords: Coordinates of superpixel centroids  
        bounds: Spatial bounds (x_min, x_max, y_min, y_max)
        
    Returns:
        Dictionary of protein_name -> 2D spatial array
        
    Raises:
        ValueError: If data shapes are incompatible or labels don't match values
    """
    logger = logging.getLogger('SpatialUtils')
    
    # Validate inputs
    if superpixel_labels.ndim != 2:
        raise ValueError(f"superpixel_labels must be 2D, got {superpixel_labels.ndim}D")
    
    unique_labels = np.unique(superpixel_labels)
    unique_labels = unique_labels[unique_labels >= 0]  # Exclude background (-1)
    n_superpixels = len(unique_labels)
    
    logger.debug(f"Converting {len(transformed_arrays)} proteins from {n_superpixels} superpixels to spatial arrays")
    
    spatial_arrays = {}
    
    for protein_name, superpixel_values in transformed_arrays.items():
        # Validate superpixel data is 1D
        if superpixel_values.ndim != 1:
            raise ValueError(f"Expected 1D superpixel values for {protein_name}, got {superpixel_values.ndim}D")
        
        # Check length compatibility
        if len(superpixel_values) != n_superpixels:
            raise ValueError(
                f"Length mismatch for {protein_name}: {len(superpixel_values)} values "
                f"for {n_superpixels} superpixels (labels: {unique_labels.min()}-{unique_labels.max()})"
            )
        
        # Create spatial array by mapping superpixel values back to spatial locations
        spatial_array = np.full_like(superpixel_labels, np.nan, dtype=float)
        
        # Fully vectorized assignment using lookup table - much faster than loops
        # Create value lookup array where index = superpixel_id, value = protein_value
        max_label = int(np.max(unique_labels))
        value_lookup = np.full(max_label + 1, np.nan)
        value_lookup[unique_labels] = superpixel_values
        
        # Apply lookup to all pixels at once - vectorized assignment
        valid_mask = superpixel_labels >= 0
        spatial_array[valid_mask] = value_lookup[superpixel_labels[valid_mask]]
        
        # Validate result
        n_assigned = np.sum(~np.isnan(spatial_array))
        n_total = np.sum(superpixel_labels >= 0)
        
        if n_assigned != n_total:
            logger.warning(f"{protein_name}: Assigned {n_assigned}/{n_total} pixels")
        
        spatial_arrays[protein_name] = spatial_array
    
    logger.info(f"Successfully converted {len(spatial_arrays)} proteins to spatial arrays")
    return spatial_arrays


def prepare_spatial_arrays_for_plotting(
    transformed_arrays: Dict[str, np.ndarray],
    superpixel_labels: Optional[np.ndarray] = None,
    superpixel_coords: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> Dict[str, np.ndarray]:
    """
    Prepare transformed arrays for plotting by ensuring they are 2D spatial arrays.
    
    This function automatically detects the data structure and handles conversion
    if needed, or returns the arrays directly if they're already spatial.
    
    Args:
        transformed_arrays: Dictionary of protein -> array data
        superpixel_labels: Optional 2D superpixel label array
        superpixel_coords: Optional superpixel coordinates
        bounds: Optional spatial bounds
        
    Returns:
        Dictionary of protein_name -> 2D spatial array ready for plotting
        
    Raises:
        ValueError: If data structure is incompatible or required parameters are missing
    """
    logger = logging.getLogger('SpatialUtils')
    
    # Validate and determine data structure
    data_type = validate_spatial_data_structure(transformed_arrays, superpixel_coords, "plotting_prep")
    
    if data_type == 'spatial_2d':
        # Data is already 2D spatial - return as is
        logger.info("Data is already in 2D spatial format - using directly")
        return {k: v.copy() for k, v in transformed_arrays.items()}
    
    elif data_type == 'superpixel_1d':
        # Need to convert from 1D superpixel data to 2D spatial
        if superpixel_labels is None or bounds is None:
            raise ValueError("superpixel_labels and bounds are required for 1D superpixel data conversion")
        
        logger.info("Converting 1D superpixel data to 2D spatial arrays")
        return create_spatial_arrays_from_superpixels(
            superpixel_labels, transformed_arrays, superpixel_coords, bounds
        )
    
    else:
        raise ValueError(f"Unknown or incompatible data structure: {data_type}")


def validate_plot_data(data: np.ndarray, data_name: str = "data") -> np.ndarray:
    """
    Validate and clean plotting data to prevent NaN/Inf visualization issues.
    
    For superpixel-based data, NaN values represent background regions and should
    be preserved for proper visualization (matplotlib handles NaN as transparent).
    
    Args:
        data: Array to validate
        data_name: Name for error messages
        
    Returns:
        Cleaned array safe for plotting
        
    Raises:
        ValueError: If data contains only NaN/Inf or has wrong dimensions
    """
    logger = logging.getLogger('SpatialUtils')
    
    if data.size == 0:
        raise ValueError(f"{data_name} is empty")
    
    if data.ndim != 2:
        raise ValueError(f"{data_name} must be 2D for plotting, got {data.ndim}D")
    
    # Check for problematic values
    n_nan = np.sum(np.isnan(data))
    n_inf = np.sum(np.isinf(data))
    n_total = data.size
    
    # Check if we have any finite values at all
    finite_mask = np.isfinite(data)
    n_finite = np.sum(finite_mask)
    
    if n_finite == 0:
        raise ValueError(f"{data_name} contains no finite values (all NaN/Inf)")
    
    # For superpixel data, NaN is expected (background regions)
    # Only fix Inf values, keep NaN for background transparency
    if n_inf > 0:
        logger.debug(f"{data_name}: Found {n_inf} Inf values, replacing with median")
        data = data.copy()
        inf_mask = np.isinf(data)
        median_val = np.median(data[finite_mask])
        data[inf_mask] = median_val
    
    if n_nan > 0:
        logger.debug(f"{data_name}: {n_nan} NaN values preserved for background regions")
    
    logger.debug(f"{data_name}: {n_finite} finite values, {n_nan} NaN (background), {n_inf} fixed Inf")
    
    return data

