"""
Reusable plotting functions for IMC data visualization.

These functions are stateless and take data as input, returning matplotlib figures or axes.
Designed to be used in Jupyter notebooks for exploratory data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple, Union
try:
    import seaborn as sns
except ImportError:
    sns = None
try:
    from skimage.segmentation import mark_boundaries
except ImportError:
    mark_boundaries = None
import warnings
import logging
try:
    from ..analysis.spatial_utils import prepare_spatial_arrays_for_plotting
    from ..analysis.spatial_utils import validate_plot_data as validate_spatial_data
except ImportError:
    # Fallback implementations
    def validate_spatial_data(data, name="data"):
        """Simple validation fallback for spatial data."""
        if data is None:
            raise ValueError(f"{name} cannot be None")
        data = np.asarray(data)
        if not np.isfinite(data).all():
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data

# Generic validation for non-spatial data (coordinates, values, etc.)
def validate_generic_data(data, name="data"):
    """Simple validation for generic data arrays."""
    if data is None:
        raise ValueError(f"{name} cannot be None")
    data = np.asarray(data)
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data
    
    def prepare_spatial_arrays_for_plotting(transformed_arrays, superpixel_labels, superpixel_coords, bounds):
        """Fallback implementation for spatial array preparation."""
        # Check if superpixel_labels is None or not a 2D array
        if superpixel_labels is None or not hasattr(superpixel_labels, 'ndim'):
            # Return empty dict if no valid labels
            return {}
            
        if superpixel_labels.ndim != 2:
            raise ValueError(f"superpixel_labels must be 2D, got {superpixel_labels.ndim}D")
        
        spatial_arrays = {}
        
        # Get unique superpixel labels (excluding background -1)
        unique_labels = np.unique(superpixel_labels[superpixel_labels >= 0])
        
        for protein, values in transformed_arrays.items():
            # Create spatial array initialized with NaN for background
            spatial_array = np.full(superpixel_labels.shape, np.nan, dtype=float)
            
            # Map superpixel values to spatial positions
            for i, superpixel_value in enumerate(values):
                if i < len(unique_labels):
                    superpixel_id = unique_labels[i]
                    # Set all pixels belonging to this superpixel
                    mask = superpixel_labels == superpixel_id
                    spatial_array[mask] = superpixel_value
            
            spatial_arrays[protein] = spatial_array
        
        return spatial_arrays



def validate_coordinate_data(coords: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate coordinate and value data together, ensuring consistency.
    
    Args:
        coords: Coordinate array (N, 2)
        values: Value array (N,)
        
    Returns:
        Tuple of cleaned (coords, values)
    """
    coords = validate_generic_data(coords, "coordinates")
    values = validate_generic_data(values, "values")
    
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Coordinates must be (N, 2) array")
    
    if values.ndim != 1:
        raise ValueError("Values must be 1D array")
    
    # Ensure consistent length after cleaning
    min_len = min(len(coords), len(values))
    if len(coords) != len(values):
        warnings.warn(
            f"Coordinate and value arrays have different lengths after cleaning: "
            f"{len(coords)} vs {len(values)}. Truncating to {min_len}."
        )
        coords = coords[:min_len]
        values = values[:min_len]
    
    return coords, values


def plot_roi_overview(
    coords: np.ndarray,
    values: np.ndarray,
    title: str = "ROI Overview",
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (8, 8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None
) -> Union[plt.Figure, plt.Axes]:
    """
    Plot spatial distribution of values across an ROI.
    
    Args:
        coords: (N, 2) array of pixel coordinates
        values: (N,) array of values to plot
        title: Plot title
        cmap: Colormap name
        figsize: Figure size if creating new figure
        vmin, vmax: Color scale limits
        ax: Existing axes to plot on (if None, creates new figure)
        
    Returns:
        Figure or Axes object
    """
    # Validate and clean input data
    coords, values = validate_coordinate_data(coords, values)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        return_fig = False
        
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], 
        c=values, s=1, cmap=cmap,
        vmin=vmin, vmax=vmax, rasterized=True
    )
    
    ax.set_title(title)
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.set_aspect('equal')
    
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    
    return fig if return_fig else ax


# Legacy square binning visualization removed - now using SLIC superpixel visualization


def plot_protein_expression(
    protein_data: Dict[str, np.ndarray],
    coords: np.ndarray,
    proteins_to_plot: Optional[List[str]] = None,
    ncols: int = 3,
    figsize_per_plot: Tuple[float, float] = (4, 4),
    cmap: str = "viridis",
    percentile_scale: Tuple[float, float] = (1, 99)
) -> plt.Figure:
    """
    Plot multiple protein expression patterns in a grid.
    
    Args:
        protein_data: Dictionary mapping protein names to expression values
        coords: (N, 2) array of coordinates
        proteins_to_plot: List of proteins to plot (default: all)
        ncols: Number of columns in grid
        figsize_per_plot: Size of each subplot
        cmap: Colormap
        percentile_scale: Percentiles for color scaling
        
    Returns:
        Figure object
    """
    if proteins_to_plot is None:
        proteins_to_plot = list(protein_data.keys())
    
    n_proteins = len(proteins_to_plot)
    nrows = (n_proteins + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )
    
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, protein in enumerate(proteins_to_plot):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        values = protein_data[protein]
        vmin, vmax = np.percentile(values, percentile_scale)
        
        plot_roi_overview(
            coords, values,
            title=protein,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            ax=ax
        )
    
    # Hide empty subplots
    for idx in range(n_proteins, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_cluster_map(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_names: Optional[Dict[int, str]] = None,
    cmap: str = "tab20",
    figsize: Tuple[int, int] = (10, 8),
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True
) -> Union[plt.Figure, plt.Axes]:
    """
    Plot spatial distribution of clusters.
    
    Args:
        coords: (N, 2) array of coordinates
        cluster_labels: (N,) array of cluster assignments
        cluster_names: Optional mapping of cluster IDs to names
        cmap: Colormap
        figsize: Figure size
        ax: Existing axes
        show_legend: Whether to show cluster legend
        
    Returns:
        Figure or Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        return_fig = False
    
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Get colors from colormap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    colors = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]
    
    # Plot each cluster
    for idx, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        label = cluster_names.get(cluster_id, f"Cluster {cluster_id}") if cluster_names else f"Cluster {cluster_id}"
        
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[idx]], s=1, label=label,
            rasterized=True
        )
    
    ax.set_title("Spatial Cluster Distribution")
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.set_aspect('equal')
    
    if show_legend and n_clusters <= 20:
        ax.legend(markerscale=5, frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))
    
    return fig if return_fig else ax


def plot_scale_comparison(
    multiscale_results: Dict[float, Dict],
    coords: np.ndarray,
    scales_to_plot: Optional[List[float]] = None,
    figsize_per_scale: Tuple[float, float] = (5, 5)
) -> plt.Figure:
    """
    Compare analysis results across multiple spatial scales.
    
    Args:
        multiscale_results: Dictionary mapping scales to results
        coords: Original coordinate array
        scales_to_plot: Scales to visualize (default: all)
        figsize_per_scale: Size of each scale subplot
        
    Returns:
        Figure object
    """
    # Validate input data
    coords = validate_generic_data(coords, "coordinates")
    
    if not multiscale_results:
        raise ValueError("multiscale_results cannot be empty")
    
    if scales_to_plot is None:
        scales_to_plot = sorted(multiscale_results.keys())
    
    n_scales = len(scales_to_plot)
    fig, axes = plt.subplots(1, n_scales, figsize=(figsize_per_scale[0] * n_scales, figsize_per_scale[1]))
    
    if n_scales == 1:
        axes = [axes]
    
    for idx, scale in enumerate(scales_to_plot):
        ax = axes[idx]
        scale_result = multiscale_results[scale]
        
        if 'cluster_map' in scale_result:
            # Plot cluster map for this scale
            cluster_map = scale_result['cluster_map']
            cluster_map = validate_spatial_data(cluster_map, f"cluster_map_scale_{scale}")
            im = ax.imshow(cluster_map, cmap='tab20', origin='lower')
            ax.set_title(f"Scale: {scale}μm")
            ax.set_xlabel("X bins")
            ax.set_ylabel("Y bins")
        elif 'cluster_labels' in scale_result and 'superpixel_coords' in scale_result:
            # Plot superpixel clusters
            sp_coords = scale_result['superpixel_coords']
            labels = scale_result['cluster_labels']
            # Validate superpixel data
            sp_coords = validate_generic_data(sp_coords, f"superpixel_coords_scale_{scale}")
            labels = validate_generic_data(labels, f"cluster_labels_scale_{scale}")
            plot_cluster_map(sp_coords, labels, ax=ax, show_legend=False)
            ax.set_title(f"Scale: {scale}μm")
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    correlation_matrix: np.ndarray,
    labels: List[str],
    title: str = "Correlation Heatmap",
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (10, 8),
    vmin: float = -1,
    vmax: float = 1,
    ax: Optional[plt.Axes] = None
) -> Union[plt.Figure, plt.Axes]:
    """
    Plot a correlation heatmap with labels.
    
    Args:
        correlation_matrix: Square correlation matrix
        labels: Labels for rows/columns
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        vmin, vmax: Color scale limits
        ax: Existing axes
        
    Returns:
        Figure or Axes object
    """
    # Validate correlation matrix
    correlation_matrix = validate_spatial_data(correlation_matrix, "correlation_matrix")
    
    if correlation_matrix.ndim != 2 or correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("Correlation matrix must be square")
    
    if len(labels) != correlation_matrix.shape[0]:
        raise ValueError(f"Number of labels ({len(labels)}) must match matrix size ({correlation_matrix.shape[0]})")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        return_fig = False
    
    im = ax.imshow(correlation_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Set ticks and labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    ax.set_title(title)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add text annotations for values
    if len(labels) <= 15:  # Only add text if matrix is small enough
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    return fig if return_fig else ax



def plot_segmentation_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    bounds: Tuple[float, float, float, float],
    transformed_arrays: Dict[str, np.ndarray],
    cofactors_used: Dict[str, float],
    config: 'Config',
    superpixel_coords: Optional[np.ndarray] = None,
    title: str = "Multi-Channel Validation",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Create comprehensive multi-channel validation plot showing all protein markers.
    
    Displays DNA composite, key biological channels, and segmentation overlay
    with arcsinh coefficients and proper biological grouping from config.
    
    Args:
        image: The 2D DNA composite image (DNA1+DNA2)
        labels: The 2D integer array of segmentation labels
        bounds: Tuple of (x_min, x_max, y_min, y_max) for axis scaling
        transformed_arrays: Dictionary of protein_name -> transformed values (1D superpixel data)
        cofactors_used: Dictionary of protein_name -> arcsinh coefficient
        config: Configuration object containing channel groups and visualization settings
        superpixel_coords: Optional superpixel coordinates for spatial mapping
        title: Main figure title
        ax: Ignored (creates custom multi-subplot layout)
        
    Returns:
        Figure object with config-driven biological validation layout
    """
    # Validate inputs
    image = validate_spatial_data(image, "DNA composite image")
    
    # Convert to spatial arrays using smart detection and validation
    logger = logging.getLogger('Visualization')
    try:
        spatial_protein_arrays = prepare_spatial_arrays_for_plotting(
            transformed_arrays=transformed_arrays,
            superpixel_labels=labels,
            superpixel_coords=superpixel_coords,
            bounds=bounds
        )
        logger.debug(f"Successfully prepared {len(spatial_protein_arrays)} spatial arrays for plotting")
    except Exception as e:
        logger.error(f"Failed to prepare spatial arrays: {e}")
        raise ValueError(f"Cannot prepare spatial arrays for plotting: {e}")
    
    # Get visualization configuration with fallbacks
    if hasattr(config, 'visualization'):
        viz_config = config.visualization.get('validation_plots', {})
    else:
        # Fallback configuration
        viz_config = {
            'primary_markers': {'immune_markers': 'CD45', 'vascular_markers': 'CD31', 'stromal_markers': 'CD140a'},
            'colormaps': {'immune_markers': 'Reds', 'vascular_markers': 'Blues', 'stromal_markers': 'Greens', 'default': 'viridis'},
            'always_include': ['CD206', 'CD44'],
            'max_additional_channels': 5
        }
    layout_config = viz_config.get('layout', {})
    figsize = tuple(layout_config.get('figsize', [20, 12]))
    
    # Get channel groups from config with fallback
    if hasattr(config, 'channel_groups'):
        channel_groups = config.channel_groups
    else:
        # Fallback channel groups
        channel_groups = {
            'immune_markers': {'pan_leukocyte': ['CD45'], 'myeloid': ['CD11b', 'Ly6G', 'CD206']},
            'vascular_markers': ['CD31', 'CD34'],
            'stromal_markers': ['CD140a', 'CD140b'],
            'adhesion_markers': ['CD44']
        }
    primary_markers = viz_config.get('primary_markers', {})
    colormaps = viz_config.get('colormaps', {})
    max_additional = viz_config.get('max_additional_channels', 5)
    
    # Create figure with custom subplot layout
    fig = plt.figure(figsize=figsize)
    
    # Define equal-sized subplot positions with proper spacing
    panel_width = 0.16    # Slightly smaller to allow more spacing
    panel_height = 0.32   # Slightly smaller height for better proportions
    h_spacing = 0.04      # Horizontal spacing between panels
    
    # Calculate positions with consistent spacing
    left_margin = 0.03
    top_y = 0.52      # Moved down to reduce gap
    bottom_y = 0.12   # Keep bottom row position
    v_spacing = top_y - bottom_y - panel_height  # Vertical spacing between rows
    
    # Top row: DNA composite and segmentation side by side, then 2 primary proteins
    top_positions = []
    for i in range(5):
        x_pos = left_margin + i * (panel_width + h_spacing)
        top_positions.append((x_pos, top_y, panel_width, panel_height))
    
    # Bottom row: Additional protein channels, all equal size and spacing
    bottom_positions = []
    for i in range(5):
        x_pos = left_margin + i * (panel_width + h_spacing)
        bottom_positions.append((x_pos, bottom_y, panel_width, panel_height))
    
    # Get spatial extent
    x_min, x_max, y_min, y_max = bounds
    extent = [x_min, x_max, y_min, y_max]
    
    # Apply proper transformation to DNA composite for visualization
    try:
        from src.analysis.slic_segmentation import prepare_dna_for_visualization
        img_normalized = prepare_dna_for_visualization(image)
    except ImportError:
        # Fallback normalization
        img_normalized = image.astype(float)
        if img_normalized.max() > img_normalized.min():
            img_normalized = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min())
        else:
            img_normalized = np.zeros_like(image, dtype=float)
    
    # Helper function to format cofactor text
    def format_cofactor(protein_name: str) -> str:
        cofactor = cofactors_used.get(protein_name, 'N/A')
        return f"arcsinh(x/{cofactor:.1f})" if isinstance(cofactor, (int, float)) else f"arcsinh(x/{cofactor})"
    
    # Helper function to get group description for a protein
    def get_group_description(protein_name: str) -> str:
        for group_name, group_data in channel_groups.items():
            if isinstance(group_data, dict):
                # Handle nested structure like immune_markers.pan_leukocyte
                for subgroup_name, proteins in group_data.items():
                    if protein_name in proteins:
                        return subgroup_name.replace('_', ' ').title()
            elif isinstance(group_data, list) and protein_name in group_data:
                # Handle flat list structure
                return group_name.replace('_', ' ').title()
        return protein_name
    
    # Helper function to get colormap for a protein from config
    def get_marker_colormap(protein_name: str, config: 'Config') -> str:
        # Try to find group for this protein and get its colormap
        for group_name, group_data in channel_groups.items():
            protein_found = False
            if isinstance(group_data, dict):
                # Handle nested structure like immune_markers.pan_leukocyte
                for subgroup_name, proteins in group_data.items():
                    if protein_name in proteins:
                        protein_found = True
                        break
            elif isinstance(group_data, list) and protein_name in group_data:
                protein_found = True
            
            if protein_found:
                return colormaps.get(group_name, colormaps.get('default', 'viridis'))
        
        # Default fallback
        return colormaps.get('default', 'viridis')
    
    # TOP ROW - Reference channels
    
    # Panel 1: DNA Composite
    ax1 = fig.add_axes(top_positions[0])
    im1 = ax1.imshow(img_normalized, extent=extent, origin='lower', cmap='gray')
    ax1.set_title("DNA Composite", fontsize=11, fontweight='bold')
    ax1.set_xlabel("X (μm)", fontsize=9)
    ax1.set_ylabel("Y (μm)", fontsize=9)
    ax1.set_aspect('equal')
    ax1.grid(False)
    
    # Panel 2: Segmentation overlay (next to DNA composite)
    ax2 = fig.add_axes(top_positions[1])
    # Create thinner segmentation boundaries by dilating the labels minimally
    try:
        from scipy.ndimage import binary_dilation
        from skimage.segmentation import find_boundaries
        
        # Find boundaries and make them thinner
        boundaries = find_boundaries(labels, mode='inner')
        # Create minimal dilation for 0.5px effect (thinnest possible)
        thin_boundaries = binary_dilation(boundaries, structure=np.ones((1,1)))
    except ImportError:
        # Fallback boundary detection
        # Simple edge detection
        dy = np.diff(labels, axis=0, prepend=labels[0:1,:])
        dx = np.diff(labels, axis=1, prepend=labels[:,0:1])
        boundaries = (dy != 0) | (dx != 0)
        thin_boundaries = boundaries
    
    # Create overlay with custom thin yellow lines
    overlay_img = img_normalized.copy()
    if len(overlay_img.shape) == 2:
        # Convert grayscale to RGB for colored overlay
        overlay_img = np.stack([overlay_img, overlay_img, overlay_img], axis=-1)
    overlay_img[thin_boundaries] = [1, 1, 0]  # Yellow boundaries
    
    ax2.imshow(overlay_img, extent=extent, origin='lower')
    ax2.set_title("SLIC Segmentation", fontsize=11, fontweight='bold')
    ax2.set_xlabel("X (μm)", fontsize=9)
    ax2.set_yticklabels([])  # Remove y-axis labels since DNA composite already has them
    ax2.set_aspect('equal')
    ax2.grid(False)
    
    # Add segment count
    n_segments = len(np.unique(labels[labels >= 0]))
    ax2.text(0.02, 0.98, f"Segments: {n_segments}", transform=ax2.transAxes, 
             fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="white", alpha=0.8))
    
    # Dynamically select primary markers from config
    primary_panels = []
    for group_name, primary_marker in primary_markers.items():
        if primary_marker in spatial_protein_arrays:
            primary_panels.append((primary_marker, group_name))
    
    # Panel 3, 4, & 5: Primary markers (up to 3) 
    protein_start_idx = 2  # Start after DNA and segmentation
    max_primary_panels = min(len(primary_panels), len(top_positions) - protein_start_idx)
    for panel_idx, (marker, group_name) in enumerate(primary_panels[:max_primary_panels]):
        ax = fig.add_axes(top_positions[protein_start_idx + panel_idx])
        
        marker_data = spatial_protein_arrays[marker]
        cmap = colormaps.get(group_name, 'viridis')
        
        im = ax.imshow(marker_data, extent=extent, origin='lower', cmap=cmap)
        # Smaller colorbar with better positioning to avoid overlap
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)
        
        description = get_group_description(marker)
        cofactor_text = format_cofactor(marker)
        
        ax.set_title(f"{marker}\n{cofactor_text}", fontsize=10, fontweight='bold')
        ax.set_xlabel("X (μm)", fontsize=8)
        
        # Only show y-axis labels on leftmost plots
        if protein_start_idx + panel_idx == 2:  # First protein panel
            ax.set_ylabel("Y (μm)", fontsize=8)
        else:
            ax.set_yticklabels([])

        ax.set_aspect('equal')
        ax.grid(False)

    # Fill empty primary panel slots if needed
    for panel_idx in range(max_primary_panels, len(top_positions) - protein_start_idx):
        if protein_start_idx + panel_idx < len(top_positions):
            ax = fig.add_axes(top_positions[protein_start_idx + panel_idx])
            ax.text(0.5, 0.5, 'Marker\nN/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
            ax.set_title("N/A", fontsize=10, fontweight='bold')
            ax.set_xlabel("X (μm)", fontsize=8)
            ax.set_aspect('equal')
            ax.grid(False)
    
    
    # BOTTOM ROW - Additional available protein channels with prioritization
    # Get primary markers that were already shown
    shown_markers = set(marker for marker, _ in primary_panels[:max_primary_panels])

    # Get protein channels from config to filter out background/bead channels
    if hasattr(config, 'channels'):
        protein_channels_list = config.channels.get('protein_channels', [])
        bead_channels_list = config.channels.get('calibration_channels', [])
    else:
        # Fallback: assume all channels are proteins
        protein_channels_list = list(spatial_protein_arrays.keys())
        bead_channels_list = []

    # Get critical markers that must be included
    always_include = viz_config.get('always_include', [])
    priority_markers = [m for m in always_include if m in spatial_protein_arrays and m not in shown_markers]

    # Get remaining proteins - FILTER to only include actual protein channels
    remaining_proteins = [p for p in spatial_protein_arrays.keys()
                         if p not in shown_markers
                         and p not in priority_markers
                         and p in protein_channels_list]  # NEW: filter by protein channels

    # Combine priority markers first, then remaining
    available_proteins = priority_markers + remaining_proteins

    # Get bead channels to plot separately
    bead_channels_to_plot = [b for b in bead_channels_list if b in spatial_protein_arrays]

    # Show ALL protein channels (no artificial limit)
    max_panels = len(available_proteins)  # Changed from config limit

    # Expand bottom_positions if we have more proteins than available slots
    panels_per_row = 5
    if max_panels > len(bottom_positions):
        # Create additional rows for all protein channels
        rows_needed = (max_panels + panels_per_row - 1) // panels_per_row
        for row_idx in range(1, rows_needed):
            y_pos = bottom_y - (row_idx * (panel_height + 0.02))  # Stack rows downward
            for col_idx in range(panels_per_row):
                x_pos = left_margin + col_idx * (panel_width + h_spacing)
                bottom_positions.append((x_pos, y_pos, panel_width, panel_height))

    for i, protein in enumerate(available_proteins):
        if i >= len(bottom_positions):
            break

        ax = fig.add_axes(bottom_positions[i])
        protein_data = spatial_protein_arrays[protein]
        
        # Determine colormap from config using helper function
        cmap = get_marker_colormap(protein, config)
        
        im = ax.imshow(protein_data, extent=extent, origin='lower', cmap=cmap)
        # Smaller colorbar with better positioning to avoid overlap
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)
        
        cofactor_text = format_cofactor(protein)
        ax.set_title(f"{protein}\n{cofactor_text}", fontsize=10, fontweight='bold')
        ax.set_xlabel("X (μm)", fontsize=8)
        
        # Only show y-axis labels on leftmost plot
        if i == 0:
            ax.set_ylabel("Y (μm)", fontsize=8)
        else:
            ax.set_yticklabels([])

        ax.set_aspect('equal')
        ax.grid(False)

    # BEAD/CALIBRATION CHANNELS - Display after all protein channels
    if bead_channels_to_plot:
        # Calculate starting position for bead channels (continuation from proteins)
        bead_start_idx = len(available_proteins)

        # Expand positions if needed for bead channels
        total_channels = len(available_proteins) + len(bead_channels_to_plot)
        while len(bottom_positions) < total_channels:
            row_idx = len(bottom_positions) // panels_per_row
            col_idx = len(bottom_positions) % panels_per_row
            y_pos = bottom_y - (row_idx * (panel_height + 0.02))
            x_pos = left_margin + col_idx * (panel_width + h_spacing)
            bottom_positions.append((x_pos, y_pos, panel_width, panel_height))

        for i, bead_channel in enumerate(bead_channels_to_plot):
            panel_idx = bead_start_idx + i
            if panel_idx >= len(bottom_positions):
                break

            ax = fig.add_axes(bottom_positions[panel_idx])
            bead_data = spatial_protein_arrays[bead_channel]

            # Use viridis colormap for bead channels
            im = ax.imshow(bead_data, extent=extent, origin='lower', cmap='plasma')
            plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)

            cofactor_text = format_cofactor(bead_channel)
            ax.set_title(f"{bead_channel} (Bead)\n{cofactor_text}", fontsize=10, fontweight='bold', color='purple')
            ax.set_xlabel("X (μm)", fontsize=8)

            # Only show y-axis labels on leftmost plot of each row
            if panel_idx % panels_per_row == 0:
                ax.set_ylabel("Y (μm)", fontsize=8)
            else:
                ax.set_yticklabels([])

            ax.set_aspect('equal')
            ax.grid(False)

    # Simplified main title
    roi_name = title.split(' - ')[1] if ' - ' in title else title
    scale_text = title.split(' - ')[-1] if ' - ' in title else ""
    fig.suptitle(f"{roi_name} {scale_text}", fontsize=12, fontweight='bold', y=0.95)
    
    return fig