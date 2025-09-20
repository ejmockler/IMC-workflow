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
import seaborn as sns
from skimage.segmentation import mark_boundaries
import warnings


def validate_plot_data(data: np.ndarray, data_name: str = "data") -> np.ndarray:
    """
    Validate and clean plotting data to prevent NaN/Inf visualization issues.
    
    Args:
        data: Input data array
        data_name: Name for error reporting
        
    Returns:
        Cleaned data array
        
    Raises:
        ValueError: If data is invalid after cleaning
    """
    if data is None:
        raise ValueError(f"{data_name} cannot be None")
        
    data = np.asarray(data)
    
    if data.size == 0:
        raise ValueError(f"{data_name} cannot be empty")
    
    # Check for NaN/Inf values
    n_nan = np.sum(np.isnan(data))
    n_inf = np.sum(np.isinf(data))
    
    if n_nan > 0 or n_inf > 0:
        warnings.warn(
            f"{data_name} contains {n_nan} NaN and {n_inf} Inf values. "
            f"These will be filtered out, which may indicate upstream data corruption."
        )
        
        # Replace NaN/Inf with finite values or filter them out
        if data.ndim > 1:
            # For 2D+ arrays, replace with median
            finite_mask = np.isfinite(data)
            if np.any(finite_mask):
                median_val = np.median(data[finite_mask])
                data = np.where(np.isfinite(data), data, median_val)
            else:
                # All values are invalid
                raise ValueError(f"{data_name} contains only NaN/Inf values")
        else:
            # For 1D arrays, filter out invalid values
            finite_mask = np.isfinite(data)
            if np.any(finite_mask):
                data = data[finite_mask]
            else:
                raise ValueError(f"{data_name} contains only NaN/Inf values")
    
    return data


def validate_coordinate_data(coords: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate coordinate and value data together, ensuring consistency.
    
    Args:
        coords: Coordinate array (N, 2)
        values: Value array (N,)
        
    Returns:
        Tuple of cleaned (coords, values)
    """
    coords = validate_plot_data(coords, "coordinates")
    values = validate_plot_data(values, "values")
    
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


def plot_spatial_heatmap(
    coords: np.ndarray,
    values: np.ndarray,
    bin_size: float = 20.0,
    title: str = "Spatial Heatmap",
    cmap: str = "hot",
    figsize: Tuple[int, int] = (10, 8),
    ax: Optional[plt.Axes] = None
) -> Union[plt.Figure, plt.Axes]:
    """
    Create a binned heatmap of spatial data.
    
    Args:
        coords: (N, 2) array of coordinates
        values: (N,) array of values
        bin_size: Size of spatial bins in micrometers
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        ax: Existing axes
        
    Returns:
        Figure or Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        return_fig = False
    
    # Create bins
    x_edges = np.arange(coords[:, 0].min(), coords[:, 0].max() + bin_size, bin_size)
    y_edges = np.arange(coords[:, 1].min(), coords[:, 1].max() + bin_size, bin_size)
    
    # Compute 2D histogram
    H, xedges, yedges = np.histogram2d(
        coords[:, 0], coords[:, 1], 
        bins=[x_edges, y_edges],
        weights=values
    )
    
    # Count pixels per bin for averaging
    counts, _, _ = np.histogram2d(
        coords[:, 0], coords[:, 1],
        bins=[x_edges, y_edges]
    )
    
    # Average values per bin
    H = np.divide(H, counts, where=counts > 0)
    
    # Plot
    im = ax.imshow(
        H.T, origin='lower', cmap=cmap,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect='equal'
    )
    
    ax.set_title(title)
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    return fig if return_fig else ax


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
    coords = validate_plot_data(coords, "coordinates")
    
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
            cluster_map = validate_plot_data(cluster_map, f"cluster_map_scale_{scale}")
            im = ax.imshow(cluster_map, cmap='tab20', origin='lower')
            ax.set_title(f"Scale: {scale}μm")
            ax.set_xlabel("X bins")
            ax.set_ylabel("Y bins")
        elif 'cluster_labels' in scale_result and 'superpixel_coords' in scale_result:
            # Plot superpixel clusters
            sp_coords = scale_result['superpixel_coords']
            labels = scale_result['cluster_labels']
            # Validate superpixel data
            sp_coords = validate_plot_data(sp_coords, f"superpixel_coords_scale_{scale}")
            labels = validate_plot_data(labels, f"cluster_labels_scale_{scale}")
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
    correlation_matrix = validate_plot_data(correlation_matrix, "correlation_matrix")
    
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
    title: str = "Segmentation Overlay",
    figsize: Tuple[int, int] = (8, 8),
    ax: Optional[plt.Axes] = None
) -> Union[plt.Figure, plt.Axes]:
    """
    Overlays segmentation boundaries on an image for visual validation.
    
    Critical for assessing whether superpixels align with biological structures.
    Low inter-scale ARI is EXPECTED - different scales capture different biology.
    
    Args:
        image: The 2D image to display (e.g., DNA channel)
        labels: The 2D integer array of segmentation labels
        bounds: Tuple of (x_min, x_max, y_min, y_max) for axis scaling
        title: Plot title
        figsize: Figure size if creating a new figure
        ax: Existing axes to plot on
        
    Returns:
        Figure or Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        return_fig = False
    
    # Normalize image for display if needed
    if image.size > 0:
        img_normalized = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10)
    else:
        img_normalized = image
    
    # Create the boundary image using yellow lines for visibility
    overlay_img = mark_boundaries(img_normalized, labels, color=(1, 1, 0), mode='thick')
    
    # Display the image with correct spatial extent
    x_min, x_max, y_min, y_max = bounds
    ax.imshow(overlay_img, extent=[x_min, x_max, y_min, y_max], origin='lower')
    
    ax.set_title(title)
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.set_aspect('equal')
    
    # Add scale information
    n_segments = len(np.unique(labels))
    ax.text(0.02, 0.98, f"Segments: {n_segments}", 
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig if return_fig else ax