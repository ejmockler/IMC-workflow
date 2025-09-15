"""Multi-scale neighborhood visualization extension for ROI analysis."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def plot_multiscale_neighborhoods_row(grid, roi_data: Dict, row: int):
    """Plot multi-scale neighborhood analysis in the specified row."""
    multiscale = roi_data.get('multiscale_neighborhoods', {})
    
    if not multiscale:
        return
    
    # Col 0: Multi-scale neighborhood map
    ax = grid.get(row, 0)
    plot_multiscale_neighborhood_map(ax, roi_data)
    
    # Col 1: Scale-dependent fragmentation
    ax = grid.get(row, 1)
    plot_scale_fragmentation(ax, multiscale)
    
    # Col 2: Cross-scale correlation
    ax = grid.get(row, 2)
    plot_scale_correlation(ax, multiscale, roi_data)
    
    # Col 3: Neighborhood hierarchy
    ax = grid.get(row, 3)
    plot_neighborhood_hierarchy(ax, multiscale)


def plot_multiscale_neighborhood_map(ax, roi_data: Dict):
    """Overlay neighborhoods at different scales."""
    coords = np.array(roi_data['coords'])
    multiscale = roi_data.get('multiscale_neighborhoods', {})
    
    # Color scheme for different scales
    scale_colors = {
        'cellular': 'red',
        'microenvironment': 'blue', 
        'functional_unit': 'green',
        'tissue_region': 'purple'
    }
    
    # Plot base tissue
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=0.5, alpha=0.3)
    
    # Overlay boundaries from each scale
    for scale_name, neighborhoods in multiscale.items():
        if neighborhoods and 'pixel_assignments' in neighborhoods:
            color = scale_colors.get(scale_name, 'black')
            assignments = neighborhoods['pixel_assignments']
            
            # Find boundary pixels (simplified)
            boundary_mask = np.zeros(len(coords), dtype=bool)
            for i in range(1, len(coords)):
                if assignments[i] != assignments[i-1]:
                    boundary_mask[i] = True
            
            boundary_coords = coords[boundary_mask]
            if len(boundary_coords) > 0:
                ax.scatter(boundary_coords[:, 0], boundary_coords[:, 1], 
                         c=color, s=1, alpha=0.7, label=scale_name)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Multi-Scale Neighborhood Boundaries')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')


def plot_scale_fragmentation(ax, multiscale: Dict):
    """Plot fragmentation metrics across scales."""
    scales = []
    n_neighborhoods = []
    radii = []
    
    # Order scales by radius
    scale_order = ['cellular', 'microenvironment', 'functional_unit', 'tissue_region']
    
    for scale_name in scale_order:
        if scale_name in multiscale:
            data = multiscale[scale_name]
            scales.append(scale_name)
            n_neighborhoods.append(data.get('n_neighborhoods', 0))
            radii.append(data.get('scale_radius', 0))
    
    if scales:
        x = np.arange(len(scales))
        ax.bar(x, n_neighborhoods, color='steelblue', alpha=0.7)
        
        # Add radius labels
        for i, (scale, radius) in enumerate(zip(scales, radii)):
            ax.text(i, n_neighborhoods[i] + 0.5, f'{radius}μm', 
                   ha='center', fontsize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(scales, rotation=45, ha='right')
        ax.set_ylabel('Number of Neighborhoods')
        ax.set_title('Scale-Dependent Fragmentation')
        ax.grid(axis='y', alpha=0.3)


def plot_scale_correlation(ax, multiscale: Dict, roi_data: Dict):
    """Plot correlation between scales."""
    scale_order = ['cellular', 'microenvironment', 'functional_unit', 'tissue_region']
    available_scales = [s for s in scale_order if s in multiscale]
    
    if len(available_scales) < 2:
        ax.text(0.5, 0.5, 'Insufficient scales for correlation', 
               ha='center', va='center')
        ax.axis('off')
        return
    
    # Create correlation matrix based on neighborhood overlap
    n_scales = len(available_scales)
    correlation_matrix = np.zeros((n_scales, n_scales))
    
    for i, scale1 in enumerate(available_scales):
        for j, scale2 in enumerate(available_scales):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                # Calculate Jaccard similarity between neighborhood assignments
                assign1 = multiscale[scale1].get('pixel_assignments', [])
                assign2 = multiscale[scale2].get('pixel_assignments', [])
                
                if len(assign1) > 0 and len(assign2) > 0:
                    # Simple overlap metric
                    same_neighborhood = (assign1 == assign2).sum()
                    total = len(assign1)
                    correlation_matrix[i, j] = same_neighborhood / total if total > 0 else 0
    
    # Plot heatmap
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n_scales))
    ax.set_yticks(range(n_scales))
    ax.set_xticklabels(available_scales, rotation=45, ha='right')
    ax.set_yticklabels(available_scales)
    ax.set_title('Cross-Scale Correlation')
    
    # Add text annotations
    for i in range(n_scales):
        for j in range(n_scales):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                         ha='center', va='center', 
                         color='black' if correlation_matrix[i, j] > 0.5 else 'white',
                         fontsize=8)
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_neighborhood_hierarchy(ax, multiscale: Dict):
    """Plot hierarchical organization of neighborhoods."""
    scale_order = ['cellular', 'microenvironment', 'functional_unit', 'tissue_region']
    
    y_pos = 0
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    for scale_idx, scale_name in enumerate(scale_order):
        if scale_name in multiscale:
            data = multiscale[scale_name]
            neighborhoods = data.get('neighborhoods', {})
            
            # Plot each neighborhood as a bar
            for nbhd_id, nbhd_data in neighborhoods.items():
                coverage = nbhd_data.get('coverage', 0)
                dominant = nbhd_data.get('dominant_pairs', ['Unknown'])[0]
                
                ax.barh(y_pos, coverage, height=0.8, 
                       color=colors[nbhd_id % len(colors)],
                       alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Add label if coverage is significant
                if coverage > 0.05:
                    ax.text(coverage/2, y_pos, dominant[:10], 
                           ha='center', va='center', fontsize=6)
                
                y_pos += 1
            
            # Add scale separator
            if neighborhoods:
                ax.axhline(y_pos - 0.5, color='gray', linestyle='--', alpha=0.5)
                ax.text(-0.02, y_pos - len(neighborhoods)/2 - 0.5, scale_name, 
                       transform=ax.transData, rotation=0, ha='right', fontsize=8)
    
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Neighborhoods by Scale')
    ax.set_title('Hierarchical Neighborhood Organization')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)