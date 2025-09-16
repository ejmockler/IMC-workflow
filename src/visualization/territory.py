"""
Territory-focused visualization for MSPT framework

This module creates visualizations that leverage the unified MSPT approach,
specifically designed for Nature Methods publication standards.

Key visualizations:
- Spatial territory maps showing protein expression patterns
- Multi-scale interaction heatmaps with statistical significance
- Territory abundance changes over time/conditions
- Territory-specific protein signatures
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import warnings

from src.config import Config


class TerritoryVisualizer:
    """Creates territory-focused visualizations for MSPT analysis."""
    
    def __init__(self, config: Config):
        """Initialize territory visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.figure_size = config.raw.get('visualization', {}).get('figure_size', [16, 12])
        self.dpi = config.raw.get('visualization', {}).get('dpi', 300)
        self.color_palette = config.raw.get('visualization', {}).get('color_palette', 'viridis')
    
    def create_roi_territory_map(self, roi_result: Dict) -> plt.Figure:
        """Create spatial territory map for single ROI.
        
        Args:
            roi_result: MSPT analysis result for single ROI
            
        Returns:
            Figure showing spatial territory distribution
        """
        if 'territory_discovery' not in roi_result:
            return self._create_empty_figure("No territory data available")
        
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        fig.suptitle(f"Protein Territory Map: {roi_result.get('filename', 'Unknown ROI')}", 
                    fontsize=14, fontweight='bold')
        
        territories = roi_result['territory_discovery']['territory_definitions']
        assignments = np.array(roi_result['territory_discovery']['assignments'])
        
        if len(territories) == 0:
            return self._create_empty_figure("No territories discovered")
        
        # Extract spatial coordinates (we'll need to reconstruct from spatial pixels)
        # For now, create dummy coordinates - this should be enhanced with actual pixel coordinates
        n_pixels = len(assignments)
        coords = self._generate_dummy_coordinates(n_pixels)
        
        # Plot 1: Territory spatial map
        ax1 = axes[0, 0]
        scatter = ax1.scatter(coords[:, 0], coords[:, 1], c=assignments, 
                            cmap=self.color_palette, s=1, alpha=0.7)
        ax1.set_title('Spatial Territory Distribution')
        ax1.set_xlabel('X position (μm)')
        ax1.set_ylabel('Y position (μm)')
        plt.colorbar(scatter, ax=ax1, label='Territory ID')
        
        # Plot 2: Territory abundance
        ax2 = axes[0, 1]
        territory_ids = [t['territory_id'] for t in territories]
        territory_names = [t['name'][:20] + '...' if len(t['name']) > 20 else t['name'] 
                          for t in territories]
        territory_sizes = [t['n_pixels'] for t in territories]
        
        bars = ax2.bar(range(len(territory_ids)), territory_sizes, 
                      color=plt.cm.get_cmap(self.color_palette)(np.linspace(0, 1, len(territory_ids))))
        ax2.set_title('Territory Abundance')
        ax2.set_xlabel('Territory')
        ax2.set_ylabel('Number of Spatial Pixels')
        ax2.set_xticks(range(len(territory_ids)))
        ax2.set_xticklabels([f'T{tid}' for tid in territory_ids], rotation=45)
        
        # Plot 3: Protein signatures heatmap
        ax3 = axes[1, 0]
        protein_signatures = []
        protein_names = roi_result.get('protein_names', [])
        
        for territory in territories:
            signature = territory['marker_signature']
            # Ensure consistent ordering
            sig_values = [signature.get(protein, 0) for protein in protein_names]
            protein_signatures.append(sig_values)
        
        if protein_signatures:
            signature_matrix = np.array(protein_signatures)
            im = ax3.imshow(signature_matrix, cmap='RdYlBu_r', aspect='auto')
            ax3.set_title('Territory Protein Signatures')
            ax3.set_xlabel('Protein Markers')
            ax3.set_ylabel('Territory')
            ax3.set_xticks(range(len(protein_names)))
            ax3.set_xticklabels(protein_names, rotation=45, ha='right')
            ax3.set_yticks(range(len(territory_ids)))
            ax3.set_yticklabels([f'T{tid}' for tid in territory_ids])
            plt.colorbar(im, ax=ax3, label='Expression Level')
        
        # Plot 4: Nuclear context
        ax4 = axes[1, 1]
        nuclear_fractions = [t['fraction_nuclear'] for t in territories]
        bars = ax4.bar(range(len(territory_ids)), nuclear_fractions,
                      color=plt.cm.get_cmap(self.color_palette)(np.linspace(0, 1, len(territory_ids))))
        ax4.set_title('Nuclear Context by Territory')
        ax4.set_xlabel('Territory')
        ax4.set_ylabel('Fraction Nuclear')
        ax4.set_xticks(range(len(territory_ids)))
        ax4.set_xticklabels([f'T{tid}' for tid in territory_ids], rotation=45)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def create_spatial_interaction_heatmap(self, roi_result: Dict, radius: float = 25.0) -> plt.Figure:
        """Create spatial interaction heatmap for specific radius.
        
        Args:
            roi_result: MSPT analysis result
            radius: Spatial radius to visualize (μm)
            
        Returns:
            Figure showing spatial interaction matrix
        """
        if 'spatial_analysis' not in roi_result:
            return self._create_empty_figure("No spatial analysis data available")
        
        spatial_results = roi_result['spatial_analysis']
        scale_key = f'{radius}um'
        
        if scale_key not in spatial_results.get('multi_scale_interactions', {}):
            return self._create_empty_figure(f"No spatial data for {radius}μm radius")
        
        scale_data = spatial_results['multi_scale_interactions'][scale_key]
        interaction_matrix = scale_data.get('interaction_matrix', np.array([]))
        
        if interaction_matrix.size == 0:
            return self._create_empty_figure("Empty interaction matrix")
        
        territories = roi_result['territory_discovery']['territory_definitions']
        territory_names = [f"T{t['territory_id']}: {t['name'][:15]}" for t in territories]
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        # Create heatmap
        im = ax.imshow(interaction_matrix, cmap='RdBu_r', vmin=-2, vmax=2)
        
        # Set labels
        ax.set_xticks(range(len(territory_names)))
        ax.set_yticks(range(len(territory_names)))
        ax.set_xticklabels(territory_names, rotation=45, ha='right')
        ax.set_yticklabels(territory_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Spatial Enrichment Ratio', rotation=270, labelpad=20)
        
        # Add significance annotations
        significant_interactions = scale_data.get('significant_interactions', [])
        for interaction in significant_interactions:
            if interaction.is_significant:
                # Find matrix position
                territory_ids = [t['territory_id'] for t in territories]
                try:
                    i = territory_ids.index(interaction.territory_a)
                    j = territory_ids.index(interaction.territory_b)
                    
                    # Add significance marker
                    if interaction.p_value < 0.001:
                        marker = '***'
                    elif interaction.p_value < 0.01:
                        marker = '**'
                    elif interaction.p_value < 0.05:
                        marker = '*'
                    else:
                        marker = ''
                    
                    if marker:
                        ax.text(j, i, marker, ha='center', va='center', 
                               color='white', fontweight='bold', fontsize=12)
                        ax.text(i, j, marker, ha='center', va='center', 
                               color='white', fontweight='bold', fontsize=12)
                
                except ValueError:
                    continue
        
        ax.set_title(f'Spatial Territory Interactions ({radius}μm radius)\n'
                    f'*** p<0.001, ** p<0.01, * p<0.05')
        
        plt.tight_layout()
        return fig
    
    def create_multi_scale_summary(self, roi_result: Dict) -> plt.Figure:
        """Create multi-scale spatial analysis summary.
        
        Args:
            roi_result: MSPT analysis result
            
        Returns:
            Figure showing multi-scale interaction patterns
        """
        if 'spatial_analysis' not in roi_result:
            return self._create_empty_figure("No spatial analysis data available")
        
        spatial_results = roi_result['spatial_analysis']
        multi_scale = spatial_results.get('multi_scale_interactions', {})
        
        if not multi_scale:
            return self._create_empty_figure("No multi-scale data available")
        
        # Extract interaction data across scales
        scales = sorted([float(scale.replace('um', '')) for scale in multi_scale.keys()])
        scale_keys = [f'{scale}um' for scale in scales]
        
        fig, axes = plt.subplots(1, len(scales), figsize=(4*len(scales), 6), dpi=self.dpi)
        if len(scales) == 1:
            axes = [axes]
        
        fig.suptitle('Multi-Scale Spatial Territory Interactions', fontsize=14, fontweight='bold')
        
        territories = roi_result['territory_discovery']['territory_definitions']
        territory_names = [f"T{t['territory_id']}" for t in territories]
        
        for i, (scale, scale_key) in enumerate(zip(scales, scale_keys)):
            ax = axes[i]
            
            interaction_matrix = multi_scale[scale_key].get('interaction_matrix', np.array([]))
            
            if interaction_matrix.size > 0:
                im = ax.imshow(interaction_matrix, cmap='RdBu_r', vmin=-2, vmax=2)
                
                ax.set_xticks(range(len(territory_names)))
                ax.set_yticks(range(len(territory_names)))
                ax.set_xticklabels(territory_names, rotation=45)
                ax.set_yticklabels(territory_names)
                
                ax.set_title(f'{scale}μm radius')
                
                # Add significance markers
                significant_interactions = multi_scale[scale_key].get('significant_interactions', [])
                for interaction in significant_interactions:
                    if interaction.is_significant:
                        territory_ids = [t['territory_id'] for t in territories]
                        try:
                            row = territory_ids.index(interaction.territory_a)
                            col = territory_ids.index(interaction.territory_b)
                            
                            marker = '●' if interaction.p_value < 0.05 else ''
                            if marker:
                                ax.text(col, row, marker, ha='center', va='center', 
                                       color='white', fontweight='bold', fontsize=8)
                                ax.text(row, col, marker, ha='center', va='center', 
                                       color='white', fontweight='bold', fontsize=8)
                        except ValueError:
                            continue
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{scale}μm radius')
        
        # Add shared colorbar
        if len(scales) > 0 and multi_scale[scale_keys[0]].get('interaction_matrix', np.array([])).size > 0:
            cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=20)
            cbar.set_label('Spatial Enrichment Ratio', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig
    
    def create_territory_temporal_analysis(self, results: List[Dict]) -> plt.Figure:
        """Create temporal analysis of territory changes.
        
        Args:
            results: List of MSPT analysis results across timepoints
            
        Returns:
            Figure showing territory changes over time
        """
        if not results:
            return self._create_empty_figure("No results provided")
        
        # Group results by timepoint
        timepoint_data = defaultdict(list)
        for result in results:
            if 'metadata' in result and hasattr(result['metadata'], 'injury_day'):
                timepoint = result['metadata'].injury_day
                if timepoint is not None:
                    timepoint_data[timepoint].append(result)
        
        if not timepoint_data:
            return self._create_empty_figure("No temporal metadata available")
        
        timepoints = sorted(timepoint_data.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        fig.suptitle('Temporal Territory Analysis', fontsize=14, fontweight='bold')
        
        # Plot 1: Territory count over time
        ax1 = axes[0, 0]
        territory_counts_per_timepoint = []
        for tp in timepoints:
            tp_results = timepoint_data[tp]
            counts = [r['territory_discovery']['n_territories'] for r in tp_results 
                     if 'territory_discovery' in r]
            territory_counts_per_timepoint.append(counts)
        
        if any(territory_counts_per_timepoint):
            bp1 = ax1.boxplot(territory_counts_per_timepoint, positions=timepoints, patch_artist=True)
            for patch in bp1['boxes']:
                patch.set_facecolor('lightblue')
            ax1.set_xlabel('Days Post-Injury')
            ax1.set_ylabel('Number of Territories')
            ax1.set_title('Territory Diversity Over Time')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coverage over time
        ax2 = axes[0, 1]
        coverage_per_timepoint = []
        for tp in timepoints:
            tp_results = timepoint_data[tp]
            coverages = [r['coverage_metrics']['analyzed_fraction'] for r in tp_results 
                        if 'coverage_metrics' in r]
            coverage_per_timepoint.append(coverages)
        
        if any(coverage_per_timepoint):
            bp2 = ax2.boxplot(coverage_per_timepoint, positions=timepoints, patch_artist=True)
            for patch in bp2['boxes']:
                patch.set_facecolor('lightgreen')
            ax2.set_xlabel('Days Post-Injury')
            ax2.set_ylabel('Tissue Coverage Fraction')
            ax2.set_title('Analysis Coverage Over Time')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Territory abundance changes (placeholder)
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.5, 'Territory Abundance\nChanges\n(Implementation pending)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Territory Abundance Changes')
        
        # Plot 4: Spatial interaction changes (placeholder)
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, 'Spatial Interaction\nChanges\n(Implementation pending)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Spatial Interaction Changes')
        
        plt.tight_layout()
        return fig
    
    def _generate_dummy_coordinates(self, n_pixels: int) -> np.ndarray:
        """Generate dummy coordinates for visualization (placeholder).
        
        This should be replaced with actual spatial pixel coordinates.
        """
        # Create a rough grid layout
        grid_size = int(np.sqrt(n_pixels)) + 1
        x = np.repeat(np.arange(grid_size), grid_size)[:n_pixels]
        y = np.tile(np.arange(grid_size), grid_size)[:n_pixels]
        
        # Add some noise for visual appeal
        x = x + np.random.normal(0, 0.1, n_pixels)
        y = y + np.random.normal(0, 0.1, n_pixels)
        
        return np.column_stack([x, y])
    
    def _create_empty_figure(self, message: str) -> plt.Figure:
        """Create empty figure with message."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        ax.text(0.5, 0.5, message, ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return fig