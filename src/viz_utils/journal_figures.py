"""Journal-quality figure generation for IMC analysis results.

Publication-ready figures for Nature Methods/Cell/Science standards.
All functions use real data from analysis results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import logging

# Set publication-quality defaults
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

logger = logging.getLogger(__name__)

class FigureGenerator:
    """Generate publication-quality figures from IMC analysis results."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration.
        
        Args:
            config: Configuration dictionary from config.json
        """
        self.config = config
        
        # Get visualization settings
        viz_config = config.get('visualization', {})
        self.colormaps = viz_config.get('validation_plots', {}).get('colormaps', {
            'immune_markers': 'Reds',
            'vascular_markers': 'Blues',
            'stromal_markers': 'Greens',
            'adhesion_markers': 'Purples',
            'default': 'viridis'
        })
        
        # Get channel groups for categorization
        self.channel_groups = config.get('channel_groups', {})
    
    def create_figure1_overview_quality(self, metadata_df: pd.DataFrame, 
                                       quality_df: pd.DataFrame) -> plt.Figure:
        """Create Figure 1: Experimental overview and quality control.
        
        Args:
            metadata_df: Metadata DataFrame
            quality_df: Quality scores DataFrame
            
        Returns:
            Figure object
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(180/25.4, 220/25.4))  # Double column width
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # A: Experimental design overview
        ax_design = fig.add_subplot(gs[0, :])
        self._plot_experimental_design(ax_design, metadata_df)
        
        # B: Quality score distributions
        ax_quality = fig.add_subplot(gs[1, :])
        self._plot_quality_distributions(ax_quality, quality_df)
        
        # C-E: Quality by condition
        quality_metrics = ['coordinate_quality', 'ion_count_quality', 'biological_quality']
        for i, metric in enumerate(quality_metrics):
            ax = fig.add_subplot(gs[2, i])
            self._plot_quality_by_condition(ax, quality_df, metric)
        
        # F-H: Quality by timepoint
        for i, metric in enumerate(quality_metrics):
            ax = fig.add_subplot(gs[3, i])
            self._plot_quality_by_timepoint(ax, quality_df, metric)
        
        # Add panel labels
        self._add_panel_labels(fig, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        
        plt.suptitle('Figure 1: Experimental Overview and Quality Control', 
                    fontsize=10, fontweight='bold', y=1.02)
        
        return fig
    
    def create_figure2_spatial_atlas(self, expression_df: pd.DataFrame) -> plt.Figure:
        """Create Figure 2: Spatial protein expression atlas.
        
        Args:
            expression_df: Expression matrix DataFrame
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(180/25.4, 200/25.4))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Get protein channels
        protein_cols = [col for col in expression_df.columns 
                       if col in self.config.get('channels', {}).get('protein_channels', [])]
        
        # Create heatmap for each protein
        for i, protein in enumerate(protein_cols[:9]):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            
            # Get colormap for this protein
            cmap = self._get_colormap_for_protein(protein)
            
            # Create expression heatmap by condition and timepoint
            self._plot_protein_expression_heatmap(ax, expression_df, protein, cmap)
            ax.set_title(protein, fontsize=9, fontweight='bold')
        
        plt.suptitle('Figure 2: Spatial Protein Expression Atlas', 
                    fontsize=10, fontweight='bold', y=1.02)
        
        return fig
    
    def create_figure3_temporal_dynamics(self, expression_df: pd.DataFrame) -> plt.Figure:
        """Create Figure 3: Temporal dynamics of protein expression.
        
        Args:
            expression_df: Expression matrix DataFrame
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(180/25.4, 150/25.4))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Get protein channels
        protein_cols = [col for col in expression_df.columns 
                       if col in self.config.get('channels', {}).get('protein_channels', [])]
        
        # Group proteins by function
        immune_markers = ['CD45', 'CD11b', 'Ly6G', 'CD206']
        stromal_markers = ['CD140a', 'CD140b']
        vascular_markers = ['CD31', 'CD34']
        
        # A: Immune marker dynamics
        ax_immune = fig.add_subplot(gs[0, 0])
        self._plot_temporal_dynamics(ax_immune, expression_df, immune_markers, 'Immune Markers')
        
        # B: Stromal marker dynamics
        ax_stromal = fig.add_subplot(gs[0, 1])
        self._plot_temporal_dynamics(ax_stromal, expression_df, stromal_markers, 'Stromal Markers')
        
        # C: Vascular marker dynamics
        ax_vascular = fig.add_subplot(gs[0, 2])
        self._plot_temporal_dynamics(ax_vascular, expression_df, vascular_markers, 'Vascular Markers')
        
        # D-F: Statistical significance heatmaps
        for i, (markers, title) in enumerate([
            (immune_markers, 'Immune'),
            (stromal_markers, 'Stromal'),
            (vascular_markers, 'Vascular')
        ]):
            ax = fig.add_subplot(gs[1, i])
            self._plot_significance_heatmap(ax, expression_df, markers, title)
        
        self._add_panel_labels(fig, ['A', 'B', 'C', 'D', 'E', 'F'])
        
        plt.suptitle('Figure 3: Temporal Dynamics of Protein Expression', 
                    fontsize=10, fontweight='bold', y=1.02)
        
        return fig
    
    def create_figure4_spatial_neighborhoods(self, neighborhood_data: Dict) -> plt.Figure:
        """Create Figure 4: Spatial neighborhood analysis.
        
        Args:
            neighborhood_data: Dictionary with neighborhood analysis results
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(180/25.4, 180/25.4))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # This would use actual neighborhood data from the analysis
        # For now, create placeholder structure
        
        plt.suptitle('Figure 4: Spatial Neighborhood Analysis', 
                    fontsize=10, fontweight='bold', y=1.02)
        
        return fig
    
    def create_figure5_regional_analysis(self, expression_df: pd.DataFrame) -> plt.Figure:
        """Create Figure 5: Regional heterogeneity analysis.
        
        Args:
            expression_df: Expression matrix DataFrame with region info
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(180/25.4, 150/25.4))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Filter for regions
        cortex_df = expression_df[expression_df['region'] == 'Cortex']
        medulla_df = expression_df[expression_df['region'] == 'Medulla']
        
        # A-B: Region-specific expression patterns
        ax_cortex = fig.add_subplot(gs[0, 0])
        self._plot_regional_expression(ax_cortex, cortex_df, 'Cortex')
        
        ax_medulla = fig.add_subplot(gs[0, 1])
        self._plot_regional_expression(ax_medulla, medulla_df, 'Medulla')
        
        # C: Regional comparison
        ax_compare = fig.add_subplot(gs[0, 2])
        self._plot_regional_comparison(ax_compare, expression_df)
        
        # D-F: Statistical tests for regional differences
        protein_groups = [
            (['CD45', 'CD11b'], 'Immune'),
            (['CD31', 'CD34'], 'Vascular'),
            (['CD140a', 'CD140b'], 'Stromal')
        ]
        
        for i, (proteins, title) in enumerate(protein_groups):
            ax = fig.add_subplot(gs[1, i])
            self._plot_regional_statistics(ax, expression_df, proteins, title)
        
        self._add_panel_labels(fig, ['A', 'B', 'C', 'D', 'E', 'F'])
        
        plt.suptitle('Figure 5: Regional Heterogeneity Analysis', 
                    fontsize=10, fontweight='bold', y=1.02)
        
        return fig
    
    # Helper methods
    def _plot_experimental_design(self, ax, metadata_df):
        """Plot experimental design overview."""
        # Count samples by condition and timepoint
        design_matrix = metadata_df.groupby(['condition', 'injury_day']).size().unstack(fill_value=0)
        
        # Create heatmap
        sns.heatmap(design_matrix, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': 'N samples'}, ax=ax)
        ax.set_xlabel('Injury Day')
        ax.set_ylabel('Condition')
        ax.set_title('Experimental Design', fontweight='bold')
    
    def _plot_quality_distributions(self, ax, quality_df):
        """Plot quality score distributions."""
        if quality_df.empty:
            ax.text(0.5, 0.5, 'No quality data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        quality_cols = ['coordinate_quality', 'ion_count_quality', 
                       'biological_quality', 'overall_quality']
        
        # Create violin plots
        data_to_plot = []
        labels = []
        for col in quality_cols:
            if col in quality_df.columns:
                data_to_plot.append(quality_df[col].dropna())
                labels.append(col.replace('_quality', '').title())
        
        if data_to_plot:
            parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                                 showmeans=True, showmedians=True)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Quality Score')
            ax.set_ylim([0, 1])
            ax.set_title('Quality Score Distributions', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def _plot_quality_by_condition(self, ax, quality_df, metric):
        """Plot quality metric by condition."""
        if quality_df.empty or metric not in quality_df.columns:
            ax.text(0.5, 0.5, f'No {metric} data', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        conditions = quality_df['condition'].unique()
        colors = sns.color_palette('Set2', len(conditions))
        
        for i, condition in enumerate(conditions):
            data = quality_df[quality_df['condition'] == condition][metric].dropna()
            if len(data) > 0:
                ax.boxplot(data, positions=[i], widths=0.6, 
                          patch_artist=True, 
                          boxprops=dict(facecolor=colors[i]))
        
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_quality", "").title()} by Condition')
        ax.grid(True, alpha=0.3)
    
    def _plot_quality_by_timepoint(self, ax, quality_df, metric):
        """Plot quality metric by timepoint."""
        if quality_df.empty or metric not in quality_df.columns:
            ax.text(0.5, 0.5, f'No {metric} data', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Group by injury day
        grouped = quality_df.groupby('injury_day')[metric].agg(['mean', 'std'])
        
        x = grouped.index
        y = grouped['mean']
        yerr = grouped['std']
        
        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5)
        ax.set_xlabel('Injury Day')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_quality", "").title()} Over Time')
        ax.grid(True, alpha=0.3)
    
    def _plot_protein_expression_heatmap(self, ax, expression_df, protein, cmap):
        """Plot protein expression heatmap."""
        if protein not in expression_df.columns:
            ax.text(0.5, 0.5, f'No {protein} data', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Pivot data for heatmap
        pivot_data = expression_df.pivot_table(
            values=protein, 
            index='condition', 
            columns='injury_day',
            aggfunc='mean'
        )
        
        if not pivot_data.empty:
            sns.heatmap(pivot_data, cmap=cmap, ax=ax, 
                       cbar_kws={'label': 'Expression'})
            ax.set_xlabel('Injury Day')
            ax.set_ylabel('Condition')
    
    def _plot_temporal_dynamics(self, ax, expression_df, proteins, title):
        """Plot temporal dynamics for a group of proteins."""
        for protein in proteins:
            if protein not in expression_df.columns:
                continue
            
            # Group by timepoint and calculate mean
            grouped = expression_df.groupby('injury_day')[protein].agg(['mean', 'sem'])
            
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['sem'],
                       label=protein, marker='o', capsize=3)
        
        ax.set_xlabel('Injury Day')
        ax.set_ylabel('Expression Level')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    def _plot_significance_heatmap(self, ax, expression_df, proteins, title):
        """Plot effect-size heatmap (log2 fold change between consecutive timepoints).

        Note: Does NOT display p-values. With n=2 mice per timepoint, p-values
        from pixel-level tests are pseudoreplicated and misleading. Effect sizes
        (fold changes) are descriptive and honest at any sample size.
        """
        timepoints = sorted(expression_df['injury_day'].unique())
        n_timepoints = len(timepoints)

        fc_matrix = np.zeros((len(proteins), n_timepoints - 1))

        for i, protein in enumerate(proteins):
            if protein not in expression_df.columns:
                continue

            for j in range(n_timepoints - 1):
                t1_data = expression_df[expression_df['injury_day'] == timepoints[j]][protein].dropna()
                t2_data = expression_df[expression_df['injury_day'] == timepoints[j+1]][protein].dropna()

                if len(t1_data) > 0 and len(t2_data) > 0:
                    eps = 1e-6
                    fc_matrix[i, j] = np.log2((t2_data.mean() + eps) / (t1_data.mean() + eps))

        im = ax.imshow(fc_matrix, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
        ax.set_xticks(range(n_timepoints - 1))
        ax.set_xticklabels([f'D{timepoints[i]} vs D{timepoints[i+1]}'
                           for i in range(n_timepoints - 1)], rotation=45, ha='right')
        ax.set_yticks(range(len(proteins)))
        ax.set_yticklabels(proteins)
        ax.set_title(f'{title} Log2 Fold Change', fontweight='bold')

        plt.colorbar(im, ax=ax, label='Log2 FC')
    
    def _plot_regional_expression(self, ax, region_df, region_name):
        """Plot expression patterns for a specific region."""
        proteins = [col for col in region_df.columns 
                   if col in self.config.get('channels', {}).get('protein_channels', [])]
        
        if not proteins or region_df.empty:
            ax.text(0.5, 0.5, f'No data for {region_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calculate mean expression
        mean_expression = region_df[proteins].mean()
        
        ax.bar(range(len(proteins)), mean_expression)
        ax.set_xticks(range(len(proteins)))
        ax.set_xticklabels(proteins, rotation=45, ha='right')
        ax.set_ylabel('Mean Expression')
        ax.set_title(f'{region_name} Expression', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_regional_comparison(self, ax, expression_df):
        """Plot comparison between regions."""
        proteins = [col for col in expression_df.columns 
                   if col in self.config.get('channels', {}).get('protein_channels', [])]
        
        if not proteins:
            return
        
        cortex_means = expression_df[expression_df['region'] == 'Cortex'][proteins].mean()
        medulla_means = expression_df[expression_df['region'] == 'Medulla'][proteins].mean()
        
        # Calculate fold change
        fold_change = np.log2((medulla_means + 1) / (cortex_means + 1))
        
        colors = ['red' if fc > 0 else 'blue' for fc in fold_change]
        ax.bar(range(len(proteins)), fold_change, color=colors)
        ax.set_xticks(range(len(proteins)))
        ax.set_xticklabels(proteins, rotation=45, ha='right')
        ax.set_ylabel('Log2 Fold Change (Medulla/Cortex)')
        ax.set_title('Regional Differences', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_regional_statistics(self, ax, expression_df, proteins, title):
        """Plot regional fold-change bar chart (no p-values).

        Displays log2 fold change (Medulla/Cortex) for each protein.
        P-values removed: with nested ROIs within n=2 mice, t-tests on
        pixel-level data are pseudoreplicated.
        """
        fold_changes = []
        valid_proteins = []

        for protein in proteins:
            if protein not in expression_df.columns:
                continue

            cortex_data = expression_df[expression_df['region'] == 'Cortex'][protein].dropna()
            medulla_data = expression_df[expression_df['region'] == 'Medulla'][protein].dropna()

            if len(cortex_data) > 0 and len(medulla_data) > 0:
                eps = 1e-6
                fc = np.log2((medulla_data.mean() + eps) / (cortex_data.mean() + eps))
                fold_changes.append(fc)
                valid_proteins.append(protein)

        if fold_changes:
            colors = ['#E63946' if fc > 0 else '#457B9D' for fc in fold_changes]
            ax.barh(range(len(valid_proteins)), fold_changes, color=colors, alpha=0.8)
            ax.set_yticks(range(len(valid_proteins)))
            ax.set_yticklabels(valid_proteins)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_xlabel('Log2 Fold Change (Medulla / Cortex)')
        ax.set_title(f'{title} Regional Differences', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    def _get_colormap_for_protein(self, protein: str) -> str:
        """Get appropriate colormap for protein based on its function."""
        # Check channel groups
        for group_name, group_info in self.channel_groups.items():
            if isinstance(group_info, dict):
                for subgroup, proteins in group_info.items():
                    if protein in proteins:
                        if 'immune' in group_name.lower():
                            return self.colormaps.get('immune_markers', 'Reds')
                        elif 'vascular' in group_name.lower():
                            return self.colormaps.get('vascular_markers', 'Blues')
                        elif 'stromal' in group_name.lower():
                            return self.colormaps.get('stromal_markers', 'Greens')
                        elif 'adhesion' in group_name.lower():
                            return self.colormaps.get('adhesion_markers', 'Purples')
        
        return self.colormaps.get('default', 'viridis')
    
    def _add_panel_labels(self, fig, labels: List[str]):
        """Add panel labels (A, B, C, etc.) to figure."""
        axes = fig.get_axes()
        for ax, label in zip(axes[:len(labels)], labels):
            ax.text(-0.15, 1.15, label, transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='top')