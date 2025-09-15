"""ROI visualization module for single ROI analysis."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from collections import Counter
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list

from src.config import Config
from src.utils.helpers import PlotGrid, add_dendrogram_to_heatmap
from src.visualization.components import (
    plot_domain_signatures_heatmap,
    plot_spatial_contact_matrix,
    plot_domain_size_distribution,
    plot_top_domain_contacts,
    plot_spatial_domains,
    plot_tissue_neighborhoods,
    plot_neighborhood_entropy_map,
    plot_neighborhood_composition,
    plot_neighborhood_statistics
)
from src.visualization.roi_multiscale import (
    plot_multiscale_neighborhoods_row
)


class ROIVisualizer:
    """Creates visualizations for single ROI."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_figure(self, roi_data: Dict, extended=True) -> plt.Figure:
        """Create comprehensive ROI visualization.
        
        Args:
            roi_data: ROI analysis results
            extended: If True, use 4x4 grid with multi-scale neighborhoods. If False, use 3x3.
        """
        # Check if we have multi-scale data
        has_multiscale = 'multiscale_neighborhoods' in roi_data and roi_data['multiscale_neighborhoods']
        
        if extended and has_multiscale:
            # Extended 4x4 grid for multi-scale analysis
            grid = PlotGrid(4, 4, figsize=(32, 24))
        else:
            # Standard 3x3 grid
            grid = PlotGrid(3, 3, figsize=(24, 18))
        
        # Row 0: All heatmaps  
        ax = grid.get(0, 0)
        self._plot_domain_signatures(ax, roi_data)
        
        ax = grid.get(0, 1)
        self._plot_spatial_contacts(ax, roi_data)
        
        ax = grid.get(0, 2)
        self._plot_colocalization_matrix(ax, roi_data)
        
        # Row 1: Spatial maps (including neighborhoods)
        ax = grid.get(1, 0)
        plot_spatial_domains(ax, roi_data['coords'], 
                           roi_data['blob_labels'],
                           roi_data['blob_signatures'])
        
        ax = grid.get(1, 1)
        # Show neighborhoods if enabled, otherwise show spatial autocorrelation
        if roi_data.get('neighborhoods') is not None:
            plot_tissue_neighborhoods(ax, roi_data['coords'],
                                    roi_data.get('neighborhoods'),
                                    roi_data['blob_signatures'])
        else:
            self._plot_spatial_autocorrelation(ax, roi_data)
        
        ax = grid.get(1, 2)
        self._plot_functional_groups(ax, roi_data)
        
        # Row 2: Distribution plots
        ax = grid.get(2, 0)
        plot_domain_size_distribution(ax, roi_data['blob_signatures'],
                                     roi_data['protein_names'])
        
        ax = grid.get(2, 1)
        plot_top_domain_contacts(ax, roi_data['blob_contacts'],
                                roi_data['blob_signatures'])
        
        ax = grid.get(2, 2)
        # Show neighborhood composition if enabled, otherwise show organization scales
        if roi_data.get('neighborhoods') is not None:
            plot_neighborhood_composition(ax, roi_data.get('neighborhoods'),
                                         roi_data['blob_signatures'])
        else:
            self._plot_organization_scales(ax, roi_data)
        
        # Row 3: Multi-scale neighborhood analysis (if extended grid)
        if extended and has_multiscale:
            plot_multiscale_neighborhoods_row(grid, roi_data, row=3)
        
        # Title
        meta = roi_data['metadata']
        # Handle both dict and Metadata object formats
        if isinstance(meta, dict):
            condition = meta.get('condition', 'Unknown')
            day = meta.get('injury_day') or meta.get('timepoint', '?')
            region = meta.get('tissue_region') or meta.get('region', 'Unknown')
            display_name = f"{condition} D{day} {region}"
        else:
            display_name = meta.display_name
        title = f"Spatial Tissue Organization: {display_name} — {roi_data['filename']}"
        grid.fig.suptitle(title, fontsize=14, fontweight='bold')
        
        return grid.fig
    
    def _plot_domain_signatures(self, ax, roi_data):
        """Plot domain signature heatmap with dendrogram."""
        # Build matrix domains x proteins from blob_signatures
        blob_compositions = []
        blob_names = []
        valid_blobs = sorted(roi_data['blob_signatures'].keys(),
                             key=lambda x: roi_data['blob_signatures'][x]['size'], reverse=True)[:15]
        for key in valid_blobs:
            sig = roi_data['blob_signatures'][key]
            blob_names.append('+'.join(sig['dominant_proteins'][:2]))
            blob_compositions.append(sig['mean_expression'])
        
        if blob_compositions:
            M = np.array(blob_compositions)
            if M.max() > 0:
                M = M / M.max()
            # Add dendrogram on rows (proteins)
            heat_ax, Z, order = add_dendrogram_to_heatmap(ax, M.T)
            proteins = roi_data['protein_names'][:M.shape[1]]
            proteins_ordered = [proteins[i] for i in order]
            M_ordered = (M.T)[order, :]
            im = heat_ax.imshow(M_ordered, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            heat_ax.set_yticks(range(len(proteins_ordered)))
            heat_ax.set_yticklabels(proteins_ordered, fontsize=8)
            heat_ax.set_xticks(range(len(blob_names)))
            heat_ax.set_xticklabels(blob_names, rotation=45, ha='right', fontsize=8)
            heat_ax.set_title('Domain Protein Signatures')
            plt.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04)
        else:
            plot_domain_signatures_heatmap(ax, roi_data['blob_signatures'], roi_data['protein_names'])
    
    def _plot_spatial_contacts(self, ax, roi_data):
        """Plot spatial contact matrix with dendrogram."""
        M, row_labels, col_labels = plot_spatial_contact_matrix(
            ax,
            roi_data['blob_labels'],
            roi_data['blob_signatures'],
            roi_data['coords'],
            radius=self.config.contact_radius,
            sample_step=1,
            max_neighbors=100,
            blob_type_mapping=roi_data['blob_type_mapping'],
            protein_names=roi_data['protein_names'],
            return_data=True
        ) or (None, None, None)
        
        if M is not None:
            heat_ax, Z, order = add_dendrogram_to_heatmap(ax, M)
            M_ordered = M[order, :]
            rows_ordered = [row_labels[i] for i in order]
            im = heat_ax.imshow(M_ordered, cmap='Blues', aspect='auto', vmin=0)
            heat_ax.set_yticks(range(len(rows_ordered)))
            heat_ax.set_yticklabels(rows_ordered, fontsize=8)
            heat_ax.set_xticks(range(len(col_labels)))
            heat_ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
            heat_ax.set_title('Protein-Domain Spatial Contacts (Adjacency frequency)')
            plt.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04)
    
    def _plot_functional_groups(self, ax, roi_data):
        """Plot functional group composition analysis."""
        functional_groups = self.config.functional_groups
        
        # Calculate functional group percentages
        group_sizes = {}
        total_pixels = roi_data['total_pixels']
        
        for group_name, proteins in functional_groups.items():
            if group_name == 'structural_controls':
                continue
                
            group_size = 0
            for sig in roi_data['blob_signatures'].values():
                domain_proteins = sig['dominant_proteins'][:2]
                # Check if domain belongs to this functional group
                if any(p in proteins for p in domain_proteins):
                    group_size += sig['size']
            
            group_sizes[group_name] = 100 * group_size / total_pixels
        
        # Create pie chart
        if group_sizes:
            labels = [name.replace('_', ' ').title() for name in group_sizes.keys()]
            sizes = list(group_sizes.values())
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(sizes)]
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
            ax.set_title('Functional Group Composition')
            
            # Enhance text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax.text(0.5, 0.5, 'No functional group data', ha='center', va='center')
            ax.axis('off')
    
    def _plot_spatial_autocorrelation(self, ax, roi_data):
        """Plot spatial autocorrelation by distance (top proteins by organization)."""
        pipeline_file = roi_data['filename'].replace('.txt', '_pipeline_analysis.json')
        pipeline_path = self.config.data_dir / pipeline_file
        
        if pipeline_path.exists():
            with open(pipeline_path, 'r') as f:
                pipeline_data = json.load(f)
            
            autocorr_data = pipeline_data.get('spatial_autocorrelation', {})
            if autocorr_data:
                # Determine available distances from data
                sample = next(iter(autocorr_data.values())) if autocorr_data else {}
                try:
                    distances = sorted([int(d) for d in sample.keys()])
                except Exception:
                    distances = [5, 10, 25, 50]
                
                # Rank proteins by mean autocorrelation
                means = []
                for protein, values in autocorr_data.items():
                    corrs = [values.get(str(d), 0) for d in distances]
                    mean_corr = float(np.mean(corrs)) if corrs else 0.0
                    means.append((protein.split('(')[0], mean_corr))
                means.sort(key=lambda x: x[1], reverse=True)
                
                top = means[:8]
                for protein, _ in top:
                    values = autocorr_data.get(protein, {}) or next((v for k, v in autocorr_data.items() if k.startswith(protein)), {})
                    corrs = [values.get(str(d), 0) for d in distances]
                    ax.plot(distances, corrs, marker='o', label=protein, linewidth=2)
                
                ax.set_xlabel('Distance (μm)')
                ax.set_ylabel('Spatial Autocorrelation')
                ax.set_title('Top Protein Spatial Organization')
                if top:
                    ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No autocorrelation data', ha='center', va='center')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Pipeline data not found', ha='center', va='center')
            ax.axis('off')
    
    def _plot_colocalization_matrix(self, ax, roi_data):
        """Plot proteins (rows) vs top colocalized pairs (columns), with row clustering."""
        pipeline_file = roi_data['filename'].replace('.txt', '_pipeline_analysis.json')
        pipeline_path = self.config.data_dir / pipeline_file
        
        if pipeline_path.exists():
            with open(pipeline_path, 'r') as f:
                pipeline_data = json.load(f)
            
            coloc_data = pipeline_data.get('colocalization', {})
            if coloc_data:
                # Build list of pairs and scores
                pairs = []  # (label, p1, p2, score)
                for _, data in coloc_data.items():
                    score = data.get('colocalization_score', 0.0)
                    p1 = data.get('protein_1', '').split('(')[0]
                    p2 = data.get('protein_2', '').split('(')[0]
                    if p1 and p2:
                        label = f"{p1}↔{p2}"
                        pairs.append((label, p1, p2, score))
                # Sort and filter
                pairs.sort(key=lambda x: x[3], reverse=True)
                pairs = [p for p in pairs if p[3] > 0.05][:20]
                
                if pairs:
                    # Columns: pair labels
                    pair_labels = [label for label, _, _, _ in pairs]
                    # Rows: unique proteins involved
                    prot_counts = Counter([p1 for _, p1, _, _ in pairs] + [p2 for _, _, p2, _ in pairs])
                    proteins = [p for p, _ in prot_counts.most_common()]
                    n_rows = len(proteins)
                    n_cols = len(pair_labels)
                    matrix = np.zeros((n_rows, n_cols))
                    row_index = {p: i for i, p in enumerate(proteins)}
                    for j, (_, p1, p2, s) in enumerate(pairs):
                        if p1 in row_index:
                            matrix[row_index[p1], j] = s
                        if p2 in row_index:
                            matrix[row_index[p2], j] = s
                    # Cluster rows (proteins)
                    if n_rows > 1:
                        dist = pdist(matrix, metric='euclidean')
                        Z = linkage(dist, method='average')
                        order = leaves_list(Z)
                        matrix = matrix[order, :]
                        proteins = [proteins[i] for i in order]
                    
                    heat_ax, Z, order = add_dendrogram_to_heatmap(ax, matrix)
                    matrix = matrix[order, :]
                    proteins = [proteins[i] for i in order]
                    im = heat_ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
                    heat_ax.set_yticks(range(len(proteins)))
                    heat_ax.set_yticklabels(proteins, fontsize=8)
                    heat_ax.set_xticks(range(len(pair_labels)))
                    heat_ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
                    heat_ax.set_xlabel('Top colocalized pairs')
                    heat_ax.set_title('Protein vs top colocalized pairs (Pearson r)')
                    plt.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04, label='Pearson r')
                else:
                    ax.text(0.5, 0.5, 'No protein pairs found', ha='center', va='center')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No colocalization data', ha='center', va='center')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Pipeline data not found', ha='center', va='center')
            ax.axis('off')
    
    def _plot_organization_scales(self, ax, roi_data):
        """Plot organization scales by protein (top-N)."""
        pipeline_file = roi_data['filename'].replace('.txt', '_pipeline_analysis.json')
        pipeline_path = self.config.data_dir / pipeline_file
        
        if pipeline_path.exists():
            with open(pipeline_path, 'r') as f:
                pipeline_data = json.load(f)
            
            org_data = pipeline_data.get('spatial_organization', {})
            if org_data:
                items = []
                for protein, data in org_data.items():
                    protein_clean = protein.split('(')[0]
                    items.append((protein_clean, data.get('organization_scale', 0)))
                items.sort(key=lambda x: x[1], reverse=True)
                
                proteins = [p for p, _ in items[:10]]
                scales = [s for _, s in items[:10]]
                
                if proteins:
                    bars = ax.bar(proteins, scales, color='lightcoral')
                    ax.set_ylabel('Organization Scale (μm)')
                    ax.set_title('Top Organization Scales')
                    ax.tick_params(axis='x', rotation=45)
                    
                    for bar, scale in zip(bars, scales):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{scale}μm', ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No organization data', ha='center', va='center')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No organization data', ha='center', va='center')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Pipeline data not found', ha='center', va='center')
            ax.axis('off')