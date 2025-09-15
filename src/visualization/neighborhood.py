"""Neighborhood analysis visualization for publication-quality figures.

This module creates comprehensive neighborhood visualizations across
experimental variables (time, region, replicates) for scientific publication.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from collections import defaultdict, Counter
from scipy import stats

from src.config import Config


class NeighborhoodVisualizer:
    """Creates publication-quality neighborhood analysis figures."""
    
    def __init__(self, config: Config):
        self.config = config
        self.scale_order = ['cellular', 'microenvironment', 'functional_unit', 'tissue_region']
    
    def create_multiscale_dynamics_figure(self, results: List[Dict]) -> plt.Figure:
        """Create 3x3 figure showing multi-scale neighborhood dynamics.
        
        Rows: Different scales (cellular, microenvironment, functional)
        Cols: Temporal progression, phenotypes, fragmentation
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Group results by timepoint
        by_timepoint = self._group_by_timepoint(results)
        
        # Process each scale
        for row_idx, scale_name in enumerate(self.scale_order[:3]):
            # Col 0: Temporal progression of neighborhood count
            ax = axes[row_idx, 0]
            self._plot_scale_temporal_progression(ax, by_timepoint, scale_name)
            
            # Col 1: Dominant phenotypes heatmap
            ax = axes[row_idx, 1]
            self._plot_scale_phenotypes(ax, by_timepoint, scale_name)
            
            # Col 2: Fragmentation by region
            ax = axes[row_idx, 2]
            self._plot_scale_regional_fragmentation(ax, results, scale_name)
        
        fig.suptitle('Multi-Scale Neighborhood Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_temporal_evolution_figure(self, results: List[Dict]) -> plt.Figure:
        """Create 2x4 figure showing temporal neighborhood evolution."""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        by_timepoint = self._group_by_timepoint(results)
        timepoints = ['Sham', 'Day1', 'Day3', 'Day7']
        
        # Top row: Quantitative metrics
        for col_idx, metric in enumerate(['count', 'size', 'diversity', 'transitions']):
            ax = axes[0, col_idx]
            if metric == 'count':
                self._plot_neighborhood_counts(ax, by_timepoint)
            elif metric == 'size':
                self._plot_mean_neighborhood_size(ax, by_timepoint)
            elif metric == 'diversity':
                self._plot_phenotypic_diversity(ax, by_timepoint)
            else:
                self._plot_phenotypic_transitions(ax, by_timepoint)
        
        # Bottom row: Representative spatial patterns
        for col_idx, tp in enumerate(timepoints):
            ax = axes[1, col_idx]
            self._plot_representative_neighborhoods(ax, by_timepoint.get(tp, []), tp)
        
        fig.suptitle('Temporal Neighborhood Evolution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_regional_comparison_figure(self, results: List[Dict]) -> plt.Figure:
        """Create 2x3 figure comparing Cortex vs Medulla neighborhoods."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Separate by region
        cortex_results = [r for r in results if self._get_region(r) == 'Cortex']
        medulla_results = [r for r in results if self._get_region(r) == 'Medulla']
        
        regions = [('Cortex', cortex_results), ('Medulla', medulla_results)]
        
        for row_idx, (region_name, region_data) in enumerate(regions):
            # Col 0: Scale-dependent organization
            ax = axes[row_idx, 0]
            self._plot_regional_scale_organization(ax, region_data, region_name)
            
            # Col 1: Protein pair enrichment
            ax = axes[row_idx, 1]
            self._plot_regional_protein_enrichment(ax, region_data, region_name)
            
            # Col 2: Replicate concordance
            ax = axes[row_idx, 2]
            self._plot_regional_replicate_concordance(ax, region_data, region_name)
        
        fig.suptitle('Regional Neighborhood Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    # Helper methods
    def _group_by_timepoint(self, results: List[Dict]) -> Dict:
        """Group results by timepoint."""
        by_timepoint = defaultdict(list)
        
        for roi in results:
            meta = roi.get('metadata', {})
            if isinstance(meta, dict):
                day = meta.get('injury_day') or meta.get('timepoint')
            else:
                day = getattr(meta, 'injury_day', None) or getattr(meta, 'timepoint', None)
            
            if day is not None:
                if day == 0:
                    key = 'Sham'
                else:
                    key = f'Day{day}'
                by_timepoint[key].append(roi)
        
        return dict(by_timepoint)
    
    def _get_region(self, roi: Dict) -> str:
        """Extract region from ROI metadata."""
        meta = roi.get('metadata', {})
        if isinstance(meta, dict):
            return meta.get('tissue_region') or meta.get('region', 'Unknown')
        else:
            return getattr(meta, 'tissue_region', None) or getattr(meta, 'region', 'Unknown')
    
    def _get_scale_data(self, roi: Dict, scale_name: str) -> Dict:
        """Extract specific scale data from multiscale neighborhoods."""
        multiscale = roi.get('multiscale_neighborhoods', {})
        return multiscale.get(scale_name, {})
    
    def _plot_scale_temporal_progression(self, ax, by_timepoint: Dict, scale_name: str):
        """Plot neighborhood count progression for a specific scale."""
        timepoints = ['Sham', 'Day1', 'Day3', 'Day7']
        means = []
        stds = []
        
        for tp in timepoints:
            rois = by_timepoint.get(tp, [])
            counts = []
            for roi in rois:
                scale_data = self._get_scale_data(roi, scale_name)
                if scale_data:
                    counts.append(scale_data.get('n_neighborhoods', 0))
            
            if counts:
                means.append(np.mean(counts))
                stds.append(np.std(counts))
            else:
                means.append(0)
                stds.append(0)
        
        x = np.arange(len(timepoints))
        ax.errorbar(x, means, yerr=stds, marker='o', capsize=5, capthick=2)
        ax.set_xticks(x)
        ax.set_xticklabels(timepoints)
        ax.set_ylabel('Number of Neighborhoods')
        ax.set_title(f'{scale_name.title()} Scale ({self._get_scale_radius(scale_name)}μm)')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_scale_phenotypes(self, ax, by_timepoint: Dict, scale_name: str):
        """Plot dominant phenotypes heatmap for a specific scale."""
        # Collect all phenotypes across timepoints
        phenotype_counts = defaultdict(lambda: defaultdict(int))
        timepoints = ['Sham', 'Day1', 'Day3', 'Day7']
        
        for tp in timepoints:
            rois = by_timepoint.get(tp, [])
            for roi in rois:
                scale_data = self._get_scale_data(roi, scale_name)
                if scale_data and 'neighborhoods' in scale_data:
                    for nbhd_data in scale_data['neighborhoods'].values():
                        dominant = nbhd_data.get('dominant_pairs', [])
                        if dominant:
                            phenotype_counts[dominant[0]][tp] += 1
        
        if not phenotype_counts:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.axis('off')
            return
        
        # Create matrix
        phenotypes = list(phenotype_counts.keys())[:10]  # Top 10
        matrix = np.zeros((len(phenotypes), len(timepoints)))
        
        for i, pheno in enumerate(phenotypes):
            for j, tp in enumerate(timepoints):
                matrix[i, j] = phenotype_counts[pheno][tp]
        
        # Normalize by column (timepoint)
        col_sums = matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1
        matrix = matrix / col_sums
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=matrix.max())
        ax.set_xticks(range(len(timepoints)))
        ax.set_xticklabels(timepoints, rotation=45)
        ax.set_yticks(range(len(phenotypes)))
        ax.set_yticklabels(phenotypes, fontsize=8)
        ax.set_title(f'{scale_name.title()} Phenotypes')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_scale_regional_fragmentation(self, ax, results: List[Dict], scale_name: str):
        """Plot fragmentation by region for a specific scale."""
        regions = ['Cortex', 'Medulla']
        fragmentation_data = {r: [] for r in regions}
        
        for roi in results:
            region = self._get_region(roi)
            if region in regions:
                scale_data = self._get_scale_data(roi, scale_name)
                if scale_data:
                    n_nbhd = scale_data.get('n_neighborhoods', 0)
                    coverage = scale_data.get('coverage', 0)
                    if coverage > 0:
                        # Fragmentation index: neighborhoods per unit coverage
                        frag = n_nbhd / coverage
                        fragmentation_data[region].append(frag)
        
        # Create violin plot
        data_for_plot = []
        labels = []
        for region in regions:
            if fragmentation_data[region]:
                data_for_plot.append(fragmentation_data[region])
                labels.append(region)
        
        if data_for_plot:
            parts = ax.violinplot(data_for_plot, showmeans=True, showmedians=True)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_ylabel('Fragmentation Index')
            ax.set_title(f'{scale_name.title()} Regional Fragmentation')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.axis('off')
    
    def _plot_neighborhood_counts(self, ax, by_timepoint: Dict):
        """Plot neighborhood counts across timepoints (all scales)."""
        timepoints = ['Sham', 'Day1', 'Day3', 'Day7']
        scale_data = {scale: {'means': [], 'stds': []} for scale in self.scale_order[:3]}
        
        for tp in timepoints:
            rois = by_timepoint.get(tp, [])
            for scale in self.scale_order[:3]:
                counts = []
                for roi in rois:
                    s_data = self._get_scale_data(roi, scale)
                    if s_data:
                        counts.append(s_data.get('n_neighborhoods', 0))
                
                if counts:
                    scale_data[scale]['means'].append(np.mean(counts))
                    scale_data[scale]['stds'].append(np.std(counts))
                else:
                    scale_data[scale]['means'].append(0)
                    scale_data[scale]['stds'].append(0)
        
        x = np.arange(len(timepoints))
        width = 0.25
        
        for i, scale in enumerate(self.scale_order[:3]):
            offset = (i - 1) * width
            ax.bar(x + offset, scale_data[scale]['means'], width, 
                  yerr=scale_data[scale]['stds'], label=scale, alpha=0.7, capsize=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(timepoints)
        ax.set_ylabel('Neighborhood Count')
        ax.set_title('Neighborhood Count by Scale')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_mean_neighborhood_size(self, ax, by_timepoint: Dict):
        """Plot mean neighborhood size across timepoints."""
        timepoints = ['Sham', 'Day1', 'Day3', 'Day7']
        
        # Use microenvironment scale as default
        means = []
        stds = []
        
        for tp in timepoints:
            rois = by_timepoint.get(tp, [])
            sizes = []
            for roi in rois:
                scale_data = self._get_scale_data(roi, 'microenvironment')
                if scale_data and 'neighborhoods' in scale_data:
                    for nbhd in scale_data['neighborhoods'].values():
                        sizes.append(nbhd.get('size', 0))
            
            if sizes:
                means.append(np.mean(sizes))
                stds.append(np.std(sizes))
            else:
                means.append(0)
                stds.append(0)
        
        x = np.arange(len(timepoints))
        ax.errorbar(x, means, yerr=stds, marker='s', capsize=5, capthick=2, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(timepoints)
        ax.set_ylabel('Mean Size (pixels)')
        ax.set_title('Mean Neighborhood Size')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_phenotypic_diversity(self, ax, by_timepoint: Dict):
        """Plot Shannon entropy of neighborhood phenotypes."""
        timepoints = ['Sham', 'Day1', 'Day3', 'Day7']
        diversities = []
        
        for tp in timepoints:
            rois = by_timepoint.get(tp, [])
            phenotype_counts = Counter()
            
            for roi in rois:
                scale_data = self._get_scale_data(roi, 'microenvironment')
                if scale_data and 'neighborhoods' in scale_data:
                    for nbhd in scale_data['neighborhoods'].values():
                        dominant = nbhd.get('dominant_pairs', [])
                        if dominant:
                            phenotype_counts[dominant[0]] += 1
            
            # Calculate Shannon entropy
            if phenotype_counts:
                total = sum(phenotype_counts.values())
                probs = [count/total for count in phenotype_counts.values()]
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                diversities.append(entropy)
            else:
                diversities.append(0)
        
        x = np.arange(len(timepoints))
        ax.bar(x, diversities, color='coral', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(timepoints)
        ax.set_ylabel('Shannon Entropy')
        ax.set_title('Phenotypic Diversity')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_phenotypic_transitions(self, ax, by_timepoint: Dict):
        """Plot Sankey-style transitions between phenotypes."""
        # Simplified version - show top phenotypes at each timepoint
        timepoints = ['Sham', 'Day1', 'Day3', 'Day7']
        top_phenotypes = []
        
        for tp in timepoints:
            rois = by_timepoint.get(tp, [])
            phenotype_counts = Counter()
            
            for roi in rois:
                scale_data = self._get_scale_data(roi, 'microenvironment')
                if scale_data and 'neighborhoods' in scale_data:
                    for nbhd in scale_data['neighborhoods'].values():
                        dominant = nbhd.get('dominant_pairs', [])
                        if dominant:
                            phenotype_counts[dominant[0]] += 1
            
            # Get top 3 phenotypes
            top_3 = [p for p, _ in phenotype_counts.most_common(3)]
            top_phenotypes.append(top_3)
        
        # Plot connections
        for i in range(len(timepoints) - 1):
            x1 = i
            x2 = i + 1
            
            for j, p1 in enumerate(top_phenotypes[i]):
                if p1 in top_phenotypes[i + 1]:
                    j2 = top_phenotypes[i + 1].index(p1)
                    ax.plot([x1, x2], [j, j2], 'b-', alpha=0.3, linewidth=2)
        
        # Add phenotype labels
        for i, (tp, phenotypes) in enumerate(zip(timepoints, top_phenotypes)):
            for j, pheno in enumerate(phenotypes):
                ax.text(i, j, pheno[:10], ha='center', va='center', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_xticks(range(len(timepoints)))
        ax.set_xticklabels(timepoints)
        ax.set_ylim(-0.5, 2.5)
        ax.set_title('Phenotypic Transitions')
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_representative_neighborhoods(self, ax, rois: List[Dict], timepoint: str):
        """Plot representative spatial pattern for a timepoint."""
        if not rois:
            ax.text(0.5, 0.5, f'No data for {timepoint}', ha='center', va='center')
            ax.axis('off')
            return
        
        # Use first ROI as representative
        roi = rois[0]
        coords = np.array(roi.get('coords', []))
        
        if len(coords) == 0:
            ax.text(0.5, 0.5, 'No spatial data', ha='center', va='center')
            ax.axis('off')
            return
        
        # Get microenvironment scale neighborhoods
        scale_data = self._get_scale_data(roi, 'microenvironment')
        
        if scale_data and 'pixel_assignments' in scale_data:
            assignments = scale_data['pixel_assignments']
            n_neighborhoods = scale_data.get('n_neighborhoods', 0)
            
            # Color by neighborhood
            colors = plt.cm.Set3(np.linspace(0, 1, max(12, n_neighborhoods)))
            
            for nbhd_id in range(n_neighborhoods):
                mask = assignments == nbhd_id
                if np.any(mask):
                    nbhd_coords = coords[mask]
                    ax.scatter(nbhd_coords[:, 0], nbhd_coords[:, 1],
                             c=[colors[nbhd_id]], s=0.5, alpha=0.6)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], c='gray', s=0.5, alpha=0.3)
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(timepoint)
        ax.set_aspect('equal')
    
    def _plot_regional_scale_organization(self, ax, region_data: List[Dict], region_name: str):
        """Plot scale-dependent organization for a region."""
        scale_metrics = defaultdict(list)
        
        for roi in region_data:
            for scale in self.scale_order[:3]:
                scale_data = self._get_scale_data(roi, scale)
                if scale_data:
                    n_nbhd = scale_data.get('n_neighborhoods', 0)
                    scale_metrics[scale].append(n_nbhd)
        
        # Create boxplot
        data = []
        labels = []
        for scale in self.scale_order[:3]:
            if scale_metrics[scale]:
                data.append(scale_metrics[scale])
                labels.append(f'{scale}\n({self._get_scale_radius(scale)}μm)')
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax.set_ylabel('Number of Neighborhoods')
            ax.set_title(f'{region_name} - Scale Organization')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.axis('off')
    
    def _plot_regional_protein_enrichment(self, ax, region_data: List[Dict], region_name: str):
        """Plot protein pair enrichment for a region."""
        phenotype_counts = Counter()
        
        for roi in region_data:
            scale_data = self._get_scale_data(roi, 'microenvironment')
            if scale_data and 'neighborhoods' in scale_data:
                for nbhd in scale_data['neighborhoods'].values():
                    dominant = nbhd.get('dominant_pairs', [])
                    if dominant:
                        phenotype_counts[dominant[0]] += nbhd.get('coverage', 0)
        
        if phenotype_counts:
            # Get top 10
            top_phenotypes = phenotype_counts.most_common(10)
            phenotypes, coverages = zip(*top_phenotypes)
            
            y_pos = np.arange(len(phenotypes))
            ax.barh(y_pos, coverages, color='steelblue', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(phenotypes, fontsize=8)
            ax.set_xlabel('Total Coverage')
            ax.set_title(f'{region_name} - Protein Enrichment')
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.axis('off')
    
    def _plot_regional_replicate_concordance(self, ax, region_data: List[Dict], region_name: str):
        """Plot replicate concordance for a region."""
        # Separate by replicate
        ms1_data = []
        ms2_data = []
        
        for roi in region_data:
            meta = roi.get('metadata', {})
            if isinstance(meta, dict):
                replicate = meta.get('mouse_replicate') or meta.get('mouse_id', '')
            else:
                replicate = getattr(meta, 'mouse_replicate', '') or getattr(meta, 'mouse_id', '')
            
            scale_data = self._get_scale_data(roi, 'microenvironment')
            if scale_data:
                n_nbhd = scale_data.get('n_neighborhoods', 0)
                if 'MS1' in str(replicate):
                    ms1_data.append(n_nbhd)
                elif 'MS2' in str(replicate):
                    ms2_data.append(n_nbhd)
        
        if ms1_data and ms2_data:
            # Scatter plot
            min_len = min(len(ms1_data), len(ms2_data))
            ax.scatter(ms1_data[:min_len], ms2_data[:min_len], alpha=0.6)
            
            # Add correlation
            if min_len > 1:
                r, p = stats.pearsonr(ms1_data[:min_len], ms2_data[:min_len])
                ax.text(0.05, 0.95, f'r = {r:.2f}\np = {p:.3f}',
                       transform=ax.transAxes, va='top')
            
            # Add identity line
            max_val = max(max(ms1_data), max(ms2_data))
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
            
            ax.set_xlabel('MS1 Neighborhoods')
            ax.set_ylabel('MS2 Neighborhoods')
            ax.set_title(f'{region_name} - Replicate Concordance')
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient replicate data', ha='center', va='center')
            ax.axis('off')
    
    def _get_scale_radius(self, scale_name: str) -> int:
        """Get radius for a scale from config."""
        scales = self.config.raw.get('neighborhood_analysis', {}).get('scales', {})
        if scale_name in scales:
            return scales[scale_name].get('radius', 0)
        return 0