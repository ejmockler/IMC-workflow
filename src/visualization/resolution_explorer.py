"""
Resolution Explorer Visualization

Provides diagnostic plots for scientists to make informed decisions
about clustering resolution at each scale. Implements the Factory pattern
for different visualization types.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional
import seaborn as sns


class ResolutionExplorer:
    """
    Exploratory visualization for resolution parameter selection.
    Empowers scientists with evidence rather than automated "optimal" values.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 20)
        
    def plot_stability_analysis(
        self,
        stability_results: Dict,
        scale_um: float,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create diagnostic plots for resolution selection.
        
        Shows stability and cluster count across resolutions,
        letting scientists choose based on domain knowledge.
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        resolutions = stability_results['resolutions']
        stability_scores = stability_results['stability_scores']
        mean_n_clusters = stability_results['mean_n_clusters']
        
        # Plot 1: Stability vs Resolution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(resolutions, stability_scores, 'b-', linewidth=2, label='Stability (ARI)')
        ax1.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='Min acceptable')
        ax1.set_xlabel('Resolution Parameter', fontsize=10)
        ax1.set_ylabel('Stability Score', fontsize=10)
        ax1.set_title(f'A. Clustering Stability at {scale_um}μm', fontweight='bold', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        # Highlight stable plateaus
        if 'stable_resolutions' in stability_results:
            for res in stability_results['stable_resolutions']:
                idx = np.argmin(np.abs(np.array(resolutions) - res))
                ax1.plot(res, stability_scores[idx], 'go', markersize=8, label='Stable plateau')
        
        # Plot 2: Number of Clusters vs Resolution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(resolutions, mean_n_clusters, 'g-', linewidth=2)
        ax2.set_xlabel('Resolution Parameter', fontsize=10)
        ax2.set_ylabel('Number of Clusters', fontsize=10)
        ax2.set_title(f'B. Cluster Count at {scale_um}μm', fontweight='bold', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Add scale-appropriate annotations
        if scale_um <= 10:
            ax2.axhspan(10, 20, alpha=0.1, color='green', label='Expected for cells')
        elif scale_um <= 20:
            ax2.axhspan(5, 12, alpha=0.1, color='blue', label='Expected for neighborhoods')
        else:
            ax2.axhspan(2, 6, alpha=0.1, color='orange', label='Expected for regions')
        
        # Plot 3: Combined view with dual axis
        ax3 = fig.add_subplot(gs[0, 2])
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(resolutions, stability_scores, 'b-', linewidth=2, label='Stability')
        line2 = ax3_twin.plot(resolutions, mean_n_clusters, 'g-', linewidth=2, label='N clusters')
        
        ax3.set_xlabel('Resolution Parameter', fontsize=10)
        ax3.set_ylabel('Stability Score', color='b', fontsize=10)
        ax3_twin.set_ylabel('Number of Clusters', color='g', fontsize=10)
        ax3.set_title(f'C. Combined View at {scale_um}μm', fontweight='bold', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper right', fontsize=9)
        
        # Plot 4: Derivative of stability (to find plateaus)
        ax4 = fig.add_subplot(gs[1, 0])
        if len(stability_scores) > 1:
            stability_diff = np.diff(stability_scores)
            res_mid = (np.array(resolutions[:-1]) + np.array(resolutions[1:])) / 2
            ax4.plot(res_mid, stability_diff, 'r-', linewidth=1.5)
            ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax4.set_xlabel('Resolution Parameter', fontsize=10)
            ax4.set_ylabel('Δ Stability', fontsize=10)
            ax4.set_title('D. Stability Change Rate', fontweight='bold', fontsize=11)
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Resolution recommendation zones
        ax5 = fig.add_subplot(gs[1, 1:])
        self._plot_recommendation_zones(ax5, resolutions, stability_scores, 
                                       mean_n_clusters, scale_um)
        
        plt.suptitle(f'Resolution Selection for {scale_um}μm Scale\n'
                    f'Choose based on stability plateaus and expected biology',
                    fontsize=13, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
    
    def _plot_recommendation_zones(
        self,
        ax: plt.Axes,
        resolutions: List[float],
        stability_scores: List[float],
        mean_n_clusters: List[float],
        scale_um: float
    ):
        """
        Plot recommendation zones based on stability and biological expectations.
        """
        # Define zones based on scale
        if scale_um <= 10:
            expected_k_range = (10, 20)
            zone_label = "Fine scale: Expect 10-20 cellular phenotypes"
        elif scale_um <= 20:
            expected_k_range = (5, 12)
            zone_label = "Medium scale: Expect 5-12 neighborhoods"
        else:
            expected_k_range = (2, 6)
            zone_label = "Coarse scale: Expect 2-6 tissue regions"
        
        # Find resolutions that satisfy both stability and k expectations
        good_zones = []
        for i, res in enumerate(resolutions):
            if (stability_scores[i] >= 0.6 and 
                expected_k_range[0] <= mean_n_clusters[i] <= expected_k_range[1]):
                good_zones.append((res, stability_scores[i], mean_n_clusters[i]))
        
        # Plot zones
        ax.set_xlabel('Resolution Parameter', fontsize=10)
        ax.set_ylabel('Quality Score', fontsize=10)
        ax.set_title('E. Recommended Resolution Zones', fontweight='bold', fontsize=11)
        
        # Background zones
        ax.axhspan(0, 0.6, alpha=0.1, color='red', label='Low stability')
        ax.axhspan(0.6, 1.0, alpha=0.1, color='green', label='Good stability')
        
        # Plot combined quality score
        quality_scores = []
        for i in range(len(resolutions)):
            # Penalize if outside expected k range
            k_penalty = 0
            if mean_n_clusters[i] < expected_k_range[0]:
                k_penalty = (expected_k_range[0] - mean_n_clusters[i]) / expected_k_range[0]
            elif mean_n_clusters[i] > expected_k_range[1]:
                k_penalty = (mean_n_clusters[i] - expected_k_range[1]) / expected_k_range[1]
            
            quality = stability_scores[i] * (1 - k_penalty * 0.5)
            quality_scores.append(max(0, quality))
        
        ax.plot(resolutions, quality_scores, 'b-', linewidth=2, label='Combined quality')
        
        # Mark good zones
        for res, stab, k in good_zones:
            idx = resolutions.index(res)
            ax.plot(res, quality_scores[idx], 'go', markersize=10)
        
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        # Add text annotation
        ax.text(0.02, 0.98, zone_label, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def plot_spatial_coherence(
        self,
        cluster_maps: Dict[float, np.ndarray],
        spatial_coherence_scores: Dict[float, float],
        scale_um: float,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize spatial coherence of clusters at different resolutions.
        """
        n_resolutions = len(cluster_maps)
        n_cols = min(4, n_resolutions)
        n_rows = (n_resolutions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (resolution, cluster_map) in enumerate(sorted(cluster_maps.items())):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            im = ax.imshow(cluster_map, cmap='tab20', interpolation='nearest')
            
            coherence = spatial_coherence_scores.get(resolution, 0)
            n_clusters = len(np.unique(cluster_map[cluster_map >= 0]))
            
            ax.set_title(f'Res={resolution:.2f}\n'
                        f'k={n_clusters}, Moran\'s I={coherence:.2f}',
                        fontsize=10)
            ax.axis('off')
            
            # Add colorbar for each
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide empty subplots
        for idx in range(n_resolutions, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Spatial Coherence at {scale_um}μm Scale\n'
                    f'Higher Moran\'s I indicates more coherent spatial clusters',
                    fontsize=13, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
    
    def plot_hierarchical_relationship(
        self,
        hierarchy: Dict,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize parent-child relationships across scales.
        """
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        scale_keys = sorted([k for k in hierarchy.keys() if k.startswith('scale_')])
        
        for idx, scale_key in enumerate(scale_keys):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            
            scale_data = hierarchy[scale_key]
            labels = scale_data['labels']
            coords = scale_data['coords']
            scale_um = scale_data['scale_um']
            
            # Plot clusters
            unique_labels = np.unique(labels[labels >= 0])
            for label in unique_labels:
                mask = labels == label
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=[self.color_palette[label % len(self.color_palette)]],
                          s=20, alpha=0.6, label=f'Cluster {label}')
            
            ax.set_title(f'{scale_um}μm Scale\n{len(unique_labels)} clusters',
                        fontweight='bold', fontsize=11)
            ax.set_xlabel('X (μm)', fontsize=9)
            ax.set_ylabel('Y (μm)', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Don't show legend if too many clusters
            if len(unique_labels) <= 8:
                ax.legend(fontsize=8, loc='upper right')
        
        plt.suptitle('Hierarchical Multi-Scale Clustering\n'
                    'Shows parent-child relationships across scales',
                    fontsize=13, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
    
    def create_summary_report(
        self,
        all_results: Dict,
        output_path: str
    ):
        """
        Create a summary PDF report for resolution selection.
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(output_path) as pdf:
            # Page 1: Overview
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle('Multi-Scale Clustering Analysis Report', fontsize=16, fontweight='bold')
            
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            summary_text = """
            This report provides diagnostic information for selecting clustering resolutions
            at multiple spatial scales. Rather than automatically selecting "optimal" parameters,
            these visualizations empower you to make informed decisions based on:
            
            1. Stability Analysis: How consistent are clusters under resampling?
            2. Spatial Coherence: Do clusters form spatially meaningful groups?  
            3. Biological Expectations: Does cluster count match expected biology?
            
            For each scale, review the stability curves and choose resolutions where:
            - Stability plateaus (flat regions in the curve)
            - Cluster count matches biological expectations
            - Spatial coherence (Moran's I) is high
            
            Remember: There is no single "correct" answer. Choose based on your
            scientific question and domain knowledge.
            """
            
            ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='center')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Additional pages with detailed results
            for scale_um, results in all_results.items():
                if 'stability_plot' in results:
                    pdf.savefig(results['stability_plot'], bbox_inches='tight')
                if 'coherence_plot' in results:
                    pdf.savefig(results['coherence_plot'], bbox_inches='tight')
        
        print(f"Resolution exploration report saved to {output_path}")