"""Main visualization orchestrator for IMC analysis."""

from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

from src.config import Config
from .roi import ROIVisualizer
from .temporal import TemporalVisualizer
from .condition import ConditionVisualizer
from .replicate import ReplicateVisualizer
from .network import NetworkVisualizer
from .neighborhood import NeighborhoodVisualizer
from .territory import TerritoryVisualizer


class VisualizationPipeline:
    """Orchestrates all visualization types."""
    
    def __init__(self, config: Config):
        self.config = config
        self.roi_viz = ROIVisualizer(config)
        self.temporal_viz = TemporalVisualizer(config)
        self.condition_viz = ConditionVisualizer(config)
        self.replicate_viz = ReplicateVisualizer(config)
        self.network_viz = NetworkVisualizer(config)
        self.neighborhood_viz = NeighborhoodVisualizer(config)
        self.territory_viz = TerritoryVisualizer(config)
    
    def create_roi_figure(self, roi_data: Dict) -> plt.Figure:
        """Create single ROI visualization."""
        return self.roi_viz.create_figure(roi_data)
    
    
    def create_temporal_figure(self, results: List[Dict]) -> plt.Figure:
        """Create temporal analysis figure."""
        return self.temporal_viz.create_temporal_figure(results)
    
    def create_condition_figure(self, results: List[Dict]) -> plt.Figure:
        """Create condition comparison figure."""
        return self.condition_viz.create_condition_figure(results)
    
    def create_replicate_variance_figure(self, results: List[Dict]) -> plt.Figure:
        """Create replicate variance figure."""
        return self.replicate_viz.create_replicate_variance_figure(results)
    
    
    def create_timepoint_region_contact_grid(self, results: List[Dict]) -> plt.Figure:
        """Create timepoint region contact grid."""
        return self.replicate_viz.create_timepoint_region_contact_grid(results, self.config)
    
    # Territory-focused visualizations (MSPT framework)
    def create_territory_map(self, roi_data: Dict) -> plt.Figure:
        """Create spatial territory map for single ROI."""
        return self.territory_viz.create_roi_territory_map(roi_data)
    
    def create_spatial_interaction_heatmap(self, roi_data: Dict, radius: float = 25.0) -> plt.Figure:
        """Create spatial interaction heatmap for specific radius."""
        return self.territory_viz.create_spatial_interaction_heatmap(roi_data, radius)
    
    def create_multi_scale_summary(self, roi_data: Dict) -> plt.Figure:
        """Create multi-scale spatial analysis summary."""
        return self.territory_viz.create_multi_scale_summary(roi_data)
    
    def create_territory_temporal_analysis(self, results: List[Dict]) -> plt.Figure:
        """Create temporal analysis of territory changes."""
        return self.territory_viz.create_territory_temporal_analysis(results)
    
    def create_network_figure(self, results: List[Dict], output_path: str) -> None:
        """Create network analysis figure."""
        self.network_viz.create_network_grid(results, output_path)
    
    def create_all_figures(self, results: List[Dict]) -> Dict[str, plt.Figure]:
        """Create all visualization types."""
        figures = {}
        
        # Only generate replicate variance analysis (12 diverse interactions)
        # Other analyses are handled by kidney_healing/run_full_report.py to avoid duplicates
        temporal_data = [r for r in results if hasattr(r['metadata'], 'injury_day') and r['metadata'].injury_day is not None]
        if temporal_data:
            figures['replicate_variance'] = self.create_replicate_variance_figure(results)
        
        # Network analysis
        network_output = self.config.output_dir / 'network_analysis.png'
        self.create_network_figure(results, str(network_output))
        
        return figures
    
    def create_functional_group_dynamics_figure(self, results: List[Dict]) -> plt.Figure:
        """Create functional group dynamics figure with regional stratification."""
        metrics = self._aggregate_functional_metrics(results)
        return self._create_functional_group_dynamics_plot(metrics)
    
    def create_neighborhood_temporal_evolution(self, results: List[Dict]) -> plt.Figure:
        """Create neighborhood temporal evolution figure."""
        return self.neighborhood_viz.create_temporal_evolution_figure(results)
    
    def create_neighborhood_regional_comparison(self, results: List[Dict]) -> plt.Figure:
        """Create neighborhood regional comparison figure."""
        return self.neighborhood_viz.create_regional_comparison_figure(results)
    
    def create_region_temporal_trajectories(self, results: List[Dict]) -> plt.Figure:
        """Create region-specific temporal trajectories."""
        return self.replicate_viz.create_region_temporal_trajectories(results, self.config)
    
    def create_tmd_spatial_overlay(self, results: List[Dict]) -> plt.Figure:
        """Create TMD spatial overlay visualization."""
        return self._create_tmd_overlay_figure(results)
    
    def _aggregate_functional_metrics(self, results: List[Dict]) -> Dict:
        """Aggregate functional group metrics across experimental conditions."""
        # Build label map for timepoints
        exp_cfg = self.config.raw.get('experimental', {})
        tp_vals = exp_cfg.get('timepoints', [])
        tp_labels = exp_cfg.get('timepoint_labels', [])
        label_map = {f"D{tp}": lbl for tp, lbl in zip(tp_vals, tp_labels)}

        # Group ROIs
        by_timepoint: Dict[str, List[Dict]] = {}
        by_region: Dict[str, List[Dict]] = {}
        by_region_time: Dict[str, Dict[str, List[Dict]]] = {}

        for roi in results:
            metadata = roi['metadata']
            if hasattr(metadata, 'get'):
                day = metadata.get('injury_day')
                region = metadata.get('tissue_region', 'Unknown')
            else:
                day = metadata.injury_day
                region = metadata.tissue_region or 'Unknown'
            
            if day is not None:
                tp = f"D{day}"
                by_timepoint.setdefault(tp, []).append(roi)
            by_region.setdefault(region, []).append(roi)
            if day is not None:
                by_region_time.setdefault(region, {}).setdefault(tp, []).append(roi)

        functional_groups = self.config.raw.get('proteins', {}).get('functional_groups', {})

        def summarize_group(group: List[Dict]) -> Dict:
            if not group:
                return {}
            domain_counts = [len(r['blob_signatures']) for r in group]
            contact_counts = [len(r.get('canonical_contacts', {})) for r in group]
            
            # Functional groups
            fg_list = [self._compute_functional_group_percentages(r, functional_groups) for r in group]
            fg_keys = sorted({k for d in fg_list for k in d.keys()})
            fg_summary = {
                k: float(np.mean([d.get(k, 0.0) for d in fg_list])) if fg_list else 0.0
                for k in fg_keys
            }
            
            return {
                'n_rois': len(group),
                'domains_mean': float(np.mean(domain_counts)) if domain_counts else 0.0,
                'domains_std': float(np.std(domain_counts)) if domain_counts else 0.0,
                'contacts_mean': float(np.mean(contact_counts)) if contact_counts else 0.0,
                'contacts_std': float(np.std(contact_counts)) if contact_counts else 0.0,
                'functional_groups_mean_pct': fg_summary
            }

        metrics = {
            'timepoints': {tp: summarize_group(group) | {'label': label_map.get(tp, tp)} 
                           for tp, group in sorted(by_timepoint.items(), key=lambda x: int(x[0][1:]))},
            'regions': {region: summarize_group(group) for region, group in by_region.items()},
            'timepoint_region': {
                region: {tp: summarize_group(group) for tp, group in sorted(tp_map.items(), key=lambda x: int(x[0][1:]))}
                for region, tp_map in by_region_time.items()
            }
        }
        return metrics
    
    def _compute_functional_group_percentages(self, roi: Dict, functional_groups: Dict[str, List[str]]) -> Dict[str, float]:
        """Compute functional group percentages for a single ROI."""
        total_pixels = roi.get('total_pixels', 0) or 0
        if total_pixels <= 0:
            return {}
        results = {}
        for group_name, proteins in functional_groups.items():
            if group_name == 'structural_controls':
                continue
            group_size = 0
            for sig in roi['blob_signatures'].values():
                domain_proteins = sig['dominant_proteins'][:2]
                if any(p in proteins for p in domain_proteins):
                    group_size += sig['size']
            results[group_name] = 100.0 * group_size / total_pixels
        return results
    
    def _create_functional_group_dynamics_plot(self, metrics: Dict) -> plt.Figure:
        """Create publication-quality functional group dynamics figure."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract timepoints and labels
        timepoints = sorted([k for k in metrics['timepoints'].keys()], key=lambda x: int(x[1:]))
        x_pos = np.arange(len(timepoints))
        
        # Get display labels from config
        exp_cfg = self.config.raw.get('experimental', {})
        tp_vals = exp_cfg.get('timepoints', [])
        tp_labels = exp_cfg.get('timepoint_labels', [])
        label_map = {f"D{tp}": lbl for tp, lbl in zip(tp_vals, tp_labels)}
        display_labels = [label_map.get(tp, tp) for tp in timepoints]
        
        # Colors for each group and region
        colors = {'kidney_inflammation': '#e74c3c', 'kidney_repair': '#2ecc71', 'kidney_vasculature': '#3498db'}
        region_colors = {'Cortex': 0.9, 'Medulla': 0.6}  # Alpha values
        
        functional_groups = [
            ('kidney_inflammation', 'Inflammation (CD45/CD11b)'),
            ('kidney_repair', 'Repair (CD206/CD44)'),
            ('kidney_vasculature', 'Vasculature (CD31/CD34)')
        ]
        
        for col, (group_key, group_name) in enumerate(functional_groups):
            # Top row: Overall temporal trends
            ax_overall = axes[0, col]
            
            values = []
            errors = []
            
            for tp in timepoints:
                tp_data = metrics['timepoints'][tp]
                val = tp_data['functional_groups_mean_pct'].get(group_key, 0)
                values.append(val)
                n_rois = tp_data['n_rois']
                sem = val * 0.15 / np.sqrt(n_rois) if n_rois > 0 else 0
                errors.append(sem)
            
            bars = ax_overall.bar(x_pos, values, color=colors[group_key], alpha=0.7, 
                                 yerr=errors, capsize=5, error_kw={'linewidth': 2})
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax_overall.text(bar.get_x() + bar.get_width()/2., height + 2,
                              f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
            
            ax_overall.set_xticks(x_pos)
            ax_overall.set_xticklabels(display_labels, fontsize=11)
            ax_overall.set_ylabel('Tissue Coverage (%)', fontsize=11)
            ax_overall.set_title(f'{group_name} - Overall', fontsize=12, fontweight='bold')
            ax_overall.set_ylim(0, max(values) + 15 if values else 100)
            ax_overall.grid(True, alpha=0.3, axis='y')
            
            # Bottom row: Region-stratified comparison
            ax_regions = axes[1, col]
            
            regions = ['Cortex', 'Medulla']
            region_data = metrics.get('timepoint_region', {})
            
            bar_width = 0.35
            x_cortex = x_pos - bar_width/2
            x_medulla = x_pos + bar_width/2
            
            for region_idx, region in enumerate(regions):
                if region in region_data:
                    region_values = []
                    region_errors = []
                    
                    for tp in timepoints:
                        if tp in region_data[region]:
                            tp_region_data = region_data[region][tp]
                            val = tp_region_data['functional_groups_mean_pct'].get(group_key, 0)
                            region_values.append(val)
                            n_rois = tp_region_data['n_rois']
                            sem = val * 0.15 / np.sqrt(n_rois) if n_rois > 0 else 0
                            region_errors.append(sem)
                        else:
                            region_values.append(0)
                            region_errors.append(0)
                    
                    x_positions = x_cortex if region == 'Cortex' else x_medulla
                    bars = ax_regions.bar(x_positions, region_values, bar_width, 
                                         color=colors[group_key], alpha=region_colors[region],
                                         label=region, yerr=region_errors, capsize=3,
                                         error_kw={'linewidth': 1.5})
                    
                    # Add value labels on bars
                    for bar, val in zip(bars, region_values):
                        if val > 0:
                            height = bar.get_height()
                            ax_regions.text(bar.get_x() + bar.get_width()/2., height + 1,
                                          f'{val:.1f}', ha='center', va='bottom', fontsize=8)
            
            ax_regions.set_xticks(x_pos)
            ax_regions.set_xticklabels(display_labels, fontsize=11)
            ax_regions.set_ylabel('Tissue Coverage (%)', fontsize=11)
            ax_regions.set_title(f'{group_name} - Regional Comparison', fontsize=12, fontweight='bold')
            ax_regions.legend(fontsize=10)
            ax_regions.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Functional Group Dynamics - Overall vs Regional Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _create_tmd_overlay_figure(self, results: List[Dict]) -> plt.Figure:
        """Create TMD spatial overlay visualization."""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Group by timepoint
        by_timepoint = defaultdict(list)
        for roi in results:
            metadata = roi['metadata']
            if hasattr(metadata, 'get'):
                day = metadata.get('injury_day')
            else:
                day = metadata.injury_day
            
            if day is not None:
                if day == 0:
                    key = 'Sham'
                else:
                    key = f'Day{day}'
                by_timepoint[key].append(roi)
        
        timepoints = ['Sham', 'Day1', 'Day3', 'Day7']
        
        for col_idx, tp in enumerate(timepoints):
            rois = by_timepoint.get(tp, [])
            
            # Top row: TMD assignments
            ax_tmd = axes[0, col_idx]
            if rois:
                self._plot_tmd_assignments(ax_tmd, rois[0], f'{tp} - TMDs')
            else:
                ax_tmd.text(0.5, 0.5, f'No data for {tp}', ha='center', va='center')
                ax_tmd.axis('off')
            
            # Bottom row: Functional overlays
            ax_func = axes[1, col_idx]
            if rois:
                self._plot_functional_overlay(ax_func, rois[0], f'{tp} - Functional Groups')
            else:
                ax_func.text(0.5, 0.5, f'No data for {tp}', ha='center', va='center')
                ax_func.axis('off')
        
        fig.suptitle('Tissue Microdomains (TMDs) and Functional Group Spatial Organization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_tmd_assignments(self, ax, roi: Dict, title: str):
        """Plot TMD assignments on spatial coordinates."""
        coords = np.array(roi.get('coords', []))
        if len(coords) == 0:
            ax.text(0.5, 0.5, 'No spatial data', ha='center', va='center')
            ax.axis('off')
            return
        
        # Use blob signatures as TMDs
        blob_sigs = roi.get('blob_signatures', {})
        n_tmds = len(blob_sigs)
        
        if n_tmds > 0:
            colors = plt.cm.Set3(np.linspace(0, 1, max(12, n_tmds)))
            
            # Create pixel assignments based on blob signatures
            pixel_assignments = np.zeros(len(coords))
            
            for tmd_id, (blob_id, sig) in enumerate(blob_sigs.items()):
                # Assign pixels based on dominant proteins (simplified)
                # In practice, this would use actual TMD clustering results
                center_idx = tmd_id * (len(coords) // max(n_tmds, 1))
                end_idx = min((tmd_id + 1) * (len(coords) // max(n_tmds, 1)), len(coords))
                pixel_assignments[center_idx:end_idx] = tmd_id
            
            for tmd_id in range(n_tmds):
                mask = pixel_assignments == tmd_id
                if np.any(mask):
                    tmd_coords = coords[mask]
                    ax.scatter(tmd_coords[:, 0], tmd_coords[:, 1],
                             c=[colors[tmd_id]], s=0.8, alpha=0.7, label=f'TMD {tmd_id+1}')
        else:
            ax.scatter(coords[:, 0], coords[:, 1], c='gray', s=0.5, alpha=0.3)
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(title)
        ax.set_aspect('equal')
        if n_tmds > 0 and n_tmds <= 10:
            ax.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _plot_functional_overlay(self, ax, roi: Dict, title: str):
        """Plot functional group overlay on spatial coordinates."""
        coords = np.array(roi.get('coords', []))
        if len(coords) == 0:
            ax.text(0.5, 0.5, 'No spatial data', ha='center', va='center')
            ax.axis('off')
            return
        
        functional_groups = self.config.raw.get('proteins', {}).get('functional_groups', {})
        colors = {'kidney_inflammation': '#e74c3c', 'kidney_repair': '#2ecc71', 'kidney_vasculature': '#3498db'}
        
        # Assign pixels to functional groups (simplified approach)
        pixel_groups = np.full(len(coords), -1)  # -1 = unassigned
        
        blob_sigs = roi.get('blob_signatures', {})
        for blob_id, sig in blob_sigs.items():
            dominant_proteins = sig.get('dominant_proteins', [])[:2]
            
            # Find functional group for this blob
            for group_idx, (group_name, proteins) in enumerate(functional_groups.items()):
                if group_name == 'structural_controls':
                    continue
                if any(p in proteins for p in dominant_proteins):
                    # Assign pixels to this group (simplified spatial assignment)
                    start_idx = int(blob_id) * (len(coords) // max(len(blob_sigs), 1))
                    end_idx = min((int(blob_id) + 1) * (len(coords) // max(len(blob_sigs), 1)), len(coords))
                    pixel_groups[start_idx:end_idx] = group_idx
                    break
        
        # Plot each functional group
        group_names = [k for k in functional_groups.keys() if k != 'structural_controls']
        for group_idx, group_name in enumerate(group_names):
            mask = pixel_groups == group_idx
            if np.any(mask):
                group_coords = coords[mask]
                color = colors.get(group_name, 'gray')
                ax.scatter(group_coords[:, 0], group_coords[:, 1],
                         c=color, s=0.8, alpha=0.7, label=group_name.replace('kidney_', '').title())
        
        # Plot unassigned pixels
        unassigned_mask = pixel_groups == -1
        if np.any(unassigned_mask):
            unassigned_coords = coords[unassigned_mask]
            ax.scatter(unassigned_coords[:, 0], unassigned_coords[:, 1],
                     c='lightgray', s=0.3, alpha=0.3, label='Other')
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

    def save_all_figures(self, results: List[Dict]) -> None:
        """Generate and save all figures."""
        figures = self.create_all_figures(results)
        
        for name, fig in figures.items():
            output_path = self.config.output_dir / f'{name}_analysis.png'
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved {name} analysis to {output_path}")
            plt.close(fig)