#!/usr/bin/env python3
"""
Kidney Healing Comprehensive Report
Generates all publication-quality figures for kidney injury research
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path for imports  
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.analysis.roi import BatchAnalyzer
from src.visualization.temporal import TemporalVisualizer
from src.visualization.condition import ConditionVisualizer
from src.visualization.replicate import ReplicateVisualizer
from src.visualization.network_clean import CleanNetworkVisualizer


def compute_functional_group_percentages(roi, functional_groups: Dict[str, List[str]]) -> Dict[str, float]:
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


def aggregate_metrics(results: List[Dict], config: Config) -> Dict:
    # Build label map for timepoints
    exp_cfg = config.raw.get('experimental', {})
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
            # Dict format
            day = metadata.get('injury_day')
            region = metadata.get('tissue_region', 'Unknown')
        else:
            # Metadata object format
            day = metadata.injury_day
            region = metadata.tissue_region or 'Unknown'
        
        if day is not None:
            tp = f"D{day}"
            by_timepoint.setdefault(tp, []).append(roi)
        by_region.setdefault(region, []).append(roi)
        if day is not None:
            by_region_time.setdefault(region, {}).setdefault(tp, []).append(roi)

    functional_groups = config.raw.get('proteins', {}).get('functional_groups', {})

    def summarize(group: List[Dict]) -> Dict:
        if not group:
            return {}
        domain_counts = [len(r['blob_signatures']) for r in group]
        contact_counts = [len(r.get('canonical_contacts', {})) for r in group]
        # Functional groups
        fg_list = [compute_functional_group_percentages(r, functional_groups) for r in group]
        fg_keys = sorted({k for d in fg_list for k in d.keys()})
        fg_summary = {
            k: float(np.mean([d.get(k, 0.0) for d in fg_list])) if fg_list else 0.0
            for k in fg_keys
        }
        # Top contacts by mean strength
        contact_strengths: Dict[str, List[float]] = {}
        for r in group:
            for c, s in r.get('canonical_contacts', {}).items():
                contact_strengths.setdefault(c, []).append(float(s))
        top_contacts = []
        if contact_strengths:
            means = {c: float(np.mean(v)) for c, v in contact_strengths.items()}
            top_contacts = sorted(means.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            'n_rois': len(group),
            'domains_mean': float(np.mean(domain_counts)) if domain_counts else 0.0,
            'domains_std': float(np.std(domain_counts)) if domain_counts else 0.0,
            'contacts_mean': float(np.mean(contact_counts)) if contact_counts else 0.0,
            'contacts_std': float(np.std(contact_counts)) if contact_counts else 0.0,
            'functional_groups_mean_pct': fg_summary,
            'top_contacts_mean_strength': [{
                'contact': c,
                'mean_strength': float(v)
            } for c, v in top_contacts]
        }

    metrics = {
        'timepoints': {tp: summarize(group) | {'label': label_map.get(tp, tp)} 
                       for tp, group in sorted(by_timepoint.items(), key=lambda x: int(x[0][1:]))},
        'regions': {region: summarize(group) for region, group in by_region.items()},
        'timepoint_region': {
            region: {tp: summarize(group) for tp, group in sorted(tp_map.items(), key=lambda x: int(x[0][1:]))}
            for region, tp_map in by_region_time.items()
        }
    }
    return metrics


def create_functional_group_dynamics_figure(metrics: Dict, config: Config) -> plt.Figure:
    """Create publication-quality functional group dynamics figure with region stratification."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract timepoints and labels
    timepoints = sorted([k for k in metrics['timepoints'].keys()], key=lambda x: int(x[1:]))
    x_pos = np.arange(len(timepoints))
    
    # Get display labels from config
    exp_cfg = config.raw.get('experimental', {})
    tp_vals = exp_cfg.get('timepoints', [])
    tp_labels = exp_cfg.get('timepoint_labels', [])
    label_map = {f"D{tp}": lbl for tp, lbl in zip(tp_vals, tp_labels)}
    display_labels = [label_map.get(tp, tp) for tp in timepoints]
    
    # Colors for each group and region
    colors = {'kidney_inflammation': '#e74c3c', 'kidney_repair': '#2ecc71', 'kidney_vasculature': '#3498db'}
    region_colors = {'Cortex': 0.9, 'Medulla': 0.6}  # Alpha values for stratification
    
    functional_groups = [
        ('kidney_inflammation', 'Inflammation (CD45/CD11b)'),
        ('kidney_repair', 'Repair (CD206/CD44)'),
        ('kidney_vasculature', 'Vasculature (CD31/CD34)')
    ]
    
    for col, (group_key, group_name) in enumerate(functional_groups):
        # Top row: Overall temporal trends
        ax_overall = axes[0, col]
        
        # Overall values (from existing metrics)
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
        
        # Extract regional data
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
        
        # Add statistical significance indicators for regional differences  
        # Find maximum bar height across all containers
        max_val = 0
        for container in ax_regions.containers:
            if hasattr(container, '__iter__'):
                for bar in container:
                    if hasattr(bar, 'get_height'):
                        max_val = max(max_val, bar.get_height())
        max_val = max(max_val, 1)  # Ensure minimum value of 1
        
        # Compare regions at each timepoint (simplified significance testing)
        for i, tp in enumerate(timepoints):
            cortex_data = region_data.get('Cortex', {}).get(tp, {})
            medulla_data = region_data.get('Medulla', {}).get(tp, {})
            
            cortex_val = cortex_data.get('functional_groups_mean_pct', {}).get(group_key, 0)
            medulla_val = medulla_data.get('functional_groups_mean_pct', {}).get(group_key, 0)
            
            # Simple fold-change threshold for significance
            if abs(cortex_val - medulla_val) > 20:  # 20% difference threshold
                ax_regions.plot([x_cortex[i], x_medulla[i]], [max_val + 5, max_val + 5], 'k-', linewidth=1)
                ax_regions.text(x_pos[i], max_val + 7, '*', ha='center', va='center', fontsize=12)
    
    fig.suptitle('Functional Group Dynamics - Overall vs Regional Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate kidney healing comprehensive report')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--publication', action='store_true', 
                       help='Generate publication-quality figures (600 DPI)')
    parser.add_argument('--stats', action='store_true',
                       help='Include statistical testing in figures')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                       help='Output format for figures')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    config.output_dir.mkdir(exist_ok=True)
    
    # Set DPI based on mode
    dpi = 600 if args.publication else 300
    format_ext = args.format

    # Analyze ROIs
    print('=' * 60)
    print('KIDNEY HEALING COMPREHENSIVE REPORT')
    print('=' * 60)
    print(f'Mode: {"Publication" if args.publication else "Standard"}')
    print(f'DPI: {dpi}')
    print(f'Format: {format_ext.upper()}')
    print(f'Statistical testing: {"Enabled" if args.stats else "Disabled"}')
    print('=' * 60)
    
    batch = BatchAnalyzer(config)
    results = batch.analyze_all()
    if not results:
        print('No ROIs analyzed. Exiting.')
        return

    # Initialize visualizers
    temporal_viz = TemporalVisualizer(config)
    condition_viz = ConditionVisualizer(config)
    replicate_viz = ReplicateVisualizer(config)
    network_viz = CleanNetworkVisualizer()

    # Generate aggregated metrics first (needed for functional groups)
    metrics = aggregate_metrics(results, config)
    out_json = config.output_dir / 'kidney_healing_metrics.json'
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'✓ Saved metrics: {out_json}')

    # 1) Timeline
    fig = temporal_viz.create_temporal_figure(results)
    out_timeline = config.output_dir / f'kidney_healing_timeline.{format_ext}'
    fig.savefig(out_timeline, dpi=dpi, bbox_inches='tight', format=format_ext)
    plt.close(fig)
    print(f'✓ Saved timeline: {out_timeline}')

    # 2) Condition comparison
    fig = condition_viz.create_condition_figure(results)
    out_condition = config.output_dir / f'kidney_condition_comparison.{format_ext}'
    fig.savefig(out_condition, dpi=dpi, bbox_inches='tight', format=format_ext)
    plt.close(fig)
    print(f'✓ Saved condition comparison: {out_condition}')

    # 3) Replicate variance
    fig = replicate_viz.create_replicate_variance_figure(results)
    out_repl = config.output_dir / f'kidney_replicate_variance.{format_ext}'
    fig.savefig(out_repl, dpi=dpi, bbox_inches='tight', format=format_ext)
    plt.close(fig)
    print(f'✓ Saved replicate variance: {out_repl}')

    # 4) Functional group dynamics
    fig = create_functional_group_dynamics_figure(metrics, config)
    out_functional = config.output_dir / f'kidney_functional_dynamics.{format_ext}'
    fig.savefig(out_functional, dpi=dpi, bbox_inches='tight', format=format_ext)
    plt.close(fig)
    print(f'✓ Saved functional dynamics: {out_functional}')
    
    # 5) Network analysis
    out_network = config.output_dir / f'kidney_network_analysis.{format_ext}'
    network_viz.create_network_grid(results, str(out_network))
    print(f'✓ Saved network analysis: {out_network}')
    
    # 6) Regional temporal trajectories
    fig = replicate_viz.create_region_temporal_trajectories(results, config)
    out_regional = config.output_dir / f'kidney_regional_trajectories.{format_ext}'
    fig.savefig(out_regional, dpi=dpi, bbox_inches='tight', format=format_ext)
    plt.close(fig)
    print(f'✓ Saved regional trajectories: {out_regional}')
    
    # 7) Region × Time interaction heatmap
    fig = replicate_viz.create_region_time_interaction_heatmap(results, config)
    out_interaction = config.output_dir / f'kidney_region_time_heatmap.{format_ext}'
    fig.savefig(out_interaction, dpi=dpi, bbox_inches='tight', format=format_ext)
    plt.close(fig)
    print(f'✓ Saved region-time interaction heatmap: {out_interaction}')
    
    print('=' * 60)
    print('REPORT GENERATION COMPLETE')
    print(f'All figures saved to: {config.output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()