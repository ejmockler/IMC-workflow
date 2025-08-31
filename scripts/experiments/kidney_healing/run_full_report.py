#!/usr/bin/env python3
"""
Kidney Healing Comprehensive Report
Generates all publication-quality figures for kidney injury research
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path for imports  
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.analysis.roi import BatchAnalyzer
from src.visualization.temporal import TemporalVisualizer
from src.visualization.condition import ConditionVisualizer
from src.visualization.replicate import ReplicateVisualizer


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


def main():
    # Load config
    config = Config('config.json')
    config.output_dir.mkdir(exist_ok=True)

    # Analyze ROIs
    batch = BatchAnalyzer(config)
    results = batch.analyze_all()
    if not results:
        print('No ROIs analyzed. Exiting.')
        return

    # Visual figures
    temporal_viz = TemporalVisualizer(config)
    condition_viz = ConditionVisualizer(config)
    replicate_viz = ReplicateVisualizer(config)

    # 1) Timeline
    fig = temporal_viz.create_temporal_figure(results)
    out_timeline = config.output_dir / 'kidney_healing_timeline.png'
    fig.savefig(out_timeline, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_timeline}')

    # 2) Condition comparison
    fig = condition_viz.create_condition_figure(results)
    out_condition = config.output_dir / 'kidney_condition_comparison.png'
    fig.savefig(out_condition, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_condition}')

    # 3) Replicate variance
    fig = replicate_viz.create_replicate_variance_figure(results)
    out_repl = config.output_dir / 'kidney_replicate_variance.png'
    fig.savefig(out_repl, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_repl}')

    # 4) Region × time grid
    fig = replicate_viz.create_timepoint_region_contact_grid(results, config)
    out_grid = config.output_dir / 'kidney_region_time_grid.png'
    fig.savefig(out_grid, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_grid}')

    # Aggregated metrics JSON
    metrics = aggregate_metrics(results, config)
    out_json = config.output_dir / 'kidney_healing_metrics.json'
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved: {out_json}')


if __name__ == '__main__':
    main()