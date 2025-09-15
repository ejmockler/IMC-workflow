#!/usr/bin/env python3
"""
Generate kidney healing neighborhood visualizations.
Uses existing analysis results or runs fresh analysis as needed.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.analysis.roi import BatchAnalyzer, ROIAnalyzer
from src.visualization.roi import ROIVisualizer
from src.visualization.neighborhood import NeighborhoodVisualizer
from src.utils.helpers import find_roi_files


def main():
    print("=" * 80)
    print("KIDNEY HEALING NEIGHBORHOOD VISUALIZATION PIPELINE")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    
    config = Config('config.json')
    
    # Try to load existing results first
    results_file = config.output_dir / 'analysis_results_multiscale.json'
    
    if results_file.exists():
        print("\n✓ Loading existing multi-scale analysis results...")
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} ROI results")
    else:
        print("\n⚠ No existing results found. Running quick analysis on subset...")
        # Analyze first 6 ROIs for demonstration
        roi_analyzer = ROIAnalyzer(config)
        roi_files = find_roi_files(config.data_dir)[:6]  # Subset for speed
        
        results = []
        for i, roi_file in enumerate(roi_files, 1):
            print(f"  [{i}/{len(roi_files)}] Analyzing {roi_file.name}")
            try:
                result = roi_analyzer.analyze(roi_file)
                results.append(result)
            except Exception as e:
                print(f"    Error: {e}")
        
        # Save for reuse
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Saved {len(results)} results to {results_file}")
    
    # Check multi-scale data availability
    ms_count = sum(1 for r in results if 'multiscale_neighborhoods' in r)
    print(f"\n📊 Multi-scale data available: {ms_count}/{len(results)} ROIs")
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING NEIGHBORHOOD VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Extended ROI visualizations (select representative ones)
    print("\n1. EXTENDED 4×4 ROI VISUALIZATIONS")
    print("-" * 40)
    
    roi_viz = ROIVisualizer(config)
    extended_dir = config.output_dir / 'kidney_extended_rois'
    extended_dir.mkdir(exist_ok=True)
    
    # Select representative ROIs (one per condition/timepoint)
    selected_rois = []
    conditions_seen = set()
    
    for roi in results:
        if 'multiscale_neighborhoods' not in roi:
            continue
            
        meta = roi.get('metadata', {})
        if isinstance(meta, dict):
            condition = meta.get('condition', 'Unknown')
            timepoint = meta.get('timepoint', -1)
        else:
            condition = getattr(meta, 'condition', 'Unknown')
            timepoint = getattr(meta, 'timepoint', -1)
        
        key = f"{condition}_D{timepoint}"
        if key not in conditions_seen and len(selected_rois) < 4:
            selected_rois.append(roi)
            conditions_seen.add(key)
    
    for roi in selected_rois:
        filename = roi.get('filename', 'unknown')
        print(f"  Generating: {filename}")
        try:
            fig = roi_viz.create_figure(roi, extended=True)
            clean_name = filename.replace('.txt', '').replace('IMC_241218_Alun_', '')
            output_path = extended_dir / f"{clean_name}_extended.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    ✓ Saved to {output_path.name}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # 2. Publication-ready summary figures
    print("\n2. PUBLICATION-READY SUMMARY FIGURES")
    print("-" * 40)
    
    nbhd_viz = NeighborhoodVisualizer(config)
    
    figures = [
        ('Multi-scale Dynamics', 'kidney_neighborhood_multiscale.png', 
         nbhd_viz.create_multiscale_dynamics_figure),
        ('Temporal Evolution', 'kidney_neighborhood_temporal.png',
         nbhd_viz.create_temporal_evolution_figure),
        ('Regional Comparison', 'kidney_neighborhood_regional.png',
         nbhd_viz.create_regional_comparison_figure)
    ]
    
    for name, filename, create_func in figures:
        print(f"  Creating {name}...")
        try:
            fig = create_func(results)
            output_path = config.output_dir / filename
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    ✓ Saved to {filename}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # 3. Summary statistics
    print("\n3. NEIGHBORHOOD STATISTICS SUMMARY")
    print("-" * 40)
    
    # Collect statistics by scale
    scale_stats = {}
    for scale in ['cellular', 'microenvironment', 'functional_unit', 'tissue_region']:
        n_neighborhoods = []
        coverages = []
        
        for roi in results:
            if 'multiscale_neighborhoods' in roi:
                scale_data = roi['multiscale_neighborhoods'].get(scale, {})
                if scale_data:
                    n_neighborhoods.append(scale_data.get('n_neighborhoods', 0))
                    coverages.append(scale_data.get('coverage', 0))
        
        if n_neighborhoods:
            scale_stats[scale] = {
                'mean_neighborhoods': np.mean(n_neighborhoods),
                'std_neighborhoods': np.std(n_neighborhoods),
                'mean_coverage': np.mean(coverages),
                'std_coverage': np.std(coverages),
                'min_neighborhoods': min(n_neighborhoods),
                'max_neighborhoods': max(n_neighborhoods)
            }
    
    # Print statistics
    for scale, stats in scale_stats.items():
        print(f"\n  {scale.upper()} ({config.raw['neighborhood_analysis']['scales'][scale]['radius']}μm):")
        print(f"    Neighborhoods: {stats['mean_neighborhoods']:.1f} ± {stats['std_neighborhoods']:.1f}")
        print(f"    Coverage: {stats['mean_coverage']:.1%} ± {stats['std_coverage']:.1%}")
        print(f"    Range: {stats['min_neighborhoods']}-{stats['max_neighborhoods']} neighborhoods")
    
    # Save statistics
    stats_file = config.output_dir / 'kidney_neighborhood_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(scale_stats, f, indent=2)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION PIPELINE COMPLETE")
    print("=" * 80)
    print(f"End: {datetime.now().strftime('%H:%M:%S')}")
    print(f"\nOutputs generated:")
    print(f"  • Extended ROI visualizations in {extended_dir}")
    print(f"  • Neighborhood summary figures in {config.output_dir}")
    print(f"  • Statistics saved to {stats_file}")


if __name__ == '__main__':
    main()