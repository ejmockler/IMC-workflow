#!/usr/bin/env python3
"""
Complete end-to-end multi-scale neighborhood analysis pipeline.
Analyzes all ROIs, generates extended visualizations, and creates publication figures.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.analysis.roi import BatchAnalyzer
from src.visualization.roi import ROIVisualizer
from src.visualization.neighborhood import NeighborhoodVisualizer
from src.visualization.temporal import TemporalVisualizer
from src.visualization.condition import ConditionVisualizer
from src.visualization.replicate import ReplicateVisualizer


def main():
    start_time = datetime.now()
    
    print("=" * 80)
    print("COMPLETE END-TO-END MULTI-SCALE NEIGHBORHOOD ANALYSIS")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load configuration
    config = Config('config.json')
    
    # Phase 1: Run complete analysis on all ROIs
    print("PHASE 1: ANALYZING ALL ROIs WITH MULTI-SCALE NEIGHBORHOODS")
    print("-" * 80)
    
    batch_analyzer = BatchAnalyzer(config)
    results = batch_analyzer.analyze_all()
    
    print(f"\n✓ Analyzed {len(results)} ROIs")
    
    # Save results
    results_file = config.output_dir / 'analysis_results_multiscale.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Saved results to {results_file}")
    
    # Check multi-scale data
    ms_count = sum(1 for r in results if 'multiscale_neighborhoods' in r)
    print(f"✓ Multi-scale data available for {ms_count}/{len(results)} ROIs")
    
    # Phase 2: Generate extended ROI visualizations
    print("\nPHASE 2: GENERATING EXTENDED 4×4 ROI VISUALIZATIONS")
    print("-" * 80)
    
    roi_viz = ROIVisualizer(config)
    extended_dir = config.output_dir / 'extended_rois'
    extended_dir.mkdir(exist_ok=True)
    
    # Generate for all ROIs with multi-scale data
    generated = 0
    for i, roi_data in enumerate(results):
        if 'multiscale_neighborhoods' in roi_data:
            filename = roi_data.get('filename', f'ROI_{i}')
            clean_name = filename.replace('.txt', '').replace('IMC_241218_Alun_', '')
            
            try:
                fig = roi_viz.create_figure(roi_data, extended=True)
                output_path = extended_dir / f"{clean_name}_extended.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                generated += 1
                
                if generated % 5 == 0:
                    print(f"  Generated {generated} visualizations...")
                    
            except Exception as e:
                print(f"  ⚠ Error with {filename}: {e}")
    
    print(f"\n✓ Generated {generated} extended ROI visualizations")
    
    # Phase 3: Create publication-ready summary figures
    print("\nPHASE 3: CREATING PUBLICATION-READY SUMMARY FIGURES")
    print("-" * 80)
    
    nbhd_viz = NeighborhoodVisualizer(config)
    
    # 1. Multi-scale dynamics
    print("  Creating multi-scale dynamics figure...")
    try:
        fig = nbhd_viz.create_multiscale_dynamics_figure(results)
        output_path = config.output_dir / 'kidney_neighborhood_multiscale.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved to {output_path.name}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # 2. Temporal evolution
    print("  Creating temporal evolution figure...")
    try:
        fig = nbhd_viz.create_temporal_evolution_figure(results)
        output_path = config.output_dir / 'kidney_neighborhood_temporal.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved to {output_path.name}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # 3. Regional comparison
    print("  Creating regional comparison figure...")
    try:
        fig = nbhd_viz.create_regional_comparison_figure(results)
        output_path = config.output_dir / 'kidney_neighborhood_regional.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved to {output_path.name}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # Phase 4: Generate standard kidney healing figures
    print("\nPHASE 4: GENERATING STANDARD KIDNEY HEALING FIGURES")
    print("-" * 80)
    
    # Temporal analysis
    temporal_viz = TemporalVisualizer(config)
    print("  Creating temporal progression figure...")
    try:
        fig = temporal_viz.create_temporal_figure(results)
        output_path = config.output_dir / 'kidney_healing_timeline.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved to {output_path.name}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # Condition comparison
    condition_viz = ConditionVisualizer(config)
    print("  Creating condition comparison figure...")
    try:
        fig = condition_viz.create_condition_figure(results)
        output_path = config.output_dir / 'kidney_condition_comparison.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved to {output_path.name}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # Replicate variance
    replicate_viz = ReplicateVisualizer(config)
    print("  Creating replicate variance figure...")
    try:
        fig = replicate_viz.create_replicate_variance_figure(results)
        output_path = config.output_dir / 'kidney_replicate_variance.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved to {output_path.name}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # Summary statistics
    print("\nPHASE 5: SUMMARY STATISTICS")
    print("-" * 80)
    
    # Collect neighborhood statistics across scales
    scale_stats = {'cellular': [], 'microenvironment': [], 'functional_unit': [], 'tissue_region': []}
    
    for roi in results:
        if 'multiscale_neighborhoods' in roi:
            for scale_name, scale_data in roi['multiscale_neighborhoods'].items():
                if scale_name in scale_stats:
                    scale_stats[scale_name].append({
                        'n_neighborhoods': scale_data.get('n_neighborhoods', 0),
                        'coverage': scale_data.get('coverage', 0),
                        'roi': roi.get('filename', 'unknown')
                    })
    
    # Print statistics
    for scale_name, data_list in scale_stats.items():
        if data_list:
            import numpy as np
            n_nbhds = [d['n_neighborhoods'] for d in data_list]
            coverages = [d['coverage'] for d in data_list]
            
            print(f"\n  {scale_name.upper()} SCALE:")
            print(f"    Neighborhoods: {np.mean(n_nbhds):.1f} ± {np.std(n_nbhds):.1f}")
            print(f"    Coverage: {np.mean(coverages):.1%} ± {np.std(coverages):.1%}")
            print(f"    Range: {min(n_nbhds)}-{max(n_nbhds)} neighborhoods")
    
    # Save summary statistics
    stats_file = config.output_dir / 'neighborhood_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(scale_stats, f, indent=2)
    print(f"\n✓ Saved statistics to {stats_file}")
    
    # Complete
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration.total_seconds():.1f} seconds")
    print("\nGenerated outputs:")
    print(f"  • {generated} extended ROI visualizations in {extended_dir}")
    print(f"  • 3 neighborhood summary figures")
    print(f"  • 3 kidney healing figures")
    print(f"  • Statistical summary in {stats_file}")


if __name__ == '__main__':
    main()