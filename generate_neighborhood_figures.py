#!/usr/bin/env python3
"""
Generate multi-scale neighborhood analysis figures.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.analysis.roi import ROIAnalyzer
from src.visualization.roi import ROIVisualizer
from src.visualization.neighborhood import NeighborhoodVisualizer
from src.utils.helpers import find_roi_files


def main():
    print("=" * 60)
    print("MULTI-SCALE NEIGHBORHOOD ANALYSIS")
    print("=" * 60)
    
    # Load config
    config = Config('config.json')
    
    # Check for existing results
    results_file = config.output_dir / 'analysis_results.json'
    
    if results_file.exists():
        print(f"Loading existing results from {results_file}")
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} ROI results")
    else:
        print("No existing results found. Running new analysis...")
        # Run analysis
        roi_analyzer = ROIAnalyzer(config)
        roi_files = find_roi_files(config.data_dir)
        
        results = []
        for i, roi_file in enumerate(roi_files, 1):
            print(f"  [{i}/{len(roi_files)}] {roi_file.name}")
            try:
                result = roi_analyzer.analyze(roi_file)
                results.append(result)
            except Exception as e:
                print(f"    ERROR: {e}")
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved {len(results)} results to {results_file}")
    
    # Generate individual ROI visualizations (extended 4x4 grid)
    print("\n" + "=" * 60)
    print("GENERATING EXTENDED ROI VISUALIZATIONS")
    print("=" * 60)
    
    roi_viz = ROIVisualizer(config)
    extended_dir = config.output_dir / 'extended_rois'
    extended_dir.mkdir(exist_ok=True)
    
    # Generate for first 3 ROIs as examples
    for i, roi_data in enumerate(results[:3]):
        filename = roi_data.get('filename', f'ROI_{i}')
        print(f"  Generating extended visualization for {filename}")
        
        try:
            # Check if we have multi-scale data
            if 'multiscale_neighborhoods' in roi_data:
                fig = roi_viz.create_figure(roi_data, extended=True)
                output_path = extended_dir / f"{filename.replace('.txt', '')}_extended.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"    ✓ Saved to {output_path}")
            else:
                print(f"    ⚠ No multi-scale data found")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Generate publication-ready summary figures
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION SUMMARY FIGURES")
    print("=" * 60)
    
    nbhd_viz = NeighborhoodVisualizer(config)
    
    # 1. Multi-scale dynamics figure
    print("  Creating multi-scale dynamics figure...")
    try:
        fig = nbhd_viz.create_multiscale_dynamics_figure(results)
        output_path = config.output_dir / 'kidney_neighborhood_multiscale.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved to {output_path}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # 2. Temporal evolution figure
    print("  Creating temporal evolution figure...")
    try:
        fig = nbhd_viz.create_temporal_evolution_figure(results)
        output_path = config.output_dir / 'kidney_neighborhood_temporal.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved to {output_path}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # 3. Regional comparison figure
    print("  Creating regional comparison figure...")
    try:
        fig = nbhd_viz.create_regional_comparison_figure(results)
        output_path = config.output_dir / 'kidney_neighborhood_regional.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved to {output_path}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()