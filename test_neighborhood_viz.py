#!/usr/bin/env python3
"""
Test multi-scale neighborhood visualization with a few ROIs.
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
    print("TEST MULTI-SCALE NEIGHBORHOOD VISUALIZATION")
    print("=" * 60)
    
    # Load config
    config = Config('config.json')
    
    # Analyze just 3 ROIs for testing
    roi_analyzer = ROIAnalyzer(config)
    roi_files = find_roi_files(config.data_dir)[:3]  # Just first 3
    
    print(f"Analyzing {len(roi_files)} test ROIs...")
    results = []
    for i, roi_file in enumerate(roi_files, 1):
        print(f"  [{i}/{len(roi_files)}] {roi_file.name}")
        try:
            result = roi_analyzer.analyze(roi_file)
            results.append(result)
            
            # Check multi-scale data
            if 'multiscale_neighborhoods' in result:
                scales = result['multiscale_neighborhoods'].keys()
                print(f"    ✓ Multi-scale data: {list(scales)}")
                for scale_name, scale_data in result['multiscale_neighborhoods'].items():
                    n_nbhd = scale_data.get('n_neighborhoods', 0)
                    coverage = scale_data.get('coverage', 0)
                    print(f"      - {scale_name}: {n_nbhd} neighborhoods, {coverage:.1%} coverage")
        except Exception as e:
            print(f"    ERROR: {e}")
    
    if not results:
        print("No results to visualize!")
        return
    
    # Test extended ROI visualization
    print("\n" + "=" * 60)
    print("TESTING EXTENDED ROI VISUALIZATION")
    print("=" * 60)
    
    roi_viz = ROIVisualizer(config)
    test_dir = config.output_dir / 'test_extended'
    test_dir.mkdir(exist_ok=True)
    
    roi_data = results[0]
    filename = roi_data.get('filename', 'test_roi')
    
    print(f"  Creating extended visualization for {filename}")
    try:
        fig = roi_viz.create_figure(roi_data, extended=True)
        output_path = test_dir / f"{filename.replace('.txt', '')}_extended.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved to {output_path}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test summary figures with limited data
    print("\n" + "=" * 60)
    print("TESTING NEIGHBORHOOD SUMMARY FIGURES")
    print("=" * 60)
    
    nbhd_viz = NeighborhoodVisualizer(config)
    
    # Test temporal evolution figure (even with limited data)
    print("  Creating test temporal evolution figure...")
    try:
        fig = nbhd_viz.create_temporal_evolution_figure(results)
        output_path = test_dir / 'test_neighborhood_temporal.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved to {output_path}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()