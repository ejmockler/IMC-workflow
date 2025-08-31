#!/usr/bin/env python3
"""
Per-ROI Detailed Visualization Script

Creates comprehensive 3x3 grid visualizations for individual ROIs from IMC analysis.
Includes spatial domain maps, protein signatures, contact matrices, and functional analysis.

Usage:
    python run_per_roi_viz.py --full-analysis                    # Visualize all ROIs (detailed)
    python run_per_roi_viz.py --roi-index 5 --full-analysis      # Visualize specific ROI by index  
    python run_per_roi_viz.py --roi-name "D7_M1" --full-analysis # Visualize by filename pattern
    python run_per_roi_viz.py --full-analysis --output-dir detailed_rois # Custom output directory

Note: --full-analysis is required to generate spatial data for proper ROI visualizations.
Without it, the script will only inform you to use --full-analysis.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.config import Config
from src.visualization.main import VisualizationPipeline


def main():
    parser = argparse.ArgumentParser(description='Generate per-ROI visualizations')
    parser.add_argument('--config', default='config.json', 
                       help='Configuration file path (default: config.json)')
    parser.add_argument('--results', default='results/analysis_results.json',
                       help='Analysis results JSON file (default: results/analysis_results.json)')
    parser.add_argument('--full-analysis', action='store_true',
                       help='Run full ROI analysis to get spatial data (REQUIRED for proper ROI visualizations)')
    parser.add_argument('--roi-index', type=int, 
                       help='Index of specific ROI to visualize (0-based)')
    parser.add_argument('--roi-name', type=str,
                       help='Filename pattern to match for ROI selection')
    parser.add_argument('--output-dir', default='results/per_roi',
                       help='Output directory for ROI visualizations (default: results/per_roi)')
    parser.add_argument('--list-rois', action='store_true',
                       help='List all available ROIs and exit')
    
    args = parser.parse_args()
    
    # Load configuration and results
    config = Config(args.config)
    
    if not Path(args.results).exists():
        print(f"❌ Results file not found: {args.results}")
        print("Run the analysis first: python imc_pipeline.py data/241218_IMC_Alun/")
        return
    
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    if not results:
        print("❌ No ROI results found in analysis file")
        return
    
    # List ROIs if requested
    if args.list_rois:
        print(f"📋 Found {len(results)} ROIs:")
        for i, roi in enumerate(results):
            filename = roi.get('filename', f'ROI_{i}')
            metadata = roi.get('metadata', {})
            condition = metadata.get('condition', 'Unknown')
            timepoint = metadata.get('injury_day', metadata.get('timepoint', '?'))
            region = metadata.get('tissue_region', metadata.get('region', 'Unknown'))
            mouse = metadata.get('mouse_replicate', metadata.get('mouse_id', 'Unknown'))
            
            print(f"  [{i:2d}] {filename:<35} | {condition:<8} | Day {timepoint:<2} | {region:<8} | {mouse}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize visualization pipeline
    pipeline = VisualizationPipeline(config)
    
    # Check if we need full analysis for proper ROI visualizations
    if args.full_analysis:
        print("🔬 Running full ROI analysis to get spatial data...")
        from src.analysis.roi import ROIAnalyzer
        from src.utils.helpers import find_roi_files
        
        roi_analyzer = ROIAnalyzer(config)
        roi_files = find_roi_files(config.data_dir)
        
        # Create filename-to-data mapping
        roi_data_map = {}
        for roi_file in roi_files:
            try:
                roi_data = roi_analyzer.analyze(roi_file)
                roi_data_map[roi_file.name] = roi_data
                print(f"  ✅ Analyzed {roi_file.name}")
            except Exception as e:
                print(f"  ❌ Failed {roi_file.name}: {e}")
        
        # Update results with full data
        for roi in results:
            filename = roi.get('filename')
            if filename in roi_data_map:
                # Merge the full analysis data
                roi.update(roi_data_map[filename])
    
    # Determine which ROIs to process
    if args.roi_index is not None:
        if args.roi_index >= len(results):
            print(f"❌ ROI index {args.roi_index} out of range (0-{len(results)-1})")
            return
        target_rois = [args.roi_index]
        print(f"🎯 Visualizing ROI index {args.roi_index}")
        
    elif args.roi_name:
        target_rois = []
        for i, roi in enumerate(results):
            filename = roi.get('filename', '')
            if args.roi_name.lower() in filename.lower():
                target_rois.append(i)
        
        if not target_rois:
            print(f"❌ No ROIs found matching pattern: {args.roi_name}")
            return
        print(f"🎯 Found {len(target_rois)} ROIs matching '{args.roi_name}'")
        
    else:
        target_rois = list(range(len(results)))
        print(f"🎯 Visualizing all {len(results)} ROIs")
    
    # Generate visualizations
    for i, roi_idx in enumerate(target_rois):
        roi_data = results[roi_idx]
        filename = roi_data.get('filename', f'ROI_{roi_idx}')
        
        print(f"  [{i+1}/{len(target_rois)}] Processing {filename}")
        
        try:
            # Check if we have spatial data for proper ROI visualization
            has_spatial_data = all(key in roi_data for key in ['coords', 'blob_labels', 'blob_type_mapping'])
            
            if has_spatial_data:
                # Use the proper ROI visualizer
                fig = pipeline.create_roi_figure(roi_data)
                suffix = "_detailed"
            else:
                print(f"    ⚠️  Missing spatial data for {filename}. Use --full-analysis for complete ROI visualization.")
                continue
            
            # Generate output filename
            clean_filename = filename.replace('.txt', '').replace('IMC_241218_Alun_', '')
            output_path = output_dir / f"{clean_filename}{suffix}.png"
            
            # Save figure
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"    ✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"    ❌ Error processing {filename}: {str(e)}")
            continue


if __name__ == '__main__':
    main()