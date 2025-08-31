#!/usr/bin/env python3
"""Main visualization runner for IMC data."""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.visualization.main import VisualizationPipeline


def main():
    """Run IMC visualization pipeline."""
    
    # Initialize configuration
    config = Config('config.json')
    
    print("=" * 60)
    print("IMC Visualization Pipeline v2.0")
    print("=" * 60)
    
    # Load analysis results
    results_file = config.output_dir / 'analysis_results.json'
    if not results_file.exists():
        print(f"ERROR: Results file not found at {results_file}")
        print("Please run run_analysis.py first")
        return 1
    
    with open(results_file, 'r') as f:
        results_raw = json.load(f)
    
    # Convert metadata dicts back to objects for visualization compatibility
    from src.utils.helpers import Metadata
    results = []
    for r in results_raw:
        r_copy = r.copy()
        if isinstance(r_copy['metadata'], dict):
            meta = r_copy['metadata']
            r_copy['metadata'] = Metadata(
                condition=meta.get('condition', 'Unknown'),
                injury_day=meta.get('injury_day'),
                tissue_region=meta.get('tissue_region', 'Unknown'),
                mouse_replicate=meta.get('mouse_replicate', 'Unknown')
            )
        results.append(r_copy)
    
    print(f"\nLoaded {len(results)} ROI results")
    
    # Initialize visualization pipeline
    viz_pipeline = VisualizationPipeline(config)
    
    print("\nGenerating visualizations...")
    
    # Generate all figures
    viz_pipeline.save_all_figures(results)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Outputs saved to {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()