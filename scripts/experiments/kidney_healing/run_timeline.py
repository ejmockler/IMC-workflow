#!/usr/bin/env python3
"""
Kidney Healing Timeline Analysis
Generates temporal progression visualization for kidney injury recovery
"""

import sys
import json
from pathlib import Path

# Add project root to path for imports  
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.analysis.roi import BatchAnalyzer
from src.visualization.temporal import TemporalVisualizer


def main():
    """Generate kidney healing timeline visualization."""
    # Load config and ensure output directory exists
    config = Config('config.json')
    config.output_dir.mkdir(exist_ok=True)

    # Analyze all ROIs
    batch = BatchAnalyzer(config)
    results = batch.analyze_all()
    if not results:
        print('No ROIs analyzed. Exiting.')
        return

    # Build the kidney healing timeline (temporal progression)
    viz = TemporalVisualizer(config)
    fig = viz.create_temporal_figure(results)

    # Save a single coherent figure
    out_path = config.output_dir / 'kidney_healing_timeline.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()