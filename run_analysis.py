#!/usr/bin/env python3
"""Main analysis runner for IMC data."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.analysis import BatchAnalyzer


def main():
    """Run complete IMC analysis pipeline."""
    
    # Initialize configuration
    config = Config('config.json')
    
    print("=" * 60)
    print("IMC Analysis Pipeline v2.0")
    print("=" * 60)
    
    # Run spatial blob analysis
    print("\nRunning spatial blob analysis...")
    batch_analyzer = BatchAnalyzer(config)
    results = batch_analyzer.analyze_all()
    
    print(f"✅ Analyzed {len(results)} ROIs successfully")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()