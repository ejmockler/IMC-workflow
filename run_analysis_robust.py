#!/usr/bin/env python3
"""
Robust IMC Analysis Runner

Uses the new superpixel-based spatial analysis engine with squidpy integration
for publication-ready spatial statistics.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.analysis.roi_robust import BatchAnalyzer


def main():
    """Run complete IMC analysis pipeline with robust spatial engine."""
    
    # Initialize configuration
    config = Config('config.json')
    
    print("=" * 60)
    print("IMC Analysis Pipeline v3.0 - ROBUST ENGINE")
    print("Superpixel parcellation + squidpy spatial statistics")
    print("=" * 60)
    
    # Check spatial libraries
    try:
        import squidpy as sq
        import scanpy as sc
        print(f"✅ Spatial libraries available: squidpy {sq.__version__}, scanpy {sc.__version__}")
    except ImportError as e:
        print(f"❌ Missing spatial libraries: {e}")
        print("Install with: pip install squidpy scanpy statsmodels")
        sys.exit(1)
    
    # Run spatial analysis with robust engine
    print("\nRunning robust spatial analysis...")
    batch_analyzer = BatchAnalyzer(config)
    results = batch_analyzer.analyze_all()
    
    print(f"✅ Analyzed {len(results)} ROIs successfully with robust engine")
    
    # Summary of robust analysis features
    if results:
        first_result = results[0]
        
        print("\n📊 Analysis Summary:")
        if 'superpixel_summary' in first_result:
            n_superpixels = first_result['superpixel_summary'].get('n_superpixels', 'unknown')
            print(f"   • Superpixel parcellation: {n_superpixels} meta-cells per ROI")
            
        if 'spatial_parameters' in first_result:
            params = first_result['spatial_parameters']
            cellular_r = params.get('cellular_radius', 'unknown')
            functional_r = params.get('functional_radius', 'unknown')
            k_neighbors = params.get('k_neighbors', 'unknown')
            print(f"   • Literature-informed scales: {cellular_r}μm cellular, {functional_r}μm functional")
            print(f"   • Spatial graph: k-NN with k={k_neighbors}")
            
        if 'statistical_summary' in first_result:
            stats = first_result['statistical_summary']
            correction = stats.get('multiple_testing_correction', 'unknown')
            n_proteins = stats.get('n_proteins_significant_autocorr', 0)
            n_pairs = stats.get('n_pairs_significant_coloc', 0)
            print(f"   • Multiple testing correction: {correction}")
            print(f"   • Significant results: {n_proteins} proteins, {n_pairs} colocalization pairs")
    
    print("\n" + "=" * 60)
    print("Robust analysis complete!")
    print(f"Results saved to {config.output_dir}/analysis_results_robust.json")
    print("=" * 60)


if __name__ == "__main__":
    main()