#!/usr/bin/env python3
"""
Production Single Experiment Runner

Executes focused IMC analysis on specific ROI subsets with comprehensive
multi-scale analysis and batch correction capabilities.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import logging
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.analysis.multiscale_analysis import (
    perform_multiscale_analysis, 
    compute_scale_consistency,
    summarize_multiscale_analysis
)
from src.analysis.batch_correction import sham_anchored_normalize


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiment_analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_roi_data(roi_file: Path) -> Dict[str, Any]:
    """Load single ROI data from file."""
    try:
        roi_data = pd.read_csv(roi_file, sep='\t')
        
        # Extract coordinates and protein data
        coords = roi_data[['X', 'Y']].values
        
        # Get protein channels (exclude coordinates and DNA)
        protein_channels = [col for col in roi_data.columns 
                          if col not in ['X', 'Y'] and 'DNA' not in col]
        
        ion_counts = {
            channel.split('(')[0]: roi_data[channel].values
            for channel in protein_channels
        }
        
        # Get DNA channels for morphological information
        dna1_col = [col for col in roi_data.columns if 'DNA1' in col]
        dna2_col = [col for col in roi_data.columns if 'DNA2' in col]
        
        dna1_intensities = roi_data[dna1_col[0]].values if dna1_col else np.zeros(len(coords))
        dna2_intensities = roi_data[dna2_col[0]].values if dna2_col else np.zeros(len(coords))
        
        return {
            'coords': coords,
            'ion_counts': ion_counts,
            'dna1_intensities': dna1_intensities,
            'dna2_intensities': dna2_intensities,
            'n_pixels': len(coords),
            'filename': roi_file.name,
            'roi_name': roi_file.stem
        }
        
    except Exception as e:
        raise ValueError(f"Failed to load ROI data from {roi_file}: {e}")


def analyze_roi_set(roi_files: List[Path], config: Config) -> Dict[str, Any]:
    """Analyze a set of ROI files with multi-scale analysis."""
    logger = logging.getLogger('ExperimentAnalysis')
    
    # Get analysis parameters
    multiscale_config = config.raw.get('multiscale_analysis', {})
    scales_um = multiscale_config.get('scales_um', [10.0, 20.0, 40.0])
    
    ion_config = config.raw.get('ion_count_processing', {})
    n_clusters = ion_config.get('n_clusters', 8)
    use_slic = ion_config.get('use_slic_segmentation', True)
    
    roi_results = []
    
    for roi_file in roi_files:
        logger.info(f"Analyzing {roi_file.name}...")
        
        try:
            # Load ROI data
            roi_data = load_roi_data(roi_file)
            
            # Perform multi-scale analysis
            multiscale_results = perform_multiscale_analysis(
                coords=roi_data['coords'],
                ion_counts=roi_data['ion_counts'],
                dna1_intensities=roi_data['dna1_intensities'],
                dna2_intensities=roi_data['dna2_intensities'],
                scales_um=scales_um,
                n_clusters=n_clusters,
                use_slic=use_slic
            )
            
            # Compute scale consistency
            consistency_results = compute_scale_consistency(multiscale_results)
            
            roi_result = {
                'roi_file': str(roi_file),
                'roi_name': roi_data['roi_name'],
                'multiscale_results': multiscale_results,
                'consistency_results': consistency_results,
                'roi_metadata': {
                    'n_pixels': roi_data['n_pixels'],
                    'n_proteins': len(roi_data['ion_counts']),
                    'filename': roi_data['filename']
                }
            }
            
            roi_results.append(roi_result)
            logger.info(f"✅ Completed {roi_file.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed {roi_file.name}: {e}")
            continue
    
    return {
        'roi_results': roi_results,
        'summary': {
            'total_rois': len(roi_files),
            'successful_rois': len(roi_results),
            'failed_rois': len(roi_files) - len(roi_results),
            'scales_analyzed': scales_um,
            'analysis_method': 'slic' if use_slic else 'square'
        }
    }


def generate_experiment_summary(results: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """Generate comprehensive experiment summary."""
    roi_results = results['roi_results']
    
    if not roi_results:
        return {'error': 'No successful ROI analyses to summarize'}
    
    # Aggregate multi-scale results
    all_multiscale_results = {}
    all_consistency_results = {}
    
    for roi_result in roi_results:
        roi_name = roi_result['roi_name']
        all_multiscale_results[roi_name] = roi_result['multiscale_results']
        all_consistency_results[roi_name] = roi_result['consistency_results']
    
    # Compute experiment-level statistics
    experiment_summary = {}
    
    # Scale consistency statistics across all ROIs
    consistency_scores = []
    for roi_consistency in all_consistency_results.values():
        overall = roi_consistency.get('overall', {})
        if 'mean_ari' in overall:
            consistency_scores.append(overall['mean_ari'])
    
    if consistency_scores:
        experiment_summary['scale_consistency'] = {
            'mean_ari_across_rois': float(np.mean(consistency_scores)),
            'std_ari_across_rois': float(np.std(consistency_scores)),
            'min_ari': float(np.min(consistency_scores)),
            'max_ari': float(np.max(consistency_scores)),
            'n_rois': len(consistency_scores)
        }
    
    # Cluster statistics
    cluster_counts = []
    for roi_result in roi_results:
        for scale_result in roi_result['multiscale_results'].values():
            if 'cluster_centroids' in scale_result:
                cluster_counts.append(len(scale_result['cluster_centroids']))
    
    if cluster_counts:
        experiment_summary['clustering_statistics'] = {
            'mean_clusters_per_scale': float(np.mean(cluster_counts)),
            'std_clusters_per_scale': float(np.std(cluster_counts)),
            'total_cluster_observations': len(cluster_counts)
        }
    
    # Add configuration summary
    experiment_summary['analysis_parameters'] = {
        'scales_analyzed': config.raw.get('multiscale_analysis', {}).get('scales_um', []),
        'n_clusters_target': config.raw.get('ion_count_processing', {}).get('n_clusters', 8),
        'use_slic_segmentation': config.raw.get('ion_count_processing', {}).get('use_slic_segmentation', True)
    }
    
    return experiment_summary


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Production IMC single experiment analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --roi-pattern "ROI_T1_*"      # Analyze Timepoint 1 ROIs
  python run_experiment.py --roi-pattern "ROI_*_R1_*"    # Analyze Replicate 1 ROIs  
  python run_experiment.py --config custom.json         # Use custom configuration
  python run_experiment.py --output-dir results/exp1/   # Custom output directory
        """
    )
    
    parser.add_argument('--config', default='config.json',
                       help='Configuration file path (default: config.json)')
    parser.add_argument('--data-dir', type=str, default='data/241218_IMC_Alun',
                       help='Directory containing ROI files')
    parser.add_argument('--roi-pattern', type=str, default='*.txt',
                       help='Pattern to match ROI files (e.g., "ROI_D1_*")')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory (overrides config)')
    parser.add_argument('--batch-correction', action='store_true', default=False,
                       help='Apply batch correction (default: False for single experiments)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Generate only summary statistics, skip detailed results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger('ExperimentAnalysis')
    
    print('=' * 80)
    print('PRODUCTION SINGLE EXPERIMENT ANALYSIS')
    print('=' * 80)
    print('• Multi-scale spatial analysis (10μm, 20μm, 40μm)')
    print('• Scale consistency validation')
    print('• Focused ROI subset analysis')
    print('=' * 80)
    
    try:
        # Load configuration
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Override output directory if specified
        if args.output_dir:
            config.raw['output']['results_dir'] = args.output_dir
            logger.info(f"Output directory overridden to: {args.output_dir}")
        
        # Find ROI files matching pattern
        data_dir = Path(args.data_dir)
        roi_files = list(data_dir.glob(args.roi_pattern))
        logger.info(f"Found {len(roi_files)} ROI files matching '{args.roi_pattern}'")
        
        if not roi_files:
            logger.error(f"No ROI files found matching pattern '{args.roi_pattern}' in {data_dir}")
            return 1
        
        print(f"\nROI Files Selected:")
        for roi_file in roi_files:
            print(f"  • {roi_file.name}")
        
        # Run analysis
        logger.info("Starting experiment analysis...")
        results = analyze_roi_set(roi_files, config)
        
        if results['summary']['successful_rois'] == 0:
            logger.error("No successful ROI analyses completed")
            return 1
        
        # Generate experiment summary
        logger.info("Generating experiment summary...")
        experiment_summary = generate_experiment_summary(results, config)
        
        # Save results
        output_config = config.raw.get('output', {})
        results_dir = Path(output_config.get('results_dir', 'results/single_experiment'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results unless summary-only
        if not args.summary_only:
            detailed_results_file = results_dir / 'experiment_results_detailed.json'
            with open(detailed_results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Detailed results saved to: {detailed_results_file}")
        
        # Save experiment summary
        summary_file = results_dir / 'experiment_summary.json'
        full_summary = {
            'experiment_metadata': {
                'roi_pattern': args.roi_pattern,
                'data_directory': str(data_dir),
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'total_pixels_analyzed': sum(r['roi_metadata']['n_pixels'] 
                                           for r in results['roi_results'])
            },
            'analysis_summary': results['summary'],
            'experiment_summary': experiment_summary,
            'configuration': config.raw
        }
        
        with open(summary_file, 'w') as f:
            json.dump(full_summary, f, indent=2, default=str)
        
        # Print summary
        summary = results['summary']
        print(f"\n" + "=" * 80)
        print("EXPERIMENT ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"📊 ROIs analyzed: {summary['successful_rois']}/{summary['total_rois']}")
        print(f"🔬 Scales analyzed: {summary['scales_analyzed']} μm")
        print(f"🧬 Analysis method: {summary['analysis_method'].upper()}")
        
        total_pixels = sum(r['roi_metadata']['n_pixels'] for r in results['roi_results'])
        print(f"📈 Total pixels: {total_pixels:,}")
        
        if 'scale_consistency' in experiment_summary:
            consistency = experiment_summary['scale_consistency']
            print(f"📏 Scale consistency (ARI): {consistency['mean_ari_across_rois']:.3f} ± {consistency['std_ari_across_rois']:.3f}")
        
        if 'clustering_statistics' in experiment_summary:
            clustering = experiment_summary['clustering_statistics']
            print(f"🔗 Avg clusters/scale: {clustering['mean_clusters_per_scale']:.1f}")
        
        print(f"💾 Results saved to: {results_dir}")
        print("=" * 80)
        
        # Print ROI-specific results
        if args.verbose:
            print("\nROI-Specific Results:")
            for roi_result in results['roi_results']:
                consistency = roi_result['consistency_results'].get('overall', {})
                mean_ari = consistency.get('mean_ari', 'N/A')
                print(f"  {roi_result['roi_name']}: ARI={mean_ari}")
        
        print("\n" + "=" * 80)
        print("STUDY LIMITATIONS")
        print("=" * 80)
        print("⚠️  Single experiment analysis - results are descriptive")
        print("⚠️  No statistical significance testing performed")
        print("⚠️  Findings require validation in independent datasets")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Experiment analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    main()