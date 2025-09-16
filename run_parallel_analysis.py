#!/usr/bin/env python3
"""
Production Parallel IMC Analysis Runner

Processes multiple ROIs in parallel using the production IMC pipeline with
batch correction, multi-scale analysis, and enhanced validation.
"""

import argparse
import logging
from pathlib import Path
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append('.')

from src.config import Config
from src.analysis.multiscale_analysis import perform_multiscale_analysis
from src.analysis.batch_correction import sham_anchored_normalize


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('parallel_analysis.log'),
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
            'filename': roi_file.name
        }
        
    except Exception as e:
        raise ValueError(f"Failed to load ROI data from {roi_file}: {e}")


def analyze_single_roi_parallel(args) -> Dict[str, Any]:
    """Analyze single ROI - designed for parallel execution."""
    roi_file, config_dict = args
    
    try:
        # Load ROI data
        roi_data = load_roi_data(roi_file)
        
        # Get analysis parameters
        multiscale_config = config_dict.get('multiscale_analysis', {})
        scales_um = multiscale_config.get('scales_um', [10.0, 20.0, 40.0])
        
        ion_config = config_dict.get('ion_count_processing', {})
        n_clusters = ion_config.get('n_clusters', 8)
        use_slic = ion_config.get('use_slic_segmentation', True)
        
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
        
        return {
            'roi_file': str(roi_file),
            'success': True,
            'multiscale_results': multiscale_results,
            'roi_metadata': {
                'n_pixels': roi_data['n_pixels'],
                'n_proteins': len(roi_data['ion_counts']),
                'filename': roi_data['filename']
            }
        }
        
    except Exception as e:
        return {
            'roi_file': str(roi_file),
            'success': False,
            'error': str(e)
        }


def run_parallel_analysis(roi_files: List[Path], config: Config, n_processes: int = None) -> Dict[str, Any]:
    """Run parallel analysis across multiple ROI files."""
    logger = logging.getLogger('ParallelAnalysis')
    
    # Determine number of processes
    if n_processes is None:
        import os
        n_processes = min(len(roi_files), os.cpu_count())
    
    logger.info(f"Starting parallel analysis with {n_processes} processes")
    
    start_time = time.time()
    results = []
    
    # Prepare arguments for parallel execution
    config_dict = config.raw  # Pass serializable config
    args_list = [(roi_file, config_dict) for roi_file in roi_files]
    
    # Execute in parallel
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        future_to_roi = {
            executor.submit(analyze_single_roi_parallel, args): args[0] 
            for args in args_list
        }
        
        for future in as_completed(future_to_roi):
            roi_file = future_to_roi[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    logger.info(f"✅ Completed {roi_file.name}")
                else:
                    logger.warning(f"❌ Failed {roi_file.name}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"❌ Exception for {roi_file.name}: {e}")
                results.append({
                    'roi_file': str(roi_file),
                    'success': False,
                    'error': str(e)
                })
    
    analysis_time = time.time() - start_time
    
    # Compile summary statistics
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    summary = {
        'total_rois': len(roi_files),
        'successful_rois': len(successful_results),
        'failed_rois': len(failed_results),
        'analysis_time_seconds': analysis_time,
        'rois_per_second': len(successful_results) / analysis_time,
        'n_processes_used': n_processes
    }
    
    return {
        'results': results,
        'successful_results': successful_results,
        'failed_results': failed_results,
        'summary': summary
    }


def main():
    parser = argparse.ArgumentParser(description='Run parallel IMC analysis with batch correction')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data/241218_IMC_Alun',
                       help='Directory containing ROI files')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of processes to use (default: auto-detect)')
    parser.add_argument('--batch-correction', action='store_true', default=True,
                       help='Apply batch correction (default: True)')
    parser.add_argument('--roi-pattern', type=str, default='*.txt',
                       help='Pattern to match ROI files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory from config')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger('ParallelAnalysis')
    
    print("=" * 80)
    print("PRODUCTION PARALLEL IMC ANALYSIS")
    print("=" * 80)
    print("• Multi-scale analysis (10μm, 20μm, 40μm)")
    print("• Batch correction with sham-anchored normalization")
    print("• Parallel processing for enhanced performance")
    print("=" * 80)
    
    try:
        # Load configuration
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Override output directory if specified
        if args.output_dir:
            config.raw['output']['results_dir'] = args.output_dir
            logger.info(f"Output directory overridden to: {args.output_dir}")
        
        # Find ROI files
        data_dir = Path(args.data_dir)
        roi_files = list(data_dir.glob(args.roi_pattern))
        logger.info(f"Found {len(roi_files)} ROI files in {data_dir}")
        
        if not roi_files:
            logger.error("No ROI files found to process")
            return 1
        
        # Run parallel analysis
        logger.info("Starting parallel ROI analysis...")
        parallel_results = run_parallel_analysis(roi_files, config, args.processes)
        
        # Apply batch correction if requested
        if args.batch_correction and parallel_results['successful_results']:
            logger.info("Applying batch correction...")
            
            # Organize data by batches for correction
            batch_data = {}
            batch_metadata = {}
            
            for result in parallel_results['successful_results']:
                # Determine batch from filename
                filename = result['roi_metadata']['filename']
                batch_id = _determine_batch_from_filename(filename)
                
                if batch_id not in batch_data:
                    batch_data[batch_id] = {}
                    batch_metadata[batch_id] = []
                
                # Extract ion counts from multiscale results (use finest scale)
                multiscale_res = result['multiscale_results']
                finest_scale = min(multiscale_res.keys())
                scale_result = multiscale_res[finest_scale]
                
                if 'superpixel_counts' in scale_result:
                    roi_ion_counts = scale_result['superpixel_counts']
                    roi_name = Path(result['roi_file']).stem
                    
                    # Aggregate for batch correction
                    for protein, counts in roi_ion_counts.items():
                        if protein not in batch_data[batch_id]:
                            batch_data[batch_id][protein] = []
                        if hasattr(counts, 'flatten'):
                            batch_data[batch_id][protein].append(counts.flatten())
                        else:
                            batch_data[batch_id][protein].append(np.array(counts))
            
            # Convert lists to concatenated arrays
            for batch_id in batch_data:
                for protein in batch_data[batch_id]:
                    batch_data[batch_id][protein] = np.concatenate(batch_data[batch_id][protein])
                batch_metadata[batch_id] = {'roi_count': len(batch_metadata[batch_id])}
            
            # Apply sham-anchored normalization
            if len(batch_data) > 1:
                try:
                    corrected_data, correction_stats = sham_anchored_normalize(
                        batch_data, batch_metadata,
                        sham_condition='Sham', sham_timepoint=0
                    )
                    parallel_results['batch_correction'] = correction_stats
                    logger.info("✅ Batch correction completed")
                except Exception as e:
                    logger.warning(f"⚠️ Batch correction failed: {e}")
                    parallel_results['batch_correction'] = {'error': str(e)}
            else:
                logger.info("Only one batch detected - skipping batch correction")
        
        # Save results
        output_config = config.raw.get('output', {})
        results_dir = Path(output_config.get('results_dir', 'results/parallel_analysis'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / 'parallel_analysis_results.json'
        
        # Create serializable results
        serializable_results = {
            'summary': parallel_results['summary'],
            'batch_correction': parallel_results.get('batch_correction', {}),
            'successful_rois': [r['roi_file'] for r in parallel_results['successful_results']],
            'failed_rois': [{'roi_file': r['roi_file'], 'error': r['error']} 
                           for r in parallel_results['failed_results']],
            'analysis_parameters': config.raw
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Print summary
        summary = parallel_results['summary']
        print(f"\n" + "=" * 80)
        print("PARALLEL ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"📊 ROIs processed: {summary['successful_rois']}/{summary['total_rois']}")
        print(f"⏱️  Analysis time: {summary['analysis_time_seconds']:.1f}s")
        print(f"🚀 Processing rate: {summary['rois_per_second']:.2f} ROIs/sec")
        print(f"💻 Processes used: {summary['n_processes_used']}")
        
        if args.batch_correction and 'batch_correction' in parallel_results:
            bc_stats = parallel_results['batch_correction']
            if 'improvement_metrics' in bc_stats:
                improvement = bc_stats['improvement_metrics']['improvement_ratio']
                print(f"🔧 Batch correction: {improvement:.1%} improvement")
        
        print(f"💾 Results saved to: {results_file}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Parallel analysis failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


def _determine_batch_from_filename(filename: str) -> str:
    """Extract batch ID from filename."""
    # Example: IMC_241218_Alun_ROI_D1_M1_01_9.txt -> D1_M1
    parts = filename.split('_')
    if len(parts) >= 6:
        timepoint = parts[4]  # D1, D3, etc.
        mouse = parts[5]      # M1, M2, etc.
        return f"{timepoint}_{mouse}"
    return "batch_unknown"


if __name__ == "__main__":
    sys.exit(main())