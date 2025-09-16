#!/usr/bin/env python3
"""
Parallel ROI Analysis Runner

Processes multiple ROIs in parallel with optional parameter sensitivity analysis.
Designed for efficient batch processing across experimental conditions.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('.')

from src.core.parallel import BatchAnalysisRunner, ParallelROIProcessor
from src.utils.helpers import find_roi_files
from src.config import Config


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


def main():
    parser = argparse.ArgumentParser(description='Run parallel ROI analysis')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing ROI files')
    parser.add_argument('--n-processes', type=int, default=None,
                       help='Number of processes to use (default: auto-detect)')
    parser.add_argument('--parameter-sensitivity', action='store_true',
                       help='Run parameter sensitivity analysis')
    parser.add_argument('--group-by-condition', action='store_true', default=True,
                       help='Group results by experimental condition')
    parser.add_argument('--roi-files', nargs='*', type=str,
                       help='Specific ROI files to process (default: find all in data-dir)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger('ParallelAnalysis')
    
    try:
        # Load configuration
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Find ROI files
        if args.roi_files:
            roi_files = [Path(f) for f in args.roi_files]
            logger.info(f"Processing specified ROI files: {len(roi_files)} files")
        else:
            roi_files = find_roi_files(Path(args.data_dir))
            logger.info(f"Found {len(roi_files)} ROI files in {args.data_dir}")
        
        if not roi_files:
            logger.error("No ROI files found to process")
            return 1
        
        # Create batch analysis runner
        runner = BatchAnalysisRunner(args.config)
        
        # Configure parallel processor
        if args.n_processes:
            runner.processor.n_processes = args.n_processes
            logger.info(f"Using {args.n_processes} processes")
        
        # Run analysis
        logger.info("Starting batch analysis...")
        results = runner.run_full_study(
            roi_files,
            parameter_sensitivity=args.parameter_sensitivity,
            group_by_condition=args.group_by_condition
        )
        
        # Report summary
        summary = results.get('summary_statistics', {})
        logger.info("Analysis complete!")
        logger.info(f"Successfully processed: {summary.get('successful_rois', 0)} ROIs")
        logger.info(f"Failed analyses: {summary.get('failed_rois', 0)} ROIs")
        logger.info(f"Conditions found: {summary.get('conditions_found', [])}")
        
        if args.parameter_sensitivity:
            logger.info(f"Parameter combinations tested: {summary.get('n_parameter_combinations', 0)}")
        
        logger.info(f"Results saved to {config.output_dir}/batch_analysis_results.json")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())