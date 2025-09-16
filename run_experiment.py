#!/usr/bin/env python3
"""
Generalized experiment execution script for IMC analysis.

This script auto-detects experiment types and generates comprehensive
visualizations using the modular experiment framework.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.analysis.roi import BatchAnalyzer
from src.experiments import ExperimentFramework, KidneyHealingExperiment


def detect_experiment_type(config: Config) -> str:
    """Auto-detect experiment type from configuration and data.
    
    Args:
        config: Configuration object
        
    Returns:
        String identifier for experiment type
    """
    # Check for kidney healing markers
    proteins = config.raw.get('proteins', {})
    functional_groups = proteins.get('functional_groups', {})
    
    kidney_markers = {'kidney_inflammation', 'kidney_repair', 'kidney_vasculature'}
    if kidney_markers.issubset(set(functional_groups.keys())):
        return 'kidney_healing'
    
    # Check for timepoint-based injury study
    experimental = config.raw.get('experimental', {})
    timepoints = experimental.get('timepoints', [])
    if 0 in timepoints and len(timepoints) > 2:  # Sham + multiple injury timepoints
        return 'injury_recovery'
    
    # Default to generic temporal analysis
    return 'temporal_analysis'


def create_experiment_framework(experiment_type: str, config: Config) -> ExperimentFramework:
    """Create appropriate experiment framework based on type.
    
    Args:
        experiment_type: Type of experiment detected
        config: Configuration object
        
    Returns:
        Concrete experiment framework instance
    """
    if experiment_type in ['kidney_healing', 'injury_recovery']:
        return KidneyHealingExperiment(config)
    else:
        # Default to kidney healing framework for now
        # Future: implement generic temporal framework
        print(f"Warning: Experiment type '{experiment_type}' not fully supported. Using kidney healing framework.")
        return KidneyHealingExperiment(config)


def run_analysis_pipeline(config: Config, force_reanalysis: bool = False) -> list:
    """Run the analysis pipeline or load existing results.
    
    Args:
        config: Configuration object
        force_reanalysis: Force re-analysis even if results exist
        
    Returns:
        List of ROI analysis results
    """
    results_file = config.output_dir / 'analysis_results.json'
    
    if results_file.exists() and not force_reanalysis:
        print(f"Loading existing analysis results from {results_file}")
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} ROI analysis results")
        return results
    else:
        print("Running complete analysis pipeline...")
        batch_analyzer = BatchAnalyzer(config)
        results = batch_analyzer.analyze_all()
        print(f"Analysis complete. Generated {len(results)} ROI results")
        return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generalized IMC experiment analysis and visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py                          # Auto-detect experiment, use default config
  python run_experiment.py --config custom.json    # Use custom configuration
  python run_experiment.py --force-analysis        # Force re-analysis even if results exist
  python run_experiment.py --publication           # Generate publication-quality figures (600 DPI)
  python run_experiment.py --output-dir results/   # Specify custom output directory
        """
    )
    
    parser.add_argument('--config', default='config.json',
                       help='Configuration file path (default: config.json)')
    parser.add_argument('--force-analysis', action='store_true',
                       help='Force re-analysis even if results exist')
    parser.add_argument('--publication', action='store_true',
                       help='Generate publication-quality figures (600 DPI)')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory (overrides config)')
    parser.add_argument('--experiment-type', type=str,
                       choices=['kidney_healing', 'injury_recovery', 'temporal_analysis'],
                       help='Force specific experiment type (overrides auto-detection)')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                       help='Output format for figures (default: png)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"Error loading configuration from {args.config}: {e}")
        sys.exit(1)
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
        config.output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set DPI based on publication mode
    dpi = 600 if args.publication else 300
    
    print('=' * 80)
    print('GENERALIZED IMC EXPERIMENT ANALYSIS')
    print('=' * 80)
    print(f'Configuration: {args.config}')
    print(f'Output directory: {config.output_dir}')
    print(f'Publication mode: {"Enabled" if args.publication else "Disabled"}')
    print(f'DPI: {dpi}')
    print(f'Format: {args.format.upper()}')
    print('=' * 80)
    
    # Detect or override experiment type
    if args.experiment_type:
        experiment_type = args.experiment_type
        print(f'Experiment type: {experiment_type} (forced)')
    else:
        experiment_type = detect_experiment_type(config)
        print(f'Experiment type: {experiment_type} (auto-detected)')
    
    # Create experiment framework
    try:
        experiment = create_experiment_framework(experiment_type, config)
        print(f'Created experiment framework: {experiment.__class__.__name__}')
    except Exception as e:
        print(f"Error creating experiment framework: {e}")
        sys.exit(1)
    
    # Run analysis pipeline
    try:
        results = run_analysis_pipeline(config, args.force_analysis)
        if not results:
            print("No analysis results available. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"Error running analysis pipeline: {e}")
        sys.exit(1)
    
    # Validate experimental design
    print("\nValidating experimental design...")
    if not experiment.validate_experimental_design(results):
        print("Warning: Experimental design validation failed. Proceeding with available data.")
    else:
        print("✓ Experimental design validation passed")
    
    # Generate experiment-specific report
    print(f"\nGenerating comprehensive experiment report...")
    try:
        output_paths = experiment.generate_experiment_report(
            results, 
            output_dir=str(config.output_dir)
        )
        
        print("\n" + "=" * 80)
        print("EXPERIMENT REPORT GENERATION COMPLETE")
        print("=" * 80)
        print(f"Generated {len(output_paths)} outputs:")
        
        for output_name, output_path in output_paths.items():
            print(f"  ✓ {output_name}: {output_path}")
        
        print(f"\nAll outputs saved to: {config.output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error generating experiment report: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()