#!/usr/bin/env python3
"""
Robust IMC Analysis Runner

This script runs the complete IMC analysis pipeline, including critical
pre-analysis validation to ensure data integrity and quality.
The pipeline will NOT run if validation fails.
"""

import sys
import json
from pathlib import Path
import logging
import pandas as pd
import numpy as np

from src.config import Config
from typing import List

# Core pipeline imports
from src.analysis.main_pipeline import run_complete_analysis

# Practical validation framework imports  
from src.validation.practical_pipeline import create_practical_pipeline


def main():
    """Run IMC analysis with integrated high-performance validation."""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load configuration to get correct data directory
    config = Config('config.json')
    data_dir = config.data_dir
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Create robust configuration
    config_data = {
        "data": {
            "raw_data_dir": str(data_dir)
        },
        "channels": {
            "protein_channels": ["CD45", "CD11b", "CD31", "CD140a", "CD140b", "CD206"],
            "dna_channels": ["DNA1", "DNA2"]
        },
        "analysis": {
            "clustering": {
                "method": "leiden",
                "resolution_range": [0.5, 1.5]
            },
            "multiscale": {
                "enable": True,
                "scales_um": [10, 20, 40]
            }
        },
        "output": {
            "results_dir": "results",
            "save_intermediate": True
        },
        "validation": {
            "enable": True,
            "fail_on_critical": True,
            "quality_gates": True
        }
    }
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Save config file
    config_path = output_dir / "analysis_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    logger.info(f"Configuration saved to: {config_path}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Find ROI files
    roi_files = list(data_dir.glob("*.txt"))
    if not roi_files:
        logger.error(f"No .txt files found in data directory: {data_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(roi_files)} ROI files")
    
    # --- CRITICAL: Pre-analysis validation with practical pipeline ---
    logger.info("=== Starting Pre-Analysis Validation (Practical Approach) ===")
    
    practical_pipeline = create_practical_pipeline()
    validation_report = practical_pipeline.validate_dataset(roi_files)
    
    can_proceed = validation_report.get('can_proceed', False)
    critical_failures = validation_report.get('critical_failures', 0)
    total_issues = validation_report.get('total_issues', 0)
    usable_rois = validation_report.get('usable_rois', 0)
    
    if not can_proceed:
        logger.error(f"VALIDATION FAILED: {critical_failures} critical failures detected")
        logger.error("Fix critical issues before proceeding with analysis")
        # Save detailed report
        report_path = output_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        logger.error(f"Detailed validation report saved to {report_path}")
        sys.exit(1)
    
    logger.info(f"✓ VALIDATION PASSED: {usable_rois}/{validation_report['total_rois']} ROIs can be analyzed")
    if total_issues > 0:
        logger.warning(f"Note: {total_issues} quality issues detected - see validation log for details")
    
    # Save validation report for reference
    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    logger.info(f"Validation report saved to {report_path}")
    
    # --- Core analysis pipeline ---
    logger.info("=== Starting Core Analysis Pipeline ===")
    
    try:
        # Run complete analysis with validation disabled (already performed by HP pipeline)
        summary = run_complete_analysis(
            config_path=str(config_path),
            roi_directory=str(data_dir),
            output_directory=str(output_dir),
            run_validation=False  # Validation already performed by HP pipeline
        )
        
        logger.info("Analysis completed successfully!")
        logger.info(f"Summary: {summary}")
        
        # Save summary
        summary_path = output_dir / "run_summary.json" 
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Run summary saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()