#!/usr/bin/env python3
"""
Deviation Workflow Integration Example and Demo

Shows how to integrate the lightweight deviation workflow
with the existing IMC analysis pipeline.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config
from src.analysis.deviation_workflow import (
    DeviationWorkflow, 
    quick_technical_change, 
    quick_scientific_change,
    show_recent_deviations
)
from src.analysis.main_pipeline import IMCAnalysisPipeline


def demo_basic_usage():
    """Demo basic deviation workflow usage."""
    print("🔧 DEVIATION WORKFLOW DEMO")
    print("=" * 50)
    
    # Load config
    config = Config("config.json")
    
    # Create deviation workflow
    workflow = DeviationWorkflow()
    
    print("\n1. Technical Changes (Auto-approved)")
    print("-" * 30)
    
    # Technical change - lower DNA threshold due to poor signal quality
    modified_config = workflow.apply_technical_deviation(
        config, 
        "processing.dna_processing.tissue_threshold", 
        0.05,  # Was 0.1
        "poor signal quality in ROIs 1-3",
        analysis_id="kidney_batch_001"
    )
    
    # Another technical change - increase memory limit
    modified_config = workflow.apply_technical_deviation(
        modified_config,
        "performance.memory_limit_gb",
        12.0,  # Was 8.0
        "large ROI files exceed 8GB limit",
        analysis_id="kidney_batch_001"
    )
    
    print("\n2. Scientific Changes (Flagged but applied)")
    print("-" * 40)
    
    # Scientific change - adjust clustering resolution for tumor heterogeneity
    modified_config = workflow.apply_scientific_deviation(
        modified_config,
        "analysis.clustering.resolution_range",
        [0.3, 1.5],  # Was [0.5, 2.0]
        "tumor heterogeneity requires finer clustering resolution",
        analysis_id="kidney_batch_001"
    )
    
    # Scientific change - change segmentation scale
    modified_config = workflow.apply_scientific_deviation(
        modified_config,
        "segmentation.scales_um",
        [8.0, 15.0, 30.0],  # Was [10.0, 20.0, 40.0]
        "smaller scales needed for capillary network analysis",
        analysis_id="kidney_batch_001"
    )
    
    print("\n3. Auto-classification")
    print("-" * 20)
    
    # Let the system auto-classify parameter type
    modified_config = workflow.apply_auto_deviation(
        modified_config,
        "quality_control.thresholds.signal_to_background.min_snr",
        2.0,  # Was 3.0
        "relaxing SNR for dim markers",
        analysis_id="kidney_batch_001"
    )
    
    print("\n4. Review Changes")
    print("-" * 15)
    
    # Show what happened
    show_recent_deviations(analysis_id="kidney_batch_001")
    
    return modified_config


def demo_pipeline_integration():
    """Demo integration with the main analysis pipeline."""
    print("\n\n🔄 PIPELINE INTEGRATION DEMO")
    print("=" * 50)
    
    # Start with base config
    config = Config("config.json")
    
    # Apply quick fixes during analysis
    print("\n1. Quick Technical Fix During Analysis")
    print("-" * 40)
    
    # Discover memory issue during batch processing
    config = quick_technical_change(
        config,
        "performance.chunk_size",
        2500,  # Was 5000
        "reduce chunk size due to memory pressure"
    )
    
    print("\n2. Scientific Adjustment Based on Initial Results")
    print("-" * 50)
    
    # Based on initial clustering results, adjust parameters
    config = quick_scientific_change(
        config,
        "analysis.clustering.optimization_method",
        "modularity",  # Was "stability"
        "modularity optimization better for sparse immune infiltration"
    )
    
    print("\n3. Initialize Pipeline with Modified Config")
    print("-" * 45)
    
    # Create pipeline with modified config
    pipeline = IMCAnalysisPipeline(config)
    print("✓ Pipeline initialized with deviation-modified config")
    
    # The pipeline will use the modified parameters
    print(f"✓ Chunk size: {config.performance.get('chunk_size', 'default')}")
    print(f"✓ Clustering method: {config.analysis.get('clustering', {}).get('optimization_method', 'default')}")
    
    # Show final deviation summary
    print("\n4. Final Deviation Summary")
    print("-" * 25)
    show_recent_deviations()


def demo_emergency_workflow():
    """Demo emergency deviation workflow."""
    print("\n\n🚨 EMERGENCY WORKFLOW DEMO")
    print("=" * 50)
    
    config = Config("config.json")
    workflow = DeviationWorkflow()
    
    print("\nEmergency: Critical data corruption detected!")
    print("Need to bypass normal QC to salvage analysis")
    
    # Emergency bypass of QC
    config = workflow.apply_emergency_deviation(
        config,
        "quality_control.thresholds.total_ion_counts.min_tic_percentile",
        1.0,  # Was 10
        "EMERGENCY: Data corruption in 30% of pixels, bypassing TIC QC to salvage remaining data",
        analysis_id="emergency_recovery_001"
    )
    
    # Emergency relaxation of segmentation QC
    config = workflow.apply_emergency_deviation(
        config,
        "quality_control.thresholds.segmentation_quality.min_tissue_coverage_percent",
        5.0,  # Was 10
        "EMERGENCY: Poor tissue coverage due to acquisition failure, minimum viable analysis",
        analysis_id="emergency_recovery_001"
    )
    
    print("\n🚨 Emergency deviations logged and applied")
    show_recent_deviations(analysis_id="emergency_recovery_001")


def demo_convenience_functions():
    """Demo convenience functions for common workflows."""
    print("\n\n⚡ CONVENIENCE FUNCTIONS DEMO")
    print("=" * 50)
    
    config = Config("config.json")
    
    print("\n1. Mass Technical Adjustments")
    print("-" * 30)
    
    # Quick adjustments for poor data quality batch
    technical_adjustments = [
        ("processing.dna_processing.tissue_threshold", 0.05, "weak DNA signal"),
        ("quality_control.thresholds.signal_to_background.min_snr", 2.0, "low SNR batch"),
        ("performance.memory_limit_gb", 16.0, "large file batch"),
        ("processing.arcsinh_transform.percentile_threshold", 3.0, "high background noise")
    ]
    
    for param_path, new_value, reason in technical_adjustments:
        config = quick_technical_change(
            config, param_path, new_value, reason,
            analysis_id="poor_quality_batch"
        )
    
    print("\n2. Scientific Parameter Sweep")
    print("-" * 30)
    
    # Scientific adjustments for specific research question
    scientific_adjustments = [
        ("segmentation.scales_um", [5.0, 12.0, 25.0], "fine-scale vascular analysis"),
        ("analysis.clustering.resolution_range", [0.8, 3.0], "high-resolution immune phenotyping"),
        ("analysis.clustering.use_coabundance_features", False, "marker-independent clustering for discovery")
    ]
    
    for param_path, new_value, reason in scientific_adjustments:
        config = quick_scientific_change(
            config, param_path, new_value, reason,
            analysis_id="vascular_immune_study"
        )
    
    print("\n3. Review All Changes")
    print("-" * 20)
    
    print("\nPoor quality batch adjustments:")
    show_recent_deviations(analysis_id="poor_quality_batch")
    
    print("\nVascular immune study adjustments:")
    show_recent_deviations(analysis_id="vascular_immune_study")


if __name__ == "__main__":
    """Run complete deviation workflow demo."""
    
    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    
    # Run all demos
    modified_config = demo_basic_usage()
    demo_pipeline_integration()
    demo_emergency_workflow()
    demo_convenience_functions()
    
    print("\n\n✅ DEMO COMPLETE")
    print("=" * 50)
    print("Check results/deviations/ for JSON logs")
    print("All parameter changes logged with timestamps and rationale")
    print("Zero bureaucracy - changes applied immediately while maintaining audit trail")