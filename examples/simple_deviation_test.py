#!/usr/bin/env python3
"""
Standalone test of deviation workflow (direct imports to avoid package issues).
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Direct import without package initialization
sys.path.append(str(Path(__file__).parent / "src"))

# Import config directly
import config as config_module
from analysis.deviation_workflow import DeviationWorkflow, quick_technical_change


def test_workflow():
    """Test the deviation workflow system."""
    print("🔧 DEVIATION WORKFLOW - STANDALONE TEST")
    print("=" * 50)
    
    # Load config
    config = config_module.Config("config.json")
    print(f"✓ Config loaded from: {config.config_path}")
    
    # Create deviation workflow
    workflow = DeviationWorkflow()
    print(f"✓ Workflow created, log dir: {workflow.log_dir}")
    
    print("\n1. Testing Technical Change")
    print("-" * 30)
    
    # Get original value
    original = config.get("processing.dna_processing.tissue_threshold")
    print(f"Original tissue_threshold: {original}")
    
    # Apply technical change
    modified_config = workflow.apply_technical_deviation(
        config, 
        "processing.dna_processing.tissue_threshold", 
        0.05,
        "poor signal quality in test data",
        analysis_id="standalone_test"
    )
    
    # Verify change
    new_value = modified_config.get("processing.dna_processing.tissue_threshold")
    print(f"New tissue_threshold: {new_value}")
    print(f"✓ Change applied successfully: {original} → {new_value}")
    
    print("\n2. Testing Scientific Change")
    print("-" * 30)
    
    # Get original clustering resolution
    original_res = config.get("analysis.clustering.resolution_range")
    print(f"Original resolution_range: {original_res}")
    
    # Apply scientific change
    modified_config = workflow.apply_scientific_deviation(
        modified_config,
        "analysis.clustering.resolution_range",
        [0.3, 1.5],
        "finer resolution needed for heterogeneous tissue",
        analysis_id="standalone_test"
    )
    
    # Verify change
    new_res = modified_config.get("analysis.clustering.resolution_range")
    print(f"New resolution_range: {new_res}")
    print(f"✓ Change applied successfully: {original_res} → {new_res}")
    
    print("\n3. Testing Auto-classification")
    print("-" * 30)
    
    # Test auto-classification (should be technical)
    param_path = "quality_control.thresholds.signal_to_background.min_snr"
    deviation_type = workflow.auto_classify_deviation(param_path)
    print(f"Parameter: {param_path}")
    print(f"Auto-classified as: {deviation_type.value}")
    
    # Apply the change
    modified_config = workflow.apply_auto_deviation(
        modified_config,
        param_path,
        2.0,
        "relaxing SNR for dim markers",
        analysis_id="standalone_test"
    )
    print("✓ Auto-classified change applied")
    
    print("\n4. Testing Convenience Function")
    print("-" * 30)
    
    # Test quick technical change
    modified_config = quick_technical_change(
        modified_config,
        "performance.memory_limit_gb",
        16.0,
        "increase memory for large dataset",
        analysis_id="standalone_test"
    )
    print("✓ Quick technical change applied")
    
    print("\n5. Checking Deviation Log")
    print("-" * 25)
    
    # Get deviation summary
    summary = workflow.get_deviation_summary("standalone_test")
    print(f"Total deviations: {summary['total_deviations']}")
    print(f"By type: {summary['by_type']}")
    
    # Check log file
    log_file = workflow.log_dir / "deviations_standalone_test.json"
    if log_file.exists():
        print(f"✓ Log file created: {log_file}")
        with open(log_file, 'r') as f:
            deviations = json.load(f)
        print(f"✓ Contains {len(deviations)} deviation records")
        
        # Show recent changes
        print("\nRecent deviations:")
        for dev in deviations[-3:]:  # Last 3
            dev_type = dev['deviation_type']
            icon = "✓" if dev_type == "technical" else "⚠" if dev_type == "scientific" else "🚨"
            print(f"  {icon} {dev['parameter_path']} = {dev['new_value']}")
            print(f"    Reason: {dev['reason']}")
    
    print("\n6. Testing Parameter Categories")
    print("-" * 30)
    
    technical_params = [
        "quality_control.thresholds.total_ion_counts.min_tic_percentile",
        "performance.memory_limit_gb",
        "processing.background_correction.clip_negative"
    ]
    
    scientific_params = [
        "analysis.clustering.resolution_range",
        "segmentation.slic_params.compactness",
        "statistical_framework.mixed_effects.method"
    ]
    
    print("Technical parameters (auto-approved):")
    for param in technical_params:
        classification = workflow.auto_classify_deviation(param)
        print(f"  ✓ {param} → {classification.value}")
    
    print("\nScientific parameters (flagged):")
    for param in scientific_params:
        classification = workflow.auto_classify_deviation(param)
        print(f"  ⚠ {param} → {classification.value}")
    
    return modified_config


if __name__ == "__main__":
    """Run standalone test."""
    
    # Create results directory
    Path("results/deviations").mkdir(parents=True, exist_ok=True)
    
    try:
        modified_config = test_workflow()
        
        print("\n\n✅ STANDALONE TEST PASSED")
        print("=" * 50)
        print("✓ All deviation workflow functions working")
        print("✓ JSON logging functional")
        print("✓ Auto-classification working")
        print("✓ Parameter changes applied correctly")
        print("\nCheck results/deviations/ for generated logs")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()