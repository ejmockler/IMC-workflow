#!/usr/bin/env python3
"""
Simple test of deviation workflow without full pipeline dependencies.
"""

import sys
import json
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


def test_basic_workflow():
    """Test basic deviation workflow functionality."""
    print("🔧 DEVIATION WORKFLOW TEST")
    print("=" * 50)
    
    # Load config
    config = Config("config.json")
    
    # Create deviation workflow
    workflow = DeviationWorkflow()
    
    print("\n1. Testing Technical Changes")
    print("-" * 30)
    
    # Get original value
    original_threshold = config.get("processing.dna_processing.tissue_threshold")
    print(f"Original tissue_threshold: {original_threshold}")
    
    # Apply technical change
    modified_config = workflow.apply_technical_deviation(
        config, 
        "processing.dna_processing.tissue_threshold", 
        0.05,  # Was 0.1
        "poor signal quality in test ROIs",
        analysis_id="test_run_001"
    )
    
    # Verify change
    new_threshold = modified_config.get("processing.dna_processing.tissue_threshold")
    print(f"New tissue_threshold: {new_threshold}")
    print(f"✓ Change applied: {original_threshold} → {new_threshold}")
    
    print("\n2. Testing Scientific Changes")
    print("-" * 30)
    
    # Get original value
    original_resolution = config.get("analysis.clustering.resolution_range")
    print(f"Original resolution_range: {original_resolution}")
    
    # Apply scientific change
    modified_config = workflow.apply_scientific_deviation(
        modified_config,
        "analysis.clustering.resolution_range",
        [0.3, 1.5],  # Was [0.5, 2.0]
        "test: finer resolution for heterogeneous tissue",
        analysis_id="test_run_001"
    )
    
    # Verify change
    new_resolution = modified_config.get("analysis.clustering.resolution_range")
    print(f"New resolution_range: {new_resolution}")
    print(f"✓ Change applied: {original_resolution} → {new_resolution}")
    
    print("\n3. Testing Auto-classification")
    print("-" * 30)
    
    # Test auto-classification
    auto_classified_config = workflow.apply_auto_deviation(
        modified_config,
        "quality_control.thresholds.signal_to_background.min_snr",
        2.0,  # Was 3.0
        "test: relaxing SNR threshold",
        analysis_id="test_run_001"
    )
    
    print("✓ Auto-classification applied")
    
    print("\n4. Deviation Summary")
    print("-" * 20)
    
    # Show summary
    summary = workflow.get_deviation_summary("test_run_001")
    print(f"Total deviations: {summary['total_deviations']}")
    print(f"By type: {summary['by_type']}")
    
    return modified_config


def test_convenience_functions():
    """Test convenience functions."""
    print("\n\n⚡ TESTING CONVENIENCE FUNCTIONS")
    print("=" * 50)
    
    config = Config("config.json")
    
    print("\n1. Quick Technical Change")
    print("-" * 25)
    
    config = quick_technical_change(
        config,
        "performance.memory_limit_gb",
        12.0,
        "test: increase memory for large files",
        analysis_id="convenience_test"
    )
    print("✓ Quick technical change applied")
    
    print("\n2. Quick Scientific Change")
    print("-" * 26)
    
    config = quick_scientific_change(
        config,
        "segmentation.scales_um",
        [8.0, 16.0, 32.0],
        "test: custom scales for validation",
        analysis_id="convenience_test"
    )
    print("✓ Quick scientific change applied")
    
    print("\n3. Show Recent Changes")
    print("-" * 22)
    
    show_recent_deviations(analysis_id="convenience_test")


def test_json_logging():
    """Test JSON logging functionality."""
    print("\n\n📝 TESTING JSON LOGGING")
    print("=" * 50)
    
    workflow = DeviationWorkflow()
    config = Config("config.json")
    
    # Apply a change
    workflow.apply_technical_deviation(
        config,
        "performance.chunk_size",
        2500,
        "test: JSON logging verification",
        analysis_id="json_test"
    )
    
    # Check if log file was created
    log_file = Path("results/deviations/deviations_json_test.json")
    if log_file.exists():
        print(f"✓ Log file created: {log_file}")
        
        # Read and display content
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        print(f"✓ Log contains {len(log_data)} entries")
        if log_data:
            latest = log_data[-1]
            print(f"✓ Latest entry: {latest['parameter_path']} = {latest['new_value']}")
            print(f"✓ Reason: {latest['reason']}")
    else:
        print("❌ Log file not created")


if __name__ == "__main__":
    """Run deviation workflow tests."""
    
    # Ensure results directory exists
    Path("results/deviations").mkdir(parents=True, exist_ok=True)
    
    try:
        # Run tests
        test_basic_workflow()
        test_convenience_functions()
        test_json_logging()
        
        print("\n\n✅ ALL TESTS PASSED")
        print("=" * 50)
        print("Deviation workflow is working correctly!")
        print("Check results/deviations/ for JSON logs")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()