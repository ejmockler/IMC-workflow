#!/usr/bin/env python3
"""
Minimal test to verify deviation workflow works independently.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import copy

# Minimal config mock to avoid full imports
class MinimalConfig:
    def __init__(self, config_path="config.json"):
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.raw = json.load(f)
        else:
            self.raw = {}
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.raw
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update(self, key, value):
        keys = key.split('.')
        config = self.raw
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value


# Import the deviation workflow safely
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'analysis'))
import deviation_workflow


def test_minimal_workflow():
    """Test with minimal dependencies."""
    print("🔧 MINIMAL DEVIATION WORKFLOW TEST")
    print("=" * 50)
    
    # Create minimal config
    config = MinimalConfig("config.json")
    print(f"✓ Config loaded: {len(config.raw)} sections")
    
    # Create deviation workflow
    workflow = deviation_workflow.DeviationWorkflow()
    print(f"✓ Workflow created")
    
    print("\n1. Technical Change Test")
    print("-" * 25)
    
    original = config.get("processing.dna_processing.tissue_threshold")
    print(f"Original: {original}")
    
    # Apply technical change
    modified_config = workflow.apply_technical_deviation(
        config,
        "processing.dna_processing.tissue_threshold",
        0.05,
        "test: poor signal quality",
        analysis_id="minimal_test"
    )
    
    new_value = modified_config.get("processing.dna_processing.tissue_threshold")
    print(f"New value: {new_value}")
    print(f"✓ Technical change: {original} → {new_value}")
    
    print("\n2. Scientific Change Test")
    print("-" * 25)
    
    original_res = config.get("analysis.clustering.resolution_range")
    print(f"Original: {original_res}")
    
    modified_config = workflow.apply_scientific_deviation(
        modified_config,
        "analysis.clustering.resolution_range",
        [0.3, 1.5],
        "test: finer resolution for heterogeneous regions",
        analysis_id="minimal_test"
    )
    
    new_res = modified_config.get("analysis.clustering.resolution_range")
    print(f"New value: {new_res}")
    print(f"⚠ Scientific change: {original_res} → {new_res}")
    
    print("\n3. Auto-classification Test")
    print("-" * 26)
    
    # Test technical parameter
    tech_param = "performance.memory_limit_gb"
    tech_type = workflow.auto_classify_deviation(tech_param)
    print(f"✓ {tech_param} → {tech_type.value}")
    
    # Test scientific parameter
    sci_param = "analysis.clustering.method"
    sci_type = workflow.auto_classify_deviation(sci_param)
    print(f"⚠ {sci_param} → {sci_type.value}")
    
    print("\n4. Emergency Change Test")
    print("-" * 24)
    
    modified_config = workflow.apply_emergency_deviation(
        modified_config,
        "quality_control.thresholds.total_ion_counts.min_tic_percentile",
        1.0,
        "EMERGENCY: data corruption detected, bypassing QC",
        analysis_id="minimal_test"
    )
    print("🚨 Emergency deviation applied")
    
    print("\n5. Deviation Summary")
    print("-" * 20)
    
    summary = workflow.get_deviation_summary("minimal_test")
    print(f"Total deviations: {summary['total_deviations']}")
    print(f"By type: {summary['by_type']}")
    
    # Check log file
    log_file = Path("results/deviations/deviations_minimal_test.json")
    if log_file.exists():
        print(f"✓ Log file: {log_file}")
        with open(log_file, 'r') as f:
            deviations = json.load(f)
        
        print(f"✓ {len(deviations)} deviations logged")
        
        # Show each deviation
        for i, dev in enumerate(deviations, 1):
            dev_type = dev['deviation_type']
            icon = "✓" if dev_type == "technical" else "⚠" if dev_type == "scientific" else "🚨"
            print(f"  {i}. {icon} {dev['parameter_path']} = {dev['new_value']}")
            print(f"     Reason: {dev['reason']}")
    
    return modified_config


def test_convenience_functions():
    """Test convenience functions."""
    print("\n\n⚡ CONVENIENCE FUNCTIONS TEST")
    print("=" * 50)
    
    config = MinimalConfig("config.json")
    
    # Test quick functions
    config = quick_technical_change(
        config,
        "performance.chunk_size", 
        2500,
        "test: reduce chunk size for memory",
        analysis_id="convenience_test"
    )
    print("✓ Quick technical change")
    
    config = quick_scientific_change(
        config,
        "segmentation.scales_um",
        [8.0, 16.0, 32.0],
        "test: custom scales for validation",
        analysis_id="convenience_test"
    )
    print("⚠ Quick scientific change")
    
    # Show results
    print("\nConvenience test results:")
    show_recent_deviations(analysis_id="convenience_test")


if __name__ == "__main__":
    """Run minimal test."""
    
    # Create output directory
    Path("results/deviations").mkdir(parents=True, exist_ok=True)
    
    try:
        test_minimal_workflow()
        test_convenience_functions()
        
        print("\n\n✅ MINIMAL TEST SUCCESSFUL")
        print("=" * 50)
        print("🎯 Deviation workflow is working correctly!")
        print("📁 Check results/deviations/ for JSON logs")
        print("⚡ Zero bureaucracy - changes applied immediately")
        print("📋 Full audit trail maintained automatically")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()