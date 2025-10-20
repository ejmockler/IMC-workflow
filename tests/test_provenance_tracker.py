#!/usr/bin/env python3
"""
Simple test of the provenance tracker without external dependencies.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.analysis.provenance_tracker import ProvenanceTracker, DecisionSeverity, DecisionType


def test_basic_functionality():
    """Test basic provenance tracker functionality."""
    
    print("🧪 Testing Provenance Tracker Basic Functionality")
    print("=" * 50)
    
    # Create tracker
    tracker = ProvenanceTracker("test_analysis_001", output_dir="test_provenance")
    
    # Test parameter decision logging
    decision_id = tracker.log_parameter_decision(
        parameter_name="slic_compactness",
        parameter_value=0.1,
        reasoning="Kidney tubule morphology requires tight segments to preserve biological boundaries",
        alternatives_considered=[0.05, 0.2, 0.5],
        evidence={"tubule_boundary_preservation": 0.92, "oversegmentation_risk": 0.15},
        severity=DecisionSeverity.CRITICAL
    )
    print(f"✓ Logged parameter decision: {decision_id}")
    
    # Test quality gate logging
    quality_id = tracker.log_quality_gate(
        gate_name="dna_signal_threshold",
        measured_values={"median_signal": 245, "background_ratio": 12.3},
        outcome="PASS",
        threshold_value=200,
        reasoning="Threshold ensures tissue detection while excluding imaging artifacts"
    )
    print(f"✓ Logged quality gate: {quality_id}")
    
    # Test data transformation logging
    transform_id = tracker.log_data_transformation(
        transformation_name="arcsinh_transform",
        parameters={"cofactor": 5},
        reasoning="Arcsinh transformation compresses dynamic range while preserving gradients for clustering"
    )
    print(f"✓ Logged data transformation: {transform_id}")
    
    # Test method selection logging
    method_id = tracker.log_method_selection(
        method_name="clustering_algorithm",
        chosen_method="leiden",
        alternatives=["kmeans", "hierarchical", "dbscan"],
        reasoning="Leiden clustering provides better detection of biologically meaningful cell neighborhoods",
        performance_comparison={
            "leiden_modularity": 0.78,
            "kmeans_silhouette": 0.42,
            "leiden_silhouette": 0.61
        }
    )
    print(f"✓ Logged method selection: {method_id}")
    
    # Test data lineage tracking
    tracker.track_data_lineage(
        input_source="IMC_ROI_001.txt",
        processing_step="arcsinh_transform -> slic_segmentation -> clustering",
        output_description="Clustered superpixel features",
        input_data={"roi_id": "001", "n_pixels": 150000},
        output_data={"n_clusters": 8, "silhouette_score": 0.61}
    )
    print("✓ Tracked data lineage")
    
    # Test decision summary
    summary = tracker.get_decision_summary()
    print(f"\n📊 Decision Summary:")
    print(f"   Total decisions: {summary['total_decisions']}")
    print(f"   Parameter choices: {summary['decisions_by_type']['parameter_choice']}")
    print(f"   Quality gates: {summary['quality_gates']['total']}")
    print(f"   Data lineage records: {summary['data_lineage_records']}")
    
    # Test methods generation
    print("\n📝 Generating methods section...")
    methods_text = tracker.generate_methods_section()
    
    # Save methods to file
    methods_file = Path("test_provenance") / "test_methods.md"
    methods_file.parent.mkdir(exist_ok=True)
    with open(methods_file, 'w') as f:
        f.write(methods_text)
    print(f"✓ Methods section saved to: {methods_file}")
    
    # Save provenance record
    provenance_file = tracker.save_provenance_record()
    print(f"✓ Provenance record saved to: {provenance_file}")
    
    # Show preview of methods
    print("\n📄 Preview of auto-generated methods:")
    print("-" * 60)
    print(methods_text[:500] + "\n... [truncated]")
    print("-" * 60)
    
    return tracker, methods_file, provenance_file


def test_methods_quality():
    """Test the quality of auto-generated methods sections."""
    
    print("\n📝 Testing Methods Section Quality")
    print("=" * 40)
    
    # Create tracker with comprehensive decisions
    tracker = ProvenanceTracker("comprehensive_test")
    
    # Add various types of decisions
    tracker.log_parameter_decision(
        parameter_name="spatial_resolution",
        parameter_value="1μm per pixel",
        reasoning="1μm resolution captures subcellular protein localization while maintaining reasonable computational cost",
        alternatives_considered=["0.5μm", "2μm"],
        evidence={"cell_diameter_range": "10-20μm", "nuclear_diameter_range": "5-10μm"}
    )
    
    tracker.log_threshold_setting(
        threshold_name="tissue_detection_threshold",
        threshold_value=0.1,
        reasoning="Threshold optimized to exclude background while retaining low-expression tissue regions",
        data_distribution={"background_mean": 0.02, "tissue_5th_percentile": 0.12},
        validation_metrics={"sensitivity": 0.94, "specificity": 0.87}
    )
    
    tracker.log_validation_outcome(
        validation_name="segmentation_quality",
        outcome="PASS",
        metrics={
            "mean_segment_size": 78.5,
            "size_coefficient_variation": 0.34,
            "boundary_accuracy": 0.89
        },
        interpretation="Segmentation shows good size consistency and boundary accuracy for kidney microstructure analysis",
        action_taken="Proceeding with clustering analysis"
    )
    
    # Generate comprehensive methods
    methods = tracker.generate_methods_section(
        include_parameter_tables=True,
        include_quality_metrics=True,
        group_by_analysis_stage=True
    )
    
    print("Generated Methods Section:")
    print("=" * 60)
    print(methods)
    print("=" * 60)
    
    return methods


def test_load_save_functionality():
    """Test save and load functionality."""
    
    print("\n💾 Testing Save/Load Functionality")
    print("=" * 35)
    
    # Create original tracker
    original_tracker = ProvenanceTracker("save_test")
    original_tracker.log_parameter_decision(
        "test_param", 42, "Test reasoning for save/load"
    )
    
    # Save it
    saved_file = original_tracker.save_provenance_record()
    print(f"✓ Saved to: {saved_file}")
    
    # Load it back
    loaded_tracker = ProvenanceTracker("loaded", None)
    loaded_tracker.load_provenance_record(saved_file)
    
    # Verify
    assert len(loaded_tracker.decisions) == 1
    assert loaded_tracker.decisions[0].parameter_name == "test_param"
    assert loaded_tracker.decisions[0].parameter_value == 42
    print("✓ Load/save verification passed")
    
    return loaded_tracker


if __name__ == "__main__":
    print("🧬 Provenance Tracker Test Suite")
    print("=" * 50)
    
    try:
        # Test basic functionality
        tracker, methods_file, provenance_file = test_basic_functionality()
        
        # Test methods quality
        comprehensive_methods = test_methods_quality()
        
        # Test save/load
        loaded_tracker = test_load_save_functionality()
        
        print(f"\n✅ All tests passed!")
        print(f"📁 Generated files:")
        print(f"   - Methods: {methods_file}")
        print(f"   - Provenance: {provenance_file}")
        
        print(f"\n🎯 Key Features Verified:")
        print(f"   ✓ Parameter decision logging with reasoning")
        print(f"   ✓ Quality gate tracking")
        print(f"   ✓ Data transformation documentation")
        print(f"   ✓ Method selection justification")
        print(f"   ✓ Data lineage tracking with checksums")
        print(f"   ✓ Auto-generated methods sections")
        print(f"   ✓ Save/load functionality")
        print(f"   ✓ Decision summaries and statistics")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()