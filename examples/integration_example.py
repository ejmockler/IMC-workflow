#!/usr/bin/env python3
"""
Integration Example: Adding Intelligent Provenance Tracking to IMC Pipeline

Shows how to integrate the ProvenanceTracker into the existing analysis pipeline
to capture scientific decisions and auto-generate methods sections.
"""

import numpy as np
from pathlib import Path
from src.analysis.provenance_tracker import ProvenanceTracker, DecisionSeverity
from src.config import Config


def example_integration_with_pipeline():
    """
    Example showing how to integrate provenance tracking into the IMC pipeline.
    
    This demonstrates capturing real scientific decisions that matter for 
    reproducibility, not just computational logging.
    """
    
    # 1. Initialize provenance tracker for this analysis run
    analysis_id = "kidney_healing_example_001"
    tracker = ProvenanceTracker(analysis_id, output_dir="provenance_records")
    
    print(f"🔬 Starting provenance-tracked analysis: {analysis_id}")
    
    # 2. Load configuration and log key parameter decisions with reasoning
    config = Config("config.json")
    
    # Log SLIC segmentation parameters with biological reasoning
    slic_compactness = config.segmentation.get("slic_params", {}).get("compactness", 10.0)
    tracker.log_parameter_decision(
        parameter_name="slic_compactness",
        parameter_value=slic_compactness,
        reasoning="Kidney tubules require tight segment boundaries (high compactness) to avoid mixing tubular and interstitial signals. Compactness=10 preserves biological structure integrity.",
        alternatives_considered=[5.0, 20.0, 50.0],
        evidence={
            "tubule_boundary_preservation": 0.89,
            "interstitial_separation": 0.92,
            "computational_cost": "acceptable"
        },
        severity=DecisionSeverity.CRITICAL,
        references=["Schapiro et al. Nature Methods 2017", "Jackson et al. Cell Systems 2020"]
    )
    
    # Log arcsinh cofactor choice
    cofactor = config.processing.get("arcsinh_transform", {}).get("percentile_threshold", 5.0)
    tracker.log_parameter_decision(
        parameter_name="arcsinh_cofactor",
        parameter_value=cofactor,
        reasoning="Cofactor=5 provides optimal balance between noise reduction and gradient preservation for IMC protein signals. Maintains linear response in biological range while compressing noise floor.",
        alternatives_considered=[1.0, 3.0, 10.0],
        evidence={
            "signal_to_noise_improvement": 3.2,
            "gradient_preservation": 0.94,
            "clustering_stability": 0.87
        },
        severity=DecisionSeverity.IMPORTANT
    )
    
    # 3. Log method selection decisions
    clustering_method = config.analysis.get("clustering", {}).get("method", "leiden")
    tracker.log_method_selection(
        method_name="clustering_algorithm",
        chosen_method=clustering_method,
        alternatives=["kmeans", "hierarchical", "dbscan"],
        reasoning="Leiden clustering provides better detection of biologically meaningful cell neighborhoods compared to k-means, with superior handling of varying cluster densities in tissue architecture.",
        performance_comparison={
            "leiden_modularity": 0.78,
            "kmeans_silhouette": 0.42,
            "leiden_silhouette": 0.61,
            "biological_interpretability": "high"
        }
    )
    
    # 4. Simulate some analysis and log quality gates
    
    # Example: DNA signal quality check
    simulated_dna_signal = np.random.lognormal(mean=5.5, sigma=0.8, size=1000)
    median_signal = np.median(simulated_dna_signal)
    background_ratio = median_signal / np.percentile(simulated_dna_signal, 5)
    
    dna_threshold = 200
    quality_outcome = "PASS" if median_signal > dna_threshold else "FAIL"
    
    tracker.log_quality_gate(
        gate_name="dna_signal_quality",
        measured_values={
            "median_signal": float(median_signal),
            "background_ratio": float(background_ratio),
            "signal_range": float(np.ptp(simulated_dna_signal))
        },
        outcome=quality_outcome,
        threshold_value=dna_threshold,
        reasoning="DNA signal threshold ensures sufficient nuclear segmentation quality. Median signal >200 indicates adequate tissue preservation and staining quality for reliable SLIC segmentation."
    )
    
    # Example: Segmentation quality assessment
    simulated_segment_sizes = np.random.gamma(shape=2, scale=50, size=500)
    mean_segment_size = np.mean(simulated_segment_sizes)
    size_cv = np.std(simulated_segment_sizes) / mean_segment_size
    
    tracker.log_validation_outcome(
        validation_name="segmentation_quality",
        outcome="PASS" if size_cv < 0.8 else "WARNING",
        metrics={
            "mean_segment_size_um2": float(mean_segment_size),
            "size_coefficient_variation": float(size_cv),
            "oversegmentation_index": 0.12,
            "undersegmentation_index": 0.08
        },
        interpretation=f"Segmentation shows good size consistency (CV={size_cv:.2f}). Segment sizes align with expected kidney microstructure scale (tubules ~50-100 μm²).",
        action_taken="Proceeding with analysis" if size_cv < 0.8 else "Flagged for manual review"
    )
    
    # 5. Log data transformations with scientific context
    tracker.log_data_transformation(
        transformation_name="arcsinh_normalization",
        parameters={
            "cofactor": cofactor,
            "per_channel": True,
            "preserve_zero": True
        },
        reasoning="Arcsinh transformation handles IMC count data heteroscedasticity while preserving biological zero values. Per-channel normalization accounts for varying protein expression levels.",
        input_description="Raw IMC ion counts (Poisson-distributed)",
        output_description="Variance-stabilized protein expression values"
    )
    
    # 6. Track data lineage for reproducibility
    tracker.track_data_lineage(
        input_source="IMC_ROI_kidney_001.txt",
        processing_step="Background correction → Arcsinh transform → SLIC segmentation → Leiden clustering",
        output_description="Spatial protein expression clusters with biological annotations",
        input_data={"roi_id": "kidney_001", "n_pixels": 150000, "n_proteins": 9},
        output_data={"n_clusters": 8, "silhouette_score": 0.61, "modularity": 0.78}
    )
    
    # 7. Generate methods section automatically
    print("\n📝 Generating methods section...")
    methods_text = tracker.generate_methods_section(
        include_parameter_tables=True,
        include_quality_metrics=True,
        group_by_analysis_stage=True
    )
    
    # Save methods to file
    methods_file = Path("provenance_records") / f"methods_{analysis_id}.md"
    methods_file.parent.mkdir(exist_ok=True)
    with open(methods_file, 'w') as f:
        f.write(methods_text)
    
    print(f"📋 Methods section saved to: {methods_file}")
    
    # 8. Save complete provenance record
    provenance_file = tracker.save_provenance_record()
    print(f"🔍 Complete provenance record saved to: {provenance_file}")
    
    # 9. Display decision summary
    summary = tracker.get_decision_summary()
    print(f"\n📊 Decision Summary:")
    print(f"   Total decisions logged: {summary['total_decisions']}")
    print(f"   Parameter choices: {summary['decisions_by_type']['parameter_choice']}")
    print(f"   Method selections: {summary['decisions_by_type']['method_selection']}")
    print(f"   Quality gates: {summary['quality_gates']['total']} ({summary['quality_gates']['passed']} passed)")
    
    # 10. Show preview of auto-generated methods
    print(f"\n📄 Preview of auto-generated methods section:")
    print("="*60)
    print(methods_text[:800] + "\n... [truncated]")
    print("="*60)
    
    return tracker, methods_file, provenance_file


def example_integration_in_existing_pipeline():
    """
    Example showing minimal integration into existing IMCAnalysisPipeline.
    
    Shows key integration points without modifying core pipeline logic.
    """
    
    print("\n🔧 Integration Example: Minimal Changes to Existing Pipeline")
    
    # This would be added to IMCAnalysisPipeline.__init__()
    analysis_id = "kidney_experiment_20250104"
    tracker = ProvenanceTracker(analysis_id)
    
    # Example integrations in key pipeline methods:
    
    # 1. In load_roi_data() - track data input
    roi_file = "data/ROI_001.txt"
    tracker.track_data_lineage(
        input_source=roi_file,
        processing_step="IMC data loading and channel extraction",
        output_description="Structured protein expression data with coordinates"
    )
    
    # 2. In ion_count_pipeline() - log transformation decisions
    tracker.log_data_transformation(
        transformation_name="ion_count_processing",
        parameters={"method": "arcsinh", "cofactor": 5.0},
        reasoning="Ion count aggregation followed by arcsinh transformation for variance stabilization"
    )
    
    # 3. In slic_pipeline() - log segmentation parameters
    tracker.log_parameter_decision(
        parameter_name="slic_n_segments",
        parameter_value=1000,
        reasoning="Target ~1000 segments per ROI to capture kidney microstructure at tubular scale (~20μm)",
        severity=DecisionSeverity.CRITICAL
    )
    
    # 4. In clustering optimization - log method selection
    tracker.log_method_selection(
        method_name="clustering_optimization",
        chosen_method="leiden",
        alternatives=["kmeans", "hierarchical"],
        reasoning="Leiden provides superior biological interpretability for spatial protein neighborhoods"
    )
    
    # 5. In validation - log quality outcomes
    tracker.log_validation_outcome(
        validation_name="clustering_stability",
        outcome="PASS",
        metrics={"silhouette_score": 0.68, "calinski_harabasz": 234.5},
        interpretation="Clustering shows good separation and biological coherence"
    )
    
    # At the end, save provenance
    provenance_file = tracker.save_provenance_record()
    methods_text = tracker.generate_methods_section()
    
    print(f"✅ Integration complete - provenance saved to: {provenance_file}")
    
    return tracker


def demonstrate_methods_generation():
    """Demonstrate the auto-generated methods section quality."""
    
    print("\n📝 Methods Generation Demo")
    
    # Create a tracker with realistic decisions
    tracker = ProvenanceTracker("methods_demo")
    
    # Add several realistic decisions
    tracker.log_parameter_decision(
        parameter_name="spatial_resolution",
        parameter_value="1μm per pixel",
        reasoning="1μm resolution captures subcellular protein localization while maintaining reasonable file sizes for analysis",
        evidence={"cell_diameter_mean": 15.2, "nuclear_diameter_mean": 8.1}
    )
    
    tracker.log_threshold_setting(
        threshold_name="tissue_detection_threshold",
        threshold_value=0.1,
        reasoning="Threshold optimized to exclude background while retaining low-expression tissue regions",
        data_distribution={"background_mean": 0.02, "tissue_5th_percentile": 0.12}
    )
    
    tracker.log_quality_gate(
        gate_name="image_quality_check",
        measured_values={"signal_to_noise": 12.3, "focus_metric": 0.89},
        outcome="PASS",
        reasoning="Image quality sufficient for reliable protein expression analysis"
    )
    
    # Generate and display methods
    methods = tracker.generate_methods_section()
    print("\n" + "="*60)
    print("AUTO-GENERATED METHODS SECTION:")
    print("="*60)
    print(methods)
    print("="*60)


if __name__ == "__main__":
    print("🧬 IMC Provenance Tracking Integration Examples")
    print("=" * 50)
    
    # Run full integration example
    tracker, methods_file, provenance_file = example_integration_with_pipeline()
    
    # Show minimal integration points
    example_integration_in_existing_pipeline()
    
    # Demonstrate methods generation
    demonstrate_methods_generation()
    
    print(f"\n✨ Integration examples complete!")
    print(f"📁 Files generated:")
    print(f"   - Methods: {methods_file}")
    print(f"   - Provenance: {provenance_file}")
    print(f"\n💡 Key Benefits:")
    print(f"   - Captures WHY decisions were made, not just WHAT")
    print(f"   - Auto-generates publication-ready methods sections")
    print(f"   - Provides data lineage for reproducibility")
    print(f"   - Minimal integration burden on existing code")