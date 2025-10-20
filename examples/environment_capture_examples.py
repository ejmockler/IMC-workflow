"""
Environment Capture Integration Examples

Demonstrates how environment capture integrates with:
1. Analysis manifest system
2. Provenance tracking 
3. Reproducibility framework
4. Pipeline validation

KISS PRINCIPLE: Simple, practical examples for real usage.
"""

import json
from pathlib import Path
from datetime import datetime

# Example 1: Basic Environment Capture
def example_basic_capture():
    """Basic environment capture and snapshot."""
    print("=== Basic Environment Capture ===")
    
    # Import environment capture
    from src.analysis.environment_capture import (
        capture_execution_environment,
        save_environment_snapshot,
        EnvironmentCapture
    )
    
    # Capture current environment
    env_info = capture_execution_environment(analysis_id="example_001")
    
    print(f"Environment Hash: {env_info.fingerprint_hash}")
    print(f"OS: {env_info.system_info.os_name}")
    print(f"Python: {env_info.system_info.python_version.split()[0]}")
    print(f"BLAS Backend: {env_info.computational_env.blas_backend}")
    
    # Save snapshot
    output_dir = Path("output/environment_snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_path = save_environment_snapshot(
        output_dir / f"env_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        env_info
    )
    print(f"Snapshot saved: {snapshot_path}")
    
    # Generate human-readable report
    capture = EnvironmentCapture(analysis_id="example_001")
    report = capture.generate_environment_report(env_info)
    
    with open(output_dir / "environment_report.md", 'w') as f:
        f.write(report)
    
    print("Environment report saved to: output/environment_snapshots/environment_report.md")


# Example 2: Integration with Analysis Manifest
def example_manifest_integration():
    """Show how environment capture integrates with analysis manifest."""
    print("\n=== Analysis Manifest Integration ===")
    
    from src.analysis.environment_capture import capture_execution_environment
    from src.analysis.analysis_manifest import AnalysisManifest
    
    # Create analysis manifest
    manifest = AnalysisManifest(
        analysis_id="kidney_analysis_001",
        output_dir="output/kidney_analysis"
    )
    
    # Capture environment for this analysis
    env_info = capture_execution_environment(analysis_id="kidney_analysis_001")
    
    # Add environment info to manifest
    manifest.add_environment_info({
        "environment_hash": env_info.fingerprint_hash,
        "capture_timestamp": env_info.capture_timestamp,
        "system_summary": {
            "os": f"{env_info.system_info.os_name} {env_info.system_info.os_release}",
            "python": env_info.system_info.python_version.split()[0],
            "architecture": env_info.system_info.architecture,
            "blas_backend": env_info.computational_env.blas_backend
        },
        "dependency_summary": {
            "numpy": env_info.dependency_versions.numpy_version,
            "scipy": env_info.dependency_versions.scipy_version,
            "sklearn": env_info.dependency_versions.sklearn_version
        }
    })
    
    # Save environment snapshot alongside manifest
    env_snapshot_path = manifest.output_dir / f"environment_{env_info.fingerprint_hash}.json"
    from src.analysis.environment_capture import save_environment_snapshot
    save_environment_snapshot(env_snapshot_path, env_info)
    
    print(f"Environment integrated into manifest: {manifest.analysis_id}")
    print(f"Environment hash: {env_info.fingerprint_hash}")


# Example 3: Integration with Provenance Tracker
def example_provenance_integration():
    """Show how environment capture integrates with provenance tracking."""
    print("\n=== Provenance Tracker Integration ===")
    
    from src.analysis.environment_capture import EnvironmentCapture
    from src.analysis.provenance_tracker import ProvenanceTracker
    
    # Create provenance tracker
    tracker = ProvenanceTracker(
        analysis_id="reproducibility_test_001",
        output_dir="output/provenance"
    )
    
    # Create environment capture
    env_capture = EnvironmentCapture(analysis_id="reproducibility_test_001")
    env_info = env_capture.capture_execution_environment()
    
    # Log environment capture as a critical decision
    tracker.log_parameter_decision(
        parameter_name="execution_environment",
        parameter_value=env_info.fingerprint_hash,
        reasoning="Environment fingerprint captured for reproducibility validation",
        evidence={
            "system_os": f"{env_info.system_info.os_name} {env_info.system_info.os_release}",
            "python_version": env_info.system_info.python_version.split()[0],
            "blas_backend": env_info.computational_env.blas_backend or "unknown",
            "deterministic_threading": all([
                env_info.computational_env.omp_num_threads == '1',
                env_info.computational_env.mkl_num_threads == '1'
            ]),
            "numpy_seeded": env_info.computational_env.numpy_random_seed_set
        },
        severity=tracker.__class__.__dict__.get('DecisionSeverity', 
                                               type('DS', (), {'CRITICAL': 'critical'})).CRITICAL
    )
    
    # Log any environment issues as quality gates
    if env_info.computational_env.omp_num_threads != '1':
        tracker.log_quality_gate(
            gate_name="deterministic_threading",
            measured_values={"omp_num_threads": env_info.computational_env.omp_num_threads or "unset"},
            outcome="WARNING",
            reasoning="Threading not set to deterministic mode - may affect reproducibility"
        )
    
    # Save provenance record with environment info
    provenance_file = tracker.save_provenance_record()
    
    # Save detailed environment snapshot alongside
    env_snapshot_path = Path(provenance_file).parent / f"environment_{env_info.fingerprint_hash}.json"
    env_capture.save_environment_snapshot(env_snapshot_path, env_info)
    
    print(f"Provenance record saved: {provenance_file}")
    print(f"Environment snapshot saved: {env_snapshot_path}")


# Example 4: Environment Compatibility Validation
def example_environment_compatibility():
    """Show how to validate environment compatibility between runs."""
    print("\n=== Environment Compatibility Validation ===")
    
    from src.analysis.environment_capture import (
        capture_execution_environment,
        validate_environment_compatibility,
        save_environment_snapshot
    )
    
    # Simulate two different analysis runs
    env1 = capture_execution_environment(analysis_id="run_001")
    env2 = capture_execution_environment(analysis_id="run_002")
    
    # Save both environments
    output_dir = Path("output/compatibility_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env1_path = save_environment_snapshot(output_dir / "env_run_001.json", env1)
    env2_path = save_environment_snapshot(output_dir / "env_run_002.json", env2)
    
    # Validate compatibility
    compatibility = validate_environment_compatibility(env1, env2, strict=False)
    
    print(f"Environment compatibility: {'COMPATIBLE' if compatibility['is_compatible'] else 'INCOMPATIBLE'}")
    print(f"Compatibility score: {compatibility['compatibility_score']:.2f}")
    
    if compatibility['incompatibilities']:
        print("Incompatibilities found:")
        for issue in compatibility['incompatibilities']:
            print(f"  - {issue}")
    
    if compatibility['recommendations']:
        print("Recommendations:")
        for rec in compatibility['recommendations']:
            print(f"  - {rec}")


# Example 5: Integration with Reproducibility Framework  
def example_reproducibility_integration():
    """Show how environment capture works with reproducibility framework."""
    print("\n=== Reproducibility Framework Integration ===")
    
    from src.analysis.environment_capture import EnvironmentCapture
    from src.analysis.reproducibility_framework import ReproducibilityFramework
    
    # Create both frameworks
    env_capture = EnvironmentCapture(analysis_id="repro_test_001")
    repro_framework = ReproducibilityFramework(seed=42)
    
    # Capture environment before setting deterministic mode
    env_before = env_capture.capture_execution_environment()
    print(f"Environment before deterministic setup:")
    print(f"  OMP threads: {env_before.computational_env.omp_num_threads or 'unset'}")
    print(f"  Random seed set: {env_before.computational_env.numpy_random_seed_set}")
    
    # Set deterministic environment
    repro_framework.ensure_deterministic_env()
    
    # Capture environment after deterministic setup
    env_after = env_capture.capture_execution_environment(force_refresh=True)
    print(f"Environment after deterministic setup:")
    print(f"  OMP threads: {env_after.computational_env.omp_num_threads or 'unset'}")
    print(f"  Random seed set: {env_after.computational_env.numpy_random_seed_set}")
    
    # Compare environments
    compatibility = validate_environment_compatibility(env_before, env_after)
    print(f"Environment change compatibility: {compatibility['compatibility_score']:.2f}")
    
    # Generate comprehensive reproducibility report
    output_dir = Path("output/reproducibility_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save both environment snapshots
    env_capture.save_environment_snapshot(
        output_dir / "env_before_deterministic.json", 
        env_before
    )
    env_capture.save_environment_snapshot(
        output_dir / "env_after_deterministic.json", 
        env_after
    )
    
    # Generate reproducibility report that includes environment info
    repro_report = repro_framework.generate_reproducibility_report(
        output_dir / "reproducibility_report.json"
    )
    
    # Add environment comparison to report
    extended_report = {
        **repro_report,
        "environment_comparison": {
            "before_deterministic": env_before.fingerprint_hash,
            "after_deterministic": env_after.fingerprint_hash,
            "compatibility_score": compatibility['compatibility_score'],
            "deterministic_improvements": [
                f"OMP threads: {env_before.computational_env.omp_num_threads or 'unset'} -> {env_after.computational_env.omp_num_threads or 'unset'}",
                f"Random seed: {env_before.computational_env.numpy_random_seed_set} -> {env_after.computational_env.numpy_random_seed_set}"
            ]
        }
    }
    
    with open(output_dir / "extended_reproducibility_report.json", 'w') as f:
        json.dump(extended_report, f, indent=2, default=str)
    
    print(f"Extended reproducibility report saved: {output_dir / 'extended_reproducibility_report.json'}")
    
    # Restore original environment
    repro_framework.restore_environment()


# Example 6: Pipeline Integration Pattern
def example_pipeline_integration():
    """Show recommended pattern for integrating environment capture in analysis pipeline."""
    print("\n=== Pipeline Integration Pattern ===")
    
    from src.analysis.environment_capture import EnvironmentCapture
    
    def run_analysis_with_environment_tracking(analysis_id: str, config: dict):
        """Example analysis function with environment tracking."""
        
        # Step 1: Capture environment at start
        env_capture = EnvironmentCapture(analysis_id=analysis_id)
        env_info = env_capture.capture_execution_environment()
        
        print(f"Starting analysis: {analysis_id}")
        print(f"Environment hash: {env_info.fingerprint_hash}")
        
        # Step 2: Set up output directory
        output_dir = Path(f"output/{analysis_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 3: Save environment snapshot immediately
        env_snapshot_path = env_capture.save_environment_snapshot(
            output_dir / "execution_environment.json",
            env_info
        )
        
        # Step 4: Add environment info to analysis metadata
        analysis_metadata = {
            "analysis_id": analysis_id,
            "start_time": datetime.now().isoformat(),
            "environment": {
                "fingerprint_hash": env_info.fingerprint_hash,
                "snapshot_path": str(env_snapshot_path),
                "system_summary": f"{env_info.system_info.os_name} {env_info.system_info.python_version.split()[0]}",
                "reproducibility_ready": all([
                    env_info.computational_env.omp_num_threads == '1',
                    env_info.computational_env.mkl_num_threads == '1',
                    env_info.computational_env.numpy_random_seed_set
                ])
            },
            "config": config
        }
        
        # Step 5: Save analysis metadata
        with open(output_dir / "analysis_metadata.json", 'w') as f:
            json.dump(analysis_metadata, f, indent=2, default=str)
        
        # Step 6: Run actual analysis (simulated)
        print("Running analysis...")
        analysis_results = {
            "status": "completed",
            "environment_hash": env_info.fingerprint_hash,
            "results": {"dummy": "results"}
        }
        
        # Step 7: Save results with environment linkage
        with open(output_dir / "analysis_results.json", 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"Analysis completed. Environment tracked in: {env_snapshot_path}")
        return analysis_results
    
    # Run example analysis
    config = {"method": "slic", "scales": [10, 20, 40]}
    results = run_analysis_with_environment_tracking("demo_pipeline_001", config)
    
    print(f"Analysis results: {results['status']}")


if __name__ == "__main__":
    print("IMC Environment Capture - Integration Examples")
    print("=" * 60)
    
    # Run all examples
    try:
        example_basic_capture()
        example_manifest_integration()
        example_provenance_integration()
        example_environment_compatibility()
        example_reproducibility_integration()
        example_pipeline_integration()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("Check the 'output/' directory for generated files.")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()