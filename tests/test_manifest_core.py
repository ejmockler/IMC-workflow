#!/usr/bin/env python3
"""
Core test for AnalysisManifest functionality without dependencies.

Tests the core manifest functionality that doesn't require numpy/pandas.
"""

import tempfile
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Test just the core manifest functionality
def test_basic_manifest():
    """Test basic manifest creation and serialization."""
    print("Testing basic manifest functionality...")
    
    try:
        from analysis.analysis_manifest import (
            AnalysisManifest, ParameterProfile, ScientificObjectives, 
            ManifestVersion, SignatureMethod
        )
        
        # Test manifest creation
        manifest = AnalysisManifest()
        assert manifest.manifest_id is not None
        assert manifest.version == ManifestVersion.V1_0
        print(f"✓ Created manifest: {manifest.manifest_id}")
        
        # Test parameter profile
        profile = ParameterProfile(
            name="test_profile",
            description="Test parameter profile",
            tissue_type="kidney",
            expected_markers=["CD45", "CD31", "CD11b"],
            segmentation_params={"scales_um": [10, 20, 40]},
            clustering_params={"method": "leiden"}
        )
        manifest.set_parameter_profile(profile)
        assert manifest.parameter_profile.name == "test_profile"
        print("✓ Parameter profile set")
        
        # Test scientific objectives
        objectives = ScientificObjectives(
            primary_research_question="How does spatial organization change?",
            hypotheses=["Hypothesis 1", "Hypothesis 2"],
            target_cell_types=["CD45+", "CD31+"]
        )
        manifest.set_scientific_objectives(objectives)
        assert manifest.scientific_objectives.primary_research_question == "How does spatial organization change?"
        print("✓ Scientific objectives set")
        
        # Test serialization
        manifest_dict = manifest.to_dict()
        assert isinstance(manifest_dict, dict)
        assert 'manifest_id' in manifest_dict
        print("✓ Serialization working")
        
        # Test deserialization
        restored_manifest = AnalysisManifest.from_dict(manifest_dict)
        assert restored_manifest.manifest_id == manifest.manifest_id
        assert restored_manifest.parameter_profile.name == "test_profile"
        print("✓ Deserialization working")
        
        # Test file save/load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            manifest_path = f.name
        
        try:
            manifest.save(manifest_path)
            print("✓ Manifest saved to file")
            
            loaded_manifest = AnalysisManifest.load(manifest_path)
            assert loaded_manifest.manifest_id == manifest.manifest_id
            print("✓ Manifest loaded from file")
            
        finally:
            Path(manifest_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_fingerprint():
    """Test dataset fingerprinting."""
    print("\nTesting dataset fingerprinting...")
    
    try:
        from analysis.analysis_manifest import AnalysisManifest, DatasetFingerprint
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock data files
            for i in range(3):
                data_file = temp_path / f"roi_{i}.txt"
                data_file.write_text(f"X\tY\tCD45\n{i}\t{i}\t{i*10}\n")
            
            manifest = AnalysisManifest()
            manifest.set_dataset_fingerprint(temp_path, "*.txt")
            
            assert manifest.dataset_fingerprint.total_files == 3
            assert len(manifest.dataset_fingerprint.files) == 3
            print("✓ Dataset fingerprinted correctly")
            
            # Test hash computation
            overall_hash = manifest.dataset_fingerprint.compute_overall_hash()
            assert len(overall_hash) == 64  # SHA256 hex string
            print("✓ Overall hash computed")
            
            return True
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_execution_logging():
    """Test execution and deviation logging."""
    print("\nTesting execution logging...")
    
    try:
        from analysis.analysis_manifest import AnalysisManifest
        
        manifest = AnalysisManifest()
        
        # Test execution step logging
        manifest.log_execution_step(
            step_name="test_analysis",
            step_type="analysis",
            parameters={"scales_um": [10, 20, 40], "method": "leiden"},
            results_summary={"n_clusters": 8, "silhouette_score": 0.75}
        )
        
        assert len(manifest.execution_history) == 1
        step = manifest.execution_history[0]
        assert step['step_name'] == "test_analysis"
        assert step['step_type'] == "analysis"
        assert step['parameters']['method'] == "leiden"
        print("✓ Execution step logged")
        
        # Test parameter deviation logging
        manifest.log_parameter_deviation(
            parameter_path="clustering.method",
            original_value="leiden",
            new_value="kmeans",
            reason="Better performance for this dataset"
        )
        
        assert len(manifest.deviation_log) == 1
        deviation = manifest.deviation_log[0]
        assert deviation['parameter_path'] == "clustering.method"
        assert deviation['new_value'] == "kmeans"
        print("✓ Parameter deviation logged")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_summary_report():
    """Test summary report generation."""
    print("\nTesting summary report...")
    
    try:
        from analysis.analysis_manifest import (
            AnalysisManifest, ParameterProfile, ScientificObjectives
        )
        
        # Create a complete manifest
        manifest = AnalysisManifest()
        
        profile = ParameterProfile(
            name="test_profile",
            description="Test profile",
            expected_markers=["CD45", "CD31"]
        )
        manifest.set_parameter_profile(profile)
        
        objectives = ScientificObjectives(
            primary_research_question="Test question",
            hypotheses=["Hypothesis 1"]
        )
        manifest.set_scientific_objectives(objectives)
        
        # Mock some fingerprint data
        manifest.dataset_fingerprint.total_files = 5
        manifest.dataset_fingerprint.total_size_bytes = 1024 * 1024  # 1MB
        
        summary = manifest.get_summary_report()
        
        assert 'manifest_id' in summary
        assert 'dataset_summary' in summary
        assert 'parameter_profile' in summary
        assert 'scientific_objectives' in summary
        assert summary['dataset_summary']['total_files'] == 5
        assert summary['parameter_profile']['name'] == "test_profile"
        print("✓ Summary report generated")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run core tests."""
    print("Starting AnalysisManifest core tests...\n")
    
    tests = [
        test_basic_manifest,
        test_dataset_fingerprint,
        test_execution_logging,
        test_summary_report
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All core tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)