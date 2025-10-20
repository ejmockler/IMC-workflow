"""
Pipeline Integration Validation - KISS Implementation

Validates that all new systems work together:
- Manifest → Profiles → Analysis → Deviation → Provenance → Results
- End-to-end reproducibility testing
- Integration health checks
"""

import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import hashlib
import warnings

# Import core pipeline components
try:
    from .main_pipeline import IMCAnalysisPipeline, run_complete_analysis
    from .analysis_manifest import (
        AnalysisManifest, ParameterProfile, ScientificObjectives, 
        create_manifest_from_config, validate_manifest_compatibility
    )
    from .deviation_workflow import DeviationWorkflow, DeviationType
    from .provenance_tracker import ProvenanceTracker, DecisionType, DecisionSeverity
    from ..config import Config
    from ..validation.framework import ValidationSeverity, ValidationCategory, ValidationResult
except ImportError as e:
    # Fallback for testing
    warnings.warn(f"Import error in integration validation: {e}")
    

class IntegrationValidationResult:
    """Results from integration validation."""
    
    def __init__(self):
        self.all_systems_working = True
        self.component_status: Dict[str, Dict[str, Any]] = {}
        self.reproducibility_validated = False
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.test_results: Dict[str, Any] = {}
        self.execution_time_seconds = 0.0
        
    def add_component_result(self, component: str, status: str, details: Dict[str, Any]):
        """Add result for a component test."""
        self.component_status[component] = {
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        if status == 'FAIL':
            self.all_systems_working = False
            
    def add_error(self, error_msg: str):
        """Add error message."""
        self.errors.append(error_msg)
        self.all_systems_working = False
        
    def add_warning(self, warning_msg: str):
        """Add warning message."""
        self.warnings.append(warning_msg)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'all_systems_working': self.all_systems_working,
            'reproducibility_validated': self.reproducibility_validated,
            'component_status': self.component_status,
            'test_results': self.test_results,
            'errors': self.errors,
            'warnings': self.warnings,
            'execution_time_seconds': self.execution_time_seconds,
            'validation_timestamp': datetime.now().isoformat()
        }


def create_test_manifest(config: Config, test_data_dir: str, temp_dir: str) -> AnalysisManifest:
    """Create test manifest for integration validation."""
    # Create scientific objectives for testing
    objectives = ScientificObjectives(
        primary_research_question="Integration validation test for IMC pipeline systems",
        hypotheses=["All systems integrate correctly", "End-to-end workflow functions"],
        target_cell_types=["test_cell_type"],
        spatial_scales_of_interest=["10um", "20um"],
        success_metrics=["manifest_creation", "parameter_application", "deviation_logging", "provenance_tracking"]
    )
    
    # Create manifest
    manifest = create_manifest_from_config(
        config=config,
        data_directory=test_data_dir,
        profile_name="integration_test_profile",
        scientific_objectives=objectives,
        description="Test manifest for integration validation",
        tissue_type="test_tissue"
    )
    
    # Save manifest
    manifest_path = Path(temp_dir) / f"test_manifest_{manifest.manifest_id}.json"
    manifest.save(manifest_path)
    
    return manifest


def test_manifest_system(config: Config, test_data_dir: str, temp_dir: str) -> Dict[str, Any]:
    """Test manifest creation and compatibility validation."""
    test_result = {
        'manifest_created': False,
        'compatibility_validated': False,
        'dataset_fingerprint_created': False,
        'parameter_profile_extracted': False,
        'errors': []
    }
    
    try:
        # Create manifest
        manifest = create_test_manifest(config, test_data_dir, temp_dir)
        test_result['manifest_created'] = True
        test_result['manifest_id'] = manifest.manifest_id
        
        # Test dataset fingerprint
        if manifest.dataset_fingerprint.total_files > 0:
            test_result['dataset_fingerprint_created'] = True
            test_result['dataset_files'] = manifest.dataset_fingerprint.total_files
        
        # Test parameter profile
        if manifest.parameter_profile:
            test_result['parameter_profile_extracted'] = True
            test_result['profile_name'] = manifest.parameter_profile.name
        
        # Test compatibility validation
        compatibility = validate_manifest_compatibility(manifest, config)
        test_result['compatibility_validated'] = compatibility['compatible']
        test_result['compatibility_warnings'] = compatibility.get('warnings', [])
        test_result['compatibility_errors'] = compatibility.get('errors', [])
        
        return test_result
        
    except Exception as e:
        test_result['errors'].append(f"Manifest system test failed: {str(e)}")
        return test_result


def test_deviation_workflow(config: Config, temp_dir: str) -> Dict[str, Any]:
    """Test parameter deviation workflow."""
    test_result = {
        'technical_deviation_applied': False,
        'scientific_deviation_applied': False,
        'auto_classification_working': False,
        'deviation_logged': False,
        'errors': []
    }
    
    try:
        # Create deviation workflow
        workflow = DeviationWorkflow(log_dir=temp_dir)
        
        # Test technical deviation
        modified_config = workflow.apply_technical_deviation(
            config=config,
            parameter_path="quality_control.thresholds.signal_to_background.min_snr",
            new_value=2.5,
            reason="Test technical parameter change",
            analysis_id="integration_test"
        )
        test_result['technical_deviation_applied'] = True
        
        # Test scientific deviation
        modified_config = workflow.apply_scientific_deviation(
            config=modified_config,
            parameter_path="analysis.clustering.resolution_range",
            new_value=[0.3, 1.5],
            reason="Test scientific parameter change for clustering optimization",
            analysis_id="integration_test"
        )
        test_result['scientific_deviation_applied'] = True
        
        # Test auto-classification
        classification = workflow.auto_classify_deviation("processing.background_correction.clip_negative")
        if classification == DeviationType.TECHNICAL:
            test_result['auto_classification_working'] = True
        
        # Check if deviations were logged
        deviations = workflow.get_deviation_log("integration_test")
        if len(deviations) >= 2:
            test_result['deviation_logged'] = True
            test_result['total_deviations'] = len(deviations)
        
        return test_result
        
    except Exception as e:
        test_result['errors'].append(f"Deviation workflow test failed: {str(e)}")
        return test_result


def test_provenance_tracking(temp_dir: str) -> Dict[str, Any]:
    """Test provenance tracking system."""
    test_result = {
        'tracker_created': False,
        'parameter_decision_logged': False,
        'quality_gate_logged': False,
        'methods_section_generated': False,
        'provenance_saved': False,
        'errors': []
    }
    
    try:
        # Create provenance tracker
        tracker = ProvenanceTracker("integration_test", temp_dir)
        test_result['tracker_created'] = True
        test_result['analysis_id'] = tracker.analysis_id
        
        # Log parameter decision
        decision_id = tracker.log_parameter_decision(
            parameter_name="test_clustering_resolution",
            parameter_value=1.0,
            reasoning="Test parameter for integration validation",
            alternatives_considered=[0.5, 1.5, 2.0],
            evidence={"validation_metric": 0.85},
            severity=DecisionSeverity.IMPORTANT
        )
        test_result['parameter_decision_logged'] = True
        test_result['decision_id'] = decision_id
        
        # Log quality gate
        tracker.log_quality_gate(
            gate_name="test_quality_threshold",
            measured_values={"metric_value": 0.92, "threshold": 0.8},
            outcome="PASS",
            threshold_value=0.8,
            reasoning="Test quality gate for integration validation"
        )
        test_result['quality_gate_logged'] = True
        
        # Generate methods section
        methods_text = tracker.generate_methods_section()
        if "Methods" in methods_text and len(methods_text) > 100:
            test_result['methods_section_generated'] = True
            test_result['methods_length'] = len(methods_text)
        
        # Save provenance record
        provenance_file = tracker.save_provenance_record()
        if Path(provenance_file).exists():
            test_result['provenance_saved'] = True
            test_result['provenance_file'] = provenance_file
        
        return test_result
        
    except Exception as e:
        test_result['errors'].append(f"Provenance tracking test failed: {str(e)}")
        return test_result


def test_pipeline_integration(config: Config, manifest: AnalysisManifest, temp_dir: str) -> Dict[str, Any]:
    """Test main pipeline integration with all systems."""
    test_result = {
        'pipeline_initialized': False,
        'manifest_integration_working': False,
        'execution_steps_logged': False,
        'parameter_deviations_tracked': False,
        'errors': []
    }
    
    try:
        # Initialize pipeline with manifest
        pipeline = IMCAnalysisPipeline(config, manifest)
        test_result['pipeline_initialized'] = True
        
        # Test manifest integration
        if pipeline.analysis_manifest and pipeline.analysis_manifest.manifest_id == manifest.manifest_id:
            test_result['manifest_integration_working'] = True
        
        # Test execution logging
        pipeline._log_execution_step(
            step_name="integration_test_step",
            step_type="validation",
            parameters={"test_param": "test_value"},
            results_summary={"test_result": True}
        )
        
        # Check if execution was logged in manifest
        if len(manifest.execution_history) > 0:
            test_result['execution_steps_logged'] = True
            test_result['logged_steps'] = len(manifest.execution_history)
        
        # Test parameter deviation logging
        pipeline._log_parameter_deviation(
            parameter_path="test.parameter",
            original_value="original",
            new_value="modified",
            reason="Integration test parameter change"
        )
        
        # Check if deviation was logged in manifest
        if len(manifest.deviation_log) > 0:
            test_result['parameter_deviations_tracked'] = True
            test_result['logged_deviations'] = len(manifest.deviation_log)
        
        return test_result
        
    except Exception as e:
        test_result['errors'].append(f"Pipeline integration test failed: {str(e)}")
        return test_result


def create_test_data(test_data_dir: str) -> Dict[str, Any]:
    """Create minimal test data for integration validation."""
    test_data_info = {
        'files_created': 0,
        'total_size_bytes': 0,
        'files': []
    }
    
    test_dir = Path(test_data_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal test ROI files
    for i in range(2):
        roi_file = test_dir / f"test_roi_{i:02d}.txt"
        
        # Create minimal IMC-style data
        test_data = {
            'X': [10 + j for j in range(100)],
            'Y': [20 + j for j in range(100)],
            'DNA1(Ir191)': [100 + j * 2 for j in range(100)],
            'DNA2(Ir193)': [90 + j * 2 for j in range(100)],
            'CD45(Sm154)': [50 + j for j in range(100)],
            'CD31(Yb173)': [30 + j for j in range(100)]
        }
        
        import pandas as pd
        df = pd.DataFrame(test_data)
        df.to_csv(roi_file, sep='\t', index=False)
        
        file_size = roi_file.stat().st_size
        test_data_info['files_created'] += 1
        test_data_info['total_size_bytes'] += file_size
        test_data_info['files'].append(str(roi_file))
    
    return test_data_info


def test_end_to_end_reproducibility(config: Config, test_data_dir: str, temp_dir: str) -> Dict[str, Any]:
    """Test end-to-end reproducibility with result comparison."""
    reproducibility_result = {
        'first_run_completed': False,
        'second_run_completed': False,
        'results_identical': False,
        'hash_comparison': None,
        'errors': []
    }
    
    try:
        # Create two separate output directories
        output_dir_1 = Path(temp_dir) / "run_1"
        output_dir_2 = Path(temp_dir) / "run_2"
        
        # First run
        try:
            results_1 = run_test_analysis(config, test_data_dir, str(output_dir_1))
            reproducibility_result['first_run_completed'] = True
            reproducibility_result['first_run_files'] = len(list(output_dir_1.rglob("*")))
        except Exception as e:
            reproducibility_result['errors'].append(f"First run failed: {str(e)}")
            return reproducibility_result
        
        # Second run with identical parameters
        try:
            results_2 = run_test_analysis(config, test_data_dir, str(output_dir_2))
            reproducibility_result['second_run_completed'] = True
            reproducibility_result['second_run_files'] = len(list(output_dir_2.rglob("*")))
        except Exception as e:
            reproducibility_result['errors'].append(f"Second run failed: {str(e)}")
            return reproducibility_result
        
        # Compare results using hash comparison
        hash_1 = compute_directory_hash(output_dir_1)
        hash_2 = compute_directory_hash(output_dir_2)
        
        reproducibility_result['hash_comparison'] = {
            'run_1_hash': hash_1,
            'run_2_hash': hash_2,
            'identical': hash_1 == hash_2
        }
        
        reproducibility_result['results_identical'] = hash_1 == hash_2
        
        return reproducibility_result
        
    except Exception as e:
        reproducibility_result['errors'].append(f"Reproducibility test failed: {str(e)}")
        return reproducibility_result


def run_test_analysis(config: Config, test_data_dir: str, output_dir: str) -> Dict[str, Any]:
    """Run minimal analysis for reproducibility testing."""
    # Create pipeline
    pipeline = IMCAnalysisPipeline(config)
    
    # Find test ROI files
    roi_files = list(Path(test_data_dir).glob("*.txt"))
    if not roi_files:
        raise ValueError("No test ROI files found")
    
    # Run batch analysis with minimal parameters
    results, errors = pipeline.run_batch_analysis(
        roi_file_paths=[str(f) for f in roi_files],
        protein_names=['CD45', 'CD31'],
        output_dir=output_dir,
        scales_um=[10.0, 20.0],
        analysis_params={'n_clusters': 3, 'use_slic': False},
        generate_plots=False
    )
    
    if errors:
        raise ValueError(f"Analysis completed with errors: {errors}")
    
    return results


def compute_directory_hash(directory: Path) -> str:
    """Compute hash of directory contents for reproducibility comparison."""
    hash_md5 = hashlib.md5()
    
    # Sort files for consistent ordering
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            # Include file path relative to directory
            relative_path = file_path.relative_to(directory)
            hash_md5.update(str(relative_path).encode())
            
            # Include file content
            try:
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
            except Exception:
                # Skip files that can't be read
                hash_md5.update(b"unreadable_file")
    
    return hash_md5.hexdigest()


def run_integration_validation(
    config_path: str,
    test_data_dir: Optional[str] = None,
    output_dir: str = "validation_results",
    run_reproducibility_test: bool = True
) -> Dict[str, Any]:
    """
    Run complete integration validation for IMC pipeline systems.
    
    Tests the complete workflow: Manifest → Profiles → Analysis → Deviation → Provenance → Results
    
    Args:
        config_path: Path to configuration file
        test_data_dir: Directory with test data (creates minimal data if None)
        output_dir: Output directory for validation results
        run_reproducibility_test: Whether to run expensive reproducibility test
        
    Returns:
        Integration validation report
    """
    start_time = datetime.now()
    
    # Initialize result tracker
    result = IntegrationValidationResult()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('IntegrationValidation')
    
    try:
        # Load configuration
        config = Config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test data if not provided
            if test_data_dir is None:
                test_data_dir = str(temp_path / "test_data")
                test_data_info = create_test_data(test_data_dir)
                result.test_results['test_data_created'] = test_data_info
                logger.info(f"Created test data: {test_data_info['files_created']} files")
            
            # Test 1: Manifest System
            logger.info("Testing manifest system...")
            manifest_result = test_manifest_system(config, test_data_dir, temp_dir)
            if manifest_result.get('errors'):
                result.add_component_result('manifest_system', 'FAIL', manifest_result)
                for error in manifest_result['errors']:
                    result.add_error(f"Manifest system: {error}")
            else:
                result.add_component_result('manifest_system', 'PASS', manifest_result)
            
            # Test 2: Deviation Workflow
            logger.info("Testing deviation workflow...")
            deviation_result = test_deviation_workflow(config, temp_dir)
            if deviation_result.get('errors'):
                result.add_component_result('deviation_workflow', 'FAIL', deviation_result)
                for error in deviation_result['errors']:
                    result.add_error(f"Deviation workflow: {error}")
            else:
                result.add_component_result('deviation_workflow', 'PASS', deviation_result)
            
            # Test 3: Provenance Tracking
            logger.info("Testing provenance tracking...")
            provenance_result = test_provenance_tracking(temp_dir)
            if provenance_result.get('errors'):
                result.add_component_result('provenance_tracking', 'FAIL', provenance_result)
                for error in provenance_result['errors']:
                    result.add_error(f"Provenance tracking: {error}")
            else:
                result.add_component_result('provenance_tracking', 'PASS', provenance_result)
            
            # Test 4: Pipeline Integration
            logger.info("Testing pipeline integration...")
            if manifest_result.get('manifest_created'):
                # Reload manifest for integration test
                manifest_id = manifest_result.get('manifest_id')
                manifest_file = temp_path / f"test_manifest_{manifest_id}.json"
                if manifest_file.exists():
                    manifest = AnalysisManifest.load(manifest_file)
                    integration_result = test_pipeline_integration(config, manifest, temp_dir)
                    if integration_result.get('errors'):
                        result.add_component_result('pipeline_integration', 'FAIL', integration_result)
                        for error in integration_result['errors']:
                            result.add_error(f"Pipeline integration: {error}")
                    else:
                        result.add_component_result('pipeline_integration', 'PASS', integration_result)
                else:
                    result.add_error("Manifest file not found for integration test")
                    result.add_component_result('pipeline_integration', 'FAIL', {'error': 'manifest_file_missing'})
            else:
                result.add_warning("Skipping pipeline integration test due to manifest creation failure")
                result.add_component_result('pipeline_integration', 'SKIP', {'reason': 'manifest_creation_failed'})
            
            # Test 5: End-to-End Reproducibility (optional, expensive)
            if run_reproducibility_test and result.all_systems_working:
                logger.info("Testing end-to-end reproducibility...")
                reproducibility_result = test_end_to_end_reproducibility(config, test_data_dir, temp_dir)
                if reproducibility_result.get('errors'):
                    result.add_component_result('reproducibility', 'FAIL', reproducibility_result)
                    for error in reproducibility_result['errors']:
                        result.add_error(f"Reproducibility: {error}")
                else:
                    result.reproducibility_validated = reproducibility_result.get('results_identical', False)
                    result.add_component_result('reproducibility', 'PASS', reproducibility_result)
            else:
                if not result.all_systems_working:
                    result.add_warning("Skipping reproducibility test due to component failures")
                result.add_component_result('reproducibility', 'SKIP', {'reason': 'not_requested_or_component_failures'})
        
        # Calculate execution time
        end_time = datetime.now()
        result.execution_time_seconds = (end_time - start_time).total_seconds()
        
        # Save validation report
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"integration_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Integration validation completed in {result.execution_time_seconds:.1f}s")
        logger.info(f"All systems working: {result.all_systems_working}")
        logger.info(f"Reproducibility validated: {result.reproducibility_validated}")
        logger.info(f"Report saved to: {report_file}")
        
        return result.to_dict()
        
    except Exception as e:
        result.add_error(f"Integration validation failed: {str(e)}")
        return result.to_dict()


def run_integration_health_check(config_path: str) -> Dict[str, Any]:
    """
    Quick integration health check (no test data creation or expensive tests).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Health check report
    """
    health_check = {
        'overall_health': 'UNKNOWN',
        'components_available': {},
        'import_status': {},
        'configuration_valid': False,
        'timestamp': datetime.now().isoformat()
    }
    
    # Test imports
    try:
        from .analysis_manifest import AnalysisManifest
        health_check['import_status']['analysis_manifest'] = 'OK'
        health_check['components_available']['manifest_system'] = True
    except ImportError as e:
        health_check['import_status']['analysis_manifest'] = f'FAIL: {str(e)}'
        health_check['components_available']['manifest_system'] = False
    
    try:
        from .deviation_workflow import DeviationWorkflow
        health_check['import_status']['deviation_workflow'] = 'OK'
        health_check['components_available']['deviation_workflow'] = True
    except ImportError as e:
        health_check['import_status']['deviation_workflow'] = f'FAIL: {str(e)}'
        health_check['components_available']['deviation_workflow'] = False
    
    try:
        from .provenance_tracker import ProvenanceTracker
        health_check['import_status']['provenance_tracker'] = 'OK'
        health_check['components_available']['provenance_tracking'] = True
    except ImportError as e:
        health_check['import_status']['provenance_tracker'] = f'FAIL: {str(e)}'
        health_check['components_available']['provenance_tracking'] = False
    
    try:
        from .main_pipeline import IMCAnalysisPipeline
        health_check['import_status']['main_pipeline'] = 'OK'
        health_check['components_available']['main_pipeline'] = True
    except ImportError as e:
        health_check['import_status']['main_pipeline'] = f'FAIL: {str(e)}'
        health_check['components_available']['main_pipeline'] = False
    
    # Test configuration loading
    try:
        config = Config(config_path)
        health_check['configuration_valid'] = True
        health_check['config_sections'] = {
            'has_analysis': hasattr(config, 'analysis'),
            'has_segmentation': hasattr(config, 'segmentation'),
            'has_channels': hasattr(config, 'channels'),
            'has_storage': hasattr(config, 'storage')
        }
    except Exception as e:
        health_check['configuration_valid'] = False
        health_check['config_error'] = str(e)
    
    # Determine overall health
    if all(health_check['components_available'].values()) and health_check['configuration_valid']:
        health_check['overall_health'] = 'HEALTHY'
    elif any(health_check['components_available'].values()) and health_check['configuration_valid']:
        health_check['overall_health'] = 'DEGRADED'
    else:
        health_check['overall_health'] = 'UNHEALTHY'
    
    return health_check


# Convenience functions for easy integration
def validate_pipeline_integration(config_path: str, output_dir: str = "validation_results") -> bool:
    """
    Simple integration validation that returns True/False.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
        
    Returns:
        True if all systems are working correctly
    """
    result = run_integration_validation(config_path, output_dir=output_dir, run_reproducibility_test=False)
    return result.get('all_systems_working', False)


def quick_health_check(config_path: str) -> str:
    """
    Quick health check that returns status string.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Health status: 'HEALTHY', 'DEGRADED', or 'UNHEALTHY'
    """
    health = run_integration_health_check(config_path)
    return health.get('overall_health', 'UNKNOWN')


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        print("Running integration validation...")
        result = run_integration_validation(config_path)
        print(f"All systems working: {result['all_systems_working']}")
        print(f"Reproducibility validated: {result['reproducibility_validated']}")
    else:
        print("Usage: python integration_validation.py <config_path>")
        print("Example: python integration_validation.py config.json")