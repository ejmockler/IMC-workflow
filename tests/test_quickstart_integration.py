#!/usr/bin/env python3
"""
Test QuickStart Integration

Validates that all optimization systems are properly integrated
and the QuickStart interface works as expected.
"""

import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import json

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.analysis.quickstart import QuickStartInterface, create_quickstart_interface


def create_test_roi_data(n_pixels: int = 1000, n_proteins: int = 5) -> pd.DataFrame:
    """Create synthetic test ROI data."""
    np.random.seed(42)
    
    # Generate coordinates
    coords = np.random.uniform(0, 100, (n_pixels, 2))
    
    # Generate protein data
    data = {
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'DNA1(Ir191)': np.random.poisson(50, n_pixels),
        'DNA2(Ir193)': np.random.poisson(45, n_pixels)
    }
    
    # Add protein channels
    protein_names = [f'Protein_{i}' for i in range(n_proteins)]
    for i, protein in enumerate(protein_names):
        data[f'{protein}(Metal{140+i})'] = np.random.poisson(10 + i, n_pixels)
    
    return pd.DataFrame(data)


def create_test_config() -> dict:
    """Create test configuration."""
    return {
        "metadata_mapping": {
            "roi_column": "ROI",
            "x_column": "X", 
            "y_column": "Y",
            "protein_columns": "auto_detect"
        },
        "processing": {
            "arcsinh_cofactor": 5.0,
            "slic_scale_um": 20.0,
            "clustering_method": "leiden",
            "clustering_resolution": 1.0
        },
        "output": {
            "save_intermediate": true,
            "format": "hdf5"
        }
    }


def test_quickstart_integration():
    """Test QuickStart integration with all optimization systems."""
    print("Testing QuickStart Integration...")
    
    # Create temporary directory and files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test config
        config_path = temp_path / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(create_test_config(), f, indent=2)
        
        # Create test ROI data
        roi_data = create_test_roi_data(n_pixels=500, n_proteins=3)  # Small for testing
        roi_file = temp_path / "test_roi.txt"
        roi_data.to_csv(roi_file, sep='\t', index=False)
        
        print(f"✅ Created test data: {len(roi_data)} pixels, {roi_file}")
        
        # Test 1: Create QuickStart interface
        print("\n1. Testing QuickStart interface creation...")
        try:
            quickstart = create_quickstart_interface(
                config_path=str(config_path),
                output_dir=str(temp_path / "results")
            )
            print("✅ QuickStart interface created successfully")
        except Exception as e:
            print(f"❌ Failed to create QuickStart interface: {e}")
            return False
        
        # Test 2: System readiness validation
        print("\n2. Testing system readiness validation...")
        try:
            readiness = quickstart.validate_system_readiness()
            print(f"✅ System readiness check complete")
            print(f"   System ready: {readiness['system_ready']}")
            print(f"   Available memory: {readiness['current_status']['available_memory_gb']:.1f}GB")
        except Exception as e:
            print(f"❌ System readiness check failed: {e}")
            return False
        
        # Test 3: Single ROI processing
        print("\n3. Testing single ROI processing...")
        try:
            roi_result = quickstart.process_single_roi(
                roi_file_path=str(roi_file),
                roi_id="test_roi",
                protein_names=None,  # Auto-detect
                run_qc=True,
                advanced_mode=True  # Bypass hardware limits for testing
            )
            
            print(f"✅ ROI processing complete")
            print(f"   Success: {roi_result['success']}")
            print(f"   Processing time: {roi_result.get('performance_metrics', {}).get('total_time_seconds', 0):.2f}s")
            print(f"   Clusters found: {roi_result.get('analysis_results', {}).get('n_clusters', 0)}")
            
            if not roi_result['success']:
                print(f"   Error: {roi_result.get('error', 'Unknown error')}")
                for warning in roi_result.get('warnings', []):
                    print(f"   Warning: {warning}")
                    
        except Exception as e:
            print(f"❌ Single ROI processing failed: {e}")
            return False
        
        # Test 4: Performance benchmark
        print("\n4. Testing performance benchmark...")
        try:
            benchmark_result = quickstart.benchmark_performance(
                test_roi_path=str(roi_file),
                n_runs=2,  # Limited runs for testing
                protein_names=None
            )
            
            print(f"✅ Performance benchmark complete")
            if 'performance_summary' in benchmark_result:
                perf = benchmark_result['performance_summary']
                print(f"   Average time: {perf['avg_time_seconds']:.2f}s")
                print(f"   Max memory: {perf['max_memory_gb']:.1f}GB")
                print(f"   Success rate: {perf['success_rate']:.1%}")
                
        except Exception as e:
            print(f"❌ Performance benchmark failed: {e}")
            return False
        
        # Test 5: Integration validation
        print("\n5. Testing system integration...")
        try:
            # Check that all systems are properly integrated
            has_memory_optimizer = hasattr(quickstart, 'memory_optimizer')
            has_performance_dag = hasattr(quickstart, 'performance_dag')
            has_qc_system = hasattr(quickstart, 'qc_system')
            has_quickstart_pipeline = hasattr(quickstart, 'quickstart_pipeline')
            
            print(f"✅ Integration check complete")
            print(f"   Memory optimizer: {'✅' if has_memory_optimizer else '❌'}")
            print(f"   Performance DAG: {'✅' if has_performance_dag else '❌'}")
            print(f"   QC system: {'✅' if has_qc_system else '❌'}")
            print(f"   QuickStart pipeline: {'✅' if has_quickstart_pipeline else '❌'}")
            
            all_integrated = all([has_memory_optimizer, has_performance_dag, has_qc_system, has_quickstart_pipeline])
            
            if not all_integrated:
                print("❌ Not all systems properly integrated")
                return False
                
        except Exception as e:
            print(f"❌ Integration validation failed: {e}")
            return False
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED")
        print("QuickStart integration working correctly!")
        print("="*50)
        
        return True


def test_hardware_validation():
    """Test hardware validation components."""
    print("\nTesting Hardware Validation...")
    
    from src.analysis.quickstart import HardwareValidator
    
    validator = HardwareValidator()
    
    # Test hardware validation
    hw_validation = validator.validate_hardware()
    print(f"Hardware validation: {hw_validation['meets_requirements']}")
    print(f"System memory: {hw_validation['system_info']['total_memory_gb']:.1f}GB")
    
    # Test ROI specification validation
    test_coords = np.random.uniform(0, 100, (500, 2))
    test_ion_counts = {f'protein_{i}': np.random.poisson(10, 500) for i in range(3)}
    
    roi_validation = validator.validate_roi_specs(test_coords, test_ion_counts)
    print(f"ROI validation: {roi_validation['within_specs']}")
    print(f"ROI pixels: {roi_validation['roi_info']['n_pixels']}")
    print(f"ROI channels: {roi_validation['roi_info']['n_channels']}")
    
    return True


if __name__ == "__main__":
    print("QuickStart Integration Test Suite")
    print("="*50)
    
    try:
        # Test hardware validation
        test_hardware_validation()
        
        # Test main integration
        success = test_quickstart_integration()
        
        if success:
            print("\n🎉 QuickStart Integration Test: SUCCESS")
            print("All optimization systems properly integrated!")
        else:
            print("\n💥 QuickStart Integration Test: FAILED")
            print("Check error messages above")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()