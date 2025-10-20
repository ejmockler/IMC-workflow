#!/usr/bin/env python3
"""
Example: Integrating Reproducibility Framework with IMC Analysis

This example shows how to integrate the reproducibility framework
into existing IMC analysis workflows for validation and debugging.

Usage:
    python example_reproducibility_integration.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from analysis.reproducibility_framework import (
        ReproducibilityFramework,
        run_reproducibility_test
    )
    from config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("This example requires the full IMC analysis environment")
    sys.exit(1)


def create_reproducibility_config():
    """Create configuration for reproducibility testing."""
    return {
        # Analysis parameters that should be deterministic
        'analysis': {
            'clustering': {
                'method': 'leiden',
                'resolution_range': [0.5, 2.0],
                'random_state': 42  # CRITICAL for reproducibility
            },
            'ion_count_processing': {
                'cofactor': 5.0,
                'transformation': 'arcsinh'
            },
            'segmentation': {
                'compactness': 10,
                'n_segments': 400,
                'random_state': 42  # CRITICAL for reproducibility
            }
        },
        
        # Quality control thresholds
        'quality_control': {
            'min_ion_counts': 100,
            'max_zero_fraction': 0.8,
            'spatial_coherence_threshold': 0.3
        },
        
        # Performance settings - important for reproducibility
        'performance': {
            'n_jobs': 1,  # Force single-threaded for determinism
            'parallel_processes': 1,
            'enable_parallel': False
        },
        
        # Reproducibility settings
        'reproducibility': {
            'tolerance': {
                'rtol': 1e-10,
                'atol': 1e-12
            },
            'environment': {
                'single_threaded_blas': True,
                'deterministic_algorithms': True
            }
        }
    }


def example_analysis_function(data, config):
    """
    Example analysis function that should be reproducible.
    
    This is a simplified version of what a real IMC analysis function
    might look like.
    """
    import numpy as np
    
    # Extract data
    coords = data['coords']
    ion_counts = data['ion_counts']
    
    # Set random seed from config (CRITICAL)
    random_state = config['analysis']['clustering']['random_state']
    np.random.seed(random_state)
    
    # Step 1: Ion count processing
    cofactor = config['analysis']['ion_count_processing']['cofactor']
    processed_counts = {}
    
    for protein, counts in ion_counts.items():
        # Arcsinh transformation
        processed_counts[protein] = np.arcsinh(counts / cofactor)
    
    # Step 2: Feature matrix construction
    feature_names = list(processed_counts.keys())
    features = np.column_stack([processed_counts[name] for name in feature_names])
    
    # Step 3: Simple clustering (deterministic with fixed seed)
    from sklearn.cluster import KMeans
    n_clusters = 5
    clusterer = KMeans(
        n_clusters=n_clusters, 
        random_state=random_state,
        n_init=10  # Deterministic initialization
    )
    labels = clusterer.fit_predict(features)
    
    # Step 4: Compute cluster statistics
    cluster_stats = {}
    for i in range(n_clusters):
        mask = labels == i
        if np.sum(mask) > 0:
            cluster_coords = coords[mask]
            cluster_features = features[mask]
            
            cluster_stats[f'cluster_{i}'] = {
                'size': int(np.sum(mask)),
                'centroid': np.mean(cluster_coords, axis=0),
                'mean_expression': np.mean(cluster_features, axis=0),
                'std_expression': np.std(cluster_features, axis=0)
            }
    
    # Step 5: Spatial statistics  
    from scipy.spatial.distance import pdist
    distances = pdist(coords)
    
    spatial_stats = {
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'density': len(coords) / (np.ptp(coords[:, 0]) * np.ptp(coords[:, 1]))
    }
    
    return {
        'labels': labels,
        'features': features,
        'cluster_stats': cluster_stats,
        'spatial_stats': spatial_stats,
        'feature_names': feature_names,
        'processing_metadata': {
            'cofactor': cofactor,
            'n_clusters': n_clusters,
            'random_state': random_state
        }
    }


def create_test_data():
    """Create synthetic test data for reproducibility testing."""
    import numpy as np
    
    np.random.seed(42)  # Fixed seed for consistent test data
    
    n_points = 1000
    coords = np.random.uniform(0, 100, (n_points, 2))
    
    # Create protein expression data
    protein_names = ['CD45', 'CD31', 'CD3', 'CD68', 'Ki67']
    ion_counts = {}
    
    for protein in protein_names:
        # Create realistic Poisson-distributed ion counts
        base_expression = np.random.poisson(50, n_points)
        
        # Add some spatial structure
        for center_x, center_y in [(25, 25), (75, 75), (25, 75)]:
            distances = np.sqrt((coords[:, 0] - center_x)**2 + (coords[:, 1] - center_y)**2)
            boost = np.exp(-distances / 15) * np.random.poisson(30, n_points)
            base_expression += boost.astype(int)
        
        ion_counts[protein] = base_expression.astype(float)
    
    return {
        'coords': coords,
        'ion_counts': ion_counts,
        'metadata': {
            'roi_id': 'test_roi_001',
            'n_points': n_points,
            'proteins': protein_names
        }
    }


def example_pipeline_validation():
    """Example of validating pipeline reproducibility."""
    print("IMC Pipeline Reproducibility Validation")
    print("=" * 50)
    
    # Create test configuration
    config = create_reproducibility_config()
    
    # Create test data
    test_data = create_test_data()
    print(f"Created test data: {test_data['metadata']['n_points']} points")
    print(f"Proteins: {', '.join(test_data['metadata']['proteins'])}")
    
    # Test reproducibility
    print("\nTesting pipeline reproducibility...")
    
    result = run_reproducibility_test(
        analysis_func=example_analysis_function,
        data=test_data,
        config=config,
        n_runs=3,  # Test 3 runs
        seed=42,
        rtol=1e-10
    )
    
    print(f"Reproducibility: {'PASSED' if result.is_reproducible else 'FAILED'}")
    print(f"Max difference: {result.max_difference:.2e}")
    print(f"Tolerance: {result.tolerance_used:.2e}")
    
    if result.failed_keys:
        print(f"Failed keys: {result.failed_keys[:3]}...")  # Show first 3
    
    # Generate detailed report
    framework = ReproducibilityFramework(seed=42)
    framework.validation_results = [result]
    
    report_path = Path("reproducibility_validation_report.json")
    report = framework.generate_reproducibility_report(report_path)
    
    print(f"\nDetailed report saved to: {report_path}")
    print("\nEnvironment information:")
    env = result.environment_fingerprint
    print(f"  Platform: {env.platform_system}")
    print(f"  Python: {env.python_version.split()[0]}")
    print(f"  NumPy: {env.numpy_version}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    return result.is_reproducible


def example_environment_debugging():
    """Example of using framework for environment debugging."""
    print("\nEnvironment Debugging Example")
    print("=" * 40)
    
    framework = ReproducibilityFramework()
    
    # Capture current environment
    env_before = framework.capture_environment()
    print("Environment before setup:")
    print(f"  OMP_NUM_THREADS: {env_before.omp_num_threads}")
    print(f"  MKL_NUM_THREADS: {env_before.mkl_num_threads}")
    
    # Setup deterministic environment
    framework.ensure_deterministic_env()
    
    env_after = framework.capture_environment()
    print("\nEnvironment after setup:")
    print(f"  OMP_NUM_THREADS: {env_after.omp_num_threads}")
    print(f"  MKL_NUM_THREADS: {env_after.mkl_num_threads}")
    
    # Compare environment hashes
    print(f"\nEnvironment hash before: {env_before.to_hash()[:8]}...")
    print(f"Environment hash after:  {env_after.to_hash()[:8]}...")
    
    # Restore environment
    framework.restore_environment()
    
    env_restored = framework.capture_environment() 
    print(f"Environment hash restored: {env_restored.to_hash()[:8]}...")
    
    print("\n✅ Environment debugging complete")


def main():
    """Main function demonstrating reproducibility integration."""
    
    try:
        # Check if we have required packages
        import numpy as np
        import scipy
        import sklearn
        
        print("Required packages available:")
        print(f"  NumPy: {np.__version__}")
        print(f"  SciPy: {scipy.__version__}")
        print(f"  Scikit-learn: {sklearn.__version__}")
        
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install: pip install numpy scipy scikit-learn")
        return 1
    
    # Run pipeline validation
    success = example_pipeline_validation()
    
    # Run environment debugging example
    example_environment_debugging()
    
    print(f"\nOverall result: {'SUCCESS' if success else 'FAILED'}")
    
    print("\nIntegration Guidelines:")
    print("1. Add 'random_state' parameters to all stochastic algorithms")
    print("2. Use ReproducibilityFramework in your test suite")
    print("3. Set deterministic environment in production pipelines")
    print("4. Validate reproducibility after any algorithm changes")
    print("5. Include environment fingerprints in result metadata")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)