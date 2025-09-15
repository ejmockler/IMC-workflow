#!/usr/bin/env python3
"""
Test clustering on actual ROI data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.config import Config
from src.utils.helpers import find_roi_files
from src.analysis.spatial import load_roi_data
from src.analysis.clustering import ClustererFactory, benchmark_clusterers
import json


def test_clustering_on_roi():
    """Test different clustering algorithms on real ROI data"""
    
    print("Testing Clustering on Real ROI Data")
    print("=" * 50)
    
    # Load config and find ROI
    config = Config('config.json')
    roi_files = find_roi_files(config.data_dir)
    
    if not roi_files:
        print("No ROI files found!")
        return
    
    # Use first ROI
    test_roi = roi_files[0]
    print(f"Testing on: {test_roi.name}")
    
    # Load data
    coords, values, protein_names = load_roi_data(test_roi, 'config.json')
    print(f"Data shape: {values.shape}")
    print(f"Proteins: {protein_names[:5]}...")
    print()
    
    # Test different algorithms
    algorithms_to_test = ['kmeans', 'minibatch', 'flowsom', 'leiden']
    n_clusters = 30  # Typical for our protein panel
    
    print("Testing Algorithms:")
    print("-" * 50)
    
    for algo_name in algorithms_to_test:
        print(f"\n{algo_name.upper()}:")
        
        try:
            # Create clusterer
            clusterer = ClustererFactory.create(algo_name)
            
            # Test on subset for speed
            subset_size = min(5000, len(values))
            subset_values = values[:subset_size]
            
            # Cluster
            if algo_name in ['leiden']:
                result = clusterer.fit_predict(subset_values)
            else:
                result = clusterer.fit_predict(subset_values, n_clusters=n_clusters)
            
            print(f"  Clusters found: {result.n_clusters}")
            
            # Cluster size distribution
            import numpy as np
            unique, counts = np.unique(result.labels, return_counts=True)
            print(f"  Min cluster size: {counts.min()}")
            print(f"  Max cluster size: {counts.max()}")
            print(f"  Mean cluster size: {counts.mean():.1f}")
            
            # Validate
            from src.analysis.validation import SilhouetteValidator
            validator = SilhouetteValidator()
            val_result = validator.validate(subset_values, result.labels)
            print(f"  Silhouette score: {val_result.score:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 50)
    print("Benchmarking on ROI subset:")
    
    # Benchmark on smaller subset
    benchmark_size = min(2000, len(values))
    benchmark_values = values[:benchmark_size]
    
    results = benchmark_clusterers(
        benchmark_values, 
        ['kmeans', 'minibatch', 'flowsom'],
        n_clusters=20
    )
    
    print(f"\n{'Algorithm':<15} {'Time (s)':<10} {'Quality':<10}")
    print("-" * 35)
    for algo, metrics in results.items():
        if 'error' not in metrics:
            print(f"{algo:<15} {metrics['time_seconds']:<10.3f} {metrics['silhouette_score']:<10.3f}")


def test_config_switching():
    """Test switching clustering algorithms via config"""
    
    print("\n\nTesting Config-Based Algorithm Switching")
    print("=" * 50)
    
    # Save original config
    with open('config.json', 'r') as f:
        original_config = json.load(f)
    
    # Test different algorithms
    test_algorithms = ['kmeans', 'flowsom', 'minibatch']
    
    for algo in test_algorithms:
        print(f"\nSetting algorithm to: {algo}")
        
        # Update config
        test_config = original_config.copy()
        test_config['clustering']['algorithm'] = algo
        
        # Save temporary config
        with open('temp_test_config.json', 'w') as f:
            json.dump(test_config, f)
        
        # Test identify_expression_blobs with this config
        from src.analysis.spatial import identify_expression_blobs
        
        config = Config('config.json')
        roi_files = find_roi_files(config.data_dir)
        
        if roi_files:
            coords, values, protein_names = load_roi_data(roi_files[0], 'config.json')
            
            # Use small subset
            subset_size = min(1000, len(values))
            
            result = identify_expression_blobs(
                coords[:subset_size],
                values[:subset_size],
                protein_names,
                config_path='temp_test_config.json',
                validate=False
            )
            
            blob_labels = result[0]
            n_blobs = len(set(blob_labels))
            print(f"  ✓ Created {n_blobs} blobs using {algo}")
    
    # Cleanup
    Path('temp_test_config.json').unlink(missing_ok=True)
    
    # Restore original config
    with open('config.json', 'w') as f:
        json.dump(original_config, f, indent=2)


if __name__ == "__main__":
    test_clustering_on_roi()
    test_config_switching()
    
    print("\n✓ ROI clustering tests completed!")