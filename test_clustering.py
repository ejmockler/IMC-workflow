#!/usr/bin/env python3
"""
Test advanced clustering implementations
"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.analysis.clustering import (
    ClustererFactory, 
    auto_select_clusterer,
    benchmark_clusterers
)


def test_individual_clusterers():
    """Test each clustering algorithm individually"""
    
    print("Testing Advanced Clustering Algorithms")
    print("=" * 50)
    
    # Generate test data - 3 well-separated clusters
    np.random.seed(42)
    n_samples = 500
    n_features = 7  # Like our protein panel
    
    # Create distinct clusters
    cluster1 = np.random.randn(n_samples // 3, n_features) + [2, 0, 0, 0, 0, 0, 0]
    cluster2 = np.random.randn(n_samples // 3, n_features) + [0, 2, 0, 0, 0, 0, 0]
    cluster3 = np.random.randn(n_samples // 3, n_features) + [0, 0, 2, 0, 0, 0, 0]
    
    data = np.vstack([cluster1, cluster2, cluster3])
    
    print(f"Test data: {data.shape[0]} samples, {data.shape[1]} features")
    print()
    
    # Test each available clusterer
    algorithms = ClustererFactory.list_algorithms()
    
    for algo_name in algorithms:
        print(f"Testing {algo_name}:")
        
        try:
            clusterer = ClustererFactory.create(algo_name)
            
            # Cluster with appropriate parameters
            if algo_name in ['phenograph', 'leiden']:
                # These determine their own n_clusters
                result = clusterer.fit_predict(data)
            else:
                # These need n_clusters specified
                result = clusterer.fit_predict(data, n_clusters=3)
            
            print(f"  ✓ Found {result.n_clusters} clusters")
            print(f"  Algorithm: {result.algorithm}")
            print(f"  Converged: {result.converged}")
            
            if result.iterations:
                print(f"  Iterations: {result.iterations}")
            
            # Check cluster sizes
            unique, counts = np.unique(result.labels, return_counts=True)
            print(f"  Cluster sizes: {dict(zip(unique, counts))}")
            
        except ImportError as e:
            print(f"  ⚠ Not installed: {e}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()


def test_auto_selection():
    """Test automatic algorithm selection"""
    
    print("Testing Auto-Selection")
    print("=" * 50)
    
    test_cases = [
        (100, 5, "Small dataset"),
        (1000, 10, "Medium dataset"),
        (10000, 7, "Large dataset"),
        (100000, 3, "Very large dataset"),
        (500, 60, "High-dimensional dataset")
    ]
    
    for n_samples, n_features, description in test_cases:
        # Create dummy data with specified shape
        data = np.random.randn(n_samples, n_features)
        
        selected = auto_select_clusterer(data)
        print(f"{description} ({n_samples}×{n_features}): {selected}")
    
    print()


def test_benchmarking():
    """Test clustering benchmark functionality"""
    
    print("Benchmarking Clustering Algorithms")
    print("=" * 50)
    
    # Generate larger test dataset
    np.random.seed(42)
    n_samples = 2000
    n_features = 7
    
    # Create overlapping clusters (more realistic)
    centers = np.random.randn(5, n_features) * 2
    data = []
    
    for center in centers:
        cluster_data = np.random.randn(n_samples // 5, n_features) * 0.8 + center
        data.append(cluster_data)
    
    data = np.vstack(data)
    
    print(f"Benchmark dataset: {data.shape}")
    print()
    
    # Run benchmark on fast algorithms only
    algorithms = ['kmeans', 'minibatch', 'flowsom']
    
    print("Running benchmark (this may take a moment)...")
    results = benchmark_clusterers(data, algorithms, n_clusters=5)
    
    # Display results table
    print("\nBenchmark Results:")
    print("-" * 60)
    print(f"{'Algorithm':<15} {'Time (s)':<12} {'Clusters':<10} {'Silhouette':<12}")
    print("-" * 60)
    
    for algo_name, metrics in sorted(results.items()):
        if 'error' in metrics:
            print(f"{algo_name:<15} {metrics['error']}")
        else:
            print(f"{algo_name:<15} {metrics['time_seconds']:<12.3f} "
                  f"{metrics['n_clusters']:<10} {metrics['silhouette_score']:<12.3f}")
    
    # Find best performer
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_time = min(valid_results.items(), key=lambda x: x[1]['time_seconds'])
        best_quality = max(valid_results.items(), key=lambda x: x[1]['silhouette_score'])
        
        print("\nSummary:")
        print(f"  Fastest: {best_time[0]} ({best_time[1]['time_seconds']:.3f}s)")
        print(f"  Best quality: {best_quality[0]} (silhouette: {best_quality[1]['silhouette_score']:.3f})")


def test_config_integration():
    """Test configuration-based clustering"""
    
    print("\nTesting Configuration Integration")
    print("=" * 50)
    
    import json
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    clustering_config = config.get('clustering', {})
    
    print(f"Default algorithm: {clustering_config.get('algorithm', 'kmeans')}")
    print(f"Auto-select enabled: {clustering_config.get('auto_select', False)}")
    print(f"Benchmark enabled: {clustering_config.get('benchmark_algorithms', False)}")
    print()
    
    print("Configured algorithms:")
    for algo_name in clustering_config.get('algorithms', {}).keys():
        print(f"  - {algo_name}")
    
    # Test creating clusterer from config
    print("\nCreating KMeans from config:")
    kmeans = ClustererFactory.create('kmeans', clustering_config.get('algorithms', {}))
    print(f"  n_init: {kmeans.n_init}")
    print(f"  max_iter: {kmeans.max_iter}")
    print(f"  random_state: {kmeans.random_state}")


if __name__ == "__main__":
    test_individual_clusterers()
    test_auto_selection()
    test_benchmarking()
    test_config_integration()
    
    print("\n✓ All clustering tests completed!")