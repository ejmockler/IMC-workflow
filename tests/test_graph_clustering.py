"""
Test Script for Graph-Based Clustering Baseline

Validates the graph clustering implementation and integration
with the existing IMC analysis pipeline.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.analysis.graph_clustering import GraphClusteringBaseline, create_graph_clustering_baseline
from src.analysis.clustering_comparison import ClusteringEvaluator, compare_graph_vs_spatial_clustering
from src.analysis.spatial_clustering import perform_spatial_clustering


def test_basic_graph_construction():
    """Test basic graph construction methods."""
    print("Testing Graph Construction Methods...")
    
    # Create test data
    np.random.seed(42)
    n_samples = 100
    n_proteins = 5
    feature_matrix = np.random.normal(0, 1, (n_samples, n_proteins))
    protein_names = [f'Protein_{i}' for i in range(n_proteins)]
    spatial_coords = np.random.uniform(0, 50, (n_samples, 2))
    
    baseline = GraphClusteringBaseline(random_state=42)
    
    # Test kNN graph
    try:
        knn_graph = baseline.graph_builder.build_knn_graph(feature_matrix, k_neighbors=10)
        assert knn_graph.shape == (n_samples, n_samples)
        assert knn_graph.nnz > 0
        print("✓ kNN graph construction works")
    except Exception as e:
        print(f"✗ kNN graph construction failed: {e}")
        return False
    
    # Test correlation graph
    try:
        corr_graph = baseline.graph_builder.build_correlation_graph(feature_matrix, threshold=0.3)
        assert corr_graph.shape == (n_samples, n_samples)
        print("✓ Correlation graph construction works")
    except Exception as e:
        print(f"✗ Correlation graph construction failed: {e}")
        return False
    
    # Test spatial weighting
    try:
        spatial_graph = baseline.graph_builder.add_spatial_weights(
            knn_graph, spatial_coords, spatial_weight=0.3
        )
        assert spatial_graph.shape == (n_samples, n_samples)
        print("✓ Spatial weighting works")
    except Exception as e:
        print(f"✗ Spatial weighting failed: {e}")
        return False
    
    return True


def test_clustering_algorithms():
    """Test clustering algorithms."""
    print("\nTesting Clustering Algorithms...")
    
    # Create test data with clear clusters
    np.random.seed(42)
    
    # Create 3 distinct clusters
    cluster1 = np.random.normal([2, 2, 0, 0, 0], 0.5, (30, 5))
    cluster2 = np.random.normal([0, 0, 2, 2, 0], 0.5, (30, 5))
    cluster3 = np.random.normal([0, 0, 0, 0, 2], 0.5, (30, 5))
    
    feature_matrix = np.vstack([cluster1, cluster2, cluster3])
    protein_names = [f'Protein_{i}' for i in range(5)]
    
    baseline = GraphClusteringBaseline(random_state=42)
    
    # Build graph
    knn_graph = baseline.graph_builder.build_knn_graph(feature_matrix, k_neighbors=8)
    
    # Test Leiden clustering
    try:
        labels, info = baseline.community_detector.leiden_clustering(knn_graph, resolution=1.0)
        n_clusters = len(np.unique(labels))
        assert n_clusters >= 2, f"Expected at least 2 clusters, got {n_clusters}"
        print(f"✓ Leiden clustering works (found {n_clusters} clusters)")
    except Exception as e:
        print(f"✗ Leiden clustering failed: {e}")
        return False
    
    # Test spectral clustering
    try:
        labels, info = baseline.community_detector.spectral_clustering(knn_graph, n_clusters=3)
        assert len(np.unique(labels)) == 3, "Spectral clustering should find exactly 3 clusters"
        print("✓ Spectral clustering works")
    except Exception as e:
        print(f"✗ Spectral clustering failed: {e}")
        return False
    
    return True


def test_full_clustering_pipeline():
    """Test complete clustering pipeline."""
    print("\nTesting Full Clustering Pipeline...")
    
    # Create realistic test data
    np.random.seed(42)
    n_samples = 200
    n_proteins = 8
    feature_matrix = np.random.lognormal(0, 0.5, (n_samples, n_proteins))
    protein_names = [f'Protein_{i}' for i in range(n_proteins)]
    spatial_coords = np.random.uniform(0, 100, (n_samples, 2))
    
    baseline = GraphClusteringBaseline(random_state=42)
    
    try:
        results = baseline.cluster_protein_expression(
            feature_matrix, protein_names, spatial_coords,
            graph_method='knn', clustering_method='leiden'
        )
        
        # Validate results structure
        required_keys = [
            'cluster_labels', 'cluster_centroids', 'graph_adjacency',
            'graph_metrics', 'quality_metrics', 'metadata'
        ]
        
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
        
        # Validate cluster labels
        cluster_labels = results['cluster_labels']
        assert len(cluster_labels) == n_samples, "Cluster labels length mismatch"
        
        n_clusters = results['metadata']['n_clusters']
        assert n_clusters > 0, "No clusters found"
        
        print(f"✓ Full pipeline works (found {n_clusters} clusters)")
        
        # Test with different methods
        for method in ['louvain', 'spectral']:
            try:
                method_results = baseline.cluster_protein_expression(
                    feature_matrix, protein_names, spatial_coords,
                    graph_method='knn', clustering_method=method
                )
                print(f"✓ {method} method works")
            except Exception as e:
                print(f"⚠ {method} method failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Full pipeline failed: {e}")
        return False


def test_clustering_comparison():
    """Test clustering comparison framework."""
    print("\nTesting Clustering Comparison Framework...")
    
    # Create test data
    np.random.seed(42)
    n_samples = 150
    n_proteins = 6
    feature_matrix = np.random.normal(0, 1, (n_samples, n_proteins))
    protein_names = [f'Protein_{i}' for i in range(n_proteins)]
    spatial_coords = np.random.uniform(0, 50, (n_samples, 2))
    
    evaluator = ClusteringEvaluator(random_state=42)
    
    try:
        # Test basic comparison
        methods = ['spatial_leiden', 'graph_leiden']
        comparison_results = evaluator.compare_clustering_methods(
            feature_matrix, protein_names, spatial_coords, methods
        )
        
        # Validate structure
        assert 'method_results' in comparison_results
        assert 'pairwise_comparisons' in comparison_results
        assert 'method_ranking' in comparison_results
        
        print("✓ Basic clustering comparison works")
        
        # Test direct graph vs spatial comparison
        direct_comparison = compare_graph_vs_spatial_clustering(
            feature_matrix, protein_names, spatial_coords
        )
        
        assert 'method_results' in direct_comparison
        print("✓ Direct graph vs spatial comparison works")
        
        return True
        
    except Exception as e:
        print(f"✗ Clustering comparison failed: {e}")
        return False


def test_integration_with_existing_pipeline():
    """Test integration with existing spatial clustering."""
    print("\nTesting Integration with Existing Pipeline...")
    
    # Create test data
    np.random.seed(42)
    n_samples = 100
    n_proteins = 5
    feature_matrix = np.random.normal(0, 1, (n_samples, n_proteins))
    protein_names = [f'Protein_{i}' for i in range(n_proteins)]
    spatial_coords = np.random.uniform(0, 50, (n_samples, 2))
    
    try:
        # Test existing spatial clustering
        spatial_labels, spatial_info = perform_spatial_clustering(
            feature_matrix, spatial_coords, method='leiden', resolution=1.0
        )
        
        assert len(spatial_labels) == n_samples
        print("✓ Spatial clustering integration works")
        
        # Test graph baseline creation
        graph_baseline = create_graph_clustering_baseline(
            feature_matrix, protein_names, spatial_coords
        )
        
        assert 'cluster_labels' in graph_baseline
        assert len(graph_baseline['cluster_labels']) == n_samples
        print("✓ Graph baseline creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline integration failed: {e}")
        return False


def test_parameter_optimization():
    """Test parameter optimization functionality."""
    print("\nTesting Parameter Optimization...")
    
    # Create test data with clear structure
    np.random.seed(42)
    
    # Create well-separated clusters
    cluster1 = np.random.normal([3, 0, 0], 0.3, (25, 3))
    cluster2 = np.random.normal([0, 3, 0], 0.3, (25, 3))
    cluster3 = np.random.normal([0, 0, 3], 0.3, (25, 3))
    
    feature_matrix = np.vstack([cluster1, cluster2, cluster3])
    protein_names = ['Protein_A', 'Protein_B', 'Protein_C']
    spatial_coords = np.random.uniform(0, 30, (75, 2))
    
    baseline = GraphClusteringBaseline(random_state=42)
    
    try:
        optimization_results = baseline.parameter_optimization(
            feature_matrix, protein_names, spatial_coords,
            graph_method='knn', clustering_method='leiden',
            n_trials=5  # Small number for testing
        )
        
        assert 'best_score' in optimization_results
        assert 'best_params' in optimization_results
        assert optimization_results['best_score'] > 0
        
        print(f"✓ Parameter optimization works (best score: {optimization_results['best_score']:.3f})")
        return True
        
    except Exception as e:
        print(f"✗ Parameter optimization failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting Edge Cases...")
    
    baseline = GraphClusteringBaseline(random_state=42)
    
    # Test empty data
    try:
        results = baseline.cluster_protein_expression(
            np.array([]).reshape(0, 3), ['A', 'B', 'C'], None
        )
        assert results['metadata']['n_clusters'] == 0
        print("✓ Empty data handling works")
    except Exception as e:
        print(f"✗ Empty data handling failed: {e}")
        return False
    
    # Test single sample
    try:
        results = baseline.cluster_protein_expression(
            np.array([[1, 2, 3]]), ['A', 'B', 'C'], np.array([[0, 0]])
        )
        assert len(results['cluster_labels']) == 1
        print("✓ Single sample handling works")
    except Exception as e:
        print(f"✗ Single sample handling failed: {e}")
        return False
    
    # Test mismatched dimensions
    try:
        baseline.cluster_protein_expression(
            np.random.normal(0, 1, (10, 3)), ['A', 'B'], None  # Wrong number of protein names
        )
        print("⚠ Dimension mismatch should have failed but didn't")
    except Exception:
        print("✓ Dimension mismatch handling works")
    
    return True


def main():
    """Run all tests."""
    print("Graph-Based Clustering Baseline Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_graph_construction,
        test_clustering_algorithms,
        test_full_clustering_pipeline,
        test_clustering_comparison,
        test_integration_with_existing_pipeline,
        test_parameter_optimization,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Graph clustering baseline is ready for use.")
        return True
    else:
        print("❌ Some tests failed. Check implementation before use.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)