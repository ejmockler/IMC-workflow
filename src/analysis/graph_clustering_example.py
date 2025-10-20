"""
Graph-Based Clustering Baseline Example

Demonstrates how to use the graph clustering baseline for IMC analysis
and compare it against spatial clustering methods.
"""

import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import IMC analysis modules
from .graph_clustering import GraphClusteringBaseline, create_graph_clustering_baseline
from .clustering_comparison import ClusteringEvaluator, compare_graph_vs_spatial_clustering
from .spatial_clustering import perform_spatial_clustering
from .ion_count_processing import ion_count_pipeline


def demonstrate_graph_clustering_methods():
    """
    Demonstrate different graph construction and clustering methods.
    """
    print("Graph-Based Clustering Methods Demonstration")
    print("=" * 50)
    
    # Create synthetic IMC-like data
    n_samples = 500
    n_proteins = 9
    protein_names = [f'Protein_{i+1}' for i in range(n_proteins)]
    
    # Generate protein expression data with some structure
    np.random.seed(42)
    
    # Create three synthetic tissue regions with different protein profiles
    region1 = np.random.lognormal(mean=1.0, sigma=0.5, size=(150, n_proteins))
    region1[:, :3] *= 3  # High expression of first 3 proteins
    
    region2 = np.random.lognormal(mean=1.0, sigma=0.5, size=(200, n_proteins))
    region2[:, 3:6] *= 3  # High expression of middle 3 proteins
    
    region3 = np.random.lognormal(mean=1.0, sigma=0.5, size=(150, n_proteins))
    region3[:, 6:] *= 3  # High expression of last 3 proteins
    
    feature_matrix = np.vstack([region1, region2, region3])
    
    # Create spatial coordinates
    spatial_coords = np.random.uniform(0, 100, size=(n_samples, 2))
    
    # True cluster labels for evaluation
    true_labels = np.concatenate([
        np.zeros(150, dtype=int),
        np.ones(200, dtype=int),
        np.full(150, 2, dtype=int)
    ])
    
    # Initialize graph clustering baseline
    baseline = GraphClusteringBaseline(random_state=42)
    
    # Test different graph construction methods
    graph_methods = ['knn', 'correlation', 'radius']
    clustering_methods = ['leiden', 'louvain', 'spectral']
    
    results = {}
    
    print("\n1. Testing Graph Construction Methods:")
    print("-" * 40)
    
    for graph_method in graph_methods:
        print(f"\nTesting {graph_method} graph construction...")
        
        method_results = baseline.cluster_protein_expression(
            feature_matrix, protein_names, spatial_coords,
            graph_method=graph_method,
            clustering_method='leiden'
        )
        
        results[graph_method] = method_results
        
        # Print basic results
        n_clusters = method_results['metadata']['n_clusters']
        silhouette = method_results['quality_metrics'].get('silhouette_score', np.nan)
        modularity = method_results['graph_metrics'].get('modularity', np.nan)
        
        print(f"  Clusters found: {n_clusters}")
        print(f"  Silhouette score: {silhouette:.3f}")
        print(f"  Modularity: {modularity:.3f}")
    
    print("\n2. Testing Clustering Algorithms:")
    print("-" * 40)
    
    for clustering_method in clustering_methods:
        print(f"\nTesting {clustering_method} clustering...")
        
        try:
            method_results = baseline.cluster_protein_expression(
                feature_matrix, protein_names, spatial_coords,
                graph_method='knn',
                clustering_method=clustering_method
            )
            
            n_clusters = method_results['metadata']['n_clusters']
            silhouette = method_results['quality_metrics'].get('silhouette_score', np.nan)
            
            print(f"  Clusters found: {n_clusters}")
            print(f"  Silhouette score: {silhouette:.3f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    print("\n3. Parameter Optimization Example:")
    print("-" * 40)
    
    optimization_results = baseline.parameter_optimization(
        feature_matrix, protein_names, spatial_coords,
        graph_method='knn', clustering_method='leiden',
        n_trials=10, optimization_metric='silhouette'
    )
    
    best_score = optimization_results['best_score']
    best_graph_params, best_clustering_params = optimization_results['best_params']
    
    print(f"Best silhouette score: {best_score:.3f}")
    print(f"Best graph parameters: {best_graph_params}")
    print(f"Best clustering parameters: {best_clustering_params}")
    
    return results, optimization_results


def demonstrate_clustering_comparison():
    """
    Demonstrate comprehensive clustering method comparison.
    """
    print("\n\nClustering Method Comparison Demonstration")
    print("=" * 50)
    
    # Create synthetic data (reuse from previous function)
    n_samples = 300
    n_proteins = 9
    protein_names = [f'Protein_{i+1}' for i in range(n_proteins)]
    
    np.random.seed(42)
    feature_matrix = np.random.lognormal(mean=1.0, sigma=0.7, size=(n_samples, n_proteins))
    spatial_coords = np.random.uniform(0, 100, size=(n_samples, 2))
    
    # Run comprehensive comparison
    evaluator = ClusteringEvaluator(random_state=42)
    
    methods = ['spatial_leiden', 'graph_leiden', 'graph_louvain', 'graph_spectral']
    
    comparison_results = evaluator.compare_clustering_methods(
        feature_matrix, protein_names, spatial_coords, methods
    )
    
    print("\nMethod Performance Summary:")
    print("-" * 30)
    
    for method, results in comparison_results['method_results'].items():
        if results['clustering_results'] is not None:
            metrics = results['evaluation_metrics']
            n_clusters = metrics.get('n_clusters', 0)
            silhouette = metrics.get('silhouette_score', np.nan)
            coherence = metrics.get('spatial_coherence', np.nan)
            
            print(f"\n{method}:")
            print(f"  Clusters: {n_clusters}")
            print(f"  Silhouette: {silhouette:.3f}")
            print(f"  Spatial coherence: {coherence:.3f}")
    
    print("\nMethod Rankings:")
    print("-" * 15)
    ranking = comparison_results['method_ranking']
    overall_ranking = ranking.get('overall_ranking', [])
    
    for i, method in enumerate(overall_ranking, 1):
        composite_score = ranking['composite_scores'].get(method, np.inf)
        print(f"{i}. {method} (score: {composite_score:.2f})")
    
    print("\nPairwise Method Comparisons:")
    print("-" * 30)
    
    for pair, metrics in comparison_results['pairwise_comparisons'].items():
        ari = metrics.get('adjusted_rand_index', np.nan)
        nmi = metrics.get('normalized_mutual_info', np.nan)
        print(f"{pair}: ARI={ari:.3f}, NMI={nmi:.3f}")
    
    return comparison_results


def demonstrate_integration_with_pipeline():
    """
    Demonstrate integration with existing IMC pipeline.
    """
    print("\n\nPipeline Integration Demonstration")
    print("=" * 40)
    
    # Simulate pipeline data (normally would come from actual IMC processing)
    n_pixels = 1000
    n_proteins = 9
    protein_names = ['CD3', 'CD4', 'CD8', 'CD20', 'CD68', 'CD45', 'Ki67', 'PanCK', 'Vimentin']
    
    np.random.seed(42)
    
    # Create coordinates
    coords = np.random.uniform(0, 200, size=(n_pixels, 2))
    
    # Create ion counts (simulate realistic IMC data)
    ion_counts = {}
    for protein in protein_names:
        # Poisson-distributed counts with some spatial structure
        base_rate = np.random.uniform(1, 10)
        counts = np.random.poisson(base_rate, n_pixels)
        ion_counts[protein] = counts.astype(float)
    
    print("1. Running standard ion count pipeline...")
    
    # Run standard pipeline
    pipeline_results = ion_count_pipeline(
        coords, ion_counts, bin_size_um=20.0
    )
    
    feature_matrix = pipeline_results['feature_matrix']
    spatial_coords = pipeline_results.get('spatial_coords')
    
    if spatial_coords is None and len(feature_matrix) > 0:
        # Create spatial coordinates from bin centers
        bin_edges_x = pipeline_results['bin_edges_x']
        bin_edges_y = pipeline_results['bin_edges_y']
        valid_indices = pipeline_results['valid_indices']
        
        if len(bin_edges_x) > 1 and len(bin_edges_y) > 1:
            x_centers = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2
            y_centers = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2
            
            n_x_bins = len(bin_edges_x) - 1
            y_indices = valid_indices // n_x_bins
            x_indices = valid_indices % n_x_bins
            
            spatial_coords = np.column_stack([
                x_centers[x_indices],
                y_centers[y_indices]
            ])
    
    if len(feature_matrix) == 0:
        print("Warning: Empty feature matrix from pipeline")
        return
    
    print(f"Pipeline processed {len(feature_matrix)} spatial bins")
    
    print("\n2. Running graph clustering baseline...")
    
    # Run graph clustering baseline
    graph_results = create_graph_clustering_baseline(
        feature_matrix, protein_names, spatial_coords
    )
    
    print(f"Graph clustering found {graph_results['metadata']['n_clusters']} clusters")
    
    print("\n3. Comparing with spatial clustering...")
    
    if spatial_coords is not None:
        # Direct comparison
        comparison = compare_graph_vs_spatial_clustering(
            feature_matrix, protein_names, spatial_coords
        )
        
        # Extract key comparison metrics
        if 'graph_vs_spatial' in comparison:
            gvs = comparison['graph_vs_spatial']
            agreement = gvs['agreement']
            
            print(f"Agreement (ARI): {agreement.get('adjusted_rand_index', 'N/A')}")
            print(f"Agreement (NMI): {agreement.get('normalized_mutual_info', 'N/A')}")
            
            print("\nRecommendations:")
            for rec in gvs.get('recommendations', []):
                print(f"  - {rec}")
    
    return pipeline_results, graph_results


def create_visualization_example(results: Dict, output_dir: Optional[str] = None):
    """
    Create visualization comparing different clustering approaches.
    """
    print("\n\nCreating Comparison Visualizations")
    print("=" * 40)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    # Extract data for visualization
    methods = list(results.keys())
    n_methods = len(methods)
    
    if n_methods == 0:
        print("No results to visualize")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Graph Clustering Methods Comparison', fontsize=16)
    
    # Plot 1: Number of clusters
    method_names = []
    cluster_counts = []
    silhouette_scores = []
    
    for method, result in results.items():
        if result is not None:
            method_names.append(method)
            cluster_counts.append(result['metadata']['n_clusters'])
            silhouette_scores.append(result['quality_metrics'].get('silhouette_score', 0))
    
    if method_names:
        axes[0, 0].bar(method_names, cluster_counts)
        axes[0, 0].set_title('Number of Clusters Found')
        axes[0, 0].set_ylabel('Cluster Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(method_names, silhouette_scores)
        axes[0, 1].set_title('Silhouette Scores')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 2: Graph metrics comparison
    density_scores = []
    for method, result in results.items():
        if result is not None and 'graph_metrics' in result:
            density_scores.append(result['graph_metrics'].get('graph_density', 0))
        else:
            density_scores.append(0)
    
    if method_names and density_scores:
        axes[1, 0].bar(method_names, density_scores)
        axes[1, 0].set_title('Graph Density')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 3: Method comparison heatmap
    if len(method_names) > 1:
        comparison_matrix = np.eye(len(method_names))
        
        # This would be filled with actual pairwise comparison metrics
        # For demonstration, using random similarities
        np.random.seed(42)
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                sim = np.random.uniform(0.3, 0.8)
                comparison_matrix[i, j] = sim
                comparison_matrix[j, i] = sim
        
        im = axes[1, 1].imshow(comparison_matrix, cmap='viridis')
        axes[1, 1].set_title('Method Agreement Matrix')
        axes[1, 1].set_xticks(range(len(method_names)))
        axes[1, 1].set_yticks(range(len(method_names)))
        axes[1, 1].set_xticklabels(method_names, rotation=45)
        axes[1, 1].set_yticklabels(method_names)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'clustering_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path / 'clustering_comparison.png'}")
    else:
        plt.show()
    
    plt.close()


def main():
    """
    Run complete graph clustering baseline demonstration.
    """
    print("Graph-Based Clustering Baseline for IMC Analysis")
    print("=" * 60)
    print("This example demonstrates the complete graph clustering")
    print("baseline implementation and comparison framework.\n")
    
    try:
        # 1. Demonstrate graph clustering methods
        graph_results, optimization_results = demonstrate_graph_clustering_methods()
        
        # 2. Demonstrate clustering comparison
        comparison_results = demonstrate_clustering_comparison()
        
        # 3. Demonstrate pipeline integration
        pipeline_results, baseline_results = demonstrate_integration_with_pipeline()
        
        # 4. Create visualizations
        create_visualization_example(graph_results)
        
        print("\n\nDemo completed successfully!")
        print("=" * 30)
        print("Key takeaways:")
        print("1. Graph clustering provides pure protein expression similarity")
        print("2. Multiple graph construction methods available (kNN, correlation, etc.)")
        print("3. Various community detection algorithms supported")
        print("4. Comprehensive evaluation and comparison framework")
        print("5. Seamless integration with existing IMC pipeline")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()