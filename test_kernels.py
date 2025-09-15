#!/usr/bin/env python3
"""
Test spatial kernel implementations
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from src.analysis.kernels import (
    GaussianKernel, AdaptiveKernel, LaplacianKernel,
    compute_augmented_features, benchmark_kernels
)


def test_individual_kernels():
    """Test each kernel type individually"""
    
    print("Testing Spatial Kernels")
    print("=" * 50)
    
    # Generate test data - 3 clusters in space
    np.random.seed(42)
    n_points = 300
    
    # Create spatially organized clusters
    cluster_centers = np.array([[0, 0], [100, 0], [50, 86.6]])  # Triangle
    coords = []
    values = []
    
    for i, center in enumerate(cluster_centers):
        cluster_coords = np.random.randn(n_points // 3, 2) * 20 + center
        coords.append(cluster_coords)
        
        # Different expression patterns per cluster
        cluster_values = np.random.randn(n_points // 3, 5) * 0.5
        cluster_values[:, i % 5] += 2  # Boost one protein per cluster
        values.append(cluster_values)
    
    coords = np.vstack(coords)
    values = np.vstack(values)
    
    print(f"Test data: {len(coords)} points, {values.shape[1]} features")
    print()
    
    # Test each kernel
    kernels = {
        'Gaussian (σ=30)': GaussianKernel(sigma=30),
        'Gaussian (σ=50)': GaussianKernel(sigma=50),
        'Adaptive': AdaptiveKernel(min_neighbors=10),
        'Laplacian (σ=40)': LaplacianKernel(sigma=40)
    }
    
    # Test point in dense region
    test_idx_dense = 50  # In first cluster
    # Test point in sparse region (between clusters)
    test_idx_sparse = 150  # In second cluster, edge
    
    print("Kernel Weight Analysis:")
    print("-" * 50)
    
    for kernel_name, kernel in kernels.items():
        print(f"\n{kernel_name}:")
        
        # Dense region
        result_dense = kernel.compute_weights(coords, test_idx_dense)
        print(f"  Dense region: {result_dense.n_neighbors} neighbors, "
              f"radius={result_dense.effective_radius:.1f}μm")
        
        # Sparse region  
        result_sparse = kernel.compute_weights(coords, test_idx_sparse)
        print(f"  Sparse region: {result_sparse.n_neighbors} neighbors, "
              f"radius={result_sparse.effective_radius:.1f}μm")
        
        # Weight distribution
        if result_dense.n_neighbors > 1:
            print(f"  Weight range: [{result_dense.weights.min():.3f}, "
                  f"{result_dense.weights.max():.3f}]")


def test_feature_augmentation():
    """Test BANKSY-style feature augmentation"""
    
    print("\n" + "=" * 50)
    print("Testing Feature Augmentation")
    print("-" * 50)
    
    # Simple test data
    np.random.seed(42)
    coords = np.random.rand(100, 2) * 100
    values = np.random.randn(100, 5)
    
    kernel = GaussianKernel(sigma=30)
    
    # Test different lambda values
    lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for lambda_val in lambdas:
        augmented = compute_augmented_features(coords, values, kernel, lambda_val)
        print(f"λ={lambda_val}: {values.shape} → {augmented.shape}")


def test_kernel_benchmark():
    """Benchmark different kernels"""
    
    print("\n" + "=" * 50)
    print("Kernel Benchmarking")
    print("-" * 50)
    
    # Larger dataset for meaningful benchmark
    np.random.seed(42)
    coords = np.random.rand(1000, 2) * 200
    values = np.random.randn(1000, 7)  # 7 proteins
    
    # Create kernels to test
    kernels_to_test = {
        'gaussian_30': GaussianKernel(sigma=30),
        'gaussian_50': GaussianKernel(sigma=50),
        'adaptive': AdaptiveKernel(),
        'laplacian_40': LaplacianKernel(sigma=40)
    }
    
    # Run benchmark
    results = benchmark_kernels(coords, values, kernels_to_test)
    
    # Display results
    print(f"\n{'Kernel':<15} {'Time (s)':<12} {'Silhouette':<12}")
    print("-" * 40)
    
    for kernel_name, metrics in sorted(results.items(), 
                                      key=lambda x: x[1]['silhouette_score'],
                                      reverse=True):
        print(f"{kernel_name:<15} {metrics['time_seconds']:<12.3f} "
              f"{metrics['silhouette_score']:<12.3f}")
    
    # Find best
    best = max(results.items(), key=lambda x: x[1]['silhouette_score'])
    print(f"\nBest kernel: {best[0]} (score: {best[1]['silhouette_score']:.3f})")


def visualize_kernel_effects():
    """Visualize kernel weight distributions"""
    
    print("\n" + "=" * 50)
    print("Generating Kernel Visualization")
    
    # Create grid for visualization
    grid_size = 100
    x = np.linspace(-100, 100, grid_size)
    y = np.linspace(-100, 100, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Center point
    center = np.array([[0, 0]])
    all_coords = np.vstack([center, grid_coords])
    
    # Compute weights for each kernel
    kernels = {
        'Gaussian σ=30': GaussianKernel(sigma=30, cutoff_radius=100),
        'Adaptive': AdaptiveKernel(max_radius=100),
        'Laplacian σ=40': LaplacianKernel(sigma=40, cutoff_radius=100)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (name, kernel) in zip(axes, kernels.items()):
        # Compute weights from center
        result = kernel.compute_weights(all_coords, 0)
        
        # Create weight image
        weight_image = np.zeros(len(all_coords))
        weight_image[result.neighbor_indices] = result.weights
        weight_image = weight_image[1:].reshape(grid_size, grid_size)
        
        # Plot
        im = ax.imshow(weight_image, extent=[-100, 100, -100, 100],
                      origin='lower', cmap='hot')
        ax.set_title(name)
        ax.set_xlabel('Distance (μm)')
        ax.set_ylabel('Distance (μm)')
        ax.plot(0, 0, 'c*', markersize=10)  # Mark center
        plt.colorbar(im, ax=ax, label='Weight')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'kernel_weights_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    test_individual_kernels()
    test_feature_augmentation()
    test_kernel_benchmark()
    visualize_kernel_effects()
    
    print("\n✓ All kernel tests completed successfully!")