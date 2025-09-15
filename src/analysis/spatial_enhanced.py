"""
Enhanced Spatial Analysis with Kernel Methods
Integrates spatial kernels for improved neighborhood detection
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

from src.analysis.kernels import (
    SpatialKernel, KernelFactory, 
    compute_augmented_features, benchmark_kernels
)
from src.analysis.validation import ValidationSuite


def identify_spatial_domains(coords: np.ndarray,
                            values: np.ndarray,
                            protein_names: list,
                            config_path: str = 'config.json',
                            validate: bool = True) -> Tuple:
    """
    Identify spatial domains using kernel-augmented features
    Enhanced version of identify_expression_blobs with spatial kernels
    
    Args:
        coords: Spatial coordinates (n_points, 2)
        values: Expression values (n_points, n_features)
        protein_names: List of protein names
        config_path: Path to configuration
        validate: Whether to validate clustering
        
    Returns:
        Tuple of (domain_labels, domain_signatures, domain_mapping, [validation_results])
    """
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    kernel_config = config.get('spatial_kernels', {})
    
    if not kernel_config.get('enabled', False):
        # Fall back to original method
        from src.analysis.spatial import identify_expression_blobs
        return identify_expression_blobs(coords, values, protein_names, 
                                        config_path=config_path, validate=validate)
    
    print("      Using kernel-enhanced spatial domain detection")
    
    # Create kernel
    kernel_type = kernel_config.get('default_kernel', 'gaussian')
    kernel = KernelFactory.create_kernel(
        kernel_type, 
        kernel_config.get(kernel_type, {})
    )
    print(f"      Kernel: {kernel.name()}")
    
    # Compute augmented features
    lambda_param = kernel_config.get('lambda_mixing', 0.5)
    augmented_features = compute_augmented_features(
        coords, values, kernel, lambda_param
    )
    print(f"      Feature augmentation: {values.shape[1]} → {augmented_features.shape[1]} features")
    
    # Determine optimal number of clusters
    n_domains = _calculate_optimal_domains(len(protein_names), augmented_features.shape[1])
    
    # Cluster augmented features
    kmeans = KMeans(n_clusters=n_domains, random_state=42, n_init=10)
    domain_labels = kmeans.fit_predict(augmented_features)
    
    # Optional benchmarking
    if kernel_config.get('benchmark_kernels', False):
        _run_kernel_benchmark(coords, values, kernel_config)
    
    # Validation
    validation_results = None
    if validate and config.get('analysis', {}).get('validation', {}).get('enabled', False):
        suite = ValidationSuite()
        validation_results = suite.validate_all(
            data=augmented_features,
            labels=domain_labels,
            coords=coords
        )
        
        if config.get('analysis', {}).get('validation', {}).get('report_format') == 'detailed':
            print("      " + suite.format_report(validation_results).replace('\n', '\n      '))
    
    # Extract domain signatures (similar to blob signatures)
    domain_signatures = _extract_domain_signatures(
        coords, values, domain_labels, protein_names, augmented_features
    )
    
    # Create mapping
    domain_mapping = {i: i for i in range(n_domains)}
    
    print(f"      Found {n_domains} spatial domains")
    for domain_id, sig in domain_signatures.items():
        pct = (sig['size'] / len(coords)) * 100
        print(f"         Domain {domain_id}: {pct:.1f}% of tissue")
    
    if validation_results is not None:
        return domain_labels, domain_signatures, domain_mapping, validation_results
    
    return domain_labels, domain_signatures, domain_mapping


def analyze_kernel_neighborhoods(coords: np.ndarray,
                                domain_labels: np.ndarray,
                                domain_signatures: Dict,
                                kernel: SpatialKernel,
                                scale_radius: float = 50.0) -> Dict:
    """
    Analyze neighborhoods using kernel-weighted connections
    Enhanced version of identify_tissue_neighborhoods
    
    Returns:
        Dictionary with kernel-based neighborhood analysis
    """
    
    n_domains = len(domain_signatures)
    tree = cKDTree(coords)
    
    # Build kernel-weighted adjacency matrix between domains
    domain_adjacency = np.zeros((n_domains, n_domains))
    
    # Sample points for efficiency
    sample_size = min(1000, len(coords))
    sample_indices = np.random.choice(len(coords), sample_size, replace=False)
    
    for idx in sample_indices:
        domain_i = domain_labels[idx]
        
        # Get kernel weights for this point
        kernel_result = kernel.compute_weights(coords, idx, tree)
        
        # Accumulate weights by domain
        for neighbor_idx, weight in zip(kernel_result.neighbor_indices, 
                                       kernel_result.weights):
            domain_j = domain_labels[neighbor_idx]
            if domain_i != domain_j:
                domain_adjacency[domain_i, domain_j] += weight
    
    # Normalize adjacency matrix
    row_sums = domain_adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    domain_adjacency = domain_adjacency / row_sums
    
    # Identify connected components (neighborhoods)
    neighborhoods = _extract_neighborhoods_from_adjacency(
        domain_adjacency, domain_signatures, threshold=0.1
    )
    
    # Calculate coverage
    pixels_in_neighborhoods = sum(n['size'] for n in neighborhoods.values())
    coverage = pixels_in_neighborhoods / len(coords)
    
    return {
        'neighborhoods': neighborhoods,
        'domain_adjacency': domain_adjacency,
        'n_neighborhoods': len(neighborhoods),
        'coverage': coverage,
        'kernel_type': kernel.name(),
        'scale_radius': scale_radius
    }


def _calculate_optimal_domains(n_proteins: int, n_augmented_features: int) -> int:
    """Calculate optimal number of spatial domains"""
    # Account for augmented feature space
    if n_augmented_features > n_proteins:
        # We have neighborhood features too
        base_domains = n_proteins * 3  # More domains for richer features
    else:
        base_domains = n_proteins * 2
    
    # Reasonable bounds
    return max(10, min(50, base_domains))


def _extract_domain_signatures(coords: np.ndarray,
                              values: np.ndarray,
                              domain_labels: np.ndarray,
                              protein_names: list,
                              augmented_features: np.ndarray) -> Dict:
    """Extract signatures for each spatial domain"""
    
    n_domains = len(np.unique(domain_labels))
    signatures = {}
    
    for domain_id in range(n_domains):
        mask = domain_labels == domain_id
        if np.sum(mask) < 10:  # Skip tiny domains
            continue
        
        # Original expression in domain
        domain_values = values[mask]
        mean_expression = domain_values.mean(axis=0)
        
        # Find dominant proteins
        top_indices = np.argsort(mean_expression)[-2:][::-1]
        dominant_proteins = [protein_names[i] for i in top_indices]
        
        # Spatial characteristics
        domain_coords = coords[mask]
        centroid = domain_coords.mean(axis=0)
        spread = np.std(domain_coords, axis=0).mean()
        
        signatures[domain_id] = {
            'mean_expression': mean_expression,
            'dominant_proteins': dominant_proteins,
            'size': np.sum(mask),
            'centroid': centroid,
            'spatial_spread': spread,
            'coords': domain_coords
        }
    
    return signatures


def _extract_neighborhoods_from_adjacency(adjacency: np.ndarray,
                                         domain_signatures: Dict,
                                         threshold: float = 0.1) -> Dict:
    """Extract neighborhoods from domain adjacency matrix"""
    
    # Find strongly connected domains
    strong_connections = adjacency > threshold
    
    # Use simple connected components
    visited = set()
    neighborhoods = {}
    neighborhood_id = 0
    
    for start_domain in domain_signatures.keys():
        if start_domain in visited:
            continue
        
        # BFS to find connected component
        component = set()
        queue = [start_domain]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            component.add(current)
            
            # Add strongly connected neighbors
            for neighbor in range(len(adjacency)):
                if (strong_connections[current, neighbor] or 
                    strong_connections[neighbor, current]):
                    if neighbor not in visited and neighbor in domain_signatures:
                        queue.append(neighbor)
        
        if len(component) >= 2:  # Neighborhood needs multiple domains
            # Calculate neighborhood properties
            total_size = sum(domain_signatures[d]['size'] for d in component)
            dominant_proteins = []
            
            for domain_id in component:
                dominant_proteins.extend(domain_signatures[domain_id]['dominant_proteins'])
            
            # Get unique proteins
            protein_counts = {}
            for protein in dominant_proteins:
                protein_counts[protein] = protein_counts.get(protein, 0) + 1
            
            top_proteins = sorted(protein_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            neighborhoods[neighborhood_id] = {
                'domains': list(component),
                'size': total_size,
                'n_domains': len(component),
                'dominant_proteins': [p for p, _ in top_proteins],
                'mean_adjacency': adjacency[list(component)][:, list(component)].mean()
            }
            
            neighborhood_id += 1
    
    return neighborhoods


def _run_kernel_benchmark(coords: np.ndarray, 
                         values: np.ndarray,
                         kernel_config: Dict) -> None:
    """Run kernel benchmarking if enabled"""
    
    print("\n      Running kernel benchmark...")
    
    # Create kernels to test
    kernels_to_test = {}
    
    # Gaussian with different sigmas
    for sigma in [20, 30, 50]:
        kernel = KernelFactory.create_kernel('gaussian', {'sigma': sigma, 'cutoff_radius': sigma * 3})
        kernels_to_test[f'gaussian_σ{sigma}'] = kernel
    
    # Adaptive
    kernel = KernelFactory.create_kernel('adaptive', kernel_config.get('adaptive', {}))
    kernels_to_test['adaptive'] = kernel
    
    # Laplacian
    kernel = KernelFactory.create_kernel('laplacian', kernel_config.get('laplacian', {}))
    kernels_to_test['laplacian'] = kernel
    
    # Run benchmark
    results = benchmark_kernels(coords, values, kernels_to_test)
    
    # Print results
    print("\n      Kernel Benchmark Results:")
    print("      " + "-" * 60)
    print(f"      {'Kernel':<20} {'Time (s)':<12} {'Silhouette':<12}")
    print("      " + "-" * 60)
    
    for kernel_name, metrics in results.items():
        print(f"      {kernel_name:<20} {metrics['time_seconds']:<12.3f} {metrics['silhouette_score']:<12.3f}")
    
    # Find best kernel
    best_kernel = max(results.items(), key=lambda x: x[1]['silhouette_score'])
    print(f"\n      Best kernel: {best_kernel[0]} (silhouette: {best_kernel[1]['silhouette_score']:.3f})")


def compare_methods(coords: np.ndarray,
                   values: np.ndarray, 
                   protein_names: list,
                   config_path: str = 'config.json') -> Dict:
    """
    Compare kernel-enhanced vs traditional clustering
    
    Returns:
        Comparison metrics between methods
    """
    from src.analysis.spatial import identify_expression_blobs
    from src.analysis.validation import ValidationSuite
    import time
    
    results = {}
    suite = ValidationSuite()
    
    # Traditional method
    start = time.time()
    trad_result = identify_expression_blobs(coords, values, protein_names, 
                                           config_path=config_path, validate=False)
    trad_labels = trad_result[0]
    trad_time = time.time() - start
    
    trad_validation = suite.validate_all(values, trad_labels, coords=coords)
    
    results['traditional'] = {
        'time': trad_time,
        'n_clusters': len(np.unique(trad_labels)),
        'validation': suite.get_summary_score(trad_validation)
    }
    
    # Kernel-enhanced method
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Temporarily enable kernels
    config['spatial_kernels']['enabled'] = True
    temp_config = Path('temp_config.json')
    with open(temp_config, 'w') as f:
        json.dump(config, f)
    
    start = time.time()
    kernel_result = identify_spatial_domains(coords, values, protein_names,
                                            config_path=str(temp_config), validate=False)
    kernel_labels = kernel_result[0]
    kernel_time = time.time() - start
    
    # Get augmented features for validation
    kernel_config = config.get('spatial_kernels', {})
    kernel = KernelFactory.create_kernel(
        kernel_config.get('default_kernel', 'gaussian'),
        kernel_config.get(kernel_config.get('default_kernel', 'gaussian'), {})
    )
    augmented = compute_augmented_features(coords, values, kernel, 
                                          kernel_config.get('lambda_mixing', 0.5))
    
    kernel_validation = suite.validate_all(augmented, kernel_labels, coords=coords)
    
    results['kernel_enhanced'] = {
        'time': kernel_time,
        'n_clusters': len(np.unique(kernel_labels)),
        'validation': suite.get_summary_score(kernel_validation)
    }
    
    # Cleanup
    temp_config.unlink()
    
    # Calculate improvement
    improvement = (results['kernel_enhanced']['validation'] - 
                  results['traditional']['validation']) / results['traditional']['validation']
    
    results['improvement_percent'] = improvement * 100
    
    return results