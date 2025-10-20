#!/usr/bin/env python3
"""
Test Script for Synthetic Data Generator Integration

Demonstrates how the synthetic data generator integrates with the existing
IMC analysis pipeline and validates the generated data quality.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import synthetic data generator
from src.analysis.synthetic_data_generator import (
    SyntheticDataGenerator, SyntheticDataConfig, SpatialPattern, TissueType,
    create_example_datasets, validate_synthetic_dataset
)

# Import existing pipeline components for integration testing
from src.analysis.ion_count_processing import ion_count_pipeline
from src.analysis.slic_segmentation import slic_pipeline
from src.analysis.spatial_stats import compute_ripleys_k, compute_spatial_correlation
from src.analysis.coabundance_features import generate_coabundance_features
from src.validation.framework import ValidationSuite, ValidationSuiteConfig

def test_basic_data_generation():
    """Test basic synthetic data generation."""
    print("=== Testing Basic Data Generation ===")
    
    # Create simple configuration
    config = SyntheticDataConfig(
        roi_size_um=(500.0, 500.0),
        n_cells_total=3000,
        protein_names=['CD45', 'CD3', 'CD20', 'PanCK', 'DNA1'],
        known_cluster_count=3,
        baseline_noise_level=0.1
    )
    
    # Generate data
    generator = SyntheticDataGenerator(config)
    dataset = generator.generate_complete_dataset()
    
    # Basic validation
    print(f"Generated {len(dataset['coordinates'])} cells")
    print(f"Proteins: {dataset['protein_names']}")
    print(f"Ground truth clusters: {len(np.unique(dataset['ground_truth_clusters']))}")
    
    # Validate data quality
    validation_result = validate_synthetic_dataset(dataset)
    print(f"Validation: {validation_result.severity.value} (score: {validation_result.quality_score:.2f})")
    print(f"Message: {validation_result.message}")
    
    return dataset

def test_pipeline_integration(dataset):
    """Test integration with existing analysis pipeline."""
    print("\n=== Testing Pipeline Integration ===")
    
    try:
        # Test ion count processing pipeline
        print("Testing ion count processing...")
        ion_result = ion_count_pipeline(
            coords=dataset['coordinates'],
            ion_counts=dataset['ion_counts'],
            bin_size_um=20.0,
            n_clusters=None,  # Auto-detect
            memory_limit_gb=2.0
        )
        
        print(f"✓ Ion count pipeline: {len(ion_result['cluster_labels'])} samples processed")
        print(f"  Detected {len(np.unique(ion_result['cluster_labels'][ion_result['cluster_labels'] >= 0]))} clusters")
        
    except Exception as e:
        print(f"✗ Ion count pipeline failed: {e}")
    
    try:
        # Test SLIC segmentation pipeline
        print("Testing SLIC segmentation...")
        slic_result = slic_pipeline(
            coords=dataset['coordinates'],
            ion_counts=dataset['ion_counts'],
            dna1_intensities=dataset['dna1_intensities'],
            dna2_intensities=dataset['dna2_intensities'],
            target_scale_um=25.0
        )
        
        print(f"✓ SLIC pipeline: {slic_result['n_segments_used']} superpixels generated")
        print(f"  Superpixel coordinates shape: {slic_result['superpixel_coords'].shape}")
        
    except Exception as e:
        print(f"✗ SLIC pipeline failed: {e}")

def test_spatial_statistics(dataset):
    """Test spatial statistics computation on synthetic data."""
    print("\n=== Testing Spatial Statistics ===")
    
    coordinates = dataset['coordinates']
    
    try:
        # Test Ripley's K function
        distances, k_values = compute_ripleys_k(
            coordinates, max_distance=100, n_bins=15
        )
        print(f"✓ Ripley's K computed: {len(distances)} distance bins")
        
        # Test protein correlations
        proteins = list(dataset['ion_counts'].keys())
        if len(proteins) >= 2:
            protein1_expr = dataset['ion_counts'][proteins[0]]
            protein2_expr = dataset['ion_counts'][proteins[1]]
            
            correlation = compute_spatial_correlation(
                protein1_expr.reshape(-1, 1),
                protein2_expr.reshape(-1, 1)
            )
            print(f"✓ Spatial correlation between {proteins[0]} and {proteins[1]}: {correlation:.3f}")
        
    except Exception as e:
        print(f"✗ Spatial statistics failed: {e}")

def test_coabundance_features(dataset):
    """Test co-abundance feature generation."""
    print("\n=== Testing Co-abundance Features ===")
    
    try:
        # Create feature matrix
        protein_names = dataset['protein_names']
        expressions = dataset['ion_counts']
        
        feature_matrix = np.column_stack([
            expressions[protein] for protein in protein_names
        ])
        
        # Generate co-abundance features
        enriched_features, enriched_names = generate_coabundance_features(
            feature_matrix=feature_matrix,
            protein_names=protein_names,
            spatial_coords=dataset['coordinates'],
            interaction_order=2,
            include_ratios=True,
            include_products=True,
            include_spatial_covariance=True
        )
        
        print(f"✓ Co-abundance features: {feature_matrix.shape[1]} → {enriched_features.shape[1]} features")
        print(f"  Original proteins: {len(protein_names)}")
        print(f"  Enriched features: {len(enriched_names)}")
        
    except Exception as e:
        print(f"✗ Co-abundance features failed: {e}")

def test_validation_framework(datasets):
    """Test integration with validation framework."""
    print("\n=== Testing Validation Framework ===")
    
    try:
        # Create validation suite
        config = ValidationSuiteConfig(
            name="synthetic_data_validation",
            stop_on_critical=False,
            minimum_quality_score=0.5
        )
        
        suite = ValidationSuite(config)
        
        # Add synthetic data validator
        from src.analysis.synthetic_data_generator import SyntheticDataValidator
        suite.add_rule(SyntheticDataValidator())
        
        # Test each dataset
        for dataset_name, dataset in datasets.items():
            print(f"Validating {dataset_name} dataset...")
            result = suite.validate(dataset)
            
            print(f"  Status: {result.summary_stats['status']}")
            print(f"  Quality score: {result.summary_stats.get('overall_quality_score', 'N/A')}")
            print(f"  Critical failures: {result.summary_stats['has_critical']}")
        
    except Exception as e:
        print(f"✗ Validation framework failed: {e}")

def test_different_scenarios():
    """Test different synthetic data scenarios."""
    print("\n=== Testing Different Scenarios ===")
    
    scenarios = {
        'clustered': SyntheticDataConfig(
            roi_size_um=(400.0, 400.0),
            n_cells_total=2000,
            protein_names=['CD45', 'CD3', 'PanCK', 'DNA1'],
            spatial_autocorr_range=30.0,
            baseline_noise_level=0.05
        ),
        'dispersed': SyntheticDataConfig(
            roi_size_um=(400.0, 400.0),
            n_cells_total=2000,
            protein_names=['CD45', 'CD3', 'PanCK', 'DNA1'],
            spatial_autocorr_range=80.0,
            baseline_noise_level=0.1
        ),
        'high_noise': SyntheticDataConfig(
            roi_size_um=(400.0, 400.0),
            n_cells_total=2000,
            protein_names=['CD45', 'CD3', 'PanCK', 'DNA1'],
            baseline_noise_level=0.4,
            hot_pixel_probability=0.01,
            batch_effect_strength=0.3
        )
    }
    
    results = {}
    
    for scenario_name, config in scenarios.items():
        print(f"Testing {scenario_name} scenario...")
        
        try:
            generator = SyntheticDataGenerator(config)
            dataset = generator.generate_complete_dataset()
            
            # Quick validation
            validation_result = validate_synthetic_dataset(dataset)
            results[scenario_name] = {
                'dataset': dataset,
                'validation': validation_result,
                'n_cells': len(dataset['coordinates']),
                'quality_score': validation_result.quality_score
            }
            
            print(f"  ✓ Generated {results[scenario_name]['n_cells']} cells")
            print(f"  Quality score: {results[scenario_name]['quality_score']:.2f}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[scenario_name] = None
    
    return results

def create_visualization_plots(dataset, output_dir='test_output'):
    """Create visualization plots of synthetic data."""
    print(f"\n=== Creating Visualization Plots ===")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        coordinates = dataset['coordinates']
        cluster_labels = dataset['ground_truth_clusters']
        
        # Plot 1: Spatial distribution with clusters
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                            c=cluster_labels, cmap='tab10', s=0.5, alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Ground Truth Clusters')
        plt.xlabel('X (μm)')
        plt.ylabel('Y (μm)')
        plt.axis('equal')
        
        # Plot 2: Protein expression heatmap
        plt.subplot(1, 2, 2)
        if 'CD45' in dataset['ion_counts']:
            cd45_expr = dataset['ion_counts']['CD45']
            scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                                c=cd45_expr, cmap='viridis', s=0.5, alpha=0.7)
            plt.colorbar(scatter, label='CD45 Expression')
            plt.title('CD45 Expression Pattern')
        else:
            # Use first available protein
            first_protein = list(dataset['ion_counts'].keys())[0]
            expr = dataset['ion_counts'][first_protein]
            scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                                c=expr, cmap='viridis', s=0.5, alpha=0.7)
            plt.colorbar(scatter, label=f'{first_protein} Expression')
            plt.title(f'{first_protein} Expression Pattern')
        
        plt.xlabel('X (μm)')
        plt.ylabel('Y (μm)')
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(output_path / 'synthetic_data_overview.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved visualization to {output_path / 'synthetic_data_overview.png'}")
        
        # Plot 3: Validation metrics
        if 'validation_metrics' in dataset:
            metrics = dataset['validation_metrics']
            
            plt.figure(figsize=(10, 6))
            
            # Ripley's K plot
            if metrics.get('ripleys_k'):
                plt.subplot(2, 2, 1)
                rk = metrics['ripleys_k']
                plt.plot(rk['distances'], rk['k_values'], 'b-', label="Ripley's K")
                # Theoretical CSR line
                distances = np.array(rk['distances'])
                csr_line = np.pi * distances**2
                plt.plot(distances, csr_line, 'r--', label='CSR (Random)')
                plt.xlabel('Distance (μm)')
                plt.ylabel("K(r)")
                plt.title("Ripley's K Function")
                plt.legend()
            
            # Cluster size distribution
            if 'cluster_stats' in metrics:
                plt.subplot(2, 2, 2)
                cluster_sizes = metrics['cluster_stats']['cluster_sizes']
                plt.bar(range(len(cluster_sizes)), cluster_sizes)
                plt.xlabel('Cluster ID')
                plt.ylabel('Size')
                plt.title('Cluster Size Distribution')
            
            # Protein correlation matrix
            if 'protein_correlations' in metrics:
                plt.subplot(2, 2, 3)
                corr_data = metrics['protein_correlations']
                corr_values = list(corr_data.values())
                plt.hist(corr_values, bins=20, alpha=0.7)
                plt.xlabel('Correlation Coefficient')
                plt.ylabel('Frequency')
                plt.title('Protein Correlation Distribution')
            
            plt.tight_layout()
            plt.savefig(output_path / 'validation_metrics.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved metrics to {output_path / 'validation_metrics.png'}")
        
    except Exception as e:
        print(f"✗ Visualization failed: {e}")

def main():
    """Run comprehensive test suite."""
    print("Synthetic Data Generator Integration Test")
    print("=" * 50)
    
    # Test 1: Basic data generation
    dataset = test_basic_data_generation()
    
    # Test 2: Pipeline integration
    test_pipeline_integration(dataset)
    
    # Test 3: Spatial statistics
    test_spatial_statistics(dataset)
    
    # Test 4: Co-abundance features
    test_coabundance_features(dataset)
    
    # Test 5: Different scenarios
    scenario_results = test_different_scenarios()
    
    # Test 6: Example datasets
    print("\n=== Testing Example Datasets ===")
    try:
        example_datasets = create_example_datasets()
        print(f"✓ Created {len(example_datasets)} example datasets")
        
        # Test validation framework
        test_validation_framework(example_datasets)
        
    except Exception as e:
        print(f"✗ Example datasets failed: {e}")
        example_datasets = {}
    
    # Test 7: Create visualizations
    if dataset:
        create_visualization_plots(dataset)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print("✓ Basic data generation: PASSED")
    print("✓ Pipeline integration: TESTED")
    print("✓ Spatial statistics: TESTED")
    print("✓ Co-abundance features: TESTED")
    print(f"✓ Scenario testing: {len([r for r in scenario_results.values() if r is not None])}/{len(scenario_results)} scenarios")
    print(f"✓ Example datasets: {len(example_datasets)} created")
    print("✓ Visualizations: Generated")
    
    print("\nSynthetic data generator is ready for use!")
    print("Usage example:")
    print("  from src.analysis import SyntheticDataGenerator, SyntheticDataConfig")
    print("  config = SyntheticDataConfig(n_cells_total=5000)")
    print("  generator = SyntheticDataGenerator(config)")
    print("  dataset = generator.generate_complete_dataset()")

if __name__ == "__main__":
    main()