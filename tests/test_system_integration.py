"""
Integration Tests for Cross-System Functionality

Tests the complete integration of Phase 2D and Phase 1B systems,
ensuring all components work together seamlessly and provide
accurate, reproducible results.
"""

import pytest
import numpy as np
import warnings
from typing import Dict, Any, List
from pathlib import Path

# Import the integration system
try:
    from src.analysis.system_integration import (
        SystemIntegrator, MethodFactory, IntegrationConfig, IntegrationMethod,
        EvaluationMode, create_system_integrator
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    pytestmark = pytest.mark.skip("System integration not available")

# Test utilities
def create_mock_data(n_cells: int = 1000) -> Dict[str, Any]:
    """Create mock IMC data for testing."""
    np.random.seed(42)
    
    coords = np.random.uniform(0, 500, (n_cells, 2))
    
    protein_names = ['CD45', 'CD3', 'CD4', 'CD8', 'CD20', 'CD68', 'PanCK', 'Vimentin']
    ion_counts = {}
    
    # Create realistic ion count distributions
    for i, protein in enumerate(protein_names):
        # Different proteins have different expression patterns
        base_level = 50 + i * 20
        noise_level = base_level * 0.3
        counts = np.random.poisson(base_level, n_cells) + np.random.normal(0, noise_level, n_cells)
        ion_counts[protein] = np.maximum(counts, 0).astype(float)
    
    dna1 = np.random.poisson(200, n_cells).astype(float)
    dna2 = np.random.poisson(180, n_cells).astype(float)
    
    return {
        'coords': coords,
        'ion_counts': ion_counts,
        'dna1_intensities': dna1,
        'dna2_intensities': dna2
    }


def create_mock_ground_truth(n_cells: int = 1000, n_clusters: int = 5) -> Dict[str, Any]:
    """Create mock ground truth data for testing."""
    np.random.seed(42)
    
    # Create spatially coherent clusters
    cluster_labels = np.random.randint(0, n_clusters, n_cells)
    
    return {
        'ground_truth_clusters': cluster_labels,
        'n_clusters': n_clusters,
        'cluster_sizes': [np.sum(cluster_labels == i) for i in range(n_clusters)]
    }


@pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration system not available")
class TestSystemIntegrator:
    """Test the main SystemIntegrator class."""
    
    def test_integrator_initialization(self):
        """Test integrator initialization."""
        integrator = create_system_integrator()
        assert integrator is not None
        assert isinstance(integrator, SystemIntegrator)
        assert integrator.config is not None
        assert integrator.method_factory is not None
    
    def test_integrator_with_config(self):
        """Test integrator with custom configuration."""
        config = IntegrationConfig(
            methods_to_evaluate=[IntegrationMethod.SLIC, IntegrationMethod.GRID],
            use_synthetic_validation=False,
            quality_threshold=0.7
        )
        
        integrator = create_system_integrator(config)
        assert integrator.config.quality_threshold == 0.7
        assert len(integrator.config.methods_to_evaluate) == 2
        assert not integrator.config.use_synthetic_validation
    
    def test_method_factory_initialization(self):
        """Test method factory initialization."""
        factory = MethodFactory()
        assert factory is not None
        assert factory.config is None  # No config provided
    
    @pytest.mark.parametrize("method", [
        IntegrationMethod.SLIC,
        IntegrationMethod.GRID,
        IntegrationMethod.WATERSHED,
        IntegrationMethod.GRAPH_LEIDEN
    ])
    def test_individual_methods(self, method):
        """Test individual method execution."""
        data = create_mock_data(500)  # Smaller dataset for faster testing
        factory = MethodFactory()
        
        try:
            result = factory.run_method(method, data)
            
            # Basic result validation
            assert isinstance(result, dict)
            assert 'method' in result
            assert result['method'] == method.value
            
            # Check for expected outputs
            if method in [IntegrationMethod.SLIC, IntegrationMethod.GRID, IntegrationMethod.WATERSHED]:
                assert 'superpixel_labels' in result or 'cell_labels' in result
                assert 'superpixel_coords' in result or 'cell_coords' in result
            elif method in [IntegrationMethod.GRAPH_LEIDEN, IntegrationMethod.GRAPH_LOUVAIN]:
                assert 'cluster_labels' in result
            
        except RuntimeError as e:
            if "not available" in str(e):
                pytest.skip(f"Method {method.value} dependencies not available")
            else:
                raise
    
    @pytest.mark.slow
    def test_comprehensive_evaluation(self):
        """Test comprehensive method evaluation."""
        data = create_mock_data(800)
        ground_truth = create_mock_ground_truth(800)
        
        config = IntegrationConfig(
            methods_to_evaluate=[IntegrationMethod.SLIC, IntegrationMethod.GRID],
            use_synthetic_validation=False,  # Skip for faster testing
            use_ablation_studies=False
        )
        
        integrator = create_system_integrator(config)
        
        result = integrator.evaluate_all_methods(
            data['coords'],
            data['ion_counts'],
            data['dna1_intensities'],
            data['dna2_intensities'],
            ground_truth_data=ground_truth
        )
        
        # Validate result structure
        assert isinstance(result.method_results, dict)
        assert isinstance(result.evaluation_metrics, dict)
        assert isinstance(result.method_ranking, list)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.metadata, dict)
        
        # Check metadata
        assert 'evaluation_time' in result.metadata
        assert 'n_methods_tested' in result.metadata
        assert 'data_size' in result.metadata
        assert result.metadata['data_size'] == 800
    
    @pytest.mark.slow
    def test_method_ranking(self):
        """Test method ranking functionality."""
        data = create_mock_data(600)
        
        config = IntegrationConfig(
            methods_to_evaluate=[IntegrationMethod.SLIC, IntegrationMethod.GRID],
            use_synthetic_validation=False,
            use_ablation_studies=False
        )
        
        integrator = create_system_integrator(config)
        
        result = integrator.evaluate_all_methods(
            data['coords'],
            data['ion_counts'],
            data['dna1_intensities'],
            data['dna2_intensities']
        )
        
        # Validate ranking
        assert len(result.method_ranking) >= 1
        for method_name, score in result.method_ranking:
            assert isinstance(method_name, str)
            assert isinstance(score, (int, float))
            assert 0 <= score <= 1  # Assuming normalized scores
    
    def test_validation_framework(self):
        """Test validation framework integration."""
        data = create_mock_data(500)
        
        integrator = create_system_integrator()
        
        # Check validators are initialized
        assert hasattr(integrator, 'validators')
        assert isinstance(integrator.validators, list)
        
        # Test validation execution
        mock_results = {
            'test_method': {
                'superpixel_labels': np.random.randint(0, 5, 500),
                'superpixel_coords': data['coords'],
                'method': 'test_method'
            }
        }
        
        validation_results = integrator._run_validation_suite(mock_results, None)
        assert isinstance(validation_results, list)
    
    def test_statistical_analysis(self):
        """Test statistical analysis functionality."""
        integrator = create_system_integrator()
        
        # Mock evaluation metrics
        evaluation_metrics = {
            'method_a': {'boundary_quality': 0.8, 'silhouette_score': 0.6},
            'method_b': {'boundary_quality': 0.7, 'silhouette_score': 0.5},
            'method_c': {'boundary_quality': 0.9, 'silhouette_score': 0.7}
        }
        
        analysis = integrator._perform_statistical_analysis(evaluation_metrics)
        
        assert isinstance(analysis, dict)
        if 'pairwise_comparisons' in analysis:
            assert isinstance(analysis['pairwise_comparisons'], dict)
        if 'score_distribution' in analysis:
            assert 'mean' in analysis['score_distribution']
            assert 'std' in analysis['score_distribution']


@pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration system not available")
class TestSyntheticDataIntegration:
    """Test synthetic data generation and validation."""
    
    def test_synthetic_dataset_generation(self):
        """Test synthetic dataset generation."""
        integrator = create_system_integrator()
        
        try:
            dataset = integrator.generate_synthetic_validation_dataset(
                n_cells=1000,
                roi_size_um=(500.0, 500.0),
                tissue_type='simple'
            )
            
            # Validate dataset structure
            assert 'coordinates' in dataset
            assert 'ion_counts' in dataset
            assert 'ground_truth_clusters' in dataset
            assert 'dna1_intensities' in dataset
            assert 'dna2_intensities' in dataset
            
            # Validate data sizes
            n_cells = len(dataset['coordinates'])
            assert len(dataset['ground_truth_clusters']) == n_cells
            assert len(dataset['dna1_intensities']) == n_cells
            
            for protein, counts in dataset['ion_counts'].items():
                assert len(counts) == n_cells
                
        except RuntimeError as e:
            if "not available" in str(e):
                pytest.skip("Synthetic data generator not available")
            else:
                raise
    
    @pytest.mark.slow
    def test_synthetic_validation_workflow(self):
        """Test complete synthetic validation workflow."""
        integrator = create_system_integrator()
        real_data = create_mock_data(400)
        
        try:
            validation_results = integrator.run_comprehensive_validation(
                real_data['coords'],
                real_data['ion_counts'],
                real_data['dna1_intensities'],
                real_data['dna2_intensities'],
                use_synthetic=True
            )
            
            assert 'real_data' in validation_results
            assert isinstance(validation_results['real_data'], dict)
            
            if 'synthetic_data' in validation_results:
                assert isinstance(validation_results['synthetic_data'], dict)
                
            if 'real_vs_synthetic_comparison' in validation_results:
                comparison = validation_results['real_vs_synthetic_comparison']
                assert isinstance(comparison, dict)
                
        except Exception as e:
            if "not available" in str(e):
                pytest.skip("Synthetic validation not available")
            else:
                raise


@pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration system not available")
class TestReferenceStandards:
    """Test reference standards integration."""
    
    def test_reference_standards_evaluation(self):
        """Test reference standards evaluation."""
        integrator = create_system_integrator()
        data = create_mock_data(500)
        
        reference_results = integrator._evaluate_reference_standards(data)
        
        assert isinstance(reference_results, dict)
        
        # Check for expected components
        possible_components = ['bead_normalization', 'mi_imc_schema', 'single_stain_protocols']
        for component in possible_components:
            if component in reference_results:
                assert isinstance(reference_results[component], dict)
    
    def test_mi_imc_schema_integration(self):
        """Test MI-IMC schema integration."""
        integrator = create_system_integrator()
        
        if integrator.mi_imc_schema:
            assert integrator.mi_imc_schema is not None
            assert hasattr(integrator.mi_imc_schema, 'version')
            
            # Test that schema can be populated
            if hasattr(integrator.mi_imc_schema, 'antibody_panel'):
                # Should have been populated from base config or defaults
                assert isinstance(integrator.mi_imc_schema.antibody_panel, list)


@pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration system not available")
class TestErrorHandling:
    """Test error handling and robustness."""
    
    def test_method_failure_handling(self):
        """Test handling of method execution failures."""
        data = create_mock_data(100)
        factory = MethodFactory()
        
        # Test with invalid method (should raise ValueError)
        with pytest.raises(ValueError):
            factory.run_method("invalid_method", data)
    
    def test_empty_data_handling(self):
        """Test handling of empty or invalid data."""
        empty_data = {
            'coords': np.array([]).reshape(0, 2),
            'ion_counts': {},
            'dna1_intensities': np.array([]),
            'dna2_intensities': np.array([])
        }
        
        integrator = create_system_integrator()
        
        # Should not crash, but may produce warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = integrator.evaluate_all_methods(
                empty_data['coords'],
                empty_data['ion_counts'],
                empty_data['dna1_intensities'],
                empty_data['dna2_intensities']
            )
        
        assert isinstance(result.method_results, dict)
        assert isinstance(result.recommendations, list)
    
    def test_partial_system_availability(self):
        """Test behavior when some systems are not available."""
        # This test ensures the integration still works when some components are missing
        config = IntegrationConfig(
            methods_to_evaluate=[IntegrationMethod.SLIC],  # Only test available method
            use_synthetic_validation=False,
            use_ablation_studies=False
        )
        
        integrator = create_system_integrator(config)
        data = create_mock_data(200)
        
        # Should work even with limited functionality
        result = integrator.evaluate_all_methods(
            data['coords'],
            data['ion_counts'],
            data['dna1_intensities'],
            data['dna2_intensities']
        )
        
        assert result is not None
        assert isinstance(result.method_results, dict)


@pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration system not available")
class TestPerformance:
    """Test performance characteristics of integration system."""
    
    @pytest.mark.slow
    def test_evaluation_performance(self):
        """Test evaluation performance on larger datasets."""
        # Test with larger dataset to check performance
        data = create_mock_data(2000)
        
        config = IntegrationConfig(
            methods_to_evaluate=[IntegrationMethod.SLIC, IntegrationMethod.GRID],
            use_synthetic_validation=False,
            use_ablation_studies=False
        )
        
        integrator = create_system_integrator(config)
        
        import time
        start_time = time.time()
        
        result = integrator.evaluate_all_methods(
            data['coords'],
            data['ion_counts'],
            data['dna1_intensities'],
            data['dna2_intensities']
        )
        
        execution_time = time.time() - start_time
        
        # Check that evaluation completed in reasonable time (< 60 seconds)
        assert execution_time < 60, f"Evaluation took too long: {execution_time:.2f} seconds"
        
        # Check that timing was recorded
        assert 'evaluation_time' in result.metadata
        assert result.metadata['evaluation_time'] > 0
    
    @pytest.mark.slow
    def test_memory_usage(self):
        """Test memory usage characteristics."""
        data = create_mock_data(1500)
        
        config = IntegrationConfig(
            methods_to_evaluate=[IntegrationMethod.SLIC],
            use_synthetic_validation=False,
            use_ablation_studies=False
        )
        
        integrator = create_system_integrator(config)
        
        # Simple memory test - ensure it doesn't crash
        result = integrator.evaluate_all_methods(
            data['coords'],
            data['ion_counts'],
            data['dna1_intensities'],
            data['dna2_intensities']
        )
        
        assert result is not None
        
        # Check if memory usage was tracked
        for method_name, method_result in result.method_results.items():
            if 'performance_comparison' in method_result:
                perf = method_result['performance_comparison']
                if 'memory_usage_mb' in perf:
                    assert perf['memory_usage_mb'] >= 0


@pytest.mark.slow
def test_integration_example():
    """Test the integration example function."""
    if not INTEGRATION_AVAILABLE:
        pytest.skip("Integration system not available")

    from src.analysis.system_integration import run_integration_example
    
    # Should run without errors
    result = run_integration_example()
    
    assert result is not None
    assert hasattr(result, 'method_results')
    assert hasattr(result, 'method_ranking')
    assert hasattr(result, 'recommendations')


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])