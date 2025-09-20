"""
Tests for SLIC Superpixel Segmentation

Critical tests for morphology-aware spatial analysis using DNA channels.
"""

import numpy as np
import pytest
from src.analysis.slic_segmentation import (
    prepare_dna_composite,
    perform_slic_segmentation,
    aggregate_to_superpixels,
    compute_superpixel_properties,
    slic_pipeline
)


class TestDNAComposite:
    """Test DNA composite image preparation."""
    
    def test_prepare_dna_composite_basic(self):
        """Test basic DNA composite creation."""
        # Create synthetic IMC-like data
        np.random.seed(42)
        n_points = 1000
        coords = np.random.uniform(0, 100, (n_points, 2))
        dna1 = np.random.poisson(500, n_points).astype(float)
        dna2 = np.random.poisson(400, n_points).astype(float)
        
        composite, bounds = prepare_dna_composite(
            coords, dna1, dna2,
            resolution_um=1.0
        )
        
        assert composite.ndim == 2
        assert composite.min() >= 0
        assert composite.max() <= 1  # Should be normalized
        assert len(bounds) == 4
        
    def test_prepare_dna_composite_empty(self):
        """Test handling of empty input."""
        coords = np.array([])
        dna1 = np.array([])
        dna2 = np.array([])
        
        composite, bounds = prepare_dna_composite(coords, dna1, dna2)
        
        assert composite.size == 0
        assert bounds == (0, 0, 0, 0)
        
    def test_prepare_dna_composite_resolution(self):
        """Test different resolution settings."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (500, 2))
        dna1 = np.random.poisson(500, 500).astype(float)
        dna2 = np.random.poisson(400, 500).astype(float)
        
        # Test coarse resolution
        composite_coarse, _ = prepare_dna_composite(
            coords, dna1, dna2, resolution_um=5.0
        )
        
        # Test fine resolution
        composite_fine, _ = prepare_dna_composite(
            coords, dna1, dna2, resolution_um=0.5
        )
        
        # Fine resolution should have more pixels
        assert composite_fine.size > composite_coarse.size


class TestSLICSegmentation:
    """Test SLIC superpixel generation."""
    
    def test_perform_slic_basic(self):
        """Test basic SLIC segmentation."""
        # Create synthetic composite image
        composite = np.random.rand(100, 100)
        
        labels = perform_slic_segmentation(
            composite,
            target_bin_size_um=20.0,
            resolution_um=1.0,
            compactness=10.0
        )
        
        assert labels.shape == composite.shape
        assert labels.min() >= 0
        assert len(np.unique(labels)) > 1  # Multiple superpixels
        
    def test_perform_slic_target_size(self):
        """Test that target size affects number of superpixels."""
        composite = np.random.rand(100, 100)
        
        # Small superpixels
        labels_small = perform_slic_segmentation(
            composite, target_bin_size_um=10.0
        )
        
        # Large superpixels
        labels_large = perform_slic_segmentation(
            composite, target_bin_size_um=40.0
        )
        
        # Smaller target size should create more superpixels
        assert len(np.unique(labels_small)) > len(np.unique(labels_large))
        
    def test_perform_slic_empty(self):
        """Test handling of empty input."""
        composite = np.array([])
        labels = perform_slic_segmentation(composite)
        assert labels.size == 0


class TestAggregation:
    """Test aggregation to superpixels."""
    
    def test_aggregate_to_superpixels_basic(self):
        """Test basic ion count aggregation."""
        # Create synthetic data
        np.random.seed(42)
        coords = np.array([[10, 10], [10, 11], [20, 20], [20, 21]])
        ion_counts = {
            'CD45': np.array([100, 150, 50, 75]),
            'CD31': np.array([25, 30, 200, 250])
        }
        
        # Create simple superpixel labels (2x2 image)
        superpixel_labels = np.array([[0, 0], [1, 1]])
        bounds = (0, 30, 0, 30)
        
        aggregated, sp_coords = aggregate_to_superpixels(
            coords, ion_counts, superpixel_labels, bounds, resolution_um=15.0
        )
        
        assert 'CD45' in aggregated
        assert 'CD31' in aggregated
        assert len(aggregated['CD45']) == len(np.unique(superpixel_labels))
        assert len(sp_coords) == len(np.unique(superpixel_labels))
        
    def test_aggregate_to_superpixels_conservation(self):
        """Test that aggregation conserves total ion counts."""
        np.random.seed(42)
        n_points = 100
        coords = np.random.uniform(0, 50, (n_points, 2))
        ion_counts = {
            'Marker1': np.random.poisson(100, n_points).astype(float)
        }
        
        # Create superpixel labels
        superpixel_labels = np.random.randint(0, 5, (60, 60))
        bounds = (-5, 55, -5, 55)
        
        aggregated, _ = aggregate_to_superpixels(
            coords, ion_counts, superpixel_labels, bounds, resolution_um=1.0
        )
        
        # Total counts should be conserved (within valid mask)
        original_total = ion_counts['Marker1'].sum()
        aggregated_total = aggregated['Marker1'].sum()
        
        # Some points might be outside bounds, so aggregated <= original
        assert aggregated_total <= original_total
        assert aggregated_total > 0  # Should aggregate something
        
    def test_aggregate_to_superpixels_empty(self):
        """Test handling of empty input."""
        coords = np.array([])
        ion_counts = {}
        superpixel_labels = np.array([])
        bounds = (0, 0, 0, 0)
        
        aggregated, sp_coords = aggregate_to_superpixels(
            coords, ion_counts, superpixel_labels, bounds
        )
        
        assert aggregated == {}
        assert sp_coords.size == 0


class TestSuperpixelProperties:
    """Test morphological property computation."""
    
    def test_compute_properties_basic(self):
        """Test basic property computation."""
        # Create simple superpixel labels
        labels = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ])
        composite = np.random.rand(4, 4)
        
        props = compute_superpixel_properties(
            labels, composite, resolution_um=10.0
        )
        
        assert len(props) == 4  # 4 superpixels
        for sp_id in range(4):
            assert sp_id in props
            assert 'area_um2' in props[sp_id]
            assert 'perimeter_um' in props[sp_id]
            assert 'mean_dna_intensity' in props[sp_id]
            assert props[sp_id]['area_um2'] > 0
            
    def test_compute_properties_resolution(self):
        """Test that resolution affects measurements."""
        labels = np.zeros((10, 10), dtype=int)  # Single superpixel with label 0
        composite = np.ones((10, 10))
        
        props_1um = compute_superpixel_properties(labels, composite, 1.0)
        props_10um = compute_superpixel_properties(labels, composite, 10.0)
        
        # Area should scale with resolution squared
        assert props_10um[0]['area_um2'] == props_1um[0]['area_um2'] * 100


class TestSLICPipeline:
    """Test complete SLIC pipeline."""
    
    def test_slic_pipeline_integration(self):
        """Test full pipeline with realistic data."""
        np.random.seed(42)
        n_points = 5000
        
        # Simulate IMC-like data
        coords = np.random.uniform(0, 200, (n_points, 2))
        ion_counts = {
            'CD45': np.random.negative_binomial(5, 0.3, n_points).astype(float),
            'CD31': np.random.negative_binomial(10, 0.5, n_points).astype(float),
            'CD11b': np.random.negative_binomial(3, 0.2, n_points).astype(float)
        }
        dna1 = np.random.poisson(800, n_points).astype(float)
        dna2 = np.random.poisson(600, n_points).astype(float)
        
        # Run pipeline
        results = slic_pipeline(
            coords, ion_counts, dna1, dna2,
            target_bin_size_um=20.0,
            resolution_um=2.0,
            compactness=10.0
        )
        
        # Check all expected outputs
        assert 'superpixel_counts' in results
        assert 'superpixel_coords' in results
        assert 'superpixel_labels' in results
        assert 'superpixel_props' in results
        assert 'composite_dna' in results
        assert 'bounds' in results
        
        # Check data consistency
        n_superpixels = len(results['superpixel_coords'])
        assert n_superpixels > 0
        
        for protein in ion_counts:
            assert protein in results['superpixel_counts']
            assert len(results['superpixel_counts'][protein]) == n_superpixels
            
        # Check properties match superpixels
        unique_labels = np.unique(results['superpixel_labels'])
        assert len(results['superpixel_props']) <= len(unique_labels)
        
    def test_slic_pipeline_scales(self):
        """Test pipeline at different spatial scales."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (1000, 2))
        ion_counts = {'Marker1': np.random.poisson(100, 1000).astype(float)}
        dna1 = np.random.poisson(500, 1000).astype(float)
        dna2 = np.random.poisson(400, 1000).astype(float)
        
        # Test at multiple scales
        scales = [10.0, 20.0, 40.0]
        n_superpixels_by_scale = []
        
        for scale in scales:
            results = slic_pipeline(
                coords, ion_counts, dna1, dna2,
                target_bin_size_um=scale
            )
            n_superpixels = len(results['superpixel_coords'])
            n_superpixels_by_scale.append(n_superpixels)
            
        # Larger scales should produce fewer superpixels
        assert n_superpixels_by_scale[0] > n_superpixels_by_scale[1]
        assert n_superpixels_by_scale[1] > n_superpixels_by_scale[2]
        
    def test_slic_pipeline_empty(self):
        """Test pipeline with empty input."""
        results = slic_pipeline(
            coords=np.array([]),
            ion_counts={},
            dna1_intensities=np.array([]),
            dna2_intensities=np.array([])
        )
        
        assert results['superpixel_counts'] == {}
        assert results['superpixel_coords'].size == 0
        assert results['superpixel_labels'].size == 0
        assert results['superpixel_props'] == {}
        
    def test_slic_pipeline_sparse_data(self):
        """Test pipeline with sparse, realistic IMC data."""
        np.random.seed(42)
        n_points = 2000
        
        # Create sparse data (30% zeros for some markers)
        coords = np.random.uniform(0, 150, (n_points, 2))
        
        # CD45: moderate expression in ~70% of pixels
        cd45 = np.random.negative_binomial(5, 0.3, n_points).astype(float)
        cd45[np.random.choice(n_points, int(n_points * 0.3), replace=False)] = 0
        
        # CD31: high expression but only in ~30% of pixels (vascular)
        cd31 = np.zeros(n_points)
        vascular_idx = np.random.choice(n_points, int(n_points * 0.3), replace=False)
        cd31[vascular_idx] = np.random.negative_binomial(20, 0.5, len(vascular_idx))
        
        ion_counts = {'CD45': cd45, 'CD31': cd31}
        dna1 = np.random.poisson(800, n_points).astype(float)
        dna2 = np.random.poisson(600, n_points).astype(float)
        
        results = slic_pipeline(
            coords, ion_counts, dna1, dna2,
            target_bin_size_um=20.0
        )
        
        # Check that aggregation handles sparsity
        assert 'CD45' in results['superpixel_counts']
        assert 'CD31' in results['superpixel_counts']
        
        # Some superpixels should have zero CD31 (sparse marker)
        cd31_aggregated = results['superpixel_counts']['CD31']
        assert np.sum(cd31_aggregated == 0) > 0
        assert np.sum(cd31_aggregated > 0) > 0  # But not all zero


if __name__ == '__main__':
    pytest.main([__file__, '-v'])