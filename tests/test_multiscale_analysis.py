"""
Test suite for multiscale analysis module.

Tests the multi-scale spatial analysis functionality that compares tissue
microenvironments at 10μm, 20μm, and 40μm spatial scales.
"""

import numpy as np
import pytest
from typing import Dict, List

from src.analysis.multiscale_analysis import (
    perform_multiscale_analysis,
    compute_scale_consistency,
    compute_adjusted_rand_index,
    compute_normalized_mutual_info,
    identify_scale_dependent_features,
    summarize_multiscale_analysis
)


class TestMultiscaleAnalysis:
    """Test multiscale analysis functionality."""
    
    def setup_method(self):
        """Setup realistic IMC data for multiscale testing."""
        np.random.seed(42)
        
        # Create 200x200 μm tissue region (reduced size for speed)
        self.coords = np.random.uniform(0, 200, (1000, 2))
        
        # Create realistic protein expression with spatial patterns
        self.ion_counts = {
            'CD45': np.random.poisson(8, 1000),    # Pan-leukocyte
            'CD11b': np.random.poisson(3, 1000),   # Myeloid  
            'CD206': np.random.poisson(5, 1000),   # M2 macrophages
        }
        
        # Add spatial structure - immune infiltration in one region
        immune_region = (self.coords[:, 0] < 100) & (self.coords[:, 1] < 100)
        self.ion_counts['CD45'][immune_region] += np.random.poisson(10, np.sum(immune_region))
        self.ion_counts['CD11b'][immune_region] += np.random.poisson(5, np.sum(immune_region))
        
        # DNA channels for morphological information
        self.dna1_intensities = np.random.gamma(2, 50, 1000)  # Typical DNA distribution
        self.dna2_intensities = np.random.gamma(2, 45, 1000)
        
        # Standard analysis parameters
        self.scales_um = [10.0, 20.0, 40.0]
        self.n_clusters = 5
    
    def test_perform_multiscale_analysis_basic(self):
        """Test basic multiscale analysis execution."""
        results = perform_multiscale_analysis(
            coords=self.coords,
            ion_counts=self.ion_counts,
            dna1_intensities=self.dna1_intensities,
            dna2_intensities=self.dna2_intensities,
            scales_um=self.scales_um,
            n_clusters=self.n_clusters,
            use_slic=False  # Use square binning for predictable structure
        )
        
        # Check that results exist for all scales
        assert isinstance(results, dict)
        assert set(results.keys()) == set(self.scales_um)
        
        # Check structure for each scale
        for scale_um, result in results.items():
            assert 'scale_um' in result
            assert result['scale_um'] == scale_um
            assert 'method' in result
            assert result['method'] == 'square'  # We're using square binning
            
            # Should have basic analysis components  
            # Check for actual keys that are commonly present
            assert 'aggregated_counts' in result or 'superpixel_coords' in result, f"Missing aggregated data for scale {scale_um}"
            assert 'method' in result
            assert 'scale_um' in result
    
    def test_perform_multiscale_analysis_different_methods(self):
        """Test multiscale analysis with different methods."""
        # Test SLIC method
        slic_results = perform_multiscale_analysis(
            self.coords, self.ion_counts, self.dna1_intensities, self.dna2_intensities,
            scales_um=[20.0], n_clusters=3, use_slic=True
        )
        
        # Test square binning method
        square_results = perform_multiscale_analysis(
            self.coords, self.ion_counts, self.dna1_intensities, self.dna2_intensities,
            scales_um=[20.0], n_clusters=3, use_slic=False
        )
        
        # Both should complete successfully
        assert 20.0 in slic_results
        assert 20.0 in square_results
        
        # Methods should be labeled correctly
        assert slic_results[20.0]['method'] == 'slic'
        assert square_results[20.0]['method'] == 'square'
    
    def test_compute_scale_consistency(self):
        """Test scale consistency computation."""
        # Run multiscale analysis with square binning to ensure cluster_map exists
        multiscale_results = perform_multiscale_analysis(
            self.coords, self.ion_counts, self.dna1_intensities, self.dna2_intensities,
            scales_um=self.scales_um, n_clusters=self.n_clusters, use_slic=False
        )
        
        # Compute consistency
        consistency_results = compute_scale_consistency(
            multiscale_results,
            consistency_metrics=['ari', 'nmi', 'cluster_stability']
        )
        
        # Check structure
        assert isinstance(consistency_results, dict)
        assert 'overall' in consistency_results
        
        # Check pairwise comparisons
        scales = sorted(multiscale_results.keys())
        for i, scale1 in enumerate(scales):
            for scale2 in scales[i+1:]:
                pair_key = f"{scale1}um_vs_{scale2}um"
                assert pair_key in consistency_results
                
                pair_metrics = consistency_results[pair_key]
                assert 'adjusted_rand_index' in pair_metrics
                assert 'normalized_mutual_info' in pair_metrics
                assert 'centroid_stability' in pair_metrics
        
        # Check overall metrics
        overall = consistency_results['overall']
        assert 'mean_ari' in overall
        assert 'mean_nmi' in overall
    
    def test_compute_adjusted_rand_index(self):
        """Test ARI computation between scales."""
        # Create mock results with cluster maps
        result1 = {
            'cluster_map': np.random.randint(0, 3, (50, 50))
        }
        result2 = {
            'cluster_map': np.random.randint(0, 3, (50, 50))
        }
        
        # Compute ARI
        ari = compute_adjusted_rand_index(result1, result2)
        
        # ARI should be a valid score
        assert isinstance(ari, float)
        assert -1 <= ari <= 1
        
        # Test with identical maps
        identical_result = {
            'cluster_map': result1['cluster_map'].copy()
        }
        identical_ari = compute_adjusted_rand_index(result1, identical_result)
        assert identical_ari == pytest.approx(1.0, abs=1e-10)
    
    def test_compute_normalized_mutual_info(self):
        """Test NMI computation between scales."""
        # Create mock results
        result1 = {
            'cluster_map': np.random.randint(0, 4, (30, 30))
        }
        result2 = {
            'cluster_map': np.random.randint(0, 4, (30, 30))
        }
        
        nmi = compute_normalized_mutual_info(result1, result2)
        
        # NMI should be valid
        assert isinstance(nmi, float)
        assert 0 <= nmi <= 1
        
        # Test with identical maps
        identical_nmi = compute_normalized_mutual_info(result1, result1)
        assert identical_nmi == pytest.approx(1.0, abs=1e-10)
    
    def test_identify_scale_dependent_features(self):
        """Test identification of scale-dependent protein features."""
        # Run multiscale analysis
        multiscale_results = perform_multiscale_analysis(
            self.coords, self.ion_counts, self.dna1_intensities, self.dna2_intensities,
            scales_um=self.scales_um, n_clusters=self.n_clusters
        )
        
        # Identify scale-dependent features
        protein_names = list(self.ion_counts.keys())
        scale_features = identify_scale_dependent_features(multiscale_results, protein_names)
        
        # Check structure
        assert isinstance(scale_features, dict)
        assert set(scale_features.keys()) == set(protein_names)
        
        # Each protein should have metrics for each scale
        for protein_name, scale_metrics in scale_features.items():
            assert isinstance(scale_metrics, dict)
            # Should have metrics for scales where data exists
            for scale_um in self.scales_um:
                if scale_um in scale_metrics:
                    assert isinstance(scale_metrics[scale_um], float)
                    assert scale_metrics[scale_um] >= 0  # CV should be non-negative
    
    def test_summarize_multiscale_analysis(self):
        """Test multiscale analysis summary generation."""
        # Run full analysis pipeline
        multiscale_results = perform_multiscale_analysis(
            self.coords, self.ion_counts, self.dna1_intensities, self.dna2_intensities,
            scales_um=self.scales_um, n_clusters=self.n_clusters
        )
        
        consistency_results = compute_scale_consistency(multiscale_results)
        
        # Create summary
        summary = summarize_multiscale_analysis(multiscale_results, consistency_results)
        
        # Check summary structure
        assert isinstance(summary, dict)
        assert 'scales_analyzed' in summary
        assert 'consistency_summary' in summary
        assert 'scale_specific_summaries' in summary
        
        # Check scales
        assert summary['scales_analyzed'] == sorted(self.scales_um)
        
        # Check scale-specific summaries
        scale_summaries = summary['scale_specific_summaries']
        for scale_um in self.scales_um:
            assert scale_um in scale_summaries
            scale_summary = scale_summaries[scale_um]
            
            assert 'n_clusters_found' in scale_summary
            assert 'n_spatial_bins' in scale_summary
            assert 'method_used' in scale_summary
            
            # Values should be reasonable
            assert isinstance(scale_summary['n_clusters_found'], int)
            assert scale_summary['n_clusters_found'] >= 0
            assert isinstance(scale_summary['n_spatial_bins'], int)
            assert scale_summary['n_spatial_bins'] >= 0


class TestMultiscaleAnalysisEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_coords = np.array([]).reshape(0, 2)
        empty_ion_counts = {}
        empty_dna = np.array([])
        
        results = perform_multiscale_analysis(
            empty_coords, empty_ion_counts, empty_dna, empty_dna,
            scales_um=[20.0], n_clusters=3
        )
        
        # Should handle empty data gracefully
        assert isinstance(results, dict)
        if 20.0 in results:
            # Should have some indication of empty processing
            assert results[20.0] is not None
    
    def test_single_scale_analysis(self):
        """Test analysis with single scale."""
        coords = np.random.uniform(0, 100, (500, 2))
        ion_counts = {'CD45': np.random.poisson(5, 500)}
        dna1 = np.random.gamma(2, 50, 500)
        dna2 = np.random.gamma(2, 45, 500)
        
        results = perform_multiscale_analysis(
            coords, ion_counts, dna1, dna2,
            scales_um=[20.0], n_clusters=3
        )
        
        assert 20.0 in results
        assert results[20.0]['scale_um'] == 20.0
    
    def test_consistency_with_insufficient_scales(self):
        """Test consistency computation with insufficient scales."""
        # Single scale result
        single_scale_results = {
            20.0: {'cluster_map': np.random.randint(0, 3, (10, 10))}
        }
        
        consistency = compute_scale_consistency(single_scale_results)
        
        # Should handle gracefully
        assert isinstance(consistency, dict)
        # No pairwise comparisons possible, but overall should exist
        assert 'overall' in consistency
    
    def test_different_cluster_map_sizes(self):
        """Test consistency computation with different sized cluster maps."""
        result1 = {'cluster_map': np.random.randint(0, 3, (20, 20))}
        result2 = {'cluster_map': np.random.randint(0, 3, (40, 40))}
        
        # Should handle different sizes via resampling
        ari = compute_adjusted_rand_index(result1, result2)
        nmi = compute_normalized_mutual_info(result1, result2)
        
        assert isinstance(ari, float)
        assert isinstance(nmi, float)
        assert not np.isnan(ari)
        assert not np.isnan(nmi)


if __name__ == "__main__":
    pytest.main([__file__])