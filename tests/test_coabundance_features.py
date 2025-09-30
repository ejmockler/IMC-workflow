"""
Tests for co-abundance feature engineering.

Critical path protection: This module generates 153 features from 9 proteins.
A silent bug here poisons every downstream result.
"""

import numpy as np
import pytest
from scipy.spatial import KDTree

from src.analysis.coabundance_features import (
    generate_coabundance_features,
    compute_local_covariance,
    select_informative_coabundance_features,
    compute_protein_modules,
    create_hierarchical_features
)


class TestCoabundanceFeatureGeneration:
    """Test the core combinatorial logic that expands 9→153 features."""
    
    @pytest.fixture
    def sample_protein_data(self):
        """Standard 9-protein IMC panel."""
        np.random.seed(42)
        n_samples = 100
        
        # Realistic protein names from config
        protein_names = ['CD45', 'CD11b', 'Ly6G', 'CD140a', 'CD140b', 
                        'CD31', 'CD34', 'CD206', 'CD44']
        
        # Create protein expression matrix with some spatial structure
        feature_matrix = np.random.gamma(2, 3, (n_samples, 9))
        
        # Add spatial coordinates
        spatial_coords = np.random.uniform(0, 100, (n_samples, 2))
        
        return feature_matrix, protein_names, spatial_coords
    
    def test_feature_count_invariants(self, sample_protein_data):
        """Test the core combinatorial mathematics: 9 proteins → 153 features."""
        feature_matrix, protein_names, spatial_coords = sample_protein_data
        
        enriched_features, feature_names = generate_coabundance_features(
            feature_matrix, protein_names, spatial_coords,
            interaction_order=2,
            include_ratios=True,
            include_products=True,
            include_spatial_covariance=True
        )
        
        # Core invariant: exact feature count
        n_proteins = len(protein_names)  # 9
        
        expected_features = (
            n_proteins +                           # 9 original proteins
            (n_proteins * (n_proteins - 1) // 2) + # 36 pairwise products: C(9,2)
            (n_proteins * (n_proteins - 1)) +      # 72 ratios: 9×8  
            (n_proteins * (n_proteins - 1) // 2)   # 36 local covariances: C(9,2)
        )
        # Total: 9 + 36 + 72 + 36 = 153
        
        assert enriched_features.shape[1] == expected_features == 153
        assert len(feature_names) == 153
        assert enriched_features.shape[0] == feature_matrix.shape[0]  # Same samples
    
    def test_pairwise_products_logic(self, sample_protein_data):
        """Test pairwise product generation: C(9,2) = 36 products."""
        feature_matrix, protein_names, _ = sample_protein_data
        
        enriched_features, feature_names = generate_coabundance_features(
            feature_matrix, protein_names, 
            spatial_coords=None,  # Skip spatial covariance
            include_ratios=False,  # Only test products
            include_products=True,
            include_spatial_covariance=False
        )
        
        # Should have: 9 original + 36 products = 45 features
        assert enriched_features.shape[1] == 9 + 36
        
        # Count product features by name pattern
        product_features = [name for name in feature_names if '*' in name]
        assert len(product_features) == 36
        
        # Verify specific combinations exist
        expected_products = [
            'CD45*CD11b', 'CD45*Ly6G', 'CD31*CD34',  # Sample pairs
            'CD140a*CD140b'  # Fibroblast markers
        ]
        for expected in expected_products:
            assert expected in product_features
    
    def test_ratio_features_logic(self, sample_protein_data):
        """Test ratio generation: 9×8 = 72 ratios."""
        feature_matrix, protein_names, _ = sample_protein_data
        
        enriched_features, feature_names = generate_coabundance_features(
            feature_matrix, protein_names,
            spatial_coords=None,
            include_ratios=True,
            include_products=False,  # Only test ratios
            include_spatial_covariance=False
        )
        
        # Should have: 9 original + 72 ratios = 81 features
        assert enriched_features.shape[1] == 9 + 72
        
        # Count ratio features
        ratio_features = [name for name in feature_names if '/' in name]
        assert len(ratio_features) == 72
        
        # Verify specific ratios exist (ordered pairs)
        expected_ratios = [
            'CD45/CD11b', 'CD11b/CD45',  # Both directions
            'CD31/CD34', 'CD34/CD31'     # Vascular markers
        ]
        for expected in expected_ratios:
            assert expected in ratio_features
    
    def test_spatial_covariance_logic(self, sample_protein_data):
        """Test local spatial covariance: C(9,2) = 36 covariances."""
        feature_matrix, protein_names, spatial_coords = sample_protein_data
        
        enriched_features, feature_names = generate_coabundance_features(
            feature_matrix, protein_names, spatial_coords,
            include_ratios=False,
            include_products=False,
            include_spatial_covariance=True  # Only test covariance
        )
        
        # Should have: 9 original + 36 covariances = 45 features  
        assert enriched_features.shape[1] == 9 + 36
        
        # Count covariance features
        cov_features = [name for name in feature_names if 'cov(' in name]
        assert len(cov_features) == 36
        
        # Verify specific covariances exist
        expected_covs = [
            'cov(CD45,CD11b)', 'cov(CD31,CD34)', 'cov(CD140a,CD140b)'
        ]
        for expected in expected_covs:
            assert expected in cov_features
    
    def test_feature_matrix_properties(self, sample_protein_data):
        """Test mathematical properties of generated features."""
        feature_matrix, protein_names, spatial_coords = sample_protein_data
        
        enriched_features, feature_names = generate_coabundance_features(
            feature_matrix, protein_names, spatial_coords
        )
        
        # All features should be finite
        assert np.all(np.isfinite(enriched_features))
        
        # No NaN values
        assert not np.any(np.isnan(enriched_features))
        
        # Original proteins should be preserved in first 9 columns
        np.testing.assert_array_equal(
            enriched_features[:, :9], feature_matrix
        )
    
    def test_empty_input_handling(self):
        """Test graceful handling of edge cases."""
        # Empty data
        empty_matrix = np.array([]).reshape(0, 9)
        protein_names = ['CD45', 'CD11b', 'Ly6G', 'CD140a', 'CD140b',
                        'CD31', 'CD34', 'CD206', 'CD44']
        empty_coords = np.array([]).reshape(0, 2)
        
        enriched_features, feature_names = generate_coabundance_features(
            empty_matrix, protein_names, empty_coords
        )
        
        # Should handle gracefully
        assert enriched_features.shape[0] == 0
        assert enriched_features.shape[1] == 153  # Still correct feature count
        assert len(feature_names) == 153


class TestLocalCovariance:
    """Test spatial covariance computation separately."""
    
    def test_covariance_spatial_logic(self):
        """Test that local covariance captures spatial relationships."""
        np.random.seed(42)
        
        # Create spatially structured data
        coords = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])  # Two clusters
        
        # Proteins that co-occur in space
        protein_data = np.array([
            [10, 2],   # High CD45, low CD31
            [12, 3],   # High CD45, low CD31  
            [2, 15],   # Low CD45, high CD31
            [1, 14]    # Low CD45, high CD31
        ])
        
        protein_names = ['CD45', 'CD31']
        
        local_cov = compute_local_covariance(
            protein_data, protein_names, coords, radius=5.0
        )
        
        # Should compute covariance for each sample
        assert local_cov.shape == (4, 1)  # 4 samples, 1 protein pair
        
        # Covariances should be finite
        assert np.all(np.isfinite(local_cov))


class TestFeatureSelection:
    """Test selection of most informative features."""
    
    def test_feature_selection_preserves_count(self):
        """Test that selection returns requested number of features."""
        np.random.seed(42)
        
        # Create large feature matrix
        enriched_features = np.random.normal(0, 1, (100, 153))
        feature_names = [f"feature_{i}" for i in range(153)]
        
        # Select top 50 features
        selected_features, selected_names = select_informative_coabundance_features(
            enriched_features, feature_names, target_n_features=50
        )
        
        assert selected_features.shape[1] == 50
        assert len(selected_names) == 50
        assert selected_features.shape[0] == 100  # Same samples
    
    def test_selection_methods(self):
        """Test different selection methods work."""
        np.random.seed(42)
        enriched_features = np.random.normal(0, 1, (50, 153))
        feature_names = [f"feature_{i}" for i in range(153)]
        
        methods = ['variance', 'mutual_info', 'lasso']
        
        for method in methods:
            selected_features, selected_names = select_informative_coabundance_features(
                enriched_features, feature_names, 
                target_n_features=20, method=method
            )
            
            assert selected_features.shape[1] == 20
            assert len(selected_names) == 20


class TestProteinModules:
    """Test protein module discovery."""
    
    def test_module_identification(self):
        """Test that NMF finds co-expressed protein groups."""
        np.random.seed(42)
        
        protein_names = ['CD45', 'CD11b', 'CD31', 'CD34']
        
        # Create data with two modules:
        # Module 1: CD45 + CD11b (immune)
        # Module 2: CD31 + CD34 (vascular)
        n_samples = 100
        
        # Base expression
        feature_matrix = np.random.gamma(1, 1, (n_samples, 4))
        
        # Add module structure
        immune_samples = np.random.choice(n_samples, 50, replace=False)
        feature_matrix[immune_samples, 0] += 5  # CD45
        feature_matrix[immune_samples, 1] += 4  # CD11b
        
        vascular_samples = np.random.choice(n_samples, 40, replace=False)
        feature_matrix[vascular_samples, 2] += 6  # CD31
        feature_matrix[vascular_samples, 3] += 5  # CD34
        
        module_scores, module_compositions = compute_protein_modules(
            feature_matrix, protein_names, n_modules=2
        )
        
        # Should find 2 modules
        assert module_scores.shape[1] == 2
        assert len(module_compositions) == 2
        
        # Each module should have contributing proteins
        for module_name, proteins in module_compositions.items():
            assert len(proteins) > 0
            assert all(protein in protein_names for protein in proteins)


class TestHierarchicalFeatures:
    """Test hierarchical feature generation."""
    
    def test_hierarchical_grouping(self):
        """Test that hierarchical features group proteins correctly."""
        np.random.seed(42)
        
        protein_names = ['CD45', 'CD11b', 'CD31', 'CD34']
        feature_matrix = np.random.gamma(2, 1, (50, 4))
        
        # Define hierarchy
        hierarchy = {
            'immune': ['CD45', 'CD11b'],
            'vascular': ['CD31', 'CD34']
        }
        
        hierarchical_features, hierarchical_names = create_hierarchical_features(
            feature_matrix, protein_names, hierarchy
        )
        
        # Should create features for each group
        expected_features = [
            'immune_mean', 'immune_max', 'immune_heterogeneity',
            'vascular_mean', 'vascular_max', 'vascular_heterogeneity'
        ]
        
        assert len(hierarchical_names) == 6
        for expected in expected_features:
            assert expected in hierarchical_names
        
        assert hierarchical_features.shape == (50, 6)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])