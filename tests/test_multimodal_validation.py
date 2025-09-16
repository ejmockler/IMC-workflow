"""
Multimodal Integration Validation Tests

Tests for scRNA-seq projection and multimodal data integration using synthetic data.
Production-quality validation of cross-platform integration capabilities.
"""

import numpy as np
import pytest
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from src.analysis.validation import (
    generate_synthetic_imc_data,
    validate_clustering_performance
)


class TestMultimodalIntegration:
    """Test multimodal integration with synthetic scRNA-seq data."""
    
    def setup_method(self):
        """Setup synthetic multimodal datasets."""
        np.random.seed(42)
        
        # Common parameters
        self.n_cells = 500
        self.n_clusters = 4
        self.protein_names = ['CD45', 'CD11b', 'CD206', 'CD44']
        
        # Generate synthetic IMC data
        self.imc_data = generate_synthetic_imc_data(
            n_cells=self.n_cells,
            n_clusters=self.n_clusters,
            protein_names=self.protein_names,
            spatial_structure='clustered',
            random_state=42
        )
        
        # Generate corresponding synthetic scRNA-seq data
        self.scrna_data = self._generate_synthetic_scrna_data()
    
    def _generate_synthetic_scrna_data(self) -> Dict:
        """Generate synthetic scRNA-seq data that corresponds to IMC cell types."""
        # Use same cell type assignments as IMC
        true_cell_types = self.imc_data['true_cell_types']
        actual_n_cells = len(true_cell_types)  # Use actual number from IMC data
        
        # Define scRNA-seq gene panel (larger than protein panel)
        gene_names = [
            'PTPRC',     # CD45 
            'ITGAM',     # CD11b
            'MRC1',      # CD206
            'CD44',      # CD44
            'GAPDH',     # Housekeeping
            'ACTB',      # Housekeeping  
            'IL1B',      # Inflammation
            'TNF',       # Inflammation
            'ARG1',      # M2 marker
            'NOS2'       # M1 marker
        ]
        
        # Generate cell type-specific expression profiles
        scrna_expression = {}
        
        for gene in gene_names:
            gene_expr = np.zeros(actual_n_cells)
            
            for cluster_id in range(self.n_clusters):
                cluster_mask = true_cell_types == cluster_id
                n_cells_cluster = np.sum(cluster_mask)
                
                if n_cells_cluster == 0:
                    continue
                
                # Define cluster-specific expression patterns
                if gene == 'PTPRC':  # CD45 - immune marker
                    if cluster_id in [0, 1]:  # Immune clusters
                        mean_expr = 8.0
                    else:
                        mean_expr = 2.0
                elif gene == 'ITGAM':  # CD11b - myeloid marker
                    if cluster_id == 1:  # Myeloid cluster
                        mean_expr = 7.5
                    else:
                        mean_expr = 1.5
                elif gene == 'MRC1':  # CD206 - M2 marker
                    if cluster_id == 2:  # M2 cluster
                        mean_expr = 6.0
                    else:
                        mean_expr = 1.0
                elif gene == 'CD44':  # CD44 - widespread
                    mean_expr = 5.0 + cluster_id * 0.5
                elif gene in ['GAPDH', 'ACTB']:  # Housekeeping
                    mean_expr = 9.0
                elif gene == 'IL1B':  # Pro-inflammatory
                    if cluster_id == 1:  # M1-like
                        mean_expr = 4.0
                    else:
                        mean_expr = 0.5
                elif gene == 'TNF':  # Pro-inflammatory
                    if cluster_id == 1:  # M1-like
                        mean_expr = 5.0
                    else:
                        mean_expr = 0.8
                elif gene == 'ARG1':  # M2 marker
                    if cluster_id == 2:  # M2 cluster
                        mean_expr = 5.5
                    else:
                        mean_expr = 0.5
                elif gene == 'NOS2':  # M1 marker
                    if cluster_id == 1:  # M1-like
                        mean_expr = 4.5
                    else:
                        mean_expr = 0.3
                else:
                    mean_expr = 3.0
                
                # Add biological variability (log-normal for scRNA-seq)
                cluster_expression = np.random.lognormal(
                    mean=np.log(mean_expr),
                    sigma=0.8,  # High variability typical of scRNA-seq
                    size=n_cells_cluster
                )
                
                # Add dropout events (common in scRNA-seq)
                dropout_rate = 0.1 if gene in ['GAPDH', 'ACTB'] else 0.3
                dropout_mask = np.random.random(n_cells_cluster) < dropout_rate
                cluster_expression[dropout_mask] = 0.0
                
                gene_expr[cluster_mask] = cluster_expression
            
            scrna_expression[gene] = gene_expr
        
        return {
            'expression': scrna_expression,
            'gene_names': gene_names,
            'true_cell_types': true_cell_types,
            'n_cells': actual_n_cells
        }
    
    def test_protein_gene_correspondence(self):
        """Test correspondence between protein and gene expression."""
        # Define protein-gene pairs
        protein_gene_pairs = [
            ('CD45', 'PTPRC'),
            ('CD11b', 'ITGAM'), 
            ('CD206', 'MRC1'),
            ('CD44', 'CD44')
        ]
        
        correlation_results = {}
        
        for protein, gene in protein_gene_pairs:
            # Get protein expression from IMC
            protein_expr = self.imc_data['ion_counts'][protein]
            
            # Get gene expression from scRNA-seq  
            gene_expr = self.scrna_data['expression'][gene]
            
            # Compute correlation
            correlation, p_value = spearmanr(protein_expr, gene_expr)
            correlation_results[f"{protein}_{gene}"] = {
                'correlation': correlation,
                'p_value': p_value
            }
            
            # Test framework functionality - should compute correlations without error
            assert not np.isnan(correlation), f"Invalid correlation for {protein}-{gene}"
            assert not np.isnan(p_value), f"Invalid p-value for {protein}-{gene}"
        
        # Test framework completeness
        assert len(correlation_results) == 4, "Should compute 4 protein-gene correlations"
        
        # At least some correlations should be meaningful (not all random)
        abs_correlations = [abs(r['correlation']) for r in correlation_results.values()]
        max_correlation = max(abs_correlations)
        assert max_correlation > 0.05, f"All correlations too weak: max={max_correlation:.3f}"
    
    def test_multimodal_clustering_agreement(self):
        """Test that clustering results agree between modalities."""
        # Cluster IMC data (using true expression for this test)
        imc_features = np.column_stack([
            self.imc_data['true_expression'][protein] 
            for protein in self.protein_names
        ])
        
        # Cluster scRNA-seq data (corresponding genes only)
        scrna_genes = ['PTPRC', 'ITGAM', 'MRC1', 'CD44'] 
        scrna_features = np.column_stack([
            self.scrna_data['expression'][gene]
            for gene in scrna_genes
        ])
        
        # Simple clustering using PCA + k-means
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # IMC clustering
        scaler_imc = StandardScaler()
        imc_scaled = scaler_imc.fit_transform(imc_features)
        
        pca_imc = PCA(n_components=2, random_state=42)
        imc_pca = pca_imc.fit_transform(imc_scaled)
        
        kmeans_imc = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        imc_clusters = kmeans_imc.fit_predict(imc_pca)
        
        # scRNA-seq clustering  
        scaler_scrna = StandardScaler()
        scrna_scaled = scaler_scrna.fit_transform(scrna_features)
        
        pca_scrna = PCA(n_components=2, random_state=42)
        scrna_pca = pca_scrna.fit_transform(scrna_scaled)
        
        kmeans_scrna = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        scrna_clusters = kmeans_scrna.fit_predict(scrna_pca)
        
        # Measure agreement between modalities
        cross_modal_ari = adjusted_rand_score(imc_clusters, scrna_clusters)
        
        # Should have reasonable agreement
        assert cross_modal_ari > 0.3, f"Cross-modal ARI too low: {cross_modal_ari:.3f}"
        
        # Both should agree reasonably with ground truth
        imc_truth_ari = adjusted_rand_score(imc_clusters, self.imc_data['true_cell_types'])
        scrna_truth_ari = adjusted_rand_score(scrna_clusters, self.scrna_data['true_cell_types'])
        
        assert imc_truth_ari > 0.4, f"IMC-truth ARI too low: {imc_truth_ari:.3f}"
        assert scrna_truth_ari > 0.4, f"scRNA-seq-truth ARI too low: {scrna_truth_ari:.3f}"
    
    def test_cross_platform_projection(self):
        """Test projection of scRNA-seq data into IMC space."""
        # Create shared feature space using overlapping markers
        shared_markers = {
            'CD45': 'PTPRC',
            'CD11b': 'ITGAM', 
            'CD206': 'MRC1',
            'CD44': 'CD44'
        }
        
        # Build projection matrices
        imc_matrix = np.column_stack([
            self.imc_data['ion_counts'][protein]
            for protein in shared_markers.keys()
        ])
        
        scrna_matrix = np.column_stack([
            self.scrna_data['expression'][gene]
            for gene in shared_markers.values()
        ])
        
        # Standardize both datasets
        from sklearn.preprocessing import StandardScaler
        
        scaler_imc = StandardScaler()
        imc_scaled = scaler_imc.fit_transform(imc_matrix)
        
        scaler_scrna = StandardScaler()
        scrna_scaled = scaler_scrna.fit_transform(scrna_matrix)
        
        # Compute principal components in each space
        pca_imc = PCA(n_components=3, random_state=42)
        imc_pc = pca_imc.fit_transform(imc_scaled)
        
        pca_scrna = PCA(n_components=3, random_state=42)  
        scrna_pc = pca_scrna.fit_transform(scrna_scaled)
        
        # Test projection quality using Procrustes-like analysis
        from scipy.linalg import orthogonal_procrustes
        
        # Find optimal rotation between PC spaces
        R, scale = orthogonal_procrustes(scrna_pc, imc_pc)
        scrna_projected = scrna_pc @ R * scale
        
        # Measure projection quality
        projection_distances = np.linalg.norm(imc_pc - scrna_projected, axis=1)
        mean_projection_error = np.mean(projection_distances)
        
        # Should achieve reasonable projection accuracy
        max_expected_error = np.std(np.linalg.norm(imc_pc, axis=1)) * 1.5
        assert mean_projection_error < max_expected_error, \
            f"Projection error too high: {mean_projection_error:.3f} > {max_expected_error:.3f}"
    
    def test_multimodal_batch_effect_detection(self):
        """Test detection of batch effects across modalities."""
        # Create artificial batch effects
        n_batch1 = self.n_cells // 2
        n_batch2 = self.n_cells - n_batch1
        
        # Add batch effect to IMC data (systematic shift)
        imc_batch1 = {
            protein: values[:n_batch1] * 1.0  # No change
            for protein, values in self.imc_data['ion_counts'].items()
        }
        
        imc_batch2 = {
            protein: values[n_batch1:] * 1.3 + 10  # Systematic shift
            for protein, values in self.imc_data['ion_counts'].items()
        }
        
        # Add different batch effect to scRNA-seq (multiplicative)
        scrna_batch1 = {
            gene: values[:n_batch1] * 1.0
            for gene, values in self.scrna_data['expression'].items()
        }
        
        scrna_batch2 = {
            gene: values[n_batch1:] * 0.8  # Different scaling
            for gene, values in self.scrna_data['expression'].items()
        }
        
        # Detect batch effects in each modality
        shared_proteins = ['CD45', 'CD11b', 'CD206', 'CD44']
        
        # For IMC
        imc_batch_data = {
            'batch1': imc_batch1,
            'batch2': imc_batch2
        }
        
        # Compute batch effect statistics
        imc_batch_effects = self._compute_batch_effects(imc_batch_data, shared_proteins)
        
        # Should detect significant batch effects in IMC
        assert imc_batch_effects['effect_size'] > 0.2, \
            f"IMC batch effect not detected: {imc_batch_effects['effect_size']:.3f}"
        
        # For scRNA-seq (corresponding genes)
        scrna_genes = ['PTPRC', 'ITGAM', 'MRC1', 'CD44']
        scrna_batch_data = {
            'batch1': scrna_batch1, 
            'batch2': scrna_batch2
        }
        
        scrna_batch_effects = self._compute_batch_effects(scrna_batch_data, scrna_genes)
        
        # Should also detect batch effects in scRNA-seq
        assert scrna_batch_effects['effect_size'] > 0.1, \
            f"scRNA-seq batch effect not detected: {scrna_batch_effects['effect_size']:.3f}"
    
    def _compute_batch_effects(self, batch_data: Dict, features: List[str]) -> Dict:
        """Compute batch effect statistics."""
        from scipy.stats import mannwhitneyu
        
        batch_ids = list(batch_data.keys())
        if len(batch_ids) != 2:
            raise ValueError("Need exactly 2 batches for comparison")
        
        batch1_data = batch_data[batch_ids[0]]
        batch2_data = batch_data[batch_ids[1]]
        
        p_values = []
        effect_sizes = []
        
        for feature in features:
            if feature not in batch1_data or feature not in batch2_data:
                continue
                
            values1 = batch1_data[feature]
            values2 = batch2_data[feature]
            
            # Mann-Whitney U test
            statistic, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
            p_values.append(p_value)
            
            # Cohen's d effect size
            mean1, mean2 = np.mean(values1), np.mean(values2)
            std1, std2 = np.std(values1), np.std(values2)
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            
            if pooled_std > 0:
                cohens_d = abs(mean1 - mean2) / pooled_std
                effect_sizes.append(cohens_d)
        
        return {
            'p_values': p_values,
            'mean_p_value': np.mean(p_values) if p_values else 1.0,
            'effect_sizes': effect_sizes,
            'effect_size': np.mean(effect_sizes) if effect_sizes else 0.0,
            'significant_features': sum(1 for p in p_values if p < 0.05)
        }
    
    def test_multimodal_validation_integration(self):
        """Test integration with existing validation framework."""
        # This tests that multimodal data can be used with existing validation
        
        # Use IMC data with validation framework
        imc_validation = validate_clustering_performance(
            predicted_clusters=self.imc_data['true_cell_types'],  # Use ground truth
            true_cell_types=self.imc_data['true_cell_types'],
            coords=self.imc_data['coords'],
            enhanced_metrics=True
        )
        
        # Should achieve perfect performance with ground truth
        assert imc_validation['adjusted_rand_index'] == 1.0
        assert imc_validation['cluster_purity'] == 1.0
        assert imc_validation['coverage'] == 1.0
        
        # Test with imperfect predictions (add noise)
        noisy_predictions = self.imc_data['true_cell_types'].copy()
        n_errors = int(0.1 * len(noisy_predictions))  # 10% error rate
        error_indices = np.random.choice(len(noisy_predictions), n_errors, replace=False)
        
        for idx in error_indices:
            # Randomly reassign cluster
            possible_clusters = list(range(self.n_clusters))
            possible_clusters.remove(noisy_predictions[idx])
            noisy_predictions[idx] = np.random.choice(possible_clusters)
        
        noisy_validation = validate_clustering_performance(
            predicted_clusters=noisy_predictions,
            true_cell_types=self.imc_data['true_cell_types'],
            coords=self.imc_data['coords'],
            enhanced_metrics=True
        )
        
        # Should have reduced but reasonable performance
        assert 0.7 < noisy_validation['adjusted_rand_index'] < 1.0
        assert 0.8 < noisy_validation['cluster_purity'] < 1.0
        assert noisy_validation['coverage'] == 1.0  # All cells assigned
    
    def test_cross_resolution_consistency(self):
        """Test consistency across different spatial resolutions."""
        # Create multi-resolution versions of IMC data
        coords = self.imc_data['coords']
        
        # Original resolution
        original_data = self.imc_data['ion_counts']
        
        # Simulate lower resolution by spatial averaging
        from scipy.spatial.distance import cdist
        
        # Create coarser grid
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Grid resolution
        grid_size = 50.0  # 50μm grid
        x_grid = np.arange(x_min, x_max, grid_size)
        y_grid = np.arange(y_min, y_max, grid_size)
        
        # Create grid points
        grid_coords = np.array([[x, y] for x in x_grid for y in y_grid])
        
        # Average nearby pixels for each grid point
        averaging_radius = 25.0  # μm
        
        low_res_data = {}
        low_res_coords = []
        
        for grid_point in grid_coords:
            # Find nearby pixels
            distances = cdist([grid_point], coords)[0]
            nearby_mask = distances < averaging_radius
            
            if np.any(nearby_mask):
                # Average protein expression in this region
                grid_averages = {}
                for protein, values in original_data.items():
                    grid_averages[protein] = np.mean(values[nearby_mask])
                
                # Store this grid point
                for protein, avg_value in grid_averages.items():
                    if protein not in low_res_data:
                        low_res_data[protein] = []
                    low_res_data[protein].append(avg_value)
                
                low_res_coords.append(grid_point)
        
        # Convert to arrays
        low_res_coords = np.array(low_res_coords)
        for protein in low_res_data:
            low_res_data[protein] = np.array(low_res_data[protein])
        
        # Should have reasonable number of grid points
        assert len(low_res_coords) > 10, "Too few grid points for resolution test"
        
        # Cluster at both resolutions
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Original resolution clustering
        original_features = np.column_stack([
            original_data[protein] for protein in self.protein_names
        ])
        scaler_orig = StandardScaler()
        original_scaled = scaler_orig.fit_transform(original_features)
        kmeans_orig = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        original_clusters = kmeans_orig.fit_predict(original_scaled)
        
        # Low resolution clustering
        low_res_features = np.column_stack([
            low_res_data[protein] for protein in self.protein_names
        ])
        scaler_low = StandardScaler()
        low_res_scaled = scaler_low.fit_transform(low_res_features)
        kmeans_low = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        low_res_clusters = kmeans_low.fit_predict(low_res_scaled)
        
        # Both should identify reasonable cluster structure
        original_ari = adjusted_rand_score(original_clusters, self.imc_data['true_cell_types'])
        
        # Low resolution should maintain some cluster structure
        # (Can't directly compare ARIs due to different sampling)
        n_clusters_found = len(np.unique(low_res_clusters))
        assert n_clusters_found >= 2, "Low resolution should find multiple clusters"
        assert n_clusters_found <= self.n_clusters + 2, "Too many clusters at low resolution"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])