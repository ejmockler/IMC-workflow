"""
Hierarchical FDR Control Under Spatial Dependence

Implements multiple testing correction methods that are valid under
spatial correlation. Standard FDR methods fail catastrophically with
spatial data - this module provides theoretically sound alternatives.

Critical for preventing false discoveries in IMC spatial analyses.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.spatial import distance_matrix


@dataclass 
class FDRConfig:
    """Configuration for spatial FDR control."""
    method: str = 'benjamini_yekutieli'  # 'simes', 'benjamini_yekutieli', 'tree_structured'
    alpha: float = 0.05
    dependence_assumption: str = 'arbitrary'  # 'independent', 'prds', 'arbitrary'
    use_spatial_weights: bool = True
    adaptive_weights: bool = False  # Use IHW or AdaPT
    cache_computations: bool = True


class SpatialFDR:
    """
    FDR control methods valid under spatial dependence.
    
    Implements Benjamini-Yekutieli for arbitrary dependence,
    tree-structured FDR for hierarchical testing, and
    spatial weighting methods.
    """
    
    def __init__(self, config: FDRConfig):
        self.config = config
        self._cache = {} if config.cache_computations else None
        
    def control_fdr(self,
                   p_values: np.ndarray,
                   spatial_coords: Optional[np.ndarray] = None,
                   hierarchy: Optional[Dict[str, Any]] = None,
                   weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Main FDR control interface.
        
        Args:
            p_values: Array of p-values to correct
            spatial_coords: Spatial coordinates for spatial methods
            hierarchy: Hierarchical structure for tree-structured FDR
            weights: Optional weights for weighted FDR
            
        Returns:
            Boolean array of discoveries (True = significant)
        """
        # Input validation
        p_values = np.asarray(p_values)
        n_tests = len(p_values)
        
        if n_tests == 0:
            return np.array([], dtype=bool)
        
        # Check for NaN or invalid p-values
        valid_mask = np.isfinite(p_values) & (p_values >= 0) & (p_values <= 1)
        if not np.all(valid_mask):
            warnings.warn(f"Found {np.sum(~valid_mask)} invalid p-values. Setting to 1.0")
            p_values = p_values.copy()
            p_values[~valid_mask] = 1.0
        
        # Apply appropriate FDR method
        if self.config.method == 'benjamini_yekutieli':
            discoveries = self.benjamini_yekutieli_spatial(
                p_values, spatial_coords, weights
            )
        elif self.config.method == 'simes':
            if self.config.dependence_assumption == 'arbitrary':
                warnings.warn("Simes procedure may not control FDR under arbitrary dependence")
            discoveries = self.simes_spatial(p_values, weights)
        elif self.config.method == 'tree_structured':
            if hierarchy is None:
                raise ValueError("Tree-structured FDR requires hierarchy")
            discoveries = self.tree_structured_fdr(p_values, hierarchy, spatial_coords)
        else:
            raise ValueError(f"Unknown FDR method: {self.config.method}")
        
        return discoveries
    
    def benjamini_yekutieli_spatial(self,
                                   p_values: np.ndarray,
                                   spatial_coords: Optional[np.ndarray] = None,
                                   weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Benjamini-Yekutieli procedure for arbitrary dependence.
        
        Valid under any dependence structure including spatial correlation.
        More conservative than BH but guarantees FDR control.
        
        Args:
            p_values: P-values to correct
            spatial_coords: Optional spatial coordinates for adaptive weights
            weights: Optional importance weights
            
        Returns:
            Boolean array of discoveries
        """
        n = len(p_values)
        
        if n == 0:
            return np.array([], dtype=bool)
        
        # Compute c(n) correction factor for arbitrary dependence
        c_n = np.sum(1.0 / np.arange(1, n + 1))  # Harmonic sum
        
        # Adaptive weights based on spatial covariates if requested
        if self.config.adaptive_weights and spatial_coords is not None:
            weights = self._compute_spatial_weights(p_values, spatial_coords)
        elif weights is None:
            weights = np.ones(n)
        
        # Normalize weights
        weights = weights / np.sum(weights) * n
        
        # Apply weighted BY procedure
        sorted_idx = np.argsort(p_values)
        sorted_pvals = p_values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        # BY threshold with weights: p_i <= (i * alpha * w_i) / (n * c(n))
        thresholds = np.arange(1, n + 1) * self.config.alpha * sorted_weights / (n * c_n)
        
        # Find largest i where p_(i) <= threshold_i
        discoveries_sorted = sorted_pvals <= thresholds
        
        if np.any(discoveries_sorted):
            # Find the largest i satisfying the condition
            max_idx = np.where(discoveries_sorted)[0][-1]
            discoveries_sorted[:max_idx + 1] = True
        else:
            discoveries_sorted[:] = False
        
        # Map back to original order
        discoveries = np.zeros(n, dtype=bool)
        discoveries[sorted_idx] = discoveries_sorted
        
        return discoveries
    
    def simes_spatial(self,
                     p_values: np.ndarray,
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simes procedure for PRDS (Positive Regression Dependency on Subsets).
        
        Valid for many spatial settings but not arbitrary dependence.
        Less conservative than BY.
        
        Args:
            p_values: P-values to correct
            weights: Optional importance weights
            
        Returns:
            Boolean array of discoveries
        """
        n = len(p_values)
        
        if n == 0:
            return np.array([], dtype=bool)
        
        # Check PRDS assumption if spatial structure available
        if self.config.dependence_assumption == 'arbitrary':
            warnings.warn("Simes may not control FDR under arbitrary dependence. "
                         "Consider using Benjamini-Yekutieli instead.")
        
        if weights is None:
            weights = np.ones(n)
        
        # Normalize weights
        weights = weights / np.sum(weights) * n
        
        # Sort p-values
        sorted_idx = np.argsort(p_values)
        sorted_pvals = p_values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        # Simes threshold: p_i <= (i * alpha * w_i) / n
        thresholds = np.arange(1, n + 1) * self.config.alpha * sorted_weights / n
        
        # Find discoveries
        discoveries_sorted = sorted_pvals <= thresholds
        
        if np.any(discoveries_sorted):
            max_idx = np.where(discoveries_sorted)[0][-1]
            discoveries_sorted[:max_idx + 1] = True
        else:
            discoveries_sorted[:] = False
        
        # Map back
        discoveries = np.zeros(n, dtype=bool)
        discoveries[sorted_idx] = discoveries_sorted
        
        return discoveries
    
    def tree_structured_fdr(self,
                          p_values: np.ndarray,
                          hierarchy: Dict[str, Any],
                          spatial_coords: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Tree-structured FDR for hierarchical hypotheses.
        
        Tests proceed top-down: parent must be significant before testing children.
        Controls selective inference error rate.
        
        Args:
            p_values: P-values for all nodes in tree
            hierarchy: Tree structure with 'parents' and 'children' for each node
            spatial_coords: Optional spatial coordinates
            
        Returns:
            Boolean array of discoveries
        """
        n = len(p_values)
        discoveries = np.zeros(n, dtype=bool)
        
        # Extract tree structure
        parents = hierarchy.get('parents', {})
        children = hierarchy.get('children', {})
        roots = hierarchy.get('roots', [])
        
        if not roots:
            # Find roots (nodes with no parents)
            roots = [i for i in range(n) if i not in parents or parents[i] is None]
        
        # Top-down testing
        tested_nodes = set()
        significant_nodes = set()
        
        # BFS traversal
        from collections import deque
        queue = deque(roots)
        
        while queue:
            node = queue.popleft()
            
            if node in tested_nodes:
                continue
            
            # Check if parent is significant (if has parent)
            if node in parents and parents[node] is not None:
                if parents[node] not in significant_nodes:
                    continue  # Parent not significant, skip
            
            # Test this node
            tested_nodes.add(node)
            
            # Apply FDR at this level
            if node in children and children[node]:
                # Test all children together
                child_indices = children[node]
                child_pvals = p_values[child_indices]
                
                # Apply Simes or BY at this family
                if self.config.dependence_assumption == 'arbitrary':
                    child_discoveries = self.benjamini_yekutieli_spatial(
                        child_pvals, spatial_coords
                    )
                else:
                    child_discoveries = self.simes_spatial(child_pvals)
                
                # Mark discoveries
                for i, child_idx in enumerate(child_indices):
                    if child_discoveries[i]:
                        discoveries[child_idx] = True
                        significant_nodes.add(child_idx)
                        # Add children to queue
                        if child_idx in children:
                            queue.extend(children[child_idx])
            else:
                # Leaf node - test individually
                if p_values[node] <= self.config.alpha:
                    discoveries[node] = True
                    significant_nodes.add(node)
        
        return discoveries
    
    def spatial_pvalue_adjustment(self,
                                 p_values: np.ndarray,
                                 spatial_coords: np.ndarray,
                                 method: str = 'effective_tests') -> np.ndarray:
        """
        Adjust p-values based on spatial correlation structure.
        
        Args:
            p_values: Raw p-values
            spatial_coords: Spatial coordinates
            method: 'effective_tests' or 'spectral'
            
        Returns:
            Adjusted p-values accounting for spatial dependence
        """
        n = len(p_values)
        
        if method == 'effective_tests':
            # Estimate effective number of independent tests
            n_eff = self._estimate_effective_tests(spatial_coords)
            
            # Bonferroni-type adjustment with effective N
            adjusted_pvals = p_values * n_eff
            adjusted_pvals = np.minimum(adjusted_pvals, 1.0)
            
        elif method == 'spectral':
            # Spectral adjustment based on eigenvalues of correlation matrix
            corr_matrix = self._compute_spatial_correlation_matrix(spatial_coords)
            eigenvals = np.linalg.eigvalsh(corr_matrix)
            
            # Effective number via Cheverud-Nyholt method
            var_eigenvals = np.var(np.abs(eigenvals))
            n_eff = 1 + (n - 1) * (1 - var_eigenvals / n)
            
            adjusted_pvals = p_values * n_eff
            adjusted_pvals = np.minimum(adjusted_pvals, 1.0)
            
        else:
            raise ValueError(f"Unknown adjustment method: {method}")
        
        return adjusted_pvals
    
    def check_prds(self, spatial_weights: np.ndarray) -> bool:
        """
        Check if Positive Regression Dependency on Subsets (PRDS) holds.
        
        PRDS is satisfied for many spatial weight matrices with positive weights
        and certain monotone transformations.
        
        Args:
            spatial_weights: Spatial weights matrix
            
        Returns:
            True if PRDS likely holds
        """
        # Check if all weights are non-negative
        if np.any(spatial_weights < 0):
            return False
        
        # Check if matrix is symmetric (typical for spatial weights)
        if not np.allclose(spatial_weights, spatial_weights.T):
            return False
        
        # Check if row-standardized (rows sum to 1)
        row_sums = spatial_weights.sum(axis=1)
        if np.allclose(row_sums, 1.0):
            # Row-standardized positive weights often satisfy PRDS
            return True
        
        # For general positive weights, PRDS may hold but harder to verify
        # Conservative: return False if unsure
        return False
    
    def _compute_spatial_weights(self,
                                p_values: np.ndarray,
                                spatial_coords: np.ndarray) -> np.ndarray:
        """
        Compute adaptive weights based on spatial covariates.
        
        Uses Independent Hypothesis Weighting (IHW) approach with
        spatial features as covariates.
        """
        n = len(p_values)
        
        # Compute spatial features
        features = self._extract_spatial_features(spatial_coords)
        
        # Bin features and estimate optimal weights per bin
        n_bins = min(10, n // 50)  # Adaptive number of bins
        
        if n_bins < 2:
            return np.ones(n)
        
        # Discretize features
        feature_quantiles = np.percentile(features, np.linspace(0, 100, n_bins + 1))
        feature_bins = np.digitize(features, feature_quantiles[1:-1])
        
        # Estimate weights per bin (simplified IHW)
        weights = np.ones(n)
        
        for bin_idx in range(n_bins):
            bin_mask = feature_bins == bin_idx
            if np.sum(bin_mask) > 0:
                # Weight proportional to enrichment of small p-values
                bin_pvals = p_values[bin_mask]
                enrichment = np.mean(bin_pvals < 0.1) / 0.1  # Enrichment at 0.1
                weights[bin_mask] = np.clip(enrichment, 0.5, 2.0)
        
        return weights
    
    def _extract_spatial_features(self, spatial_coords: np.ndarray) -> np.ndarray:
        """Extract spatial features for adaptive weighting."""
        n = len(spatial_coords)
        
        # Distance to centroid (tests near center vs edge)
        centroid = np.mean(spatial_coords, axis=0)
        dist_to_center = np.linalg.norm(spatial_coords - centroid, axis=1)
        
        # Local density (number of neighbors within radius)
        from scipy.spatial import cKDTree
        tree = cKDTree(spatial_coords)
        radius = np.percentile(dist_to_center, 25)
        local_density = np.array([
            len(tree.query_ball_point(coord, radius)) for coord in spatial_coords
        ])
        
        # Combine features (use first principal component)
        features = np.column_stack([dist_to_center, local_density])
        
        # Normalize
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-10)
        
        # Return first PC
        U, s, Vt = np.linalg.svd(features, full_matrices=False)
        return U[:, 0] * s[0]
    
    def _estimate_effective_tests(self, spatial_coords: np.ndarray) -> float:
        """
        Estimate effective number of independent tests.
        
        Based on eigenvalue variance of spatial correlation matrix.
        """
        corr_matrix = self._compute_spatial_correlation_matrix(spatial_coords)
        
        # Eigenvalue decomposition
        eigenvals = np.linalg.eigvalsh(corr_matrix)
        eigenvals = np.abs(eigenvals)  # Handle numerical issues
        
        # Galwey's method
        n = len(eigenvals)
        sum_eigenvals = np.sum(eigenvals)
        sum_squared_eigenvals = np.sum(eigenvals ** 2)
        
        if sum_squared_eigenvals > 0:
            n_eff = sum_eigenvals ** 2 / sum_squared_eigenvals
        else:
            n_eff = n
        
        return float(np.clip(n_eff, 1, n))
    
    def _compute_spatial_correlation_matrix(self, spatial_coords: np.ndarray) -> np.ndarray:
        """Compute spatial correlation matrix from coordinates using sparse KDTree approach."""
        from scipy.spatial import cKDTree
        
        n = len(spatial_coords)
        
        # Use KDTree for efficient neighbor finding
        tree = cKDTree(spatial_coords)
        
        # Estimate median distance from sample to set decay rate
        sample_size = min(1000, n)
        sample_idx = np.random.choice(n, sample_size, replace=False)
        sample_distances = tree.query(spatial_coords[sample_idx], k=2)[0][:, 1]
        median_dist = np.median(sample_distances)
        decay_rate = 1.0 / median_dist
        
        # Define correlation cutoff to avoid storing very small correlations
        corr_cutoff = 0.01  # Only store correlations > 1%
        max_distance = -np.log(corr_cutoff) / decay_rate
        
        # Build sparse correlation matrix
        from scipy import sparse
        row_indices = []
        col_indices = []
        corr_data = []
        
        for i in range(n):
            # Find neighbors within correlation distance
            neighbors = tree.query_ball_point(spatial_coords[i], max_distance)
            for j in neighbors:
                if i <= j:  # Only upper triangular + diagonal to avoid duplicates
                    if i == j:
                        corr = 1.0
                    else:
                        dist = np.linalg.norm(spatial_coords[i] - spatial_coords[j])
                        corr = np.exp(-decay_rate * dist)
                    
                    if corr >= corr_cutoff:
                        row_indices.extend([i, j] if i != j else [i])
                        col_indices.extend([j, i] if i != j else [i])
                        corr_data.extend([corr, corr] if i != j else [corr])
        
        # Create sparse matrix and convert to dense (needed for eigendecomposition)
        corr_sparse = sparse.csr_matrix((corr_data, (row_indices, col_indices)), shape=(n, n))
        
        # For small matrices, convert to dense; for large matrices, use approximation
        if n <= 5000:
            return corr_sparse.toarray()
        else:
            # For very large matrices, use block-diagonal approximation
            # This trades accuracy for memory efficiency
            block_size = 1000
            corr_matrix = np.eye(n)  # Start with identity
            
            for start_idx in range(0, n, block_size):
                end_idx = min(start_idx + block_size, n)
                block_slice = slice(start_idx, end_idx)
                
                # Extract dense block
                block = corr_sparse[block_slice, block_slice].toarray()
                corr_matrix[block_slice, block_slice] = block
            
            return corr_matrix


def fisher_combination_spatial(p_values_list: List[np.ndarray],
                              spatial_coords_list: List[np.ndarray],
                              method: str = 'dependent') -> Tuple[float, float]:
    """
    Combine p-values across multiple ROIs accounting for spatial structure.
    
    Args:
        p_values_list: List of p-value arrays per ROI
        spatial_coords_list: List of coordinate arrays per ROI
        method: 'independent' or 'dependent'
        
    Returns:
        Tuple of (combined_statistic, combined_pvalue)
    """
    # Flatten all p-values
    all_pvals = np.concatenate(p_values_list)
    k = len(all_pvals)
    
    if k == 0:
        return np.nan, np.nan
    
    # Fisher's method: -2 * sum(log(p_i))
    with np.errstate(divide='ignore'):
        log_pvals = np.log(np.maximum(all_pvals, 1e-300))
    
    fisher_stat = -2 * np.sum(log_pvals)
    
    if method == 'independent':
        # Chi-squared distribution with 2k degrees of freedom
        combined_pval = 1 - stats.chi2.cdf(fisher_stat, df=2*k)
        
    else:  # dependent
        # Adjust for correlation using Brown's method
        if len(p_values_list) > 1:
            # Estimate correlation between ROIs
            correlations = []
            for i in range(len(p_values_list) - 1):
                for j in range(i + 1, len(p_values_list)):
                    if len(p_values_list[i]) > 0 and len(p_values_list[j]) > 0:
                        # Sample correlation
                        n_sample = min(len(p_values_list[i]), len(p_values_list[j]), 100)
                        idx_i = np.random.choice(len(p_values_list[i]), n_sample, replace=False)
                        idx_j = np.random.choice(len(p_values_list[j]), n_sample, replace=False)
                        
                        corr = np.corrcoef(
                            -2 * np.log(p_values_list[i][idx_i]),
                            -2 * np.log(p_values_list[j][idx_j])
                        )[0, 1]
                        
                        if np.isfinite(corr):
                            correlations.append(corr)
            
            if correlations:
                mean_corr = np.mean(correlations)
                # Brown's approximation
                c = mean_corr * (3.25 + 0.75 * mean_corr)
                df_brown = 2 * k / c if c > 0 else 2 * k
                combined_pval = 1 - stats.chi2.cdf(fisher_stat / c, df=df_brown)
            else:
                # Fallback to independent
                combined_pval = 1 - stats.chi2.cdf(fisher_stat, df=2*k)
        else:
            combined_pval = 1 - stats.chi2.cdf(fisher_stat, df=2*k)
    
    return float(fisher_stat), float(combined_pval)