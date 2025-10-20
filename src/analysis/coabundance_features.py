"""
Protein Co-abundance Feature Engineering

Exploits combinatorial complexity of protein markers to create richer
clustering substrate from limited marker panels. Instead of just using
9 individual protein intensities, we generate interaction terms, ratios,
and local co-expression patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial import KDTree
from itertools import combinations
import warnings


def generate_coabundance_features(
    feature_matrix: np.ndarray,
    protein_names: List[str],
    spatial_coords: Optional[np.ndarray] = None,
    interaction_order: int = 2,
    include_ratios: bool = True,
    include_products: bool = True,
    include_spatial_covariance: bool = True,
    neighborhood_radius: float = 20.0,
    min_expression_percentile: float = 25.0
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate co-abundance features from protein expression matrix.
    
    From N proteins, we can generate:
    - N*(N-1)/2 pairwise products (co-expression)
    - N*(N-1) ratios (relative abundance)
    - N*(N-1)/2 local spatial covariances
    
    For 9 proteins: 9 original + 36 products + 72 ratios + 36 covariances = 153 features
    
    Args:
        feature_matrix: N_samples x N_proteins expression matrix
        protein_names: List of protein names
        spatial_coords: N_samples x 2 spatial coordinates
        interaction_order: Maximum order of interactions (2 for pairs, 3 for triplets)
        include_ratios: Whether to include protein ratios
        include_products: Whether to include protein products
        include_spatial_covariance: Whether to include local co-expression
        neighborhood_radius: Radius for spatial covariance calculation
        min_expression_percentile: Minimum expression to avoid ratio artifacts
        
    Returns:
        Tuple of (enriched_features, feature_names)
    """
    n_samples, n_proteins = feature_matrix.shape
    features = [feature_matrix]
    feature_names = protein_names.copy()
    
    # Safety check: ensure protein_names matches feature_matrix dimensions
    if len(protein_names) != n_proteins:
        raise ValueError(f"Mismatch: protein_names has {len(protein_names)} entries "
                        f"but feature_matrix has {n_proteins} columns")
    
    # Avoid division by zero in ratios
    if feature_matrix.size > 0:
        positive_values = feature_matrix[feature_matrix > 0]
        if len(positive_values) > 0:
            min_expression = np.percentile(positive_values, min_expression_percentile)
        else:
            min_expression = 1.0  # Default for all-zero data
    else:
        min_expression = 1.0  # Default for empty data
    safe_matrix = feature_matrix + min_expression
    
    # Generate pairwise products (co-expression strength)
    if include_products and interaction_order >= 2:
        for i, j in combinations(range(n_proteins), 2):
            product = feature_matrix[:, i] * feature_matrix[:, j]
            # Normalize by geometric mean to maintain scale
            product = product / np.sqrt(np.mean(feature_matrix[:, i]**2) * 
                                        np.mean(feature_matrix[:, j]**2) + 1e-10)
            features.append(product.reshape(-1, 1))
            feature_names.append(f"{protein_names[i]}*{protein_names[j]}")
    
    # Generate ratios (relative abundance)
    if include_ratios:
        for i in range(n_proteins):
            for j in range(n_proteins):
                if i != j:
                    ratio = safe_matrix[:, i] / safe_matrix[:, j]
                    # Log-transform ratios for symmetry
                    ratio = np.log1p(ratio)
                    # Handle any remaining NaN/inf values
                    ratio = np.nan_to_num(ratio, nan=0.0, posinf=10.0, neginf=-10.0)
                    features.append(ratio.reshape(-1, 1))
                    feature_names.append(f"{protein_names[i]}/{protein_names[j]}")
    
    # Generate local spatial covariance
    if include_spatial_covariance and spatial_coords is not None:
        covariance_features = compute_local_covariance(
            feature_matrix, protein_names, spatial_coords, neighborhood_radius
        )
        features.append(covariance_features)
        
        for i, j in combinations(range(n_proteins), 2):
            feature_names.append(f"cov({protein_names[i]},{protein_names[j]})")
    
    # Higher-order interactions if requested
    if interaction_order >= 3 and n_proteins >= 3:
        for proteins in combinations(range(n_proteins), 3):
            triplet = feature_matrix[:, proteins[0]]
            for p in proteins[1:]:
                triplet = triplet * feature_matrix[:, p]
            # Normalize by geometric mean
            norm = np.prod([np.mean(feature_matrix[:, p]**2) for p in proteins]) ** (1/(2*3))
            triplet = triplet / (norm + 1e-10)
            features.append(triplet.reshape(-1, 1))
            
            triplet_name = "*".join([protein_names[p] for p in proteins])
            feature_names.append(triplet_name)
    
    # Concatenate all features
    enriched_matrix = np.hstack(features)
    
    # Final cleanup of any NaN/inf values
    enriched_matrix = np.nan_to_num(enriched_matrix, nan=0.0, posinf=10.0, neginf=-10.0)
    
    return enriched_matrix, feature_names


def compute_local_covariance(
    feature_matrix: np.ndarray,
    protein_names: List[str], 
    spatial_coords: np.ndarray,
    radius: float = 20.0
) -> np.ndarray:
    """
    Compute local spatial covariance between protein pairs.
    
    This captures whether proteins tend to co-occur in spatial neighborhoods,
    which is distinct from their pixel-level correlation.
    
    Args:
        feature_matrix: N_samples x N_proteins
        protein_names: Protein names
        spatial_coords: N_samples x 2 coordinates
        radius: Neighborhood radius in spatial units
        
    Returns:
        N_samples x N_pairs local covariance matrix
    """
    n_samples, n_proteins = feature_matrix.shape
    
    # Build spatial index
    tree = KDTree(spatial_coords)
    
    # Pre-compute neighborhoods
    neighborhoods = tree.query_ball_tree(tree, r=radius)
    
    # Compute local covariances
    n_pairs = n_proteins * (n_proteins - 1) // 2
    local_cov = np.zeros((n_samples, n_pairs))
    
    pair_idx = 0
    for i, j in combinations(range(n_proteins), 2):
        for sample_idx in range(n_samples):
            neighbors = neighborhoods[sample_idx]
            if len(neighbors) > 1:
                # Get protein values in neighborhood
                local_i = feature_matrix[neighbors, i]
                local_j = feature_matrix[neighbors, j]
                
                # Compute local covariance
                cov = np.cov(local_i, local_j)[0, 1]
                # Handle NaN covariance (constant values)
                if np.isnan(cov):
                    cov = 0.0
                local_cov[sample_idx, pair_idx] = cov
        
        pair_idx += 1
    
    return local_cov


def select_informative_coabundance_features(
    enriched_features: np.ndarray,
    feature_names: List[str],
    target_n_features: int = 50,
    method: str = 'variance',
    options: Optional[Dict] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Select most informative co-abundance features.

    Args:
        enriched_features: Full feature matrix
        feature_names: Names of all features
        target_n_features: Number of features to select
        method: Selection method ('variance', 'mutual_info', 'lasso')
        options: Optional dict with method-specific options (e.g., lasso_max_iter)

    Returns:
        Tuple of (selected_features, selected_names)
    """
    if options is None:
        options = {}
    n_features = enriched_features.shape[1]
    
    if target_n_features >= n_features:
        return enriched_features, feature_names
    
    if method == 'variance':
        # Select features with highest variance
        variances = np.var(enriched_features, axis=0)
        top_indices = np.argsort(variances)[-target_n_features:]
        
    elif method == 'mutual_info':
        # Use mutual information between features
        from sklearn.feature_selection import mutual_info_regression
        # Use first PC as target for unsupervised selection
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        target = pca.fit_transform(enriched_features).ravel()
        
        mi_scores = mutual_info_regression(enriched_features, target)
        top_indices = np.argsort(mi_scores)[-target_n_features:]
        
    elif method == 'lasso':
        # Use L1 regularization for feature selection
        from sklearn.linear_model import LassoCV
        from sklearn.decomposition import PCA

        # Create target from first PCs
        pca = PCA(n_components=1)
        target = pca.fit_transform(enriched_features).ravel()

        # Get max_iter from options (default: 5000 for robust convergence)
        max_iter = options.get('lasso_max_iter', 5000)

        # Fit Lasso with cross-validation
        lasso = LassoCV(cv=5, max_iter=max_iter)
        lasso.fit(enriched_features, target)
        
        # Select features with non-zero coefficients
        coef_abs = np.abs(lasso.coef_)
        top_indices = np.argsort(coef_abs)[-target_n_features:]
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    selected_features = enriched_features[:, top_indices]
    selected_names = [feature_names[i] for i in top_indices]
    
    return selected_features, selected_names


def compute_protein_modules(
    feature_matrix: np.ndarray,
    protein_names: List[str],
    n_modules: int = 3,
    method: str = 'nmf'
) -> Tuple[np.ndarray, Dict[str, List[str]]]:
    """
    Identify co-expressed protein modules using matrix factorization.
    
    Instead of treating proteins independently, find groups that co-vary.
    
    Args:
        feature_matrix: N_samples x N_proteins
        protein_names: Protein names
        n_modules: Number of modules to identify
        method: 'nmf' or 'ica'
        
    Returns:
        Tuple of (module_scores, module_compositions)
    """
    from sklearn.decomposition import NMF, FastICA
    
    # Ensure non-negative for NMF
    if method == 'nmf':
        # Shift to non-negative if needed
        min_val = np.min(feature_matrix)
        if min_val < 0:
            matrix = feature_matrix - min_val
        else:
            matrix = feature_matrix
            
        model = NMF(n_components=n_modules, init='nndsvda', random_state=42)
        module_scores = model.fit_transform(matrix)
        loadings = model.components_
        
    elif method == 'ica':
        model = FastICA(n_components=n_modules, random_state=42)
        module_scores = model.fit_transform(feature_matrix)
        loadings = model.components_
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Identify which proteins contribute to each module
    module_compositions = {}
    for module_idx in range(n_modules):
        # Get proteins with high loadings
        module_loadings = np.abs(loadings[module_idx])
        threshold = np.percentile(module_loadings, 75)
        
        contributing_proteins = [
            protein_names[i] 
            for i, loading in enumerate(module_loadings)
            if loading > threshold
        ]
        
        module_compositions[f"Module_{module_idx}"] = contributing_proteins
    
    return module_scores, module_compositions


def create_hierarchical_features(
    feature_matrix: np.ndarray,
    protein_names: List[str],
    hierarchy: Optional[Dict[str, List[str]]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Create hierarchical features based on known protein relationships.
    
    For example:
    - Immune markers -> T-cell, B-cell, Myeloid
    - Structural markers -> ECM, Cytoskeleton
    
    Args:
        feature_matrix: N_samples x N_proteins
        protein_names: Protein names
        hierarchy: Optional hierarchical grouping of proteins
        
    Returns:
        Tuple of (hierarchical_features, feature_names)
    """
    if hierarchy is None:
        # Default hierarchy for common IMC panels
        hierarchy = {
            'pan_immune': ['CD45', 'CD45RA', 'CD45RO'],
            't_cell': ['CD3', 'CD4', 'CD8'],
            'b_cell': ['CD20', 'CD19'],
            'myeloid': ['CD68', 'CD163', 'CD11b'],
            'structural': ['Vimentin', 'SMA', 'Collagen'],
            'proliferation': ['Ki67', 'PCNA'],
            'epithelial': ['PanCK', 'ECadherin']
        }
    
    hierarchical_features = []
    hierarchical_names = []
    
    for group_name, group_proteins in hierarchy.items():
        # Find indices of proteins in this group
        indices = []
        for protein in group_proteins:
            if protein in protein_names:
                indices.append(protein_names.index(protein))
        
        if indices:
            # Compute group-level features
            # Mean expression
            group_mean = np.mean(feature_matrix[:, indices], axis=1)
            hierarchical_features.append(group_mean.reshape(-1, 1))
            hierarchical_names.append(f"{group_name}_mean")
            
            # Max expression (any marker present)
            group_max = np.max(feature_matrix[:, indices], axis=1)
            hierarchical_features.append(group_max.reshape(-1, 1))
            hierarchical_names.append(f"{group_name}_max")
            
            # Variance within group (heterogeneity)
            if len(indices) > 1:
                group_var = np.var(feature_matrix[:, indices], axis=1)
                hierarchical_features.append(group_var.reshape(-1, 1))
                hierarchical_names.append(f"{group_name}_heterogeneity")
    
    if hierarchical_features:
        return np.hstack(hierarchical_features), hierarchical_names
    else:
        return np.zeros((feature_matrix.shape[0], 0)), []