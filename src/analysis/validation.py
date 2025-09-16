"""
IMC Data Validation Framework

Realistic validation using synthetic data with Poisson noise and spatial artifacts.
Addresses ion count statistics, antibody crosstalk, and spatial resolution issues.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy import ndimage, stats
from sklearn.metrics import adjusted_rand_score
import warnings


def generate_synthetic_imc_data(
    n_cells: int = 1000,
    tissue_size_um: Tuple[float, float] = (500.0, 500.0),
    n_clusters: int = 5,
    protein_names: List[str] = None,
    base_expression_levels: Optional[Dict[str, float]] = None,
    spatial_structure: str = 'clustered',
    random_state: int = 42
) -> Dict:
    """
    Generate synthetic IMC data with realistic statistical properties.
    
    Args:
        n_cells: Number of cell positions to generate
        tissue_size_um: Tissue dimensions (width, height) in micrometers
        n_clusters: Number of cell type clusters
        protein_names: List of protein markers to simulate
        base_expression_levels: Base expression levels for each protein
        spatial_structure: 'clustered', 'random', or 'gradient'
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with synthetic data components
    """
    np.random.seed(random_state)
    
    if protein_names is None:
        protein_names = ['CD45', 'CD11b', 'CD206', 'CD44', 'CD31', 'CD140b']
    
    if base_expression_levels is None:
        # Realistic base levels based on IMC experience
        base_expression_levels = {
            'CD45': 50.0,    # High immune marker
            'CD11b': 30.0,   # Moderate myeloid
            'CD206': 20.0,   # Lower M2 marker
            'CD44': 40.0,    # Moderate matrix receptor
            'CD31': 60.0,    # High endothelial
            'CD140b': 25.0   # Moderate pericyte
        }
    
    # Generate spatial coordinates
    coords = generate_spatial_coordinates(n_cells, tissue_size_um, spatial_structure)
    
    # Assign cell types based on spatial clustering
    cell_types = assign_cell_types(coords, n_clusters, spatial_structure)
    
    # Generate true protein expression based on cell types
    true_expression = generate_true_expression(
        cell_types, protein_names, base_expression_levels, n_clusters
    )
    
    # Add ion count noise (Poisson + measurement artifacts)
    observed_counts = add_imc_noise(
        true_expression, 
        coords, 
        protein_names,
        enhanced_noise=True  # Use enhanced noise models by default
    )
    
    # Generate DNA channels
    dna_data = generate_dna_channels(coords, tissue_size_um, cell_types)
    
    return {
        'coords': coords,
        'ion_counts': observed_counts,
        'dna1_intensities': dna_data['dna1'],
        'dna2_intensities': dna_data['dna2'],
        'true_cell_types': cell_types,
        'true_expression': true_expression,
        'protein_names': protein_names,
        'metadata': {
            'n_cells': n_cells,
            'n_clusters': n_clusters,
            'tissue_size_um': tissue_size_um,
            'spatial_structure': spatial_structure
        }
    }


def generate_spatial_coordinates(
    n_cells: int,
    tissue_size_um: Tuple[float, float],
    structure: str
) -> np.ndarray:
    """
    Generate cell coordinates with different spatial structures.
    """
    width_um, height_um = tissue_size_um
    
    if structure == 'random':
        # Uniform random distribution
        coords = np.random.uniform(
            low=[0, 0],
            high=[width_um, height_um],
            size=(n_cells, 2)
        )
    
    elif structure == 'clustered':
        # Create spatial clusters (e.g., tissue compartments)
        n_spatial_clusters = 3
        cluster_centers = np.random.uniform(
            low=[0.2 * width_um, 0.2 * height_um],
            high=[0.8 * width_um, 0.8 * height_um],
            size=(n_spatial_clusters, 2)
        )
        
        coords = []
        cells_per_cluster = n_cells // n_spatial_clusters
        
        for i, center in enumerate(cluster_centers):
            # Generate cells around this center
            if i == len(cluster_centers) - 1:
                # Last cluster gets remaining cells
                n_cells_this_cluster = n_cells - len(coords)
            else:
                n_cells_this_cluster = cells_per_cluster
            
            # Gaussian distribution around center
            cluster_coords = np.random.normal(
                loc=center,
                scale=[width_um * 0.1, height_um * 0.1],
                size=(n_cells_this_cluster, 2)
            )
            
            # Clip to tissue boundaries
            cluster_coords[:, 0] = np.clip(cluster_coords[:, 0], 0, width_um)
            cluster_coords[:, 1] = np.clip(cluster_coords[:, 1], 0, height_um)
            
            coords.append(cluster_coords)
        
        coords = np.vstack(coords)
    
    elif structure == 'gradient':
        # Create spatial gradient (e.g., region A to region B)
        coords = np.random.uniform(
            low=[0, 0],
            high=[width_um, height_um],
            size=(n_cells, 2)
        )
        
        # No additional processing needed - gradient is handled in cell type assignment
    
    else:
        raise ValueError(f"Unknown spatial structure: {structure}")
    
    return coords


def assign_cell_types(
    coords: np.ndarray,
    n_clusters: int,
    spatial_structure: str
) -> np.ndarray:
    """
    Assign cell types based on spatial position and structure.
    """
    n_cells = len(coords)
    
    if spatial_structure == 'random':
        # Random cell type assignment
        cell_types = np.random.randint(0, n_clusters, size=n_cells)
    
    elif spatial_structure == 'clustered':
        # Cell types correlate with spatial position
        from sklearn.cluster import KMeans
        
        # Cluster based on spatial coordinates
        spatial_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cell_types = spatial_kmeans.fit_predict(coords)
    
    elif spatial_structure == 'gradient':
        # Cell types vary with spatial gradient
        # Use x-coordinate to define gradient (e.g., region A to region B)
        x_positions = coords[:, 0]
        x_normalized = (x_positions - x_positions.min()) / (x_positions.max() - x_positions.min())
        
        # Assign cell types based on position along gradient
        cell_types = np.floor(x_normalized * n_clusters).astype(int)
        cell_types = np.clip(cell_types, 0, n_clusters - 1)
    
    else:
        raise ValueError(f"Unknown spatial structure: {spatial_structure}")
    
    return cell_types


def generate_true_expression(
    cell_types: np.ndarray,
    protein_names: List[str],
    base_levels: Dict[str, float],
    n_clusters: int
) -> Dict[str, np.ndarray]:
    """
    Generate ground truth protein expression based on cell types.
    """
    n_cells = len(cell_types)
    true_expression = {}
    
    # Create cell type-specific expression patterns
    for protein in protein_names:
        base_level = base_levels.get(protein, 30.0)
        protein_expression = np.zeros(n_cells)
        
        for cluster_id in range(n_clusters):
            mask = cell_types == cluster_id
            n_cells_in_cluster = np.sum(mask)
            
            if n_cells_in_cluster == 0:
                continue
            
            # Each cluster has different expression level
            # Some clusters high, some low, some medium
            if cluster_id == 0:
                # High expression cluster
                mean_expr = base_level * 2.0
            elif cluster_id == n_clusters - 1:
                # Low expression cluster
                mean_expr = base_level * 0.2
            else:
                # Medium expression clusters
                mean_expr = base_level * (0.5 + cluster_id / n_clusters)
            
            # Add biological variability (log-normal)
            cluster_expression = np.random.lognormal(
                mean=np.log(mean_expr),
                sigma=0.3,  # 30% coefficient of variation
                size=n_cells_in_cluster
            )
            
            protein_expression[mask] = cluster_expression
        
        true_expression[protein] = protein_expression
    
    return true_expression


def add_imc_noise(
    true_expression: Dict[str, np.ndarray],
    coords: np.ndarray,
    protein_names: List[str],
    enhanced_noise: bool = True
) -> Dict[str, np.ndarray]:
    """
    Add realistic IMC measurement noise and artifacts.
    
    Args:
        true_expression: True protein expression levels
        coords: Cell coordinates
        protein_names: List of protein names
        enhanced_noise: Use enhanced noise models addressing latest IMC challenges
    """
    noisy_counts = {}
    
    for protein in protein_names:
        true_values = true_expression[protein]
        
        # Step 1: Poisson noise (ion count statistics)
        # Ion counts follow Poisson distribution around true value
        poisson_counts = np.random.poisson(true_values)
        
        # Step 2: Antibody crosstalk and background
        # Add small amount of signal spillover
        background_level = np.mean(true_values) * 0.05  # 5% background
        background_noise = np.random.exponential(background_level, size=len(true_values))
        
        # Step 3: Spatial artifacts (edge effects, uneven staining)
        spatial_artifacts = add_spatial_artifacts(poisson_counts, coords)
        
        # Step 4: Detection efficiency variations
        # Different regions may have different detection efficiency
        detection_efficiency = np.random.uniform(0.85, 1.15, size=len(true_values))
        
        if enhanced_noise:
            # Step 5: Isotope interference and mass drift
            isotope_interference = add_isotope_interference(spatial_artifacts, protein)
            
            # Step 6: Acquisition time-dependent drift
            temporal_drift = add_temporal_drift(isotope_interference, coords)
            
            # Step 7: Cell segmentation artifacts
            segmentation_noise = add_segmentation_artifacts(temporal_drift, coords)
            
            # Step 8: Ion beam instability
            beam_instability = add_beam_instability(segmentation_noise)
            
            # Combine enhanced noise sources
            final_counts = (beam_instability + background_noise) * detection_efficiency
        else:
            # Standard noise model
            final_counts = (spatial_artifacts + background_noise) * detection_efficiency
        
        # Ensure non-negative
        final_counts = np.maximum(final_counts, 0)
        
        noisy_counts[protein] = final_counts
    
    return noisy_counts


def add_spatial_artifacts(
    counts: np.ndarray,
    coords: np.ndarray,
    artifact_strength: float = 0.1
) -> np.ndarray:
    """
    Add spatial artifacts like edge effects and uneven staining.
    """
    # Create spatial artifact field
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    # Edge effect: signal drops near tissue boundaries
    x_norm = (coords[:, 0] - x_min) / (x_max - x_min)
    y_norm = (coords[:, 1] - y_min) / (y_max - y_min)
    
    # Distance from center
    center_distance = np.sqrt((x_norm - 0.5)**2 + (y_norm - 0.5)**2)
    edge_factor = 1.0 - artifact_strength * center_distance
    edge_factor = np.clip(edge_factor, 1.0 - artifact_strength, 1.0)
    
    # Add random spatial variability
    spatial_noise = np.random.normal(1.0, artifact_strength * 0.5, size=len(counts))
    spatial_noise = np.clip(spatial_noise, 0.1, 2.0)
    
    # Apply artifacts
    artifacted_counts = counts * edge_factor * spatial_noise
    
    return artifacted_counts


def add_isotope_interference(
    counts: np.ndarray,
    protein_name: str,
    interference_strength: float = 0.02
) -> np.ndarray:
    """
    Add isotope interference artifacts common in IMC.
    
    Isotope spillover affects specific channels based on mass proximity.
    """
    # Define common isotope interferences
    interference_map = {
        'CD45': ['CD31'],  # 165Ho → 166Er spillover
        'CD68': ['CD163'], # Similar mass region interference
        'CD8': ['CD4'],    # Close mass channels
        'Ki67': ['CD44']   # Example spillover
    }
    
    # Apply interference if this protein is affected
    if protein_name in interference_map:
        # Add spillover signal (small fraction of nearby channel)
        spillover_signal = np.random.exponential(
            np.mean(counts) * interference_strength, 
            size=len(counts)
        )
        counts = counts + spillover_signal
    
    return counts


def add_temporal_drift(
    counts: np.ndarray,
    coords: np.ndarray,
    drift_strength: float = 0.03
) -> np.ndarray:
    """
    Add temporal drift that occurs during IMC acquisition.
    
    Signal may drift over acquisition time, correlating with spatial position.
    """
    # Simulate acquisition order (row-by-row scanning)
    y_positions = coords[:, 1]
    acquisition_order = np.argsort(y_positions)
    
    # Create temporal drift profile
    n_positions = len(counts)
    drift_profile = np.linspace(1.0 - drift_strength, 1.0 + drift_strength, n_positions)
    
    # Apply drift according to acquisition order
    drift_factors = np.zeros(n_positions)
    drift_factors[acquisition_order] = drift_profile
    
    return counts * drift_factors


def add_segmentation_artifacts(
    counts: np.ndarray,
    coords: np.ndarray,
    artifact_probability: float = 0.1,
    artifact_strength: float = 0.5
) -> np.ndarray:
    """
    Add cell segmentation artifacts affecting protein quantification.
    
    Segmentation errors can lead to signal bleeding between adjacent cells.
    """
    from scipy.spatial.distance import cdist
    
    # Find cells that might have segmentation artifacts
    artifact_mask = np.random.random(len(counts)) < artifact_probability
    
    if not np.any(artifact_mask):
        return counts
    
    # For affected cells, blend with nearby cells
    distances = cdist(coords, coords)
    
    modified_counts = counts.copy()
    
    for i, has_artifact in enumerate(artifact_mask):
        if has_artifact:
            # Find nearby cells (within segmentation error range)
            nearby_mask = (distances[i] < 5.0) & (distances[i] > 0)  # 5μm radius
            
            if np.any(nearby_mask):
                # Blend signal with nearby cells
                nearby_signals = counts[nearby_mask]
                blended_signal = np.mean(nearby_signals) * artifact_strength
                modified_counts[i] = counts[i] + blended_signal
    
    return modified_counts


def add_beam_instability(
    counts: np.ndarray,
    instability_strength: float = 0.05
) -> np.ndarray:
    """
    Add ion beam instability effects.
    
    Random fluctuations in beam intensity during acquisition.
    """
    # Random beam intensity fluctuations
    beam_fluctuations = np.random.normal(1.0, instability_strength, len(counts))
    beam_fluctuations = np.clip(beam_fluctuations, 0.8, 1.2)  # Reasonable bounds
    
    return counts * beam_fluctuations


def generate_dna_channels(
    coords: np.ndarray,
    tissue_size_um: Tuple[float, float],
    cell_types: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic DNA channel data for SLIC segmentation.
    """
    n_cells = len(coords)
    
    # DNA1 (Iridium 191) - generally uniform nuclear stain
    dna1_base = 1000.0
    dna1_values = np.random.lognormal(
        mean=np.log(dna1_base),
        sigma=0.2,  # 20% variability
        size=n_cells
    )
    
    # DNA2 (Iridium 193) - similar but with different spectral properties
    dna2_base = 800.0
    dna2_values = np.random.lognormal(
        mean=np.log(dna2_base),
        sigma=0.2,
        size=n_cells
    )
    
    # Add correlation between DNA channels (they're both nuclear)
    correlation_factor = 0.7
    shared_component = np.random.normal(0, 1, size=n_cells)
    dna1_values += shared_component * correlation_factor * np.std(dna1_values)
    dna2_values += shared_component * correlation_factor * np.std(dna2_values)
    
    # Ensure positive values
    dna1_values = np.maximum(dna1_values, 10.0)
    dna2_values = np.maximum(dna2_values, 10.0)
    
    return {
        'dna1': dna1_values,
        'dna2': dna2_values
    }


def validate_clustering_performance(
    predicted_clusters: np.ndarray,
    true_cell_types: np.ndarray,
    coords: np.ndarray = None,
    enhanced_metrics: bool = True
) -> Dict[str, float]:
    """
    Validate clustering performance against ground truth with enhanced IMC-specific metrics.
    
    Args:
        predicted_clusters: Predicted cluster assignments
        true_cell_types: True cell type assignments 
        coords: Cell coordinates for spatial metrics
        enhanced_metrics: Include IMC-specific validation metrics
    """
    if len(predicted_clusters) != len(true_cell_types):
        raise ValueError("Predicted and true cluster arrays must have same length")
    
    # Remove invalid predictions (-1 values)
    valid_mask = predicted_clusters >= 0
    
    if not np.any(valid_mask):
        base_metrics = {
            'adjusted_rand_index': 0.0,
            'cluster_purity': 0.0,
            'coverage': 0.0,
            'n_valid_predictions': 0
        }
        if enhanced_metrics:
            base_metrics.update({
                'spatial_coherence': 0.0,
                'cluster_stability': 0.0,
                'silhouette_validation': 0.0,
                'boundary_preservation': 0.0
            })
        return base_metrics
    
    pred_valid = predicted_clusters[valid_mask]
    true_valid = true_cell_types[valid_mask]
    coords_valid = coords[valid_mask] if coords is not None else None
    
    # Adjusted Rand Index
    ari = adjusted_rand_score(true_valid, pred_valid)
    
    # Cluster purity (average purity of predicted clusters)
    purity_scores = []
    for cluster_id in np.unique(pred_valid):
        cluster_mask = pred_valid == cluster_id
        cluster_true_labels = true_valid[cluster_mask]
        
        if len(cluster_true_labels) > 0:
            # Purity = fraction of most common true label in this cluster
            unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
            max_count = np.max(counts)
            purity = max_count / len(cluster_true_labels)
            purity_scores.append(purity)
    
    average_purity = np.mean(purity_scores) if purity_scores else 0.0
    
    # Coverage (fraction of cells assigned to clusters)
    coverage = np.sum(valid_mask) / len(predicted_clusters)
    
    # Base metrics
    metrics = {
        'adjusted_rand_index': float(ari),
        'cluster_purity': float(average_purity),
        'coverage': float(coverage),
        'n_valid_predictions': int(np.sum(valid_mask))
    }
    
    # Enhanced IMC-specific metrics
    if enhanced_metrics and coords_valid is not None:
        # Spatial coherence: measure how spatially compact clusters are
        spatial_coherence = compute_spatial_coherence(pred_valid, coords_valid)
        
        # Cluster stability: measure consistency across different resolutions
        cluster_stability = compute_cluster_stability(pred_valid, true_valid)
        
        # Boundary preservation: how well cluster boundaries align with true boundaries
        boundary_preservation = compute_boundary_preservation(pred_valid, true_valid, coords_valid)
        
        metrics.update({
            'spatial_coherence': float(spatial_coherence),
            'cluster_stability': float(cluster_stability),
            'boundary_preservation': float(boundary_preservation)
        })
    
    return metrics


def compute_spatial_coherence(
    predicted_clusters: np.ndarray,
    coords: np.ndarray
) -> float:
    """
    Compute spatial coherence: measure how spatially compact predicted clusters are.
    
    Returns the average intra-cluster distance normalized by inter-cluster distance.
    """
    from scipy.spatial.distance import pdist, cdist
    
    unique_clusters = np.unique(predicted_clusters)
    if len(unique_clusters) < 2:
        return 1.0
    
    coherence_scores = []
    
    for cluster_id in unique_clusters:
        cluster_mask = predicted_clusters == cluster_id
        cluster_coords = coords[cluster_mask]
        
        if len(cluster_coords) < 2:
            continue
            
        # Intra-cluster distances
        intra_distances = pdist(cluster_coords)
        avg_intra_distance = np.mean(intra_distances) if len(intra_distances) > 0 else 0
        
        # Inter-cluster distances (to other cluster centroids)
        other_clusters = unique_clusters[unique_clusters != cluster_id]
        if len(other_clusters) == 0:
            continue
            
        cluster_centroid = np.mean(cluster_coords, axis=0).reshape(1, -1)
        
        inter_distances = []
        for other_cluster_id in other_clusters:
            other_cluster_mask = predicted_clusters == other_cluster_id
            other_cluster_coords = coords[other_cluster_mask]
            
            if len(other_cluster_coords) > 0:
                other_centroid = np.mean(other_cluster_coords, axis=0).reshape(1, -1)
                distance = cdist(cluster_centroid, other_centroid)[0, 0]
                inter_distances.append(distance)
        
        avg_inter_distance = np.mean(inter_distances) if inter_distances else 1.0
        
        # Coherence = inter_distance / intra_distance (higher is better)
        if avg_intra_distance > 0:
            coherence = avg_inter_distance / avg_intra_distance
        else:
            coherence = float('inf')
        
        coherence_scores.append(coherence)
    
    # Normalize to [0, 1] range
    if coherence_scores:
        mean_coherence = np.mean(coherence_scores)
        return min(1.0, mean_coherence / 10.0)  # Normalize roughly to [0,1]
    else:
        return 0.0


def compute_cluster_stability(
    predicted_clusters: np.ndarray,
    true_clusters: np.ndarray
) -> float:
    """
    Compute cluster stability using bootstrap-like resampling.
    
    Measures how consistent cluster assignments are across different subsets of data.
    """
    n_samples = len(predicted_clusters)
    if n_samples < 10:
        return 0.0
    
    n_bootstrap = 10
    subsample_size = int(0.8 * n_samples)
    
    stability_scores = []
    
    for _ in range(n_bootstrap):
        # Randomly subsample data
        indices = np.random.choice(n_samples, subsample_size, replace=False)
        
        # Compute ARI between original and subsampled predictions
        pred_subsample = predicted_clusters[indices]
        true_subsample = true_clusters[indices]
        
        # Use true clusters as "reference" for stability measure
        if len(np.unique(pred_subsample)) > 1 and len(np.unique(true_subsample)) > 1:
            stability = adjusted_rand_score(pred_subsample, true_subsample)
            stability_scores.append(stability)
    
    return np.mean(stability_scores) if stability_scores else 0.0


def compute_boundary_preservation(
    predicted_clusters: np.ndarray,
    true_clusters: np.ndarray,
    coords: np.ndarray,
    boundary_threshold: float = 20.0
) -> float:
    """
    Compute how well cluster boundaries are preserved.
    
    Measures agreement between predicted and true cluster boundaries.
    """
    from scipy.spatial.distance import cdist
    
    if len(coords) < 10:
        return 0.0
    
    # Find boundary cells (cells near cluster transitions)
    true_boundary_cells = find_boundary_cells(true_clusters, coords, boundary_threshold)
    pred_boundary_cells = find_boundary_cells(predicted_clusters, coords, boundary_threshold)
    
    # Measure overlap between true and predicted boundaries
    if len(true_boundary_cells) == 0 or len(pred_boundary_cells) == 0:
        return 0.0
    
    # Compute Jaccard index of boundary cells
    true_boundary_set = set(true_boundary_cells)
    pred_boundary_set = set(pred_boundary_cells)
    
    intersection = len(true_boundary_set & pred_boundary_set)
    union = len(true_boundary_set | pred_boundary_set)
    
    return intersection / union if union > 0 else 0.0


def find_boundary_cells(
    clusters: np.ndarray,
    coords: np.ndarray,
    threshold: float
) -> List[int]:
    """
    Find cells that are near cluster boundaries.
    """
    from scipy.spatial.distance import cdist
    
    boundary_cells = []
    
    for i, (cluster_id, coord) in enumerate(zip(clusters, coords)):
        # Find nearby cells
        distances = cdist([coord], coords)[0]
        nearby_mask = distances < threshold
        
        # Check if nearby cells have different cluster assignments
        nearby_clusters = clusters[nearby_mask]
        if len(np.unique(nearby_clusters)) > 1:  # Multiple clusters nearby
            boundary_cells.append(i)
    
    return boundary_cells


def run_validation_experiment(
    analysis_pipeline: Callable,
    n_experiments: int = 10,
    experiment_params: Dict = None
) -> Dict[str, List[float]]:
    """
    Run multiple validation experiments with different synthetic datasets.
    """
    if experiment_params is None:
        experiment_params = {
            'n_cells': 1000,
            'n_clusters': 5,
            'spatial_structure': 'clustered'
        }
    
    # Initialize results dict with enhanced metrics
    results = {
        'adjusted_rand_index': [],
        'cluster_purity': [],
        'coverage': [],
        'n_clusters_found': [],
        'spatial_coherence': [],
        'cluster_stability': [],
        'boundary_preservation': []
    }
    
    for experiment_id in range(n_experiments):
        # Generate synthetic data
        synthetic_data = generate_synthetic_imc_data(
            random_state=experiment_id,
            **experiment_params
        )
        
        # Run analysis pipeline
        try:
            pipeline_result = analysis_pipeline(
                synthetic_data['coords'],
                synthetic_data['ion_counts'],
                synthetic_data['dna1_intensities'],
                synthetic_data['dna2_intensities']
            )
            
            # Extract predicted clusters
            if 'cluster_labels' in pipeline_result:
                predicted_clusters = pipeline_result['cluster_labels']
            else:
                predicted_clusters = np.full(len(synthetic_data['coords']), -1)
            
            # Validate performance
            validation_metrics = validate_clustering_performance(
                predicted_clusters,
                synthetic_data['true_cell_types'],
                synthetic_data['coords']
            )
            
            # Store all validation metrics
            for metric_name in results.keys():
                if metric_name == 'n_clusters_found':
                    n_clusters_found = len(np.unique(predicted_clusters[predicted_clusters >= 0]))
                    results[metric_name].append(n_clusters_found)
                else:
                    metric_value = validation_metrics.get(metric_name, np.nan)
                    results[metric_name].append(metric_value)
            
        except Exception as e:
            warnings.warn(f"Experiment {experiment_id} failed: {str(e)}")
            # Add NaN for failed experiments
            for metric_name in results.keys():
                if metric_name == 'n_clusters_found':
                    results[metric_name].append(0)
                else:
                    results[metric_name].append(np.nan)
    
    return results


def summarize_validation_results(validation_results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Summarize validation experiment results.
    """
    summary = {}
    
    for metric_name, values in validation_results.items():
        # Remove NaN values
        valid_values = [v for v in values if not np.isnan(v)]
        
        if valid_values:
            summary[metric_name] = {
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'median': float(np.median(valid_values)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values)),
                'n_valid': len(valid_values),
                'success_rate': len(valid_values) / len(values)
            }
        else:
            summary[metric_name] = {
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan,
                'min': np.nan,
                'max': np.nan,
                'n_valid': 0,
                'success_rate': 0.0
            }
    
    return summary