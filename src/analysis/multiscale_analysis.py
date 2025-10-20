"""
Multi-Scale Analysis Framework

Compare tissue microenvironments at multiple spatial scales to address heterogeneity.
Implements 10μm, 20μm, and 40μm scale analysis with consistency metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING, Any
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from .ion_count_processing import ion_count_pipeline
from .slic_segmentation import slic_pipeline
from .spatial_clustering import perform_spatial_clustering, stability_analysis, compute_spatial_coherence
from .hierarchical_multiscale import build_multiscale_hierarchy, add_neighbor_composition_features
from .graph_clustering import create_graph_clustering_baseline
from .clustering_comparison import compare_graph_vs_spatial_clustering

if TYPE_CHECKING:
    from ..config import Config

def _extract_stability_config(config: Optional['Config']) -> Dict[str, Any]:
    """
    Safely extract stability-analysis optimization settings from config-like objects.
    """
    if config is None:
        return {}

    if hasattr(config, 'optimization'):
        optimization_section = getattr(config, 'optimization', {})
        if isinstance(optimization_section, dict):
            return optimization_section.get('stability_analysis', {}) or {}

    if isinstance(config, dict):
        return config.get('optimization', {}).get('stability_analysis', {}) or {}

    return {}


def _compute_scale_adaptive_k_neighbors(n_samples: int, scale_um: float, config: Optional['Config']) -> int:
    """
    Compute scale-adaptive k_neighbors to prevent over-connected graphs at coarse scales.

    Uses 2×log(N) rule as starting point, then adjusts based on scale and config overrides.

    Args:
        n_samples: Number of superpixels at this scale
        scale_um: Spatial scale in micrometers
        config: Config object with optional k_neighbors_by_scale overrides

    Returns:
        Optimal k value for this scale
    """
    # Extract scale-specific k values from config if provided
    k_params = _extract_algorithm_params(config, 'k_neighbors_by_scale')

    # Check for direct scale mapping (e.g., "10.0": 14)
    if k_params and str(scale_um) in k_params:
        return int(k_params[str(scale_um)])

    # Otherwise use data-driven heuristic: 2×log(N)
    # This ensures graphs aren't over-connected at coarse scales
    k_heuristic = max(8, min(15, int(2 * np.log(n_samples))))

    return k_heuristic


def _extract_algorithm_params(config: Optional['Config'], param_name: str) -> Dict[str, Any]:
    """
    Safely extract algorithm parameters from config.

    BUG FIX #9: Externalized hardcoded scientific parameters for reproducibility.

    Args:
        config: Config object or dict
        param_name: Name of parameter set (e.g., 'spatial_weight', 'resolution_range')

    Returns:
        Dictionary of parameters with defaults if not in config
    """
    if config is None:
        return {}

    # Try to extract from Config object
    if hasattr(config, 'optimization'):
        optimization_section = getattr(config, 'optimization', {})
        if isinstance(optimization_section, dict):
            algo_params = optimization_section.get('algorithm_parameters', {})
            if isinstance(algo_params, dict):
                return algo_params.get(param_name, {})

    # Try to extract from dict
    if isinstance(config, dict):
        algo_params = config.get('optimization', {}).get('algorithm_parameters', {})
        if isinstance(algo_params, dict):
            return algo_params.get(param_name, {})

    return {}


def perform_multiscale_analysis(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    dna1_intensities: np.ndarray,
    dna2_intensities: np.ndarray,
    scales_um: List[float] = [10.0, 20.0, 40.0],
    method: str = 'leiden',
    use_slic: bool = True,
    segmentation_method: str = 'slic',  # New parameter: 'slic', 'watershed', or 'grid'
    config: Optional['Config'] = None,
    cached_cofactors: Optional[Dict[str, float]] = None,
    plots_dir: Optional[str] = None,
    roi_id: Optional[str] = None,
    include_graph_baseline: bool = False,  # New parameter: include graph-based clustering baseline
    compare_clustering_methods: bool = False  # New parameter: compare multiple clustering approaches
) -> Dict:
    """
    Perform hierarchical multi-scale analysis with data-driven clustering.
    
    This new approach:
    1. Performs spatial aggregation at each scale
    2. Runs stability analysis to find optimal resolution
    3. Builds hierarchical relationships between scales
    4. Provides diagnostic visualizations for scientist review
    5. Optionally includes graph-based clustering baseline for comparison
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        dna1_intensities: DNA1 channel data
        dna2_intensities: DNA2 channel data
        scales_um: List of spatial scales to analyze
        method: Clustering method ('leiden', 'hdbscan')
        use_slic: Whether to use SLIC segmentation (deprecated, use segmentation_method)
        segmentation_method: Segmentation method ('slic', 'watershed', or 'grid')
        config: Configuration object
        cached_cofactors: Pre-computed arcsinh cofactors
        plots_dir: Optional directory for validation plots
        roi_id: Optional ROI identifier for plots
        include_graph_baseline: Include graph-based clustering baseline for comparison
        compare_clustering_methods: Run comprehensive clustering method comparison
        
    Returns:
        Dictionary with hierarchical analysis results and optional baseline comparisons
    """
    import logging
    logger = logging.getLogger('MultiscaleAnalysis')
    logger.info(f"Starting hierarchical multi-scale analysis with {len(coords)} pixels")
    
    # Handle backward compatibility
    if not use_slic and segmentation_method == 'slic':
        segmentation_method = 'grid'
        logger.info("use_slic=False detected, switching to grid segmentation")
    
    logger.info(f"Using {segmentation_method} segmentation method")
    
    results = {}
    
    # Process each scale
    for scale_idx, scale_um in enumerate(scales_um):
        logger.info(f"Processing scale {scale_um}μm with {segmentation_method} segmentation...")
        
        # Step 1: Spatial aggregation with method selection
        if segmentation_method.lower() == 'slic':
            from .slic_segmentation import slic_pipeline
            aggregation_result = slic_pipeline(
                coords=coords,
                ion_counts=ion_counts,
                dna1_intensities=dna1_intensities,
                dna2_intensities=dna2_intensities,
                target_scale_um=scale_um,
                n_segments=None,  # Data-driven
                config=config,
                cached_cofactors=cached_cofactors
            )
        elif segmentation_method.lower() == 'watershed':
            from .watershed_segmentation import watershed_pipeline
            aggregation_result = watershed_pipeline(
                coords=coords,
                ion_counts=ion_counts,
                dna1_intensities=dna1_intensities,
                dna2_intensities=dna2_intensities,
                target_scale_um=scale_um,
                resolution_um=1.0,
                config=config,
                cached_cofactors=cached_cofactors
            )
        elif segmentation_method.lower() == 'grid':
            from .grid_segmentation import grid_pipeline
            aggregation_result = grid_pipeline(
                coords=coords,
                ion_counts=ion_counts,
                dna1_intensities=dna1_intensities,
                dna2_intensities=dna2_intensities,
                target_scale_um=scale_um,
                n_segments=None,  # Data-driven
                config=config,
                cached_cofactors=cached_cofactors
            )
        else:
            raise ValueError(f"Unknown segmentation method: {segmentation_method}. Use 'slic', 'watershed', or 'grid'.")
        
        # Continue with existing logic for spatial aggregation
        if segmentation_method.lower() in ['slic', 'watershed', 'grid']:
            # Check if we have valid superpixel results
            if ('superpixel_counts' not in aggregation_result or
                len(aggregation_result.get('superpixel_coords', [])) == 0):
                # Empty or invalid data - return minimal result
                return {
                    'scale_um': scale_um,
                    'method': segmentation_method,
                    'segmentation_method': segmentation_method,
                    'features': np.array([]).reshape(0, len(ion_counts)),
                    'spatial_coords': np.array([]).reshape(0, 2),
                    'cluster_labels': np.array([]),
                    'clustering_info': {'n_clusters': 0, 'method': 'none'},
                    'superpixel_labels': np.array([])
                }

            # Extract features from superpixels/cells
            features = np.array([aggregation_result['superpixel_counts'][protein]
                                for protein in ion_counts.keys()]).T
            spatial_coords = aggregation_result['superpixel_coords']
        else:
            aggregation_result = ion_count_pipeline(
                coords=coords,
                ion_counts=ion_counts,
                bin_size_um=scale_um
            )
            features = aggregation_result['feature_matrix']
            # Create spatial coordinates from bin centers
            bin_edges_x = aggregation_result['bin_edges_x']
            bin_edges_y = aggregation_result['bin_edges_y']
            valid_indices = aggregation_result['valid_indices']
            
            # Create bin center coordinates for valid bins
            if len(bin_edges_x) > 1 and len(bin_edges_y) > 1:
                x_centers = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2
                y_centers = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2
                
                # Convert valid indices to 2D coordinates  
                n_x_bins = len(bin_edges_x) - 1
                y_indices = valid_indices // n_x_bins
                x_indices = valid_indices % n_x_bins
                
                spatial_coords = np.column_stack([
                    x_centers[x_indices],
                    y_centers[y_indices]
                ])
            else:
                spatial_coords = np.zeros((len(features), 2))
        
        # Get protein names
        protein_names = list(ion_counts.keys())
        
        # Step 2: Add spatial features for coarse scales
        enhanced_protein_names = protein_names.copy()
        if scale_um >= 20 and features.size > 0 and len(features.shape) == 2:
            original_n_features = features.shape[1]
            features = add_neighbor_composition_features(
                features, spatial_coords, 
                np.zeros(len(features)),  # Placeholder labels
                radius_um=scale_um
            )
            # Add feature names for the composition features
            n_composition_features = features.shape[1] - original_n_features
            for i in range(n_composition_features):
                enhanced_protein_names.append(f"neighborhood_comp_{i}")
        
        # Step 3: Stability analysis for resolution selection
        stability_config = _extract_stability_config(config)
        stability_kwargs = {
            'n_resolutions': stability_config.get('n_resolutions', 15),
            'n_bootstrap': stability_config.get('n_bootstrap_iterations', 5),
            'use_graph_caching': stability_config.get('use_graph_caching', True),
            'parallel': stability_config.get('parallel_execution', True),
            'n_workers': stability_config.get('n_workers'),
            'adaptive_search': stability_config.get('adaptive_search', False),
            'adaptive_target_stability': stability_config.get('adaptive_target_stability', 0.6),
            'adaptive_tolerance': stability_config.get('adaptive_tolerance', 0.05),
            'adaptive_max_evaluations': stability_config.get('adaptive_max_evaluations')
        }

        # BUG FIX #9: Read resolution range from config instead of hardcoding
        resolution_range_params = _extract_algorithm_params(config, 'resolution_range')
        threshold = resolution_range_params.get('fine_scale_threshold_um', 20.0)
        fine_range = resolution_range_params.get('fine_scale_range', [0.5, 2.0])
        coarse_range = resolution_range_params.get('coarse_scale_range', [0.2, 1.0])
        resolution_range = tuple(fine_range) if scale_um <= threshold else tuple(coarse_range)

        stability_result = stability_analysis(
            features, spatial_coords,
            method=method,
            resolution_range=resolution_range,
            **stability_kwargs
        )

        optimal_resolution = stability_result['optimal_resolution']

        # Step 4: Perform clustering at optimal resolution with co-abundance features
        # Extract coabundance options from config
        coabundance_opts = {}
        if config is not None:
            if hasattr(config, 'analysis') and hasattr(config.analysis, 'clustering'):
                if hasattr(config.analysis.clustering, 'coabundance_options'):
                    coabundance_opts = config.analysis.clustering.coabundance_options
            elif isinstance(config, dict):
                coabundance_opts = config.get('analysis', {}).get('clustering', {}).get('coabundance_options', {})

        # BUG FIX #9: Read spatial weight from config instead of hardcoding
        spatial_weight_params = _extract_algorithm_params(config, 'spatial_weight')
        sw_threshold = spatial_weight_params.get('fine_scale_threshold_um', 20.0)
        fine_weight = spatial_weight_params.get('fine_scale_weight', 0.2)
        coarse_weight = spatial_weight_params.get('coarse_scale_weight', 0.4)
        spatial_weight = fine_weight if scale_um <= sw_threshold else coarse_weight

        # Compute scale-adaptive k_neighbors to prevent over-connected graphs
        n_samples = features.shape[0]
        k_neighbors = _compute_scale_adaptive_k_neighbors(n_samples, scale_um, config)

        cluster_labels, clustering_info = perform_spatial_clustering(
            features, spatial_coords,
            method=method,
            resolution=optimal_resolution,
            spatial_weight=spatial_weight,
            k_neighbors=k_neighbors,  # Scale-adaptive k instead of fixed k=15
            use_coabundance=True,  # Enable co-abundance feature generation
            protein_names=enhanced_protein_names,
            coabundance_options=coabundance_opts
        )

        # Use enriched features if they were generated
        final_features = clustering_info.get('enriched_features', features)

        # Step 5: Compute spatial coherence
        coherence = compute_spatial_coherence(cluster_labels, spatial_coords)

        results[scale_um] = {
            'features': final_features,
            'spatial_coords': spatial_coords,
            'cluster_labels': cluster_labels,
            'clustering_info': clustering_info,
            'stability_analysis': stability_result,
            'spatial_coherence': coherence,
            'scale_um': scale_um,
            'method': segmentation_method,
            'segmentation_method': segmentation_method  # Explicit method tracking
        }
        
        # Add segmentation-specific data for visualization (works for both SLIC and Grid)
        if 'superpixel_labels' in aggregation_result:
            results[scale_um].update({
                'superpixel_labels': aggregation_result['superpixel_labels'],
                'superpixel_coords': aggregation_result['superpixel_coords'],
                'composite_dna': aggregation_result.get('composite_dna'),
                'bounds': aggregation_result.get('bounds'),
                'transformed_arrays': aggregation_result.get('transformed_arrays'),
                'cofactors_used': aggregation_result.get('cofactors_used')
            })
            
            # Add method-specific metrics
            if segmentation_method.lower() == 'grid':
                results[scale_um].update({
                    'grid_metrics': aggregation_result.get('metrics'),
                    'boundary_quality': aggregation_result.get('boundary_quality'),
                    'performance_comparison': aggregation_result.get('performance_comparison')
                })
        
        # Generate validation plots if requested (works for both methods)
        if plots_dir and roi_id and 'superpixel_labels' in aggregation_result:
            _generate_validation_plot(
                aggregation_result, scale_um, plots_dir, roi_id, config
            )
        elif plots_dir and roi_id:
            # Debug: Log why plot wasn't generated
            import logging
            logger = logging.getLogger('MultiscaleAnalysis')
            logger.warning(f"Validation plot skipped for {roi_id} at {scale_um}μm: "
                         f"superpixel_labels {'present' if 'superpixel_labels' in aggregation_result else 'MISSING'}")
        
        # Add graph-based clustering baseline for comparison
        if include_graph_baseline:
            logger.info(f"Computing graph-based clustering baseline for {scale_um}μm scale...")
            try:
                graph_baseline = create_graph_clustering_baseline(
                    features, list(ion_counts.keys()), spatial_coords,
                    config={'clustering_method': 'leiden', 'graph_method': 'knn'}
                )
                results[scale_um]['graph_baseline'] = graph_baseline
                
                # Add comparison metrics
                spatial_labels = cluster_labels
                graph_labels = graph_baseline['cluster_labels']
                
                if len(spatial_labels) == len(graph_labels):
                    # Agreement metrics
                    valid_mask = (spatial_labels >= 0) & (graph_labels >= 0)
                    if np.sum(valid_mask) > 1:
                        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                        ari = adjusted_rand_score(spatial_labels[valid_mask], graph_labels[valid_mask])
                        nmi = normalized_mutual_info_score(spatial_labels[valid_mask], graph_labels[valid_mask])
                        
                        results[scale_um]['clustering_comparison'] = {
                            'spatial_vs_graph_ari': ari,
                            'spatial_vs_graph_nmi': nmi,
                            'spatial_n_clusters': len(np.unique(spatial_labels[spatial_labels >= 0])),
                            'graph_n_clusters': len(np.unique(graph_labels[graph_labels >= 0])),
                            'spatial_coherence': coherence,
                            'graph_coherence': graph_baseline['quality_metrics'].get('spatial_coherence', np.nan)
                        }
                        
            except Exception as e:
                logger.warning(f"Graph baseline failed for scale {scale_um}μm: {e}")
        
        # Comprehensive clustering method comparison
        if compare_clustering_methods:
            logger.info(f"Running comprehensive clustering comparison for {scale_um}μm scale...")
            try:
                comparison_results = compare_graph_vs_spatial_clustering(
                    features, list(ion_counts.keys()), spatial_coords
                )
                results[scale_um]['method_comparison'] = comparison_results
                
            except Exception as e:
                logger.warning(f"Clustering comparison failed for scale {scale_um}μm: {e}")
    
    # Step 6: Build hierarchy if we have multiple scales
    if len(scales_um) > 1:
        # Use finest scale as base
        finest_scale = min(scales_um)
        hierarchy = build_multiscale_hierarchy(
            results[finest_scale]['features'],
            results[finest_scale]['spatial_coords'],
            results[finest_scale]['cluster_labels'],
            coarsening_factor=2.0,
            n_scales=len(scales_um)
        )
        results['hierarchy'] = hierarchy
    
    return results


# OLD perform_multiscale_analysis DELETED - Now using hierarchical approach only


def _generate_validation_plot(
    aggregation_result: Dict,
    scale_um: float,
    plots_dir: str,
    roi_id: str,
    config: Optional['Config']
) -> None:
    """
    Generate multichannel validation plot for a given scale.
    
    Args:
        aggregation_result: SLIC pipeline results
        scale_um: Current analysis scale
        plots_dir: Directory to save plots
        roi_id: ROI identifier
        config: Configuration object
    """
    from pathlib import Path
    import numpy as np
    from ..viz_utils.plotting import plot_segmentation_overlay
    
    # Ensure plots directory exists
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    
    # Check we have required data
    if not all(k in aggregation_result for k in ['composite_dna', 'superpixel_labels', 'bounds', 
                                                   'transformed_arrays', 'cofactors_used']):
        return  # Skip if missing required data
    
    try:
        # Calculate quality metrics for title
        labels = aggregation_result['superpixel_labels']
        composite_dna = aggregation_result['composite_dna']
        
        # 1. Tissue coverage
        tissue_pixels = np.sum(labels >= 0)
        total_pixels = labels.size
        tissue_coverage = (tissue_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # 2. Segmentation granularity
        unique_superpixels = np.unique(labels[labels >= 0])
        n_superpixels = len(unique_superpixels)
        mean_superpixel_size = tissue_pixels / n_superpixels if n_superpixels > 0 else 0
        mean_superpixel_um2 = mean_superpixel_size  # Assuming 1μm resolution
        
        # 3. Spatial heterogeneity (using coefficient of variation of superpixel sizes)
        if n_superpixels > 1:
            superpixel_sizes = [np.sum(labels == sp) for sp in unique_superpixels]
            size_cv = np.std(superpixel_sizes) / np.mean(superpixel_sizes) if np.mean(superpixel_sizes) > 0 else 0
            heterogeneity = f"CV={size_cv:.2f}"
        else:
            heterogeneity = "N/A"
        
        # Create informative title
        title = (f"{roi_id} | {scale_um}μm scale | "
                f"Coverage: {tissue_coverage:.1f}% | "
                f"Segments: {n_superpixels} ({mean_superpixel_um2:.0f}μm²) | "
                f"Heterogeneity: {heterogeneity}")
        
        # Generate the plot
        fig = plot_segmentation_overlay(
            image=composite_dna,
            labels=labels,
            bounds=aggregation_result['bounds'],
            transformed_arrays=aggregation_result['transformed_arrays'],
            cofactors_used=aggregation_result['cofactors_used'],
            config=config,
            superpixel_coords=aggregation_result.get('superpixel_coords'),
            title=title
        )
        
        # Save the plot
        output_file = plots_path / f"{roi_id}_scale_{scale_um}_multichannel_validation.png"
        fig.savefig(output_file, dpi=100, bbox_inches='tight')
        
        import matplotlib.pyplot as plt
        plt.close(fig)  # Free memory
        
        import logging
        logger = logging.getLogger('MultiscaleAnalysis')
        logger.info(f"Saved validation plot to {output_file}")
        
    except Exception as e:
        import logging
        logger = logging.getLogger('MultiscaleAnalysis')
        logger.warning(f"Could not generate validation plot: {e}")


def compute_scale_consistency(
    multiscale_results: Dict[float, Dict]
) -> Dict[str, Dict]:
    """
    Validate relationships between scales using actual analysis results.

    Args:
        multiscale_results: Results from perform_multiscale_analysis

    Returns:
        Dictionary of scale consistency metrics
    """
    # Extract numeric scale keys (exclude 'hierarchy' and other non-scale keys)
    scales = sorted([k for k in multiscale_results.keys() if isinstance(k, (int, float))])

    if len(scales) == 0:
        return {
            'scales_analyzed': [],
            'n_scales': 0,
            'overall': {},
            'scale_progression': {},
            'note': 'No scales found'
        }

    # Extract cluster counts and other metrics from each scale
    cluster_counts = []
    n_bins = []

    for scale in scales:
        result = multiscale_results[scale]
        # Get n_clusters from clustering_info or cluster_labels
        if 'clustering_info' in result and 'n_clusters' in result['clustering_info']:
            n_clusters = result['clustering_info']['n_clusters']
        elif 'cluster_labels' in result:
            n_clusters = len(np.unique(result['cluster_labels'][result['cluster_labels'] >= 0]))
        else:
            n_clusters = 0
        cluster_counts.append(n_clusters)

        # Get number of spatial bins (superpixels)
        if 'superpixel_labels' in result:
            n_bins.append(len(np.unique(result['superpixel_labels'])))
        elif 'features' in result:
            n_bins.append(len(result['features']))
        else:
            n_bins.append(0)

    # Compute consistency metrics
    consistency_metrics = {
        'scales_analyzed': scales,
        'n_scales': len(scales),
        'scale_progression': {
            'n_clusters_per_scale': cluster_counts,
            'n_bins_per_scale': n_bins,
            'scales_um': scales
        }
    }

    # Overall consistency statistics
    if len(scales) >= 2:
        consistency_metrics['overall'] = {
            'cluster_count_stability': float(np.std(cluster_counts) / (np.mean(cluster_counts) + 1e-8)),
            'spatial_coverage_consistency': float(np.std(n_bins) / (np.mean(n_bins) + 1e-8)),
            'mean_clusters_per_scale': float(np.mean(cluster_counts)),
            'mean_bins_per_scale': float(np.mean(n_bins))
        }
    else:
        consistency_metrics['overall'] = {}

    return consistency_metrics


# Old pairwise comparison functions removed - using hierarchical validation instead

def compute_adjusted_rand_index(
    result1: Dict, 
    result2: Dict,
    overlap_threshold: float = 0.5
) -> float:
    """
    Compute Adjusted Rand Index between two scale clusterings.
    
    For different spatial scales, we need to handle overlapping regions.
    """
    if 'cluster_map' not in result1 or 'cluster_map' not in result2:
        return np.nan
    
    map1 = result1['cluster_map']
    map2 = result2['cluster_map']
    
    if map1.size == 0 or map2.size == 0:
        return np.nan
    
    # Resample to common resolution (use finer scale)
    if map1.shape != map2.shape:
        from scipy.ndimage import zoom
        
        # Determine target shape (finer resolution)
        if map1.size > map2.size:
            target_shape = map1.shape
            map2_resampled = zoom(map2.astype(float), 
                                  np.array(target_shape) / np.array(map2.shape), 
                                  order=0, mode='nearest').astype(int)
            map1_resampled = map1
        else:
            target_shape = map2.shape
            map1_resampled = zoom(map1.astype(float), 
                                  np.array(target_shape) / np.array(map1.shape), 
                                  order=0, mode='nearest').astype(int)
            map2_resampled = map2
    else:
        map1_resampled = map1
        map2_resampled = map2
    
    # Only compare valid regions (both scales have valid assignments)
    valid_mask = (map1_resampled >= 0) & (map2_resampled >= 0)
    
    if not np.any(valid_mask):
        return np.nan
    
    labels1 = map1_resampled[valid_mask]
    labels2 = map2_resampled[valid_mask]
    
    return adjusted_rand_score(labels1, labels2)


def compute_normalized_mutual_info(result1: Dict, result2: Dict) -> float:
    """
    Compute Normalized Mutual Information between scale clusterings.
    """
    if 'cluster_map' not in result1 or 'cluster_map' not in result2:
        return np.nan
    
    map1 = result1['cluster_map']
    map2 = result2['cluster_map']
    
    if map1.size == 0 or map2.size == 0:
        return np.nan
    
    # Resample to common resolution (similar to ARI computation)
    if map1.shape != map2.shape:
        from scipy.ndimage import zoom
        
        if map1.size > map2.size:
            target_shape = map1.shape
            map2_resampled = zoom(map2.astype(float), 
                                  np.array(target_shape) / np.array(map2.shape), 
                                  order=0, mode='nearest').astype(int)
            map1_resampled = map1
        else:
            target_shape = map2.shape
            map1_resampled = zoom(map1.astype(float), 
                                  np.array(target_shape) / np.array(map1.shape), 
                                  order=0, mode='nearest').astype(int)
            map2_resampled = map2
    else:
        map1_resampled = map1
        map2_resampled = map2
    
    # Only compare valid regions
    valid_mask = (map1_resampled >= 0) & (map2_resampled >= 0)
    
    if not np.any(valid_mask):
        return np.nan
    
    labels1 = map1_resampled[valid_mask]
    labels2 = map2_resampled[valid_mask]
    
    return normalized_mutual_info_score(labels1, labels2)


def compute_cluster_centroid_stability(result1: Dict, result2: Dict) -> Dict[str, float]:
    """
    Compare cluster centroids between scales using protein profiles.
    """
    if 'cluster_centroids' not in result1 or 'cluster_centroids' not in result2:
        return {}
    
    centroids1 = result1['cluster_centroids']
    centroids2 = result2['cluster_centroids']
    
    if not centroids1 or not centroids2:
        return {}
    
    # Get common proteins
    all_proteins1 = set()
    all_proteins2 = set()
    
    for cluster_data in centroids1.values():
        all_proteins1.update(cluster_data.keys())
    
    for cluster_data in centroids2.values():
        all_proteins2.update(cluster_data.keys())
    
    common_proteins = list(all_proteins1 & all_proteins2)
    
    if not common_proteins:
        return {}
    
    # Create centroid matrices
    n_clusters1 = len(centroids1)
    n_clusters2 = len(centroids2)
    n_proteins = len(common_proteins)
    
    matrix1 = np.zeros((n_clusters1, n_proteins))
    matrix2 = np.zeros((n_clusters2, n_proteins))
    
    # Fill matrices
    for i, (cluster_id, cluster_data) in enumerate(centroids1.items()):
        for j, protein in enumerate(common_proteins):
            matrix1[i, j] = cluster_data.get(protein, 0.0)
    
    for i, (cluster_id, cluster_data) in enumerate(centroids2.items()):
        for j, protein in enumerate(common_proteins):
            matrix2[i, j] = cluster_data.get(protein, 0.0)
    
    # Compute pairwise distances between all centroids
    from scipy.spatial.distance import cdist
    distances = cdist(matrix1, matrix2, metric='euclidean')
    
    # Find best matches (Hungarian algorithm would be better, but this is simpler)
    min_distances = []
    for i in range(n_clusters1):
        min_dist = np.min(distances[i])
        min_distances.append(min_dist)
    
    stability_metrics = {
        'mean_centroid_distance': float(np.mean(min_distances)),
        'max_centroid_distance': float(np.max(min_distances)),
        'std_centroid_distance': float(np.std(min_distances))
    }
    
    return stability_metrics


def compute_overall_consistency(consistency_results: Dict[str, Dict]) -> Dict[str, float]:
    """
    Compute overall consistency metrics across all scale pairs.
    """
    ari_scores = []
    nmi_scores = []
    centroid_distances = []
    
    for pair_key, metrics in consistency_results.items():
        if pair_key == 'overall':  # Skip self
            continue
        
        if 'adjusted_rand_index' in metrics and not np.isnan(metrics['adjusted_rand_index']):
            ari_scores.append(metrics['adjusted_rand_index'])
        
        if 'normalized_mutual_info' in metrics and not np.isnan(metrics['normalized_mutual_info']):
            nmi_scores.append(metrics['normalized_mutual_info'])
        
        if 'centroid_stability' in metrics and 'mean_centroid_distance' in metrics['centroid_stability']:
            centroid_distances.append(metrics['centroid_stability']['mean_centroid_distance'])
    
    overall = {}
    
    if ari_scores:
        overall['mean_ari'] = float(np.mean(ari_scores))
        overall['std_ari'] = float(np.std(ari_scores))
    
    if nmi_scores:
        overall['mean_nmi'] = float(np.mean(nmi_scores))
        overall['std_nmi'] = float(np.std(nmi_scores))
    
    if centroid_distances:
        overall['mean_centroid_distance'] = float(np.mean(centroid_distances))
        overall['std_centroid_distance'] = float(np.std(centroid_distances))
    
    return overall


def identify_scale_dependent_features(
    multiscale_results: Dict[float, Dict],
    protein_names: List[str]
) -> Dict[str, Dict[float, float]]:
    """
    Identify proteins that show scale-dependent spatial organization.
    
    Args:
        multiscale_results: Results from multiscale analysis
        protein_names: List of protein names to analyze
        
    Returns:
        Dictionary mapping protein -> scale -> spatial_metric
    """
    scale_features = {}
    
    for protein_name in protein_names:
        scale_features[protein_name] = {}
        
        for scale_um, result in multiscale_results.items():
            # Extract spatial coefficient of variation for this protein at this scale
            if ('standardized_arrays' in result and 
                protein_name in result['standardized_arrays']):
                
                protein_array = result['standardized_arrays'][protein_name]
                
                if protein_array.size > 0:
                    spatial_cv = float(np.std(protein_array) / (np.mean(np.abs(protein_array)) + 1e-10))
                    scale_features[protein_name][scale_um] = spatial_cv
    
    return scale_features


def summarize_multiscale_analysis(
    multiscale_results: Dict[float, Dict],
    consistency_results: Dict[str, Dict]
) -> Dict[str, any]:
    """
    Create comprehensive summary of multiscale analysis.
    """
    # Filter out non-numeric keys (like 'hierarchy')
    numeric_scales = [k for k in multiscale_results.keys() if isinstance(k, (int, float))]
    
    summary = {
        'scales_analyzed': sorted(numeric_scales),
        'consistency_summary': consistency_results.get('scale_progression', {}),
        'scale_specific_summaries': {}
    }
    
    # Summarize each scale
    for scale_um in numeric_scales:
        result = multiscale_results[scale_um]
        scale_summary = {
            'n_clusters_found': len(result.get('cluster_centroids', {})),
            'n_spatial_bins': len(result.get('cluster_labels', [])),
            'method_used': result.get('method', 'unknown')
        }
        
        # Add cluster size statistics
        if 'cluster_labels' in result and len(result['cluster_labels']) > 0:
            unique_labels, counts = np.unique(result['cluster_labels'], return_counts=True)
            scale_summary['cluster_size_stats'] = {
                'mean': float(np.mean(counts)),
                'std': float(np.std(counts)),
                'min': int(np.min(counts)),
                'max': int(np.max(counts))
            }
        
        summary['scale_specific_summaries'][scale_um] = scale_summary
    
    return summary
