"""
Analysis module for IMC data processing.

Production-quality implementation of ion count processing, morphology-aware segmentation,
and multi-scale analysis for IMC data.
"""

# Core ion count processing
from .ion_count_processing import (
    ion_count_pipeline,
    aggregate_ion_counts,
    apply_arcsinh_transform,
    standardize_features,
    create_feature_matrix,
    perform_clustering,
    estimate_optimal_cofactor,
    optimize_cofactors_for_dataset
)

# Spatial clustering (NEW - replaces old clustering_optimization)
from .spatial_clustering import (
    perform_spatial_clustering,
    stability_analysis,
    compute_spatial_coherence,
    select_resolution_headless
)

# Co-abundance feature engineering
from .coabundance_features import (
    generate_coabundance_features,
    compute_local_covariance,
    select_informative_coabundance_features,
    compute_protein_modules,
    create_hierarchical_features
)

# Hierarchical multi-scale analysis
from .hierarchical_multiscale import (
    build_multiscale_hierarchy,
    coarsen_graph,
    add_neighbor_composition_features,
    compute_texture_features,
    validate_hierarchy
)

# Memory management
from .memory_management import (
    MemoryEfficientPipeline,
    estimate_memory_requirements,
    check_memory_availability,
    get_memory_profile
)

# Configuration management
# Config management now consolidated in src/config.py

# SLIC morphology-aware segmentation
from .slic_segmentation import (
    slic_pipeline,
    prepare_dna_composite,
    perform_slic_segmentation,
    aggregate_to_superpixels
)

# Multi-scale analysis
try:
    from .multiscale_analysis import (
        perform_multiscale_analysis,
        compute_scale_consistency,
        identify_scale_dependent_features,
        summarize_multiscale_analysis,
        compute_adjusted_rand_index,
        compute_normalized_mutual_info
    )
except ImportError:
    # Multiscale analysis temporarily disabled due to import issues
    pass

# Batch correction
from .batch_correction import (
    sham_anchored_normalize,
    detect_batch_effects
)

# Note: Validation moved to src/validation/core for clean separation

# Parallel processing
from .parallel_processing import (
    parallel_roi_analysis,
    create_roi_batch_processor,
    get_optimal_process_count
)

# Spatial statistics
from .spatial_stats import (
    compute_spatial_correlation,
    compute_region_difference,
    compute_ripleys_k,
    spatial_bootstrap
)

# Threshold analysis
from .threshold_analysis import (
    extract_threshold_features,
    compute_spatial_clustering
)

# Note: Validation metrics moved to src/validation/ framework

# Main pipeline orchestrator
from .main_pipeline import (
    run_complete_analysis
)

# Efficient storage system
try:
    from .data_storage import (
        create_storage_backend,
        HDF5Storage,
        ParquetStorage,
        HybridStorage,
        CompressedJSONStorage
    )
    _storage_available = True
except ImportError:
    create_storage_backend = None
    HDF5Storage = None
    ParquetStorage = None
    HybridStorage = None
    CompressedJSONStorage = None
    _storage_available = False

__all__ = [
    # Core Ion Count Processing
    'ion_count_pipeline',
    'aggregate_ion_counts',
    'apply_arcsinh_transform',
    'standardize_features',
    'create_feature_matrix',
    'perform_clustering',
    'estimate_optimal_cofactor',
    'optimize_cofactors_for_dataset',
    
    # Spatial Clustering (NEW)
    'perform_spatial_clustering',
    'stability_analysis',
    'compute_spatial_coherence',
    'select_resolution_headless',
    
    # Co-abundance Features
    'generate_coabundance_features',
    'compute_local_covariance',
    'select_informative_coabundance_features',
    'compute_protein_modules',
    'create_hierarchical_features',
    
    # Hierarchical Multi-scale
    'build_multiscale_hierarchy',
    'coarsen_graph',
    'add_neighbor_composition_features',
    'compute_texture_features',
    'validate_hierarchy',
    
    # Memory Management
    'MemoryEfficientPipeline',
    'estimate_memory_requirements',
    'check_memory_availability',
    'get_memory_profile',
    
    # Configuration Management
    # Config management consolidated in src/config.py
    
    # SLIC Segmentation
    'slic_pipeline',
    'prepare_dna_composite',
    'perform_slic_segmentation',
    'aggregate_to_superpixels',
    
    # Multi-scale Analysis
    'perform_multiscale_analysis',
    'compute_scale_consistency',
    'identify_scale_dependent_features',
    'summarize_multiscale_analysis',
    'compute_adjusted_rand_index',
    'compute_normalized_mutual_info',
    
    # Batch Correction
    'sham_anchored_normalize',
    'detect_batch_effects',
    'correct_batch_effects',
    
    
    # Parallel Processing
    'parallel_roi_analysis',
    'create_roi_batch_processor',
    'get_optimal_process_count',
    
    # Spatial Statistics
    'compute_spatial_correlation',
    'compute_region_difference',
    'compute_ripleys_k',
    'spatial_bootstrap',
    
    # Threshold Analysis
    'extract_threshold_features',
    'compute_spatial_clustering',
    
    # Note: Validation metrics moved to src/validation/ framework
    
    # Main Pipeline
    'run_production_pipeline',
    'validate_pipeline_inputs',
    'generate_pipeline_report',
]

# Add storage components if available
if _storage_available:
    __all__.extend([
        'create_storage_backend',
        'HDF5Storage',
        'ParquetStorage', 
        'HybridStorage',
        'CompressedJSONStorage'
    ])