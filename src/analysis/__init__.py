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

# Clustering optimization
from .clustering_optimization import (
    optimize_clustering_parameters,
    elbow_method,
    silhouette_analysis,
    gap_statistic,
    biological_validation_score
)

# Memory management
from .memory_management import (
    MemoryEfficientPipeline,
    estimate_memory_requirements,
    check_memory_availability,
    get_memory_profile
)

# Configuration management
from .config_management import (
    IMCAnalysisConfig,
    ConfigurationManager,
    ArcSinhConfig,
    ClusteringConfig,
    MemoryConfig,
    SLICConfig,
    MultiScaleConfig
)

# SLIC morphology-aware segmentation
from .slic_segmentation import (
    slic_pipeline,
    prepare_dna_composite,
    perform_slic_segmentation,
    aggregate_to_superpixels
)

# Multi-scale analysis
from .multiscale_analysis import (
    perform_multiscale_analysis,
    compute_scale_consistency,
    identify_scale_dependent_features,
    summarize_multiscale_analysis,
    compute_adjusted_rand_index,
    compute_normalized_mutual_info
)

# Batch correction
from .batch_correction import (
    sham_anchored_normalize,
    detect_batch_effects
)

# Validation framework
from .validation import (
    generate_synthetic_imc_data,
    validate_clustering_performance,
    run_validation_experiment,
    summarize_validation_results
)

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

# Metrics
from .metrics import (
    ValidationResult,
    ClusterValidator,
    SilhouetteValidator,
    SpatialCoherenceValidator,
    ValidationSuite
)

# Main pipeline orchestrator
from .main_pipeline import (
    run_complete_analysis
)

# Efficient storage system
try:
    from .efficient_storage import (
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
    
    # Clustering Optimization
    'optimize_clustering_parameters',
    'elbow_method',
    'silhouette_analysis',
    'gap_statistic',
    'biological_validation_score',
    
    # Memory Management
    'MemoryEfficientPipeline',
    'estimate_memory_requirements',
    'check_memory_availability',
    'get_memory_profile',
    
    # Configuration Management
    'IMCAnalysisConfig',
    'ConfigurationManager',
    'ArcSinhConfig',
    'ClusteringConfig',
    'MemoryConfig',
    'SLICConfig',
    'MultiScaleConfig',
    
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
    
    # Validation
    'generate_synthetic_imc_data',
    'validate_clustering_performance',
    'run_validation_experiment',
    'summarize_validation_results',
    
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
    
    # Metrics
    'compute_ari',
    'compute_nmi',
    'compute_silhouette',
    'compute_calinski_harabasz',
    'compute_davies_bouldin',
    
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