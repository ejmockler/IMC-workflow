"""
Analysis module for IMC data processing.

Implements ion count processing, morphology-aware segmentation, and multi-scale analysis
for IMC data using proper statistical methods for n=2 pilot studies.
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
    summarize_multiscale_analysis
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

# Spatial statistics (descriptive only)
from .spatial_stats import (
    compute_spatial_correlation,
    compute_region_difference,
    compute_ripleys_k,
    spatial_bootstrap
)

# Threshold analysis (alternative approach)
from .threshold_analysis import (
    extract_threshold_features,
    compute_spatial_clustering
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

# Legacy components (maintained for backward compatibility)
# Note: Some legacy components may have broken imports after PFD module removal
try:
    from .pipeline import PFDPipeline, run_pfd_analysis
except ImportError:
    PFDPipeline = None
    run_pfd_analysis = None

try:
    from .roi_main import BatchAnalyzer, ROIAnalyzer
except ImportError:
    BatchAnalyzer = None
    ROIAnalyzer = None

try:
    from .network import NetworkAnalyzer
except ImportError:
    NetworkAnalyzer = None

try:
    from .metadata_driven import MetadataDrivenAnalyzer
except ImportError:
    MetadataDrivenAnalyzer = None

try:
    from .biological_interpretation import BiologicalInterpreter
except ImportError:
    BiologicalInterpreter = None

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
    
    # Legacy (Backward Compatibility)
    # Note: These may be None if imports failed
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

# Add legacy components if they were successfully imported
legacy_components = [
    ('PFDPipeline', PFDPipeline),
    ('run_pfd_analysis', run_pfd_analysis),
    ('BatchAnalyzer', BatchAnalyzer),
    ('ROIAnalyzer', ROIAnalyzer),
    ('NetworkAnalyzer', NetworkAnalyzer),
    ('MetadataDrivenAnalyzer', MetadataDrivenAnalyzer),
    ('BiologicalInterpreter', BiologicalInterpreter)
]

for name, component in legacy_components:
    if component is not None:
        __all__.append(name)