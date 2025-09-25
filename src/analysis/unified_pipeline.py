"""
Unified Analysis Pipeline

Consolidates SLIC-based and square binning approaches into a single, coherent interface.
Provides consistent data flow, error handling, and optimization across all analysis modes.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .ion_count_processing import (
    apply_arcsinh_transform, standardize_features, create_feature_matrix,
    perform_clustering, compute_cluster_centroids
)
from .slic_segmentation import slic_pipeline
from .ion_count_processing import ion_count_pipeline
from .error_handling import ErrorHandler, DataValidationError, ProcessingError


class AnalysisMode(Enum):
    """Analysis mode selection."""
    SLIC = "slic"           # Morphology-aware superpixel segmentation
    SQUARE = "square"       # Regular square binning
    AUTO = "auto"           # Automatic selection based on data characteristics


@dataclass
class PipelineConfig:
    """Unified pipeline configuration."""
    # Spatial parameters
    target_scale_um: float = 20.0
    resolution_um: float = 1.0
    
    # SLIC-specific parameters
    compactness: float = 10.0
    
    # Clustering parameters
    n_clusters: Optional[int] = None
    
    # Optimization parameters
    memory_limit_gb: float = 4.0
    use_cached_cofactors: bool = True
    
    # Processing options
    mode: AnalysisMode = AnalysisMode.AUTO
    
    # Error handling
    fail_on_missing_markers: bool = True
    min_valid_pixels: int = 100


@dataclass
class PipelineResult:
    """Unified pipeline result structure."""
    # Core results (always present)
    transformed_arrays: Dict[str, np.ndarray]
    cofactors_used: Dict[str, float]
    standardized_arrays: Dict[str, np.ndarray]
    scalers: Dict[str, Any]
    feature_matrix: np.ndarray
    protein_names: list
    cluster_labels: np.ndarray
    cluster_centroids: Dict[str, Dict[str, float]]
    
    # Metadata
    analysis_mode: str
    scale_um: float
    n_clusters_found: int
    n_spatial_bins: int
    
    # Mode-specific results (optional)
    superpixel_labels: Optional[np.ndarray] = None
    superpixel_coords: Optional[np.ndarray] = None
    superpixel_counts: Optional[Dict[str, np.ndarray]] = None
    composite_dna: Optional[np.ndarray] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    cluster_map: Optional[np.ndarray] = None
    bin_coords: Optional[np.ndarray] = None
    
    # Optimization results
    optimization_results: Optional[Dict] = None
    memory_profile: Optional[Dict] = None


class UnifiedPipeline:
    """
    Unified analysis pipeline that consolidates SLIC and square binning approaches.
    
    Provides consistent interfaces, error handling, and optimization across different
    spatial analysis methods while maintaining backwards compatibility.
    """
    
    def __init__(self, config: Optional['Config'] = None):
        """Initialize unified pipeline with optional configuration."""
        self.config = config
        self.logger = logging.getLogger('UnifiedPipeline')
        self._cached_cofactors = {}
        self.error_handler = ErrorHandler('UnifiedPipeline')
    
    def analyze(
        self,
        coords: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        dna1_intensities: np.ndarray,
        dna2_intensities: np.ndarray,
        pipeline_config: PipelineConfig
    ) -> PipelineResult:
        """
        Execute unified analysis pipeline.
        
        Args:
            coords: Nx2 coordinate array
            ion_counts: Dictionary of protein ion counts
            dna1_intensities: DNA1 channel data  
            dna2_intensities: DNA2 channel data
            pipeline_config: Pipeline configuration
            
        Returns:
            Unified pipeline result
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If analysis fails
        """
        self.logger.info(f"Starting unified analysis (mode: {pipeline_config.mode.value}, scale: {pipeline_config.target_scale_um}μm)")
        
        # Validate inputs using error handler
        self.error_handler.validate_coordinates(coords, "pipeline_input")
        self.error_handler.validate_ion_counts(ion_counts, len(coords), "pipeline_input")
        self._validate_pipeline_config(pipeline_config, len(coords))
        
        # Determine analysis mode
        analysis_mode = self._determine_analysis_mode(pipeline_config, coords, ion_counts)
        
        # Execute analysis with error handling
        if analysis_mode == AnalysisMode.SLIC:
            return self.error_handler.safe_execute(
                self._execute_slic_analysis,
                "SLIC_analysis",
                ProcessingError,
                coords=coords,
                ion_counts=ion_counts,
                dna1_intensities=dna1_intensities,
                dna2_intensities=dna2_intensities,
                pipeline_config=pipeline_config
            )
        else:
            return self.error_handler.safe_execute(
                self._execute_square_analysis,
                "square_analysis", 
                ProcessingError,
                coords=coords,
                ion_counts=ion_counts,
                pipeline_config=pipeline_config
            )
    
    def _validate_pipeline_config(self, pipeline_config: PipelineConfig, n_pixels: int) -> None:
        """Validate pipeline configuration."""
        if n_pixels < pipeline_config.min_valid_pixels:
            raise DataValidationError(
                self.error_handler._create_error(
                    f"Insufficient pixels: {n_pixels} < {pipeline_config.min_valid_pixels}",
                    self.error_handler.ErrorSeverity.CRITICAL,
                    "INSUFFICIENT_PIXELS",
                    {"n_pixels": n_pixels, "min_required": pipeline_config.min_valid_pixels}
                ).pipeline_error
            )
    
    def _determine_analysis_mode(
        self,
        pipeline_config: PipelineConfig,
        coords: np.ndarray,
        ion_counts: Dict[str, np.ndarray]
    ) -> AnalysisMode:
        """Determine optimal analysis mode based on data characteristics."""
        if pipeline_config.mode != AnalysisMode.AUTO:
            return pipeline_config.mode
        
        # Auto-selection logic
        n_pixels = len(coords)
        n_proteins = len(ion_counts)
        
        # Use SLIC for larger datasets with more proteins (better morphology preservation)
        # Use square binning for smaller datasets (faster, simpler)
        if n_pixels > 50000 and n_proteins >= 8:
            self.logger.info("Auto-selecting SLIC mode for large, high-dimensional dataset")
            return AnalysisMode.SLIC
        else:
            self.logger.info("Auto-selecting square binning mode for smaller dataset")
            return AnalysisMode.SQUARE
    
    def _execute_slic_analysis(
        self,
        coords: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        dna1_intensities: np.ndarray,
        dna2_intensities: np.ndarray,
        pipeline_config: PipelineConfig
    ) -> PipelineResult:
        """Execute SLIC-based analysis."""
        self.logger.info("Executing SLIC-based morphology-aware analysis")
        
        # Use existing SLIC pipeline
        slic_result = slic_pipeline(
            coords, ion_counts, dna1_intensities, dna2_intensities,
            target_bin_size_um=pipeline_config.target_scale_um,
            resolution_um=pipeline_config.resolution_um,
            compactness=pipeline_config.compactness,
            config=self.config
        )
        
        # Apply downstream processing to superpixel data
        downstream_result = self._apply_downstream_processing(
            slic_result['superpixel_counts'], pipeline_config
        )
        
        # Combine results
        return PipelineResult(
            # Core results
            transformed_arrays=downstream_result['transformed_arrays'],
            cofactors_used=downstream_result['cofactors_used'],
            standardized_arrays=downstream_result['standardized_arrays'],
            scalers=downstream_result['scalers'],
            feature_matrix=downstream_result['feature_matrix'],
            protein_names=downstream_result['protein_names'],
            cluster_labels=downstream_result['cluster_labels'],
            cluster_centroids=downstream_result['cluster_centroids'],
            
            # Metadata
            analysis_mode="slic",
            scale_um=pipeline_config.target_scale_um,
            n_clusters_found=len(downstream_result['cluster_centroids']),
            n_spatial_bins=len(downstream_result['cluster_labels']),
            
            # SLIC-specific results
            superpixel_labels=slic_result['superpixel_labels'],
            superpixel_coords=slic_result['superpixel_coords'],
            superpixel_counts=slic_result['superpixel_counts'],
            composite_dna=slic_result.get('composite_dna'),
            bounds=slic_result.get('bounds'),
            
            # Optimization results
            optimization_results=downstream_result.get('optimization_results')
        )
    
    def _execute_square_analysis(
        self,
        coords: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        pipeline_config: PipelineConfig
    ) -> PipelineResult:
        """Execute square binning analysis."""
        self.logger.info("Executing square binning analysis")
        
        # Use existing ion count pipeline
        square_result = ion_count_pipeline(
            coords, ion_counts,
            bin_size_um=pipeline_config.target_scale_um,
            n_clusters=pipeline_config.n_clusters,
            memory_limit_gb=pipeline_config.memory_limit_gb
        )
        
        return PipelineResult(
            # Core results
            transformed_arrays=square_result['transformed_arrays'],
            cofactors_used=square_result['cofactors_used'],
            standardized_arrays=square_result['standardized_arrays'],
            scalers=square_result['scalers'],
            feature_matrix=square_result['feature_matrix'],
            protein_names=square_result['protein_names'],
            cluster_labels=square_result['cluster_labels'],
            cluster_centroids=square_result['cluster_centroids'],
            
            # Metadata
            analysis_mode="square",
            scale_um=pipeline_config.target_scale_um,
            n_clusters_found=len(square_result['cluster_centroids']),
            n_spatial_bins=len(square_result['cluster_labels']),
            
            # Square-specific results
            cluster_map=square_result.get('cluster_map'),
            bin_coords=square_result.get('bin_coords'),
            
            # Optimization results
            optimization_results=square_result.get('optimization_results'),
            memory_profile=square_result.get('memory_profile')
        )
    
    def _apply_downstream_processing(
        self,
        aggregated_data: Dict[str, np.ndarray],
        pipeline_config: PipelineConfig
    ) -> Dict:
        """Apply standardized downstream processing to aggregated data."""
        # Step 1: ArcSinh transformation with caching
        if pipeline_config.use_cached_cofactors and self._cached_cofactors:
            transformed_arrays, cofactors_used = apply_arcsinh_transform(
                aggregated_data,
                optimization_method="cached",
                cached_cofactors=self._cached_cofactors
            )
        else:
            transformed_arrays, cofactors_used = apply_arcsinh_transform(
                aggregated_data,
                optimization_method="percentile",
                percentile_threshold=5.0
            )
            # Cache for future use
            self._cached_cofactors = cofactors_used
        
        # Step 2: Create mask (all aggregated regions are valid)
        first_protein = next(iter(transformed_arrays.keys()))
        n_regions = len(transformed_arrays[first_protein])
        mask = np.ones(n_regions, dtype=bool)
        
        # Step 3: Standardize features
        standardized_arrays, scalers = standardize_features(transformed_arrays, mask)
        
        # Step 4: Create feature matrix
        feature_matrix, protein_names, valid_indices = create_feature_matrix(standardized_arrays, mask)
        
        # Step 5: Clustering
        cluster_labels, kmeans_model, optimization_results = perform_clustering(
            feature_matrix, protein_names, 
            n_clusters=pipeline_config.n_clusters, 
            random_state=42
        )
        
        # Step 6: Compute centroids
        cluster_centroids = compute_cluster_centroids(feature_matrix, cluster_labels, protein_names)
        
        return {
            'transformed_arrays': transformed_arrays,
            'cofactors_used': cofactors_used,
            'standardized_arrays': standardized_arrays,
            'scalers': scalers,
            'feature_matrix': feature_matrix,
            'protein_names': protein_names,
            'cluster_labels': cluster_labels,
            'kmeans_model': kmeans_model,
            'cluster_centroids': cluster_centroids,
            'optimization_results': optimization_results
        }
    
    def clear_cache(self) -> None:
        """Clear cached cofactors."""
        self._cached_cofactors.clear()
        self.logger.info("Cleared cofactor cache")


# Convenience functions for backwards compatibility
def unified_analysis(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    dna1_intensities: np.ndarray,
    dna2_intensities: np.ndarray,
    target_scale_um: float = 20.0,
    mode: Union[str, AnalysisMode] = AnalysisMode.AUTO,
    config: Optional['Config'] = None
) -> PipelineResult:
    """
    Convenience function for unified analysis.
    
    Args:
        coords: Nx2 coordinate array
        ion_counts: Dictionary of protein ion counts
        dna1_intensities: DNA1 channel data
        dna2_intensities: DNA2 channel data
        target_scale_um: Target spatial scale in micrometers
        mode: Analysis mode ('slic', 'square', or 'auto')
        config: Optional configuration object
        
    Returns:
        Unified pipeline result
    """
    if isinstance(mode, str):
        mode = AnalysisMode(mode)
    
    pipeline_config = PipelineConfig(
        target_scale_um=target_scale_um,
        mode=mode
    )
    
    pipeline = UnifiedPipeline(config)
    return pipeline.analyze(coords, ion_counts, dna1_intensities, dna2_intensities, pipeline_config)