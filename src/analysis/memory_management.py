"""
Memory Management and Chunked Processing

Addresses Gemini's critique about memory explosion with large IMC datasets.
Implements efficient memory usage, chunked processing, and monitoring.
"""

import numpy as np
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Iterator, Any
from scipy import sparse
import warnings
from dataclasses import dataclass


@dataclass
class MemoryProfile:
    """Memory usage profile for monitoring."""
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    process_memory_gb: float


def get_memory_profile() -> MemoryProfile:
    """Get current system and process memory usage."""
    # System memory
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    used_gb = mem.used / (1024**3)
    percent_used = mem.percent
    
    # Process memory
    process = psutil.Process()
    process_memory_gb = process.memory_info().rss / (1024**3)
    
    return MemoryProfile(
        total_gb=total_gb,
        available_gb=available_gb,
        used_gb=used_gb,
        percent_used=percent_used,
        process_memory_gb=process_memory_gb
    )


def estimate_memory_requirements(
    n_coords: int,
    n_proteins: int,
    bin_size_um: float,
    coord_bounds: Tuple[float, float, float, float]
) -> Dict[str, float]:
    """
    Estimate memory requirements for ion count processing.
    
    Args:
        n_coords: Number of coordinate points
        n_proteins: Number of proteins
        bin_size_um: Spatial binning size
        coord_bounds: (x_min, x_max, y_min, y_max) coordinate bounds
        
    Returns:
        Dictionary with memory estimates in GB
    """
    x_min, x_max, y_min, y_max = coord_bounds
    
    # Estimate number of spatial bins
    n_bins_x = int((x_max - x_min) / bin_size_um) + 1
    n_bins_y = int((y_max - y_min) / bin_size_um) + 1
    n_total_bins = n_bins_x * n_bins_y
    
    # Memory estimates (in bytes)
    estimates = {}
    
    # Input data
    coords_memory = n_coords * 2 * 8  # x, y coordinates (float64)
    ion_counts_memory = n_coords * n_proteins * 8  # Ion counts (float64)
    
    # Aggregated arrays
    aggregated_memory = n_total_bins * n_proteins * 8
    
    # Transformed arrays (same size as aggregated)
    transformed_memory = aggregated_memory
    
    # Standardized arrays (same size)
    standardized_memory = aggregated_memory
    
    # Feature matrix (bins × proteins)
    feature_matrix_memory = n_total_bins * n_proteins * 8
    
    # Cluster maps and labels
    cluster_memory = n_total_bins * 4  # int32 labels
    
    # Total memory
    total_memory = (coords_memory + ion_counts_memory + aggregated_memory + 
                   transformed_memory + standardized_memory + 
                   feature_matrix_memory + cluster_memory)
    
    # Convert to GB and add safety factor
    safety_factor = 2.0  # 2x safety margin for intermediate calculations
    
    estimates = {
        'input_data_gb': (coords_memory + ion_counts_memory) / (1024**3),
        'processing_arrays_gb': (aggregated_memory + transformed_memory + 
                                standardized_memory) / (1024**3),
        'feature_matrix_gb': feature_matrix_memory / (1024**3),
        'clustering_gb': cluster_memory / (1024**3),
        'total_estimated_gb': total_memory / (1024**3) * safety_factor,
        'n_spatial_bins': n_total_bins
    }
    
    return estimates


def check_memory_availability(required_gb: float, buffer_gb: float = 1.0) -> bool:
    """
    Check if sufficient memory is available for processing.
    
    Args:
        required_gb: Required memory in GB
        buffer_gb: Buffer memory to keep available
        
    Returns:
        True if sufficient memory available
    """
    profile = get_memory_profile()
    available_with_buffer = profile.available_gb - buffer_gb
    
    return available_with_buffer >= required_gb


def suggest_chunk_size(
    total_coords: int,
    n_proteins: int,
    target_memory_gb: float = 2.0
) -> int:
    """
    Suggest chunk size for processing based on memory constraints.
    
    Args:
        total_coords: Total number of coordinate points
        n_proteins: Number of proteins
        target_memory_gb: Target memory usage per chunk
        
    Returns:
        Suggested chunk size (number of coordinates per chunk)
    """
    # Estimate memory per coordinate
    memory_per_coord = (2 + n_proteins) * 8  # coords + ion counts in bytes
    
    # Add overhead for processing
    processing_overhead = 3.0  # 3x overhead for intermediate arrays
    memory_per_coord *= processing_overhead
    
    # Calculate chunk size
    target_memory_bytes = target_memory_gb * (1024**3)
    chunk_size = int(target_memory_bytes / memory_per_coord)
    
    # Ensure reasonable bounds
    chunk_size = max(1000, min(chunk_size, total_coords))
    
    return chunk_size


class ChunkedProcessor:
    """
    Process large datasets in memory-efficient chunks.
    """
    
    def __init__(
        self, 
        memory_limit_gb: float = 4.0,
        monitoring_enabled: bool = True
    ):
        self.memory_limit_gb = memory_limit_gb
        self.monitoring_enabled = monitoring_enabled
        self.memory_snapshots = []
        
    def _take_memory_snapshot(self, label: str):
        """Take memory snapshot if monitoring enabled."""
        if self.monitoring_enabled:
            profile = get_memory_profile()
            self.memory_snapshots.append({
                'label': label,
                'profile': profile
            })
    
    def chunk_coordinates(
        self,
        coords: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        chunk_size: Optional[int] = None
    ) -> Iterator[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Yield chunks of coordinates and ion counts.
        
        Args:
            coords: Full coordinate array
            ion_counts: Full ion counts dictionary
            chunk_size: Size of each chunk (auto-calculated if None)
            
        Yields:
            Tuples of (coord_chunk, ion_counts_chunk)
        """
        if chunk_size is None:
            chunk_size = suggest_chunk_size(
                len(coords), len(ion_counts), self.memory_limit_gb / 2
            )
        
        n_coords = len(coords)
        n_chunks = (n_coords + chunk_size - 1) // chunk_size
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_coords)
            
            # Extract coordinate chunk
            coord_chunk = coords[start_idx:end_idx]
            
            # Extract ion count chunks
            ion_count_chunk = {}
            for protein_name, counts in ion_counts.items():
                ion_count_chunk[protein_name] = counts[start_idx:end_idx]
            
            self._take_memory_snapshot(f'chunk_{i+1}/{n_chunks}')
            
            yield coord_chunk, ion_count_chunk
    
    def process_roi_chunks(
        self,
        coords: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        processing_function: callable,
        **processing_kwargs
    ) -> List[Any]:
        """
        Process ROI data in chunks and combine results.
        
        Args:
            coords: Coordinate array
            ion_counts: Ion counts dictionary
            processing_function: Function to apply to each chunk
            **processing_kwargs: Additional arguments for processing function
            
        Returns:
            List of results from each chunk
        """
        self._take_memory_snapshot('start_chunked_processing')
        
        # Check if chunking is necessary
        memory_est = estimate_memory_requirements(
            len(coords), len(ion_counts),
            processing_kwargs.get('bin_size_um', 20.0),
            (coords[:, 0].min(), coords[:, 0].max(),
             coords[:, 1].min(), coords[:, 1].max())
        )
        
        if memory_est['total_estimated_gb'] <= self.memory_limit_gb:
            # Process without chunking
            self._take_memory_snapshot('single_chunk_processing')
            result = processing_function(coords, ion_counts, **processing_kwargs)
            self._take_memory_snapshot('single_chunk_complete')
            return [result]
        
        # Process in chunks
        chunk_results = []
        
        for i, (coord_chunk, ion_count_chunk) in enumerate(
            self.chunk_coordinates(coords, ion_counts)
        ):
            self._take_memory_snapshot(f'processing_chunk_{i}')
            
            # Process chunk
            chunk_result = processing_function(
                coord_chunk, ion_count_chunk, **processing_kwargs
            )
            chunk_results.append(chunk_result)
            
            # Force garbage collection after each chunk
            gc.collect()
            
            self._take_memory_snapshot(f'chunk_{i}_complete')
        
        self._take_memory_snapshot('all_chunks_complete')
        return chunk_results
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate memory usage report from snapshots."""
        if not self.memory_snapshots:
            return {'error': 'No memory snapshots available'}
        
        peak_memory = max(
            snapshot['profile'].process_memory_gb 
            for snapshot in self.memory_snapshots
        )
        
        memory_trajectory = [
            {
                'label': snapshot['label'],
                'memory_gb': snapshot['profile'].process_memory_gb,
                'available_gb': snapshot['profile'].available_gb
            }
            for snapshot in self.memory_snapshots
        ]
        
        return {
            'peak_memory_gb': peak_memory,
            'memory_limit_gb': self.memory_limit_gb,
            'within_limit': peak_memory <= self.memory_limit_gb,
            'trajectory': memory_trajectory,
            'n_snapshots': len(self.memory_snapshots)
        }


def estimate_memory_usage(coords: np.ndarray, ion_counts: Dict[str, np.ndarray]) -> float:
    """
    Estimate memory usage for ROI data in MB.
    
    Args:
        coords: Coordinate array
        ion_counts: Ion counts dictionary
        
    Returns:
        Estimated memory usage in MB
    """
    # Calculate memory for coordinates (float64)
    coords_memory = coords.nbytes
    
    # Calculate memory for ion counts
    ion_counts_memory = sum(arr.nbytes for arr in ion_counts.values())
    
    # Add processing overhead (2x for intermediate calculations)
    total_memory = (coords_memory + ion_counts_memory) * 2
    
    # Convert to MB
    return total_memory / (1024 * 1024)


# Alias for backward compatibility
MemoryAwareProcessor = ChunkedProcessor


def create_sparse_aggregation_matrix(
    coords: np.ndarray,
    bin_edges_x: np.ndarray,
    bin_edges_y: np.ndarray
) -> sparse.csr_matrix:
    """
    Create sparse matrix for memory-efficient aggregation.
    
    For very sparse IMC data, using sparse matrices can significantly
    reduce memory usage during aggregation.
    
    Args:
        coords: Coordinate array
        bin_edges_x: X bin edges
        bin_edges_y: Y bin edges
        
    Returns:
        Sparse aggregation matrix
    """
    # Digitize coordinates
    x_indices = np.digitize(coords[:, 0], bin_edges_x) - 1
    y_indices = np.digitize(coords[:, 1], bin_edges_y) - 1
    
    # Create flat bin indices
    n_bins_x = len(bin_edges_x) - 1
    n_bins_y = len(bin_edges_y) - 1
    
    # Valid indices only
    valid_mask = (x_indices >= 0) & (x_indices < n_bins_x) & \
                 (y_indices >= 0) & (y_indices < n_bins_y)
    
    if not np.any(valid_mask):
        # Return empty sparse matrix
        return sparse.csr_matrix((len(coords), n_bins_x * n_bins_y))
    
    # Flat bin indices
    flat_bin_indices = y_indices[valid_mask] * n_bins_x + x_indices[valid_mask]
    coord_indices = np.where(valid_mask)[0]
    
    # Create aggregation matrix
    row_indices = coord_indices
    col_indices = flat_bin_indices
    data = np.ones(len(row_indices))
    
    aggregation_matrix = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(coords), n_bins_x * n_bins_y)
    )
    
    return aggregation_matrix


def sparse_aggregate_ion_counts(
    ion_counts: Dict[str, np.ndarray],
    aggregation_matrix: sparse.csr_matrix,
    output_shape: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """
    Perform memory-efficient sparse aggregation of ion counts.
    
    Args:
        ion_counts: Dictionary of protein ion counts
        aggregation_matrix: Sparse aggregation matrix
        output_shape: Shape of output arrays (n_bins_y, n_bins_x)
        
    Returns:
        Dictionary of aggregated count arrays
    """
    aggregated_counts = {}
    
    for protein_name, counts in ion_counts.items():
        # Sparse matrix multiplication for aggregation
        aggregated_flat = aggregation_matrix.T.dot(counts)
        
        # Reshape to 2D
        aggregated_2d = aggregated_flat.reshape(output_shape)
        
        aggregated_counts[protein_name] = aggregated_2d
    
    return aggregated_counts


class MemoryEfficientPipeline:
    """
    Memory-efficient version of the ion count processing pipeline.
    """
    
    def __init__(self, memory_limit_gb: float = 4.0):
        self.processor = ChunkedProcessor(memory_limit_gb)
        
    def process_large_roi(
        self,
        coords: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        bin_size_um: float = 20.0,
        use_sparse: bool = True,
        **kwargs
    ) -> Dict:
        """
        Process large ROI with memory management.
        
        Args:
            coords: Coordinate array
            ion_counts: Ion counts dictionary
            bin_size_um: Spatial bin size
            use_sparse: Whether to use sparse matrices for aggregation
            **kwargs: Additional processing parameters
            
        Returns:
            Processing results dictionary
        """
        # Estimate memory requirements
        bounds = (coords[:, 0].min(), coords[:, 0].max(),
                 coords[:, 1].min(), coords[:, 1].max())
        
        memory_est = estimate_memory_requirements(
            len(coords), len(ion_counts), bin_size_um, bounds
        )
        
        # Check if memory-efficient processing is needed
        if memory_est['total_estimated_gb'] > self.processor.memory_limit_gb:
            warnings.warn(
                f"Large dataset detected ({memory_est['total_estimated_gb']:.1f} GB estimated). "
                f"Using memory-efficient processing with limit {self.processor.memory_limit_gb:.1f} GB."
            )
            
            if use_sparse and memory_est['total_estimated_gb'] > 8.0:
                # Use sparse processing for very large datasets
                return self._process_with_sparse_methods(
                    coords, ion_counts, bin_size_um, **kwargs
                )
            else:
                # Use chunked processing
                return self._process_with_chunks(
                    coords, ion_counts, bin_size_um, **kwargs
                )
        else:
            # Standard processing
            from .ion_count_processing import ion_count_pipeline
            return ion_count_pipeline(coords, ion_counts, bin_size_um, **kwargs)
    
    def _process_with_sparse_methods(
        self, 
        coords: np.ndarray, 
        ion_counts: Dict[str, np.ndarray],
        bin_size_um: float,
        **kwargs
    ) -> Dict:
        """Process using sparse matrices for very large datasets."""
        self.processor._take_memory_snapshot('sparse_processing_start')
        
        # Create spatial bins
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        bin_edges_x = np.arange(x_min, x_max + bin_size_um, bin_size_um)
        bin_edges_y = np.arange(y_min, y_max + bin_size_um, bin_size_um)
        
        # Create sparse aggregation matrix
        aggregation_matrix = create_sparse_aggregation_matrix(
            coords, bin_edges_x, bin_edges_y
        )
        
        self.processor._take_memory_snapshot('sparse_matrix_created')
        
        # Sparse aggregation
        output_shape = (len(bin_edges_y) - 1, len(bin_edges_x) - 1)
        aggregated_counts = sparse_aggregate_ion_counts(
            ion_counts, aggregation_matrix, output_shape
        )
        
        self.processor._take_memory_snapshot('sparse_aggregation_complete')
        
        # Continue with standard pipeline for smaller aggregated data
        from .ion_count_processing import apply_arcsinh_transform, standardize_features
        from .ion_count_processing import create_feature_matrix, perform_clustering
        from .ion_count_processing import create_cluster_map, compute_cluster_centroids
        
        # Apply transformations
        transformed_arrays, cofactors_used = apply_arcsinh_transform(aggregated_counts)
        standardized_arrays, scalers = standardize_features(transformed_arrays)
        
        # Create feature matrix and cluster
        feature_matrix, protein_names, valid_indices = create_feature_matrix(standardized_arrays)
        
        cluster_labels, kmeans_model, optimization_results = perform_clustering(
            feature_matrix, list(ion_counts.keys()), **kwargs
        )
        
        # Create outputs
        cluster_map = create_cluster_map(cluster_labels, valid_indices, output_shape)
        cluster_centroids = compute_cluster_centroids(feature_matrix, cluster_labels, protein_names)
        
        self.processor._take_memory_snapshot('sparse_processing_complete')
        
        return {
            'aggregated_counts': aggregated_counts,
            'transformed_arrays': transformed_arrays,
            'standardized_arrays': standardized_arrays,
            'scalers': scalers,
            'cofactors_used': cofactors_used,
            'feature_matrix': feature_matrix,
            'cluster_labels': cluster_labels,
            'cluster_map': cluster_map,
            'cluster_centroids': cluster_centroids,
            'kmeans_model': kmeans_model,
            'optimization_results': optimization_results,
            'bin_edges_x': bin_edges_x,
            'bin_edges_y': bin_edges_y,
            'protein_names': protein_names,
            'valid_indices': valid_indices,
            'bin_size_um': bin_size_um,
            'processing_method': 'sparse',
            'memory_report': self.processor.get_memory_report()
        }
    
    def _process_with_chunks(
        self, 
        coords: np.ndarray, 
        ion_counts: Dict[str, np.ndarray],
        bin_size_um: float,
        **kwargs
    ) -> Dict:
        """Process using chunked approach for large datasets."""
        # For chunked processing, we need to carefully combine results
        # This is more complex and would require significant refactoring
        # For now, raise an informative error
        raise NotImplementedError(
            "Chunked processing for full pipeline not yet implemented. "
            "Consider using smaller bin sizes or sparse processing for very large datasets."
        )