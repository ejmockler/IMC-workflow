"""
Memory-Efficient Streaming Validation

Processes large validation datasets without loading everything into memory.
Integrates with ValidationCache for optimal performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple
import logging
import mmap
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

from .cache import ValidationCache
from .framework import ValidationResult, ValidationSeverity

logger = logging.getLogger(__name__)


class StreamingDataLoader:
    """Memory-efficient data loader for large IMC datasets."""
    
    def __init__(self, cache: ValidationCache, max_memory_mb: int = 2000):
        """Initialize streaming data loader.
        
        Args:
            cache: ValidationCache instance
            max_memory_mb: Maximum memory usage in MB
        """
        self.cache = cache
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_usage = 0
        
        logger.info(f"StreamingDataLoader initialized with {max_memory_mb}MB memory limit")
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        process = psutil.Process()
        return process.memory_info().rss
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        current_usage = self.get_memory_usage()
        return current_usage < self.max_memory_bytes
    
    def stream_roi_data(self, roi_files: List[Path], chunk_size: int = 5) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Stream ROI data in memory-efficient chunks.
        
        Args:
            roi_files: List of ROI file paths
            chunk_size: Number of ROIs to process simultaneously
            
        Yields:
            (roi_id, data) tuples
        """
        for i in range(0, len(roi_files), chunk_size):
            chunk_files = roi_files[i:i+chunk_size]
            
            # Check memory before processing chunk
            if not self.check_memory_limit():
                logger.warning(f"Memory limit exceeded, reducing chunk size")
                chunk_size = max(1, chunk_size // 2)
                chunk_files = roi_files[i:i+chunk_size]
            
            # Process chunk
            for roi_file in chunk_files:
                try:
                    roi_id = roi_file.stem
                    
                    # Try cache first
                    file_hash = self.cache.compute_file_hash(roi_file)
                    cached_data = self._try_load_from_cache(roi_id, file_hash)
                    
                    if cached_data:
                        yield roi_id, cached_data
                    else:
                        # Load from file with memory mapping
                        data = self._load_roi_file_memory_mapped(roi_file)
                        if data:
                            # Cache for future use
                            self._cache_roi_data(roi_id, data, file_hash)
                            yield roi_id, data
                            
                except Exception as e:
                    logger.error(f"Error streaming ROI {roi_file}: {e}")
                    continue
    
    def _try_load_from_cache(self, roi_id: str, file_hash: str) -> Optional[Dict[str, Any]]:
        """Try to load ROI data from cache."""
        
        # Try geometric metrics
        geometric = self.cache.load_geometric_metrics(roi_id)
        if geometric and self.cache.is_cached(roi_id, "geometric", file_hash):
            data = {'geometric_metrics': geometric}
            
            # Try spatial adjacency
            spatial = self.cache.load_spatial_adjacency(roi_id)
            if spatial:
                data['spatial_adjacency'] = spatial
            
            # Try ion count stats
            ion_stats = self.cache.load_ion_count_stats(roi_id, file_hash)
            if ion_stats:
                data['ion_count_stats'] = ion_stats
            
            logger.debug(f"Loaded {roi_id} from cache")
            return data
        
        return None
    
    def _load_roi_file_memory_mapped(self, roi_file: Path) -> Optional[Dict[str, Any]]:
        """Load ROI file using memory mapping for efficiency."""
        
        try:
            if roi_file.suffix.lower() == '.txt':
                return self._load_text_file_chunked(roi_file)
            elif roi_file.suffix.lower() == '.json':
                return self._load_json_file_streamed(roi_file)
            elif roi_file.suffix.lower() in ['.npz', '.npy']:
                return self._load_numpy_file_mapped(roi_file)
            else:
                logger.warning(f"Unsupported file format: {roi_file}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading {roi_file}: {e}")
            return None
    
    def _load_text_file_chunked(self, roi_file: Path) -> Optional[Dict[str, Any]]:
        """Load large text files in chunks."""
        
        try:
            # Use pandas with chunking for large files
            chunk_size = 10000  # Process 10k rows at a time
            
            # Read file info first
            with open(roi_file, 'r') as f:
                header = f.readline().strip().split('\t')
                
            # Identify coordinate and protein columns
            coord_cols = ['X', 'Y']
            protein_cols = [col for col in header if any(marker in col for marker in 
                          ['CD', 'DNA', 'Ki67', 'αSMA', 'Collagen', 'Vimentin'])]
            
            # Stream process the file
            coords_list = []
            ion_counts = {protein: [] for protein in protein_cols}
            
            for chunk in pd.read_csv(roi_file, sep='\t', chunksize=chunk_size):
                # Extract coordinates
                if all(col in chunk.columns for col in coord_cols):
                    coords_list.append(chunk[coord_cols].values)
                
                # Extract ion counts
                for protein in protein_cols:
                    if protein in chunk.columns:
                        ion_counts[protein].append(chunk[protein].values)
            
            # Combine chunks
            data = {}
            if coords_list:
                data['coords'] = np.vstack(coords_list)
            
            if any(ion_counts.values()):
                data['ion_counts'] = {
                    protein: np.concatenate(values) if values else np.array([])
                    for protein, values in ion_counts.items()
                }
            
            return data if data else None
            
        except Exception as e:
            logger.error(f"Error loading text file {roi_file}: {e}")
            return None
    
    def _load_json_file_streamed(self, roi_file: Path) -> Optional[Dict[str, Any]]:
        """Load JSON files with streaming parser for large files."""
        
        try:
            # For small files, load normally
            if roi_file.stat().st_size < 10 * 1024 * 1024:  # 10MB
                with open(roi_file, 'r') as f:
                    return json.load(f)
            
            # For large files, use streaming approach
            # This is a placeholder - in practice, you'd need a streaming JSON parser
            # like ijson for very large files
            with open(roi_file, 'r') as f:
                data = json.load(f)
                return data
                
        except Exception as e:
            logger.error(f"Error loading JSON file {roi_file}: {e}")
            return None
    
    def _load_numpy_file_mapped(self, roi_file: Path) -> Optional[Dict[str, Any]]:
        """Load numpy files with memory mapping."""
        
        try:
            if roi_file.suffix.lower() == '.npz':
                # Load npz with selective reading
                npz_data = np.load(roi_file)
                
                # Only load arrays we need, use memory mapping when possible
                data = {}
                for key in npz_data.files:
                    # Load small arrays directly, memory map large ones
                    array = npz_data[key]
                    if array.nbytes < 50 * 1024 * 1024:  # 50MB limit
                        data[key] = array
                    else:
                        # For very large arrays, store path for lazy loading
                        data[key + '_path'] = str(roi_file)
                        data[key + '_key'] = key
                
                return data
                
            elif roi_file.suffix.lower() == '.npy':
                # Memory map numpy arrays
                return {'array': np.load(roi_file, mmap_mode='r')}
            
        except Exception as e:
            logger.error(f"Error loading numpy file {roi_file}: {e}")
            return None
    
    def _cache_roi_data(self, roi_id: str, data: Dict[str, Any], file_hash: str):
        """Cache processed ROI data."""
        
        try:
            # Cache geometric data if available
            if 'segmentation' in data:
                segmentation = data['segmentation']
                geometric_metrics = self._compute_basic_geometric_metrics(segmentation)
                self.cache.save_geometric_metrics(roi_id, segmentation, geometric_metrics, file_hash)
            
            # Cache ion count stats if available
            if 'ion_counts' in data:
                ion_stats = self._compute_basic_ion_stats(data['ion_counts'])
                self.cache.save_ion_count_stats(roi_id, ion_stats, file_hash)
                
        except Exception as e:
            logger.warning(f"Error caching data for {roi_id}: {e}")
    
    def _compute_basic_geometric_metrics(self, segmentation: np.ndarray) -> Dict[str, Any]:
        """Compute basic geometric metrics for caching."""
        
        unique_segments, counts = np.unique(segmentation, return_counts=True)
        valid_mask = unique_segments >= 0
        
        return {
            'n_segments': int(np.sum(valid_mask)),
            'segment_sizes': counts[valid_mask].tolist(),
            'mean_size': float(np.mean(counts[valid_mask])),
            'std_size': float(np.std(counts[valid_mask])),
            'total_pixels': int(segmentation.size)
        }
    
    def _compute_basic_ion_stats(self, ion_counts: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compute basic ion count statistics for caching."""
        
        stats = {}
        for protein, counts in ion_counts.items():
            if len(counts) > 0:
                stats[protein] = {
                    'mean': float(np.mean(counts)),
                    'std': float(np.std(counts)),
                    'min': float(np.min(counts)),
                    'max': float(np.max(counts)),
                    'positive_fraction': float(np.sum(counts > 0) / len(counts)),
                    'n_points': int(len(counts))
                }
            else:
                stats[protein] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                    'positive_fraction': 0.0, 'n_points': 0
                }
        
        return stats


class StreamingValidator:
    """Streaming validation processor."""
    
    def __init__(self, cache: ValidationCache, max_workers: int = 4):
        """Initialize streaming validator.
        
        Args:
            cache: ValidationCache instance
            max_workers: Maximum parallel workers
        """
        self.cache = cache
        self.max_workers = max_workers
        self.data_loader = StreamingDataLoader(cache)
    
    def validate_dataset_streaming(self, roi_files: List[Path], 
                                 validators: List[Any]) -> Iterator[ValidationResult]:
        """Validate entire dataset using streaming approach.
        
        Args:
            roi_files: List of ROI files to validate
            validators: List of validation rules to apply
            
        Yields:
            ValidationResult objects
        """
        logger.info(f"Starting streaming validation of {len(roi_files)} ROIs")
        
        # Process in batches to manage memory
        batch_size = min(10, max(1, self.max_workers * 2))
        
        for roi_id, data in self.data_loader.stream_roi_data(roi_files, batch_size):
            try:
                # Run all validators on this ROI
                for validator in validators:
                    try:
                        result = validator.validate(data)
                        result.context['roi_id'] = roi_id
                        yield result
                        
                    except Exception as e:
                        logger.error(f"Validator {validator.name} failed for {roi_id}: {e}")
                        
                        # Create error result
                        yield ValidationResult(
                            rule_name=validator.name,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"Validation failed: {str(e)}",
                            quality_score=0.0,
                            context={'roi_id': roi_id, 'error': str(e)}
                        )
                        
            except Exception as e:
                logger.error(f"Error processing ROI {roi_id}: {e}")
                continue
    
    def validate_parallel_streaming(self, roi_files: List[Path], 
                                  validators: List[Any]) -> List[ValidationResult]:
        """Validate dataset using parallel streaming.
        
        Args:
            roi_files: List of ROI files to validate
            validators: List of validation rules to apply
            
        Returns:
            List of all validation results
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit validation tasks
            future_to_roi = {}
            
            for roi_id, data in self.data_loader.stream_roi_data(roi_files):
                for validator in validators:
                    future = executor.submit(self._validate_single, validator, data, roi_id)
                    future_to_roi[future] = (roi_id, validator.name)
            
            # Collect results
            for future in as_completed(future_to_roi):
                roi_id, validator_name = future_to_roi[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Validation failed for {roi_id}, {validator_name}: {e}")
                    
                    # Create error result
                    results.append(ValidationResult(
                        rule_name=validator_name,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Validation failed: {str(e)}",
                        quality_score=0.0,
                        context={'roi_id': roi_id, 'error': str(e)}
                    ))
        
        return results
    
    def _validate_single(self, validator: Any, data: Dict[str, Any], roi_id: str) -> ValidationResult:
        """Validate single ROI with single validator."""
        
        result = validator.validate(data)
        result.context['roi_id'] = roi_id
        return result