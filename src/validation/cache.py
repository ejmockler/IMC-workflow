"""
High-Performance Validation Cache System

Pre-computes expensive validation metrics and stores them on disk.
Eliminates redundant computation and enables memory-efficient streaming.
"""

import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from scipy import sparse
import h5py
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CacheMetadata:
    """Metadata for cached validation results."""
    roi_id: str
    file_hash: str
    cache_version: str
    created_at: str
    data_shape: Tuple[int, ...]
    metrics_computed: List[str]


class ValidationCache:
    """High-performance cache for validation metrics and intermediate results."""
    
    def __init__(self, cache_dir: str = ".validation_cache", cache_version: str = "v2.0"):
        """Initialize validation cache.
        
        Args:
            cache_dir: Directory for cache storage
            cache_version: Version string for cache compatibility
        """
        self.cache_dir = Path(cache_dir)
        self.cache_version = cache_version
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        (self.cache_dir / "geometric").mkdir(exist_ok=True)
        (self.cache_dir / "spatial").mkdir(exist_ok=True)
        (self.cache_dir / "adjacency").mkdir(exist_ok=True)
        (self.cache_dir / "checksums").mkdir(exist_ok=True)
        
        logger.info(f"ValidationCache initialized: {self.cache_dir}")
    
    def compute_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Compute SHA-256 hash of file for integrity checking.
        
        Args:
            file_path: Path to file
            chunk_size: Chunk size for reading large files
            
        Returns:
            Hexadecimal SHA-256 hash
        """
        hasher = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return ""
    
    def get_cache_key(self, roi_id: str, metric_type: str) -> str:
        """Generate cache key for ROI and metric type."""
        return f"{roi_id}_{metric_type}_{self.cache_version}"
    
    def is_cached(self, roi_id: str, metric_type: str, file_hash: str = None) -> bool:
        """Check if metric is cached and valid.
        
        Args:
            roi_id: ROI identifier
            metric_type: Type of metric ('geometric', 'spatial', 'adjacency')
            file_hash: Optional file hash for integrity check
            
        Returns:
            True if cached and valid
        """
        cache_key = self.get_cache_key(roi_id, metric_type)
        cache_path = self.cache_dir / metric_type / f"{cache_key}.h5"
        metadata_path = self.cache_dir / "metadata" / f"{cache_key}.json"
        
        if not (cache_path.exists() and metadata_path.exists()):
            return False
        
        try:
            # Check metadata
            with open(metadata_path, 'r') as f:
                metadata = CacheMetadata(**json.load(f))
            
            # Version check
            if metadata.cache_version != self.cache_version:
                return False
            
            # Hash check if provided
            if file_hash and metadata.file_hash != file_hash:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache validation failed for {cache_key}: {e}")
            return False
    
    def save_geometric_metrics(self, roi_id: str, segmentation: np.ndarray, 
                              metrics: Dict[str, Any], file_hash: str = "") -> bool:
        """Save pre-computed geometric metrics to cache.
        
        Args:
            roi_id: ROI identifier
            segmentation: Segmentation array
            metrics: Computed metrics
            file_hash: File integrity hash
            
        Returns:
            Success status
        """
        try:
            cache_key = self.get_cache_key(roi_id, "geometric")
            cache_path = self.cache_dir / "geometric" / f"{cache_key}.h5"
            metadata_path = self.cache_dir / "metadata" / f"{cache_key}.json"
            
            # Save metrics to HDF5
            with h5py.File(cache_path, 'w') as f:
                # Save segmentation if small enough
                if segmentation.nbytes < 100 * 1024 * 1024:  # 100MB limit
                    f.create_dataset('segmentation', data=segmentation, compression='gzip')
                
                # Save all metrics
                metrics_group = f.create_group('metrics')
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_group.attrs[key] = value
                    elif isinstance(value, np.ndarray):
                        metrics_group.create_dataset(key, data=value, compression='gzip')
                    elif isinstance(value, dict):
                        # Save nested dictionaries as JSON strings
                        metrics_group.attrs[key] = json.dumps(value)
            
            # Save metadata
            metadata = CacheMetadata(
                roi_id=roi_id,
                file_hash=file_hash,
                cache_version=self.cache_version,
                created_at=datetime.now().isoformat(),
                data_shape=segmentation.shape,
                metrics_computed=list(metrics.keys())
            )
            
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            logger.debug(f"Cached geometric metrics for {roi_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching geometric metrics for {roi_id}: {e}")
            return False
    
    def load_geometric_metrics(self, roi_id: str) -> Optional[Dict[str, Any]]:
        """Load cached geometric metrics.
        
        Args:
            roi_id: ROI identifier
            
        Returns:
            Cached metrics or None if not found
        """
        try:
            cache_key = self.get_cache_key(roi_id, "geometric")
            cache_path = self.cache_dir / "geometric" / f"{cache_key}.h5"
            
            if not cache_path.exists():
                return None
            
            metrics = {}
            with h5py.File(cache_path, 'r') as f:
                if 'segmentation' in f:
                    metrics['segmentation'] = f['segmentation'][:]
                
                if 'metrics' in f:
                    metrics_group = f['metrics']
                    
                    # Load attributes
                    for key, value in metrics_group.attrs.items():
                        if isinstance(value, str) and value.startswith('{'):
                            # Try to parse JSON
                            try:
                                metrics[key] = json.loads(value)
                            except:
                                metrics[key] = value
                        else:
                            metrics[key] = value
                    
                    # Load datasets
                    for key in metrics_group.keys():
                        metrics[key] = metrics_group[key][:]
            
            logger.debug(f"Loaded cached geometric metrics for {roi_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error loading geometric metrics for {roi_id}: {e}")
            return None
    
    def save_spatial_adjacency(self, roi_id: str, adjacency_matrix: sparse.csr_matrix,
                              segment_properties: Dict[str, np.ndarray], 
                              file_hash: str = "") -> bool:
        """Save pre-computed spatial adjacency graph.
        
        Args:
            roi_id: ROI identifier
            adjacency_matrix: Sparse adjacency matrix
            segment_properties: Segment properties (sizes, centroids, etc.)
            file_hash: File integrity hash
            
        Returns:
            Success status
        """
        try:
            cache_key = self.get_cache_key(roi_id, "adjacency")
            cache_path = self.cache_dir / "adjacency" / f"{cache_key}.npz"
            metadata_path = self.cache_dir / "metadata" / f"{cache_key}.json"
            
            # Save sparse matrix and properties
            save_data = {
                'adjacency_data': adjacency_matrix.data,
                'adjacency_indices': adjacency_matrix.indices,
                'adjacency_indptr': adjacency_matrix.indptr,
                'adjacency_shape': np.array(adjacency_matrix.shape)
            }
            
            # Add segment properties
            for key, value in segment_properties.items():
                save_data[f"prop_{key}"] = value
            
            np.savez_compressed(cache_path, **save_data)
            
            # Save metadata
            metadata = CacheMetadata(
                roi_id=roi_id,
                file_hash=file_hash,
                cache_version=self.cache_version,
                created_at=datetime.now().isoformat(),
                data_shape=adjacency_matrix.shape,
                metrics_computed=list(segment_properties.keys())
            )
            
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            logger.debug(f"Cached spatial adjacency for {roi_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching spatial adjacency for {roi_id}: {e}")
            return False
    
    def load_spatial_adjacency(self, roi_id: str) -> Optional[Dict[str, Any]]:
        """Load cached spatial adjacency data.
        
        Args:
            roi_id: ROI identifier
            
        Returns:
            Dictionary with adjacency_matrix and segment properties
        """
        try:
            cache_key = self.get_cache_key(roi_id, "adjacency")
            cache_path = self.cache_dir / "adjacency" / f"{cache_key}.npz"
            
            if not cache_path.exists():
                return None
            
            data = np.load(cache_path)
            
            # Reconstruct sparse matrix
            adjacency_matrix = sparse.csr_matrix(
                (data['adjacency_data'], data['adjacency_indices'], data['adjacency_indptr']),
                shape=tuple(data['adjacency_shape'])
            )
            
            # Extract segment properties
            segment_properties = {}
            for key in data.keys():
                if key.startswith('prop_'):
                    prop_name = key[5:]  # Remove 'prop_' prefix
                    segment_properties[prop_name] = data[key]
            
            result = {
                'adjacency_matrix': adjacency_matrix,
                'segment_properties': segment_properties
            }
            
            logger.debug(f"Loaded cached spatial adjacency for {roi_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading spatial adjacency for {roi_id}: {e}")
            return None
    
    def save_ion_count_stats(self, roi_id: str, protein_stats: Dict[str, Dict[str, float]],
                           file_hash: str = "") -> bool:
        """Save pre-computed ion count statistics.
        
        Args:
            roi_id: ROI identifier
            protein_stats: Statistics for each protein
            file_hash: File integrity hash
            
        Returns:
            Success status
        """
        try:
            cache_key = self.get_cache_key(roi_id, "ion_stats")
            cache_path = self.cache_dir / "metadata" / f"{cache_key}_stats.json"
            
            # Prepare data for JSON serialization
            serializable_stats = {}
            for protein, stats in protein_stats.items():
                serializable_stats[protein] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in stats.items()
                }
            
            save_data = {
                'roi_id': roi_id,
                'file_hash': file_hash,
                'cache_version': self.cache_version,
                'created_at': datetime.now().isoformat(),
                'protein_stats': serializable_stats
            }
            
            with open(cache_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.debug(f"Cached ion count stats for {roi_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching ion count stats for {roi_id}: {e}")
            return False
    
    def load_ion_count_stats(self, roi_id: str, file_hash: str = None) -> Optional[Dict[str, Dict[str, float]]]:
        """Load cached ion count statistics.
        
        Args:
            roi_id: ROI identifier
            file_hash: Optional file hash for validation
            
        Returns:
            Protein statistics or None if not found
        """
        try:
            cache_key = self.get_cache_key(roi_id, "ion_stats")
            cache_path = self.cache_dir / "metadata" / f"{cache_key}_stats.json"
            
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            # Validate hash if provided
            if file_hash and data.get('file_hash') != file_hash:
                return None
            
            # Validate version
            if data.get('cache_version') != self.cache_version:
                return None
            
            logger.debug(f"Loaded cached ion count stats for {roi_id}")
            return data.get('protein_stats', {})
            
        except Exception as e:
            logger.error(f"Error loading ion count stats for {roi_id}: {e}")
            return None
    
    def clear_cache(self, roi_id: str = None, metric_type: str = None) -> int:
        """Clear cache entries.
        
        Args:
            roi_id: Specific ROI to clear (None for all)
            metric_type: Specific metric type to clear (None for all)
            
        Returns:
            Number of files removed
        """
        removed_count = 0
        
        try:
            if roi_id and metric_type:
                # Clear specific cache entry
                cache_key = self.get_cache_key(roi_id, metric_type)
                patterns = [
                    self.cache_dir / metric_type / f"{cache_key}.*",
                    self.cache_dir / "metadata" / f"{cache_key}*"
                ]
            elif roi_id:
                # Clear all metrics for specific ROI
                patterns = [
                    self.cache_dir / "*" / f"*{roi_id}*",
                    self.cache_dir / "metadata" / f"*{roi_id}*"
                ]
            elif metric_type:
                # Clear specific metric type for all ROIs
                patterns = [
                    self.cache_dir / metric_type / "*",
                    self.cache_dir / "metadata" / f"*{metric_type}*"
                ]
            else:
                # Clear entire cache
                patterns = [self.cache_dir / "*" / "*"]
            
            for pattern in patterns:
                for path in Path().glob(str(pattern)):
                    if path.is_file():
                        path.unlink()
                        removed_count += 1
            
            logger.info(f"Cleared {removed_count} cache files")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
        
        return removed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'total_files': 0,
            'total_size_mb': 0.0,
            'by_type': {},
            'oldest_file': None,
            'newest_file': None
        }
        
        try:
            oldest_time = float('inf')
            newest_time = 0
            
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    type_files = 0
                    type_size = 0
                    
                    for file_path in subdir.glob("*"):
                        if file_path.is_file():
                            file_stat = file_path.stat()
                            file_size = file_stat.st_size
                            file_time = file_stat.st_mtime
                            
                            type_files += 1
                            type_size += file_size
                            
                            if file_time < oldest_time:
                                oldest_time = file_time
                                stats['oldest_file'] = str(file_path)
                            
                            if file_time > newest_time:
                                newest_time = file_time
                                stats['newest_file'] = str(file_path)
                    
                    stats['by_type'][subdir.name] = {
                        'files': type_files,
                        'size_mb': type_size / (1024 * 1024)
                    }
                    
                    stats['total_files'] += type_files
                    stats['total_size_mb'] += type_size / (1024 * 1024)
            
        except Exception as e:
            logger.error(f"Error computing cache stats: {e}")
        
        return stats