"""
Efficient Data Storage

Replaces JSON storage with HDF5/Parquet for scalability.
Addresses Gemini's critique about JSON bottlenecks for large datasets.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict
import warnings
from datetime import datetime
import gzip
import re

# Security function for ROI ID sanitization
def _sanitize_roi_id(roi_id: str, max_length: int = 100) -> str:
    """
    Sanitize ROI ID to prevent path traversal and injection attacks.
    
    Args:
        roi_id: Raw ROI identifier
        max_length: Maximum allowed length
        
    Returns:
        Sanitized ROI ID safe for use in filenames and HDF5 group names
        
    Raises:
        ValueError: If ROI ID cannot be safely sanitized
    """
    if not roi_id or not isinstance(roi_id, str):
        raise ValueError("ROI ID must be a non-empty string")
    
    # Remove any path separators and dangerous characters
    # Allow only alphanumeric, underscore, hyphen, and dot
    sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', roi_id)
    
    # Remove leading/trailing dots and underscores to prevent hidden files
    sanitized = sanitized.strip('._')
    
    # Ensure it doesn't start with hyphen (can be problematic in some contexts)
    if sanitized.startswith('-'):
        sanitized = 'roi_' + sanitized
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Ensure we still have a valid identifier
    if not sanitized or sanitized == '.' or sanitized == '..':
        raise ValueError(f"ROI ID '{roi_id}' cannot be safely sanitized")
    
    # Final validation - must not contain any remaining dangerous patterns
    dangerous_patterns = ['..', '//', '\\\\', '.\\', './']
    if any(pattern in sanitized for pattern in dangerous_patterns):
        raise ValueError(f"ROI ID '{roi_id}' contains dangerous patterns after sanitization")
    
    return sanitized


# Optional dependencies with graceful fallbacks
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    # Create stub for type annotations
    class h5py:
        class Group:
            pass
    warnings.warn("h5py not available. HDF5 storage will be disabled. Install with: pip install h5py")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Create stub for type annotations
    class pd:
        DataFrame = type('DataFrame', (), {})
    warnings.warn("pandas not available. Parquet storage will be disabled. Install with: pip install pandas")


class HDF5Storage:
    """
    HDF5-based storage for IMC analysis results.
    Provides efficient storage, compression, and querying for large datasets.
    """
    
    def __init__(self, storage_path: Union[str, Path], compression: str = 'gzip'):
        if not HDF5_AVAILABLE:
            raise ImportError("HDF5Storage requires h5py. Install with: pip install h5py")
        
        self.storage_path = Path(storage_path)
        self.compression = compression
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    def save_analysis_results(
        self, 
        results: Dict[str, Any], 
        roi_id: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save analysis results for single ROI to HDF5.
        
        Args:
            results: Analysis results dictionary
            roi_id: ROI identifier
            metadata: Optional metadata dictionary
        """
        with h5py.File(self.storage_path, 'a') as f:
            # Sanitize ROI ID for security
            safe_roi_id = _sanitize_roi_id(roi_id)
            
            # Create group for this ROI
            roi_group = f.create_group(safe_roi_id) if safe_roi_id not in f else f[safe_roi_id]
            
            # Store original ROI ID as metadata if different
            if safe_roi_id != roi_id:
                roi_group.attrs['original_roi_id'] = roi_id
            
            # Store timestamp
            roi_group.attrs['timestamp'] = datetime.now().isoformat()
            
            # Store metadata if provided
            if metadata:
                roi_group.attrs['metadata'] = json.dumps(metadata)
            
            # Store different types of data
            self._store_arrays(roi_group, results)
            self._store_scalars(roi_group, results)
            self._store_dictionaries(roi_group, results)
    
    def _store_arrays(self, group: h5py.Group, results: Dict[str, Any]) -> None:
        """Store numpy arrays with compression."""
        array_keys = [
            'feature_matrix', 'cluster_labels', 'cluster_map', 
            'bin_edges_x', 'bin_edges_y', 'valid_indices'
        ]
        
        for key in array_keys:
            if key in results and isinstance(results[key], np.ndarray):
                if results[key].size > 0:
                    group.create_dataset(
                        key, 
                        data=results[key],
                        compression=self.compression,
                        compression_opts=9 if self.compression == 'gzip' else None
                    )
    
    def _store_scalars(self, group: h5py.Group, results: Dict[str, Any]) -> None:
        """Store scalar values as attributes."""
        scalar_keys = ['bin_size_um', 'processing_method']
        
        for key in scalar_keys:
            if key in results:
                group.attrs[key] = results[key]
    
    def _store_dictionaries(self, group: h5py.Group, results: Dict[str, Any]) -> None:
        """Store dictionary data as JSON strings or separate groups."""
        dict_keys = [
            'aggregated_counts', 'transformed_arrays', 'standardized_arrays',
            'cofactors_used', 'cluster_centroids', 'optimization_results',
            'memory_report', 'configuration_used'
        ]
        
        for key in dict_keys:
            if key in results and results[key]:
                if key in ['aggregated_counts', 'transformed_arrays', 'standardized_arrays']:
                    # Store protein arrays as separate datasets
                    array_group = group.create_group(key)
                    for protein_name, array in results[key].items():
                        if isinstance(array, np.ndarray) and array.size > 0:
                            array_group.create_dataset(
                                protein_name,
                                data=array,
                                compression=self.compression,
                                compression_opts=9 if self.compression == 'gzip' else None
                            )
                else:
                    # Store as JSON string attribute
                    try:
                        json_str = json.dumps(results[key], default=self._json_serializer)
                        group.attrs[key] = json_str
                    except (TypeError, ValueError) as e:
                        warnings.warn(f"Could not serialize {key}: {e}")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return str(obj)
    
    def load_roi_results(self, roi_id: str) -> Dict[str, Any]:
        """Load analysis results for single ROI."""
        if not self.storage_path.exists():
            raise FileNotFoundError(f"Storage file not found: {self.storage_path}")
        
        results = {}
        
        with h5py.File(self.storage_path, 'r') as f:
            # Try sanitized ROI ID first
            safe_roi_id = _sanitize_roi_id(roi_id)
            
            if safe_roi_id in f:
                roi_group = f[safe_roi_id]
            elif roi_id in f:
                # Fallback to original for backward compatibility
                roi_group = f[roi_id]
            else:
                raise KeyError(f"ROI '{roi_id}' (sanitized: '{safe_roi_id}') not found in storage")
            
            # Load arrays
            for key in roi_group.keys():
                if isinstance(roi_group[key], h5py.Dataset):
                    results[key] = roi_group[key][:]
                elif isinstance(roi_group[key], h5py.Group):
                    # Load dictionary of arrays
                    results[key] = {}
                    for subkey in roi_group[key].keys():
                        results[key][subkey] = roi_group[key][subkey][:]
            
            # Load attributes
            for key, value in roi_group.attrs.items():
                if key in ['metadata', 'cofactors_used', 'cluster_centroids', 
                          'optimization_results', 'memory_report', 'configuration_used']:
                    try:
                        results[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        results[key] = value
                else:
                    results[key] = value
        
        return results
    
    def list_rois(self) -> List[str]:
        """List all ROI IDs in storage."""
        if not self.storage_path.exists():
            return []
        
        with h5py.File(self.storage_path, 'r') as f:
            return list(f.keys())
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage file."""
        if not self.storage_path.exists():
            return {'exists': False}
        
        file_size = self.storage_path.stat().st_size
        
        with h5py.File(self.storage_path, 'r') as f:
            n_rois = len(f.keys())
            
            # Estimate uncompressed size
            total_uncompressed = 0
            for roi_id in f.keys():
                for item in f[roi_id].values():
                    if isinstance(item, h5py.Dataset):
                        total_uncompressed += item.size * item.dtype.itemsize
        
        compression_ratio = total_uncompressed / file_size if file_size > 0 else 0
        
        return {
            'exists': True,
            'file_size_mb': file_size / (1024**2),
            'n_rois': n_rois,
            'compression_ratio': compression_ratio
        }


class ParquetStorage:
    """
    Parquet-based storage for tabular IMC analysis data.
    Uses partitioning by roi_id to avoid O(N²) rewrite behavior.
    """
    
    def __init__(self, storage_dir: Union[str, Path]):
        if not PANDAS_AVAILABLE:
            raise ImportError("ParquetStorage requires pandas. Install with: pip install pandas pyarrow")
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Use partitioned directories for O(1) updates
        self.features_dir = self.storage_dir / "roi_features_partitioned"
        self.metadata_path = self.storage_dir / "roi_metadata.parquet"
        self.clustering_dir = self.storage_dir / "clustering_results_partitioned"
        
        self.features_dir.mkdir(exist_ok=True)
        self.clustering_dir.mkdir(exist_ok=True)
    
    def save_roi_features(
        self, 
        roi_id: str, 
        feature_matrix: np.ndarray, 
        protein_names: List[str],
        metadata: Optional[Dict] = None
    ) -> None:
        """Save ROI feature matrix as partitioned parquet file (O(1) operation)."""
        if feature_matrix.size == 0:
            return
        
        # Create feature DataFrame
        feature_df = pd.DataFrame(
            feature_matrix, 
            columns=protein_names
        )
        feature_df['roi_id'] = roi_id
        feature_df['spatial_bin_id'] = range(len(feature_df))
        
        # Add metadata columns if provided
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    feature_df[f'meta_{key}'] = value
        
        # Sanitize ROI ID for security
        safe_roi_id = _sanitize_roi_id(roi_id)
        
        # Save to partitioned file - O(1) operation!
        roi_file = self.features_dir / f"roi_{safe_roi_id}.parquet"
        feature_df.to_parquet(roi_file, compression='snappy')
    
    def save_roi_metadata(self, roi_metadata_list: List[Dict]) -> None:
        """Save ROI-level metadata."""
        metadata_df = pd.DataFrame(roi_metadata_list)
        metadata_df.to_parquet(self.metadata_path, compression='snappy')
    
    def save_clustering_results(self, roi_id: str, clustering_results: Dict) -> None:
        """Save clustering results as partitioned parquet (O(1) operation)."""
        if 'cluster_centroids' not in clustering_results:
            return
        
        centroids_data = []
        for cluster_id, centroid in clustering_results['cluster_centroids'].items():
            for protein, value in centroid.items():
                centroids_data.append({
                    'roi_id': roi_id,
                    'cluster_id': cluster_id,
                    'protein': protein,
                    'centroid_value': value
                })
        
        if not centroids_data:
            return
        
        centroids_df = pd.DataFrame(centroids_data)
        
        # Add optimization metadata
        if 'optimization_results' in clustering_results:
            opt_results = clustering_results['optimization_results']
            centroids_df['optimal_k'] = opt_results.get('final_n_clusters', -1)
            centroids_df['optimization_method'] = opt_results.get('recommendation', 'unknown')
        
        # Sanitize ROI ID for security
        safe_roi_id = _sanitize_roi_id(roi_id)
        
        # Save to partitioned file - O(1) operation!
        roi_file = self.clustering_dir / f"roi_{safe_roi_id}.parquet"
        centroids_df.to_parquet(roi_file, compression='snappy')
    
    def load_features_for_analysis(
        self, 
        roi_ids: Optional[List[str]] = None,
        proteins: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load feature data from partitioned storage."""
        if not self.features_dir.exists():
            return pd.DataFrame()
        
        # Load only requested ROIs from partitioned files
        dfs = []
        if roi_ids:
            # Load specific ROIs - O(k) where k = # of requested ROIs
            for roi_id in roi_ids:
                roi_file = self.features_dir / f"roi_{roi_id}.parquet"
                if roi_file.exists():
                    dfs.append(pd.read_parquet(roi_file))
        else:
            # Load all ROIs
            for roi_file in self.features_dir.glob("roi_*.parquet"):
                dfs.append(pd.read_parquet(roi_file))
        
        if not dfs:
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Filter by proteins
        if proteins:
            # Get all columns that are either in proteins list or are metadata
            protein_cols = [col for col in df.columns if col in proteins]
            meta_cols = [col for col in df.columns if col.startswith('meta_') or col in ['roi_id', 'spatial_bin_id']]
            selected_cols = protein_cols + meta_cols
            df = df[selected_cols]
        
        return df
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get summary of stored data."""
        summary = {
            'features_dir_exists': self.features_dir.exists(),
            'metadata_file_exists': self.metadata_path.exists(),
            'clustering_dir_exists': self.clustering_dir.exists()
        }
        
        if self.features_dir.exists():
            roi_files = list(self.features_dir.glob("roi_*.parquet"))
            if roi_files:
                # Sample first file to get structure
                sample_df = pd.read_parquet(roi_files[0])
                total_size = sum(f.stat().st_size for f in roi_files)
                summary.update({
                    'n_rois': len(roi_files),
                    'n_proteins': len([col for col in sample_df.columns if not col.startswith('meta_') and col not in ['roi_id', 'spatial_bin_id']]),
                    'features_total_size_mb': total_size / (1024**2)
            })
        
        return summary


class HybridStorage:
    """
    Hybrid storage system combining HDF5 and Parquet.
    Uses HDF5 for complex nested data and arrays, Parquet for tabular analysis data.
    """
    
    def __init__(self, base_path: Union[str, Path], compression: bool = True):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage backends
        hdf5_path = self.base_path / "analysis_results.h5"
        parquet_dir = self.base_path / "tabular_data"
        
        self.hdf5_storage = HDF5Storage(hdf5_path, compression='gzip' if compression else None)
        self.parquet_storage = ParquetStorage(parquet_dir)
    
    def save_roi_analysis(self, roi_id: str, results: Dict[str, Any]) -> None:
        """Save ROI analysis using appropriate storage backend."""
        # Save complete results to HDF5
        self.hdf5_storage.save_analysis_results(results, roi_id)
        
        # Extract tabular data for Parquet storage
        if 'feature_matrix' in results and 'protein_names' in results:
            self.parquet_storage.save_roi_features(
                roi_id=roi_id,
                feature_matrix=results['feature_matrix'],
                protein_names=results['protein_names'],
                metadata={
                    'bin_size_um': results.get('bin_size_um', 20.0),
                    'processing_method': results.get('processing_method', 'unknown'),
                    'n_clusters': results.get('optimization_results', {}).get('final_n_clusters', -1)
                }
            )
        
        # Save clustering results
        if 'cluster_centroids' in results or 'optimization_results' in results:
            self.parquet_storage.save_clustering_results(roi_id, results)
    
    def load_roi_complete(self, roi_id: str) -> Dict[str, Any]:
        """Load complete ROI analysis from HDF5."""
        return self.hdf5_storage.load_roi_results(roi_id)
    
    def load_features_for_comparison(
        self, 
        roi_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load feature data optimized for comparative analysis."""
        return self.parquet_storage.load_features_for_analysis(roi_ids)
    
    def create_analysis_summary(self) -> Dict[str, Any]:
        """Create comprehensive analysis summary."""
        hdf5_info = self.hdf5_storage.get_storage_info()
        parquet_info = self.parquet_storage.get_storage_summary()
        
        return {
            'storage_type': 'hybrid',
            'base_path': str(self.base_path),
            'hdf5_storage': hdf5_info,
            'parquet_storage': parquet_info,
            'total_rois': max(
                hdf5_info.get('n_rois', 0),
                parquet_info.get('n_rois', 0)
            ),
            'created_at': datetime.now().isoformat()
        }
    
    def export_for_publication(self, export_path: Union[str, Path]) -> List[str]:
        """Export data in publication-ready formats."""
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        # Export feature matrix as CSV
        features_df = self.parquet_storage.load_features_for_analysis()
        if not features_df.empty:
            csv_path = export_path / "roi_features.csv"
            features_df.to_csv(csv_path, index=False)
            exported_files.append(str(csv_path))
        
        # Export clustering summary
        if self.parquet_storage.clustering_dir.exists():
            # Load all clustering files from the partitioned directory
            clustering_files = list(self.parquet_storage.clustering_dir.glob("roi_*.parquet"))
            if clustering_files:
                clustering_dfs = []
                for file in clustering_files:
                    df = pd.read_parquet(file)
                    clustering_dfs.append(df)
                clustering_df = pd.concat(clustering_dfs, ignore_index=True)
                
                # Create summary by ROI
                roi_summary = clustering_df.groupby('roi_id').agg({
                    'optimal_k': 'first',
                    'optimization_method': 'first'
                }).reset_index()
                
                summary_path = export_path / "clustering_summary.csv"
                roi_summary.to_csv(summary_path, index=False)
                exported_files.append(str(summary_path))
        
        # Export analysis metadata
        metadata = {
            'export_date': datetime.now().isoformat(),
            'storage_summary': self.create_analysis_summary(),
            'files_exported': [Path(f).name for f in exported_files]
        }
        
        metadata_path = export_path / "analysis_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        exported_files.append(str(metadata_path))
        
        return exported_files


class CompressedJSONStorage:
    """
    Fallback storage using compressed JSON when HDF5/Parquet unavailable.
    More efficient than plain JSON while maintaining compatibility.
    """
    
    def __init__(self, storage_dir: Union[str, Path], compression: bool = True):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.roi_dir = self.storage_dir / "roi_data"
        self.roi_dir.mkdir(exist_ok=True)
    
    def save_roi_analysis(self, roi_id: str, results: Dict[str, Any]) -> None:
        """Save ROI analysis using compressed JSON."""
        # Sanitize ROI ID for security
        safe_roi_id = _sanitize_roi_id(roi_id)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_for_json(results)
        
        # Add metadata
        json_results['_metadata'] = {
            'roi_id': roi_id,  # Keep original for reference
            'safe_roi_id': safe_roi_id,  # Include sanitized version
            'timestamp': datetime.now().isoformat(),
            'storage_format': 'compressed_json'
        }
        
        # Save to file with sanitized name
        filename = f"roi_{safe_roi_id}.json"
        if self.compression:
            filename += ".gz"
        
        filepath = self.roi_dir / filename
        
        if self.compression:
            with gzip.open(filepath, 'wt') as f:
                json.dump(json_results, f, indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=2)
    
    def load_roi_complete(self, roi_id: str) -> Dict[str, Any]:
        """Load complete ROI analysis from compressed JSON."""
        filename = f"roi_{roi_id}.json"
        
        # Try compressed version first
        filepath_gz = self.roi_dir / (filename + ".gz")
        filepath = self.roi_dir / filename
        
        if filepath_gz.exists():
            with gzip.open(filepath_gz, 'rt') as f:
                results = json.load(f)
        elif filepath.exists():
            with open(filepath, 'r') as f:
                results = json.load(f)
        else:
            raise FileNotFoundError(f"ROI '{roi_id}' not found in storage")
        
        # Convert lists back to numpy arrays
        return self._restore_from_json(results)
    
    def _prepare_for_json(self, obj):
        """Convert numpy arrays and other objects for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return {
                '__numpy_array__': True,
                'dtype': str(obj.dtype),
                'shape': obj.shape,
                'data': obj.tolist()
            }
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__class__') and 'sklearn' in str(obj.__class__):
            # Handle sklearn objects by converting to description
            return {
                '__sklearn_object__': True,
                'class_name': obj.__class__.__name__,
                'module': obj.__class__.__module__,
                'description': str(obj)
            }
        elif hasattr(obj, 'tolist'):
            # Handle other array-like objects
            return self._prepare_for_json(obj.tolist())
        else:
            try:
                # Try direct JSON serialization
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                # Fallback to string representation
                return {
                    '__string_repr__': True,
                    'type': str(type(obj)),
                    'value': str(obj)
                }
    
    def _restore_from_json(self, obj):
        """Convert JSON back to numpy arrays and other objects."""
        if isinstance(obj, dict):
            if '__numpy_array__' in obj:
                return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
            elif '__sklearn_object__' in obj:
                # Return a description dict for sklearn objects (can't fully restore)
                return {
                    'sklearn_class': obj['class_name'],
                    'module': obj['module'],
                    'description': obj['description']
                }
            elif '__string_repr__' in obj:
                # Return description for non-serializable objects
                return {
                    'object_type': obj['type'],
                    'string_value': obj['value']
                }
            else:
                return {k: self._restore_from_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._restore_from_json(item) for item in obj]
        else:
            return obj
    
    def create_analysis_summary(self) -> Dict[str, Any]:
        """Create analysis summary from JSON files."""
        roi_files = list(self.roi_dir.glob("roi_*.json*"))
        
        total_size = sum(f.stat().st_size for f in roi_files)
        n_rois = len(roi_files)
        
        return {
            'storage_type': 'compressed_json',
            'base_path': str(self.storage_dir),
            'n_rois': n_rois,
            'total_size_mb': total_size / (1024**2),
            'compression_enabled': self.compression,
            'created_at': datetime.now().isoformat()
        }
    
    def save_batch_analysis(self, batch_results: Dict[str, Any]) -> None:
        """Save multiple ROI analyses in batch."""
        for roi_id, results in batch_results.items():
            self.save_roi_analysis(roi_id, results)
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get summary of all stored analyses."""
        return self.create_analysis_summary()
    
    def export_for_publication(self, export_path: Union[str, Path]) -> List[str]:
        """Export data in publication-ready formats."""
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Create summary CSV of all ROI results
        roi_files = list(self.roi_dir.glob("roi_*.json*"))
        
        if not roi_files:
            return []
        
        # Extract key metrics from each ROI
        roi_summaries = []
        
        for roi_file in roi_files:
            roi_id = roi_file.stem.replace('roi_', '').replace('.json', '')
            
            try:
                results = self.load_roi_complete(roi_id)
                
                summary = {
                    'roi_id': roi_id,
                    'n_clusters': results.get('optimization_results', {}).get('final_n_clusters', -1),
                    'bin_size_um': results.get('bin_size_um', 20.0),
                    'processing_method': results.get('processing_method', 'unknown'),
                    'n_spatial_bins': results.get('feature_matrix', np.array([])).shape[0] if 'feature_matrix' in results else 0,
                    'n_proteins': len(results.get('protein_names', []))
                }
                
                # Add cofactors used
                cofactors = results.get('cofactors_used', {})
                for protein, cofactor in cofactors.items():
                    summary[f'cofactor_{protein}'] = cofactor
                
                roi_summaries.append(summary)
                
            except Exception as e:
                warnings.warn(f"Could not process ROI {roi_id}: {e}")
        
        # Save summary CSV
        exported_files = []
        if roi_summaries:
            import csv
            
            csv_path = export_path / "roi_analysis_summary.csv"
            
            with open(csv_path, 'w', newline='') as csvfile:
                if roi_summaries:
                    fieldnames = roi_summaries[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(roi_summaries)
            
            exported_files.append(str(csv_path))
        
        return exported_files


def create_storage_backend(
    storage_config: Dict[str, Any],
    base_path: Union[str, Path]
) -> Union[HDF5Storage, ParquetStorage, HybridStorage, CompressedJSONStorage]:
    """
    Factory function to create appropriate storage backend.
    
    Args:
        storage_config: Storage configuration dictionary
        base_path: Base path for storage
        
    Returns:
        Appropriate storage backend instance
    """
    storage_format = storage_config.get('format', 'hybrid').lower()
    compression = storage_config.get('compression', True)
    
    try:
        if storage_format == 'hdf5' and HDF5_AVAILABLE:
            return HDF5Storage(Path(base_path) / "analysis_results.h5", 
                              compression='gzip' if compression else None)
        elif storage_format == 'parquet' and PANDAS_AVAILABLE:
            return ParquetStorage(base_path)
        elif storage_format == 'hybrid' and HDF5_AVAILABLE and PANDAS_AVAILABLE:
            return HybridStorage(base_path, compression)
        else:
            # Fallback to compressed JSON
            warnings.warn(
                f"Requested storage format '{storage_format}' not available. "
                f"HDF5 available: {HDF5_AVAILABLE}, Pandas available: {PANDAS_AVAILABLE}. "
                "Falling back to compressed JSON storage."
            )
            return CompressedJSONStorage(base_path, compression)
    
    except ImportError as e:
        warnings.warn(f"Storage backend creation failed: {e}. Using compressed JSON fallback.")
        return CompressedJSONStorage(base_path, compression)