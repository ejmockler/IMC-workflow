"""
Tests for data storage layer

Verifies HDF5/Parquet/JSON backends, partitioning, and pickle removal.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import json
from pathlib import Path
import pickle
from unittest.mock import patch, Mock

from src.analysis.data_storage import (
    create_storage_backend,
    HDF5Storage,
    ParquetStorage,
    HybridStorage,
    CompressedJSONStorage
)


class TestStorageFactory:
    """Test storage backend factory."""
    
    def test_create_hdf5_backend(self):
        """Test HDF5 backend creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'format': 'hdf5',
                'compression': True
            }
            
            backend = create_storage_backend(config, tmpdir)
            assert isinstance(backend, HDF5Storage)
    
    def test_create_parquet_backend(self):
        """Test Parquet backend creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'format': 'parquet'
            }
            
            backend = create_storage_backend(config, tmpdir)
            assert isinstance(backend, ParquetStorage)
    
    def test_create_json_backend(self):
        """Test JSON backend creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'format': 'json',
                'compression': True
            }
            
            backend = create_storage_backend(config, tmpdir)
            assert isinstance(backend, CompressedJSONStorage)
    
    def test_fallback_to_json_on_missing_deps(self):
        """Test fallback to JSON when optional deps missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'format': 'hdf5'
            }
            
            # Mock missing h5py
            with patch.dict('sys.modules', {'h5py': None}):
                backend = create_storage_backend(config, tmpdir)
                assert isinstance(backend, CompressedJSONStorage)
    
    def test_no_pickle_format(self):
        """Test that pickle format is not supported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'format': 'pickle'  # Should not be valid
            }
            
            # Should fall back to JSON
            backend = create_storage_backend(config, tmpdir)
            assert isinstance(backend, CompressedJSONStorage)


class TestParquetPartitioning:
    """Test Parquet partitioned storage (no O(N²) behavior)."""
    
    def test_partitioned_storage(self):
        """Test that Parquet uses partitioned files."""
        pd = pytest.importorskip('pandas')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ParquetStorage(tmpdir)
            
            # Save multiple ROIs
            for roi_id in ['roi1', 'roi2', 'roi3']:
                feature_matrix = np.random.rand(100, 5)
                protein_names = [f'protein_{i}' for i in range(5)]
                
                storage.save_roi_features(
                    roi_id, feature_matrix, protein_names
                )
            
            # Check that individual files were created (partitioned)
            features_dir = Path(tmpdir) / "roi_features_partitioned"
            assert features_dir.exists()
            
            roi_files = list(features_dir.glob("roi_*.parquet"))
            assert len(roi_files) == 3
            
            # Verify each file contains correct ROI
            for roi_file in roi_files:
                df = pd.read_parquet(roi_file)
                assert 'roi_id' in df.columns
                assert df['roi_id'].nunique() == 1
    
    def test_no_read_modify_write_cycle(self):
        """Test that adding new ROI doesn't read/rewrite existing ones."""
        pd = pytest.importorskip('pandas')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ParquetStorage(tmpdir)
            
            # Save first ROI
            storage.save_roi_features(
                'roi1', np.random.rand(100, 3), ['p1', 'p2', 'p3']
            )
            
            # Get initial file modification time
            roi1_file = Path(tmpdir) / "roi_features_partitioned" / "roi_roi1.parquet"
            initial_mtime = roi1_file.stat().st_mtime
            
            # Save second ROI
            storage.save_roi_features(
                'roi2', np.random.rand(100, 3), ['p1', 'p2', 'p3']
            )
            
            # First file should not have been modified
            assert roi1_file.stat().st_mtime == initial_mtime


class TestPickleRemoval:
    """Test that pickle support has been removed."""
    
    def test_no_pickle_save(self):
        """Test that we cannot save pickle files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try various storage backends
            for format_type in ['hdf5', 'parquet', 'json']:
                config = {'format': format_type}
                backend = create_storage_backend(config, tmpdir)
                
                # Save some data
                test_data = {'roi_id': 'test', 'data': np.random.rand(10)}
                
                if hasattr(backend, 'save_roi_analysis'):
                    backend.save_roi_analysis('test', test_data)
                
                # Verify no .pkl files were created
                pkl_files = list(Path(tmpdir).rglob("*.pkl"))
                assert len(pkl_files) == 0, f"Found pickle files: {pkl_files}"
    
    def test_pickle_load_fails(self):
        """Test that attempting to load pickle fails gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a pickle file (simulating old data)
            pkl_path = Path(tmpdir) / "old_data.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump({'data': 'test'}, f)
            
            # Storage backends should not be able to load it
            config = {'format': 'json'}
            backend = create_storage_backend(config, tmpdir)
            
            # Loading pickle should fail or return empty
            if hasattr(backend, 'load_roi_analysis'):
                # Should either raise error or return None/empty
                try:
                    result = backend.load_roi_analysis('old_data')
                    assert result is None or result == {}
                except (ValueError, TypeError, KeyError):
                    pass  # Expected to fail


class TestHDF5Storage:
    """Test HDF5 storage backend."""
    
    def test_save_and_load(self):
        """Test basic save and load operations."""
        h5py = pytest.importorskip('h5py')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = HDF5Storage(Path(tmpdir) / "test.h5")
            
            # Save ROI data
            roi_data = {
                'feature_matrix': np.random.rand(100, 10),
                'cluster_labels': np.random.randint(0, 5, 100),
                'metadata': {
                    'roi_id': 'test_roi',
                    'n_clusters': 5
                }
            }
            
            storage.save_roi_analysis('test_roi', roi_data)
            
            # Load back
            loaded = storage.load_roi_analysis('test_roi')
            
            assert 'feature_matrix' in loaded
            assert loaded['feature_matrix'].shape == (100, 10)
            assert 'cluster_labels' in loaded
            assert loaded['metadata']['n_clusters'] == 5
    
    def test_compression(self):
        """Test that compression reduces file size."""
        h5py = pytest.importorskip('h5py')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Large random data
            large_data = {
                'matrix': np.random.rand(1000, 100)
            }
            
            # Save with compression
            storage_compressed = HDF5Storage(
                Path(tmpdir) / "compressed.h5",
                compression='gzip'
            )
            storage_compressed.save_roi_analysis('test', large_data)
            
            # Save without compression
            storage_uncompressed = HDF5Storage(
                Path(tmpdir) / "uncompressed.h5",
                compression=None
            )
            storage_uncompressed.save_roi_analysis('test', large_data)
            
            # Compressed should be smaller
            compressed_size = (Path(tmpdir) / "compressed.h5").stat().st_size
            uncompressed_size = (Path(tmpdir) / "uncompressed.h5").stat().st_size
            
            assert compressed_size < uncompressed_size


class TestJSONStorage:
    """Test JSON storage backend."""
    
    def test_json_serialization(self):
        """Test JSON serialization of numpy arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CompressedJSONStorage(Path(tmpdir))
            
            # Data with numpy arrays
            roi_data = {
                'counts': np.array([1, 2, 3, 4, 5]),
                'matrix': np.random.rand(10, 5),
                'metadata': {
                    'roi_id': 'test',
                    'value': 42.5
                }
            }
            
            storage.save_roi_summary('test', roi_data)
            
            # Verify JSON file created
            json_files = list(Path(tmpdir).glob("*.json*"))
            assert len(json_files) > 0
            
            # Load and verify
            loaded = storage.load_roi_summary('test')
            assert 'counts' in loaded
            assert len(loaded['counts']) == 5
            assert loaded['metadata']['value'] == 42.5


class TestEdgeCases:
    """Test edge cases in storage."""
    
    def test_empty_data(self):
        """Test handling of empty data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for backend_class in [HDF5Storage, CompressedJSONStorage]:
                if backend_class == HDF5Storage:
                    pytest.importorskip('h5py')
                    storage = backend_class(Path(tmpdir) / "test.h5")
                else:
                    storage = backend_class(Path(tmpdir))
                
                # Save empty data
                empty_data = {
                    'matrix': np.array([]),
                    'labels': np.array([])
                }
                
                if hasattr(storage, 'save_roi_analysis'):
                    storage.save_roi_analysis('empty', empty_data)
                elif hasattr(storage, 'save_roi_summary'):
                    storage.save_roi_summary('empty', empty_data)
                
                # Should handle gracefully
                assert True  # If we get here, it didn't crash
    
    def test_overwrite_existing(self):
        """Test overwriting existing data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CompressedJSONStorage(Path(tmpdir))
            
            # Save initial data
            storage.save_roi_summary('test', {'value': 1})
            
            # Overwrite with new data
            storage.save_roi_summary('test', {'value': 2})
            
            # Should have new value
            loaded = storage.load_roi_summary('test')
            assert loaded['value'] == 2


@pytest.mark.slow
class TestPerformance:
    """Performance tests for storage backends."""
    
    def test_large_dataset(self):
        """Test handling of large datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'format': 'hdf5', 'compression': True}
            backend = create_storage_backend(config, tmpdir)
            
            # Large dataset (reduced for CI)
            large_data = {
                'feature_matrix': np.random.rand(1000, 50),
                'cluster_labels': np.random.randint(0, 10, 1000)
            }
            
            # Should complete in reasonable time
            import time
            start = time.time()
            
            if hasattr(backend, 'save_roi_analysis'):
                backend.save_roi_analysis('large', large_data)
                loaded = backend.load_roi_analysis('large')
            
            elapsed = time.time() - start
            assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s"