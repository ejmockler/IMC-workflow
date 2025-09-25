"""
Tests for parallel processing and memory-aware scheduling

Verifies file-based processing, memory management, and worker scheduling.
"""

import numpy as np
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import multiprocessing as mp

from src.analysis.parallel_processing import (
    MemoryAwareScheduler,
    get_optimal_process_count,
    parallel_roi_analysis_files,
    process_single_roi_file,
    parallel_roi_analysis,
    ProgressTracker
)


# Module-level function for pickling in tests
def simple_analysis_function(roi_data):
    """Simple analysis function for testing."""
    return roi_data


class TestMemoryAwareScheduler:
    """Test dynamic memory-aware worker scheduling."""
    
    def test_scheduler_initialization(self):
        """Test scheduler setup."""
        scheduler = MemoryAwareScheduler(
            target_memory_percent=70.0,
            min_workers=1,
            max_workers=8
        )
        
        assert scheduler.target_memory_percent == 70.0
        assert scheduler.min_workers == 1
        assert scheduler.max_workers == 8
        assert scheduler.current_workers == 1
    
    @patch('psutil.virtual_memory')
    def test_memory_usage_monitoring(self, mock_vmem):
        """Test memory usage monitoring."""
        # Mock memory at 50%
        mock_vmem.return_value = Mock(percent=50.0)
        
        scheduler = MemoryAwareScheduler()
        usage = scheduler.get_current_memory_usage()
        
        assert usage == 50.0
    
    @patch('psutil.virtual_memory')
    def test_worker_adjustment_high_memory(self, mock_vmem):
        """Test worker reduction when memory is high."""
        # Start with 4 workers
        scheduler = MemoryAwareScheduler()
        scheduler.current_workers = 4
        
        # Simulate high memory usage
        mock_vmem.return_value = Mock(percent=85.0)
        
        # Should reduce workers
        new_count = scheduler.adjust_worker_count()
        assert new_count == 3
    
    @patch('psutil.virtual_memory')
    @patch('multiprocessing.cpu_count')
    def test_worker_adjustment_low_memory(self, mock_cpu, mock_vmem):
        """Test worker increase when memory is available."""
        mock_cpu.return_value = 8
        
        scheduler = MemoryAwareScheduler()
        scheduler.current_workers = 2
        
        # Simulate low memory usage (plenty of headroom)
        mock_vmem.return_value = Mock(percent=30.0)
        
        # Should increase workers
        new_count = scheduler.adjust_worker_count()
        assert new_count == 3
    
    def test_roi_memory_estimation(self):
        """Test memory estimation for ROI files."""
        scheduler = MemoryAwareScheduler()
        
        with tempfile.NamedTemporaryFile() as tmp:
            # Write 1MB of data
            tmp.write(b'x' * 1024 * 1024)
            tmp.flush()
            
            # Should estimate ~10MB in memory (10x heuristic)
            estimate = scheduler.estimate_roi_memory(tmp.name)
            assert 0.009 < estimate < 0.011  # ~0.01 GB
    
    @patch('time.sleep')
    @patch('psutil.virtual_memory')
    def test_wait_for_memory(self, mock_vmem, mock_sleep):
        """Test waiting for memory to become available."""
        scheduler = MemoryAwareScheduler()
        
        # Simulate memory above threshold initially, then dropping
        mock_vmem.return_value = Mock(percent=85.0)
        
        scheduler.wait_for_memory(threshold_percent=80.0, timeout=1)
        
        # Should have called sleep at least once
        assert mock_sleep.called


class TestOptimalProcessCount:
    """Test process count optimization."""
    
    @patch('psutil.virtual_memory')
    @patch('multiprocessing.cpu_count')
    def test_static_process_count(self, mock_cpu, mock_vmem):
        """Test static process count calculation."""
        mock_cpu.return_value = 8
        mock_vmem.return_value = Mock(available=8 * 1024**3)  # 8GB available
        
        count = get_optimal_process_count(max_memory_gb=4.0)
        
        # Should be limited by memory or CPU
        assert 1 <= count <= 8
    
    @patch('psutil.virtual_memory')
    def test_dynamic_process_count(self, mock_vmem):
        """Test dynamic process count with scheduler."""
        mock_vmem.return_value = Mock(percent=50.0)
        
        count = get_optimal_process_count(use_dynamic=True)
        
        assert count >= 1


class TestFileBasedProcessing:
    """Test file-based parallel processing."""
    
    def test_process_single_roi_file(self):
        """Test processing a single ROI file."""
        # Create mock analysis function
        def mock_analysis(file_path, roi_id=None, **kwargs):
            return f"processed_{roi_id}.json"
        
        # Test with mock file
        result_file, error = process_single_roi_file(
            ('roi1', '/path/to/roi1.txt'),
            mock_analysis,
            {}
        )
        
        assert result_file == "processed_roi1.json"
        assert error is None
    
    def test_process_single_roi_file_error(self):
        """Test error handling in single ROI processing."""
        # Create failing analysis function
        def failing_analysis(file_path, **kwargs):
            raise ValueError("Test error")
        
        result_file, error = process_single_roi_file(
            ('roi1', '/path/to/roi1.txt'),
            failing_analysis,
            {}
        )
        
        assert result_file is None
        assert "Test error" in error
    
    @patch('src.analysis.parallel_processing.Pool')
    def test_parallel_roi_analysis_files(self, mock_pool_class):
        """Test parallel file-based processing."""
        # Mock pool
        mock_pool = MagicMock()
        mock_pool_class.return_value.__enter__.return_value = mock_pool
        mock_pool.map.return_value = [
            ("result1.json", None),
            ("result2.json", None),
            (None, "Error in roi3")
        ]
        
        # Test files
        roi_files = [
            ('roi1', '/path/to/roi1.txt'),
            ('roi2', '/path/to/roi2.txt'),
            ('roi3', '/path/to/roi3.txt')
        ]
        
        # Mock analysis function
        def mock_analysis(path, **kwargs):
            return f"processed_{path}.json"
        
        result_files, errors = parallel_roi_analysis_files(
            roi_files,
            mock_analysis,
            n_processes=2,
            batch_size=3,  # Process all in one batch
            use_memory_scheduler=False
        )
        
        assert len(result_files) == 2
        assert len(errors) == 1
        assert "Error in roi3" in errors[0]
    
    @patch('src.analysis.parallel_processing.MemoryAwareScheduler')
    @patch('src.analysis.parallel_processing.Pool')
    def test_memory_aware_processing(self, mock_pool_class, mock_scheduler_class):
        """Test processing with memory-aware scheduling."""
        # Mock scheduler
        mock_scheduler = Mock()
        mock_scheduler.adjust_worker_count.return_value = 2
        mock_scheduler.wait_for_memory.return_value = None
        mock_scheduler_class.return_value = mock_scheduler
        
        # Mock pool
        mock_pool = MagicMock()
        mock_pool_class.return_value.__enter__.return_value = mock_pool
        mock_pool.map.return_value = [("result1.json", None)]
        
        roi_files = [('roi1', '/path/to/roi1.txt')]
        
        result_files, errors = parallel_roi_analysis_files(
            roi_files,
            lambda x, **k: x,
            use_memory_scheduler=True
        )
        
        # Scheduler should be used
        assert mock_scheduler.adjust_worker_count.called
        assert mock_scheduler.wait_for_memory.called


class TestDeprecatedInterface:
    """Test deprecated parallel_roi_analysis function."""
    
    def test_deprecation_warning(self):
        """Test that deprecated function warns."""
        with pytest.warns(DeprecationWarning, match="memory issues"):
            # Should warn about deprecation
            roi_data = {'roi1': {'data': [1, 2, 3]}}
            parallel_roi_analysis(
                roi_data,
                simple_analysis_function,
                n_processes=1
            )


class TestProgressTracking:
    """Test progress tracking functionality."""
    
    @patch('time.time')
    def test_progress_tracker(self, mock_time):
        """Test progress tracker updates."""
        mock_time.side_effect = [0, 5, 10]  # Start, first update, second update
        
        tracker = ProgressTracker(total_steps=100, update_interval=5.0)
        
        # First update
        with patch('builtins.print') as mock_print:
            tracker.update(0.5, "Halfway")
            mock_print.assert_called()
            
            # Check ETA calculation
            call_args = mock_print.call_args[0][0]
            assert "50.0%" in call_args
            assert "Halfway" in call_args
    
    def test_progress_callback_integration(self):
        """Test progress callback in parallel processing."""
        progress_updates = []
        
        def progress_callback(progress, message):
            progress_updates.append((progress, message))
        
        # Create simple test with mock pool
        with patch('src.analysis.parallel_processing.Pool') as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value.__enter__.return_value = mock_pool
            mock_pool.map.return_value = [("result.json", None)]
            
            roi_files = [('roi1', '/path/to/roi1.txt')]
            
            parallel_roi_analysis_files(
                roi_files,
                lambda x, **k: x,
                progress_callback=progress_callback,
                use_memory_scheduler=False
            )
            
            # Should have progress updates
            assert len(progress_updates) > 0
            assert progress_updates[0][0] == 1.0  # 100% after one batch


class TestEdgeCases:
    """Test edge cases in parallel processing."""
    
    def test_empty_roi_list(self):
        """Test handling of empty ROI list."""
        result_files, errors = parallel_roi_analysis_files(
            [],
            lambda x, **k: x,
            use_memory_scheduler=False
        )
        
        assert result_files == []
        assert errors == []
    
    @patch('src.analysis.parallel_processing.Pool')
    def test_single_roi_processing(self, mock_pool_class):
        """Test processing single ROI."""
        mock_pool = MagicMock()
        mock_pool_class.return_value.__enter__.return_value = mock_pool
        mock_pool.map.return_value = [("result.json", None)]
        
        roi_files = [('roi1', '/path/to/roi1.txt')]
        
        result_files, errors = parallel_roi_analysis_files(
            roi_files,
            lambda x, **k: "result.json",
            use_memory_scheduler=False
        )
        
        assert len(result_files) == 1
        assert len(errors) == 0
    
    def test_output_directory_creation(self):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            
            with patch('src.analysis.parallel_processing.Pool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value.__enter__.return_value = mock_pool
                mock_pool.map.return_value = []
                
                parallel_roi_analysis_files(
                    [],
                    lambda x, **k: x,
                    output_dir=str(output_dir),
                    use_memory_scheduler=False
                )
                
                assert output_dir.exists()


@pytest.mark.slow
class TestPerformanceAndScale:
    """Performance tests for parallel processing."""
    
    @patch('src.analysis.parallel_processing.Pool')
    def test_large_batch_processing(self, mock_pool_class):
        """Test processing large number of ROIs."""
        # Create 100 mock ROIs
        roi_files = [(f'roi{i}', f'/path/to/roi{i}.txt') for i in range(100)]
        
        # Mock pool to return results
        mock_pool = MagicMock()
        mock_pool_class.return_value.__enter__.return_value = mock_pool
        mock_pool.map.return_value = [(f"result{i}.json", None) for i in range(10)]
        
        result_files, errors = parallel_roi_analysis_files(
            roi_files,
            lambda x, **k: x,
            batch_size=10,
            n_processes=4,
            use_memory_scheduler=False
        )
        
        # Should process in batches
        assert mock_pool.map.call_count == 10  # 100 ROIs / 10 batch_size
    
    def test_memory_estimation_accuracy(self):
        """Test accuracy of memory estimation."""
        scheduler = MemoryAwareScheduler()
        
        # Test with various file sizes
        test_sizes = [1024, 1024*1024, 10*1024*1024]  # 1KB, 1MB, 10MB
        
        for size in test_sizes:
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(b'x' * size)
                tmp.flush()
                
                estimate_gb = scheduler.estimate_roi_memory(tmp.name)
                expected_gb = (size * 10) / (1024**3)  # 10x heuristic
                
                # Should be within 1% of expected
                assert abs(estimate_gb - expected_gb) / expected_gb < 0.01