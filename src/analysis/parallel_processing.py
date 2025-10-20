"""
Parallel Processing Framework

Simple multiprocessing-based parallelization for ROI-level analysis.
Memory-efficient processing with batch management and error handling.
"""

import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
import json
import os
import traceback
import warnings
from functools import partial
import psutil


class MemoryAwareScheduler:
    """
    Dynamic memory-aware worker scheduling for parallel processing.
    Monitors system resources and adjusts worker count in real-time.
    """
    
    def __init__(self, 
                 target_memory_percent: float = 70.0,
                 min_workers: int = 1,
                 max_workers: int = 8):
        """
        Initialize memory-aware scheduler.
        
        Args:
            target_memory_percent: Target memory usage (0-100)
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
        """
        self.target_memory_percent = target_memory_percent
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self._memory_history = []
        
    def get_current_memory_usage(self) -> float:
        """Get current system memory usage percentage."""
        return psutil.virtual_memory().percent
    
    def estimate_roi_memory(self, roi_file_path: str = None) -> float:
        """Estimate memory requirement for processing an ROI file."""
        if roi_file_path and os.path.exists(roi_file_path):
            # Estimate based on file size (rough heuristic: 10x file size in memory)
            file_size_gb = os.path.getsize(roi_file_path) / (1024**3)
            return file_size_gb * 10
        else:
            # Default estimate: 500MB per ROI
            return 0.5
    
    def adjust_worker_count(self) -> int:
        """
        Dynamically adjust worker count based on memory usage.
        
        Returns:
            Updated number of workers to use
        """
        current_memory = self.get_current_memory_usage()
        self._memory_history.append(current_memory)
        
        # Keep only recent history (last 10 measurements)
        if len(self._memory_history) > 10:
            self._memory_history.pop(0)
        
        # Calculate trend
        avg_memory = sum(self._memory_history) / len(self._memory_history)
        
        if avg_memory > self.target_memory_percent:
            # Reduce workers if memory pressure is high
            self.current_workers = max(self.min_workers, self.current_workers - 1)
        elif avg_memory < self.target_memory_percent - 20:
            # Increase workers if we have headroom
            n_cpu = mp.cpu_count()
            self.current_workers = min(
                self.max_workers,
                n_cpu - 1,
                self.current_workers + 1
            )
        
        return self.current_workers
    
    def wait_for_memory(self, threshold_percent: float = 80.0, timeout: int = 60):
        """
        Wait for memory to become available.
        
        Args:
            threshold_percent: Wait until memory usage is below this
            timeout: Maximum seconds to wait
        """
        import time
        start_time = time.time()
        
        while self.get_current_memory_usage() > threshold_percent:
            if time.time() - start_time > timeout:
                warnings.warn(f"Memory threshold {threshold_percent}% not reached after {timeout}s")
                break
            time.sleep(1)


def get_optimal_process_count(max_memory_gb: float = 4.0, 
                            use_dynamic: bool = False) -> int:
    """
    Determine optimal number of processes based on system resources.
    
    Args:
        max_memory_gb: Maximum memory to use in GB
        use_dynamic: Use dynamic memory-aware scheduling
        
    Returns:
        Number of processes to use
    """
    if use_dynamic:
        scheduler = MemoryAwareScheduler(target_memory_percent=70.0)
        return scheduler.adjust_worker_count()
    
    # Static calculation (original implementation)
    n_cpu = mp.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Conservative estimates
    memory_per_process_gb = 0.5  # Assume 500MB per ROI analysis
    max_processes_by_memory = int(min(max_memory_gb, available_memory_gb) / memory_per_process_gb)
    
    # Use minimum of CPU count and memory constraint
    optimal_processes = min(n_cpu - 1, max_processes_by_memory, 8)  # Cap at 8 for stability
    
    return max(1, optimal_processes)


def process_single_roi(
    roi_data: Tuple[str, Dict],
    analysis_function: Callable,
    analysis_params: Dict
) -> Tuple[str, Dict, Optional[str]]:
    """
    Process a single ROI with error handling.

    Args:
        roi_data: Tuple of (roi_id, roi_data_dict)
        analysis_function: Function to apply to ROI
        analysis_params: Parameters for analysis function (passed as override_config)

    Returns:
        Tuple of (roi_id, results_dict, error_message)
    """
    roi_id, data = roi_data

    try:
        # Apply analysis function
        # Pass analysis_params as override_config to maintain compatibility
        # with analyze_single_roi(roi_data, override_config=None, plots_dir=None, roi_id=None)
        result = analysis_function(
            data,
            override_config=analysis_params,
            roi_id=roi_id
        )
        return roi_id, result, None

    except Exception as e:
        error_msg = f"ROI {roi_id} failed: {str(e)}\n{traceback.format_exc()}"
        return roi_id, {}, error_msg


def parallel_roi_analysis_files(
    roi_files: List[Tuple[str, str]],  # List of (roi_id, file_path)
    analysis_function: Callable,
    analysis_params: Dict = None,
    n_processes: Optional[int] = None,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    output_dir: Optional[str] = None,
    use_memory_scheduler: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Process multiple ROIs in parallel using file paths (memory-efficient).
    
    Args:
        roi_files: List of (roi_id, file_path) tuples
        analysis_function: Function that takes file_path and saves results
        analysis_params: Parameters for analysis function
        n_processes: Number of processes (auto-detected if None)
        batch_size: Batch size for processing (auto-calculated if None)
        progress_callback: Optional function to call with progress updates
        output_dir: Directory to save results
        
    Returns:
        Tuple of (result_files, error_messages)
    """
    if analysis_params is None:
        analysis_params = {}
    
    # Initialize memory scheduler if requested
    scheduler = None
    if use_memory_scheduler:
        scheduler = MemoryAwareScheduler(
            target_memory_percent=70.0,
            min_workers=1,
            max_workers=n_processes or 8
        )
        if n_processes is None:
            n_processes = scheduler.adjust_worker_count()
    elif n_processes is None:
        n_processes = get_optimal_process_count()
    
    n_rois = len(roi_files)
    if batch_size is None:
        batch_size = max(1, n_rois // n_processes)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        analysis_params['output_dir'] = output_dir
    
    result_files = []
    errors = []
    
    # Create partial function with fixed parameters
    process_func = partial(
        process_single_roi_file,
        analysis_function=analysis_function,
        analysis_params=analysis_params
    )
    
    # Use single persistent pool for all processing to avoid thrashing
    with Pool(processes=n_processes) as pool:
        # Process in batches to manage memory, but use streaming with chunksize
        n_batches = (n_rois + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            # Dynamically adjust memory monitoring if using scheduler
            if scheduler:
                scheduler.wait_for_memory(threshold_percent=80.0)
                # Note: Can't change pool size mid-execution, but can throttle
            
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_rois)
            batch_items = roi_files[start_idx:end_idx]
            
            # Process batch with optimal chunksize for better load balancing
            chunksize = max(1, len(batch_items) // n_processes)
            batch_results = pool.map(process_func, batch_items, chunksize=chunksize)
            
            # Collect results
            for result_file, error in batch_results:
                if error is None:
                    result_files.append(result_file)
                else:
                    errors.append(error)
            
            # Progress callback
            if progress_callback:
                progress = (batch_idx + 1) / n_batches
                progress_callback(progress, f"Processed batch {batch_idx + 1}/{n_batches}")
    
    return result_files, errors


def process_single_roi_file(
    roi_file_info: Tuple[str, str],
    analysis_function: Callable,
    analysis_params: Dict
) -> Tuple[Optional[str], Optional[str]]:
    """
    Process a single ROI file with error handling.
    
    Args:
        roi_file_info: Tuple of (roi_id, file_path)
        analysis_function: Function to apply to ROI file
        analysis_params: Parameters for analysis function
        
    Returns:
        Tuple of (result_file_path, error_message)
    """
    roi_id, file_path = roi_file_info
    
    try:
        # Analysis function should load data, process, and save results
        result_file = analysis_function(file_path, roi_id=roi_id, **analysis_params)
        return result_file, None
        
    except Exception as e:
        error_msg = f"ROI {roi_id} failed: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg


def parallel_roi_analysis(
    roi_data_dict: Dict[str, Dict],
    analysis_function: Callable,
    analysis_params: Dict = None,
    n_processes: Optional[int] = None,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[Dict[str, Dict], List[str]]:
    """
    [DEPRECATED - Use parallel_roi_analysis_files instead for memory efficiency]
    Process multiple ROIs in parallel.
    
    Args:
        roi_data_dict: Dictionary mapping roi_id -> data
        analysis_function: Function to apply to each ROI
        analysis_params: Parameters for analysis function
        n_processes: Number of processes (auto-detected if None)
        batch_size: Batch size for processing (auto-calculated if None)
        progress_callback: Optional function to call with progress updates
        
    Returns:
        Tuple of (results_dict, error_messages)
    """
    warnings.warn(
        "parallel_roi_analysis is deprecated due to memory issues. "
        "Use parallel_roi_analysis_files instead which passes file paths.",
        DeprecationWarning,
        stacklevel=2
    )
    if analysis_params is None:
        analysis_params = {}
    
    if n_processes is None:
        n_processes = get_optimal_process_count()
    
    n_rois = len(roi_data_dict)
    if batch_size is None:
        batch_size = max(1, n_rois // n_processes)
    
    # Prepare data for processing
    roi_items = list(roi_data_dict.items())
    results = {}
    errors = []
    
    # Create partial function with fixed parameters
    process_func = partial(
        process_single_roi,
        analysis_function=analysis_function,
        analysis_params=analysis_params
    )
    
    # Use single persistent pool to avoid thrashing (even in deprecated function)
    with Pool(processes=n_processes) as pool:
        # Process in batches to manage memory
        n_batches = (n_rois + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_rois)
            batch_items = roi_items[start_idx:end_idx]
            
            # Process batch with optimal chunksize
            chunksize = max(1, len(batch_items) // n_processes)
            batch_results = pool.map(process_func, batch_items, chunksize=chunksize)
            
            # Collect results
            for roi_id, result, error in batch_results:
                if error is None:
                    results[roi_id] = result
                else:
                    errors.append(error)
            
            # Progress callback
            if progress_callback:
                progress = (batch_idx + 1) / n_batches
                progress_callback(progress, f"Processed batch {batch_idx + 1}/{n_batches}")
    
    return results, errors


def save_parallel_results(
    results: Dict[str, Dict],
    output_dir: str,
    format: str = 'json',
    compress: bool = True
) -> List[str]:
    """
    Save parallel processing results to disk.
    
    Args:
        results: Results dictionary from parallel processing
        output_dir: Output directory path
        format: Output format ('json', 'csv')
        compress: Whether to compress output files
        
    Returns:
        List of created file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    created_files = []
    
    if format == 'json':
        # Save as JSON (separate file per ROI for memory efficiency)
        for roi_id, roi_results in results.items():
            filename = f"roi_{roi_id}_results.json"
            if compress:
                filename += '.gz'
            
            filepath = os.path.join(output_dir, filename)
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = serialize_for_json(roi_results)
            
            if compress:
                import gzip
                with gzip.open(filepath, 'wt') as f:
                    json.dump(json_results, f, indent=2, default=str)
            else:
                with open(filepath, 'w') as f:
                    json.dump(json_results, f, indent=2, default=str)
            
            created_files.append(filepath)
    
    
    elif format == 'csv':
        # Flatten results to CSV (for tabular data only)
        flattened_data = []
        
        for roi_id, roi_results in results.items():
            flat_row = {'roi_id': roi_id}
            flat_row.update(flatten_dict(roi_results))
            flattened_data.append(flat_row)
        
        if flattened_data:
            df = pd.DataFrame(flattened_data)
            filename = "parallel_results.csv"
            if compress:
                filename += '.gz'
            
            filepath = os.path.join(output_dir, filename)
            
            if compress:
                df.to_csv(filepath, index=False, compression='gzip')
            else:
                df.to_csv(filepath, index=False)
            
            created_files.append(filepath)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return created_files


def load_parallel_results(
    input_path: str,
    format: str = None
) -> Dict[str, Dict]:
    """
    Load parallel processing results from disk.
    
    Args:
        input_path: Path to results file or directory
        format: Format to expect ('json', 'csv', or None for auto-detect)
        
    Returns:
        Results dictionary
    """
    if format is None:
        # Auto-detect format
        if input_path.endswith('.json') or input_path.endswith('.json.gz'):
            format = 'json'
        elif input_path.endswith('.csv') or input_path.endswith('.csv.gz'):
            format = 'csv'
        elif os.path.isdir(input_path):
            format = 'json'  # Assume directory contains JSON files
        else:
            raise ValueError("Cannot auto-detect format. Please specify format explicitly.")
    
    if format == 'json':
        if os.path.isdir(input_path):
            # Load multiple JSON files
            results = {}
            for filename in os.listdir(input_path):
                if filename.startswith('roi_') and (filename.endswith('.json') or filename.endswith('.json.gz')):
                    filepath = os.path.join(input_path, filename)
                    roi_id = filename.replace('roi_', '').replace('_results.json', '').replace('.gz', '')
                    
                    if filename.endswith('.gz'):
                        import gzip
                        with gzip.open(filepath, 'rt') as f:
                            roi_results = json.load(f)
                    else:
                        with open(filepath, 'r') as f:
                            roi_results = json.load(f)
                    
                    results[roi_id] = deserialize_from_json(roi_results)
        else:
            # Single JSON file
            if input_path.endswith('.gz'):
                import gzip
                with gzip.open(input_path, 'rt') as f:
                    results = json.load(f)
            else:
                with open(input_path, 'r') as f:
                    results = json.load(f)
            
            results = deserialize_from_json(results)
    
    
    elif format == 'csv':
        if input_path.endswith('.gz'):
            df = pd.read_csv(input_path, compression='gzip')
        else:
            df = pd.read_csv(input_path)
        
        # Convert back to nested dictionary format
        results = {}
        for _, row in df.iterrows():
            roi_id = row['roi_id']
            row_dict = row.drop('roi_id').to_dict()
            results[roi_id] = unflatten_dict(row_dict)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return results


def serialize_for_json(obj):
    """
    Convert numpy arrays and other non-JSON types for serialization.
    Handles all NumPy scalar types robustly, including dictionary keys.
    """
    if isinstance(obj, dict):
        # CRITICAL: Convert both keys AND values
        # Dictionary keys can also be NumPy types (e.g., int64 indices)
        return {
            serialize_for_json(k): serialize_for_json(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return {
            '__numpy_array__': True,
            'dtype': str(obj.dtype),
            'shape': obj.shape,
            'data': obj.flatten().tolist()
        }
    elif isinstance(obj, np.generic):
        # Handles all NumPy scalar types (int64, float64, bool_, etc.)
        return obj.item()  # Convert to native Python type
    elif hasattr(obj, 'tolist'):
        # Fallback for any NumPy-like object with tolist()
        return obj.tolist()
    else:
        return obj


def deserialize_from_json(obj):
    """
    Convert serialized numpy arrays back to numpy arrays.
    """
    if isinstance(obj, dict):
        if '__numpy_array__' in obj:
            # Reconstruct numpy array
            data = np.array(obj['data'], dtype=obj['dtype'])
            return data.reshape(obj['shape'])
        else:
            return {k: deserialize_from_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deserialize_from_json(item) for item in obj]
    else:
        return obj


def flatten_dict(d: Dict, prefix: str = '') -> Dict:
    """
    Flatten nested dictionary for CSV export.
    """
    flattened = {}
    
    for k, v in d.items():
        new_key = f"{prefix}{k}" if prefix else k
        
        if isinstance(v, dict):
            flattened.update(flatten_dict(v, f"{new_key}_"))
        elif isinstance(v, (list, tuple, np.ndarray)):
            # Convert arrays/lists to summary statistics
            try:
                arr = np.array(v)
                if arr.dtype.kind in ['i', 'f']:  # Numeric data
                    flattened[f"{new_key}_mean"] = float(np.mean(arr))
                    flattened[f"{new_key}_std"] = float(np.std(arr))
                    flattened[f"{new_key}_min"] = float(np.min(arr))
                    flattened[f"{new_key}_max"] = float(np.max(arr))
                    flattened[f"{new_key}_size"] = int(arr.size)
                else:
                    flattened[f"{new_key}_size"] = len(v)
            except:
                flattened[f"{new_key}_size"] = len(v) if hasattr(v, '__len__') else 1
        else:
            flattened[new_key] = v
    
    return flattened


def unflatten_dict(d: Dict) -> Dict:
    """
    Unflatten dictionary (partial reconstruction).
    """
    # This is a simplified version - full reconstruction is complex
    # For now, just return the flattened dict
    return d


class ProgressTracker:
    """
    Simple progress tracker for parallel processing.
    """
    
    def __init__(self, total_steps: int, update_interval: float = 5.0):
        self.total_steps = total_steps
        self.current_step = 0
        self.update_interval = update_interval
        self.last_update = 0
        import time
        self.start_time = time.time()
    
    def update(self, progress: float, message: str = ""):
        import time
        current_time = time.time()
        
        if current_time - self.last_update >= self.update_interval or progress >= 1.0:
            elapsed_time = current_time - self.start_time
            
            if progress > 0:
                eta = elapsed_time * (1.0 - progress) / progress
                eta_str = f" (ETA: {eta:.1f}s)"
            else:
                eta_str = ""
            
            print(f"Progress: {progress*100:.1f}% {message} - Elapsed: {elapsed_time:.1f}s{eta_str}")
            self.last_update = current_time


def create_roi_batch_processor(
    analysis_function: Callable,
    n_processes: Optional[int] = None,
    batch_size: Optional[int] = None,
    output_dir: Optional[str] = None,
    save_format: str = 'json'
) -> Callable:
    """
    Create a configured batch processor for ROI analysis.
    
    Args:
        analysis_function: Function to apply to each ROI
        n_processes: Number of parallel processes
        batch_size: Batch size for processing
        output_dir: Directory to save results (optional)
        save_format: Format for saving results
        
    Returns:
        Configured batch processor function
    """
    
    def batch_processor(
        roi_data_dict: Dict[str, Dict],
        analysis_params: Dict = None,
        show_progress: bool = True
    ) -> Tuple[Dict[str, Dict], List[str]]:
        
        # Set up progress tracking
        progress_tracker = None
        if show_progress:
            progress_tracker = ProgressTracker(len(roi_data_dict))
            progress_callback = progress_tracker.update
        else:
            progress_callback = None
        
        # Run parallel analysis
        results, errors = parallel_roi_analysis(
            roi_data_dict=roi_data_dict,
            analysis_function=analysis_function,
            analysis_params=analysis_params or {},
            n_processes=n_processes,
            batch_size=batch_size,
            progress_callback=progress_callback
        )
        
        # Save results if output directory specified
        if output_dir:
            saved_files = save_parallel_results(
                results, output_dir, save_format
            )
            print(f"Results saved to: {saved_files}")
        
        # Print error summary
        if errors:
            print(f"Encountered {len(errors)} errors during processing:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        
        return results, errors
    
    return batch_processor