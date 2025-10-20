"""
Memory Optimizer - Surgical Precision Memory Surgery

MISSION: Eliminate performance bottlenecks through dtype optimization and copy reduction.
Implements float32 discipline, eliminates unnecessary .copy() calls, and provides
memory profiling to achieve 50% memory reduction while maintaining scientific validity.

Key Features:
- Float32 discipline throughout the pipeline
- Elimination of unnecessary .copy() calls
- Memory-efficient dtype selection (int16 for labels, float32 for data)
- Memory profiling and monitoring
- Validation that results remain scientifically equivalent
"""

import numpy as np
import psutil
import gc
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from contextlib import contextmanager
import time


@dataclass
class MemoryOptimizationReport:
    """Comprehensive memory optimization report."""
    before_memory_gb: float
    after_memory_gb: float
    memory_reduction_percent: float
    dtype_conversions: Dict[str, int]
    copy_eliminations: int
    processing_time_ms: float
    validation_passed: bool
    warnings: List[str]


class DtypeOptimizer:
    """Optimizes numpy array dtypes for memory efficiency."""
    
    # Optimal dtype mappings for different data types
    DTYPE_MAPPINGS = {
        'float64': np.float32,
        'int64': np.int32,
        'int32': np.int16,  # For labels/indices when possible
        'complex128': np.complex64,
    }
    
    # Minimum values for dtype conversions
    MIN_VALUES = {
        np.int16: -32768,
        np.int32: -2147483648,
        np.uint16: 0,
        np.uint32: 0,
    }
    
    MAX_VALUES = {
        np.int16: 32767,
        np.int32: 2147483647,
        np.uint16: 65535,
        np.uint32: 4294967295,
    }
    
    @classmethod
    def optimize_array_dtype(cls, arr: np.ndarray, 
                           force_dtype: Optional[np.dtype] = None,
                           preserve_precision: bool = True) -> Tuple[np.ndarray, str]:
        """
        Optimize array dtype while preserving data integrity.
        
        Args:
            arr: Input array
            force_dtype: Force specific dtype (overrides optimization)
            preserve_precision: Whether to preserve floating point precision
            
        Returns:
            Tuple of (optimized_array, conversion_info)
        """
        if arr.size == 0:
            return arr, "empty_array"
        
        original_dtype = arr.dtype
        
        # Use forced dtype if specified
        if force_dtype is not None:
            if cls._is_safe_conversion(arr, force_dtype):
                return arr.astype(force_dtype), f"{original_dtype} -> {force_dtype} (forced)"
            else:
                warnings.warn(f"Unsafe forced conversion {original_dtype} -> {force_dtype}, skipping")
                return arr, f"{original_dtype} (unsafe_forced_conversion)"
        
        # Float64 -> Float32 optimization
        if original_dtype == np.float64:
            if preserve_precision:
                # Check if conversion is safe (no overflow/underflow)
                if cls._is_safe_float32_conversion(arr):
                    return arr.astype(np.float32), f"float64 -> float32"
                else:
                    warnings.warn("Float64 values outside float32 range, preserving precision")
                    return arr, "float64 (preserved_precision)"
            else:
                return arr.astype(np.float32), f"float64 -> float32 (forced)"
        
        # Integer optimization
        elif original_dtype in [np.int64, np.int32]:
            optimal_dtype = cls._find_optimal_int_dtype(arr)
            if optimal_dtype != original_dtype:
                return arr.astype(optimal_dtype), f"{original_dtype} -> {optimal_dtype}"
        
        # Complex optimization
        elif original_dtype == np.complex128:
            return arr.astype(np.complex64), f"complex128 -> complex64"
        
        return arr, f"{original_dtype} (no_optimization)"
    
    @classmethod
    def _is_safe_float32_conversion(cls, arr: np.ndarray) -> bool:
        """Check if float64 array can be safely converted to float32."""
        if arr.dtype != np.float64:
            return True
        
        # Check for values outside float32 range
        float32_min = np.finfo(np.float32).min
        float32_max = np.finfo(np.float32).max
        
        finite_mask = np.isfinite(arr)
        if not np.any(finite_mask):
            return True  # Only inf/nan values, safe to convert
        
        finite_values = arr[finite_mask]
        return (np.all(finite_values >= float32_min) and 
                np.all(finite_values <= float32_max))
    
    @classmethod
    def _find_optimal_int_dtype(cls, arr: np.ndarray) -> np.dtype:
        """Find optimal integer dtype for array."""
        if arr.size == 0:
            return arr.dtype
        
        min_val = np.min(arr)
        max_val = np.max(arr)
        
        # Check if int16 is sufficient
        if (min_val >= cls.MIN_VALUES[np.int16] and 
            max_val <= cls.MAX_VALUES[np.int16]):
            return np.int16
        
        # Check if int32 is sufficient
        if (min_val >= cls.MIN_VALUES[np.int32] and 
            max_val <= cls.MAX_VALUES[np.int32]):
            return np.int32
        
        # Keep as int64
        return np.int64
    
    @classmethod
    def _is_safe_conversion(cls, arr: np.ndarray, target_dtype: np.dtype) -> bool:
        """Check if conversion to target dtype is safe."""
        if arr.size == 0:
            return True
        
        # For integer conversions, check range
        if np.issubdtype(target_dtype, np.integer):
            if target_dtype in cls.MIN_VALUES:
                min_val = np.min(arr)
                max_val = np.max(arr)
                return (min_val >= cls.MIN_VALUES[target_dtype] and 
                       max_val <= cls.MAX_VALUES[target_dtype])
        
        # For float conversions, check precision
        elif np.issubdtype(target_dtype, np.floating):
            if target_dtype == np.float32:
                return cls._is_safe_float32_conversion(arr)
        
        return True


class CopyEliminator:
    """Eliminates unnecessary array copies and promotes view usage."""
    
    @staticmethod
    def safe_view_or_copy(arr: np.ndarray, 
                         force_copy: bool = False,
                         writeable: bool = True) -> np.ndarray:
        """
        Return view if safe, copy only when necessary.
        
        Args:
            arr: Input array
            force_copy: Force copy even if view is safe
            writeable: Whether result needs to be writeable
            
        Returns:
            View or copy of array
        """
        if force_copy:
            result = arr.copy()
        else:
            # Try to use view
            if arr.flags.c_contiguous or arr.flags.f_contiguous:
                # Array is contiguous, view is safe
                result = arr.view()
            else:
                # Non-contiguous, need copy
                result = arr.copy()
        
        if writeable and not result.flags.writeable:
            result = result.copy()
        
        return result
    
    @staticmethod
    def eliminate_redundant_copy(arr: np.ndarray, 
                               operation: str = "unknown") -> np.ndarray:
        """
        Eliminate redundant copy operations.
        
        Args:
            arr: Array that would be copied
            operation: Description of operation for logging
            
        Returns:
            Array (view if possible, copy only if necessary)
        """
        # Check if array is already a copy or can be modified in place
        if arr.flags.owndata and arr.flags.writeable:
            # Array owns its data and is writeable, no copy needed
            return arr
        
        # Check if a view is sufficient
        if not arr.flags.writeable:
            # Need writeable copy
            return arr.copy()
        
        # Return view if possible
        return arr.view()


class MemoryProfiler:
    """Memory profiling and monitoring utilities."""
    
    def __init__(self):
        self.snapshots = []
        self.process = psutil.Process()
    
    def take_snapshot(self, label: str) -> Dict[str, float]:
        """Take memory snapshot with timestamp."""
        memory_info = self.process.memory_info()
        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'rss_gb': memory_info.rss / (1024**3),
            'vms_gb': memory_info.vms / (1024**3),
        }
        
        # Add system memory info
        sys_memory = psutil.virtual_memory()
        snapshot.update({
            'system_available_gb': sys_memory.available / (1024**3),
            'system_used_percent': sys_memory.percent
        })
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_memory_delta(self, start_label: str, end_label: str) -> float:
        """Get memory change between two snapshots."""
        start_snapshot = next((s for s in self.snapshots if s['label'] == start_label), None)
        end_snapshot = next((s for s in self.snapshots if s['label'] == end_label), None)
        
        if start_snapshot and end_snapshot:
            return end_snapshot['rss_gb'] - start_snapshot['rss_gb']
        return 0.0
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage across all snapshots."""
        if not self.snapshots:
            return 0.0
        return max(s['rss_gb'] for s in self.snapshots)
    
    def clear_snapshots(self):
        """Clear all snapshots."""
        self.snapshots.clear()


@contextmanager
def memory_optimization_context(label: str = "optimization"):
    """Context manager for memory optimization tracking."""
    profiler = MemoryProfiler()
    profiler.take_snapshot(f"{label}_start")
    
    try:
        yield profiler
    finally:
        profiler.take_snapshot(f"{label}_end")
        gc.collect()  # Force garbage collection


class PipelineMemoryOptimizer:
    """Main memory optimizer for the entire pipeline."""
    
    def __init__(self, 
                 target_dtype: str = 'float32',
                 preserve_precision: bool = False,
                 validate_results: bool = True):
        self.target_dtype = getattr(np, target_dtype)
        self.preserve_precision = preserve_precision
        self.validate_results = validate_results
        
        self.dtype_optimizer = DtypeOptimizer()
        self.copy_eliminator = CopyEliminator()
        self.profiler = MemoryProfiler()
        
        self.conversion_stats = {
            'dtype_conversions': {},
            'copy_eliminations': 0,
            'warnings': []
        }
    
    def optimize_coordinate_data(self, 
                               coords: np.ndarray,
                               ion_counts: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Optimize coordinate and ion count data dtypes.
        
        Args:
            coords: Coordinate array
            ion_counts: Ion count dictionary
            
        Returns:
            Tuple of (optimized_coords, optimized_ion_counts)
        """
        with memory_optimization_context("coordinate_optimization") as profiler:
            # Optimize coordinates
            opt_coords, coord_info = self.dtype_optimizer.optimize_array_dtype(
                coords, force_dtype=self.target_dtype, 
                preserve_precision=self.preserve_precision
            )
            self._record_conversion('coordinates', coord_info)
            
            # Optimize ion counts
            opt_ion_counts = {}
            for protein_name, counts in ion_counts.items():
                opt_counts, count_info = self.dtype_optimizer.optimize_array_dtype(
                    counts, force_dtype=self.target_dtype,
                    preserve_precision=self.preserve_precision
                )
                opt_ion_counts[protein_name] = opt_counts
                self._record_conversion(f'ion_counts_{protein_name}', count_info)
            
            return opt_coords, opt_ion_counts
    
    def optimize_aggregated_arrays(self, 
                                 aggregated_counts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Optimize aggregated count arrays."""
        opt_aggregated = {}
        for protein_name, counts in aggregated_counts.items():
            # Aggregated counts should use float32
            opt_counts, info = self.dtype_optimizer.optimize_array_dtype(
                counts, force_dtype=self.target_dtype
            )
            opt_aggregated[protein_name] = opt_counts
            self._record_conversion(f'aggregated_{protein_name}', info)
        
        return opt_aggregated
    
    def optimize_cluster_labels(self, 
                              cluster_labels: np.ndarray,
                              cluster_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize cluster labels and maps to minimal integer dtypes."""
        # Use int16 for cluster labels if possible
        opt_labels, label_info = self.dtype_optimizer.optimize_array_dtype(
            cluster_labels, preserve_precision=False
        )
        self._record_conversion('cluster_labels', label_info)
        
        opt_map, map_info = self.dtype_optimizer.optimize_array_dtype(
            cluster_map, preserve_precision=False
        )
        self._record_conversion('cluster_map', map_info)
        
        return opt_labels, opt_map
    
    def eliminate_pipeline_copies(self, 
                                pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Eliminate unnecessary copies in pipeline data."""
        optimized_data = {}
        
        for key, value in pipeline_data.items():
            if isinstance(value, np.ndarray):
                # Use view instead of copy where possible
                optimized_data[key] = self.copy_eliminator.eliminate_redundant_copy(
                    value, operation=f"pipeline_{key}"
                )
                self.conversion_stats['copy_eliminations'] += 1
            
            elif isinstance(value, dict) and key in ['aggregated_counts', 'transformed_arrays', 'standardized_arrays']:
                # Optimize dictionary of arrays
                optimized_dict = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        optimized_dict[subkey] = self.copy_eliminator.eliminate_redundant_copy(
                            subvalue, operation=f"{key}_{subkey}"
                        )
                        self.conversion_stats['copy_eliminations'] += 1
                    else:
                        optimized_dict[subkey] = subvalue
                optimized_data[key] = optimized_dict
            
            else:
                # Keep non-array data as is
                optimized_data[key] = value
        
        return optimized_data
    
    def validate_optimization(self, 
                            original_data: Dict[str, Any],
                            optimized_data: Dict[str, Any],
                            tolerance: float = 1e-6) -> bool:
        """
        Validate that optimization preserves scientific accuracy.
        
        Args:
            original_data: Original pipeline data
            optimized_data: Optimized pipeline data
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if optimization is valid
        """
        try:
            # Compare key arrays
            array_keys = ['feature_matrix', 'cluster_labels']
            
            for key in array_keys:
                if key in original_data and key in optimized_data:
                    orig_arr = original_data[key]
                    opt_arr = optimized_data[key]
                    
                    if not self._arrays_equivalent(orig_arr, opt_arr, tolerance):
                        self.conversion_stats['warnings'].append(
                            f"Validation failed for {key}: arrays not equivalent"
                        )
                        return False
            
            # Compare aggregated counts
            if 'aggregated_counts' in original_data and 'aggregated_counts' in optimized_data:
                orig_counts = original_data['aggregated_counts']
                opt_counts = optimized_data['aggregated_counts']
                
                for protein_name in orig_counts:
                    if protein_name in opt_counts:
                        if not self._arrays_equivalent(
                            orig_counts[protein_name], 
                            opt_counts[protein_name], 
                            tolerance
                        ):
                            self.conversion_stats['warnings'].append(
                                f"Validation failed for {protein_name}: counts not equivalent"
                            )
                            return False
            
            return True
            
        except Exception as e:
            self.conversion_stats['warnings'].append(f"Validation error: {str(e)}")
            return False
    
    def _arrays_equivalent(self, 
                          arr1: np.ndarray, 
                          arr2: np.ndarray, 
                          tolerance: float = 1e-6) -> bool:
        """Check if two arrays are numerically equivalent."""
        if arr1.shape != arr2.shape:
            return False
        
        # Handle different dtypes
        if arr1.dtype != arr2.dtype:
            # Convert to common dtype for comparison
            common_dtype = np.result_type(arr1.dtype, arr2.dtype)
            arr1_cmp = arr1.astype(common_dtype)
            arr2_cmp = arr2.astype(common_dtype)
        else:
            arr1_cmp = arr1
            arr2_cmp = arr2
        
        # Handle special values
        nan_mask = np.isnan(arr1_cmp) | np.isnan(arr2_cmp)
        inf_mask = np.isinf(arr1_cmp) | np.isinf(arr2_cmp)
        
        # Check that NaN and inf patterns match
        if not np.array_equal(np.isnan(arr1_cmp), np.isnan(arr2_cmp)):
            return False
        if not np.array_equal(np.isinf(arr1_cmp), np.isinf(arr2_cmp)):
            return False
        
        # Compare finite values
        finite_mask = ~(nan_mask | inf_mask)
        if np.any(finite_mask):
            finite1 = arr1_cmp[finite_mask]
            finite2 = arr2_cmp[finite_mask]
            
            # Use relative tolerance for better float32 comparison
            if not np.allclose(finite1, finite2, rtol=tolerance, atol=tolerance):
                return False
        
        return True
    
    def _record_conversion(self, key: str, conversion_info: str):
        """Record dtype conversion for reporting."""
        if conversion_info not in self.conversion_stats['dtype_conversions']:
            self.conversion_stats['dtype_conversions'][conversion_info] = 0
        self.conversion_stats['dtype_conversions'][conversion_info] += 1
    
    def generate_optimization_report(self, 
                                   before_memory_gb: float,
                                   after_memory_gb: float,
                                   processing_time_ms: float,
                                   validation_passed: bool) -> MemoryOptimizationReport:
        """Generate comprehensive optimization report."""
        memory_reduction = ((before_memory_gb - after_memory_gb) / before_memory_gb) * 100
        
        return MemoryOptimizationReport(
            before_memory_gb=before_memory_gb,
            after_memory_gb=after_memory_gb,
            memory_reduction_percent=memory_reduction,
            dtype_conversions=self.conversion_stats['dtype_conversions'].copy(),
            copy_eliminations=self.conversion_stats['copy_eliminations'],
            processing_time_ms=processing_time_ms,
            validation_passed=validation_passed,
            warnings=self.conversion_stats['warnings'].copy()
        )


def optimize_ion_count_pipeline(
    coords: np.ndarray,
    ion_counts: Dict[str, np.ndarray],
    pipeline_function: callable,
    target_dtype: str = 'float32',
    validate_results: bool = True,
    **pipeline_kwargs
) -> Tuple[Dict[str, Any], MemoryOptimizationReport]:
    """
    Optimize memory usage for the complete ion count pipeline.
    
    Args:
        coords: Coordinate array
        ion_counts: Ion count dictionary
        pipeline_function: Pipeline function to optimize
        target_dtype: Target dtype for optimization
        validate_results: Whether to validate results
        **pipeline_kwargs: Additional pipeline arguments
        
    Returns:
        Tuple of (optimized_results, optimization_report)
    """
    optimizer = PipelineMemoryOptimizer(
        target_dtype=target_dtype,
        validate_results=validate_results
    )
    
    start_time = time.time()
    
    # Take initial memory snapshot
    initial_memory = optimizer.profiler.take_snapshot('optimization_start')['rss_gb']
    
    # Step 1: Optimize input data
    opt_coords, opt_ion_counts = optimizer.optimize_coordinate_data(coords, ion_counts)
    
    # Step 2: Run original pipeline for validation (if requested)
    original_results = None
    if validate_results:
        original_results = pipeline_function(coords, ion_counts, **pipeline_kwargs)
    
    # Step 3: Run optimized pipeline
    optimized_results = pipeline_function(opt_coords, opt_ion_counts, **pipeline_kwargs)
    
    # Step 4: Further optimize pipeline outputs
    if 'aggregated_counts' in optimized_results:
        optimized_results['aggregated_counts'] = optimizer.optimize_aggregated_arrays(
            optimized_results['aggregated_counts']
        )
    
    if 'cluster_labels' in optimized_results and 'cluster_map' in optimized_results:
        opt_labels, opt_map = optimizer.optimize_cluster_labels(
            optimized_results['cluster_labels'],
            optimized_results['cluster_map']
        )
        optimized_results['cluster_labels'] = opt_labels
        optimized_results['cluster_map'] = opt_map
    
    # Step 5: Eliminate unnecessary copies
    optimized_results = optimizer.eliminate_pipeline_copies(optimized_results)
    
    # Step 6: Validate results
    validation_passed = True
    if validate_results and original_results is not None:
        validation_passed = optimizer.validate_optimization(
            original_results, optimized_results
        )
    
    # Take final memory snapshot
    final_memory = optimizer.profiler.take_snapshot('optimization_end')['rss_gb']
    
    # Generate report
    processing_time_ms = (time.time() - start_time) * 1000
    
    report = optimizer.generate_optimization_report(
        before_memory_gb=initial_memory,
        after_memory_gb=final_memory,
        processing_time_ms=processing_time_ms,
        validation_passed=validation_passed
    )
    
    return optimized_results, report


def scan_codebase_for_memory_issues(base_path: str = "/Users/noot/Documents/IMC/src/analysis") -> Dict[str, Any]:
    """
    Scan codebase for common memory issues.
    
    Args:
        base_path: Base path to scan
        
    Returns:
        Dictionary with scan results
    """
    import os
    import re
    
    issues = {
        'float64_usage': [],
        'unnecessary_copies': [],
        'large_array_allocations': [],
        'memory_leaks': []
    }
    
    # Patterns to search for
    patterns = {
        'float64': re.compile(r'dtype\s*=\s*np\.float64|dtype\s*=\s*\'float64\'|\.astype\(np\.float64\)'),
        'copy_calls': re.compile(r'\.copy\(\)'),
        'zeros_large': re.compile(r'np\.zeros\([^)]*\d{4,}[^)]*\)'),  # Large allocations
        'empty_large': re.compile(r'np\.empty\([^)]*\d{4,}[^)]*\)'),
    }
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for line_num, line in enumerate(lines, 1):
                            # Check for float64 usage
                            if patterns['float64'].search(line):
                                issues['float64_usage'].append({
                                    'file': filepath,
                                    'line': line_num,
                                    'content': line.strip()
                                })
                            
                            # Check for .copy() calls
                            if patterns['copy_calls'].search(line):
                                issues['unnecessary_copies'].append({
                                    'file': filepath,
                                    'line': line_num,
                                    'content': line.strip()
                                })
                            
                            # Check for large array allocations
                            if (patterns['zeros_large'].search(line) or 
                                patterns['empty_large'].search(line)):
                                issues['large_array_allocations'].append({
                                    'file': filepath,
                                    'line': line_num,
                                    'content': line.strip()
                                })
                
                except Exception as e:
                    issues['memory_leaks'].append({
                        'file': filepath,
                        'error': str(e)
                    })
    
    return issues


# Factory function for creating optimized versions of existing functions
def create_memory_optimized_function(original_function: callable, 
                                   optimization_config: Optional[Dict[str, Any]] = None):
    """Create memory-optimized wrapper for existing functions."""
    def optimized_wrapper(*args, **kwargs):
        # Extract arrays from arguments
        array_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                array_args.append(arg)
        
        # Optimize arrays
        optimizer = DtypeOptimizer()
        optimized_args = []
        
        for arg in args:
            if isinstance(arg, np.ndarray):
                opt_arg, _ = optimizer.optimize_array_dtype(arg)
                optimized_args.append(opt_arg)
            else:
                optimized_args.append(arg)
        
        # Call original function with optimized arguments
        return original_function(*optimized_args, **kwargs)
    
    return optimized_wrapper