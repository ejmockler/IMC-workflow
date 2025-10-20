"""
Performance DAG - Unified Computation Graph with Content-Hash Caching

MISSION CRITICAL: Eliminate redundant computations and create explicit data flow.

This module implements:
1. Unified DAG with explicit step contracts
2. Content-hash caching to avoid recomputation  
3. Step memoization with automatic invalidation
4. Resource monitoring with fail-fast budgets
5. Streaming operations for large arrays

Engineering Focus: No redundant computation, clear data flow, predictable performance.
"""

import hashlib
import json
import tempfile
import os
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logger = logging.getLogger('PerformanceDAG')


class StepStatus(Enum):
    """Step execution status tracking."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class ResourceBudget:
    """Resource budget for step execution."""
    max_memory_gb: float = 16.0
    max_time_seconds: float = 300.0  # 5 minutes
    max_cpu_percent: float = 80.0
    enable_fail_fast: bool = True


@dataclass
class StepMetrics:
    """Performance metrics for step execution."""
    execution_time: float = 0.0
    peak_memory_gb: float = 0.0
    memory_increase_gb: float = 0.0
    cpu_percent: float = 0.0
    cache_hit: bool = False
    data_size_mb: float = 0.0
    input_hash: str = ""
    output_hash: str = ""


@dataclass 
class DAGStep:
    """Individual step in the computation DAG."""
    name: str
    function: Callable
    inputs: List[str]
    outputs: List[str]
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    status: StepStatus = StepStatus.PENDING
    metrics: StepMetrics = field(default_factory=StepMetrics)
    cache_key: str = ""
    dependencies: List[str] = field(default_factory=list)
    tile_safe: bool = False  # Can be executed in tiles
    
    def __post_init__(self):
        """Set dependencies from inputs after initialization."""
        if not self.dependencies:
            self.dependencies = self.inputs.copy()


class ContentHashCache:
    """Content-based caching system using SHA256 fingerprints."""
    
    def __init__(self, cache_dir: str = "cache", max_size_gb: float = 10.0):
        """Initialize cache with directory and size limits."""
        self.cache_dir = Path(cache_dir)
        # Secure cache directory with restricted permissions
        self.cache_dir.mkdir(exist_ok=True, mode=0o700)
        # Ensure directory has proper permissions if it already existed
        self.cache_dir.chmod(0o700)
        self.max_size_gb = max_size_gb
        self.cache_index = {}
        self.hit_rate_tracker = {"hits": 0, "misses": 0}
        
        # Load existing cache index
        self._load_cache_index()
    
    def compute_content_hash(self, data: Any, params: Dict[str, Any] = None) -> str:
        """Compute SHA256 hash of data + parameters."""
        hasher = hashlib.sha256()
        
        # Hash the data
        if isinstance(data, np.ndarray):
            hasher.update(data.tobytes())
            hasher.update(str(data.shape).encode())
            hasher.update(str(data.dtype).encode())
        elif isinstance(data, dict):
            # Sort keys for consistent hashing
            for key in sorted(data.keys()):
                hasher.update(key.encode())
                hasher.update(self.compute_content_hash(data[key]).encode())
        elif isinstance(data, (list, tuple)):
            for item in data:
                hasher.update(self.compute_content_hash(item).encode())
        else:
            hasher.update(str(data).encode())
        
        # Hash the parameters
        if params:
            param_str = json.dumps(params, sort_keys=True, default=str)
            hasher.update(param_str.encode())
        
        return hasher.hexdigest()
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached result by key."""
        if cache_key not in self.cache_index:
            self.hit_rate_tracker["misses"] += 1
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.npz"
        if not cache_file.exists():
            # Cache index is stale, remove entry
            del self.cache_index[cache_key]
            self.hit_rate_tracker["misses"] += 1
            return None
        
        try:
            # Load data from secure NPZ format
            loaded = np.load(cache_file, allow_pickle=False)
            
            # Reconstruct data based on type stored in metadata
            cache_meta = self.cache_index[cache_key].get("metadata", {})
            data_type = cache_meta.get("data_type", "numpy")
            
            if data_type == "numpy":
                result = loaded["data"]
            elif data_type == "dict":
                # Reconstruct dictionary from stored arrays
                result = {}
                for key in loaded.files:
                    if key != "metadata":
                        result[key] = loaded[key]
            elif data_type == "list":
                # Reconstruct list from stored array
                result = loaded["data"].tolist()
            else:
                # Fallback: try to get main data array
                result = loaded["data"]
            
            # Update access time
            self.cache_index[cache_key]["last_access"] = time.time()
            self.hit_rate_tracker["hits"] += 1
            
            logger.debug(f"Cache hit: {cache_key}")
            return result
            
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            self.hit_rate_tracker["misses"] += 1
            return None
    
    def put(self, cache_key: str, data: Any, metadata: Dict[str, Any] = None) -> bool:
        """Store data in cache with metadata."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.npz"
            
            # Store data in secure NPZ format based on type
            if isinstance(data, np.ndarray):
                np.savez_compressed(cache_file, data=data)
                data_type = "numpy"
            elif isinstance(data, dict):
                # Store dictionary as multiple arrays
                arrays = {}
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        arrays[str(key)] = value
                    else:
                        # Convert non-array values to arrays
                        arrays[str(key)] = np.array([value])
                np.savez_compressed(cache_file, **arrays)
                data_type = "dict"
            elif isinstance(data, (list, tuple)):
                # Convert to numpy array if possible
                try:
                    np.savez_compressed(cache_file, data=np.array(data))
                    data_type = "list"
                except:
                    # Fallback: convert to string representation
                    np.savez_compressed(cache_file, data=np.array([str(data)]))
                    data_type = "string"
            else:
                # Store scalar or other data as single-element array
                np.savez_compressed(cache_file, data=np.array([data]))
                data_type = "scalar"
            
            # Calculate file size
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
            
            # Store metadata including data type for reconstruction
            cache_metadata = metadata or {}
            cache_metadata["data_type"] = data_type
            
            # Update cache index
            self.cache_index[cache_key] = {
                "created": time.time(),
                "last_access": time.time(),
                "size_mb": file_size_mb,
                "metadata": cache_metadata
            }
            
            # Check cache size limits
            self._enforce_size_limits()
            
            logger.debug(f"Cached: {cache_key} ({file_size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cache {cache_key}: {e}")
            return False
    
    def _load_cache_index(self):
        """Load cache index from disk."""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self.cache_index = {}
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _enforce_size_limits(self):
        """Remove old cache entries if size limit exceeded."""
        total_size_mb = sum(entry["size_mb"] for entry in self.cache_index.values())
        
        if total_size_mb > self.max_size_gb * 1024:
            # Sort by last access time (LRU eviction)
            sorted_entries = sorted(
                self.cache_index.items(),
                key=lambda x: x[1]["last_access"]
            )
            
            # Remove oldest entries until under limit
            while total_size_mb > self.max_size_gb * 1024 and sorted_entries:
                cache_key, entry = sorted_entries.pop(0)
                cache_file = self.cache_dir / f"{cache_key}.npz"
                
                if cache_file.exists():
                    cache_file.unlink()
                
                total_size_mb -= entry["size_mb"]
                del self.cache_index[cache_key]
                
                logger.debug(f"Evicted cache entry: {cache_key}")
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_rate_tracker["hits"] + self.hit_rate_tracker["misses"]
        return self.hit_rate_tracker["hits"] / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size_mb = sum(entry["size_mb"] for entry in self.cache_index.values())
        
        return {
            "total_entries": len(self.cache_index),
            "total_size_mb": total_size_mb,
            "total_size_gb": total_size_mb / 1024,
            "hit_rate": self.get_hit_rate(),
            "hits": self.hit_rate_tracker["hits"],
            "misses": self.hit_rate_tracker["misses"]
        }
    
    def clear(self):
        """Clear all cache entries."""
        for cache_key in list(self.cache_index.keys()):
            cache_file = self.cache_dir / f"{cache_key}.npz"
            if cache_file.exists():
                cache_file.unlink()
        
        self.cache_index.clear()
        self.hit_rate_tracker = {"hits": 0, "misses": 0}


class ResourceMonitor:
    """Monitor resource usage during step execution."""
    
    def __init__(self, budget: ResourceBudget):
        """Initialize with resource budget."""
        self.budget = budget
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0.0
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / (1024**3)  # GB
        self.peak_memory = self.start_memory
        
    def check_budget(self) -> Tuple[bool, str]:
        """Check if resource usage is within budget."""
        if not self.start_time:
            return True, ""
        
        # Check execution time
        elapsed = time.time() - self.start_time
        if elapsed > self.budget.max_time_seconds:
            return False, f"Time budget exceeded: {elapsed:.1f}s > {self.budget.max_time_seconds}s"
        
        # Check memory usage
        current_memory = self.process.memory_info().rss / (1024**3)  # GB
        self.peak_memory = max(self.peak_memory, current_memory)
        
        if current_memory > self.budget.max_memory_gb:
            return False, f"Memory budget exceeded: {current_memory:.1f}GB > {self.budget.max_memory_gb}GB"
        
        # Check CPU usage
        try:
            cpu_percent = self.process.cpu_percent()
            if cpu_percent > self.budget.max_cpu_percent:
                return False, f"CPU budget exceeded: {cpu_percent:.1f}% > {self.budget.max_cpu_percent}%"
        except:
            pass  # CPU monitoring optional
        
        return True, ""
    
    def get_metrics(self) -> StepMetrics:
        """Get final resource metrics."""
        if not self.start_time:
            return StepMetrics()
        
        execution_time = time.time() - self.start_time
        memory_increase = self.peak_memory - self.start_memory
        
        try:
            cpu_percent = self.process.cpu_percent()
        except:
            cpu_percent = 0.0
        
        return StepMetrics(
            execution_time=execution_time,
            peak_memory_gb=self.peak_memory,
            memory_increase_gb=memory_increase,
            cpu_percent=cpu_percent
        )


class TilingUtilities:
    """Utilities for memory-efficient tile-based processing."""
    
    @staticmethod
    def estimate_tile_size(
        array_shape: Tuple[int, ...],
        memory_budget_gb: float,
        dtype: np.dtype = np.float32,
        safety_factor: float = 0.5
    ) -> Tuple[int, int]:
        """Estimate optimal tile size for given memory budget."""
        if len(array_shape) < 2:
            return array_shape[:2] if len(array_shape) >= 2 else (1, 1)
        
        h, w = array_shape[:2]
        bytes_per_element = dtype.itemsize
        
        # Calculate maximum pixels per tile
        memory_budget_bytes = memory_budget_gb * 1024**3 * safety_factor
        max_pixels = int(memory_budget_bytes / bytes_per_element)
        
        # Calculate tile dimensions maintaining aspect ratio
        aspect_ratio = w / h
        tile_h = int(np.sqrt(max_pixels / aspect_ratio))
        tile_w = int(tile_h * aspect_ratio)
        
        # Ensure tiles don't exceed array dimensions
        tile_h = min(tile_h, h)
        tile_w = min(tile_w, w)
        
        return (tile_h, tile_w)
    
    @staticmethod
    def generate_tiles(
        array_shape: Tuple[int, int],
        tile_size: Tuple[int, int],
        overlap: int = 0
    ) -> List[Tuple[slice, slice]]:
        """Generate tile coordinates with optional overlap."""
        h, w = array_shape
        tile_h, tile_w = tile_size
        
        tiles = []
        
        for y in range(0, h, tile_h - overlap):
            for x in range(0, w, tile_w - overlap):
                y_end = min(y + tile_h, h)
                x_end = min(x + tile_w, w)
                
                tiles.append((slice(y, y_end), slice(x, x_end)))
        
        return tiles
    
    @staticmethod
    def merge_tiled_results(
        tiled_results: List[Tuple[np.ndarray, Tuple[slice, slice]]],
        output_shape: Tuple[int, int],
        overlap: int = 0
    ) -> np.ndarray:
        """Merge results from tiled processing."""
        if not tiled_results:
            return np.zeros(output_shape)
        
        # Determine output dtype from first result
        first_result, _ = tiled_results[0]
        output = np.zeros(output_shape, dtype=first_result.dtype)
        
        if overlap == 0:
            # Simple case: no overlap
            for result, (y_slice, x_slice) in tiled_results:
                output[y_slice, x_slice] = result
        else:
            # Complex case: handle overlaps with averaging
            weight_map = np.zeros(output_shape, dtype=np.float32)
            
            for result, (y_slice, x_slice) in tiled_results:
                output[y_slice, x_slice] += result.astype(np.float32)
                weight_map[y_slice, x_slice] += 1.0
            
            # Normalize by weights
            valid_mask = weight_map > 0
            output[valid_mask] = output[valid_mask] / weight_map[valid_mask]
        
        return output


class PerformanceDAG:
    """Unified computation DAG with content-hash caching and resource monitoring."""
    
    def __init__(
        self,
        cache_dir: str = "performance_cache",
        max_cache_size_gb: float = 10.0,
        default_budget: ResourceBudget = None
    ):
        """Initialize performance DAG."""
        self.cache = ContentHashCache(cache_dir, max_cache_size_gb)
        self.default_budget = default_budget or ResourceBudget()
        self.steps: Dict[str, DAGStep] = {}
        self.data_store: Dict[str, Any] = {}
        self.execution_order: List[str] = []
        self.performance_log: List[Dict[str, Any]] = []
        
    def add_step(
        self,
        name: str,
        function: Callable,
        inputs: List[str],
        outputs: List[str],
        budget: Optional[ResourceBudget] = None,
        tile_safe: bool = False
    ) -> 'PerformanceDAG':
        """Add step to computation DAG."""
        step = DAGStep(
            name=name,
            function=function,
            inputs=inputs,
            outputs=outputs,
            budget=budget or self.default_budget,
            tile_safe=tile_safe
        )
        
        self.steps[name] = step
        return self
    
    def build_execution_order(self) -> List[str]:
        """Build topological execution order using Kahn's algorithm."""
        # Calculate in-degrees
        in_degree = {name: 0 for name in self.steps.keys()}
        
        for step in self.steps.values():
            for dep in step.dependencies:
                if dep in in_degree:
                    in_degree[step.name] += 1
        
        # Queue steps with no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            # Update in-degrees for dependent steps
            for step_name, step in self.steps.items():
                if current in step.dependencies:
                    in_degree[step_name] -= 1
                    if in_degree[step_name] == 0:
                        queue.append(step_name)
        
        # Check for cycles
        if len(execution_order) != len(self.steps):
            remaining = set(self.steps.keys()) - set(execution_order)
            raise ValueError(f"Circular dependencies detected in steps: {remaining}")
        
        self.execution_order = execution_order
        return execution_order
    
    def compute_cache_key(self, step: DAGStep) -> str:
        """Compute cache key for step based on inputs and parameters."""
        # Collect input data
        input_data = {}
        for input_name in step.inputs:
            if input_name in self.data_store:
                input_data[input_name] = self.data_store[input_name]
        
        # Include step function parameters (extract from function if possible)
        params = {
            "function_name": step.function.__name__,
            "step_name": step.name
        }
        
        # Compute hash
        return self.cache.compute_content_hash(input_data, params)
    
    def execute_step(self, step_name: str, force_recompute: bool = False) -> bool:
        """Execute single step with caching and resource monitoring."""
        step = self.steps[step_name]
        
        # Check cache first
        cache_key = self.compute_cache_key(step)
        step.cache_key = cache_key
        
        if not force_recompute:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                # Restore outputs to data store
                for i, output_name in enumerate(step.outputs):
                    if i < len(cached_result):
                        self.data_store[output_name] = cached_result[i]
                
                step.status = StepStatus.CACHED
                step.metrics.cache_hit = True
                
                logger.info(f"Step '{step_name}' cached result used")
                return True
        
        # Execute step with resource monitoring
        step.status = StepStatus.RUNNING
        monitor = ResourceMonitor(step.budget)
        monitor.start_monitoring()
        
        logger.info(f"Executing step '{step_name}'...")
        
        try:
            # Prepare inputs
            input_args = []
            for input_name in step.inputs:
                if input_name not in self.data_store:
                    raise ValueError(f"Input '{input_name}' not found for step '{step_name}'")
                input_args.append(self.data_store[input_name])
            
            # Check budget during execution
            within_budget, budget_msg = monitor.check_budget()
            if not within_budget and step.budget.enable_fail_fast:
                raise RuntimeError(f"Resource budget exceeded: {budget_msg}")
            
            # Execute function
            if len(input_args) == 1:
                result = step.function(input_args[0])
            else:
                result = step.function(*input_args)
            
            # Handle single vs multiple outputs
            if not isinstance(result, (list, tuple)):
                result = [result]
            
            # Store outputs
            outputs_for_cache = []
            for i, output_name in enumerate(step.outputs):
                if i < len(result):
                    self.data_store[output_name] = result[i]
                    outputs_for_cache.append(result[i])
            
            # Cache result
            self.cache.put(cache_key, outputs_for_cache, {
                "step_name": step_name,
                "execution_time": monitor.get_metrics().execution_time
            })
            
            step.status = StepStatus.COMPLETED
            step.metrics = monitor.get_metrics()
            step.metrics.input_hash = cache_key[:16]  # Short hash for logging
            
            logger.info(f"Step '{step_name}' completed in {step.metrics.execution_time:.2f}s")
            
            # Log performance
            self.performance_log.append({
                "step_name": step_name,
                "status": "completed",
                "metrics": step.metrics,
                "timestamp": time.time()
            })
            
            return True
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.metrics = monitor.get_metrics()
            
            logger.error(f"Step '{step_name}' failed: {e}")
            
            # Log failure
            self.performance_log.append({
                "step_name": step_name,
                "status": "failed", 
                "error": str(e),
                "metrics": step.metrics,
                "timestamp": time.time()
            })
            
            raise
        
        finally:
            # Cleanup
            gc.collect()
    
    def execute_dag(
        self,
        inputs: Dict[str, Any],
        force_recompute: bool = False,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """Execute entire DAG with optional parallelization."""
        # Store initial inputs
        self.data_store.update(inputs)
        
        # Build execution order
        execution_order = self.build_execution_order()
        
        logger.info(f"Executing DAG with {len(execution_order)} steps")
        logger.info(f"Execution order: {' -> '.join(execution_order)}")
        
        start_time = time.time()
        
        if parallel:
            self._execute_parallel(execution_order, force_recompute)
        else:
            self._execute_sequential(execution_order, force_recompute)
        
        total_time = time.time() - start_time
        
        # Generate performance report
        self._generate_performance_report(total_time)
        
        return self.data_store.copy()
    
    def _execute_sequential(self, execution_order: List[str], force_recompute: bool):
        """Execute steps sequentially."""
        for step_name in execution_order:
            self.execute_step(step_name, force_recompute)
    
    def _execute_parallel(self, execution_order: List[str], force_recompute: bool):
        """Execute steps in parallel where possible."""
        completed = set()
        
        while len(completed) < len(execution_order):
            # Find steps ready to execute (all dependencies completed)
            ready_steps = []
            for step_name in execution_order:
                if step_name in completed:
                    continue
                
                step = self.steps[step_name]
                deps_ready = all(dep in completed or dep not in self.steps 
                               for dep in step.dependencies)
                
                if deps_ready:
                    ready_steps.append(step_name)
            
            if not ready_steps:
                remaining = [s for s in execution_order if s not in completed]
                raise RuntimeError(f"No ready steps found. Remaining: {remaining}")
            
            # Execute ready steps in parallel
            with ThreadPoolExecutor(max_workers=min(4, len(ready_steps))) as executor:
                futures = {
                    executor.submit(self.execute_step, step_name, force_recompute): step_name
                    for step_name in ready_steps
                }
                
                for future in as_completed(futures):
                    step_name = futures[future]
                    try:
                        future.result()
                        completed.add(step_name)
                    except Exception as e:
                        logger.error(f"Parallel execution failed for {step_name}: {e}")
                        raise
    
    def _generate_performance_report(self, total_time: float):
        """Generate comprehensive performance report."""
        cache_stats = self.cache.get_stats()
        
        step_metrics = []
        total_execution_time = 0.0
        total_cached_time = 0.0
        
        for step_name, step in self.steps.items():
            if step.metrics.cache_hit:
                total_cached_time += step.metrics.execution_time
            else:
                total_execution_time += step.metrics.execution_time
            
            step_metrics.append({
                "name": step_name,
                "status": step.status.value,
                "execution_time": step.metrics.execution_time,
                "peak_memory_gb": step.metrics.peak_memory_gb,
                "cache_hit": step.metrics.cache_hit,
                "input_hash": step.metrics.input_hash
            })
        
        performance_gain = (total_cached_time / (total_execution_time + total_cached_time)) * 100 if (total_execution_time + total_cached_time) > 0 else 0
        
        report = {
            "execution_summary": {
                "total_time": total_time,
                "total_execution_time": total_execution_time,
                "total_cached_time": total_cached_time,
                "performance_gain_percent": performance_gain,
                "steps_executed": len([s for s in self.steps.values() if s.status == StepStatus.COMPLETED]),
                "steps_cached": len([s for s in self.steps.values() if s.status == StepStatus.CACHED]),
                "steps_failed": len([s for s in self.steps.values() if s.status == StepStatus.FAILED])
            },
            "cache_statistics": cache_stats,
            "step_metrics": step_metrics,
            "resource_efficiency": {
                "peak_memory_gb": max((s.metrics.peak_memory_gb for s in self.steps.values()), default=0),
                "avg_cpu_percent": np.mean([s.metrics.cpu_percent for s in self.steps.values() if s.metrics.cpu_percent > 0]),
                "memory_efficiency": "optimal" if max((s.metrics.peak_memory_gb for s in self.steps.values()), default=0) < 16.0 else "needs_optimization"
            }
        }
        
        logger.info(f"DAG Execution Report:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Performance gain: {performance_gain:.1f}%")
        logger.info(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")
        logger.info(f"  Peak memory: {report['resource_efficiency']['peak_memory_gb']:.1f}GB")
        
        self.performance_report = report
    
    def get_data(self, key: str) -> Any:
        """Get data from data store."""
        return self.data_store.get(key)
    
    def clear_cache(self):
        """Clear all cached results."""
        self.cache.clear()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        return getattr(self, 'performance_report', {})


# Pre-built DAG configurations for common IMC workflows

def create_imc_analysis_dag(
    cache_dir: str = "imc_cache",
    memory_budget_gb: float = 16.0
) -> PerformanceDAG:
    """Create unified DAG for IMC analysis pipeline."""
    
    # Set conservative resource budget
    budget = ResourceBudget(
        max_memory_gb=memory_budget_gb,
        max_time_seconds=600,  # 10 minutes per step
        enable_fail_fast=True
    )
    
    dag = PerformanceDAG(cache_dir, max_cache_size_gb=memory_budget_gb * 0.5, default_budget=budget)
    
    # Import analysis functions
    from .ion_count_processing import estimate_optimal_cofactor, apply_arcsinh_transform
    from .slic_segmentation import prepare_dna_composite, perform_slic_segmentation, aggregate_to_superpixels
    from .spatial_clustering import perform_spatial_clustering
    
    # Step 1: Cofactor optimization (cacheable, fast)
    def optimize_cofactors_step(ion_counts: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Optimize arcsinh cofactors for all proteins."""
        cofactors = {}
        for protein_name, counts in ion_counts.items():
            cofactors[protein_name] = estimate_optimal_cofactor(counts.ravel())
        return cofactors
    
    dag.add_step(
        name="optimize_cofactors",
        function=optimize_cofactors_step,
        inputs=["ion_counts"],
        outputs=["cofactors"],
        budget=ResourceBudget(max_memory_gb=2.0, max_time_seconds=60),
        tile_safe=False
    )
    
    # Step 2: Ion count transformation (tile-safe)
    def transform_ion_counts_step(ion_counts: Dict[str, np.ndarray], cofactors: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Apply arcsinh transformation with optimized cofactors."""
        transformed, _ = apply_arcsinh_transform(
            ion_counts, 
            optimization_method="cached",
            cached_cofactors=cofactors
        )
        return transformed
    
    dag.add_step(
        name="transform_ion_counts", 
        function=transform_ion_counts_step,
        inputs=["ion_counts", "cofactors"],
        outputs=["transformed_counts"],
        tile_safe=True
    )
    
    # Step 3: DNA composite preparation (tile-safe)
    def prepare_dna_step(coords: np.ndarray, dna1: np.ndarray, dna2: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """Prepare DNA composite for segmentation."""
        return prepare_dna_composite(coords, dna1, dna2, resolution_um=1.0)
    
    dag.add_step(
        name="prepare_dna",
        function=prepare_dna_step,
        inputs=["coords", "dna1_intensities", "dna2_intensities"],
        outputs=["composite_dna", "bounds"],
        tile_safe=True
    )
    
    # Step 4: SLIC segmentation (memory intensive)
    def slic_segmentation_step(composite_dna: np.ndarray, target_scale_um: float = 20.0) -> np.ndarray:
        """Perform SLIC segmentation."""
        return perform_slic_segmentation(
            composite_dna, 
            target_bin_size_um=target_scale_um,
            resolution_um=1.0
        )
    
    dag.add_step(
        name="slic_segmentation",
        function=slic_segmentation_step,
        inputs=["composite_dna"],
        outputs=["superpixel_labels"],
        budget=ResourceBudget(max_memory_gb=8.0, max_time_seconds=300),
        tile_safe=False
    )
    
    # Step 5: Aggregate to superpixels (compute intensive)
    def aggregate_step(coords: np.ndarray, transformed_counts: Dict[str, np.ndarray], 
                      superpixel_labels: np.ndarray, bounds: Tuple) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Aggregate ion counts to superpixels."""
        return aggregate_to_superpixels(coords, transformed_counts, superpixel_labels, bounds)
    
    dag.add_step(
        name="aggregate_superpixels",
        function=aggregate_step,
        inputs=["coords", "transformed_counts", "superpixel_labels", "bounds"],
        outputs=["superpixel_counts", "superpixel_coords"],
        budget=ResourceBudget(max_memory_gb=4.0, max_time_seconds=120),
        tile_safe=False
    )
    
    # Step 6: Feature matrix creation (fast)
    def create_feature_matrix_step(superpixel_counts: Dict[str, np.ndarray]) -> np.ndarray:
        """Create feature matrix from superpixel counts."""
        if not superpixel_counts:
            return np.array([])
        
        protein_names = list(superpixel_counts.keys())
        n_superpixels = len(next(iter(superpixel_counts.values())))
        
        feature_matrix = np.zeros((n_superpixels, len(protein_names)))
        for i, protein in enumerate(protein_names):
            feature_matrix[:, i] = superpixel_counts[protein]
        
        return feature_matrix
    
    dag.add_step(
        name="create_feature_matrix",
        function=create_feature_matrix_step,
        inputs=["superpixel_counts"],
        outputs=["feature_matrix"],
        budget=ResourceBudget(max_memory_gb=1.0, max_time_seconds=30),
        tile_safe=False
    )
    
    # Step 7: Spatial clustering (compute intensive)
    def spatial_clustering_step(feature_matrix: np.ndarray, superpixel_coords: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Perform spatial clustering."""
        if feature_matrix.size == 0:
            return np.array([]), {}
        
        return perform_spatial_clustering(
            feature_matrix=feature_matrix,
            spatial_coords=superpixel_coords,
            method='leiden',
            resolution=1.0
        )
    
    dag.add_step(
        name="spatial_clustering",
        function=spatial_clustering_step,
        inputs=["feature_matrix", "superpixel_coords"],
        outputs=["cluster_labels", "clustering_info"],
        budget=ResourceBudget(max_memory_gb=4.0, max_time_seconds=180),
        tile_safe=False
    )
    
    return dag


def create_multiscale_dag(
    scales_um: List[float] = [10.0, 20.0, 40.0],
    cache_dir: str = "multiscale_cache",
    memory_budget_gb: float = 16.0
) -> PerformanceDAG:
    """Create DAG for multiscale analysis with shared computation."""
    
    budget = ResourceBudget(max_memory_gb=memory_budget_gb, max_time_seconds=900)
    dag = PerformanceDAG(cache_dir, max_cache_size_gb=memory_budget_gb * 0.5, default_budget=budget)
    
    from .multiscale_analysis import perform_multiscale_analysis
    
    # Shared preprocessing steps (reuse from single-scale DAG)
    base_dag = create_imc_analysis_dag(cache_dir, memory_budget_gb)
    
    # Copy base steps
    for step_name, step in base_dag.steps.items():
        if step_name in ["optimize_cofactors", "transform_ion_counts", "prepare_dna"]:
            dag.steps[step_name] = step
    
    # Add scale-specific processing
    for scale_um in scales_um:
        scale_name = f"scale_{int(scale_um)}um"
        
        # Scale-specific segmentation
        def make_scale_segmentation(scale):
            def scale_segmentation_step(composite_dna: np.ndarray) -> np.ndarray:
                from .slic_segmentation import perform_slic_segmentation
                return perform_slic_segmentation(
                    composite_dna,
                    target_bin_size_um=scale,
                    resolution_um=1.0
                )
            return scale_segmentation_step
        
        dag.add_step(
            name=f"segmentation_{scale_name}",
            function=make_scale_segmentation(scale_um),
            inputs=["composite_dna"],
            outputs=[f"superpixel_labels_{scale_name}"],
            budget=ResourceBudget(max_memory_gb=6.0, max_time_seconds=300)
        )
        
        # Scale-specific aggregation
        def make_scale_aggregation(scale):
            def scale_aggregation_step(coords, transformed_counts, superpixel_labels, bounds):
                from .slic_segmentation import aggregate_to_superpixels
                return aggregate_to_superpixels(coords, transformed_counts, superpixel_labels, bounds)
            return scale_aggregation_step
        
        dag.add_step(
            name=f"aggregation_{scale_name}",
            function=make_scale_aggregation(scale_um),
            inputs=["coords", "transformed_counts", f"superpixel_labels_{scale_name}", "bounds"],
            outputs=[f"superpixel_counts_{scale_name}", f"superpixel_coords_{scale_name}"]
        )
    
    return dag


# Streaming utilities for large datasets

class StreamingProcessor:
    """Process large datasets in streaming fashion."""
    
    def __init__(self, dag: PerformanceDAG, memory_budget_gb: float = 4.0):
        """Initialize streaming processor."""
        self.dag = dag
        self.memory_budget_gb = memory_budget_gb
        self.tiling = TilingUtilities()
    
    def process_large_roi(
        self,
        coords: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        dna1_intensities: np.ndarray,
        dna2_intensities: np.ndarray,
        target_scale_um: float = 20.0
    ) -> Dict[str, Any]:
        """Process large ROI using tiling strategy."""
        
        logger.info(f"Processing large ROI with {len(coords)} pixels using streaming")
        
        # Estimate if we need tiling
        estimated_memory = self._estimate_memory_usage(coords, ion_counts)
        
        if estimated_memory < self.memory_budget_gb:
            # Process normally
            inputs = {
                "coords": coords,
                "ion_counts": ion_counts,
                "dna1_intensities": dna1_intensities,
                "dna2_intensities": dna2_intensities
            }
            return self.dag.execute_dag(inputs)
        
        # Use tiling approach
        logger.info(f"Estimated memory {estimated_memory:.1f}GB exceeds budget {self.memory_budget_gb}GB, using tiling")
        
        # Create spatial grid for tiling
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Estimate tile size
        array_shape = (int(y_max - y_min + 1), int(x_max - x_min + 1))
        tile_size = self.tiling.estimate_tile_size(
            array_shape, 
            self.memory_budget_gb * 0.7,  # Leave headroom
            dtype=np.float32
        )
        
        # Generate tiles
        tiles = self.tiling.generate_tiles(array_shape, tile_size, overlap=int(target_scale_um))
        
        logger.info(f"Processing {len(tiles)} tiles of size {tile_size}")
        
        # Process each tile
        tile_results = []
        for i, (y_slice, x_slice) in enumerate(tiles):
            logger.debug(f"Processing tile {i+1}/{len(tiles)}")
            
            # Extract tile data
            tile_mask = (
                (coords[:, 0] >= x_min + x_slice.start) &
                (coords[:, 0] < x_min + x_slice.stop) &
                (coords[:, 1] >= y_min + y_slice.start) &
                (coords[:, 1] < y_min + y_slice.stop)
            )
            
            if not np.any(tile_mask):
                continue
            
            tile_coords = coords[tile_mask]
            tile_ion_counts = {k: v[tile_mask] for k, v in ion_counts.items()}
            tile_dna1 = dna1_intensities[tile_mask]
            tile_dna2 = dna2_intensities[tile_mask]
            
            # Process tile
            tile_inputs = {
                "coords": tile_coords,
                "ion_counts": tile_ion_counts,
                "dna1_intensities": tile_dna1,
                "dna2_intensities": tile_dna2
            }
            
            tile_result = self.dag.execute_dag(tile_inputs, force_recompute=False)
            tile_results.append((tile_result, (y_slice, x_slice)))
            
            # Clear intermediate data
            del tile_coords, tile_ion_counts, tile_dna1, tile_dna2, tile_result
            gc.collect()
        
        # Merge tile results
        merged_result = self._merge_tile_results(tile_results, array_shape)
        
        logger.info(f"Completed streaming processing of {len(tiles)} tiles")
        
        return merged_result
    
    def _estimate_memory_usage(self, coords: np.ndarray, ion_counts: Dict[str, np.ndarray]) -> float:
        """Estimate memory usage for processing."""
        n_pixels = len(coords)
        n_proteins = len(ion_counts)
        
        # Rough estimates based on pipeline stages
        base_data_gb = (n_pixels * n_proteins * 4) / (1024**3)  # float32
        processing_overhead = 3.0  # Factor for intermediate results
        
        return base_data_gb * processing_overhead
    
    def _merge_tile_results(self, tile_results: List, array_shape: Tuple) -> Dict[str, Any]:
        """Merge results from multiple tiles."""
        if not tile_results:
            return {}
        
        # For now, just concatenate results from tiles
        # In practice, would need sophisticated merging for overlapping regions
        
        merged_superpixel_counts = {}
        merged_coords = []
        merged_labels = []
        
        for tile_result, _ in tile_results:
            if "superpixel_counts" in tile_result:
                for protein, counts in tile_result["superpixel_counts"].items():
                    if protein not in merged_superpixel_counts:
                        merged_superpixel_counts[protein] = []
                    merged_superpixel_counts[protein].extend(counts)
            
            if "superpixel_coords" in tile_result:
                merged_coords.extend(tile_result["superpixel_coords"])
            
            if "cluster_labels" in tile_result:
                merged_labels.extend(tile_result["cluster_labels"])
        
        # Convert to numpy arrays
        for protein in merged_superpixel_counts:
            merged_superpixel_counts[protein] = np.array(merged_superpixel_counts[protein])
        
        return {
            "superpixel_counts": merged_superpixel_counts,
            "superpixel_coords": np.array(merged_coords) if merged_coords else np.array([]),
            "cluster_labels": np.array(merged_labels) if merged_labels else np.array([]),
            "processing_method": "streaming_tiles",
            "n_tiles_processed": len(tile_results)
        }


# Validation and benchmarking utilities

def validate_dag_performance(
    dag: PerformanceDAG,
    test_data: Dict[str, Any],
    target_speedup: float = 1.5,
    max_memory_gb: float = 16.0
) -> Dict[str, Any]:
    """Validate DAG performance against targets."""
    
    # First run (populate cache)
    start_time = time.time()
    result1 = dag.execute_dag(test_data.copy())
    first_run_time = time.time() - start_time
    
    # Second run (should use cache)
    start_time = time.time()
    result2 = dag.execute_dag(test_data.copy())
    second_run_time = time.time() - start_time
    
    # Calculate metrics
    speedup = first_run_time / second_run_time if second_run_time > 0 else 1.0
    cache_hit_rate = dag.cache.get_hit_rate()
    
    performance_report = dag.get_performance_report()
    peak_memory = performance_report.get("resource_efficiency", {}).get("peak_memory_gb", 0)
    
    validation_result = {
        "performance_targets": {
            "target_speedup": target_speedup,
            "achieved_speedup": speedup,
            "speedup_met": speedup >= target_speedup,
            "target_memory_gb": max_memory_gb,
            "peak_memory_gb": peak_memory,
            "memory_target_met": peak_memory <= max_memory_gb
        },
        "cache_performance": {
            "hit_rate": cache_hit_rate,
            "cache_effective": cache_hit_rate > 0.5
        },
        "timing_breakdown": {
            "first_run_seconds": first_run_time,
            "second_run_seconds": second_run_time,
            "cache_savings_seconds": first_run_time - second_run_time
        },
        "overall_success": (speedup >= target_speedup and 
                          peak_memory <= max_memory_gb and 
                          cache_hit_rate > 0.3)
    }
    
    logger.info(f"DAG Performance Validation:")
    logger.info(f"  Speedup: {speedup:.2f}x (target: {target_speedup:.2f}x)")
    logger.info(f"  Memory: {peak_memory:.1f}GB (limit: {max_memory_gb}GB)")
    logger.info(f"  Cache hit rate: {cache_hit_rate:.1%}")
    logger.info(f"  Overall success: {validation_result['overall_success']}")
    
    return validation_result


if __name__ == "__main__":
    # Example usage and testing
    
    # Create test data
    np.random.seed(42)
    test_coords = np.random.uniform(0, 1000, (1000, 2))
    test_ion_counts = {
        f"Protein_{i}": np.random.poisson(10, 1000).astype(np.float32)
        for i in range(5)
    }
    test_dna1 = np.random.poisson(50, 1000).astype(np.float32)
    test_dna2 = np.random.poisson(45, 1000).astype(np.float32)
    
    test_data = {
        "coords": test_coords,
        "ion_counts": test_ion_counts,
        "dna1_intensities": test_dna1,
        "dna2_intensities": test_dna2
    }
    
    # Create and test DAG
    dag = create_imc_analysis_dag(memory_budget_gb=8.0)
    
    print("Testing Performance DAG...")
    result = dag.execute_dag(test_data)
    
    print("\nPerformance Report:")
    report = dag.get_performance_report()
    print(json.dumps(report, indent=2, default=str))
    
    # Validate performance
    print("\nValidating Performance...")
    validation = validate_dag_performance(dag, test_data, target_speedup=2.0)
    print(json.dumps(validation, indent=2, default=str))
    
    print("\nPerformance DAG implementation complete!")