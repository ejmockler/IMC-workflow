"""
Performance profiling utilities for the IMC pipeline.

Provides lightweight timing helpers that can be reused across the
optimization effort without imposing mandatory logging or dependencies.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

logger = logging.getLogger("PerformanceProfiling")


class PerformanceTimer:
    """Context manager for measuring execution time of critical sections."""

    def __init__(self, name: str, log_result: bool = True):
        self.name = name
        self.log_result = log_result
        self.elapsed: Optional[float] = None
        self._start: Optional[float] = None

    def __enter__(self) -> "PerformanceTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.elapsed = time.perf_counter() - (self._start or time.perf_counter())
        if self.log_result:
            logger.info("%s: %.3fs", self.name, self.elapsed)


def profile_function(func: Callable) -> Callable:
    """Decorator that records elapsed time and attaches it to dict results."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with PerformanceTimer(func.__name__) as timer:
            result = func(*args, **kwargs)

        if isinstance(result, dict):
            result = dict(result)
            result["_performance_timing_seconds"] = timer.elapsed

        return result

    return wrapper


_profiling_stats: Dict[str, list[float]] = {}


def record_timing(operation: str, elapsed: float) -> None:
    """Collect elapsed times for aggregation after long runs."""
    _profiling_stats.setdefault(operation, []).append(elapsed)


def merge_timings(operation: str, timings: Iterable[float]) -> None:
    """Bulk-add previously captured timings for the same operation."""
    _profiling_stats.setdefault(operation, []).extend(timings)


def get_profiling_summary() -> Dict[str, Dict[str, float]]:
    """Return descriptive statistics for recorded timings."""
    summary: Dict[str, Dict[str, float]] = {}
    for operation, timings in _profiling_stats.items():
        if not timings:
            continue
        data = np.asarray(timings, dtype=float)
        summary[operation] = {
            "count": float(data.size),
            "total_seconds": float(data.sum()),
            "mean_seconds": float(data.mean()),
            "std_seconds": float(data.std(ddof=0)),
            "min_seconds": float(data.min()),
            "max_seconds": float(data.max()),
        }
    return summary


def clear_profiling_stats() -> None:
    """Reset all accumulated profiling statistics."""
    _profiling_stats.clear()
