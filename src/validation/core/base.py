"""
Pure Segmentation Validation Framework

This module contains NO biological knowledge. It validates segmentation quality
through geometric and spatial metrics that apply to ANY segmented image,
whether satellite imagery, CT scans, or microscopy.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np


class Metric(ABC):
    """
    Abstract base for segmentation quality metrics.
    
    Metrics are pure functions that assess geometric or spatial properties
    without any domain-specific assumptions.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this metric."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this metric measures."""
        pass
    
    @abstractmethod
    def compute(self, 
                segmentation: np.ndarray, 
                context: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Compute metric value(s) for given segmentation.
        
        Args:
            segmentation: 2D array of integer segment labels
            context: Optional auxiliary data (e.g., intensity images)
                    No assumptions made about keys or meaning
        
        Returns:
            Dictionary of metric values (can be single or multiple values)
        """
        pass
    
    def requires_context(self) -> bool:
        """Whether this metric needs auxiliary context data."""
        return False
    
    def validate_input(self, segmentation: np.ndarray) -> bool:
        """Check if input is valid for this metric."""
        if segmentation.ndim != 2:
            return False
        if not np.issubdtype(segmentation.dtype, np.integer):
            return False
        return True


class SegmentationValidator:
    """
    Core validation framework for segmentation quality assessment.
    
    This validator is completely domain-agnostic and can assess
    segmentation quality for any type of image data.
    """
    
    def __init__(self, metrics: Optional[List[Metric]] = None):
        """
        Initialize validator with pluggable metrics.
        
        Args:
            metrics: List of metrics to compute. If None, no metrics computed.
        """
        self.metrics = metrics or []
        self._results_cache = {}
    
    def add_metric(self, metric: Metric) -> None:
        """Add a metric to the validation pipeline."""
        if not isinstance(metric, Metric):
            raise TypeError(f"Expected Metric, got {type(metric)}")
        self.metrics.append(metric)
    
    def remove_metric(self, metric_name: str) -> None:
        """Remove a metric by name."""
        self.metrics = [m for m in self.metrics if m.name != metric_name]
    
    def validate(self, 
                 segmentation: np.ndarray,
                 context: Optional[Dict[str, np.ndarray]] = None,
                 cache_results: bool = False) -> Dict[str, Any]:
        """
        Validate segmentation quality using configured metrics.
        
        Args:
            segmentation: 2D array of segment labels
            context: Optional auxiliary data for context-aware metrics
            cache_results: Whether to cache results for later retrieval
            
        Returns:
            Dictionary with metric results and metadata
        """
        # Input validation
        if segmentation.ndim != 2:
            raise ValueError(f"Expected 2D segmentation, got {segmentation.ndim}D")
        
        # Compute basic segmentation statistics
        unique_segments = np.unique(segmentation)
        n_segments = len(unique_segments[unique_segments >= 0])
        
        results = {
            'metadata': {
                'shape': segmentation.shape,
                'n_segments': n_segments,
                'has_context': context is not None,
                'n_metrics': len(self.metrics)
            },
            'metrics': {}
        }
        
        # Compute each metric
        for metric in self.metrics:
            # Skip metrics that need context if none provided
            if metric.requires_context() and context is None:
                results['metrics'][metric.name] = {
                    'status': 'skipped',
                    'reason': 'requires_context'
                }
                continue
            
            # Validate input for this metric
            if not metric.validate_input(segmentation):
                results['metrics'][metric.name] = {
                    'status': 'failed',
                    'reason': 'invalid_input'
                }
                continue
            
            try:
                # Compute metric
                metric_values = metric.compute(segmentation, context)
                results['metrics'][metric.name] = {
                    'status': 'success',
                    'values': metric_values,
                    'description': metric.description
                }
            except Exception as e:
                results['metrics'][metric.name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Cache if requested
        if cache_results:
            cache_key = f"{id(segmentation)}_{id(context)}"
            self._results_cache[cache_key] = results
        
        return results
    
    def get_cached_results(self) -> Dict[str, Any]:
        """Retrieve cached validation results."""
        return self._results_cache.copy()
    
    def clear_cache(self) -> None:
        """Clear cached results."""
        self._results_cache.clear()
    
    def summarize(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Create summary statistics from validation results.
        
        Args:
            results: Output from validate() method
            
        Returns:
            Flattened dictionary of summary values
        """
        summary = {}
        
        # Add metadata
        summary['n_segments'] = results['metadata']['n_segments']
        summary['n_successful_metrics'] = sum(
            1 for m in results['metrics'].values() 
            if m.get('status') == 'success'
        )
        
        # Flatten metric values
        for metric_name, metric_data in results['metrics'].items():
            if metric_data.get('status') == 'success':
                values = metric_data['values']
                if isinstance(values, dict):
                    for key, value in values.items():
                        summary[f"{metric_name}_{key}"] = value
                else:
                    summary[metric_name] = values
        
        return summary


class ValidationFactory:
    """
    Factory for creating configured validators.
    
    This factory manages the creation of validators with appropriate
    metrics based on configuration, maintaining separation between
    framework mechanics and domain-specific knowledge.
    """
    
    @staticmethod
    def create_validator(config: Dict[str, Any]) -> SegmentationValidator:
        """
        Create validator from configuration.
        
        Args:
            config: Configuration dictionary specifying metrics
            
        Returns:
            Configured SegmentationValidator instance
        """
        validator = SegmentationValidator()
        
        # Load metrics specified in config
        metrics_config = config.get('metrics', [])
        
        for metric_spec in metrics_config:
            if isinstance(metric_spec, str):
                # Load by name from registered metrics
                metric = ValidationFactory._load_metric_by_name(metric_spec)
            elif isinstance(metric_spec, dict):
                # Load with parameters
                metric_type = metric_spec.get('type')
                metric_params = metric_spec.get('params', {})
                metric = ValidationFactory._load_metric_by_name(metric_type, **metric_params)
            else:
                raise ValueError(f"Invalid metric specification: {metric_spec}")
            
            if metric:
                validator.add_metric(metric)
        
        return validator
    
    @staticmethod
    def _load_metric_by_name(name: str, **params) -> Optional[Metric]:
        """
        Load a metric by name.
        
        This method will be extended to support dynamic loading
        of metrics from plugins.
        """
        # Import here to avoid circular dependencies
        from .metrics import REGISTERED_METRICS
        
        metric_class = REGISTERED_METRICS.get(name)
        if metric_class:
            return metric_class(**params)
        
        return None
    
    @staticmethod
    def register_metric(name: str, metric_class: type) -> None:
        """
        Register a new metric type.
        
        This allows plugins to add new metrics without modifying core code.
        """
        from .metrics import REGISTERED_METRICS
        REGISTERED_METRICS[name] = metric_class