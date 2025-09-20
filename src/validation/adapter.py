"""
Adapter Layer for Legacy Code

This module bridges the old segmentation_validation.py functions
with the new pure validation framework, allowing gradual migration.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from .core import SegmentationValidator, ValidationFactory
from .core.metrics import REGISTERED_METRICS


class LegacyAdapter:
    """
    Adapter to use old validation functions with new framework.
    
    This allows existing code to continue working while we migrate
    to the new architecture.
    """
    
    def __init__(self):
        """Initialize with core validator."""
        self.validator = SegmentationValidator()
        self._setup_default_metrics()
    
    def _setup_default_metrics(self):
        """Add default metrics matching old validation."""
        # Add core metrics
        for metric_name in ['compactness', 'size_uniformity', 
                           'boundary_regularity', 'spatial_autocorrelation']:
            if metric_name in REGISTERED_METRICS:
                metric_class = REGISTERED_METRICS[metric_name]
                self.validator.add_metric(metric_class())
    
    def validate_segmentation_quality(self,
                                     segments: np.ndarray,
                                     reference_channels: Dict[str, np.ndarray],
                                     scale_um: float,
                                     metrics: List[str] = None) -> Dict[str, float]:
        """
        Legacy interface for segmentation validation.
        
        Maintains backward compatibility with old API while using
        new framework internally.
        
        Args:
            segments: 2D array of segment labels
            reference_channels: Dict of reference images
            scale_um: Spatial scale (ignored in new framework)
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of validation metrics
        """
        # Run validation with new framework
        results = self.validator.validate(segments, context=reference_channels)
        
        # Flatten results to match old format
        flat_results = {}
        
        # Add metadata
        flat_results['scale_um'] = scale_um
        flat_results['n_segments'] = results['metadata']['n_segments']
        
        # Flatten metric results
        for metric_name, metric_data in results['metrics'].items():
            if metric_data.get('status') == 'success':
                values = metric_data['values']
                if isinstance(values, dict):
                    # Add individual values
                    for key, value in values.items():
                        if key == 'mean':  # Old format used base name for mean
                            flat_results[metric_name] = value
                        else:
                            flat_results[f'{metric_name}_{key}'] = value
                else:
                    flat_results[metric_name] = values
        
        # Add signal adherence if reference channels provided
        if reference_channels and 'signal_adherence' not in flat_results:
            from .core.metrics import SignalAdherence
            sig_metric = SignalAdherence()
            sig_results = sig_metric.compute(segments, reference_channels)
            flat_results['boundary_adherence_mean'] = sig_results.get('adherence', 0.0)
        
        return flat_results
    
    def validate_biological_correspondence(self,
                                          segments: np.ndarray,
                                          marker_data: Dict[str, np.ndarray],
                                          known_structures: Dict[str, Any],
                                          scale_um: float) -> Dict[str, float]:
        """
        Legacy biological validation interface.
        
        This should NOT be used in new code. It exists only for
        backward compatibility. Use experiment-specific validators instead.
        
        Args:
            segments: Segment labels
            marker_data: Protein intensity data
            known_structures: Expected biological structures
            scale_um: Spatial scale
            
        Returns:
            Biological validation metrics
        """
        # Log warning about deprecated usage
        import warnings
        warnings.warn(
            "validate_biological_correspondence is deprecated. "
            "Use experiment-specific validators from src/experiments/",
            DeprecationWarning
        )
        
        # Return empty results - biological validation moved to plugins
        return {
            'warning': 'biological_validation_deprecated',
            'scale_um': scale_um
        }


# Create global adapter instance for backward compatibility
_legacy_adapter = LegacyAdapter()


def validate_segmentation_quality(segments: np.ndarray,
                                 reference_channels: Dict[str, np.ndarray],
                                 scale_um: float,
                                 metrics: List[str] = None) -> Dict[str, float]:
    """
    Legacy function for backward compatibility.
    
    Deprecated: Use SegmentationValidator directly.
    """
    return _legacy_adapter.validate_segmentation_quality(
        segments, reference_channels, scale_um, metrics
    )


def validate_biological_correspondence(segments: np.ndarray,
                                      marker_data: Dict[str, np.ndarray],
                                      known_structures: Dict[str, Any],
                                      scale_um: float) -> Dict[str, float]:
    """
    Legacy function for backward compatibility.
    
    Deprecated: Use experiment-specific validators.
    """
    return _legacy_adapter.validate_biological_correspondence(
        segments, marker_data, known_structures, scale_um
    )