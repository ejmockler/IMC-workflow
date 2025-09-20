"""
Core Validation Framework

Pure segmentation validation without any domain knowledge.
Works for satellite imagery, medical scans, microscopy, or any segmented images.
"""

from .base import (
    Metric,
    SegmentationValidator,
    ValidationFactory
)

from .metrics import (
    GeometricCompactness,
    SizeUniformity,
    BoundaryRegularity,
    SpatialAutocorrelation,
    SignalAdherence,
    REGISTERED_METRICS
)

__all__ = [
    # Base classes
    'Metric',
    'SegmentationValidator', 
    'ValidationFactory',
    
    # Metrics
    'GeometricCompactness',
    'SizeUniformity',
    'BoundaryRegularity',
    'SpatialAutocorrelation',
    'SignalAdherence',
    'REGISTERED_METRICS'
]

__version__ = '2.0.0'  # Major version bump for breaking changes