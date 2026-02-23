"""
Centralized configuration for IMC Quality Control.

Provides ThresholdConfig, QCConfig, SPCConfig, and sensible defaults.
Consumed by quality_gates.py and statistical_monitoring.py.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ThresholdConfig:
    """Threshold configuration for quality gate decisions."""

    # Per-ROI quality score thresholds (0-1 scale)
    min_coordinate_quality: float = 0.3
    min_ion_count_quality: float = 0.4
    min_biological_quality: float = 0.3
    min_overall_quality: float = 0.5

    # Technical minimums
    min_pixels: int = 1000
    min_proteins: int = 3
    min_protein_completeness: float = 0.5

    # Batch-level thresholds
    max_batch_cv: float = 0.3
    min_batch_size: int = 2

    # Trend detection
    min_trend_window: int = 5
    max_declining_trend: float = -0.05


@dataclass
class SPCConfig:
    """Statistical process control configuration for QualityMonitor."""

    min_roi_per_batch: int = 2
    control_limit_sigma: float = 3.0
    warning_limit_sigma: float = 2.0
    min_samples_for_limits: int = 10


@dataclass
class QCConfig:
    """Top-level QC configuration aggregating all sub-configs."""

    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    spc: SPCConfig = field(default_factory=SPCConfig)


# Module-level default used when no config is provided
DEFAULT_CONFIG = QCConfig()
