"""
Centralized Configuration for Quality Control System

All thresholds, magic numbers, and configuration parameters are defined here
to ensure consistency across the quality control modules.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ThresholdConfig:
    """Quality thresholds for decision making - rebalanced for scientific rigor."""
    # Individual metric thresholds - aligned with validation framework
    min_coordinate_quality: float = 0.15  # Only truly corrupted data flagged as critical
    min_ion_count_quality: float = 0.15   # More permissive for real-world data
    min_biological_quality: float = 0.15  # Aligned with BiologicalValidator critical threshold
    min_overall_quality: float = 0.25     # Balanced overall threshold
    
    # Technical thresholds
    min_pixels: int = 100
    min_proteins: int = 3
    min_protein_completeness: float = 0.7
    
    # Batch consistency thresholds
    max_batch_cv: float = 0.4
    min_batch_size: int = 2
    
    # Trend detection thresholds
    max_declining_trend: float = -0.05
    min_trend_window: int = 5
    trend_stable_threshold: float = 0.01
    
    # Control limit parameters
    control_limit_sigma: float = 3.0  # 99.7% confidence
    warning_limit_sigma: float = 2.0  # 95% confidence
    min_samples_for_limits: int = 10
    
    # Batch consistency CV thresholds
    cv_excellent: float = 0.1  # CV < 10%
    cv_good: float = 0.2      # CV < 20%
    cv_acceptable: float = 0.3  # CV < 30%


@dataclass 
class MetricDirectionality:
    """Define directionality for each metric type."""
    # Map metric names to their directionality
    # "lower_only": High values good, only low values problematic (quality metrics)
    # "upper_only": Low values good, only high values problematic (error metrics)
    # "bilateral": Both extremes problematic (traditional SPC)
    
    METRIC_DIRECTIONS = {
        'overall_quality': 'lower_only',
        'coordinate_quality': 'lower_only',
        'ion_count_quality': 'lower_only',
        'biological_quality': 'lower_only',
        'protein_completeness': 'lower_only',
        'clustering_quality': 'lower_only',
        'segmentation_quality': 'lower_only',
        # Future error/noise metrics would be 'upper_only'
    }
    
    @classmethod
    def get_direction(cls, metric_name: str) -> str:
        """Get directionality for a metric, defaulting to lower_only for safety."""
        return cls.METRIC_DIRECTIONS.get(metric_name, 'lower_only')


@dataclass
class SPCConfig:
    """Statistical Process Control configuration."""
    # CUSUM parameters
    cusum_h: float = 4.0  # Decision interval
    cusum_k: float = 0.5  # Reference value (shift to detect)
    
    # EWMA parameters
    ewma_lambda: float = 0.2  # Smoothing parameter (0-1)
    ewma_L: float = 3.0  # Control limit multiplier
    
    # Western Electric rules
    enable_we_rules: bool = True
    we_rule_1_points: int = 1  # Points beyond 3σ
    we_rule_2_points: int = 2  # Consecutive points beyond 2σ
    we_rule_3_points: int = 4  # Consecutive points beyond 1σ
    we_rule_4_points: int = 8  # Consecutive points on same side of center
    
    # Trend detection
    min_points_for_trend: int = 7  # Minimum consecutive points for trend


@dataclass
class ReportConfig:
    """Reporting and visualization configuration."""
    # Plot parameters
    max_points_in_plot: int = 1000  # Downsample if more points
    figure_dpi: int = 150
    figure_width: float = 12.0
    figure_height: float = 8.0
    
    # Privacy settings
    anonymize_roi_ids: bool = False
    hash_salt: str = "imc_qc_2024"  # Salt for ROI ID hashing
    
    # Alert thresholds
    max_alerts_per_report: int = 10
    max_warnings_per_report: int = 20
    
    # Summary settings
    recent_decisions_count: int = 10
    quality_bin_edges: list = None  # Set in __post_init__
    
    def __post_init__(self):
        if self.quality_bin_edges is None:
            self.quality_bin_edges = [0.0, 0.5, 0.7, 0.85, 1.0]


@dataclass
class QCConfig:
    """Master configuration for the entire quality control system."""
    thresholds: ThresholdConfig = None
    spc: SPCConfig = None
    reporting: ReportConfig = None
    
    def __post_init__(self):
        """Initialize sub-configs with defaults if not provided."""
        if self.thresholds is None:
            self.thresholds = ThresholdConfig()
        if self.spc is None:
            self.spc = SPCConfig()
        if self.reporting is None:
            self.reporting = ReportConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'QCConfig':
        """Create configuration from dictionary."""
        thresholds = ThresholdConfig(**config_dict.get('thresholds', {}))
        spc = SPCConfig(**config_dict.get('spc', {}))
        reporting = ReportConfig(**config_dict.get('reporting', {}))
        return cls(thresholds=thresholds, spc=spc, reporting=reporting)
    
    def to_dict(self) -> Dict:
        """Export configuration to dictionary."""
        return {
            'thresholds': self.thresholds.__dict__,
            'spc': self.spc.__dict__,
            'reporting': self.reporting.__dict__
        }


# Global default configuration
DEFAULT_CONFIG = QCConfig()


def load_config_from_file(filepath: str) -> QCConfig:
    """Load configuration from JSON file."""
    import json
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return QCConfig.from_dict(config_dict)


def save_config_to_file(config: QCConfig, filepath: str) -> None:
    """Save configuration to JSON file."""
    import json
    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)