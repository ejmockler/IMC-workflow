"""
Quality Control System for IMC Analysis

Provides statistical monitoring and automated quality gates for ensuring
analysis reliability.

Usage:
    from src.quality_control import QualityMonitor, QualityGateEngine

    # Initialize quality monitoring
    monitor = QualityMonitor("quality_history.json")
    gate_engine = QualityGateEngine()

    # Add quality metrics for each ROI
    for validation_result in roi_validations:
        quality_metrics = extract_quality_metrics_from_validation(
            validation_result, roi_id, batch_id
        )
        monitor.add_roi_quality(quality_metrics)

        # Check quality gates
        decision = gate_engine.evaluate_roi_quality(quality_metrics)
        if decision[0] == GateDecision.FAIL:
            print(f"ROI {roi_id} failed quality gates: {decision[1]}")

    # Update control limits periodically
    monitor.update_control_limits()
"""

from .statistical_monitoring import (
    QualityMetrics,
    QualityLimits,
    QualityMonitor,
    extract_quality_metrics_from_validation
)

from .quality_gates import (
    GateDecision,
    QualityThresholds,
    QualityGateEngine,
    create_quality_gate_engine,
    evaluate_roi_for_analysis
)

from .config import QCConfig, ThresholdConfig, SPCConfig, DEFAULT_CONFIG

__all__ = [
    # Configuration
    'QCConfig',
    'ThresholdConfig',
    'SPCConfig',
    'DEFAULT_CONFIG',

    # Statistical monitoring
    'QualityMetrics',
    'QualityLimits',
    'QualityMonitor',
    'extract_quality_metrics_from_validation',

    # Quality gates
    'GateDecision',
    'QualityThresholds',
    'QualityGateEngine',
    'create_quality_gate_engine',
    'evaluate_roi_for_analysis',
]

# Version information
__version__ = "1.0.0"
__author__ = "IMC Analysis Pipeline"
__description__ = "Quality control system for IMC analysis"
