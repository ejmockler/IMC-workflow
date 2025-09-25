"""
Quality Control System for IMC Analysis

Provides statistical monitoring, automated quality gates, and essential reporting
for ensuring analysis reliability without overengineering.

Usage:
    from src.quality_control import QualityMonitor, QualityGateEngine, generate_quality_reports
    
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
    
    # Generate comprehensive reports
    reports = generate_quality_reports(monitor, gate_engine)
    print(f"Generated reports: {list(reports.keys())}")
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

from .reporting import (
    QualityReporter,
    generate_quality_reports
)

__all__ = [
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
    
    # Reporting
    'QualityReporter',
    'generate_quality_reports'
]

# Version information
__version__ = "1.0.0"
__author__ = "IMC Analysis Pipeline"
__description__ = "Quality control system for IMC analysis"