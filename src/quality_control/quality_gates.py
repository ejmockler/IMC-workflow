"""
Automated Quality Gates for IMC Analysis

Provides practical quality gates and decision rules for determining
analysis continuation, ROI filtering, and systematic issue escalation.
Now uses centralized configuration for all thresholds.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging

from .statistical_monitoring import QualityMetrics, QualityMonitor
from .config import QCConfig, ThresholdConfig


class GateDecision(Enum):
    """Quality gate decision outcomes."""
    PASS = "pass"           # Continue analysis
    WARN = "warn"           # Continue with warning
    FAIL = "fail"           # Skip this ROI
    ABORT = "abort"         # Stop entire analysis


# Legacy QualityThresholds for backward compatibility
# Now delegates to centralized config
class QualityThresholds(ThresholdConfig):
    """Legacy wrapper for ThresholdConfig - use QCConfig.thresholds instead."""
    pass


class QualityGateEngine:
    """
    Automated quality gate engine for IMC analysis.
    
    Provides practical, conservative decision rules without overengineering.
    Focus on catching real problems that would compromise analysis quality.
    Now uses centralized configuration for consistency.
    """
    
    def __init__(self, config: QCConfig = None):
        """Initialize quality gate engine with centralized config.
        
        Args:
            config: Quality control configuration. If None, uses defaults.
        """
        if config is None:
            from .config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        
        self.config = config
        self.thresholds = config.thresholds  # For backward compatibility
        self.logger = logging.getLogger('QualityGates')
        
        # Track decisions for analysis
        self.decisions_history: List[Tuple[str, GateDecision, str]] = []
    
    def evaluate_roi_quality(self, quality_metrics: QualityMetrics) -> Tuple[GateDecision, str, Dict[str, Any]]:
        """
        Evaluate ROI quality and return gate decision.
        
        Returns:
            Tuple of (decision, reason, details)
        """
        issues = []
        warnings = []
        details = {
            'roi_id': quality_metrics.roi_id,
            'overall_quality': quality_metrics.overall_quality(),
            'thresholds_checked': []
        }
        
        # Check individual quality metrics
        checks = [
            ('coordinate_quality', quality_metrics.coordinate_quality, self.thresholds.min_coordinate_quality),
            ('ion_count_quality', quality_metrics.ion_count_quality, self.thresholds.min_ion_count_quality),
            ('biological_quality', quality_metrics.biological_quality, self.thresholds.min_biological_quality),
            ('overall_quality', quality_metrics.overall_quality(), self.thresholds.min_overall_quality)
        ]
        
        for metric_name, value, threshold in checks:
            details['thresholds_checked'].append({
                'metric': metric_name,
                'value': value,
                'threshold': threshold,
                'passed': value >= threshold
            })
            
            if value < threshold:
                if threshold >= 0.5:  # Critical thresholds
                    issues.append(f"{metric_name}: {value:.3f} < {threshold:.3f}")
                else:  # Warning thresholds
                    warnings.append(f"{metric_name}: {value:.3f} < {threshold:.3f}")
        
        # Check technical requirements
        if quality_metrics.n_pixels < self.thresholds.min_pixels:
            issues.append(f"insufficient pixels: {quality_metrics.n_pixels} < {self.thresholds.min_pixels}")
        
        if quality_metrics.n_proteins < self.thresholds.min_proteins:
            issues.append(f"insufficient proteins: {quality_metrics.n_proteins} < {self.thresholds.min_proteins}")
        
        if quality_metrics.protein_completeness < self.thresholds.min_protein_completeness:
            if quality_metrics.protein_completeness < 0.5:
                issues.append(f"poor protein completeness: {quality_metrics.protein_completeness:.2f}")
            else:
                warnings.append(f"moderate protein completeness: {quality_metrics.protein_completeness:.2f}")
        
        # Make decision
        if issues:
            decision = GateDecision.FAIL
            reason = f"Quality gate failure: {'; '.join(issues[:3])}"  # Limit to top 3 issues
        elif warnings:
            decision = GateDecision.WARN
            reason = f"Quality warnings: {'; '.join(warnings[:2])}"  # Limit to top 2 warnings
        else:
            decision = GateDecision.PASS
            reason = "All quality checks passed"
        
        # Log decision
        self.decisions_history.append((quality_metrics.roi_id, decision, reason))
        self.logger.debug(f"ROI {quality_metrics.roi_id}: {decision.value} - {reason}")
        
        return decision, reason, details
    
    def evaluate_batch_quality(self, quality_monitor: QualityMonitor, batch_id: str) -> Tuple[GateDecision, str, Dict[str, Any]]:
        """
        Evaluate batch-level quality consistency.
        
        Returns:
            Tuple of (decision, reason, details)
        """
        batch_analysis = quality_monitor.analyze_batch_consistency(batch_id)
        
        if batch_analysis['status'] == 'insufficient_data':
            return GateDecision.PASS, "Insufficient data for batch evaluation", batch_analysis
        
        details = batch_analysis.copy()
        issues = []
        warnings = []
        
        # Check batch consistency
        cv_quality = batch_analysis['cv_quality']
        if cv_quality > self.thresholds.max_batch_cv:
            if cv_quality > 0.6:  # Very high variation
                issues.append(f"high batch variation: CV={cv_quality:.2f}")
            else:
                warnings.append(f"moderate batch variation: CV={cv_quality:.2f}")
        
        # Check batch size
        n_rois = batch_analysis['n_rois']
        if n_rois < self.thresholds.min_batch_size:
            warnings.append(f"small batch size: {n_rois} ROIs")
        
        # Check overall batch quality
        mean_quality = batch_analysis['mean_quality']
        if mean_quality < self.thresholds.min_overall_quality:
            issues.append(f"low batch quality: {mean_quality:.3f}")
        
        # Make decision
        if issues:
            decision = GateDecision.WARN  # Batch issues are warnings, not failures
            reason = f"Batch quality concerns: {'; '.join(issues)}"
        elif warnings:
            decision = GateDecision.WARN
            reason = f"Batch quality warnings: {'; '.join(warnings)}"
        else:
            decision = GateDecision.PASS
            reason = "Batch quality acceptable"
        
        self.logger.info(f"Batch {batch_id}: {decision.value} - {reason}")
        return decision, reason, details
    
    def evaluate_systematic_trends(self, quality_monitor: QualityMonitor) -> Tuple[GateDecision, str, Dict[str, Any]]:
        """
        Evaluate systematic quality trends across the analysis.
        
        Returns:
            Tuple of (decision, reason, details)
        """
        trend_analysis = quality_monitor.detect_quality_trends(window_size=self.thresholds.min_trend_window)
        
        if trend_analysis['status'] == 'insufficient_data':
            return GateDecision.PASS, "Insufficient data for trend analysis", trend_analysis
        
        details = trend_analysis.copy()
        issues = []
        warnings = []
        
        # Check for declining trends
        if trend_analysis['status'] == 'declining':
            slope = trend_analysis['slope']
            if slope < self.thresholds.max_declining_trend:
                issues.append(f"significant quality decline: slope={slope:.3f}")
            else:
                warnings.append(f"quality declining: slope={slope:.3f}")
        
        # Check recent quality level
        recent_mean = trend_analysis['recent_mean']
        if recent_mean < self.thresholds.min_overall_quality * 0.8:  # 20% below threshold
            issues.append(f"recent quality low: {recent_mean:.3f}")
        
        # Make decision
        if issues:
            decision = GateDecision.ABORT  # Systematic issues are serious
            reason = f"Systematic quality issues detected: {'; '.join(issues)}"
        elif warnings:
            decision = GateDecision.WARN
            reason = f"Quality trend warnings: {'; '.join(warnings)}"
        else:
            decision = GateDecision.PASS
            reason = "Quality trends acceptable"
        
        self.logger.info(f"Trend analysis: {decision.value} - {reason}")
        return decision, reason, details
    
    def should_continue_analysis(self, quality_monitor: QualityMonitor, current_roi_quality: QualityMetrics) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Comprehensive decision on whether to continue analysis.
        
        Returns:
            Tuple of (should_continue, reason, details)
        """
        # Evaluate current ROI
        roi_decision, roi_reason, roi_details = self.evaluate_roi_quality(current_roi_quality)
        
        # Evaluate batch consistency if enough data
        batch_decision, batch_reason, batch_details = self.evaluate_batch_quality(
            quality_monitor, current_roi_quality.batch_id
        )
        
        # Evaluate systematic trends if enough data
        trend_decision, trend_reason, trend_details = self.evaluate_systematic_trends(quality_monitor)
        
        # Combine decisions
        all_decisions = [roi_decision, batch_decision, trend_decision]
        
        # Abort if any component says abort
        if GateDecision.ABORT in all_decisions:
            return False, f"Analysis abort: {trend_reason}", {
                'roi': roi_details,
                'batch': batch_details,
                'trend': trend_details,
                'final_decision': 'abort'
            }
        
        # Continue but note if ROI failed
        roi_failed = roi_decision == GateDecision.FAIL
        has_warnings = GateDecision.WARN in all_decisions
        
        if roi_failed:
            reason = f"Continue analysis but skip ROI: {roi_reason}"
        elif has_warnings:
            reason = f"Continue with warnings: {roi_reason if roi_decision == GateDecision.WARN else batch_reason}"
        else:
            reason = "All quality checks passed"
        
        return True, reason, {
            'roi': roi_details,
            'batch': batch_details, 
            'trend': trend_details,
            'final_decision': 'continue',
            'skip_current_roi': roi_failed
        }
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of all quality gate decisions."""
        if not self.decisions_history:
            return {'status': 'no_decisions'}
        
        decision_counts = {}
        for decision in GateDecision:
            decision_counts[decision.value] = sum(1 for _, d, _ in self.decisions_history if d == decision)
        
        total_decisions = len(self.decisions_history)
        pass_rate = decision_counts.get('pass', 0) / total_decisions if total_decisions > 0 else 0
        
        return {
            'total_decisions': total_decisions,
            'decision_distribution': decision_counts,
            'pass_rate': pass_rate,
            'recent_decisions': [(roi_id, decision.value, reason) for roi_id, decision, reason in self.decisions_history[-10:]]
        }


# Convenience functions for integration
def create_quality_gate_engine(
    min_overall_quality: float = 0.5,
    min_coordinate_quality: float = 0.3,
    min_ion_count_quality: float = 0.4,
    **kwargs
) -> QualityGateEngine:
    """Create quality gate engine with custom thresholds."""
    thresholds = QualityThresholds(
        min_overall_quality=min_overall_quality,
        min_coordinate_quality=min_coordinate_quality,
        min_ion_count_quality=min_ion_count_quality,
        **kwargs
    )
    return QualityGateEngine(thresholds)


def evaluate_roi_for_analysis(
    quality_metrics: QualityMetrics,
    quality_monitor: QualityMonitor,
    gate_engine: QualityGateEngine = None
) -> Dict[str, Any]:
    """
    Simple interface for evaluating ROI quality.
    
    Returns:
        Dictionary with decision and details
    """
    if gate_engine is None:
        gate_engine = create_quality_gate_engine()
    
    should_continue, reason, details = gate_engine.should_continue_analysis(quality_monitor, quality_metrics)
    
    return {
        'continue_analysis': should_continue,
        'skip_current_roi': details.get('skip_current_roi', False),
        'reason': reason,
        'roi_quality': details['roi']['overall_quality'],
        'decision_details': details
    }