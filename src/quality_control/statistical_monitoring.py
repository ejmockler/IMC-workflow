"""
Statistical Quality Monitoring for IMC Analysis

Provides essential statistical process control for tracking quality trends,
detecting systematic issues, and ensuring batch consistency without overengineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import json


@dataclass
class QualityMetrics:
    """Core quality metrics for monitoring with comprehensive validation."""
    roi_id: str
    batch_id: str
    timestamp: str
    
    # Data quality
    coordinate_quality: float
    ion_count_quality: float
    biological_quality: float
    
    # Analysis quality (if available)
    clustering_quality: Optional[float] = None
    segmentation_quality: Optional[float] = None
    
    # Technical metrics
    n_pixels: int = 0
    n_proteins: int = 0
    protein_completeness: float = 0.0
    
    def __post_init__(self):
        """Validate all metrics after initialization."""
        # Validate string fields
        if not self.roi_id or not isinstance(self.roi_id, str):
            raise ValueError(f"Invalid roi_id: {self.roi_id}")
        if not self.batch_id or not isinstance(self.batch_id, str):
            raise ValueError(f"Invalid batch_id: {self.batch_id}")
        
        # Sanitize ROI ID to prevent injection
        if not self.roi_id.replace('_', '').replace('-', '').replace('.', '').isalnum():
            raise ValueError(f"ROI ID contains invalid characters: {self.roi_id}")
        
        # Validate quality scores [0,1]
        for metric_name, value in [
            ('coordinate_quality', self.coordinate_quality),
            ('ion_count_quality', self.ion_count_quality),
            ('biological_quality', self.biological_quality),
            ('protein_completeness', self.protein_completeness)
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{metric_name}={value} outside valid range [0,1]")
        
        # Validate optional quality scores
        for metric_name, value in [
            ('clustering_quality', self.clustering_quality),
            ('segmentation_quality', self.segmentation_quality)
        ]:
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"{metric_name}={value} outside valid range [0,1]")
        
        # Validate technical metrics
        if self.n_pixels < 0:
            raise ValueError(f"n_pixels cannot be negative: {self.n_pixels}")
        if self.n_proteins < 0:
            raise ValueError(f"n_proteins cannot be negative: {self.n_proteins}")
    
    def overall_quality(self) -> float:
        """Calculate overall quality score with validation."""
        base_metrics = [self.coordinate_quality, self.ion_count_quality, self.biological_quality]
        
        # Add analysis metrics if available
        analysis_metrics = []
        if self.clustering_quality is not None:
            analysis_metrics.append(self.clustering_quality)
        if self.segmentation_quality is not None:
            analysis_metrics.append(self.segmentation_quality)
        
        # Weight base metrics more heavily
        all_metrics = base_metrics + [m * 0.5 for m in analysis_metrics]
        overall = float(np.mean(all_metrics))
        
        # Ensure result is in valid range
        return max(0.0, min(1.0, overall))


@dataclass
class QualityLimits:
    """Statistical control limits for quality monitoring with directionality support."""
    metric_name: str
    center_line: float
    upper_control_limit: float
    lower_control_limit: float
    upper_warning_limit: float
    lower_warning_limit: float
    directionality: str = "lower_only"  # "lower_only", "upper_only", or "bilateral"
    
    def evaluate(self, value: float) -> str:
        """Evaluate a value against control limits considering metric directionality.
        
        For quality metrics (0-1 scale), typically only low values are problematic.
        High values indicate good quality and should not trigger warnings.
        """
        # For metrics where only low values are bad (most quality metrics)
        if self.directionality == "lower_only":
            if value < self.lower_control_limit:
                return "out_of_control"
            elif value < self.lower_warning_limit:
                return "warning"
            else:
                return "in_control"
        
        # For metrics where only high values are bad (e.g., error rates)
        elif self.directionality == "upper_only":
            if value > self.upper_control_limit:
                return "out_of_control"
            elif value > self.upper_warning_limit:
                return "warning"
            else:
                return "in_control"
        
        # For metrics where both extremes are bad (traditional SPC)
        else:  # bilateral
            if value > self.upper_control_limit or value < self.lower_control_limit:
                return "out_of_control"
            elif value > self.upper_warning_limit or value < self.lower_warning_limit:
                return "warning"
            else:
                return "in_control"


class QualityMonitor:
    """
    Statistical quality monitoring for IMC analysis.
    
    Tracks quality trends, detects systematic issues, and provides
    actionable feedback without complex statistical machinery.
    """
    
    def __init__(self, history_file: Optional[str] = None):
        """Initialize quality monitor."""
        self.logger = logging.getLogger('QualityMonitor')
        self.history_file = Path(history_file) if history_file else None
        self.quality_history: List[QualityMetrics] = []
        self.control_limits: Dict[str, QualityLimits] = {}
        
        # Load existing history if available
        if self.history_file and self.history_file.exists():
            self._load_history()
    
    def add_roi_quality(self, quality_metrics: QualityMetrics) -> None:
        """Add quality metrics for a single ROI."""
        self.quality_history.append(quality_metrics)
        self.logger.debug(f"Added quality metrics for {quality_metrics.roi_id}")
    
    def update_control_limits(self, min_samples: int = 10) -> None:
        """Update statistical control limits based on recent history."""
        if len(self.quality_history) < min_samples:
            self.logger.info(f"Insufficient history for control limits ({len(self.quality_history)} < {min_samples})")
            return
        
        # Extract key metrics
        metrics = {
            'overall_quality': [qm.overall_quality() for qm in self.quality_history],
            'coordinate_quality': [qm.coordinate_quality for qm in self.quality_history],
            'ion_count_quality': [qm.ion_count_quality for qm in self.quality_history],
            'biological_quality': [qm.biological_quality for qm in self.quality_history],
            'protein_completeness': [qm.protein_completeness for qm in self.quality_history]
        }
        
        # Define metric directionality - quality metrics are "lower_only" 
        # (high values are good, only low values indicate problems)
        metric_directionality = {
            'overall_quality': 'lower_only',
            'coordinate_quality': 'lower_only', 
            'ion_count_quality': 'lower_only',
            'biological_quality': 'lower_only',
            'protein_completeness': 'lower_only'
        }
        
        # Calculate control limits for each metric
        for metric_name, values in metrics.items():
            if values and len(values) >= min_samples:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Handle constant values (zero variance)
                if std_val == 0:
                    std_val = 0.01  # Small epsilon to avoid division issues
                
                # 3-sigma control limits (99.7% of normal distribution)
                # Clamp to [0,1] for quality metrics
                ucl = min(1.0, mean_val + 3 * std_val)
                lcl = max(0.0, mean_val - 3 * std_val)
                
                # 2-sigma warning limits (95% of normal distribution)
                # Clamp to [0,1] for quality metrics
                uwl = min(1.0, mean_val + 2 * std_val)
                lwl = max(0.0, mean_val - 2 * std_val)
                
                self.control_limits[metric_name] = QualityLimits(
                    metric_name=metric_name,
                    center_line=mean_val,
                    upper_control_limit=ucl,
                    lower_control_limit=lcl,
                    upper_warning_limit=uwl,
                    lower_warning_limit=lwl,
                    directionality=metric_directionality.get(metric_name, 'lower_only')
                )
        
        self.logger.info(f"Updated control limits for {len(self.control_limits)} metrics")
    
    def check_quality_status(self, quality_metrics: QualityMetrics) -> Dict[str, Any]:
        """Check quality status against control limits."""
        status = {
            'roi_id': quality_metrics.roi_id,
            'overall_status': 'in_control',
            'alerts': [],
            'warnings': [],
            'metric_evaluations': {}
        }
        
        # Evaluate against control limits
        test_metrics = {
            'overall_quality': quality_metrics.overall_quality(),
            'coordinate_quality': quality_metrics.coordinate_quality,
            'ion_count_quality': quality_metrics.ion_count_quality,
            'biological_quality': quality_metrics.biological_quality,
            'protein_completeness': quality_metrics.protein_completeness
        }
        
        for metric_name, value in test_metrics.items():
            if metric_name in self.control_limits:
                limits = self.control_limits[metric_name]
                evaluation = limits.evaluate(value)
                status['metric_evaluations'][metric_name] = {
                    'value': value,
                    'status': evaluation,
                    'center_line': limits.center_line
                }
                
                if evaluation == 'out_of_control':
                    status['overall_status'] = 'out_of_control'
                    status['alerts'].append(f"{metric_name}: {value:.3f} (expected: {limits.center_line:.3f})")
                elif evaluation == 'warning' and status['overall_status'] == 'in_control':
                    status['overall_status'] = 'warning'
                    status['warnings'].append(f"{metric_name}: {value:.3f} (center: {limits.center_line:.3f})")
        
        return status
    
    def analyze_batch_consistency(self, batch_id: str) -> Dict[str, Any]:
        """Analyze quality consistency within a batch."""
        batch_metrics = [qm for qm in self.quality_history if qm.batch_id == batch_id]
        
        if len(batch_metrics) < 2:
            return {
                'batch_id': batch_id,
                'status': 'insufficient_data',
                'n_rois': len(batch_metrics)
            }
        
        # Calculate consistency metrics
        overall_qualities = [qm.overall_quality() for qm in batch_metrics]
        mean_quality = np.mean(overall_qualities)
        std_quality = np.std(overall_qualities)
        
        # Handle division by zero - use infinity for undefined CV
        if mean_quality > 0:
            cv_quality = std_quality / mean_quality
        else:
            cv_quality = float('inf')  # Undefined CV when mean is zero
            self.logger.warning(f"Batch {batch_id} has zero mean quality - CV undefined")
        
        # Determine consistency status
        if cv_quality < 0.1:  # CV < 10%
            consistency_status = 'excellent'
        elif cv_quality < 0.2:  # CV < 20%
            consistency_status = 'good'
        elif cv_quality < 0.3:  # CV < 30%
            consistency_status = 'acceptable'
        else:
            consistency_status = 'poor'
        
        return {
            'batch_id': batch_id,
            'status': consistency_status,
            'n_rois': len(batch_metrics),
            'mean_quality': mean_quality,
            'cv_quality': cv_quality,
            'min_quality': np.min(overall_qualities),
            'max_quality': np.max(overall_qualities)
        }
    
    def detect_quality_trends(self, window_size: int = 10) -> Dict[str, Any]:
        """Detect trends in quality metrics over recent ROIs."""
        if len(self.quality_history) < window_size:
            return {'status': 'insufficient_data'}
        
        # Get recent metrics
        recent_metrics = self.quality_history[-window_size:]
        overall_qualities = [qm.overall_quality() for qm in recent_metrics]
        
        # Robust trend detection using linear regression
        x = np.arange(len(overall_qualities))
        y = np.array(overall_qualities)
        
        # Calculate slope using robust polyfit to avoid NaN issues
        if len(y) > 1 and np.std(y) > 0:
            try:
                # Use polyfit for robust linear regression
                coefficients = np.polyfit(x, y, 1)
                slope = float(coefficients[0])
            except (np.linalg.LinAlgError, ValueError):
                # If polyfit fails, assume no trend
                slope = 0.0
        else:
            # Constant values have no trend
            slope = 0.0
        
        # Ensure slope is finite
        if not np.isfinite(slope):
            slope = 0.0
            self.logger.warning(f"Non-finite slope detected in trend analysis, defaulting to 0.0")
        
        # Determine trend status with clear thresholds
        if abs(slope) < 0.01:
            trend_status = 'stable'
        elif slope > 0.01:
            trend_status = 'improving'
        elif slope < -0.01:
            trend_status = 'declining'
        else:
            trend_status = 'stable'  # Default to stable for edge cases
        
        return {
            'status': trend_status,
            'slope': slope,
            'recent_mean': np.mean(overall_qualities),
            'window_size': window_size,
            'quality_range': [np.min(overall_qualities), np.max(overall_qualities)]
        }
    
    def generate_quality_summary(self) -> Dict[str, Any]:
        """Generate comprehensive quality summary."""
        if not self.quality_history:
            return {'status': 'no_data'}
        
        # Overall statistics
        overall_qualities = [qm.overall_quality() for qm in self.quality_history]
        
        # Batch-level analysis
        unique_batches = list(set(qm.batch_id for qm in self.quality_history))
        batch_analyses = [self.analyze_batch_consistency(batch_id) for batch_id in unique_batches]
        
        # Recent trend analysis
        trend_analysis = self.detect_quality_trends()
        
        # Quality distribution
        quality_bins = np.histogram(overall_qualities, bins=[0, 0.5, 0.7, 0.85, 1.0])[0]
        
        summary = {
            'total_rois': len(self.quality_history),
            'unique_batches': len(unique_batches),
            'overall_quality_stats': {
                'mean': np.mean(overall_qualities),
                'std': np.std(overall_qualities),
                'min': np.min(overall_qualities),
                'max': np.max(overall_qualities),
                'median': np.median(overall_qualities)
            },
            'quality_distribution': {
                'poor_(<0.5)': int(quality_bins[0]),
                'fair_(0.5-0.7)': int(quality_bins[1]),
                'good_(0.7-0.85)': int(quality_bins[2]),
                'excellent_(>0.85)': int(quality_bins[3])
            },
            'batch_consistency': batch_analyses,
            'trend_analysis': trend_analysis,
            'control_limits_available': len(self.control_limits) > 0
        }
        
        return summary
    
    def save_history(self) -> None:
        """Save quality history to file using atomic write to prevent corruption."""
        if not self.history_file:
            return
        
        # Convert to serializable format
        history_data = []
        for qm in self.quality_history:
            history_data.append({
                'roi_id': qm.roi_id,
                'batch_id': qm.batch_id,
                'timestamp': qm.timestamp,
                'coordinate_quality': qm.coordinate_quality,
                'ion_count_quality': qm.ion_count_quality,
                'biological_quality': qm.biological_quality,
                'clustering_quality': qm.clustering_quality,
                'segmentation_quality': qm.segmentation_quality,
                'n_pixels': qm.n_pixels,
                'n_proteins': qm.n_proteins,
                'protein_completeness': qm.protein_completeness
            })
        
        # Save with control limits - include directionality
        save_data = {
            'version': '2.0',  # Schema version for future migration
            'quality_history': history_data,
            'control_limits': {
                name: {
                    'metric_name': limits.metric_name,
                    'center_line': limits.center_line,
                    'upper_control_limit': limits.upper_control_limit,
                    'lower_control_limit': limits.lower_control_limit,
                    'upper_warning_limit': limits.upper_warning_limit,
                    'lower_warning_limit': limits.lower_warning_limit,
                    'directionality': limits.directionality
                } for name, limits in self.control_limits.items()
            }
        }
        
        # Atomic write: write to temp file then rename
        temp_file = self.history_file.with_suffix('.tmp')
        try:
            # Write to temporary file
            with open(temp_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            # Atomic rename (replaces existing file atomically)
            temp_file.replace(self.history_file)
            self.logger.info(f"Saved quality history to {self.history_file} (atomic)")
            
        except Exception as e:
            # Clean up temp file if write failed
            if temp_file.exists():
                temp_file.unlink()
            self.logger.error(f"Failed to save quality history: {e}")
            raise
    
    def _load_history(self) -> None:
        """Load quality history from file with schema validation."""
        try:
            with open(self.history_file, 'r') as f:
                save_data = json.load(f)
            
            # Check schema version
            version = save_data.get('version', '1.0')
            if version not in ['1.0', '2.0']:
                self.logger.warning(f"Unknown schema version {version}, attempting best-effort load")
            
            # Validate and load quality history
            for item in save_data.get('quality_history', []):
                # Validate required fields
                if not all(k in item for k in ['roi_id', 'batch_id', 'timestamp']):
                    self.logger.warning(f"Skipping invalid history item: missing required fields")
                    continue
                
                # Validate quality scores are in range [0,1]
                for metric in ['coordinate_quality', 'ion_count_quality', 'biological_quality']:
                    if metric in item:
                        value = item[metric]
                        if not (0 <= value <= 1):
                            self.logger.warning(f"Invalid {metric}={value} for {item['roi_id']}, clamping")
                            item[metric] = max(0.0, min(1.0, value))
                
                qm = QualityMetrics(
                    roi_id=str(item['roi_id']),  # Ensure string
                    batch_id=str(item['batch_id']),  # Ensure string
                    timestamp=str(item['timestamp']),  # Ensure string
                    coordinate_quality=float(item.get('coordinate_quality', 0.0)),
                    ion_count_quality=float(item.get('ion_count_quality', 0.0)),
                    biological_quality=float(item.get('biological_quality', 0.0)),
                    clustering_quality=item.get('clustering_quality'),
                    segmentation_quality=item.get('segmentation_quality'),
                    n_pixels=max(0, int(item.get('n_pixels', 0))),  # Ensure non-negative
                    n_proteins=max(0, int(item.get('n_proteins', 0))),  # Ensure non-negative
                    protein_completeness=max(0.0, min(1.0, float(item.get('protein_completeness', 0.0))))
                )
                self.quality_history.append(qm)
            
            # Load control limits with backward compatibility
            for name, limits_data in save_data.get('control_limits', {}).items():
                # Add default directionality if missing (backward compatibility)
                if 'directionality' not in limits_data:
                    limits_data['directionality'] = 'lower_only'
                
                self.control_limits[name] = QualityLimits(**limits_data)
            
            self.logger.info(f"Loaded {len(self.quality_history)} quality records from {self.history_file} (v{version})")
            
        except FileNotFoundError:
            self.logger.info(f"No history file found at {self.history_file}, starting fresh")
        except json.JSONDecodeError as e:
            self.logger.error(f"Corrupt history file: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to load quality history: {str(e)}")


def extract_quality_metrics_from_validation(validation_result, roi_id: str, batch_id: str) -> QualityMetrics:
    """Extract quality metrics from validation result.
    
    CRITICAL: Uses None for missing metrics instead of optimistic defaults.
    This ensures failures are not masked as passing scores.
    """
    import datetime
    
    # Use None for missing metrics - fail-safe approach
    coordinate_quality = None
    ion_count_quality = None
    biological_quality = None
    
    # Technical metrics - conservative defaults
    n_pixels = 0
    n_proteins = 0
    protein_completeness = 0.0  # Conservative: assume incomplete until proven
    
    # Parse validation results for actual metrics
    if hasattr(validation_result, 'results'):
        for result in validation_result.results:
            if result.rule_name == 'coordinate_validation':
                # Only use actual score, never default to passing value
                if result.quality_score is not None:
                    coordinate_quality = float(result.quality_score)
                else:
                    coordinate_quality = 0.0  # Missing = fail
                    
                if hasattr(result, 'metrics') and 'n_points' in result.metrics:
                    n_pixels = result.metrics['n_points'].value
                    
            elif result.rule_name == 'ion_count_validation':
                # Only use actual score, never default to passing value
                if result.quality_score is not None:
                    ion_count_quality = float(result.quality_score)
                else:
                    ion_count_quality = 0.0  # Missing = fail
                    
                if hasattr(result, 'metrics'):
                    if 'n_proteins' in result.metrics:
                        n_proteins = result.metrics['n_proteins'].value
                    if 'data_completeness' in result.metrics:
                        protein_completeness = result.metrics['data_completeness'].value
                    
            elif result.rule_name == 'biological_validation':
                # Only use actual score, never default to passing value
                if result.quality_score is not None:
                    biological_quality = float(result.quality_score)
                else:
                    biological_quality = 0.0  # Missing = fail
    
    # If no validation results, use explicit failure values
    if coordinate_quality is None:
        coordinate_quality = 0.0
    if ion_count_quality is None:
        ion_count_quality = 0.0
    if biological_quality is None:
        biological_quality = 0.0
    
    return QualityMetrics(
        roi_id=roi_id,
        batch_id=batch_id,
        timestamp=datetime.datetime.now().isoformat(),
        coordinate_quality=coordinate_quality,
        ion_count_quality=ion_count_quality,
        biological_quality=biological_quality,
        n_pixels=n_pixels,
        n_proteins=n_proteins,
        protein_completeness=protein_completeness
    )