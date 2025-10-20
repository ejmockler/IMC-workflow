"""
Lightweight Parameter Deviation Workflow - KISS Implementation

Simple, fast parameter changes with automatic documentation.
Zero bureaucracy for technical changes, light flagging for scientific changes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from enum import Enum
import copy

try:
    from ..config import Config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import Config


class DeviationType(Enum):
    """Types of parameter deviations."""
    TECHNICAL = "technical"      # Auto-approved: QC thresholds, background correction, outlier detection
    SCIENTIFIC = "scientific"    # Flagged: Core clustering, segmentation, statistical methods
    EMERGENCY = "emergency"      # Critical fixes that bypass normal workflow


@dataclass
class Deviation:
    """Single parameter deviation record."""
    timestamp: str
    parameter_path: str
    original_value: Any
    new_value: Any
    deviation_type: str
    reason: str
    analysis_id: Optional[str] = None
    approved_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class DeviationWorkflow:
    """Lightweight deviation workflow manager."""
    
    # Auto-approved technical parameters (no approval needed)
    TECHNICAL_PARAMETERS = {
        # Quality control thresholds
        'quality_control.thresholds.total_ion_counts.min_tic_percentile',
        'quality_control.thresholds.total_ion_counts.max_low_tic_pixels_percent',
        'quality_control.thresholds.calibration_drift.max_drift_percent',
        'quality_control.thresholds.calibration_drift.max_cv_across_rois',
        'quality_control.thresholds.segmentation_quality.min_dna_signal',
        'quality_control.thresholds.segmentation_quality.min_tissue_coverage_percent',
        'quality_control.thresholds.signal_to_background.min_snr',
        
        # Background and preprocessing
        'processing.background_correction.clip_negative',
        'processing.dna_processing.tissue_threshold',
        'processing.arcsinh_transform.percentile_threshold',
        'processing.dna_processing.arcsinh_transform.noise_floor_percentile',
        'processing.dna_processing.arcsinh_transform.cofactor_multiplier',
        
        # Performance settings
        'performance.memory_limit_gb',
        'performance.chunk_size',
        'performance.parallel_processes',
        
        # Output settings
        'output.save_intermediate',
        'output.compression',
        'visualization.validation_plots.dpi',
        
        # Outlier detection
        'quality_control.thresholds.spatial_artifacts.edge_distance_fraction',
        'quality_control.thresholds.spatial_artifacts.significance_threshold',
        'quality_control.thresholds.spatial_artifacts.fold_change_threshold',
    }
    
    # Scientific parameters that get flagged (still auto-applied, just logged differently)
    SCIENTIFIC_PARAMETERS = {
        # Core clustering parameters
        'analysis.clustering.resolution_range',
        'analysis.clustering.method',
        'analysis.clustering.optimization_method',
        'analysis.clustering.use_coabundance_features',
        
        # Segmentation parameters
        'segmentation.slic_params.n_segments_per_mm2',
        'segmentation.slic_params.compactness',
        'segmentation.slic_params.sigma',
        'segmentation.scales_um',
        
        # Statistical methods
        'statistical_framework.mixed_effects.method',
        'statistical_framework.cross_validation.method',
        'statistical_framework.multiple_testing.correction_method',
        
        # Batch correction
        'analysis.batch_correction.method',
        'analysis.batch_correction.enabled',
    }
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize deviation workflow manager.
        
        Args:
            log_dir: Directory for deviation logs (defaults to results/deviations)
        """
        self.log_dir = Path(log_dir) if log_dir else Path("results/deviations")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('DeviationWorkflow')
    
    def apply_technical_deviation(
        self, 
        config: Config, 
        parameter_path: str, 
        new_value: Any, 
        reason: str,
        analysis_id: Optional[str] = None
    ) -> Config:
        """Apply auto-approved technical parameter change.
        
        Args:
            config: Configuration object to modify
            parameter_path: Dot-separated parameter path (e.g., 'quality_control.thresholds.min_snr')
            new_value: New parameter value
            reason: Brief reason for change
            analysis_id: Optional analysis identifier
            
        Returns:
            Modified configuration object
        """
        return self._apply_deviation(
            config, parameter_path, new_value, reason, 
            DeviationType.TECHNICAL, analysis_id
        )
    
    def apply_scientific_deviation(
        self, 
        config: Config, 
        parameter_path: str, 
        new_value: Any, 
        reason: str,
        analysis_id: Optional[str] = None
    ) -> Config:
        """Apply flagged scientific parameter change.
        
        Still works immediately, just logged with more visibility.
        
        Args:
            config: Configuration object to modify
            parameter_path: Dot-separated parameter path
            new_value: New parameter value
            reason: Brief reason for change (should explain scientific rationale)
            analysis_id: Optional analysis identifier
            
        Returns:
            Modified configuration object
        """
        return self._apply_deviation(
            config, parameter_path, new_value, reason, 
            DeviationType.SCIENTIFIC, analysis_id
        )
    
    def apply_emergency_deviation(
        self, 
        config: Config, 
        parameter_path: str, 
        new_value: Any, 
        reason: str,
        analysis_id: Optional[str] = None
    ) -> Config:
        """Apply emergency parameter change.
        
        For critical fixes that bypass normal workflow.
        
        Args:
            config: Configuration object to modify
            parameter_path: Dot-separated parameter path
            new_value: New parameter value
            reason: Detailed reason for emergency change
            analysis_id: Optional analysis identifier
            
        Returns:
            Modified configuration object
        """
        return self._apply_deviation(
            config, parameter_path, new_value, reason, 
            DeviationType.EMERGENCY, analysis_id
        )
    
    def _apply_deviation(
        self, 
        config: Config, 
        parameter_path: str, 
        new_value: Any, 
        reason: str,
        deviation_type: DeviationType,
        analysis_id: Optional[str] = None
    ) -> Config:
        """Internal method to apply parameter deviation.
        
        Args:
            config: Configuration object to modify
            parameter_path: Dot-separated parameter path
            new_value: New parameter value
            reason: Reason for change
            deviation_type: Type of deviation
            analysis_id: Optional analysis identifier
            
        Returns:
            Modified configuration object
        """
        # Get original value
        original_value = config.get(parameter_path)
        
        # Create modified config (don't modify original)
        modified_config = copy.deepcopy(config)
        modified_config.update(parameter_path, new_value)
        
        # Log the deviation
        deviation = Deviation(
            timestamp=datetime.now().isoformat(),
            parameter_path=parameter_path,
            original_value=original_value,
            new_value=new_value,
            deviation_type=deviation_type.value,
            reason=reason,
            analysis_id=analysis_id
        )
        
        self._log_deviation(deviation)
        
        # Different logging based on type
        if deviation_type == DeviationType.TECHNICAL:
            self.logger.info(f"✓ Technical change: {parameter_path} = {new_value} ({reason})")
        elif deviation_type == DeviationType.SCIENTIFIC:
            self.logger.warning(f"⚠ Scientific change: {parameter_path} = {new_value} ({reason})")
        elif deviation_type == DeviationType.EMERGENCY:
            self.logger.error(f"🚨 Emergency change: {parameter_path} = {new_value} ({reason})")
        
        return modified_config
    
    def auto_classify_deviation(self, parameter_path: str) -> DeviationType:
        """Automatically classify parameter deviation type.
        
        Args:
            parameter_path: Dot-separated parameter path
            
        Returns:
            Automatically determined deviation type
        """
        if parameter_path in self.TECHNICAL_PARAMETERS:
            return DeviationType.TECHNICAL
        elif parameter_path in self.SCIENTIFIC_PARAMETERS:
            return DeviationType.SCIENTIFIC
        else:
            # Default to scientific for unknown parameters
            return DeviationType.SCIENTIFIC
    
    def apply_auto_deviation(
        self, 
        config: Config, 
        parameter_path: str, 
        new_value: Any, 
        reason: str,
        analysis_id: Optional[str] = None
    ) -> Config:
        """Apply parameter deviation with automatic classification.
        
        Args:
            config: Configuration object to modify
            parameter_path: Dot-separated parameter path
            new_value: New parameter value
            reason: Reason for change
            analysis_id: Optional analysis identifier
            
        Returns:
            Modified configuration object
        """
        deviation_type = self.auto_classify_deviation(parameter_path)
        return self._apply_deviation(
            config, parameter_path, new_value, reason, 
            deviation_type, analysis_id
        )
    
    def _log_deviation(self, deviation: Deviation) -> None:
        """Log deviation to JSON file.
        
        Args:
            deviation: Deviation record to log
        """
        # Use analysis_id if provided, otherwise use date
        if deviation.analysis_id:
            log_file = self.log_dir / f"deviations_{deviation.analysis_id}.json"
        else:
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = self.log_dir / f"deviations_{date_str}.json"
        
        # Load existing deviations
        deviations = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    deviations = json.load(f)
            except (json.JSONDecodeError, IOError):
                deviations = []
        
        # Add new deviation
        deviations.append(deviation.to_dict())
        
        # Save updated deviations
        with open(log_file, 'w') as f:
            json.dump(deviations, f, indent=2, default=str)
    
    def get_deviation_log(self, analysis_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get deviation log for analysis.
        
        Args:
            analysis_id: Analysis identifier (uses today's date if None)
            
        Returns:
            List of deviation records
        """
        if analysis_id:
            log_file = self.log_dir / f"deviations_{analysis_id}.json"
        else:
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = self.log_dir / f"deviations_{date_str}.json"
        
        if not log_file.exists():
            return []
        
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    
    def get_deviation_summary(self, analysis_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of deviations for analysis.
        
        Args:
            analysis_id: Analysis identifier (uses today's date if None)
            
        Returns:
            Summary of deviations
        """
        deviations = self.get_deviation_log(analysis_id)
        
        if not deviations:
            return {
                'total_deviations': 0,
                'by_type': {},
                'recent_changes': []
            }
        
        # Count by type
        by_type = {}
        for dev in deviations:
            dev_type = dev.get('deviation_type', 'unknown')
            by_type[dev_type] = by_type.get(dev_type, 0) + 1
        
        # Get recent changes (last 5)
        recent_changes = sorted(
            deviations, 
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )[:5]
        
        return {
            'total_deviations': len(deviations),
            'by_type': by_type,
            'recent_changes': recent_changes,
            'log_file': f"deviations_{analysis_id or datetime.now().strftime('%Y%m%d')}.json"
        }
    
    def clear_deviations(self, analysis_id: Optional[str] = None) -> bool:
        """Clear deviation log for analysis.
        
        Args:
            analysis_id: Analysis identifier (uses today's date if None)
            
        Returns:
            True if log was cleared, False if no log existed
        """
        if analysis_id:
            log_file = self.log_dir / f"deviations_{analysis_id}.json"
        else:
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = self.log_dir / f"deviations_{date_str}.json"
        
        if log_file.exists():
            log_file.unlink()
            return True
        return False


# Convenience functions for easy use
def create_deviation_workflow(log_dir: Optional[str] = None) -> DeviationWorkflow:
    """Create deviation workflow manager.
    
    Args:
        log_dir: Directory for deviation logs
        
    Returns:
        Configured deviation workflow manager
    """
    return DeviationWorkflow(log_dir)


def quick_technical_change(
    config: Config,
    parameter_path: str,
    new_value: Any,
    reason: str,
    analysis_id: Optional[str] = None,
    log_dir: Optional[str] = None
) -> Config:
    """Quick technical parameter change with minimal setup.
    
    Args:
        config: Configuration to modify
        parameter_path: Parameter to change
        new_value: New value
        reason: Brief reason
        analysis_id: Optional analysis ID
        log_dir: Optional log directory
        
    Returns:
        Modified configuration
    """
    workflow = DeviationWorkflow(log_dir)
    return workflow.apply_technical_deviation(
        config, parameter_path, new_value, reason, analysis_id
    )


def quick_scientific_change(
    config: Config,
    parameter_path: str,
    new_value: Any,
    reason: str,
    analysis_id: Optional[str] = None,
    log_dir: Optional[str] = None
) -> Config:
    """Quick scientific parameter change with minimal setup.
    
    Args:
        config: Configuration to modify
        parameter_path: Parameter to change
        new_value: New value
        reason: Scientific rationale
        analysis_id: Optional analysis ID
        log_dir: Optional log directory
        
    Returns:
        Modified configuration
    """
    workflow = DeviationWorkflow(log_dir)
    return workflow.apply_scientific_deviation(
        config, parameter_path, new_value, reason, analysis_id
    )


def show_recent_deviations(
    analysis_id: Optional[str] = None,
    log_dir: Optional[str] = None
) -> None:
    """Show recent parameter deviations.
    
    Args:
        analysis_id: Analysis identifier
        log_dir: Log directory
    """
    workflow = DeviationWorkflow(log_dir)
    summary = workflow.get_deviation_summary(analysis_id)
    
    print(f"\n📋 Deviation Summary")
    print(f"Total deviations: {summary['total_deviations']}")
    
    if summary['by_type']:
        print("\nBy type:")
        for dev_type, count in summary['by_type'].items():
            icon = "✓" if dev_type == "technical" else "⚠" if dev_type == "scientific" else "🚨"
            print(f"  {icon} {dev_type}: {count}")
    
    if summary['recent_changes']:
        print("\nRecent changes:")
        for change in summary['recent_changes']:
            dev_type = change.get('deviation_type', 'unknown')
            icon = "✓" if dev_type == "technical" else "⚠" if dev_type == "scientific" else "🚨"
            timestamp = change.get('timestamp', 'unknown')[:19]  # Remove microseconds
            print(f"  {icon} {timestamp}: {change.get('parameter_path')} = {change.get('new_value')}")
            print(f"    Reason: {change.get('reason')}")
    
    print(f"\nLog file: {summary['log_file']}")