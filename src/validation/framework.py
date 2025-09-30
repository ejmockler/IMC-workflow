"""
Comprehensive Validation Framework for IMC Analysis Pipeline

Provides scientific-grade validation infrastructure with modular rules,
quality metrics, and automated decision-making for robust analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import json


class ValidationSeverity(Enum):
    """Validation result severity levels."""
    CRITICAL = "critical"    # Pipeline must stop
    WARNING = "warning"      # Issue noted but can continue  
    INFO = "info"           # Informational only
    PASS = "pass"           # Validation passed


class ValidationCategory(Enum):
    """Categories of validation checks."""
    DATA_INTEGRITY = "data_integrity"
    SCIENTIFIC_QUALITY = "scientific_quality"
    PIPELINE_STATE = "pipeline_state"
    OUTPUT_COMPLIANCE = "output_compliance"


@dataclass
class ValidationMetric:
    """Individual validation metric result."""
    name: str
    value: Union[float, int, bool, str]
    expected_range: Optional[Tuple[float, float]] = None
    units: Optional[str] = None
    description: Optional[str] = None


@dataclass 
class ValidationResult:
    """Comprehensive validation result with metrics and recommendations."""
    rule_name: str
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    
    # Quantitative metrics
    metrics: Dict[str, ValidationMetric] = field(default_factory=dict)
    quality_score: Optional[float] = None  # 0-1 scale
    
    # Context and recommendations
    context: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    # Technical details
    rule_version: str = "1.0"
    execution_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'rule_name': self.rule_name,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'quality_score': self.quality_score,
            'metrics': {
                name: {
                    'value': metric.value,
                    'expected_range': metric.expected_range,
                    'units': metric.units,
                    'description': metric.description
                } for name, metric in self.metrics.items()
            },
            'context': self.context,
            'recommendations': self.recommendations,
            'rule_version': self.rule_version,
            'execution_time_ms': self.execution_time_ms
        }


class ValidationRule(ABC):
    """
    Abstract base class for validation rules.
    
    Each rule implements specific validation logic and returns structured results.
    Rules are modular, testable, and can be combined into validation suites.
    """
    
    def __init__(self, name: str, category: ValidationCategory, version: str = "1.0"):
        """Initialize validation rule."""
        self.name = name
        self.category = category
        self.version = version
        self.logger = logging.getLogger(f'ValidationRule.{name}')
    
    @abstractmethod
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """
        Execute validation rule.
        
        Args:
            data: Data to validate (format depends on rule)
            context: Additional context for validation
            
        Returns:
            Validation result with metrics and recommendations
        """
        pass
    
    def _create_result(
        self,
        severity: ValidationSeverity,
        message: str,
        quality_score: Optional[float] = None,
        metrics: Dict[str, ValidationMetric] = None,
        recommendations: List[str] = None,
        context: Dict[str, Any] = None
    ) -> ValidationResult:
        """Helper to create validation results."""
        return ValidationResult(
            rule_name=self.name,
            category=self.category,
            severity=severity,
            message=message,
            quality_score=quality_score,
            metrics=metrics or {},
            recommendations=recommendations or [],
            context=context or {},
            rule_version=self.version
        )


@dataclass
class ValidationSuiteConfig:
    """Configuration for validation suite execution."""
    # Basic info
    name: str = "validation_suite"
    
    # Execution control
    stop_on_critical: bool = True  # Default behavior: Stop on critical failures
    parallel_execution: bool = False
    timeout_seconds: int = 300
    
    # Rule filtering
    enabled_categories: List[ValidationCategory] = field(default_factory=lambda: list(ValidationCategory))
    disabled_rules: List[str] = field(default_factory=list)
    
    # Quality thresholds
    minimum_quality_score: float = 0.5
    warning_quality_threshold: float = 0.7
    
    # Output control
    save_detailed_results: bool = True
    generate_summary_report: bool = True


class ValidationSuite:
    """
    Orchestrates multiple validation rules and provides comprehensive quality assessment.
    
    Features:
    - Modular rule execution with dependency management
    - Quality scoring and automated decision-making
    - Detailed reporting and recommendations
    - Configurable execution policies
    """
    
    def __init__(self, config: ValidationSuiteConfig = None):
        """Initialize validation suite."""
        self.config = config or ValidationSuiteConfig()
        self.rules: List[ValidationRule] = []
        self.logger = logging.getLogger('ValidationSuite')
        self.execution_history = []
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add validation rule to suite."""
        if rule.category in self.config.enabled_categories:
            if rule.name not in self.config.disabled_rules:
                self.rules.append(rule)
                self.logger.debug(f"Added validation rule: {rule.name}")
            else:
                self.logger.debug(f"Skipped disabled rule: {rule.name}")
        else:
            self.logger.debug(f"Skipped rule in disabled category: {rule.name} ({rule.category.value})")
    
    def validate(
        self, 
        data: Dict[str, Any], 
        context: Dict[str, Any] = None
    ) -> 'ValidationSuiteResult':
        """
        Execute all validation rules on provided data.
        
        Args:
            data: Data to validate
            context: Additional validation context
            
        Returns:
            Comprehensive validation suite result
        """
        context = context or {}
        results = []
        execution_start = pd.Timestamp.now()
        
        self.logger.info(f"Starting validation suite with {len(self.rules)} rules")
        
        for rule in self.rules:
            try:
                rule_start = pd.Timestamp.now()
                
                self.logger.debug(f"Executing rule: {rule.name}")
                result = rule.validate(data, context)
                
                # Add execution timing
                execution_time = (pd.Timestamp.now() - rule_start).total_seconds() * 1000
                result.execution_time_ms = execution_time
                
                results.append(result)
                
                # Check for early termination
                if result.severity == ValidationSeverity.CRITICAL and self.config.stop_on_critical:
                    self.logger.error(f"Critical validation failure in {rule.name}: {result.message}")
                    self.logger.info("Stopping validation suite due to critical failure")
                    break
                    
                self.logger.debug(f"Completed rule {rule.name} in {execution_time:.1f}ms")
                
            except Exception as e:
                self.logger.error(f"Exception in validation rule {rule.name}: {str(e)}")
                
                # Create error result
                error_result = ValidationResult(
                    rule_name=rule.name,
                    category=rule.category,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Validation rule failed with exception: {str(e)}",
                    context={"exception": str(e), "rule_version": rule.version}
                )
                results.append(error_result)
                
                if self.config.stop_on_critical:
                    break
        
        total_execution_time = (pd.Timestamp.now() - execution_start).total_seconds() * 1000
        
        # Create suite result
        suite_result = ValidationSuiteResult(
            results=results,
            config=self.config,
            execution_time_ms=total_execution_time,
            context=context
        )
        
        self.execution_history.append(suite_result)
        
        self.logger.info(f"Validation suite completed in {total_execution_time:.1f}ms")
        self.logger.info(f"Results: {suite_result.summary_stats}")
        
        return suite_result
    
    def run(
        self, 
        data: Dict[str, Any], 
        context: Dict[str, Any] = None
    ) -> 'ValidationSuiteResult':
        """Alias for validate() method for backward compatibility."""
        return self.validate(data, context)


class ValidationSuiteResult:
    """Comprehensive result from validation suite execution."""
    
    def __init__(
        self, 
        results: List[ValidationResult] = None,
        config: ValidationSuiteConfig = None,
        execution_time_ms: float = 0.0,
        context: Dict[str, Any] = None,
        # Legacy parameters
        suite_name: str = None,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None
    ):
        """Initialize ValidationSuiteResult with flexible signature."""
        self.results = results or []
        self.config = config or ValidationSuiteConfig()
        self.context = context or {}
        
        # Handle legacy constructor
        if suite_name is not None:
            self.suite_name = suite_name
            self.config.name = suite_name
        
        if start_time is not None and end_time is not None:
            self.start_time = start_time
            self.end_time = end_time
            self.execution_time_ms = (end_time - start_time).total_seconds() * 1000
        else:
            self.execution_time_ms = execution_time_ms
    
    @property
    def summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {"total_rules": 0, "status": "no_rules"}
        
        severity_counts = {}
        for severity in ValidationSeverity:
            severity_counts[severity] = sum(1 for r in self.results if r.severity == severity)
        
        quality_scores = [r.quality_score for r in self.results if r.quality_score is not None]
        
        return {
            "total_rules": len(self.results),
            "severity_distribution": severity_counts,
            "severity_counts": severity_counts,  # For backward compatibility
            "has_critical": severity_counts.get(ValidationSeverity.CRITICAL, 0) > 0,
            "has_warnings": severity_counts.get(ValidationSeverity.WARNING, 0) > 0,
            "overall_quality_score": np.mean(quality_scores) if quality_scores else None,
            "min_quality_score": np.min(quality_scores) if quality_scores else None,
            "execution_time_ms": self.execution_time_ms,
            "status": self._determine_status()
        }
    
    def _determine_status(self) -> str:
        """Determine overall validation status."""
        if any(r.severity == ValidationSeverity.CRITICAL for r in self.results):
            return "critical"
        elif any(r.severity == ValidationSeverity.WARNING for r in self.results):
            return "warning"
        else:
            return "pass"
    
    def get_critical_failures(self) -> List[ValidationResult]:
        """Get all critical validation failures."""
        return [r for r in self.results if r.severity == ValidationSeverity.CRITICAL]
    
    def get_recommendations(self) -> List[str]:
        """Get all recommendations from validation results."""
        recommendations = []
        for result in self.results:
            recommendations.extend(result.recommendations)
        return list(set(recommendations))  # Remove duplicates
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "summary": self.summary_stats,
            "results": [r.to_dict() for r in self.results],
            "recommendations": self.get_recommendations(),
            "config": {
                "stop_on_critical": self.config.stop_on_critical,
                "minimum_quality_score": self.config.minimum_quality_score,
                "enabled_categories": [c.value for c in self.config.enabled_categories]
            },
            "context": self.context
        }
    
    def save_report(self, output_path: Union[str, Path]) -> None:
        """Save validation report to file."""
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logging.getLogger('ValidationSuite').info(f"Validation report saved to {output_path}")


# Backward compatibility alias with different constructor signature
class ValidationReport(ValidationSuiteResult):
    """Backward compatibility wrapper for ValidationSuiteResult."""
    
    def __init__(
        self, 
        suite_name: str, 
        results: List[ValidationResult], 
        start_time: pd.Timestamp, 
        end_time: pd.Timestamp,
        context: Dict[str, Any] = None
    ):
        """Initialize validation report with legacy signature."""
        # Convert to new signature
        super().__init__(
            results=results,
            config=ValidationSuiteConfig(),  # Default config
            execution_time_ms=(end_time - start_time).total_seconds() * 1000,
            context=context or {}
        )
        self.suite_name = suite_name
        self.start_time = start_time
        self.end_time = end_time


# Quality scoring utilities
class QualityMetrics:
    """Standardized quality metrics for IMC analysis validation."""
    
    @staticmethod
    def calculate_data_completeness(
        expected_markers: List[str], 
        found_markers: List[str]
    ) -> float:
        """Calculate data completeness score (0-1)."""
        if not expected_markers:
            return 1.0
        return len(set(found_markers) & set(expected_markers)) / len(expected_markers)
    
    @staticmethod
    def calculate_spatial_quality(
        coords: np.ndarray,
        min_density: float = 100.0,
        max_outlier_fraction: float = 0.05
    ) -> float:
        """Calculate spatial data quality score (0-1)."""
        if len(coords) == 0:
            return 0.0
        
        # Density check
        area = (coords[:, 0].max() - coords[:, 0].min()) * (coords[:, 1].max() - coords[:, 1].min())
        density = len(coords) / area if area > 0 else 0
        density_score = min(1.0, density / min_density)
        
        # Outlier check
        q25, q75 = np.percentile(coords, [25, 75], axis=0)
        iqr = q75 - q25
        outlier_bounds = np.column_stack([q25 - 1.5 * iqr, q75 + 1.5 * iqr])
        
        outliers = np.any((coords < outlier_bounds[:, 0]) | (coords > outlier_bounds[:, 1]), axis=1)
        outlier_fraction = np.sum(outliers) / len(coords)
        outlier_score = max(0.0, 1.0 - outlier_fraction / max_outlier_fraction)
        
        return (density_score + outlier_score) / 2
    
    @staticmethod
    def calculate_distribution_quality(
        data: np.ndarray,
        expected_distribution: str = "poisson"
    ) -> float:
        """Calculate distribution quality score (0-1)."""
        if len(data) == 0:
            return 0.0
        
        if expected_distribution == "poisson":
            # For Poisson: variance should approximately equal mean
            mean_val = np.mean(data)
            var_val = np.var(data)
            
            if mean_val == 0:
                return 1.0 if var_val == 0 else 0.0
            
            # Calculate how close variance/mean is to 1
            ratio = var_val / mean_val
            # Score decreases as ratio deviates from 1
            score = np.exp(-abs(np.log(ratio))**2)
            return float(score)
        
        return 0.5  # Default for unknown distributions


# Factory function for creating validation suites
def create_validation_suite(
    categories: List[ValidationCategory] = None,
    **config_kwargs
) -> ValidationSuite:
    """
    Factory function to create validation suite with standard configuration.
    
    Args:
        categories: List of validation categories to enable
        **config_kwargs: Additional configuration options
        
    Returns:
        Configured validation suite
    """
    if categories is None:
        categories = list(ValidationCategory)
    
    config = ValidationSuiteConfig(
        enabled_categories=categories,
        **config_kwargs
    )
    
    return ValidationSuite(config)