"""
Comprehensive Validation Framework for IMC Analysis

This package provides scientific-grade validation infrastructure for IMC analysis
pipelines with modular rules, quality metrics, and automated decision-making.

Usage:
    from src.validation import create_validation_suite, ValidationCategory
    
    # Create validation suite for all categories
    suite = create_validation_suite()
    
    # Add specific validation rules
    from src.validation.data_integrity import CoordinateValidator, IonCountValidator
    from src.validation.scientific_quality import BiologicalValidator
    
    suite.add_rule(CoordinateValidator())
    suite.add_rule(IonCountValidator())
    suite.add_rule(BiologicalValidator())
    
    # Run validation
    result = suite.validate(data, context)
    
    # Check results
    if result.summary_stats['has_critical']:
        print("Critical validation failures detected")
    
    # Save report
    result.save_report('validation_report.json')
"""

from .framework import (
    ValidationRule,
    ValidationResult, 
    ValidationSeverity,
    ValidationCategory,
    ValidationMetric,
    ValidationSuite,
    ValidationSuiteConfig,
    ValidationSuiteResult,
    QualityMetrics,
    create_validation_suite
)

from .data_integrity import (
    CoordinateValidator,
    IonCountValidator,
    TransformationValidator
)

from .scientific_quality import (
    BiologicalValidator,
    SpatialValidator
)

from .pipeline_state import (
    PreprocessingValidator,
    SegmentationValidator,
    ClusteringValidator
)

__all__ = [
    # Core framework
    'ValidationRule',
    'ValidationResult',
    'ValidationSeverity', 
    'ValidationCategory',
    'ValidationMetric',
    'ValidationSuite',
    'ValidationSuiteConfig',
    'ValidationSuiteResult',
    'QualityMetrics',
    'create_validation_suite',
    
    # Data integrity validators
    'CoordinateValidator',
    'IonCountValidator', 
    'TransformationValidator',
    
    # Scientific quality validators
    'BiologicalValidator',
    'SpatialValidator',
    
    # Pipeline state validators
    'PreprocessingValidator',
    'SegmentationValidator',
    'ClusteringValidator'
]

# Version information
__version__ = "1.0.0"
__author__ = "IMC Analysis Pipeline"
__description__ = "Comprehensive validation framework for IMC analysis"