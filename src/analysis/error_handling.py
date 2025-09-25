"""
Unified Error Handling for IMC Analysis Pipeline

Provides consistent exception handling, validation, and recovery patterns
across all analysis modules to improve robustness and debugging.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable, Type
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"       # Pipeline-stopping errors
    WARNING = "warning"         # Issues that can be worked around
    INFO = "info"              # Informational messages


@dataclass
class PipelineError:
    """Structured error information."""
    message: str
    severity: ErrorSeverity
    component: str
    error_code: str
    context: Dict[str, Any]
    original_exception: Optional[Exception] = None
    recovery_suggestion: Optional[str] = None


class IMCAnalysisError(Exception):
    """Base exception for IMC analysis pipeline."""
    
    def __init__(self, pipeline_error: PipelineError):
        self.pipeline_error = pipeline_error
        super().__init__(pipeline_error.message)


class DataValidationError(IMCAnalysisError):
    """Exception for data validation failures."""
    pass


class ProcessingError(IMCAnalysisError):
    """Exception for processing failures."""
    pass


class ConfigurationError(IMCAnalysisError):
    """Exception for configuration issues."""
    pass


class MemoryError(IMCAnalysisError):
    """Exception for memory-related issues.""" 
    pass


class ErrorHandler:
    """Centralized error handling and validation."""
    
    def __init__(self, component_name: str):
        """Initialize error handler for a specific component."""
        self.component_name = component_name
        self.logger = logging.getLogger(f'ErrorHandler.{component_name}')
        self.error_history = []
    
    def validate_coordinates(self, coords: np.ndarray, context: str = "") -> None:
        """Validate coordinate array."""
        if coords is None:
            raise self._create_error(
                "Coordinates cannot be None",
                ErrorSeverity.CRITICAL,
                "COORD_NULL",
                {"context": context}
            )
        
        if not isinstance(coords, np.ndarray):
            raise self._create_error(
                f"Coordinates must be numpy array, got {type(coords)}",
                ErrorSeverity.CRITICAL,
                "COORD_TYPE",
                {"context": context, "actual_type": str(type(coords))}
            )
        
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise self._create_error(
                f"Coordinates must be Nx2 array, got shape {coords.shape}",
                ErrorSeverity.CRITICAL,
                "COORD_SHAPE",
                {"context": context, "shape": coords.shape}
            )
        
        if len(coords) == 0:
            raise self._create_error(
                "Coordinates array is empty",
                ErrorSeverity.CRITICAL,
                "COORD_EMPTY",
                {"context": context}
            )
        
        # Check for NaN/Inf values
        if not np.isfinite(coords).all():
            n_invalid = np.sum(~np.isfinite(coords))
            raise self._create_error(
                f"Coordinates contain {n_invalid} non-finite values",
                ErrorSeverity.CRITICAL,
                "COORD_INVALID",
                {"context": context, "n_invalid": n_invalid},
                recovery_suggestion="Remove or interpolate non-finite coordinate values"
            )
    
    def validate_ion_counts(
        self, 
        ion_counts: Dict[str, np.ndarray], 
        n_expected: Optional[int] = None,
        context: str = ""
    ) -> None:
        """Validate ion counts dictionary."""
        if not isinstance(ion_counts, dict):
            raise self._create_error(
                f"Ion counts must be dictionary, got {type(ion_counts)}",
                ErrorSeverity.CRITICAL,
                "ION_TYPE",
                {"context": context, "actual_type": str(type(ion_counts))}
            )
        
        if len(ion_counts) == 0:
            raise self._create_error(
                "Ion counts dictionary is empty",
                ErrorSeverity.CRITICAL,
                "ION_EMPTY",
                {"context": context}
            )
        
        # Validate each protein array
        for protein, counts in ion_counts.items():
            if not isinstance(counts, np.ndarray):
                raise self._create_error(
                    f"Ion counts for {protein} must be numpy array, got {type(counts)}",
                    ErrorSeverity.CRITICAL,
                    "ION_PROTEIN_TYPE",
                    {"context": context, "protein": protein, "actual_type": str(type(counts))}
                )
            
            if counts.ndim != 1:
                raise self._create_error(
                    f"Ion counts for {protein} must be 1D array, got {counts.ndim}D",
                    ErrorSeverity.CRITICAL,
                    "ION_PROTEIN_DIM",
                    {"context": context, "protein": protein, "ndim": counts.ndim}
                )
            
            if n_expected is not None and len(counts) != n_expected:
                raise self._create_error(
                    f"Ion counts for {protein} length mismatch: {len(counts)} != {n_expected}",
                    ErrorSeverity.CRITICAL,
                    "ION_LENGTH_MISMATCH",
                    {"context": context, "protein": protein, "actual": len(counts), "expected": n_expected}
                )
            
            # Check for negative values (ion counts should be non-negative)
            if np.any(counts < 0):
                n_negative = np.sum(counts < 0)
                self._log_warning(
                    f"Ion counts for {protein} contain {n_negative} negative values",
                    "ION_NEGATIVE",
                    {"context": context, "protein": protein, "n_negative": n_negative},
                    recovery_suggestion="Consider background correction or data preprocessing"
                )
            
            # Check for all-zero data
            if np.all(counts == 0):
                self._log_warning(
                    f"Ion counts for {protein} are all zero",
                    "ION_ALL_ZERO",
                    {"context": context, "protein": protein},
                    recovery_suggestion="Verify protein marker presence and data processing"
                )
    
    def validate_memory_requirements(
        self, 
        estimated_mb: float, 
        limit_mb: float,
        context: str = ""
    ) -> None:
        """Validate memory requirements against limits."""
        if estimated_mb > limit_mb:
            raise self._create_error(
                f"Estimated memory {estimated_mb:.1f}MB exceeds limit {limit_mb:.1f}MB",
                ErrorSeverity.CRITICAL,
                "MEMORY_EXCEEDED",
                {"context": context, "estimated_mb": estimated_mb, "limit_mb": limit_mb},
                recovery_suggestion="Increase memory limit or use memory optimization"
            )
        
        # Warn if approaching limit
        if estimated_mb > 0.8 * limit_mb:
            self._log_warning(
                f"Memory usage {estimated_mb:.1f}MB approaching limit {limit_mb:.1f}MB",
                "MEMORY_HIGH",
                {"context": context, "estimated_mb": estimated_mb, "limit_mb": limit_mb},
                recovery_suggestion="Consider enabling memory optimization"
            )
    
    def safe_execute(
        self,
        operation: Callable,
        operation_name: str,
        error_type: Type[IMCAnalysisError] = ProcessingError,
        **kwargs
    ) -> Any:
        """Safely execute an operation with error handling."""
        try:
            self.logger.debug(f"Executing {operation_name}")
            result = operation(**kwargs)
            self.logger.debug(f"Successfully completed {operation_name}")
            return result
            
        except IMCAnalysisError:
            # Re-raise our structured errors
            raise
            
        except Exception as e:
            # Wrap unhandled exceptions
            error = PipelineError(
                message=f"Unexpected error in {operation_name}: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                component=self.component_name,
                error_code="UNEXPECTED_ERROR",
                context={"operation": operation_name, "exception_type": type(e).__name__},
                original_exception=e,
                recovery_suggestion="Check input data and operation parameters"
            )
            
            self.logger.error(f"Unexpected error in {operation_name}: {str(e)}")
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            
            raise error_type(error) from e
    
    def _create_error(
        self,
        message: str,
        severity: ErrorSeverity,
        error_code: str,
        context: Dict[str, Any],
        recovery_suggestion: Optional[str] = None
    ) -> DataValidationError:
        """Create structured error."""
        error = PipelineError(
            message=message,
            severity=severity,
            component=self.component_name,
            error_code=error_code,
            context=context,
            recovery_suggestion=recovery_suggestion
        )
        
        self.error_history.append(error)
        self.logger.error(f"[{error_code}] {message}")
        
        return DataValidationError(error)
    
    def _log_warning(
        self,
        message: str,
        error_code: str,
        context: Dict[str, Any],
        recovery_suggestion: Optional[str] = None
    ) -> None:
        """Log warning with structured information."""
        warning = PipelineError(
            message=message,
            severity=ErrorSeverity.WARNING,
            component=self.component_name,
            error_code=error_code,
            context=context,
            recovery_suggestion=recovery_suggestion
        )
        
        self.error_history.append(warning)
        self.logger.warning(f"[{error_code}] {message}")
        
        if recovery_suggestion:
            self.logger.info(f"Recovery suggestion: {recovery_suggestion}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors and warnings."""
        critical_errors = [e for e in self.error_history if e.severity == ErrorSeverity.CRITICAL]
        warnings = [e for e in self.error_history if e.severity == ErrorSeverity.WARNING]
        
        return {
            "component": self.component_name,
            "total_errors": len(critical_errors),
            "total_warnings": len(warnings),
            "critical_errors": [e.error_code for e in critical_errors],
            "warnings": [e.error_code for e in warnings],
            "last_error": critical_errors[-1].message if critical_errors else None
        }


def create_error_handler(component_name: str) -> ErrorHandler:
    """Factory function to create error handlers."""
    return ErrorHandler(component_name)