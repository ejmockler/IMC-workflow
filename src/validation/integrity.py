"""
Data Integrity and Security Validation

Comprehensive data integrity checks with cryptographic verification.
Prevents silent data corruption and ensures validation reproducibility.
"""

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import mmap

from .framework import (
    ValidationRule, ValidationResult, ValidationSeverity,
    ValidationCategory, ValidationMetric
)

logger = logging.getLogger(__name__)


class DataIntegrityValidator(ValidationRule):
    """Validates data integrity using cryptographic checksums."""
    
    def __init__(self):
        super().__init__("data_integrity_validation", ValidationCategory.DATA_INTEGRITY)
        self.known_hashes = {}  # Cache of known good hashes
        
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Validate data integrity with checksums."""
        
        if context is None:
            context = {}
        
        file_path = context.get('file_path')
        if not file_path:
            return self._create_result(
                ValidationSeverity.WARNING,
                "No file path provided for integrity validation",
                quality_score=0.5
            )
        
        metrics = {}
        recommendations = []
        issues = []
        
        try:
            # Compute file hash
            file_hash = self._compute_file_hash_secure(Path(file_path))
            metrics['file_hash'] = ValidationMetric(
                'file_hash',
                file_hash[:16],  # First 16 chars for display
                description="SHA-256 hash of source file (truncated)"
            )
            
            # Check file size
            file_size = Path(file_path).stat().st_size
            metrics['file_size_bytes'] = ValidationMetric(
                'file_size_bytes',
                file_size,
                description="File size in bytes"
            )
            
            # Validate against known hash if available
            expected_hash = context.get('expected_hash')
            hash_valid = True
            
            if expected_hash:
                hash_valid = file_hash == expected_hash
                if not hash_valid:
                    issues.append("File hash mismatch - possible data corruption")
                    recommendations.append("Re-acquire data from original source")
                    
                metrics['hash_match'] = ValidationMetric(
                    'hash_match',
                    hash_valid,
                    description="Whether file hash matches expected value"
                )
            
            # Validate data array integrity
            array_integrity = self._validate_array_integrity(data)
            metrics.update(array_integrity['metrics'])
            issues.extend(array_integrity['issues'])
            recommendations.extend(array_integrity['recommendations'])
            
            # Check for suspicious patterns
            pattern_check = self._check_suspicious_patterns(data)
            metrics.update(pattern_check['metrics'])
            issues.extend(pattern_check['issues'])
            recommendations.extend(pattern_check['recommendations'])
            
            # Overall integrity score
            integrity_components = [
                1.0 if hash_valid else 0.0,
                array_integrity['integrity_score'],
                pattern_check['integrity_score']
            ]
            
            overall_integrity = np.mean(integrity_components)
            
            metrics['overall_integrity'] = ValidationMetric(
                'overall_integrity',
                overall_integrity,
                expected_range=(0.9, 1.0),
                description="Overall data integrity score"
            )
            
            # Determine severity
            if not hash_valid or overall_integrity < 0.7:
                severity = ValidationSeverity.CRITICAL
                message = f"Critical data integrity issues detected (score: {overall_integrity:.3f})"
            elif overall_integrity < 0.9 or issues:
                severity = ValidationSeverity.WARNING
                message = f"Data integrity concerns detected (score: {overall_integrity:.3f})"
            else:
                severity = ValidationSeverity.PASS
                message = f"Data integrity validated (score: {overall_integrity:.3f})"
            
            return self._create_result(
                severity=severity,
                message=message,
                quality_score=overall_integrity,
                metrics=metrics,
                recommendations=recommendations,
                context={
                    'file_hash': file_hash,
                    'file_size': file_size,
                    'integrity_issues': issues
                }
            )
            
        except Exception as e:
            logger.error(f"Data integrity validation failed: {e}")
            return self._create_result(
                ValidationSeverity.CRITICAL,
                f"Data integrity validation failed: {str(e)}",
                quality_score=0.0,
                context={'error': str(e)}
            )
    
    def _compute_file_hash_secure(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Compute SHA-256 hash of file using secure chunked reading."""
        
        hasher = hashlib.sha256()
        
        try:
            # Use memory mapping for large files
            if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                with open(file_path, 'rb') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        for i in range(0, len(mm), chunk_size):
                            chunk = mm[i:i+chunk_size]
                            hasher.update(chunk)
            else:
                # Regular chunked reading for smaller files
                with open(file_path, 'rb') as f:
                    while chunk := f.read(chunk_size):
                        hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Error computing file hash: {e}")
            return ""
    
    def _validate_array_integrity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate integrity of data arrays."""
        
        metrics = {}
        issues = []
        recommendations = []
        integrity_scores = []
        
        # Check coordinate arrays
        if 'coords' in data:
            coords = data['coords']
            coord_integrity = self._check_array_integrity(coords, 'coordinates')
            metrics.update({f"coord_{k}": v for k, v in coord_integrity['metrics'].items()})
            issues.extend([f"Coordinates: {issue}" for issue in coord_integrity['issues']])
            integrity_scores.append(coord_integrity['integrity_score'])
        
        # Check ion count arrays
        if 'ion_counts' in data:
            for protein, counts in data['ion_counts'].items():
                if isinstance(counts, np.ndarray):
                    protein_integrity = self._check_array_integrity(counts, f'{protein}_counts')
                    metrics.update({f"{protein}_{k}": v for k, v in protein_integrity['metrics'].items()})
                    issues.extend([f"{protein}: {issue}" for issue in protein_integrity['issues']])
                    integrity_scores.append(protein_integrity['integrity_score'])
        
        # Check segmentation arrays
        if 'segmentation' in data:
            seg = data['segmentation']
            seg_integrity = self._check_array_integrity(seg, 'segmentation')
            metrics.update({f"seg_{k}": v for k, v in seg_integrity['metrics'].items()})
            issues.extend([f"Segmentation: {issue}" for issue in seg_integrity['issues']])
            integrity_scores.append(seg_integrity['integrity_score'])
        
        overall_integrity = np.mean(integrity_scores) if integrity_scores else 1.0
        
        if overall_integrity < 0.8:
            recommendations.append("Regenerate arrays from original source data")
        
        return {
            'metrics': metrics,
            'issues': issues,
            'recommendations': recommendations,
            'integrity_score': overall_integrity
        }
    
    def _check_array_integrity(self, array: np.ndarray, array_name: str) -> Dict[str, Any]:
        """Check integrity of individual array."""
        
        metrics = {}
        issues = []
        integrity_components = []
        
        # Check for NaN/Inf values
        finite_mask = np.isfinite(array)
        finite_fraction = np.mean(finite_mask)
        
        metrics['finite_fraction'] = ValidationMetric(
            'finite_fraction',
            finite_fraction,
            expected_range=(0.99, 1.0),
            description=f"Fraction of finite values in {array_name}"
        )
        
        if finite_fraction < 0.95:
            issues.append(f"High fraction of non-finite values: {1-finite_fraction:.2%}")
        
        integrity_components.append(finite_fraction)
        
        # Check data type consistency
        expected_dtypes = {
            'coordinates': [np.float32, np.float64],
            'counts': [np.int32, np.int64, np.float32, np.float64],
            'segmentation': [np.int32, np.int64, np.uint32, np.uint64]
        }
        
        dtype_valid = True
        for pattern, valid_types in expected_dtypes.items():
            if pattern in array_name.lower():
                if array.dtype.type not in valid_types:
                    issues.append(f"Unexpected data type: {array.dtype}")
                    dtype_valid = False
                break
        
        integrity_components.append(1.0 if dtype_valid else 0.5)
        
        # Check for memory layout issues
        is_contiguous = array.flags.c_contiguous or array.flags.f_contiguous
        if not is_contiguous:
            issues.append("Array is not contiguous in memory - possible corruption")
            integrity_components.append(0.5)
        else:
            integrity_components.append(1.0)
        
        # Check array shape consistency
        if array.ndim == 0:
            issues.append("Array has zero dimensions")
            integrity_components.append(0.0)
        elif array.size == 0:
            issues.append("Array is empty")
            integrity_components.append(0.0)
        else:
            integrity_components.append(1.0)
        
        metrics['dtype_valid'] = ValidationMetric('dtype_valid', dtype_valid, description="Data type is appropriate")
        metrics['is_contiguous'] = ValidationMetric('is_contiguous', is_contiguous, description="Array memory is contiguous")
        metrics['array_size'] = ValidationMetric('array_size', array.size, description="Total array size")
        
        integrity_score = np.mean(integrity_components)
        
        return {
            'metrics': metrics,
            'issues': issues,
            'integrity_score': integrity_score
        }
    
    def _check_suspicious_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for suspicious data patterns that might indicate corruption."""
        
        metrics = {}
        issues = []
        recommendations = []
        suspicion_scores = []
        
        # Check for repeated values (possible corruption)
        for key, value in data.items():
            if isinstance(value, np.ndarray) and value.size > 100:
                suspicion = self._detect_suspicious_repetition(value, key)
                metrics.update(suspicion['metrics'])
                issues.extend(suspicion['issues'])
                suspicion_scores.append(suspicion['score'])
        
        # Check for impossible values
        if 'coords' in data:
            coords = data['coords']
            if coords.size > 0:
                # Check for negative coordinates (might be valid in some cases)
                negative_frac = np.mean(coords < 0)
                if negative_frac > 0.1:
                    issues.append(f"High fraction of negative coordinates: {negative_frac:.2%}")
                    recommendations.append("Verify coordinate system origin")
                
                # Check for extremely large coordinates
                max_coord = np.max(np.abs(coords))
                if max_coord > 1e6:  # Larger than 1 million units
                    issues.append(f"Extremely large coordinate values: max={max_coord:.0f}")
                    recommendations.append("Check coordinate units and scaling")
        
        # Check ion counts for impossible values
        if 'ion_counts' in data:
            for protein, counts in data['ion_counts'].items():
                if isinstance(counts, np.ndarray) and counts.size > 0:
                    # Negative ion counts are impossible
                    negative_counts = np.sum(counts < 0)
                    if negative_counts > 0:
                        issues.append(f"{protein}: {negative_counts} negative ion counts detected")
                        recommendations.append(f"Investigate {protein} data preprocessing")
                    
                    # Extremely high counts might indicate saturation
                    max_count = np.max(counts)
                    if max_count > 1e6:
                        issues.append(f"{protein}: extremely high count detected: {max_count}")
                        recommendations.append(f"Check {protein} detector saturation")
        
        # Overall suspicion score (lower is more suspicious)
        overall_suspicion = np.mean(suspicion_scores) if suspicion_scores else 1.0
        
        metrics['suspicion_score'] = ValidationMetric(
            'suspicion_score',
            overall_suspicion,
            expected_range=(0.8, 1.0),
            description="Data suspicion score (1.0 = not suspicious)"
        )
        
        return {
            'metrics': metrics,
            'issues': issues,
            'recommendations': recommendations,
            'integrity_score': overall_suspicion
        }
    
    def _detect_suspicious_repetition(self, array: np.ndarray, array_name: str) -> Dict[str, Any]:
        """Detect suspicious repetition patterns in array."""
        
        metrics = {}
        issues = []
        
        # Sample array for efficiency
        sample_size = min(10000, array.size)
        sample = array.flat[:sample_size]
        
        # Check for excessive repetition
        unique_values, counts = np.unique(sample, return_counts=True)
        
        if len(unique_values) > 0:
            max_repetition = np.max(counts)
            repetition_fraction = max_repetition / len(sample)
            
            metrics['max_repetition_fraction'] = ValidationMetric(
                'max_repetition_fraction',
                repetition_fraction,
                expected_range=(0.0, 0.5),
                description=f"Maximum repetition fraction in {array_name}"
            )
            
            if repetition_fraction > 0.8:
                issues.append(f"Extremely high repetition: {repetition_fraction:.1%} of values are identical")
                score = 0.0
            elif repetition_fraction > 0.5:
                issues.append(f"High repetition detected: {repetition_fraction:.1%}")
                score = 0.5
            else:
                score = 1.0
            
            # Check for perfect patterns (e.g., all zeros, all same value)
            if len(unique_values) == 1:
                issues.append(f"All values identical: {unique_values[0]}")
                score = 0.0
            elif len(unique_values) < sample_size * 0.01:  # Less than 1% unique
                issues.append(f"Very low diversity: only {len(unique_values)} unique values")
                score = min(score, 0.3)
        else:
            score = 1.0
        
        return {
            'metrics': metrics,
            'issues': issues,
            'score': score
        }


class IntegrityManager:
    """Manages data integrity across the validation pipeline."""
    
    def __init__(self, integrity_db_path: str = ".integrity_hashes.json"):
        """Initialize integrity manager.
        
        Args:
            integrity_db_path: Path to integrity database file
        """
        self.integrity_db_path = Path(integrity_db_path)
        self.known_hashes = self._load_integrity_db()
        
    def _load_integrity_db(self) -> Dict[str, str]:
        """Load known good hashes from database."""
        
        if self.integrity_db_path.exists():
            try:
                with open(self.integrity_db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading integrity database: {e}")
        
        return {}
    
    def save_integrity_db(self):
        """Save integrity database to disk."""
        
        try:
            with open(self.integrity_db_path, 'w') as f:
                json.dump(self.known_hashes, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving integrity database: {e}")
    
    def register_file(self, file_path: Path, expected_hash: str = None):
        """Register file with known good hash.
        
        Args:
            file_path: Path to file
            expected_hash: Known good hash (computed if not provided)
        """
        
        if expected_hash is None:
            validator = DataIntegrityValidator()
            expected_hash = validator._compute_file_hash_secure(file_path)
        
        self.known_hashes[str(file_path)] = {
            'hash': expected_hash,
            'registered_at': datetime.now().isoformat(),
            'file_size': file_path.stat().st_size
        }
        
        self.save_integrity_db()
        logger.info(f"Registered file integrity: {file_path}")
    
    def verify_file(self, file_path: Path) -> bool:
        """Verify file integrity against known hash.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is verified
        """
        
        file_key = str(file_path)
        if file_key not in self.known_hashes:
            return False
        
        validator = DataIntegrityValidator()
        current_hash = validator._compute_file_hash_secure(file_path)
        expected_hash = self.known_hashes[file_key]['hash']
        
        return current_hash == expected_hash
    
    def get_expected_hash(self, file_path: Path) -> Optional[str]:
        """Get expected hash for file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Expected hash or None if not registered
        """
        
        file_key = str(file_path)
        if file_key in self.known_hashes:
            return self.known_hashes[file_key]['hash']
        return None