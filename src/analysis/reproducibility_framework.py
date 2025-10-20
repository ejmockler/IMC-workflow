"""
KISS Tolerance-Based Reproducibility Framework

MISSION: Build realistic numerical reproducibility - NOT bit-perfect fantasy.
CORE PRINCIPLE: Numerical equivalence within scientific precision tolerances.

This framework provides:
1. Deterministic environment setup (single-thread BLAS, fixed seeds)
2. Numerical tolerance validation using scientific precision
3. Environment fingerprinting for debugging
4. Simple pass/fail reproducibility reporting

REALISTIC GOALS:
- Numerical equivalence within 1e-10 relative tolerance
- Same architecture reproducibility guaranteed  
- Environment capture for debugging
- Fast validation without bureaucracy
"""

import os
import sys
import json
import time
import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import pandas as pd


@dataclass
class EnvironmentFingerprint:
    """Capture environment state that affects reproducibility."""
    
    # Core system info
    python_version: str
    platform_system: str
    platform_release: str
    platform_machine: str
    
    # Critical environment variables
    omp_num_threads: Optional[str]
    mkl_num_threads: Optional[str] 
    openblas_num_threads: Optional[str]
    
    # NumPy/SciPy BLAS info
    numpy_version: str
    numpy_blas_info: Dict[str, Any]
    
    # Package versions
    package_versions: Dict[str, str]
    
    # Random state info
    numpy_random_state: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_hash(self) -> str:
        """Generate hash of environment for quick comparison."""
        env_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(env_str.encode()).hexdigest()


@dataclass 
class ReproducibilityResult:
    """Results of reproducibility validation."""
    
    is_reproducible: bool
    tolerance_used: float
    max_difference: float
    failed_keys: List[str]
    environment_fingerprint: EnvironmentFingerprint
    validation_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['environment_fingerprint'] = self.environment_fingerprint.to_dict()
        return result


class ReproducibilityFramework:
    """
    KISS Tolerance-Based Reproducibility Framework
    
    Core workflow:
    1. framework = ReproducibilityFramework(seed=42)
    2. framework.ensure_deterministic_env()
    3. result1 = run_analysis(data, config)
    4. result2 = run_analysis(data, config) 
    5. is_reproducible = framework.validate_reproducibility(result1, result2)
    6. report = framework.generate_reproducibility_report()
    """
    
    def __init__(self, 
                 seed: int = 42,
                 rtol: float = 1e-10,
                 atol: float = 1e-12):
        """
        Initialize reproducibility framework.
        
        Args:
            seed: Global random seed for reproducibility
            rtol: Relative tolerance for numerical comparisons
            atol: Absolute tolerance for numerical comparisons
        """
        self.seed = seed
        self.rtol = rtol
        self.atol = atol
        self.validation_results: List[ReproducibilityResult] = []
        self._original_env: Dict[str, Optional[str]] = {}
        
    def capture_environment(self) -> EnvironmentFingerprint:
        """
        Capture current environment state that affects reproducibility.
        
        Returns:
            EnvironmentFingerprint containing environment state
        """
        # Get BLAS info safely
        try:
            import numpy as np
            blas_info = np.__config__.show()
            # Parse the BLAS info string into a dictionary
            blas_dict = {}
            if hasattr(np.__config__, 'blas_opt_info'):
                blas_dict['blas_opt_info'] = dict(np.__config__.blas_opt_info)
            if hasattr(np.__config__, 'lapack_opt_info'):  
                blas_dict['lapack_opt_info'] = dict(np.__config__.lapack_opt_info)
        except Exception:
            blas_dict = {"error": "Could not retrieve BLAS info"}
        
        # Get package versions
        package_versions = {}
        important_packages = [
            'numpy', 'scipy', 'pandas', 'scikit-learn', 
            'skimage', 'sklearn', 'joblib'
        ]
        
        for pkg in important_packages:
            try:
                mod = __import__(pkg)
                package_versions[pkg] = getattr(mod, '__version__', 'unknown')
            except ImportError:
                package_versions[pkg] = 'not_installed'
        
        # Get random state info
        random_state_info = None
        try:
            # Get current numpy random state
            random_state_info = str(np.random.get_state()[1][:5])  # First 5 elements for brevity
        except Exception:
            random_state_info = "unavailable"
        
        return EnvironmentFingerprint(
            python_version=sys.version,
            platform_system=platform.system(),
            platform_release=platform.release(),
            platform_machine=platform.machine(),
            omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
            mkl_num_threads=os.environ.get('MKL_NUM_THREADS'),
            openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
            numpy_version=np.__version__,
            numpy_blas_info=blas_dict,
            package_versions=package_versions,
            numpy_random_state=random_state_info
        )
    
    def ensure_deterministic_env(self) -> None:
        """
        Set environment variables for deterministic execution.
        
        Forces single-threaded BLAS operations and stable sorts to ensure
        reproducibility across runs on the same architecture.
        """
        # Store original environment for restoration
        env_vars = [
            'OMP_NUM_THREADS', 
            'MKL_NUM_THREADS', 
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'
        ]
        
        for var in env_vars:
            self._original_env[var] = os.environ.get(var)
            
        # Set deterministic environment
        deterministic_env = {
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 
            'OPENBLAS_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'VECLIB_MAXIMUM_THREADS': '1'
        }
        
        for var, value in deterministic_env.items():
            os.environ[var] = value
            
        # Set global random seeds
        np.random.seed(self.seed)
        
        # Try to set other random seeds if packages are available
        try:
            import random
            random.seed(self.seed)
        except ImportError:
            pass
            
        # Warn about limitations
        warnings.warn(
            "Deterministic environment set. Note: GPU operations and some "
            "multithreaded libraries may still introduce non-determinism.",
            UserWarning
        )
    
    def restore_environment(self) -> None:
        """Restore original environment variables."""
        for var, original_value in self._original_env.items():
            if original_value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = original_value
    
    def _compare_values(self, 
                       val1: Any, 
                       val2: Any, 
                       key_path: str = "") -> Tuple[bool, float]:
        """
        Compare two values with appropriate tolerance.
        
        Args:
            val1, val2: Values to compare
            key_path: Path to this value in nested structure
            
        Returns:
            Tuple of (is_equal, max_difference)
        """
        # Handle None values
        if val1 is None and val2 is None:
            return True, 0.0
        if val1 is None or val2 is None:
            return False, float('inf')
            
        # Handle numpy arrays
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if val1.shape != val2.shape:
                return False, float('inf')
            if val1.size == 0 and val2.size == 0:
                return True, 0.0
            try:
                is_close = np.allclose(val1, val2, rtol=self.rtol, atol=self.atol)
                max_diff = np.max(np.abs(val1 - val2)) if val1.size > 0 else 0.0
                return is_close, float(max_diff)
            except (TypeError, ValueError):
                # Handle non-numeric arrays
                return np.array_equal(val1, val2), 0.0
                
        # Handle pandas DataFrames
        if isinstance(val1, pd.DataFrame) and isinstance(val2, pd.DataFrame):
            if val1.shape != val2.shape:
                return False, float('inf')
            if not val1.columns.equals(val2.columns):
                return False, float('inf')
            
            max_diff = 0.0
            for col in val1.columns:
                try:
                    if pd.api.types.is_numeric_dtype(val1[col]):
                        col_close = np.allclose(val1[col], val2[col], 
                                              rtol=self.rtol, atol=self.atol)
                        if not col_close:
                            return False, float('inf')
                        col_diff = np.max(np.abs(val1[col] - val2[col]))
                        max_diff = max(max_diff, col_diff)
                    else:
                        if not val1[col].equals(val2[col]):
                            return False, float('inf')
                except Exception:
                    return False, float('inf')
            return True, max_diff
            
        # Handle dictionaries recursively
        if isinstance(val1, dict) and isinstance(val2, dict):
            if set(val1.keys()) != set(val2.keys()):
                return False, float('inf')
            
            max_diff = 0.0
            for key in val1.keys():
                sub_path = f"{key_path}.{key}" if key_path else key
                is_equal, sub_diff = self._compare_values(val1[key], val2[key], sub_path)
                if not is_equal:
                    return False, float('inf')
                max_diff = max(max_diff, sub_diff)
            return True, max_diff
            
        # Handle lists/tuples
        if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            if len(val1) != len(val2):
                return False, float('inf')
            
            max_diff = 0.0
            for i, (item1, item2) in enumerate(zip(val1, val2)):
                sub_path = f"{key_path}[{i}]" if key_path else f"[{i}]"
                is_equal, sub_diff = self._compare_values(item1, item2, sub_path)
                if not is_equal:
                    return False, float('inf')
                max_diff = max(max_diff, sub_diff)
            return True, max_diff
            
        # Handle numeric scalars
        if isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
            try:
                is_close = np.allclose([val1], [val2], rtol=self.rtol, atol=self.atol)
                diff = abs(float(val1) - float(val2))
                return is_close, diff
            except (TypeError, ValueError):
                return val1 == val2, 0.0
                
        # Handle other types with exact equality
        return val1 == val2, 0.0
    
    def validate_reproducibility(self, 
                               result1: Dict[str, Any], 
                               result2: Dict[str, Any]) -> ReproducibilityResult:
        """
        Validate that two analysis results are numerically equivalent.
        
        Args:
            result1, result2: Analysis results to compare
            
        Returns:
            ReproducibilityResult with validation outcome
        """
        failed_keys = []
        max_difference = 0.0
        
        # Compare all keys in both results
        all_keys = set(result1.keys()) | set(result2.keys())
        
        for key in all_keys:
            if key not in result1 or key not in result2:
                failed_keys.append(f"{key} (missing in one result)")
                max_difference = float('inf')
                continue
                
            is_equal, diff = self._compare_values(result1[key], result2[key], key)
            if not is_equal:
                failed_keys.append(key)
            max_difference = max(max_difference, diff)
        
        # Create result
        result = ReproducibilityResult(
            is_reproducible=len(failed_keys) == 0,
            tolerance_used=self.rtol,
            max_difference=max_difference,
            failed_keys=failed_keys,
            environment_fingerprint=self.capture_environment(),
            validation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.validation_results.append(result)
        return result
    
    def generate_reproducibility_report(self, 
                                       output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive reproducibility report.
        
        Args:
            output_path: Optional path to save report as JSON
            
        Returns:
            Dictionary containing full reproducibility report
        """
        if not self.validation_results:
            return {"error": "No validation results available"}
            
        latest_result = self.validation_results[-1]
        
        # Generate summary statistics
        all_results = [r.is_reproducible for r in self.validation_results]
        success_rate = sum(all_results) / len(all_results)
        
        report = {
            "reproducibility_framework": {
                "version": "1.0.0",
                "seed": self.seed,
                "tolerances": {
                    "relative_tolerance": self.rtol,
                    "absolute_tolerance": self.atol
                }
            },
            "latest_validation": latest_result.to_dict(),
            "validation_history": {
                "total_validations": len(self.validation_results),
                "success_rate": success_rate,
                "recent_results": [r.is_reproducible for r in self.validation_results[-10:]]
            },
            "environment": latest_result.environment_fingerprint.to_dict(),
            "recommendations": self._generate_recommendations(latest_result)
        }
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
        return report
    
    def _generate_recommendations(self, 
                                 result: ReproducibilityResult) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if not result.is_reproducible:
            recommendations.append(
                f"REPRODUCIBILITY FAILED: {len(result.failed_keys)} keys differ"
            )
            recommendations.append(
                f"Maximum difference: {result.max_difference:.2e}"
            )
            recommendations.append(
                f"Failed keys: {', '.join(result.failed_keys[:5])}"
            )
            
            # Environment-specific recommendations
            env = result.environment_fingerprint
            if env.omp_num_threads != '1':
                recommendations.append(
                    "Set OMP_NUM_THREADS=1 for deterministic BLAS operations"
                )
            if env.mkl_num_threads != '1':
                recommendations.append(
                    "Set MKL_NUM_THREADS=1 for deterministic Intel MKL operations"
                )
                
        else:
            recommendations.append("REPRODUCIBILITY PASSED: Results are numerically equivalent")
            recommendations.append(
                f"Maximum difference: {result.max_difference:.2e} (within tolerance)"
            )
            
        return recommendations


def run_reproducibility_test(analysis_func, 
                           data: Any,
                           config: Any,
                           n_runs: int = 2,
                           seed: int = 42,
                           rtol: float = 1e-10) -> ReproducibilityResult:
    """
    Convenience function to test reproducibility of an analysis function.
    
    Args:
        analysis_func: Function to test (should accept data and config)
        data: Input data for analysis
        config: Configuration for analysis
        n_runs: Number of runs to compare (minimum 2)
        seed: Random seed for reproducibility
        rtol: Relative tolerance for comparisons
        
    Returns:
        ReproducibilityResult for the validation
    """
    framework = ReproducibilityFramework(seed=seed, rtol=rtol)
    framework.ensure_deterministic_env()
    
    try:
        # Run analysis multiple times
        results = []
        for i in range(n_runs):
            # Reset random state for each run
            np.random.seed(seed)
            result = analysis_func(data, config)
            results.append(result)
        
        # Compare first two results
        validation_result = framework.validate_reproducibility(results[0], results[1])
        
        # If we have more than 2 results, verify they all match
        if n_runs > 2:
            for i in range(2, n_runs):
                additional_validation = framework.validate_reproducibility(results[0], results[i])
                if not additional_validation.is_reproducible:
                    validation_result.is_reproducible = False
                    validation_result.failed_keys.extend(additional_validation.failed_keys)
                    
    finally:
        framework.restore_environment()
        
    return validation_result


if __name__ == "__main__":
    # Example usage
    print("Reproducibility Framework - Example Usage")
    print("=" * 50)
    
    # Create framework
    framework = ReproducibilityFramework(seed=42)
    
    # Capture environment
    env = framework.capture_environment()
    print(f"Environment hash: {env.to_hash()}")
    print(f"Python version: {env.python_version.split()[0]}")
    print(f"NumPy version: {env.numpy_version}")
    
    # Set deterministic environment  
    framework.ensure_deterministic_env()
    
    # Create some test data
    np.random.seed(42)
    data1 = {
        'values': np.random.randn(100),
        'coords': np.random.rand(100, 2),
        'metadata': {'n_points': 100}
    }
    
    np.random.seed(42)  # Reset seed
    data2 = {
        'values': np.random.randn(100),
        'coords': np.random.rand(100, 2), 
        'metadata': {'n_points': 100}
    }
    
    # Validate reproducibility
    result = framework.validate_reproducibility(data1, data2)
    
    print(f"\nReproducibility test: {'PASSED' if result.is_reproducible else 'FAILED'}")
    print(f"Max difference: {result.max_difference:.2e}")
    
    # Generate report
    report = framework.generate_reproducibility_report()
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    # Restore environment
    framework.restore_environment()
    print("\nOriginal environment restored.")