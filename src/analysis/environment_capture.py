"""
Environment Capture Utilities - KISS Implementation

MISSION: Capture execution environment for reproducibility and debugging.
FOCUS: What matters for IMC analysis numerical results and system compatibility.

This module provides:
1. System fingerprinting (OS, hardware, BLAS backends)
2. Dependency versions for critical packages
3. Computational environment (threading, determinism flags)
4. Environment comparison and compatibility checks
5. Integration with analysis manifest and provenance systems

CORE PRINCIPLE: Capture what affects results, not everything.
"""

import os
import sys
import json
import time
import platform
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

# Import numpy safely
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


@dataclass
class SystemInfo:
    """Core system information that affects numerical computations."""
    
    # Operating system details
    os_name: str
    os_version: str
    os_release: str
    architecture: str
    machine: str
    
    # Python environment
    python_version: str
    python_executable: str
    python_implementation: str
    
    # Hardware that affects computation
    cpu_count: Optional[int]
    total_memory_gb: Optional[float]
    
    # Platform-specific details
    platform_processor: Optional[str]
    platform_platform: str


@dataclass 
class ComputationalEnvironment:
    """Computational environment settings affecting determinism."""
    
    # Threading configuration
    omp_num_threads: Optional[str]
    mkl_num_threads: Optional[str]
    openblas_num_threads: Optional[str]
    numexpr_num_threads: Optional[str]
    veclib_maximum_threads: Optional[str]
    
    # BLAS/LAPACK backend information
    numpy_blas_info: Dict[str, Any]
    blas_backend: Optional[str]
    
    # Random state and determinism
    numpy_random_seed_set: bool
    numpy_random_state_info: Optional[str]
    
    # GPU availability (if relevant)
    cuda_available: bool
    gpu_devices: List[str]


@dataclass
class DependencyVersions:
    """Critical package versions for IMC analysis."""
    
    # Core scientific computing
    numpy_version: str
    pandas_version: str
    scipy_version: str
    
    # Machine learning
    sklearn_version: str
    
    # Image processing
    skimage_version: str
    
    # Statistical packages
    statsmodels_version: str
    
    # Performance packages
    numba_version: str
    
    # Optional packages (marked as such if missing)
    h5py_version: Optional[str] = None
    pyarrow_version: Optional[str] = None
    hdbscan_version: Optional[str] = None
    leidenalg_version: Optional[str] = None
    igraph_version: Optional[str] = None


@dataclass
class EnvironmentFingerprint:
    """Complete environment fingerprint for reproducibility."""
    
    # Core components
    system_info: SystemInfo
    computational_env: ComputationalEnvironment
    dependency_versions: DependencyVersions
    
    # Metadata
    capture_timestamp: str
    analysis_id: Optional[str] = None
    fingerprint_hash: Optional[str] = None
    
    def __post_init__(self):
        """Generate fingerprint hash after initialization."""
        if self.fingerprint_hash is None:
            self.fingerprint_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of environment for quick comparison."""
        # Create a deterministic representation
        env_dict = asdict(self)
        env_dict.pop('capture_timestamp', None)  # Exclude timestamp from hash
        env_dict.pop('fingerprint_hash', None)   # Exclude self-reference
        env_dict.pop('analysis_id', None)        # Exclude analysis ID
        
        env_str = json.dumps(env_dict, sort_keys=True, default=str)
        return hashlib.sha256(env_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def is_compatible_with(self, other: 'EnvironmentFingerprint', 
                          strict: bool = False) -> Tuple[bool, List[str]]:
        """
        Check compatibility with another environment.
        
        Args:
            other: Other environment fingerprint to compare
            strict: If True, require exact version matches
            
        Returns:
            Tuple of (is_compatible, incompatibility_reasons)
        """
        incompatibilities = []
        
        # Check OS compatibility
        if self.system_info.os_name != other.system_info.os_name:
            incompatibilities.append(f"OS mismatch: {self.system_info.os_name} vs {other.system_info.os_name}")
        
        if self.system_info.architecture != other.system_info.architecture:
            incompatibilities.append(f"Architecture mismatch: {self.system_info.architecture} vs {other.system_info.architecture}")
        
        # Check BLAS backend compatibility
        if self.computational_env.blas_backend != other.computational_env.blas_backend:
            incompatibilities.append(f"BLAS backend mismatch: {self.computational_env.blas_backend} vs {other.computational_env.blas_backend}")
        
        # Check critical package versions
        critical_packages = ['numpy_version', 'scipy_version', 'sklearn_version']
        
        for package in critical_packages:
            self_version = getattr(self.dependency_versions, package)
            other_version = getattr(other.dependency_versions, package)
            
            if strict:
                if self_version != other_version:
                    incompatibilities.append(f"{package} version mismatch: {self_version} vs {other_version}")
            else:
                # Check major version compatibility
                try:
                    self_major = self_version.split('.')[0]
                    other_major = other_version.split('.')[0]
                    if self_major != other_major:
                        incompatibilities.append(f"{package} major version mismatch: {self_major} vs {other_major}")
                except (AttributeError, IndexError):
                    incompatibilities.append(f"{package} version format issue: {self_version} vs {other_version}")
        
        return len(incompatibilities) == 0, incompatibilities


class EnvironmentCapture:
    """
    Environment capture utility for IMC analysis reproducibility.
    
    FOCUS: Capture environment state that affects numerical results.
    KISS PRINCIPLE: Simple, fast, and focused on what matters.
    """
    
    def __init__(self, analysis_id: Optional[str] = None):
        """
        Initialize environment capture utility.
        
        Args:
            analysis_id: Optional analysis identifier for tracking
        """
        self.analysis_id = analysis_id
        self._cached_fingerprint: Optional[EnvironmentFingerprint] = None
    
    def capture_system_info(self) -> SystemInfo:
        """Capture core system information."""
        # Get memory info safely
        total_memory_gb = None
        try:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Try alternative methods
            try:
                if platform.system() == "Darwin":  # macOS
                    result = subprocess.run(['sysctl', 'hw.memsize'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        memory_bytes = int(result.stdout.split(':')[1].strip())
                        total_memory_gb = memory_bytes / (1024**3)
                elif platform.system() == "Linux":
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if line.startswith('MemTotal:'):
                                memory_kb = int(line.split()[1])
                                total_memory_gb = memory_kb / (1024**2)
                                break
            except Exception:
                pass
        
        return SystemInfo(
            os_name=platform.system(),
            os_version=platform.version(),
            os_release=platform.release(),
            architecture=platform.architecture()[0],
            machine=platform.machine(),
            python_version=sys.version,
            python_executable=sys.executable,
            python_implementation=platform.python_implementation(),
            cpu_count=os.cpu_count(),
            total_memory_gb=total_memory_gb,
            platform_processor=platform.processor() or "unknown",
            platform_platform=platform.platform()
        )
    
    def capture_computational_environment(self) -> ComputationalEnvironment:
        """Capture computational environment affecting determinism."""
        # Get BLAS info
        numpy_blas_info = {}
        blas_backend = None
        
        if NUMPY_AVAILABLE:
            try:
                # Try to get BLAS configuration
                if hasattr(np.__config__, 'blas_opt_info'):
                    numpy_blas_info['blas_opt_info'] = dict(np.__config__.blas_opt_info)
                    # Infer backend from library names
                    libraries = numpy_blas_info['blas_opt_info'].get('libraries', [])
                    if any('mkl' in lib.lower() for lib in libraries):
                        blas_backend = 'mkl'
                    elif any('openblas' in lib.lower() for lib in libraries):
                        blas_backend = 'openblas'
                    elif any('blas' in lib.lower() for lib in libraries):
                        blas_backend = 'generic_blas'
                
                if hasattr(np.__config__, 'lapack_opt_info'):
                    numpy_blas_info['lapack_opt_info'] = dict(np.__config__.lapack_opt_info)
            except Exception as e:
                numpy_blas_info = {"error": f"Could not retrieve BLAS info: {str(e)}"}
        else:
            numpy_blas_info = {"error": "NumPy not available"}
        
        # Check random state
        numpy_random_seed_set = False
        numpy_random_state_info = None
        if NUMPY_AVAILABLE:
            try:
                # Check if random state looks like it was set (not the default)
                state = np.random.get_state()
                # Get first few elements of random state for fingerprinting
                numpy_random_state_info = str(state[1][:5].tolist())
                # Heuristic: if state is not at default position, likely seeded
                numpy_random_seed_set = state[2] != 0
            except Exception:
                numpy_random_state_info = "unavailable"
        else:
            numpy_random_state_info = "numpy_not_available"
        
        # Check GPU availability
        cuda_available = False
        gpu_devices = []
        try:
            # Try to detect CUDA without importing heavy libraries
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                cuda_available = True
                gpu_devices = result.stdout.strip().split('\n')
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return ComputationalEnvironment(
            omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
            mkl_num_threads=os.environ.get('MKL_NUM_THREADS'),
            openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
            numexpr_num_threads=os.environ.get('NUMEXPR_NUM_THREADS'),
            veclib_maximum_threads=os.environ.get('VECLIB_MAXIMUM_THREADS'),
            numpy_blas_info=numpy_blas_info,
            blas_backend=blas_backend,
            numpy_random_seed_set=numpy_random_seed_set,
            numpy_random_state_info=numpy_random_state_info,
            cuda_available=cuda_available,
            gpu_devices=gpu_devices
        )
    
    def capture_dependency_versions(self) -> DependencyVersions:
        """Capture versions of critical packages."""
        def get_package_version(package_name: str) -> Optional[str]:
            """Get package version safely."""
            try:
                module = __import__(package_name)
                return getattr(module, '__version__', 'unknown')
            except ImportError:
                return None
        
        # Core packages (required)
        core_versions = {}
        core_packages = ['numpy', 'pandas', 'scipy', 'sklearn', 'skimage', 'statsmodels', 'numba']
        
        for package in core_packages:
            version = get_package_version(package)
            if version is None:
                version = "not_installed"
            core_versions[f"{package}_version"] = version
        
        # Handle scikit-learn special case
        if core_versions['sklearn_version'] == "not_installed":
            # Try alternative import
            try:
                import sklearn
                core_versions['sklearn_version'] = sklearn.__version__
            except ImportError:
                pass
        
        # Handle scikit-image special case  
        if core_versions['skimage_version'] == "not_installed":
            try:
                import skimage
                core_versions['skimage_version'] = skimage.__version__
            except ImportError:
                pass
        
        # Special handling for NumPy if not available globally
        if not NUMPY_AVAILABLE and core_versions['numpy_version'] == "not_installed":
            core_versions['numpy_version'] = "not_available"
        
        # Optional packages
        optional_packages = {
            'h5py': 'h5py_version',
            'pyarrow': 'pyarrow_version', 
            'hdbscan': 'hdbscan_version',
            'leidenalg': 'leidenalg_version',
            'igraph': 'igraph_version'
        }
        
        optional_versions = {}
        for package, version_key in optional_packages.items():
            optional_versions[version_key] = get_package_version(package)
        
        return DependencyVersions(
            **core_versions,
            **optional_versions
        )
    
    def capture_execution_environment(self, 
                                    force_refresh: bool = False) -> EnvironmentFingerprint:
        """
        Capture complete execution environment.
        
        Args:
            force_refresh: If True, capture fresh environment (ignore cache)
            
        Returns:
            Complete environment fingerprint
        """
        if self._cached_fingerprint is not None and not force_refresh:
            return self._cached_fingerprint
        
        system_info = self.capture_system_info()
        computational_env = self.capture_computational_environment()
        dependency_versions = self.capture_dependency_versions()
        
        fingerprint = EnvironmentFingerprint(
            system_info=system_info,
            computational_env=computational_env,
            dependency_versions=dependency_versions,
            capture_timestamp=datetime.now().isoformat(),
            analysis_id=self.analysis_id
        )
        
        self._cached_fingerprint = fingerprint
        return fingerprint
    
    def save_environment_snapshot(self, 
                                output_path: Union[str, Path],
                                env_info: Optional[EnvironmentFingerprint] = None) -> str:
        """
        Save environment snapshot to JSON file.
        
        Args:
            output_path: Path to save environment snapshot
            env_info: Environment info to save (captures fresh if None)
            
        Returns:
            Path to saved file
        """
        if env_info is None:
            env_info = self.capture_execution_environment()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive snapshot
        snapshot = {
            "metadata": {
                "capture_tool": "IMC Environment Capture v1.0",
                "capture_timestamp": env_info.capture_timestamp,
                "analysis_id": env_info.analysis_id,
                "fingerprint_hash": env_info.fingerprint_hash
            },
            "environment": env_info.to_dict(),
            "summary": {
                "os": f"{env_info.system_info.os_name} {env_info.system_info.os_version}",
                "python": env_info.system_info.python_version.split()[0],
                "numpy": env_info.dependency_versions.numpy_version,
                "blas_backend": env_info.computational_env.blas_backend or "unknown",
                "deterministic_setup": all([
                    env_info.computational_env.omp_num_threads == '1',
                    env_info.computational_env.mkl_num_threads == '1',
                    env_info.computational_env.numpy_random_seed_set
                ])
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        
        return str(output_path)
    
    def load_environment_snapshot(self, snapshot_path: Union[str, Path]) -> EnvironmentFingerprint:
        """
        Load environment snapshot from JSON file.
        
        Args:
            snapshot_path: Path to environment snapshot file
            
        Returns:
            Loaded environment fingerprint
        """
        with open(snapshot_path, 'r') as f:
            snapshot = json.load(f)
        
        env_dict = snapshot['environment']
        
        # Reconstruct dataclasses
        system_info = SystemInfo(**env_dict['system_info'])
        computational_env = ComputationalEnvironment(**env_dict['computational_env'])
        dependency_versions = DependencyVersions(**env_dict['dependency_versions'])
        
        return EnvironmentFingerprint(
            system_info=system_info,
            computational_env=computational_env,
            dependency_versions=dependency_versions,
            capture_timestamp=env_dict['capture_timestamp'],
            analysis_id=env_dict.get('analysis_id'),
            fingerprint_hash=env_dict.get('fingerprint_hash')
        )
    
    def validate_environment_compatibility(self, 
                                         env1: EnvironmentFingerprint,
                                         env2: EnvironmentFingerprint,
                                         strict: bool = False) -> Dict[str, Any]:
        """
        Validate compatibility between two environments.
        
        Args:
            env1: First environment fingerprint
            env2: Second environment fingerprint  
            strict: If True, require exact version matches
            
        Returns:
            Compatibility report
        """
        is_compatible, incompatibilities = env1.is_compatible_with(env2, strict=strict)
        
        # Generate detailed comparison
        compatibility_score = 1.0 - (len(incompatibilities) / 10.0)  # Rough scoring
        compatibility_score = max(0.0, min(1.0, compatibility_score))
        
        return {
            "is_compatible": is_compatible,
            "compatibility_score": compatibility_score,
            "incompatibilities": incompatibilities,
            "comparison_details": {
                "env1_hash": env1.fingerprint_hash,
                "env2_hash": env2.fingerprint_hash,
                "timestamps": {
                    "env1": env1.capture_timestamp,
                    "env2": env2.capture_timestamp
                },
                "system_match": env1.system_info.os_name == env2.system_info.os_name,
                "blas_match": env1.computational_env.blas_backend == env2.computational_env.blas_backend,
                "python_version_match": env1.system_info.python_version == env2.system_info.python_version
            },
            "recommendations": self._generate_compatibility_recommendations(incompatibilities)
        }
    
    def _generate_compatibility_recommendations(self, 
                                              incompatibilities: List[str]) -> List[str]:
        """Generate recommendations to improve compatibility."""
        recommendations = []
        
        if not incompatibilities:
            recommendations.append("Environments are compatible for reproducible analysis")
            return recommendations
        
        for issue in incompatibilities:
            if "OS mismatch" in issue:
                recommendations.append("OS differences may affect binary dependencies - verify results")
            elif "Architecture mismatch" in issue:
                recommendations.append("Architecture differences may cause numerical precision variations")
            elif "BLAS backend mismatch" in issue:
                recommendations.append("Different BLAS backends may cause numerical differences - use deterministic environment")
            elif "version mismatch" in issue:
                recommendations.append(f"Package version difference detected: {issue}")
        
        recommendations.append("Consider using containerization (Docker) for exact environment matching")
        return recommendations
    
    def generate_environment_report(self, 
                                  env_info: Optional[EnvironmentFingerprint] = None,
                                  include_recommendations: bool = True) -> str:
        """
        Generate human-readable environment report.
        
        Args:
            env_info: Environment info to report (captures fresh if None)
            include_recommendations: Include setup recommendations
            
        Returns:
            Formatted environment report
        """
        if env_info is None:
            env_info = self.capture_execution_environment()
        
        report_lines = []
        
        # Header
        report_lines.extend([
            "# IMC Analysis Environment Report",
            f"Generated: {env_info.capture_timestamp}",
            f"Analysis ID: {env_info.analysis_id or 'N/A'}",
            f"Environment Hash: {env_info.fingerprint_hash}",
            ""
        ])
        
        # System Information
        report_lines.extend([
            "## System Information",
            f"OS: {env_info.system_info.os_name} {env_info.system_info.os_version}",
            f"Architecture: {env_info.system_info.architecture}",
            f"Machine: {env_info.system_info.machine}",
            f"Python: {env_info.system_info.python_version.split()[0]} ({env_info.system_info.python_implementation})",
            f"CPU Cores: {env_info.system_info.cpu_count}",
            f"Memory: {env_info.system_info.total_memory_gb:.1f} GB" if env_info.system_info.total_memory_gb else "Memory: Unknown",
            ""
        ])
        
        # Computational Environment
        report_lines.extend([
            "## Computational Environment",
            f"BLAS Backend: {env_info.computational_env.blas_backend or 'Unknown'}",
            f"OMP Threads: {env_info.computational_env.omp_num_threads or 'Not set'}",
            f"MKL Threads: {env_info.computational_env.mkl_num_threads or 'Not set'}",
            f"OpenBLAS Threads: {env_info.computational_env.openblas_num_threads or 'Not set'}",
            f"NumPy Random Seed Set: {env_info.computational_env.numpy_random_seed_set}",
            f"CUDA Available: {env_info.computational_env.cuda_available}",
            ""
        ])
        
        # Package Versions
        report_lines.extend([
            "## Critical Package Versions",
            f"NumPy: {env_info.dependency_versions.numpy_version}",
            f"SciPy: {env_info.dependency_versions.scipy_version}",
            f"Pandas: {env_info.dependency_versions.pandas_version}",
            f"Scikit-learn: {env_info.dependency_versions.sklearn_version}",
            f"Scikit-image: {env_info.dependency_versions.skimage_version}",
            f"Statsmodels: {env_info.dependency_versions.statsmodels_version}",
            f"Numba: {env_info.dependency_versions.numba_version}",
            ""
        ])
        
        # Optional packages
        optional_packages = [
            ("H5PY", env_info.dependency_versions.h5py_version),
            ("PyArrow", env_info.dependency_versions.pyarrow_version),
            ("HDBSCAN", env_info.dependency_versions.hdbscan_version),
            ("Leiden Algorithm", env_info.dependency_versions.leidenalg_version),
            ("igraph", env_info.dependency_versions.igraph_version)
        ]
        
        installed_optional = [(name, version) for name, version in optional_packages if version is not None]
        if installed_optional:
            report_lines.append("## Optional Packages")
            for name, version in installed_optional:
                report_lines.append(f"{name}: {version}")
            report_lines.append("")
        
        # Recommendations
        if include_recommendations:
            report_lines.extend([
                "## Reproducibility Recommendations",
                self._generate_reproducibility_recommendations(env_info),
                ""
            ])
        
        return "\n".join(report_lines)
    
    def _generate_reproducibility_recommendations(self, 
                                                env_info: EnvironmentFingerprint) -> str:
        """Generate reproducibility setup recommendations."""
        recommendations = []
        
        # Check deterministic setup
        if not env_info.computational_env.numpy_random_seed_set:
            recommendations.append("- Set NumPy random seed for reproducible results")
        
        if env_info.computational_env.omp_num_threads != '1':
            recommendations.append("- Set OMP_NUM_THREADS=1 for deterministic BLAS operations")
        
        if env_info.computational_env.mkl_num_threads != '1':
            recommendations.append("- Set MKL_NUM_THREADS=1 for deterministic Intel MKL operations")
        
        if env_info.computational_env.openblas_num_threads != '1':
            recommendations.append("- Set OPENBLAS_NUM_THREADS=1 for deterministic OpenBLAS operations")
        
        # Check package versions
        if env_info.dependency_versions.numpy_version.startswith("1."):
            recommendations.append("- Consider upgrading to NumPy 2.x for better performance")
        
        if env_info.dependency_versions.numba_version == "not_installed":
            recommendations.append("- Install Numba for JIT compilation performance improvements")
        
        # GPU recommendations
        if env_info.computational_env.cuda_available:
            recommendations.append("- GPU detected: ensure CUDA determinism flags if using GPU computing")
        
        if not recommendations:
            recommendations.append("- Environment is well-configured for reproducible analysis")
        
        return "\n".join(recommendations)


# Convenience functions for quick usage
def capture_execution_environment(analysis_id: Optional[str] = None) -> EnvironmentFingerprint:
    """
    Quick function to capture current execution environment.
    
    Args:
        analysis_id: Optional analysis identifier
        
    Returns:
        Environment fingerprint
    """
    capture = EnvironmentCapture(analysis_id)
    return capture.capture_execution_environment()


def save_environment_snapshot(output_path: Union[str, Path],
                            env_info: Optional[EnvironmentFingerprint] = None,
                            analysis_id: Optional[str] = None) -> str:
    """
    Quick function to save environment snapshot.
    
    Args:
        output_path: Path to save snapshot
        env_info: Environment info to save (captures fresh if None)
        analysis_id: Analysis identifier if capturing fresh
        
    Returns:
        Path to saved file
    """
    capture = EnvironmentCapture(analysis_id)
    return capture.save_environment_snapshot(output_path, env_info)


def validate_environment_compatibility(env1_path: Union[str, Path, EnvironmentFingerprint],
                                     env2_path: Union[str, Path, EnvironmentFingerprint],
                                     strict: bool = False) -> Dict[str, Any]:
    """
    Quick function to validate environment compatibility.
    
    Args:
        env1_path: Path to first environment snapshot or fingerprint object
        env2_path: Path to second environment snapshot or fingerprint object
        strict: If True, require exact version matches
        
    Returns:
        Compatibility report
    """
    capture = EnvironmentCapture()
    
    # Load environments if needed
    if isinstance(env1_path, (str, Path)):
        env1 = capture.load_environment_snapshot(env1_path)
    else:
        env1 = env1_path
        
    if isinstance(env2_path, (str, Path)):
        env2 = capture.load_environment_snapshot(env2_path)
    else:
        env2 = env2_path
    
    return capture.validate_environment_compatibility(env1, env2, strict=strict)


# Example integration with existing provenance/manifest systems
def integrate_with_provenance_tracker(tracker, env_capture: EnvironmentCapture):
    """
    Integrate environment capture with existing provenance tracker.
    
    Args:
        tracker: ProvenanceTracker instance
        env_capture: EnvironmentCapture instance
    """
    env_info = env_capture.capture_execution_environment()
    
    # Log environment as a critical decision
    tracker.log_parameter_decision(
        parameter_name="execution_environment",
        parameter_value=env_info.fingerprint_hash,
        reasoning="Environment fingerprint captured for reproducibility",
        evidence={
            "os": f"{env_info.system_info.os_name} {env_info.system_info.os_version}",
            "python": env_info.system_info.python_version.split()[0],
            "numpy": env_info.dependency_versions.numpy_version,
            "blas_backend": env_info.computational_env.blas_backend,
            "deterministic_setup": all([
                env_info.computational_env.omp_num_threads == '1',
                env_info.computational_env.mkl_num_threads == '1'
            ])
        },
        severity=tracker.__class__.__dict__.get('DecisionSeverity', type('DS', (), {'CRITICAL': 'critical'})).CRITICAL
    )


if __name__ == "__main__":
    # Example usage
    print("IMC Environment Capture - Example Usage")
    print("=" * 50)
    
    # Capture current environment
    capture = EnvironmentCapture(analysis_id="demo_001")
    env_info = capture.capture_execution_environment()
    
    print(f"Environment Hash: {env_info.fingerprint_hash}")
    print(f"OS: {env_info.system_info.os_name} {env_info.system_info.os_version}")
    print(f"Python: {env_info.system_info.python_version.split()[0]}")
    print(f"NumPy: {env_info.dependency_versions.numpy_version}")
    print(f"BLAS Backend: {env_info.computational_env.blas_backend}")
    
    # Save snapshot
    snapshot_path = save_environment_snapshot("demo_env_snapshot.json", env_info)
    print(f"\nEnvironment snapshot saved to: {snapshot_path}")
    
    # Generate report
    report = capture.generate_environment_report(env_info)
    print(f"\nEnvironment Report:\n{report}")
    
    # Test compatibility with itself (should be perfect match)
    compatibility = validate_environment_compatibility(env_info, env_info)
    print(f"\nSelf-compatibility check: {'PASS' if compatibility['is_compatible'] else 'FAIL'}")
    print(f"Compatibility score: {compatibility['compatibility_score']:.2f}")