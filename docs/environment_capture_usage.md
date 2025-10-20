# Environment Capture Usage Guide

## Overview

The Environment Capture utilities provide KISS (Keep It Simple, Stupid) environment fingerprinting for IMC analysis reproducibility. This system captures **what matters for numerical results**, not everything.

## Core Functions

### Quick Start (2 lines of code)

```python
from src.analysis.environment_capture import capture_execution_environment, save_environment_snapshot

# Capture current environment
env_info = capture_execution_environment('my_analysis_001')

# Save snapshot for reproducibility 
save_environment_snapshot('my_analysis_env.json', env_info)
```

### What Gets Captured

#### System Information
- OS, architecture, Python version
- CPU cores, memory (affects performance)
- Platform details for compatibility

#### Computational Environment  
- BLAS/LAPACK backend (affects numerical results)
- Threading configuration (OMP, MKL, OpenBLAS)
- Random seed status
- GPU availability

#### Critical Package Versions
- NumPy, SciPy, Pandas (core scientific computing)
- Scikit-learn, Scikit-image (machine learning/image processing)
- Statsmodels, Numba (statistical analysis/performance)
- Optional: H5PY, PyArrow, HDBSCAN, Leiden

## Integration Patterns

### With Analysis Pipeline

```python
from src.analysis.environment_capture import EnvironmentCapture

def run_analysis_with_env_tracking(analysis_id: str):
    # Capture environment at start
    env_capture = EnvironmentCapture(analysis_id)
    env_info = env_capture.capture_execution_environment()
    
    # Save environment snapshot
    env_capture.save_environment_snapshot(f"{analysis_id}_env.json")
    
    # Add environment hash to results
    results = run_your_analysis()
    results['environment_hash'] = env_info.fingerprint_hash
    
    return results
```

### With Provenance Tracker

```python
from src.analysis.provenance_tracker import ProvenanceTracker
from src.analysis.environment_capture import EnvironmentCapture

tracker = ProvenanceTracker('analysis_001')
env_capture = EnvironmentCapture('analysis_001')

# Capture and log environment
env_info = env_capture.capture_execution_environment()
tracker.log_parameter_decision(
    parameter_name="execution_environment",
    parameter_value=env_info.fingerprint_hash,
    reasoning="Environment captured for reproducibility",
    evidence={
        "os": env_info.system_info.os_name,
        "blas": env_info.computational_env.blas_backend,
        "deterministic": env_info.computational_env.omp_num_threads == '1'
    }
)
```

### With Reproducibility Framework

```python
from src.analysis.reproducibility_framework import ReproducibilityFramework
from src.analysis.environment_capture import EnvironmentCapture

# Capture environment before/after deterministic setup
env_capture = EnvironmentCapture('repro_test')
env_before = env_capture.capture_execution_environment()

# Set deterministic environment
repro = ReproducibilityFramework(seed=42)
repro.ensure_deterministic_env()

# Capture environment after
env_after = env_capture.capture_execution_environment(force_refresh=True)

# Compare environments
from src.analysis.environment_capture import validate_environment_compatibility
compatibility = validate_environment_compatibility(env_before, env_after)
```

## Environment Comparison

### Check Compatibility Between Runs

```python
from src.analysis.environment_capture import validate_environment_compatibility

# Load two environment snapshots
env1_path = "analysis_001_env.json"
env2_path = "analysis_002_env.json"

compatibility = validate_environment_compatibility(env1_path, env2_path, strict=False)

if not compatibility['is_compatible']:
    print("Environments incompatible:")
    for issue in compatibility['incompatibilities']:
        print(f"  - {issue}")
    
    print("Recommendations:")
    for rec in compatibility['recommendations']:
        print(f"  - {rec}")
```

### Strict vs Lenient Comparison

- **Lenient** (`strict=False`): Allows minor version differences, focuses on major compatibility
- **Strict** (`strict=True`): Requires exact version matches for all packages

## Environment Reports

### Generate Human-Readable Report

```python
from src.analysis.environment_capture import EnvironmentCapture

capture = EnvironmentCapture('my_analysis')
env_info = capture.capture_execution_environment()

# Generate markdown report
report = capture.generate_environment_report(env_info)

# Save report
with open('environment_report.md', 'w') as f:
    f.write(report)
```

### Report Contents

- System information (OS, Python, hardware)
- Computational environment (BLAS, threading)
- Package versions
- Reproducibility recommendations

## Key Features

### KISS Principle
- **2 lines** for basic usage
- **No configuration** required
- **Graceful degradation** when packages missing
- **Focus on what matters** for numerical results

### Reproducibility Focus
- Captures BLAS backend (affects numerical results)
- Tracks threading configuration (affects determinism)
- Records random seed status
- Identifies version incompatibilities

### Integration Ready
- Works with existing provenance tracking
- Integrates with reproducibility framework
- Compatible with analysis manifests
- Standalone operation when needed

## File Formats

### Environment Snapshot JSON Structure

```json
{
  "metadata": {
    "capture_tool": "IMC Environment Capture v1.0",
    "capture_timestamp": "2025-10-04T...",
    "analysis_id": "my_analysis",
    "fingerprint_hash": "d1c3913de7aac5f6"
  },
  "environment": {
    "system_info": { ... },
    "computational_env": { ... },
    "dependency_versions": { ... }
  },
  "summary": {
    "os": "Darwin 24.6.0",
    "python": "3.13.3",
    "blas_backend": "mkl",
    "deterministic_setup": true
  }
}
```

## Best Practices

### When to Capture Environment

1. **At analysis start** - Capture before any computation
2. **After environment setup** - Capture after setting deterministic flags
3. **Before critical computations** - Capture before BLAS-heavy operations
4. **For result validation** - Capture when creating baseline results

### Environment Fingerprinting

- **Hash changes** when environment affects results
- **Same hash** for compatible environments
- **Use hash** for quick environment comparison
- **Store hash** with analysis results

### Reproducibility Setup

```python
# Recommended pattern for reproducible analysis
from src.analysis.environment_capture import EnvironmentCapture
from src.analysis.reproducibility_framework import ReproducibilityFramework

def reproducible_analysis(analysis_id: str, seed: int = 42):
    # 1. Capture original environment
    env_capture = EnvironmentCapture(analysis_id)
    env_original = env_capture.capture_execution_environment()
    
    # 2. Set deterministic environment
    repro = ReproducibilityFramework(seed=seed)
    repro.ensure_deterministic_env()
    
    # 3. Capture deterministic environment
    env_deterministic = env_capture.capture_execution_environment(force_refresh=True)
    
    # 4. Save both environments
    env_capture.save_environment_snapshot(f"{analysis_id}_env_original.json", env_original)
    env_capture.save_environment_snapshot(f"{analysis_id}_env_deterministic.json", env_deterministic)
    
    try:
        # 5. Run analysis
        results = your_analysis_function()
        results['environment_hash'] = env_deterministic.fingerprint_hash
        return results
    finally:
        # 6. Restore original environment
        repro.restore_environment()
```

## Troubleshooting

### Missing Dependencies

Environment capture works even when scientific packages are missing:
- Records "not_installed" for missing packages
- Still captures system and Python information
- Provides installation recommendations

### BLAS Backend Detection

If BLAS backend shows as "Unknown":
- NumPy may not be installed
- BLAS configuration may be non-standard
- Check NumPy installation: `python -c "import numpy; print(numpy.__config__.show())"`

### Threading Configuration

For deterministic results:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

### Memory Detection

If memory shows as "Unknown":
- Install `psutil` for accurate memory detection
- Memory detection uses platform-specific fallbacks
- Not critical for reproducibility

## Examples

See `examples/environment_capture_examples.py` for comprehensive integration examples.

## API Reference

See `src/analysis/environment_capture.py` docstrings for complete API documentation.