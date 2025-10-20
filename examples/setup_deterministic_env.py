#!/usr/bin/env python3
"""
Setup Deterministic Environment Utility

Simple utility to configure environment variables for reproducible IMC analysis.

Usage:
    # As a script
    python setup_deterministic_env.py
    
    # As a module  
    from setup_deterministic_env import setup_deterministic_environment
    setup_deterministic_environment(seed=42)
"""

import os
import sys
import warnings
from typing import Optional, Dict


def setup_deterministic_environment(seed: int = 42, 
                                  verbose: bool = True,
                                  backup_env: bool = True) -> Dict[str, Optional[str]]:
    """
    Configure environment for deterministic numerical computations.
    
    Sets environment variables to force single-threaded BLAS operations
    and configures random seeds for reproducibility.
    
    Args:
        seed: Random seed for numpy/python random
        verbose: Print configuration messages
        backup_env: Return dict of original environment values
        
    Returns:
        Dictionary of original environment values (if backup_env=True)
    """
    
    # Environment variables that affect numerical reproducibility
    deterministic_vars = {
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 
        'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'PYTHONHASHSEED': str(seed)
    }
    
    # Backup original environment
    original_env = {}
    if backup_env:
        for var in deterministic_vars.keys():
            original_env[var] = os.environ.get(var)
    
    # Set deterministic environment
    for var, value in deterministic_vars.items():
        old_value = os.environ.get(var)
        os.environ[var] = value
        
        if verbose and old_value != value:
            if old_value is None:
                print(f"Setting {var}={value}")
            else:
                print(f"Changing {var}: {old_value} → {value}")
    
    # Set random seeds
    try:
        import numpy as np
        np.random.seed(seed)
        if verbose:
            print(f"Set NumPy random seed: {seed}")
    except ImportError:
        if verbose:
            print("NumPy not available - skipping numpy seed")
    
    try:
        import random
        random.seed(seed)
        if verbose:
            print(f"Set Python random seed: {seed}")
    except ImportError:
        pass
    
    # Warning about limitations
    if verbose:
        warnings.warn(
            "Deterministic environment configured. Note: GPU operations, "
            "some multithreaded libraries, and different hardware may still "
            "introduce non-determinism.",
            UserWarning
        )
    
    return original_env


def restore_environment(original_env: Dict[str, Optional[str]], 
                       verbose: bool = True) -> None:
    """
    Restore original environment variables.
    
    Args:
        original_env: Dictionary of original environment values
        verbose: Print restoration messages
    """
    for var, original_value in original_env.items():
        if original_value is None:
            if var in os.environ:
                del os.environ[var]
                if verbose:
                    print(f"Removed {var}")
        else:
            os.environ[var] = original_value
            if verbose:
                print(f"Restored {var}={original_value}")


def check_environment_reproducibility(verbose: bool = True) -> Dict[str, str]:
    """
    Check current environment configuration for reproducibility.
    
    Args:
        verbose: Print environment status
        
    Returns:
        Dictionary of environment status
    """
    
    # Check important environment variables
    important_vars = [
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS', 
        'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'PYTHONHASHSEED'
    ]
    
    env_status = {}
    issues = []
    
    for var in important_vars:
        value = os.environ.get(var, 'NOT_SET')
        env_status[var] = value
        
        if verbose:
            print(f"{var}: {value}")
        
        # Check for potential reproducibility issues
        if var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
            if value not in ['1', 'NOT_SET']:
                issues.append(f"{var}={value} (recommend 1 for determinism)")
    
    # Check NumPy BLAS configuration
    try:
        import numpy as np
        if verbose:
            print(f"NumPy version: {np.__version__}")
            
        # Try to get BLAS info
        try:
            import numpy.distutils.system_info as sysinfo
            blas_info = sysinfo.get_info('blas_opt')
            if blas_info:
                libraries = blas_info.get('libraries', [])
                if verbose:
                    print(f"BLAS libraries: {libraries}")
                env_status['blas_libraries'] = str(libraries)
        except Exception:
            if verbose:
                print("Could not determine BLAS configuration")
            env_status['blas_libraries'] = 'UNKNOWN'
            
    except ImportError:
        if verbose:
            print("NumPy not available")
        env_status['numpy_version'] = 'NOT_INSTALLED'
    
    # Report issues
    if issues:
        env_status['reproducibility_issues'] = issues
        if verbose:
            print("\nPotential reproducibility issues:")
            for issue in issues:
                print(f"  ⚠️  {issue}")
    else:
        env_status['reproducibility_issues'] = []
        if verbose:
            print("\n✅ Environment looks good for reproducibility")
    
    return env_status


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup deterministic environment for reproducible IMC analysis"
    )
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check current environment, do not modify')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimize output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if verbose:
        print("IMC Analysis - Deterministic Environment Setup")
        print("=" * 50)
    
    if args.check_only:
        if verbose:
            print("Checking current environment...")
        env_status = check_environment_reproducibility(verbose=verbose)
        
        # Return appropriate exit code
        if env_status.get('reproducibility_issues', []):
            if verbose:
                print("\nSome reproducibility issues detected.")
            return 1
        else:
            if verbose:
                print("\nEnvironment configured for reproducibility.")
            return 0
    
    else:
        if verbose:
            print(f"Setting up deterministic environment (seed={args.seed})...")
        
        # Setup environment
        original_env = setup_deterministic_environment(
            seed=args.seed, 
            verbose=verbose
        )
        
        if verbose:
            print("\n" + "="*50)
            print("Environment configured for deterministic execution!")
            print("="*50)
            print("\nTo restore original environment in Python:")
            print("from setup_deterministic_env import restore_environment")
            print(f"restore_environment({original_env})")
        
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)