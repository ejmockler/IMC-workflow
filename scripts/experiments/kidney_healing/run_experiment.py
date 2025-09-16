#!/usr/bin/env python3
"""
Kidney healing experiment runner using the generalized framework.

This script is a convenience wrapper around the main run_experiment.py
specifically configured for kidney healing studies.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import subprocess


def main():
    """Run kidney healing experiment using the generalized framework."""
    # Path to the main experiment runner
    main_script = project_root / 'run_experiment.py'
    
    # Force kidney healing experiment type
    cmd = [
        sys.executable, str(main_script),
        '--experiment-type', 'kidney_healing',
        '--publication',  # Enable publication quality by default
        '--verbose'
    ]
    
    # Pass through any additional arguments
    cmd.extend(sys.argv[1:])
    
    print("Running kidney healing experiment using generalized framework...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Execute the main script
    result = subprocess.run(cmd, cwd=project_root)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()