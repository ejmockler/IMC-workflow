#!/usr/bin/env python3
"""
Fix QualityMetrics constructor calls in test files.
Adds required batch_id and timestamp parameters.
"""

import re
from pathlib import Path
from datetime import datetime

def fix_quality_metrics_in_file(filepath):
    """Fix QualityMetrics calls in a single file."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to match QualityMetrics constructor
    pattern = r'QualityMetrics\(\s*roi_id="([^"]+)",'
    
    # Check if file needs fixing
    if 'QualityMetrics(' not in content:
        return False
    
    # Check if batch_id already present (already fixed)
    if 'batch_id=' in content:
        print(f"  {filepath.name} already fixed")
        return False
    
    # Add batch_id and timestamp after roi_id
    def add_required_params(match):
        roi_id = match.group(1)
        # Extract batch from ROI name or use default
        if 'batch' in roi_id.lower():
            batch_id = 'batch1'
        else:
            batch_id = 'test_batch'
        
        return f'QualityMetrics(\n            roi_id="{roi_id}",\n            batch_id="{batch_id}",\n            timestamp="{datetime.now().isoformat()}",'
    
    new_content = re.sub(pattern, add_required_params, content)
    
    # Write back if changed
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"  Fixed {filepath.name}")
        return True
    
    return False

def main():
    """Fix all test files."""
    test_dir = Path('tests')
    
    test_files = [
        'test_quality_gates.py',
        'test_qc_monitoring.py'
    ]
    
    print("Fixing QualityMetrics constructor calls...")
    
    fixed_count = 0
    for test_file in test_files:
        filepath = test_dir / test_file
        if filepath.exists():
            if fix_quality_metrics_in_file(filepath):
                fixed_count += 1
        else:
            print(f"  {test_file} not found")
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()