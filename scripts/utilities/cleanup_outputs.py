#!/usr/bin/env python3
"""
Cleanup script to remove all output files
Useful before git operations or when starting fresh
"""

import os
import shutil
from pathlib import Path


def cleanup_outputs():
    """Remove all output files and directories"""
    print("🧹 Cleaning up output files...")
    
    # Directories to remove
    output_dirs = [
        'results',
        'plots', 
        'demo_results',
        'test_output',
        'final_demo',
        'final_test',
        'universe_test',
        'demo_output',
        'output_plots',
        'logs'
    ]
    
    # File patterns to remove
    file_patterns = [
        '*.png',
        '*.pdf', 
        '*analysis*.json',
        '*results*.json',
        '*experimental*.json',
        '*network*.json',
        'batch_experimental_results.json',
        '*.log',
        '*.pkl',
        '*.pickle',
        '*insights*.md'
    ]
    
    removed_count = 0
    
    # Remove directories
    for dir_name in output_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"   Removing directory: {dir_name}")
            shutil.rmtree(dir_path)
            removed_count += 1
    
    # Remove files by pattern
    for pattern in file_patterns:
        for file_path in Path('.').glob(pattern):
            if file_path.name != 'config.json':  # Preserve config
                print(f"   Removing file: {file_path}")
                file_path.unlink()
                removed_count += 1
    
    # Remove files in data directory (analysis outputs)
    data_dir = Path('data/241218_IMC_Alun')
    if data_dir.exists():
        for file_path in data_dir.glob('*analysis*.json'):
            print(f"   Removing analysis file: {file_path}")
            file_path.unlink()
            removed_count += 1
    
    print(f"✅ Cleanup complete! Removed {removed_count} items")
    
    # Show what's left
    remaining_files = []
    for pattern in ['*.png', '*.json', '*.txt']:
        remaining_files.extend(list(Path('.').glob(pattern)))
    
    # Filter out expected files
    expected_files = {'config.json', '.vscode/launch.json'}
    unexpected_files = [f for f in remaining_files 
                       if str(f) not in expected_files and f.name != 'config.json']
    
    if unexpected_files:
        print(f"⚠️  {len(unexpected_files)} files still present:")
        for f in unexpected_files[:10]:  # Show first 10
            print(f"   {f}")
        if len(unexpected_files) > 10:
            print(f"   ... and {len(unexpected_files) - 10} more")
    else:
        print("✨ All output files removed!")


def main():
    """Main cleanup function"""
    response = input("🗑️  Remove all output files and results? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        cleanup_outputs()
    else:
        print("Cleanup cancelled.")


if __name__ == '__main__':
    main()