#!/usr/bin/env python3
"""
Check for large files before git commit
Helps prevent accidentally committing output files
"""

import os
import subprocess
from pathlib import Path


def get_file_size_mb(filepath):
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except:
        return 0


def check_staged_files():
    """Check staged files for large files"""
    try:
        # Get staged files
        result = subprocess.run(['git', 'diff', '--cached', '--name-only'], 
                              capture_output=True, text=True)
        staged_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        large_files = []
        output_files = []
        
        for file_path in staged_files:
            if not file_path or not os.path.exists(file_path):
                continue
                
            size_mb = get_file_size_mb(file_path)
            
            # Check for large files (>1MB)
            if size_mb > 1:
                large_files.append((file_path, size_mb))
            
            # Check for output file patterns
            if any(pattern in file_path.lower() for pattern in [
                'result', 'output', 'analysis', 'network', 'temporal', 
                'discovery', 'evolution', 'experimental', 'replicate'
            ]):
                if file_path.endswith(('.png', '.json', '.txt', '.md')) and 'config.json' not in file_path:
                    output_files.append(file_path)
        
        return large_files, output_files
        
    except subprocess.CalledProcessError:
        return [], []


def main():
    """Main check function"""
    print("🔍 Checking for large files before commit...")
    
    large_files, output_files = check_staged_files()
    
    if large_files:
        print("\n⚠️  WARNING: Large files detected in staging area:")
        for file_path, size_mb in large_files:
            print(f"   {file_path}: {size_mb:.1f}MB")
        print("\n   These files may be too large for GitHub.")
        print("   Consider adding them to .gitignore if they are output files.")
    
    if output_files:
        print("\n⚠️  WARNING: Potential output files detected:")
        for file_path in output_files:
            print(f"   {file_path}")
        print("\n   These appear to be analysis output files.")
        print("   Consider if they should be in .gitignore instead.")
    
    if large_files or output_files:
        print("\n💡 To exclude these files, you can:")
        print("   1. Add patterns to .gitignore")
        print("   2. Use 'git reset HEAD <file>' to unstage")
        print("   3. Use 'git rm --cached <file>' to untrack")
        return False
    else:
        print("✅ No large files or obvious output files detected.")
        return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)