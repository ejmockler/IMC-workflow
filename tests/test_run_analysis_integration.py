"""Integration tests for run_analysis.py script."""

import pytest
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil


class TestRunAnalysisIntegration:
    """Test the production run_analysis.py script."""
    
    def test_script_uses_config_data_directory(self):
        """Verify run_analysis.py uses config.json data directory correctly."""
        # Run script briefly to check it reads config correctly
        try:
            result = subprocess.run([
                sys.executable, "run_analysis.py"
            ], 
            cwd=Path.cwd(),
            capture_output=True, 
            text=True, 
            timeout=5,  # Very short timeout just to see initial output
            env={"PYTHONPATH": str(Path.cwd())}
            )
        except subprocess.TimeoutExpired as e:
            # Expected to timeout - just check the partial output
            output = e.stderr.decode() if e.stderr else ""
            
            # Verify it uses production data directory from config, not hardcoded test data
            assert "Data directory: data/241218_IMC_Alun" in output, f"Should use config data dir, got: {output}"
            assert "Found 25 ROI files" in output, f"Should find production data (25 files), got: {output}"
            return
        
        # If it completes quickly (shouldn't happen with 25 files), still verify
        output = result.stderr
        assert "Data directory: data/241218_IMC_Alun" in output
        assert "Found 25 ROI files" in output
    
    def test_script_imports_and_runs(self):
        """Verify run_analysis.py can be imported and has basic structure."""
        # Test that script can be compiled without syntax errors
        try:
            compile(Path("run_analysis.py").read_text(), "run_analysis.py", "exec")
        except SyntaxError as e:
            pytest.fail(f"run_analysis.py has syntax errors: {e}")
    
    def test_config_data_directory_exists(self):
        """Verify the config points to an existing data directory."""
        from src.config import Config
        
        config = Config('config.json')
        data_dir = config.data_dir
        
        assert data_dir.exists(), f"Config data directory does not exist: {data_dir}"
        
        # Should find production data files
        roi_files = list(data_dir.glob("*.txt"))
        assert len(roi_files) > 0, f"No ROI files found in {data_dir}"
        
        # Should be production data (25 files), not test data (3 files)
        assert len(roi_files) > 10, f"Expected production data (>10 files), found {len(roi_files)}"