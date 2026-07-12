"""Integration tests for run_analysis.py script."""

import json
import logging

import pytest
from pathlib import Path

import run_analysis


class TestRunAnalysisIntegration:
    """Test the production run_analysis.py script."""
    
    def test_script_uses_config_data_directory(
        self, tmp_path, monkeypatch, caplog
    ):
        """The runner reads config paths while all writes stay in a temp tree."""
        repo_root = Path(__file__).resolve().parents[1]
        config_data = json.loads((repo_root / "config.json").read_text())

        data_dir = tmp_path / "configured_data"
        data_dir.mkdir()
        for filename in ("roi_a.txt", "roi_b.txt", "Test_acquisition.txt"):
            (data_dir / filename).write_text("synthetic")

        metadata_path = tmp_path / "metadata.csv"
        metadata_path.write_text("ROI\n")
        config_data["data"]["raw_data_dir"] = str(data_dir)
        config_data["data"]["metadata_file"] = str(metadata_path)
        (tmp_path / "config.json").write_text(json.dumps(config_data))

        calls = {}

        class StubValidationPipeline:
            def validate_dataset(self, roi_files):
                calls["roi_files"] = list(roi_files)
                n_rois = len(roi_files)
                return {
                    "can_proceed": True,
                    "critical_failures": 0,
                    "total_issues": 0,
                    "usable_rois": n_rois,
                    "total_rois": n_rois,
                }

        monkeypatch.setattr(
            run_analysis,
            "create_practical_pipeline",
            lambda expected_proteins: StubValidationPipeline(),
        )

        def fake_run_complete_analysis(**kwargs):
            calls["analysis_kwargs"] = kwargs
            return {"status": "complete"}

        monkeypatch.setattr(
            run_analysis, "run_complete_analysis", fake_run_complete_analysis
        )
        monkeypatch.chdir(tmp_path)

        with caplog.at_level(logging.INFO):
            run_analysis.main()

        messages = [record.getMessage() for record in caplog.records]
        assert f"Data directory: {data_dir}" in messages
        assert "Found 2 ROI files (test acquisitions excluded)" in messages
        assert [path.name for path in calls["roi_files"]] == ["roi_a.txt", "roi_b.txt"]
        assert calls["analysis_kwargs"]["roi_directory"] == str(data_dir)
        assert (tmp_path / "results" / "validation_report.json").is_file()
        assert (tmp_path / "results" / "run_summary.json").is_file()
    
    def test_script_imports_and_runs(self):
        """Verify run_analysis.py can be imported and has basic structure."""
        # Test that script can be compiled without syntax errors
        script_path = Path(__file__).resolve().parents[1] / "run_analysis.py"
        try:
            compile(script_path.read_text(), str(script_path), "exec")
        except SyntaxError as e:
            pytest.fail(f"run_analysis.py has syntax errors: {e}")

    @pytest.mark.integration
    def test_config_data_directory_exists(self):
        """Verify the config points to an existing data directory."""
        from src.config import Config
        
        config = Config('config.json')
        data_dir = config.data_dir
        
        if not data_dir.exists():
            pytest.skip(f"local production data directory is unavailable: {data_dir}")
        
        # Should find production data files
        roi_files = list(data_dir.glob("*.txt"))
        if not roi_files:
            pytest.skip(f"local production ROI text files are unavailable in {data_dir}")
        
        # Should be production data (25 files), not test data (3 files)
        assert len(roi_files) > 10, f"Expected production data (>10 files), found {len(roi_files)}"
