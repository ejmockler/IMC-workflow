"""Tests for src.utils.metadata cache isolation (Gate 6 seam closure)."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.utils.metadata import parse_roi_metadata, clear_metadata_cache


@pytest.fixture(autouse=True)
def _clear_cache_between_tests():
    clear_metadata_cache()
    yield
    clear_metadata_cache()


def _write_fake_metadata_csv(tmp_path: Path, rows: list[dict]) -> Path:
    csv_path = tmp_path / "fake_metadata.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_custom_csv_does_not_poison_default_path(tmp_path):
    """A call with a custom metadata_csv must not alter what the default
    path returns on a subsequent call.

    This is the test isolation hazard the Gate 1 brutalist flagged.
    """
    # Fake CSV with fully-formed schema (Injury Day / Details / Mouse) but
    # a non-matching ROI. The lookup will miss and fall through to fallback
    # parsing; what matters is that the fake DataFrame gets cached under
    # ITS OWN key, not as "the default CSV".
    fake = _write_fake_metadata_csv(tmp_path, [
        {"File Name": "IMC_Other_ROI_X_01_1", "Injury Day": 1,
         "Details": "fabricated", "Mouse": "MS1"},
    ])
    # First call: custom CSV. roi_id not in fake table → fallback parser fires.
    custom = parse_roi_metadata("IMC_Some_Sam1_Path_01_1", metadata_csv=fake)
    assert custom["timepoint"] == "Sham"  # fallback recognized "Sam"
    # Second call: default CSV, real ROI — must lookup in REAL metadata,
    # not in the fake DataFrame that was cached on the first call.
    default = parse_roi_metadata("IMC_241218_Alun_ROI_D1_M1_01_9")
    assert default["timepoint"] == "D1"
    assert default["mouse"] == "MS1"
    assert default["region"] == "Medulla"  # real CSV value, NOT "fabricated"


def test_cache_clear_removes_all_entries(tmp_path):
    from src.utils.metadata import _load_metadata_df_cached

    # Prime the cache with two different paths
    parse_roi_metadata("IMC_241218_Alun_ROI_D1_M1_01_9")
    fake = _write_fake_metadata_csv(tmp_path, [{"File Name": "x", "Details": "y"}])
    try:
        parse_roi_metadata("IMC_Fake_ROI_Sam1_01_1", metadata_csv=fake)
    except Exception:
        pass
    info_before = _load_metadata_df_cached.cache_info()
    assert info_before.currsize >= 1
    clear_metadata_cache()
    info_after = _load_metadata_df_cached.cache_info()
    assert info_after.currsize == 0


def test_default_path_caches_on_repeated_calls():
    """Second default-path call should hit the cache (no I/O)."""
    from src.utils.metadata import _load_metadata_df_cached

    parse_roi_metadata("IMC_241218_Alun_ROI_D1_M1_01_9")
    info1 = _load_metadata_df_cached.cache_info()
    parse_roi_metadata("IMC_241218_Alun_ROI_D3_M1_01_2")
    info2 = _load_metadata_df_cached.cache_info()
    # Both calls use default path, so second should be a cache hit
    assert info2.hits > info1.hits
    assert info2.currsize == 1
