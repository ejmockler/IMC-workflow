"""Tests for VizConfig cross-file key validation.

Rationale: prior to the viz/analysis config split, the project mixed three
labeling systems in one config.json — the remold anchored everything to the
15 config-defined cell types. These tests prevent future drift by ensuring
VizConfig.load() hard-fails whenever viz.json disagrees with config.json on
load-bearing keys.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.viz_utils import VizConfig, VizConfigValidationError


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_canonical_configs() -> tuple[dict, dict]:
    with open(REPO_ROOT / 'config.json') as f:
        cfg = json.load(f)
    with open(REPO_ROOT / 'viz.json') as f:
        viz = json.load(f)
    return cfg, viz


def _write_pair(tmp_path: Path, config: dict, viz: dict) -> Path:
    with open(tmp_path / 'config.json', 'w') as f:
        json.dump(config, f)
    viz_path = tmp_path / 'viz.json'
    with open(viz_path, 'w') as f:
        json.dump(viz, f)
    return viz_path


def test_canonical_configs_agree() -> None:
    """The project's own config.json + viz.json must pass validation — any
    divergence between them is the bug this validator is designed to catch."""
    v = VizConfig.load()
    assert len(v.cell_type_colors) == 15
    assert v.timepoint_order == ['Sham', 'D1', 'D3', 'D7']


def test_missing_cell_type_in_viz_raises(tmp_path: Path) -> None:
    cfg, viz = _read_canonical_configs()
    # Drop one cell type from viz.json — simulating a renamed type that
    # wasn't propagated to viz.json
    dropped = 'neutrophil'
    del viz['cell_type_display'][dropped]
    viz_path = _write_pair(tmp_path, cfg, viz)
    with pytest.raises(VizConfigValidationError) as exc:
        VizConfig.load(viz_path)
    assert 'neutrophil' in str(exc.value)
    assert 'missing from viz.json' in str(exc.value)


def test_extra_cell_type_in_viz_raises(tmp_path: Path) -> None:
    cfg, viz = _read_canonical_configs()
    viz['cell_type_display']['ghost_type'] = {'label': 'Ghost', 'color': '#000000'}
    viz_path = _write_pair(tmp_path, cfg, viz)
    with pytest.raises(VizConfigValidationError) as exc:
        VizConfig.load(viz_path)
    assert 'ghost_type' in str(exc.value)
    assert 'not in config.json' in str(exc.value)


def test_timepoint_order_mismatch_raises(tmp_path: Path) -> None:
    cfg, viz = _read_canonical_configs()
    viz['timepoint_display']['order'] = ['Sham', 'D1', 'D7', 'D3']  # reordered
    viz_path = _write_pair(tmp_path, cfg, viz)
    with pytest.raises(VizConfigValidationError) as exc:
        VizConfig.load(viz_path)
    assert 'timepoint_display.order' in str(exc.value)


def test_validate_false_permits_drift(tmp_path: Path) -> None:
    """Test fixtures sometimes need a partial VizConfig — validate=False
    must bypass the cross-file check."""
    cfg, viz = _read_canonical_configs()
    del viz['cell_type_display']['neutrophil']
    viz_path = _write_pair(tmp_path, cfg, viz)
    # Should not raise
    v = VizConfig.load(viz_path, validate=False)
    assert 'neutrophil' not in v.cell_type_display


def test_multiple_mismatches_reported_together(tmp_path: Path) -> None:
    """When several invariants fail, the error should enumerate all of them
    rather than stop at the first — saves the user a round-trip per fix."""
    cfg, viz = _read_canonical_configs()
    del viz['cell_type_display']['neutrophil']
    viz['cell_type_display']['ghost_type'] = {'label': 'Ghost', 'color': '#000000'}
    viz['timepoint_display']['order'] = ['Sham']
    viz_path = _write_pair(tmp_path, cfg, viz)
    with pytest.raises(VizConfigValidationError) as exc:
        VizConfig.load(viz_path)
    msg = str(exc.value)
    assert 'neutrophil' in msg
    assert 'ghost_type' in msg
    assert 'timepoint_display.order' in msg
