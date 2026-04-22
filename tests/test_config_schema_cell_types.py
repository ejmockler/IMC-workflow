"""Tests for the cell_type_annotation Pydantic schema.

The Config class previously tolerated ``extra='allow'`` across the whole
cell_type_annotation section — a typo in a gate marker (e.g., ``CD451`` instead
of ``CD45``) would silently produce an empty gate. These tests lock in the
typo-detection guarantees added to src/config_schema.py.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config_schema import (
    CellTypeAnnotationConfig,
    CellTypeDefinition,
    CompositeLabelThresholds,
    load_validated_config,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_canonical() -> dict:
    with open(REPO_ROOT / 'config.json') as f:
        return json.load(f)


def test_canonical_config_validates() -> None:
    """The repo's own config.json must pass — any drift here is a regression."""
    cfg = load_validated_config(str(REPO_ROOT / 'config.json'))
    assert len(cfg.cell_type_annotation.cell_types) == 15
    thr = cfg.cell_type_annotation.membership_axes.composite_label_thresholds
    assert 0 < thr.lineage < 1
    assert thr.dominance >= 1.0


def test_cell_type_missing_required_field() -> None:
    with pytest.raises(ValidationError) as exc:
        CellTypeDefinition(
            positive_markers=['CD45'],
            # family missing
        )
    assert 'family' in str(exc.value).lower()


def test_cell_type_marker_in_both_positive_and_negative() -> None:
    with pytest.raises(ValidationError) as exc:
        CellTypeDefinition(
            positive_markers=['CD45', 'CD31'],
            negative_markers=['CD31'],  # conflict
            family='immune',
        )
    msg = str(exc.value)
    assert 'CD31' in msg
    assert 'both' in msg.lower()


def test_cell_type_typo_field_rejected() -> None:
    """A typo like 'positive_makers' must not silently become a no-op."""
    with pytest.raises(ValidationError) as exc:
        CellTypeDefinition(
            positive_markers=['CD45'],
            positive_makers=['CD11b'],  # typo; should be rejected
            family='immune',
        )
    assert 'positive_makers' in str(exc.value) or 'unexpected' in str(exc.value).lower()


def test_cell_type_comment_field_permitted() -> None:
    """_comment keys are legal (JSON convention in this repo)."""
    d = CellTypeDefinition(
        positive_markers=['CD45', 'Ly6G'],
        negative_markers=['CD31'],
        family='immune_neutrophil',
        _comment='Gated first because Ly6G is definitive',
    )
    assert d.family == 'immune_neutrophil'


def test_composite_thresholds_out_of_range() -> None:
    with pytest.raises(ValidationError):
        CompositeLabelThresholds(lineage=1.5, activation=0.3, dominance=2.0)
    with pytest.raises(ValidationError):
        CompositeLabelThresholds(lineage=0.3, activation=0.0, dominance=2.0)
    with pytest.raises(ValidationError):
        CompositeLabelThresholds(lineage=0.3, activation=0.3, dominance=0.5)


def test_composite_thresholds_typo_rejected() -> None:
    """Catches the 'linage' / 'dominace' typos that would silently fall back
    to defaults under the old dict-based config path."""
    with pytest.raises(ValidationError) as exc:
        CompositeLabelThresholds(
            lineage=0.3,
            activation=0.3,
            dominance=2.0,
            linage=0.5,  # typo
        )
    assert 'linage' in str(exc.value).lower() or 'unexpected' in str(exc.value).lower()


def test_cell_type_marker_not_in_protein_channels(tmp_path: Path) -> None:
    """Full-config load: a gate using a marker absent from protein_channels
    must fail at load time."""
    cfg = _load_canonical()
    cfg['cell_type_annotation']['cell_types']['neutrophil']['positive_markers'] = ['CD451']  # typo
    cfg_path = tmp_path / 'config.json'
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f)
    with pytest.raises(Exception) as exc:  # pydantic wraps it
        load_validated_config(str(cfg_path))
    assert 'CD451' in str(exc.value)


def test_composite_thresholds_default_sanity() -> None:
    """The canonical (0.3, 0.3, 2.0) tuple — if these change we want to know."""
    cfg = _load_canonical()
    thr = cfg['cell_type_annotation']['membership_axes']['composite_label_thresholds']
    assert thr['lineage'] == 0.3
    assert thr['activation'] == 0.3
    assert thr['dominance'] == 2.0
