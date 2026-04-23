"""Integration tests for batch annotation orchestration (the Seam-1 seam).

Covers the pieces brutalist Codex #4 flagged as untested: load_sham_reference
validation, archive_prior_annotations behavior, and the transformed_arrays
bug-fix regression.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src.analysis.cell_type_annotation import annotate_roi_from_results

# Import the orchestration helpers without triggering the __main__ block
import importlib.util

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "batch_annotate_all_rois", _ROOT / "batch_annotate_all_rois.py"
)
_BATCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BATCH)


def _write_config(tmp_path: Path) -> Path:
    """Minimal config.json with just the fields load_sham_reference reads."""
    cfg = {
        'channels': {'protein_channels': ['CD45', 'CD31', 'CD140a']},
        'cell_type_annotation': {
            'enabled': True,
            'positivity_threshold': {
                'method': 'percentile', 'percentile': 60,
                'per_marker_override': {},
            },
            'cell_types': {},
            'membership_axes': {
                'normalization': 'sigmoid_threshold',
                'sigmoid_steepness': 10.0,
                'lineages': {'immune': {'markers': ['CD45'], 'aggregation': 'max'}},
                'subtypes': {'subtype_threshold': 0.3, 'definitions': {}},
                'activation': {'markers': {}},
                'composite_label_thresholds': {
                    'lineage': 0.3, 'activation': 0.3, 'dominance': 2.0,
                },
            },
        },
        'segmentation': {
            'method': 'slic', 'slic_input_channels': ['DNA1', 'DNA2'],
            'slic_params': {}, 'scales_um': [10.0],
        },
        'processing': {},
        'analysis': {}, 'quality_control': {}, 'output': {},
        'performance': {}, 'validation': {}, 'metadata_tracking': {},
    }
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(cfg))
    return config_path


def _write_artifact(tmp_path: Path, config_path: Path, **overrides) -> Path:
    sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    metadata = {
        'scale_um': 10.0,
        'marker_order': ['CD45', 'CD31', 'CD140a'],
        'percentile': 60.0,
        'aggregation': 'per_mouse',
        'config_sha256': sha,
    }
    metadata.update(overrides)
    artifact = {
        '_metadata': metadata,
        'reference': {
            'CD45':   {'threshold': 1.6, 'scale': 0.7},
            'CD31':   {'threshold': 2.4, 'scale': 1.2},
            'CD140a': {'threshold': 2.2, 'scale': 1.0},
        },
    }
    path = tmp_path / 'sham_reference_10.0um.json'
    path.write_text(json.dumps(artifact))
    return path


# ---------------------------------------------------------------------------
# load_sham_reference hardening
# ---------------------------------------------------------------------------

def test_load_sham_reference_happy_path(tmp_path):
    config_path = _write_config(tmp_path)
    artifact_path = _write_artifact(tmp_path, config_path)
    config = Config(str(config_path))
    ref = _BATCH.load_sham_reference(
        scale_um=10.0,
        protein_channels=['CD45', 'CD31', 'CD140a'],
        config=config, config_path=config_path,
        reference_path=artifact_path,
    )
    assert set(ref.keys()) == {'CD45', 'CD31', 'CD140a'}
    assert ref['CD45'] == {'threshold': 1.6, 'scale': 0.7}


def test_load_sham_reference_config_hash_mismatch(tmp_path):
    config_path = _write_config(tmp_path)
    artifact_path = _write_artifact(
        tmp_path, config_path, config_sha256='deadbeef' * 8,
    )
    config = Config(str(config_path))
    with pytest.raises(ValueError, match="config_sha256 does not match"):
        _BATCH.load_sham_reference(
            scale_um=10.0, protein_channels=['CD45', 'CD31', 'CD140a'],
            config=config, config_path=config_path,
            reference_path=artifact_path,
        )


def test_load_sham_reference_percentile_mismatch(tmp_path):
    config_path = _write_config(tmp_path)
    artifact_path = _write_artifact(tmp_path, config_path, percentile=75.0)
    config = Config(str(config_path))
    with pytest.raises(ValueError, match="percentile 75.0 does not match"):
        _BATCH.load_sham_reference(
            scale_um=10.0, protein_channels=['CD45', 'CD31', 'CD140a'],
            config=config, config_path=config_path,
            reference_path=artifact_path,
        )


def test_load_sham_reference_aggregation_mismatch(tmp_path):
    config_path = _write_config(tmp_path)
    artifact_path = _write_artifact(tmp_path, config_path, aggregation='pool')
    config = Config(str(config_path))
    with pytest.raises(ValueError, match="aggregation 'pool' is not"):
        _BATCH.load_sham_reference(
            scale_um=10.0, protein_channels=['CD45', 'CD31', 'CD140a'],
            config=config, config_path=config_path,
            reference_path=artifact_path,
        )


def test_load_sham_reference_marker_order_mismatch(tmp_path):
    config_path = _write_config(tmp_path)
    artifact_path = _write_artifact(
        tmp_path, config_path,
        marker_order=['CD31', 'CD45', 'CD140a'],  # order permuted
    )
    config = Config(str(config_path))
    with pytest.raises(ValueError, match="marker_order mismatch"):
        _BATCH.load_sham_reference(
            scale_um=10.0, protein_channels=['CD45', 'CD31', 'CD140a'],
            config=config, config_path=config_path,
            reference_path=artifact_path,
        )


def test_load_sham_reference_missing_scale_key(tmp_path):
    """A marker entry missing 'scale' must raise (Codex #2)."""
    config_path = _write_config(tmp_path)
    artifact_path = _write_artifact(tmp_path, config_path)
    artifact = json.loads(artifact_path.read_text())
    del artifact['reference']['CD45']['scale']
    artifact_path.write_text(json.dumps(artifact))
    config = Config(str(config_path))
    with pytest.raises(ValueError, match="missing key 'scale'"):
        _BATCH.load_sham_reference(
            scale_um=10.0, protein_channels=['CD45', 'CD31', 'CD140a'],
            config=config, config_path=config_path,
            reference_path=artifact_path,
        )


def test_load_sham_reference_missing_marker_entry(tmp_path):
    config_path = _write_config(tmp_path)
    artifact_path = _write_artifact(tmp_path, config_path)
    artifact = json.loads(artifact_path.read_text())
    del artifact['reference']['CD31']
    artifact_path.write_text(json.dumps(artifact))
    config = Config(str(config_path))
    with pytest.raises(ValueError, match="missing entry for marker 'CD31'"):
        _BATCH.load_sham_reference(
            scale_um=10.0, protein_channels=['CD45', 'CD31', 'CD140a'],
            config=config, config_path=config_path,
            reference_path=artifact_path,
        )


def test_load_sham_reference_scale_mismatch(tmp_path):
    config_path = _write_config(tmp_path)
    artifact_path = _write_artifact(tmp_path, config_path, scale_um=20.0)
    config = Config(str(config_path))
    with pytest.raises(ValueError, match="scale 20.0 does not match"):
        _BATCH.load_sham_reference(
            scale_um=10.0, protein_channels=['CD45', 'CD31', 'CD140a'],
            config=config, config_path=config_path,
            reference_path=artifact_path,
        )


def test_load_sham_reference_file_not_found(tmp_path):
    config_path = _write_config(tmp_path)
    config = Config(str(config_path))
    with pytest.raises(FileNotFoundError, match="generate_sham_reference"):
        _BATCH.load_sham_reference(
            scale_um=10.0, protein_channels=['CD45', 'CD31', 'CD140a'],
            config=config, config_path=config_path,
            reference_path=tmp_path / 'nonexistent.json',
        )


# ---------------------------------------------------------------------------
# archive_prior_annotations opt-in behavior
# ---------------------------------------------------------------------------

def test_archive_prior_empty_dir_returns_none(tmp_path):
    out = tmp_path / 'annotations'
    out.mkdir()
    assert _BATCH.archive_prior_annotations(out) is None


def test_detect_prior_annotations_finds_parquets_and_json(tmp_path):
    out = tmp_path / 'annotations'
    out.mkdir()
    (out / 'roi_1_cell_types.parquet').write_bytes(b'stub')
    (out / 'roi_1_annotation_metadata.json').write_text('{}')
    found = _BATCH.detect_prior_annotations(out)
    assert len(found) == 2
    assert {p.name for p in found} == {
        'roi_1_cell_types.parquet', 'roi_1_annotation_metadata.json',
    }


# ---------------------------------------------------------------------------
# End-to-end: annotate_roi_from_results with transformed_arrays (the bug-fix
# regression — was Codex #4 and Claude's testability-gap #1)
# ---------------------------------------------------------------------------

def _make_roi_results(n_sp=50):
    """Minimal roi_results dict covering both transformed_arrays and features.

    ``features`` holds raw ion counts (cols 0-8) + placeholder derived cols;
    ``transformed_arrays`` holds arcsinh per marker. Drastically different
    scales so a bug that mistakenly used ``features`` for continuous
    memberships would saturate the sigmoid.
    """
    rng = np.random.default_rng(0)
    markers = ['CD45', 'CD31', 'CD140a']
    # Raw: ion counts in 0..1000 range
    raw_cols = np.stack([
        rng.uniform(0, 1000, n_sp),  # CD45 raw
        rng.uniform(0, 1000, n_sp),  # CD31 raw
        rng.uniform(0, 1000, n_sp),  # CD140a raw
    ], axis=1)
    features_mat = np.hstack([raw_cols, rng.normal(0, 1, (n_sp, 5))])
    # Arcsinh per marker: ~1-5 range
    arc = {m: np.arcsinh(raw_cols[:, i]) for i, m in enumerate(markers)}
    coords = np.stack([np.arange(n_sp), np.arange(n_sp)], axis=1)
    return {
        'multiscale_results': {
            '10.0': {
                'features': features_mat,
                'transformed_arrays': arc,
                'spatial_coords': coords,
                'superpixel_coords': coords,
                'cluster_labels': np.zeros(n_sp, dtype=int),
                'scale_um': 10.0,
            }
        }
    }


def test_annotate_uses_arcsinh_not_raw_ion_counts(tmp_path):
    """Regression: the features→transformed_arrays bug-fix must hold.

    If annotate_roi_from_results reverted to using ``features`` (raw ion
    counts in 0..1000 range), the sigmoid applied with a ~1.6 threshold
    from Sham-reference would saturate to 1.0 for almost every row.
    """
    config_path = _write_config(tmp_path)
    config = Config(str(config_path))
    roi_results = _make_roi_results(n_sp=50)

    # Reference built on arcsinh-scale values
    reference = {
        'CD45':   {'threshold': 3.0, 'scale': 1.0},
        'CD31':   {'threshold': 3.0, 'scale': 1.0},
        'CD140a': {'threshold': 3.0, 'scale': 1.0},
    }

    result = annotate_roi_from_results(
        roi_results=roi_results, config=config, scale='10.0',
        reference_distribution=reference,
    )
    assert 'memberships' in result
    scores = result['memberships']['lineage_scores']['immune']
    # Under arcsinh scale (1-5 range), the ~3.0 threshold sees a reasonable
    # spread of scores. Under raw scale (0-1000 range), every value would
    # be ≫ 3.0, saturating sigmoid to ~1.0.
    assert np.std(scores) > 0.1, (
        f"Sigmoid saturated — std={np.std(scores):.3f}. "
        f"Likely regression: using raw features instead of arcsinh."
    )
    assert 0.0 <= scores.min() < scores.max() <= 1.0
