"""Unit tests for the shared Sham-reference primitive.

Covers the public surface of src.analysis.sham_reference: per-mouse vs
pool aggregation, per-marker overrides, hard gates, and the combined
``build_reference_distribution`` convenience wrapper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.sham_reference import (
    build_reference_distribution,
    experiment_wide_iqr,
    experiment_wide_percentile,
    sham_reference_thresholds,
)


def _make_annotations(
    sham_ms1=None, sham_ms2=None, d7_ms1=None, d7_ms2=None,
    marker='CD45',
):
    """Minimal fixture: variable-length Sham/D7 per mouse."""
    rows = []
    for vals, tp, ms in (
        (sham_ms1, 'Sham', 'MS1'),
        (sham_ms2, 'Sham', 'MS2'),
        (d7_ms1, 'D7', 'MS1'),
        (d7_ms2, 'D7', 'MS2'),
    ):
        if vals is None:
            continue
        for v in vals:
            rows.append({marker: v, 'timepoint': tp, 'mouse': ms})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sham_reference_thresholds — happy paths
# ---------------------------------------------------------------------------

def test_per_mouse_averages_across_biological_replicates():
    """Per-mouse 50th pct for MS1 (median of [0..100]) = 50; for MS2 (median
    of [50..150]) = 100. Mean across mice = 75."""
    df = _make_annotations(
        sham_ms1=list(range(0, 101)),     # 0..100; median=50
        sham_ms2=list(range(50, 151)),    # 50..150; median=100
    )
    thr = sham_reference_thresholds(df, ['CD45'], percentile=50.0)
    assert thr['CD45'] == pytest.approx(75.0)


def test_pool_aggregation_matches_legacy_behavior():
    """Pool of MS1+MS2 = [0..100, 50..150] — 50th pct of 202 values ~ 75."""
    df = _make_annotations(
        sham_ms1=list(range(0, 101)),
        sham_ms2=list(range(50, 151)),
    )
    thr = sham_reference_thresholds(
        df, ['CD45'], percentile=50.0, aggregation='pool',
    )
    # With these fixtures per_mouse and pool coincide — intentional to show
    # that they diverge only when mouse distributions differ in shape, not
    # location. The divergence case is in the next test.
    assert thr['CD45'] == pytest.approx(75.0, abs=1.0)


def test_per_mouse_diverges_from_pool_under_size_imbalance():
    """5 Sham values from MS1, 500 from MS2 — pool is dominated by MS2."""
    df = _make_annotations(
        sham_ms1=[100.0] * 5,            # MS1 baseline = 100
        sham_ms2=[1.0] * 500,            # MS2 baseline = 1
    )
    pooled = sham_reference_thresholds(
        df, ['CD45'], percentile=50.0, aggregation='pool',
    )
    per_mouse = sham_reference_thresholds(
        df, ['CD45'], percentile=50.0, aggregation='per_mouse',
    )
    # Pool median is dominated by MS2's 500 points -> ~1.0
    assert pooled['CD45'] == pytest.approx(1.0)
    # Per-mouse: median(MS1)=100, median(MS2)=1, mean=50.5
    assert per_mouse['CD45'] == pytest.approx(50.5)


def test_ignores_non_sham_timepoints():
    """Injury timepoints should not enter the reference."""
    df = _make_annotations(
        sham_ms1=[1.0, 2.0, 3.0],
        sham_ms2=[1.0, 2.0, 3.0],
        d7_ms1=[100.0] * 50,  # should not affect Sham threshold
        d7_ms2=[100.0] * 50,
    )
    thr = sham_reference_thresholds(df, ['CD45'], percentile=75.0)
    assert thr['CD45'] < 10.0  # injury 100s ignored


# ---------------------------------------------------------------------------
# per-marker overrides
# ---------------------------------------------------------------------------

def test_per_marker_override_changes_percentile():
    df = pd.DataFrame({
        'CD206': np.arange(100, dtype=float),  # 0..99
        'timepoint': ['Sham'] * 100,
        'mouse': ['MS1'] * 50 + ['MS2'] * 50,
    })
    # Without override: 60th pct per-mouse then averaged
    base = sham_reference_thresholds(df, ['CD206'], percentile=60.0)
    # With override: 50th pct
    overridden = sham_reference_thresholds(
        df, ['CD206'], percentile=60.0,
        per_marker_overrides={'CD206': {'method': 'percentile', 'percentile': 50}},
    )
    assert overridden['CD206'] < base['CD206']


def test_per_marker_override_rejects_absolute_method():
    df = pd.DataFrame({
        'CD206': [1.0, 2.0], 'timepoint': ['Sham', 'Sham'],
        'mouse': ['MS1', 'MS2'],
    })
    with pytest.raises(ValueError, match="only method='percentile' is supported"):
        sham_reference_thresholds(
            df, ['CD206'], percentile=60.0,
            per_marker_overrides={'CD206': {'method': 'absolute', 'value': 0.5}},
        )


def test_per_marker_override_missing_marker_uses_default():
    """Overrides for markers not in the list are ignored."""
    df = pd.DataFrame({
        'CD45': np.arange(10, dtype=float),
        'timepoint': ['Sham'] * 10,
        'mouse': ['MS1'] * 5 + ['MS2'] * 5,
    })
    thr = sham_reference_thresholds(
        df, ['CD45'], percentile=50.0,
        per_marker_overrides={'CD206': {'method': 'percentile', 'percentile': 99}},
    )
    # Should use default 50th pct, not the CD206 override
    assert 2.0 <= thr['CD45'] <= 7.0


# ---------------------------------------------------------------------------
# hard gates
# ---------------------------------------------------------------------------

def test_empty_sham_raises():
    df = pd.DataFrame({
        'CD45': [1.0, 2.0], 'timepoint': ['D1', 'D7'], 'mouse': ['MS1', 'MS2'],
    })
    with pytest.raises(ValueError, match="no superpixels with"):
        sham_reference_thresholds(df, ['CD45'], percentile=60.0)


def test_missing_timepoint_column_raises():
    df = pd.DataFrame({'CD45': [1.0], 'mouse': ['MS1']})
    with pytest.raises(ValueError, match="missing 'timepoint'"):
        sham_reference_thresholds(df, ['CD45'], percentile=60.0)


def test_missing_mouse_column_per_mouse_raises():
    df = pd.DataFrame({'CD45': [1.0], 'timepoint': ['Sham']})
    with pytest.raises(ValueError, match="requires 'mouse' column"):
        sham_reference_thresholds(df, ['CD45'], percentile=60.0)


def test_single_mouse_per_mouse_raises():
    df = _make_annotations(sham_ms1=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="require >= 2"):
        sham_reference_thresholds(df, ['CD45'], percentile=60.0)


def test_all_nan_marker_raises():
    df = _make_annotations(
        sham_ms1=[np.nan, np.nan], sham_ms2=[np.nan, np.nan],
    )
    with pytest.raises(ValueError, match="no finite Sham"):
        sham_reference_thresholds(df, ['CD45'], percentile=60.0)


def test_missing_marker_raises():
    df = _make_annotations(sham_ms1=[1.0], sham_ms2=[2.0])
    with pytest.raises(ValueError, match="marker 'CD999' missing"):
        sham_reference_thresholds(df, ['CD999'], percentile=60.0)


def test_unknown_aggregation_raises():
    df = _make_annotations(sham_ms1=[1.0], sham_ms2=[2.0])
    with pytest.raises(ValueError, match="unknown aggregation"):
        sham_reference_thresholds(
            df, ['CD45'], percentile=60.0, aggregation='magic',
        )


# ---------------------------------------------------------------------------
# experiment_wide helpers
# ---------------------------------------------------------------------------

def test_experiment_wide_percentile_ignores_timepoint():
    df = _make_annotations(
        sham_ms1=[1.0] * 10,
        sham_ms2=[1.0] * 10,
        d7_ms1=[100.0] * 10,
        d7_ms2=[100.0] * 10,
    )
    thr = experiment_wide_percentile(df, ['CD45'], percentile=75.0)
    # 75th pct of [1]*20 + [100]*20 spans into the injury distribution
    assert thr['CD45'] >= 50.0


def test_experiment_wide_iqr_zero_variance_falls_back():
    df = pd.DataFrame({'CD45': [5.0] * 20})
    scales = experiment_wide_iqr(df, ['CD45'])
    # IQR=0 → fallback (max-min)/4 = 0 → degenerate floor
    assert scales['CD45'] > 0


def test_experiment_wide_iqr_computes_q75_minus_q25():
    df = pd.DataFrame({'CD45': np.arange(101, dtype=float)})
    scales = experiment_wide_iqr(df, ['CD45'])
    # IQR of 0..100 = 50
    assert scales['CD45'] == pytest.approx(50.0, abs=1.0)


# ---------------------------------------------------------------------------
# build_reference_distribution integration
# ---------------------------------------------------------------------------

def test_build_reference_distribution_returns_threshold_and_scale_per_marker():
    df = _make_annotations(
        sham_ms1=list(range(0, 21)),
        sham_ms2=list(range(10, 31)),
        d7_ms1=list(range(100, 121)),  # inflates scale but not threshold
        d7_ms2=list(range(100, 121)),
    )
    ref = build_reference_distribution(df, ['CD45'], percentile=60.0)
    assert 'threshold' in ref['CD45']
    assert 'scale' in ref['CD45']
    # Threshold only sees Sham (low); scale sees Sham+injury (wide)
    assert ref['CD45']['threshold'] < 30.0
    assert ref['CD45']['scale'] > 30.0
