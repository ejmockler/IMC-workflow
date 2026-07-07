"""Cross-check + structure tests for the descriptive CD44 compartment script.

Proves the reuse is correct: the recomputed 10um pooled compartment rates
match the frozen pipeline's compartment_activation_temporal.parquet, and the
recomputed thresholds match sham_reference_thresholds.parquet. Also checks CSV
structure and that the new compartments are present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import collab_cd44_compartments as cc
from run_temporal_interface_analysis import load_annotations_with_markers

REPO_ROOT = Path(__file__).resolve().parents[1]
TI_DIR = REPO_ROOT / 'results' / 'biological_analysis' / 'temporal_interfaces'
OUT_DIR = REPO_ROOT / 'results' / 'biological_analysis' / 'collab_cd44'
EXPECTED_COLUMNS = ['compartment', 'timepoint', 'mouse', 'region', 'cd44_rate', 'n_support']


@pytest.fixture(scope='module')
def annotations_10um():
    return load_annotations_with_markers(scale_um=10.0)


@pytest.fixture(scope='module')
def pooled_mouse_10um(annotations_10um):
    """Recompute the pooled (region-agnostic) mouse-level table via the reuse
    path, exactly as the script does before regional slicing."""
    from src.analysis import temporal_interface_analysis as tia
    per_roi, _ = cc.compute_per_roi(annotations_10um, include_neutrophil=True)
    return tia.aggregate_compartment_activation_to_mouse(per_roi)


def test_10um_overlap_rates_match_pipeline(pooled_mouse_10um):
    """CD45/CD31/CD140b/neutrophil pooled rates reproduce the frozen
    compartment_activation_temporal.parquet within 1e-6 (reuse correctness)."""
    ref = pd.read_parquet(TI_DIR / 'compartment_activation_temporal.parquet')
    got = pooled_mouse_10um.set_index(['timepoint', 'mouse'])
    ref = ref.set_index(['timepoint', 'mouse'])
    cols = [
        'CD45_compartment_cd44_rate',
        'CD31_compartment_cd44_rate',
        'CD140b_compartment_cd44_rate',
        'neutrophil_compartment_cd44_rate',
    ]
    # Align on the shared index (8 mouse x timepoint rows)
    assert set(got.index) == set(ref.index)
    for col in cols:
        diff = (got.loc[ref.index, col] - ref[col]).abs().max()
        assert diff < 1e-6, f"{col} mismatch: max abs diff {diff:.3e}"


def test_thresholds_match_sham_reference(annotations_10um):
    """Recomputed CD45/CD31/CD140b/CD44 thresholds equal
    sham_reference_thresholds.parquet within ~1e-9 (identical recipe extends
    to CD206/CD34)."""
    from src.analysis import temporal_interface_analysis as tia
    thr = tia.compute_sham_reference_thresholds(
        annotations_10um, cc.THRESHOLD_MARKERS, percentile=cc.SHAM_PERCENTILE,
    )
    ref = pd.read_parquet(TI_DIR / 'sham_reference_thresholds.parquet').iloc[0]
    for marker in ('CD45', 'CD31', 'CD140b', 'CD44'):
        assert abs(thr[marker] - float(ref[marker])) < 1e-9, (
            f"{marker} threshold {thr[marker]} != parquet {ref[marker]}"
        )
    # The new markers exist and are finite by the same recipe.
    assert np.isfinite(thr['CD206']) and np.isfinite(thr['CD34'])


@pytest.mark.parametrize('scale', ['10um', '40um'])
def test_csv_structure(scale):
    df = pd.read_csv(OUT_DIR / f'cd44_compartment_rates_{scale}.csv')
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) > 0
    assert set(df['region'].unique()) == {'cortex', 'medulla', 'pooled'}
    # Every row carries an n_support count.
    assert df['n_support'].notna().all()


def test_10um_new_compartments_present():
    df = pd.read_csv(OUT_DIR / 'cd44_compartment_rates_10um.csv')
    compartments = set(df['compartment'].unique())
    for expected in ('CD45', 'CD206', 'CD31', 'CD34', 'CD140b',
                     'endothelial_cd31cd34', 'neutrophil'):
        assert expected in compartments, f"missing compartment {expected}"
    # 7 total at 10um (the CD31&CD34 endothelial AND-compartment included).
    assert len(compartments) == 7


def test_40um_thinning_visible_and_no_neutrophil():
    df10 = pd.read_csv(OUT_DIR / 'cd44_compartment_rates_10um.csv')
    df40 = pd.read_csv(OUT_DIR / 'cd44_compartment_rates_40um.csv')
    # Neutrophil compartment has no 40um cell-type basis — must be absent.
    assert 'neutrophil' not in set(df40['compartment'].unique())
    # Thinning: 40um support is far smaller than 10um for the same compartment.
    med10 = df10.loc[df10['compartment'] == 'CD45', 'n_support'].sum()
    med40 = df40.loc[df40['compartment'] == 'CD45', 'n_support'].sum()
    assert med40 < med10
