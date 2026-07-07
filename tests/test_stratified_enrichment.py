"""Contract tests for the stratified-enrichment abstraction ``aggregate_strata``.

These pin the DESIGN (stratified-enrichment re-mold) decisions directly against the
already-landed production function ``spatial_neighborhood_analysis.aggregate_strata``
(N2). No production code is exercised except that function and its metadata join.

The suite is deterministic, I/O-free and permutation-free: it monkeypatches the
module-level ``parse_roi_metadata`` binding (spatial_neighborhood_analysis.py:26) with a
tiny in-memory map and feeds hand-built one-row frames that mirror the
``analyze_roi_neighborhoods`` output schema (spatial_neighborhood_analysis.py:239-249).
``aggregate_strata`` only reads ``log2_enrichment`` / ``n_focal_cells`` / ``roi_id`` plus
the derived timepoint/region/mouse, so the frames are minimal-but-realistic and the whole
suite runs in well under a second — no parquet reads, no metadata CSV, no 1000-perm run.

The five (six) required assertions:
  1. ``['timepoint']`` reproduces the timepoint marginal and ``['region']`` the region
     marginal (mouse-of-mouse), with matching sign/order and the point bounded by the
     observed per-mouse spread.
  2. Mouse-of-mouse, NOT ``n_focal`` weighting: a 100x-denser ROI/mouse does not dominate.
  3. Support ledger cols present on every row; a below-min-support / single-mouse stratum
     is EMITTED with its flag and a NaN estimate, never dropped.
  4. No significance columns (p_value / q_value / fraction_significant_*) at the lens.
  5. Observed per-mouse spread cols (range_min / range_max / mouse_values), no bootstrap.
  6. Effective-mice guard: a stratum where only one mouse yields a defined per-mouse
     estimate is insufficient_support even though n_mice == 2.
"""

import numpy as np
import pandas as pd
import pytest

import spatial_neighborhood_analysis as sna
from src.analysis.temporal_interface_analysis import DEFAULT_MIN_SUPPORT


# --------------------------------------------------------------------------- #
# Fixtures / helpers                                                          #
# --------------------------------------------------------------------------- #

def _is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def _mkframe(roi_id, log2, n_focal, focal='A', neighbor='B'):
    """One-row DataFrame mirroring analyze_roi_neighborhoods output schema.

    Columns match spatial_neighborhood_analysis.py:239-249. ``aggregate_strata`` reads
    only log2_enrichment / n_focal_cells / roi_id (+ derived metadata); the remaining
    columns are populated for realism but are never consumed by the aggregation.
    """
    enr = 0.0 if _is_nan(log2) else float(2.0 ** log2)
    exp_prop = 0.1
    obs_prop = exp_prop * enr
    return pd.DataFrame([{
        'roi_id': roi_id,
        'focal_cell_type': focal,
        'neighbor_cell_type': neighbor,
        'observed_proportion': obs_prop,
        'expected_proportion': exp_prop,
        'enrichment_score': enr,
        'log2_enrichment': (np.nan if _is_nan(log2) else float(log2)),
        'p_value': 0.5,
        'n_focal_cells': int(n_focal),
    }])


def _build(specs, focal='A', neighbor='B'):
    """Turn (roi_id, timepoint, region, mouse, log2, n_focal) tuples into
    (list-of-frames, meta-dict). The meta dict mirrors the keys returned by
    src.utils.metadata.parse_roi_metadata (roi_id/timepoint/region/replicate/mouse)."""
    frames = []
    meta = {}
    for roi_id, tp, region, mouse, log2, n_focal in specs:
        frames.append(_mkframe(roi_id, log2, n_focal, focal, neighbor))
        meta[roi_id] = {
            'roi_id': roi_id,
            'timepoint': tp,
            'region': region,
            'replicate': roi_id,
            'mouse': mouse,
        }
    return frames, meta


def _patch_meta(monkeypatch, meta):
    """Replace the module-level parse_roi_metadata binding so the suite is I/O-free."""
    monkeypatch.setattr(sna, 'parse_roi_metadata', lambda roi_id, *a, **k: meta[roi_id])


def _marginal_dataset():
    """Multi-region, multi-timepoint, 2-mice-per-cell dataset with hand-picked log2s.

    Each of 4 mice contributes exactly one Cortex ROI and one Medulla ROI, so both the
    timepoint marginal and the region marginal are analytically known.
    """
    return _build([
        # roi_id,   timepoint, region,    mouse, log2, n_focal
        ('m1_ctx',  'D1',      'Cortex',  'M1',  +2.0, 100),
        ('m1_med',  'D1',      'Medulla', 'M1',  +1.0, 100),
        ('m2_ctx',  'D1',      'Cortex',  'M2',  +1.0, 100),
        ('m2_med',  'D1',      'Medulla', 'M2',   0.0, 100),
        ('m3_ctx',  'D3',      'Cortex',  'M3',  -1.0, 100),
        ('m3_med',  'D3',      'Medulla', 'M3',  -2.0, 100),
        ('m4_ctx',  'D3',      'Cortex',  'M4',   0.0, 100),
        ('m4_med',  'D3',      'Medulla', 'M4',  -1.0, 100),
    ])


def _row(df, **conds):
    """Return the single row matching all column==value conditions."""
    mask = pd.Series(True, index=df.index)
    for col, val in conds.items():
        mask &= (df[col] == val)
    sub = df[mask]
    assert len(sub) == 1, f"expected exactly one row for {conds}, got {len(sub)}"
    return sub.iloc[0]


# --------------------------------------------------------------------------- #
# ASSERTION 1 — marginal reproduction (mouse-of-mouse, sign/order, bounded)   #
# --------------------------------------------------------------------------- #

def test_timepoint_marginal_reproduces_mouse_of_mouse(monkeypatch):
    frames, meta = _marginal_dataset()
    _patch_meta(monkeypatch, meta)

    out = sna.aggregate_strata(frames, ['timepoint'])

    # One row per (timepoint, focal, neighbor); single A-B pair -> 2 timepoints.
    assert len(out) == 2
    assert set(out['timepoint']) == {'D1', 'D3'}

    # Hand-computed mouse-of-mouse marginals:
    #   D1: M1=mean(2,1)=1.5, M2=mean(1,0)=0.5  -> mean over mice = 1.0
    #   D3: M3=mean(-1,-2)=-1.5, M4=mean(0,-1)=-0.5 -> mean over mice = -1.0
    d1 = _row(out, timepoint='D1', focal_cell_type='A', neighbor_cell_type='B')
    d3 = _row(out, timepoint='D3', focal_cell_type='A', neighbor_cell_type='B')

    assert np.isclose(d1['log2_enrichment'], 1.0)
    assert np.isclose(d3['log2_enrichment'], -1.0)

    # Sign: D1 enriched (>0), D3 depleted (<0). Order: D1 > D3.
    assert d1['log2_enrichment'] > 0 > d3['log2_enrichment']
    assert d1['log2_enrichment'] > d3['log2_enrichment']

    # Mouse-level delta bounded by observed spread on every non-flagged row.
    for r in (d1, d3):
        assert not r['insufficient_support'] and not r['below_min_support']
        assert r['range_min'] <= r['log2_enrichment'] <= r['range_max']

    # Exact per-mouse spread D1 = [0.5, 1.5], D3 = [-1.5, -0.5].
    assert np.isclose(d1['range_min'], 0.5) and np.isclose(d1['range_max'], 1.5)
    assert np.isclose(d3['range_min'], -1.5) and np.isclose(d3['range_max'], -0.5)


def test_region_marginal_reproduces_mouse_of_mouse(monkeypatch):
    frames, meta = _marginal_dataset()
    _patch_meta(monkeypatch, meta)

    out = sna.aggregate_strata(frames, ['region'])

    assert len(out) == 2
    assert set(out['region']) == {'Cortex', 'Medulla'}

    # Region marginal (each mouse contributes one ROI per region -> per-mouse == that ROI):
    #   Cortex: [M1=2, M2=1, M3=-1, M4=0]  -> mean = 0.5
    #   Medulla:[M1=1, M2=0, M3=-2, M4=-1] -> mean = -0.5
    ctx = _row(out, region='Cortex', focal_cell_type='A', neighbor_cell_type='B')
    med = _row(out, region='Medulla', focal_cell_type='A', neighbor_cell_type='B')

    assert np.isclose(ctx['log2_enrichment'], 0.5)
    assert np.isclose(med['log2_enrichment'], -0.5)

    # Sign: Cortex enriched, Medulla depleted. Order: Cortex > Medulla.
    assert ctx['log2_enrichment'] > 0 > med['log2_enrichment']
    assert ctx['log2_enrichment'] > med['log2_enrichment']

    for r in (ctx, med):
        assert not r['insufficient_support'] and not r['below_min_support']
        assert r['range_min'] <= r['log2_enrichment'] <= r['range_max']

    assert np.isclose(ctx['range_min'], -1.0) and np.isclose(ctx['range_max'], 2.0)
    assert np.isclose(med['range_min'], -2.0) and np.isclose(med['range_max'], 1.0)


# --------------------------------------------------------------------------- #
# ASSERTION 2 — MOUSE-of-mouse, not n_focal weighting                         #
# --------------------------------------------------------------------------- #

def test_dense_roi_does_not_dominate_between_mice(monkeypatch):
    # One stratum, two single-ROI mice differing 1000x in focal cell count.
    frames, meta = _build([
        ('big',   'D1', 'Cortex', 'M1', +3.0, 10000),
        ('small', 'D1', 'Cortex', 'M2', -1.0,    10),
    ])
    _patch_meta(monkeypatch, meta)

    out = sna.aggregate_strata(frames, ['timepoint'])
    assert len(out) == 1
    row = out.iloc[0]

    unweighted = float(np.mean([3.0, -1.0]))            # == 1.0
    n_focal_weighted = float(np.average([3.0, -1.0], weights=[10000, 10]))  # ~2.996

    assert np.isclose(row['log2_enrichment'], unweighted)
    assert np.isclose(row['log2_enrichment'], 1.0)
    # Discriminating: a regression to pseudoreplication (n_focal weighting) would fail here.
    assert not np.isclose(row['log2_enrichment'], n_focal_weighted)
    assert sorted(float(v) for v in row['mouse_values']) == [-1.0, 3.0]


def test_dense_roi_does_not_dominate_within_a_mouse(monkeypatch):
    # M1 has two ROIs differing 100x in focal count; the per-mouse mean must be UNWEIGHTED.
    frames, meta = _build([
        ('m1a', 'D1', 'Cortex', 'M1', +2.0, 10000),
        ('m1b', 'D1', 'Cortex', 'M1',  0.0,   100),
        ('m2',  'D1', 'Cortex', 'M2', +0.5,    50),
    ])
    _patch_meta(monkeypatch, meta)

    out = sna.aggregate_strata(frames, ['timepoint'])
    assert len(out) == 1
    row = out.iloc[0]

    m1_unweighted = float(np.mean([2.0, 0.0]))                              # 1.0
    m1_within_weighted = float(np.average([2.0, 0.0], weights=[10000, 100]))  # ~1.980
    stratum_unweighted = float(np.mean([m1_unweighted, 0.5]))              # 0.75
    stratum_if_within_weighted = float(np.mean([m1_within_weighted, 0.5]))  # ~1.240

    assert np.isclose(row['log2_enrichment'], stratum_unweighted)
    assert not np.isclose(row['log2_enrichment'], stratum_if_within_weighted)

    # The per-mouse point for M1 is the unweighted 1.0, never the cell-count-weighted 1.98.
    mv = sorted(float(v) for v in row['mouse_values'])
    assert np.allclose(mv, [0.5, 1.0])
    assert not any(np.isclose(v, m1_within_weighted, atol=1e-3) for v in mv)


# --------------------------------------------------------------------------- #
# ASSERTION 3 (+6) — support ledger + emit-not-drop + effective-mice guard    #
# --------------------------------------------------------------------------- #

def test_support_ledger_emit_not_drop_and_effective_mice(monkeypatch):
    # Four distinct strata in ONE call, covering every support regime:
    #   TA: well-supported (2 effective mice, n_focal >= min_support)
    #   TB: below-min-support (2 mice, summed n_focal < min_support)
    #   TC: single-mouse (n_mice == 1)
    #   TD: effective-mice catch (2 mice have rows, but one mouse's log2 is NaN)
    frames, meta = _build([
        ('ta1', 'TA', 'R', 'M1', +1.0,  50),
        ('ta2', 'TA', 'R', 'M2', +2.0,  50),
        ('tb1', 'TB', 'R', 'M3', +1.0,   5),
        ('tb2', 'TB', 'R', 'M4', +2.0,   5),
        ('tc1', 'TC', 'R', 'M5', +1.0, 100),
        ('td1', 'TD', 'R', 'M6', +1.0, 100),
        ('td2', 'TD', 'R', 'M7', np.nan, 100),
    ])
    _patch_meta(monkeypatch, meta)

    out = sna.aggregate_strata(frames, ['timepoint'])

    # Emit-not-drop: exactly one row per distinct input stratum key; nothing removed.
    assert len(out) == 4
    assert set(out['timepoint']) == {'TA', 'TB', 'TC', 'TD'}

    # Support ledger columns present and NON-NULL on EVERY row.
    ledger = ['n_focal_cells', 'n_rois', 'n_mice', 'n_mice_effective']
    for col in ledger:
        assert col in out.columns
        assert out[col].notna().all(), f"{col} has nulls"

    est_cols = ['log2_enrichment', 'enrichment_score', 'range_min', 'range_max']

    # (a) well-supported: estimate computed, no flags.
    ta = _row(out, timepoint='TA')
    assert not ta['below_min_support'] and not ta['insufficient_support']
    assert np.isclose(ta['log2_enrichment'], 1.5)
    assert ta['n_mice'] == 2 and ta['n_mice_effective'] == 2

    # (b) below-min-support: flagged, estimate NaN'd, but row + counts retained.
    tb = _row(out, timepoint='TB')
    assert tb['below_min_support'] is True or tb['below_min_support'] == True  # noqa: E712
    assert tb['n_focal_cells'] == 10 and tb['n_mice'] == 2
    for c in est_cols:
        assert _is_nan(tb[c]), f"TB estimate {c} should be NaN"

    # (c) single-mouse: insufficient_support, estimate NaN, but emitted with counts.
    tc = _row(out, timepoint='TC')
    assert tc['insufficient_support'] == True  # noqa: E712
    assert tc['n_mice'] == 1 and tc['n_mice_effective'] == 1
    assert _is_nan(tc['log2_enrichment'])

    # (d) EFFECTIVE-MICE GUARD: two mice have rows, but only one yields a defined
    #     per-mouse estimate -> n_mice == 2, n_mice_effective == 1, insufficient_support.
    td = _row(out, timepoint='TD')
    assert td['n_mice'] == 2
    assert td['n_mice_effective'] == 1
    assert td['insufficient_support'] == True  # noqa: E712
    assert _is_nan(td['log2_enrichment'])


# --------------------------------------------------------------------------- #
# ASSERTION 4 — NO significance surface at the lens                           #
# --------------------------------------------------------------------------- #

FORBIDDEN_SIGNIFICANCE_COLS = {
    'p_value', 'p_value_fdr', 'q_value',
    'fraction_significant_raw', 'fraction_significant_fdr', 'significant_fdr',
}


@pytest.mark.parametrize('strata', [
    ['region', 'timepoint'],
    ['timepoint'],
    ['region'],
    [],
])
def test_no_significance_columns_at_lens(monkeypatch, strata):
    frames, meta = _marginal_dataset()
    _patch_meta(monkeypatch, meta)

    out = sna.aggregate_strata(frames, strata)

    leaked = FORBIDDEN_SIGNIFICANCE_COLS.intersection(out.columns)
    assert not leaked, f"significance columns leaked into strata={strata} lens: {leaked}"


# --------------------------------------------------------------------------- #
# ASSERTION 5 — observed per-mouse spread cols (range_min/range_max), no boot #
# --------------------------------------------------------------------------- #

def test_observed_range_columns_present_no_bootstrap(monkeypatch):
    frames, meta = _marginal_dataset()
    _patch_meta(monkeypatch, meta)

    out = sna.aggregate_strata(frames, ['timepoint'])

    for col in ('range_min', 'range_max', 'mouse_values'):
        assert col in out.columns

    # DESIGN Decision 2: the band is the OBSERVED per-mouse spread, not a bootstrap.
    assert not any('bootstrap' in c for c in out.columns)

    row = _row(out, timepoint='D1', focal_cell_type='A', neighbor_cell_type='B')
    assert not row['insufficient_support'] and not row['below_min_support']

    mv = row['mouse_values']
    assert isinstance(mv, list)
    assert len(mv) == 2  # two mice contributed defined per-mouse estimates
    assert np.isclose(row['range_min'], float(min(mv)))
    assert np.isclose(row['range_max'], float(max(mv)))
