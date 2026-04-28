"""
Phase 7 D1 — guard tests for round-1/2 dropped designs.

These tests encode round-1/2 brutalist findings as machine-checkable
invariants. They prevent silent re-introduction of designs that were
explicitly ruled out:

1. `composite_label_v1` deprecation column (round 1: dropped because no
   release process existed to amortize the shim against).
2. ALR-style CLR with a chosen reference category (round 1: math error
   confused with CLR; revised CLR has no reference).
3. Confidence sweep `{0.5, 0.7, 0.9}` (round 1: empirically meaningless
   because actual values are `{0.0, 0.333, 0.5, 1.0}`).
4. Raw-marker corroborating path for Family A_v2
   `classify_celltype_per_superpixel_global_markers` (round 1: shared
   the entire taxonomy with the primary path; not real corroboration;
   spec defaults to single-path Option H1).

Failure to keep these invariants alive means a future engineer (or a future
self) can silently rebuild a previously-dropped design. Round-3 F8 process
attack: at what point does iteration become its own bad engineering?
The honest answer: when guard tests stop running.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_SUMMARY_PATH = REPO_ROOT / 'results' / 'biological_analysis' / 'temporal_interfaces' / 'endpoint_summary.csv'
ANNOTATION_PARQUETS = list((REPO_ROOT / 'results' / 'biological_analysis' / 'cell_type_annotations').glob('*_cell_types.parquet'))


def test_no_composite_label_v1_column_in_endpoint_summary():
    """Dropped post-round-1: deprecation column with no release process to
    amortize against. The single-commit rename pattern replaces it. If a
    future engineer adds `composite_label_v1`, this test fails loudly."""
    if not ENDPOINT_SUMMARY_PATH.exists():
        pytest.skip(f"endpoint_summary.csv not found at {ENDPOINT_SUMMARY_PATH}")
    df = pd.read_csv(ENDPOINT_SUMMARY_PATH, nrows=1)
    assert 'composite_label_v1' not in df.columns, (
        "composite_label_v1 column was dropped post-round-1 (no release "
        "process to amortize the deprecation shim). If you need it back, "
        "first re-open the round-1 finding in Appendix A and document why "
        "the rationale no longer applies."
    )


def test_no_composite_label_v1_in_annotation_parquets():
    """Same dropped design, checked at the annotation parquet level."""
    if not ANNOTATION_PARQUETS:
        pytest.skip("No annotation parquets found")
    for parquet_path in ANNOTATION_PARQUETS[:3]:  # Sample 3 ROIs
        df = pd.read_parquet(parquet_path, columns=None)
        assert 'composite_label_v1' not in df.columns, (
            f"composite_label_v1 column resurrected in {parquet_path.name}; "
            "see test_no_composite_label_v1_column_in_endpoint_summary docstring."
        )


def test_no_confidence_floor_sweep_column_in_endpoint_summary():
    """Dropped post-round-1: empirical confidence values are {0.0, 0.333,
    0.5, 1.0}, so floors of 0.7 and 0.9 partition identically. The sweep
    is meaningless. Phase 7 v2 uses `min_prevalence_sweep_value` instead."""
    if not ENDPOINT_SUMMARY_PATH.exists():
        pytest.skip(f"endpoint_summary.csv not found at {ENDPOINT_SUMMARY_PATH}")
    df = pd.read_csv(ENDPOINT_SUMMARY_PATH, nrows=1)
    assert 'confidence_floor' not in df.columns, (
        "confidence_floor column was dropped post-round-1 (sweep was "
        "empirically meaningless). Phase 7 uses min_prevalence_sweep_value "
        "instead; see Appendix A.1 round-1 findings."
    )


def test_no_clr_reference_category_column():
    """Dropped post-round-1: CLR has no reference category (geometric mean
    denominator, all N coordinates). The original `clr_reference_category`
    schema was ALR semantics confused with CLR."""
    if not ENDPOINT_SUMMARY_PATH.exists():
        pytest.skip(f"endpoint_summary.csv not found at {ENDPOINT_SUMMARY_PATH}")
    df = pd.read_csv(ENDPOINT_SUMMARY_PATH, nrows=1)
    assert 'clr_reference_category' not in df.columns, (
        "clr_reference_category column was dropped post-round-1 (CLR has no "
        "reference; the field name was an ALR/CLR confusion). See round-1 "
        "Codex Critical #1 disposition in Appendix A."
    )


def test_no_raw_marker_corroborating_path_function_in_temporal_interface_module():
    """Dropped post-round-1: classify_celltype_per_superpixel_global_markers
    would have been a "second path" sharing the entire taxonomy with the
    primary path; round-1 found this is not corroboration. Spec §1.1 locks
    Option H1 (single-path A_v2). If this function appears, the H2 design
    was reintroduced without re-evaluating the round-1 reasoning."""
    from src.analysis import temporal_interface_analysis as tia
    assert not hasattr(tia, 'classify_celltype_per_superpixel_global_markers'), (
        "classify_celltype_per_superpixel_global_markers was dropped post-round-1 "
        "(would not be corroboration; shared taxonomy with primary path). See "
        "Appendix A.4 in the Phase 7 spec for the rejection rationale."
    )


def test_phase7_required_columns_present_in_endpoint_summary():
    """Positive-side guard: confirm every Phase 7 schema-delta column from
    spec §6.3 is present. If any go missing in a future change, downstream
    consumers (notebooks, audit scripts) will break silently."""
    if not ENDPOINT_SUMMARY_PATH.exists():
        pytest.skip(f"endpoint_summary.csv not found at {ENDPOINT_SUMMARY_PATH}")
    df = pd.read_csv(ENDPOINT_SUMMARY_PATH, nrows=1)
    required = {
        'endpoint_axis',
        'stratifier_basis',
        'min_prevalence_sweep_value',
        'headline_rule_version',
        'headline_demoted_reason',
        'is_headline',
        'unassigned_rate_mouse_mean_1',
        'unassigned_rate_mouse_mean_2',
    }
    missing = required - set(df.columns)
    assert not missing, (
        f"Phase 7 required columns missing from endpoint_summary.csv: {sorted(missing)}. "
        "See spec §6.3 for the locked schema."
    )


def test_composite_label_values_all_carry_c_prefix_in_annotation_parquets():
    """Phase 7 §4.3 lock: every composite_label value emitted by the
    producer must carry the `c:` prefix. Catches a future change that
    silently reverts the prefix."""
    if not ANNOTATION_PARQUETS:
        pytest.skip("No annotation parquets found")
    seen: set = set()
    for parquet_path in ANNOTATION_PARQUETS[:5]:  # Sample 5 ROIs is enough
        df = pd.read_parquet(parquet_path, columns=['composite_label'])
        seen |= set(df['composite_label'].unique())
    unprefixed = [v for v in seen if not v.startswith('c:')]
    assert not unprefixed, (
        f"composite_label values without 'c:' prefix found: {sorted(unprefixed)[:5]}. "
        "Phase 7 §4.3 requires every composite_label value to carry the prefix; "
        "regenerate annotation parquets via batch_annotate_all_rois.py."
    )
