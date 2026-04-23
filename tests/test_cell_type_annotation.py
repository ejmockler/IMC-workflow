"""Regression + unit tests for the continuous-membership engine.

Key gate: the Sham-reference sigmoid must not compress temporal dynamics.
We assert this via a synthetic fixture where Sham and injury ROIs share a
panel but injury has elevated stromal marker expression. The frozen
reference (Sham-only) must then score injury superpixels higher on the
stromal lineage than Sham superpixels — the property that per-ROI sigmoid
normalization failed to preserve (Gate 6 seam).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.cell_type_annotation import compute_continuous_memberships
from src.analysis.sham_reference import build_reference_distribution


PROTEIN_CHANNELS = ['CD45', 'CD31', 'CD34', 'CD140a', 'CD140b', 'CD44',
                    'CD11b', 'Ly6G', 'CD206']

MEMBERSHIP_CONFIG = {
    'normalization': 'sigmoid_threshold',
    'sigmoid_steepness': 10.0,
    'lineages': {
        'immune': {'markers': ['CD45'], 'aggregation': 'max'},
        'endothelial': {'markers': ['CD31', 'CD34'], 'aggregation': 'mean'},
        'stromal': {'markers': ['CD140a'], 'aggregation': 'max'},
    },
    'subtypes': {
        'subtype_threshold': 0.3,
        'definitions': {},  # keep minimal for these tests
    },
    'activation': {
        'markers': {'cd44': 'CD44', 'cd140b': 'CD140b'},
    },
    'composite_label_thresholds': {
        'lineage': 0.3, 'activation': 0.3, 'dominance': 2.0,
    },
}

THRESHOLD_CONFIG = {
    'method': 'percentile',
    'percentile': 60,
    'per_marker_override': {},
}


def _synth_panel(n_rows, cd45=0.0, cd31=0.0, cd34=0.0, cd140a=0.0,
                 cd140b=0.0, cd44=0.0, cd11b=0.0, ly6g=0.0, cd206=0.0,
                 rng=None):
    """Build [n_rows, 9] expression matrix with the given mean per marker."""
    if rng is None:
        rng = np.random.default_rng(0)
    means = np.array([cd45, cd31, cd34, cd140a, cd140b, cd44, cd11b, ly6g, cd206])
    return means + rng.normal(0, 0.1, size=(n_rows, 9))


def _annotations_frame(sham_matrix, injury_matrix):
    """Build a long-form DataFrame for reference building."""
    rows = []
    for i, row in enumerate(sham_matrix):
        d = {m: row[j] for j, m in enumerate(PROTEIN_CHANNELS)}
        d['timepoint'] = 'Sham'
        d['mouse'] = 'MS1' if i < len(sham_matrix) // 2 else 'MS2'
        rows.append(d)
    for i, row in enumerate(injury_matrix):
        d = {m: row[j] for j, m in enumerate(PROTEIN_CHANNELS)}
        d['timepoint'] = 'D3'
        d['mouse'] = 'MS1' if i < len(injury_matrix) // 2 else 'MS2'
        rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regression fixture — the core Gate-6 property
# ---------------------------------------------------------------------------

def test_sham_reference_preserves_injury_elevation():
    """Synthetic dataset: Sham has CD140a ~ 0.1, D3 has CD140a ~ 1.5.

    Under the Sham-reference sigmoid, injury superpixels must score higher
    on stromal lineage than Sham superpixels. Under the old per-ROI path,
    each ROI centers on its own 60th percentile, so this distinction
    collapses. This test would have failed under the old engine.
    """
    rng = np.random.default_rng(42)
    sham = _synth_panel(
        100, cd45=0.1, cd31=0.1, cd34=0.1, cd140a=0.1, rng=rng,
    )
    injury = _synth_panel(
        100, cd45=0.1, cd31=0.1, cd34=0.1, cd140a=1.5, rng=rng,
    )

    # Build reference from Sham only
    ann_df = _annotations_frame(sham, injury)
    reference = build_reference_distribution(
        ann_df, PROTEIN_CHANNELS, percentile=60.0,
    )

    # Apply to each ROI independently
    sham_mem = compute_continuous_memberships(
        sham, PROTEIN_CHANNELS, MEMBERSHIP_CONFIG, THRESHOLD_CONFIG,
        reference_distribution=reference,
    )
    injury_mem = compute_continuous_memberships(
        injury, PROTEIN_CHANNELS, MEMBERSHIP_CONFIG, THRESHOLD_CONFIG,
        reference_distribution=reference,
    )

    sham_stromal_mean = float(sham_mem['lineage_scores']['stromal'].mean())
    injury_stromal_mean = float(injury_mem['lineage_scores']['stromal'].mean())

    assert injury_stromal_mean > 0.8, (
        f"injury stromal should score high on Sham-ref sigmoid "
        f"(CD140a=1.5 well above Sham pct60); got {injury_stromal_mean:.3f}"
    )
    assert sham_stromal_mean < 0.6, (
        f"Sham stromal should score near 0.5 (centered on Sham percentile) "
        f"or lower; got {sham_stromal_mean:.3f}"
    )
    # The headline property: injury dynamically above Sham
    assert injury_stromal_mean - sham_stromal_mean > 0.3


def test_sham_reference_is_shared_across_rois():
    """Two injury ROIs with identical data must produce identical scores.

    Under per-ROI normalization, two ROIs with different internal ranges
    get different sigmoid parameters and thus different scores for the same
    raw value. Under Sham-reference, they must agree.
    """
    rng = np.random.default_rng(42)
    sham = _synth_panel(100, cd45=0.1, cd140a=0.1, cd31=0.1, cd34=0.1, rng=rng)

    ann_df = _annotations_frame(sham, sham)  # placeholder injury=Sham copy
    reference = build_reference_distribution(
        ann_df, PROTEIN_CHANNELS, percentile=60.0,
    )

    # Two ROIs with identical data but different rng draws for other markers
    roi_a = _synth_panel(50, cd45=0.5, cd140a=0.8, rng=np.random.default_rng(1))
    roi_b = _synth_panel(50, cd45=0.5, cd140a=0.8, rng=np.random.default_rng(2))

    mem_a = compute_continuous_memberships(
        roi_a, PROTEIN_CHANNELS, MEMBERSHIP_CONFIG, THRESHOLD_CONFIG,
        reference_distribution=reference,
    )
    mem_b = compute_continuous_memberships(
        roi_b, PROTEIN_CHANNELS, MEMBERSHIP_CONFIG, THRESHOLD_CONFIG,
        reference_distribution=reference,
    )

    # Different noise but same means → means of lineage scores close
    a_mean = float(mem_a['lineage_scores']['stromal'].mean())
    b_mean = float(mem_b['lineage_scores']['stromal'].mean())
    assert abs(a_mean - b_mean) < 0.05


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------

def test_missing_reference_distribution_raises():
    panel = _synth_panel(10)
    with pytest.raises(TypeError):
        # Required positional arg — calling without it raises TypeError.
        compute_continuous_memberships(
            panel, PROTEIN_CHANNELS, MEMBERSHIP_CONFIG, THRESHOLD_CONFIG,
        )


def test_none_reference_distribution_raises():
    panel = _synth_panel(10)
    with pytest.raises(ValueError, match="reference_distribution is required"):
        compute_continuous_memberships(
            panel, PROTEIN_CHANNELS, MEMBERSHIP_CONFIG, THRESHOLD_CONFIG,
            reference_distribution=None,
        )


def test_empty_reference_distribution_raises():
    panel = _synth_panel(10)
    with pytest.raises(ValueError, match="non-empty dict"):
        compute_continuous_memberships(
            panel, PROTEIN_CHANNELS, MEMBERSHIP_CONFIG, THRESHOLD_CONFIG,
            reference_distribution={},
        )


def test_reference_missing_needed_marker_raises():
    panel = _synth_panel(10)
    # Reference has everything except CD140a (stromal lineage marker)
    ref = {
        m: {'threshold': 0.5, 'scale': 1.0}
        for m in PROTEIN_CHANNELS if m != 'CD140a'
    }
    with pytest.raises(ValueError, match="CD140a"):
        compute_continuous_memberships(
            panel, PROTEIN_CHANNELS, MEMBERSHIP_CONFIG, THRESHOLD_CONFIG,
            reference_distribution=ref,
        )


def test_reference_entry_missing_scale_raises():
    panel = _synth_panel(10)
    ref = {m: {'threshold': 0.5, 'scale': 1.0} for m in PROTEIN_CHANNELS}
    # Strip scale from one entry
    ref['CD45'] = {'threshold': 0.5}
    with pytest.raises(ValueError, match="must have both 'threshold' and 'scale'"):
        compute_continuous_memberships(
            panel, PROTEIN_CHANNELS, MEMBERSHIP_CONFIG, THRESHOLD_CONFIG,
            reference_distribution=ref,
        )


def test_panel_missing_needed_marker_raises():
    panel = _synth_panel(10)[:, :5]  # drop last 4 columns
    short_panel = PROTEIN_CHANNELS[:5]
    ref = {m: {'threshold': 0.5, 'scale': 1.0} for m in PROTEIN_CHANNELS}
    with pytest.raises(ValueError, match="missing from panel"):
        compute_continuous_memberships(
            panel, short_panel, MEMBERSHIP_CONFIG, THRESHOLD_CONFIG,
            reference_distribution=ref,
        )


# ---------------------------------------------------------------------------
# Score shape & range
# ---------------------------------------------------------------------------

def test_lineage_scores_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    panel = _synth_panel(200, cd45=0.5, cd140a=0.8, cd31=0.3, cd34=0.3,
                         cd11b=0.4, rng=rng)
    ref = {m: {'threshold': 0.5, 'scale': 0.2} for m in PROTEIN_CHANNELS}
    mem = compute_continuous_memberships(
        panel, PROTEIN_CHANNELS, MEMBERSHIP_CONFIG, THRESHOLD_CONFIG,
        reference_distribution=ref,
    )
    for lineage, scores in mem['lineage_scores'].items():
        assert np.all((scores >= 0.0) & (scores <= 1.0)), (
            f"lineage {lineage} produced out-of-range scores: "
            f"min={scores.min()}, max={scores.max()}"
        )


def test_normalization_params_record_threshold_scale():
    """Provenance: output should record the threshold + scale from reference."""
    panel = _synth_panel(10)
    ref = {m: {'threshold': 0.42, 'scale': 0.17} for m in PROTEIN_CHANNELS}
    mem = compute_continuous_memberships(
        panel, PROTEIN_CHANNELS, MEMBERSHIP_CONFIG, THRESHOLD_CONFIG,
        reference_distribution=ref,
    )
    norm = mem['normalization_params']
    # CD45 is used by immune lineage -> should have sham_reference_sigmoid
    assert norm['CD45']['method'] == 'sham_reference_sigmoid'
    assert norm['CD45']['threshold'] == pytest.approx(0.42)
    assert norm['CD45']['scale_denom'] == pytest.approx(0.17)
