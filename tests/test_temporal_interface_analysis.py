"""Tests for src.analysis.temporal_interface_analysis.

Covers all three endpoint families plus statistical primitives. Uses synthetic
data with controlled compositional structure, known spatial patterns, and
known effect sizes.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import temporal_interface_analysis as tia


# ---------------------------------------------------------------------------
# Fixtures: synthetic annotations
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_annotations():
    """24 superpixels split across 4 ROIs, controlled lineage scores.

    Per-ROI design (6 superpixels each):
      ROI 0: 2 immune-only, 2 endo-only, 2 immune+endo
      ROI 1: 2 stromal-only, 2 immune+stromal, 2 triple
      ROI 2: 2 endo+stromal, 2 none, 2 immune-only
      ROI 3: 1 of each category (8 categories - reuse "none" twice)
    """
    rng = np.random.RandomState(0)
    rows = []
    layouts = {
        'roi_0': [('immune',)] * 2 + [('endothelial',)] * 2 + [('immune', 'endothelial')] * 2,
        'roi_1': [('stromal',)] * 2 + [('immune', 'stromal')] * 2 + [('immune', 'endothelial', 'stromal')] * 2,
        'roi_2': [('endothelial', 'stromal')] * 2 + [tuple()] * 2 + [('immune',)] * 2,
        'roi_3': [
            ('immune',), ('endothelial',), ('stromal',),
            ('immune', 'endothelial'), ('immune', 'stromal'), ('endothelial', 'stromal'),
        ],
    }
    sid = 0
    for roi, layout in layouts.items():
        for pos_lineages in layout:
            row = {
                'superpixel_id': f'sp_{sid}',
                'roi_id': f'IMC_241218_Alun_ROI_{roi}',  # fake; we'll override metadata
                'x': float(sid % 4),
                'y': float(sid // 4),
                'lineage_immune': 0.7 if 'immune' in pos_lineages else 0.1,
                'lineage_endothelial': 0.7 if 'endothelial' in pos_lineages else 0.1,
                'lineage_stromal': 0.7 if 'stromal' in pos_lineages else 0.1,
                'CD45': 5.0 if 'immune' in pos_lineages else 0.5,
                'CD31': 5.0 if 'endothelial' in pos_lineages else 0.5,
                'CD140b': 5.0 if 'stromal' in pos_lineages else 0.5,
                'CD44': rng.uniform(0, 8),
                'composite_label': '+'.join(sorted(pos_lineages)) if pos_lineages else 'none',
            }
            rows.append(row)
            sid += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Family A — Interface composition
# ---------------------------------------------------------------------------

def test_classify_interface_assigns_all_categories(synthetic_annotations):
    labels = tia.classify_interface_per_superpixel(synthetic_annotations, threshold=0.3)
    expected_cats = set(tia.INTERFACE_CATEGORIES)
    assigned = set(labels.unique())
    assert assigned.issubset(expected_cats), f"Unexpected labels: {assigned - expected_cats}"
    assert len(labels) == len(synthetic_annotations)


def test_interface_fractions_sum_to_one(synthetic_annotations):
    fractions = tia.compute_interface_fractions_per_roi(synthetic_annotations, threshold=0.3)
    fraction_cols = [c for c in tia.INTERFACE_CATEGORIES if c in fractions.columns]
    sums = fractions[fraction_cols].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)


def test_bayesian_multiplicative_replacement_preserves_row_sum():
    props = np.array([
        [0.5, 0.3, 0.2, 0.0],
        [0.0, 0.4, 0.6, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ])
    replaced = tia.bayesian_multiplicative_zero_replacement(props, n_samples=np.array([100, 100, 100]))
    np.testing.assert_allclose(replaced.sum(axis=1), 1.0, atol=1e-9)
    assert (replaced > 0).all(), "Zero replacement must yield strictly positive entries"


def test_clr_transform_zero_sum_property():
    props = np.array([
        [0.5, 0.3, 0.2],
        [0.1, 0.4, 0.5],
    ])
    clr = tia.clr_transform(props)
    np.testing.assert_allclose(clr.sum(axis=1), 0.0, atol=1e-9)


def test_clr_transform_rejects_zeros():
    with pytest.raises(ValueError):
        tia.clr_transform(np.array([[0.5, 0.0, 0.5]]))


def test_min_prevalence_filter_collapses_rare(synthetic_annotations):
    fractions = tia.compute_interface_fractions_per_roi(synthetic_annotations, threshold=0.3)
    annotated = tia.attach_roi_metadata(fractions)
    mouse_fractions = tia.aggregate_fractions_to_mouse_level(annotated)
    # 'none' is rare in this synthetic data; use a high threshold to force collapse
    filtered, collapsed = tia.apply_min_prevalence_filter(mouse_fractions, threshold=0.50)
    if collapsed:
        assert 'other_rare' in filtered.columns
        for c in collapsed:
            assert c not in filtered.columns


# ---------------------------------------------------------------------------
# Family B — Continuous neighborhood
# ---------------------------------------------------------------------------

def test_neighbor_minus_self_zero_in_homogeneous_region():
    """In a homogeneous region, neighbor mean equals self → delta ≈ 0."""
    n = 30
    coords = np.column_stack([np.arange(n), np.zeros(n)])
    df = pd.DataFrame({
        'superpixel_id': [f's{i}' for i in range(n)],
        'x': coords[:, 0], 'y': coords[:, 1],
        'lineage_immune': np.full(n, 0.7),
        'lineage_endothelial': np.full(n, 0.1),
        'lineage_stromal': np.full(n, 0.1),
    })
    deltas = tia.compute_knn_neighbor_lineage_scores(df, k=5)
    for col in tia.LINEAGE_COLS:
        np.testing.assert_allclose(deltas[f'delta_{col}'].values, 0.0, atol=1e-9)


def test_min_support_marks_below_threshold_labels(synthetic_annotations):
    """Below-min-support labels are emitted with flag, not silently dropped."""
    per_roi = tia.compute_neighbor_minus_self_per_roi(
        synthetic_annotations, roi_id='IMC_241218_Alun_ROI_roi_0',
        k=2, min_support=3,
    )
    if per_roi.empty:
        return
    # Below-support rows have flag set + NaN deltas
    below = per_roi[per_roi['below_min_support']]
    above = per_roi[~per_roi['below_min_support']]
    if not below.empty:
        delta_cols = [c for c in per_roi.columns if c.startswith('mean_delta_')]
        assert below[delta_cols].isna().all().all()
        assert (below['n_superpixels'] < 3).all()
    if not above.empty:
        assert (above['n_superpixels'] >= 3).all()


# ---------------------------------------------------------------------------
# Family C — Compartment activation with Sham-reference threshold
# ---------------------------------------------------------------------------

def test_sham_reference_threshold_uses_only_sham():
    """If a marker is high only in Sham, threshold should reflect Sham distribution."""
    df = pd.DataFrame({
        'CD45': np.concatenate([
            np.full(50, 10.0),  # Sham — high
            np.full(50, 1.0),   # D7 — low
        ]),
        'timepoint': ['Sham'] * 50 + ['D7'] * 50,
    })
    thresholds = tia.compute_sham_reference_thresholds(df, ['CD45'], percentile=75)
    # Sham 75th percentile of [10, 10, ...] = 10
    assert thresholds['CD45'] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Spatial coherence — join-count + Moran's I
# ---------------------------------------------------------------------------

def test_join_count_clustered_pattern_has_positive_z():
    """A spatially clustered binary pattern should produce z > 2."""
    rng = np.random.RandomState(0)
    n = 200
    coords = rng.uniform(0, 100, size=(n, 2))
    # Mark all points in left half as positive (spatially clustered)
    binary = (coords[:, 0] < 50).astype(int)
    result = tia.compute_join_count_bb(binary, coords, k=10, n_perm=200, seed=42)
    assert result['sufficient']
    assert result['z_score'] > 2.0, f"Expected z > 2 for clustered pattern, got {result['z_score']}"


def test_join_count_random_pattern_has_small_z():
    """A spatially random binary pattern should have |z| typically small."""
    rng = np.random.RandomState(1)
    n = 200
    coords = rng.uniform(0, 100, size=(n, 2))
    binary = rng.randint(0, 2, size=n)
    result = tia.compute_join_count_bb(binary, coords, k=10, n_perm=200, seed=7)
    assert result['sufficient']
    assert abs(result['z_score']) < 3.0, f"Expected |z| < 3 for random pattern, got {result['z_score']}"


def test_join_count_insufficient_positives():
    coords = np.random.uniform(0, 100, size=(100, 2))
    binary = np.zeros(100, dtype=int)
    binary[:5] = 1
    result = tia.compute_join_count_bb(binary, coords, k=10, n_perm=50, min_positive=10)
    assert not result['sufficient']
    assert np.isnan(result['z_score'])


def test_morans_i_clustered_positive():
    """Spatially smooth values give Moran's I > 0."""
    rng = np.random.RandomState(0)
    n = 200
    coords = rng.uniform(0, 100, size=(n, 2))
    # Smooth gradient: values increase with x
    values = coords[:, 0] + rng.normal(0, 1, size=n)
    moran = tia.compute_morans_i_continuous(values, coords, k=10)
    assert moran > 0.2, f"Expected I > 0.2 for smooth field, got {moran}"


def test_morans_i_random_near_zero():
    rng = np.random.RandomState(2)
    n = 200
    coords = rng.uniform(0, 100, size=(n, 2))
    values = rng.normal(size=n)
    moran = tia.compute_morans_i_continuous(values, coords, k=10)
    assert abs(moran) < 0.15, f"Expected |I| < 0.15 for random field, got {moran}"


# ---------------------------------------------------------------------------
# Statistics — Hedges' g, bootstrap, n_required, BH-FDR
# ---------------------------------------------------------------------------

def test_hedges_g_zero_for_identical_groups():
    g = tia.hedges_g(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert g == pytest.approx(0.0)


def test_hedges_g_positive_for_higher_second_group():
    # g2 > g1 → positive g (matches differential_abundance_analysis convention)
    g = tia.hedges_g(np.array([1.0, 2.0]), np.array([10.0, 11.0]))
    assert g > 0


def test_pathological_flag_fires_on_near_zero_variance():
    """Mimics the g=-4.87 stromal Sham_vs_D7 case."""
    g1 = np.array([0.4072, 0.4073])  # std ≈ 0.00007
    g2 = np.array([0.4083, 0.4084])  # std ≈ 0.00007
    diag = tia.compute_hedges_g_with_diagnostics(g1, g2)
    assert abs(diag.hedges_g) > 3, "Expected pathological |g| > 3"
    assert diag.pooled_std < 0.01
    assert diag.g_pathological is True


def test_pathological_flag_silent_for_normal_data():
    g1 = np.array([0.3, 0.5])
    g2 = np.array([0.7, 0.9])
    diag = tia.compute_hedges_g_with_diagnostics(g1, g2)
    assert diag.g_pathological is False


def test_bootstrap_range_at_n2_has_few_unique_values():
    g1 = np.array([0.1, 0.2])
    g2 = np.array([0.5, 0.6])
    result = tia.bootstrap_range_g(g1, g2, n_iter=5000, seed=42)
    # With n=2 per group and replacement, only 16 resample combinations exist
    # → at most ~9 unique Hedges' g values
    assert result['n_unique_values'] <= 16, f"Got {result['n_unique_values']} unique values"
    assert result['range_min'] <= result['range_max']


def test_bootstrap_deterministic():
    g1 = np.array([0.1, 0.2, 0.3])
    g2 = np.array([0.4, 0.5, 0.6])
    r1 = tia.bootstrap_range_g(g1, g2, n_iter=1000, seed=123)
    r2 = tia.bootstrap_range_g(g1, g2, n_iter=1000, seed=123)
    assert r1 == r2


def test_n_required_decreases_with_larger_g():
    n_small = tia.n_required_for_g(0.3)
    n_medium = tia.n_required_for_g(0.5)
    n_large = tia.n_required_for_g(1.0)
    assert n_small > n_medium > n_large
    assert n_large >= 2


def test_bayesian_shrinkage_zero_observed_returns_zero():
    """Zero observed g should shrink to exactly zero under any prior."""
    for prior_sd in (0.5, 1.0, 2.0):
        assert tia.bayesian_shrinkage(0.0, n_per_group=2, prior_sd=prior_sd) == 0.0


def test_bayesian_shrinkage_huge_g_more_under_skeptical():
    """A huge observed g shrinks more under a skeptical prior than optimistic."""
    g = 5.0
    g_skeptical = tia.bayesian_shrinkage(g, n_per_group=2, prior_sd=0.5)
    g_optimistic = tia.bayesian_shrinkage(g, n_per_group=2, prior_sd=2.0)
    assert abs(g_skeptical) < abs(g_optimistic) < abs(g), \
        f"Expected |skeptical| < |optimistic| < |g|; got {g_skeptical}, {g_optimistic}, {g}"


def test_bayesian_shrinkage_larger_n_reduces_shrinkage():
    """Larger n -> smaller sampling variance -> less shrinkage toward zero."""
    g = 1.0
    g_n2 = tia.bayesian_shrinkage(g, n_per_group=2, prior_sd=1.0)
    g_n10 = tia.bayesian_shrinkage(g, n_per_group=10, prior_sd=1.0)
    g_n50 = tia.bayesian_shrinkage(g, n_per_group=50, prior_sd=1.0)
    assert abs(g_n2) < abs(g_n10) < abs(g_n50) <= abs(g)


def test_bayesian_shrinkage_signs_preserved():
    """Shrinkage preserves sign (only shrinks magnitude toward zero)."""
    assert tia.bayesian_shrinkage(2.0, 2, 1.0) > 0
    assert tia.bayesian_shrinkage(-2.0, 2, 1.0) < 0


def test_bayesian_shrinkage_handles_nan():
    assert np.isnan(tia.bayesian_shrinkage(float('nan'), 2, 1.0))


def test_compute_hedges_g_with_diagnostics_emits_three_shrunk_values():
    """GDiagnostics now exposes a shrinkage range, not a single Type-M scalar."""
    diag = tia.compute_hedges_g_with_diagnostics(np.array([0.1, 0.2]), np.array([0.7, 0.8]))
    # All three shrunk values should be present, ordered: |skeptical| <= |neutral| <= |optimistic| <= |g|
    assert hasattr(diag, 'g_shrunk_skeptical')
    assert hasattr(diag, 'g_shrunk_neutral')
    assert hasattr(diag, 'g_shrunk_optimistic')
    assert abs(diag.g_shrunk_skeptical) <= abs(diag.g_shrunk_neutral) <= abs(diag.g_shrunk_optimistic) <= abs(diag.hedges_g)


def test_pathological_rows_emit_nan_shrunk_values():
    """Variance-collapse artifacts (pathological) should return NaN shrunk values,
    not plausible-looking shrunk numbers."""
    # Mimics the pathological stromal Sham_vs_D7 case (means differ by 0.001, std<0.001)
    g1 = np.array([0.4072, 0.4073])
    g2 = np.array([0.4083, 0.4084])
    diag = tia.compute_hedges_g_with_diagnostics(g1, g2)
    assert diag.g_pathological is True
    assert np.isnan(diag.g_shrunk_skeptical)
    assert np.isnan(diag.g_shrunk_neutral)
    assert np.isnan(diag.g_shrunk_optimistic)


def test_bayesian_shrinkage_numerical_regression():
    """Pin numerical values so silent formula changes fail the test.

    Using textbook Hedges & Olkin variance: v(g, n) = 2/n + g**2/(4n)
    For g=2.0, n=2 per group: v = 2/2 + 4/(4*2) = 1.0 + 0.5 = 1.5
    Posterior means with shrinkage factor = prior_var / (prior_var + v):
      prior_sd=0.5 (prior_var=0.25): 2.0 * 0.25/(0.25+1.5) = 2.0 * 0.1428... = 0.2857142857
      prior_sd=1.0 (prior_var=1.00): 2.0 * 1.00/(1.00+1.5) = 2.0 * 0.4       = 0.8
      prior_sd=2.0 (prior_var=4.00): 2.0 * 4.00/(4.00+1.5) = 2.0 * 0.7272... = 1.4545454545
    """
    g, n = 2.0, 2
    assert abs(tia.bayesian_shrinkage(g, n, prior_sd=0.5) - 0.2857142857) < 1e-6
    assert abs(tia.bayesian_shrinkage(g, n, prior_sd=1.0) - 0.8) < 1e-6
    assert abs(tia.bayesian_shrinkage(g, n, prior_sd=2.0) - 1.4545454545) < 1e-6


def test_hedges_g_sampling_variance_numerical_regression():
    """Pin the textbook Hedges & Olkin sampling variance formula."""
    # v(g=0, n=2) = 2/2 + 0 = 1.0
    assert abs(tia.hedges_g_sampling_variance(0.0, 2) - 1.0) < 1e-9
    # v(g=2, n=2) = 2/2 + 4/8 = 1.0 + 0.5 = 1.5
    assert abs(tia.hedges_g_sampling_variance(2.0, 2) - 1.5) < 1e-9
    # v(g=2, n=10) = 2/10 + 4/40 = 0.2 + 0.1 = 0.3
    assert abs(tia.hedges_g_sampling_variance(2.0, 10) - 0.3) < 1e-9


def test_benjamini_hochberg_monotone():
    p = [0.001, 0.01, 0.05, 0.1, 0.5]
    q = tia.benjamini_hochberg(p)
    # BH q-values should be non-decreasing in original p order
    assert all(q[i] <= q[i + 1] + 1e-9 for i in range(len(q) - 1))
    assert (q >= np.array(p)).all(), "BH adjustment must increase p"


def test_benjamini_hochberg_handles_nan():
    p = [0.001, np.nan, 0.05, 0.1]
    q = tia.benjamini_hochberg(p)
    assert np.isnan(q[1])
    assert not np.isnan(q[0])


# ---------------------------------------------------------------------------
# Edge cases added in response to Gate 1 brutalist review
# ---------------------------------------------------------------------------

def test_bayesian_replacement_all_zero_row_sums_to_one():
    """All-zero row should become uniform composition (1/D) summing to 1."""
    props = np.array([[0.0, 0.0, 0.0, 0.0]])
    out = tia.bayesian_multiplicative_zero_replacement(props, n_samples=np.array([100]))
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(out[0], 0.25, atol=1e-9)


def test_bayesian_replacement_zero_count_input():
    """When n_samples=0, output should still sum to 1 (degenerate but not crash)."""
    props = np.array([[0.5, 0.5, 0.0]])
    out = tia.bayesian_multiplicative_zero_replacement(props, n_samples=np.array([0.0]))
    # With N=0, posterior_num = alpha = 0.5, denom = 3*alpha = 1.5; each = 0.333
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)
    assert (out > 0).all()


def test_bayesian_replacement_preserves_proportional_structure():
    """Non-zero categories should remain near their original proportions for large N."""
    props = np.array([[0.5, 0.3, 0.2]])
    out = tia.bayesian_multiplicative_zero_replacement(props, n_samples=np.array([10000.0]))
    np.testing.assert_allclose(out[0], props[0], atol=0.01)


def test_clr_table_exclude_none_path():
    """compute_interface_clr_table with exclude_none=True drops 'none' before CLR."""
    df = pd.DataFrame({
        'timepoint': ['Sham', 'Sham', 'D7', 'D7'],
        'mouse': ['M1', 'M2', 'M1', 'M2'],
        'immune': [0.3, 0.4, 0.5, 0.45],
        'endothelial': [0.2, 0.25, 0.2, 0.15],
        'stromal': [0.2, 0.15, 0.2, 0.25],
        'endothelial+immune': [0.05, 0.05, 0.0, 0.05],
        'immune+stromal': [0.05, 0.05, 0.05, 0.05],
        'endothelial+stromal': [0.05, 0.05, 0.0, 0.0],
        'endothelial+immune+stromal': [0.05, 0.0, 0.0, 0.0],
        'none': [0.10, 0.05, 0.05, 0.05],
        'n_total': [1000, 1000, 1000, 1000],
    })
    with_none = tia.compute_interface_clr_table(df, exclude_none=False)
    without_none = tia.compute_interface_clr_table(df, exclude_none=True)
    assert 'none_clr' in with_none.columns
    assert 'none_clr' not in without_none.columns
    assert without_none.shape[0] == with_none.shape[0]


def test_hedges_g_with_nan_input_returns_nan():
    g = tia.hedges_g(np.array([1.0, np.nan]), np.array([2.0, 3.0]))
    assert np.isnan(g)


def test_pairwise_endpoint_table_emits_insufficient_row():
    """Contrasts with only 1 mouse must yield insufficient_support=True, not be dropped."""
    mouse_df = pd.DataFrame({
        'timepoint': ['Sham', 'Sham', 'D7'],  # D7 has only 1 mouse
        'mouse': ['M1', 'M2', 'M1'],
        'fraction': [0.3, 0.4, 0.5],
    })
    out = tia.pairwise_endpoint_table(
        mouse_df, value_cols=['fraction'], family='test',
        contrasts=[('Sham', 'D7')],
    )
    assert len(out) == 1
    assert bool(out['insufficient_support'].iloc[0]) is True
    assert np.isnan(out['hedges_g'].iloc[0])
    assert int(out['n_mice_1'].iloc[0]) == 2
    assert int(out['n_mice_2'].iloc[0]) == 1


def test_pairwise_endpoint_table_extra_index_cols():
    """extra_index_cols groups by composite_label as well as endpoint (Family B path)."""
    mouse_df = pd.DataFrame({
        'composite_label': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'timepoint': ['Sham', 'Sham', 'D7', 'D7', 'Sham', 'Sham', 'D7', 'D7'],
        'mouse': ['M1', 'M2', 'M1', 'M2'] * 2,
        'delta': [0.1, 0.15, 0.3, 0.35, 0.05, 0.06, 0.05, 0.07],
    })
    out = tia.pairwise_endpoint_table(
        mouse_df, value_cols=['delta'], family='B',
        contrasts=[('Sham', 'D7')],
        extra_index_cols=['composite_label'],
    )
    assert set(out['composite_label'].unique()) == {'A', 'B'}
    g_a = out.loc[out['composite_label'] == 'A', 'hedges_g'].iloc[0]
    g_b = out.loc[out['composite_label'] == 'B', 'hedges_g'].iloc[0]
    assert g_a > g_b, "Label A has larger Sham→D7 shift than B; expect larger g"


def test_apply_trajectory_filter_excludes_label_missing_in_any_timepoint():
    df = pd.DataFrame({
        'composite_label': ['A', 'A', 'A', 'A', 'B', 'B'],
        'timepoint': ['Sham', 'D1', 'D3', 'D7', 'Sham', 'D7'],
        'mouse': ['M1', 'M1', 'M1', 'M1', 'M1', 'M1'],
        'mean_delta_lineage_immune': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })
    filtered, missingness = tia.apply_trajectory_filter(df)
    assert set(filtered['composite_label'].unique()) == {'A'}
    b_rows = missingness[missingness['composite_label'] == 'B']
    assert (b_rows['kept_in_trajectory'] == False).all()
    assert int(b_rows[b_rows['timepoint'] == 'D1']['n_mice_with_valid_delta'].iloc[0]) == 0
    # Without per_roi_raw, status falls back to absent_biology vs sufficient
    assert b_rows[b_rows['timepoint'] == 'D1']['status'].iloc[0] == 'absent_biology'


def test_apply_trajectory_filter_distinguishes_below_support_from_absent():
    """When per_roi_raw is supplied, status must distinguish absent_biology
    from below_min_support (label appeared but had too few superpixels)."""
    mouse_deltas = pd.DataFrame({
        'composite_label': ['A', 'A'],
        'timepoint': ['Sham', 'D7'],
        'mouse': ['M1', 'M1'],
        'mean_delta_lineage_immune': [0.1, 0.4],
    })
    # Label A is missing from D1/D3 in mouse_deltas, but present in raw with too few superpixels
    per_roi_raw = pd.DataFrame({
        'roi_id': ['IMC_241218_Alun_ROI_D1_M1_01_9', 'IMC_241218_Alun_ROI_D3_M1_01_2'],
        'composite_label': ['A', 'A'],
        'n_superpixels': [3, 2],
        'below_min_support': [True, True],
        'mean_delta_lineage_immune': [np.nan, np.nan],
    })
    _, missingness = tia.apply_trajectory_filter(mouse_deltas, per_roi_raw=per_roi_raw)
    a_rows = missingness[missingness['composite_label'] == 'A']
    d1_status = a_rows[a_rows['timepoint'] == 'D1']['status'].iloc[0]
    assert d1_status == 'below_min_support'


def test_join_count_saturated_indicator_flagged():
    """All-positive binary should be flagged saturated, not just sufficient."""
    coords = np.random.RandomState(0).uniform(0, 100, size=(50, 2))
    binary = np.ones(50, dtype=int)
    out = tia.compute_join_count_bb(binary, coords, k=5, n_perm=50)
    assert out['saturated'] is True
    assert out['sufficient'] is False
    assert np.isnan(out['z_score'])


def test_morans_i_with_nan_values():
    rng = np.random.RandomState(0)
    n = 100
    coords = rng.uniform(0, 100, size=(n, 2))
    values = coords[:, 0].astype(float)
    values[5] = np.nan  # one NaN should not poison the result
    moran = tia.compute_morans_i_continuous(values, coords, k=10)
    assert not np.isnan(moran), "NaN values should be dropped, not propagate"


def test_compute_sham_reference_thresholds_raises_on_no_sham():
    df = pd.DataFrame({'CD45': [1.0, 2.0], 'timepoint': ['D1', 'D7']})
    with pytest.raises(ValueError, match="No Sham"):
        tia.compute_sham_reference_thresholds(df, ['CD45'])


def test_compute_sham_reference_thresholds_raises_on_all_nan_marker():
    df = pd.DataFrame({
        'CD45': [np.nan, np.nan, 1.0, 2.0],
        'timepoint': ['Sham', 'Sham', 'D7', 'D7'],
    })
    with pytest.raises(ValueError, match="no finite Sham"):
        tia.compute_sham_reference_thresholds(df, ['CD45'])


def test_add_pooled_fdr_proxy_removed():
    """Gate 6 seam closure: the add_pooled_fdr_proxy helper has been removed.
    At n=2 per group, no real p-value exists; FDR-adjusted proxies invite
    misinterpretation regardless of disclaimers.
    """
    assert not hasattr(tia, 'add_pooled_fdr_proxy'), \
        "add_pooled_fdr_proxy should be removed (Gate 6 seam closure)"


# ---------------------------------------------------------------------------
# Normalization sensitivity (Gate 6 seam closure)
# ---------------------------------------------------------------------------

def test_compute_global_marker_thresholds_sham_only_default():
    """Default sham_only=True restricts threshold computation to Sham timepoint.

    Critical: pooling across all timepoints lets injury-elevated markers drive
    the threshold (outcome contamination). Sham-only matches Family C
    philosophy and avoids this.
    """
    df = pd.DataFrame({
        'CD45': np.concatenate([np.full(50, 1.0), np.full(50, 10.0)]),  # sham low, D7 high
        'CD31': np.full(100, 2.0),
        'CD34': np.full(100, 4.0),
        'CD140a': np.arange(100, dtype=float),
        'timepoint': ['Sham'] * 50 + ['D7'] * 50,
    })
    sham_thr = tia.compute_global_marker_thresholds(df, percentile=75.0, sham_only=True)
    # Sham CD45 is constant 1.0; 75th percentile = 1.0 (unaffected by D7=10)
    assert abs(sham_thr['immune'] - 1.0) < 1e-9


def test_compute_global_marker_thresholds_pooled_includes_d7():
    """Explicit sham_only=False pools across timepoints; D7 elevated markers
    raise the threshold (the outcome-contamination Gate 6 critic flagged)."""
    df = pd.DataFrame({
        'CD45': np.concatenate([np.full(50, 1.0), np.full(50, 10.0)]),
        'CD31': np.full(100, 2.0),
        'CD34': np.full(100, 4.0),
        'CD140a': np.arange(100, dtype=float),
        'timepoint': ['Sham'] * 50 + ['D7'] * 50,
    })
    pooled_thr = tia.compute_global_marker_thresholds(df, percentile=75.0, sham_only=False)
    # Pooled 75th percentile of [1]*50 + [10]*50 = 10.0 (driven by D7)
    assert pooled_thr['immune'] > 5.0


def test_compute_global_marker_thresholds_sham_only_requires_timepoint():
    df = pd.DataFrame({'CD45': [1.0], 'CD31': [1.0], 'CD34': [1.0], 'CD140a': [1.0]})
    with pytest.raises(ValueError, match="sham_only=True requires"):
        tia.compute_global_marker_thresholds(df, sham_only=True)


def test_compute_global_marker_thresholds_missing_markers_errors():
    with pytest.raises(ValueError, match="Missing required raw-marker"):
        tia.compute_global_marker_thresholds(pd.DataFrame({'CD45': [1.0]}))


def test_classify_interface_global_markers_assigns_expected_categories():
    """Given raw-marker expressions and global thresholds, classification
    should match the standard interface category set."""
    df = pd.DataFrame({
        'CD45': [10, 0, 0, 10, 10, 0, 10, 0],   # rows 0,3,4,6 are immune+
        'CD31': [0, 10, 0, 10, 0, 10, 10, 0],   # rows 1,3,5,6 are endo+
        'CD34': [0, 10, 0, 10, 0, 10, 10, 0],
        'CD140a': [0, 0, 10, 0, 10, 10, 10, 0], # rows 2,4,5,6 are stromal+
    })
    thresholds = {'immune': 5.0, 'endothelial': 5.0, 'stromal': 5.0}
    labels = tia.classify_interface_per_superpixel_global_markers(df, thresholds)
    expected = [
        'immune',
        'endothelial',
        'stromal',
        'endothelial+immune',
        'immune+stromal',
        'endothelial+stromal',
        'endothelial+immune+stromal',
        'none',
    ]
    assert labels.tolist() == expected


def test_pairwise_endpoint_table_seed_determinism():
    mouse_df = pd.DataFrame({
        'timepoint': ['Sham', 'Sham', 'D7', 'D7'],
        'mouse': ['M1', 'M2', 'M1', 'M2'],
        'fraction': [0.3, 0.4, 0.5, 0.6],
    })
    out_a = tia.pairwise_endpoint_table(
        mouse_df, value_cols=['fraction'], family='test',
        contrasts=[('Sham', 'D7')], bootstrap_seed=99, bootstrap_iter=500,
    )
    out_b = tia.pairwise_endpoint_table(
        mouse_df, value_cols=['fraction'], family='test',
        contrasts=[('Sham', 'D7')], bootstrap_seed=99, bootstrap_iter=500,
    )
    pd.testing.assert_frame_equal(out_a, out_b)
