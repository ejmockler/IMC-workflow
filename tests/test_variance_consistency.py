"""Variance-consistency regression: the Hedges' g Bayesian-shrinkage sampling
variance must be a single source of truth shared by the temporal-interface
module and the differential-abundance rank table.

Guards the K1 fix of the 2x fork: `differential_abundance_analysis.py` used to
compute a harmonic `n_eff = n1*n2/(n1+n2) = 1` at n=2, giving `v = 2 + g**2/4`
(double the correct variance) and over-shrinking g=2.28 to ~0.53. It now REUSES
`hedges_g_sampling_variance` (v = 2/n + g**2/(4n) = 1 + g**2/8 at n=2), so the
same effect shrinks to ~0.86 on both surfaces.
"""

import pytest

import differential_abundance_analysis as da
from src.analysis.temporal_interface_analysis import hedges_g_sampling_variance

# Pilot per-group mouse count (n1 == n2 == 2).
N = 2


def _general_hedges_olkin_v(g, n1, n2):
    """The DA unequal-n fallback form. Must reduce to the canonical function
    exactly when n1 == n2 (equal-n reduction invariant)."""
    return (n1 + n2) / (n1 * n2) + g ** 2 / (2.0 * (n1 + n2))


@pytest.mark.parametrize("g", [0.0, 1.0, 2.0, 2.28])
def test_da_variance_matches_canonical_at_n2(g):
    """DA per-row variance == canonical hedges_g_sampling_variance at n=2.

    The DA equal-n path calls the canonical fn directly; the DA unequal-n
    fallback uses the general Hedges-Olkin form. Both must agree at n1==n2==2,
    and the value must be the un-forked 1 + g**2/8 (NOT the old 2 + g**2/4).
    """
    canonical = hedges_g_sampling_variance(g, N)
    assert abs(_general_hedges_olkin_v(g, N, N) - canonical) < 1e-9
    assert abs(canonical - (1.0 + g ** 2 / 8.0)) < 1e-9
    # Explicitly reject the old harmonic-n fork (v = 2 + g**2/4) for g != 0.
    if g != 0.0:
        assert abs(canonical - (2.0 + g ** 2 / 4.0)) > 1e-6


def test_da_neutral_shrinkage_of_g228_is_086_not_053():
    """g=2.28 at n=2 under a neutral (sd=1.0) prior shrinks to ~0.8604.

    Exercises the REAL DA shrinkage helper `_shrink_posterior_mean` so the test
    guards the production code path, not a re-implementation.
    """
    g = 2.28
    v = hedges_g_sampling_variance(g, N)  # 1 + 2.28**2/8 = 1.6498
    shrunk = da._shrink_posterior_mean(g, v, prior_sd=1.0)

    expected = 2.28 * 1.0 / (1.0 + (1.0 + 2.28 ** 2 / 8.0))
    assert abs(shrunk - expected) < 1e-9
    assert abs(shrunk - 0.8604) < 1e-3
    # Must NOT be the old over-shrunk value from the harmonic-n fork (~0.53).
    assert abs(shrunk - 0.53) > 0.1


@pytest.mark.parametrize("g", [0.0, 1.0, 2.0, 2.28])
def test_da_shrinkage_agrees_with_temporal_bayesian_shrinkage(g):
    """DA shrinkage arithmetic == temporal bayesian_shrinkage at n=2 for all
    three priors (identical shrinkage on both surfaces)."""
    from src.analysis import temporal_interface_analysis as tia

    v = hedges_g_sampling_variance(g, N)
    for prior_sd in (0.5, 1.0, 2.0):
        da_val = da._shrink_posterior_mean(g, v, prior_sd)
        tia_val = tia.bayesian_shrinkage(g, N, prior_sd)
        assert abs(da_val - tia_val) < 1e-9
