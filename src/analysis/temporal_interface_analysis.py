"""
Temporal Multi-Lineage Interface Analysis.

Implements the three pre-registered endpoint families from
`analysis_plans/temporal_interfaces_plan.md`:

  Family A: Interface composition (8 categories, CLR-transformed)
  Family B: Continuous neighborhood lineage shifts (neighbor-minus-self delta)
  Family C: Cross-compartment activation trajectories (Sham-reference threshold)

Plus spatial coherence (join-count statistics + Moran's I on continuous scores)
and a single endpoint_summary table for PI/reviewer consumption.

Pure-function contract: every function takes DataFrames/arrays in and returns
DataFrames/arrays out. No prints, no file I/O. Determinism via explicit seeds.

Note: the statistical primitives (hedges_g, bootstrap, BH-FDR) are reimplemented
here rather than imported from the root-level differential_abundance_analysis.py
script. A future refactor could promote them to src/analysis/statistical_utils.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gmean

from src.utils.metadata import parse_roi_metadata


# ---------------------------------------------------------------------------
# Constants (frozen by analysis_plans/temporal_interfaces_plan.md)
# ---------------------------------------------------------------------------

LINEAGE_COLS: Tuple[str, ...] = ('lineage_immune', 'lineage_endothelial', 'lineage_stromal')
LINEAGE_NAMES: Tuple[str, ...] = ('immune', 'endothelial', 'stromal')

INTERFACE_CATEGORIES: Tuple[str, ...] = (
    'immune', 'endothelial', 'stromal',
    'endothelial+immune', 'immune+stromal', 'endothelial+stromal',
    'endothelial+immune+stromal',
    'none',
)

DEFAULT_LINEAGE_THRESHOLD: float = 0.3
DEFAULT_MIN_SUPPORT: int = 20
DEFAULT_MIN_PREVALENCE: float = 0.01
DEFAULT_BOOTSTRAP_ITER: int = 10_000
DEFAULT_PERMUTATIONS: int = 1_000
DEFAULT_K_NEIGHBORS: int = 10
DEFAULT_SHAM_PERCENTILE: float = 75.0

PATHOLOGY_G_THRESHOLD: float = 3.0
PATHOLOGY_STD_THRESHOLD: float = 0.01

# Type-M shrinkage factor. Gelman & Carlin (2014) design analysis shows that for
# pilot studies that fail to reject (our case at n=2), the observed effect size
# is a noisy estimate whose magnitude is on average inflated. Their framework
# does NOT prescribe a universal scalar; the true exaggeration ratio depends on
# the unknown true effect size, n, and alpha. We use 0.65 as a midpoint
# ballpark for moderate-to-large g at very small n, derived from Figure 2 of
# Gelman & Carlin (2014). This is a conservative *lower-bound* shrinkage:
# realistic Type-M ratios at n=2 are likely 2-4x (i.e. true effect ~0.25-0.5x
# observed), not 1/0.65=1.54x. The column g_type_m_corrected in
# endpoint_summary.csv should therefore be read as "one plausible shrinkage
# under a permissive assumption", NOT as a calibrated estimate of the true
# effect. Realistic study-design planning should assume effect sizes 50-75%
# smaller than observed Hedges' g, and treat n_required_80pct_type_m as a
# lower bound on the required sample size.
TYPE_M_CORRECTION: float = 0.65

TIMEPOINT_ORDER: Tuple[str, ...] = ('Sham', 'D1', 'D3', 'D7')
PAIRWISE_CONTRASTS: Tuple[Tuple[str, str], ...] = (
    ('Sham', 'D1'), ('Sham', 'D3'), ('Sham', 'D7'),
    ('D1', 'D3'), ('D1', 'D7'),
    ('D3', 'D7'),
)


# ---------------------------------------------------------------------------
# Family A — Interface composition
# ---------------------------------------------------------------------------

def classify_interface_per_superpixel(
    annotations: pd.DataFrame,
    threshold: float = DEFAULT_LINEAGE_THRESHOLD,
) -> pd.Series:
    """Assign each superpixel to one of the 8 interface categories."""
    above = {ln: annotations[col] > threshold for ln, col in zip(LINEAGE_NAMES, LINEAGE_COLS)}
    n_above = sum(above.values())

    labels = pd.Series('none', index=annotations.index, dtype=object)
    for ln in LINEAGE_NAMES:
        only_this = above[ln] & (n_above == 1)
        labels[only_this] = ln

    for i, ln1 in enumerate(LINEAGE_NAMES):
        for ln2 in LINEAGE_NAMES[i + 1:]:
            both = above[ln1] & above[ln2] & (n_above == 2)
            labels[both] = '+'.join(sorted([ln1, ln2]))

    triple = above[LINEAGE_NAMES[0]] & above[LINEAGE_NAMES[1]] & above[LINEAGE_NAMES[2]]
    labels[triple] = '+'.join(sorted(LINEAGE_NAMES))

    return labels


def compute_interface_fractions_per_roi(
    annotations: pd.DataFrame,
    roi_id_col: str = 'roi_id',
    threshold: float = DEFAULT_LINEAGE_THRESHOLD,
) -> pd.DataFrame:
    """ROI x category fraction table. Includes lineage-positive denominator counts."""
    labels = classify_interface_per_superpixel(annotations, threshold=threshold)
    df = annotations[[roi_id_col]].copy()
    df['interface'] = labels.values

    counts = (
        df.groupby([roi_id_col, 'interface'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(INTERFACE_CATEGORIES), fill_value=0)
    )
    totals = counts.sum(axis=1)
    fractions = counts.div(totals, axis=0)
    fractions['n_total'] = totals
    fractions['n_lineage_positive'] = totals - counts['none']
    fractions = fractions.reset_index()
    return fractions


def attach_roi_metadata(roi_fractions: pd.DataFrame, roi_id_col: str = 'roi_id') -> pd.DataFrame:
    """Add timepoint and mouse columns by parsing each roi_id."""
    metadata = roi_fractions[roi_id_col].apply(parse_roi_metadata).apply(pd.Series)
    return pd.concat(
        [roi_fractions, metadata[['timepoint', 'mouse']]],
        axis=1,
    )


def aggregate_fractions_to_mouse_level(roi_fractions: pd.DataFrame) -> pd.DataFrame:
    """Mouse-level mean of ROI fractions; sums for count columns."""
    fraction_cols = [c for c in INTERFACE_CATEGORIES if c in roi_fractions.columns]
    count_cols = [c for c in ('n_total', 'n_lineage_positive') if c in roi_fractions.columns]

    agg_map = {c: 'mean' for c in fraction_cols}
    agg_map.update({c: 'sum' for c in count_cols})
    mouse_df = (
        roi_fractions.groupby(['timepoint', 'mouse'], dropna=False)
        .agg(agg_map)
        .reset_index()
    )
    return mouse_df


def bayesian_multiplicative_zero_replacement(
    proportions: np.ndarray,
    n_samples: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Bayesian zero replacement under a Dirichlet(alpha, ..., alpha) prior.

    Posterior mean for category j given count c_j and total N over D categories:
        p_j_new = (c_j + alpha) / (N + D * alpha)

    For zero counts this yields alpha / (N + D*alpha) > 0; for non-zero counts
    it yields a slightly-shrunk version of the original proportion. The row
    sums to 1 by construction (sum of posterior counts = N + D*alpha).

    Default alpha=0.5 corresponds to the Jeffreys prior. References:
    Palarea-Albaladejo & Martin-Fernandez (2008), "A modified EM
    alr-algorithm for replacing rounded zeros in compositional data sets."

    Args:
        proportions: shape (n_rows, n_categories) or (n_categories,)
        n_samples: per-row total count (e.g., superpixels). If None, defaults
            to a generic 100 — caller should provide actual counts.
        alpha: Dirichlet concentration. 0.5 = Jeffreys.

    Returns:
        Replaced proportions, strictly positive, row-summing to 1.
    """
    arr = np.asarray(proportions, dtype=float)
    one_dim_input = arr.ndim == 1
    if one_dim_input:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1- or 2-D input; got ndim={arr.ndim}")
    n_rows, n_cats = arr.shape
    if n_samples is None:
        n_per_row = np.full(n_rows, 100.0)
    else:
        n_per_row = np.asarray(n_samples, dtype=float).reshape(n_rows)

    # Convert proportions back to expected counts; apply posterior; renormalize.
    # All-zero rows (sum=0) carry no information; return uniform 1/D as a
    # degenerate but valid composition rather than a non-summing vector.
    counts = arr * n_per_row[:, None]
    posterior_num = counts + alpha
    posterior_denom = n_per_row[:, None] + n_cats * alpha
    out = posterior_num / posterior_denom

    row_sums_in = arr.sum(axis=1)
    all_zero = row_sums_in == 0
    if all_zero.any():
        out[all_zero] = 1.0 / n_cats

    return out.ravel() if one_dim_input else out


def clr_transform(proportions: np.ndarray) -> np.ndarray:
    """Centered log-ratio transform. Input must already have zeros replaced."""
    arr = np.asarray(proportions, dtype=float)
    one_dim_input = arr.ndim == 1
    if one_dim_input:
        arr = arr.reshape(1, -1)
    if (arr <= 0).any():
        raise ValueError("CLR requires strictly positive entries; apply zero replacement first.")
    geo = gmean(arr, axis=1, keepdims=True)
    clr = np.log(arr / geo)
    return clr.ravel() if one_dim_input else clr


def apply_min_prevalence_filter(
    mouse_fractions: pd.DataFrame,
    threshold: float = DEFAULT_MIN_PREVALENCE,
) -> Tuple[pd.DataFrame, List[str]]:
    """Categories below `threshold` in *all* timepoints are merged into 'other_rare'.

    Returns (filtered_df, list_of_collapsed_categories).
    """
    fraction_cols = [c for c in INTERFACE_CATEGORIES if c in mouse_fractions.columns]
    timepoint_means = mouse_fractions.groupby('timepoint')[fraction_cols].mean()
    rare = [
        c for c in fraction_cols
        if (timepoint_means[c] < threshold).all()
    ]
    if not rare:
        return mouse_fractions.copy(), []

    out = mouse_fractions.copy()
    out['other_rare'] = out[rare].sum(axis=1)
    out = out.drop(columns=rare)
    return out, rare


def compute_interface_clr_table(
    mouse_fractions: pd.DataFrame,
    exclude_none: bool = False,
) -> pd.DataFrame:
    """Compute CLR-transformed interface composition table.

    If exclude_none=True, the 'none' category is dropped before CLR (sensitivity).
    """
    fraction_cols = [
        c for c in mouse_fractions.columns
        if c in INTERFACE_CATEGORIES or c == 'other_rare'
    ]
    if exclude_none and 'none' in fraction_cols:
        fraction_cols = [c for c in fraction_cols if c != 'none']

    props = mouse_fractions[fraction_cols].values
    n_per_row = mouse_fractions['n_total'].values if 'n_total' in mouse_fractions.columns else None
    replaced = bayesian_multiplicative_zero_replacement(props, n_samples=n_per_row)
    clr_vals = clr_transform(replaced)

    clr_df = pd.DataFrame(
        clr_vals,
        columns=[f'{c}_clr' for c in fraction_cols],
        index=mouse_fractions.index,
    )
    out = pd.concat(
        [mouse_fractions[['timepoint', 'mouse']].reset_index(drop=True),
         clr_df.reset_index(drop=True)],
        axis=1,
    )
    return out


# ---------------------------------------------------------------------------
# Family B — Continuous neighborhood (neighbor-minus-self)
# ---------------------------------------------------------------------------

def compute_knn_neighbor_lineage_scores(
    annotations: pd.DataFrame,
    k: int = DEFAULT_K_NEIGHBORS,
) -> pd.DataFrame:
    """For each superpixel, mean lineage scores of k nearest neighbors.

    Returns DataFrame with columns: superpixel_id, neighbor_lineage_immune,
    neighbor_lineage_endothelial, neighbor_lineage_stromal.
    """
    from scipy.spatial import cKDTree

    coords = annotations[['x', 'y']].values
    if len(coords) <= k:
        return pd.DataFrame()

    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # +1 includes self; drop column 0
    neighbors = idx[:, 1:]

    out = {'superpixel_id': annotations['superpixel_id'].values}
    for col in LINEAGE_COLS:
        scores = annotations[col].values
        out[f'neighbor_{col}'] = scores[neighbors].mean(axis=1)
        out[f'self_{col}'] = scores
        out[f'delta_{col}'] = out[f'neighbor_{col}'] - scores
    return pd.DataFrame(out)


def compute_neighbor_minus_self_per_roi(
    annotations: pd.DataFrame,
    roi_id: str,
    k: int = DEFAULT_K_NEIGHBORS,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> pd.DataFrame:
    """Per (composite_label x neighbor_lineage), mean delta within ROI.

    Emits ALL composite labels present in the ROI (not just those above
    min_support) with a `below_min_support` flag. This preserves the
    distinction between "absent biology" (label never appeared) and
    "present but below support threshold" (label existed but had too few
    superpixels for stable mean).
    """
    if 'composite_label' not in annotations.columns:
        return pd.DataFrame()
    deltas = compute_knn_neighbor_lineage_scores(annotations, k=k)
    if deltas.empty:
        return pd.DataFrame()
    deltas['composite_label'] = annotations['composite_label'].values

    rows = []
    for label, group in deltas.groupby('composite_label'):
        n = len(group)
        below_min = n < min_support
        row: Dict[str, object] = {
            'roi_id': roi_id,
            'composite_label': label,
            'n_superpixels': n,
            'below_min_support': bool(below_min),
        }
        for col in LINEAGE_COLS:
            if below_min:
                row[f'mean_delta_{col}'] = np.nan
            else:
                row[f'mean_delta_{col}'] = float(group[f'delta_{col}'].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_neighbor_delta_to_mouse(per_roi: pd.DataFrame) -> pd.DataFrame:
    """Mouse-level mean of per-ROI neighbor-minus-self deltas."""
    if per_roi.empty:
        return per_roi
    annotated = attach_roi_metadata(per_roi)
    delta_cols = [c for c in annotated.columns if c.startswith('mean_delta_')]
    agg_map = {c: 'mean' for c in delta_cols}
    agg_map['n_superpixels'] = 'sum'
    return (
        annotated.groupby(['composite_label', 'timepoint', 'mouse'], dropna=False)
        .agg(agg_map)
        .reset_index()
    )


def apply_trajectory_filter(
    mouse_deltas: pd.DataFrame,
    required_timepoints: Sequence[str] = TIMEPOINT_ORDER,
    label_col: str = 'composite_label',
    per_roi_raw: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Enforce plan rule: a label must have ≥1 mouse in EVERY required timepoint
    or it is excluded from the entire trajectory analysis.

    If `per_roi_raw` is supplied (the unfiltered per-ROI table including
    below_min_support rows), the missingness report will distinguish
    'absent_biology' (label never appeared in ROI) from 'below_min_support'
    (label appeared but with too few superpixels).

    Returns (filtered_df, missingness_df).
    """
    if mouse_deltas.empty:
        return mouse_deltas, pd.DataFrame()

    # Has-data presence (label has at least one mouse with valid delta in this timepoint)
    presence = (
        mouse_deltas.groupby([label_col, 'timepoint']).size().unstack(fill_value=0)
        .reindex(columns=list(required_timepoints), fill_value=0)
    )
    keep = (presence > 0).all(axis=1)
    kept_labels = presence.index[keep].tolist()

    # Raw presence (label appeared at all in ROI, even below support)
    if per_roi_raw is not None and not per_roi_raw.empty:
        raw_with_meta = attach_roi_metadata(per_roi_raw)
        raw_presence = (
            raw_with_meta.groupby([label_col, 'timepoint']).size().unstack(fill_value=0)
            .reindex(columns=list(required_timepoints), fill_value=0)
        )
    else:
        raw_presence = presence

    missingness_rows = []
    for label in presence.index:
        for tp in required_timepoints:
            n_with_data = int(presence.loc[label, tp]) if label in presence.index else 0
            n_raw = int(raw_presence.loc[label, tp]) if label in raw_presence.index else 0
            if n_with_data > 0:
                status = 'sufficient'
            elif n_raw > 0:
                status = 'below_min_support'
            else:
                status = 'absent_biology'
            missingness_rows.append({
                label_col: label,
                'timepoint': tp,
                'n_mice_with_valid_delta': n_with_data,
                'n_mice_present_in_raw': n_raw,
                'status': status,
                'kept_in_trajectory': label in kept_labels,
            })
    missingness = pd.DataFrame(missingness_rows)
    filtered = mouse_deltas[mouse_deltas[label_col].isin(kept_labels)].copy()
    return filtered, missingness


def compute_delta_vs_sham(mouse_deltas: pd.DataFrame) -> pd.DataFrame:
    """Subtract Sham-mean delta from each timepoint's delta, per (label x lineage)."""
    if mouse_deltas.empty:
        return mouse_deltas
    delta_cols = [c for c in mouse_deltas.columns if c.startswith('mean_delta_')]
    sham = (
        mouse_deltas[mouse_deltas['timepoint'] == 'Sham']
        .groupby('composite_label')[delta_cols]
        .mean()
    )
    out = mouse_deltas.copy()
    for col in delta_cols:
        sham_lookup = sham[col].to_dict()
        out[f'vs_sham_{col}'] = out.apply(
            lambda r: r[col] - sham_lookup.get(r['composite_label'], np.nan),
            axis=1,
        )
    # Note: Sham vs_sham residuals are individual mouse deviations from the
    # Sham group mean (NOT zero per mouse). The group mean of these residuals
    # is mathematically zero; floating-point gives ~1e-17 in the endpoint table.
    return out


# ---------------------------------------------------------------------------
# Family C — Cross-compartment activation (Sham-reference threshold)
# ---------------------------------------------------------------------------

def compute_sham_reference_thresholds(
    annotations_with_metadata: pd.DataFrame,
    markers: Sequence[str],
    percentile: float = DEFAULT_SHAM_PERCENTILE,
) -> Dict[str, float]:
    """Global per-marker threshold from Sham superpixels only.

    Uses nanpercentile to ignore NaN entries; raises if a marker has zero
    finite Sham values (otherwise the threshold would silently be NaN and
    every downstream comparison would evaluate False).
    """
    sham = annotations_with_metadata[annotations_with_metadata['timepoint'] == 'Sham']
    if sham.empty:
        raise ValueError("No Sham superpixels found; cannot compute reference thresholds.")
    thresholds: Dict[str, float] = {}
    for m in markers:
        if m not in sham.columns:
            raise ValueError(f"Marker '{m}' missing from annotations.")
        finite = sham[m].values[np.isfinite(sham[m].values)]
        if finite.size == 0:
            raise ValueError(f"Marker '{m}' has no finite Sham values.")
        thresholds[m] = float(np.percentile(finite, percentile))
    return thresholds


def compute_compartment_activation_per_roi(
    annotations: pd.DataFrame,
    sham_thresholds: Dict[str, float],
    compartment_markers: Sequence[str] = ('CD45', 'CD31', 'CD140b'),
    activation_marker: str = 'CD44',
    roi_id_col: str = 'roi_id',
) -> pd.DataFrame:
    """Per ROI: CD44+ rate within each compartment + triple-overlap fraction.

    Compartments are not exhaustive: superpixels can belong to >1 compartment.
    """
    df = annotations.copy()
    for m in list(compartment_markers) + [activation_marker]:
        df[f'{m}_pos'] = df[m] > sham_thresholds[m]

    rows = []
    for roi, group in df.groupby(roi_id_col):
        n_total = len(group)
        row: Dict[str, object] = {roi_id_col: roi, 'n_total': n_total}
        for m in compartment_markers:
            mask = group[f'{m}_pos']
            n_in = int(mask.sum())
            row[f'{m}_compartment_n'] = n_in
            row[f'{m}_compartment_cd44_rate'] = (
                float(group.loc[mask, f'{activation_marker}_pos'].mean()) if n_in > 0 else np.nan
            )
        bg_mask = ~group[[f'{m}_pos' for m in compartment_markers]].any(axis=1)
        n_bg = int(bg_mask.sum())
        row['background_compartment_n'] = n_bg
        row['background_compartment_cd44_rate'] = (
            float(group.loc[bg_mask, f'{activation_marker}_pos'].mean()) if n_bg > 0 else np.nan
        )
        triple_mask = group[[f'{m}_pos' for m in compartment_markers]].all(axis=1)
        row['triple_overlap_fraction'] = float(triple_mask.mean()) if n_total > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_compartment_activation_to_mouse(per_roi: pd.DataFrame) -> pd.DataFrame:
    """Mouse-level mean of per-ROI CD44+ rates per compartment.

    Empty compartments (rate=NaN) are skipped by .mean() — this is
    intentional but creates asymmetric weighting. Per-mouse counts of
    contributing ROIs are exposed via n_rois_with_<compartment> columns
    so reviewers can see how much each mouse's mean is supported by.
    """
    annotated = attach_roi_metadata(per_roi)
    rate_cols = [c for c in annotated.columns if c.endswith('_cd44_rate') or c == 'triple_overlap_fraction']
    count_cols = [c for c in annotated.columns if c.endswith('_compartment_n')]

    # Mean of rates (skips NaN); count of non-NaN ROIs per compartment
    grouped = annotated.groupby(['timepoint', 'mouse'], dropna=False)
    rate_means = grouped[rate_cols].mean().reset_index()
    n_rois_supporting = grouped[rate_cols].count().add_suffix('_n_rois').reset_index()
    if count_cols:
        compartment_n = grouped[count_cols].sum().reset_index()
        out = rate_means.merge(n_rois_supporting, on=['timepoint', 'mouse']).merge(
            compartment_n, on=['timepoint', 'mouse']
        )
    else:
        out = rate_means.merge(n_rois_supporting, on=['timepoint', 'mouse'])
    return out


# ---------------------------------------------------------------------------
# Spatial coherence — join-count + Moran's I on continuous lineage scores
# ---------------------------------------------------------------------------

def _knn_adjacency(coords: np.ndarray, k: int) -> np.ndarray:
    """k-NN adjacency; returns (n, k) integer index array of neighbors."""
    from scipy.spatial import cKDTree
    if len(coords) <= k:
        return np.empty((0, k), dtype=int)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    return idx[:, 1:]


def compute_join_count_bb(
    binary: np.ndarray,
    coords: np.ndarray,
    k: int = DEFAULT_K_NEIGHBORS,
    n_perm: int = DEFAULT_PERMUTATIONS,
    seed: int = 42,
    min_positive: int = 10,
) -> Dict[str, float]:
    """Black-Black join count for a binary indicator vs CSR null.

    Returns observed count, permutation null mean/std, z-score, n_positive,
    and sufficient flag. Saturated indicators (all-positive or all-negative)
    are flagged as not sufficient — the permutation null collapses to a
    point mass and the z-score is undefined.
    """
    n = len(binary)
    n_pos = int(binary.sum())
    saturated = n_pos == 0 or n_pos == n
    sufficient = (n_pos >= min_positive) and not saturated
    out = {
        'n_total': n, 'n_positive': n_pos, 'observed_bb': np.nan,
        'null_mean': np.nan, 'null_std': np.nan, 'z_score': np.nan,
        'sufficient': sufficient, 'saturated': saturated,
    }
    if not sufficient or n <= k:
        return out

    neighbors = _knn_adjacency(coords, k)
    if neighbors.size == 0:
        return out

    bin_arr = np.asarray(binary, dtype=bool)

    def _bb_count(b: np.ndarray) -> int:
        return int(np.sum(b[:, None] & b[neighbors]))

    observed = _bb_count(bin_arr)
    rng = np.random.RandomState(seed)
    null_counts = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = bin_arr.copy()
        rng.shuffle(shuffled)
        null_counts[i] = _bb_count(shuffled)

    null_mean = float(null_counts.mean())
    null_std = float(null_counts.std(ddof=1))
    z = (observed - null_mean) / null_std if null_std > 0 else np.nan
    out.update({
        'observed_bb': float(observed),
        'null_mean': null_mean, 'null_std': null_std, 'z_score': float(z),
    })
    return out


def compute_morans_i_continuous(
    values: np.ndarray,
    coords: np.ndarray,
    k: int = DEFAULT_K_NEIGHBORS,
) -> float:
    """Moran's I on continuous values using k-NN adjacency.

    Notes:
    - k-NN is asymmetric (j may be in nbr(i) without i in nbr(j)). Standard
      Moran's I assumes symmetric weights; we accept the asymmetry for spatial
      analysis at large n.
    - NaN values are dropped before computing mean/deviation; if fewer than
      k+1 finite values remain, returns NaN.
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    if finite.sum() <= k:
        return np.nan
    v_clean = v[finite]
    coords_clean = np.asarray(coords)[finite]
    neighbors = _knn_adjacency(coords_clean, k)
    if neighbors.size == 0:
        return np.nan
    v_mean = v_clean.mean()
    deviation = v_clean - v_mean
    denominator = float((deviation ** 2).sum())
    if denominator == 0:
        return np.nan
    # Row-normalized k-NN weights (w_ij = 1/k for j in nbr(i)); sum of all weights = n.
    # Standard Moran's I = (n / W) * cross / denominator; with W = n this reduces to cross / denominator.
    cross = float(np.sum(deviation[:, None] * deviation[neighbors] / k))
    return cross / denominator


# ---------------------------------------------------------------------------
# Statistics — Hedges' g, bootstrap range, BH-FDR, n_required
# ---------------------------------------------------------------------------

def cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = float(np.mean(g1)), float(np.mean(g2))
    s1, s2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))
    pooled_var = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return np.nan
    return (m2 - m1) / np.sqrt(pooled_var)


def hedges_g(g1: np.ndarray, g2: np.ndarray) -> float:
    d = cohens_d(g1, g2)
    if np.isnan(d):
        return np.nan
    df = len(g1) + len(g2) - 2
    if df <= 0:
        return np.nan
    correction = 1 - (3 / (4 * df - 1))
    return d * correction


@dataclass
class GDiagnostics:
    hedges_g: float
    pooled_std: float
    g_pathological: bool
    g_type_m_corrected: float


def compute_hedges_g_with_diagnostics(
    g1: np.ndarray,
    g2: np.ndarray,
    g_threshold: float = PATHOLOGY_G_THRESHOLD,
    std_threshold: float = PATHOLOGY_STD_THRESHOLD,
    type_m_factor: float = TYPE_M_CORRECTION,
) -> GDiagnostics:
    """Hedges' g + pathology flag + Type M-corrected estimate."""
    g = hedges_g(g1, g2)
    s1 = float(np.var(g1, ddof=1)) if len(g1) > 1 else 0.0
    s2 = float(np.var(g2, ddof=1)) if len(g2) > 1 else 0.0
    pooled = float(np.sqrt((s1 + s2) / 2.0))
    pathological = bool((not np.isnan(g)) and abs(g) > g_threshold and pooled < std_threshold)
    return GDiagnostics(
        hedges_g=g,
        pooled_std=pooled,
        g_pathological=pathological,
        g_type_m_corrected=g * type_m_factor if not np.isnan(g) else np.nan,
    )


def bootstrap_range_g(
    g1: np.ndarray,
    g2: np.ndarray,
    n_iter: int = DEFAULT_BOOTSTRAP_ITER,
    seed: int = 42,
) -> Dict[str, float]:
    """Percentile-bootstrap range. At n=2 per group, ~9 unique g values are possible.

    Reported as range_min, range_max, n_unique_values — NOT as a 95% CI.
    """
    if len(g1) < 2 or len(g2) < 2:
        return {'range_min': np.nan, 'range_max': np.nan, 'n_unique_values': 0}
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_iter):
        b1 = rng.choice(g1, size=len(g1), replace=True)
        b2 = rng.choice(g2, size=len(g2), replace=True)
        gv = hedges_g(b1, b2)
        if not np.isnan(gv):
            boot.append(gv)
    if not boot:
        return {'range_min': np.nan, 'range_max': np.nan, 'n_unique_values': 0}
    arr = np.asarray(boot)
    return {
        'range_min': float(arr.min()),
        'range_max': float(arr.max()),
        'n_unique_values': int(np.unique(np.round(arr, decimals=8)).size),
    }


def n_required_for_g(
    g: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Two-sample t-test sample size per group needed to detect effect size g.

    For small g, n_required grows quickly. Returns inf when g is non-positive
    or when computation fails. Uses two-sided test, equal group sizes.
    """
    if not np.isfinite(g) or abs(g) < 1e-6:
        return float('inf')
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / abs(g)) ** 2 * 2
    return float(np.ceil(n))


def n_required_type_m_adjusted(
    g: float,
    type_m_factor: float = TYPE_M_CORRECTION,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """n_required assuming the true effect is g * type_m_factor."""
    return n_required_for_g(g * type_m_factor, alpha=alpha, power=power)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """BH-FDR adjusted q-values; preserves input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    valid = ~np.isnan(p)
    if not valid.any():
        return p
    order = np.argsort(p[valid])
    ranked = np.arange(1, valid.sum() + 1)
    adj_sorted = p[valid][order] * valid.sum() / ranked
    # Enforce monotonicity
    for i in range(len(adj_sorted) - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj = np.full(n, np.nan)
    adj_valid = np.empty(valid.sum())
    adj_valid[order] = np.minimum(adj_sorted, 1.0)
    adj[valid] = adj_valid
    return adj


# ---------------------------------------------------------------------------
# Orchestration — endpoint summary table
# ---------------------------------------------------------------------------

def pairwise_endpoint_table(
    mouse_df: pd.DataFrame,
    value_cols: Sequence[str],
    family: str,
    contrasts: Sequence[Tuple[str, str]] = PAIRWISE_CONTRASTS,
    extra_index_cols: Sequence[str] = (),
    bootstrap_seed: int = 42,
    bootstrap_iter: int = DEFAULT_BOOTSTRAP_ITER,
) -> pd.DataFrame:
    """Compute Hedges' g + diagnostics + n_required per (endpoint x contrast).

    Contrasts where one or both groups have <2 observations emit a row with
    `insufficient_support=True` and NaN statistics — they are NOT silently
    dropped. This preserves the missingness signal in the output.

    `extra_index_cols` lets Family B group by composite_label as well as endpoint.
    `bootstrap_seed` is forwarded to bootstrap_range_g for reproducibility.
    """
    rows: List[Dict[str, object]] = []
    grouping_cols = list(extra_index_cols)
    if grouping_cols:
        groups = list(mouse_df.groupby(grouping_cols))
    else:
        groups = [(None, mouse_df)]
    for key, group in groups:
        for value_col in value_cols:
            if value_col not in group.columns:
                continue
            for tp1, tp2 in contrasts:
                v1 = group.loc[group['timepoint'] == tp1, value_col].dropna().values
                v2 = group.loc[group['timepoint'] == tp2, value_col].dropna().values
                insufficient = len(v1) < 2 or len(v2) < 2
                if insufficient:
                    row = {
                        'family': family,
                        'endpoint': value_col,
                        'contrast': f'{tp1}_vs_{tp2}',
                        'tp1': tp1, 'tp2': tp2,
                        'n_mice_1': len(v1), 'n_mice_2': len(v2),
                        'insufficient_support': True,
                        'mouse_mean_1': float(v1.mean()) if len(v1) else np.nan,
                        'mouse_mean_2': float(v2.mean()) if len(v2) else np.nan,
                        'mouse_range_1': float(v1.max() - v1.min()) if len(v1) else np.nan,
                        'mouse_range_2': float(v2.max() - v2.min()) if len(v2) else np.nan,
                        'hedges_g': np.nan,
                        'g_type_m_corrected': np.nan,
                        'pooled_std': np.nan,
                        'g_pathological': False,
                        'bootstrap_range_min': np.nan,
                        'bootstrap_range_max': np.nan,
                        'n_unique_resamples': 0,
                        'n_required_80pct': np.nan,
                        'n_required_80pct_type_m': np.nan,
                    }
                else:
                    diag = compute_hedges_g_with_diagnostics(v1, v2)
                    br = bootstrap_range_g(v1, v2, n_iter=bootstrap_iter, seed=bootstrap_seed)
                    row = {
                        'family': family,
                        'endpoint': value_col,
                        'contrast': f'{tp1}_vs_{tp2}',
                        'tp1': tp1, 'tp2': tp2,
                        'n_mice_1': len(v1), 'n_mice_2': len(v2),
                        'insufficient_support': False,
                        'mouse_mean_1': float(v1.mean()),
                        'mouse_mean_2': float(v2.mean()),
                        'mouse_range_1': float(v1.max() - v1.min()),
                        'mouse_range_2': float(v2.max() - v2.min()),
                        'hedges_g': diag.hedges_g,
                        'g_type_m_corrected': diag.g_type_m_corrected,
                        'pooled_std': diag.pooled_std,
                        'g_pathological': diag.g_pathological,
                        'bootstrap_range_min': br['range_min'],
                        'bootstrap_range_max': br['range_max'],
                        'n_unique_resamples': br['n_unique_values'],
                        'n_required_80pct': n_required_for_g(diag.hedges_g),
                        'n_required_80pct_type_m': n_required_type_m_adjusted(diag.hedges_g),
                    }
                if grouping_cols:
                    if isinstance(key, tuple):
                        for col, val in zip(grouping_cols, key):
                            row[col] = val
                    else:
                        row[grouping_cols[0]] = key
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # BH within family + threshold-sensitive flag will be added by orchestrator
    return df


def add_pooled_fdr_proxy(endpoint_df: pd.DataFrame) -> pd.DataFrame:
    """Add q_proxy_pooled column applying BH-FDR mechanics across all rows.

    AT n=2 PER GROUP, NO REAL p-VALUE EXISTS. This function derives a sortable
    p-value proxy from |hedges_g| via the standard normal CDF and applies BH
    mechanics. The output column is named `q_proxy_pooled` (not q_value_pooled)
    to make explicit that this is a researcher-degree-of-freedom audit
    quantity, not a coverage-bearing q-value. It exists solely to compare
    rank-orderings under within-family vs pooled FDR — to detect whether the
    family partition acts as multiplicity arbitrage.

    Reviewers should not interpret q_proxy values as FDR-controlled discoveries.
    """
    if endpoint_df.empty:
        return endpoint_df
    z = endpoint_df['hedges_g'].abs().fillna(0).values
    approx_p = 2 * (1 - stats.norm.cdf(z))
    endpoint_df = endpoint_df.copy()
    endpoint_df['p_proxy_from_g'] = approx_p
    endpoint_df['q_proxy_pooled'] = benjamini_hochberg(approx_p)
    return endpoint_df
