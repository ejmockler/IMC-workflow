"""
Shared Sham-reference normalization primitive.

Before this module: three separate Sham-threshold paths drift silently when
CD206/Ly6G overrides change in config.json — compute_global_marker_thresholds
and compute_sham_reference_thresholds in temporal_interface_analysis.py, and
an implicit per-ROI sigmoid path in compute_continuous_memberships.

After this module: one primitive (``sham_reference_thresholds``) with explicit
aggregation mode (per-mouse vs pooled superpixels) and per-marker override
support. All call sites converge here.

``per_mouse`` is the statistically defensible default under this study's
design (n=2 Sham mice × 3 ROIs × ~1000 superpixels): pooling ~6000
correlated superpixels understates uncertainty and weights ROIs by size
rather than biological replicate. The ``pool`` aggregation mode is retained
for legacy comparison, not as a recommended path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


_SHAM_LABEL_DEFAULT = 'Sham'
_MIN_SHAM_MICE = 2


def _filter_sham(
    annotations: pd.DataFrame,
    timepoint_col: str,
    sham_label: str,
) -> pd.DataFrame:
    if timepoint_col not in annotations.columns:
        raise ValueError(
            f"sham_reference: annotations missing '{timepoint_col}' column; "
            f"required to identify Sham superpixels"
        )
    sham = annotations[annotations[timepoint_col] == sham_label]
    if sham.empty:
        raise ValueError(
            f"sham_reference: no superpixels with "
            f"{timepoint_col}=='{sham_label}'"
        )
    return sham


def _effective_percentile(
    marker: str,
    default_percentile: float,
    per_marker_overrides: Optional[Mapping[str, Mapping[str, Any]]],
) -> float:
    if not per_marker_overrides or marker not in per_marker_overrides:
        return default_percentile
    override = per_marker_overrides[marker]
    method = override.get('method', 'percentile')
    if method != 'percentile':
        raise ValueError(
            f"sham_reference: marker '{marker}' uses method='{method}' — "
            f"only method='percentile' is supported for Sham-reference "
            f"thresholds (absolute thresholds are a discrete-gating concern)"
        )
    return float(override.get('percentile', default_percentile))


def _pct_pool(values: np.ndarray, percentile: float, marker: str) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(
            f"sham_reference: marker '{marker}' has no finite Sham values"
        )
    return float(np.percentile(finite, percentile))


def _pct_per_mouse(
    sham: pd.DataFrame,
    marker: str,
    percentile: float,
    mouse_col: str,
) -> float:
    if mouse_col not in sham.columns:
        raise ValueError(
            f"sham_reference: aggregation='per_mouse' requires "
            f"'{mouse_col}' column in annotations"
        )
    per_mouse: List[float] = []
    for mouse_id, group in sham.groupby(mouse_col):
        vals = group[marker].values
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            raise ValueError(
                f"sham_reference: mouse '{mouse_id}' has no finite "
                f"Sham values for marker '{marker}'"
            )
        per_mouse.append(float(np.percentile(finite, percentile)))
    if len(per_mouse) < _MIN_SHAM_MICE:
        raise ValueError(
            f"sham_reference: only {len(per_mouse)} Sham mouse(s) found "
            f"for marker '{marker}'; require >= {_MIN_SHAM_MICE} for "
            f"per-mouse aggregation (biological replication unit)"
        )
    return float(np.mean(per_mouse))


def sham_reference_thresholds(
    annotations: pd.DataFrame,
    markers: Sequence[str],
    percentile: float,
    per_marker_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    aggregation: Literal['per_mouse', 'pool'] = 'per_mouse',
    timepoint_col: str = 'timepoint',
    mouse_col: str = 'mouse',
    sham_label: str = _SHAM_LABEL_DEFAULT,
) -> Dict[str, float]:
    """Per-marker Sham-reference threshold, with config-aware overrides.

    Args:
        annotations: per-superpixel DataFrame containing each marker column,
            ``timepoint_col``, and (for ``per_mouse``) ``mouse_col``.
        markers: marker columns to compute thresholds for. Derived columns
            (e.g., ``(CD31 + CD34) / 2``) should be pre-computed on the
            DataFrame before calling — this preserves
            ``percentile(mean(a, b))`` semantics, which is not the same as
            ``mean(percentile(a), percentile(b))``.
        percentile: default percentile (e.g., 60.0 for cell-type gating,
            75.0 for Family A/C sensitivity sweeps).
        per_marker_overrides: optional mapping mirroring
            ``config.positivity_threshold.per_marker_override``. Honored
            only when ``method='percentile'`` (absolute overrides are a
            discrete-gating concern and raise).
        aggregation: ``'per_mouse'`` (default) computes the percentile per
            Sham mouse then averages — respects n=2 biological replicates,
            does not pseudo-replicate superpixels. ``'pool'`` concatenates
            all Sham superpixels (legacy; retained for comparison).
        timepoint_col, mouse_col, sham_label: metadata column names / value.

    Returns:
        ``{marker: threshold}`` dict.

    Raises:
        ValueError: empty Sham pool; missing columns; all-NaN marker;
            < 2 Sham mice (``per_mouse``); unsupported override method.
    """
    sham = _filter_sham(annotations, timepoint_col, sham_label)
    out: Dict[str, float] = {}
    for marker in markers:
        if marker not in sham.columns:
            raise ValueError(
                f"sham_reference: marker '{marker}' missing from annotations "
                f"(have {len(sham.columns)} columns; first 10: "
                f"{list(sham.columns)[:10]})"
            )
        pct = _effective_percentile(marker, percentile, per_marker_overrides)
        if aggregation == 'per_mouse':
            out[marker] = _pct_per_mouse(sham, marker, pct, mouse_col)
        elif aggregation == 'pool':
            out[marker] = _pct_pool(sham[marker].values, pct, marker)
        else:
            raise ValueError(
                f"sham_reference: unknown aggregation '{aggregation}'; "
                f"expected 'per_mouse' or 'pool'"
            )
    return out


def experiment_wide_percentile(
    annotations: pd.DataFrame,
    markers: Sequence[str],
    percentile: float,
) -> Dict[str, float]:
    """Per-marker percentile pooled across ALL rows (all timepoints).

    Companion to ``sham_reference_thresholds`` for call sites that want a
    non-Sham-filtered threshold (e.g., ``compute_global_marker_thresholds``
    with ``sham_only=False``, a diagnostic comparison path).
    """
    out: Dict[str, float] = {}
    for marker in markers:
        if marker not in annotations.columns:
            raise ValueError(
                f"experiment_wide_percentile: marker '{marker}' missing"
            )
        out[marker] = _pct_pool(annotations[marker].values, percentile, marker)
    return out


def experiment_wide_iqr(
    annotations: pd.DataFrame,
    markers: Sequence[str],
    floor: float = 1e-6,
) -> Dict[str, float]:
    """Per-marker IQR across ALL rows (all timepoints).

    Used as the sigmoid-steepness denominator in
    ``compute_continuous_memberships``. Pooling across all timepoints avoids
    the Sham-only-IQR pathology (if Sham runs hot or flat, the sigmoid
    degenerates): temporal signal is captured in the threshold *center*
    position via ``sham_reference_thresholds``; scale just needs to reflect
    the full experimental dynamic range so injury-driven departures aren't
    clipped.

    For zero-variance markers (IQR < ``floor``), falls back to
    ``(max - min) / 4.0``. For fully degenerate markers (all equal), uses
    ``floor`` — produces sigmoid scores near 0.5 everywhere, which is the
    correct no-information behavior.
    """
    out: Dict[str, float] = {}
    for marker in markers:
        if marker not in annotations.columns:
            raise ValueError(
                f"experiment_wide_iqr: marker '{marker}' missing"
            )
        vals = annotations[marker].values
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            raise ValueError(
                f"experiment_wide_iqr: marker '{marker}' has no finite values"
            )
        iqr = float(np.percentile(finite, 75) - np.percentile(finite, 25))
        if iqr < floor:
            iqr = float((finite.max() - finite.min()) / 4.0)
            if iqr < floor:
                iqr = floor
        out[marker] = iqr
    return out


def build_reference_distribution(
    annotations: pd.DataFrame,
    markers: Sequence[str],
    percentile: float,
    per_marker_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    aggregation: Literal['per_mouse', 'pool'] = 'per_mouse',
    timepoint_col: str = 'timepoint',
    mouse_col: str = 'mouse',
) -> Dict[str, Dict[str, float]]:
    """Convenience: returns the ``{marker: {threshold, scale}}`` dict
    consumed by ``compute_continuous_memberships(reference_distribution=...)``.

    Threshold uses ``sham_reference_thresholds`` (Sham-only, per-mouse);
    scale uses ``experiment_wide_iqr`` (all timepoints pooled).
    Persisting this artifact is the caller's responsibility — see
    ``generate_sham_reference.py``.
    """
    thresholds = sham_reference_thresholds(
        annotations, markers, percentile,
        per_marker_overrides=per_marker_overrides,
        aggregation=aggregation,
        timepoint_col=timepoint_col,
        mouse_col=mouse_col,
    )
    scales = experiment_wide_iqr(annotations, markers)
    return {m: {'threshold': thresholds[m], 'scale': scales[m]} for m in markers}
