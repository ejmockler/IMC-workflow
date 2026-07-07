"""Descriptive: CD44+ compartment activation rates by mouse x timepoint x region.

Standalone, DESCRIPTIVE analysis (NOT a new pre-registered endpoint). Computes
the CD44+ positivity rate within a set of marker/cell-type compartments, per
mouse x timepoint x region {cortex, medulla, pooled}, at 10um and 40um.

It REUSES the frozen pipeline's compute functions and threshold recipe:
  - ``compute_sham_reference_thresholds`` (Sham-only, per-mouse, 75th pct)
  - ``compute_compartment_activation_per_roi`` (CD44+ rate within {marker}+)
  - ``compute_neutrophil_compartment_activation_per_roi`` (cell_type=='neutrophil')
  - ``aggregate_compartment_activation_to_mouse`` (per-ROI -> mouse level)
all imported from ``src.analysis.temporal_interface_analysis``. NO CD44-rate or
positivity math is re-implemented here.

The four existing Family C thresholds (CD45/CD31/CD140b/CD44) are reproduced
exactly by the same recipe; CD206 and CD34 thresholds are computed by the
IDENTICAL recipe (same percentile, same Sham per-mouse pooling). The pooled
region reproduces the pipeline's region-agnostic compartment rates bit-for-bit
(this is the cross-check test's anchor).

Outputs (new files only):
  results/biological_analysis/collab_cd44/cd44_compartment_rates_10um.csv
  results/biological_analysis/collab_cd44/cd44_compartment_rates_40um.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analysis import temporal_interface_analysis as tia
from src.utils.canonical_loader import build_superpixel_dataframe, load_all_rois

REPO_ROOT = Path(__file__).resolve().parent
ROI_RESULTS_DIR = REPO_ROOT / 'results' / 'roi_results'
ABUNDANCES_CSV = (
    REPO_ROOT / 'results' / 'biological_analysis' /
    'differential_abundance' / 'roi_abundances.csv'
)
OUTPUT_DIR = REPO_ROOT / 'results' / 'biological_analysis' / 'collab_cd44'

# Identical threshold recipe across all six markers and both scales.
COMPARTMENT_MARKERS: Tuple[str, ...] = ('CD45', 'CD206', 'CD31', 'CD34', 'CD140b')
ACTIVATION_MARKER = 'CD44'
THRESHOLD_MARKERS: List[str] = list(COMPARTMENT_MARKERS) + [ACTIVATION_MARKER]
SHAM_PERCENTILE = tia.DEFAULT_SHAM_PERCENTILE  # 75.0
ENDOTHELIAL_COL = 'endothelial_cd31cd34'  # CD31+ AND CD34+ derived compartment

# (tidy compartment name, marker fed to compute_compartment_activation_per_roi)
MARKER_COMPARTMENTS: Tuple[str, ...] = COMPARTMENT_MARKERS + (ENDOTHELIAL_COL,)
NEUTROPHIL_COMPARTMENT = 'neutrophil'

TIDY_COLUMNS = ['compartment', 'timepoint', 'mouse', 'region', 'cd44_rate', 'n_support']

# region label in roi_abundances.csv -> tidy output value
REGION_LABELS: Tuple[Tuple[str, Optional[str]], ...] = (
    ('cortex', 'Cortex'),
    ('medulla', 'Medulla'),
    ('pooled', None),  # None => all rows (region-agnostic)
)


def load_region_map() -> Dict[str, str]:
    """roi_id (unprefixed, loader-style) -> region ('Cortex'/'Medulla').

    roi_abundances.csv carries a ``roi_``-prefixed roi_id; the canonical loader
    emits the unprefixed form (e.g. ``IMC_241218_Alun_ROI_D1_M1_01_9``). Strip
    the prefix so the join key matches the loader.
    """
    abund = pd.read_csv(ABUNDANCES_CSV)
    abund = abund.copy()
    abund['roi_key'] = abund['roi_id'].str.replace(r'^roi_', '', regex=True)
    return dict(zip(abund['roi_key'], abund['region']))


def _add_endothelial_column(df: pd.DataFrame, thr: Dict[str, float]) -> pd.DataFrame:
    """Derived positivity column: CD31+ AND CD34+ (float 0/1), threshold 0.5.

    compute_compartment_activation_per_roi does ``df[m] > sham_thresholds[m]``;
    with values in {0.0, 1.0} and threshold 0.5 this recovers the AND mask.
    """
    df = df.copy()
    df[ENDOTHELIAL_COL] = (
        (df['CD31'] > thr['CD31']) & (df['CD34'] > thr['CD34'])
    ).astype(float)
    return df


def compute_per_roi(
    df: pd.DataFrame,
    include_neutrophil: bool,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Per-ROI compartment CD44+ rates for one scale.

    Reuses the frozen compute for every marker compartment and (when
    cell-type annotations exist) the neutrophil compartment. Returns
    (per_roi, thresholds).
    """
    thr = tia.compute_sham_reference_thresholds(
        df, THRESHOLD_MARKERS, percentile=SHAM_PERCENTILE,
    )
    thr[ENDOTHELIAL_COL] = 0.5  # paired threshold for the derived AND column

    df_ext = _add_endothelial_column(df, thr)
    per_roi = tia.compute_compartment_activation_per_roi(
        df_ext,
        sham_thresholds=thr,
        compartment_markers=MARKER_COMPARTMENTS,
        activation_marker=ACTIVATION_MARKER,
    )

    if include_neutrophil:
        neutro = tia.compute_neutrophil_compartment_activation_per_roi(
            df, sham_thresholds=thr, activation_marker=ACTIVATION_MARKER,
        )
        per_roi = per_roi.merge(
            neutro[['roi_id', 'neutrophil_compartment_n',
                    'neutrophil_compartment_cd44_rate']],
            on='roi_id', how='left',
        )
    return per_roi, thr


def _compartment_columns(has_neutrophil: bool) -> List[Tuple[str, str, str]]:
    """(tidy compartment name, rate column, support column)."""
    cols = [
        (name, f'{name}_compartment_cd44_rate', f'{name}_compartment_n')
        for name in MARKER_COMPARTMENTS
    ]
    if has_neutrophil:
        cols.append((
            NEUTROPHIL_COMPARTMENT,
            'neutrophil_compartment_cd44_rate',
            'neutrophil_compartment_n',
        ))
    return cols


def _tidy_from_mouse(
    mouse: pd.DataFrame,
    region_value: str,
    compartments: Sequence[Tuple[str, str, str]],
) -> pd.DataFrame:
    """Melt the wide mouse-level table into tidy long rows.

    Empty/thin compartments (cd44_rate NaN from the frozen compute) are kept
    with their n_support so the thinning stays visible; compartments whose
    columns are absent for this scale (e.g. neutrophil at 40um) are skipped
    rather than fabricated.
    """
    rows: List[Dict[str, object]] = []
    for name, rate_col, n_col in compartments:
        if rate_col not in mouse.columns:
            continue  # compartment has no basis at this scale — skip, don't fake
        for _, r in mouse.iterrows():
            rate = r[rate_col]
            n_support = int(r[n_col]) if (n_col in mouse.columns and pd.notna(r[n_col])) else 0
            rows.append({
                'compartment': name,
                'timepoint': r['timepoint'],
                'mouse': r['mouse'],
                'region': region_value,
                'cd44_rate': float(rate) if pd.notna(rate) else np.nan,
                'n_support': n_support,
            })
    return pd.DataFrame(rows, columns=TIDY_COLUMNS)


def build_tidy_table(
    per_roi: pd.DataFrame,
    region_map: Dict[str, str],
    has_neutrophil: bool,
) -> pd.DataFrame:
    """Aggregate per-ROI -> mouse x timepoint per region and emit tidy rows."""
    per_roi = per_roi.copy()
    per_roi['region'] = per_roi['roi_id'].map(region_map)
    unjoined = per_roi[per_roi['region'].isna()]['roi_id'].tolist()
    if unjoined:
        raise ValueError(
            f"region join incomplete — {len(unjoined)} ROI(s) missing a region: "
            f"{unjoined}"
        )

    compartments = _compartment_columns(has_neutrophil)
    frames: List[pd.DataFrame] = []
    for region_value, region_filter in REGION_LABELS:
        subset = per_roi if region_filter is None else per_roi[per_roi['region'] == region_filter]
        if subset.empty:
            continue
        # aggregate_compartment_activation_to_mouse re-derives timepoint/mouse
        # via attach_roi_metadata and is region-agnostic; feeding it a region
        # subset yields that region's mouse-level means.
        mouse = tia.aggregate_compartment_activation_to_mouse(
            subset.drop(columns=['region'])
        )
        frames.append(_tidy_from_mouse(mouse, region_value, compartments))
    return pd.concat(frames, ignore_index=True)


def run_10um(region_map: Dict[str, str]) -> pd.DataFrame:
    from run_temporal_interface_analysis import load_annotations_with_markers
    ann = load_annotations_with_markers(scale_um=10.0)
    has_neutrophil = 'cell_type' in ann.columns
    per_roi, thr = compute_per_roi(ann, include_neutrophil=has_neutrophil)
    tidy = build_tidy_table(per_roi, region_map, has_neutrophil=has_neutrophil)
    print(f"[10um] {len(ann)} superpixels, thresholds: "
          + ", ".join(f"{m}={thr[m]:.6f}" for m in THRESHOLD_MARKERS))
    return tidy


def run_grid_scale(scale_um: float, region_map: Dict[str, str]) -> pd.DataFrame:
    """Grid-superpixel scales (20um tubular, 40um domain): marker compartments only.

    These coarser scales average expression over multiple cells and have NO
    cell-type annotation basis, so the neutrophil (cell_type) compartment is
    skipped rather than fabricated. Loaded from the persisted multiscale
    superpixel matrices in roi_results; thresholds recomputed by the identical
    Sham per-mouse 75th-pct recipe on each scale's own Sham superpixels.
    """
    results = load_all_rois(str(ROI_RESULTS_DIR))
    df = build_superpixel_dataframe(results, scale=scale_um).rename(columns={'roi': 'roi_id'})
    has_neutrophil = 'cell_type' in df.columns  # False at grid scales
    per_roi, thr = compute_per_roi(df, include_neutrophil=has_neutrophil)
    tidy = build_tidy_table(per_roi, region_map, has_neutrophil=has_neutrophil)
    print(f"[{scale_um:g}um] {len(df)} superpixels (vs ~58,137 at 10um — thinning "
          f"visible in n_support); thresholds: "
          + ", ".join(f"{m}={thr[m]:.6f}" for m in THRESHOLD_MARKERS))
    if not has_neutrophil:
        print(f"[{scale_um:g}um] neutrophil (cell_type) compartment skipped — no "
              "cell-type annotation basis at grid scales (not fabricated)")
    return tidy


def compute_crossscale_summary(allscales: pd.DataFrame) -> pd.DataFrame:
    """The cross-scale READING, per compartment x region.

    For each compartment x region, the mouse-mean CD44+ rate at Sham and D7 and
    the Sham->D7 delta at each scale, plus a plain-language scale reading:
      - ``scale_robust``   : present at all scales AND the Sham->D7 direction
                             agrees across 10/20/40um (the finding holds as the
                             unit coarsens from ~1 cell to a tissue domain);
      - ``scale_dependent``: present at all scales but the direction flips with
                             scale (an artifact of the unit, not a robust shift);
      - ``thins_out``      : at least one scale has no supporting superpixels
                             (the compartment collapses as patches enlarge).
    This is the actual cross-scale depth: whether the descriptive CD44 story
    survives the 10um->40um coarsening, not just whether a 40um number exists.
    """
    scales = sorted(allscales['scale_um'].unique())
    rows: List[Dict[str, object]] = []
    for (comp, region), g in allscales.groupby(['compartment', 'region']):
        rec: Dict[str, object] = {'compartment': comp, 'region': region}
        deltas: Dict[float, float] = {}
        for s in scales:
            gs = g[g['scale_um'] == s]
            sham = gs[gs['timepoint'] == 'Sham']['cd44_rate'].mean()
            d7 = gs[gs['timepoint'] == 'D7']['cd44_rate'].mean()
            delta = d7 - sham
            rec[f'sham_{s:g}um'] = sham
            rec[f'd7_{s:g}um'] = d7
            rec[f'delta_{s:g}um'] = delta
            rec[f'n_sham_{s:g}um'] = gs[gs['timepoint'] == 'Sham']['n_support'].mean()
            deltas[s] = delta
        finite = {s: d for s, d in deltas.items() if pd.notna(d)}
        signs = {int(np.sign(d)) for d in finite.values() if d != 0}
        rec['n_scales_present'] = len(finite)
        rec['direction_consistent'] = bool(len(signs) <= 1 and len(finite) == len(scales))
        if len(finite) < len(scales):
            rec['scale_reading'] = 'thins_out'
        elif rec['direction_consistent']:
            rec['scale_reading'] = 'scale_robust'
        else:
            rec['scale_reading'] = 'scale_dependent'
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    region_map = load_region_map()

    # 10um (cellular): the only scale with a cell-type / neutrophil basis.
    per_scale: Dict[float, pd.DataFrame] = {}
    per_scale[10.0] = run_10um(region_map)
    # 20um (tubular cross-section) + 40um (tissue domain): grid scales, marker
    # compartments only.
    per_scale[20.0] = run_grid_scale(20.0, region_map)
    per_scale[40.0] = run_grid_scale(40.0, region_map)

    for scale_um, tidy in per_scale.items():
        out = OUTPUT_DIR / f'cd44_compartment_rates_{scale_um:g}um.csv'
        tidy.to_csv(out, index=False)
        print(f"  wrote {out} ({len(tidy)} rows, "
              f"{tidy['compartment'].nunique()} compartments)")

    # Cross-scale: one long table (adds a scale_um column) + the cross-scale reading.
    allscales = pd.concat(
        [t.assign(scale_um=s) for s, t in per_scale.items()], ignore_index=True
    )
    out_all = OUTPUT_DIR / 'cd44_compartment_rates_allscales.csv'
    allscales.to_csv(out_all, index=False)
    print(f"  wrote {out_all} ({len(allscales)} rows across "
          f"{allscales['scale_um'].nunique()} scales)")

    summary = compute_crossscale_summary(allscales)
    out_sum = OUTPUT_DIR / 'cd44_crossscale_summary.csv'
    summary.to_csv(out_sum, index=False)
    reading = summary['scale_reading'].value_counts().to_dict()
    print(f"  wrote {out_sum} ({len(summary)} compartment x region rows; "
          f"scale_reading: {reading})")


if __name__ == '__main__':
    main()
