"""
Tissue-mask non-compositional density: empirical-closure audit (Phase 5.1).

Closure scope: closes the **area-based** density column on the present
acquisition design. Does NOT close per-nucleus or DNA-intensity-integral
alternatives, which remain untested.

Originally proposed (FROZEN_PREREG, 2026-04-23) as a Phase 1.5 follow-up.
The earlier retracted version computed ``density = count / (n_total / 2500)``
which reduced algebraically to ``2500 × proportion``. This audit tests
whether the *measured* eroded DNA-mask tissue area (``np.sum(superpixel_labels >= 0) × resolution_um²``
from the persisted per-ROI artifacts) escapes the tautology.

**Lead finding (algebraic, not statistical)**: every ROI in this pilot is
acquired at the same ~500×500 µm field-of-view, so ``tissue_area_mm2`` is
dataset-constant. Therefore ``density = count / tissue_area_mm2 ≈ const ×
proportion``, regardless of which primitive computes the area. The CV bar
below is supporting diagnostic; the algebraic reduction is what closes the
door for area-based denominators on this acquisition.

**Pre-registered non-degeneracy gates** (this audit):
- CV(tissue_area_mm2) > 0.05 across ROIs (necessary, not sufficient).
- Pearson |r| < 0.95 between ``superpixels_per_mm2`` and a constant
  ``superpixels`` proxy across ROIs (sanity check on the algebraic
  reduction).

**Untested alternatives** (NOT closed by this audit, flagged for follow-up):
- Per-nucleus density via watershed segmentation (``src/analysis/watershed_segmentation.py``
  exists but is not wired to production); denominator becomes ``n_nuclei``
  rather than ``tissue_area_mm2``.
- DNA-intensity integral over the eroded mask as denominator
  (cellularity-weighted volume).
- Variable-extent re-acquisition on a follow-up cohort (whole-section IMC,
  panoramic montage) restores meaningful CV(tissue_area_mm2).

Output: ``results/biological_analysis/tissue_area_audit.csv`` — one row per
ROI with ``tissue_area_mm2``, ``n_superpixels``, ``superpixels_per_mm2``,
plus the algebraic reduction, the gate verdict, and the scope language at
the bottom.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
ROI_RESULTS_DIR = REPO_ROOT / 'results' / 'roi_results'
OUT_PATH = REPO_ROOT / 'results' / 'biological_analysis' / 'tissue_area_audit.csv'

SCALE_UM_KEY = '10.0'  # primary scale (matches the rest of the pilot)
RESOLUTION_UM = 1.0    # native IMC resolution; tissue_area_um2 = n_pixels × this²


def parse_timepoint(roi_id: str) -> tuple[str, str]:
    """Return (timepoint, mouse_id). Sham is encoded as 'Sam1'/'Sam2'."""
    m = re.search(r'ROI_(Sam\d|D\d)_?M?(\d)?', roi_id)
    if not m:
        return ('NA', 'NA')
    tp_raw = m.group(1)
    mouse_digit = m.group(2) or 'NA'
    timepoint = 'Sham' if tp_raw.startswith('Sam') else tp_raw
    if tp_raw.startswith('Sam'):
        # 'Sam1' / 'Sam2' encodes mouse in the first token, no separate M digit
        mouse = f'M{tp_raw[-1]}'
    else:
        mouse = f'M{mouse_digit}'
    return (timepoint, mouse)


CELL_TYPE_SUMMARY_PATH = (
    REPO_ROOT / 'results' / 'biological_analysis'
    / 'cell_type_annotations' / 'batch_annotation_summary.json'
)


def _compute_pearson_density_vs_proportion(area_df: pd.DataFrame) -> float:
    """Pick the cell type with the highest median count across ROIs and
    return Pearson |r| between density (count/tissue_area_mm2) and
    proportion (count/n_superpixels) across the ROIs in ``area_df``.

    Returns NaN if the cell-type summary is missing or no overlap with the
    ROIs in ``area_df``."""
    if not CELL_TYPE_SUMMARY_PATH.exists():
        return float('nan')
    import json
    with CELL_TYPE_SUMMARY_PATH.open() as fh:
        summary = json.load(fh)
    roi_summaries = summary.get('roi_summaries', {})
    rows = []
    for roi_id, payload in roi_summaries.items():
        counts = payload.get('cell_type_counts', {})
        n_total = payload.get('n_superpixels', 0)
        # Match against area_df by stripping 'roi_' prefix if present.
        area_match = area_df[area_df['roi_id'] == roi_id]
        if area_match.empty:
            roi_short = roi_id.replace('roi_', '')
            area_match = area_df[area_df['roi_id'] == roi_short]
        if area_match.empty:
            continue
        tissue_mm2 = float(area_match.iloc[0]['tissue_area_mm2'])
        for cell_type, count in counts.items():
            rows.append({
                'roi_id': roi_id,
                'cell_type': cell_type,
                'count': count,
                'n_total': n_total,
                'tissue_area_mm2': tissue_mm2,
            })
    if not rows:
        return float('nan')
    long = pd.DataFrame(rows)
    long['proportion'] = long['count'] / long['n_total'].where(long['n_total'] > 0, 1)
    long['density_per_mm2'] = long['count'] / long['tissue_area_mm2']
    # Pick the cell type with the highest median count (most measurement
    # signal) — small-N cell types have noisy correlations.
    medians = long.groupby('cell_type')['count'].median().sort_values(ascending=False)
    if medians.empty:
        return float('nan')
    chosen = medians.index[0]
    sub = long[long['cell_type'] == chosen]
    if len(sub) < 3:
        return float('nan')
    return float(np.corrcoef(sub['density_per_mm2'], sub['proportion'])[0, 1])


def main() -> int:
    print('=' * 72)
    print('Tissue-mask non-compositional density: closure audit')
    print('=' * 72)

    files = sorted(ROI_RESULTS_DIR.glob('roi_IMC_*.json.gz'))
    if not files:
        raise SystemExit(f'no per-ROI artifacts found under {ROI_RESULTS_DIR}')

    records = []
    for path in files:
        with gzip.open(path) as fh:
            payload = json.load(fh)
        scale_payload = payload.get('multiscale_results', {}).get(SCALE_UM_KEY)
        if not scale_payload:
            continue
        sl_dict = scale_payload.get('superpixel_labels', {})
        if not sl_dict or 'data' not in sl_dict:
            continue
        arr = np.array(sl_dict['data']).reshape(sl_dict['shape'])
        n_tissue_px = int((arr >= 0).sum())
        unique_sp = np.unique(arr[arr >= 0])
        n_superpixels = int(unique_sp.size)
        roi_id = payload.get('roi_id', path.stem)
        timepoint, mouse = parse_timepoint(roi_id)
        tissue_area_mm2 = n_tissue_px * (RESOLUTION_UM ** 2) / 1e6
        records.append({
            'roi_id': roi_id,
            'timepoint': timepoint,
            'mouse': mouse,
            'n_tissue_px': n_tissue_px,
            'n_superpixels': n_superpixels,
            'tissue_area_mm2': tissue_area_mm2,
            'superpixels_per_mm2': (
                n_superpixels / tissue_area_mm2 if tissue_area_mm2 > 0 else np.nan
            ),
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit('no scale-10.0 superpixel_labels found in any ROI artifact')

    cv_area = df['tissue_area_mm2'].std() / df['tissue_area_mm2'].mean()
    cv_ratio = (
        df['superpixels_per_mm2'].std() / df['superpixels_per_mm2'].mean()
    )

    # Pearson |r| between density and proportion **for a real cell-type
    # column**. The brutalist non-degeneracy gate asks whether a candidate
    # density column would carry information independent of the
    # corresponding proportion column. By the algebraic identity
    # density_K = (n_total / tissue_area_mm2) × proportion_K, both columns
    # are linear in proportion_K modulo the (CV ~0.014) multiplier, so
    # |r| ≈ 1 by construction for any cell type with non-trivial counts.
    # Compute against an actual per-ROI cell-type column from the
    # batch-annotation summary so the gate is empirical, not algebraic-only.
    pearson_r = _compute_pearson_density_vs_proportion(df)

    NON_DEGENERACY_BAR_CV = 0.05      # CV(tissue_area_mm2) must exceed this
    NON_DEGENERACY_BAR_R = 0.95       # Pearson |r| with proportion-proxy must fall below this
    gate_cv_passed = cv_area > NON_DEGENERACY_BAR_CV
    gate_r_passed = abs(pearson_r) < NON_DEGENERACY_BAR_R
    gate_passed = gate_cv_passed and gate_r_passed

    # Algebraic-reduction headline (lead finding, not the CV).
    multiplier = float(df['superpixels_per_mm2'].mean())

    print(f'\n  ROIs analyzed: {len(df)}')
    print(f'  Scale: {SCALE_UM_KEY} µm')
    print(f'  Resolution: {RESOLUTION_UM} µm/pixel')
    print('\n  Algebraic reduction (lead finding):')
    print(f'    density = count / tissue_area_mm2 ≈ {multiplier:.0f} × proportion '
          f'± {cv_ratio*100:.1f}% (CV of multiplier)')
    print('\n  tissue_area_mm2 distribution (all ROIs):')
    print(df['tissue_area_mm2'].describe().to_string())
    print(f'\n  CV(tissue_area_mm2) = {cv_area:.4f}  '
          f'[gate: > {NON_DEGENERACY_BAR_CV} ⇒ {"PASS" if gate_cv_passed else "FAIL"}]')
    print(f'  Pearson |r|(density_per_mm2, proportion) on dominant cell type = '
          f'{abs(pearson_r):.4f}  '
          f'[gate: < {NON_DEGENERACY_BAR_R} ⇒ {"PASS" if gate_r_passed else "FAIL"}]')
    print(f'\n  Non-degeneracy verdict: {"PASS (density would be independent)" if gate_passed else "FAIL (density is rescaled proportion)"}')
    if not gate_passed:
        print('  ↳ Closure scope: area-based density on this acquisition design.')
        print('  ↳ NOT closed: per-nucleus density (watershed), DNA-intensity '
              'integral, or variable-extent re-acquisition cohorts.')

    # Stamp the verdicts + scope at the end of the CSV so the artifact carries
    # its own conclusion (not just raw numbers).
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open('w') as fh:
        df.to_csv(fh, index=False)
        fh.write('\n')
        fh.write(f'# CV_tissue_area_mm2,{cv_area:.6f}\n')
        fh.write(f'# CV_superpixels_per_mm2,{cv_ratio:.6f}\n')
        fh.write(f'# pearson_abs_r_density_vs_proportion_dominant_celltype,{abs(pearson_r):.6f}\n')
        fh.write(f'# multiplier_density_per_proportion,{multiplier:.2f}\n')
        fh.write(f'# non_degeneracy_bar_CV,{NON_DEGENERACY_BAR_CV}\n')
        fh.write(f'# non_degeneracy_bar_pearson_r,{NON_DEGENERACY_BAR_R}\n')
        fh.write(f'# gate_cv_passed,{gate_cv_passed}\n')
        fh.write(f'# gate_pearson_passed,{gate_r_passed}\n')
        fh.write(f'# gate_passed,{gate_passed}\n')
        if not gate_passed:
            fh.write('# closure_scope,'
                     '"area-based density on the present acquisition '
                     'design (constant field-of-view); does NOT close '
                     'per-nucleus density, DNA-intensity integral, or '
                     'variable-extent re-acquisition cohorts."\n')
            fh.write('# corroboration,'
                     '"Family C raw-marker CD44+ compartment activation '
                     'remains the only currently-implemented non-compositional '
                     'corroboration for Family A CLR findings."\n')

    print(f'\n✓ Wrote {OUT_PATH}')
    return 0  # closure is a successful audit, not a test failure


if __name__ == '__main__':
    raise SystemExit(main())
