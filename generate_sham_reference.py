"""
Generate the Sham-reference normalization artifact for continuous memberships.

Produces ``results/biological_analysis/sham_reference_10.0um.json`` — a
frozen per-marker {threshold, scale} reference consumed by
``batch_annotate_all_rois.py`` → ``compute_continuous_memberships``.

Threshold is Sham-only, per-mouse-aggregated percentile (biological
replication unit, not superpixel-pooled). Scale is experiment-wide IQR
(all timepoints pooled), so injury-driven departures aren't clipped by a
narrow Sham-only denominator (the previous per-ROI IQR pathology).

Hard gates abort on any deviation from the pilot design (2 Sham mice,
6 Sham ROIs, 24 total ROIs, all 9 protein channels). Silent partial-state
generation would reintroduce the confound this script exists to close.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

from src.config import Config
from src.analysis.sham_reference import build_reference_distribution
from src.utils.canonical_loader import build_superpixel_dataframe, load_all_rois
from src.utils.paths import get_paths


REPO_ROOT = Path(__file__).resolve().parent
SCALE_UM = 10.0

# Pilot design invariants — if these change, re-audit everything downstream
# before relaxing the gates.
EXPECTED_N_SHAM_MICE = 2
EXPECTED_N_SHAM_ROIS = 6
EXPECTED_N_TOTAL_ROIS = 24


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"generate_sham_reference: {message}")


def _load_per_marker_overrides(threshold_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    overrides_raw = threshold_config.get('per_marker_override', {}) or {}
    clean: Dict[str, Dict[str, Any]] = {}
    for marker, spec in overrides_raw.items():
        if marker.startswith('_'):
            continue
        clean[marker] = {
            k: v for k, v in spec.items() if not k.startswith('_')
        }
    return clean


def _git_hash_and_dirty(repo_root: Path) -> Dict[str, Any]:
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_root,
        ).decode().strip()
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=repo_root,
        ).decode().strip()
    except Exception:
        return {'git_hash': None, 'git_dirty': None}
    return {'git_hash': git_hash, 'git_dirty': bool(status)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(
    scale_um: float = SCALE_UM,
    output_path: Path | None = None,
    config_path: Path | None = None,
) -> Path:
    paths = get_paths()
    config_path = config_path or (REPO_ROOT / 'config.json')
    output_path = output_path or (
        paths.biological_analysis_dir / f'sham_reference_{scale_um}um.json'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load config ----
    config = Config(str(config_path))
    annotation_cfg = config.raw.get('cell_type_annotation', {})
    threshold_cfg = annotation_cfg.get('positivity_threshold', {})
    percentile = float(threshold_cfg.get('percentile', 60.0))
    per_marker_overrides = _load_per_marker_overrides(threshold_cfg)

    protein_channels: Sequence[str] = tuple(
        config.channels.get('protein_channels', [])
    )
    _require(
        len(protein_channels) > 0,
        "config.channels.protein_channels is empty",
    )

    # ---- Load canonical superpixels ----
    results = load_all_rois(str(paths.roi_results_dir))
    _require(
        len(results) == EXPECTED_N_TOTAL_ROIS,
        f"loaded {len(results)} ROI results; pilot expects "
        f"{EXPECTED_N_TOTAL_ROIS} (24 analyzed, 1 Test excluded pre-analysis)",
    )
    df = build_superpixel_dataframe(results, scale=scale_um)

    missing_markers = [m for m in protein_channels if m not in df.columns]
    _require(
        not missing_markers,
        f"canonical superpixels missing protein channels {missing_markers}",
    )
    for needed in ('timepoint', 'mouse'):
        _require(needed in df.columns,
                 f"canonical superpixels missing '{needed}' column")

    sham_df = df[df['timepoint'] == 'Sham']
    n_sham_mice = int(sham_df['mouse'].nunique())
    n_sham_rois = int(sham_df['roi'].nunique())
    n_sham_superpixels = int(len(sham_df))

    _require(
        n_sham_mice == EXPECTED_N_SHAM_MICE,
        f"found {n_sham_mice} Sham mice; pilot expects "
        f"{EXPECTED_N_SHAM_MICE} (MS1, MS2). Partial Sham would silently "
        f"shift the reference baseline.",
    )
    _require(
        n_sham_rois == EXPECTED_N_SHAM_ROIS,
        f"found {n_sham_rois} Sham ROIs; pilot expects "
        f"{EXPECTED_N_SHAM_ROIS} (3/mouse × 2 mice).",
    )

    # ---- Build reference ----
    reference = build_reference_distribution(
        df,
        markers=protein_channels,
        percentile=percentile,
        per_marker_overrides=per_marker_overrides,
        aggregation='per_mouse',
    )

    # ---- Provenance ----
    provenance: Dict[str, Any] = {
        'scale_um': scale_um,
        'marker_order': list(protein_channels),
        'percentile': percentile,
        'per_marker_overrides': per_marker_overrides,
        'aggregation': 'per_mouse',
        'n_sham_mice': n_sham_mice,
        'n_sham_rois': n_sham_rois,
        'n_sham_superpixels': n_sham_superpixels,
        'n_total_rois': len(results),
        'n_total_superpixels': int(len(df)),
        'config_sha256': _sha256(config_path),
        'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python_version': sys.version.split()[0],
        'generator_script': 'generate_sham_reference.py',
    }
    provenance.update(_git_hash_and_dirty(REPO_ROOT))

    artifact = {
        '_metadata': provenance,
        'reference': {
            m: {
                'threshold': float(reference[m]['threshold']),
                'scale': float(reference[m]['scale']),
            }
            for m in protein_channels
        },
    }

    with open(output_path, 'w') as f:
        json.dump(artifact, f, indent=2)

    return output_path


def main() -> int:
    print('=' * 80)
    print(f'Generating Sham-reference artifact @ scale {SCALE_UM}µm')
    print('=' * 80)

    path = generate()

    with open(path) as f:
        artifact = json.load(f)

    meta = artifact['_metadata']
    print(f'\n✓ Wrote {path}')
    print(f'  scale              : {meta["scale_um"]} µm')
    print(f'  percentile         : {meta["percentile"]}')
    print(f'  aggregation        : {meta["aggregation"]}')
    print(f'  Sham mice / ROIs   : {meta["n_sham_mice"]} / {meta["n_sham_rois"]}')
    print(f'  Sham superpixels   : {meta["n_sham_superpixels"]:,}')
    print(f'  total superpixels  : {meta["n_total_superpixels"]:,}')
    print(f'  config_sha256      : {meta["config_sha256"][:16]}…')
    print(f'  git_hash           : {(meta.get("git_hash") or "?")[:12]}'
          f'{" (dirty)" if meta.get("git_dirty") else ""}')
    print(f'\n  reference (threshold, scale):')
    for m, vals in artifact['reference'].items():
        print(f'    {m:<8s}  threshold={vals["threshold"]:+.4f}  '
              f'scale={vals["scale"]:.4f}')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
