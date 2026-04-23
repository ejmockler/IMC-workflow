"""
Batch Cell Type Annotation for All ROIs

Two annotation modes:
1. Discrete boolean gating: per-ROI percentile gates → 15 cell types.
   Stays per-ROI because discrete gating calls cell identity, not
   population shift.
2. Continuous multi-label memberships: Sham-reference sigmoid on raw
   markers → lineage / subtype / activation scores.  Reference is frozen
   across all ROIs via a pre-generated artifact (see
   ``generate_sham_reference.py``).

Outputs per ROI:
- Parquet: discrete labels + continuous lineage/activation scores per superpixel
- JSON: annotation metadata, cluster annotations, membership summary
- Batch summary: aggregate statistics across all ROIs
"""

import argparse
import datetime
import hashlib
import json
import gzip
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from src.config import Config
from src.analysis.cell_type_annotation import annotate_roi_from_results
from src.utils.paths import get_paths

_PATHS = get_paths()


def load_roi_results(roi_file: Path) -> dict:
    """Load ROI analysis results from gzipped JSON."""
    with gzip.open(roi_file, 'rt') as f:
        return json.load(f)


def _expected_aggregation() -> str:
    return 'per_mouse'


def load_sham_reference(
    scale_um: float,
    protein_channels: list,
    config: Config,
    config_path: Path,
    reference_path: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Load + hard-validate the Sham-reference artifact.

    Enforces every provenance field the generator recorded so a stale or
    hand-edited artifact cannot silently shift continuous memberships:
      * ``scale_um`` matches the current run
      * ``marker_order`` matches ``config.channels.protein_channels`` exactly
      * ``config_sha256`` matches the active ``config.json`` hash
      * ``percentile`` matches ``cell_type_annotation.positivity_threshold.percentile``
      * ``aggregation`` is the current expected value ('per_mouse')
      * each marker entry has both ``threshold`` and ``scale`` keys

    Any failure raises ``ValueError`` with remediation pointing to
    ``python generate_sham_reference.py``.
    """
    if reference_path is None:
        reference_path = (
            _PATHS.biological_analysis_dir / f'sham_reference_{scale_um}um.json'
        )
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Sham-reference artifact not found at {reference_path}. "
            f"Run `python generate_sham_reference.py` first."
        )
    with open(reference_path) as f:
        artifact = json.load(f)

    meta = artifact.get('_metadata', {})

    if float(meta.get('scale_um', -1)) != float(scale_um):
        raise ValueError(
            f"sham_reference artifact scale {meta.get('scale_um')} does not "
            f"match requested {scale_um}. Regenerate with "
            f"`python generate_sham_reference.py`."
        )

    ref_order = tuple(meta.get('marker_order', []))
    cur_order = tuple(protein_channels)
    if ref_order != cur_order:
        raise ValueError(
            f"sham_reference marker_order mismatch.\n"
            f"  artifact: {ref_order}\n"
            f"  current : {cur_order}\n"
            f"  regenerate with `python generate_sham_reference.py`"
        )

    current_config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    artifact_config_sha = meta.get('config_sha256')
    if artifact_config_sha != current_config_sha:
        raise ValueError(
            f"sham_reference config_sha256 does not match current "
            f"config.json.\n"
            f"  artifact : {artifact_config_sha}\n"
            f"  current  : {current_config_sha}\n"
            f"  regenerate with `python generate_sham_reference.py`."
        )

    expected_pct = float(
        config.raw['cell_type_annotation']['positivity_threshold']['percentile']
    )
    artifact_pct = float(meta.get('percentile', -1))
    if artifact_pct != expected_pct:
        raise ValueError(
            f"sham_reference percentile {artifact_pct} does not match "
            f"config.positivity_threshold.percentile {expected_pct}. "
            f"Regenerate the artifact."
        )

    artifact_agg = meta.get('aggregation')
    if artifact_agg != _expected_aggregation():
        raise ValueError(
            f"sham_reference aggregation '{artifact_agg}' is not the "
            f"expected '{_expected_aggregation()}'. The pool-mode artifact "
            f"is diagnostic only and must not drive production annotation."
        )

    reference = artifact.get('reference', {})
    for marker in protein_channels:
        entry = reference.get(marker)
        if entry is None:
            raise ValueError(
                f"sham_reference missing entry for marker '{marker}'"
            )
        for key in ('threshold', 'scale'):
            if key not in entry:
                raise ValueError(
                    f"sham_reference['{marker}'] missing key '{key}' "
                    f"(got {entry}); regenerate the artifact."
                )
    return reference


def detect_prior_annotations(output_dir: Path) -> list:
    """Return the list of pre-existing annotation artifacts in the output dir."""
    return [
        p for p in output_dir.glob('*')
        if p.is_file() and p.suffix in ('.parquet', '.json')
    ]


def archive_prior_annotations(output_dir: Path) -> Optional[Path]:
    """Move existing annotations to ``results/archive/pre_sham_ref_<ts>/``.

    Opt-in via ``--archive-prior``. The default behavior is to fail fast if
    the output dir already contains parquets/JSONs, so that a silent
    re-run does not interleave old per-ROI-sigmoid rows with new
    Sham-reference rows in the same directory. Returns the archive path,
    or None if nothing was moved.
    """
    existing = detect_prior_annotations(output_dir)
    if not existing:
        return None
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        '%Y%m%dT%H%M%SZ'
    )
    archive_dir = _PATHS.results_dir / 'archive' / f'pre_sham_ref_{timestamp}'
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.move(str(path), str(archive_dir / path.name))
    return archive_dir

def deserialize_numpy_arrays(data):
    """Recursively deserialize numpy arrays from JSON representation."""
    if isinstance(data, dict):
        if data.get('__numpy_array__'):
            arr = np.array(data['data'], dtype=data['dtype'])
            return arr.reshape(data['shape'])
        else:
            return {k: deserialize_numpy_arrays(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deserialize_numpy_arrays(item) for item in data]
    else:
        return data

def _parse_cli_args():
    ap = argparse.ArgumentParser(
        description='Batch cell type annotation for all ROIs.'
    )
    ap.add_argument(
        '--archive-prior', action='store_true',
        help='Archive existing annotation outputs to results/archive/ '
             'before running. Default: fail fast if the output dir is not empty.'
    )
    return ap.parse_args()


def main():
    args = _parse_cli_args()

    print("="*80)
    print("Batch Cell Type Annotation - All ROIs")
    print("="*80)

    # Load config
    config_path = Path('config.json')
    config = Config(str(config_path))
    protein_channels = config.channels.get('protein_channels', [])
    print(f"\n✓ Config loaded")
    print(f"  Global threshold: {config.raw['cell_type_annotation']['positivity_threshold']['percentile']}th percentile")
    print(f"  Segmentation: {config.raw['segmentation']['slic_input_channels']}")
    print(f"  Cell types defined: {len(config.raw['cell_type_annotation']['cell_types'])}")
    print(f"  Protein channels : {protein_channels}")

    # Scale to analyze (finest scale for cell-level annotation)
    scale = '10.0'

    # Load + validate Sham-reference artifact (required by continuous
    # memberships). Fast-fail before any ROI work so stale/missing
    # artifacts don't produce half-annotated output.
    membership_configured = bool(
        config.raw.get('cell_type_annotation', {}).get('membership_axes')
    )
    if membership_configured:
        reference_distribution = load_sham_reference(
            scale_um=float(scale),
            protein_channels=protein_channels,
            config=config,
            config_path=config_path,
        )
        print(f"\n✓ Sham reference loaded "
              f"({len(reference_distribution)} markers, scale={scale}µm)")
    else:
        reference_distribution = None
        print("\n  (membership_axes not configured — skipping Sham reference)")

    # Find all ROI results
    results_dir = _PATHS.roi_results_dir
    roi_files = sorted(results_dir.glob('roi_*_results.json.gz'))

    print(f"\n✓ Found {len(roi_files)} ROI results to process")

    # Output directory — fail fast if prior annotations exist unless user
    # opts in to archiving. Silent overwrite would interleave old per-ROI
    # and new Sham-reference parquets; silent archive would shuffle
    # reviewer-facing artifacts under a timestamp nobody thinks to look for.
    output_dir = _PATHS.annotations_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prior = detect_prior_annotations(output_dir)
    if prior:
        if args.archive_prior:
            archive_path = archive_prior_annotations(output_dir)
            print(f"✓ Archived {len(prior)} prior artifact(s) to {archive_path}")
        else:
            raise SystemExit(
                f"Refusing to run: {output_dir} contains {len(prior)} prior "
                f"annotation file(s). Re-run with --archive-prior to move "
                f"them to results/archive/, or clear the directory manually."
            )

    # Track statistics
    batch_stats = {
        'n_rois_processed': 0,
        'n_rois_failed': 0,
        'roi_summaries': {},
        'errors': [],
        'normalization_mode': (
            'sham_reference_v2' if membership_configured else 'discrete_only'
        ),
    }

    for roi_file in roi_files:
        roi_id = roi_file.stem.replace('_results.json', '')

        print(f"\n{'─'*80}")
        print(f"Processing: {roi_id}")
        print(f"{'─'*80}")

        try:
            # Load ROI results
            print(f"  Loading results...")
            roi_results = load_roi_results(roi_file)
            roi_results = deserialize_numpy_arrays(roi_results)

            # Check that scale exists
            if scale not in roi_results.get('multiscale_results', {}):
                print(f"  ❌ Scale {scale}μm not found in results")
                batch_stats['n_rois_failed'] += 1
                batch_stats['errors'].append(f"{roi_id}: Missing scale {scale}")
                continue

            # Annotate using existing infrastructure
            print(f"  Annotating cell types...")
            annotations = annotate_roi_from_results(
                roi_results=roi_results,
                config=config,
                scale=scale,
                reference_distribution=reference_distribution,
            )

            if annotations.get('status') == 'disabled':
                print(f"  ⚠️  Cell type annotation disabled in config")
                batch_stats['n_rois_failed'] += 1
                batch_stats['errors'].append(f"{roi_id}: Annotation disabled")
                continue

            # Extract components
            cell_type_labels = annotations.get('cell_type_labels')
            confidence_scores = annotations.get('confidence_scores')
            annotation_metadata = annotations.get('annotation_metadata')
            cluster_annotations = annotations.get('cluster_annotations')

            # Save annotations
            print(f"  Saving annotations...")

            # Get coordinates from scale results
            scale_results = roi_results['multiscale_results'][scale]
            coords = scale_results.get('superpixel_coords', np.zeros((len(cell_type_labels), 2)))

            # Build annotation dataframe
            import pandas as pd
            annotation_df = pd.DataFrame({
                'superpixel_id': np.arange(len(cell_type_labels)),
                'x': coords[:, 0],
                'y': coords[:, 1],
                'cell_type': cell_type_labels,
                'confidence': confidence_scores
            })

            # Add continuous membership columns if available
            memberships = annotations.get('memberships')
            if memberships is not None:
                for lineage_name, scores in memberships['lineage_scores'].items():
                    annotation_df[f'lineage_{lineage_name}'] = scores
                annotation_df['subtype'] = memberships['subtype_labels']
                for act_name, scores in memberships['activation_scores'].items():
                    annotation_df[f'activation_{act_name}'] = scores
                annotation_df['composite_label'] = memberships['composite_labels']
                # Provenance: stamp every superpixel row with the mode so
                # mixed-mode parquets in downstream dataframes are impossible.
                annotation_df['normalization_mode'] = 'sham_reference_v2'
            else:
                annotation_df['normalization_mode'] = 'discrete_only'

            parquet_file = output_dir / f"{roi_id}_cell_types.parquet"
            annotation_df.to_parquet(parquet_file, index=False)

            # Save metadata (JSON)
            metadata_file = output_dir / f"{roi_id}_annotation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(annotation_metadata, f, indent=2, default=str)

            # Save cluster annotations (JSON)
            if cluster_annotations is not None:
                cluster_file = output_dir / f"{roi_id}_cluster_annotations.json"
                with open(cluster_file, 'w') as f:
                    json.dump(cluster_annotations, f, indent=2, default=str)

            # Compute summary statistics
            n_total = len(cell_type_labels)
            n_unassigned = (cell_type_labels == 'unassigned').sum()
            n_ambiguous = annotation_metadata.get('n_multi_assigned', 0)

            # Count each cell type (use annotation_metadata which has counts)
            cell_type_counts = annotation_metadata.get('assignment_counts', {})

            summary = {
                'n_superpixels': n_total,
                'n_assigned': int(n_total - n_unassigned),
                'n_unassigned': int(n_unassigned),
                'n_ambiguous': int(n_ambiguous),
                'assignment_rate': float((n_total - n_unassigned) / n_total) if n_total > 0 else 0.0,
                'ambiguity_rate': float(n_ambiguous / n_total) if n_total > 0 else 0.0,
                'cell_type_counts': cell_type_counts,
                'top_cell_type': max(cell_type_counts.items(), key=lambda x: x[1])[0] if cell_type_counts else None,
                'top_cell_type_fraction': float(max(cell_type_counts.values()) / n_total) if cell_type_counts and n_total > 0 else 0.0
            }

            if cluster_annotations is not None and isinstance(cluster_annotations, dict):
                summary['n_clusters'] = len(cluster_annotations)
                confidences = [ann.get('confidence', 0.0) for ann in cluster_annotations.values()]
                summary['cluster_confidence_mean'] = float(np.mean(confidences)) if confidences else 0.0

            # Add continuous membership summary
            if memberships is not None:
                ls = memberships['lineage_scores']
                # Use the composite-label lineage threshold (the one that
                # actually classifies a superpixel as single/multi/no-lineage
                # in the parquet output) rather than the subtype gate, which
                # governs a different decision.
                lineage_threshold = float(
                    config.raw['cell_type_annotation']['membership_axes']
                    ['composite_label_thresholds']['lineage']
                )
                above_thresh = {ln: (scores > lineage_threshold).sum() for ln, scores in ls.items()}
                n_lineages = sum((scores > lineage_threshold).astype(int) for scores in ls.values())
                summary['membership'] = {
                    'lineage_means': {ln: float(scores.mean()) for ln, scores in ls.items()},
                    'lineage_above_threshold': {ln: int(c) for ln, c in above_thresh.items()},
                    'n_mixed': int((n_lineages >= 2).sum()),
                    'n_single_lineage': int((n_lineages == 1).sum()),
                    'n_no_lineage': int((n_lineages == 0).sum()),
                    'composite_counts': dict(zip(*np.unique(memberships['composite_labels'], return_counts=True))),
                    'subtype_counts': dict(zip(*np.unique(memberships['subtype_labels'], return_counts=True))),
                }
                # Convert numpy int64 to int for JSON serialization
                for key in ['composite_counts', 'subtype_counts']:
                    summary['membership'][key] = {
                        str(k): int(v) for k, v in summary['membership'][key].items()
                    }

            batch_stats['roi_summaries'][roi_id] = summary
            batch_stats['n_rois_processed'] += 1

            n_mixed = summary.get('membership', {}).get('n_mixed', 0)
            n_single = summary.get('membership', {}).get('n_single_lineage', 0)
            n_no_lin = summary.get('membership', {}).get('n_no_lineage', 0)

            print(f"  ✓ Complete")
            print(f"    Discrete: {summary['n_assigned']}/{n_total} assigned ({summary['assignment_rate']*100:.1f}%)")
            print(f"    Continuous: {n_single} single-lineage, {n_mixed} mixed, {n_no_lin} unresolved")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            batch_stats['n_rois_failed'] += 1
            batch_stats['errors'].append(f"{roi_id}: {str(e)}")

    # Save batch summary
    print(f"\n{'='*80}")
    print("Batch Processing Complete")
    print(f"{'='*80}")
    print(f"\nProcessed: {batch_stats['n_rois_processed']}/{len(roi_files)} ROIs")
    print(f"Failed: {batch_stats['n_rois_failed']} ROIs")

    if batch_stats['errors']:
        print(f"\nErrors:")
        for error in batch_stats['errors']:
            print(f"  - {error}")

    # Overall statistics
    if batch_stats['roi_summaries']:
        all_assignment_rates = [s['assignment_rate'] for s in batch_stats['roi_summaries'].values()]
        all_ambiguity_rates = [s['ambiguity_rate'] for s in batch_stats['roi_summaries'].values()]

        print(f"\nOverall Statistics:")
        print(f"  Assignment rate: {np.mean(all_assignment_rates)*100:.1f}% ± {np.std(all_assignment_rates)*100:.1f}%")
        print(f"  Ambiguity rate: {np.mean(all_ambiguity_rates)*100:.1f}% ± {np.std(all_ambiguity_rates)*100:.1f}%")

        # Count discrete cell types across all ROIs
        all_cell_types = {}
        for roi_summary in batch_stats['roi_summaries'].values():
            for ct, count in roi_summary['cell_type_counts'].items():
                all_cell_types[ct] = all_cell_types.get(ct, 0) + count

        print(f"\nDiscrete Cell Type Distribution (All ROIs):")
        for ct, count in sorted(all_cell_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ct:40s}: {count:6d}")

        # Continuous membership summary
        any_membership = any('membership' in s for s in batch_stats['roi_summaries'].values())
        if any_membership:
            total_mixed = sum(s.get('membership', {}).get('n_mixed', 0) for s in batch_stats['roi_summaries'].values())
            total_single = sum(s.get('membership', {}).get('n_single_lineage', 0) for s in batch_stats['roi_summaries'].values())
            total_no = sum(s.get('membership', {}).get('n_no_lineage', 0) for s in batch_stats['roi_summaries'].values())
            total_sp = total_mixed + total_single + total_no
            print(f"\nContinuous Membership Summary (All ROIs):")
            print(f"  Single-lineage:  {total_single:6d} ({100*total_single/total_sp:.1f}%)")
            print(f"  Multi-lineage:   {total_mixed:6d} ({100*total_mixed/total_sp:.1f}%)")
            print(f"  No lineage:      {total_no:6d} ({100*total_no/total_sp:.1f}%)")

    # Save batch summary
    summary_file = output_dir / 'batch_annotation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(batch_stats, f, indent=2, default=str)

    print(f"\n✓ Batch summary saved to: {summary_file}")
    print(f"✓ Annotations saved to: {output_dir}")
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()
