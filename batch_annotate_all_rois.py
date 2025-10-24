"""
Batch Cell Type Annotation for All ROIs

Processes all 25 ROIs from the kidney injury time course with optimized parameters:
- 60th percentile global threshold (validated to capture sparse IMC signal)
- DNA-based segmentation (consistent across ROI heterogeneity)
- Config-driven cell type definitions

Outputs:
- Per-ROI cell type annotations (Parquet format)
- Per-ROI cluster-to-celltype mappings (JSON)
- Batch summary statistics
"""

import json
import gzip
from pathlib import Path
import numpy as np
from src.config import Config
from src.analysis.cell_type_annotation import annotate_roi_from_results

def load_roi_results(roi_file: Path) -> dict:
    """Load ROI analysis results from gzipped JSON."""
    with gzip.open(roi_file, 'rt') as f:
        return json.load(f)

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

def main():
    print("="*80)
    print("Batch Cell Type Annotation - All ROIs")
    print("="*80)

    # Load config
    config = Config('config.json')
    print(f"\n✓ Config loaded")
    print(f"  Global threshold: {config.raw['cell_type_annotation']['positivity_threshold']['percentile']}th percentile")
    print(f"  Segmentation: {config.raw['segmentation']['slic_input_channels']}")
    print(f"  Cell types defined: {len(config.raw['cell_type_annotation']['cell_types'])}")

    # Find all ROI results
    results_dir = Path('results/roi_results')
    roi_files = sorted(results_dir.glob('roi_*_results.json.gz'))

    print(f"\n✓ Found {len(roi_files)} ROI results to process")

    # Output directory
    output_dir = Path('results/biological_analysis/cell_type_annotations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track statistics
    batch_stats = {
        'n_rois_processed': 0,
        'n_rois_failed': 0,
        'roi_summaries': {},
        'errors': []
    }

    # Get scale to analyze (use middle scale for annotation)
    scale = '10.0'  # Use finest scale for cell-level annotation

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
                scale=scale
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

            # Save superpixel-level annotations (JSON)
            import pandas as pd
            annotation_df = pd.DataFrame({
                'superpixel_id': np.arange(len(cell_type_labels)),
                'x': coords[:, 0],
                'y': coords[:, 1],
                'cell_type': cell_type_labels,
                'confidence': confidence_scores
            })

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
                # Get mean confidence across cluster annotations
                confidences = [ann.get('confidence', 0.0) for ann in cluster_annotations.values()]
                summary['cluster_confidence_mean'] = float(np.mean(confidences)) if confidences else 0.0

            batch_stats['roi_summaries'][roi_id] = summary
            batch_stats['n_rois_processed'] += 1

            print(f"  ✓ Complete")
            print(f"    Assigned: {summary['n_assigned']}/{n_total} ({summary['assignment_rate']*100:.1f}%)")
            print(f"    Ambiguous: {n_ambiguous} ({summary['ambiguity_rate']*100:.1f}%)")
            print(f"    Top type: {summary['top_cell_type']} ({summary['top_cell_type_fraction']*100:.1f}%)")

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

        # Count cell types across all ROIs
        all_cell_types = {}
        for roi_summary in batch_stats['roi_summaries'].values():
            for ct, count in roi_summary['cell_type_counts'].items():
                all_cell_types[ct] = all_cell_types.get(ct, 0) + count

        print(f"\nCell Type Distribution (All ROIs):")
        for ct, count in sorted(all_cell_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ct:40s}: {count:6d}")

    # Save batch summary
    summary_file = output_dir / 'batch_annotation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(batch_stats, f, indent=2, default=str)

    print(f"\n✓ Batch summary saved to: {summary_file}")
    print(f"✓ Annotations saved to: {output_dir}")
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()
