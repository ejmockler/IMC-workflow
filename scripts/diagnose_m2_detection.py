#!/usr/bin/env python3
"""
Diagnostic script for M2 macrophage detection issue.

SWE Principle: Separate exploratory diagnostics from production pipeline.
This script investigates WHY M2 macrophages aren't detected, without modifying
the pipeline. Findings inform configuration decisions.

Usage:
    python scripts/diagnose_m2_detection.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import MarkerProfiler class directly to avoid package import issues
from analysis.marker_profiling import MarkerProfiler


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results' / 'biological_analysis'

    ion_counts_path = results_dir / 'cell_type_annotations' / 'batch_ion_counts.parquet'
    annotations_path = results_dir / 'cell_type_annotations' / 'batch_annotation_summary.json'
    output_dir = base_dir / 'results' / 'diagnostics' / 'm2_macrophage'

    # Check if files exist
    if not ion_counts_path.exists():
        print(f"❌ Ion counts not found: {ion_counts_path}")
        print("   Run cell type annotation pipeline first")
        return 1

    if not annotations_path.exists():
        print(f"❌ Annotations not found: {annotations_path}")
        return 1

    # Run comprehensive diagnostic
    print("🔬 M2 Macrophage Detection Diagnostic")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    ion_counts = pd.read_parquet(ion_counts_path)

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Get M2 definition
    m2_def = annotations['cell_type_definitions']['M2_macrophage']

    profiler = MarkerProfiler(ion_counts, metadata=None)

    # 1. CD206 distribution analysis
    print("\n" + "="*70)
    print("1. CD206 Distribution Analysis")
    print("="*70)

    cd206_profile = profiler.profile_marker('CD206')

    print(f"\nGlobal Statistics (n={cd206_profile['n_measurements']} superpixels):")
    print(f"  Mean: {cd206_profile['mean']:.4f}")
    print(f"  Median: {cd206_profile['median']:.4f}")
    print(f"  Std: {cd206_profile['std']:.4f}")
    print(f"  Min: {cd206_profile['min']:.4f}")
    print(f"  Max: {cd206_profile['max']:.4f}")
    print(f"  Zero fraction: {cd206_profile['zero_fraction']:.2%}")

    print(f"\nPercentile Thresholds:")
    for p, val in sorted(cd206_profile['percentiles'].items()):
        print(f"  {p}th: {val:.4f}")

    # 2. ROI-level variation
    print("\n" + "="*70)
    print("2. ROI-Level CD206 Expression")
    print("="*70)

    roi_stats = pd.DataFrame(cd206_profile['roi_statistics'])
    print(f"\nCD206 expression across {len(roi_stats)} ROIs:")
    print(roi_stats.describe().to_string())

    # Find ROIs with highest CD206
    top_rois = roi_stats.nlargest(5, 'mean')
    print(f"\nTop 5 ROIs by CD206 mean expression:")
    print(top_rois[['roi', 'mean', 'median', 'p60', 'p70']].to_string(index=False))

    # 3. Threshold sensitivity
    print("\n" + "="*70)
    print("3. Threshold Sensitivity Analysis")
    print("="*70)

    sensitivity = profiler.threshold_sensitivity_analysis('CD206', percentile_range=(30, 90), step=5)
    print("\nDetection rate vs. threshold percentile:")
    print(sensitivity.to_string(index=False))

    # 4. Compare to other immune markers
    print("\n" + "="*70)
    print("4. Comparison to Other Markers")
    print("="*70)

    immune_markers = ['CD206', 'CD11b', 'CD45', 'Ly6G', 'CD44']
    comparison = profiler.compare_markers(immune_markers)
    print("\nRelative expression levels:")
    print(comparison.to_string(index=False))

    # 5. Simulate detection with different thresholds
    print("\n" + "="*70)
    print("5. Simulated Detection at Different Thresholds")
    print("="*70)

    for percentile in [40, 50, 60, 70, 80]:
        diagnosis = profiler.diagnose_cell_type_detection(m2_def, threshold_percentile=percentile)
        n_detected = diagnosis['detection_summary']['n_positive_superpixels']
        frac = diagnosis['detection_summary']['fraction_positive']
        n_rois = diagnosis['detection_summary']['n_rois_with_detection']
        print(f"  {percentile}th percentile: {n_detected} cells ({frac:.2%}) in {n_rois}/25 ROIs")

    # 6. Check for co-expression with macrophage markers
    print("\n" + "="*70)
    print("6. CD206 Co-expression with Macrophage Markers")
    print("="*70)

    # Check CD206+ cells for CD11b and CD45 expression
    cd206_threshold = np.percentile(ion_counts['CD206'], 60)
    cd206_positive = ion_counts[ion_counts['CD206'] >= cd206_threshold]

    if len(cd206_positive) > 0:
        print(f"\nCD206+ cells (60th percentile, n={len(cd206_positive)}):")
        print(f"  Mean CD11b: {cd206_positive['CD11b'].mean():.4f}")
        print(f"  Mean CD45: {cd206_positive['CD45'].mean():.4f}")
        print(f"  Mean Ly6G: {cd206_positive['Ly6G'].mean():.4f}")

        # Check how many are also macrophage-like (CD11b+CD45+)
        cd11b_thresh = np.percentile(ion_counts['CD11b'], 60)
        cd45_thresh = np.percentile(ion_counts['CD45'], 60)

        double_positive = cd206_positive[
            (cd206_positive['CD11b'] >= cd11b_thresh) &
            (cd206_positive['CD45'] >= cd45_thresh)
        ]

        print(f"\n  CD206+CD11b+CD45+ (canonical macrophage): {len(double_positive)} ({len(double_positive)/len(cd206_positive):.1%})")
    else:
        print("\n❌ No CD206+ cells detected at 60th percentile threshold")

    # 7. Biological interpretation
    print("\n" + "="*70)
    print("7. Biological Interpretation")
    print("="*70)

    print("""
Current M2 Macrophage Definition:
  Markers: CD206 (single marker)
  Logic: CD206 positivity
  Threshold: 60th percentile

Possible Explanations for Zero Detection:
  1. BIOLOGICAL: M2 macrophages genuinely absent/rare in this injury model
     - Early timepoints (D1-D7) may be M1-dominant (pro-inflammatory)
     - M2 polarization typically occurs during resolution (D14+)

  2. METHODOLOGICAL: Single-marker definition insufficient
     - CD206 alone may not distinguish M2 from other cell types
     - Need: CD206 + CD11b + CD45 (+ maybe CD163 if available)

  3. TECHNICAL: Threshold too stringent for rare population
     - 60th percentile appropriate for abundant cells
     - Rare populations may need 40th-50th percentile

  4. PANEL LIMITATION: Missing key M2 markers
     - CD163, CD204 (scavenger receptors) not in panel
     - Cannot definitively identify M2 phenotype with current markers

Recommendations:
  A. Check literature: Are M2 macrophages expected at D1-D7 post-IRI?
  B. Refine definition: CD206 + CD11b + CD45 (multi-marker)
  C. Lower threshold: Try 50th percentile for CD206
  D. Validate: If still zero, this may be biological reality
    """)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all data
    with open(output_dir / 'cd206_profile.json', 'w') as f:
        json.dump(cd206_profile, f, indent=2)

    roi_stats.to_csv(output_dir / 'roi_cd206_stats.csv', index=False)
    sensitivity.to_csv(output_dir / 'threshold_sensitivity.csv', index=False)
    comparison.to_csv(output_dir / 'marker_comparison.csv', index=False)

    print(f"\n✅ Diagnostic complete. Results saved to {output_dir}")
    print("\nNext steps:")
    print("  1. Review diagnostic outputs")
    print("  2. Consult literature on M2 macrophage kinetics in kidney IRI")
    print("  3. Decide: biological reality vs. methodological refinement needed")
    print("  4. If refinement needed, update config.json with new definition")

    return 0


if __name__ == '__main__':
    sys.exit(main())
