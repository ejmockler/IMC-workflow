"""
Marker Profiling - Diagnostic tool for investigating cell type detection issues.

This module provides distribution analysis for individual protein markers,
helping diagnose why certain cell types may not be detected.

Separation of Concerns:
- This is a DIAGNOSTIC tool, not part of the production pipeline
- Used for exploratory analysis and threshold calibration
- Outputs inform configuration decisions, but doesn't modify them

Standalone module - no internal dependencies to avoid package import issues.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class MarkerProfiler:
    """Profile protein marker distributions to diagnose detection issues."""

    def __init__(self, ion_counts: pd.DataFrame, metadata: pd.DataFrame):
        """
        Initialize profiler with ion count data.

        Args:
            ion_counts: Superpixel-level protein measurements
            metadata: ROI-level metadata (timepoint, region, etc.)
        """
        self.ion_counts = ion_counts
        self.metadata = metadata

    def profile_marker(self, marker: str) -> Dict:
        """
        Generate comprehensive profile for a single marker.

        Returns distribution statistics, percentiles, presence across ROIs.
        """
        if marker not in self.ion_counts.columns:
            raise ValueError(f"Marker {marker} not found in data")

        values = self.ion_counts[marker].values

        # Basic statistics
        profile = {
            'marker': marker,
            'n_measurements': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'percentiles': {
                p: float(np.percentile(values, p))
                for p in [50, 60, 65, 70, 75, 80, 85, 90, 95]
            },
            'zero_fraction': float(np.mean(values == 0)),
            'dynamic_range': float(np.max(values) - np.min(values))
        }

        # ROI-level statistics
        roi_stats = []
        for roi in self.ion_counts['roi'].unique():
            roi_values = self.ion_counts[self.ion_counts['roi'] == roi][marker].values
            roi_stats.append({
                'roi': roi,
                'mean': float(np.mean(roi_values)),
                'median': float(np.median(roi_values)),
                'p60': float(np.percentile(roi_values, 60)),
                'p70': float(np.percentile(roi_values, 70)),
                'n_superpixels': len(roi_values)
            })

        profile['roi_statistics'] = roi_stats

        return profile

    def compare_markers(self, markers: List[str]) -> pd.DataFrame:
        """
        Compare distribution statistics across multiple markers.

        Useful for understanding relative expression levels.
        """
        comparisons = []

        for marker in markers:
            if marker not in self.ion_counts.columns:
                continue

            values = self.ion_counts[marker].values
            comparisons.append({
                'marker': marker,
                'mean': np.mean(values),
                'median': np.median(values),
                'p60': np.percentile(values, 60),
                'p70': np.percentile(values, 70),
                'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0,
                'zero_fraction': np.mean(values == 0)
            })

        return pd.DataFrame(comparisons)

    def diagnose_cell_type_detection(
        self,
        cell_type_def: Dict,
        threshold_percentile: int = 60
    ) -> Dict:
        """
        Diagnose why a cell type definition may not be detecting cells.

        Args:
            cell_type_def: Cell type definition from config (markers + logic)
            threshold_percentile: Percentile threshold for positivity

        Returns:
            Diagnostic report with detection statistics
        """
        markers = cell_type_def['markers']
        logic = cell_type_def['logic']

        # Get marker profiles
        marker_profiles = {m: self.profile_marker(m) for m in markers}

        # Simulate boolean gating to estimate detection rate
        thresholds = {
            m: np.percentile(self.ion_counts[m].values, threshold_percentile)
            for m in markers
        }

        # Evaluate logic expression
        positivity = {}
        for m in markers:
            positivity[m] = (self.ion_counts[m] >= thresholds[m]).values

        # Parse logic and compute final positivity
        # For simplicity, handle common patterns
        if ' & ' in logic:
            # AND logic
            marker_list = logic.split(' & ')
            final_positivity = np.ones(len(self.ion_counts), dtype=bool)
            for m in marker_list:
                final_positivity &= positivity[m.strip()]
        elif logic in markers:
            # Single marker
            final_positivity = positivity[logic]
        else:
            # Complex logic - skip for now
            final_positivity = np.zeros(len(self.ion_counts), dtype=bool)

        n_positive = np.sum(final_positivity)
        fraction_positive = n_positive / len(self.ion_counts)

        # ROI-level detection
        roi_detection = []
        for roi in self.ion_counts['roi'].unique():
            roi_mask = self.ion_counts['roi'] == roi
            n_roi_positive = np.sum(final_positivity[roi_mask])
            roi_detection.append({
                'roi': roi,
                'n_detected': int(n_roi_positive),
                'fraction_detected': float(n_roi_positive / np.sum(roi_mask))
            })

        return {
            'cell_type_definition': cell_type_def,
            'threshold_percentile': threshold_percentile,
            'thresholds_used': thresholds,
            'marker_profiles': marker_profiles,
            'detection_summary': {
                'n_positive_superpixels': int(n_positive),
                'fraction_positive': float(fraction_positive),
                'n_rois_with_detection': int(sum(1 for r in roi_detection if r['n_detected'] > 0)),
                'total_rois': len(roi_detection)
            },
            'roi_detection': roi_detection
        }

    def threshold_sensitivity_analysis(
        self,
        marker: str,
        percentile_range: Tuple[int, int] = (50, 90),
        step: int = 5
    ) -> pd.DataFrame:
        """
        Analyze how detection rate changes with threshold percentile.

        Useful for calibrating thresholds when dealing with rare populations.
        """
        results = []

        for percentile in range(percentile_range[0], percentile_range[1] + 1, step):
            threshold = np.percentile(self.ion_counts[marker].values, percentile)
            n_positive = np.sum(self.ion_counts[marker] >= threshold)
            fraction_positive = n_positive / len(self.ion_counts)

            results.append({
                'percentile': percentile,
                'threshold': threshold,
                'n_positive': n_positive,
                'fraction_positive': fraction_positive
            })

        return pd.DataFrame(results)


def profile_all_markers(
    ion_counts_path: Path,
    output_dir: Path,
    markers: Optional[List[str]] = None
) -> None:
    """
    Generate comprehensive marker profiling report.

    This is a convenience function for batch processing.
    """
    # Load data
    ion_counts = pd.read_parquet(ion_counts_path)

    if markers is None:
        # Profile all numeric columns except metadata
        markers = [col for col in ion_counts.columns
                  if col not in ['roi', 'superpixel_id', 'x', 'y']]

    profiler = MarkerProfiler(ion_counts, metadata=None)

    # Generate individual marker profiles
    profiles = {}
    for marker in markers:
        print(f"Profiling {marker}...")
        profiles[marker] = profiler.profile_marker(marker)

    # Save comprehensive report
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'marker_profiles.json', 'w') as f:
        json.dump(profiles, f, indent=2)

    # Generate comparison table
    comparison = profiler.compare_markers(markers)
    comparison.to_csv(output_dir / 'marker_comparison.csv', index=False)

    print(f"✅ Profiling complete. Results saved to {output_dir}")


def diagnose_m2_macrophage_detection(
    ion_counts_path: Path,
    cell_type_annotations_path: Path,
    output_dir: Path
) -> None:
    """
    Specific diagnostic for M2 macrophage detection issue.

    This function is purpose-built to investigate why M2 macrophages
    are not being detected in the kidney injury data.
    """
    # Load data
    ion_counts = pd.read_parquet(ion_counts_path)

    with open(cell_type_annotations_path, 'r') as f:
        annotations = json.load(f)

    # Get M2 definition
    m2_def = annotations['cell_type_definitions']['M2_macrophage']

    profiler = MarkerProfiler(ion_counts, metadata=None)

    # Full diagnostic
    print("🔍 Diagnosing M2 Macrophage Detection")
    print("=" * 60)

    diagnosis = profiler.diagnose_cell_type_detection(m2_def, threshold_percentile=60)

    print(f"\nCell Type: M2_macrophage")
    print(f"Markers: {m2_def['markers']}")
    print(f"Logic: {m2_def['logic']}")
    print(f"\nThreshold (60th percentile): {diagnosis['thresholds_used']}")
    print(f"\nDetection Summary:")
    print(f"  Positive superpixels: {diagnosis['detection_summary']['n_positive_superpixels']}")
    print(f"  Fraction: {diagnosis['detection_summary']['fraction_positive']:.4f}")
    print(f"  ROIs with detection: {diagnosis['detection_summary']['n_rois_with_detection']}/{diagnosis['detection_summary']['total_rois']}")

    # CD206 sensitivity analysis
    print("\n📊 CD206 Threshold Sensitivity:")
    sensitivity = profiler.threshold_sensitivity_analysis('CD206', percentile_range=(40, 80), step=5)
    print(sensitivity.to_string(index=False))

    # Compare CD206 to other markers
    print("\n📈 Marker Comparison (relative expression):")
    comparison = profiler.compare_markers(['CD206', 'CD11b', 'CD45', 'Ly6G', 'CD44'])
    print(comparison.to_string(index=False))

    # Save full report
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'm2_macrophage_diagnostic.json', 'w') as f:
        json.dump(diagnosis, f, indent=2)

    sensitivity.to_csv(output_dir / 'm2_cd206_sensitivity.csv', index=False)
    comparison.to_csv(output_dir / 'm2_marker_comparison.csv', index=False)

    print(f"\n✅ Diagnostic complete. Full report saved to {output_dir}")
