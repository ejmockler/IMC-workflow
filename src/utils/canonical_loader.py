"""
Canonical loader for IMC analysis results (gzipped JSON format).

This module provides the official way to load and work with IMC analysis results
according to the schema documented in docs/DATA_SCHEMA.md.

Author: IMC Analysis Pipeline
Version: 1.0
"""

import gzip
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ROIMetadata:
    """Structured ROI metadata."""
    roi_id: str
    timepoint: str
    mouse: str
    condition: str
    roi_number: str
    replicate: str

    @classmethod
    def from_filename(cls, filename: str) -> 'ROIMetadata':
        """Parse metadata from standard filename pattern.

        Pattern: roi_IMC_241218_Alun_ROI_{timepoint}_{mouse}_{roi_number}_{replicate}_results.json.gz
        """
        # Extract ROI ID from filename
        stem = Path(filename).stem
        if stem.endswith('_results'):
            stem = stem[:-8]  # Remove '_results'
        if stem.endswith('.json'):
            stem = stem[:-5]  # Remove '.json'
        if stem.startswith('roi_'):
            stem = stem[4:]  # Remove 'roi_' prefix

        # Parse components
        parts = stem.split('_ROI_')
        if len(parts) < 2:
            raise ValueError(f"Cannot parse filename: {filename}")

        # Handle Sham samples (format: Sam1_01_2 or Sam2_03_8)
        if 'Sam' in parts[1]:
            sham_parts = parts[1].split('_')
            if len(sham_parts) >= 3:
                timepoint = 'Sham'
                mouse = sham_parts[0]  # Sam1 or Sam2
                roi_number = sham_parts[1]
                replicate = sham_parts[2]
                condition = 'Sham'
            else:
                raise ValueError(f"Cannot parse Sham ROI: {filename}")
        else:
            # Standard UUO format: D1_M1_01_9
            roi_parts = parts[1].split('_')
            if len(roi_parts) < 4:
                raise ValueError(f"Cannot parse ROI components: {filename}")

            timepoint = roi_parts[0]
            mouse = roi_parts[1]
            roi_number = roi_parts[2]
            replicate = roi_parts[3]
            condition = 'UUO'

        return cls(
            roi_id=stem,
            timepoint=timepoint,
            mouse=mouse,
            condition=condition,
            roi_number=roi_number,
            replicate=replicate
        )


def deserialize_array(arr_dict: Any) -> Any:
    """Convert serialized numpy array back to numpy array.

    Args:
        arr_dict: Either a dict with '__numpy_array__' key or a regular value

    Returns:
        Numpy array if arr_dict is serialized array, otherwise arr_dict unchanged
    """
    if isinstance(arr_dict, dict) and '__numpy_array__' in arr_dict:
        return np.array(
            arr_dict['data'],
            dtype=arr_dict['dtype']
        ).reshape(arr_dict['shape'])
    return arr_dict


def load_roi_results(roi_file: Union[str, Path]) -> Dict[str, Any]:
    """Load a single ROI result file.

    Args:
        roi_file: Path to gzipped JSON result file

    Returns:
        Dictionary with complete result structure

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is corrupted
    """
    roi_file = Path(roi_file)
    if not roi_file.exists():
        raise FileNotFoundError(f"Result file not found: {roi_file}")

    try:
        with gzip.open(roi_file, 'rt') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {roi_file}: {e}")
        raise


def load_all_rois(results_dir: Union[str, Path] = 'results/roi_results') -> Dict[str, Dict[str, Any]]:
    """Load all ROI results from directory.

    Args:
        results_dir: Directory containing roi_*_results.json.gz files

    Returns:
        Dictionary mapping roi_id to result data
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_results = {}
    roi_files = sorted(results_dir.glob('roi_*_results.json.gz'))

    if not roi_files:
        logger.warning(f"No result files found in {results_dir}")
        return all_results

    for roi_file in roi_files:
        try:
            roi_id = roi_file.stem.replace('roi_', '').replace('_results.json', '').replace('.json', '')
            all_results[roi_id] = load_roi_results(roi_file)
        except Exception as e:
            logger.error(f"Skipping {roi_file.name}: {e}")
            continue

    logger.info(f"Loaded {len(all_results)} ROI result files")
    return all_results


def extract_scale_data(results: Dict[str, Any], scale: float = 10.0) -> Optional[Dict[str, Any]]:
    """Extract data for a specific scale from results.

    Args:
        results: Complete result dictionary from load_roi_results
        scale: Scale in micrometers (10.0, 20.0, or 40.0)

    Returns:
        Scale-specific data dictionary or None if scale not found
    """
    if 'multiscale_results' not in results:
        return None

    scale_key = str(float(scale))
    return results['multiscale_results'].get(scale_key)


def build_superpixel_dataframe(
    results_dict: Dict[str, Dict[str, Any]],
    scale: float = 10.0,
    include_dna: bool = False
) -> pd.DataFrame:
    """Convert results to pandas DataFrame for analysis.

    Args:
        results_dict: Dictionary mapping roi_id to result data (from load_all_rois)
        scale: Scale to extract (10.0, 20.0, or 40.0)
        include_dna: Whether to include DNA markers (130Ba, 131Xe)

    Returns:
        DataFrame with superpixel-level data including:
        - roi, timepoint, mouse, condition (metadata)
        - superpixel_id, x, y (identity & location)
        - cluster (Leiden cluster label)
        - marker intensities (arcsinh-transformed)
    """
    all_rows = []

    for roi_id, results in results_dict.items():
        # Extract scale data
        scale_data = extract_scale_data(results, scale)
        if scale_data is None:
            logger.warning(f"Scale {scale}μm not found for ROI {roi_id}")
            continue

        # Parse metadata from roi_id (biological metadata not in file)
        try:
            parsed = ROIMetadata.from_filename(roi_id)
            metadata = {
                'timepoint': parsed.timepoint,
                'mouse': parsed.mouse,
                'condition': parsed.condition
            }
        except ValueError:
            logger.warning(f"Cannot parse metadata for {roi_id}")
            metadata = {'timepoint': 'Unknown', 'mouse': 'Unknown', 'condition': 'Unknown'}

        # Deserialize arrays
        try:
            coords = deserialize_array(scale_data['spatial_coords'])
            clusters = deserialize_array(scale_data['cluster_labels'])
        except KeyError as e:
            logger.error(f"Missing required array in {roi_id}: {e}")
            continue

        # Extract marker data
        markers = {}
        for marker, arr_dict in scale_data.get('transformed_arrays', {}).items():
            if not include_dna and marker in ['130Ba', '131Xe']:
                continue
            markers[marker] = deserialize_array(arr_dict)

        # Build rows
        n_superpixels = len(coords)
        for i in range(n_superpixels):
            row = {
                'roi': roi_id,
                'timepoint': metadata.get('timepoint', 'Unknown'),
                'mouse': metadata.get('mouse', 'Unknown'),
                'condition': metadata.get('condition', 'Unknown'),
                'superpixel_id': i,
                'x': coords[i, 0],
                'y': coords[i, 1],
                'cluster': int(clusters[i])
            }

            # Add marker values
            for marker, values in markers.items():
                if i < len(values):
                    row[marker] = values[i]
                else:
                    row[marker] = np.nan

            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    logger.info(f"Created DataFrame with {len(df)} superpixels from {df['roi'].nunique()} ROIs")
    return df


def get_cluster_profiles(
    results: Dict[str, Any],
    scale: float = 10.0,
    include_dna: bool = False
) -> pd.DataFrame:
    """Get mean marker expression per cluster for a single ROI.

    Args:
        results: Result dictionary from load_roi_results
        scale: Scale to analyze
        include_dna: Whether to include DNA markers

    Returns:
        DataFrame with clusters as rows and markers as columns
    """
    scale_data = extract_scale_data(results, scale)
    if scale_data is None:
        return pd.DataFrame()

    # Deserialize arrays
    clusters = deserialize_array(scale_data['cluster_labels'])
    markers = {
        name: deserialize_array(arr)
        for name, arr in scale_data['transformed_arrays'].items()
        if include_dna or name not in ['130Ba', '131Xe']
    }

    # Build DataFrame
    df = pd.DataFrame(markers)
    df['cluster'] = clusters

    # Group by cluster
    profiles = df.groupby('cluster').mean()

    return profiles


def compare_timepoints(
    all_results: Dict[str, Dict[str, Any]],
    marker: str,
    scale: float = 10.0,
    aggregation: str = 'mean'
) -> pd.DataFrame:
    """Compare marker expression across timepoints.

    Args:
        all_results: Dictionary from load_all_rois
        marker: Marker to analyze
        scale: Scale to use
        aggregation: 'mean', 'median', 'std', or 'all'

    Returns:
        DataFrame with timepoints as rows and statistics as columns
    """
    timepoint_data = {}

    for roi_id, results in all_results.items():
        # Get metadata
        metadata = results.get('metadata', {})
        if not metadata:
            try:
                parsed = ROIMetadata.from_filename(roi_id)
                tp = parsed.timepoint
            except ValueError:
                continue
        else:
            tp = metadata.get('timepoint', 'Unknown')

        if tp == 'Unknown':
            continue

        # Extract marker data
        scale_data = extract_scale_data(results, scale)
        if scale_data is None:
            continue

        marker_values = deserialize_array(
            scale_data.get('transformed_arrays', {}).get(marker)
        )
        if marker_values is None:
            continue

        if tp not in timepoint_data:
            timepoint_data[tp] = []
        timepoint_data[tp].extend(marker_values)

    # Compute statistics
    stats = {}
    for tp, values in timepoint_data.items():
        values = np.array(values)
        if aggregation == 'mean':
            stats[tp] = {'value': np.mean(values)}
        elif aggregation == 'median':
            stats[tp] = {'value': np.median(values)}
        elif aggregation == 'std':
            stats[tp] = {'value': np.std(values)}
        else:  # 'all'
            stats[tp] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'n_superpixels': len(values)
            }

    return pd.DataFrame(stats).T


def validate_result_file(results: Dict[str, Any]) -> bool:
    """Validate result file structure and invariants.

    Args:
        results: Result dictionary from load_roi_results

    Returns:
        True if valid, raises AssertionError if invalid
    """
    # Check top-level keys
    required_keys = {'multiscale_results', 'metadata', 'roi_id'}
    assert required_keys.issubset(results.keys()), \
        f"Missing keys: {required_keys - results.keys()}"

    # Check scales
    expected_scales = {'10.0', '20.0', '40.0'}
    found_scales = set(results['multiscale_results'].keys())
    assert expected_scales == found_scales, \
        f"Expected scales {expected_scales}, found {found_scales}"

    # Validate each scale
    for scale, scale_data in results['multiscale_results'].items():
        # Deserialize key arrays
        clusters = deserialize_array(scale_data['cluster_labels'])
        coords = deserialize_array(scale_data['spatial_coords'])

        # Check shapes match
        n_superpixels = len(clusters)
        assert len(coords) == n_superpixels, \
            f"Coord length mismatch at scale {scale}: {len(coords)} != {n_superpixels}"

        # Check marker arrays
        for marker, arr_dict in scale_data['transformed_arrays'].items():
            values = deserialize_array(arr_dict)
            assert len(values) == n_superpixels, \
                f"{marker} length mismatch at scale {scale}: {len(values)} != {n_superpixels}"
            assert (values >= 0).all(), \
                f"{marker} has negative values at scale {scale}"

        # Check cluster labels are sequential
        unique_clusters = np.unique(clusters)
        expected_clusters = np.arange(len(unique_clusters))
        assert np.array_equal(unique_clusters, expected_clusters), \
            f"Cluster labels not sequential at scale {scale}"

        # Check spatial coordinates are positive
        assert (coords >= 0).all(), \
            f"Negative spatial coordinates at scale {scale}"

    return True


def get_roi_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get summary statistics for a single ROI.

    Args:
        results: Result dictionary from load_roi_results

    Returns:
        Dictionary with summary metrics
    """
    summary = {
        'roi_id': results['roi_id'],
        'metadata': results.get('metadata', {}),
        'scales': {}
    }

    for scale, scale_data in results['multiscale_results'].items():
        clusters = deserialize_array(scale_data['cluster_labels'])
        coords = deserialize_array(scale_data['spatial_coords'])

        summary['scales'][f'{scale}um'] = {
            'n_superpixels': len(clusters),
            'n_clusters': len(np.unique(clusters)),
            'spatial_coherence': scale_data.get('spatial_coherence', 0.0),
            'optimal_resolution': scale_data.get('stability_analysis', {}).get('optimal_resolution', 0.0),
            'bounds_um': scale_data.get('bounds', [0, 0, 0, 0])
        }

    return summary


# Convenience function for common use case
def quick_load(
    results_dir: str = 'results/roi_results',
    scale: float = 10.0,
    as_dataframe: bool = True
) -> Union[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Quick load with sensible defaults.

    Args:
        results_dir: Directory containing result files
        scale: Scale to extract for DataFrame
        as_dataframe: If True, return DataFrame; if False, return dict

    Returns:
        Either superpixel DataFrame or results dictionary
    """
    all_results = load_all_rois(results_dir)

    if as_dataframe:
        return build_superpixel_dataframe(all_results, scale=scale)
    else:
        return all_results


# Example usage as module-level documentation
__doc__ += """

Example Usage
=============

Basic Loading
-------------
>>> from src.utils.canonical_loader import load_all_rois, build_superpixel_dataframe
>>>
>>> # Load all ROI results
>>> results = load_all_rois('results/roi_results')
>>> print(f"Loaded {len(results)} ROIs")
>>>
>>> # Convert to DataFrame for analysis
>>> df = build_superpixel_dataframe(results, scale=10.0)
>>> print(df.head())

Quick Load
----------
>>> from src.utils.canonical_loader import quick_load
>>>
>>> # One-liner to get DataFrame
>>> df = quick_load(scale=10.0)

Single ROI Analysis
-------------------
>>> from src.utils.canonical_loader import load_roi_results, get_cluster_profiles
>>>
>>> # Load specific ROI
>>> roi_path = 'results/roi_results/roi_IMC_241218_Alun_ROI_D1_M1_01_9_results.json.gz'
>>> results = load_roi_results(roi_path)
>>>
>>> # Get cluster profiles
>>> profiles = get_cluster_profiles(results, scale=10.0)
>>> print(profiles[['CD45', 'CD31', 'CD140b']])

Temporal Comparison
-------------------
>>> from src.utils.canonical_loader import load_all_rois, compare_timepoints
>>>
>>> results = load_all_rois()
>>> temporal = compare_timepoints(results, marker='CD44', scale=10.0, aggregation='all')
>>> print(temporal)

Validation
----------
>>> from src.utils.canonical_loader import load_roi_results, validate_result_file
>>>
>>> results = load_roi_results('results/roi_results/roi_..._results.json.gz')
>>> validate_result_file(results)  # Raises AssertionError if invalid
>>> print("✓ Result file is valid")

See docs/DATA_SCHEMA.md for complete documentation.
"""
