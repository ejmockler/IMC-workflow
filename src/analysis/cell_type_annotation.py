"""
Cell Type Annotation Engine

Boolean gating strategy for assigning cell types to superpixels based on
marker expression profiles. Supports hierarchical annotation and probabilistic
assignment when gating criteria are ambiguous.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from pathlib import Path
import json

if TYPE_CHECKING:
    from ..config import Config


def compute_marker_positivity(
    expression_matrix: np.ndarray,
    marker_names: List[str],
    thresholds: Dict[str, Dict[str, Any]],
    global_threshold_config: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Determine marker positivity for each superpixel using percentile thresholding.

    Args:
        expression_matrix: [n_superpixels, n_markers] arcsinh-transformed expression
        marker_names: List of marker names corresponding to columns
        thresholds: Per-marker threshold overrides from config
        global_threshold_config: Global threshold configuration

    Returns:
        Tuple of:
        - positivity_matrix: [n_superpixels, n_markers] boolean array
        - thresholds_used: Dictionary mapping marker name to threshold value
    """
    n_superpixels, n_markers = expression_matrix.shape
    positivity_matrix = np.zeros((n_superpixels, n_markers), dtype=bool)
    thresholds_used = {}

    # Get global threshold settings
    method = global_threshold_config.get('method', 'percentile')
    global_percentile = global_threshold_config.get('percentile', 75)

    for i, marker in enumerate(marker_names):
        marker_values = expression_matrix[:, i]

        # Check for per-marker override
        if marker in thresholds:
            override = thresholds[marker]
            threshold_method = override.get('method', method)
            threshold_percentile = override.get('percentile', global_percentile)
        else:
            threshold_method = method
            threshold_percentile = global_percentile

        # Compute threshold
        if threshold_method == 'percentile':
            threshold = np.percentile(marker_values, threshold_percentile)
        elif threshold_method == 'absolute':
            threshold = thresholds.get(marker, {}).get('value', 0.5)
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")

        # Apply threshold
        positivity_matrix[:, i] = marker_values > threshold
        thresholds_used[marker] = float(threshold)

    return positivity_matrix, thresholds_used


def apply_boolean_gate(
    positivity_matrix: np.ndarray,
    marker_names: List[str],
    positive_markers: List[str],
    negative_markers: List[str]
) -> np.ndarray:
    """
    Apply boolean gating logic to identify cells matching a phenotype.

    Implements: (positive_markers ALL True) AND (negative_markers ALL False)

    Args:
        positivity_matrix: [n_superpixels, n_markers] boolean array
        marker_names: List of marker names corresponding to columns
        positive_markers: Markers that must be positive
        negative_markers: Markers that must be negative

    Returns:
        Boolean array [n_superpixels] indicating which cells match the gate
    """
    n_superpixels = positivity_matrix.shape[0]

    # Initialize mask as all True
    gate_mask = np.ones(n_superpixels, dtype=bool)

    # Apply positive marker requirements (AND logic)
    for marker in positive_markers:
        if marker not in marker_names:
            raise ValueError(f"Positive marker '{marker}' not found in expression matrix")
        marker_idx = marker_names.index(marker)
        gate_mask &= positivity_matrix[:, marker_idx]

    # Apply negative marker requirements (AND NOT logic)
    for marker in negative_markers:
        if marker not in marker_names:
            raise ValueError(f"Negative marker '{marker}' not found in expression matrix")
        marker_idx = marker_names.index(marker)
        gate_mask &= ~positivity_matrix[:, marker_idx]

    return gate_mask


def annotate_cell_types(
    expression_matrix: np.ndarray,
    marker_names: List[str],
    cell_type_definitions: Dict[str, Dict[str, Any]],
    threshold_config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Annotate superpixels with cell types using boolean gating strategy.

    Args:
        expression_matrix: [n_superpixels, n_markers] arcsinh-transformed expression
        marker_names: List of marker names
        cell_type_definitions: Cell type gating definitions from config
        threshold_config: Positivity threshold configuration

    Returns:
        Tuple of:
        - cell_type_labels: String array [n_superpixels] with cell type names
        - confidence_scores: Float array [n_superpixels] with annotation confidence
        - annotation_metadata: Dictionary with QC metrics and threshold info
    """
    n_superpixels = expression_matrix.shape[0]

    # Compute marker positivity
    positivity_matrix, thresholds_used = compute_marker_positivity(
        expression_matrix,
        marker_names,
        threshold_config.get('per_marker_override', {}),
        threshold_config
    )

    # Initialize outputs
    cell_type_labels = np.array(['unassigned'] * n_superpixels, dtype=object)
    confidence_scores = np.zeros(n_superpixels, dtype=float)
    assignment_counts = {cell_type: 0 for cell_type in cell_type_definitions.keys()}
    assignment_counts['unassigned'] = n_superpixels

    # Track multi-assignments for ambiguity detection
    assignment_matrix = np.zeros((n_superpixels, len(cell_type_definitions)), dtype=bool)

    # Apply each cell type gate
    for cell_type_idx, (cell_type_name, cell_type_def) in enumerate(cell_type_definitions.items()):
        positive_markers = cell_type_def.get('positive_markers', [])
        negative_markers = cell_type_def.get('negative_markers', [])

        # Apply gate
        gate_mask = apply_boolean_gate(
            positivity_matrix,
            marker_names,
            positive_markers,
            negative_markers
        )

        assignment_matrix[:, cell_type_idx] = gate_mask

    # Resolve assignments (priority: first defined cell type wins if multiple match)
    for cell_type_idx, (cell_type_name, _) in enumerate(cell_type_definitions.items()):
        gate_mask = assignment_matrix[:, cell_type_idx]

        # Only assign if not already assigned (priority order)
        unassigned_mask = cell_type_labels == 'unassigned'
        final_assignment = gate_mask & unassigned_mask

        # Apply assignments
        cell_type_labels[final_assignment] = cell_type_name

        # Confidence: 1.0 for single assignment, lower for ambiguous
        n_assignments = assignment_matrix.sum(axis=1)
        confidence_scores[final_assignment] = 1.0 / n_assignments[final_assignment]

        # Track counts
        assignment_counts[cell_type_name] = int(final_assignment.sum())

    # Update unassigned count
    assignment_counts['unassigned'] = int((cell_type_labels == 'unassigned').sum())

    # Compute ambiguity metrics
    n_multi_assigned = int((assignment_matrix.sum(axis=1) > 1).sum())
    ambiguity_rate = n_multi_assigned / n_superpixels if n_superpixels > 0 else 0.0

    # Build metadata
    annotation_metadata = {
        'n_superpixels': n_superpixels,
        'thresholds_used': thresholds_used,
        'assignment_counts': assignment_counts,
        'ambiguity_rate': ambiguity_rate,
        'n_multi_assigned': n_multi_assigned,
        'unassigned_fraction': assignment_counts['unassigned'] / n_superpixels if n_superpixels > 0 else 0.0
    }

    return cell_type_labels, confidence_scores, annotation_metadata


def hierarchical_annotation(
    expression_matrix: np.ndarray,
    marker_names: List[str],
    hierarchical_config: Dict[str, Any],
    threshold_config: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Perform hierarchical cell type annotation (lineage -> activation -> cell type).

    Args:
        expression_matrix: [n_superpixels, n_markers] arcsinh-transformed expression
        marker_names: List of marker names
        hierarchical_config: Hierarchical annotation configuration
        threshold_config: Positivity threshold configuration

    Returns:
        Dictionary with hierarchical labels at each level
    """
    n_superpixels = expression_matrix.shape[0]

    # Compute marker positivity
    positivity_matrix, _ = compute_marker_positivity(
        expression_matrix,
        marker_names,
        threshold_config.get('per_marker_override', {}),
        threshold_config
    )

    results = {}

    # Level 1: Lineage classification
    if 'level_1_lineage' in hierarchical_config:
        lineage_labels = np.array(['unknown'] * n_superpixels, dtype=object)

        for lineage_name, lineage_def in hierarchical_config['level_1_lineage'].items():
            gate_mask = apply_boolean_gate(
                positivity_matrix,
                marker_names,
                lineage_def.get('positive', []),
                lineage_def.get('negative', [])
            )
            # Priority: first match wins
            unassigned = lineage_labels == 'unknown'
            lineage_labels[gate_mask & unassigned] = lineage_name

        results['lineage'] = lineage_labels

    # Level 2: Activation state
    if 'level_2_activation' in hierarchical_config:
        activation_labels = np.array(['unknown'] * n_superpixels, dtype=object)

        for activation_state, activation_def in hierarchical_config['level_2_activation'].items():
            if 'any_positive' in activation_def:
                # Any marker positive
                any_positive_mask = np.zeros(n_superpixels, dtype=bool)
                for marker in activation_def['any_positive']:
                    if marker in marker_names:
                        marker_idx = marker_names.index(marker)
                        any_positive_mask |= positivity_matrix[:, marker_idx]

                unassigned = activation_labels == 'unknown'
                activation_labels[any_positive_mask & unassigned] = activation_state

            elif 'all_negative' in activation_def:
                # All markers negative
                all_negative_mask = np.ones(n_superpixels, dtype=bool)
                for marker in activation_def['all_negative']:
                    if marker in marker_names:
                        marker_idx = marker_names.index(marker)
                        all_negative_mask &= ~positivity_matrix[:, marker_idx]

                unassigned = activation_labels == 'unknown'
                activation_labels[all_negative_mask & unassigned] = activation_state

        results['activation'] = activation_labels

    return results


def annotate_clusters_by_enrichment(
    cluster_labels: np.ndarray,
    expression_matrix: np.ndarray,
    marker_names: List[str],
    cell_type_definitions: Dict[str, Dict[str, Any]],
    enrichment_threshold: float = 1.5,
    min_marker_fraction: float = 0.3
) -> Dict[int, Dict[str, Any]]:
    """
    Annotate unsupervised clusters with cell types based on marker enrichment.

    Enables comparison of clustering-based vs gating-based cell type discovery.

    Args:
        cluster_labels: [n_superpixels] cluster assignments
        expression_matrix: [n_superpixels, n_markers] expression values
        marker_names: List of marker names
        cell_type_definitions: Cell type definitions with positive/negative markers
        enrichment_threshold: Minimum fold-enrichment for marker to be "high"
        min_marker_fraction: Minimum fraction of defining markers that must be high

    Returns:
        Dictionary mapping cluster_id to annotation info
    """
    unique_clusters = np.unique(cluster_labels)
    cluster_annotations = {}

    # Compute global marker means for enrichment calculation
    global_means = expression_matrix.mean(axis=0)

    for cluster_id in unique_clusters:
        if cluster_id < 0:  # Skip noise cluster
            continue

        cluster_mask = cluster_labels == cluster_id
        cluster_expression = expression_matrix[cluster_mask]
        cluster_means = cluster_expression.mean(axis=0)

        # Compute fold enrichment for each marker
        enrichment = cluster_means / (global_means + 1e-10)  # Avoid division by zero

        # Match to cell types
        best_match = None
        best_score = 0.0

        for cell_type_name, cell_type_def in cell_type_definitions.items():
            positive_markers = cell_type_def.get('positive_markers', [])
            negative_markers = cell_type_def.get('negative_markers', [])

            # Score: fraction of positive markers enriched + fraction of negative markers depleted
            n_positive_enriched = 0
            n_negative_depleted = 0

            for marker in positive_markers:
                if marker in marker_names:
                    marker_idx = marker_names.index(marker)
                    if enrichment[marker_idx] >= enrichment_threshold:
                        n_positive_enriched += 1

            for marker in negative_markers:
                if marker in marker_names:
                    marker_idx = marker_names.index(marker)
                    if enrichment[marker_idx] < 1.0 / enrichment_threshold:
                        n_negative_depleted += 1

            # Compute match score
            total_markers = len(positive_markers) + len(negative_markers)
            if total_markers > 0:
                score = (n_positive_enriched + n_negative_depleted) / total_markers

                if score > best_score and score >= min_marker_fraction:
                    best_score = score
                    best_match = cell_type_name

        # Build annotation
        cluster_annotations[int(cluster_id)] = {
            'cell_type': best_match if best_match else 'unknown',
            'confidence': float(best_score),
            'n_cells': int(cluster_mask.sum()),
            'marker_enrichment': {
                marker_names[i]: float(enrichment[i])
                for i in range(len(marker_names))
            }
        }

    return cluster_annotations


def save_cell_type_annotations(
    roi_id: str,
    cell_type_labels: np.ndarray,
    confidence_scores: np.ndarray,
    superpixel_ids: np.ndarray,
    annotation_metadata: Dict[str, Any],
    output_dir: Path
) -> Path:
    """
    Save cell type annotations to disk in Parquet format.

    Args:
        roi_id: ROI identifier
        cell_type_labels: [n_superpixels] cell type assignments
        confidence_scores: [n_superpixels] confidence scores
        superpixel_ids: [n_superpixels] superpixel identifiers
        annotation_metadata: Annotation QC metadata
        output_dir: Output directory path

    Returns:
        Path to saved Parquet file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame({
        'roi_id': roi_id,
        'superpixel_id': superpixel_ids,
        'cell_type': cell_type_labels,
        'confidence': confidence_scores
    })

    # Save to Parquet
    output_path = output_dir / f"{roi_id}_cell_types.parquet"
    df.to_parquet(output_path, compression='snappy', index=False)

    # Save metadata separately
    metadata_path = output_dir / f"{roi_id}_annotation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(annotation_metadata, f, indent=2)

    return output_path


def annotate_roi_from_results(
    roi_results: Dict[str, Any],
    config: 'Config',
    scale: str = '10.0'
) -> Dict[str, Any]:
    """
    Annotate a single ROI's superpixels with cell types from existing results.

    This is the main entry point for cell type annotation, consuming the output
    from multiscale_analysis and applying gating strategies from config.

    Args:
        roi_results: Dictionary containing multiscale analysis results
        config: Configuration object with cell type annotation settings
        scale: Which spatial scale to annotate (default: '10.0' µm)

    Returns:
        Dictionary containing:
        - cell_type_labels: Cell type assignments
        - confidence_scores: Annotation confidence
        - hierarchical_labels: Hierarchical annotations (if enabled)
        - cluster_annotations: Cluster-to-celltype mapping
        - annotation_metadata: QC metrics
    """
    # Extract configuration
    annotation_config = config.raw.get('cell_type_annotation', {})

    if not annotation_config.get('enabled', True):
        return {'status': 'disabled'}

    # Extract multiscale results for specified scale
    multiscale_results = roi_results.get('multiscale_results', {})
    if scale not in multiscale_results:
        raise ValueError(f"Scale {scale} not found in multiscale results")

    scale_results = multiscale_results[scale]

    # Extract features (arcsinh-transformed expression matrix)
    features = scale_results.get('features')
    if features is None:
        raise ValueError("No features found in scale results")

    # Get marker names (protein channels)
    marker_names = config.channels.get('protein_channels', [])

    # Extract cluster labels (for cluster annotation)
    cluster_labels = scale_results.get('cluster_labels')

    # Get cell type definitions
    cell_type_definitions = annotation_config.get('cell_types', {})
    threshold_config = annotation_config.get('positivity_threshold', {})

    # Perform main annotation
    cell_type_labels, confidence_scores, annotation_metadata = annotate_cell_types(
        features,
        marker_names,
        cell_type_definitions,
        threshold_config
    )

    results = {
        'cell_type_labels': cell_type_labels,
        'confidence_scores': confidence_scores,
        'annotation_metadata': annotation_metadata
    }

    # Hierarchical annotation (if configured)
    hierarchical_config = annotation_config.get('hierarchical_annotation', {})
    if hierarchical_config:
        hierarchical_labels = hierarchical_annotation(
            features,
            marker_names,
            hierarchical_config,
            threshold_config
        )
        results['hierarchical_labels'] = hierarchical_labels

    # Cluster annotation (if clustering was performed)
    if cluster_labels is not None:
        cluster_annotation_config = config.raw.get('biological_analysis', {}).get('cluster_annotation', {})
        if cluster_annotation_config.get('enabled', True):
            cluster_annotations = annotate_clusters_by_enrichment(
                cluster_labels,
                features,
                marker_names,
                cell_type_definitions,
                enrichment_threshold=cluster_annotation_config.get('enrichment_threshold', 1.5),
                min_marker_fraction=cluster_annotation_config.get('min_marker_fraction', 0.3)
            )
            results['cluster_annotations'] = cluster_annotations

    return results


def batch_annotate_rois(
    roi_results_dict: Dict[str, Dict[str, Any]],
    config: 'Config',
    scale: str = '10.0',
    save_outputs: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Annotate multiple ROIs in batch.

    Args:
        roi_results_dict: Dictionary mapping ROI ID to results
        config: Configuration object
        scale: Spatial scale to annotate
        save_outputs: Whether to save annotations to disk

    Returns:
        Dictionary mapping ROI ID to annotation results
    """
    all_annotations = {}

    output_dir = Path(config.output_dir) / 'biological_analysis' / 'cell_type_annotations'

    for roi_id, roi_results in roi_results_dict.items():
        try:
            annotations = annotate_roi_from_results(roi_results, config, scale)
            all_annotations[roi_id] = annotations

            # Save if requested
            if save_outputs and 'cell_type_labels' in annotations:
                # Generate superpixel IDs
                n_superpixels = len(annotations['cell_type_labels'])
                superpixel_ids = np.arange(n_superpixels)

                save_cell_type_annotations(
                    roi_id,
                    annotations['cell_type_labels'],
                    annotations['confidence_scores'],
                    superpixel_ids,
                    annotations['annotation_metadata'],
                    output_dir
                )

        except Exception as e:
            print(f"Warning: Failed to annotate ROI {roi_id}: {e}")
            all_annotations[roi_id] = {'status': 'failed', 'error': str(e)}

    # Save summary
    if save_outputs:
        summary = {
            'n_rois_annotated': len([a for a in all_annotations.values() if 'cell_type_labels' in a]),
            'n_rois_failed': len([a for a in all_annotations.values() if a.get('status') == 'failed']),
            'scale_um': float(scale),
            'cell_type_definitions': config.raw.get('cell_type_annotation', {}).get('cell_types', {})
        }

        summary_path = output_dir / 'annotation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    return all_annotations
