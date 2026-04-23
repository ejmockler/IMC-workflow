"""
Cell Type Annotation Engine

Two annotation modes:
1. Boolean gating: Discrete cell type assignment via positive/negative marker gates.
2. Continuous membership: Multi-label decomposition into lineage scores (non-exclusive),
   within-lineage subtypes, and activation overlays. Superpixels at tissue interfaces
   score on multiple lineages simultaneously.

Both modes are config-driven and experiment-agnostic.
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
    global_percentile = global_threshold_config.get('percentile', 60)

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


def compute_continuous_memberships(
    expression_matrix: np.ndarray,
    marker_names: List[str],
    membership_config: Dict[str, Any],
    threshold_config: Dict[str, Any],
    reference_distribution: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Compute continuous multi-label memberships for each superpixel.

    Decomposes annotation into three independent axes:
    - Lineage: non-exclusive continuous scores (immune, endothelial, stromal)
    - Subtype: within-lineage categorical refinement (neutrophil, M2, myeloid, ...)
    - Activation: independent continuous overlay (CD44, CD140b)

    A superpixel at a tissue interface scores on multiple lineages simultaneously.
    Scores are normalized expression values in [0, 1].

    Args:
        expression_matrix: [n_superpixels, n_markers] arcsinh-transformed expression
        marker_names: List of marker names corresponding to columns
        membership_config: membership_axes config section
        threshold_config: Positivity threshold config (for discrete-gating
            subtype logic — NOT for continuous normalization)
        reference_distribution: REQUIRED. ``{marker: {threshold, scale}}`` dict
            produced by ``src.analysis.sham_reference.build_reference_distribution``.
            The sigmoid for each lineage/subtype/activation marker is centered on
            ``threshold`` (pooled-Sham per-mouse percentile) and its steepness is
            ``sigmoid_steepness / scale`` where ``scale`` is the experiment-wide
            IQR. Separating threshold (center) from scale (steepness) is the
            Gate-6 seam closure: the previous per-ROI IQR-scaled sigmoid
            compressed temporal dynamics because each ROI re-centered on its own
            distribution. No silent fallback is permitted — production callers
            must pass an explicit reference; see ``generate_sham_reference.py``.

    Returns:
        Dictionary with:
        - lineage_scores: Dict[lineage_name → np.ndarray[n_superpixels] float in [0,1]]
        - subtype_labels: np.ndarray[n_superpixels] string (best-matching subtype or 'none')
        - subtype_scores: Dict[subtype_name → np.ndarray[n_superpixels] float in [0,1]]
        - activation_scores: Dict[activation_name → np.ndarray[n_superpixels] float in [0,1]]
        - composite_labels: np.ndarray[n_superpixels] string (derived strict type or 'mixed'/'unassigned')
        - normalization_params: per-marker threshold/scale used (for provenance)
        - thresholds_used: alias of normalization_params for back-compat
    """
    n_superpixels = expression_matrix.shape[0]

    # Validate reference_distribution shape before anything else — missing keys
    # here would produce silently-wrong sigmoid scores with correct-looking
    # config_sha256. Match the hard-validation style already used by
    # composite_label_thresholds below.
    if not isinstance(reference_distribution, dict) or not reference_distribution:
        raise ValueError(
            "compute_continuous_memberships: reference_distribution is required "
            "and must be a non-empty dict. Build via "
            "src.analysis.sham_reference.build_reference_distribution."
        )

    # Collect markers actually consumed by the membership engine
    needed_markers: set = set()
    for lineage_def in membership_config.get('lineages', {}).values():
        if isinstance(lineage_def, dict):
            needed_markers.update(lineage_def.get('markers', []))
    for subtype_def in membership_config.get('subtypes', {}).get('definitions', {}).values():
        needed_markers.update(subtype_def.get('positive', []))
        needed_markers.update(subtype_def.get('negative', []))
    for act_marker in membership_config.get('activation', {}).get('markers', {}).values():
        needed_markers.add(act_marker)

    # Validate the panel covers them
    marker_set = set(marker_names)
    missing_panel = needed_markers - marker_set
    if missing_panel:
        raise ValueError(
            f"compute_continuous_memberships: markers {sorted(missing_panel)} "
            f"referenced in membership_config but missing from panel {marker_names}"
        )

    # Validate the reference covers them
    missing_ref = needed_markers - set(reference_distribution.keys())
    if missing_ref:
        raise ValueError(
            f"compute_continuous_memberships: reference_distribution missing "
            f"markers {sorted(missing_ref)} required for sigmoid normalization"
        )
    for m in needed_markers:
        ref_entry = reference_distribution[m]
        if 'threshold' not in ref_entry or 'scale' not in ref_entry:
            raise ValueError(
                f"compute_continuous_memberships: reference_distribution['{m}'] "
                f"must have both 'threshold' and 'scale' keys; got {ref_entry}"
            )

    # --- Normalize expression to [0, 1] via Sham-reference sigmoid ---
    # threshold (center) from Sham-reference per-mouse percentile;
    # scale (steepness denominator) from experiment-wide IQR. Separating them
    # avoids the Sham-IQR-inflates-compresses-injury pathology.
    sigmoid_steepness = float(membership_config.get('sigmoid_steepness', 10.0))
    norm_params: Dict[str, Dict[str, float]] = {}
    normalized = np.zeros_like(expression_matrix, dtype=float)

    for i, marker in enumerate(marker_names):
        col = expression_matrix[:, i]
        if marker in reference_distribution:
            ref = reference_distribution[marker]
            threshold = float(ref['threshold'])
            scale_denom = float(ref['scale'])
            scale = sigmoid_steepness / max(scale_denom, 1e-6)
            z = np.clip(scale * (col - threshold), -500, 500)
            normalized[:, i] = 1.0 / (1.0 + np.exp(-z))
            norm_params[marker] = {
                'method': 'sham_reference_sigmoid',
                'threshold': threshold,
                'scale_denom': scale_denom,
                'steepness': sigmoid_steepness,
            }
        else:
            # Marker is in the panel but not consumed by memberships — leave
            # at zero (will not be read). Still record for provenance.
            norm_params[marker] = {'method': 'unused_by_memberships'}

    def _marker_index(marker: str) -> int:
        return marker_names.index(marker)

    def _marker_score(marker: str) -> np.ndarray:
        return normalized[:, _marker_index(marker)]

    # --- Axis 1: Lineage scores (positive-only, non-exclusive) ---
    lineage_config = membership_config.get('lineages', {})
    lineage_scores = {}
    for lineage_name, lineage_def in lineage_config.items():
        if lineage_name.startswith('_'):
            continue
        markers = lineage_def.get('markers', [])
        agg = lineage_def.get('aggregation', 'max')
        if not markers:
            lineage_scores[lineage_name] = np.zeros(n_superpixels)
            continue
        marker_vals = np.column_stack([_marker_score(m) for m in markers])
        if agg == 'mean':
            lineage_scores[lineage_name] = marker_vals.mean(axis=1)
        elif agg == 'max':
            lineage_scores[lineage_name] = marker_vals.max(axis=1)
        elif agg == 'min':
            lineage_scores[lineage_name] = marker_vals.min(axis=1)
        else:
            lineage_scores[lineage_name] = marker_vals.max(axis=1)

    # --- Axis 2: Subtype (within-lineage categorical refinement) ---
    subtype_config = membership_config.get('subtypes', {})
    subtype_threshold = subtype_config.get('subtype_threshold', 0.3)
    subtype_defs = subtype_config.get('definitions', {})

    # thresholds_used kept as an alias of norm_params so existing downstream
    # readers (batch_annotate_all_rois.py output metadata) don't break on the
    # key change. norm_params now records Sham-reference threshold + scale.
    thresholds_used = norm_params

    # Compute subtype scores using geometric mean to normalize across marker count.
    # Raw product penalizes subtypes with more markers (each factor < 1 shrinks the
    # product). Geometric mean (product^(1/n)) makes scores comparable regardless of
    # how many markers define each subtype.
    subtype_scores = {}
    for subtype_name, subtype_def in subtype_defs.items():
        parent = subtype_def.get('parent_lineage', '')
        pos_markers = subtype_def.get('positive', [])
        neg_markers = subtype_def.get('negative', [])

        parent_score = lineage_scores.get(parent, np.zeros(n_superpixels))
        eligible = parent_score >= subtype_threshold

        # Collect all factor scores
        factors = []
        for m in pos_markers:
            factors.append(_marker_score(m))
        for m in neg_markers:
            factors.append(1.0 - _marker_score(m))

        if not factors:
            # No positive or negative markers defined — this is a catch-all.
            # Score = parent lineage score (decays with distance from lineage).
            score = parent_score.copy()
        else:
            # Geometric mean: product^(1/n) — normalizes for marker count
            product = np.ones(n_superpixels, dtype=float)
            for f in factors:
                product *= f
            score = np.power(np.maximum(product, 0.0), 1.0 / len(factors))

        score[~eligible] = 0.0
        subtype_scores[subtype_name] = score

    # Assign best subtype per superpixel (vectorized)
    subtype_labels = np.array(['none'] * n_superpixels, dtype=object)
    if subtype_scores:
        score_matrix = np.column_stack(list(subtype_scores.values()))
        subtype_names_list = list(subtype_scores.keys())
        best_idx = score_matrix.argmax(axis=1)
        best_val = score_matrix.max(axis=1)
        mask = best_val > 0
        subtype_labels[mask] = np.array(subtype_names_list)[best_idx[mask]]

    # --- Axis 3: Activation scores (independent overlay) ---
    activation_config = membership_config.get('activation', {})
    activation_markers = activation_config.get('markers', {})
    activation_scores = {}
    for act_name, act_marker in activation_markers.items():
        activation_scores[act_name] = _marker_score(act_marker)

    # --- Derive composite discrete labels for backward compatibility ---
    # Hard-require the three threshold keys; silent fallback to defaults on a
    # typo like 'dominace' would produce unintended composite_labels with
    # correct-looking config hashes.
    if 'composite_label_thresholds' not in membership_config:
        raise KeyError(
            "config.cell_type_annotation.membership_axes.composite_label_thresholds "
            "is required; expected {lineage, activation, dominance}"
        )
    composite_cfg = membership_config['composite_label_thresholds']
    for k in ('lineage', 'activation', 'dominance'):
        if k not in composite_cfg:
            raise KeyError(f"composite_label_thresholds missing required key '{k}'")
    lin_thr = float(composite_cfg['lineage'])
    act_thr = float(composite_cfg['activation'])
    dom_ratio = float(composite_cfg['dominance'])
    if not 0.0 < lin_thr < 1.0:
        raise ValueError(f"composite_label_thresholds.lineage must be in (0,1); got {lin_thr}")
    if not 0.0 < act_thr < 1.0:
        raise ValueError(f"composite_label_thresholds.activation must be in (0,1); got {act_thr}")
    if dom_ratio < 1.0:
        raise ValueError(f"composite_label_thresholds.dominance must be >= 1.0; got {dom_ratio}")
    composite_labels = _derive_composite_labels(
        lineage_scores, subtype_labels, activation_scores,
        dominance_ratio=dom_ratio,
        lineage_threshold=lin_thr,
        activation_threshold=act_thr,
    )

    return {
        'lineage_scores': lineage_scores,
        'subtype_labels': subtype_labels,
        'subtype_scores': subtype_scores,
        'activation_scores': activation_scores,
        'composite_labels': composite_labels,
        'normalization_params': norm_params,
        'thresholds_used': thresholds_used,
    }


def _derive_composite_labels(
    lineage_scores: Dict[str, np.ndarray],
    subtype_labels: np.ndarray,
    activation_scores: Dict[str, np.ndarray],
    dominance_ratio: float = 2.0,
    lineage_threshold: float = 0.3,
    activation_threshold: float = 0.3,
) -> np.ndarray:
    # NOTE: these defaults must stay aligned with the canonical values in
    # config.cell_type_annotation.membership_axes.composite_label_thresholds.
    # Production code paths hard-require the config keys; the defaults exist
    # only so direct unit-test calls can override them explicitly.
    """
    Derive discrete composite labels from continuous membership axes.

    Rules:
    - 'unassigned': no lineage scores above lineage_threshold
    - 'mixed': multiple lineages above threshold with no single dominant
      (dominant = highest score >= dominance_ratio * second highest)
    - Otherwise: lineage + subtype + activation → composite name
    """
    n = len(subtype_labels)
    labels = np.array(['unassigned'] * n, dtype=object)

    if not lineage_scores:
        return labels

    lineage_names = list(lineage_scores.keys())
    score_matrix = np.column_stack([lineage_scores[ln] for ln in lineage_names])

    # Sort scores descending per superpixel
    sorted_scores = np.sort(score_matrix, axis=1)[:, ::-1]
    top_score = sorted_scores[:, 0]
    second_score = sorted_scores[:, 1] if score_matrix.shape[1] > 1 else np.zeros(n)
    dominant_idx = score_matrix.argmax(axis=1)

    for i in range(n):
        if top_score[i] < lineage_threshold:
            # No lineage signal
            labels[i] = 'unassigned'
        elif second_score[i] >= lineage_threshold and top_score[i] < dominance_ratio * second_score[i]:
            # Multiple lineages, no dominant
            labels[i] = 'mixed'
        else:
            # Single dominant lineage
            lineage = lineage_names[dominant_idx[i]]
            subtype = subtype_labels[i]

            # Build composite name
            active = []
            for act_name, act_score in activation_scores.items():
                if act_score[i] >= activation_threshold:
                    active.append(act_name)

            if subtype != 'none':
                base = subtype
            else:
                base = lineage

            if active:
                labels[i] = f"activated_{base}_{'_'.join(sorted(active))}"
            else:
                labels[i] = base

    return labels


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


def annotate_roi_from_results(
    roi_results: Dict[str, Any],
    config: 'Config',
    scale: str = '10.0',
    reference_distribution: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Annotate a single ROI's superpixels with cell types from existing results.

    This is the main entry point for cell type annotation, consuming the output
    from multiscale_analysis and applying gating strategies from config.

    Args:
        roi_results: Dictionary containing multiscale analysis results
        config: Configuration object with cell type annotation settings
        scale: Which spatial scale to annotate (default: '10.0' µm)
        reference_distribution: ``{marker: {threshold, scale}}`` dict required
            when ``cell_type_annotation.membership_axes`` is configured. Built
            once per pipeline run by ``generate_sham_reference.py`` and passed
            to every ROI so all ROIs share the same sigmoid centering.
            Discrete-gating annotation is unaffected by this argument.

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

    # Get marker names (protein channels)
    marker_names = config.channels.get('protein_channels', [])

    # Build the arcsinh expression matrix for gating + continuous memberships.
    # The ``features`` matrix in roi_results is the raw coabundance-augmented
    # feature set used for clustering (30 cols: 9 raw markers + 21 derived);
    # its leading columns are RAW ion counts, not arcsinh. Using it for
    # continuous memberships would apply a Sham-reference built on arcsinh
    # values to raw-count inputs — the sigmoid would saturate for every row.
    # Use ``transformed_arrays`` (arcsinh per marker) so the engine and the
    # reference agree on scale.
    transformed = scale_results.get('transformed_arrays')
    if not transformed:
        raise ValueError(
            f"No transformed_arrays in scale {scale} results; expected per-marker "
            f"arcsinh arrays produced by multiscale_analysis"
        )
    missing = [m for m in marker_names if m not in transformed]
    if missing:
        raise ValueError(
            f"transformed_arrays missing protein channels {missing}; "
            f"check that multiscale_analysis ran with the current config"
        )
    expression_matrix = np.column_stack(
        [np.asarray(transformed[m], dtype=float) for m in marker_names]
    )

    # Cluster-annotation-by-enrichment reads the original clustering feature
    # matrix (raw coabundance). Preserved separately to avoid a silent
    # cluster-annotation scale shift: fold-change (cluster_mean/global_mean)
    # is NOT invariant under arcsinh, so switching the cluster path to
    # arcsinh would silently flip cluster-to-celltype mappings against a
    # 1.5-enrichment threshold calibrated for raw counts.
    cluster_feature_matrix = scale_results.get('features')

    # Extract cluster labels (for cluster annotation)
    cluster_labels = scale_results.get('cluster_labels')

    # Get cell type definitions
    cell_type_definitions = annotation_config.get('cell_types', {})
    threshold_config = annotation_config.get('positivity_threshold', {})

    # Perform main annotation (on arcsinh expression — percentile gating is
    # scale-invariant so this matches the legacy boolean assignments exactly).
    cell_type_labels, confidence_scores, annotation_metadata = annotate_cell_types(
        expression_matrix,
        marker_names,
        cell_type_definitions,
        threshold_config
    )

    results = {
        'cell_type_labels': cell_type_labels,
        'confidence_scores': confidence_scores,
        'annotation_metadata': annotation_metadata
    }

    # Continuous multi-label memberships (if configured). Uses the arcsinh
    # expression matrix + Sham reference.
    membership_config = annotation_config.get('membership_axes', {})
    if membership_config:
        if reference_distribution is None:
            raise ValueError(
                "annotate_roi_from_results: cell_type_annotation.membership_axes "
                "is configured but reference_distribution was not provided. "
                "Build one via src.analysis.sham_reference.build_reference_distribution "
                "(or load results/biological_analysis/sham_reference_10.0um.json) "
                "and pass it here. Silent fallback would reintroduce the per-ROI "
                "sigmoid confound."
            )
        memberships = compute_continuous_memberships(
            expression_matrix,
            marker_names,
            membership_config,
            threshold_config,
            reference_distribution=reference_distribution,
        )
        results['memberships'] = memberships

    # Cluster annotation (if clustering was performed). Uses the ORIGINAL raw
    # coabundance feature matrix — fold-change enrichment isn't invariant
    # under arcsinh, so migrating this path would silently shift the
    # 1.5-enrichment cluster-to-celltype mapping.
    if cluster_labels is not None and cluster_feature_matrix is not None:
        cluster_annotation_config = config.raw.get('biological_analysis', {}).get('cluster_annotation', {})
        if cluster_annotation_config.get('enabled', True):
            cluster_annotations = annotate_clusters_by_enrichment(
                cluster_labels,
                cluster_feature_matrix,
                marker_names,
                cell_type_definitions,
                enrichment_threshold=cluster_annotation_config.get('enrichment_threshold', 1.5),
                min_marker_fraction=cluster_annotation_config.get('min_marker_fraction', 0.3)
            )
            results['cluster_annotations'] = cluster_annotations

    return results
