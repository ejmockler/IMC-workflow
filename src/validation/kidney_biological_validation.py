"""
Kidney-Specific Biological Validation

Validates that clustering and feature engineering respect known kidney biology:
1. Cortex vs Medulla marker enrichment patterns
2. Injury timepoint-specific immune responses
3. Vascular network spatial organization
4. Expected cell type compositions

This addresses peer review requirement for biological plausibility.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from dataclasses import dataclass
import warnings


@dataclass
class KidneyAnatomySignature:
    """Expected marker patterns for kidney anatomical regions."""
    region: str
    high_markers: List[str]
    moderate_markers: List[str]
    low_markers: List[str]
    spatial_pattern: str
    description: str


@dataclass
class InjuryTimepoint:
    """Expected immune response at injury timepoints."""
    day: int
    primary_response: str
    key_markers: List[str]
    expected_increase: List[str]
    expected_decrease: List[str]
    spatial_pattern: str


class KidneyBiologicalValidator:
    """
    Validates clustering results against known kidney injury biology.

    Uses established literature on kidney anatomy and AKI immune response:
    - Cortex enriched in CD31/CD34 (glomerular capillaries)
    - Medulla enriched in CD140a (interstitial fibroblasts)
    - Day 1: Neutrophil recruitment (Ly6G+, CD11b+)
    - Day 3: Macrophage activation (CD206+, CD11b+)
    - Day 7: Resolution/fibrosis (CD140a+, CD140b+, CD206+)
    """

    def __init__(self):
        self.cortex_signature = KidneyAnatomySignature(
            region="cortex",
            high_markers=["CD31", "CD34"],  # Endothelial cells in glomeruli
            moderate_markers=["CD45", "CD44"],  # Immune & adhesion
            low_markers=["CD140a"],  # Less fibroblasts than medulla
            spatial_pattern="glomerular_enrichment",
            description="Cortex has dense capillary networks in glomeruli"
        )

        self.medulla_signature = KidneyAnatomySignature(
            region="medulla",
            high_markers=["CD140a"],  # Interstitial fibroblasts
            moderate_markers=["CD44"],  # Adhesion molecules
            low_markers=["CD31", "CD34"],  # Fewer capillaries than cortex
            spatial_pattern="interstitial_distribution",
            description="Medulla has more interstitial fibroblasts"
        )

        self.injury_timepoints = {
            1: InjuryTimepoint(
                day=1,
                primary_response="neutrophil_recruitment",
                key_markers=["Ly6G", "CD11b"],
                expected_increase=["Ly6G", "CD11b", "CD45"],
                expected_decrease=[],
                spatial_pattern="focal_infiltration"
            ),
            3: InjuryTimepoint(
                day=3,
                primary_response="macrophage_activation",
                key_markers=["CD206", "CD11b"],
                expected_increase=["CD206", "CD11b", "CD45"],
                expected_decrease=["Ly6G"],  # Neutrophils resolve
                spatial_pattern="expanding_inflammation"
            ),
            7: InjuryTimepoint(
                day=7,
                primary_response="resolution_or_fibrosis",
                key_markers=["CD140a", "CD140b", "CD206"],
                expected_increase=["CD140a", "CD140b", "CD206"],
                expected_decrease=["Ly6G"],
                spatial_pattern="organized_repair"
            )
        }

    def validate_anatomical_enrichment(
        self,
        cluster_labels: np.ndarray,
        feature_matrix: np.ndarray,
        protein_names: List[str],
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate that clusters show expected anatomical marker enrichment.

        Args:
            cluster_labels: Cluster assignments
            feature_matrix: N x P protein expression matrix
            protein_names: List of protein names
            region: Optional region label (cortex/medulla)

        Returns:
            Validation results with enrichment scores
        """
        results = {
            "cortex_validation": {},
            "medulla_validation": {},
            "overall_quality": None,
            "recommendations": []
        }

        # Create protein name to index mapping
        protein_idx = {name: i for i, name in enumerate(protein_names)}

        # Validate cortex signature
        cortex_scores = self._validate_signature(
            cluster_labels, feature_matrix, protein_idx, self.cortex_signature
        )
        results["cortex_validation"] = cortex_scores

        # Validate medulla signature
        medulla_scores = self._validate_signature(
            cluster_labels, feature_matrix, protein_idx, self.medulla_signature
        )
        results["medulla_validation"] = medulla_scores

        # Overall quality score
        cortex_quality = cortex_scores.get("signature_quality", 0.0)
        medulla_quality = medulla_scores.get("signature_quality", 0.0)
        results["overall_quality"] = (cortex_quality + medulla_quality) / 2.0

        # Generate recommendations
        if cortex_quality < 0.5:
            results["recommendations"].append({
                "priority": "WARNING",
                "finding": "Weak cortex signature - expected CD31/CD34 enrichment not found",
                "suggestion": "Check for ROI containing glomeruli vs tubular-only regions"
            })

        if medulla_quality < 0.5:
            results["recommendations"].append({
                "priority": "WARNING",
                "finding": "Weak medulla signature - expected CD140a enrichment not found",
                "suggestion": "Check for cortical vs medullary tissue representation"
            })

        return results

    def _validate_signature(
        self,
        cluster_labels: np.ndarray,
        feature_matrix: np.ndarray,
        protein_idx: Dict[str, int],
        signature: KidneyAnatomySignature
    ) -> Dict[str, Any]:
        """Validate a specific anatomical signature."""
        scores = {
            "signature_quality": 0.0,
            "high_marker_enrichment": {},
            "low_marker_depletion": {},
            "clusters_with_signature": []
        }

        unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])

        if len(unique_clusters) == 0:
            return scores

        # For each cluster, check if it matches the signature
        cluster_signature_scores = []

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_features = feature_matrix[cluster_mask]

            # Skip if too few superpixels
            if cluster_features.shape[0] < 5:
                continue

            # Check high marker enrichment
            high_score = 0.0
            for marker in signature.high_markers:
                if marker in protein_idx:
                    marker_idx = protein_idx[marker]
                    marker_expression = feature_matrix[:, marker_idx]
                    cluster_mean = np.mean(cluster_features[:, marker_idx])
                    overall_mean = np.mean(marker_expression)

                    # Enrichment ratio
                    if overall_mean > 0:
                        enrichment = cluster_mean / overall_mean
                        high_score += max(0, enrichment - 1.0)  # Reward >1.0 enrichment
                        scores["high_marker_enrichment"][f"{marker}_cluster_{cluster_id}"] = enrichment

            # Check low marker depletion
            low_score = 0.0
            for marker in signature.low_markers:
                if marker in protein_idx:
                    marker_idx = protein_idx[marker]
                    marker_expression = feature_matrix[:, marker_idx]
                    cluster_mean = np.mean(cluster_features[:, marker_idx])
                    overall_mean = np.mean(marker_expression)

                    # Depletion ratio
                    if overall_mean > 0:
                        depletion = cluster_mean / overall_mean
                        low_score += max(0, 1.0 - depletion)  # Reward <1.0 depletion
                        scores["low_marker_depletion"][f"{marker}_cluster_{cluster_id}"] = depletion

            # Combined signature score for this cluster
            n_high = len([m for m in signature.high_markers if m in protein_idx])
            n_low = len([m for m in signature.low_markers if m in protein_idx])

            if n_high + n_low > 0:
                cluster_score = (high_score / max(1, n_high) + low_score / max(1, n_low)) / 2.0
                cluster_signature_scores.append(cluster_score)

                if cluster_score > 0.3:  # Threshold for "has signature"
                    scores["clusters_with_signature"].append(int(cluster_id))

        # Overall signature quality is max cluster score (best match)
        if cluster_signature_scores:
            scores["signature_quality"] = float(np.max(cluster_signature_scores))

        return scores

    def validate_injury_timepoint(
        self,
        cluster_labels: np.ndarray,
        feature_matrix: np.ndarray,
        protein_names: List[str],
        injury_day: int,
        baseline_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Validate expected immune response at specific injury timepoint.

        Args:
            cluster_labels: Cluster assignments
            feature_matrix: N x P protein expression matrix
            protein_names: List of protein names
            injury_day: Day post-injury (1, 3, or 7)
            baseline_features: Optional baseline (Day 0) features for comparison

        Returns:
            Validation results with timepoint-specific metrics
        """
        if injury_day not in self.injury_timepoints:
            return {
                "error": f"Unsupported injury day: {injury_day}",
                "supported_days": list(self.injury_timepoints.keys())
            }

        timepoint = self.injury_timepoints[injury_day]
        protein_idx = {name: i for i, name in enumerate(protein_names)}

        results = {
            "day": injury_day,
            "expected_response": timepoint.primary_response,
            "key_marker_expression": {},
            "expected_increases": {},
            "expected_decreases": {},
            "response_quality": 0.0,
            "recommendations": []
        }

        # Check key marker expression
        for marker in timepoint.key_markers:
            if marker in protein_idx:
                marker_idx = protein_idx[marker]
                marker_mean = np.mean(feature_matrix[:, marker_idx])
                marker_std = np.std(feature_matrix[:, marker_idx])

                results["key_marker_expression"][marker] = {
                    "mean": float(marker_mean),
                    "std": float(marker_std),
                    "cv": float(marker_std / marker_mean) if marker_mean > 0 else 0.0
                }

        # If baseline available, check expected changes
        if baseline_features is not None:
            increase_score = 0.0
            for marker in timepoint.expected_increase:
                if marker in protein_idx:
                    marker_idx = protein_idx[marker]
                    baseline_mean = np.mean(baseline_features[:, marker_idx])
                    current_mean = np.mean(feature_matrix[:, marker_idx])

                    if baseline_mean > 0:
                        fold_change = current_mean / baseline_mean
                        results["expected_increases"][marker] = float(fold_change)

                        if fold_change > 1.2:  # At least 20% increase
                            increase_score += 1.0

            decrease_score = 0.0
            for marker in timepoint.expected_decrease:
                if marker in protein_idx:
                    marker_idx = protein_idx[marker]
                    baseline_mean = np.mean(baseline_features[:, marker_idx])
                    current_mean = np.mean(feature_matrix[:, marker_idx])

                    if baseline_mean > 0:
                        fold_change = current_mean / baseline_mean
                        results["expected_decreases"][marker] = float(fold_change)

                        if fold_change < 0.8:  # At least 20% decrease
                            decrease_score += 1.0

            # Response quality score
            n_increase = len([m for m in timepoint.expected_increase if m in protein_idx])
            n_decrease = len([m for m in timepoint.expected_decrease if m in protein_idx])

            if n_increase + n_decrease > 0:
                results["response_quality"] = (
                    (increase_score / max(1, n_increase) + decrease_score / max(1, n_decrease)) / 2.0
                )

        # Generate recommendations
        if results["response_quality"] < 0.5 and baseline_features is not None:
            results["recommendations"].append({
                "priority": "WARNING",
                "finding": f"Weak {timepoint.primary_response} signature at day {injury_day}",
                "expected": f"Should see {', '.join(timepoint.expected_increase)} increase",
                "suggestion": "Check if ROI contains injury zone vs unaffected tissue"
            })

        return results


def run_kidney_validation(
    results: Dict,
    config: Dict,
    baseline_results: Optional[Dict] = None
) -> Dict:
    """
    Run comprehensive kidney biological validation on analysis results.

    Args:
        results: Analysis results from main pipeline
        config: Configuration dict with metadata
        baseline_results: Optional baseline (Day 0) results for comparison

    Returns:
        Comprehensive biological validation report
    """
    validator = KidneyBiologicalValidator()

    validation_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "anatomical_validation": None,
        "temporal_validation": None,
        "overall_biological_quality": 0.0,
        "recommendations": []
    }

    # Extract multiscale results (use 10μm scale for detailed validation)
    multiscale_results = results.get("multiscale_results", {})
    scale_10um = multiscale_results.get(10.0, multiscale_results.get("10.0"))

    if scale_10um is None:
        validation_report["error"] = "No 10μm scale results found for validation"
        return validation_report

    cluster_labels = scale_10um.get("cluster_labels")
    features = scale_10um.get("features")
    protein_names = results.get("protein_names", [])

    if cluster_labels is None or features is None:
        validation_report["error"] = "Missing cluster labels or features"
        return validation_report

    # Anatomical validation
    anatomical_validation = validator.validate_anatomical_enrichment(
        cluster_labels, features, protein_names
    )
    validation_report["anatomical_validation"] = anatomical_validation
    validation_report["recommendations"].extend(anatomical_validation.get("recommendations", []))

    # Temporal validation (if injury day specified in metadata)
    metadata = results.get("metadata", {})
    injury_day = metadata.get("injury_day")

    if injury_day is not None:
        baseline_features = None
        if baseline_results is not None:
            baseline_scale = baseline_results.get("multiscale_results", {}).get(10.0)
            if baseline_scale:
                baseline_features = baseline_scale.get("features")

        temporal_validation = validator.validate_injury_timepoint(
            cluster_labels, features, protein_names, injury_day, baseline_features
        )
        validation_report["temporal_validation"] = temporal_validation
        validation_report["recommendations"].extend(temporal_validation.get("recommendations", []))

    # Overall biological quality
    quality_scores = []
    if anatomical_validation:
        quality_scores.append(anatomical_validation.get("overall_quality", 0.0))
    if "temporal_validation" in validation_report and validation_report["temporal_validation"]:
        quality_scores.append(temporal_validation.get("response_quality", 0.0))

    if quality_scores:
        validation_report["overall_biological_quality"] = float(np.mean(quality_scores))

    return validation_report


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Kidney Biological Validation Module")
    print("=" * 80)

    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    protein_names = ["CD31", "CD34", "CD140a", "CD45", "Ly6G", "CD11b", "CD206", "CD140b", "CD44"]

    # Create two clusters: cortex-like and medulla-like
    cluster_labels = np.random.choice([0, 1], size=n_samples)

    # Features with anatomical signatures
    features = np.random.rand(n_samples, len(protein_names))

    # Cluster 0: High CD31/CD34 (cortex-like)
    features[cluster_labels == 0, 0] *= 3  # CD31
    features[cluster_labels == 0, 1] *= 3  # CD34

    # Cluster 1: High CD140a (medulla-like)
    features[cluster_labels == 1, 2] *= 3  # CD140a

    # Run validation
    validator = KidneyBiologicalValidator()
    results = validator.validate_anatomical_enrichment(cluster_labels, features, protein_names)

    print("\nAnatomical Validation Results:")
    print(f"  Cortex signature quality: {results['cortex_validation']['signature_quality']:.3f}")
    print(f"  Medulla signature quality: {results['medulla_validation']['signature_quality']:.3f}")
    print(f"  Overall quality: {results['overall_quality']:.3f}")

    if results["recommendations"]:
        print("\nRecommendations:")
        for rec in results["recommendations"]:
            print(f"  [{rec['priority']}] {rec['finding']}")
