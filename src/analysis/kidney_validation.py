"""
Kidney-Specific Validation for IMC Spatial Analysis

Focused validation for kidney injury studies using known anatomical and
temporal patterns. Designed for hypothesis generation, not definitive claims.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import ndimage, stats
from dataclasses import dataclass


@dataclass 
class KidneyValidationResult:
    """Results from kidney-specific validation metrics."""
    anatomical_concordance: float
    temporal_consistency: float  
    spatial_pattern_score: float
    glomerular_detection: float
    immune_infiltration_pattern: float
    confidence_notes: str


class KidneyIMCValidator:
    """
    Kidney-specific validation for IMC superpixel analysis.
    
    Validates whether superpixel segmentation captures meaningful
    kidney anatomy and injury response patterns. Designed for
    hypothesis generation with n=2 pilot data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with kidney experiment configuration.
        
        Args:
            config: Kidney experiment configuration from config.json
        """
        self.config = config
        self.anatomical_expectations = config.get('anatomical_validation', {})
        self.temporal_expectations = config.get('temporal_expectations', {})
        self.scale_config = config.get('superpixel_scales', {})
        
    def validate_kidney_segmentation(self,
                                   superpixel_labels: np.ndarray,
                                   protein_data: Dict[str, np.ndarray],
                                   metadata: Dict[str, Any],
                                   scale_um: float) -> KidneyValidationResult:
        """
        Validate superpixel segmentation for kidney analysis.
        
        Args:
            superpixel_labels: 2D array of superpixel assignments
            protein_data: Dictionary of protein intensities
            metadata: ROI metadata (region, timepoint, etc.)
            scale_um: Analysis scale
            
        Returns:
            Kidney-specific validation results
        """
        # Get anatomical region from metadata
        anatomical_region = metadata.get('Details', 'Unknown')
        injury_day = metadata.get('Injury Day', 0)
        
        # Anatomical concordance validation
        anatomical_score = self._validate_anatomical_signatures(
            superpixel_labels, protein_data, anatomical_region
        )
        
        # Temporal consistency validation
        temporal_score = self._validate_temporal_patterns(
            superpixel_labels, protein_data, injury_day
        )
        
        # Spatial pattern validation
        spatial_score = self._validate_spatial_organization(
            superpixel_labels, protein_data, scale_um
        )
        
        # Glomerular detection (for cortical samples)
        glomerular_score = 0.0
        if anatomical_region == 'Cortex' and scale_um >= 150:
            glomerular_score = self._detect_glomerular_structures(
                superpixel_labels, protein_data
            )
        
        # Immune infiltration pattern
        immune_score = self._validate_immune_patterns(
            superpixel_labels, protein_data, injury_day, anatomical_region
        )
        
        # Generate confidence notes
        confidence_notes = self._generate_confidence_assessment(
            anatomical_score, temporal_score, spatial_score, 
            metadata, scale_um
        )
        
        return KidneyValidationResult(
            anatomical_concordance=anatomical_score,
            temporal_consistency=temporal_score,
            spatial_pattern_score=spatial_score,
            glomerular_detection=glomerular_score,
            immune_infiltration_pattern=immune_score,
            confidence_notes=confidence_notes
        )
    
    def _validate_anatomical_signatures(self,
                                      superpixel_labels: np.ndarray,
                                      protein_data: Dict[str, np.ndarray],
                                      region: str) -> float:
        """Validate anatomical marker enrichment patterns."""
        
        if region not in ['Cortex', 'Medulla']:
            return 0.0
            
        expected_signatures = self.anatomical_expectations.get(
            f'{region.lower()}_signatures', {}
        )
        expected_high = expected_signatures.get('expected_high', [])
        
        if not expected_high:
            return 0.0
        
        enrichment_scores = []
        
        # Check each expected marker
        for marker in expected_high:
            if marker in protein_data:
                # Compute coefficient of variation across superpixels
                unique_superpixels = np.unique(superpixel_labels)
                unique_superpixels = unique_superpixels[unique_superpixels >= 0]
                
                if len(unique_superpixels) > 1:
                    superpixel_means = []
                    for sp_id in unique_superpixels:
                        mask = superpixel_labels == sp_id
                        if np.sum(mask) > 0:
                            mean_intensity = np.mean(protein_data[marker][mask])
                            superpixel_means.append(mean_intensity)
                    
                    if len(superpixel_means) > 1:
                        cv = np.std(superpixel_means) / (np.mean(superpixel_means) + 1e-10)
                        enrichment_scores.append(cv)
        
        return float(np.mean(enrichment_scores)) if enrichment_scores else 0.0
    
    def _validate_temporal_patterns(self,
                                  superpixel_labels: np.ndarray,
                                  protein_data: Dict[str, np.ndarray],
                                  injury_day: int) -> float:
        """Validate expected temporal injury response patterns."""
        
        day_key = f'day_{injury_day}'
        if day_key not in self.temporal_expectations:
            return 0.0
        
        expected_markers = self.temporal_expectations[day_key].get('key_markers', [])
        expected_pattern = self.temporal_expectations[day_key].get('spatial_pattern', '')
        
        if not expected_markers:
            return 0.0
        
        pattern_scores = []
        
        for marker in expected_markers:
            if marker in protein_data:
                if expected_pattern == 'focal_infiltration':
                    # Expect high clustering
                    clustering_score = self._compute_spatial_clustering(
                        protein_data[marker], superpixel_labels
                    )
                    pattern_scores.append(clustering_score)
                    
                elif expected_pattern == 'expanding_inflammation':
                    # Expect moderate clustering with spread
                    clustering_score = self._compute_spatial_clustering(
                        protein_data[marker], superpixel_labels
                    )
                    # Moderate clustering is optimal
                    optimal_score = 1.0 - abs(clustering_score - 0.5) * 2
                    pattern_scores.append(optimal_score)
                    
                elif expected_pattern == 'organized_repair':
                    # Expect distributed pattern
                    distribution_score = self._compute_spatial_distribution(
                        protein_data[marker], superpixel_labels
                    )
                    pattern_scores.append(distribution_score)
        
        return float(np.mean(pattern_scores)) if pattern_scores else 0.0
    
    def _validate_spatial_organization(self,
                                     superpixel_labels: np.ndarray,
                                     protein_data: Dict[str, np.ndarray],
                                     scale_um: float) -> float:
        """Validate spatial organization at given scale."""
        
        # Find scale configuration
        scale_config = None
        for scale_name, config in self.scale_config.items():
            if abs(config['target_size_um'] - scale_um) < 5:
                scale_config = config
                break
        
        if not scale_config:
            return 0.0
        
        expected_features = scale_config.get('biological_features', [])
        organization_scores = []
        
        # Validate expected biological features at this scale
        for feature in expected_features:
            if 'CD31' in feature and 'CD31' in protein_data:
                # Vascular network organization
                vascular_score = self._assess_vascular_organization(
                    protein_data['CD31'], superpixel_labels
                )
                organization_scores.append(vascular_score)
                
            elif 'immune' in feature and 'CD45' in protein_data:
                # Immune cell spatial organization
                immune_score = self._assess_immune_organization(
                    protein_data['CD45'], superpixel_labels
                )
                organization_scores.append(immune_score)
        
        return float(np.mean(organization_scores)) if organization_scores else 0.0
    
    def _detect_glomerular_structures(self,
                                    superpixel_labels: np.ndarray,
                                    protein_data: Dict[str, np.ndarray]) -> float:
        """Attempt to detect glomerular structures using CD31/CD34."""
        
        if 'CD31' not in protein_data or 'CD34' not in protein_data:
            return 0.0
        
        # Glomeruli should show co-enrichment of CD31 and CD34
        cd31_intensity = protein_data['CD31']
        cd34_intensity = protein_data['CD34']
        
        unique_superpixels = np.unique(superpixel_labels)
        unique_superpixels = unique_superpixels[unique_superpixels >= 0]
        
        colocalization_scores = []
        
        for sp_id in unique_superpixels:
            mask = superpixel_labels == sp_id
            if np.sum(mask) > 10:  # Sufficient pixels
                cd31_mean = np.mean(cd31_intensity[mask])
                cd34_mean = np.mean(cd34_intensity[mask])
                
                # High in both = potential glomerular signal
                if cd31_mean > np.percentile(cd31_intensity, 75) and \
                   cd34_mean > np.percentile(cd34_intensity, 75):
                    colocalization_scores.append(1.0)
                else:
                    colocalization_scores.append(0.0)
        
        # Fraction of superpixels with potential glomerular signal
        return float(np.mean(colocalization_scores)) if colocalization_scores else 0.0
    
    def _validate_immune_patterns(self,
                                superpixel_labels: np.ndarray,
                                protein_data: Dict[str, np.ndarray],
                                injury_day: int,
                                region: str) -> float:
        """Validate immune infiltration patterns."""
        
        if 'CD45' not in protein_data:
            return 0.0
        
        cd45_intensity = protein_data['CD45']
        
        # Expected immune patterns vary by timepoint
        if injury_day == 0:  # Sham
            # Expect low, distributed immune signal
            return self._compute_spatial_distribution(cd45_intensity, superpixel_labels)
            
        elif injury_day <= 3:  # Acute response
            # Expect focal clustering
            return self._compute_spatial_clustering(cd45_intensity, superpixel_labels)
            
        else:  # Resolution phase
            # Expect organized, moderate clustering
            clustering = self._compute_spatial_clustering(cd45_intensity, superpixel_labels)
            return 1.0 - abs(clustering - 0.6) * 2.5  # Optimal around 0.6
    
    def _compute_spatial_clustering(self,
                                  intensity: np.ndarray,
                                  superpixel_labels: np.ndarray) -> float:
        """Compute spatial clustering of intensity signal."""
        
        # Threshold to identify high-intensity regions
        threshold = np.percentile(intensity, 75)
        high_intensity_mask = intensity > threshold
        
        if np.sum(high_intensity_mask) == 0:
            return 0.0
        
        # Find connected components of high-intensity superpixels
        unique_superpixels = np.unique(superpixel_labels)
        high_intensity_superpixels = set()
        
        for sp_id in unique_superpixels:
            if sp_id >= 0:
                mask = superpixel_labels == sp_id
                if np.mean(high_intensity_mask[mask]) > 0.5:
                    high_intensity_superpixels.add(sp_id)
        
        if len(high_intensity_superpixels) == 0:
            return 0.0
        
        # Simple clustering metric: fewer large clusters = higher clustering
        binary_map = np.isin(superpixel_labels, list(high_intensity_superpixels))
        labeled_regions, n_regions = ndimage.label(binary_map)
        
        if n_regions == 0:
            return 0.0
        
        # Clustering score = inverse of number of separate regions
        return 1.0 / (1.0 + n_regions)
    
    def _compute_spatial_distribution(self,
                                    intensity: np.ndarray,
                                    superpixel_labels: np.ndarray) -> float:
        """Compute spatial distribution uniformity."""
        
        unique_superpixels = np.unique(superpixel_labels)
        unique_superpixels = unique_superpixels[unique_superpixels >= 0]
        
        if len(unique_superpixels) <= 1:
            return 0.0
        
        superpixel_means = []
        for sp_id in unique_superpixels:
            mask = superpixel_labels == sp_id
            if np.sum(mask) > 0:
                mean_intensity = np.mean(intensity[mask])
                superpixel_means.append(mean_intensity)
        
        if len(superpixel_means) <= 1:
            return 0.0
        
        # Distribution score = inverse coefficient of variation
        cv = np.std(superpixel_means) / (np.mean(superpixel_means) + 1e-10)
        return 1.0 / (1.0 + cv)
    
    def _assess_vascular_organization(self,
                                    cd31_intensity: np.ndarray,
                                    superpixel_labels: np.ndarray) -> float:
        """Assess vascular network organization."""
        
        # Vessels should form connected networks
        threshold = np.percentile(cd31_intensity, 70)
        vascular_mask = cd31_intensity > threshold
        
        if np.sum(vascular_mask) == 0:
            return 0.0
        
        # Connectivity metric
        labeled_regions, n_regions = ndimage.label(vascular_mask)
        
        if n_regions == 0:
            return 0.0
        
        # Good vascular organization = fewer, larger connected components
        region_sizes = [np.sum(labeled_regions == i) for i in range(1, n_regions + 1)]
        
        if not region_sizes:
            return 0.0
        
        # Connectivity score based on largest component
        largest_component_fraction = max(region_sizes) / sum(region_sizes)
        return float(largest_component_fraction)
    
    def _assess_immune_organization(self,
                                  cd45_intensity: np.ndarray,
                                  superpixel_labels: np.ndarray) -> float:
        """Assess immune cell spatial organization."""
        
        # Similar to clustering but focusing on organizational structure
        return self._compute_spatial_clustering(cd45_intensity, superpixel_labels)
    
    def _generate_confidence_assessment(self,
                                      anatomical_score: float,
                                      temporal_score: float,
                                      spatial_score: float,
                                      metadata: Dict[str, Any],
                                      scale_um: float) -> str:
        """Generate confidence assessment for validation results."""
        
        confidence_notes = []
        
        # Scale appropriateness
        if scale_um < 20:
            confidence_notes.append("Scale may be too fine for kidney architecture")
        elif scale_um > 250:
            confidence_notes.append("Scale may miss cellular interactions")
        
        # Sample type assessment
        region = metadata.get('Details', 'Unknown')
        if region == 'Unknown':
            confidence_notes.append("Unknown anatomical region limits validation")
        
        # Injury day assessment
        injury_day = metadata.get('Injury Day', -1)
        if injury_day < 0:
            confidence_notes.append("Missing injury timepoint information")
        
        # Overall confidence
        overall_score = np.mean([anatomical_score, temporal_score, spatial_score])
        if overall_score > 0.7:
            confidence_notes.append("High confidence in segmentation quality")
        elif overall_score > 0.4:
            confidence_notes.append("Moderate confidence - suitable for hypothesis generation")
        else:
            confidence_notes.append("Low confidence - interpret with caution")
        
        # Sample size reminder
        confidence_notes.append("Results based on n=2 pilot data - hypothesis generating only")
        
        return "; ".join(confidence_notes)


def create_kidney_validator(config: Dict[str, Any]) -> KidneyIMCValidator:
    """
    Create kidney-specific validator from configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Configured KidneyIMCValidator
    """
    kidney_config = config.get('kidney_experiment', {})
    return KidneyIMCValidator(kidney_config)