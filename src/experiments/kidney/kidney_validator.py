"""
Kidney-Specific Biological Validation

This module contains kidney injury model-specific validation logic.
It imports from the core framework but the core never imports from here.
This maintains clean separation of concerns.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from scipy import ndimage


class KidneyInjuryValidator:
    """
    Kidney injury model-specific biological validation.
    
    This class contains domain knowledge about kidney anatomy,
    expected protein patterns, etc. It's completely separate from
    the core validation framework.
    """
    
    def __init__(self, experiment_config: Dict[str, Any]):
        """
        Initialize with kidney-specific configuration.
        
        Args:
            experiment_config: Kidney experiment parameters including
                              marker mappings, anatomical regions, etc.
        """
        self.config = experiment_config
        self.markers = experiment_config.get('markers', {})
        self.anatomical_model = experiment_config.get('anatomy', {})
        
    def validate_kidney_structures(self,
                                  segmentation: np.ndarray,
                                  protein_data: Dict[str, np.ndarray],
                                  metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate kidney-specific biological structures.
        
        This method contains domain knowledge about kidney anatomy
        and expected protein expression patterns.
        
        Args:
            segmentation: Segment labels
            protein_data: Dictionary of protein intensities
            metadata: Experimental metadata (timepoint, condition, etc.)
            
        Returns:
            Kidney-specific validation metrics
        """
        results = {}
        
        # Validate cortex/medulla organization if present
        if 'Details' in metadata:
            region = metadata['Details']
            if region in ['Cortex', 'Medulla']:
                results[f'{region.lower()}_validation'] = self._validate_region(
                    segmentation, protein_data, region
                )
        
        # Check vascular patterns if CD31/CD34 present
        if 'CD31' in protein_data:
            results['vascular_continuity'] = self._check_vascular_continuity(
                segmentation, protein_data['CD31']
            )
        
        # Check immune infiltration patterns if CD45 present
        if 'CD45' in protein_data and 'Injury Day' in metadata:
            day = metadata['Injury Day']
            results['immune_infiltration'] = self._check_immune_pattern(
                segmentation, protein_data['CD45'], day
            )
        
        return results
    
    def _validate_region(self,
                        segmentation: np.ndarray,
                        protein_data: Dict[str, np.ndarray],
                        region: str) -> float:
        """Validate cortex or medulla specific patterns."""
        
        # This contains kidney-specific biological knowledge
        # that doesn't belong in the core framework
        
        if region == 'Cortex':
            # Cortex should have more CD31 (glomerular capillaries)
            if 'CD31' in protein_data:
                cd31_density = np.mean(protein_data['CD31'] > 0)
                return float(cd31_density)
        
        elif region == 'Medulla':
            # Medulla has different vascular organization
            if 'CD31' in protein_data:
                # Check for vasa recta patterns (linear structures)
                gradient = ndimage.sobel(protein_data['CD31'])
                linearity = np.mean(gradient > np.percentile(gradient, 75))
                return float(linearity)
        
        return 0.0
    
    def _check_vascular_continuity(self,
                                   segmentation: np.ndarray,
                                   cd31_intensity: np.ndarray) -> float:
        """Check if CD31+ structures form continuous vessels."""
        
        # Threshold CD31 to identify vascular structures
        threshold = np.percentile(cd31_intensity, 75)
        vascular_mask = cd31_intensity > threshold
        
        # Check connectivity
        labeled, n_components = ndimage.label(vascular_mask)
        
        if n_components > 0:
            # Compute size of largest component
            sizes = [np.sum(labeled == i) for i in range(1, n_components + 1)]
            max_size = max(sizes)
            total_size = sum(sizes)
            
            # Continuity = fraction in largest component
            continuity = max_size / total_size if total_size > 0 else 0.0
            return float(continuity)
        
        return 0.0
    
    def _check_immune_pattern(self,
                              segmentation: np.ndarray,
                              cd45_intensity: np.ndarray,
                              injury_day: int) -> float:
        """Check if immune infiltration matches expected temporal pattern."""
        
        # Kidney injury model-specific knowledge:
        # Day 1-3: Peak neutrophil infiltration (clustered)
        # Day 7+: Resolution phase (more distributed)
        
        threshold = np.percentile(cd45_intensity, 75)
        immune_mask = cd45_intensity > threshold
        
        if injury_day <= 3:
            # Expect clustered pattern
            labeled, n_clusters = ndimage.label(immune_mask)
            clustering_score = n_clusters / np.sum(immune_mask) if np.sum(immune_mask) > 0 else 0
            return float(1.0 - clustering_score)  # Lower score = more clustered
            
        else:
            # Expect distributed pattern
            # Use coefficient of variation
            unique_segs = np.unique(segmentation)
            seg_means = []
            for seg in unique_segs[unique_segs >= 0]:
                mask = segmentation == seg
                if np.sum(mask) > 0:
                    seg_means.append(np.mean(cd45_intensity[mask]))
            
            if len(seg_means) > 1:
                cv = np.std(seg_means) / np.mean(seg_means)
                return float(1.0 / (1.0 + cv))  # Higher = more uniform
        
        return 0.5