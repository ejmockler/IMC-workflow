"""
Hierarchical Data Structure Handling for IMC Analysis

Provides utilities for managing and validating nested data structures,
variance decomposition, and proper aggregation preserving hierarchical relationships.

Implements the core data structures needed for mixed-effects modeling.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from pathlib import Path

from .mixed_effects_models import HierarchicalDataStructure


@dataclass
class VarianceComponents:
    """Results from variance decomposition analysis."""
    total_variance: float
    between_subject_variance: float
    between_slide_variance: float
    within_variance: float
    
    subject_icc: float  # Subject-level ICC
    slide_icc: float    # Slide-level ICC (conditional on subject)
    
    effective_sample_sizes: Dict[str, float] = field(default_factory=dict)
    design_effects: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived statistics."""
        if self.total_variance > 0:
            self.subject_proportion = self.between_subject_variance / self.total_variance
            self.slide_proportion = self.between_slide_variance / self.total_variance  
            self.within_proportion = self.within_variance / self.total_variance
        else:
            self.subject_proportion = 0.0
            self.slide_proportion = 0.0
            self.within_proportion = 0.0


class NestedDataValidator:
    """
    Validates integrity of hierarchical data structure and identifies issues.
    """
    
    def __init__(self, hierarchy: HierarchicalDataStructure):
        self.hierarchy = hierarchy
        
    def validate_data_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive validation of hierarchical data integrity.
        
        Args:
            data: DataFrame with hierarchical structure
            
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'statistics': {}
        }
        
        # Check required columns exist
        required_columns = [
            self.hierarchy.subject_column,
            self.hierarchy.roi_column
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
            return validation_results
        
        # Validate hierarchical structure
        hierarchy_issues = self._validate_hierarchy_structure(data)
        validation_results['issues'].extend(hierarchy_issues)
        
        # Check for pseudoreplication patterns
        pseudorep_warnings = self._check_pseudoreplication_patterns(data)
        validation_results['warnings'].extend(pseudorep_warnings)
        
        # Calculate summary statistics
        stats = self._calculate_structure_statistics(data)
        validation_results['statistics'] = stats
        
        # Generate recommendations
        recommendations = self._generate_recommendations(stats)
        validation_results['recommendations'] = recommendations
        
        # Overall validity
        if validation_results['issues']:
            validation_results['is_valid'] = False
        
        return validation_results
    
    def _validate_hierarchy_structure(self, data: pd.DataFrame) -> List[str]:
        """Validate the nested structure is coherent."""
        issues = []
        
        # Check subject-slide consistency if slide column exists
        if self.hierarchy.slide_column in data.columns:
            # Each slide should belong to only one subject
            slide_subjects = data.groupby(self.hierarchy.slide_column)[
                self.hierarchy.subject_column
            ].nunique()
            
            multi_subject_slides = slide_subjects[slide_subjects > 1]
            if len(multi_subject_slides) > 0:
                issues.append(f"Slides belonging to multiple subjects: {list(multi_subject_slides.index)}")
        
        # Check ROI-slide consistency if slide column exists
        if self.hierarchy.slide_column in data.columns:
            roi_slides = data.groupby(self.hierarchy.roi_column)[
                self.hierarchy.slide_column
            ].nunique()
            
            multi_slide_rois = roi_slides[roi_slides > 1]
            if len(multi_slide_rois) > 0:
                issues.append(f"ROIs spanning multiple slides: {list(multi_slide_rois.index)}")
        
        # Check subject-ROI relationship
        roi_subjects = data.groupby(self.hierarchy.roi_column)[
            self.hierarchy.subject_column
        ].nunique()
        
        multi_subject_rois = roi_subjects[roi_subjects > 1]
        if len(multi_subject_rois) > 0:
            issues.append(f"ROIs belonging to multiple subjects: {list(multi_subject_rois.index)}")
        
        return issues
    
    def _check_pseudoreplication_patterns(self, data: pd.DataFrame) -> List[str]:
        """Check for patterns that indicate pseudoreplication risk."""
        warnings_list = []
        
        # Check observation counts per subject
        obs_per_subject = data[self.hierarchy.subject_column].value_counts()
        
        max_obs = obs_per_subject.max()
        min_obs = obs_per_subject.min()
        imbalance_ratio = max_obs / min_obs if min_obs > 0 else np.inf
        
        if imbalance_ratio > 10:
            warnings_list.append(
                f"Severe imbalance in observations per subject (ratio: {imbalance_ratio:.1f}). "
                "This may cause issues in mixed-effects modeling."
            )
        
        # Check for single-observation subjects
        single_obs_subjects = obs_per_subject[obs_per_subject == 1]
        if len(single_obs_subjects) > 0:
            warnings_list.append(
                f"{len(single_obs_subjects)} subjects have only single observations. "
                "Random effects estimation may be unreliable."
            )
        
        # Check total subject count for power
        n_subjects = data[self.hierarchy.subject_column].nunique()
        if n_subjects < 10:
            warnings_list.append(
                f"Only {n_subjects} subjects available. "
                "This is quite small for mixed-effects modeling and statistical power will be limited."
            )
        elif n_subjects < 20:
            warnings_list.append(
                f"Only {n_subjects} subjects available. "
                "Consider this when interpreting results - power may be limited."
            )
        
        return warnings_list
    
    def _calculate_structure_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics of hierarchical structure."""
        stats = {}
        
        # Subject-level statistics
        n_subjects = data[self.hierarchy.subject_column].nunique()
        obs_per_subject = data[self.hierarchy.subject_column].value_counts()
        
        stats['n_subjects'] = n_subjects
        stats['obs_per_subject'] = {
            'mean': float(obs_per_subject.mean()),
            'std': float(obs_per_subject.std()),
            'min': int(obs_per_subject.min()),
            'max': int(obs_per_subject.max()),
            'median': float(obs_per_subject.median())
        }
        
        # ROI-level statistics
        n_rois = data[self.hierarchy.roi_column].nunique()
        obs_per_roi = data[self.hierarchy.roi_column].value_counts()
        
        stats['n_rois'] = n_rois
        stats['obs_per_roi'] = {
            'mean': float(obs_per_roi.mean()),
            'std': float(obs_per_roi.std()),
            'min': int(obs_per_roi.min()),
            'max': int(obs_per_roi.max())
        }
        
        # Cross-tabulation statistics
        rois_per_subject = data.groupby(self.hierarchy.subject_column)[
            self.hierarchy.roi_column
        ].nunique()
        
        stats['rois_per_subject'] = {
            'mean': float(rois_per_subject.mean()),
            'std': float(rois_per_subject.std()),
            'min': int(rois_per_subject.min()),
            'max': int(rois_per_subject.max())
        }
        
        # Slide statistics if available
        if self.hierarchy.slide_column in data.columns:
            n_slides = data[self.hierarchy.slide_column].nunique()
            slides_per_subject = data.groupby(self.hierarchy.subject_column)[
                self.hierarchy.slide_column
            ].nunique()
            
            stats['n_slides'] = n_slides
            stats['slides_per_subject'] = {
                'mean': float(slides_per_subject.mean()),
                'std': float(slides_per_subject.std()),
                'min': int(slides_per_subject.min()),
                'max': int(slides_per_subject.max())
            }
        
        return stats
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on structure statistics."""
        recommendations = []
        
        # Subject count recommendations
        n_subjects = stats['n_subjects']
        if n_subjects < 5:
            recommendations.append(
                "Very small sample size. Consider descriptive analysis only. "
                "Mixed-effects modeling not recommended."
            )
        elif n_subjects < 10:
            recommendations.append(
                "Small sample size. Use conservative statistical approaches. "
                "Bootstrap confidence intervals recommended over p-values."
            )
        elif n_subjects < 20:
            recommendations.append(
                "Moderate sample size. Mixed-effects modeling feasible but interpret cautiously. "
                "Consider effect sizes and confidence intervals."
            )
        
        # Imbalance recommendations
        obs_ratio = stats['obs_per_subject']['max'] / stats['obs_per_subject']['min']
        if obs_ratio > 5:
            recommendations.append(
                "Significant imbalance in observations per subject. "
                "Consider weighted analysis or robust standard errors."
            )
        
        # ROI recommendations
        avg_rois_per_subject = stats['rois_per_subject']['mean']
        if avg_rois_per_subject < 2:
            recommendations.append(
                "Very few ROIs per subject on average. "
                "Random effects for slides/ROIs may not be estimable."
            )
        
        return recommendations


class HierarchicalAggregator:
    """
    Proper aggregation methods preserving hierarchical structure.
    """
    
    def __init__(self, hierarchy: HierarchicalDataStructure):
        self.hierarchy = hierarchy
        
    def aggregate_to_subject_level(self,
                                  data: pd.DataFrame,
                                  value_columns: List[str],
                                  aggregation_method: str = 'mean',
                                  preserve_metadata: bool = True) -> pd.DataFrame:
        """
        Aggregate data to subject level while preserving structure.
        
        Args:
            data: Input data with hierarchical structure
            value_columns: Columns to aggregate
            aggregation_method: 'mean', 'median', 'sum', etc.
            preserve_metadata: Whether to preserve subject-level metadata
            
        Returns:
            Subject-level aggregated data
        """
        
        # Define grouping columns
        group_cols = [self.hierarchy.subject_column]
        
        # Add metadata columns to preserve
        metadata_cols = []
        if preserve_metadata:
            potential_metadata = [
                self.hierarchy.condition_column,
                self.hierarchy.timepoint_column
            ]
            metadata_cols = [col for col in potential_metadata 
                           if col and col in data.columns]
            group_cols.extend(metadata_cols)
        
        # Remove duplicates while preserving order
        group_cols = list(dict.fromkeys(group_cols))
        
        # Perform aggregation
        if aggregation_method == 'mean':
            aggregated = data.groupby(group_cols)[value_columns].mean()
        elif aggregation_method == 'median':
            aggregated = data.groupby(group_cols)[value_columns].median()
        elif aggregation_method == 'sum':
            aggregated = data.groupby(group_cols)[value_columns].sum()
        elif aggregation_method == 'std':
            aggregated = data.groupby(group_cols)[value_columns].std()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Reset index to make grouping columns regular columns
        aggregated = aggregated.reset_index()
        
        # Add aggregation metadata
        aggregated['aggregation_method'] = aggregation_method
        aggregated['aggregation_level'] = 'subject'
        
        return aggregated
    
    def aggregate_to_roi_level(self,
                              data: pd.DataFrame,
                              value_columns: List[str],
                              aggregation_method: str = 'mean') -> pd.DataFrame:
        """Aggregate pixel-level data to ROI level."""
        
        group_cols = [
            self.hierarchy.subject_column,
            self.hierarchy.roi_column
        ]
        
        # Add slide column if available
        if self.hierarchy.slide_column in data.columns:
            group_cols.insert(1, self.hierarchy.slide_column)
        
        # Perform aggregation
        if aggregation_method == 'mean':
            aggregated = data.groupby(group_cols)[value_columns].mean()
        elif aggregation_method == 'median':
            aggregated = data.groupby(group_cols)[value_columns].median()
        elif aggregation_method == 'sum':
            aggregated = data.groupby(group_cols)[value_columns].sum()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        aggregated = aggregated.reset_index()
        aggregated['aggregation_method'] = aggregation_method
        aggregated['aggregation_level'] = 'roi'
        
        return aggregated
    
    def calculate_hierarchical_means(self,
                                   data: pd.DataFrame,
                                   value_columns: List[str]) -> Dict[str, pd.DataFrame]:
        """Calculate means at all hierarchical levels."""
        
        results = {}
        
        # Subject level
        results['subject'] = self.aggregate_to_subject_level(
            data, value_columns, 'mean'
        )
        
        # ROI level  
        results['roi'] = self.aggregate_to_roi_level(
            data, value_columns, 'mean'
        )
        
        # Slide level if available
        if self.hierarchy.slide_column in data.columns:
            slide_group_cols = [
                self.hierarchy.subject_column,
                self.hierarchy.slide_column
            ]
            results['slide'] = data.groupby(slide_group_cols)[value_columns].mean().reset_index()
            results['slide']['aggregation_method'] = 'mean'
            results['slide']['aggregation_level'] = 'slide'
        
        return results


class VarianceDecomposition:
    """
    Variance component estimation for hierarchical data.
    """
    
    def __init__(self, hierarchy: HierarchicalDataStructure):
        self.hierarchy = hierarchy
        
    def decompose_variance(self,
                          data: pd.DataFrame,
                          response_column: str,
                          method: str = 'anova') -> VarianceComponents:
        """
        Decompose variance into hierarchical components.
        
        Args:
            data: Data with hierarchical structure
            response_column: Column to decompose variance for
            method: 'anova' or 'reml' (if statsmodels available)
            
        Returns:
            VarianceComponents object with decomposition results
        """
        
        if method == 'reml':
            return self._decompose_variance_reml(data, response_column)
        else:
            return self._decompose_variance_anova(data, response_column)
    
    def _decompose_variance_anova(self,
                                 data: pd.DataFrame,
                                 response_column: str) -> VarianceComponents:
        """ANOVA-based variance decomposition."""
        
        # Calculate means at each level
        aggregator = HierarchicalAggregator(self.hierarchy)
        
        # Subject-level means
        subject_means = aggregator.aggregate_to_subject_level(
            data, [response_column], 'mean'
        )
        
        # ROI-level means
        roi_means = aggregator.aggregate_to_roi_level(
            data, [response_column], 'mean'
        )
        
        # Overall mean
        overall_mean = data[response_column].mean()
        
        # Calculate variance components
        
        # Between-subject variance
        subject_deviations = subject_means[response_column] - overall_mean
        between_subject_var = np.var(subject_deviations, ddof=1)
        
        # Within-subject, between-ROI variance
        roi_subject_means = roi_means.merge(
            subject_means[[self.hierarchy.subject_column, response_column]], 
            on=self.hierarchy.subject_column,
            suffixes=('_roi', '_subject')
        )
        roi_deviations = (roi_subject_means[f'{response_column}_roi'] - 
                         roi_subject_means[f'{response_column}_subject'])
        between_roi_var = np.var(roi_deviations, ddof=1)
        
        # Within-ROI variance (residual)
        data_with_roi_means = data.merge(
            roi_means[[self.hierarchy.roi_column, response_column]],
            on=self.hierarchy.roi_column,
            suffixes=('', '_roi_mean')
        )
        residuals = data_with_roi_means[response_column] - data_with_roi_means[f'{response_column}_roi_mean']
        within_var = np.var(residuals, ddof=1)
        
        # Total variance
        total_var = np.var(data[response_column], ddof=1)
        
        # Calculate ICCs
        subject_icc = between_subject_var / total_var if total_var > 0 else 0
        roi_icc = between_roi_var / (between_roi_var + within_var) if (between_roi_var + within_var) > 0 else 0
        
        # Calculate effective sample sizes
        n_subjects = data[self.hierarchy.subject_column].nunique()
        n_rois = data[self.hierarchy.roi_column].nunique()
        n_obs = len(data)
        
        avg_obs_per_subject = n_obs / n_subjects if n_subjects > 0 else 0
        avg_obs_per_roi = n_obs / n_rois if n_rois > 0 else 0
        
        # Design effects
        design_effect_subject = 1 + (avg_obs_per_subject - 1) * subject_icc
        design_effect_roi = 1 + (avg_obs_per_roi - 1) * roi_icc
        
        effective_sample_sizes = {
            'subjects': n_subjects / design_effect_subject,
            'rois': n_rois / design_effect_roi,
            'observations': n_obs / design_effect_subject
        }
        
        design_effects = {
            'subject_level': design_effect_subject,
            'roi_level': design_effect_roi
        }
        
        return VarianceComponents(
            total_variance=float(total_var),
            between_subject_variance=float(between_subject_var),
            between_slide_variance=float(between_roi_var),  # Using ROI as "slide" level
            within_variance=float(within_var),
            subject_icc=float(subject_icc),
            slide_icc=float(roi_icc),
            effective_sample_sizes=effective_sample_sizes,
            design_effects=design_effects
        )
    
    def _decompose_variance_reml(self,
                                data: pd.DataFrame,
                                response_column: str) -> VarianceComponents:
        """REML-based variance decomposition using statsmodels."""
        
        try:
            from statsmodels.regression.mixed_linear_model import MixedLM
            
            # Fit mixed-effects model with random intercepts
            groups = data[self.hierarchy.subject_column]
            
            model = MixedLM(
                endog=data[response_column],
                exog=np.ones(len(data)),  # Intercept only
                groups=groups
            )
            
            fitted_model = model.fit(method='lbfgs')
            
            # Extract variance components
            random_effects_var = float(fitted_model.cov_re.iloc[0, 0])
            residual_var = float(fitted_model.scale)
            total_var = random_effects_var + residual_var
            
            subject_icc = random_effects_var / total_var if total_var > 0 else 0
            
            return VarianceComponents(
                total_variance=total_var,
                between_subject_variance=random_effects_var,
                between_slide_variance=0.0,  # Not modeled in simple random intercept
                within_variance=residual_var,
                subject_icc=subject_icc,
                slide_icc=0.0
            )
            
        except ImportError:
            warnings.warn("statsmodels not available. Falling back to ANOVA method.")
            return self._decompose_variance_anova(data, response_column)
        except Exception as e:
            warnings.warn(f"REML estimation failed: {e}. Falling back to ANOVA method.")
            return self._decompose_variance_anova(data, response_column)


class ClusterCorrection:
    """
    Adjustments for clustering in statistical inference.
    """
    
    @staticmethod
    def cluster_robust_se(residuals: np.ndarray,
                         cluster_ids: np.ndarray,
                         X: np.ndarray) -> np.ndarray:
        """
        Calculate cluster-robust standard errors.
        
        Args:
            residuals: Model residuals
            cluster_ids: Cluster identifiers for each observation
            X: Design matrix
            
        Returns:
            Cluster-robust covariance matrix
        """
        
        # Get unique clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)
        
        if n_clusters <= 1:
            warnings.warn("Only one cluster found. Cannot calculate cluster-robust SE.")
            return np.eye(X.shape[1])
        
        # Calculate cluster-robust covariance matrix
        XtX_inv = np.linalg.pinv(X.T @ X)
        
        # Sum of outer products for each cluster
        cluster_sum = np.zeros((X.shape[1], X.shape[1]))
        
        for cluster in unique_clusters:
            cluster_mask = cluster_ids == cluster
            X_cluster = X[cluster_mask]
            residuals_cluster = residuals[cluster_mask]
            
            # Cluster contribution to covariance
            cluster_contrib = X_cluster.T @ np.diag(residuals_cluster**2) @ X_cluster
            cluster_sum += cluster_contrib
        
        # Finite sample correction
        correction = n_clusters / (n_clusters - 1)
        
        cluster_cov = correction * XtX_inv @ cluster_sum @ XtX_inv
        
        return cluster_cov
    
    @staticmethod
    def effective_degrees_freedom(n_clusters: int,
                                 n_parameters: int) -> float:
        """
        Calculate effective degrees of freedom for cluster-robust inference.
        
        Conservative approach: use number of clusters minus parameters.
        """
        return max(1, n_clusters - n_parameters)


def create_hierarchical_summary(data: pd.DataFrame,
                               hierarchy: HierarchicalDataStructure,
                               value_columns: List[str]) -> Dict[str, Any]:
    """
    Create comprehensive summary of hierarchical data structure.
    
    Args:
        data: Input data
        hierarchy: Hierarchical structure definition  
        value_columns: Columns to analyze
        
    Returns:
        Dictionary with comprehensive hierarchical summary
    """
    
    # Initialize components
    validator = NestedDataValidator(hierarchy)
    aggregator = HierarchicalAggregator(hierarchy)
    decomposer = VarianceDecomposition(hierarchy)
    
    summary = {}
    
    # Data validation
    summary['validation'] = validator.validate_data_integrity(data)
    
    # Structure statistics
    summary['structure_statistics'] = validator._calculate_structure_statistics(data)
    
    # Aggregated data at different levels
    summary['aggregated_data'] = {}
    for column in value_columns:
        if column in data.columns:
            summary['aggregated_data'][column] = aggregator.calculate_hierarchical_means(
                data, [column]
            )
    
    # Variance decomposition
    summary['variance_components'] = {}
    for column in value_columns:
        if column in data.columns and data[column].notna().sum() > 10:
            try:
                variance_comp = decomposer.decompose_variance(data, column)
                summary['variance_components'][column] = variance_comp
            except Exception as e:
                warnings.warn(f"Variance decomposition failed for {column}: {e}")
                summary['variance_components'][column] = None
    
    return summary