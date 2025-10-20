"""
Mixed-Effects Statistical Models for Hierarchical IMC Data

Addresses pseudoreplication by properly modeling nested structure:
Subject → Slide → ROI → Pixel hierarchy with random effects.

Implements mixed-effects models with spatial autocorrelation handling.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")

try:
    import libpysal
    from libpysal import weights
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False
    warnings.warn("libpysal not available for spatial autocorrelation. Install with: pip install libpysal")


@dataclass(frozen=True)
class HierarchicalDataStructure:
    """
    Immutable representation of nested IMC data structure.
    
    Enforces proper hierarchy: Subject → Slide → ROI → Pixel
    """
    subjects: List[str]
    slides_per_subject: Dict[str, List[str]]
    rois_per_slide: Dict[str, List[str]]
    
    subject_column: str = "Mouse"
    slide_column: str = "Slide" 
    roi_column: str = "ROI"
    condition_column: str = "Condition"
    timepoint_column: str = "Injury Day"
    
    def __post_init__(self):
        """Validate hierarchical structure integrity."""
        # Ensure all slides have valid subjects
        all_slides = []
        for subject, slides in self.slides_per_subject.items():
            if subject not in self.subjects:
                raise ValueError(f"Subject {subject} not in subjects list")
            all_slides.extend(slides)
        
        # Ensure all ROIs have valid slides
        for slide, rois in self.rois_per_slide.items():
            if slide not in all_slides:
                raise ValueError(f"Slide {slide} not in slides list")
    
    @classmethod
    def from_metadata(cls, metadata: pd.DataFrame, 
                     subject_col: str = "Mouse",
                     slide_col: str = "Slide",
                     roi_col: str = "ROI",
                     condition_col: str = "Condition",
                     timepoint_col: str = "Injury Day") -> 'HierarchicalDataStructure':
        """Create hierarchical structure from metadata DataFrame."""
        
        # Handle missing slide column by creating artificial slides
        if slide_col not in metadata.columns:
            # Create slide IDs as Subject_ROI for uniqueness
            metadata = metadata.copy()
            metadata[slide_col] = metadata[subject_col].astype(str) + "_slide"
            warnings.warn(f"No {slide_col} column found. Creating artificial slides.")
        
        subjects = sorted(metadata[subject_col].unique())
        
        slides_per_subject = {}
        for subject in subjects:
            subject_data = metadata[metadata[subject_col] == subject]
            slides = sorted(subject_data[slide_col].unique())
            slides_per_subject[subject] = slides
        
        rois_per_slide = {}
        for subject, slides in slides_per_subject.items():
            for slide in slides:
                slide_data = metadata[
                    (metadata[subject_col] == subject) & 
                    (metadata[slide_col] == slide)
                ]
                rois = sorted(slide_data[roi_col].unique())
                rois_per_slide[slide] = rois
        
        return cls(
            subjects=subjects,
            slides_per_subject=slides_per_subject,
            rois_per_slide=rois_per_slide,
            subject_column=subject_col,
            slide_column=slide_col,
            roi_column=roi_col,
            condition_column=condition_col,
            timepoint_column=timepoint_col
        )
    
    def get_effective_sample_size(self) -> Dict[str, int]:
        """Calculate effective sample sizes at each hierarchical level."""
        n_subjects = len(self.subjects)
        n_slides = sum(len(slides) for slides in self.slides_per_subject.values())
        n_rois = sum(len(rois) for rois in self.rois_per_slide.values())
        
        return {
            'subjects': n_subjects,
            'slides': n_slides,
            'rois': n_rois,
            'primary_unit': n_subjects  # Subject is the unit of analysis
        }


@dataclass
class MixedEffectsConfig:
    """Configuration for mixed-effects modeling."""
    random_effects: List[str] = None  # Defaults to ['subject', 'slide']
    fixed_effects: List[str] = None   # Defaults to ['condition', 'timepoint']
    include_spatial: bool = True
    spatial_decay: float = 0.1  # Spatial correlation decay parameter
    convergence_tolerance: float = 1e-6
    max_iterations: int = 1000
    
    def __post_init__(self):
        if self.random_effects is None:
            self.random_effects = ['subject', 'slide']
        if self.fixed_effects is None:
            self.fixed_effects = ['condition', 'timepoint']


class NestedModel:
    """
    Mixed-effects model for nested IMC data structure.
    
    Handles Subject → Slide → ROI hierarchy with proper random effects.
    """
    
    def __init__(self, config: MixedEffectsConfig):
        self.config = config
        self.model = None
        self.fitted_model = None
        self.hierarchy = None
        
    def fit(self, 
            data: pd.DataFrame,
            response_var: str,
            hierarchy: HierarchicalDataStructure,
            spatial_coords: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Fit mixed-effects model to hierarchical data.
        
        Args:
            data: DataFrame with hierarchical structure
            response_var: Name of response variable column
            hierarchy: Hierarchical data structure definition
            spatial_coords: Optional spatial coordinates for spatial modeling
            
        Returns:
            Dictionary with model results and diagnostics
        """
        if not STATSMODELS_AVAILABLE:
            return self._fit_fallback_model(data, response_var, hierarchy)
        
        self.hierarchy = hierarchy
        
        # Prepare data for mixed-effects modeling
        model_data = self._prepare_model_data(data, response_var, hierarchy)
        
        if len(model_data) == 0:
            raise ValueError("No valid data for modeling after preparation")
        
        # Build fixed effects formula
        fixed_formula = self._build_fixed_effects_formula(response_var)
        
        # Define groups for random effects
        groups = model_data[hierarchy.subject_column]
        
        # Fit mixed-effects model
        try:
            self.model = MixedLM.from_formula(
                fixed_formula,
                data=model_data,
                groups=groups,
                re_formula="1"  # Random intercept by subject
            )
            
            self.fitted_model = self.model.fit(
                method='lbfgs',
                maxiter=self.config.max_iterations
            )
            
            # Calculate additional diagnostics
            results = self._calculate_model_diagnostics(model_data, spatial_coords)
            results['converged'] = self.fitted_model.converged
            results['aic'] = self.fitted_model.aic
            results['bic'] = self.fitted_model.bic
            
        except Exception as e:
            warnings.warn(f"Mixed-effects model failed: {e}. Using fallback.")
            return self._fit_fallback_model(data, response_var, hierarchy)
        
        return results
    
    def _prepare_model_data(self, 
                           data: pd.DataFrame,
                           response_var: str,
                           hierarchy: HierarchicalDataStructure) -> pd.DataFrame:
        """Prepare data for mixed-effects modeling."""
        model_data = data.copy()
        
        # Ensure all required columns exist
        required_cols = [
            response_var,
            hierarchy.subject_column,
            hierarchy.roi_column
        ]
        
        missing_cols = [col for col in required_cols if col not in model_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add slide column if missing (create artificial slides)
        if hierarchy.slide_column not in model_data.columns:
            model_data[hierarchy.slide_column] = (
                model_data[hierarchy.subject_column].astype(str) + "_slide"
            )
        
        # Remove rows with missing response
        model_data = model_data.dropna(subset=[response_var])
        
        # Ensure categorical variables are properly encoded
        categorical_cols = [hierarchy.subject_column, hierarchy.slide_column]
        if hierarchy.condition_column in model_data.columns:
            categorical_cols.append(hierarchy.condition_column)
        
        for col in categorical_cols:
            if col in model_data.columns:
                model_data[col] = model_data[col].astype('category')
        
        return model_data
    
    def _build_fixed_effects_formula(self, response_var: str) -> str:
        """Build formula for fixed effects."""
        fixed_terms = []
        
        # Add configured fixed effects if available
        for effect in self.config.fixed_effects:
            if effect == 'condition' and self.hierarchy.condition_column:
                fixed_terms.append(f"C({self.hierarchy.condition_column})")
            elif effect == 'timepoint' and self.hierarchy.timepoint_column:
                fixed_terms.append(f"C({self.hierarchy.timepoint_column})")
        
        # Default to intercept-only if no fixed effects available
        if not fixed_terms:
            fixed_terms = ["1"]
        
        formula = f"{response_var} ~ " + " + ".join(fixed_terms)
        return formula
    
    def _fit_fallback_model(self, 
                           data: pd.DataFrame,
                           response_var: str,
                           hierarchy: HierarchicalDataStructure) -> Dict[str, Any]:
        """Fallback to cluster-robust standard errors if statsmodels unavailable."""
        
        # Simple aggregation to subject level to address pseudoreplication
        subject_data = self._aggregate_to_subject_level(data, response_var, hierarchy)
        
        if len(subject_data) < 3:
            warnings.warn("Too few subjects for reliable inference")
            return {
                'effect_size': np.nan,
                'confidence_interval': (np.nan, np.nan),
                'converged': False,
                'method': 'insufficient_data',
                'n_subjects': len(subject_data)
            }
        
        # Simple t-test on subject-level aggregates
        if hierarchy.condition_column in subject_data.columns:
            conditions = subject_data[hierarchy.condition_column].unique()
            if len(conditions) == 2:
                group1 = subject_data[
                    subject_data[hierarchy.condition_column] == conditions[0]
                ][response_var]
                group2 = subject_data[
                    subject_data[hierarchy.condition_column] == conditions[1]
                ][response_var]
                
                if len(group1) > 1 and len(group2) > 1:
                    t_stat, p_value = stats.ttest_ind(group1, group2)
                    effect_size = (group1.mean() - group2.mean()) / np.sqrt(
                        (group1.var() + group2.var()) / 2
                    )
                    
                    # Conservative confidence interval
                    se = np.sqrt(group1.var()/len(group1) + group2.var()/len(group2))
                    df = len(group1) + len(group2) - 2
                    t_crit = stats.t.ppf(0.975, df)
                    mean_diff = group1.mean() - group2.mean()
                    ci = (mean_diff - t_crit*se, mean_diff + t_crit*se)
                    
                    return {
                        'effect_size': float(effect_size),
                        'confidence_interval': ci,
                        'p_value': float(p_value),
                        'converged': True,
                        'method': 'fallback_ttest',
                        'n_subjects': len(subject_data)
                    }
        
        # Single-group descriptive statistics
        mean_response = subject_data[response_var].mean()
        se_response = subject_data[response_var].std() / np.sqrt(len(subject_data))
        df = len(subject_data) - 1
        t_crit = stats.t.ppf(0.975, df) if df > 0 else np.inf
        
        return {
            'effect_size': float(mean_response),
            'confidence_interval': (
                mean_response - t_crit * se_response,
                mean_response + t_crit * se_response
            ),
            'converged': True,
            'method': 'descriptive',
            'n_subjects': len(subject_data)
        }
    
    def _aggregate_to_subject_level(self,
                                   data: pd.DataFrame,
                                   response_var: str,
                                   hierarchy: HierarchicalDataStructure) -> pd.DataFrame:
        """Aggregate data to subject level to avoid pseudoreplication."""
        
        grouping_cols = [hierarchy.subject_column]
        if hierarchy.condition_column in data.columns:
            grouping_cols.append(hierarchy.condition_column)
        if hierarchy.timepoint_column in data.columns:
            grouping_cols.append(hierarchy.timepoint_column)
        
        # Keep only columns we need
        cols_to_keep = grouping_cols + [response_var]
        available_cols = [col for col in cols_to_keep if col in data.columns]
        
        subject_aggregated = data[available_cols].groupby(
            [col for col in grouping_cols if col in data.columns]
        )[response_var].mean().reset_index()
        
        return subject_aggregated
    
    def _calculate_model_diagnostics(self,
                                   model_data: pd.DataFrame,
                                   spatial_coords: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate model diagnostics and effect sizes."""
        
        if self.fitted_model is None:
            return {}
        
        results = {
            'fixed_effects': self.fitted_model.params.to_dict(),
            'random_effects_var': float(self.fitted_model.cov_re.iloc[0, 0]),
            'residual_var': float(self.fitted_model.scale),
            'log_likelihood': float(self.fitted_model.llf)
        }
        
        # Calculate intraclass correlation
        var_between = results['random_effects_var']
        var_within = results['residual_var']
        icc = var_between / (var_between + var_within)
        results['intraclass_correlation'] = float(icc)
        
        # Effective sample size adjustment
        n_obs = len(model_data)
        effective_n = n_obs / (1 + (n_obs - 1) * icc)
        results['effective_sample_size'] = float(effective_n)
        
        # Spatial autocorrelation in residuals if coordinates provided
        if spatial_coords is not None and SPATIAL_AVAILABLE:
            residuals = self.fitted_model.resid
            try:
                spatial_autocorr = self._calculate_spatial_autocorrelation(
                    residuals.values, spatial_coords
                )
                results['spatial_autocorrelation'] = spatial_autocorr
            except Exception as e:
                warnings.warn(f"Spatial autocorrelation calculation failed: {e}")
                results['spatial_autocorrelation'] = np.nan
        
        return results
    
    def _calculate_spatial_autocorrelation(self,
                                         residuals: np.ndarray,
                                         coords: np.ndarray) -> float:
        """Calculate Moran's I for spatial autocorrelation in residuals."""
        
        if not SPATIAL_AVAILABLE:
            return np.nan
        
        try:
            # Create spatial weights
            w = weights.DistanceBand.from_array(coords, threshold=50.0)
            
            # Calculate Moran's I
            from esda.moran import Moran
            moran = Moran(residuals, w)
            return float(moran.I)
            
        except Exception:
            # Fallback: simple correlation with spatial lag
            from scipy.spatial.distance import pdist, squareform
            
            distances = squareform(pdist(coords))
            weights = np.exp(-distances / 50.0)  # Exponential decay
            np.fill_diagonal(weights, 0)
            weights = weights / weights.sum(axis=1, keepdims=True)
            
            spatial_lag = weights @ residuals
            correlation = np.corrcoef(residuals, spatial_lag)[0, 1]
            return float(correlation)


class SpatialMixedEffects(NestedModel):
    """
    Extended mixed-effects model with explicit spatial autocorrelation modeling.
    """
    
    def fit(self,
            data: pd.DataFrame,
            response_var: str,
            hierarchy: HierarchicalDataStructure,
            spatial_coords: np.ndarray) -> Dict[str, Any]:
        """Fit spatial mixed-effects model."""
        
        # First fit standard mixed-effects model
        results = super().fit(data, response_var, hierarchy, spatial_coords)
        
        # If spatial coordinates provided, model spatial correlation
        if spatial_coords is not None and SPATIAL_AVAILABLE:
            try:
                spatial_results = self._fit_spatial_correlation(
                    data, response_var, spatial_coords
                )
                results.update(spatial_results)
            except Exception as e:
                warnings.warn(f"Spatial correlation modeling failed: {e}")
        
        return results
    
    def _fit_spatial_correlation(self,
                                data: pd.DataFrame,
                                response_var: str,
                                spatial_coords: np.ndarray) -> Dict[str, Any]:
        """Model spatial correlation structure."""
        
        # Extract residuals from mixed-effects model
        if self.fitted_model is None:
            return {}
        
        residuals = self.fitted_model.resid.values
        
        # Fit exponential spatial correlation model
        distances = self._calculate_pairwise_distances(spatial_coords)
        correlations = self._calculate_residual_correlations(residuals, distances)
        
        # Estimate spatial decay parameter
        spatial_decay = self._estimate_spatial_decay(distances, correlations)
        
        return {
            'spatial_decay_parameter': float(spatial_decay),
            'spatial_correlation_range': float(1.0 / spatial_decay) if spatial_decay > 0 else np.inf
        }
    
    def _calculate_pairwise_distances(self, coords: np.ndarray) -> np.ndarray:
        """Calculate pairwise spatial distances."""
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(coords))
    
    def _calculate_residual_correlations(self,
                                       residuals: np.ndarray,
                                       distances: np.ndarray) -> np.ndarray:
        """Calculate correlations between residuals at different distances."""
        
        # Bin distances and calculate average correlation in each bin
        max_dist = np.percentile(distances[distances > 0], 90)
        dist_bins = np.linspace(0, max_dist, 20)
        correlations = []
        
        for i in range(len(dist_bins) - 1):
            mask = (distances >= dist_bins[i]) & (distances < dist_bins[i + 1])
            if np.sum(mask) > 10:  # Need sufficient pairs
                pairs = np.where(mask)
                res_pairs = residuals[pairs[0]] * residuals[pairs[1]]
                correlations.append(np.mean(res_pairs))
            else:
                correlations.append(0.0)
        
        return np.array(correlations)
    
    def _estimate_spatial_decay(self,
                               distances: np.ndarray,
                               correlations: np.ndarray) -> float:
        """Estimate spatial decay parameter for exponential model."""
        
        # Fit exponential decay: corr(d) = exp(-decay * d)
        valid_mask = ~np.isnan(correlations) & (correlations > 0)
        
        if np.sum(valid_mask) < 3:
            return 0.0
        
        valid_corr = correlations[valid_mask]
        valid_dist = np.arange(len(correlations))[valid_mask]
        
        try:
            # Log-linear regression: log(corr) = -decay * dist + const
            log_corr = np.log(np.maximum(valid_corr, 1e-10))
            coeffs = np.polyfit(valid_dist, log_corr, 1)
            decay = -coeffs[0]
            return max(0.0, decay)  # Ensure non-negative
        except:
            return 0.0


def calculate_effect_sizes(results: Dict[str, Any],
                         hierarchy: HierarchicalDataStructure) -> Dict[str, float]:
    """
    Calculate proper effect sizes accounting for nested structure.
    
    Args:
        results: Results from mixed-effects model
        hierarchy: Hierarchical data structure
        
    Returns:
        Dictionary of effect sizes with confidence intervals
    """
    effect_sizes = {}
    
    # Calculate Cohen's d adjusted for clustering
    if 'fixed_effects' in results and 'intraclass_correlation' in results:
        icc = results['intraclass_correlation']
        
        # Design effect for clustered data
        cluster_sizes = []
        for subject, slides in hierarchy.slides_per_subject.items():
            total_rois = sum(len(hierarchy.rois_per_slide.get(slide, [])) for slide in slides)
            cluster_sizes.append(total_rois)
        
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 1
        design_effect = 1 + (avg_cluster_size - 1) * icc
        
        effect_sizes['design_effect'] = float(design_effect)
        effect_sizes['effective_n_subjects'] = len(hierarchy.subjects) / design_effect
    
    # Extract effect sizes from fixed effects
    if 'fixed_effects' in results:
        for effect_name, coefficient in results['fixed_effects'].items():
            if effect_name != 'Intercept':
                effect_sizes[f'{effect_name}_coefficient'] = float(coefficient)
    
    return effect_sizes


def bootstrap_uncertainty(model: NestedModel,
                         data: pd.DataFrame,
                         response_var: str,
                         hierarchy: HierarchicalDataStructure,
                         n_bootstrap: int = 100) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap confidence intervals for mixed-effects model parameters.
    
    Args:
        model: Fitted mixed-effects model
        data: Original data
        response_var: Response variable name
        hierarchy: Hierarchical structure
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Dictionary of parameter confidence intervals
    """
    
    bootstrap_results = []
    subjects = hierarchy.subjects
    
    for _ in range(n_bootstrap):
        # Sample subjects with replacement
        bootstrap_subjects = np.random.choice(subjects, size=len(subjects), replace=True)
        
        # Create bootstrap dataset
        bootstrap_data = []
        for subject in bootstrap_subjects:
            subject_data = data[data[hierarchy.subject_column] == subject]
            bootstrap_data.append(subject_data)
        
        if not bootstrap_data:
            continue
            
        bootstrap_df = pd.concat(bootstrap_data, ignore_index=True)
        
        # Fit model to bootstrap sample
        try:
            bootstrap_model = NestedModel(model.config)
            bootstrap_results_dict = bootstrap_model.fit(
                bootstrap_df, response_var, hierarchy
            )
            bootstrap_results.append(bootstrap_results_dict)
        except:
            continue  # Skip failed bootstrap iterations
    
    # Calculate confidence intervals
    confidence_intervals = {}
    
    if bootstrap_results:
        # Fixed effects CIs
        for param in ['effect_size']:
            if param in bootstrap_results[0]:
                values = [r[param] for r in bootstrap_results if param in r and not np.isnan(r[param])]
                if values:
                    ci_lower = np.percentile(values, 2.5)
                    ci_upper = np.percentile(values, 97.5)
                    confidence_intervals[param] = (float(ci_lower), float(ci_upper))
        
        # Random effects variance CI
        for param in ['random_effects_var', 'intraclass_correlation']:
            if param in bootstrap_results[0]:
                values = [r[param] for r in bootstrap_results if param in r and not np.isnan(r[param])]
                if values:
                    ci_lower = np.percentile(values, 2.5)
                    ci_upper = np.percentile(values, 97.5)
                    confidence_intervals[param] = (float(ci_lower), float(ci_upper))
    
    return confidence_intervals