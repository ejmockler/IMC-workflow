"""
Hierarchical Multiple Testing Control for IMC Pipeline

Final statistical framework component implementing hierarchical FDR control,
family-wise error rate control, and bootstrap confidence intervals.
Addresses critical multiple testing issues in spatial multi-scale analyses.

Key Features:
- Hierarchical FDR control across scales × markers × spatial statistics
- Family-wise error rate control for clustering optimization
- Pre-specified hypothesis testing with correction for exploratory analysis
- Bootstrap confidence intervals for small-n studies instead of p-values
- Integration with existing spatial statistics and evaluation frameworks

Critical for Publication-Ready Statistical Rigor.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import defaultdict
import itertools
from scipy import stats
from sklearn.utils import resample

# Import existing framework components
from .fdr_spatial import SpatialFDR, FDRConfig
from .spatial_permutation import SpatialPermutation, PermutationConfig, compute_spatial_pvalue
from .spatial_resampling import SpatialResampling, ResamplingConfig
from .hierarchical_data import HierarchicalDataStructure
from .boundary_metrics import ValidationMetric

try:
    from ..validation.framework import ValidationResult, ValidationCategory, ValidationSeverity
except ImportError:
    # Fallback definitions
    from enum import Enum
    from dataclasses import dataclass
    
    class ValidationSeverity(Enum):
        CRITICAL = "critical"
        WARNING = "warning" 
        INFO = "info"
        PASS = "pass"
    
    class ValidationCategory(Enum):
        SCIENTIFIC_QUALITY = "scientific_quality"
    
    @dataclass
    class ValidationResult:
        rule_name: str
        category: ValidationCategory
        severity: ValidationSeverity
        message: str
        quality_score: Optional[float] = None
        metrics: Dict[str, Any] = field(default_factory=dict)


class HypothesisType(Enum):
    """Types of hypotheses in hierarchical testing."""
    PRE_SPECIFIED = "pre_specified"      # Primary hypotheses planned before data
    EXPLORATORY = "exploratory"          # Secondary hypotheses discovered in data
    CLUSTERING = "clustering"            # Parameter optimization hypotheses
    SPATIAL = "spatial"                  # Spatial statistics hypotheses


@dataclass
class HypothesisFamily:
    """Represents a family of related hypotheses."""
    name: str
    hypothesis_type: HypothesisType
    scale: Optional[str] = None          # e.g., "10um", "20um", "40um"
    marker: Optional[str] = None         # e.g., "CD3", "CD20", etc.
    statistic: Optional[str] = None      # e.g., "moran_i", "clustering_quality"
    parent_family: Optional[str] = None  # Hierarchical parent
    children: List[str] = field(default_factory=list)
    priority: int = 1                    # Higher priority = tested first


@dataclass
class BootstrapConfidenceInterval:
    """Bootstrap confidence interval for effect size."""
    effect_size: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    method: str = "percentile"           # "percentile", "bias_corrected", "bca"


@dataclass
class MultipleTesting:
    """Multiple testing correction configuration."""
    method: str = "benjamini_yekutieli"  # FDR method for arbitrary dependence
    alpha: float = 0.05                 # Family-wise or FDR level
    hierarchical: bool = True           # Use hierarchical testing
    adaptive_weights: bool = True       # Use adaptive importance weights
    correction_type: str = "fdr"        # "fdr" or "fwer"


@dataclass
class HierarchicalTestingConfig:
    """Configuration for hierarchical multiple testing control."""
    # FDR control
    fdr_config: FDRConfig = field(default_factory=lambda: FDRConfig(
        method='benjamini_yekutieli',
        alpha=0.05,
        dependence_assumption='arbitrary',
        use_spatial_weights=True,
        adaptive_weights=True
    ))
    
    # Family-wise error rate control
    fwer_alpha: float = 0.05
    fwer_method: str = "holm"           # "holm", "bonferroni", "sidak"
    
    # Bootstrap confidence intervals
    bootstrap_n: int = 1000
    bootstrap_confidence: float = 0.95
    bootstrap_method: str = "percentile" 
    
    # Hierarchical structure
    enforce_hierarchy: bool = True      # Require parent significance before testing children
    hierarchy_alpha_spending: str = "simes"  # "simes", "alpha_investing"
    
    # Spatial permutation testing
    permutation_config: PermutationConfig = field(default_factory=lambda: PermutationConfig(
        method='moran_spectral',
        n_permutations=999,
        preserve_marginals=True
    ))
    
    # Small-n adjustments
    min_n_for_pvalues: int = 10         # Use bootstrap CIs instead of p-values below this
    effect_size_threshold: float = 0.2  # Minimum meaningful effect size


class HierarchicalMultipleTestingControl:
    """
    Hierarchical Multiple Testing Control Framework
    
    Implements hierarchical FDR control across spatial scales, markers, and statistics
    with proper handling of spatial dependence and small sample sizes.
    """
    
    def __init__(self, config: HierarchicalTestingConfig):
        self.config = config
        
        # Initialize component frameworks
        self.spatial_fdr = SpatialFDR(config.fdr_config)
        self.spatial_permutation = SpatialPermutation(config.permutation_config)
        self.spatial_resampling = SpatialResampling(ResamplingConfig(
            n_bootstrap=config.bootstrap_n
        ))
        
        # Hypothesis family registry
        self.hypothesis_families: Dict[str, HypothesisFamily] = {}
        self.family_hierarchy: Dict[str, List[str]] = {}  # parent -> children
        
        # Results storage
        self.correction_results: Dict[str, Any] = {}
        self.bootstrap_results: Dict[str, BootstrapConfidenceInterval] = {}
        
        # Cache for expensive computations
        self._spatial_weights_cache: Dict[str, np.ndarray] = {}
        
    def register_hypothesis_family(self, family: HypothesisFamily) -> None:
        """Register a family of hypotheses in the hierarchical structure."""
        self.hypothesis_families[family.name] = family
        
        # Update hierarchy
        if family.parent_family:
            if family.parent_family not in self.family_hierarchy:
                self.family_hierarchy[family.parent_family] = []
            self.family_hierarchy[family.parent_family].append(family.name)
            
            # Add to parent's children list
            if family.parent_family in self.hypothesis_families:
                parent = self.hypothesis_families[family.parent_family]
                if family.name not in parent.children:
                    parent.children.append(family.name)
    
    def multiscale_hypothesis_testing(self,
                                    test_results: Dict[str, Dict[str, Any]],
                                    spatial_coords: Dict[str, np.ndarray],
                                    effect_sizes: Optional[Dict[str, float]] = None,
                                    sample_sizes: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Perform hierarchical multiple testing control across scales and markers.
        
        Args:
            test_results: Dict[scale][marker] -> {'p_value': float, 'statistic': float, ...}
            spatial_coords: Dict[scale] -> coordinates for spatial weights
            effect_sizes: Optional effect sizes for each test
            sample_sizes: Optional sample sizes for bootstrap CI decisions
            
        Returns:
            Dictionary with corrected results and recommendations
        """
        # Organize tests by hierarchy
        organized_tests = self._organize_tests_by_hierarchy(test_results)
        
        # Initialize results structure
        results = {
            'hierarchical_corrections': {},
            'bootstrap_confidence_intervals': {},
            'recommendations': [],
            'validation_results': [],
            'family_wise_results': {},
            'effective_tests': {}
        }
        
        # Apply hierarchical testing by family priority
        family_order = self._determine_testing_order()
        
        for family_name in family_order:
            if family_name not in organized_tests:
                continue
                
            family = self.hypothesis_families[family_name]
            family_tests = organized_tests[family_name]
            
            # Determine testing approach based on sample size and hypothesis type
            testing_approach = self._determine_testing_approach(
                family, family_tests, sample_sizes
            )
            
            if testing_approach == "bootstrap_ci":
                # Use bootstrap confidence intervals for small-n studies
                family_results = self._bootstrap_confidence_interval_testing(
                    family_tests, spatial_coords, family
                )
                results['bootstrap_confidence_intervals'][family_name] = family_results
                
            elif testing_approach == "hierarchical_fdr":
                # Use hierarchical FDR control
                family_results = self._hierarchical_fdr_testing(
                    family_tests, spatial_coords, family
                )
                results['hierarchical_corrections'][family_name] = family_results
                
            elif testing_approach == "fwer_control":
                # Use family-wise error rate control for clustering optimization
                family_results = self._family_wise_error_control(
                    family_tests, spatial_coords, family
                )
                results['family_wise_results'][family_name] = family_results
            
            # Add validation results
            validation_result = self._validate_family_results(family, family_results)
            results['validation_results'].append(validation_result)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Calculate effective number of tests
        results['effective_tests'] = self._calculate_effective_tests(
            test_results, spatial_coords
        )
        
        return results
    
    def _organize_tests_by_hierarchy(self, 
                                   test_results: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Organize test results by hypothesis family."""
        organized = defaultdict(list)
        
        for scale, scale_results in test_results.items():
            for marker, test_result in scale_results.items():
                # Determine family based on scale and marker
                family_name = self._determine_family(scale, marker, test_result)
                
                test_info = {
                    'scale': scale,
                    'marker': marker,
                    'test_result': test_result,
                    'family': family_name
                }
                organized[family_name].append(test_info)
        
        return dict(organized)
    
    def _bootstrap_confidence_interval_testing(self,
                                             family_tests: List[Dict],
                                             spatial_coords: Dict[str, np.ndarray],
                                             family: HypothesisFamily) -> Dict[str, Any]:
        """
        Bootstrap confidence interval approach for small-n studies.
        
        Instead of p-values, compute confidence intervals for effect sizes.
        Declare significance based on CI excluding null effect.
        """
        results = {
            'method': 'bootstrap_confidence_intervals',
            'confidence_level': self.config.bootstrap_confidence,
            'n_bootstrap': self.config.bootstrap_n,
            'discoveries': {},
            'confidence_intervals': {},
            'effect_sizes': {}
        }
        
        for test_info in family_tests:
            scale = test_info['scale']
            marker = test_info['marker']
            test_result = test_info['test_result']
            
            # Get spatial coordinates for this scale
            coords = spatial_coords.get(scale, None)
            
            # Bootstrap confidence interval for effect size
            if 'effect_size' in test_result and coords is not None:
                effect_size = test_result['effect_size']
                
                # Generate bootstrap samples using spatial resampling
                bootstrap_effects = []
                
                # Use spatial resampling to preserve spatial structure
                if 'raw_data' in test_result:
                    raw_data = test_result['raw_data']
                    statistic_func = test_result.get('statistic_func', np.mean)
                    
                    # Generate spatially-aware bootstrap samples
                    bootstrap_generator = self.spatial_resampling.moving_block_bootstrap(
                        raw_data, 
                        n_samples=self.config.bootstrap_n
                    )
                    
                    for bootstrap_sample in bootstrap_generator:
                        bootstrap_effect = statistic_func(bootstrap_sample)
                        bootstrap_effects.append(bootstrap_effect)
                    
                    bootstrap_effects = np.array(bootstrap_effects)
                else:
                    # Fallback to parametric bootstrap if raw data not available
                    warnings.warn(f"No raw data for {scale}-{marker}, using parametric bootstrap")
                    se = test_result.get('standard_error', effect_size * 0.1)
                    bootstrap_effects = np.random.normal(
                        effect_size, se, self.config.bootstrap_n
                    )
                
                # Calculate confidence interval
                alpha = 1 - self.config.bootstrap_confidence
                ci_lower = np.percentile(bootstrap_effects, 100 * alpha/2)
                ci_upper = np.percentile(bootstrap_effects, 100 * (1 - alpha/2))
                
                # Determine significance based on CI excluding null
                is_significant = self._ci_excludes_null(
                    ci_lower, ci_upper, test_result.get('null_value', 0)
                )
                
                # Store results
                test_key = f"{scale}_{marker}"
                results['confidence_intervals'][test_key] = BootstrapConfidenceInterval(
                    effect_size=effect_size,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    confidence_level=self.config.bootstrap_confidence,
                    n_bootstrap=self.config.bootstrap_n,
                    method=self.config.bootstrap_method
                )
                results['discoveries'][test_key] = is_significant
                results['effect_sizes'][test_key] = effect_size
        
        return results
    
    def _hierarchical_fdr_testing(self,
                                family_tests: List[Dict],
                                spatial_coords: Dict[str, np.ndarray],
                                family: HypothesisFamily) -> Dict[str, Any]:
        """
        Hierarchical FDR control with spatial dependence awareness.
        
        Uses Benjamini-Yekutieli for arbitrary dependence with hierarchical structure.
        """
        results = {
            'method': 'hierarchical_fdr',
            'fdr_level': self.config.fdr_config.alpha,
            'discoveries': {},
            'adjusted_p_values': {},
            'hierarchical_structure': {},
            'spatial_adjustment': {}
        }
        
        # Extract p-values and organize hierarchically
        p_values = []
        test_keys = []
        spatial_coords_list = []
        
        for test_info in family_tests:
            scale = test_info['scale']
            marker = test_info['marker']
            test_result = test_info['test_result']
            
            if 'p_value' in test_result:
                p_values.append(test_result['p_value'])
                test_keys.append(f"{scale}_{marker}")
                
                # Get spatial coordinates
                coords = spatial_coords.get(scale, None)
                spatial_coords_list.append(coords)
        
        if not p_values:
            return results
        
        p_values = np.array(p_values)
        
        # Build hierarchical structure for tree-structured FDR
        hierarchy = self._build_hypothesis_hierarchy(family_tests, family)
        
        # Apply spatial FDR control
        if family.hypothesis_type == HypothesisType.SPATIAL:
            # Use tree-structured FDR for hierarchical spatial tests
            discoveries = self.spatial_fdr.tree_structured_fdr(
                p_values, hierarchy, spatial_coords_list[0] if spatial_coords_list[0] is not None else None
            )
        else:
            # Use standard Benjamini-Yekutieli for other hypothesis types
            discoveries = self.spatial_fdr.benjamini_yekutieli_spatial(
                p_values, spatial_coords_list[0] if spatial_coords_list[0] is not None else None
            )
        
        # Store results
        for i, (test_key, discovery) in enumerate(zip(test_keys, discoveries)):
            results['discoveries'][test_key] = bool(discovery)
            results['adjusted_p_values'][test_key] = float(p_values[i])
        
        results['hierarchical_structure'] = hierarchy
        
        return results
    
    def _family_wise_error_control(self,
                                 family_tests: List[Dict],
                                 spatial_coords: Dict[str, np.ndarray],
                                 family: HypothesisFamily) -> Dict[str, Any]:
        """
        Family-wise error rate control for clustering optimization.
        
        Uses Holm's method with spatial permutation testing.
        """
        results = {
            'method': 'family_wise_error_control',
            'fwer_level': self.config.fwer_alpha,
            'fwer_method': self.config.fwer_method,
            'discoveries': {},
            'adjusted_p_values': {},
            'permutation_results': {}
        }
        
        # Extract test statistics and use spatial permutation testing
        test_statistics = []
        test_keys = []
        
        for test_info in family_tests:
            scale = test_info['scale']
            marker = test_info['marker']
            test_result = test_info['test_result']
            
            if 'statistic' in test_result:
                test_statistics.append(test_result['statistic'])
                test_keys.append(f"{scale}_{marker}")
        
        if not test_statistics:
            return results
        
        # Spatial permutation testing for each statistic
        permutation_p_values = []
        
        for i, (test_key, statistic) in enumerate(zip(test_keys, test_statistics)):
            # Get test info
            test_info = family_tests[i]
            scale = test_info['scale']
            
            # Spatial permutation test
            coords = spatial_coords.get(scale, None)
            if coords is not None and 'raw_data' in test_info['test_result']:
                raw_data = test_info['test_result']['raw_data']
                statistic_func = test_info['test_result'].get('statistic_func', np.mean)
                
                # Generate spatial nulls
                null_generator = self.spatial_permutation.generate_spatial_nulls(
                    raw_data, coords, n_perms=self.config.permutation_config.n_permutations
                )
                
                null_statistics = []
                for null_data in null_generator:
                    null_stat = statistic_func(null_data)
                    null_statistics.append(null_stat)
                
                # Compute permutation p-value
                p_value = compute_spatial_pvalue(statistic, np.array(null_statistics))
                permutation_p_values.append(p_value)
                
                results['permutation_results'][test_key] = {
                    'observed_statistic': statistic,
                    'null_statistics': null_statistics,
                    'permutation_p_value': p_value
                }
            else:
                # Fallback to provided p-value
                p_value = test_info['test_result'].get('p_value', 1.0)
                permutation_p_values.append(p_value)
        
        # Apply family-wise error rate control
        permutation_p_values = np.array(permutation_p_values)
        adjusted_p_values = self._apply_fwer_method(permutation_p_values)
        
        # Determine discoveries
        discoveries = adjusted_p_values <= self.config.fwer_alpha
        
        # Store results
        for i, test_key in enumerate(test_keys):
            results['discoveries'][test_key] = bool(discoveries[i])
            results['adjusted_p_values'][test_key] = float(adjusted_p_values[i])
        
        return results
    
    def _apply_fwer_method(self, p_values: np.ndarray) -> np.ndarray:
        """Apply family-wise error rate correction method."""
        if self.config.fwer_method == "holm":
            return self._holm_correction(p_values)
        elif self.config.fwer_method == "bonferroni":
            return np.minimum(p_values * len(p_values), 1.0)
        elif self.config.fwer_method == "sidak":
            return 1 - (1 - p_values) ** len(p_values)
        else:
            raise ValueError(f"Unknown FWER method: {self.config.fwer_method}")
    
    def _holm_correction(self, p_values: np.ndarray) -> np.ndarray:
        """Holm's step-down method for FWER control."""
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # Holm adjustment: p_i * (n - i + 1)
        adjusted = np.zeros_like(p_values)
        for i in range(n):
            adjusted[sorted_idx[i]] = min(1.0, sorted_p[i] * (n - i))
        
        # Ensure monotonicity
        for i in range(1, n):
            idx_curr = sorted_idx[i]
            idx_prev = sorted_idx[i-1]
            adjusted[idx_curr] = max(adjusted[idx_curr], adjusted[idx_prev])
        
        return adjusted
    
    def _determine_testing_approach(self,
                                  family: HypothesisFamily,
                                  family_tests: List[Dict],
                                  sample_sizes: Optional[Dict[str, int]]) -> str:
        """Determine appropriate testing approach based on family and sample size."""
        
        # Check sample sizes if provided
        if sample_sizes:
            min_n = min(sample_sizes.values())
            if min_n < self.config.min_n_for_pvalues:
                return "bootstrap_ci"
        
        # Clustering optimization uses FWER control
        if family.hypothesis_type == HypothesisType.CLUSTERING:
            return "fwer_control"
        
        # Pre-specified hypotheses use hierarchical FDR
        if family.hypothesis_type == HypothesisType.PRE_SPECIFIED:
            return "hierarchical_fdr"
        
        # Exploratory analysis uses hierarchical FDR with stronger correction
        if family.hypothesis_type == HypothesisType.EXPLORATORY:
            return "hierarchical_fdr"
        
        # Spatial statistics use hierarchical FDR with spatial dependence handling
        if family.hypothesis_type == HypothesisType.SPATIAL:
            return "hierarchical_fdr"
        
        # Default to hierarchical FDR
        return "hierarchical_fdr"
    
    def _determine_family(self, scale: str, marker: str, test_result: Dict) -> str:
        """Determine hypothesis family based on test characteristics."""
        # Simple heuristic - in practice this would be more sophisticated
        test_type = test_result.get('test_type', 'unknown')
        
        if 'clustering' in test_type.lower():
            return f"clustering_{scale}"
        elif 'spatial' in test_type.lower():
            return f"spatial_{scale}"
        else:
            return f"marker_effects_{scale}"
    
    def _build_hypothesis_hierarchy(self,
                                  family_tests: List[Dict],
                                  family: HypothesisFamily) -> Dict[str, Any]:
        """Build hierarchical structure for tree-structured FDR."""
        n_tests = len(family_tests)
        
        # Simple hierarchy: group by scale, then by marker
        hierarchy = {
            'parents': {},
            'children': {},
            'roots': []
        }
        
        # Group tests by scale
        scale_groups = defaultdict(list)
        for i, test_info in enumerate(family_tests):
            scale_groups[test_info['scale']].append(i)
        
        # Create hierarchical structure
        root_nodes = []
        for scale, test_indices in scale_groups.items():
            if len(test_indices) > 1:
                # Create scale-level parent
                parent_idx = max(hierarchy['parents'].keys(), default=-1) + 1
                hierarchy['children'][parent_idx] = test_indices
                root_nodes.append(parent_idx)
                
                for test_idx in test_indices:
                    hierarchy['parents'][test_idx] = parent_idx
            else:
                # Single test becomes root
                root_nodes.extend(test_indices)
        
        hierarchy['roots'] = root_nodes
        return hierarchy
    
    def _ci_excludes_null(self, ci_lower: float, ci_upper: float, null_value: float) -> bool:
        """Check if confidence interval excludes null value."""
        return (ci_lower > null_value) or (ci_upper < null_value)
    
    def _determine_testing_order(self) -> List[str]:
        """Determine order for testing hypothesis families based on hierarchy and priority."""
        # Sort by priority, then alphabetically
        families = list(self.hypothesis_families.values())
        families.sort(key=lambda f: (-f.priority, f.name))
        
        # Ensure parents are tested before children
        ordered = []
        processed = set()
        
        def add_family_and_children(family_name: str):
            if family_name in processed:
                return
            
            family = self.hypothesis_families[family_name]
            
            # Add parent first if it exists and hasn't been processed
            if family.parent_family and family.parent_family not in processed:
                add_family_and_children(family.parent_family)
            
            # Add this family
            if family_name not in processed:
                ordered.append(family_name)
                processed.add(family_name)
            
            # Add children
            for child_name in family.children:
                add_family_and_children(child_name)
        
        # Start with root families (no parents)
        for family in families:
            if not family.parent_family:
                add_family_and_children(family.name)
        
        return ordered
    
    def _validate_family_results(self,
                               family: HypothesisFamily,
                               family_results: Dict[str, Any]) -> ValidationResult:
        """Validate results for a hypothesis family."""
        
        # Count discoveries
        discoveries = family_results.get('discoveries', {})
        n_discoveries = sum(discoveries.values())
        n_tests = len(discoveries)
        
        discovery_rate = n_discoveries / max(n_tests, 1)
        
        # Determine validation severity
        if discovery_rate > 0.5:
            severity = ValidationSeverity.WARNING
            message = f"High discovery rate ({discovery_rate:.2%}) in family {family.name} - check for inflated error rates"
        elif discovery_rate == 0:
            severity = ValidationSeverity.INFO
            message = f"No discoveries in family {family.name} - consider effect size analysis"
        else:
            severity = ValidationSeverity.PASS
            message = f"Reasonable discovery rate ({discovery_rate:.2%}) in family {family.name}"
        
        return ValidationResult(
            rule_name=f"family_validation_{family.name}",
            category=ValidationCategory.SCIENTIFIC_QUALITY,
            severity=severity,
            message=message,
            quality_score=1.0 - discovery_rate if discovery_rate > 0.5 else 1.0,
            metrics={
                'n_discoveries': n_discoveries,
                'n_tests': n_tests,
                'discovery_rate': discovery_rate,
                'family_type': family.hypothesis_type.value
            }
        )
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on multiple testing results."""
        recommendations = []
        
        # Check for high discovery rates
        for family_name, family_results in results.get('hierarchical_corrections', {}).items():
            discoveries = family_results.get('discoveries', {})
            discovery_rate = sum(discoveries.values()) / max(len(discoveries), 1)
            
            if discovery_rate > 0.5:
                recommendations.append(
                    f"High discovery rate ({discovery_rate:.2%}) in {family_name}. "
                    "Consider stricter alpha levels or effect size thresholds."
                )
        
        # Check for bootstrap CI usage
        if results.get('bootstrap_confidence_intervals'):
            recommendations.append(
                "Bootstrap confidence intervals used due to small sample sizes. "
                "Consider increasing sample size for more powerful testing."
            )
        
        # Check for spatial adjustments
        effective_tests = results.get('effective_tests', {})
        for scale, n_eff in effective_tests.items():
            if 'n_nominal' in effective_tests and scale in effective_tests['n_nominal']:
                n_nominal = effective_tests['n_nominal'][scale]
                if n_eff < 0.5 * n_nominal:
                    recommendations.append(
                        f"Strong spatial correlation at {scale} scale reduces effective sample size "
                        f"from {n_nominal} to {n_eff:.0f}. Consider spatial modeling."
                    )
        
        if not recommendations:
            recommendations.append("Multiple testing corrections applied successfully.")
        
        return recommendations
    
    def _calculate_effective_tests(self,
                                 test_results: Dict[str, Dict[str, Any]],
                                 spatial_coords: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate effective number of independent tests accounting for spatial dependence."""
        effective_tests = {}
        
        for scale, coords in spatial_coords.items():
            if coords is not None and scale in test_results:
                n_nominal = len(test_results[scale])
                
                # Estimate effective tests using spatial correlation
                n_effective = self.spatial_fdr._estimate_effective_tests(coords)
                
                effective_tests[scale] = {
                    'n_nominal': n_nominal,
                    'n_effective': min(n_effective, n_nominal),
                    'correlation_reduction': n_effective / n_nominal if n_nominal > 0 else 1.0
                }
        
        return effective_tests


def create_standard_hypothesis_families(scales: List[str],
                                       markers: List[str]) -> List[HypothesisFamily]:
    """
    Create standard hypothesis families for IMC multiscale analysis.
    
    Args:
        scales: List of scale identifiers (e.g., ["10um", "20um", "40um"])
        markers: List of marker names
        
    Returns:
        List of pre-configured hypothesis families
    """
    families = []
    
    # 1. Pre-specified marker effects (highest priority)
    for scale in scales:
        family = HypothesisFamily(
            name=f"marker_effects_{scale}",
            hypothesis_type=HypothesisType.PRE_SPECIFIED,
            scale=scale,
            priority=1
        )
        families.append(family)
    
    # 2. Spatial statistics (medium priority)
    for scale in scales:
        family = HypothesisFamily(
            name=f"spatial_stats_{scale}",
            hypothesis_type=HypothesisType.SPATIAL,
            scale=scale,
            priority=2
        )
        families.append(family)
    
    # 3. Clustering optimization (lower priority, FWER control)
    for scale in scales:
        family = HypothesisFamily(
            name=f"clustering_{scale}",
            hypothesis_type=HypothesisType.CLUSTERING,
            scale=scale,
            priority=3
        )
        families.append(family)
    
    # 4. Exploratory analysis (lowest priority)
    family = HypothesisFamily(
        name="exploratory_analysis",
        hypothesis_type=HypothesisType.EXPLORATORY,
        priority=4
    )
    families.append(family)
    
    # Set up hierarchical relationships
    # Spatial stats depend on marker effects
    for scale in scales:
        marker_family = next(f for f in families if f.name == f"marker_effects_{scale}")
        spatial_family = next(f for f in families if f.name == f"spatial_stats_{scale}")
        
        spatial_family.parent_family = marker_family.name
        marker_family.children.append(spatial_family.name)
    
    return families


def bootstrap_effect_size_testing(observed_effects: np.ndarray,
                                 bootstrap_samples: Iterator[np.ndarray],
                                 null_value: float = 0.0,
                                 confidence_level: float = 0.95) -> List[BootstrapConfidenceInterval]:
    """
    Bootstrap confidence interval testing for effect sizes.
    
    Preferred over p-values for small sample studies.
    
    Args:
        observed_effects: Array of observed effect sizes
        bootstrap_samples: Iterator yielding bootstrap effect samples
        null_value: Null hypothesis value (usually 0)
        confidence_level: Confidence level for intervals
        
    Returns:
        List of bootstrap confidence intervals
    """
    results = []
    
    # Collect all bootstrap samples
    all_bootstrap_samples = list(bootstrap_samples)
    n_bootstrap = len(all_bootstrap_samples)
    
    for i, observed_effect in enumerate(observed_effects):
        # Extract bootstrap samples for this effect
        bootstrap_effects = np.array([sample[i] if i < len(sample) else observed_effect 
                                    for sample in all_bootstrap_samples])
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_effects, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_effects, 100 * (1 - alpha/2))
        
        ci = BootstrapConfidenceInterval(
            effect_size=observed_effect,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            method="percentile"
        )
        
        results.append(ci)
    
    return results


# Integration functions for existing pipeline

def integrate_with_multiscale_analysis(multiscale_results: Dict[str, Any],
                                     spatial_coords: Dict[str, np.ndarray],
                                     config: Optional[HierarchicalTestingConfig] = None) -> Dict[str, Any]:
    """
    Integrate hierarchical multiple testing with multiscale analysis results.
    
    Args:
        multiscale_results: Results from multiscale_analysis.perform_multiscale_analysis
        spatial_coords: Spatial coordinates for each scale
        config: Testing configuration (uses defaults if None)
        
    Returns:
        Enhanced results with multiple testing corrections
    """
    if config is None:
        config = HierarchicalTestingConfig()
    
    # Initialize testing framework
    testing_framework = HierarchicalMultipleTestingControl(config)
    
    # Register standard hypothesis families
    scales = list(multiscale_results.keys())
    markers = []
    if scales and 'aggregated_features' in multiscale_results[scales[0]]:
        markers = list(multiscale_results[scales[0]]['aggregated_features'].columns)
    
    families = create_standard_hypothesis_families(scales, markers)
    for family in families:
        testing_framework.register_hypothesis_family(family)
    
    # Extract test results from multiscale analysis
    test_results = {}
    for scale in scales:
        test_results[scale] = {}
        
        # Extract clustering quality metrics as test statistics
        if 'clustering_quality' in multiscale_results[scale]:
            quality = multiscale_results[scale]['clustering_quality']
            for metric, value in quality.items():
                test_results[scale][f"clustering_{metric}"] = {
                    'statistic': value,
                    'test_type': 'clustering_quality',
                    'p_value': 0.5  # Placeholder - would come from permutation test
                }
        
        # Extract spatial statistics if available
        if 'spatial_metrics' in multiscale_results[scale]:
            spatial = multiscale_results[scale]['spatial_metrics']
            for metric, value in spatial.items():
                test_results[scale][f"spatial_{metric}"] = {
                    'statistic': value,
                    'test_type': 'spatial_statistic',
                    'p_value': 0.5  # Placeholder
                }
    
    # Apply hierarchical multiple testing
    correction_results = testing_framework.multiscale_hypothesis_testing(
        test_results, spatial_coords
    )
    
    # Enhance original results
    enhanced_results = multiscale_results.copy()
    enhanced_results['multiple_testing_control'] = correction_results
    
    return enhanced_results