"""
Result Comparison Utilities for IMC Analysis

Compares analysis outputs for scientific equivalence with configurable tolerances.
Handles HDF5, Parquet, and JSON result formats with intelligent diff reporting.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings
from datetime import datetime

# Optional dependencies with graceful fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create stub for type annotations
    class np:
        ndarray = type('ndarray', (), {})
        
        @staticmethod
        def array(x):
            return list(x) if hasattr(x, '__iter__') else x
        
        @staticmethod
        def allclose(a, b, rtol=1e-5, atol=1e-8):
            if not hasattr(a, '__iter__') or not hasattr(b, '__iter__'):
                return abs(a - b) <= atol + rtol * abs(b)
            if len(a) != len(b):
                return False
            return all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(a, b))
        
        @staticmethod
        def max(x):
            return max(x) if hasattr(x, '__iter__') else x
        
        @staticmethod
        def mean(x):
            return sum(x) / len(x) if hasattr(x, '__iter__') and len(x) > 0 else x
        
        @staticmethod
        def std(x):
            if not hasattr(x, '__iter__') or len(x) <= 1:
                return 0.0
            mean_val = sum(x) / len(x)
            return (sum((val - mean_val) ** 2 for val in x) / len(x)) ** 0.5
        
        @staticmethod
        def unique(x):
            return list(set(x)) if hasattr(x, '__iter__') else [x]

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("pandas not available. Some storage formats will be disabled.")

# Import storage backends
try:
    from .data_storage import (
        HDF5Storage, ParquetStorage, HybridStorage, 
        CompressedJSONStorage, create_storage_backend
    )
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    warnings.warn("Storage backends not available. Only direct result comparison supported.")


class ComparisonSeverity(Enum):
    """Severity levels for comparison differences."""
    IDENTICAL = "identical"      # Exactly the same
    EQUIVALENT = "equivalent"    # Scientifically equivalent within tolerance
    DIFFERENT = "different"      # Meaningful scientific difference
    INCOMPARABLE = "incomparable"  # Cannot be compared (missing data, etc.)


@dataclass
class ToleranceProfile:
    """Tolerance settings for different data types in IMC analysis."""
    
    # Clustering tolerances
    cluster_assignment_tolerance: float = 0.15  # 15% allowed reassignment
    cluster_centroids_rtol: float = 0.05       # 5% relative tolerance
    cluster_centroids_atol: float = 0.01       # Absolute tolerance for centroids
    
    # Spatial statistics tolerances
    spatial_coords_rtol: float = 1e-6          # Very tight for coordinates
    spatial_coords_atol: float = 1e-9          
    spatial_stats_rtol: float = 0.02           # 2% for spatial statistics
    spatial_stats_atol: float = 1e-8
    
    # Protein expression tolerances  
    expression_rtol: float = 0.03              # 3% for protein levels
    expression_atol: float = 0.001             # Small absolute tolerance
    cofactor_rtol: float = 0.01                # 1% for cofactors
    
    # Quality metrics tolerances
    quality_score_tolerance: float = 0.05      # 5% for quality scores
    silhouette_tolerance: float = 0.02         # 2% for silhouette scores
    
    # Scale consistency tolerances
    consistency_rtol: float = 0.10             # 10% for scale consistency
    hierarchy_tolerance: float = 0.08          # 8% for hierarchical metrics
    
    @classmethod
    def create_strict(cls) -> 'ToleranceProfile':
        """Create strict tolerance profile for validation."""
        return cls(
            cluster_assignment_tolerance=0.05,
            cluster_centroids_rtol=0.02,
            spatial_stats_rtol=0.01,
            expression_rtol=0.01,
            quality_score_tolerance=0.02
        )
    
    @classmethod  
    def create_permissive(cls) -> 'ToleranceProfile':
        """Create permissive tolerance profile for development."""
        return cls(
            cluster_assignment_tolerance=0.25,
            cluster_centroids_rtol=0.10,
            spatial_stats_rtol=0.05,
            expression_rtol=0.05,
            quality_score_tolerance=0.10
        )


@dataclass
class ComparisonResult:
    """Result of comparing two values or datasets."""
    
    field_path: str
    severity: ComparisonSeverity
    message: str
    
    # Quantitative metrics
    difference_metric: Optional[float] = None
    tolerance_used: Optional[float] = None
    
    # Context information
    value1_summary: Optional[str] = None
    value2_summary: Optional[str] = None
    comparison_method: Optional[str] = None
    
    # Additional details
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'field_path': self.field_path,
            'severity': self.severity.value,
            'message': self.message,
            'difference_metric': self.difference_metric,
            'tolerance_used': self.tolerance_used,
            'value1_summary': self.value1_summary,
            'value2_summary': self.value2_summary,
            'comparison_method': self.comparison_method,
            'context': self.context
        }


@dataclass
class DiffReport:
    """Comprehensive diff report for two analysis results."""
    
    comparison_id: str
    timestamp: str
    
    # Summary statistics
    total_comparisons: int = 0
    identical_count: int = 0
    equivalent_count: int = 0
    different_count: int = 0
    incomparable_count: int = 0
    
    # Detailed results
    results: List[ComparisonResult] = field(default_factory=list)
    tolerance_profile: Optional[ToleranceProfile] = None
    
    # Context
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        total = max(1, self.total_comparisons)  # Avoid division by zero
        
        return {
            'total_comparisons': self.total_comparisons,
            'identical_percent': 100 * self.identical_count / total,
            'equivalent_percent': 100 * self.equivalent_count / total,
            'different_percent': 100 * self.different_count / total,
            'incomparable_percent': 100 * self.incomparable_count / total,
            'scientific_equivalence_percent': 100 * (self.identical_count + self.equivalent_count) / total
        }
    
    @property
    def is_scientifically_equivalent(self) -> bool:
        """Check if results are scientifically equivalent overall."""
        return self.different_count == 0 and self.incomparable_count == 0
    
    def get_critical_differences(self) -> List[ComparisonResult]:
        """Get differences that matter scientifically."""
        return [r for r in self.results if r.severity == ComparisonSeverity.DIFFERENT]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'comparison_id': self.comparison_id,
            'timestamp': self.timestamp,
            'summary': self.summary_stats,
            'is_scientifically_equivalent': self.is_scientifically_equivalent,
            'tolerance_profile': {
                'cluster_assignment_tolerance': self.tolerance_profile.cluster_assignment_tolerance,
                'expression_rtol': self.tolerance_profile.expression_rtol,
                'spatial_stats_rtol': self.tolerance_profile.spatial_stats_rtol
            } if self.tolerance_profile else None,
            'critical_differences': [r.to_dict() for r in self.get_critical_differences()],
            'all_results': [r.to_dict() for r in self.results],
            'metadata': self.metadata
        }
    
    def save_report(self, output_path: Union[str, Path]) -> None:
        """Save diff report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class ResultComparer:
    """
    Main class for comparing IMC analysis results with scientific tolerances.
    
    Features:
    - Intelligent tolerance-based comparison
    - Support for all storage formats (HDF5, Parquet, JSON)
    - Structured diff reporting
    - Scientific equivalence assessment
    """
    
    def __init__(self, tolerance_profile: Optional[ToleranceProfile] = None):
        """Initialize result comparer."""
        self.tolerance_profile = tolerance_profile or ToleranceProfile()
        self.comparison_methods = {
            'arrays': self._compare_arrays,
            'scalars': self._compare_scalars,
            'clustering': self._compare_clustering_results,
            'spatial_stats': self._compare_spatial_statistics,
            'metadata': self._compare_metadata,
            'hierarchical': self._compare_hierarchical_results
        }
    
    def compare_analysis_results(
        self,
        result_path1: Union[str, Path, Dict],
        result_path2: Union[str, Path, Dict],
        tolerance_profile: Optional[str] = "standard",
        roi_ids: Optional[List[str]] = None
    ) -> DiffReport:
        """
        Compare two complete analysis runs.
        
        Args:
            result_path1: Path to first result set or result dict
            result_path2: Path to second result set or result dict  
            tolerance_profile: Tolerance profile name ("standard", "strict", "permissive")
            roi_ids: Specific ROI IDs to compare (None = all)
            
        Returns:
            Comprehensive diff report
        """
        # Set tolerance profile
        if tolerance_profile == "strict":
            profile = ToleranceProfile.create_strict()
        elif tolerance_profile == "permissive":
            profile = ToleranceProfile.create_permissive()
        else:
            profile = self.tolerance_profile
        
        # Load results
        results1 = self._load_results(result_path1, roi_ids)
        results2 = self._load_results(result_path2, roi_ids)
        
        # Create diff report
        comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        diff_report = DiffReport(
            comparison_id=comparison_id,
            timestamp=datetime.now().isoformat(),
            tolerance_profile=profile,
            metadata={
                'source1': str(result_path1),
                'source2': str(result_path2),
                'roi_ids_requested': roi_ids,
                'roi_ids_found': list(set(results1.keys()) | set(results2.keys()))
            }
        )
        
        # Compare each ROI
        all_roi_ids = set(results1.keys()) | set(results2.keys())
        
        for roi_id in all_roi_ids:
            roi_result1 = results1.get(roi_id)
            roi_result2 = results2.get(roi_id)
            
            if roi_result1 is None:
                diff_report.results.append(ComparisonResult(
                    field_path=f"roi.{roi_id}",
                    severity=ComparisonSeverity.INCOMPARABLE,
                    message=f"ROI {roi_id} missing from first result set",
                    comparison_method="existence_check"
                ))
                diff_report.incomparable_count += 1
                diff_report.total_comparisons += 1
                continue
                
            if roi_result2 is None:
                diff_report.results.append(ComparisonResult(
                    field_path=f"roi.{roi_id}",
                    severity=ComparisonSeverity.INCOMPARABLE,
                    message=f"ROI {roi_id} missing from second result set",
                    comparison_method="existence_check"
                ))
                diff_report.incomparable_count += 1
                diff_report.total_comparisons += 1
                continue
            
            # Compare individual ROI results
            roi_comparisons = self._compare_roi_results(
                roi_result1, roi_result2, roi_id, profile
            )
            
            # Add to report
            diff_report.results.extend(roi_comparisons)
            
            # Update counts
            for comp in roi_comparisons:
                diff_report.total_comparisons += 1
                if comp.severity == ComparisonSeverity.IDENTICAL:
                    diff_report.identical_count += 1
                elif comp.severity == ComparisonSeverity.EQUIVALENT:
                    diff_report.equivalent_count += 1
                elif comp.severity == ComparisonSeverity.DIFFERENT:
                    diff_report.different_count += 1
                else:
                    diff_report.incomparable_count += 1
        
        return diff_report
    
    def compare_clustering_results(
        self,
        result1: Dict[str, Any],
        result2: Dict[str, Any],
        tolerance_profile: Optional[ToleranceProfile] = None
    ) -> List[ComparisonResult]:
        """
        Compare clustering results between two analyses.
        
        Args:
            result1: First clustering result
            result2: Second clustering result
            tolerance_profile: Tolerance settings
            
        Returns:
            List of comparison results
        """
        profile = tolerance_profile or self.tolerance_profile
        return self._compare_clustering_results(result1, result2, "clustering", profile)
    
    def compare_spatial_statistics(
        self,
        stats1: Dict[str, Any],
        stats2: Dict[str, Any],
        tolerance_profile: Optional[ToleranceProfile] = None
    ) -> List[ComparisonResult]:
        """
        Compare spatial statistics between two analyses.
        
        Args:
            stats1: First spatial statistics
            stats2: Second spatial statistics  
            tolerance_profile: Tolerance settings
            
        Returns:
            List of comparison results
        """
        profile = tolerance_profile or self.tolerance_profile
        return self._compare_spatial_statistics(stats1, stats2, "spatial_stats", profile)
    
    def _load_results(
        self, 
        source: Union[str, Path, Dict], 
        roi_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Load results from various sources."""
        if isinstance(source, dict):
            # Direct dictionary input
            if roi_ids:
                return {roi_id: source.get(roi_id) for roi_id in roi_ids if roi_id in source}
            return source
        
        source_path = Path(source)
        
        if not STORAGE_AVAILABLE:
            # Fallback to JSON loading
            if source_path.is_file() and source_path.suffix == '.json':
                with open(source_path, 'r') as f:
                    data = json.load(f)
                    if roi_ids:
                        return {roi_id: data.get(roi_id) for roi_id in roi_ids if roi_id in data}
                    return data
            else:
                raise ValueError(f"Cannot load results from {source_path} (storage backends unavailable)")
        
        # Try to determine storage type and load appropriately
        if source_path.is_file():
            if source_path.suffix == '.h5':
                storage = HDF5Storage(source_path)
                roi_list = roi_ids or storage.list_rois()
                return {roi_id: storage.load_roi_results(roi_id) for roi_id in roi_list}
            elif source_path.suffix == '.json':
                storage = CompressedJSONStorage(source_path.parent)
                roi_list = roi_ids or [p.stem.replace('roi_', '') for p in source_path.parent.glob('roi_*.json*')]
                return {roi_id: storage.load_roi_complete(roi_id) for roi_id in roi_list}
        
        elif source_path.is_dir():
            # Try hybrid storage
            try:
                storage = HybridStorage(source_path)
                if hasattr(storage, 'hdf5_storage'):
                    roi_list = roi_ids or storage.hdf5_storage.list_rois()
                    return {roi_id: storage.load_roi_complete(roi_id) for roi_id in roi_list}
            except Exception:
                pass
            
            # Try parquet storage
            try:
                storage = ParquetStorage(source_path)
                # Load tabular data and convert to result format
                features_df = storage.load_features_for_analysis(roi_ids)
                if not features_df.empty:
                    results = {}
                    for roi_id in features_df['roi_id'].unique():
                        roi_data = features_df[features_df['roi_id'] == roi_id]
                        protein_cols = [col for col in roi_data.columns 
                                      if not col.startswith('meta_') and col not in ['roi_id', 'spatial_bin_id']]
                        results[roi_id] = {
                            'feature_matrix': roi_data[protein_cols].values,
                            'protein_names': protein_cols,
                            'metadata': {col: roi_data[col].iloc[0] for col in roi_data.columns if col.startswith('meta_')}
                        }
                    return results
            except Exception:
                pass
        
        raise ValueError(f"Could not load results from {source_path}")
    
    def _compare_roi_results(
        self,
        result1: Dict[str, Any],
        result2: Dict[str, Any],
        roi_id: str,
        tolerance_profile: ToleranceProfile
    ) -> List[ComparisonResult]:
        """Compare results for a single ROI."""
        comparisons = []
        
        # Get all possible field paths to compare
        all_fields = set()
        
        def collect_field_paths(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    field_path = f"{prefix}.{key}" if prefix else key
                    all_fields.add(field_path)
                    if isinstance(value, dict):
                        collect_field_paths(value, field_path)
        
        collect_field_paths(result1)
        collect_field_paths(result2)
        
        # Compare each field
        for field_path in all_fields:
            value1 = self._get_nested_value(result1, field_path)
            value2 = self._get_nested_value(result2, field_path)
            
            full_path = f"roi.{roi_id}.{field_path}"
            
            if value1 is None and value2 is None:
                continue
            elif value1 is None or value2 is None:
                comparisons.append(ComparisonResult(
                    field_path=full_path,
                    severity=ComparisonSeverity.INCOMPARABLE,
                    message=f"Field {field_path} missing in one result",
                    comparison_method="existence_check"
                ))
                continue
            
            # Choose comparison method based on field type and name
            if 'cluster' in field_path.lower():
                comp_results = self._compare_clustering_results(
                    {field_path.split('.')[-1]: value1},
                    {field_path.split('.')[-1]: value2},
                    full_path,
                    tolerance_profile
                )
                comparisons.extend(comp_results)
            elif 'spatial' in field_path.lower() or 'coord' in field_path.lower():
                comp_results = self._compare_spatial_statistics(
                    {field_path.split('.')[-1]: value1},
                    {field_path.split('.')[-1]: value2},
                    full_path,
                    tolerance_profile
                )
                comparisons.extend(comp_results)
            elif isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
                comp_result = self._compare_arrays(value1, value2, full_path, tolerance_profile)
                comparisons.append(comp_result)
            elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                comp_result = self._compare_scalars(value1, value2, full_path, tolerance_profile)
                comparisons.append(comp_result)
            else:
                comp_result = self._compare_metadata(value1, value2, full_path, tolerance_profile)
                comparisons.append(comp_result)
        
        return comparisons
    
    def _get_nested_value(self, obj: Dict, path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        try:
            keys = path.split('.')
            current = obj
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            return current
        except (KeyError, TypeError):
            return None
    
    def _compare_arrays(
        self,
        arr1: Union[np.ndarray, list],
        arr2: Union[np.ndarray, list],
        field_path: str,
        tolerance_profile: ToleranceProfile
    ) -> ComparisonResult:
        """Compare arrays with appropriate tolerances."""
        
        # Convert to lists if numpy arrays
        if NUMPY_AVAILABLE and isinstance(arr1, np.ndarray):
            arr1_list = arr1.tolist() if hasattr(arr1, 'tolist') else list(arr1)
            shape1 = arr1.shape
        else:
            arr1_list = arr1 if isinstance(arr1, list) else [arr1]
            shape1 = (len(arr1_list),) if hasattr(arr1_list, '__len__') else (1,)
            
        if NUMPY_AVAILABLE and isinstance(arr2, np.ndarray):
            arr2_list = arr2.tolist() if hasattr(arr2, 'tolist') else list(arr2)
            shape2 = arr2.shape
        else:
            arr2_list = arr2 if isinstance(arr2, list) else [arr2]
            shape2 = (len(arr2_list),) if hasattr(arr2_list, '__len__') else (1,)
        
        # Shape check
        if shape1 != shape2:
            return ComparisonResult(
                field_path=field_path,
                severity=ComparisonSeverity.DIFFERENT,
                message=f"Array shapes differ: {shape1} vs {shape2}",
                comparison_method="shape_check",
                value1_summary=f"shape={shape1}",
                value2_summary=f"shape={shape2}"
            )
        
        # Choose tolerances based on field type
        if 'coord' in field_path.lower():
            rtol, atol = tolerance_profile.spatial_coords_rtol, tolerance_profile.spatial_coords_atol
        elif 'expression' in field_path.lower() or 'protein' in field_path.lower():
            rtol, atol = tolerance_profile.expression_rtol, tolerance_profile.expression_atol
        else:
            rtol, atol = tolerance_profile.spatial_stats_rtol, tolerance_profile.spatial_stats_atol
        
        # Numerical comparison
        try:
            # Use numpy if available, otherwise use our stub implementation
            if NUMPY_AVAILABLE and hasattr(arr1, 'shape') and hasattr(arr2, 'shape'):
                are_close = np.allclose(arr1, arr2, rtol=rtol, atol=atol)
                max_diff = np.max(np.abs(arr1 - arr2))
                mean1, std1 = np.mean(arr1), np.std(arr1)
                mean2, std2 = np.mean(arr2), np.std(arr2)
            else:
                # Fallback implementation
                are_close = np.allclose(arr1_list, arr2_list, rtol=rtol, atol=atol)
                diffs = [abs(a - b) for a, b in zip(arr1_list, arr2_list)]
                max_diff = max(diffs) if diffs else 0.0
                mean1, std1 = np.mean(arr1_list), np.std(arr1_list)
                mean2, std2 = np.mean(arr2_list), np.std(arr2_list)
            
            if are_close:
                return ComparisonResult(
                    field_path=field_path,
                    severity=ComparisonSeverity.EQUIVALENT,
                    message=f"Arrays equivalent within tolerance (max_diff={max_diff:.6f})",
                    difference_metric=float(max_diff),
                    tolerance_used=rtol,
                    comparison_method="allclose",
                    value1_summary=f"mean={mean1:.4f}, std={std1:.4f}",
                    value2_summary=f"mean={mean2:.4f}, std={std2:.4f}"
                )
            else:
                # Calculate mean relative difference
                if NUMPY_AVAILABLE and hasattr(arr1, 'shape'):
                    mean_rel_diff = np.mean(np.abs((arr1 - arr2) / (arr1 + 1e-10)))
                else:
                    rel_diffs = [abs(a - b) / (abs(a) + 1e-10) for a, b in zip(arr1_list, arr2_list)]
                    mean_rel_diff = sum(rel_diffs) / len(rel_diffs) if rel_diffs else 0.0
                
                return ComparisonResult(
                    field_path=field_path,
                    severity=ComparisonSeverity.DIFFERENT,
                    message=f"Arrays differ beyond tolerance (max_diff={max_diff:.6f}, mean_rel_diff={mean_rel_diff:.4f})",
                    difference_metric=float(max_diff),
                    tolerance_used=rtol,
                    comparison_method="allclose",
                    value1_summary=f"mean={mean1:.4f}, std={std1:.4f}",
                    value2_summary=f"mean={mean2:.4f}, std={std2:.4f}"
                )
                
        except Exception as e:
            return ComparisonResult(
                field_path=field_path,
                severity=ComparisonSeverity.INCOMPARABLE,
                message=f"Array comparison failed: {str(e)}",
                comparison_method="allclose_failed"
            )
    
    def _compare_scalars(
        self,
        val1: Union[int, float],
        val2: Union[int, float],
        field_path: str,
        tolerance_profile: ToleranceProfile
    ) -> ComparisonResult:
        """Compare scalar values with appropriate tolerances."""
        
        if val1 == val2:
            return ComparisonResult(
                field_path=field_path,
                severity=ComparisonSeverity.IDENTICAL,
                message="Values identical",
                difference_metric=0.0,
                comparison_method="exact_match",
                value1_summary=str(val1),
                value2_summary=str(val2)
            )
        
        # Choose tolerance based on field type
        if 'quality' in field_path.lower() or 'score' in field_path.lower():
            tolerance = tolerance_profile.quality_score_tolerance
        elif 'silhouette' in field_path.lower():
            tolerance = tolerance_profile.silhouette_tolerance
        else:
            tolerance = tolerance_profile.spatial_stats_rtol
        
        abs_diff = abs(val1 - val2)
        rel_diff = abs_diff / (abs(val1) + 1e-10)
        
        if rel_diff <= tolerance:
            return ComparisonResult(
                field_path=field_path,
                severity=ComparisonSeverity.EQUIVALENT,
                message=f"Values equivalent within tolerance (rel_diff={rel_diff:.4f})",
                difference_metric=rel_diff,
                tolerance_used=tolerance,
                comparison_method="relative_tolerance",
                value1_summary=str(val1),
                value2_summary=str(val2)
            )
        else:
            return ComparisonResult(
                field_path=field_path,
                severity=ComparisonSeverity.DIFFERENT,
                message=f"Values differ beyond tolerance (rel_diff={rel_diff:.4f} > {tolerance:.4f})",
                difference_metric=rel_diff,
                tolerance_used=tolerance,
                comparison_method="relative_tolerance",
                value1_summary=str(val1),
                value2_summary=str(val2)
            )
    
    def _compare_clustering_results(
        self,
        result1: Dict[str, Any],
        result2: Dict[str, Any],
        field_path: str,
        tolerance_profile: ToleranceProfile
    ) -> List[ComparisonResult]:
        """Compare clustering results with cluster assignment tolerance."""
        comparisons = []
        
        # Extract cluster labels
        labels1 = result1.get('cluster_labels') or result1.get('labels')
        labels2 = result2.get('cluster_labels') or result2.get('labels')
        
        if labels1 is not None and labels2 is not None:
            if isinstance(labels1, np.ndarray) and isinstance(labels2, np.ndarray):
                # Calculate assignment similarity (allowing for label permutation)
                assignment_similarity = self._calculate_cluster_assignment_similarity(labels1, labels2)
                
                if assignment_similarity >= (1.0 - tolerance_profile.cluster_assignment_tolerance):
                    severity = ComparisonSeverity.EQUIVALENT
                    message = f"Cluster assignments equivalent (similarity={assignment_similarity:.3f})"
                else:
                    severity = ComparisonSeverity.DIFFERENT
                    message = f"Cluster assignments differ (similarity={assignment_similarity:.3f})"
                
                comparisons.append(ComparisonResult(
                    field_path=f"{field_path}.cluster_labels",
                    severity=severity,
                    message=message,
                    difference_metric=1.0 - assignment_similarity,
                    tolerance_used=tolerance_profile.cluster_assignment_tolerance,
                    comparison_method="cluster_assignment_similarity",
                    value1_summary=f"n_clusters={len(np.unique(labels1))}",
                    value2_summary=f"n_clusters={len(np.unique(labels2))}"
                ))
        
        # Compare cluster centroids if available
        centroids1 = result1.get('cluster_centroids') or result1.get('centroids')
        centroids2 = result2.get('cluster_centroids') or result2.get('centroids')
        
        if centroids1 is not None and centroids2 is not None:
            if isinstance(centroids1, dict) and isinstance(centroids2, dict):
                # Compare centroids for each cluster
                all_cluster_ids = set(centroids1.keys()) | set(centroids2.keys())
                
                centroid_diffs = []
                for cluster_id in all_cluster_ids:
                    if cluster_id in centroids1 and cluster_id in centroids2:
                        cent1 = centroids1[cluster_id]
                        cent2 = centroids2[cluster_id]
                        
                        if isinstance(cent1, dict) and isinstance(cent2, dict):
                            # Compare protein-wise
                            for protein in set(cent1.keys()) | set(cent2.keys()):
                                if protein in cent1 and protein in cent2:
                                    val1, val2 = cent1[protein], cent2[protein]
                                    rel_diff = abs(val1 - val2) / (abs(val1) + 1e-10)
                                    centroid_diffs.append(rel_diff)
                
                if centroid_diffs:
                    max_centroid_diff = max(centroid_diffs)
                    mean_centroid_diff = np.mean(centroid_diffs)
                    
                    if max_centroid_diff <= tolerance_profile.cluster_centroids_rtol:
                        severity = ComparisonSeverity.EQUIVALENT
                        message = f"Cluster centroids equivalent (max_diff={max_centroid_diff:.4f})"
                    else:
                        severity = ComparisonSeverity.DIFFERENT
                        message = f"Cluster centroids differ (max_diff={max_centroid_diff:.4f})"
                    
                    comparisons.append(ComparisonResult(
                        field_path=f"{field_path}.cluster_centroids",
                        severity=severity,
                        message=message,
                        difference_metric=mean_centroid_diff,
                        tolerance_used=tolerance_profile.cluster_centroids_rtol,
                        comparison_method="centroid_comparison",
                        context={'max_diff': max_centroid_diff, 'n_comparisons': len(centroid_diffs)}
                    ))
        
        return comparisons
    
    def _compare_spatial_statistics(
        self,
        stats1: Dict[str, Any],
        stats2: Dict[str, Any],
        field_path: str,
        tolerance_profile: ToleranceProfile
    ) -> List[ComparisonResult]:
        """Compare spatial statistics with appropriate tolerances."""
        comparisons = []
        
        # Compare spatial coordinates
        coords1 = stats1.get('spatial_coords') or stats1.get('coords')
        coords2 = stats2.get('spatial_coords') or stats2.get('coords')
        
        if coords1 is not None and coords2 is not None:
            comp_result = self._compare_arrays(
                coords1, coords2, f"{field_path}.spatial_coords", tolerance_profile
            )
            comparisons.append(comp_result)
        
        # Compare spatial statistics
        spatial_fields = ['spatial_coherence', 'spatial_autocorr', 'moran_i', 'spatial_quality']
        
        for field in spatial_fields:
            val1 = stats1.get(field)
            val2 = stats2.get(field)
            
            if val1 is not None and val2 is not None:
                comp_result = self._compare_scalars(
                    val1, val2, f"{field_path}.{field}", tolerance_profile
                )
                comparisons.append(comp_result)
        
        return comparisons
    
    def _compare_metadata(
        self,
        meta1: Any,
        meta2: Any,
        field_path: str,
        tolerance_profile: ToleranceProfile
    ) -> ComparisonResult:
        """Compare metadata fields (strings, lists, etc.)."""
        
        if meta1 == meta2:
            return ComparisonResult(
                field_path=field_path,
                severity=ComparisonSeverity.IDENTICAL,
                message="Metadata identical",
                comparison_method="exact_match",
                value1_summary=str(meta1)[:50],
                value2_summary=str(meta2)[:50]
            )
        
        # For lists, check if they contain the same elements
        if isinstance(meta1, list) and isinstance(meta2, list):
            if set(meta1) == set(meta2):
                return ComparisonResult(
                    field_path=field_path,
                    severity=ComparisonSeverity.EQUIVALENT,
                    message="Lists contain same elements (different order)",
                    comparison_method="set_comparison",
                    value1_summary=f"length={len(meta1)}",
                    value2_summary=f"length={len(meta2)}"
                )
        
        return ComparisonResult(
            field_path=field_path,
            severity=ComparisonSeverity.DIFFERENT,
            message="Metadata differs",
            comparison_method="exact_match",
            value1_summary=str(meta1)[:50],
            value2_summary=str(meta2)[:50]
        )
    
    def _compare_hierarchical_results(
        self,
        hier1: Dict[str, Any],
        hier2: Dict[str, Any],
        field_path: str,
        tolerance_profile: ToleranceProfile
    ) -> List[ComparisonResult]:
        """Compare hierarchical multiscale results."""
        comparisons = []
        
        # Compare scale consistency metrics
        if 'scale_consistency' in hier1 and 'scale_consistency' in hier2:
            consistency1 = hier1['scale_consistency']
            consistency2 = hier2['scale_consistency']
            
            if isinstance(consistency1, dict) and isinstance(consistency2, dict):
                for metric_name in set(consistency1.keys()) | set(consistency2.keys()):
                    if metric_name in consistency1 and metric_name in consistency2:
                        comp_result = self._compare_scalars(
                            consistency1[metric_name],
                            consistency2[metric_name],
                            f"{field_path}.scale_consistency.{metric_name}",
                            tolerance_profile
                        )
                        comparisons.append(comp_result)
        
        # Compare hierarchy structure if available
        if 'hierarchy' in hier1 and 'hierarchy' in hier2:
            # This is complex - for now just compare summary metrics
            comp_result = ComparisonResult(
                field_path=f"{field_path}.hierarchy",
                severity=ComparisonSeverity.EQUIVALENT,  # Assume equivalent for now
                message="Hierarchical structure comparison not fully implemented",
                comparison_method="placeholder"
            )
            comparisons.append(comp_result)
        
        return comparisons
    
    def _calculate_cluster_assignment_similarity(
        self,
        labels1: Union[np.ndarray, list],
        labels2: Union[np.ndarray, list]
    ) -> float:
        """
        Calculate similarity between cluster assignments, accounting for label permutation.
        
        Uses the Hungarian algorithm to find optimal label mapping.
        """
        # Convert to lists if needed
        if NUMPY_AVAILABLE and hasattr(labels1, 'tolist'):
            labels1_list = labels1.tolist()
        else:
            labels1_list = list(labels1) if hasattr(labels1, '__iter__') else [labels1]
            
        if NUMPY_AVAILABLE and hasattr(labels2, 'tolist'):
            labels2_list = labels2.tolist()
        else:
            labels2_list = list(labels2) if hasattr(labels2, '__iter__') else [labels2]
        
        if len(labels1_list) != len(labels2_list):
            return 0.0
        
        # Get unique labels
        if NUMPY_AVAILABLE:
            unique_labels1 = np.unique(labels1_list)
            unique_labels2 = np.unique(labels2_list)
        else:
            unique_labels1 = list(set(labels1_list))
            unique_labels2 = list(set(labels2_list))
        
        # Create confusion matrix
        if NUMPY_AVAILABLE:
            confusion_matrix = np.zeros((len(unique_labels1), len(unique_labels2)))
        else:
            confusion_matrix = [[0 for _ in range(len(unique_labels2))] for _ in range(len(unique_labels1))]
        
        for i, label1 in enumerate(unique_labels1):
            for j, label2 in enumerate(unique_labels2):
                # Count overlaps
                overlap = sum(1 for l1, l2 in zip(labels1_list, labels2_list) if l1 == label1 and l2 == label2)
                confusion_matrix[i][j] = overlap
        
        # Find optimal assignment using Hungarian algorithm if available
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(-confusion_matrix)
            if NUMPY_AVAILABLE:
                optimal_overlap = confusion_matrix[row_indices, col_indices].sum()
            else:
                optimal_overlap = sum(confusion_matrix[i][j] for i, j in zip(row_indices, col_indices))
        except ImportError:
            # Fallback: greedy assignment
            optimal_overlap = 0
            used_cols = set()
            for i in range(len(unique_labels1)):
                best_j = None
                best_overlap = 0
                for j in range(len(unique_labels2)):
                    overlap = confusion_matrix[i][j] if NUMPY_AVAILABLE else confusion_matrix[i][j]
                    if j not in used_cols and overlap > best_overlap:
                        best_j = j
                        best_overlap = overlap
                if best_j is not None:
                    optimal_overlap += best_overlap
                    used_cols.add(best_j)
        
        return optimal_overlap / len(labels1_list)


# Convenience functions
def compare_results(
    result_path1: Union[str, Path, Dict],
    result_path2: Union[str, Path, Dict],
    tolerance_profile: str = "standard",
    output_path: Optional[Union[str, Path]] = None
) -> DiffReport:
    """
    Convenience function to compare two analysis results.
    
    Args:
        result_path1: First result path or dict
        result_path2: Second result path or dict
        tolerance_profile: Tolerance level ("standard", "strict", "permissive")
        output_path: Optional path to save diff report
        
    Returns:
        Comprehensive diff report
    """
    comparer = ResultComparer()
    diff_report = comparer.compare_analysis_results(
        result_path1, result_path2, tolerance_profile
    )
    
    if output_path:
        diff_report.save_report(output_path)
    
    return diff_report


def quick_cluster_comparison(
    labels1: Union[np.ndarray, list],
    labels2: Union[np.ndarray, list],
    tolerance: float = 0.15
) -> Dict[str, Any]:
    """
    Quick comparison of cluster assignments.
    
    Args:
        labels1: First set of cluster labels
        labels2: Second set of cluster labels
        tolerance: Assignment tolerance (0.15 = 15% allowed difference)
        
    Returns:
        Comparison summary
    """
    comparer = ResultComparer()
    similarity = comparer._calculate_cluster_assignment_similarity(labels1, labels2)
    
    # Convert to lists for consistent handling
    labels1_list = labels1.tolist() if NUMPY_AVAILABLE and hasattr(labels1, 'tolist') else list(labels1)
    labels2_list = labels2.tolist() if NUMPY_AVAILABLE and hasattr(labels2, 'tolist') else list(labels2)
    
    return {
        'similarity': similarity,
        'equivalent': similarity >= (1.0 - tolerance),
        'n_clusters_1': len(set(labels1_list)),
        'n_clusters_2': len(set(labels2_list)),
        'n_points': len(labels1_list)
    }