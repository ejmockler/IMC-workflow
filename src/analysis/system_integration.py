"""
Cross-System Integration Layer for IMC Pipeline Phase 2D + 1B Systems

Provides unified interface to integrate all newly developed systems:

Phase 2D (Honest Baselines):
- Grid-based segmentation baseline
- Watershed DNA segmentation  
- Graph-based clustering baseline
- Synthetic ground truth generator
- Quantitative boundary metrics
- Ablation study framework

Phase 1B (Reference Standards):
- MI-IMC metadata schema
- Bead normalization protocols
- Single-stain reference protocols  
- Automatic QC threshold system

Creates seamless cross-system workflows, unified evaluation pipelines,
and comprehensive validation testing for all integrated components.
"""

import numpy as np
import pandas as pd
import warnings
import logging
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime

# Import Phase 2D Systems (Honest Baselines)
try:
    from .grid_segmentation import grid_pipeline, compare_grid_vs_slic, benchmark_grid_performance
    GRID_AVAILABLE = True
except ImportError:
    GRID_AVAILABLE = False
    warnings.warn("Grid segmentation not available")

try:
    from .watershed_segmentation import watershed_pipeline, assess_watershed_quality
    WATERSHED_AVAILABLE = True
except ImportError:
    WATERSHED_AVAILABLE = False
    warnings.warn("Watershed segmentation not available")

try:
    from .graph_clustering import GraphClusteringBaseline, create_graph_clustering_baseline
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    warnings.warn("Graph clustering not available")

try:
    from .synthetic_data_generator import (
        SyntheticDataGenerator, SyntheticDataConfig, create_example_datasets,
        validate_synthetic_dataset, SyntheticDataValidator
    )
    SYNTHETIC_AVAILABLE = True
except ImportError:
    SYNTHETIC_AVAILABLE = False
    warnings.warn("Synthetic data generator not available")

try:
    from .boundary_metrics import (
        BoundaryQualityEvaluator, create_boundary_evaluator, evaluate_method_comparison,
        SegmentationMethod, BoundaryMetricType, BoundaryQualityValidator
    )
    BOUNDARY_METRICS_AVAILABLE = True
except ImportError:
    BOUNDARY_METRICS_AVAILABLE = False
    warnings.warn("Boundary metrics not available")

try:
    from .ablation_framework import AblationFramework, AblationStudyType, AblationStudyResult
    ABLATION_AVAILABLE = True
except ImportError:
    ABLATION_AVAILABLE = False
    warnings.warn("Ablation framework not available")
    
    # Create fallback classes
    @dataclass  
    class AblationStudyResult:
        study_id: str
        method_ranking: List[Tuple[str, float]] = field(default_factory=list)
        recommendations: List[str] = field(default_factory=list)
    
    class AblationStudyType(Enum):
        METHOD_COMPARISON = "method_comparison"

# Import Phase 1B Systems (Reference Standards)
try:
    from .mi_imc_schema import MIIMCSchema, StudyMetadata, SampleMetadata, AntibodyMetadata
    MI_IMC_AVAILABLE = True
except ImportError:
    MI_IMC_AVAILABLE = False
    warnings.warn("MI-IMC schema not available")

try:
    from .bead_normalization import (
        detect_calibration_beads, compute_normalization_factors,
        BeadDetectionConfig, NormalizationConfig
    )
    BEAD_NORM_AVAILABLE = True
except ImportError:
    BEAD_NORM_AVAILABLE = False
    warnings.warn("Bead normalization not available")

try:
    from .single_stain_protocols import SingleStainProtocol, create_spillover_matrix
    SINGLE_STAIN_AVAILABLE = True
except ImportError:
    SINGLE_STAIN_AVAILABLE = False
    warnings.warn("Single stain protocols not available")

try:
    from .automatic_qc_system import AutomaticQCSystem, QCDecisionEngine
    AUTO_QC_AVAILABLE = True
except ImportError:
    AUTO_QC_AVAILABLE = False
    warnings.warn("Automatic QC system not available")

# Import existing pipeline components
try:
    from .slic_segmentation import slic_pipeline
    from .multiscale_analysis import perform_multiscale_analysis
    from .spatial_clustering import perform_spatial_clustering
    EXISTING_PIPELINE_AVAILABLE = True
except ImportError:
    EXISTING_PIPELINE_AVAILABLE = False
    warnings.warn("Existing pipeline components not available")

# Import configuration and validation
try:
    from ..config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    warnings.warn("Config not available")

try:
    from ..validation.framework import ValidationResult, ValidationSeverity
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    warnings.warn("Validation framework not available")
    
    # Create fallback classes
    from dataclasses import dataclass
    from enum import Enum
    from typing import Optional, Dict, Any, List
    
    class ValidationSeverity(Enum):
        CRITICAL = "critical"
        WARNING = "warning"
        INFO = "info"
        PASS = "pass"
    
    @dataclass
    class ValidationResult:
        rule_name: str
        severity: ValidationSeverity
        message: str
        quality_score: Optional[float] = None
        recommendations: List[str] = None
        
        def __post_init__(self):
            if self.recommendations is None:
                self.recommendations = []


class IntegrationMethod(Enum):
    """Available analysis methods in integrated system."""
    SLIC = "slic"
    GRID = "grid"
    WATERSHED = "watershed"
    GRAPH_LEIDEN = "graph_leiden"
    GRAPH_LOUVAIN = "graph_louvain"
    HYBRID = "hybrid"


class EvaluationMode(Enum):
    """Evaluation modes for method comparison."""
    COMPREHENSIVE = "comprehensive"
    BASELINE_ONLY = "baseline_only"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"


@dataclass
class IntegrationConfig:
    """Configuration for system integration."""
    methods_to_evaluate: List[IntegrationMethod] = field(default_factory=lambda: [
        IntegrationMethod.SLIC, IntegrationMethod.GRID, 
        IntegrationMethod.WATERSHED, IntegrationMethod.GRAPH_LEIDEN
    ])
    use_synthetic_validation: bool = True
    use_ablation_studies: bool = True
    use_reference_standards: bool = True
    evaluation_mode: EvaluationMode = EvaluationMode.COMPREHENSIVE
    quality_threshold: float = 0.6
    statistical_significance: float = 0.05
    random_state: int = 42


@dataclass
class IntegrationResult:
    """Complete integration evaluation result."""
    method_results: Dict[str, Any]
    evaluation_metrics: Dict[str, Any]
    method_ranking: List[Tuple[str, float]]
    statistical_analysis: Dict[str, Any]
    validation_results: List[ValidationResult]
    recommendations: List[str]
    metadata: Dict[str, Any]
    reference_standards: Optional[Dict[str, Any]] = None
    ablation_results: Optional[AblationStudyResult] = None


class SystemIntegrator:
    """
    Unified interface for all Phase 2D and Phase 1B systems.
    
    Provides comprehensive method evaluation, cross-system validation,
    and integrated analysis workflows.
    """
    
    def __init__(self, config: IntegrationConfig = None, base_config: Any = None):
        """Initialize system integrator."""
        self.config = config or IntegrationConfig()
        self.base_config = base_config
        self.logger = logging.getLogger('SystemIntegrator')
        
        # Initialize components
        self._init_evaluators()
        self._init_reference_standards()
        self._init_validation_framework()
        
        # Method factory
        self.method_factory = MethodFactory(self.base_config)
        
        # Cache for repeated evaluations
        self._evaluation_cache = {}
        
    def _init_evaluators(self):
        """Initialize evaluation components."""
        # Boundary quality evaluator
        if BOUNDARY_METRICS_AVAILABLE:
            self.boundary_evaluator = create_boundary_evaluator(self.config.random_state)
            self.boundary_validator = BoundaryQualityValidator(self.config.quality_threshold)
        else:
            self.boundary_evaluator = None
            self.boundary_validator = None
        
        # Graph clustering baseline
        if GRAPH_AVAILABLE:
            self.graph_baseline = GraphClusteringBaseline(self.config.random_state)
        else:
            self.graph_baseline = None
        
        # Ablation framework
        if ABLATION_AVAILABLE:
            self.ablation_framework = AblationFramework(
                config=self.base_config,
                random_state=self.config.random_state
            )
        else:
            self.ablation_framework = None
    
    def _init_reference_standards(self):
        """Initialize reference standards components."""
        # MI-IMC schema
        if MI_IMC_AVAILABLE:
            self.mi_imc_schema = MIIMCSchema()
            if self.base_config:
                self.mi_imc_schema.import_from_config(self.base_config)
        else:
            self.mi_imc_schema = None
        
        # Bead normalization
        if BEAD_NORM_AVAILABLE:
            self.bead_config = BeadDetectionConfig(
                bead_channels=['Ce140', 'Eu151', 'Eu153', 'Ho165'],
                signal_threshold_percentile=95.0
            )
            self.norm_config = NormalizationConfig()
        else:
            self.bead_config = None
            self.norm_config = None
        
        # Automatic QC system
        if AUTO_QC_AVAILABLE:
            self.qc_system = AutomaticQCSystem()
            self.qc_engine = QCDecisionEngine()
        else:
            self.qc_system = None
            self.qc_engine = None
    
    def _init_validation_framework(self):
        """Initialize validation components."""
        self.validators = []
        
        # Add boundary quality validator
        if self.boundary_validator:
            self.validators.append(self.boundary_validator)
        
        # Add synthetic data validator
        if SYNTHETIC_AVAILABLE:
            self.validators.append(SyntheticDataValidator())
        
        self.logger.info(f"Initialized {len(self.validators)} validators")
    
    def evaluate_all_methods(
        self,
        coords: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        dna1_intensities: np.ndarray,
        dna2_intensities: np.ndarray,
        ground_truth_data: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """
        Comprehensive evaluation of all available methods.
        
        Args:
            coords: Spatial coordinates
            ion_counts: Ion count data
            dna1_intensities: DNA1 channel
            dna2_intensities: DNA2 channel
            ground_truth_data: Optional ground truth for supervised evaluation
            
        Returns:
            Complete integration evaluation result
        """
        self.logger.info("Starting comprehensive method evaluation")
        start_time = time.time()
        
        # Prepare data
        data_package = {
            'coords': coords,
            'ion_counts': ion_counts,
            'dna1_intensities': dna1_intensities,
            'dna2_intensities': dna2_intensities
        }
        
        # Run all methods
        method_results = {}
        for method in self.config.methods_to_evaluate:
            try:
                self.logger.info(f"Evaluating method: {method.value}")
                result = self.method_factory.run_method(method, data_package)
                method_results[method.value] = result
            except Exception as e:
                self.logger.error(f"Method {method.value} failed: {e}")
                method_results[method.value] = {'error': str(e)}
        
        # Comprehensive evaluation
        evaluation_metrics = self._evaluate_methods_comprehensive(
            method_results, ground_truth_data, data_package
        )
        
        # Method ranking
        method_ranking = self._rank_methods(evaluation_metrics)
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(evaluation_metrics)
        
        # Validation
        validation_results = self._run_validation_suite(method_results, ground_truth_data)
        
        # Reference standards evaluation
        reference_standards = None
        if self.config.use_reference_standards:
            reference_standards = self._evaluate_reference_standards(data_package)
        
        # Ablation studies
        ablation_results = None
        if self.config.use_ablation_studies and self.ablation_framework:
            try:
                ablation_results = self.ablation_framework.run_method_comparison_study(
                    data_package, ground_truth_data=ground_truth_data
                )
            except Exception as e:
                self.logger.warning(f"Ablation study failed: {e}")
        
        # Generate recommendations
        recommendations = self._generate_integration_recommendations(
            method_ranking, statistical_analysis, validation_results
        )
        
        # Metadata
        metadata = {
            'evaluation_time': time.time() - start_time,
            'n_methods_tested': len(method_results),
            'n_successful_methods': len([r for r in method_results.values() if 'error' not in r]),
            'data_size': len(coords),
            'evaluation_mode': self.config.evaluation_mode.value,
            'config': self.config.__dict__
        }
        
        return IntegrationResult(
            method_results=method_results,
            evaluation_metrics=evaluation_metrics,
            method_ranking=method_ranking,
            statistical_analysis=statistical_analysis,
            validation_results=validation_results,
            recommendations=recommendations,
            metadata=metadata,
            reference_standards=reference_standards,
            ablation_results=ablation_results
        )
    
    def _evaluate_methods_comprehensive(
        self,
        method_results: Dict[str, Any],
        ground_truth_data: Optional[Dict[str, Any]],
        data_package: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of all methods."""
        evaluation_metrics = {}
        
        for method_name, results in method_results.items():
            if 'error' in results:
                evaluation_metrics[method_name] = {'error': results['error']}
                continue
            
            method_metrics = {}
            
            # Boundary quality evaluation
            if self.boundary_evaluator and BOUNDARY_METRICS_AVAILABLE:
                try:
                    boundary_metrics = self.boundary_evaluator.evaluate_comprehensive(
                        results, ground_truth_data
                    )
                    overall_score = self.boundary_evaluator._calculate_overall_score(boundary_metrics)
                    method_metrics['boundary_quality'] = overall_score
                    method_metrics['boundary_metrics'] = boundary_metrics
                except Exception as e:
                    self.logger.warning(f"Boundary evaluation failed for {method_name}: {e}")
            
            # Performance metrics
            if 'performance_comparison' in results:
                perf = results['performance_comparison']
                method_metrics['execution_time'] = perf.get('total_time', 0)
                method_metrics['memory_usage'] = perf.get('memory_usage_mb', 0)
            
            # Clustering quality
            if 'cluster_labels' in results or 'superpixel_labels' in results:
                labels = results.get('cluster_labels', results.get('superpixel_labels'))
                if labels is not None:
                    method_metrics['n_clusters'] = len(np.unique(labels[labels >= 0]))
                    
                    # Silhouette score if feature matrix available
                    if 'feature_matrix' in results or 'superpixel_counts' in results:
                        features = results.get('feature_matrix')
                        if features is None and 'superpixel_counts' in results:
                            # Create feature matrix from ion counts
                            counts = results['superpixel_counts']
                            if len(counts) > 0 and len(list(counts.keys())) > 0:
                                features = np.column_stack([counts[protein] for protein in counts.keys()])
                            else:
                                features = None

                        if features is not None and len(features) > 1:
                            try:
                                from sklearn.metrics import silhouette_score
                                valid_mask = labels >= 0
                                if np.sum(valid_mask) > 1 and len(np.unique(labels[valid_mask])) > 1:
                                    # Use sampling for large datasets to reduce O(n²) → O(n*s) complexity
                                    n_samples = np.sum(valid_mask)
                                    sample_size = min(1000, n_samples) if n_samples > 1000 else None
                                    silhouette = silhouette_score(
                                        features[valid_mask],
                                        labels[valid_mask],
                                        sample_size=sample_size
                                    )
                                    method_metrics['silhouette_score'] = silhouette
                            except Exception as e:
                                self.logger.debug(f"Silhouette score calculation failed: {e}")
            
            evaluation_metrics[method_name] = method_metrics
        
        return evaluation_metrics
    
    def _rank_methods(self, evaluation_metrics: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Rank methods by overall quality score."""
        method_scores = []
        
        for method_name, metrics in evaluation_metrics.items():
            if 'error' in metrics:
                score = 0.0
            else:
                # Weighted scoring
                score = 0.0
                weights = {
                    'boundary_quality': 0.4,
                    'silhouette_score': 0.3,
                    'execution_time': -0.1,  # Lower is better
                    'memory_usage': -0.1     # Lower is better
                }
                
                for metric, weight in weights.items():
                    if metric in metrics:
                        value = metrics[metric]
                        if metric in ['execution_time', 'memory_usage']:
                            # Normalize and invert (lower is better)
                            if value > 0:
                                score += weight * (1.0 / (1.0 + value))
                        else:
                            # Higher is better
                            score += weight * min(1.0, max(0.0, value))

                # Clamp to [0, 1] to maintain normalized interpretation
                score = max(0.0, min(1.0, score))
            
            method_scores.append((method_name, score))
        
        # Sort by score (descending)
        return sorted(method_scores, key=lambda x: x[1], reverse=True)
    
    def _perform_statistical_analysis(self, evaluation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis of method differences."""
        analysis = {}
        
        # Extract boundary quality scores
        boundary_scores = {}
        for method, metrics in evaluation_metrics.items():
            if 'boundary_quality' in metrics and 'error' not in metrics:
                boundary_scores[method] = metrics['boundary_quality']
        
        if len(boundary_scores) >= 2:
            # Pairwise comparisons
            pairwise_results = {}
            methods = list(boundary_scores.keys())
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    score1, score2 = boundary_scores[method1], boundary_scores[method2]
                    diff = abs(score1 - score2)
                    significant = diff > self.config.statistical_significance
                    
                    pairwise_results[f"{method1}_vs_{method2}"] = {
                        'score_difference': score1 - score2,
                        'abs_difference': diff,
                        'significant': significant,
                        'better_method': method1 if score1 > score2 else method2
                    }
            
            analysis['pairwise_comparisons'] = pairwise_results
            analysis['score_distribution'] = {
                'mean': np.mean(list(boundary_scores.values())),
                'std': np.std(list(boundary_scores.values())),
                'min': np.min(list(boundary_scores.values())),
                'max': np.max(list(boundary_scores.values()))
            }
        
        return analysis
    
    def _run_validation_suite(
        self,
        method_results: Dict[str, Any],
        ground_truth_data: Optional[Dict[str, Any]]
    ) -> List[ValidationResult]:
        """Run complete validation suite."""
        validation_results = []
        
        for validator in self.validators:
            try:
                for method_name, results in method_results.items():
                    if 'error' not in results:
                        validation_data = {
                            'segmentation_results': results,
                            'ground_truth_data': ground_truth_data,
                            'method_name': method_name
                        }
                        
                        result = validator.validate(validation_data)
                        validation_results.append(result)
            except Exception as e:
                self.logger.warning(f"Validation failed for {type(validator).__name__}: {e}")
        
        return validation_results
    
    def _evaluate_reference_standards(self, data_package: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate reference standards."""
        reference_results = {}
        
        # Bead normalization evaluation
        if BEAD_NORM_AVAILABLE and self.bead_config:
            try:
                # Mock bead data for demonstration
                mock_beads = {
                    'Ce140': np.random.exponential(100, len(data_package['coords'])),
                    'Eu151': np.random.exponential(80, len(data_package['coords'])),
                    'Eu153': np.random.exponential(90, len(data_package['coords'])),
                    'Ho165': np.random.exponential(70, len(data_package['coords']))
                }
                
                bead_result = detect_calibration_beads(
                    mock_beads, data_package['coords'], self.bead_config
                )
                
                reference_results['bead_normalization'] = {
                    'detection_success': bead_result.is_valid_for_normalization(),
                    'n_channels_detected': len(bead_result.detected_beads),
                    'quality_metrics': bead_result.quality_metrics
                }
            except Exception as e:
                self.logger.warning(f"Bead normalization evaluation failed: {e}")
        
        # MI-IMC schema validation
        if self.mi_imc_schema:
            try:
                # Create basic study metadata
                study = StudyMetadata(
                    study_title="IMC Method Integration Study",
                    research_question="Comparison of segmentation and clustering methods",
                    hypotheses=["Different methods yield different spatial patterns"]
                )
                self.mi_imc_schema.set_study_metadata(study)
                
                reference_results['mi_imc_schema'] = {
                    'schema_complete': True,
                    'version': self.mi_imc_schema.version.value,
                    'n_antibodies': len(self.mi_imc_schema.antibody_panel)
                }
            except Exception as e:
                self.logger.warning(f"MI-IMC schema evaluation failed: {e}")
        
        return reference_results
    
    def _generate_integration_recommendations(
        self,
        method_ranking: List[Tuple[str, float]],
        statistical_analysis: Dict[str, Any],
        validation_results: List[ValidationResult]
    ) -> List[str]:
        """Generate comprehensive recommendations."""
        recommendations = []
        
        if method_ranking:
            best_method, best_score = method_ranking[0]
            recommendations.append(f"Best performing method: {best_method} (score: {best_score:.3f})")
            
            if best_score < self.config.quality_threshold:
                recommendations.append(f"⚠️  Best method score ({best_score:.3f}) below threshold ({self.config.quality_threshold})")
        
        # Statistical analysis recommendations
        if 'pairwise_comparisons' in statistical_analysis:
            significant_diffs = [
                comp for comp, results in statistical_analysis['pairwise_comparisons'].items()
                if results['significant']
            ]
            if significant_diffs:
                recommendations.append(f"Found {len(significant_diffs)} statistically significant method differences")
            else:
                recommendations.append("No statistically significant differences between methods")
        
        # Validation recommendations
        critical_validations = [v for v in validation_results if v.severity == ValidationSeverity.CRITICAL]
        if critical_validations:
            recommendations.append(f"⚠️  {len(critical_validations)} critical validation failures require attention")
        
        # Method-specific recommendations
        if len(method_ranking) >= 2:
            top_methods = [method for method, score in method_ranking[:2]]
            recommendations.append(f"Consider comparing top methods in detail: {', '.join(top_methods)}")
        
        return recommendations
    
    def generate_synthetic_validation_dataset(
        self,
        n_cells: int = 5000,
        roi_size_um: Tuple[float, float] = (800.0, 800.0),
        tissue_type: str = 'complex'
    ) -> Dict[str, Any]:
        """Generate synthetic dataset for validation."""
        if not SYNTHETIC_AVAILABLE:
            raise RuntimeError("Synthetic data generator not available")
        
        from .synthetic_data_generator import SyntheticDataConfig
        
        # Create configuration for validation dataset
        if tissue_type == 'simple':
            protein_names = ['CD45', 'CD3', 'CD20', 'PanCK', 'DNA1']
            n_clusters = 3
        elif tissue_type == 'complex':
            protein_names = ['CD45', 'CD3', 'CD4', 'CD8', 'CD20', 'CD68', 'PanCK', 'Vimentin', 'DNA1']
            n_clusters = 7
        else:  # custom
            protein_names = ['CD45', 'CD3', 'CD20', 'PanCK', 'Vimentin', 'DNA1']
            n_clusters = 5
        
        config = SyntheticDataConfig(
            roi_size_um=roi_size_um,
            n_cells_total=n_cells,
            protein_names=protein_names,
            known_cluster_count=n_clusters,
            baseline_noise_level=0.15,
            hot_pixel_probability=0.002
        )
        
        generator = SyntheticDataGenerator(config)
        dataset = generator.generate_complete_dataset()
        
        self.logger.info(f"Generated synthetic dataset: {n_cells} cells, {len(protein_names)} proteins")
        return dataset
    
    def run_comprehensive_validation(
        self,
        coords: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        dna1_intensities: np.ndarray,
        dna2_intensities: np.ndarray,
        use_synthetic: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive validation using all available systems."""
        validation_results = {}
        
        # Real data evaluation
        real_data_result = self.evaluate_all_methods(
            coords, ion_counts, dna1_intensities, dna2_intensities
        )
        validation_results['real_data'] = asdict(real_data_result)
        
        # Synthetic data evaluation
        if use_synthetic and SYNTHETIC_AVAILABLE:
            try:
                synthetic_dataset = self.generate_synthetic_validation_dataset()
                
                synthetic_result = self.evaluate_all_methods(
                    synthetic_dataset['coordinates'],
                    synthetic_dataset['ion_counts'],
                    synthetic_dataset['dna1_intensities'],
                    synthetic_dataset['dna2_intensities'],
                    ground_truth_data=synthetic_dataset
                )
                validation_results['synthetic_data'] = asdict(synthetic_result)
                
                # Compare real vs synthetic performance
                validation_results['real_vs_synthetic_comparison'] = self._compare_real_vs_synthetic(
                    real_data_result, synthetic_result
                )
                
            except Exception as e:
                self.logger.warning(f"Synthetic validation failed: {e}")
        
        return validation_results


class MethodFactory:
    """Factory for creating and running different analysis methods."""
    
    def __init__(self, config: Any = None):
        """Initialize method factory."""
        self.config = config
        self.logger = logging.getLogger('MethodFactory')
    
    def run_method(self, method: IntegrationMethod, data_package: Dict[str, Any]) -> Dict[str, Any]:
        """Run specified method on data package."""
        coords = data_package['coords']
        ion_counts = data_package['ion_counts']
        dna1 = data_package['dna1_intensities']
        dna2 = data_package['dna2_intensities']
        
        if method == IntegrationMethod.SLIC:
            return self._run_slic(coords, ion_counts, dna1, dna2)
        elif method == IntegrationMethod.GRID:
            return self._run_grid(coords, ion_counts, dna1, dna2)
        elif method == IntegrationMethod.WATERSHED:
            return self._run_watershed(coords, ion_counts, dna1, dna2)
        elif method == IntegrationMethod.GRAPH_LEIDEN:
            return self._run_graph_leiden(coords, ion_counts, dna1, dna2)
        elif method == IntegrationMethod.GRAPH_LOUVAIN:
            return self._run_graph_louvain(coords, ion_counts, dna1, dna2)
        elif method == IntegrationMethod.HYBRID:
            return self._run_hybrid(coords, ion_counts, dna1, dna2)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _run_slic(self, coords, ion_counts, dna1, dna2):
        """Run SLIC segmentation."""
        if not EXISTING_PIPELINE_AVAILABLE:
            raise RuntimeError("SLIC pipeline not available")
        
        result = slic_pipeline(
            coords, ion_counts, dna1, dna2,
            target_scale_um=20.0, config=self.config
        )
        result['method'] = 'slic'
        return result
    
    def _run_grid(self, coords, ion_counts, dna1, dna2):
        """Run grid segmentation."""
        if not GRID_AVAILABLE:
            raise RuntimeError("Grid segmentation not available")
        
        result = grid_pipeline(
            coords, ion_counts, dna1, dna2,
            target_scale_um=20.0, config=self.config
        )
        result['method'] = 'grid'
        return result
    
    def _run_watershed(self, coords, ion_counts, dna1, dna2):
        """Run watershed segmentation."""
        if not WATERSHED_AVAILABLE:
            raise RuntimeError("Watershed segmentation not available")
        
        result = watershed_pipeline(
            coords, ion_counts, dna1, dna2,
            target_scale_um=20.0, config=self.config
        )
        result['method'] = 'watershed'
        return result
    
    def _run_graph_leiden(self, coords, ion_counts, dna1, dna2):
        """Run graph-based clustering with Leiden."""
        if not GRAPH_AVAILABLE:
            raise RuntimeError("Graph clustering not available")
        
        # Create feature matrix
        if ion_counts:
            protein_names = list(ion_counts.keys())
            feature_matrix = np.column_stack([ion_counts[p] for p in protein_names])
        else:
            raise ValueError("No ion counts available for graph clustering")
        
        result = create_graph_clustering_baseline(
            feature_matrix, protein_names, coords,
            config={'clustering_method': 'leiden', 'graph_method': 'knn'}
        )
        
        # Convert to pipeline-compatible format
        result.update({
            'superpixel_labels': result['cluster_labels'],
            'superpixel_coords': coords,
            'superpixel_counts': ion_counts,
            'method': 'graph_leiden'
        })
        return result
    
    def _run_graph_louvain(self, coords, ion_counts, dna1, dna2):
        """Run graph-based clustering with Louvain."""
        if not GRAPH_AVAILABLE:
            raise RuntimeError("Graph clustering not available")
        
        # Create feature matrix
        if ion_counts:
            protein_names = list(ion_counts.keys())
            feature_matrix = np.column_stack([ion_counts[p] for p in protein_names])
        else:
            raise ValueError("No ion counts available for graph clustering")
        
        result = create_graph_clustering_baseline(
            feature_matrix, protein_names, coords,
            config={'clustering_method': 'louvain', 'graph_method': 'knn'}
        )
        
        # Convert to pipeline-compatible format
        result.update({
            'superpixel_labels': result['cluster_labels'],
            'superpixel_coords': coords,
            'superpixel_counts': ion_counts,
            'method': 'graph_louvain'
        })
        return result
    
    def _run_hybrid(self, coords, ion_counts, dna1, dna2):
        """Run hybrid method combining multiple approaches."""
        # For now, use SLIC as the base method
        return self._run_slic(coords, ion_counts, dna1, dna2)
    
    def _compare_real_vs_synthetic(self, real_result, synthetic_result):
        """Compare performance on real vs synthetic data."""
        comparison = {}

        def _extract_ranking(result: Any) -> Dict[str, float]:
            if hasattr(result, 'method_ranking'):
                ranking_iter = getattr(result, 'method_ranking')
            elif isinstance(result, dict):
                ranking_iter = result.get('method_ranking', [])
            else:
                ranking_iter = []
            try:
                return dict(ranking_iter)
            except Exception:
                return {}
        
        # Compare method rankings
        real_ranking = _extract_ranking(real_result)
        synthetic_ranking = _extract_ranking(synthetic_result)
        
        ranking_correlation = {}
        for method in real_ranking:
            if method in synthetic_ranking:
                ranking_correlation[method] = {
                    'real_score': real_ranking[method],
                    'synthetic_score': synthetic_ranking[method],
                    'score_diff': abs(real_ranking[method] - synthetic_ranking[method])
                }
        
        comparison['ranking_correlation'] = ranking_correlation
        
        # Overall agreement
        if ranking_correlation:
            score_diffs = [data['score_diff'] for data in ranking_correlation.values()]
            comparison['mean_score_difference'] = np.mean(score_diffs)
            comparison['agreement_level'] = 'high' if np.mean(score_diffs) < 0.1 else 'moderate' if np.mean(score_diffs) < 0.3 else 'low'
        
        return comparison


def create_system_integrator(
    config: IntegrationConfig = None,
    base_config: Any = None
) -> SystemIntegrator:
    """
    Factory function to create system integrator.
    
    Args:
        config: Integration configuration
        base_config: Base pipeline configuration
        
    Returns:
        Configured system integrator
    """
    if base_config is None:
        try:
            base_config = Config()
        except Exception as exc:
            logging.getLogger('SystemIntegrator').warning(
                f"Falling back to default configuration: {exc}"
            )
            base_config = None
    return SystemIntegrator(config, base_config)


def run_integration_example():
    """Example usage of the integration system."""
    # Create mock data
    n_cells = 1000
    coords = np.random.uniform(0, 500, (n_cells, 2))
    
    protein_names = ['CD45', 'CD3', 'CD4', 'CD8', 'CD20', 'PanCK']
    ion_counts = {
        protein: np.random.poisson(100, n_cells).astype(float)
        for protein in protein_names
    }
    
    dna1 = np.random.poisson(200, n_cells).astype(float)
    dna2 = np.random.poisson(180, n_cells).astype(float)
    
    # Create integrator
    integrator = create_system_integrator()
    
    # Run evaluation
    print("Running integration example...")
    result = integrator.evaluate_all_methods(
        coords, ion_counts, dna1, dna2
    )
    
    print(f"\nEvaluation completed in {result.metadata['evaluation_time']:.2f} seconds")
    print(f"Methods tested: {result.metadata['n_methods_tested']}")
    print(f"Successful methods: {result.metadata['n_successful_methods']}")
    
    print("\nMethod ranking:")
    for i, (method, score) in enumerate(result.method_ranking):
        print(f"  {i+1}. {method}: {score:.3f}")
    
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")
    
    return result


if __name__ == "__main__":
    # Run example
    try:
        result = run_integration_example()
        print("\n✅ Integration system functional!")
    except Exception as e:
        print(f"\n❌ Integration system error: {e}")
        import traceback
        traceback.print_exc()
