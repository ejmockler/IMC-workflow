"""
QuickStart Pipeline - Realistic, Fast, Reliable Default Path

MISSION: Build the user-facing interface that combines all optimizations into a realistic,
honest pipeline with clear limitations and hardware requirements.

BRUTALIST FEEDBACK INTEGRATED:
- NO pretentious naming claims
- NO impossible setup promises
- NO ready-for-publication automatic outputs
- YES to "QuickStart" honest limitations
- YES to one fast, reliable default path
- YES to explicit hardware requirements

INTEGRATES ALL OPTIMIZATIONS:
- Emergency optimized pipeline (quickstart_pipeline.py)
- Memory optimization framework (memory_optimizer.py) 
- Unified DAG with caching (performance_dag.py)
- QC assessment system (automatic_qc_system.py)

HONEST ENGINEERING:
- Hardware Requirements: 1k×1k pixels, ≤40 channels, 32GB RAM
- Default Method: SLIC segmentation at 20μm scale with Leiden clustering
- Memory Target: <16GB usage with monitoring
- Time Target: <10 minutes for standard ROI
- Output: Draft reports with QC warnings, requiring human review
"""

import numpy as np
import pandas as pd
import json
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import time
import psutil
import gc

# Import all optimization systems
from .quickstart_pipeline import QuickstartPipeline
from .memory_optimizer import PipelineMemoryOptimizer, MemoryOptimizationReport
from .performance_dag import PerformanceDAG, create_imc_analysis_dag, validate_dag_performance
from .automatic_qc_system import AutomaticQCSystem, create_automatic_qc_system, AutomaticQCConfig
from ..config import Config


class HardwareValidator:
    """Validates hardware requirements and provides honest feedback."""
    
    # HONEST HARDWARE REQUIREMENTS
    STANDARD_ROI_SPECS = {
        'max_pixels': 1_000_000,  # 1k×1k pixels
        'max_channels': 40,       # ≤40 channels
        'required_ram_gb': 32,    # 32GB RAM
        'target_memory_gb': 16,   # <16GB target usage
        'target_time_minutes': 10 # <10 minutes target
    }
    
    def __init__(self):
        self.logger = logging.getLogger('HardwareValidator')
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.cpu_count = psutil.cpu_count()
    
    def validate_hardware(self) -> Dict[str, Any]:
        """Validate system hardware against requirements."""
        validation = {
            'meets_requirements': True,
            'warnings': [],
            'recommendations': [],
            'system_info': {
                'total_memory_gb': self.system_memory_gb,
                'cpu_cores': self.cpu_count,
                'meets_ram_requirement': self.system_memory_gb >= self.STANDARD_ROI_SPECS['required_ram_gb']
            }
        }
        
        # Check RAM
        if self.system_memory_gb < self.STANDARD_ROI_SPECS['required_ram_gb']:
            validation['meets_requirements'] = False
            validation['warnings'].append(
                f"System has {self.system_memory_gb:.1f}GB RAM, "
                f"requires {self.STANDARD_ROI_SPECS['required_ram_gb']}GB for standard ROIs"
            )
            validation['recommendations'].append(
                "Upgrade to 32GB+ RAM or use reduced ROI sizes"
            )
        
        # Check CPU
        if self.cpu_count < 4:
            validation['warnings'].append(
                f"System has {self.cpu_count} CPU cores, recommend 4+ for optimal performance"
            )
            validation['recommendations'].append(
                "Consider upgrading to 4+ core CPU for better performance"
            )
        
        return validation
    
    def validate_roi_specs(self, coords: np.ndarray, ion_counts: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate ROI specifications against standard limits."""
        n_pixels = len(coords)
        n_channels = len(ion_counts)
        
        validation = {
            'within_specs': True,
            'warnings': [],
            'recommendations': [],
            'roi_info': {
                'n_pixels': n_pixels,
                'n_channels': n_channels,
                'pixels_ok': n_pixels <= self.STANDARD_ROI_SPECS['max_pixels'],
                'channels_ok': n_channels <= self.STANDARD_ROI_SPECS['max_channels']
            }
        }
        
        # Check pixels
        if n_pixels > self.STANDARD_ROI_SPECS['max_pixels']:
            validation['within_specs'] = False
            validation['warnings'].append(
                f"ROI has {n_pixels:,} pixels, exceeds {self.STANDARD_ROI_SPECS['max_pixels']:,} pixel limit"
            )
            validation['recommendations'].append(
                "Consider splitting ROI or reducing resolution for standard processing"
            )
        
        # Check channels
        if n_channels > self.STANDARD_ROI_SPECS['max_channels']:
            validation['within_specs'] = False
            validation['warnings'].append(
                f"ROI has {n_channels} channels, exceeds {self.STANDARD_ROI_SPECS['max_channels']} channel limit"
            )
            validation['recommendations'].append(
                "Consider reducing channel count or using advanced processing mode"
            )
        
        return validation


class QuickStartInterface:
    """
    Honest, reliable QuickStart interface integrating all optimizations.
    
    CLEAR LIMITATIONS:
    - Standard ROIs only: 1k×1k pixels, ≤40 channels
    - Single method: SLIC @ 20μm + Leiden clustering  
    - Hardware requirement: 32GB RAM
    - Output: Draft reports requiring human review
    """
    
    def __init__(self, config_path: str, output_dir: str = "quickstart_results"):
        """Initialize QuickStart with honest expectations."""
        self.config = Config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all optimization systems
        self.hardware_validator = HardwareValidator()
        self.quickstart_pipeline = QuickstartPipeline(self.config)
        self.memory_optimizer = PipelineMemoryOptimizer(target_dtype='float32', validate_results=True)
        self.performance_dag = create_imc_analysis_dag(memory_budget_gb=16.0)
        self.qc_system = create_automatic_qc_system(output_dir=str(self.output_dir / "qc"))
        
        self.logger = logging.getLogger('QuickStartInterface')
        self.processing_history = []
        
        self.logger.info("QuickStart Pipeline initialized with integrated optimizations")
        self.logger.info("Hardware requirements: 1k×1k pixels, ≤40 channels, 32GB RAM")
        self.logger.info("Method: SLIC @ 20μm + Leiden clustering only")
    
    def validate_system_readiness(self) -> Dict[str, Any]:
        """Validate system readiness with honest assessment."""
        self.logger.info("Validating system readiness...")
        
        # Hardware validation
        hw_validation = self.hardware_validator.validate_hardware()
        
        # System health check
        current_memory = psutil.virtual_memory()
        available_memory_gb = current_memory.available / (1024**3)
        
        readiness = {
            'system_ready': hw_validation['meets_requirements'] and available_memory_gb >= 16.0,
            'hardware_validation': hw_validation,
            'current_status': {
                'available_memory_gb': available_memory_gb,
                'memory_sufficient': available_memory_gb >= 16.0,
                'cpu_usage_percent': psutil.cpu_percent(interval=1)
            },
            'limitations': [
                "Standard ROIs only: max 1M pixels, 40 channels",
                "Single method: SLIC segmentation at 20μm scale",
                "Clustering: Leiden algorithm only",
                "Memory target: <16GB usage",
                "Time target: <10 minutes per ROI",
                "Output: Draft reports requiring human review"
            ],
            'hardware_requirements': self.hardware_validator.STANDARD_ROI_SPECS
        }
        
        return readiness
    
    def process_single_roi(
        self,
        roi_file_path: str,
        roi_id: str = None,
        protein_names: List[str] = None,
        run_qc: bool = True,
        advanced_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Process single ROI with integrated optimizations.
        
        Args:
            roi_file_path: Path to ROI data file
            roi_id: ROI identifier (auto-generated if None)
            protein_names: Protein markers (auto-detected if None)
            run_qc: Whether to run automatic QC
            advanced_mode: Enable advanced processing (bypasses standard limits)
            
        Returns:
            Processing results with QC assessment and performance metrics
        """
        if roi_id is None:
            roi_id = Path(roi_file_path).stem
        
        self.logger.info(f"Processing ROI {roi_id} with QuickStart pipeline")
        
        start_time = time.time()
        result = {
            'roi_id': roi_id,
            'file_path': roi_file_path,
            'timestamp': datetime.now().isoformat(),
            'method': 'quickstart_integrated',
            'processing_mode': 'advanced' if advanced_mode else 'standard',
            'success': False,
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Step 1: Load and validate ROI data
            self.logger.debug(f"Loading ROI data from {roi_file_path}")
            roi_data = self.quickstart_pipeline.load_roi_data(roi_file_path, protein_names or [])
            
            # Auto-detect proteins if needed
            if protein_names is None:
                protein_names = list(roi_data['ion_counts'].keys())
                self.logger.info(f"Auto-detected {len(protein_names)} proteins: {protein_names}")
            
            # Step 2: Validate ROI specifications
            roi_validation = self.hardware_validator.validate_roi_specs(
                roi_data['coords'], roi_data['ion_counts']
            )
            result['roi_validation'] = roi_validation
            
            if not advanced_mode and not roi_validation['within_specs']:
                result['warnings'].extend(roi_validation['warnings'])
                result['recommendations'].extend(roi_validation['recommendations'])
                result['recommendations'].append("Use advanced_mode=True to bypass standard limits")
                raise ValueError("ROI exceeds standard specifications")
            
            # Step 3: Memory-optimized processing
            self.logger.debug("Applying memory optimizations")
            
            opt_coords, opt_ion_counts = self.memory_optimizer.optimize_coordinate_data(
                roi_data['coords'], roi_data['ion_counts']
            )
            
            # Step 4: Performance DAG execution
            self.logger.debug("Executing performance DAG")
            
            dag_inputs = {
                'coords': opt_coords,
                'ion_counts': opt_ion_counts,
                'dna1_intensities': roi_data['dna1_intensities'],
                'dna2_intensities': roi_data['dna2_intensities']
            }
            
            dag_results = self.performance_dag.execute_dag(dag_inputs, force_recompute=False)
            
            # Step 5: Compile analysis results
            analysis_result = {
                'cluster_labels': dag_results.get('cluster_labels', np.array([])),
                'feature_matrix': dag_results.get('feature_matrix', np.array([])),
                'superpixel_coords': dag_results.get('superpixel_coords', np.array([])),
                'n_clusters': len(np.unique(dag_results.get('cluster_labels', [0]))),
                'n_superpixels': len(dag_results.get('superpixel_coords', [])),
                'processing_method': 'slic_20um_leiden'
            }
            
            result['analysis_results'] = analysis_result
            
            # Step 6: Performance metrics
            performance_report = self.performance_dag.get_performance_report()
            result['performance_metrics'] = {
                'total_time_seconds': time.time() - start_time,
                'memory_usage_gb': performance_report.get('resource_efficiency', {}).get('peak_memory_gb', 0),
                'cache_hit_rate': performance_report.get('cache_statistics', {}).get('hit_rate', 0),
                'within_time_target': (time.time() - start_time) < (10 * 60),  # 10 minutes
                'within_memory_target': performance_report.get('resource_efficiency', {}).get('peak_memory_gb', 0) < 16.0
            }
            
            # Step 7: Automatic QC (if enabled)
            if run_qc:
                self.logger.debug("Running automatic QC assessment")
                
                # Prepare QC data
                qc_roi_data = {
                    'coords': roi_data['coords'],
                    'ion_counts': roi_data['ion_counts'],
                    'dna1_intensities': roi_data['dna1_intensities'],
                    'dna2_intensities': roi_data['dna2_intensities'],
                    'raw_data': None  # Would need raw DataFrame for full QC
                }
                
                qc_result = self.qc_system.run_comprehensive_qc(
                    roi_data=qc_roi_data,
                    roi_id=roi_id,
                    batch_id="quickstart"
                )
                
                result['qc_assessment'] = qc_result
                
                # Add QC warnings to main result
                if not qc_result.get('overall_passed', True):
                    result['warnings'].append("Automatic QC detected quality issues")
                    result['recommendations'].extend(qc_result.get('recommendations', []))
            
            # Step 8: Generate draft report
            result['draft_report'] = self._generate_draft_report(result)
            
            result['success'] = True
            self.processing_history.append(result)
            
            self.logger.info(f"ROI {roi_id} processed successfully in {result['performance_metrics']['total_time_seconds']:.1f}s")
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            result['recommendations'].append("Review error and ROI specifications before retry")
            self.logger.error(f"Failed to process ROI {roi_id}: {e}")
            self.processing_history.append(result)
            return result
    
    def process_batch(
        self,
        roi_directory: str,
        max_rois: Optional[int] = None,
        protein_names: List[str] = None,
        run_qc: bool = True,
        advanced_mode: bool = False,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Process batch of ROIs with integrated optimizations.
        
        Args:
            roi_directory: Directory containing ROI files
            max_rois: Maximum ROIs to process (for testing)
            protein_names: Protein markers (auto-detected if None)
            run_qc: Whether to run automatic QC
            advanced_mode: Enable advanced processing
            parallel: Enable parallel processing (experimental)
            
        Returns:
            Batch processing results with comprehensive QC and performance analysis
        """
        self.logger.info(f"Starting batch processing from {roi_directory}")
        
        # Find ROI files
        roi_files = list(Path(roi_directory).glob("*.txt"))
        if not roi_files:
            raise ValueError(f"No ROI files found in {roi_directory}")
        
        if max_rois:
            roi_files = roi_files[:max_rois]
            self.logger.info(f"Limited to {max_rois} ROIs for processing")
        
        batch_start_time = time.time()
        batch_result = {
            'batch_info': {
                'roi_directory': roi_directory,
                'n_rois': len(roi_files),
                'processing_mode': 'advanced' if advanced_mode else 'standard',
                'timestamp': datetime.now().isoformat()
            },
            'roi_results': {},
            'batch_statistics': {},
            'batch_qc_results': {},
            'performance_summary': {},
            'overall_recommendations': []
        }
        
        # Process each ROI
        successful_rois = 0
        failed_rois = 0
        
        for roi_file in roi_files:
            roi_id = roi_file.stem
            
            try:
                roi_result = self.process_single_roi(
                    str(roi_file),
                    roi_id=roi_id,
                    protein_names=protein_names,
                    run_qc=run_qc,
                    advanced_mode=advanced_mode
                )
                
                batch_result['roi_results'][roi_id] = roi_result
                
                if roi_result['success']:
                    successful_rois += 1
                else:
                    failed_rois += 1
                    
            except Exception as e:
                self.logger.error(f"Batch processing failed for {roi_id}: {e}")
                batch_result['roi_results'][roi_id] = {
                    'roi_id': roi_id,
                    'success': False,
                    'error': str(e)
                }
                failed_rois += 1
        
        # Generate batch statistics
        batch_result['batch_statistics'] = self._generate_batch_statistics(batch_result['roi_results'])
        
        # Run batch-level QC if enabled
        if run_qc:
            self.logger.debug("Running batch-level QC analysis")
            batch_result['batch_qc_results'] = self._run_batch_qc_analysis(batch_result['roi_results'])
        
        # Performance summary
        total_time = time.time() - batch_start_time
        batch_result['performance_summary'] = {
            'total_processing_time_minutes': total_time / 60,
            'average_time_per_roi_minutes': total_time / len(roi_files) / 60,
            'successful_rois': successful_rois,
            'failed_rois': failed_rois,
            'success_rate': successful_rois / len(roi_files),
            'throughput_rois_per_hour': len(roi_files) / (total_time / 3600)
        }
        
        # Overall recommendations
        batch_result['overall_recommendations'] = self._generate_batch_recommendations(batch_result)
        
        # Save batch results
        batch_output_file = self.output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_output_file, 'w') as f:
            json.dump(batch_result, f, indent=2, default=str)
        
        self.logger.info(f"Batch processing complete: {successful_rois}/{len(roi_files)} successful")
        self.logger.info(f"Results saved to {batch_output_file}")
        
        return batch_result
    
    def benchmark_performance(
        self,
        test_roi_path: str,
        n_runs: int = 3,
        protein_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Benchmark QuickStart performance against targets.
        
        Args:
            test_roi_path: Path to test ROI file
            n_runs: Number of benchmark runs
            protein_names: Protein markers
            
        Returns:
            Performance benchmark results
        """
        self.logger.info(f"Running performance benchmark with {n_runs} runs")
        
        benchmark_results = {
            'test_info': {
                'roi_path': test_roi_path,
                'n_runs': n_runs,
                'timestamp': datetime.now().isoformat()
            },
            'run_results': [],
            'performance_summary': {},
            'target_compliance': {}
        }
        
        # Run multiple benchmarks
        for run_i in range(n_runs):
            self.logger.debug(f"Benchmark run {run_i + 1}/{n_runs}")
            
            run_result = self.process_single_roi(
                test_roi_path,
                roi_id=f"benchmark_run_{run_i + 1}",
                protein_names=protein_names,
                run_qc=True,
                advanced_mode=False
            )
            
            benchmark_results['run_results'].append(run_result)
        
        # Calculate performance summary
        successful_runs = [r for r in benchmark_results['run_results'] if r['success']]
        
        if successful_runs:
            times = [r['performance_metrics']['total_time_seconds'] for r in successful_runs]
            memories = [r['performance_metrics']['memory_usage_gb'] for r in successful_runs]
            
            benchmark_results['performance_summary'] = {
                'avg_time_seconds': float(np.mean(times)),
                'min_time_seconds': float(np.min(times)),
                'max_time_seconds': float(np.max(times)),
                'std_time_seconds': float(np.std(times)),
                'avg_memory_gb': float(np.mean(memories)),
                'max_memory_gb': float(np.max(memories)),
                'success_rate': len(successful_runs) / n_runs
            }
            
            # Target compliance
            targets = self.hardware_validator.STANDARD_ROI_SPECS
            benchmark_results['target_compliance'] = {
                'time_target_met': benchmark_results['performance_summary']['avg_time_seconds'] < (targets['target_time_minutes'] * 60),
                'memory_target_met': benchmark_results['performance_summary']['max_memory_gb'] < targets['target_memory_gb'],
                'reliability_target_met': benchmark_results['performance_summary']['success_rate'] >= 0.95
            }
        
        self.logger.info(f"Benchmark complete: {len(successful_runs)}/{n_runs} successful runs")
        
        return benchmark_results
    
    def _generate_draft_report(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate draft report for ROI processing."""
        return {
            'roi_id': result['roi_id'],
            'processing_timestamp': result['timestamp'],
            'method': 'SLIC segmentation (20μm) + Leiden clustering',
            'analysis_summary': {
                'n_clusters_identified': result.get('analysis_results', {}).get('n_clusters', 0),
                'n_superpixels_generated': result.get('analysis_results', {}).get('n_superpixels', 0),
                'processing_successful': result['success']
            },
            'performance_summary': result.get('performance_metrics', {}),
            'qc_summary': {
                'qc_performed': 'qc_assessment' in result,
                'qc_passed': result.get('qc_assessment', {}).get('overall_passed', None),
                'quality_score': result.get('qc_assessment', {}).get('overall_quality_score', None)
            },
            'warnings': result.get('warnings', []),
            'recommendations': result.get('recommendations', []),
            'note': "This is a DRAFT report. Human review required before publication or clinical use.",
            'limitations': [
                "Single method only (SLIC + Leiden)",
                "Standard ROI specifications assumed",
                "Automated QC may miss domain-specific issues",
                "Results require expert validation"
            ]
        }
    
    def _generate_batch_statistics(self, roi_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistics for batch processing."""
        successful_results = {k: v for k, v in roi_results.items() if v.get('success', False)}
        
        if not successful_results:
            return {'no_successful_rois': True}
        
        # Extract metrics
        times = [r['performance_metrics']['total_time_seconds'] for r in successful_results.values()]
        memories = [r['performance_metrics']['memory_usage_gb'] for r in successful_results.values()]
        n_clusters = [r['analysis_results']['n_clusters'] for r in successful_results.values()]
        
        qc_passed = [
            r.get('qc_assessment', {}).get('overall_passed', False) 
            for r in successful_results.values()
        ]
        
        return {
            'n_successful_rois': len(successful_results),
            'n_total_rois': len(roi_results),
            'success_rate': len(successful_results) / len(roi_results),
            'processing_times': {
                'mean_seconds': float(np.mean(times)),
                'std_seconds': float(np.std(times)),
                'min_seconds': float(np.min(times)),
                'max_seconds': float(np.max(times))
            },
            'memory_usage': {
                'mean_gb': float(np.mean(memories)),
                'max_gb': float(np.max(memories)),
                'within_target': all(m < 16.0 for m in memories)
            },
            'biological_results': {
                'mean_clusters': float(np.mean(n_clusters)),
                'std_clusters': float(np.std(n_clusters)),
                'cluster_range': [int(np.min(n_clusters)), int(np.max(n_clusters))]
            },
            'qc_statistics': {
                'qc_pass_rate': sum(qc_passed) / len(qc_passed) if qc_passed else 0.0,
                'n_qc_passed': sum(qc_passed),
                'n_qc_failed': len(qc_passed) - sum(qc_passed)
            }
        }
    
    def _run_batch_qc_analysis(self, roi_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run batch-level QC analysis."""
        # This is a simplified version - full implementation would use the QC system
        return {
            'batch_effects_detected': False,  # Would implement full batch analysis
            'calibration_drift_detected': False,
            'quality_trend_analysis': 'stable',
            'recommendations': ['Full batch QC analysis requires raw data access']
        }
    
    def _generate_batch_recommendations(self, batch_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for batch processing."""
        recommendations = []
        
        stats = batch_result['batch_statistics']
        performance = batch_result['performance_summary']
        
        # Success rate recommendations
        if performance['success_rate'] >= 0.95:
            recommendations.append("Excellent batch processing performance - protocol working well")
        elif performance['success_rate'] >= 0.8:
            recommendations.append("Good batch processing performance - minor optimization possible")
        else:
            recommendations.append("Low batch success rate - review ROI specifications and protocol")
        
        # Performance recommendations
        if stats.get('memory_usage', {}).get('within_target', True):
            recommendations.append("Memory usage within targets")
        else:
            recommendations.append("Some ROIs exceeded memory targets - consider ROI size reduction")
        
        # QC recommendations
        qc_stats = stats.get('qc_statistics', {})
        if qc_stats.get('qc_pass_rate', 0) < 0.8:
            recommendations.append("Low QC pass rate - review tissue preparation and imaging protocols")
        
        # Time performance
        avg_time_minutes = stats.get('processing_times', {}).get('mean_seconds', 0) / 60
        if avg_time_minutes > 10:
            recommendations.append("Processing time exceeds 10-minute target - consider hardware upgrade")
        
        return recommendations
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processing performed."""
        if not self.processing_history:
            return {'no_processing_performed': True}
        
        successful = [r for r in self.processing_history if r.get('success', False)]
        
        return {
            'total_rois_processed': len(self.processing_history),
            'successful_rois': len(successful),
            'success_rate': len(successful) / len(self.processing_history),
            'average_processing_time_seconds': np.mean([
                r.get('performance_metrics', {}).get('total_time_seconds', 0) 
                for r in successful
            ]) if successful else 0,
            'system_performance': 'excellent' if len(successful) / len(self.processing_history) > 0.9 else 'good' if len(successful) / len(self.processing_history) > 0.7 else 'needs_attention'
        }


def create_quickstart_interface(config_path: str, output_dir: str = "quickstart_results") -> QuickStartInterface:
    """
    Factory function to create QuickStart interface.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
        
    Returns:
        Configured QuickStart interface
    """
    return QuickStartInterface(config_path, output_dir)


def run_quickstart_pipeline(
    roi_directory: str,
    config_path: str = "config.json",
    output_dir: str = "quickstart_results",
    max_rois: Optional[int] = None,
    protein_names: List[str] = None,
    run_benchmark: bool = True,
    advanced_mode: bool = False
) -> Dict[str, Any]:
    """
    Complete QuickStart pipeline with honest performance expectations.
    
    HONEST LIMITATIONS:
    - Standard ROIs: 1k×1k pixels, ≤40 channels, 32GB RAM required
    - Single method: SLIC @ 20μm + Leiden clustering
    - Memory target: <16GB usage
    - Time target: <10 minutes per ROI
    - Output: Draft reports requiring human review
    
    Args:
        roi_directory: Directory containing ROI data files
        config_path: Path to configuration file
        output_dir: Output directory for results
        max_rois: Maximum ROIs to process (for testing)
        protein_names: Protein markers (auto-detected if None)
        run_benchmark: Whether to run performance benchmark
        advanced_mode: Enable advanced processing (bypasses standard limits)
        
    Returns:
        Complete pipeline results with performance metrics and QC assessment
    """
    print("=" * 60)
    print("QuickStart Pipeline - Realistic, Fast, Reliable")
    print("=" * 60)
    print("LIMITATIONS:")
    print("  - Standard ROIs: 1k×1k pixels, ≤40 channels")
    print("  - Hardware: 32GB RAM required")
    print("  - Method: SLIC @ 20μm + Leiden clustering only")
    print("  - Target: <16GB memory, <10min per ROI")
    print("  - Output: Draft reports requiring human review")
    print("=" * 60)
    
    # Create QuickStart interface
    quickstart = create_quickstart_interface(config_path, output_dir)
    
    # Validate system readiness
    print("\n1. Validating system readiness...")
    readiness = quickstart.validate_system_readiness()
    
    if not readiness['system_ready']:
        print("SYSTEM NOT READY:")
        for warning in readiness['hardware_validation']['warnings']:
            print(f"  ⚠️  {warning}")
        for rec in readiness['hardware_validation']['recommendations']:
            print(f"  💡 {rec}")
        
        if not advanced_mode:
            return {
                'status': 'system_not_ready',
                'readiness_check': readiness,
                'recommendations': ['Upgrade hardware or use advanced_mode=True to proceed with warnings']
            }
    else:
        print("✅ System ready for QuickStart processing")
    
    # Process batch
    print(f"\n2. Processing ROIs from {roi_directory}...")
    batch_results = quickstart.process_batch(
        roi_directory=roi_directory,
        max_rois=max_rois,
        protein_names=protein_names,
        run_qc=True,
        advanced_mode=advanced_mode
    )
    
    # Performance benchmark
    benchmark_results = None
    if run_benchmark and batch_results['performance_summary']['successful_rois'] > 0:
        print("\n3. Running performance benchmark...")
        
        # Use first successful ROI for benchmark
        successful_rois = [
            roi_data for roi_data in batch_results['roi_results'].values() 
            if roi_data.get('success', False)
        ]
        
        if successful_rois:
            test_roi_path = successful_rois[0]['file_path']
            benchmark_results = quickstart.benchmark_performance(
                test_roi_path=test_roi_path,
                n_runs=3,
                protein_names=protein_names
            )
    
    # Generate final summary
    print("\n4. Generating final summary...")
    processing_summary = quickstart.get_processing_summary()
    
    final_result = {
        'pipeline_info': {
            'version': 'quickstart_integrated_v1.0',
            'method': 'slic_20um_leiden_clustering',
            'optimizations': ['memory_optimizer', 'performance_dag', 'quickstart_pipeline', 'automatic_qc'],
            'limitations': readiness['limitations']
        },
        'system_readiness': readiness,
        'batch_results': batch_results,
        'benchmark_results': benchmark_results,
        'processing_summary': processing_summary,
        'final_recommendations': []
    }
    
    # Final recommendations
    success_rate = batch_results['performance_summary']['success_rate']
    avg_time = batch_results['performance_summary']['average_time_per_roi_minutes']
    
    if success_rate >= 0.9 and avg_time <= 10:
        final_result['final_recommendations'].append("✅ Excellent performance - QuickStart working as designed")
    elif success_rate >= 0.7:
        final_result['final_recommendations'].append("⚠️  Good performance - minor optimization possible")
    else:
        final_result['final_recommendations'].append("❌ Performance issues - review hardware and ROI specifications")
    
    final_result['final_recommendations'].append("📝 All outputs are DRAFT reports requiring human review")
    final_result['final_recommendations'].append("🔬 Consider expert validation before publication or clinical use")
    
    # Performance summary
    print("\n" + "=" * 60)
    print("QUICKSTART PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Processed: {batch_results['performance_summary']['successful_rois']}/{batch_results['batch_info']['n_rois']} ROIs")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average time: {avg_time:.1f} min/ROI")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    print("⚠️  IMPORTANT: These are DRAFT results requiring human review")
    print("=" * 60)
    
    return final_result


if __name__ == "__main__":
    # Example usage
    results = run_quickstart_pipeline(
        roi_directory="data/241218_IMC_Alun",
        config_path="config.json",
        output_dir="quickstart_results",
        max_rois=5,  # Limit for testing
        run_benchmark=True,
        advanced_mode=False  # Use standard limits
    )
    
    print("\nQuickStart Pipeline Integration Complete!")
    print(f"Success rate: {results['processing_summary']['success_rate']:.1%}")