"""
Segmentation Method Benchmarking and Comparison

Provides comprehensive benchmarking tools for comparing SLIC vs Grid segmentation
methods across multiple dimensions: performance, quality, and biological relevance.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

if TYPE_CHECKING:
    from ..config import Config

# Import both segmentation methods
from .slic_segmentation import slic_pipeline
from .grid_segmentation import grid_pipeline, GridMetrics


@dataclass
class BenchmarkResult:
    """Single benchmark result for one method."""
    method: str
    target_scale_um: float
    n_segments: int
    generation_time: float
    aggregation_time: float
    total_time: float
    memory_usage_mb: float
    boundary_coherence: float
    regularity_score: float
    coverage_ratio: float
    segments_per_second: float
    

@dataclass
class ComparisonMetrics:
    """Comparison metrics between two segmentation methods."""
    speedup_factor: float
    memory_efficiency: float
    segment_count_ratio: float
    boundary_coherence_diff: float
    regularity_diff: float
    

class SegmentationBenchmark:
    """Comprehensive benchmarking suite for segmentation methods."""
    
    def __init__(self, config: Optional['Config'] = None):
        self.config = config
        self.logger = logging.getLogger('SegmentationBenchmark')
        self.results = []
        
    def benchmark_single_method(
        self,
        method: str,
        coords: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        dna1_intensities: np.ndarray,
        dna2_intensities: np.ndarray,
        target_scale_um: float,
        n_segments: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Benchmark a single segmentation method.
        
        Args:
            method: 'slic' or 'grid'
            coords: Coordinate data
            ion_counts: Ion count data
            dna1_intensities: DNA1 channel data
            dna2_intensities: DNA2 channel data
            target_scale_um: Target scale for segmentation
            n_segments: Optional fixed number of segments
            
        Returns:
            BenchmarkResult with performance and quality metrics
        """
        self.logger.info(f"Benchmarking {method} method at {target_scale_um}μm scale")
        
        start_time = time.time()
        
        if method == 'slic':
            pipeline_func = slic_pipeline
        elif method == 'grid':
            pipeline_func = grid_pipeline
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Run the pipeline
        result = pipeline_func(
            coords=coords,
            ion_counts=ion_counts,
            dna1_intensities=dna1_intensities,
            dna2_intensities=dna2_intensities,
            target_scale_um=target_scale_um,
            n_segments=n_segments,
            config=self.config
        )
        
        total_time = time.time() - start_time
        
        # Extract metrics
        n_segments_actual = result.get('n_segments_used', 0)
        
        # Performance metrics
        if method == 'grid' and 'metrics' in result:
            metrics = result['metrics']
            generation_time = metrics.generation_time
            aggregation_time = metrics.aggregation_time
            memory_usage_mb = metrics.memory_usage_mb
            boundary_coherence = metrics.boundary_coherence
            coverage_ratio = metrics.coverage_ratio
        else:
            # Extract from result structure for SLIC
            generation_time = total_time * 0.7  # Estimate (SLIC spends most time on segmentation)
            aggregation_time = total_time * 0.3  # Estimate
            memory_usage_mb = self._estimate_memory_usage(result)
            boundary_coherence = self._compute_slic_boundary_coherence(result)
            coverage_ratio = self._compute_coverage_ratio(result)
        
        # Regularity score
        regularity_score = self._compute_regularity_score(result, method)
        
        # Segments per second
        segments_per_second = n_segments_actual / total_time if total_time > 0 else 0
        
        benchmark_result = BenchmarkResult(
            method=method,
            target_scale_um=target_scale_um,
            n_segments=n_segments_actual,
            generation_time=generation_time,
            aggregation_time=aggregation_time,
            total_time=total_time,
            memory_usage_mb=memory_usage_mb,
            boundary_coherence=boundary_coherence,
            regularity_score=regularity_score,
            coverage_ratio=coverage_ratio,
            segments_per_second=segments_per_second
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def benchmark_comparison(
        self,
        coords: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        dna1_intensities: np.ndarray,
        dna2_intensities: np.ndarray,
        scales_um: List[float] = [10.0, 20.0, 40.0],
        n_trials: int = 3
    ) -> Dict[str, Any]:
        """
        Run comprehensive comparison between SLIC and Grid methods.
        
        Args:
            coords: Coordinate data
            ion_counts: Ion count data
            dna1_intensities: DNA1 channel data
            dna2_intensities: DNA2 channel data
            scales_um: List of scales to test
            n_trials: Number of trials per configuration
            
        Returns:
            Dictionary with comprehensive comparison results
        """
        self.logger.info(f"Running comparison benchmark across {len(scales_um)} scales with {n_trials} trials each")
        
        comparison_results = {
            'scales_tested': scales_um,
            'n_trials': n_trials,
            'detailed_results': [],
            'summary_statistics': {},
            'performance_comparison': {},
            'quality_comparison': {}
        }
        
        for scale_um in scales_um:
            self.logger.info(f"Testing scale {scale_um}μm")
            
            scale_results = {'scale_um': scale_um, 'trials': []}
            
            for trial in range(n_trials):
                self.logger.debug(f"Trial {trial + 1}/{n_trials}")
                
                # Benchmark SLIC
                slic_result = self.benchmark_single_method(
                    'slic', coords, ion_counts, dna1_intensities, dna2_intensities, scale_um
                )
                
                # Benchmark Grid
                grid_result = self.benchmark_single_method(
                    'grid', coords, ion_counts, dna1_intensities, dna2_intensities, scale_um
                )
                
                # Compute comparison metrics
                comparison = self._compute_comparison_metrics(slic_result, grid_result)
                
                trial_result = {
                    'trial': trial,
                    'slic': asdict(slic_result),
                    'grid': asdict(grid_result),
                    'comparison': asdict(comparison)
                }
                
                scale_results['trials'].append(trial_result)
            
            comparison_results['detailed_results'].append(scale_results)
        
        # Compute summary statistics
        comparison_results['summary_statistics'] = self._compute_summary_statistics(comparison_results)
        
        return comparison_results
    
    def benchmark_scalability(
        self,
        base_coords: np.ndarray,
        base_ion_counts: Dict[str, np.ndarray],
        base_dna1: np.ndarray,
        base_dna2: np.ndarray,
        scaling_factors: List[float] = [0.5, 1.0, 2.0, 4.0, 8.0]
    ) -> Dict[str, Any]:
        """
        Test scalability of both methods with increasing data sizes.
        
        Args:
            base_coords: Base coordinate data
            base_ion_counts: Base ion count data
            base_dna1: Base DNA1 data
            base_dna2: Base DNA2 data
            scaling_factors: Factors to scale data size
            
        Returns:
            Dictionary with scalability results
        """
        self.logger.info(f"Running scalability benchmark with factors: {scaling_factors}")
        
        scalability_results = {
            'scaling_factors': scaling_factors,
            'results': []
        }
        
        for factor in scaling_factors:
            self.logger.info(f"Testing scaling factor {factor}x")
            
            # Scale the data
            n_points = int(len(base_coords) * factor)
            if factor < 1.0:
                # Subsample
                indices = np.random.choice(len(base_coords), n_points, replace=False)
                coords = base_coords[indices]
                dna1 = base_dna1[indices]
                dna2 = base_dna2[indices]
                ion_counts = {k: v[indices] for k, v in base_ion_counts.items()}
            else:
                # Replicate with noise
                n_replicates = int(factor)
                coords_list = [base_coords]
                dna1_list = [base_dna1]
                dna2_list = [base_dna2]
                ion_counts_list = [{k: v for k, v in base_ion_counts.items()}]
                
                for rep in range(1, n_replicates):
                    # Add small spatial offset to avoid exact overlap
                    offset = np.random.normal(0, 1, base_coords.shape)
                    coords_list.append(base_coords + offset)
                    dna1_list.append(base_dna1)
                    dna2_list.append(base_dna2)
                    ion_counts_list.append({k: v for k, v in base_ion_counts.items()})
                
                coords = np.vstack(coords_list)
                dna1 = np.hstack(dna1_list)
                dna2 = np.hstack(dna2_list)
                ion_counts = {}
                for k in base_ion_counts.keys():
                    ion_counts[k] = np.hstack([ic[k] for ic in ion_counts_list])
            
            # Benchmark both methods
            slic_result = self.benchmark_single_method(
                'slic', coords, ion_counts, dna1, dna2, 20.0
            )
            
            grid_result = self.benchmark_single_method(
                'grid', coords, ion_counts, dna1, dna2, 20.0
            )
            
            scalability_results['results'].append({
                'scaling_factor': factor,
                'n_points': len(coords),
                'slic': asdict(slic_result),
                'grid': asdict(grid_result)
            })
        
        return scalability_results
    
    def _compute_comparison_metrics(
        self, 
        slic_result: BenchmarkResult, 
        grid_result: BenchmarkResult
    ) -> ComparisonMetrics:
        """Compute comparison metrics between SLIC and Grid results."""
        
        speedup_factor = slic_result.total_time / grid_result.total_time if grid_result.total_time > 0 else float('inf')
        memory_efficiency = slic_result.memory_usage_mb / grid_result.memory_usage_mb if grid_result.memory_usage_mb > 0 else 1.0
        segment_count_ratio = slic_result.n_segments / grid_result.n_segments if grid_result.n_segments > 0 else 1.0
        boundary_coherence_diff = slic_result.boundary_coherence - grid_result.boundary_coherence
        regularity_diff = slic_result.regularity_score - grid_result.regularity_score
        
        return ComparisonMetrics(
            speedup_factor=speedup_factor,
            memory_efficiency=memory_efficiency,
            segment_count_ratio=segment_count_ratio,
            boundary_coherence_diff=boundary_coherence_diff,
            regularity_diff=regularity_diff
        )
    
    def _compute_summary_statistics(self, comparison_results: Dict) -> Dict[str, Any]:
        """Compute summary statistics across all trials and scales."""
        
        all_speedups = []
        all_memory_ratios = []
        all_coherence_diffs = []
        
        for scale_result in comparison_results['detailed_results']:
            for trial in scale_result['trials']:
                comp = trial['comparison']
                all_speedups.append(comp['speedup_factor'])
                all_memory_ratios.append(comp['memory_efficiency'])
                all_coherence_diffs.append(comp['boundary_coherence_diff'])
        
        return {
            'average_speedup': np.mean(all_speedups),
            'std_speedup': np.std(all_speedups),
            'min_speedup': np.min(all_speedups),
            'max_speedup': np.max(all_speedups),
            'average_memory_ratio': np.mean(all_memory_ratios),
            'average_coherence_diff': np.mean(all_coherence_diffs),
            'std_coherence_diff': np.std(all_coherence_diffs)
        }
    
    def _estimate_memory_usage(self, result: Dict) -> float:
        """Estimate memory usage from pipeline result."""
        memory_mb = 0
        
        # Estimate from array sizes
        for key in ['superpixel_labels', 'composite_dna']:
            if key in result and isinstance(result[key], np.ndarray):
                memory_mb += result[key].nbytes / (1024**2)
        
        # Estimate from superpixel counts
        if 'superpixel_counts' in result:
            for arr in result['superpixel_counts'].values():
                if isinstance(arr, np.ndarray):
                    memory_mb += arr.nbytes / (1024**2)
        
        return memory_mb
    
    def _compute_slic_boundary_coherence(self, result: Dict) -> float:
        """Compute boundary coherence for SLIC results."""
        # This would require similar analysis to grid boundary quality
        # For now, return a placeholder based on superpixel properties
        if 'superpixel_props' in result and result['superpixel_props']:
            # Use average solidity as proxy for boundary coherence
            solidities = [props.get('solidity', 0) for props in result['superpixel_props'].values()]
            return np.mean(solidities) if solidities else 0.0
        return 0.0
    
    def _compute_coverage_ratio(self, result: Dict) -> float:
        """Compute tissue coverage ratio."""
        if 'superpixel_labels' in result:
            labels = result['superpixel_labels']
            if labels.size > 0:
                valid_pixels = np.sum(labels >= 0)
                total_pixels = labels.size
                return valid_pixels / total_pixels if total_pixels > 0 else 0.0
        return 0.0
    
    def _compute_regularity_score(self, result: Dict, method: str) -> float:
        """Compute regularity score for segmentation."""
        if method == 'grid':
            # Grid should have perfect regularity
            return result.get('boundary_quality', {}).get('grid_regularity', 1.0)
        else:
            # For SLIC, compute based on superpixel size variation
            if 'superpixel_props' in result and result['superpixel_props']:
                areas = [props.get('area_um2', 0) for props in result['superpixel_props'].values()]
                if len(areas) > 1:
                    cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
                    return 1.0 / (1.0 + cv)  # Higher score = more regular
            return 0.5  # Default middle score for SLIC
    
    def generate_benchmark_report(
        self, 
        comparison_results: Dict, 
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Generate comprehensive benchmark report with plots and tables.
        
        Args:
            comparison_results: Results from benchmark_comparison
            output_dir: Directory to save report files
            
        Returns:
            Dictionary mapping report type to file path
        """
        if output_dir is None:
            output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        report_files = {}
        
        # 1. Performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SLIC vs Grid Segmentation Performance Comparison', fontsize=16)
        
        scales = comparison_results['scales_tested']
        
        # Extract mean performance metrics across trials
        slic_times = []
        grid_times = []
        slic_memory = []
        grid_memory = []
        speedups = []
        coherence_diffs = []
        
        for scale_result in comparison_results['detailed_results']:
            scale_slic_times = [trial['slic']['total_time'] for trial in scale_result['trials']]
            scale_grid_times = [trial['grid']['total_time'] for trial in scale_result['trials']]
            scale_slic_memory = [trial['slic']['memory_usage_mb'] for trial in scale_result['trials']]
            scale_grid_memory = [trial['grid']['memory_usage_mb'] for trial in scale_result['trials']]
            scale_speedups = [trial['comparison']['speedup_factor'] for trial in scale_result['trials']]
            scale_coherence = [trial['comparison']['boundary_coherence_diff'] for trial in scale_result['trials']]
            
            slic_times.append(np.mean(scale_slic_times))
            grid_times.append(np.mean(scale_grid_times))
            slic_memory.append(np.mean(scale_slic_memory))
            grid_memory.append(np.mean(scale_grid_memory))
            speedups.append(np.mean(scale_speedups))
            coherence_diffs.append(np.mean(scale_coherence))
        
        # Plot 1: Execution time
        axes[0, 0].plot(scales, slic_times, 'o-', label='SLIC', color='blue')
        axes[0, 0].plot(scales, grid_times, 's-', label='Grid', color='red')
        axes[0, 0].set_xlabel('Scale (μm)')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].set_title('Execution Time Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Memory usage
        axes[0, 1].plot(scales, slic_memory, 'o-', label='SLIC', color='blue')
        axes[0, 1].plot(scales, grid_memory, 's-', label='Grid', color='red')
        axes[0, 1].set_xlabel('Scale (μm)')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].set_title('Memory Usage Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Speedup factor
        axes[1, 0].plot(scales, speedups, 'o-', color='green')
        axes[1, 0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Scale (μm)')
        axes[1, 0].set_ylabel('Speedup Factor (SLIC time / Grid time)')
        axes[1, 0].set_title('Grid Performance Advantage')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Boundary coherence difference
        axes[1, 1].plot(scales, coherence_diffs, 'o-', color='purple')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Scale (μm)')
        axes[1, 1].set_ylabel('Boundary Coherence Diff (SLIC - Grid)')
        axes[1, 1].set_title('Boundary Quality Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        performance_plot_path = output_dir / "performance_comparison.png"
        fig.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        report_files['performance_plot'] = performance_plot_path
        
        # 2. Summary statistics table
        summary_stats = comparison_results['summary_statistics']
        
        summary_df = pd.DataFrame([{
            'Metric': 'Average Speedup (Grid advantage)',
            'Value': f"{summary_stats['average_speedup']:.2f}x",
            'Std Dev': f"±{summary_stats['std_speedup']:.2f}",
            'Range': f"{summary_stats['min_speedup']:.2f}x - {summary_stats['max_speedup']:.2f}x"
        }, {
            'Metric': 'Memory Efficiency (SLIC/Grid ratio)',
            'Value': f"{summary_stats['average_memory_ratio']:.2f}",
            'Std Dev': '',
            'Range': ''
        }, {
            'Metric': 'Boundary Coherence Diff (SLIC advantage)',
            'Value': f"{summary_stats['average_coherence_diff']:.3f}",
            'Std Dev': f"±{summary_stats['std_coherence_diff']:.3f}",
            'Range': ''
        }])
        
        summary_path = output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        report_files['summary_table'] = summary_path
        
        # 3. Detailed results table
        detailed_data = []
        for scale_result in comparison_results['detailed_results']:
            scale_um = scale_result['scale_um']
            for trial in scale_result['trials']:
                detailed_data.append({
                    'Scale_um': scale_um,
                    'Trial': trial['trial'],
                    'SLIC_Time': trial['slic']['total_time'],
                    'Grid_Time': trial['grid']['total_time'],
                    'SLIC_Memory_MB': trial['slic']['memory_usage_mb'],
                    'Grid_Memory_MB': trial['grid']['memory_usage_mb'],
                    'SLIC_Segments': trial['slic']['n_segments'],
                    'Grid_Segments': trial['grid']['n_segments'],
                    'Speedup_Factor': trial['comparison']['speedup_factor'],
                    'Memory_Ratio': trial['comparison']['memory_efficiency'],
                    'Coherence_Diff': trial['comparison']['boundary_coherence_diff']
                })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_path = output_dir / "detailed_results.csv"
        detailed_df.to_csv(detailed_path, index=False)
        report_files['detailed_table'] = detailed_path
        
        # 4. Method comparison text report
        report_text = f"""
# Segmentation Method Comparison Report

## Summary
This report compares SLIC superpixel segmentation with regular grid segmentation
across {len(scales)} scales with {comparison_results['n_trials']} trials each.

## Key Findings

### Performance
- **Grid Advantage**: {summary_stats['average_speedup']:.2f}x faster on average
- **Speed Range**: {summary_stats['min_speedup']:.2f}x to {summary_stats['max_speedup']:.2f}x speedup
- **Memory Efficiency**: SLIC uses {summary_stats['average_memory_ratio']:.2f}x memory vs Grid

### Quality Trade-offs
- **Boundary Coherence**: SLIC shows {abs(summary_stats['average_coherence_diff']):.3f} {'advantage' if summary_stats['average_coherence_diff'] > 0 else 'disadvantage'} 
- **Regularity**: Grid provides perfect geometric regularity
- **Morphology Adaptation**: SLIC adapts to tissue morphology, Grid provides uniform sampling

## Method Characteristics

### Grid Segmentation
- **Advantages**: 
  - Predictable, fast performance
  - Perfect geometric regularity
  - Simple, interpretable results
  - Consistent segment sizes
- **Disadvantages**:
  - No adaptation to tissue morphology
  - May cut across biological boundaries
  - Less biologically relevant segmentation

### SLIC Segmentation  
- **Advantages**:
  - Adapts to tissue morphology
  - Respects DNA-based boundaries
  - More biologically relevant
  - Variable segment sizes match tissue structure
- **Disadvantages**:
  - Slower, more variable performance
  - Complex parameter tuning
  - Less predictable segment sizes

## Recommendations

1. **For exploratory analysis**: Use Grid for fast hypothesis generation
2. **For publication**: Use SLIC for morphologically-aware analysis
3. **For benchmarking**: Use Grid as honest baseline comparison
4. **For large-scale studies**: Consider Grid for computational efficiency

## Test Configuration
- Scales tested: {scales} μm
- Trials per scale: {comparison_results['n_trials']}
- Methods compared: SLIC vs Grid
"""
        
        report_path = output_dir / "comparison_report.md"
        with open(report_path, 'w') as f:
            f.write(report_text)
        report_files['text_report'] = report_path
        
        self.logger.info(f"Benchmark report generated in {output_dir}")
        return report_files


def create_segmentation_method_factory(config: Optional['Config'] = None):
    """
    Factory function to create segmentation method switcher.
    
    Returns a function that can switch between SLIC and Grid methods
    with identical interfaces.
    """
    
    def segment_with_method(
        method: str,
        coords: np.ndarray,
        ion_counts: Dict[str, np.ndarray],
        dna1_intensities: np.ndarray,
        dna2_intensities: np.ndarray,
        target_scale_um: float = 20.0,
        **kwargs
    ) -> Dict:
        """
        Run segmentation with specified method.
        
        Args:
            method: 'slic' or 'grid'
            coords: Coordinate data
            ion_counts: Ion count data
            dna1_intensities: DNA1 channel data  
            dna2_intensities: DNA2 channel data
            target_scale_um: Target scale
            **kwargs: Additional arguments passed to pipeline
            
        Returns:
            Pipeline results with identical structure
        """
        if method.lower() == 'slic':
            return slic_pipeline(
                coords, ion_counts, dna1_intensities, dna2_intensities,
                target_scale_um=target_scale_um, config=config, **kwargs
            )
        elif method.lower() == 'grid':
            return grid_pipeline(
                coords, ion_counts, dna1_intensities, dna2_intensities,
                target_scale_um=target_scale_um, config=config, **kwargs
            )
        else:
            raise ValueError(f"Unknown segmentation method: {method}. Use 'slic' or 'grid'.")
    
    return segment_with_method