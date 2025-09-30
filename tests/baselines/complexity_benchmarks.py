"""
Complexity analysis utilities for performance baseline testing.

Provides tools to measure and analyze algorithmic complexity rather than
relying on brittle absolute timing measurements.
"""

import time
import numpy as np
import tracemalloc
import gc
import psutil
import os
from typing import List, Tuple, Dict, Callable, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI
import matplotlib.pyplot as plt


@dataclass
class BenchmarkResult:
    """Result of a single benchmark measurement."""
    input_size: int
    execution_time: float
    peak_memory_mb: float
    output_size: Optional[int] = None
    metadata: Dict[str, Any] = None


@dataclass  
class ComplexityAnalysis:
    """Analysis of algorithm complexity from benchmark results."""
    algorithm_name: str
    input_sizes: List[int]
    execution_times: List[float]
    memory_usage: List[float]
    estimated_complexity: str
    growth_rate: float
    r_squared: float
    performance_grade: str
    notes: str


class ComplexityBenchmarker:
    """Utility for measuring and analyzing algorithm complexity."""
    
    def __init__(self, warmup_runs: int = 2, measurement_runs: int = 3):
        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs
        self.process = psutil.Process(os.getpid())
    
    def benchmark_function(self, 
                          func: Callable,
                          input_generator: Callable[[int], Any],
                          sizes: List[int],
                          function_name: str = "unknown") -> List[BenchmarkResult]:
        """Benchmark a function across different input sizes."""
        results = []
        
        for size in sizes:
            print(f"Benchmarking {function_name} with input size {size}...")
            
            # Generate input data
            input_data = input_generator(size)
            
            # Warmup runs
            for _ in range(self.warmup_runs):
                try:
                    func(input_data)
                except Exception as e:
                    print(f"Warning: Warmup failed for size {size}: {e}")
                    break
            
            # Measurement runs
            times = []
            memories = []
            
            for run in range(self.measurement_runs):
                # Force garbage collection
                gc.collect()
                
                # Start memory tracking
                tracemalloc.start()
                initial_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
                
                # Execute function
                start_time = time.time()
                try:
                    result = func(input_data)
                    execution_time = time.time() - start_time
                    
                    # Measure peak memory
                    current, peak = tracemalloc.get_traced_memory()
                    peak_memory = peak / (1024 * 1024)  # MB
                    
                    times.append(execution_time)
                    memories.append(peak_memory)
                    
                    # Determine output size if possible
                    output_size = None
                    if hasattr(result, '__len__'):
                        try:
                            output_size = len(result)
                        except:
                            pass
                    elif hasattr(result, 'shape'):
                        try:
                            output_size = result.shape[0] if result.shape else None
                        except:
                            pass
                
                except Exception as e:
                    print(f"Error in measurement run {run} for size {size}: {e}")
                    times.append(float('inf'))
                    memories.append(float('inf'))
                    output_size = None
                
                finally:
                    tracemalloc.stop()
            
            # Use median time and max memory from measurement runs
            median_time = np.median([t for t in times if not np.isinf(t)])
            max_memory = max([m for m in memories if not np.isinf(m)]) if memories else 0
            
            if np.isinf(median_time):
                print(f"Warning: All runs failed for size {size}")
                median_time = 0
                max_memory = 0
            
            result = BenchmarkResult(
                input_size=size,
                execution_time=median_time,
                peak_memory_mb=max_memory,
                output_size=output_size,
                metadata={
                    'function_name': function_name,
                    'all_times': times,
                    'all_memories': memories,
                    'measurement_runs': self.measurement_runs
                }
            )
            
            results.append(result)
        
        return results
    
    def analyze_complexity(self, results: List[BenchmarkResult], algorithm_name: str) -> ComplexityAnalysis:
        """Analyze complexity from benchmark results."""
        if len(results) < 2:
            return ComplexityAnalysis(
                algorithm_name=algorithm_name,
                input_sizes=[r.input_size for r in results],
                execution_times=[r.execution_time for r in results],
                memory_usage=[r.peak_memory_mb for r in results],
                estimated_complexity="insufficient_data",
                growth_rate=0.0,
                r_squared=0.0,
                performance_grade="unknown",
                notes="Insufficient data points for complexity analysis"
            )
        
        sizes = np.array([r.input_size for r in results])
        times = np.array([r.execution_time for r in results])
        
        # Filter out failed runs
        valid_mask = np.isfinite(times) & (times > 0)
        sizes = sizes[valid_mask]
        times = times[valid_mask]
        
        if len(sizes) < 2:
            return ComplexityAnalysis(
                algorithm_name=algorithm_name,
                input_sizes=sizes.tolist(),
                execution_times=times.tolist(),
                memory_usage=[r.peak_memory_mb for r in results],
                estimated_complexity="measurement_failed",
                growth_rate=0.0,
                r_squared=0.0,
                performance_grade="failed",
                notes="Too many failed measurements"
            )
        
        # Calculate growth rates between consecutive measurements
        growth_rates = []
        for i in range(1, len(sizes)):
            if times[i-1] > 0 and sizes[i-1] > 0:
                size_ratio = sizes[i] / sizes[i-1]
                time_ratio = times[i] / times[i-1]
                
                if size_ratio > 1 and time_ratio > 0:
                    growth_rate = np.log(time_ratio) / np.log(size_ratio)
                    growth_rates.append(growth_rate)
        
        avg_growth_rate = np.mean(growth_rates) if growth_rates else 0.0
        
        # Estimate complexity class
        estimated_complexity = self._classify_complexity(avg_growth_rate)
        
        # Calculate R-squared for complexity fit
        r_squared = self._calculate_complexity_fit(sizes, times, avg_growth_rate)
        
        # Grade performance
        performance_grade = self._grade_performance(avg_growth_rate, r_squared)
        
        # Generate notes
        notes = self._generate_analysis_notes(avg_growth_rate, r_squared, growth_rates)
        
        return ComplexityAnalysis(
            algorithm_name=algorithm_name,
            input_sizes=sizes.tolist(),
            execution_times=times.tolist(),
            memory_usage=[r.peak_memory_mb for r in results if r.input_size in sizes],
            estimated_complexity=estimated_complexity,
            growth_rate=avg_growth_rate,
            r_squared=r_squared,
            performance_grade=performance_grade,
            notes=notes
        )
    
    def _classify_complexity(self, growth_rate: float) -> str:
        """Classify algorithmic complexity based on growth rate."""
        if growth_rate < 0.5:
            return "O(log n) or better"
        elif growth_rate < 1.2:
            return "O(n)"
        elif growth_rate < 1.7:
            return "O(n log n)"
        elif growth_rate < 2.2:
            return "O(n^2)"
        elif growth_rate < 2.7:
            return "O(n^2.5)"
        elif growth_rate < 3.2:
            return "O(n^3)"
        else:
            return f"O(n^{growth_rate:.1f}) - potentially problematic"
    
    def _calculate_complexity_fit(self, sizes: np.ndarray, times: np.ndarray, growth_rate: float) -> float:
        """Calculate how well the data fits the estimated complexity."""
        if len(sizes) < 3:
            return 0.0
        
        try:
            # Predict times based on power law: t = a * n^growth_rate
            log_sizes = np.log(sizes)
            log_times = np.log(times)
            
            # Linear regression in log space
            coeffs = np.polyfit(log_sizes, log_times, 1)
            predicted_log_times = np.polyval(coeffs, log_sizes)
            
            # Calculate R-squared
            ss_res = np.sum((log_times - predicted_log_times) ** 2)
            ss_tot = np.sum((log_times - np.mean(log_times)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return max(0.0, r_squared)  # Ensure non-negative
            
        except Exception as e:
            print(f"Error calculating R-squared: {e}")
            return 0.0
    
    def _grade_performance(self, growth_rate: float, r_squared: float) -> str:
        """Grade algorithm performance."""
        # Penalize high complexity
        complexity_score = max(0, 100 - (growth_rate - 1) * 50)
        
        # Reward good fit (predictable performance)
        fit_score = r_squared * 100
        
        # Combined score
        overall_score = (complexity_score + fit_score) / 2
        
        if overall_score >= 85:
            return "excellent"
        elif overall_score >= 70:
            return "good"
        elif overall_score >= 50:
            return "acceptable"
        elif overall_score >= 30:
            return "poor"
        else:
            return "problematic"
    
    def _generate_analysis_notes(self, growth_rate: float, r_squared: float, growth_rates: List[float]) -> str:
        """Generate descriptive notes about the complexity analysis."""
        notes = []
        
        # Complexity assessment
        if growth_rate <= 1.0:
            notes.append("Linear or better scaling - excellent")
        elif growth_rate <= 1.5:
            notes.append("Near-linear scaling - good")
        elif growth_rate <= 2.0:
            notes.append("Quadratic-like scaling - may become problematic with large data")
        else:
            notes.append("High polynomial scaling - concerning for large datasets")
        
        # Consistency assessment  
        if r_squared >= 0.9:
            notes.append("Very consistent performance")
        elif r_squared >= 0.7:
            notes.append("Reasonably consistent performance")
        else:
            notes.append("Inconsistent performance - investigate system factors")
        
        # Variability assessment
        if growth_rates:
            variability = np.std(growth_rates)
            if variability < 0.2:
                notes.append("Low performance variability")
            elif variability < 0.5:
                notes.append("Moderate performance variability")
            else:
                notes.append("High performance variability - results may be unreliable")
        
        return "; ".join(notes)
    
    def save_benchmark_results(self, results: List[BenchmarkResult], 
                              filepath: Path, algorithm_name: str = "unknown"):
        """Save benchmark results to JSON file."""
        data = {
            'algorithm_name': algorithm_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'benchmark_settings': {
                'warmup_runs': self.warmup_runs,
                'measurement_runs': self.measurement_runs
            },
            'results': [asdict(r) for r in results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_benchmark_results(self, filepath: Path) -> List[BenchmarkResult]:
        """Load benchmark results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return [BenchmarkResult(**r) for r in data['results']]
    
    def plot_complexity_analysis(self, analysis: ComplexityAnalysis, 
                                save_path: Optional[Path] = None) -> None:
        """Create visualization of complexity analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        sizes = np.array(analysis.input_sizes)
        times = np.array(analysis.execution_times)
        
        # Plot 1: Execution time vs input size
        ax1.loglog(sizes, times, 'bo-', label='Measured')
        
        # Fit line based on estimated complexity
        if len(sizes) >= 2:
            fit_sizes = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 100)
            # Use power law fit: t = a * n^growth_rate
            a = times[0] / (sizes[0] ** analysis.growth_rate)
            fit_times = a * (fit_sizes ** analysis.growth_rate)
            ax1.loglog(fit_sizes, fit_times, 'r--', alpha=0.7, 
                      label=f'{analysis.estimated_complexity} (R²={analysis.r_squared:.3f})')
        
        ax1.set_xlabel('Input Size')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title(f'{analysis.algorithm_name} - Time Complexity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory usage vs input size
        if analysis.memory_usage:
            memory = np.array(analysis.memory_usage)
            ax2.loglog(sizes, memory, 'go-', label='Peak Memory')
            ax2.set_xlabel('Input Size')
            ax2.set_ylabel('Peak Memory (MB)')
            ax2.set_title(f'{analysis.algorithm_name} - Memory Usage')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add summary text
        summary = f"""
Algorithm: {analysis.algorithm_name}
Complexity: {analysis.estimated_complexity}
Growth Rate: {analysis.growth_rate:.2f}
Performance: {analysis.performance_grade}
Notes: {analysis.notes[:100]}...
        """.strip()
        
        plt.figtext(0.02, 0.02, summary, fontsize=8, verticalalignment='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Complexity analysis plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def create_input_generators():
    """Create standard input generators for different algorithm types."""
    
    def roi_data_generator(n_points: int) -> Dict[str, Any]:
        """Generate ROI data for testing."""
        np.random.seed(42)  # Fixed seed for consistency
        return {
            'coords': np.random.uniform(0, 100, (n_points, 2)),
            'ion_counts': {
                f'protein_{i}': np.random.poisson(5, n_points).astype(float)
                for i in range(9)
            },
            'protein_names': [f'protein_{i}' for i in range(9)]
        }
    
    def matrix_generator(n_points: int) -> np.ndarray:
        """Generate matrix data for testing."""
        np.random.seed(42)
        return np.random.randn(n_points, 10)
    
    def graph_data_generator(n_points: int) -> Dict[str, Any]:
        """Generate graph data for spatial algorithms."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (n_points, 2))
        features = np.random.randn(n_points, 5)
        return {'coords': coords, 'features': features}
    
    return {
        'roi_data': roi_data_generator,
        'matrix': matrix_generator,
        'graph_data': graph_data_generator
    }


def run_standard_benchmarks(output_dir: Path) -> Dict[str, ComplexityAnalysis]:
    """Run standard performance benchmarks for common algorithms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    benchmarker = ComplexityBenchmarker()
    input_generators = create_input_generators()
    
    # Standard test sizes - start small for CI compatibility
    small_sizes = [100, 500, 1000, 2000]
    medium_sizes = [1000, 2500, 5000, 7500, 10000]
    
    analyses = {}
    
    # Benchmark numpy operations as baseline
    def numpy_sum(data):
        return np.sum(data, axis=1)
    
    print("Benchmarking numpy operations (baseline)...")
    numpy_results = benchmarker.benchmark_function(
        func=numpy_sum,
        input_generator=input_generators['matrix'],
        sizes=medium_sizes,
        function_name="numpy_sum"
    )
    
    numpy_analysis = benchmarker.analyze_complexity(numpy_results, "numpy_sum")
    analyses['numpy_baseline'] = numpy_analysis
    
    benchmarker.save_benchmark_results(
        numpy_results, output_dir / "numpy_baseline.json", "numpy_sum"
    )
    
    benchmarker.plot_complexity_analysis(
        numpy_analysis, output_dir / "numpy_baseline_plot.png"
    )
    
    # Add more benchmarks here as needed for specific algorithms
    print("Standard benchmarks completed!")
    
    return analyses


def validate_performance_against_baseline(current_results: List[BenchmarkResult],
                                        baseline_file: Path,
                                        tolerance: float = 2.0) -> Dict[str, Any]:
    """Validate current performance against saved baseline."""
    if not baseline_file.exists():
        return {
            'status': 'no_baseline',
            'message': f'No baseline found at {baseline_file}',
            'recommendation': 'Create baseline with current results'
        }
    
    # Load baseline
    benchmarker = ComplexityBenchmarker()
    baseline_results = benchmarker.load_benchmark_results(baseline_file)
    
    # Compare results at matching input sizes
    validation_results = {
        'status': 'pass',
        'regressions': [],
        'improvements': [],
        'tolerance': tolerance
    }
    
    baseline_dict = {r.input_size: r for r in baseline_results}
    
    for current in current_results:
        if current.input_size in baseline_dict:
            baseline = baseline_dict[current.input_size]
            
            # Check timing regression
            if baseline.execution_time > 0:
                time_ratio = current.execution_time / baseline.execution_time
                if time_ratio > tolerance:
                    validation_results['regressions'].append({
                        'type': 'timing',
                        'size': current.input_size,
                        'baseline_time': baseline.execution_time,
                        'current_time': current.execution_time,
                        'ratio': time_ratio
                    })
                    validation_results['status'] = 'regression'
                elif time_ratio < 0.8:  # Significant improvement
                    validation_results['improvements'].append({
                        'type': 'timing',
                        'size': current.input_size,
                        'improvement_ratio': 1 / time_ratio
                    })
            
            # Check memory regression
            if baseline.peak_memory_mb > 0:
                memory_ratio = current.peak_memory_mb / baseline.peak_memory_mb
                if memory_ratio > tolerance:
                    validation_results['regressions'].append({
                        'type': 'memory',
                        'size': current.input_size,
                        'baseline_memory': baseline.peak_memory_mb,
                        'current_memory': current.peak_memory_mb,
                        'ratio': memory_ratio
                    })
                    validation_results['status'] = 'regression'
    
    return validation_results


if __name__ == "__main__":
    # Example usage
    print("Running complexity benchmark example...")
    
    # Create output directory
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    # Run standard benchmarks
    analyses = run_standard_benchmarks(output_dir)
    
    # Print results
    for name, analysis in analyses.items():
        print(f"\n{name.upper()} Analysis:")
        print(f"  Complexity: {analysis.estimated_complexity}")
        print(f"  Growth Rate: {analysis.growth_rate:.2f}")
        print(f"  Performance: {analysis.performance_grade}")
        print(f"  R²: {analysis.r_squared:.3f}")
        print(f"  Notes: {analysis.notes}")
    
    print(f"\nBenchmark results saved to {output_dir}")
    print("Complexity benchmarking complete!")