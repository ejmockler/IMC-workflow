"""
Benchmarking Framework for Spatial Analysis Methods
Provides comprehensive performance comparison across different approaches
"""

import numpy as np
import time
import psutil
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class BenchmarkResult:
    """Single benchmark result with metrics"""
    method_name: str
    score: float
    time_seconds: float
    memory_mb: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def method_scores(self) -> Dict[str, Dict]:
        """Access method scores from additional metrics"""
        return self.additional_metrics.get('method_scores', {})
    
    @property
    def best_method(self) -> str:
        """Get best method name"""
        return self.additional_metrics.get('best_method', self.method_name)
    
    @property
    def best_score(self) -> float:
        """Get best score"""
        return self.additional_metrics.get('best_score', self.score)
    
    @property
    def rankings(self) -> List[Tuple[str, float]]:
        """Get method rankings"""
        return self.additional_metrics.get('rankings', [])


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report"""
    timestamp: str
    data_description: Dict[str, Any]
    comparisons: Dict[str, BenchmarkResult]
    scalability_metrics: Dict[str, Dict]
    configuration: Dict[str, Any]
    summary_statistics: Dict[str, float]


class MethodComparator(ABC):
    """Abstract base for method comparison"""
    
    @abstractmethod
    def compare(self, data: np.ndarray, **kwargs) -> BenchmarkResult:
        """Compare methods on given data and return aggregated result"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return comparator name"""
        pass


class ClusteringComparator(MethodComparator):
    """Compare clustering algorithms"""
    
    def __init__(self, algorithms: Optional[List[str]] = None):
        if algorithms is None:
            self.algorithms = ['kmeans', 'minibatch']
        else:
            self.algorithms = algorithms
    
    def name(self) -> str:
        return "clustering"
    
    def compare(self, data: np.ndarray,
                coords: np.ndarray,
                n_clusters: Optional[int] = None,
                ground_truth: Optional[np.ndarray] = None) -> BenchmarkResult:
        """Compare clustering algorithms and return aggregated result"""
        from src.analysis.clustering import ClustererFactory
        from src.analysis.validation import ValidationSuite, ValidationResult
        from sklearn.metrics import adjusted_rand_score
        
        # Auto-determine n_clusters if not provided
        if n_clusters is None:
            n_clusters = min(30, max(5, len(data) // 100))
        
        method_scores = {}
        validator = ValidationSuite()
        
        for algo_name in self.algorithms:
            try:
                # Create clusterer
                clusterer = ClustererFactory.create(algo_name)
                
                # Measure performance
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                start_time = time.time()
                
                # Cluster
                if algo_name in ['phenograph', 'leiden']:
                    cluster_result = clusterer.fit_predict(data)
                else:
                    cluster_result = clusterer.fit_predict(data, n_clusters=n_clusters)
                
                elapsed = time.time() - start_time
                mem_after = process.memory_info().rss / 1024 / 1024
                memory_used = mem_after - mem_before
                
                # Validate quality
                val_results = validator.validate_all(
                    data, cluster_result.labels, coords=coords
                )
                quality_score = validator.get_summary_score(val_results)
                
                # Compile metrics for this algorithm
                metrics = {
                    'time': elapsed,
                    'memory': memory_used,
                    'quality': quality_score,
                    'n_clusters': cluster_result.n_clusters,
                    'silhouette': val_results.get('silhouette', ValidationResult(
                        'silhouette', 0, {}, '')).score if 'silhouette' in val_results else 0,
                    'consistency': val_results.get('consistency', ValidationResult(
                        'consistency', 0, {}, '')).score if 'consistency' in val_results else 0,
                    'spatial_coherence': val_results.get('spatial_coherence', ValidationResult(
                        'spatial_coherence', 0, {}, '')).score if 'spatial_coherence' in val_results else 0
                }
                
                # Add ground truth comparison if available
                if ground_truth is not None:
                    ari = adjusted_rand_score(ground_truth, cluster_result.labels)
                    metrics['ari_vs_truth'] = ari
                
                method_scores[algo_name] = metrics
                
            except ImportError as e:
                # Algorithm not installed
                method_scores[algo_name] = {
                    'time': 0.0,
                    'memory': 0.0,
                    'quality': 0.0,
                    'error': f"Not installed: {str(e)}"
                }
            except Exception as e:
                # Other errors
                method_scores[algo_name] = {
                    'time': 0.0,
                    'memory': 0.0,
                    'quality': 0.0,
                    'error': str(e)
                }
        
        # Find best method
        valid_methods = {k: v for k, v in method_scores.items() if 'error' not in v}
        if valid_methods:
            best_method = max(valid_methods.keys(), key=lambda k: valid_methods[k]['quality'])
            best_score = valid_methods[best_method]['quality']
        else:
            best_method = list(method_scores.keys())[0] if method_scores else 'none'
            best_score = 0.0
        
        # Create rankings
        rankings = sorted(
            [(k, v['quality']) for k, v in valid_methods.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return BenchmarkResult(
            method_name='clustering_comparison',
            score=best_score,
            time_seconds=sum(v.get('time', 0) for v in method_scores.values()),
            memory_mb=max(v.get('memory', 0) for v in method_scores.values()),
            additional_metrics={
                'method_scores': method_scores,
                'best_method': best_method,
                'best_score': best_score,
                'rankings': rankings
            }
        )


class KernelComparator(MethodComparator):
    """Compare spatial kernel methods"""
    
    def __init__(self, kernels: Optional[List[str]] = None):
        if kernels is None:
            self.kernels = ['gaussian', 'adaptive']
        else:
            self.kernels = kernels
    
    def name(self) -> str:
        return "kernels"
    
    def compare(self, data: np.ndarray, 
                coords: np.ndarray,
                lambda_param: float = 0.5,
                n_clusters: Optional[int] = None) -> BenchmarkResult:
        """Compare spatial kernels and return aggregated result"""
        from src.analysis.kernels import KernelFactory, compute_augmented_features
        from sklearn.cluster import KMeans
        from src.analysis.validation import SilhouetteValidator, SpatialCoherenceValidator
        
        method_scores = {}
        validator = SilhouetteValidator()
        spatial_val = SpatialCoherenceValidator()
        
        if n_clusters is None:
            n_clusters = min(30, max(5, len(data) // 100))
        
        for kernel_name in self.kernels:
            try:
                # Create kernel
                if kernel_name == 'gaussian':
                    kernel = KernelFactory.create_kernel('gaussian', {'sigma': 30})
                elif kernel_name == 'adaptive':
                    kernel = KernelFactory.create_kernel('adaptive', {})
                else:
                    kernel = KernelFactory.create_kernel('laplacian', {'sigma': 40})
                
                # Measure augmentation performance
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024
                
                start_time = time.time()
                augmented = compute_augmented_features(coords, data, kernel, lambda_param)
                augmentation_time = time.time() - start_time
                
                # Cluster augmented features
                kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
                labels = kmeans.fit_predict(augmented)
                
                # Measure clustering quality
                cluster_time = time.time() - start_time - augmentation_time
                mem_after = process.memory_info().rss / 1024 / 1024
                
                val_result = validator.validate(augmented, labels, coords=coords)
                spatial_result = spatial_val.validate(data, labels, coords=coords)
                
                # Combined score
                combined_score = val_result.score * 0.7 + spatial_result.score * 0.3
                
                method_scores[kernel_name] = {
                    'time': augmentation_time + cluster_time,
                    'memory': mem_after - mem_before,
                    'quality': combined_score,
                    'augmentation_time': augmentation_time,
                    'clustering_quality': val_result.score,
                    'spatial_coherence': spatial_result.score,
                    'lambda': lambda_param
                }
                
            except Exception as e:
                method_scores[kernel_name] = {
                    'time': 0.0,
                    'memory': 0.0,
                    'quality': 0.0,
                    'error': str(e)
                }
        
        # Find best kernel
        valid_methods = {k: v for k, v in method_scores.items() if 'error' not in v}
        if valid_methods:
            best_method = max(valid_methods.keys(), key=lambda k: valid_methods[k]['quality'])
            best_score = valid_methods[best_method]['quality']
        else:
            best_method = list(method_scores.keys())[0] if method_scores else 'none'
            best_score = 0.0
        
        # Create rankings
        rankings = sorted(
            [(k, v['quality']) for k, v in valid_methods.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return BenchmarkResult(
            method_name='kernel_comparison',
            score=best_score,
            time_seconds=sum(v.get('time', 0) for v in method_scores.values()),
            memory_mb=max(v.get('memory', 0) for v in method_scores.values()),
            additional_metrics={
                'method_scores': method_scores,
                'best_method': best_method,
                'best_score': best_score,
                'rankings': rankings
            }
        )


class IntegratedComparator(MethodComparator):
    """Compare integrated kernel+clustering approaches"""
    
    def __init__(self, 
                 clustering_algorithms: Optional[List[str]] = None,
                 kernel_types: Optional[List[str]] = None,
                 lambda_values: Optional[List[float]] = None):
        self.clustering_algorithms = clustering_algorithms or ['kmeans', 'minibatch']
        self.kernel_types = kernel_types or ['gaussian', 'adaptive']
        self.lambda_values = lambda_values or [0.3, 0.5, 0.7]
    
    def name(self) -> str:
        return "integrated"
    
    def compare(self, data: np.ndarray,
                coords: np.ndarray,
                n_clusters: Optional[int] = None) -> BenchmarkResult:
        """Compare kernel+clustering combinations and return aggregated result"""
        from src.analysis.kernels import KernelFactory, compute_augmented_features
        from src.analysis.clustering import ClustererFactory
        from src.analysis.validation import ValidationSuite
        
        if n_clusters is None:
            n_clusters = min(30, max(5, len(data) // 100))
        
        method_scores = {}
        validator = ValidationSuite()
        
        for algo_name in self.clustering_algorithms:
            for kernel_name in self.kernel_types:
                for lambda_val in self.lambda_values:
                    combo_name = f"{algo_name}+{kernel_name}(λ={lambda_val})"
                    
                    try:
                        # Create kernel
                        if kernel_name == 'gaussian':
                            kernel = KernelFactory.create_kernel('gaussian', {'sigma': 30})
                        elif kernel_name == 'adaptive':
                            kernel = KernelFactory.create_kernel('adaptive', {})
                        else:
                            kernel = KernelFactory.create_kernel('laplacian', {'sigma': 40})
                        
                        # Augment features
                        augmented = compute_augmented_features(coords, data, kernel, lambda_val)
                        
                        # Cluster
                        process = psutil.Process()
                        mem_before = process.memory_info().rss / 1024 / 1024
                        start_time = time.time()
                        
                        clusterer = ClustererFactory.create(algo_name)
                        if algo_name in ['phenograph', 'leiden']:
                            cluster_result = clusterer.fit_predict(augmented)
                        else:
                            cluster_result = clusterer.fit_predict(augmented, n_clusters=n_clusters)
                        
                        elapsed = time.time() - start_time
                        mem_after = process.memory_info().rss / 1024 / 1024
                        
                        # Validate
                        val_results = validator.validate_all(
                            augmented, cluster_result.labels, coords=coords
                        )
                        quality_score = validator.get_summary_score(val_results)
                        
                        method_scores[combo_name] = {
                            'time': elapsed,
                            'memory': mem_after - mem_before,
                            'quality': quality_score,
                            'algorithm': algo_name,
                            'kernel': kernel_name,
                            'lambda': lambda_val,
                            'n_clusters': cluster_result.n_clusters
                        }
                        
                    except Exception as e:
                        method_scores[combo_name] = {
                            'time': 0.0,
                            'memory': 0.0,
                            'quality': 0.0,
                            'error': str(e)
                        }
        
        # Find best combination
        valid_methods = {k: v for k, v in method_scores.items() if 'error' not in v}
        if valid_methods:
            best_method = max(valid_methods.keys(), key=lambda k: valid_methods[k]['quality'])
            best_score = valid_methods[best_method]['quality']
        else:
            best_method = list(method_scores.keys())[0] if method_scores else 'none'
            best_score = 0.0
        
        # Create rankings
        rankings = sorted(
            [(k, v['quality']) for k, v in valid_methods.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return BenchmarkResult(
            method_name='integrated_comparison',
            score=best_score,
            time_seconds=sum(v.get('time', 0) for v in method_scores.values()),
            memory_mb=max(v.get('memory', 0) for v in method_scores.values()),
            additional_metrics={
                'method_scores': method_scores,
                'best_method': best_method,
                'best_score': best_score,
                'rankings': rankings
            }
        )


class BenchmarkSuite:
    """Orchestrates comprehensive benchmarking"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            with open(config_path) as f:
                config = json.load(f)
                self.config = config.get('benchmarking', {})
        else:
            self.config = {}
        
        # Setup default comparators
        self.clustering_algorithms = self.config.get('clustering_algorithms', 
                                                     ['kmeans', 'minibatch'])
        self.kernel_types = self.config.get('kernel_types', 
                                           ['gaussian', 'adaptive'])
        self.subset_sizes = self.config.get('subset_sizes', 
                                           [1000, 5000, 10000])
    
    def run_full_benchmark(self, data: np.ndarray,
                          coords: np.ndarray,
                          subset_sizes: Optional[List[int]] = None) -> BenchmarkReport:
        """Run comprehensive benchmark suite"""
        import datetime
        
        if subset_sizes is None:
            subset_sizes = self.subset_sizes
        
        # Data description
        data_desc = {
            'n_samples': len(data),
            'n_features': data.shape[1],
            'memory_mb': data.nbytes / 1024 / 1024
        }
        
        comparisons = {}
        
        # 1. Compare clustering algorithms
        print("Running clustering comparison...")
        clustering_comp = ClusteringComparator(self.clustering_algorithms)
        comparisons['clustering'] = clustering_comp.compare(data, coords)
        
        # 2. Compare spatial kernels
        print("Running kernel comparison...")
        kernel_comp = KernelComparator(self.kernel_types)
        comparisons['kernels'] = kernel_comp.compare(data, coords)
        
        # 3. Test integrated approaches (on smaller subset)
        if len(data) > 5000:
            subset_idx = np.random.choice(len(data), 5000, replace=False)
            subset_data = data[subset_idx]
            subset_coords = coords[subset_idx]
        else:
            subset_data = data
            subset_coords = coords
        
        print("Running integrated comparison...")
        integrated_comp = IntegratedComparator(
            clustering_algorithms=self.clustering_algorithms[:2],  # Limit for speed
            kernel_types=self.kernel_types[:2],
            lambda_values=[0.3, 0.7]
        )
        comparisons['integrated'] = integrated_comp.compare(subset_data, subset_coords)
        
        # 4. Scalability analysis
        scalability = self._analyze_scalability(data, coords, subset_sizes)
        
        # 5. Summary statistics
        summary_stats = self._calculate_summary_statistics(comparisons)
        
        return BenchmarkReport(
            timestamp=datetime.datetime.now().isoformat(),
            data_description=data_desc,
            comparisons=comparisons,
            scalability_metrics=scalability,
            configuration={
                'clustering_algorithms': self.clustering_algorithms,
                'kernel_types': self.kernel_types,
                'subset_sizes': subset_sizes
            },
            summary_statistics=summary_stats
        )
    
    def _analyze_scalability(self, data: np.ndarray,
                            coords: np.ndarray,
                            subset_sizes: List[int]) -> Dict[str, Dict]:
        """Analyze algorithm scalability"""
        from sklearn.cluster import KMeans
        
        scalability_results = {}
        
        for algo_name in ['kmeans', 'minibatch']:
            times = []
            sizes = []
            
            for size in subset_sizes:
                if size > len(data):
                    continue
                
                idx = np.random.choice(len(data), size, replace=False)
                subset = data[idx]
                
                if algo_name == 'kmeans':
                    model = KMeans(n_clusters=10, n_init=1, random_state=42)
                else:
                    from sklearn.cluster import MiniBatchKMeans
                    model = MiniBatchKMeans(n_clusters=10, n_init=1, random_state=42)
                
                start = time.time()
                model.fit(subset)
                elapsed = time.time() - start
                
                times.append(elapsed)
                sizes.append(size)
            
            # Fit scaling curve (polynomial)
            if len(sizes) > 1:
                coeffs = np.polyfit(np.log(sizes), np.log(times), 1)
                scaling_factor = coeffs[0]  # O(n^scaling_factor)
            else:
                scaling_factor = None
            
            scalability_results[algo_name] = {
                'sizes': sizes,
                'times': times,
                'scaling_factor': scaling_factor
            }
        
        return scalability_results
    
    def _calculate_summary_statistics(self, comparisons: Dict) -> Dict[str, float]:
        """Calculate summary statistics across all comparisons"""
        stats = {}
        
        # Best overall method
        all_scores = []
        for comp_name, result in comparisons.items():
            if hasattr(result, 'best_score'):
                all_scores.append((f"{comp_name}_best", result.best_score))
        
        if all_scores:
            best_overall = max(all_scores, key=lambda x: x[1])
            stats['best_overall_method'] = best_overall[0]
            stats['best_overall_score'] = best_overall[1]
        
        # Average quality across methods
        quality_scores = []
        for comp_name, result in comparisons.items():
            if hasattr(result, 'method_scores'):
                for method, metrics in result.method_scores.items():
                    if 'quality' in metrics and 'error' not in metrics:
                        quality_scores.append(metrics['quality'])
        
        if quality_scores:
            stats['mean_quality'] = np.mean(quality_scores)
            stats['std_quality'] = np.std(quality_scores)
            stats['max_quality'] = np.max(quality_scores)
        
        return stats


class ReportGenerator:
    """Generate benchmark reports in various formats"""
    
    def generate_html(self, report: BenchmarkReport, output_path: str):
        """Generate HTML report with visualizations"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report - {report.timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
        .best {{ background-color: #d4edda; font-weight: bold; }}
        .metric {{ font-family: monospace; }}
    </style>
</head>
<body>
    <h1>IMC Analysis Benchmark Report</h1>
    <p>Generated: {report.timestamp}</p>
    
    <h2>Data Description</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Samples</td><td>{report.data_description['n_samples']:,}</td></tr>
        <tr><td>Features</td><td>{report.data_description['n_features']}</td></tr>
        <tr><td>Memory (MB)</td><td>{report.data_description['memory_mb']:.2f}</td></tr>
    </table>
"""
        
        # Add comparison results
        for comp_name, result in report.comparisons.items():
            html_content += f"""
    <h2>{comp_name.title()} Comparison</h2>
    <p>Best Method: <span class="best">{result.best_method}</span> (Score: {result.best_score:.3f})</p>
    <table>
        <tr><th>Method</th><th>Quality</th><th>Time (s)</th><th>Memory (MB)</th></tr>
"""
            for method, metrics in result.method_scores.items():
                is_best = method == result.best_method
                row_class = 'class="best"' if is_best else ''
                html_content += f"""
        <tr {row_class}>
            <td>{method}</td>
            <td class="metric">{metrics.get('quality', 0):.3f}</td>
            <td class="metric">{metrics.get('time', 0):.2f}</td>
            <td class="metric">{metrics.get('memory', 0):.1f}</td>
        </tr>
"""
            html_content += "    </table>\n"
        
        # Add summary
        if report.summary_statistics:
            html_content += """
    <h2>Summary Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
"""
            for key, value in report.summary_statistics.items():
                if isinstance(value, float):
                    html_content += f'        <tr><td>{key}</td><td class="metric">{value:.3f}</td></tr>\n'
                else:
                    html_content += f'        <tr><td>{key}</td><td>{value}</td></tr>\n'
            html_content += "    </table>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def generate_markdown(self, report: BenchmarkReport, output_path: str):
        """Generate Markdown report"""
        md_content = f"""# IMC Analysis Benchmark Report

Generated: {report.timestamp}

## Data Description

| Metric | Value |
|--------|-------|
| Samples | {report.data_description['n_samples']:,} |
| Features | {report.data_description['n_features']} |
| Memory (MB) | {report.data_description['memory_mb']:.2f} |

"""
        
        # Add comparison results
        for comp_name, result in report.comparisons.items():
            md_content += f"""## {comp_name.title()} Comparison

**Best Method:** {result.best_method} (Score: {result.best_score:.3f})

| Method | Quality | Time (s) | Memory (MB) |
|--------|---------|----------|-------------|
"""
            for method, metrics in result.method_scores.items():
                quality = metrics.get('quality', 0)
                time_s = metrics.get('time', 0)
                memory = metrics.get('memory', 0)
                is_best = method == result.best_method
                prefix = "**" if is_best else ""
                suffix = "**" if is_best else ""
                md_content += f"| {prefix}{method}{suffix} | {quality:.3f} | {time_s:.2f} | {memory:.1f} |\n"
            md_content += "\n"
        
        # Add summary
        if report.summary_statistics:
            md_content += "## Summary Statistics\n\n"
            md_content += "| Metric | Value |\n"
            md_content += "|--------|-------|\n"
            for key, value in report.summary_statistics.items():
                if isinstance(value, float):
                    md_content += f"| {key} | {value:.3f} |\n"
                else:
                    md_content += f"| {key} | {value} |\n"
        
        with open(output_path, 'w') as f:
            f.write(md_content)
    
    def generate_latex(self, report: BenchmarkReport, output_path: str):
        """Generate LaTeX report for publication"""
        latex_content = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{graphicx}
\begin{document}

\title{IMC Analysis Benchmark Report}
\date{""" + report.timestamp + r"""}
\maketitle

\section{Data Description}

\begin{table}[h]
\centering
\begin{tabular}{lr}
\toprule
Metric & Value \\
\midrule
Samples & """ + f"{report.data_description['n_samples']:,}" + r""" \\
Features & """ + str(report.data_description['n_features']) + r""" \\
Memory (MB) & """ + f"{report.data_description['memory_mb']:.2f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

"""
        
        # Add comparison results
        for comp_name, result in report.comparisons.items():
            latex_content += rf"""
\section{{{comp_name.title()} Comparison}}

Best Method: \textbf{{{result.best_method}}} (Score: {result.best_score:.3f})

\begin{{table}}[h]
\centering
\begin{{tabular}}{{lrrr}}
\toprule
Method & Quality & Time (s) & Memory (MB) \\
\midrule
"""
            for method, metrics in result.method_scores.items():
                quality = metrics.get('quality', 0)
                time_s = metrics.get('time', 0)
                memory = metrics.get('memory', 0)
                is_best = method == result.best_method
                prefix = r"\textbf{" if is_best else ""
                suffix = "}" if is_best else ""
                latex_content += f"{prefix}{method}{suffix} & {quality:.3f} & {time_s:.2f} & {memory:.1f} \\\\\n"
            
            latex_content += r"""\bottomrule
\end{tabular}
\end{table}

"""
        
        latex_content += r"\end{document}"
        
        with open(output_path, 'w') as f:
            f.write(latex_content)