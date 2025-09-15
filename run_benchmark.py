#!/usr/bin/env python3
"""
Production benchmark runner for IMC analysis methods
Compares all configured algorithms and generates comprehensive reports
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import json
import argparse
from datetime import datetime

from src.config import Config
from src.utils.helpers import find_roi_files
from src.analysis.spatial import load_roi_data
from src.analysis.benchmark import BenchmarkSuite, ReportGenerator


def run_roi_benchmark(roi_path: Path, config_path: str = 'config.json'):
    """Run benchmarks on a single ROI"""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {roi_path.name}")
    print(f"{'='*60}")
    
    # Load data
    coords, values, protein_names = load_roi_data(roi_path, config_path)
    print(f"Loaded: {len(coords)} cells, {len(protein_names)} proteins")
    
    # Load benchmark configuration
    with open(config_path) as f:
        full_config = json.load(f)
    benchmark_config = full_config.get('benchmarking', {})
    
    # Subsample if needed
    max_cells = benchmark_config.get('max_cells_for_integrated', 3000)
    if len(coords) > max_cells:
        print(f"Subsampling to {max_cells} cells for benchmarking")
        indices = np.random.choice(len(coords), max_cells, replace=False)
        coords = coords[indices]
        values = values[indices]
    
    # Run benchmark suite
    suite = BenchmarkSuite(config_path)
    print("\nRunning benchmark suite...")
    print(f"  Algorithms: {suite.clustering_algorithms}")
    print(f"  Kernels: {suite.kernel_types}")
    print(f"  Subset sizes: {suite.subset_sizes}")
    
    report = suite.run_full_benchmark(values, coords)
    
    # Print summary
    print("\n" + "-"*40)
    print("RESULTS SUMMARY")
    print("-"*40)
    
    # Clustering results
    if 'clustering' in report.comparisons:
        clustering = report.comparisons['clustering']
        print(f"\nBest clustering: {clustering.best_method} (score: {clustering.best_score:.3f})")
        for algo, score in clustering.rankings[:3]:
            metrics = clustering.method_scores[algo]
            print(f"  • {algo}: {score:.3f} ({metrics['time']:.2f}s)")
    
    # Kernel results
    if 'kernels' in report.comparisons:
        kernels = report.comparisons['kernels']
        print(f"\nBest kernel: {kernels.best_method} (score: {kernels.best_score:.3f})")
    
    # Overall statistics
    if report.summary_statistics:
        stats = report.summary_statistics
        if 'mean_quality' in stats:
            print(f"\nOverall quality: {stats['mean_quality']:.3f} ± {stats['std_quality']:.3f}")
    
    return report


def generate_reports(report, roi_name: str, config_path: str = 'config.json'):
    """Generate benchmark reports in configured formats"""
    
    with open(config_path) as f:
        full_config = json.load(f)
    benchmark_config = full_config.get('benchmarking', {})
    
    # Create output directory
    report_dir = Path(benchmark_config.get('report_directory', 'benchmark_reports'))
    report_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"benchmark_{roi_name}_{timestamp}"
    
    generator = ReportGenerator()
    formats = benchmark_config.get('output_formats', ['markdown'])
    
    generated_files = []
    
    for format_type in formats:
        if format_type == 'markdown':
            output_path = report_dir / f"{base_name}.md"
            generator.generate_markdown(report, str(output_path))
            generated_files.append(output_path)
            
        elif format_type == 'html':
            output_path = report_dir / f"{base_name}.html"
            generator.generate_html(report, str(output_path))
            generated_files.append(output_path)
            
        elif format_type == 'latex':
            output_path = report_dir / f"{base_name}.tex"
            generator.generate_latex(report, str(output_path))
            generated_files.append(output_path)
    
    print(f"\nGenerated {len(generated_files)} report(s):")
    for path in generated_files:
        print(f"  • {path}")
    
    return generated_files


def main():
    """Main benchmark runner"""
    
    parser = argparse.ArgumentParser(description='Run IMC analysis benchmarks')
    parser.add_argument('--roi', type=str, help='Specific ROI file to benchmark')
    parser.add_argument('--all', action='store_true', help='Benchmark all ROIs')
    parser.add_argument('--config', type=str, default='config.json', 
                       help='Config file path')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip report generation')
    
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("#" + " "*20 + "IMC BENCHMARK RUNNER" + " "*19 + "#")
    print("#"*60)
    
    # Find ROIs
    config = Config(args.config)
    roi_files = find_roi_files(config.data_dir)
    
    if not roi_files:
        print("No ROI files found!")
        return 1
    
    # Determine which ROIs to process
    if args.roi:
        # Find specific ROI
        target_roi = None
        for roi in roi_files:
            if args.roi in str(roi):
                target_roi = roi
                break
        
        if not target_roi:
            print(f"ROI '{args.roi}' not found")
            return 1
        
        rois_to_process = [target_roi]
        
    elif args.all:
        rois_to_process = roi_files
        
    else:
        # Default: process first ROI
        rois_to_process = [roi_files[0]]
    
    print(f"\nProcessing {len(rois_to_process)} ROI(s)")
    
    # Run benchmarks
    all_reports = []
    
    for roi_path in rois_to_process:
        try:
            report = run_roi_benchmark(roi_path, args.config)
            all_reports.append((roi_path, report))
            
            # Generate reports if requested
            if not args.no_report:
                generate_reports(report, roi_path.stem, args.config)
                
        except Exception as e:
            print(f"\nError processing {roi_path.name}: {e}")
            continue
    
    # Final summary
    print("\n" + "#"*60)
    print("#" + " "*15 + "BENCHMARK COMPLETE" + " "*16 + "#")
    print("#"*60)
    print(f"\nProcessed {len(all_reports)} ROI(s) successfully")
    
    if all_reports and len(all_reports) > 1:
        # Compare across ROIs
        print("\nCross-ROI comparison:")
        best_methods = {}
        
        for roi_path, report in all_reports:
            if 'clustering' in report.comparisons:
                best = report.comparisons['clustering'].best_method
                best_methods[best] = best_methods.get(best, 0) + 1
        
        if best_methods:
            print("Most frequently best clustering:")
            for method, count in sorted(best_methods.items(), 
                                       key=lambda x: x[1], reverse=True):
                print(f"  • {method}: {count}/{len(all_reports)} ROIs")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())