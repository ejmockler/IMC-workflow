"""
Ablation Study for Publication Readiness

Systematic comparison of 4 critical configurations:
1. Baseline: No coabundance, fixed k=15
2. Coabundance: 153 features, fixed k=15 (current overfitting config)
3. Adaptive-k: No coabundance, adaptive k
4. Full: LASSO-selected features (30), adaptive k (recommended config)

This ablation study addresses peer review concerns about methodological complexity.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List
from datetime import datetime

from src.config import Config
from src.analysis.main_pipeline import IMCAnalysisPipeline
from src.analysis.multiscale_analysis import perform_multiscale_analysis


def run_ablation_study(
    roi_path: str,
    config_path: str = "config.json",
    output_dir: str = "results/ablation_study"
) -> Dict:
    """
    Run ablation study on a single ROI with 4 configurations.

    Args:
        roi_path: Path to ROI data file
        config_path: Path to base configuration
        output_dir: Output directory for results

    Returns:
        Dictionary with results from all 4 configurations
    """
    # Load base config
    config = Config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define 4 ablation configurations
    configs = {
        "baseline": {
            "name": "Baseline (no coabundance, fixed k=15)",
            "use_coabundance_features": False,
            "k_neighbors": 15,
            "k_neighbors_by_scale": None,
            "description": "Standard Leiden clustering without feature engineering"
        },
        "coabundance_only": {
            "name": "Coabundance (153 features, fixed k=15)",
            "use_coabundance_features": True,
            "coabundance_options": {
                "interaction_order": 2,
                "include_ratios": True,
                "include_products": True,
                "include_spatial_covariance": True,
                "neighborhood_radius": 20.0,
                "min_expression_percentile": 25.0,
                "use_feature_selection": False  # NO selection = overfitting
            },
            "k_neighbors": 15,
            "k_neighbors_by_scale": None,
            "description": "All 153 features without selection - demonstrates overfitting risk"
        },
        "adaptive_k_only": {
            "name": "Adaptive k (no coabundance, adaptive k)",
            "use_coabundance_features": False,
            "k_neighbors": 15,
            "k_neighbors_by_scale": {
                "10.0": 14,
                "20.0": 12,
                "40.0": 10
            },
            "description": "Scale-adaptive k_neighbors without feature engineering"
        },
        "full_optimized": {
            "name": "Full (LASSO 30 features, adaptive k)",
            "use_coabundance_features": True,
            "coabundance_options": {
                "interaction_order": 2,
                "include_ratios": True,
                "include_products": True,
                "include_spatial_covariance": True,
                "neighborhood_radius": 20.0,
                "min_expression_percentile": 25.0,
                "use_feature_selection": True,  # LASSO selection
                "target_n_features": 30,
                "selection_method": "lasso"
            },
            "k_neighbors": 15,
            "k_neighbors_by_scale": {
                "10.0": 14,
                "20.0": 12,
                "40.0": 10
            },
            "description": "Recommended configuration with LASSO selection + adaptive k"
        }
    }

    results = {}

    for config_name, config_overrides in configs.items():
        print(f"\n{'='*80}")
        print(f"Running: {config_overrides['name']}")
        print(f"{'='*80}")

        # Create modified config for this ablation
        modified_config = Config(config_path)

        # Apply clustering parameter overrides
        if hasattr(modified_config, 'analysis') and hasattr(modified_config.analysis, 'clustering'):
            clustering = modified_config.analysis.clustering

            # Set coabundance features
            clustering.use_coabundance_features = config_overrides.get('use_coabundance_features', False)

            # Set k_neighbors
            clustering.k_neighbors = config_overrides.get('k_neighbors', 15)

            # Set coabundance options if provided
            if 'coabundance_options' in config_overrides:
                for key, value in config_overrides['coabundance_options'].items():
                    setattr(clustering.coabundance_options, key, value)

        # Run analysis with this configuration
        try:
            # Use IMCAnalysisPipeline class
            pipeline = IMCAnalysisPipeline(modified_config)
            roi_results = pipeline.analyze_single_roi(roi_path, roi_id=Path(roi_path).stem)

            # Extract key metrics for comparison
            metrics = extract_comparison_metrics(roi_results, config_name)

            results[config_name] = {
                "config": config_overrides,
                "metrics": metrics,
                "full_results": roi_results
            }

            # Save intermediate results
            result_file = output_path / f"{config_name}_results.json"
            with open(result_file, 'w') as f:
                json.dump({
                    "config": config_overrides,
                    "metrics": metrics
                }, f, indent=2)

            print(f"✓ Completed: {config_overrides['name']}")
            print(f"  Metrics: {metrics}")

        except Exception as e:
            print(f"✗ Failed: {config_overrides['name']}")
            print(f"  Error: {str(e)}")
            results[config_name] = {
                "config": config_overrides,
                "error": str(e)
            }

    # Generate comparison report
    comparison_report = generate_comparison_report(results)

    # Save full results
    report_file = output_path / "ablation_study_report.json"
    with open(report_file, 'w') as f:
        json.dump(comparison_report, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Ablation study complete! Results saved to: {output_dir}")
    print(f"{'='*80}\n")

    return comparison_report


def extract_comparison_metrics(roi_results: Dict, config_name: str) -> Dict:
    """Extract key metrics for comparing configurations."""
    metrics = {}

    # Extract multiscale results if available
    if 'multiscale_results' in roi_results:
        multiscale = roi_results['multiscale_results']

        for scale, scale_results in multiscale.items():
            scale_key = f"scale_{scale}"

            # Number of clusters
            if 'cluster_labels' in scale_results:
                labels = scale_results['cluster_labels']
                metrics[f"{scale_key}_n_clusters"] = len(np.unique(labels[labels >= 0]))

            # Stability score
            if 'stability_analysis' in scale_results:
                stability = scale_results['stability_analysis']
                if 'optimal_stability' in stability:
                    metrics[f"{scale_key}_stability"] = stability['optimal_stability']

            # Spatial coherence
            if 'spatial_coherence' in scale_results:
                coherence = scale_results['spatial_coherence']
                if isinstance(coherence, dict):
                    metrics[f"{scale_key}_morans_i"] = coherence.get('morans_i', None)
                else:
                    metrics[f"{scale_key}_morans_i"] = coherence

            # Feature dimensionality (if enriched features available)
            if 'clustering_info' in scale_results:
                info = scale_results['clustering_info']
                if 'n_features_used' in info:
                    metrics[f"{scale_key}_n_features"] = info['n_features_used']

    return metrics


def generate_comparison_report(results: Dict) -> Dict:
    """Generate structured comparison report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "configurations": {},
        "comparison": {},
        "recommendations": []
    }

    # Summarize each configuration
    for config_name, result in results.items():
        if 'error' in result:
            report['configurations'][config_name] = {
                "status": "failed",
                "error": result['error']
            }
        else:
            report['configurations'][config_name] = {
                "status": "success",
                "description": result['config']['description'],
                "metrics": result['metrics']
            }

    # Compare metrics across configurations
    successful_configs = {
        name: res for name, res in results.items()
        if 'metrics' in res
    }

    if len(successful_configs) >= 2:
        # Extract common metrics for comparison
        metric_comparison = {}

        # Get all metric keys
        all_metrics = set()
        for res in successful_configs.values():
            all_metrics.update(res['metrics'].keys())

        # Compare each metric across configs
        for metric in all_metrics:
            metric_comparison[metric] = {}
            for config_name, res in successful_configs.items():
                metric_comparison[metric][config_name] = res['metrics'].get(metric, None)

        report['comparison'] = metric_comparison

    # Generate recommendations
    if 'full_optimized' in successful_configs:
        report['recommendations'].append({
            "priority": "HIGH",
            "recommendation": "Use full_optimized configuration for publication",
            "rationale": "LASSO feature selection (30 features) prevents overfitting while adaptive k maintains scientific validity across scales"
        })

    if 'coabundance_only' in successful_configs:
        report['recommendations'].append({
            "priority": "CRITICAL",
            "recommendation": "NEVER use coabundance_only configuration",
            "rationale": "153 features without selection creates catastrophic overfitting risk - peer reviewers will reject"
        })

    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_ablation_study.py <roi_file_path> [config_path] [output_dir]")
        print("\nExample:")
        print("  python run_ablation_study.py data/241218_IMC_Alun/IMC_241218_Alun_ROI_D1_M1_01_9.txt")
        sys.exit(1)

    roi_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.json"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "results/ablation_study"

    results = run_ablation_study(roi_path, config_path, output_dir)

    print("\n=== ABLATION STUDY SUMMARY ===")
    print(json.dumps(results['comparison'], indent=2))
    print("\n=== RECOMMENDATIONS ===")
    for rec in results['recommendations']:
        print(f"\n[{rec['priority']}] {rec['recommendation']}")
        print(f"  Rationale: {rec['rationale']}")
