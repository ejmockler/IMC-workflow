"""
Visualization utilities for IMC analysis.

Lightweight, stateless plotting functions for creating common IMC visualizations
from standardized analysis outputs (HDF5, Parquet, JSON).
"""

from .plotting import (
    plot_roi_overview,
    plot_protein_expression,
    plot_cluster_map,
    plot_scale_comparison
)

from .loaders import (
    load_roi_results,
    load_multiscale_results,
    load_batch_results
)

__all__ = [
    'plot_roi_overview',
    'plot_protein_expression',
    'plot_cluster_map',
    'plot_scale_comparison',
    'load_roi_results',
    'load_multiscale_results',
    'load_batch_results'
]