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

from .comprehensive_figures import generate_all_figures
from .viz_config import VizConfig

__all__ = [
    'plot_roi_overview',
    'plot_protein_expression',
    'plot_cluster_map',
    'plot_scale_comparison',
    'generate_all_figures',
    'VizConfig',
]