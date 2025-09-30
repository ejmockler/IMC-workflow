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

# Import comprehensive figures if available
try:
    from .comprehensive_figures import ComprehensiveFigureGenerator
except ImportError:
    ComprehensiveFigureGenerator = None

__all__ = [
    'plot_roi_overview',
    'plot_protein_expression',
    'plot_cluster_map',
    'plot_scale_comparison'
]

if ComprehensiveFigureGenerator is not None:
    __all__.append('ComprehensiveFigureGenerator')