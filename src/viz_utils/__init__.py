"""
Visualization utilities for IMC analysis.

Lightweight, stateless plotting functions for creating common IMC visualizations
from standardized analysis outputs (HDF5, Parquet, JSON).

Plotting modules are lazy-loaded so that ``from src.viz_utils import VizConfig``
does not drag in matplotlib/seaborn/scipy/pandas via ``plotting`` and
``comprehensive_figures``. This matters because VizConfig is consumed by
zero-dependency config-loading paths (test fixtures, the cell-type-annotation
engine's schema validators) that should not pay the cost of the full viz
stack on import.
"""

from .viz_config import VizConfig, VizConfigValidationError

__all__ = [
    'plot_roi_overview',
    'plot_protein_expression',
    'plot_cluster_map',
    'plot_scale_comparison',
    'generate_all_figures',
    'VizConfig',
    'VizConfigValidationError',
]


def __getattr__(name):
    """Lazy-load plotting attributes on first access (PEP 562)."""
    if name in ('plot_roi_overview', 'plot_protein_expression',
                'plot_cluster_map', 'plot_scale_comparison'):
        from . import plotting
        return getattr(plotting, name)
    if name == 'generate_all_figures':
        from .comprehensive_figures import generate_all_figures
        return generate_all_figures
    raise AttributeError(f"module 'src.viz_utils' has no attribute {name!r}")