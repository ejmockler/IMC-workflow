"""Visualization module for IMC analysis."""

from .main import VisualizationPipeline
from .network import NetworkVisualizer
from .roi import ROIVisualizer
from .temporal import TemporalVisualizer
from .condition import ConditionVisualizer
from .replicate import ReplicateVisualizer
from .components import (
    plot_spatial_domains,
    plot_domain_signatures_heatmap,
    plot_spatial_contact_matrix,
    plot_domain_size_distribution,
    plot_top_domain_contacts,
    plot_aggregated_contact_matrix,
    plot_aggregated_domain_signatures
)

__all__ = [
    'VisualizationPipeline',
    'NetworkVisualizer',
    'ROIVisualizer',
    'TemporalVisualizer',
    'ConditionVisualizer',
    'ReplicateVisualizer',
    'plot_spatial_domains',
    'plot_domain_signatures_heatmap',
    'plot_spatial_contact_matrix',
    'plot_domain_size_distribution',
    'plot_top_domain_contacts',
    'plot_aggregated_contact_matrix',
    'plot_aggregated_domain_signatures'
]