"""Main visualization orchestrator for IMC analysis."""

from typing import List, Dict
import matplotlib.pyplot as plt

from src.config import Config
from .roi import ROIVisualizer
from .temporal import TemporalVisualizer
from .condition import ConditionVisualizer
from .replicate import ReplicateVisualizer
from .network_clean import CleanNetworkVisualizer


class VisualizationPipeline:
    """Orchestrates all visualization types."""
    
    def __init__(self, config: Config):
        self.config = config
        self.roi_viz = ROIVisualizer(config)
        self.temporal_viz = TemporalVisualizer(config)
        self.condition_viz = ConditionVisualizer(config)
        self.replicate_viz = ReplicateVisualizer(config)
        self.network_viz = CleanNetworkVisualizer()
    
    def create_roi_figure(self, roi_data: Dict) -> plt.Figure:
        """Create single ROI visualization."""
        return self.roi_viz.create_figure(roi_data)
    
    
    def create_temporal_figure(self, results: List[Dict]) -> plt.Figure:
        """Create temporal analysis figure."""
        return self.temporal_viz.create_temporal_figure(results)
    
    def create_condition_figure(self, results: List[Dict]) -> plt.Figure:
        """Create condition comparison figure."""
        return self.condition_viz.create_condition_figure(results)
    
    def create_replicate_variance_figure(self, results: List[Dict]) -> plt.Figure:
        """Create replicate variance figure."""
        return self.replicate_viz.create_replicate_variance_figure(results)
    
    
    def create_timepoint_region_contact_grid(self, results: List[Dict]) -> plt.Figure:
        """Create timepoint region contact grid."""
        return self.replicate_viz.create_timepoint_region_contact_grid(results, self.config)
    
    def create_network_figure(self, results: List[Dict], output_path: str) -> None:
        """Create network analysis figure."""
        self.network_viz.create_network_grid(results, output_path)
    
    def create_all_figures(self, results: List[Dict]) -> Dict[str, plt.Figure]:
        """Create all visualization types."""
        figures = {}
        
        # Only generate replicate variance analysis (12 diverse interactions)
        # Other analyses are handled by kidney_healing/run_full_report.py to avoid duplicates
        temporal_data = [r for r in results if hasattr(r['metadata'], 'injury_day') and r['metadata'].injury_day is not None]
        if temporal_data:
            figures['replicate_variance'] = self.create_replicate_variance_figure(results)
        
        # Network analysis
        network_output = self.config.output_dir / 'network_analysis_comprehensive.png'
        self.create_network_figure(results, str(network_output))
        
        return figures
    
    def save_all_figures(self, results: List[Dict]) -> None:
        """Generate and save all figures."""
        figures = self.create_all_figures(results)
        
        for name, fig in figures.items():
            output_path = self.config.output_dir / f'{name}_analysis.png'
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved {name} analysis to {output_path}")
            plt.close(fig)