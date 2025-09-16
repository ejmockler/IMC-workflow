"""Abstract base class for experiment-specific analysis frameworks."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt

from src.config import Config


class ExperimentFramework(ABC):
    """Abstract base class for experiment-specific analysis and visualization."""
    
    def __init__(self, config: Config):
        """Initialize experiment framework with configuration.
        
        Args:
            config: Configuration object containing experimental parameters
        """
        self.config = config
        self.experimental_config = config.raw.get('experimental', {})
    
    @abstractmethod
    def get_figure_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Define experiment-specific figure specifications.
        
        Returns:
            Dictionary mapping figure names to their specifications:
            {
                'figure_name': {
                    'method': 'visualization_method_name',
                    'params': {'param1': value1, ...},
                    'description': 'Figure description for scientific communication',
                    'priority': int  # Lower numbers = higher priority
                }
            }
        """
        pass
    
    @abstractmethod
    def get_experimental_groupings(self) -> Dict[str, List[str]]:
        """Define how to group ROIs for analysis.
        
        Returns:
            Dictionary mapping grouping types to their categories:
            {
                'timepoints': ['Sham', 'Day1', 'Day3', 'Day7'],
                'regions': ['Cortex', 'Medulla'],
                'replicates': ['MS1', 'MS2']
            }
        """
        pass
    
    @abstractmethod
    def get_functional_groups(self) -> Dict[str, List[str]]:
        """Define functional protein groups for biological interpretation.
        
        Returns:
            Dictionary mapping functional group names to protein lists
        """
        pass
    
    @abstractmethod
    def create_experimental_summary_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Create experiment-specific summary metrics.
        
        Args:
            results: List of ROI analysis results
            
        Returns:
            Dictionary containing experiment-specific metrics
        """
        pass
    
    def validate_experimental_design(self, results: List[Dict]) -> bool:
        """Validate that results match expected experimental design.
        
        Args:
            results: List of ROI analysis results
            
        Returns:
            True if experimental design is valid, False otherwise
        """
        # Extract unique experimental conditions from results
        conditions = set()
        for roi in results:
            metadata = roi.get('metadata', {})
            if hasattr(metadata, 'to_dict'):
                metadata = metadata.to_dict()
            elif hasattr(metadata, '_asdict'):
                metadata = metadata._asdict()
            
            timepoint = metadata.get('injury_day') or metadata.get('timepoint')
            region = metadata.get('tissue_region') or metadata.get('region')
            replicate = metadata.get('mouse_replicate') or metadata.get('mouse_id')
            
            conditions.add((timepoint, region, replicate))
        
        expected_groupings = self.get_experimental_groupings()
        expected_timepoints = set(self.experimental_config.get('timepoints', []))
        expected_regions = set(expected_groupings.get('regions', []))
        expected_replicates = set(expected_groupings.get('replicates', []))
        
        # Check if we have expected coverage
        observed_timepoints = {c[0] for c in conditions if c[0] is not None}
        observed_regions = {c[1] for c in conditions if c[1] is not None}
        observed_replicates = {c[2] for c in conditions if c[2] is not None}
        
        print(f"Experimental Design Validation:")
        print(f"  Expected timepoints: {expected_timepoints}")
        print(f"  Observed timepoints: {observed_timepoints}")
        print(f"  Expected regions: {expected_regions}")
        print(f"  Observed regions: {observed_regions}")
        print(f"  Expected replicates: {expected_replicates}")
        print(f"  Observed replicates: {observed_replicates}")
        
        return (
            len(observed_timepoints & expected_timepoints) > 0 and
            len(observed_regions & expected_regions) > 0 and
            len(observed_replicates & expected_replicates) > 0
        )
    
    def get_output_prefix(self) -> str:
        """Get experiment-specific output file prefix.
        
        Returns:
            String prefix for output files
        """
        return self.__class__.__name__.lower().replace('experiment', '')
    
    def generate_experiment_report(self, results: List[Dict], output_dir: str = None) -> Dict[str, str]:
        """Generate complete experiment report with all figures.
        
        Args:
            results: List of ROI analysis results
            output_dir: Output directory (uses config default if None)
            
        Returns:
            Dictionary mapping figure names to output file paths
        """
        if not self.validate_experimental_design(results):
            print("Warning: Experimental design validation failed. Proceeding with available data.")
        
        from src.visualization.main import VisualizationPipeline
        
        viz_pipeline = VisualizationPipeline(self.config)
        figures = self.get_figure_specifications()
        output_paths = {}
        
        output_directory = output_dir or self.config.output_dir
        prefix = self.get_output_prefix()
        
        # Sort figures by priority
        sorted_figures = sorted(figures.items(), key=lambda x: x[1].get('priority', 999))
        
        print(f"Generating {len(sorted_figures)} experiment-specific figures...")
        
        for fig_name, fig_spec in sorted_figures:
            try:
                method_name = fig_spec['method']
                params = fig_spec.get('params', {})
                
                # Get the method from visualization pipeline
                if hasattr(viz_pipeline, method_name):
                    method = getattr(viz_pipeline, method_name)
                    fig = method(results, **params)
                    
                    # Save figure
                    output_path = f"{output_directory}/{prefix}_{fig_name}.png"
                    fig.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    output_paths[fig_name] = output_path
                    print(f"  ✓ {fig_spec['description']}: {output_path}")
                else:
                    print(f"  ✗ Method '{method_name}' not found in VisualizationPipeline")
                    
            except Exception as e:
                print(f"  ✗ Error generating {fig_name}: {e}")
        
        # Generate summary metrics
        try:
            summary_metrics = self.create_experimental_summary_metrics(results)
            metrics_path = f"{output_directory}/{prefix}_summary_metrics.json"
            
            import json
            with open(metrics_path, 'w') as f:
                json.dump(summary_metrics, f, indent=2)
            
            output_paths['summary_metrics'] = metrics_path
            print(f"  ✓ Summary metrics: {metrics_path}")
            
        except Exception as e:
            print(f"  ✗ Error generating summary metrics: {e}")
        
        return output_paths