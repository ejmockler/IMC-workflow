"""Unified configuration module for IMC analysis."""

import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional


def migrate_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate old config format to new format.
    Handles removal of n_clusters and addition of resolution parameters.
    """
    modified = False
    
    if 'analysis' in config_dict:
        clustering = config_dict['analysis'].get('clustering', {})
        
        # Check for old n_clusters parameter
        if 'n_clusters' in clustering:
            warnings.warn(
                "Config contains deprecated 'n_clusters' parameter. "
                "Using resolution-based clustering instead.",
                DeprecationWarning
            )
            del clustering['n_clusters']
            modified = True
            
        # Ensure new parameters exist
        if 'resolution_range' not in clustering:
            clustering['resolution_range'] = [0.5, 2.0]
            modified = True
        if 'optimization_method' not in clustering and 'optimization_method' in clustering:
            # Fix: only set if not present
            pass
        elif 'optimization_method' not in clustering:
            clustering['optimization_method'] = 'stability'
            modified = True
            
    if 'kidney_experiment' in config_dict:
        exp = config_dict['kidney_experiment']
        clustering = exp.get('clustering', {})
        if 'n_clusters' in clustering:
            warnings.warn(
                "Removing n_clusters from kidney_experiment.clustering",
                DeprecationWarning
            )
            del clustering['n_clusters']
            modified = True
            
    if modified:
        warnings.warn("Config was migrated to new format", UserWarning)
            
    return config_dict


class Config:
    """Single source of truth for all configuration."""
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize configuration from JSON file.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config_path = Path(config_path)
        self.raw = self._load_config()
        
        # Data configuration
        self.data = self.raw.get('data', {})
        self.data_dir = Path(self.data.get('raw_data_dir', 'data/241218_IMC_Alun'))
        
        # Channel configuration - CRITICAL FOR PROPER ANALYSIS
        self.channels = self.raw.get('channels', {})
        self.channel_groups = self.raw.get('channel_groups', {})
        
        # Processing configuration
        self.processing = self.raw.get('processing', {})
        self.dna_processing = self.processing.get('dna_processing', {})
        
        # Segmentation configuration  
        self.segmentation = self.raw.get('segmentation', {})
        
        # Analysis parameters
        self.analysis = self.raw.get('analysis', {})
        
        # Quality control
        self.quality_control = self.raw.get('quality_control', {})
        
        # Output configuration
        self.output = self.raw.get('output', {})
        self.output_dir = Path(self.output.get('results_dir', 'results'))
        
        # Performance settings
        self.performance = self.raw.get('performance', {})
        
        # Metadata tracking
        self.metadata_tracking = self.raw.get('metadata_tracking', {})
        
        # Visualization configuration
        self.visualization = self.raw.get('visualization', {})
        
        # Legacy support (for backwards compatibility)
        self.proteins = self.channels.get('protein_channels', [])
        self.experimental = self.raw.get('experimental', {})
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Validate channel consistency
        self.validate_channel_consistency()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file and apply migrations.
        
        Returns:
            Dictionary containing configuration data
        """
        if not self.config_path.exists():
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return {}
        
        with open(self.config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Apply migrations to handle old format
        config_dict = migrate_config(config_dict)
        
        return config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.
        
        Args:
            key: Configuration key (supports nested keys with dots)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.raw
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update a configuration value.
        
        Args:
            key: Configuration key
            value: New value
        """
        keys = key.split('.')
        config = self.raw
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Output path (uses original path if None)
        """
        output_path = path or self.config_path
        with open(output_path, 'w') as f:
            json.dump(self.raw, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self.raw.copy()
    
    def validate_channel_consistency(self) -> None:
        """Validate that visualization config references valid channels.
        
        Raises:
            ValueError: If primary_markers reference non-existent protein channels
        """
        protein_channels = set(self.channels.get('protein_channels', []))
        primary_markers = self.visualization.get('validation_plots', {}).get('primary_markers', {})
        
        for group_name, marker in primary_markers.items():
            if marker not in protein_channels:
                print(f"Warning: Primary marker '{marker}' for group '{group_name}' not found in protein_channels")
        
        # Validate that channel_groups reference valid protein channels
        channel_groups = self.channel_groups
        for group_name, group_data in channel_groups.items():
            if isinstance(group_data, dict):
                # Handle nested structure like immune_markers.pan_leukocyte
                for subgroup_name, proteins in group_data.items():
                    if isinstance(proteins, list):
                        for protein in proteins:
                            if protein not in protein_channels:
                                print(f"Warning: Channel '{protein}' in group '{group_name}.{subgroup_name}' not found in protein_channels")
            elif isinstance(group_data, list):
                for protein in group_data:
                    if protein not in protein_channels:
                        print(f"Warning: Channel '{protein}' in group '{group_name}' not found in protein_channels")