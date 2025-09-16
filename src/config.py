"""Unified configuration module for IMC analysis."""

import json
from pathlib import Path
from typing import Dict, Any, Optional


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
        
        # Legacy support (for backwards compatibility)
        self.proteins = self.channels.get('protein_channels', [])
        self.experimental = self.raw.get('experimental', {})
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file.
        
        Returns:
            Dictionary containing configuration data
        """
        if not self.config_path.exists():
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return {}
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
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