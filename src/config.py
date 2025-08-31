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
        
        # Data paths
        self.data_dir = Path(self.raw.get('data_dir', 'data/241218_IMC_Alun'))
        self.output_dir = Path(self.raw.get('output_dir', 'results'))
        
        # Protein configuration
        self.proteins = self.raw.get('proteins', [])
        self.functional_groups = self.raw.get('proteins', {}).get('functional_groups', {})
        
        # Analysis parameters
        self.contact_radius = self.raw.get('contact_radius', 15.0)
        self.min_blob_size = self.raw.get('min_blob_size', 50)
        self.spatial_distances = self.raw.get('spatial_distances', [5, 10, 25, 50])
        self.max_pixels = self.raw.get('max_pixels', 100000)
        
        # Experimental configuration
        self.experimental = self.raw.get('experimental', {})
        self.metadata_lookup = self.raw.get('metadata_lookup', {})
        
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