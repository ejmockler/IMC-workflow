"""
Multi-Project Configuration Management

Handles loading and validating configurations for single projects
and multi-project meta-analyses.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from .config import Config


@dataclass
class ProjectInfo:
    """Project information for multi-project analyses."""
    project_id: str
    project_name: str
    version: str
    config_path: Path
    data_dir: Path
    description: Optional[str] = None


class MultiProjectConfig:
    """
    Configuration manager for multi-project analyses.
    
    Supports loading multiple project configurations and standardizing
    metadata schemas across projects for meta-analysis.
    """
    
    def __init__(self, registry_path: str = "project_registry.json"):
        """
        Initialize multi-project configuration manager.
        
        Args:
            registry_path: Path to project registry file
        """
        self.registry_path = Path(registry_path)
        self.projects: Dict[str, ProjectInfo] = {}
        self.configs: Dict[str, Config] = {}
        self.logger = logging.getLogger('MultiProjectConfig')
        
        if self.registry_path.exists():
            self._load_registry()
    
    def _load_registry(self):
        """Load project registry from file."""
        try:
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
            
            for project_data in registry.get('projects', []):
                project_info = ProjectInfo(
                    project_id=project_data['project_id'],
                    project_name=project_data['project_name'],
                    version=project_data['version'],
                    config_path=Path(project_data['config_path']),
                    data_dir=Path(project_data['data_dir']),
                    description=project_data.get('description')
                )
                self.projects[project_info.project_id] = project_info
                
            self.logger.info(f"Loaded {len(self.projects)} projects from registry")
            
        except Exception as e:
            self.logger.error(f"Failed to load project registry: {e}")
            self.projects = {}
    
    def register_project(self, 
                        project_id: str,
                        config_path: Union[str, Path], 
                        description: str = None) -> None:
        """
        Register a new project in the registry.
        
        Args:
            project_id: Unique project identifier
            config_path: Path to project config.json
            description: Optional project description
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load config to extract project info
        config = Config(str(config_path))
        
        project_info = ProjectInfo(
            project_id=project_id,
            project_name=config.raw.get('project_name', project_id),
            version=config.raw.get('project_version', '1.0'),
            config_path=config_path,
            data_dir=Path(config.data.get('raw_data_dir', '.')),
            description=description
        )
        
        self.projects[project_id] = project_info
        self._save_registry()
        
        self.logger.info(f"Registered project: {project_id}")
    
    def load_project(self, project_id: str) -> Config:
        """
        Load configuration for a specific project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Loaded configuration object
        """
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found in registry. "
                           f"Available projects: {list(self.projects.keys())}")
        
        project_info = self.projects[project_id]
        
        if project_id not in self.configs:
            self.configs[project_id] = Config(str(project_info.config_path))
            
        return self.configs[project_id]
    
    def load_all_projects(self) -> Dict[str, Config]:
        """Load all registered projects."""
        configs = {}
        for project_id in self.projects:
            configs[project_id] = self.load_project(project_id)
        return configs
    
    def get_standardized_metadata_schema(self) -> Dict[str, str]:
        """
        Generate standardized metadata schema for cross-project analysis.
        
        Returns:
            Dictionary mapping standard field names to descriptions
        """
        return {
            'project_id': 'Project identifier for cross-project analyses',
            'replicate_id': 'Biological replicate identifier (standardized across projects)',
            'timepoint': 'Experimental timepoint (standardized units)',
            'condition': 'Experimental condition (standardized names)',
            'region': 'Anatomical/spatial region (standardized names)',
            'batch_id': 'Technical batch identifier including project prefix'
        }
    
    def create_cross_project_batch_id(self, project_id: str, batch_id: str) -> str:
        """
        Create unique batch ID for cross-project analysis.
        
        Args:
            project_id: Project identifier
            batch_id: Original batch ID from project
            
        Returns:
            Cross-project unique batch ID
        """
        return f"{project_id}_{batch_id}"
    
    def validate_cross_project_compatibility(self, project_ids: List[str]) -> Dict[str, Any]:
        """
        Validate that projects can be analyzed together.
        
        Args:
            project_ids: List of project IDs to validate
            
        Returns:
            Dictionary with compatibility analysis
        """
        validation_results = {
            'compatible': True,
            'warnings': [],
            'errors': [],
            'schema_comparison': {}
        }
        
        if len(project_ids) < 2:
            validation_results['errors'].append("Need at least 2 projects for cross-project analysis")
            validation_results['compatible'] = False
            return validation_results
        
        configs = {pid: self.load_project(pid) for pid in project_ids}
        
        # Check protein panel compatibility
        protein_panels = {pid: set(config.channels.get('protein_channels', [])) 
                         for pid, config in configs.items()}
        
        # Find common proteins
        common_proteins = set.intersection(*protein_panels.values())
        all_proteins = set.union(*protein_panels.values())
        
        if len(common_proteins) == 0:
            validation_results['errors'].append("No common protein markers across projects")
            validation_results['compatible'] = False
        elif len(common_proteins) < len(all_proteins) * 0.7:  # <70% overlap
            validation_results['warnings'].append(
                f"Limited protein overlap: {len(common_proteins)}/{len(all_proteins)} common markers"
            )
        
        validation_results['schema_comparison'] = {
            'common_proteins': list(common_proteins),
            'project_specific_proteins': {
                pid: list(protein_panels[pid] - common_proteins)
                for pid in project_ids
            },
            'total_common': len(common_proteins),
            'total_unique': len(all_proteins)
        }
        
        # Check metadata schema compatibility
        metadata_schemas = {pid: config.metadata_tracking for pid, config in configs.items()}
        
        # Validate that required metadata fields are available
        required_fields = ['replicate_column', 'timepoint_column', 'condition_column']
        for pid, schema in metadata_schemas.items():
            missing_fields = [field for field in required_fields if field not in schema]
            if missing_fields:
                validation_results['errors'].append(
                    f"Project {pid} missing required metadata fields: {missing_fields}"
                )
                validation_results['compatible'] = False
        
        return validation_results
    
    def _save_registry(self):
        """Save project registry to file."""
        registry_data = {
            'version': '1.0',
            'projects': [
                {
                    'project_id': info.project_id,
                    'project_name': info.project_name,
                    'version': info.version,
                    'config_path': str(info.config_path),
                    'data_dir': str(info.data_dir),
                    'description': info.description
                }
                for info in self.projects.values()
            ]
        }
        
        with open(self.registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def list_projects(self) -> List[Dict[str, str]]:
        """List all registered projects."""
        return [
            {
                'project_id': info.project_id,
                'project_name': info.project_name,
                'version': info.version,
                'description': info.description or 'No description'
            }
            for info in self.projects.values()
        ]
    
    def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """Get detailed summary of a project."""
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project_info = self.projects[project_id]
        config = self.load_project(project_id)
        
        return {
            'project_info': {
                'project_id': project_info.project_id,
                'project_name': project_info.project_name,
                'version': project_info.version,
                'description': project_info.description
            },
            'data_info': {
                'data_dir': str(project_info.data_dir),
                'config_path': str(project_info.config_path)
            },
            'analysis_info': {
                'protein_channels': config.channels.get('protein_channels', []),
                'n_proteins': len(config.channels.get('protein_channels', [])),
                'metadata_schema': config.metadata_tracking,
                'batch_correction_enabled': config.analysis.get('batch_correction', {}).get('enabled', False)
            }
        }


def load_project_config(project_id: str, registry_path: str = "project_registry.json") -> Config:
    """
    Convenience function to load a single project configuration.
    
    Args:
        project_id: Project identifier
        registry_path: Path to project registry
        
    Returns:
        Project configuration
    """
    manager = MultiProjectConfig(registry_path)
    return manager.load_project(project_id)