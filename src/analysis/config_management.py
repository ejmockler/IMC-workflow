"""
Configuration Management System

Addresses Gemini's critique about hardcoded parameters and technical debt.
Provides centralized, validated, and versioned configuration management.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict, field
import jsonschema
from datetime import datetime
import warnings


@dataclass
class ArcSinhConfig:
    """Configuration for arcsinh transformation."""
    optimization_method: str = "percentile"
    percentile_threshold: float = 5.0
    default_cofactor: float = 1.0
    custom_cofactors: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClusteringConfig:
    """Configuration for clustering optimization."""
    optimization_method: str = "comprehensive"  # comprehensive, silhouette, gap, elbow
    k_range: List[int] = field(default_factory=lambda: [2, 15])
    use_biological_validation: bool = True
    validation_weights: Dict[str, float] = field(default_factory=lambda: {
        'elbow': 0.15,
        'silhouette': 0.30,
        'gap': 0.30,
        'calinski_harabasz': 0.15,
        'davies_bouldin': 0.10
    })
    random_state: int = 42


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    memory_limit_gb: float = 4.0
    use_optimization: bool = True
    use_sparse_methods: bool = True
    monitoring_enabled: bool = True
    chunk_size_auto: bool = True
    manual_chunk_size: Optional[int] = None


@dataclass
class SLICConfig:
    """Configuration for SLIC superpixel segmentation."""
    target_bin_size_um: float = 20.0
    resolution_um: float = 1.0
    sigma_um: float = 2.0
    compactness: float = 10.0
    use_slic: bool = True


@dataclass
class MultiScaleConfig:
    """Configuration for multi-scale analysis."""
    scales_um: List[float] = field(default_factory=lambda: [10.0, 20.0, 40.0])
    consistency_metrics: List[str] = field(default_factory=lambda: ["ari", "nmi", "cluster_stability"])
    scale_dependent_analysis: bool = True


@dataclass
class ValidationConfig:
    """Configuration for validation framework."""
    enabled: bool = True
    n_experiments: int = 10
    synthetic_data_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_cells': 1000,
        'n_clusters': 5,
        'spatial_structure': 'clustered'
    })
    enhanced_noise_models: bool = True


@dataclass
class StorageConfig:
    """Configuration for data storage."""
    format: str = "hdf5"  # hdf5, parquet, json
    compression: bool = True
    results_dir: str = "results/production_analysis"
    plots_dir: str = "plots/production_analysis"
    validation_dir: str = "validation"
    backup_enabled: bool = True


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    n_processes: Optional[int] = None  # Auto-detect if None
    batch_size: Optional[int] = None   # Auto-calculate if None
    memory_per_process_gb: float = 2.0
    progress_reporting: bool = True


@dataclass
class IMCAnalysisConfig:
    """Master configuration for IMC analysis pipeline."""
    # Core processing configurations
    arcsinh: ArcSinhConfig = field(default_factory=ArcSinhConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    slic: SLICConfig = field(default_factory=SLICConfig)
    multiscale: MultiScaleConfig = field(default_factory=MultiScaleConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    
    # Metadata
    version: str = "2.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = "IMC Analysis Configuration"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, config_path: Union[str, Path]) -> 'IMCAnalysisConfig':
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> 'IMCAnalysisConfig':
        """Create configuration from dictionary."""
        # Extract nested configurations
        config = cls()
        
        if 'arcsinh' in config_data:
            config.arcsinh = ArcSinhConfig(**config_data['arcsinh'])
        
        if 'clustering' in config_data:
            config.clustering = ClusteringConfig(**config_data['clustering'])
        
        if 'memory' in config_data:
            config.memory = MemoryConfig(**config_data['memory'])
        
        if 'slic' in config_data:
            config.slic = SLICConfig(**config_data['slic'])
        
        if 'multiscale' in config_data:
            config.multiscale = MultiScaleConfig(**config_data['multiscale'])
        
        if 'validation' in config_data:
            config.validation = ValidationConfig(**config_data['validation'])
        
        if 'storage' in config_data:
            config.storage = StorageConfig(**config_data['storage'])
        
        if 'parallel' in config_data:
            config.parallel = ParallelConfig(**config_data['parallel'])
        
        # Update metadata
        if 'version' in config_data:
            config.version = config_data['version']
        if 'created_at' in config_data:
            config.created_at = config_data['created_at']
        if 'description' in config_data:
            config.description = config_data['description']
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration parameters and return list of warnings/errors."""
        warnings_list = []
        
        # Validate arcsinh configuration
        if self.arcsinh.optimization_method not in ['percentile', 'mad', 'variance']:
            warnings_list.append(f"Invalid arcsinh optimization method: {self.arcsinh.optimization_method}")
        
        if not (0 < self.arcsinh.percentile_threshold <= 50):
            warnings_list.append(f"Invalid percentile threshold: {self.arcsinh.percentile_threshold}")
        
        # Validate clustering configuration
        if self.clustering.optimization_method not in ['comprehensive', 'silhouette', 'gap', 'elbow']:
            warnings_list.append(f"Invalid clustering optimization method: {self.clustering.optimization_method}")
        
        if len(self.clustering.k_range) != 2 or self.clustering.k_range[0] >= self.clustering.k_range[1]:
            warnings_list.append(f"Invalid k_range: {self.clustering.k_range}")
        
        if not (2 <= self.clustering.k_range[0] <= 20 and 2 <= self.clustering.k_range[1] <= 20):
            warnings_list.append(f"k_range outside reasonable bounds: {self.clustering.k_range}")
        
        # Validate memory configuration
        if not (0.1 <= self.memory.memory_limit_gb <= 64):
            warnings_list.append(f"Memory limit outside reasonable range: {self.memory.memory_limit_gb}")
        
        # Validate SLIC configuration
        if not (5.0 <= self.slic.target_bin_size_um <= 100.0):
            warnings_list.append(f"SLIC target bin size outside reasonable range: {self.slic.target_bin_size_um}")
        
        if not (0.1 <= self.slic.resolution_um <= 5.0):
            warnings_list.append(f"SLIC resolution outside reasonable range: {self.slic.resolution_um}")
        
        # Validate multi-scale configuration
        if len(self.multiscale.scales_um) < 2:
            warnings_list.append("Multi-scale analysis requires at least 2 scales")
        
        if not all(5.0 <= scale <= 100.0 for scale in self.multiscale.scales_um):
            warnings_list.append(f"Some scales outside reasonable range: {self.multiscale.scales_um}")
        
        # Validate storage configuration
        if self.storage.format not in ['hdf5', 'parquet', 'json']:
            warnings_list.append(f"Invalid storage format: {self.storage.format}")
        
        return warnings_list
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization settings."""
        return {
            'arcsinh_optimization': self.arcsinh.optimization_method,
            'clustering_optimization': self.clustering.optimization_method,
            'memory_optimization': self.memory.use_optimization,
            'use_slic': self.slic.use_slic,
            'multiscale_enabled': len(self.multiscale.scales_um) > 1,
            'validation_enabled': self.validation.enabled,
            'parallel_processing': self.parallel.n_processes != 1
        }


class ConfigurationManager:
    """
    Manages configuration versions, validation, and parameter optimization.
    """
    
    def __init__(self, config_dir: Union[str, Path] = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration schema for validation
        self.schema = self._create_validation_schema()
    
    def create_default_config(self, config_name: str = "default") -> IMCAnalysisConfig:
        """Create default configuration."""
        config = IMCAnalysisConfig()
        config.description = f"Default IMC analysis configuration - {config_name}"
        return config
    
    def create_pilot_study_config(self) -> IMCAnalysisConfig:
        """Create configuration optimized for n=2 pilot study."""
        config = IMCAnalysisConfig()
        
        # Adjust for pilot study constraints
        config.clustering.k_range = [2, 8]  # Smaller range for limited data
        config.clustering.use_biological_validation = True
        config.memory.memory_limit_gb = 2.0  # Conservative for small datasets
        config.validation.n_experiments = 5  # Fewer experiments for speed
        config.multiscale.scales_um = [15.0, 30.0]  # Two scales sufficient
        
        config.description = "Configuration optimized for n=2 pilot study"
        return config
    
    def create_production_config(self) -> IMCAnalysisConfig:
        """Create configuration for large-scale production analysis."""
        config = IMCAnalysisConfig()
        
        # Production optimizations
        config.memory.memory_limit_gb = 8.0
        config.memory.use_sparse_methods = True
        config.parallel.n_processes = None  # Auto-detect
        config.storage.format = "hdf5"
        config.storage.compression = True
        config.validation.n_experiments = 20
        
        config.description = "Configuration optimized for large-scale production analysis"
        return config
    
    def save_config(
        self, 
        config: IMCAnalysisConfig, 
        name: str,
        validate: bool = True
    ) -> Path:
        """Save configuration with validation."""
        if validate:
            validation_warnings = config.validate()
            if validation_warnings:
                warning_msg = "Configuration warnings:\n" + "\n".join(validation_warnings)
                warnings.warn(warning_msg)
        
        # Add timestamp to config
        config.created_at = datetime.now().isoformat()
        
        # Save configuration
        config_path = self.config_dir / f"{name}.json"
        config.save(config_path)
        
        return config_path
    
    def load_config(self, name: str) -> IMCAnalysisConfig:
        """Load and validate configuration."""
        config_path = self.config_dir / f"{name}.json"
        config = IMCAnalysisConfig.load(config_path)
        
        # Validate loaded configuration
        validation_warnings = config.validate()
        if validation_warnings:
            warning_msg = f"Loaded configuration '{name}' has warnings:\n" + "\n".join(validation_warnings)
            warnings.warn(warning_msg)
        
        return config
    
    def list_configs(self) -> List[str]:
        """List available configurations."""
        config_files = list(self.config_dir.glob("*.json"))
        return [f.stem for f in config_files]
    
    def compare_configs(self, config1: str, config2: str) -> Dict[str, Any]:
        """Compare two configurations and highlight differences."""
        cfg1 = self.load_config(config1)
        cfg2 = self.load_config(config2)
        
        dict1 = cfg1.to_dict()
        dict2 = cfg2.to_dict()
        
        differences = {}
        
        def compare_dicts(d1, d2, path=""):
            for key in set(d1.keys()) | set(d2.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    differences[current_path] = {"config1": "MISSING", "config2": d2[key]}
                elif key not in d2:
                    differences[current_path] = {"config1": d1[key], "config2": "MISSING"}
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dicts(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    differences[current_path] = {"config1": d1[key], "config2": d2[key]}
        
        compare_dicts(dict1, dict2)
        
        return {
            "config1": config1,
            "config2": config2,
            "differences": differences,
            "n_differences": len(differences)
        }
    
    def optimize_config_for_dataset(
        self, 
        base_config: IMCAnalysisConfig,
        n_rois: int,
        n_proteins: int,
        avg_roi_size: int,
        available_memory_gb: float
    ) -> IMCAnalysisConfig:
        """Optimize configuration parameters for specific dataset characteristics."""
        optimized_config = IMCAnalysisConfig.from_dict(base_config.to_dict())
        
        # Memory optimization based on dataset size
        estimated_memory_per_roi = (avg_roi_size * n_proteins * 8 * 5) / (1024**3)  # Rough estimate
        total_estimated_memory = estimated_memory_per_roi * n_rois
        
        if total_estimated_memory > available_memory_gb:
            # Aggressive memory optimization needed
            optimized_config.memory.memory_limit_gb = min(
                available_memory_gb * 0.8,  # Use 80% of available memory
                4.0  # Cap at 4GB for stability
            )
            optimized_config.memory.use_sparse_methods = True
            optimized_config.parallel.memory_per_process_gb = 1.0
        
        # Clustering optimization based on dataset size
        if n_rois < 10:
            # Small dataset - conservative clustering
            optimized_config.clustering.k_range = [2, 6]
        elif n_rois > 100:
            # Large dataset - allow more clusters
            optimized_config.clustering.k_range = [2, 20]
        
        # Multi-scale optimization based on ROI size
        if avg_roi_size < 10000:  # Small ROIs
            optimized_config.multiscale.scales_um = [10.0, 20.0]
        elif avg_roi_size > 100000:  # Large ROIs
            optimized_config.multiscale.scales_um = [10.0, 20.0, 40.0, 80.0]
        
        optimized_config.description = f"Auto-optimized for {n_rois} ROIs, {n_proteins} proteins"
        
        return optimized_config
    
    def _create_validation_schema(self) -> Dict[str, Any]:
        """Create JSON schema for configuration validation."""
        # This would be a comprehensive JSON schema
        # For brevity, returning a simple schema structure
        return {
            "type": "object",
            "properties": {
                "version": {"type": "string"},
                "description": {"type": "string"},
                "arcsinh": {"type": "object"},
                "clustering": {"type": "object"},
                "memory": {"type": "object"},
                "slic": {"type": "object"},
                "multiscale": {"type": "object"},
                "validation": {"type": "object"},
                "storage": {"type": "object"},
                "parallel": {"type": "object"}
            }
        }