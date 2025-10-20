"""
Pydantic Schema Validation for IMC Configuration (Priority 3)

CRITICAL: Prevents channel overlap that would invalidate biological analysis.

Validates:
- Channel overlap (protein vs calibration/carrier_gas/background)
- Coabundance feature selection enforcement
- Parameter ranges and types
- Cross-field dependencies
"""

import json
from typing import List, Dict, Optional, Any
from pathlib import Path
from pydantic import BaseModel, field_validator, Field, ValidationError, model_validator, ConfigDict


class ChannelConfig(BaseModel):
    """
    Channel configuration with strict validation.

    CRITICAL VALIDATION: Ensures protein channels don't overlap with
    technical channels (calibration, carrier gas, background).

    Failure mode without this: Calibration beads analyzed as cells,
    carrier gas analyzed as protein expression.
    """

    protein_channels: List[str] = Field(
        ...,
        min_length=1,
        description="Protein marker channels for biological analysis"
    )
    dna_channels: List[str] = Field(
        ...,
        min_length=1,
        description="DNA channels for segmentation (e.g., DNA1, DNA2)"
    )
    background_channel: str = Field(
        ...,
        description="Background channel for pixel-wise subtraction"
    )
    calibration_channels: List[str] = Field(
        default_factory=list,
        description="Calibration channels (excluded from analysis)"
    )
    carrier_gas_channel: str = Field(
        default="",
        description="Carrier gas channel (excluded from analysis)"
    )
    excluded_channels: List[str] = Field(
        default_factory=list,
        description="Additional channels to exclude"
    )

    @field_validator('protein_channels')
    @classmethod
    def validate_no_duplicates_protein(cls, v):
        """Ensure no duplicate channels in protein_channels."""
        if len(v) != len(set(v)):
            duplicates = [x for x in v if v.count(x) > 1]
            raise ValueError(
                f"Duplicate channels found in protein_channels: {set(duplicates)}"
            )
        return v

    @field_validator('dna_channels')
    @classmethod
    def validate_no_duplicates_dna(cls, v):
        """Ensure no duplicate channels in dna_channels."""
        if len(v) != len(set(v)):
            duplicates = [x for x in v if v.count(x) > 1]
            raise ValueError(
                f"Duplicate channels found in dna_channels: {set(duplicates)}"
            )
        return v

    @model_validator(mode='after')
    def validate_channel_overlaps(self):
        """
        CRITICAL VALIDATION: Ensure protein channels don't overlap
        with technical channels or DNA channels.

        This prevents catastrophic scientific errors where calibration
        beads or carrier gas are analyzed as biological signal.
        """
        technical_channels = set()

        # Collect all technical channels
        technical_channels.update(self.calibration_channels)
        if self.carrier_gas_channel:
            technical_channels.add(self.carrier_gas_channel)
        technical_channels.add(self.background_channel)
        technical_channels.update(self.excluded_channels)

        # Check protein vs technical overlap
        protein_set = set(self.protein_channels)
        overlap = protein_set & technical_channels

        if overlap:
            raise ValueError(
                f"CRITICAL ERROR: Protein channels overlap with technical channels: {overlap}.\n"
                f"This would invalidate all biological analysis!\n"
                f"Protein channels: {sorted(protein_set)}\n"
                f"Technical channels: {sorted(technical_channels)}\n"
                f"\nRemove {overlap} from protein_channels or technical channels."
            )

        # Check protein vs DNA overlap
        dna_set = set(self.dna_channels)
        dna_overlap = protein_set & dna_set

        if dna_overlap:
            raise ValueError(
                f"DNA channels overlap with protein channels: {dna_overlap}.\n"
                f"DNA channels should be separate from protein channels."
            )

        return self


class CoabundanceConfig(BaseModel):
    """Coabundance feature configuration with validation."""

    use_feature_selection: bool = Field(
        True,
        description="Use LASSO feature selection to prevent overfitting"
    )
    target_n_features: int = Field(
        30,
        ge=10,
        le=100,
        description="Target number of features after selection"
    )
    selection_method: str = Field(
        "lasso",
        pattern="^(lasso|mutual_info|variance)$",
        description="Feature selection method"
    )

    @field_validator('target_n_features')
    @classmethod
    def validate_target_features_reasonable(cls, v):
        """
        Validate target_n_features is reasonable.

        Too high → overfitting risk
        Too low → loss of biological signal
        """
        if v > 50:
            raise ValueError(
                f"target_n_features={v} is too high. "
                f"Typical datasets should use ~30 features (≈√N). "
                f"High feature counts increase overfitting risk."
            )
        return v


class ClusteringConfig(BaseModel):
    """Clustering configuration with validation."""

    method: str = Field(
        ...,
        pattern="^(leiden|hdbscan|louvain)$",
        description="Clustering method"
    )
    k_neighbors: int = Field(
        ...,
        ge=5,
        le=30,
        description="Number of neighbors for graph construction"
    )
    k_neighbors_by_scale: Optional[Dict[str, int]] = Field(
        None,
        description="Scale-specific k_neighbors (overrides default)"
    )
    spatial_weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weight for spatial component (0=feature-only, 1=spatial-only)"
    )
    random_state: int = Field(
        42,
        ge=0,
        description="Random seed for reproducibility"
    )
    use_coabundance_features: bool = Field(
        True,
        description="Use coabundance features (products, ratios, covariances)"
    )
    coabundance_options: Optional[CoabundanceConfig] = None

    @model_validator(mode='after')
    def validate_coabundance_requires_selection(self):
        """
        CRITICAL: If coabundance is enabled, feature selection must be enabled.

        Without selection: 9 proteins → 153 features = catastrophic overfitting!
        """
        if self.coabundance_options is None:
            return self

        if self.use_coabundance_features:
            if not self.coabundance_options.use_feature_selection:
                raise ValueError(
                    "CRITICAL: use_feature_selection must be True when "
                    "use_coabundance_features=True!\n"
                    "Without feature selection: 9 proteins → 153 coabundance features.\n"
                    "This creates catastrophic overfitting risk.\n"
                    "Enable feature selection or disable coabundance features."
                )

        return self

    @field_validator('k_neighbors_by_scale')
    @classmethod
    def validate_k_by_scale_ranges(cls, v):
        """Validate scale-specific k values are in valid range."""
        if v is None:
            return v

        for scale_str, k in v.items():
            # Skip comment keys
            if scale_str.startswith('_'):
                continue

            try:
                scale = float(scale_str)
                if scale <= 0:
                    raise ValueError(f"Invalid scale: {scale_str} (must be > 0)")
            except ValueError:
                if not scale_str.startswith('_'):
                    raise ValueError(f"Invalid scale format: {scale_str}")

            if not isinstance(k, int) or k < 5 or k > 30:
                raise ValueError(
                    f"k_neighbors={k} out of valid range [5, 30] for scale {scale_str}"
                )

        return v

    @field_validator('spatial_weight')
    @classmethod
    def validate_spatial_weight_reasonable(cls, v):
        """
        Validate spatial weight is reasonable for IMC data.

        IMC has both feature and spatial information - should use both.
        """
        if v == 0.0:
            raise ValueError(
                "spatial_weight=0.0 ignores all spatial information. "
                "IMC data has rich spatial structure - use spatial_weight > 0."
            )
        if v == 1.0:
            raise ValueError(
                "spatial_weight=1.0 ignores all feature information. "
                "This defeats the purpose of protein expression analysis."
            )
        return v


class ProcessingConfig(BaseModel):
    """Processing configuration validation."""

    background_correction: Dict[str, Any]
    dna_processing: Dict[str, Any]
    arcsinh_transform: Dict[str, Any]
    normalization: Optional[Dict[str, Any]] = None

    @field_validator('dna_processing')
    @classmethod
    def validate_dna_processing_params(cls, v):
        """Validate DNA processing parameters."""
        if 'resolution_um' not in v:
            raise ValueError("dna_processing.resolution_um is required")

        resolution = v['resolution_um']
        if resolution <= 0:
            raise ValueError(f"resolution_um must be > 0, got {resolution}")
        if resolution > 10:
            raise ValueError(
                f"resolution_um={resolution} is very high. "
                f"Typical IMC resolution is 1-2 μm."
            )

        return v

    @field_validator('arcsinh_transform')
    @classmethod
    def validate_arcsinh_params(cls, v):
        """Validate arcsinh transformation parameters."""
        # Allow either auto_cofactor or optimization_method
        has_auto = 'auto_cofactor' in v
        has_optimization = 'optimization_method' in v

        if not (has_auto or has_optimization):
            raise ValueError(
                "arcsinh_transform requires either 'auto_cofactor' or 'optimization_method'"
            )

        return v


class QualityControlConfig(BaseModel):
    """Quality control thresholds validation."""

    calibration: Optional[Dict[str, Any]] = None
    carrier_gas: Optional[Dict[str, Any]] = None
    dna_signal: Optional[Dict[str, Any]] = None
    thresholds: Optional[Dict[str, Any]] = None  # Allow flexible QC structure

    @field_validator('calibration')
    @classmethod
    def validate_calibration_thresholds(cls, v):
        """Validate calibration QC thresholds."""
        if v is None:
            return v

        if 'max_cv' in v:
            max_cv = v['max_cv']
            if max_cv <= 0 or max_cv >= 1:
                raise ValueError(
                    f"calibration.max_cv={max_cv} out of range (0, 1). "
                    f"Typical threshold is 0.2 (20% CV)."
                )
        return v


class OutputConfig(BaseModel):
    """Output configuration validation."""

    results_dir: str = Field(..., description="Results output directory")
    save_plots: bool = Field(True, description="Save visualization plots")

    @field_validator('results_dir')
    @classmethod
    def validate_results_dir_not_empty(cls, v):
        """Ensure results directory is specified."""
        if not v or v.strip() == "":
            raise ValueError("results_dir cannot be empty")
        return v


class PerformanceConfig(BaseModel):
    """Performance configuration validation."""

    parallel_processes: int = Field(
        8,
        ge=1,
        le=64,
        description="Number of parallel processes"
    )
    memory_limit_gb: float = Field(
        8.0,
        ge=1.0,
        le=256.0,
        description="Memory limit per process (GB)"
    )
    process_sequentially: bool = Field(
        False,
        description="Force sequential processing (debugging)"
    )

    @field_validator('parallel_processes')
    @classmethod
    def validate_parallel_processes_reasonable(cls, v):
        """Validate parallel process count is reasonable."""
        if v > 32:
            import warnings
            warnings.warn(
                f"parallel_processes={v} is very high. "
                f"Typical systems have 8-16 cores."
            )
        return v


class IMCConfig(BaseModel):
    """
    Root configuration with full validation.

    This is the main config schema that enforces all validation rules.
    """

    project_name: str = Field(..., description="Project name")
    project_id: str = Field(..., description="Unique project identifier")

    channels: ChannelConfig
    processing: ProcessingConfig
    quality_control: QualityControlConfig
    output: OutputConfig
    performance: PerformanceConfig

    # Analysis config is complex nested dict - validate key parts
    analysis: Dict[str, Any]

    # Segmentation config
    segmentation: Dict[str, Any]

    model_config = ConfigDict(extra='allow')  # Allow extra fields for backward compatibility

    @field_validator('analysis')
    @classmethod
    def validate_analysis_structure(cls, v):
        """Validate key analysis configuration structure."""
        # Check for clustering config
        if 'clustering' in v:
            clustering = v['clustering']

            # Validate using ClusteringConfig model
            try:
                ClusteringConfig(**clustering)
            except ValidationError as e:
                raise ValueError(f"Invalid clustering configuration: {e}")

        return v


def load_validated_config(config_path: str) -> IMCConfig:
    """
    Load and validate config using Pydantic.

    Args:
        config_path: Path to config.json

    Returns:
        Validated IMCConfig instance

    Raises:
        ValidationError: If config validation fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    try:
        validated_config = IMCConfig(**config_dict)
        return validated_config
    except ValidationError as e:
        print("\n❌ CONFIG VALIDATION FAILED:")
        print("=" * 80)

        # Pretty print validation errors
        for error in e.errors():
            loc = " → ".join(str(x) for x in error['loc'])
            msg = error['msg']
            print(f"\n  Location: {loc}")
            print(f"  Error: {msg}")

        print("\n" + "=" * 80)
        raise


def validate_config_file(config_path: str) -> bool:
    """
    Validate config file without raising exceptions.

    Returns:
        True if valid, False otherwise
    """
    try:
        load_validated_config(config_path)
        print(f"✅ Config validation passed: {config_path}")
        return True
    except ValidationError:
        print(f"❌ Config validation failed: {config_path}")
        return False
    except Exception as e:
        print(f"❌ Config validation error: {e}")
        return False


if __name__ == "__main__":
    """CLI for validating config files."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.config_schema <config_file>")
        print("\nExample:")
        print("  python -m src.config_schema config.json")
        sys.exit(1)

    config_file = sys.argv[1]
    success = validate_config_file(config_file)
    sys.exit(0 if success else 1)
