"""
Tests for Pydantic Schema Validation (Priority 3)

Validates that:
1. ChannelConfig prevents protein/technical channel overlap (CRITICAL)
2. CoabundanceConfig enforces feature selection when enabled
3. ClusteringConfig validates parameter ranges
4. Config validation catches configuration errors
5. Schema handles actual config.json structure
6. Validation errors provide clear feedback
"""

import pytest
from pydantic import ValidationError

from src.config_schema import (
    ChannelConfig,
    CoabundanceConfig,
    ClusteringConfig,
    ProcessingConfig,
    QualityControlConfig,
    OutputConfig,
    PerformanceConfig,
    IMCConfig,
    load_validated_config,
    validate_config_file
)


class TestChannelConfigValidation:
    """Test critical channel overlap validation."""

    def test_valid_channel_config(self):
        """Test valid channel configuration."""
        config = ChannelConfig(
            protein_channels=["CD31", "CD34", "CD45"],
            dna_channels=["DNA1", "DNA2"],
            background_channel="Ar80",
            calibration_channels=["Ce140", "Eu151"],
            carrier_gas_channel="Xe131",
            excluded_channels=[]
        )

        assert len(config.protein_channels) == 3
        assert len(config.dna_channels) == 2

    def test_protein_technical_overlap_detected(self):
        """
        CRITICAL: Detect protein/technical channel overlap.

        This prevents catastrophic errors where calibration beads
        or carrier gas are analyzed as biological signal.
        """
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig(
                protein_channels=["CD31", "CD34", "Ar80"],  # Ar80 is background!
                dna_channels=["DNA1", "DNA2"],
                background_channel="Ar80",
                calibration_channels=["Ce140"],
                carrier_gas_channel="Xe131",
                excluded_channels=[]
            )

        error_str = str(exc_info.value)
        assert "CRITICAL ERROR" in error_str
        assert "Ar80" in error_str
        assert "overlap" in error_str.lower()

    def test_protein_calibration_overlap_detected(self):
        """Detect overlap between protein channels and calibration channels."""
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig(
                protein_channels=["CD31", "Ce140", "CD45"],  # Ce140 is calibration!
                dna_channels=["DNA1", "DNA2"],
                background_channel="Ar80",
                calibration_channels=["Ce140", "Eu151"],
                carrier_gas_channel="Xe131",
                excluded_channels=[]
            )

        error_str = str(exc_info.value)
        assert "Ce140" in error_str

    def test_protein_dna_overlap_detected(self):
        """Detect overlap between protein and DNA channels."""
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig(
                protein_channels=["CD31", "DNA1", "CD45"],  # DNA1 is DNA channel!
                dna_channels=["DNA1", "DNA2"],
                background_channel="Ar80",
                calibration_channels=["Ce140"],
                carrier_gas_channel="Xe131",
                excluded_channels=[]
            )

        error_str = str(exc_info.value)
        assert "DNA1" in error_str

    def test_duplicate_protein_channels_rejected(self):
        """Reject duplicate channels in protein_channels."""
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig(
                protein_channels=["CD31", "CD34", "CD31"],  # Duplicate CD31
                dna_channels=["DNA1", "DNA2"],
                background_channel="Ar80",
                calibration_channels=["Ce140"],
                carrier_gas_channel="Xe131",
                excluded_channels=[]
            )

        error_str = str(exc_info.value)
        assert "Duplicate" in error_str or "duplicate" in error_str
        assert "CD31" in error_str


class TestCoabundanceConfigValidation:
    """Test coabundance feature selection enforcement."""

    def test_valid_coabundance_config(self):
        """Test valid coabundance configuration."""
        config = CoabundanceConfig(
            use_feature_selection=True,
            target_n_features=30,
            selection_method="lasso"
        )

        assert config.use_feature_selection is True
        assert config.target_n_features == 30
        assert config.selection_method == "lasso"

    def test_target_features_too_high_rejected(self):
        """Reject unreasonably high feature counts (overfitting risk)."""
        with pytest.raises(ValidationError) as exc_info:
            CoabundanceConfig(
                use_feature_selection=True,
                target_n_features=60,  # Too high!
                selection_method="lasso"
            )

        error_str = str(exc_info.value)
        assert "60" in error_str
        assert "high" in error_str.lower()

    def test_invalid_selection_method_rejected(self):
        """Reject invalid feature selection methods."""
        with pytest.raises(ValidationError) as exc_info:
            CoabundanceConfig(
                use_feature_selection=True,
                target_n_features=30,
                selection_method="invalid_method"
            )

        error_str = str(exc_info.value)
        # Should mention pattern validation
        assert "pattern" in error_str.lower() or "lasso" in error_str.lower()


class TestClusteringConfigValidation:
    """Test clustering parameter validation."""

    def test_valid_clustering_config(self):
        """Test valid clustering configuration."""
        config = ClusteringConfig(
            method="leiden",
            k_neighbors=15,
            spatial_weight=0.5,
            random_state=42,
            use_coabundance_features=True,
            coabundance_options=CoabundanceConfig(
                use_feature_selection=True,
                target_n_features=30,
                selection_method="lasso"
            )
        )

        assert config.method == "leiden"
        assert config.k_neighbors == 15
        assert config.spatial_weight == 0.5

    def test_coabundance_requires_feature_selection(self):
        """
        CRITICAL: Coabundance without feature selection = overfitting disaster.

        9 proteins → 153 coabundance features without selection!
        """
        with pytest.raises(ValidationError) as exc_info:
            ClusteringConfig(
                method="leiden",
                k_neighbors=15,
                spatial_weight=0.5,
                random_state=42,
                use_coabundance_features=True,
                coabundance_options=CoabundanceConfig(
                    use_feature_selection=False,  # CRITICAL ERROR!
                    target_n_features=30,
                    selection_method="lasso"
                )
            )

        error_str = str(exc_info.value)
        assert "CRITICAL" in error_str
        assert "use_feature_selection must be True" in error_str

    def test_spatial_weight_zero_rejected(self):
        """Reject spatial_weight=0.0 (ignores spatial information)."""
        with pytest.raises(ValidationError) as exc_info:
            ClusteringConfig(
                method="leiden",
                k_neighbors=15,
                spatial_weight=0.0,  # Ignores spatial info!
                random_state=42,
                use_coabundance_features=False
            )

        error_str = str(exc_info.value)
        assert "spatial_weight=0.0" in error_str

    def test_spatial_weight_one_rejected(self):
        """Reject spatial_weight=1.0 (ignores feature information)."""
        with pytest.raises(ValidationError) as exc_info:
            ClusteringConfig(
                method="leiden",
                k_neighbors=15,
                spatial_weight=1.0,  # Ignores features!
                random_state=42,
                use_coabundance_features=False
            )

        error_str = str(exc_info.value)
        assert "spatial_weight=1.0" in error_str

    def test_k_neighbors_out_of_range_rejected(self):
        """Reject k_neighbors outside valid range [5, 30]."""
        with pytest.raises(ValidationError) as exc_info:
            ClusteringConfig(
                method="leiden",
                k_neighbors=3,  # Too low!
                spatial_weight=0.5,
                random_state=42,
                use_coabundance_features=False
            )

        error_str = str(exc_info.value)
        assert "k_neighbors" in error_str or "3" in error_str

    def test_k_neighbors_by_scale_validation(self):
        """Test scale-specific k_neighbors validation."""
        config = ClusteringConfig(
            method="leiden",
            k_neighbors=15,
            k_neighbors_by_scale={
                "10.0": 10,
                "20.0": 15,
                "40.0": 20
            },
            spatial_weight=0.5,
            random_state=42,
            use_coabundance_features=False
        )

        assert config.k_neighbors_by_scale["10.0"] == 10

    def test_invalid_clustering_method_rejected(self):
        """Reject invalid clustering methods."""
        with pytest.raises(ValidationError) as exc_info:
            ClusteringConfig(
                method="invalid_method",  # Not leiden/hdbscan/louvain
                k_neighbors=15,
                spatial_weight=0.5,
                random_state=42,
                use_coabundance_features=False
            )

        error_str = str(exc_info.value)
        assert "pattern" in error_str.lower() or "leiden" in error_str.lower()


class TestProcessingConfigValidation:
    """Test processing parameter validation."""

    def test_valid_processing_config(self):
        """Test valid processing configuration."""
        config = ProcessingConfig(
            background_correction={"method": "pixel_wise"},
            dna_processing={"resolution_um": 1.0},
            arcsinh_transform={"optimization_method": "percentile", "percentile_threshold": 5.0}
        )

        assert config.dna_processing["resolution_um"] == 1.0

    def test_dna_resolution_too_high_rejected(self):
        """Reject unrealistically high resolution values."""
        with pytest.raises(ValidationError) as exc_info:
            ProcessingConfig(
                background_correction={"method": "pixel_wise"},
                dna_processing={"resolution_um": 15.0},  # Way too high!
                arcsinh_transform={"auto_cofactor": True}
            )

        error_str = str(exc_info.value)
        assert "15.0" in error_str or "resolution" in error_str.lower()

    def test_arcsinh_missing_method_rejected(self):
        """Reject arcsinh config without cofactor or optimization method."""
        with pytest.raises(ValidationError) as exc_info:
            ProcessingConfig(
                background_correction={"method": "pixel_wise"},
                dna_processing={"resolution_um": 1.0},
                arcsinh_transform={}  # Missing required field!
            )

        error_str = str(exc_info.value)
        assert "auto_cofactor" in error_str or "optimization_method" in error_str


class TestIMCConfigIntegration:
    """Test full IMCConfig validation with nested models."""

    def test_load_actual_config_file(self):
        """Test loading and validating actual config.json."""
        # This tests that our schema works with real config
        is_valid = validate_config_file("config.json")
        assert is_valid, "config.json should pass validation"

    def test_load_validated_config_returns_model(self):
        """Test load_validated_config returns IMCConfig instance."""
        config = load_validated_config("config.json")

        assert isinstance(config, IMCConfig)
        assert hasattr(config, 'channels')
        assert hasattr(config, 'processing')
        assert hasattr(config, 'analysis')

    def test_invalid_nested_clustering_detected(self):
        """Test that invalid nested clustering config is detected."""
        from pathlib import Path
        import tempfile
        import json

        # Create invalid config with bad clustering params
        invalid_config = {
            "project_name": "test",
            "project_id": "test_001",
            "channels": {
                "protein_channels": ["CD31"],
                "dna_channels": ["DNA1"],
                "background_channel": "Ar80",
                "calibration_channels": [],
                "carrier_gas_channel": "",
                "excluded_channels": []
            },
            "processing": {
                "background_correction": {},
                "dna_processing": {"resolution_um": 1.0},
                "arcsinh_transform": {"auto_cofactor": True}
            },
            "quality_control": {},
            "output": {"results_dir": "results", "save_plots": True},
            "performance": {
                "parallel_processes": 8,
                "memory_limit_gb": 8.0,
                "process_sequentially": False
            },
            "analysis": {
                "clustering": {
                    "method": "leiden",
                    "k_neighbors": 15,
                    "spatial_weight": 0.0,  # INVALID! Must be > 0
                    "random_state": 42,
                    "use_coabundance_features": False
                }
            },
            "segmentation": {}
        }

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f)
            temp_path = f.name

        try:
            # Should fail validation
            with pytest.raises(ValidationError):
                load_validated_config(temp_path)
        finally:
            Path(temp_path).unlink()


class TestValidationErrorMessages:
    """Test that validation errors provide clear, actionable feedback."""

    def test_channel_overlap_error_message_clarity(self):
        """Test that channel overlap error message is clear and actionable."""
        try:
            ChannelConfig(
                protein_channels=["CD31", "Ar80", "CD45"],
                dna_channels=["DNA1"],
                background_channel="Ar80",
                calibration_channels=["Ce140"],
                carrier_gas_channel="Xe131",
                excluded_channels=[]
            )
            assert False, "Should have raised ValidationError"
        except ValidationError as e:
            error_str = str(e)

            # Check error message contains key information
            assert "CRITICAL" in error_str
            assert "Ar80" in error_str
            assert "overlap" in error_str.lower()
            assert "protein" in error_str.lower() or "Protein" in error_str
            assert "technical" in error_str.lower() or "background" in error_str.lower()

            # Check it suggests how to fix
            assert "Remove" in error_str or "remove" in error_str

    def test_coabundance_error_message_explains_why(self):
        """Test that coabundance error explains the overfitting risk."""
        try:
            ClusteringConfig(
                method="leiden",
                k_neighbors=15,
                spatial_weight=0.5,
                random_state=42,
                use_coabundance_features=True,
                coabundance_options=CoabundanceConfig(
                    use_feature_selection=False,
                    target_n_features=30,
                    selection_method="lasso"
                )
            )
            assert False, "Should have raised ValidationError"
        except ValidationError as e:
            error_str = str(e)

            # Check error explains the consequence
            assert "CRITICAL" in error_str
            assert "153" in error_str or "overfitting" in error_str.lower()
            assert "use_feature_selection must be True" in error_str
