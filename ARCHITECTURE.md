# IMC Analysis Pipeline - Architecture

**Last Updated:** 2025-10-20
**Status:** Production-ready with 100% test pass rate (101/101 tests)

## Overview

Production-quality architecture for IMC data analysis focusing on multi-scale spatial analysis, proper ion count statistics, and comprehensive validation. All parameters configurable via `config.json`.

## Core Pipeline Modules

### Ion Count Processing
```
src/analysis/ion_count_processing.py
```
- Arcsinh transformation with per-protein cofactor optimization
- Poisson noise handling
- Spatial aggregation (binning)
- Background correction

### Multi-Scale Segmentation
```
src/analysis/slic_segmentation.py
src/analysis/threshold_analysis.py
src/analysis/grid_segmentation.py
src/analysis/watershed_segmentation.py
```
- **SLIC**: DNA-based superpixel segmentation (primary method)
- **Threshold**: Alternative threshold-based approach
- **Grid**: Simple grid-based spatial binning
- **Watershed**: Watershed segmentation for comparison

### Spatial Analysis
```
src/analysis/multiscale_analysis.py
src/analysis/hierarchical_multiscale.py
src/analysis/spatial_stats.py
src/analysis/spatial_utils.py
```
- Multi-scale consistency analysis (10μm, 20μm, 40μm)
- Hierarchical tissue organization
- Spatial statistics (Moran's I, Ripley's K)
- Boundary metrics and spatial coherence

### Clustering & Features
```
src/analysis/spatial_clustering.py
src/analysis/coabundance_features.py
```
- Leiden clustering with spatial weighting
- Coabundance feature generation (products, ratios)
- LASSO feature selection
- Resolution optimization

### Pipeline Orchestration
```
src/analysis/main_pipeline.py
```
- ROI-level analysis workflow
- Config snapshotting and provenance tracking
- Dependency version recording
- Output standardization

## Support Infrastructure

### Quality Control
```
src/analysis/quality_control.py
src/analysis/automatic_qc_system.py
src/analysis/marker_validation.py
src/analysis/kidney_validation.py
```
- Automated QC checks
- Marker panel validation
- Biological validation scoring
- Statistical monitoring

### Performance & Memory
```
src/analysis/memory_management.py
src/analysis/memory_optimizer.py
src/analysis/parallel_processing.py
src/analysis/performance_dag.py
src/analysis/performance_optimizer.py
```
- Chunked processing for large data
- Memory-aware optimization
- Parallel ROI processing
- DAG-based computation caching

### Batch Effects & Normalization
```
src/analysis/batch_correction.py
src/analysis/bead_normalization.py
src/analysis/spillover_correction.py
```
- Quantile normalization across batches
- Bead-based calibration
- Spillover compensation

### Data Storage & Provenance
```
src/analysis/data_storage.py
src/analysis/provenance_tracker.py
src/analysis/analysis_manifest.py
src/analysis/environment_capture.py
```
- HDF5/Parquet/JSON backends
- Config versioning (SHA256)
- Dependency tracking
- Reproducibility framework

### Statistical Methods
```
src/analysis/multiple_testing_control.py
src/analysis/fdr_spatial.py
src/analysis/spatial_permutation.py
src/analysis/spatial_resampling.py
src/analysis/mixed_effects_models.py
src/analysis/patient_level_cv.py
src/analysis/uncertainty_propagation.py
```
- FDR correction for spatial data
- Permutation testing
- Bootstrap resampling
- Mixed-effects modeling
- Patient-level cross-validation

### Validation & Testing
```
src/analysis/synthetic_data_generator.py
src/analysis/deviation_workflow.py
src/analysis/ablation_framework.py
src/analysis/complete_system_validation.py
```
- Synthetic IMC data generation
- Deviation detection
- Ablation studies
- System-level validation

### Utilities & Integration
```
src/analysis/error_handling.py
src/analysis/artifact_detection.py
src/analysis/boundary_metrics.py
src/analysis/result_comparison.py
src/analysis/parameter_profiles.py
src/analysis/system_integration.py
```
- Error handling utilities
- Artifact detection
- Cross-method comparison
- Parameter profiling

## Configuration System

```
src/config.py              # Main config class
src/config_schema.py       # Pydantic V2 validation
config.json               # Project configuration
```

### Key Validations (Pydantic)
- **Channel overlap prevention** (CRITICAL): Prevents calibration beads analyzed as proteins
- **Coabundance feature selection enforcement**: Prevents overfitting (9 proteins → 153 features)
- **Parameter range validation**: Ensures physically meaningful parameters

## Data Flow

```
Raw IMC Data (.txt)
    ↓
Ion Count Processing
    ↓ (arcsinh + normalization)
Multi-Scale Segmentation (SLIC at 10μm, 20μm, 40μm)
    ↓
Feature Extraction (protein expression + coabundance)
    ↓
Spatial Clustering (Leiden with spatial weight)
    ↓
Validation & QC
    ↓
Results Storage (HDF5/Parquet/JSON)
```

## Test Infrastructure

```
tests/
├── test_config_provenance.py      # Config versioning (11 tests)
├── test_pydantic_schema.py        # Schema validation (23 tests)
├── test_ion_count_core.py         # Ion count processing (20 tests)
├── test_core_algorithms.py        # Core algorithms (10 tests)
├── test_multiscale_analysis.py    # Multi-scale (11 tests)
├── test_slic_segmentation.py      # SLIC (15 tests)
└── test_spatial_clustering_properties.py  # Clustering (11 tests)
```

**Status:** 101/101 tests passing (100% pass rate)

## Visualization

```
src/viz_utils/
├── plotting.py                    # Stateless plotting functions
├── journal_figures.py             # Publication-quality figures
└── comprehensive_figures.py       # Comprehensive visualization
```

**Note:** Analysis and visualization are decoupled. Notebooks consume analysis outputs.

## Key Design Principles

1. **Configuration-driven**: All parameters in `config.json`
2. **Experiment-agnostic**: Configurable metadata mapping
3. **Proper statistics**: Ion count transformations, Poisson handling
4. **Multi-scale**: Consistent analysis across spatial scales
5. **Validation-first**: Synthetic data + comprehensive testing
6. **Reproducible**: Config snapshots + dependency tracking
7. **Production-ready**: Error handling, memory management, parallel processing

## Module Categories

### Production Pipeline (Use These)
- `main_pipeline.py`
- `ion_count_processing.py`
- `slic_segmentation.py`
- `multiscale_analysis.py`
- `spatial_clustering.py`
- `coabundance_features.py`
- `batch_correction.py`
- `quality_control.py`

### Research/Experimental (Available)
- `ablation_framework.py`
- `segmentation_benchmark.py`
- `graph_clustering.py`
- `mi_imc_integration.py`
- `single_stain_protocols.py`

### Utility/Support (Infrastructure)
- All memory, performance, provenance, and validation modules

## Performance Characteristics

- **Memory**: Chunked processing supports datasets >> RAM
- **Speed**: Parallel ROI processing scales linearly
- **Storage**: Efficient HDF5/Parquet with compression
- **Validation**: Comprehensive test suite (100% pass rate)

## See Also

- `CLAUDE.md` - Development instructions
- `README.md` - Project overview
- `METHODS.md` - Scientific methods
- `TEST_SUITE_STATUS.md` - Test results
- `COMPLETION_STATUS.md` - Infrastructure status
