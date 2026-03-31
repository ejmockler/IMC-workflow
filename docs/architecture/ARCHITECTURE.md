# IMC Analysis Pipeline - Architecture

## Overview

Production-quality architecture for IMC data analysis: multi-scale spatial analysis, proper ion count statistics, and comprehensive validation. All parameters configurable via `config.json`.

## Core Pipeline Modules

### Ion Count Processing
```
src/analysis/ion_count_processing.py
```
- Arcsinh transformation with per-protein cofactor optimization
- Poisson noise handling
- Spatial aggregation (binning)

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
- Multi-scale consistency analysis (10um, 20um, 40um)
- Hierarchical tissue organization
- Spatial statistics (Moran's I, Ripley's K)
- Boundary metrics and spatial coherence

### Clustering & Features
```
src/analysis/spatial_clustering.py
src/analysis/coabundance_features.py
```
- Leiden clustering with spatial weighting
- Coabundance feature generation (products, ratios, covariances)
- Variance-based feature selection
- Resolution optimization with bootstrap stability

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
src/quality_control/quality_gates.py
src/quality_control/statistical_monitoring.py
```

### Performance & Memory
```
src/analysis/memory_management.py
src/analysis/parallel_processing.py
```

### Batch Effects & Normalization
```
src/analysis/batch_correction.py
```
- Quantile normalization for batch effects

### Data Storage & Provenance
```
src/analysis/data_storage.py
src/analysis/analysis_manifest.py
```
- HDF5/Parquet/JSON backends
- Config versioning (SHA256)
- Dependency tracking

### Validation & Testing
```
src/validation/framework.py
src/validation/scientific_quality.py
src/validation/data_integrity.py
src/validation/pipeline_state.py
```

## Configuration System

```
src/config.py              # Main config class
src/config_schema.py       # Pydantic V2 validation
config.json               # Project configuration
```

## Data Flow

```
Raw IMC Data (.txt)
    |
Ion Count Processing
    | (arcsinh + normalization)
Multi-Scale Segmentation (SLIC at 10um, 20um, 40um)
    |
Feature Extraction (protein expression + coabundance)
    |
Spatial Clustering (Leiden with spatial weight)
    |
Validation & QC
    |
Results Storage (HDF5/Parquet/JSON)
```

## Visualization

```
src/viz_utils/
  plotting.py                    # Stateless plotting functions
  comprehensive_figures.py       # Annotation-driven figures (ternary maps, interface composition, type distributions)
```

Analysis and visualization are decoupled. Notebooks consume analysis outputs.

## Module Categories

### Production Pipeline (Use These)
- `main_pipeline.py`
- `ion_count_processing.py`
- `slic_segmentation.py`
- `multiscale_analysis.py`
- `spatial_clustering.py`
- `coabundance_features.py`
- `batch_correction.py`

### Research/Experimental (Available)
- `graph_clustering.py`
- `grid_segmentation.py`
- `watershed_segmentation.py`
- `clustering_comparison.py`

### Utility/Support (Infrastructure)
- All memory, performance, provenance, and validation modules

## See Also

- `CLAUDE.md` - Development instructions
- `README.md` - Project overview
- `METHODS.md` - Scientific methods
- `docs/DATA_SCHEMA.md` - Result file schema
