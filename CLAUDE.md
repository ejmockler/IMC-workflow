# IMC Analysis - Development Guide

## Actual Codebase Architecture

### Directory Structure (Current Implementation)
```
src/
├── analysis/                    # Core analysis modules (21 modules)
│   ├── main_pipeline.py         # Production pipeline orchestrator
│   ├── multiscale_analysis.py   # Multi-scale spatial analysis
│   ├── ion_count_processing.py  # Ion count statistics & transformations
│   ├── slic_segmentation.py     # SLIC superpixel generation
│   ├── spatial_clustering.py    # Clustering optimization
│   ├── batch_correction.py      # Quantile normalization for batches
│   ├── coabundance_features.py  # Protein co-abundance analysis
│   ├── hierarchical_multiscale.py # Hierarchical tissue organization
│   ├── threshold_analysis.py    # Alternative threshold-based methods
│   ├── spatial_stats.py         # Spatial statistics (Moran's I, etc.)
│   ├── spatial_utils.py         # Spatial helper functions
│   ├── data_storage.py          # HDF5/Parquet/JSON storage backends
│   ├── memory_management.py     # Chunked processing for large data
│   ├── parallel_processing.py   # Multi-ROI parallel processing
│   ├── analysis_manifest.py     # Scientific objectives & manifest
│   ├── cell_type_annotation.py  # Boolean gating cell type assignment
│   ├── clustering_comparison.py # Graph vs spatial clustering comparison
│   ├── graph_clustering.py      # Graph-based clustering baseline
│   ├── grid_segmentation.py     # Grid-based segmentation alternative
│   ├── watershed_segmentation.py # Watershed segmentation alternative
│   └── performance_profiling.py # Timing utilities
├── quality_control/             # QC framework
│   ├── quality_gates.py         # Quality gate decisions
│   ├── config.py                # QC configuration
│   └── statistical_monitoring.py # Statistical QC monitoring
├── validation/                  # Validation framework
│   ├── framework.py             # Main validation framework
│   ├── scientific_quality.py    # Scientific validation
│   ├── data_integrity.py        # Data integrity checks
│   ├── pipeline_state.py        # Pipeline state validation
│   ├── practical_pipeline.py    # Practical validation pipeline
│   ├── kidney_biological_validation.py # Kidney-specific validation
│   └── core/                    # Core validation components
│       ├── base.py              # Base validation classes
│       └── metrics.py           # Validation metrics
├── utils/                       # Utilities
│   ├── paths.py                 # Centralized path management
│   ├── metadata.py              # ROI metadata extraction
│   ├── imc_loader.py            # Raw .txt pixel file loader
│   ├── canonical_loader.py      # Processed gzipped JSON loader
│   └── column_matching.py       # Column matching utilities
├── viz_utils/                   # Lightweight visualization utilities
│   ├── plotting.py              # Stateless plotting functions
│   └── comprehensive_figures.py # Comprehensive figure generation
├── config.py                    # Main configuration class
└── config_schema.py             # Pydantic config validation
```

## Development Principles

### Configuration-Driven Design
- All parameters externalized to `config.json`
- `Config` class provides single source of truth
- No hardcoded values in analysis modules
- Experiment-agnostic through configurable metadata mapping

### Clean Architecture Patterns
1. **Analysis Pipeline**: Produces standardized data products (HDF5/Parquet/JSON)
2. **Visualization Layer**: Jupyter notebooks consume analysis outputs
3. **Separation of Concerns**: Analysis and visualization are completely decoupled
4. **Validation Framework**: Comprehensive testing with synthetic data

### Current Data Flow
1. **Ion Count Processing**: Arcsinh transformation with optimized cofactors
2. **Multi-Scale Analysis**: SLIC superpixels at 10μm, 20μm, 40μm scales
3. **Clustering Optimization**: Data-driven parameter selection
4. **Validation**: Scale consistency, biological validation, QC metrics
5. **Storage**: Efficient HDF5/Parquet with metadata preservation

# Important Instructions
- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing existing files to creating new ones
- NEVER proactively create documentation files