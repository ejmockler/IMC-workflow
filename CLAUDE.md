# IMC Analysis - Development Guide

## Actual Codebase Architecture

### Directory Structure (Current Implementation)
```
src/
├── analysis/                    # Core analysis modules
│   ├── main_pipeline.py         # Production pipeline orchestrator
│   ├── ion_count_processing.py  # Ion count statistics & transformations
│   ├── multiscale_analysis.py   # Multi-scale spatial analysis
│   ├── slic_segmentation.py     # SLIC superpixel generation
│   ├── batch_correction.py      # Quantile normalization for batches
│   ├── coabundance_features.py  # Protein co-abundance analysis
│   ├── hierarchical_multiscale.py # Hierarchical tissue organization
│   ├── threshold_analysis.py    # Alternative threshold-based methods
│   ├── memory_management.py     # Chunked processing for large data
│   ├── data_storage.py          # HDF5/Parquet/JSON storage backends
│   ├── spatial_stats.py         # Spatial statistics (Moran's I, etc.)
│   ├── spatial_clustering.py    # Clustering optimization
│   ├── quality_control.py       # QC metrics and validation
│   ├── kidney_validation.py     # Kidney-specific validation
│   ├── marker_validation.py     # Marker panel validation
│   ├── error_handling.py        # Error handling utilities
│   └── parallel_processing.py   # Multi-ROI parallel processing
├── quality_control/             # QC framework
│   ├── quality_gates.py         # Quality gate decisions
│   ├── reporting.py             # QC reporting
│   └── statistical_monitoring.py # Statistical QC monitoring
├── validation/                  # Validation framework
│   ├── framework.py             # Main validation framework
│   ├── scientific_quality.py    # Scientific validation
│   ├── data_integrity.py        # Data integrity checks
│   ├── pipeline_state.py        # Pipeline state validation
│   └── core/                    # Core validation components
│       ├── base.py              # Base validation classes
│       └── metrics.py           # Validation metrics
├── utils/                       # Utilities
│   ├── helpers.py               # Metadata classes and utilities
│   ├── column_matching.py       # Column matching utilities
│   ├── results_loader.py        # Results loading utilities
│   └── streamlined_loader.py    # Streamlined data loading
├── viz_utils/                   # Lightweight visualization utilities
│   ├── plotting.py              # Stateless plotting functions
│   ├── journal_figures.py       # Journal-quality figures
│   └── comprehensive_figures.py # Comprehensive figure generation
├── visualization/               # Minimal visualization components
│   └── resolution_explorer.py   # Resolution exploration tools
├── experiments/                 # Experiment-specific modules
│   └── kidney/                  # Kidney experiment components
└── config.py                    # Main configuration class
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