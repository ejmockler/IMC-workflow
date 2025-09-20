# Production IMC Analysis Pipeline

## Overview

Production-quality analysis framework for Imaging Mass Cytometry (IMC) data, implementing proper ion count statistics, multi-scale analysis, and comprehensive validation. Addresses all technical critiques through scientific rigor and engineering best practices.

## Supported Data Types

- **Protein markers**: Any IMC panel (typically 20-50 markers)
- **DNA markers**: For nuclear identification and morphology
- **Study designs**: Cross-sectional, longitudinal, or case-control
- **Scale**: From small pilot studies to large cohorts

## Key Features

### 1. Proper Ion Count Statistics
- **Arcsinh transformation** with marker-specific cofactor optimization
- **StandardScaler normalization** after transformation
- **Poisson noise handling** throughout pipeline
- **No arbitrary parameters** - all data-driven

### 2. Multi-Scale Spatial Analysis
- **SLIC superpixel segmentation** using DNA channels for morphology-aware binning
- **Multi-scale consistency** analysis (10μm, 20μm, 40μm)
- **Spatial pattern detection** with proper statistical methods
- **Scale-dependent feature identification**

### 3. Robust Clustering Optimization  
- **Systematic parameter selection** using elbow method, silhouette analysis, gap statistic
- **Biological validation scoring** for cluster quality
- **Cross-validation** with bootstrap resampling
- **No hardcoded cluster numbers**

### 4. Production Engineering
- **Configuration-driven** architecture (all parameters in `config.json`)
- **Efficient storage** with HDF5/Parquet (JSON fallback)
- **Memory management** with chunked processing
- **Parallel processing** for ROI-level analysis
- **Comprehensive error handling**

### 5. Enhanced Validation Framework
- **Realistic noise models**: Poisson statistics, spatial artifacts, isotope interference, temporal drift
- **Synthetic data generation** with proper IMC characteristics
- **Performance metrics**: ARI, purity, spatial coherence, boundary preservation
- **Parameter sensitivity analysis**

## Technical Limitations (See TECHNICAL_LIMITATIONS.md)

### Common Constraints in IMC Studies
- **Limited marker panels** - Cell type identification depends on panel comprehensiveness
- **No membrane markers** - Prevents true single-cell segmentation in some datasets
- **Pixel-level analysis** - Not equivalent to single-cell resolution
- **Statistical power** - Depends on study design and sample size

### Spatial Resolution
- **1μm pixel resolution** with ~4μm tissue thickness
- **Z-dimension averaging** across multiple cell layers
- **"Co-localization" = co-abundance in tissue volume**, not direct interaction

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt  # Install dependencies
```

### 2. Run Analysis
```bash
# Full production pipeline
python run_analysis.py --config config.json

# Parallel processing
python run_parallel_analysis.py --config config.json --processes 8

# Single experiment
python run_experiment.py --config config.json --roi-pattern "ROI_D1_*"
```

### 3. Configuration
All parameters are in `config.json`:
```json
{
  "ion_count_processing": {
    "bin_sizes_um": [10.0, 20.0, 40.0],
    "use_slic_segmentation": true,
    "clustering_params": {
      "optimization_method": "comprehensive"
    }
  },
  "multiscale_analysis": {
    "scales_um": [10.0, 20.0, 40.0],
    "consistency_metrics": ["ari", "nmi", "cluster_stability"]
  }
}
```

## Architecture

### Core Pipeline (Analysis-Only Focus)
```
Ion Count Data → Arcsinh Transform → Feature Standardization → 
Clustering Optimization → Multi-Scale Analysis → Validation → Storage
```

**Visualization is handled separately in Jupyter notebooks** - see `VISUALIZATION_GUIDE.md`

### Key Components

#### Analysis Core (`src/analysis/`)
- `main_pipeline.py` - Production pipeline orchestrator
- `ion_count_processing.py` - Ion count statistics and transformations  
- `clustering_optimization.py` - Data-driven parameter selection
- `multiscale_analysis.py` - Multi-scale spatial analysis
- `slic_segmentation.py` - Morphology-aware tissue segmentation
- `validation.py` - Enhanced validation with realistic noise models
- `batch_correction.py` - Quantile normalization for batch effects

#### Storage & Processing (`src/analysis/`)
- `efficient_storage.py` - HDF5/Parquet scalable storage
- `memory_management.py` - Chunked processing for large datasets
- `parallel_processing.py` - Multi-ROI parallel analysis
- `config_management.py` - Configuration system with validation

#### Validation & Metrics (`src/analysis/`)
- `spatial_stats.py` - Spatial statistics (Moran's I, Ripley's K)
- `threshold_analysis.py` - Alternative analysis approaches
- `metrics.py` - Performance and validation metrics

#### Visualization Utilities (`src/viz_utils/`)
- `plotting.py` - Lightweight, stateless plotting functions
- `loaders.py` - Data loading helpers for notebooks

## Output Structure

```
results/
├── roi_results/           # Per-ROI detailed results
├── validation/           # Validation study outputs  
├── analysis_summary.json # Comprehensive summary
└── plots/               # Visualization outputs
```

## Validation Best Practices

**For robust biological conclusions:**
1. **Orthogonal methods**: Validate findings with complementary techniques (Flow cytometry, IHC, RNA-seq)
2. **Adequate sample size**: Power analysis to determine appropriate n
3. **Technical replication**: Multiple ROIs per sample
4. **Biological replication**: Multiple subjects per condition

## Key Improvements Over Previous Systems

1. **Proper Ion Count Statistics** - Addresses Poisson nature of IMC data
2. **Data-Driven Parameters** - No arbitrary hardcoded values
3. **Multi-Scale Consistency** - Validates findings across spatial scales  
4. **Enhanced Validation** - Realistic noise models and comprehensive testing
5. **Production Architecture** - Scalable, configurable, maintainable
6. **Honest Limitations** - Clear documentation of what system can/cannot do

## Files Description

### Production Scripts
- `run_analysis.py` - Main analysis pipeline
- `run_experiment.py` - Single experiment runner
- `run_parallel_analysis.py` - Parallel processing wrapper

### Configuration  
- `config.json` - All analysis parameters (single configuration file)

### Documentation
- `ARCHITECTURE.md` - Clean module structure and data flow
- `TECHNICAL_LIMITATIONS.md` - Detailed technical constraints
- `VISUALIZATION_GUIDE.md` - Guide for creating visualizations in notebooks
- `PROJECT_SUMMARY.md` - Honest assessment of capabilities
- `PUBLICATION_STRATEGY.md` - Path to publication as methods paper
- `CLAUDE.md` - Development guidelines

### Notebooks
- `notebooks/templates/` - Starter notebooks for common analyses
- `notebooks/examples/` - Example analyses for specific experiments

## Contributing

See `CLAUDE.md` for development guidelines. Key principles:
- Configuration-driven design (no hardcoded parameters)
- Proper error handling and validation
- Comprehensive testing and documentation
- Scientific rigor in statistical methods

## Citation

If you use this pipeline, please cite:
- The multi-scale analysis approach for IMC data
- The SLIC-based morphology-aware segmentation method
- Appropriate statistical considerations for your study design