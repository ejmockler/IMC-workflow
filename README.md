# IMC Analysis Pipeline

Multi-scale spatial proteomics framework for Imaging Mass Cytometry data. Implements arcsinh-transformed ion count statistics, DNA-guided SLIC superpixel segmentation, Leiden clustering with spatial weighting, and coabundance feature engineering across configurable spatial scales.

## Entry Points

- **Run the pipeline**: `python run_analysis.py` (reads `config.json`)
- **Biological findings**: `notebooks/biological_narratives/kidney_injury_spatial_analysis.ipynb`
- **Methods**: `METHODS.md`
- **Result schema**: `docs/DATA_SCHEMA.md`
- **Loading results**: `src/utils/canonical_loader.py`

## Pipeline

```
Raw IMC (.txt) → Validation → Arcsinh Transform → SLIC Segmentation (10/20/40μm)
    → Coabundance Features → Leiden Clustering → Spatial Statistics → JSON.gz
```

All parameters live in `config.json`. The `Config` class (`src/config.py`) is the single source of truth.

## Core Modules (`src/analysis/`)

| Module | Role |
|--------|------|
| `main_pipeline.py` | Pipeline orchestrator |
| `ion_count_processing.py` | Arcsinh transformation, cofactor optimization |
| `slic_segmentation.py` | DNA-guided superpixel segmentation |
| `multiscale_analysis.py` | Multi-scale consistency analysis |
| `spatial_clustering.py` | Leiden clustering, resolution optimization, stability |
| `coabundance_features.py` | Product/ratio/covariance features, LASSO selection |
| `batch_correction.py` | Sham-anchored z-score normalization |
| `spatial_stats.py` | Moran's I, spatial coherence |
| `hierarchical_multiscale.py` | Hierarchical tissue organization |
| `quality_control.py` | QC metrics |

## Supporting Infrastructure

| Area | Modules |
|------|---------|
| **Storage** | `data_storage.py` (HDF5/Parquet/JSON) |
| **Memory** | `memory_management.py`, `memory_optimizer.py` |
| **Parallel** | `parallel_processing.py` |
| **Provenance** | `provenance_tracker.py`, `environment_capture.py`, `analysis_manifest.py` |
| **Statistics** | `multiple_testing_control.py`, `fdr_spatial.py`, `spatial_permutation.py`, `mixed_effects_models.py` |
| **Validation** | `src/validation/` framework, `synthetic_data_generator.py` |
| **Visualization** | `src/viz_utils/plotting.py`, `journal_figures.py`, `comprehensive_figures.py` |

## Output

```
results/
├── roi_results/           # Per-ROI results (JSON.gz)
├── validation_report.json # Pre-analysis validation
└── run_summary.json       # Pipeline summary
```

Result files are documented in `docs/DATA_SCHEMA.md`.

## Architecture

- `docs/architecture/ARCHITECTURE.md` - Module structure and data flow
- `docs/architecture/WORKFLOW_INTEGRATION.md` - Three-phase workflow (pipeline → biological analysis → visualization)
- `CLAUDE.md` - Development conventions

## Limitations

- 9-protein panel limits cell type identification
- n=2 per timepoint limits statistical power for pairwise comparisons
- 1μm pixel resolution with ~4μm tissue thickness; co-localization means co-abundance in tissue volume
- Cross-sectional design; findings are hypothesis-generating
