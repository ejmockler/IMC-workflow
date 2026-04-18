# IMC Analysis Pipeline

Multi-scale spatial proteomics framework for Imaging Mass Cytometry data, applied to murine acute kidney injury (AKI). Implements arcsinh-transformed ion count statistics, DNA-guided SLIC superpixel segmentation, Leiden clustering with spatial weighting, and coabundance feature engineering across configurable spatial scales.

**Study scope**: Hypothesis-generating pilot (n=2 mice/timepoint, 9 markers, 24 ROIs, 4 timepoints: Sham/D1/D3/D7). Effect sizes are provided for follow-up study design; no confirmatory statistical claims are made.

## Quick Start

```bash
# 1. Core pipeline (SLIC segmentation + clustering)
.venv/bin/python run_analysis.py

# 2. Cell type annotation
.venv/bin/python batch_annotate_all_rois.py            # Boolean gating annotation

# 3. Biological analysis scripts
.venv/bin/python differential_abundance_analysis.py    # Mouse-level discrete cell-type DA, FDR-corrected
.venv/bin/python spatial_neighborhood_analysis.py      # Permutation-based discrete neighborhood enrichment
.venv/bin/python run_temporal_interface_analysis.py    # Pre-registered temporal interface analysis (Family A/B/C; see analysis_plans/temporal_interfaces_plan.md)

# 4. INDRA knowledge context
.venv/bin/python build_indra_evidence_table.py         # Panel context + finding annotations

# 5. Figures
.venv/bin/python generate_enrichment_heatmaps.py       # Temporal neighborhood heatmaps
.venv/bin/python generate_power_analysis.py             # Effect size forest plot + sample size table
.venv/bin/python run_bodenmiller_benchmark.py           # Steinbock concordance validation
```

## Entry Points

- **Methods**: `METHODS.md` (includes panel justification, ablation results, limitations)
- **Results**: `RESULTS.md` (experiment-specific findings with actual numbers)
- **Result schema**: `docs/DATA_SCHEMA.md`
- **Loading results**: `src/utils/canonical_loader.py`
- **INDRA knowledge base**: `results/biological_analysis/indra_panel_context.json`

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
| `coabundance_features.py` | Product/ratio/covariance features, variance-based selection |
| `batch_correction.py` | Sham-anchored z-score normalization |
| `spatial_stats.py` | Spatial coherence, coordinate correlations |
| `hierarchical_multiscale.py` | Hierarchical tissue organization |
| `cell_type_annotation.py` | Boolean gating + continuous membership annotation |

## Supporting Infrastructure

| Area | Modules |
|------|---------|
| **Storage** | `data_storage.py` (HDF5/Parquet/JSON) |
| **Memory** | `memory_management.py` |
| **Parallel** | `parallel_processing.py` |
| **Provenance** | `analysis_manifest.py` |
| **Validation** | `src/validation/` framework |
| **Visualization** | `src/viz_utils/plotting.py`, `comprehensive_figures.py` (ternary maps, interface composition, type distributions) |

## Output

```
results/
├── roi_results/                    # Per-ROI analysis (24 x JSON.gz)
├── biological_analysis/
│   ├── cell_type_annotations/      # 24 parquet + metadata
│   ├── differential_abundance/     # Temporal + regional CSVs
│   ├── spatial_neighborhoods/      # Enrichment CSVs
│   ├── indra_panel_context.json
│   └── indra_finding_annotations.csv
├── figures/                        # Publication figures
├── power_analysis/                 # Sample size requirements
├── benchmark/                      # Bodenmiller concordance
├── validation_report.json
├── run_summary.json
└── analysis_summary.json
```

Result files are documented in `docs/DATA_SCHEMA.md`.

## Architecture

- `docs/architecture/ARCHITECTURE.md` - Module structure and data flow
- `docs/architecture/WORKFLOW_INTEGRATION.md` - Three-phase workflow (pipeline → biological analysis → visualization)
- `CLAUDE.md` - Development conventions

## Limitations

- **n=2 per group**: Zero FDR-significant findings. Effect sizes reported for follow-up study design.
- **9-marker panel**: ~79% of tissue unassigned. Coarse lineage identification only.
- **Near-zero clustering stability**: Bootstrap ARI near zero at all scales.
- **Cross-sectional design**: Temporal patterns inferred from different subjects.
- See `METHODS.md` for full limitations discussion.
