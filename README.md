# IMC Analysis Pipeline

Multi-scale spatial proteomics framework for Imaging Mass Cytometry data, applied to murine acute kidney injury (AKI). Implements arcsinh-transformed ion count statistics, DNA-guided SLIC superpixel segmentation, Leiden clustering with spatial weighting, and coabundance feature engineering across configurable spatial scales.

**Study scope**: Hypothesis-generating pilot (n=2 mice/timepoint, 9 markers, 24 ROIs, 4 timepoints: Sham/D1/D3/D7). Effect sizes are provided for follow-up study design; no confirmatory statistical claims are made.

## Quick Start

```bash
# 1. Core pipeline (SLIC segmentation + clustering)
.venv/bin/python run_analysis.py

# 2. Sham-reference normalization artifact (Phase 1; required by step 3)
.venv/bin/python generate_sham_reference.py            # Pinned Sham-reference threshold + scale per marker; writes results/biological_analysis/sham_reference_10.0um.json

# 3. Cell type annotation (consumes the Sham-reference artifact above)
.venv/bin/python batch_annotate_all_rois.py            # Boolean gating + continuous Sham-referenced memberships (use --archive-prior to roll forward)

# 4. Biological analysis scripts
.venv/bin/python differential_abundance_analysis.py    # Mouse-level cell-type DA + Bayesian-shrunk rank table (temporal_top_ranked_by_effect.csv)
.venv/bin/python spatial_neighborhood_analysis.py      # Permutation-based discrete neighborhood enrichment
.venv/bin/python run_temporal_interface_analysis.py    # Pre-registered temporal interface analysis (Family A/B/C; see analysis_plans/temporal_interfaces_plan.md)

# 5. Phase 1.5 / 5 sensitivity audits (one-off scripts; outputs land in results/biological_analysis/temporal_interfaces/ or results/biological_analysis/)
.venv/bin/python sweep_continuous_sham_pct.py          # Family A continuous Sham-percentile sweep {50, 60, 70}
.venv/bin/python audit_family_b_raw_markers.py         # Family B sigmoid-vs-raw-marker basis comparison (prints both headline counts)
.venv/bin/python audit_tissue_mask_density.py          # Tissue-mask density empirical-closure audit (Phase 5.1)
.venv/bin/python verify_frozen_prereg.py               # Recompute pinned SHAs in review_packet/FROZEN_PREREG.md; fails on drift

# 6. INDRA knowledge context
.venv/bin/python build_indra_evidence_table.py         # Panel context + finding annotations

# 7. Figures
.venv/bin/python generate_enrichment_heatmaps.py       # Temporal neighborhood heatmaps
.venv/bin/python generate_power_analysis.py             # Effect size forest plot + sample size table
```

> **Note on `run_bodenmiller_benchmark.py`**: this script computes channel-level concordance (Spearman r≈0.996) between our raw-pixel loader and Steinbock cell-level means on a single-patient Bodenmiller ROI set. It validates data I/O only, not framework generalization. External validation of the Family A/B/C framework on Bodenmiller is **closed-by-design** (Phase 5.3, not deferred): the dataset is single-patient, single-timepoint, different organ/species/panel, and Family A/B/C requires temporal sampling that does not exist in that dataset. Permanent scope boundary. See `benchmarks/STATUS.archived.md` for archived material.

## Entry Points

- **Methods**: `METHODS.md` (panel justification, statistical methods, limitations)
- **Results**: `RESULTS.md` (experiment-specific findings with actual numbers)
- **Result schema**: `docs/DATA_SCHEMA.md`
- **Config**:
  - `config.json` — analysis knobs (gating, thresholds, endpoints) loaded via `src.config.Config`
  - `viz.json` — display knobs (colors, labels, plot defaults) loaded via `src.viz_utils.VizConfig`
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
│   ├── cell_type_annotations/      # 24 parquet + metadata (12-col schema)
│   ├── differential_abundance/     # Phase 1: temporal + regional CSVs + temporal_top_ranked_by_effect.csv (Bayesian-shrunk rank companion, Phase 2)
│   ├── spatial_neighborhoods/      # Phase 1: cell-type-pair enrichment CSVs
│   ├── temporal_interfaces/        # Phase 2 + Phase 7: 22 parquets + endpoint_summary.csv (1134 rows × 46 cols) + run_provenance.json + Phase 1.5 sensitivity outputs (continuous_sham_pct_sweep.csv + family_b_raw_marker_audit.parquet + family_b_raw_marker_comparison.csv) + Phase 7 v2 outputs (celltype_fractions.parquet + celltype_clr.parquet + celltype_min_prevalence_sweep.parquet)
│   ├── tissue_area_audit.csv       # Phase 5.1: empirical-closure audit for area-based density (CV / Pearson gates + closure scope)
│   ├── sham_reference_10.0um.json  # Phase 1: pinned Sham-reference threshold + scale per marker (consumed by batch_annotate_all_rois.py)
│   ├── indra_panel_context.json
│   └── indra_finding_annotations.csv
├── figures/                        # Publication figures
├── power_analysis/                 # Sample size requirements
├── benchmark/                      # Channel-level Bodenmiller concordance (data I/O only)
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
