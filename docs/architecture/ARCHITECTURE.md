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

### Sham-reference normalization primitive (Phase 1)
```
src/analysis/sham_reference.py               # shared threshold + scale primitive (per-mouse aggregation default)
generate_sham_reference.py                   # orchestrator (root); writes results/biological_analysis/sham_reference_10.0um.json
```
- Replaces three drifting Sham-threshold implementations with a single primitive.
- Hard gates on pilot design at write time (n_sham_mice=2, n_sham_rois=6); all consumers (`batch_annotate_all_rois.py`, `temporal_interface_analysis.py`) validate full provenance (config_sha256, percentile, aggregation, per-marker threshold+scale) at load time.
- `batch_annotate_all_rois.py --archive-prior` rolls forward annotations under a new Sham reference (fail-fast default).

### Temporal Interface Analysis (Phase 2; amended through Phase 5)
```
src/analysis/temporal_interface_analysis.py
run_temporal_interface_analysis.py           # orchestrator (root)
analysis_plans/temporal_interfaces_plan.md   # pre-registration + amendment log (Phase 1.5a/b/c, Phase 5)
```
- Pre-registered, three-family endpoint framework:
  - **Family A**: interface-composition CLR with Bayesian-multiplicative zero replacement; Sham-reference sigmoid (primary) + raw-marker Sham-reference percentile (corroboration)
  - **Family B**: continuous neighborhood neighbor-minus-self lineage shifts; Phase 5.2 amendment specifies co-primary intersection-conservative reporting (sigmoid + raw-marker bases)
  - **Family C**: Sham-reference compartment activation trajectories
- Bayesian shrinkage of Hedges' g under three priors (skeptical / neutral / optimistic; neutral is planning default)
- Hedges & Olkin (1985) sampling variance `v = 2/n + g²/(4n)`
- Pathology gate (`g_pathological = |g|>3 AND pooled_std<0.01`) quarantines variance-collapse artifacts
- Per-row sensitivity flags: `support_sensitive` (Family B), `clr_none_sensitivity` (Family A), `normalization_sign_reverse` + `normalization_magnitude_disagree` (Family A)
- Join-count statistics for categorical spatial coherence; continuous Moran's I for lineage scores
- Consumes per-ROI annotation parquets; emits **19 parquets** + `endpoint_summary.csv` (348 rows × 37 cols) + `continuous_sham_pct_sweep.csv` + `family_b_raw_marker_comparison.csv` + `run_provenance.json`
- No FDR at n=2 by construction; family-arbitrage audit by `|g_shrunk_neutral|` rank in `temporal_top_ranked_by_effect.csv`

### Phase 1.5 / 5 sensitivity audits (root-level scripts)
```
sweep_continuous_sham_pct.py                 # Phase 1.5b: Family A continuous Sham-pct sweep {50, 60, 70}
audit_family_b_raw_markers.py                # Phase 1.5c: Family B sigmoid-vs-raw-marker basis comparison; prints both headline counts
audit_tissue_mask_density.py                 # Phase 5.1: empirical-closure audit for area-based density
verify_frozen_prereg.py                      # Phase 5.6: recompute pinned SHAs in review_packet/FROZEN_PREREG.md; fails on drift
```

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
Ion Count Processing (arcsinh + normalization)
    |
Multi-Scale Segmentation (SLIC at 10um, 20um, 40um)
    |
Feature Extraction (protein expression + coabundance)
    |
Spatial Clustering (Leiden with spatial weight)
    |
Validation & QC
    |
Results Storage (HDF5/Parquet/JSON)
    |
    +--> Phase 1 post-processing:
    |       generate_sham_reference.py       (Phase 1: pinned Sham-reference threshold + scale per marker)
    |       batch_annotate_all_rois.py       (boolean gating + continuous Sham-referenced memberships)
    |       differential_abundance_analysis.py
    |       spatial_neighborhood_analysis.py
    |
    +--> Phase 2 pre-registered analysis:
    |       run_temporal_interface_analysis.py  (Family A/B/C, Bayesian shrinkage)
    |
    +--> Phase 1.5 / 5 sensitivity audits:
            sweep_continuous_sham_pct.py     (Phase 1.5b: continuous Sham-pct sweep)
            audit_family_b_raw_markers.py    (Phase 1.5c: sigmoid-vs-raw-marker Family B)
            audit_tissue_mask_density.py     (Phase 5.1: area-based density empirical-closure audit)
            verify_frozen_prereg.py          (Phase 5.6: freeze-manifest SHA verifier)
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
- `cell_type_annotation.py`
- `temporal_interface_analysis.py`  (Phase 2)

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
