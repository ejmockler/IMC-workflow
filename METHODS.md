# Methods - IMC Analysis Pipeline

## Overview
This document provides detailed methodology for the IMC analysis pipeline.

**Two analytical phases.** This project proceeded in two phases sharing the same raw data and segmentation. **Phase 1** (§1-§7, §Differential Abundance, §Spatial Neighborhood Enrichment) is frequentist: boolean-gating cell types on SLIC superpixels, Mann-Whitney U with BH-FDR for differential abundance, permutation-based neighborhood enrichment. **Phase 2** (§Temporal Interface Analysis) is pre-registered and effect-size-first: three endpoint families (interface composition CLR, continuous neighborhood neighbor-minus-self, Sham-reference compartment activation) with per-endpoint Bayesian shrinkage under three priors and no FDR (n=2 per group makes coverage-bearing inference impossible). Phase 1 artifacts remain active as descriptive summaries; Phase 2 is the reviewer-facing analysis for candidate findings. See `analysis_plans/temporal_interfaces_plan.md` for the frozen Phase 2 pre-registration.

**Key Methodological Features:**
1. Multi-scale superpixel analysis with scale-adaptive graph construction
2. Variance-based feature selection for coabundance features (153 → 30 features)
3. Stability-based clustering optimization with bootstrap validation
4. Kidney-specific cell type annotation via boolean gating + continuous multi-label memberships
5. Pre-registered Phase 2 temporal interface analysis with Bayesian-shrunk effect sizes

## Study Design

**Objective:** Develop superpixel-based spatial analysis methods for kidney injury IMC data.

**Sample Size:** n=2 mice per timepoint (Sham, D1, D3, D7) — **pilot study for hypothesis generation**.

**Design Type:** Cross-sectional (different subjects at each timepoint).
- 8 biological replicates total across 4 timepoints
- Multiple ROIs per subject (24 ROIs total, 1 test acquisition excluded; ~3 per mouse)
- Anatomical regions (cortex/medulla) identified from metadata

**Statistical Limitations:** n=2 per timepoint prevents inferential statistics between specific timepoints. All findings are hypothesis-generating, not confirmatory. Validation in adequately powered cohorts (n>=10 per group) required for biological claims.

## Channel Processing and Quality Control

### Channel Classification
IMC data channels classified into functional groups:

1. **Protein Markers** (n=9): CD45, CD11b, Ly6G, CD140a, CD140b, CD31, CD34, CD206, CD44
2. **DNA Channels** (n=2): DNA1(Ir191Di), DNA2(Ir193Di) — used for SLIC segmentation only
3. **Background Channel**: 190BCKG — pixel-wise background subtraction
4. **Calibration Channels** (n=2): 130Ba, 131Xe — instrument stability monitoring
5. **Carrier Gas Channel**: 80ArAr — plasma stability monitoring

## Panel Design and Biological Coverage

### Marker Panel
Nine protein markers selected to capture the principal cellular axes of acute kidney injury:

| Marker | Gene | Biological Axis |
|--------|------|----------------|
| CD45 | PTPRC | Pan-leukocyte (immune infiltration) |
| CD11b | ITGAM | Myeloid cells (neutrophils, macrophages) |
| Ly6G | — | Murine neutrophils (no human ortholog in INDRA) |
| CD140a | PDGFRA | Fibroblasts, mesenchymal cells |
| CD140b | PDGFRB | Pericytes, vascular mural cells |
| CD31 | PECAM1 | Endothelial cells |
| CD34 | CD34 | Endothelial progenitors, hematopoietic stem cells |
| CD206 | MRC1 | M2/alternatively activated macrophages |
| CD44 | CD44 | Tissue injury, hyaluronan receptor |

### Knowledge-Grounded Panel Justification
Panel biological coverage was assessed against the INDRA/CoGEx knowledge graph (queried 2026-02-26). The 8 groundable markers (Ly6G excluded as murine-specific) encode genes with 117 known intra-panel relationships (175 raw edges aggregated by source/type/target), spanning immune infiltration (PTPRC, ITGAM), tissue injury/adhesion (CD44), vascular integrity (PECAM1, CD34), stromal/fibrotic response (PDGFRA, PDGFRB), and anti-inflammatory resolution (MRC1). Five of 8 genes are regulated by TGF-beta, the master regulator of renal fibrosis; 4/8 by VEGF. The panel was not designed for pathway enrichment analysis (n=8 genes precludes meaningful ORA).

CD44 is the only panel gene with a direct AKI disease association in INDRA (MESH:D058186). CD34 and PECAM1 share the kidney-specific GO process "glomerular endothelium development" (GO:0072011). PDGFRA and PDGFRB both participate in the nephrogenesis pathway (WP4823). All 8 grounded genes are expressed in metanephros cortex (UBERON:0010533). See `results/biological_analysis/indra_panel_context.json` for the full INDRA knowledge base.

## Data Processing Pipeline

### 1. Background Correction
For each protein channel p and pixel i:
```
Corrected_p,i = max(0, Raw_p,i - Background_i)
```

### 2. Arcsinh Transformation
```
Transformed = arcsinh(Ion_counts / cofactor)
```
- Protein markers: cofactor = 5th percentile of positive counts per marker
- DNA channels: cofactor = 10th percentile × 3 multiplier (conservative, preserves gradients)

### 3. Standardization
```
Standardized = (Transformed - mean) / std
```
Applied per-channel across all pixels.

## SLIC Superpixel Segmentation

Morphology-aware segmentation using DNA channels:

1. **DNA Composite**: Average of DNA1 and DNA2 channels with arcsinh transform
2. **SLIC Parameters** (from config.json):
   - Compactness: 10.0
   - Sigma: 1.5
   - Segment count: tissue_area / target_size^2 (scale-dependent; tissue area from eroded mask, not total image area)
   - Scales: 10um, 20um, 40um
3. **Aggregation**: Mean protein expression within each superpixel

Note: No Gaussian pre-smoothing is applied — the SLIC sigma parameter controls internal smoothing.

## Multi-Scale Analysis

### Scale Selection (kidney anatomy)
- **10um**: Peritubular capillary diameter — cellular-level features
- **20um**: Tubular cross-section — local microenvironments
- **40um**: Glomerular diameter — tissue domain organization

### Inter-Scale Consistency
Low inter-scale ARI (0.01-0.06) is expected and scientifically meaningful — each scale captures distinct organizational features.

## Coabundance Feature Engineering

### Feature Generation
For P=9 protein markers:

1. **Original features** (9): Raw protein expression per superpixel
2. **Pairwise products** (36): x_i * x_j — identifies co-expressing regions
   - Normalized by geometric mean of per-marker RMS: product / sqrt(RMS(x_i)² × RMS(x_j)²)
3. **Pairwise ratios** (72): log1p((x_i + eps) / (x_j + eps)) — relative abundances
   - Log-transformed for symmetry; eps = 25th percentile of positive values
   - Inf/-inf values replaced with ±10 (finite values uncapped)
4. **Spatial covariances** (36): Local neighborhood co-expression (r=20um)

**Total: 153 features**

### Feature Selection
Variance-based selection: retain the 30 features with highest variance across superpixels. This avoids the circularity of using PCA-derived targets as LASSO regression targets.

**Result:** 153 → 30 features (approximately sqrt(N) for N~1000 superpixels)

## Spatial Clustering

### Leiden Community Detection
Leiden algorithm (Traag et al. 2019) with resolution parameter gamma.

### Feature + Spatial Integration
Combined feature matrix:
```
F_combined = [F_protein * (1 - w_s), C_spatial * w_s]
```
where:
- F_protein = selected features (30 dimensions)
- C_spatial = spatial coordinates (2 dimensions), standardized
- w_s = spatial weight (default 0.3 from config)

Both feature and spatial components are scaled by the weight parameter. Setting w_s=0 produces purely expression-based clustering.

### kNN Graph Construction
kNN graphs are built with distance-mode edges, then converted to similarity weights via `w = 1/(1+d)` before Leiden partitioning. This ensures that closer neighbors receive stronger edge weights in the modularity optimization.

Scale-adaptive k neighbors:
- 10um (~2400 superpixels, median=2434): k=14
- 20um (~580 superpixels, median=584): k=12
- 40um (~120 superpixels, median=123): k=10

### Resolution Selection via Stability Analysis
Resolution selected via bootstrap stability:

1. For each candidate resolution:
   - Generate B=20 bootstrap samples; 85% subsampling without replacement
   - Cluster each sample at that resolution
   - Compute pairwise ARI between all bootstrap clusterings on the intersection of common points (each bootstrap pair shares ~72% of points at 85% subsampling)

2. Stability score: mean pairwise ARI across bootstrap pairs (compared only on common points)

3. Select resolution with maximum stability (target: S >= 0.30)

**Current status:** Near-zero stability scores observed across all scales and resolutions. This is reported transparently; downstream analyses that depend on cluster assignments carry this uncertainty.

> **Stability estimation.** Graph caching is enabled (`use_graph_caching=true`): a single kNN graph is built on the full dataset and subsampled per bootstrap iteration via vertex deletion. This is faster but introduces bias: subsampled points lose neighbors (especially near subsample boundaries), producing degraded graph topology that may deflate stability scores via cluster fragmentation. Alternatively, the degraded topology could produce optimistic estimates via fewer, larger communities. The direction of bias is indeterminate. With near-zero stability observed across all scales and resolutions, the caching bias is negligible relative to the fundamental instability. The parameter can be set to `false` to rebuild the kNN graph from scratch per iteration at the cost of O(n_bootstrap × n²) graph construction.

> **Feature space mismatch.** Stability analysis optimizes the resolution parameter using the original 9 protein features, but final clustering operates on 30 coabundance-augmented features (9 original + 21 pairwise coabundance). Re-running resolution selection on the augmented feature space would require O(n_resolutions × n_bootstrap) coabundance computations per scale, which is impractical. The selected resolution may not be optimal for the augmented space; this is an acknowledged architectural trade-off.

## Cell Type Annotation

### Method
Boolean gating on arcsinh-transformed superpixel-level expression (10um scale):
- Default positivity threshold: 60th percentile per marker, computed **per-ROI** (each ROI's own superpixel distribution determines the threshold). This normalizes for inter-ROI technical variation (acquisition conditions, signal degradation) but partially suppresses genuine inter-ROI expression differences.
- Per-marker overrides: CD206 at 50th, Ly6G at 70th percentile
- When a superpixel matches multiple cell type gates, the first-defined cell type in config wins (priority-order assignment). Cell type ordering in config.json determines resolution of ambiguous assignments.

### Assignment Rate
Mean assignment rate: **22.4%** (range 14.8-31.8% across 24 ROIs under the 15-type ontology; see `RESULTS.md` for per-ROI distribution). The remaining ~77% of superpixels are unassigned due to the 9-marker panel lacking sufficient markers for complete tissue annotation. Spatial analyses operate only on the identifiable fraction.

> **Proportions denominator.** Cell type proportions in differential abundance analysis use the total number of superpixels (including unassigned) as denominator. This means changes in the unassigned fraction across timepoints will shift all assigned cell type proportions without a true change in absolute abundance. An alternative denominator (assigned superpixels only) would be equally valid but measures relative composition within the identified fraction rather than tissue-wide prevalence. The total-superpixel denominator is reported; users should interpret proportion changes in the context of the ~79% unassigned tissue.

## Statistical Analysis

### Differential Abundance
- **Unit of analysis**: Mouse (biological replicate), not ROI
- ROI-level proportions averaged within each mouse before testing (temporal and regional). Each ROI contributes equally regardless of superpixel count (unweighted mean). Current ROI sizes are roughly balanced (~2400 superpixels each); if ROI sizes diverge substantially, size-weighted aggregation should be considered.
- Mann-Whitney U test on mouse-level means (n=2 per group)
- Effect sizes: Hedges' g (small-sample corrected Cohen's d) with percentile bootstrap ranges (10,000 iterations; see degeneracy disclaimer below — these are bounds on observed values, not coverage-bearing CIs)
- Multiple testing: Benjamini-Hochberg FDR across all pairwise comparisons
- **Compositional awareness**: Centered log-ratio (CLR) transformed effect sizes reported alongside raw proportions. CLR(x_i) = log(x_i / geometric_mean(x)), with pseudocount 1e-6 for zero proportions. The CLR transform includes the unassigned fraction (~79%) as a component, addressing spurious negative correlations from the shared denominator. Raw Hedges' g (unaffected by pseudocount choice) is the primary effect size; CLR-adjusted `hedges_g_clr` is supplementary. The pseudocount magnitude influences CLR values for rare cell types; sensitivity to this choice was not formally assessed in this pilot.

With n=2 per group, most comparisons are expected to be non-significant. Effect sizes with bootstrap ranges crossing zero are reported honestly.

> **Mann-Whitney U at n=2.** With two observations per group, the Mann-Whitney U test can only produce three possible p-values (approximately 0.33, 0.67, 1.0 for two-sided tests). No comparison can reach conventional significance (p < 0.05) regardless of effect magnitude. The test is retained for completeness and forward compatibility with larger cohorts; Hedges' g effect sizes and bootstrap ranges are the primary inferential quantities.

> **Bootstrap range degeneracy at n=2.** With 2 mouse-level means per group and bootstrap resampling with replacement, only 4 unique bootstrap samples exist per group ({a,a}, {a,b}, {b,a}, {b,b}), yielding 3 unique group means. This produces a maximum of 9 unique Hedges' g values per comparison. The resulting ranges reflect the discrete sample space rather than smooth sampling distributions and should be interpreted as approximate bounds on observed effect magnitude, not coverage-bearing interval estimates.

> **Regional comparisons are paired.** Cortex and medulla from the same mouse are paired observations. The current implementation uses Mann-Whitney U, which treats them as independent — between-mouse variance is conflated with between-region variance. A Wilcoxon signed-rank test would be more appropriate for paired data, but with n=2 pairs, neither test can reach significance. Effect sizes from regional comparisons should be interpreted with this caveat.

### Spatial Neighborhood Enrichment
- k-nearest neighbors (k=10) neighborhood composition per superpixel
- Permutation test (n=1000): global shuffle of cell type labels within each ROI, compare observed vs null neighbor proportions. P-values computed with Phipson & Smyth (2010) pseudocount: p = (n_extreme + 1) / (n_permutations + 1). Deterministic seeding per ROI × cell-type pair for reproducibility.
- BH FDR correction within each ROI across all focal × neighbor pairs
- Aggregation: weighted mean enrichment across ROIs (weights = n_focal_cells per ROI). Note: weighted_mean(observed/expected) ≠ weighted_sum(observed)/weighted_sum(expected) by Jensen's inequality. The per-ROI enrichment ratio is the standard estimand (consistent with histoCAT/squidpy); pooled-ratio aggregation would measure a different quantity.

> **Marker sharing and self-enrichment.** Several cell type definitions share positive markers (e.g., activated_endothelial_cd44 and activated_immune_cd44 both require CD44+; m2_macrophage and activated_immune share CD11b+). Because boolean gating assigns positivity based on continuous expression thresholds, cell types sharing markers will co-localize in marker expression space and consequently in physical space, producing self-enrichment that reflects shared gating criteria rather than independent biological co-localization. Self-enrichment scores (diagonal of the enrichment matrix) should be interpreted with this confound in mind.

> **Spatial enrichment null model.** Neighborhood enrichment significance is assessed via global permutation of cell-type labels within each ROI. This null model assumes spatial homogeneity and does not preserve regional gradients (e.g., cortico-medullary axis). For tissues with strong spatial structure, enrichment p-values may be anti-conservative. Regional stratification or toroidal shift permutation would provide a more appropriate null but were not implemented in this pilot study.

**Spatial weight ablation**: In prior runs, enrichment scores were identical (Pearson r=1.000) between spatial_weight=0.3 (default, X/Y coordinates appended to feature matrix) and spatial_weight=0 (coordinates omitted). Note: the co-abundance feature matrix already contains 36 spatial covariance features computed from 20μm KDTree neighborhoods (see Coabundance Feature Engineering above), so this ablation specifically tests the contribution of global position coordinates (2 dimensions) beyond the local spatial covariance structure (36 dimensions) already present in the 153-feature matrix. The result confirms that global tissue position does not influence enrichment patterns beyond local co-expression structure captured by spatial covariance features and boolean gating.

> **Spatial information encoding.** Local spatial structure enters the analysis through two channels: (1) 36 spatial covariance features computed via radius-based neighborhoods in the coabundance feature set, and (2) the `spatial_weight` parameter that blends raw spatial coordinates into the clustering feature matrix via kNN graphs. These encode related but non-identical spatial information (radius neighborhoods vs. k-nearest-neighbor graphs). The `spatial_weight=0` ablation removes channel (2) but not channel (1), so the ablation tests the marginal contribution of coordinate-based weighting beyond what is already captured by spatial covariance features.

### Spatial Autocorrelation
- **Marker-level**: Spearman correlation of marker expression with spatial coordinates
- **Cluster-level**: Moran's I spatial coherence computed per ROI per scale as a heuristic quality metric for cluster spatial compactness (inverse-distance weights, cKDTree neighborhood). Higher values indicate spatially contiguous clusters. Note: Moran's I treats cluster labels as numeric, which is technically inappropriate for nominal (categorical) data; used here for quality assessment only, not primary analysis. For categorical spatial analysis in Phase 2 (temporal interface analysis), join-count statistics replace Moran's I — implemented in `src/analysis/temporal_interface_analysis.py` (`compute_join_count_bb`), with `compute_morans_i_continuous` applied only to continuous lineage scores. See `analysis_plans/temporal_interfaces_plan.md` §6 for the rationale.

## Temporal Interface Analysis (Phase 2)

Pre-registered, effect-size-first analysis consuming Phase 1 continuous lineage memberships at 10 μm scale. Plan frozen 2026-04-17 (`analysis_plans/temporal_interfaces_plan.md`); subsequent changes require amendment-block entries in that file. Orchestrator: `run_temporal_interface_analysis.py`. Pure-function module: `src/analysis/temporal_interface_analysis.py`.

### Unit of analysis and sample size
- Per-ROI quantities aggregated to per-mouse means; group size n=2 mice/timepoint at Sham/D1/D3/D7.
- Mouse-level aggregation defeats superpixel-level pseudoreplication; no FDR at n=2 because formal p-values are mathematically out of reach. Family-arbitrage audit is by `|hedges_g|` rank-order in `endpoint_summary.csv`.

### Endpoint families (pre-registered)
- **Family A — Interface composition (compositional).** Mouse-level fraction of superpixels in each of 8 interface categories {`immune`, `endothelial`, `stromal`, `endothelial+immune`, `immune+stromal`, `endothelial+stromal`, `endothelial+immune+stromal`, `none`} (these are the exact column names in `interface_fractions.parquet`; "triple" is human shorthand for the 3-lineage `endothelial+immune+stromal` category). CLR transform with **Bayesian-multiplicative zero replacement** (Martín-Fernández et al. 2003, Dirichlet α=0.5) rather than additive ε. Minimum-prevalence filter (<1% in all timepoints → collapsed to "other_rare"). Two normalization paths reported side-by-side (Phase 1 replaced the earlier per-ROI sigmoid with a Sham-reference-centered sigmoid; both normalization paths are now Sham-anchored): (i) **Sham-reference sigmoid on continuous lineage memberships** (primary, normalization_mode=`sham_reference_v2_continuous`); (ii) **raw-marker Sham-reference percentile** (corroboration, normalization_mode=`sham_reference_raw_marker_per_mouse`). Sensitivity sweeps: lineage threshold {0.2, 0.3, 0.4}; raw-marker Sham percentile {65, 75, 85}; continuous Sham percentile {50, 60, 70} (Phase 1.5b — closed). Disagreement reported per row via `normalization_sign_reverse` AND symmetric `normalization_magnitude_disagree` (≥2× divergence). `clr_none_sensitivity` flags rows whose sign flips when the `none` category is excluded (Phase 1.5a — closed; 0/48 flips).
- **Family B — Continuous neighborhood lineage shifts.** Per (composite_label × neighbor_lineage), mouse-level mean of `neighbor_lineage_X − self_lineage_X` (neighbor-minus-self delta). Minimum support filter `n_superpixels ≥ 20` per row; filtered rows preserved as NaN-with-flag distinguishing `absent_biology` from `below_min_support`. Only temporal changes in the delta are interpretable (self-delta is biased by definition). Sensitivity sweep: `{10, 20, 40}` minimum support; `support_sensitive=True` flag on rows whose presence changes across the sweep (Phase 1.5a — closed; 90/270 flagged). Phase 1.5c added a sigmoid-independent **raw-marker basis** (`audit_family_b_raw_markers.py`) using config-defined raw-arcsinh composites for each lineage; Phase 5.2 amends the rule for follow-up cohorts to **co-primary intersection-conservative** reporting (both bases emit endpoints; the intersection is the conservative headline set; sigmoid-only and raw-only sets are reported but flagged as basis-dependent). Pilot result: 21 sigmoid + 18 raw-marker Sham→D7 endpoints clear `|g_shrunk_neutral|>0.5`; 14 in common; 96% sign agreement on overlapping endpoints.
- **Family C — Cross-compartment activation.** Mouse-level CD44⁺ rate within {CD45⁺, CD31⁺, CD140b⁺, background} compartments and triple-overlap fraction, where compartment positivity uses **Sham-reference** 75ᵗʰ-percentile thresholds computed once on Sham-only ROIs (prevents outcome contamination from D7-elevated markers driving the threshold). Sensitivity sweep: `{65, 75, 85}` Sham percentile.

### Effect sizes and shrinkage
- **Hedges' g** on mouse-level means (small-sample-corrected Cohen's d).
- **Bayesian shrinkage** of g under three explicit priors on the true effect δ: skeptical N(0, 0.5²), neutral N(0, 1.0²), optimistic N(0, 2.0²). Posterior mean `E[δ | g_obs] = g_obs × prior_var / (prior_var + sampling_var)`.
- **Sampling variance**: Hedges & Olkin (1985) asymptotic formula `v(g, n) = 2/n + g²/(4n)`. An earlier implementation used a non-textbook `4/n + g²/(2n)` inflation; the Gate 5 amendment switched to the standard form.
- **No default prior.** The three-prior range is a pre-registered sensitivity analysis, not a Bayesian inference. Downstream study designers pick the prior matching their own scepticism.
- **Bootstrap range**: percentile bootstrap over 10,000 iterations reported as min/max of the ~9 unique resamples at n=2 per group; not a coverage-bearing CI.
- **Pathology flag**: `g_pathological = (|g| > 3 AND pooled_std < 0.01)` quarantines variance-collapse artifacts; pathological rows emit NaN for all shrunk-g and n_required columns.

### Spatial coherence for interface labels
- **Join-count statistics** (per binary interface indicator, e.g. `is_triple`, `is_immune+endothelial`) replace Moran's I on categorical labels. Adjacency: k=10 nearest neighbors in (x, y); null: label permutation within ROI, 1000 permutations; statistic: observed BB joins standardized by permutation mean/std. Filter: ≥10 positive superpixels per ROI.
- **Moran's I** used only on *continuous* lineage scores (immune, endothelial, stromal).

### Outputs (`results/biological_analysis/temporal_interfaces/`)
**22 parquet files** (see `docs/DATA_SCHEMA.md`) + `endpoint_summary.csv` (primary reviewer-facing table, **1134 rows × 46 columns** post-Phase-7 across Families A/B/C v1+v2 and all 6 pairwise contrasts) + `continuous_sham_pct_sweep.csv` + `family_b_raw_marker_comparison.csv` + `run_provenance.json` (git commit, config hash, file sha256, seeds, parameters, continuous-Sham-reference artifact path + SHA + percentile + aggregation).

### Phase 5 closures (deferred-item resolution, 2026-04-23)
- **5.1 Tissue-mask area-based density** — closed empirically (`audit_tissue_mask_density.py` → `results/biological_analysis/tissue_area_audit.csv`). Both pre-registered gates fail: CV(tissue_area_mm2) = 0.012, Pearson |r|(density, proportion) on dominant cell type = 0.97. Algebraic reduction: density ≈ 9857 × proportion ± 1.4% because every ROI is acquired at the same ~500×500 µm field-of-view. Closure scope: area-based denominators on this acquisition design only. Untested alternatives flagged separately: per-nucleus density (`watershed_segmentation.py` not wired), DNA-intensity integral, variable-extent re-acquisition cohorts.
- **5.2 Family B lineage-source basis amendment** — co-primary intersection-conservative rule, panel-portable via config-driven raw-marker mapping (see Family B above).
- **5.3 Bodenmiller** — formally closed-by-design (not "tabled"). Permanent scope boundary; framework requires temporal sampling that does not exist in the Bodenmiller Patient1 dataset.
- **5.5 Phase 1.5c factual correction** — earlier draft claimed "0 sigmoid Family B Sham→D7 headlines"; actual is 21 (audit script printed only the raw count). Audit script now prints both counts at run time.
- **5.6 Freeze guard** — `verify_frozen_prereg.py` recomputes pinned SHAs in `review_packet/FROZEN_PREREG.md`; fails with non-zero exit on drift.

### What is explicitly out of scope
- Object-level lineage tracing (not possible with snapshot IMC).
- Significance claims or BH-FDR (n=2 precludes coverage-bearing inference).
- Extrapolation beyond UUO kidney injury on this 9-marker panel (panel-dependent).

## Quality Control

### Thresholds
- Total Ion Count: min 10th percentile, max 20% low-TIC pixels
- DNA Signal: min_dna_signal=1.0, min_tissue_coverage=10%
- Signal-to-Noise: 95th/5th percentile ratio >= 1.5 (practical_pipeline.py)
- Calibration drift: max 5% drift, max CV 0.3 across ROIs

All thresholds derived from pilot data and instrument specifications. Validation in larger cohorts required.

## Batch Correction

### Sham-Anchored Normalization
When multiple acquisition batches detected:
1. Pool sham control data as reference
2. Compute reference statistics
3. Z-score normalize all batches relative to sham reference

## Computational Implementation

- **Graph Caching**: Enabled (`use_graph_caching=true`); kNN graph built once on full dataset, subsampled per bootstrap iteration. Faster but bias direction is indeterminate (see Resolution Selection section) — moot given near-zero stability baseline
- **Random Seeds**: Fixed (random_state=42) for reproducibility
- **Configuration**: All parameters in config.json; Config class is single source of truth
- **Scale-adaptive parameters**: Spatial weight and resolution range use scale-dependent defaults (fine scales: w=0.2, range=[0.5, 2.0]; coarse scales: w=0.4, range=[0.2, 1.0]). The scalar `spatial_weight` and list `resolution_range` in config.json are overridden by these scale-adaptive heuristics
- **Provenance**: Software versions and parameter snapshots tracked per run

## Limitations

1. **Sample size (n=2 per group)**: Insufficient for frequentist hypothesis testing. Zero FDR-significant differential abundance findings. Effect sizes (Hedges' g) reported with wide CIs that typically cross zero. Pilot power analysis indicates most observed effects require n>=20 per group for 80% power. All findings are hypothesis-generating; effect sizes are provided for follow-up study design, not for confirmatory claims.
2. **Marker panel (n=9)**: Limited to coarse lineage identification. ~79% of tissue is unassigned (lacks marker combination for annotation). Cannot identify specific T cell subsets, B cells, dendritic cells, or full macrophage polarization spectrum. The panel precludes pathway enrichment analysis (n=8 groundable genes).
3. **Clustering stability**: Near-zero bootstrap ARI stability at all scales and resolutions. Cluster assignments should be interpreted cautiously; downstream analyses carry this uncertainty.
4. **Cross-sectional design**: No longitudinal tracking. Temporal patterns are inferred from different subjects at each timepoint.
5. **INDRA knowledge context**: The INDRA/CoGEx-derived biological context (shared regulators, mediated paths, mechanistic narratives) contextualizes spatial findings against known biology. It does not validate that specific spatial patterns are statistically real — it provides Discussion-level interpretation, not Results-level evidence. Note: for cell types defined by marker co-expression (e.g., neutrophil = CD45+/Ly6G+), INDRA relationships between those same markers (e.g., ITGAM-Ly6G Complex) trivially "explain" self-enrichment — this circularity should be considered when interpreting INDRA context for self-enriching cell type pairs.

## Software and Dependencies

- Python 3.12+
- NumPy, Pandas, Scikit-learn, Scikit-image, SciPy
- statsmodels (multiple testing correction)
- leidenalg (community detection)

All parameters externalized to `config.json`. See `docs/DATA_SCHEMA.md` for output file formats.
