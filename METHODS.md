# Methods - IMC Analysis Pipeline

## Overview
This document provides detailed methodology for the IMC analysis pipeline.

**Key Methodological Features:**
1. Multi-scale superpixel analysis with scale-adaptive graph construction
2. Variance-based feature selection for coabundance features (153 → 30 features)
3. Stability-based clustering optimization with bootstrap validation
4. Kidney-specific cell type annotation via boolean gating

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
   - Products are RMS-normalized: product / sqrt(mean(product^2))
3. **Pairwise ratios** (72): log1p((x_i + eps) / (x_j + eps)) — relative abundances
   - Log-transformed for symmetry; eps = 25th percentile
   - Clipped to [-10, +10] to prevent outlier domination
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

> **Stability estimation.** Graph caching is enabled (`use_graph_caching=true`): a single kNN graph is built on the full dataset and subsampled per bootstrap iteration via vertex deletion. This is faster but may produce optimistic stability estimates. With near-zero stability observed across all scales and resolutions, the caching bias is negligible relative to the fundamental instability. The parameter can be set to `false` to rebuild the kNN graph from scratch per iteration at the cost of O(n_bootstrap × n²) graph construction.

## Cell Type Annotation

### Method
Boolean gating on arcsinh-transformed superpixel-level expression (10um scale):
- Default positivity threshold: 60th percentile per marker
- Per-marker overrides: CD206 at 50th, Ly6G at 70th percentile

### Assignment Rate
Mean assignment rate: ~21% (range 10-48% across ROIs). The remaining ~79% of superpixels are unassigned due to the 9-marker panel lacking sufficient markers for complete tissue annotation. Spatial analyses operate only on the identifiable fraction.

> **Proportions denominator.** Cell type proportions in differential abundance analysis use the total number of superpixels (including unassigned) as denominator. This means changes in the unassigned fraction across timepoints will shift all assigned cell type proportions without a true change in absolute abundance. An alternative denominator (assigned superpixels only) would be equally valid but measures relative composition within the identified fraction rather than tissue-wide prevalence. The total-superpixel denominator is reported; users should interpret proportion changes in the context of the ~79% unassigned tissue.

## Statistical Analysis

### Differential Abundance
- **Unit of analysis**: Mouse (biological replicate), not ROI
- ROI-level proportions averaged within each mouse before testing (temporal and regional)
- Mann-Whitney U test on mouse-level means (n=2 per group)
- Effect sizes: Hedges' g (small-sample corrected Cohen's d) with percentile bootstrap 95% CIs (10,000 iterations)
- Multiple testing: Benjamini-Hochberg FDR across all pairwise comparisons
- **Compositional awareness**: Centered log-ratio (CLR) transformed effect sizes reported alongside raw proportions. CLR(x_i) = log(x_i / geometric_mean(x)), with pseudocount 1e-6 for zero proportions. The CLR transform includes the unassigned fraction (~79%) as a component, addressing spurious negative correlations from the shared denominator. Raw Hedges' g (unaffected by pseudocount choice) is the primary effect size; CLR-adjusted `hedges_g_clr` is supplementary. The pseudocount magnitude influences CLR values for rare cell types; sensitivity to this choice was not formally assessed in this pilot.

With n=2 per group, most comparisons are expected to be non-significant. Effect sizes with CIs crossing zero are reported honestly.

> **Mann-Whitney U at n=2.** With two observations per group, the Mann-Whitney U test can only produce three possible p-values (approximately 0.33, 0.67, 1.0 for two-sided tests). No comparison can reach conventional significance (p < 0.05) regardless of effect magnitude. The test is retained for completeness and forward compatibility with larger cohorts; Hedges' g effect sizes and bootstrap CIs are the primary inferential quantities.

> **Bootstrap CI degeneracy at n=2.** With 2 mouse-level means per group and bootstrap resampling with replacement, only 4 unique bootstrap samples exist per group ({a,a}, {a,b}, {b,a}, {b,b}), yielding 3 unique group means. This produces a maximum of 9 unique Hedges' g values per comparison. The resulting CIs reflect the discrete sample space rather than smooth sampling distributions and should be interpreted as approximate bounds on effect magnitude, not precise interval estimates.

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
- **Cluster-level**: Moran's I spatial coherence computed per ROI per scale as a quality metric for cluster spatial compactness (inverse-distance weights, cKDTree neighborhood). Higher values indicate spatially contiguous clusters. Used for quality assessment, not primary analysis.

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

- **Graph Caching**: Enabled (`use_graph_caching=true`); kNN graph built once on full dataset, subsampled per bootstrap iteration. Faster but potentially optimistic — moot given near-zero stability baseline
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
