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
- Multiple ROIs per subject (25 ROIs total, ~3 per mouse)
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
Panel biological coverage was assessed against the INDRA/CoGEx knowledge graph (queried 2026-02-20). The 8 groundable markers (Ly6G excluded as murine-specific) encode genes with 32 known intra-panel causal relationships, spanning immune infiltration (PTPRC, ITGAM), tissue injury/adhesion (CD44), vascular integrity (PECAM1, CD34), stromal/fibrotic response (PDGFRA, PDGFRB), and anti-inflammatory resolution (MRC1). Five of 8 genes are regulated by TGF-beta, the master regulator of renal fibrosis; 4/8 by VEGF. The panel was not designed for pathway enrichment analysis (n=8 genes precludes meaningful ORA).

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
   - Segment count: area / target_size^2 (scale-dependent)
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
- 10um (~2400 superpixels): k=14
- 20um (~600 superpixels): k=12
- 40um (~130 superpixels): k=10

### Resolution Selection via Stability Analysis
Resolution selected via bootstrap stability:

1. For each candidate resolution:
   - Generate B bootstrap samples (default B=5, configurable to 100; 90% subsampling)
   - Cluster each sample at that resolution
   - Compute pairwise ARI between all bootstrap clusterings

2. Stability score: mean pairwise ARI across bootstrap pairs

3. Select resolution with maximum stability (target: S >= 0.6)

**Current status:** Near-zero stability scores observed across all scales and resolutions. This is reported transparently; downstream analyses that depend on cluster assignments carry this uncertainty.

## Cell Type Annotation

### Method
Boolean gating on arcsinh-transformed superpixel-level expression (10um scale):
- Default positivity threshold: 60th percentile per marker
- Per-marker overrides: CD206 at 50th, Ly6G at 70th percentile

### Assignment Rate
Mean assignment rate: ~21% (range 10-48% across ROIs). The remaining ~79% of superpixels are unassigned due to the 9-marker panel lacking sufficient markers for complete tissue annotation. Spatial analyses operate only on the identifiable fraction.

## Statistical Analysis

### Differential Abundance
- **Unit of analysis**: Mouse (biological replicate), not ROI
- ROI-level proportions averaged within each mouse before testing
- Mann-Whitney U test on mouse-level means (n=2 per group)
- Effect sizes: Hedges' g (small-sample corrected Cohen's d) with percentile bootstrap 95% CIs (10,000 iterations)
- Multiple testing: Benjamini-Hochberg FDR across all pairwise comparisons

With n=2 per group, most comparisons are expected to be non-significant. Effect sizes with CIs crossing zero are reported honestly.

### Spatial Neighborhood Enrichment
- k-nearest neighbors (k=10) neighborhood composition per superpixel
- Permutation test (n=500): shuffle cell type labels, compare observed vs null neighbor proportions
- BH FDR correction within each ROI across all focal × neighbor pairs
- Aggregation: fraction of ROIs with FDR-significant enrichment

**Spatial weight ablation**: Enrichment scores are identical (Pearson r=1.000) between spatial_weight=0.3 (default, coordinates contribute to clustering) and spatial_weight=0 (expression only). This confirms that self-clustering and cross-type enrichment patterns reflect marker co-expression via boolean gating, not spatial weighting artifacts in the clustering step.

### Spatial Autocorrelation
Spatial coherence assessed via Spearman correlation of marker expression with spatial coordinates. Moran's I is configured but not currently implemented in the analysis pipeline.

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

- **Graph Caching**: kNN graph computed once per scale, reused across bootstrap iterations
- **Random Seeds**: Fixed (random_state=42) for reproducibility
- **Configuration**: All parameters in config.json; Config class is single source of truth
- **Provenance**: Software versions and parameter snapshots tracked per run

## Limitations

1. **Sample size (n=2 per group)**: Insufficient for frequentist hypothesis testing. Zero FDR-significant differential abundance findings. Effect sizes (Hedges' g) reported with wide CIs that typically cross zero. Pilot power analysis indicates most observed effects require n>=20 per group for 80% power. All findings are hypothesis-generating; effect sizes are provided for follow-up study design, not for confirmatory claims.
2. **Marker panel (n=9)**: Limited to coarse lineage identification. ~79% of tissue is unassigned (lacks marker combination for annotation). Cannot identify specific T cell subsets, B cells, dendritic cells, or full macrophage polarization spectrum. The panel precludes pathway enrichment analysis (n=8 groundable genes).
3. **Clustering stability**: Near-zero bootstrap ARI stability at all scales and resolutions. Cluster assignments should be interpreted cautiously; downstream analyses carry this uncertainty.
4. **Cross-sectional design**: No longitudinal tracking. Temporal patterns are inferred from different subjects at each timepoint.
5. **INDRA knowledge context**: The INDRA/CoGEx-derived biological context (shared regulators, mediated paths, mechanistic narratives) contextualizes spatial findings against known biology. It does not validate that specific spatial patterns are statistically real — it provides Discussion-level interpretation, not Results-level evidence.

## Software and Dependencies

- Python 3.8+
- NumPy, Pandas, Scikit-learn, Scikit-image, SciPy
- statsmodels (multiple testing correction)
- leidenalg (community detection)

All parameters externalized to `config.json`. See `docs/DATA_SCHEMA.md` for output file formats.
