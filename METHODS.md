# Methods - IMC Analysis Pipeline

## Overview
This document provides detailed methodology for the IMC analysis pipeline, suitable for publication methods sections.

**Key Methodological Advances:**
1. Multi-scale superpixel analysis with scale-adaptive graph construction
2. LASSO-based feature selection for coabundance features (153→30 features)
3. Stability-based clustering optimization with bootstrap validation
4. Kidney-specific biological validation framework

## Channel Processing and Quality Control

### Channel Classification
IMC data channels were rigorously classified into distinct functional groups to ensure appropriate processing:

1. **Protein Markers** (n=9): CD45, CD11b, Ly6G, CD140a, CD140b, CD31, CD34, CD206, CD44
   - These channels undergo full analytical pipeline including background correction and normalization
   
2. **DNA Channels** (n=2): DNA1(Ir191Di), DNA2(Ir193Di)  
   - Used exclusively for morphology-aware SLIC superpixel segmentation
   - Not included in protein expression analysis
   
3. **Background Channel**: 190BCKG
   - Used for pixel-wise background subtraction from all protein channels
   - Signal = Protein_raw - Background (negative values clipped to 0)
   
4. **Calibration Channels** (n=2): 130Ba, 131Xe
   - Monitored for instrument stability (CV < 0.2 threshold)
   - Excluded from biological analysis
   
5. **Carrier Gas Channel**: 80ArAr
   - Monitored for plasma stability (minimum signal > 100 counts)
   - Excluded from biological analysis

### Critical Implementation Note
**IMPORTANT**: Calibration and carrier gas channels must be explicitly excluded from protein analysis. Failure to do so results in these technical channels being incorrectly analyzed as biological markers, completely invalidating downstream analysis.

## Data Processing Pipeline

### 1. Data Loading and Channel Filtering
```python
# Correct channel filtering (from config.json)
protein_channels = ["CD45", "CD11b", "Ly6G", "CD140a", "CD140b", 
                    "CD31", "CD34", "CD206", "CD44"]
excluded = ["80ArAr", "130Ba", "131Xe", "190BCKG", 
            "Start_push", "End_push", "Pushes_duration", "Z"]
```

### 2. Background Correction
For each protein channel p and pixel i:
```
Corrected_p,i = max(0, Raw_p,i - Background_i)
```
Where Background_i is the 190BCKG channel intensity at pixel i.

### 3. Arcsinh Transformation
Ion count data undergoes variance-stabilizing transformation with automatic optimization:
```
Transformed = arcsinh(Ion_counts / cofactor)
```
- Cofactor automatically optimized per marker (5th percentile method)
- Adapts to each protein's dynamic range
- Applied after background correction

### 4. Standardization
Post-transformation standardization using StandardScaler:
```
Standardized = (Transformed - μ) / σ
```
Applied per-channel across all pixels.

## Discretization Trade-offs and Information Loss

### Continuous Measurements vs Discrete Classifications
IMC technology measures **continuous protein expression gradients** via ion counting. Our pipeline discretizes these measurements at multiple levels:

1. **Boolean gating**: Continuous marker expression → marker+ vs marker- (binary)
2. **Cluster assignment**: Gradient space → discrete clusters (hard assignment)
3. **Cell type classification**: Multi-dimensional expression → categorical labels

**Methodological Choice**: We prioritize biological interpretability and expert knowledge integration over data-driven optimization, accepting information loss as trade-off.

### Information Loss Quantification
Shannon entropy analysis quantifies information content before and after discretization:

**Continuous expression** (via histogram binning):
```
H_continuous = -Σ p(x_i) log₂ p(x_i)
```
where x_i are histogram bins (n=50) of marker expression

**Discrete gates** (binary):
```
H_discrete = -[p(+) log₂ p(+) + p(-) log₂ p(-)]
```

**Measured information loss**: ~70-80% across all markers (see `notebooks/methods_validation/01_gradient_discretization_analysis.ipynb`)

### Biological Implications

**What is preserved**:
- Major expression patterns (high vs low expressors)
- Cell type identity (discrete categorical labels)
- Boolean logic gates (e.g., CD11b+ AND CD206+ = M2 macrophage)
- Integration of domain expertise (known marker combinations)

**What is lost**:
- Gradient fine structure (expression levels within positive population)
- Continuous transitions (cells near threshold arbitrarily classified)
- Heterogeneity within cell types (all "neutrophils" treated identically)
- Subtle expression differences (compressed by binary threshold)

### Alternative Approaches
We acknowledge that gradient-aware methods exist:
- **Soft assignments**: Probabilistic cell typing (e.g., topic modeling, Gaussian mixtures)
- **Fuzzy clustering**: Membership scores rather than hard labels
- **UMAP/t-SNE**: Direct analysis in continuous embedding space
- **Deep learning**: End-to-end gradient-aware classifiers

**Our rationale for discretization**:
1. Enables Boolean logic for cell type definitions (interpretable)
2. Facilitates integration of biological knowledge (literature-defined markers)
3. Provides categorical outputs suitable for spatial enrichment analysis
4. Matches clinical/biological conventions (marker positive/negative)

### Statistical Power Considerations (n=2 Pilot Study)

**Sample size**: n=2 mice per timepoint severely limits statistical power for hypothesis testing.

**Power analysis** (two-sample t-test, α=0.05):
- For 80% power with n=2: Cohen's d ≥ 3.0 required (extreme effect)
- Medium-large effects (d=0.8-1.5): Power <30% (undetectable)
- Only **very large effects** (d>2.0) have adequate power

**Observed effect sizes** in UUO dataset:
- Median |d| ≈ 1.2 (most comparisons underpowered)
- ~15-25% of comparisons exceed d=2.0 threshold
- Confidence intervals ~5× wider than adequately powered study (n≥20)

**Justified claims** (n=2):
✅ Descriptive findings (patterns observed, trends noted)
✅ Qualitative comparisons (higher/lower, present/absent)
✅ Methods demonstration (pipeline functionality validated)
✅ Hypothesis generation (what to test in powered study)
✅ Very large effects (d>2.0) with explicit uncertainty

**Unjustified claims** (n=2):
❌ Statistical significance (p-values meaningless without power)
❌ Causal inference (no statistical support)
❌ Subtle effects (d<2.0 undetectable)
❌ Population generalization (pilot data, not confirmatory)
❌ Definitive biological conclusions

**Study framing**: This pilot study demonstrates methods capability on real biological data with limited sample size. Findings are hypothesis-generating, not confirmatory. Validation in adequately powered cohorts (n≥10 per group) required for biological claims.

See `notebooks/methods_validation/02_statistical_power_analysis.ipynb` for detailed power calculations.

## Multi-Scale Spatial Analysis

### SLIC Superpixel Segmentation
Morphology-aware segmentation using DNA channels:

1. **DNA Composite Creation**
   - Combine DNA1 and DNA2 channels
   - Apply Gaussian smoothing (σ = 2μm)
   - Normalize to [0, 1] range

2. **Superpixel Generation**
   - Target density: 2500 superpixels/mm²
   - Compactness: 10.0
   - Size scales: 10μm, 20μm, 40μm

3. **Aggregation**
   - Sum ion counts within each superpixel
   - Preserve spatial relationships

## Quality Control Metrics

### Instrument Stability
- **Calibration CV**: Must be <0.2 for 130Ba and 131Xe
- **Carrier Gas**: Median 80ArAr signal must exceed 100 counts
- **Background Levels**: Monitor 190BCKG for drift

### Signal Quality
- **Signal-to-Noise Ratio**: Protein/Background > 3.0
- **Spatial Artifacts**: Edge effect detection via comparative statistics
- **Batch Effects**: ANOVA across acquisition batches

### Data Integrity Checks
1. Verify channel classifications in config.json
2. Confirm exclusion of technical channels from analysis
3. Validate metadata preservation (Cortex/Medulla regions)
4. Monitor QC metrics for each ROI

## Clustering Optimization

### Parameter Selection
Systematic optimization using three metrics:
1. **Elbow Method**: Identify inflection in within-cluster sum of squares
2. **Silhouette Score**: Maximize cluster separation
3. **Gap Statistic**: Compare to null reference distribution

### Validation
- Bootstrap resampling (n=100)
- Scale-specific biological validation
- Visual inspection of segmentation overlays

## Multi-Scale Analysis

### Hierarchical Tissue Organization
Tissue microenvironments exhibit inherent multi-scale organization that requires analysis at different spatial resolutions:

1. **10μm Scale**: Captures cellular and subcellular features
   - Approximates single cell or small cell cluster resolution
   - Reveals local protein expression patterns
   - Identifies cellular neighborhoods

2. **20μm Scale**: Captures local microenvironments
   - Encompasses cell-cell interactions
   - Reveals immune infiltration patterns
   - Identifies transitional zones

3. **40μm Scale**: Captures tissue domain organization
   - Delineates anatomical compartments (cortex/medulla)
   - Reveals large-scale gradients
   - Identifies tissue architectural features

### Critical Note on Inter-Scale Consistency
**IMPORTANT**: Low Adjusted Rand Index (ARI) between scales (0.01-0.06) is EXPECTED and scientifically meaningful, not indicative of failure. Each scale captures distinct, complementary biological information:

- High inter-scale ARI would indicate redundancy (scales not adding information)
- Low ARI demonstrates that each scale reveals different organizational features
- This is analogous to comparing city blocks to state boundaries - both valid but different abstractions of the same geography

### Scale-Specific Validation
Rather than expecting inter-scale agreement, each scale is validated independently:
- **10μm**: Overlay on nuclear markers to verify cellular approximation
- **20μm**: Compare with known cell-cell interaction distances
- **40μm**: Align with anatomical boundaries and tissue compartments

### Visual Validation
Segmentation quality assessed through:
- Overlay plots of SLIC boundaries on DNA channels
- Visual confirmation of biologically plausible segment shapes
- Scale-appropriate feature capture verification

## Batch Correction

### Sham-Anchored Normalization
Applied when multiple acquisition batches detected:
1. Identify sham control batches (Day 0, Condition='Sham')
2. Compute reference statistics from pooled sham data
3. Normalize all batches relative to sham reference (z-score)
4. Preserves biological dynamics while correcting technical effects

### Batch Definition
Batches defined by:
- Acquisition day
- Replicate ID
- Technical replicate

## Segmentation Quality Validation

### Method-Focused Validation Approach
For this methods paper, validation focuses on the core contribution: morphology-aware tissue segmentation using SLIC on DNA channels. Rather than attempting to simulate protein expression patterns, we validate the segmentation method directly.

### Segmentation Quality Metrics

#### Morphological Metrics
1. **Compactness**: Measures how circular/compact segments are (4π × area / perimeter²)
2. **Boundary Adherence**: Quantifies how well segment boundaries align with DNA signal gradients
3. **Spatial Coherence**: Assesses whether segments form contiguous regions
4. **Size Consistency**: Validates segment sizes match expected scale (10μm, 20μm, 40μm)

#### Biological Correspondence
Without hardcoded protein assumptions, the validation framework:
- Computes marker enrichment within segments for any protein panel
- Measures colocalization between high-expression regions and segment boundaries
- Validates that biologically related markers cluster within same segments

### Visual Validation
- SLIC boundary overlays on DNA channels for direct assessment
- Multi-scale comparison showing hierarchical tissue organization
- Scale-appropriate feature capture verification

## Statistical Analysis

### Cross-Sectional Design
- 8 biological replicates total (2 per timepoint)
- Timepoints: Control (T0), T1, T3, T7 post-treatment
- Anatomical regions analyzed separately
- No longitudinal tracking (different replicates per timepoint)

### Multiple Testing Correction
- Benjamini-Hochberg FDR for protein comparisons
- Bonferroni for planned contrasts

## Software and Dependencies

### Core Libraries
- Python 3.8+
- NumPy 1.19+
- Pandas 1.3+
- Scikit-learn 1.0+
- Scikit-image 0.19+
- SciPy 1.7+

### Configuration
All parameters externalized to `config.json` for reproducibility. The pipeline is experiment-agnostic through configurable metadata mapping:

```json
{
  "metadata_tracking": {
    "replicate_column": "Mouse",
    "timepoint_column": "Injury Day", 
    "condition_column": "Condition",
    "region_column": "Details"
  }
}
```

### Quality Control
Production-ready QC includes:
- Total Ion Count (TIC) monitoring per ROI
- Calibration drift tracking across acquisition
- DNA signal quality assessment for segmentation
- Batch effect visualization and validation

## Data Availability
Raw IMC data and processed results available at [repository URL].

## Study Design and Statistical Considerations

### Experimental Design
**Objective:** Develop and validate superpixel-based spatial analysis methods for kidney injury IMC data

**Sample Size:** n=2 mice per timepoint (Day 0, 1, 3, 7) - **PILOT STUDY FOR HYPOTHESIS GENERATION**

**Design Type:** Cross-sectional (different subjects at each timepoint)
- 8 biological replicates total across 4 timepoints  
- Timepoints: Control (T0), T1, T3, T7 post-treatment
- Anatomical regions analyzed separately (cortex/medulla)
- Multiple ROIs per subject for technical replication

### Statistical Limitations and Approach
**Critical Limitation:** n=2 per timepoint prevents inferential statistics between specific timepoints

**Valid Analyses:**
- **Trend analysis** across all timepoints using regression (leverages all 8 mice)
- **Effect size estimation** with bootstrap confidence intervals
- **Pattern consistency** assessment within pilot data
- **Method validation** and parameter optimization

**Invalid Analyses:**
- P-value testing between individual timepoints
- Population-level biological significance claims
- Causal inference about injury mechanisms

### Expected Patterns and Validation
**Temporal Progression Hypotheses:**
- Day 1: Focal neutrophil infiltration (Ly6G+)
- Day 3: Macrophage activation and expansion (CD11b+/CD206+)
- Day 7: Fibroblast activation or resolution (CD140a/b patterns)

**Anatomical Hypotheses:**
- Cortex: Higher vascular density (CD31/CD34), glomerular structures
- Medulla: More fibroblast presence (CD140a), different injury patterns

## Parameter Justification and Thresholds

### Quality Control Thresholds
All thresholds derived from pilot data analysis and instrument specifications:

**Total Ion Count Monitoring:**
- `min_tic_percentile: 10` - Captures technical noise floor in IMC acquisition
- `max_low_tic_pixels_percent: 20` - Identifies ROIs with acquisition artifacts

**Calibration Drift:**
- `max_drift_percent: 5` - Based on Hyperion manufacturer specifications
- `max_cv_across_rois: 0.3` - Typical batch variation in mass cytometry

**DNA Signal Quality:**
- `min_dna_signal: 1.0` - Minimum for reliable nuclear detection
- `min_tissue_coverage_percent: 10` - Excludes edge artifacts
- `dna_threshold_std_multiplier: 2.0` - Standard outlier detection

**Signal-to-Background Ratio:**
- `min_snr: 3.0` - Standard threshold for reliable protein detection

### Segmentation Parameters
**SLIC Superpixel Settings:**
- `compactness: 10.0` - Optimized for kidney tubular morphology
- `sigma: 2.0` - 2μm smoothing approximates single-cell resolution
- `n_segments_per_mm2: 2500` - Yields ~20μm superpixels matching tubular scale

### Biological Scale Parameters
**Kidney-Specific Scales:**
- `10μm`: Peritubular capillary diameter
- `20μm`: Average tubular cross-section
- `40μm`: Glomerular diameter + Bowman's space

### Arcsinh Transformation
**Automatic Optimization:**
- `percentile_threshold: 5.0` - 5th percentile method for cofactor optimization
- Prevents over-compression while avoiding over-expansion
- Each protein marker gets optimized cofactor: `cofactor = 5th percentile of positive counts`

### Clustering Parameters
- `k_range: [2, 12]` - Based on expected kidney cell diversity
- `n_iterations: 50` - Provides stable confidence intervals
- Bootstrap iterations: 10,000 for 95% CI estimation

**Important Note:** All thresholds derived from n=2 pilot data require validation in larger cohorts.

## Coabundance Feature Engineering

### Rationale
Standard IMC analysis treats each protein marker independently, ignoring biologically relevant protein co-expression patterns (e.g., CD31+CD34+ endothelial cells, CD11b+CD206+ M2 macrophages). We developed a feature engineering approach to capture these relationships.

### Feature Generation
For a dataset with P proteins, we generate an enriched feature space:

**1. Original Features (P=9):**
$$
\mathbf{X}_{\text{original}} = [x_1, x_2, ..., x_P] \in \mathbb{R}^{N \times P}
$$

**2. Pairwise Products (P(P-1)/2=36):**
Products identify co-expressing cells:
$$
x_{i,j}^{\text{product}} = x_i \cdot x_j \quad \forall i < j
$$
Example: CD31 × CD34 identifies endothelial cells with both markers.

**3. Pairwise Ratios (P(P-1)=72):**
Ratios capture relative abundances:
$$
x_{i,j}^{\text{ratio}} = \frac{x_i + \epsilon}{x_j + \epsilon} \quad \forall i \neq j
$$
where ε = 25th percentile to prevent division by near-zero values.
Example: CD206/CD11b ratio distinguishes M2 vs M1 macrophage polarization.

**4. Spatial Covariances (P(P-1)/2=36):**
Capture local neighborhood co-expression (novel contribution):
$$
\text{Cov}_{\text{spatial}}(x_i, x_j) = \frac{1}{|N_r(s)|} \sum_{t \in N_r(s)} (x_i^t - \bar{x}_i)(x_j^t - \bar{x}_j)
$$
where N_r(s) is the r-neighborhood (r=20μm) around superpixel s.

**Total enriched features:** 9 + 36 + 72 + 36 = **153 features**

### Feature Selection (CRITICAL)
Direct use of 153 features creates catastrophic overfitting risk for ~1000 superpixel datasets. We employ LASSO-based feature selection:

$$
\hat{\beta} = \arg\min_{\beta} \left\{ \frac{1}{2N} \|\mathbf{y} - \mathbf{X}\beta\|_2^2 + \lambda \|\beta\|_1 \right\}
$$

where:
- y is the first principal component of enriched features (unsupervised target)
- λ is selected via 5-fold cross-validation
- Features with non-zero coefficients are retained

**Result:** 153 → 30 features (∼√N, following rule of thumb for N∼1000)

### Comparison to Standard Methods
- **Scanpy/Squidpy:** Use PCA on all features → captures variance but no interpretable feature selection
- **Our approach:** LASSO provides interpretable feature importance + sparsity

## Spatial Clustering with Scale-Adaptive Graphs

### Leiden Community Detection
We use Leiden algorithm (Traag et al. 2019), superior to Louvain:

$$
H = \sum_{c} \left[ e_c - \gamma \frac{K_c^2}{2m} \right]
$$

where:
- e_c = number of edges in community c
- K_c = sum of degrees in community c
- m = total edges
- γ = resolution parameter (controls granularity)

**Key advantage:** Monotonic improvement guarantee (Louvain can get stuck in suboptimal partitions)

### kNN Graph Construction
Standard practice uses fixed k neighbors. We implement **scale-adaptive k**:

$$
k_{\text{scale}} = \min\left(15, \max\left(8, \lfloor 2 \log(N_{\text{scale}}) \rfloor\right)\right)
$$

where N_scale is number of superpixels at that scale.

**Justification:**
- Fine scale (10μm, ~1000 superpixels): k=14 → 1.4% connectivity
- Coarse scale (40μm, ~100 superpixels): k=10 → 10% connectivity
- Fixed k=15 at coarse scales creates over-connected graphs (15% connectivity) → destroys community structure

### Feature + Spatial Integration
Combined feature matrix:

$$
\mathbf{F}_{\text{combined}} = [\mathbf{F}_{\text{protein}}; w_s \cdot \mathbf{C}_{\text{spatial}}]
$$

where:
- F_protein = selected features (30 dimensions after LASSO)
- C_spatial = spatial coordinates (2 dimensions)
- w_s = scale-dependent spatial weight (0.2 for ≤20μm, 0.4 for >20μm)

**Distance metric:** Euclidean distance in combined space

### Resolution Selection via Stability Analysis
Resolution γ selected via bootstrap stability:

1. For each candidate resolution γ:
   - Generate B=100 bootstrap samples (85% subsampling)
   - Cluster each sample
   - Compute pairwise ARI between all clusterings

2. Stability score:
$$
S(\gamma) = \frac{2}{B(B-1)} \sum_{i<j} \text{ARI}(C_i^{\gamma}, C_j^{\gamma})
$$

3. Select resolution with maximum stability (target: S≥0.75)

**Justification:** Lancichinetti & Fortunato (2012) established stability as gold standard for resolution selection.

## Multi-Scale Analysis

### Scale Selection
Scales chosen based on kidney anatomy:
- **10μm:** Peritubular capillary diameter → capillary-level interactions
- **20μm:** Tubular cross-section → tubular microenvironments
- **40μm:** Glomerular diameter → architectural units

### Scale Consistency Metrics
To validate multi-scale organization, we compute:

**1. Hierarchical Consistency:**
$$
H(s_1, s_2) = \text{ARI}(\text{Coarse}(C_{s_1}), C_{s_2})
$$
where Coarse(C) maps fine-scale clusters to coarse scale by majority vote.

**2. Cross-Scale Stability:**
$$
\text{Stability}_{\text{cross}} = \frac{1}{|S|(|S|-1)} \sum_{s_1 \neq s_2} H(s_1, s_2)
$$

**Expected:** Hierarchical nesting (fine clusters aggregate into coarse clusters)

## Biological Validation Framework

### Kidney Anatomical Validation
Clusters validated against known kidney marker enrichment:

**Cortex signature:**
- High: CD31, CD34 (glomerular endothelium)
- Low: CD140a (relative to medulla)

**Enrichment score:**
$$
E_{\text{cortex}}(c) = \frac{1}{|M_h|} \sum_{m \in M_h} \frac{\bar{x}_{m,c}}{\bar{x}_m} - \frac{1}{|M_l|} \sum_{m \in M_l} \frac{\bar{x}_{m,c}}{\bar{x}_m}
$$

where M_h = high markers, M_l = low markers, c = cluster

**Quality threshold:** At least one cluster with E_cortex > 0.3

### Temporal Validation (Injury Timepoints)
Expected immune responses validated:
- **Day 1:** Ly6G↑, CD11b↑ (neutrophil recruitment)
- **Day 3:** CD206↑, CD11b↑ (macrophage activation)
- **Day 7:** CD140a↑, CD140b↑ (fibrosis/resolution)

**Fold-change threshold:** ≥1.2× increase vs baseline for expected markers

## Statistical Analysis

### Multiple Testing Correction
For multi-scale comparisons, we apply Bonferroni correction:
$$
\alpha_{\text{corrected}} = \frac{\alpha}{|S|}
$$
where |S| = 3 scales → α_corrected = 0.017 for family-wise α=0.05

### Bootstrap Confidence Intervals
Stability estimates reported with 95% CI from bootstrap distribution (B=100 iterations)

### Spatial Autocorrelation
Moran's I computed to validate spatial coherence of clusters:
$$
I = \frac{N}{W} \frac{\sum_i \sum_j w_{ij}(y_i - \bar{y})(y_j - \bar{y})}{\sum_i (y_i - \bar{y})^2}
$$

where w_ij = spatial weight (1 if neighbors, 0 otherwise)

**Interpretation:** I > 0 indicates spatial clustering, I ≈ 0 random, I < 0 dispersion

## Computational Implementation

### Performance Optimization
- **Graph Caching:** kNN graph computed once, reused across bootstrap iterations (10× speedup)
- **Parallel Processing:** Bootstrap iterations distributed across cores
- **Memory Management:** Chunked processing for large ROIs (>10⁶ pixels)

### Reproducibility
- Fixed random seeds (random_state=42)
- Configuration-driven design (all parameters in config.json)
- Comprehensive provenance tracking (software versions, parameter snapshots)

## Code Availability
Analysis pipeline available at: https://github.com/[username]/imc-pipeline