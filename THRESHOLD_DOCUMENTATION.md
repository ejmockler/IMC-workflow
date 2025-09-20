# IMC Analysis Threshold Documentation

## Scientific Justification for Hardcoded Thresholds

This document provides scientific rationale for all thresholds used in the IMC analysis pipeline.
These values are based on pilot data (n=2 mice) and should be validated with larger cohorts.

## Quality Control Thresholds

### 1. Total Ion Count (TIC) Monitoring
**Location:** `config.json` lines 159-163
```json
"min_tic_percentile": 10,
"max_low_tic_pixels_percent": 20
```
**Justification:** 
- 10th percentile captures technical noise floor in IMC acquisition
- 20% threshold identifies ROIs with potential acquisition artifacts
- Based on empirical observation that >20% low-TIC pixels indicates poor ablation

### 2. Calibration Drift
**Location:** `config.json` lines 164-167
```json
"max_drift_percent": 5,
"max_cv_across_rois": 0.3
```
**Justification:**
- 5% drift tolerance based on Hyperion manufacturer specifications
- CV of 0.3 represents typical batch variation in mass cytometry (Spitzer & Nolan, 2016)

### 3. DNA Signal Quality
**Location:** `config.json` lines 169-173
```json
"min_dna_signal": 1.0,
"min_tissue_coverage_percent": 10,
"dna_threshold_std_multiplier": 2.0
```
**Justification:**
- Minimum DNA signal of 1.0 ion count ensures nuclear detection
- 10% coverage threshold excludes edge artifacts and empty regions
- 2σ threshold is standard for outlier detection in signal processing

### 4. Signal-to-Background Ratio
**Location:** `config.json` line 176
```json
"min_snr": 3.0
```
**Justification:**
- SNR ≥ 3 is standard threshold for reliable protein detection in mass spectrometry
- Corresponds to ~95% confidence in signal vs. noise discrimination

### 5. Spatial Artifacts Detection
**Location:** `config.json` lines 179-183
```json
"edge_distance_fraction": 0.1,
"significance_threshold": 0.01,
"fold_change_threshold": 2.0
```
**Justification:**
- 10% edge exclusion accounts for laser edge effects in IMC
- p < 0.01 for spatial artifact detection (Bonferroni-adjusted for multiple testing)
- 2-fold change indicates biologically significant enrichment

## Segmentation Parameters

### 6. SLIC Compactness
**Location:** `config.json` line 58
```json
"compactness": 10.0
```
**Justification:**
- Balance between spatial coherence and boundary adherence
- Value of 10 validated on kidney tissue morphology (circular tubules)

### 7. Gaussian Smoothing
**Location:** `config.json` line 59
```json
"sigma": 2.0
```
**Justification:**
- 2μm smoothing kernel approximates single-cell resolution
- Reduces pixel-level noise while preserving nuclear boundaries

### 8. Superpixels per mm²
**Location:** `config.json` line 57
```json
"n_segments_per_mm2": 2500
```
**Justification:**
- Yields ~20μm superpixels, matching tubular cross-section size
- Validated against H&E staining of matched tissue sections

## Normalization Parameters

### 9. Arcsinh Transformation
**Location:** `config.json` lines 43-46
```json
"arcsinh_transform": {
  "optimization_method": "percentile",
  "percentile_threshold": 5.0
}
```
**Justification:**
- **Automatic optimization:** Each protein marker gets its own optimized cofactor
- **5th percentile method:** Standard approach in flow/mass cytometry
- **Variance stabilization:** Automatically adapts to each marker's dynamic range
- **Implementation:** `cofactor = 5th percentile of positive ion counts`
- **Benefits:** Prevents over-compression of low-expressing proteins while avoiding over-expansion of high-expressing proteins

## Clustering Parameters

### 10. Default Cluster Range
**Location:** `config.json` line 119
```json
"k_range": [2, 12]
```
**Justification:**
- Lower bound (2): Minimum for meaningful clustering
- Upper bound (12): Expected cell types in kidney (podocytes, tubular epithelial, endothelial, fibroblasts, immune subsets)

### 11. Stability Testing Iterations
**Location:** `config.json` line 124
```json
"n_iterations": 50
```
**Justification:**
- 50 iterations provides stable confidence intervals
- Balances computational cost with statistical robustness

## Batch Correction Parameters

### 12. Bootstrap Iterations
**Location:** `config.json` line 143
```json
"n_iterations": 10000
```
**Justification:**
- 10,000 iterations for stable 95% CI estimation
- Standard for bootstrap methods (Efron & Tibshirani, 1993)

## Biological Scale Parameters

### 13. Kidney-Specific Scales
**Location:** `config.json` lines 68-83
```json
"capillary": {"target_size_um": 10},
"tubular": {"target_size_um": 75},
"architectural": {"target_size_um": 200}
```
**Justification:**
- 10μm: Peritubular capillary diameter (Kriz & Kaissling, 2007)
- 75μm: Average tubular cross-section (proximal/distal tubules)
- 200μm: Glomerular diameter + surrounding Bowman's space

## Important Notes

1. **Pilot Study Limitations:** All thresholds derived from n=2 mice pilot data
2. **Validation Required:** Larger cohort needed for robust threshold optimization
3. **Tissue Specificity:** Values optimized for kidney cortex/medulla
4. **Technical Variability:** May need adjustment for different IMC instruments

## References

- Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap
- Giesen, C. et al. (2014). Highly multiplexed imaging of tumor tissues. Nature Methods
- Kriz, W. & Kaissling, B. (2007). Structural organization of the mammalian kidney
- Spitzer, M.H. & Nolan, G.P. (2016). Mass Cytometry: Single Cells, Many Features. Cell

## Revision History

- 2024-01-19: Initial documentation for hypothesis generation study
- Note: Thresholds subject to revision based on experimental validation