# Methods - IMC Analysis Pipeline

## Overview
This document provides detailed methodology for the IMC analysis pipeline, suitable for publication methods sections.

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
Ion count data undergoes variance-stabilizing transformation:
```
Transformed = arcsinh(Corrected / cofactor)
```
- Cofactor = 1.0 (optimized for IMC dynamic range)
- Applied after background correction

### 4. Standardization
Post-transformation standardization using StandardScaler:
```
Standardized = (Transformed - μ) / σ
```
Applied per-channel across all pixels.

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
- Cross-scale consistency (ARI > 0.7)
- Biological coherence scoring

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

## Code Availability
Analysis pipeline available at: https://github.com/[username]/imc-pipeline