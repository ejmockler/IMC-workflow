# Configuration Variants for Methods Validation

This directory contains specialized configurations for demonstrating and validating the methodological innovations of the IMC analysis pipeline.

## Available Configurations

### 1. `config_square_binning.json`
Traditional square grid binning approach for baseline comparison.
- Uses fixed-size square bins at 10μm, 20μm, 40μm
- No morphological awareness (ignores DNA channels for segmentation)
- Serves as the conventional baseline method

**Usage:**
```bash
python run_analysis.py --config config_variants/config_square_binning.json
```

### 2. `config_synthetic_validation.json`
Synthetic data validation with known ground truth.
- Generates synthetic IMC data with controlled spatial patterns
- Includes realistic noise models (Poisson, spatial artifacts, etc.)
- Tests method's ability to recover known patterns
- Parameter sensitivity analysis

**Usage:**
```bash
python run_synthetic_validation.py --config config_variants/config_synthetic_validation.json
```

### 3. `config_method_comparison.json`
Head-to-head comparison of SLIC vs alternatives on real data.
- Runs both SLIC and square binning on same dataset
- Computes comprehensive comparison metrics
- Statistical testing for method superiority
- Generates side-by-side visualizations

**Usage:**
```bash
python run_method_comparison.py --config config_variants/config_method_comparison.json
```

## Key Parameters for Methods Focus

### SLIC Innovation Parameters
- `n_segments_per_mm2`: 2500 (optimal for ~20μm tissue domains)
- `compactness`: 10.0 (balances shape regularity with boundary adherence)
- `sigma`: 2.0 (smoothing for DNA composite image)

### Multi-scale Validation
- `scales_um`: [10.0, 20.0, 40.0] (test consistency across scales)
- Metrics: ARI, NMI between scales to validate robustness

### Statistical Rigor (for n=2)
- `bootstrap_iterations`: 10000 (robust confidence intervals)
- `subsample_ratios`: [0.8, 0.9, 1.0] (stability testing)
- Focus on effect sizes and trends, not p-values

## Benchmarking Strategy

To address reviewer concerns about method validation:

1. **Baseline Comparison**: SLIC vs square binning
   - Clustering quality metrics (silhouette, Davies-Bouldin)
   - Spatial coherence (intra/inter-cluster distances)
   - Stability across parameter variations

2. **Synthetic Validation**: Known ground truth recovery
   - Generate data with known spatial organization
   - Add realistic IMC noise and artifacts
   - Measure ability to recover true patterns

3. **Parameter Sensitivity**: Robustness testing
   - Vary SLIC parameters (segments, compactness)
   - Test stability of biological conclusions
   - Identify optimal parameter ranges

## Expected Outcomes

These configurations will demonstrate:

1. **SLIC Superiority**: Better spatial coherence than square binning
2. **Robustness**: Consistent results across parameter ranges
3. **Biological Relevance**: Better separation of tissue regions
4. **Computational Efficiency**: Scalability metrics

## Notes on Statistical Limitations

All configurations acknowledge:
- n=2 biological replicates per timepoint
- No p-values for pairwise comparisons
- Focus on trends and effect sizes
- Bootstrap confidence intervals for uncertainty

## Integration with Main Pipeline

These configurations use the same pipeline code but with different parameters:
- Same data processing (arcsinh, standardization)
- Same batch correction (sham-anchored)
- Same clustering optimization
- Different segmentation methods for comparison