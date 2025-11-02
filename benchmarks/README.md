# IMC Pipeline Benchmarking

This directory contains infrastructure for comparing our multi-scale pipeline against established IMC analysis tools (primarily Steinbock).

## Directory Structure

```
benchmarks/
├── data/                          # Public IMC datasets for validation
│   ├── bodenmiller_kidney/        # Bodenmiller lab kidney dataset (if available)
│   ├── jackson_breast/            # Jackson et al. breast cancer data
│   └── dataset_metadata.json      # Metadata for all benchmark datasets
│
├── steinbock_outputs/             # Results from Steinbock pipeline
│   ├── dataset_name/
│   │   ├── img/                   # Preprocessed images
│   │   ├── masks/                 # Cell segmentation masks
│   │   ├── intensities/           # Single-cell intensity measurements
│   │   └── neighbors/             # Spatial neighborhood data
│   └── run_metadata.json          # Steinbock version, parameters, runtime
│
├── our_outputs/                   # Results from our pipeline
│   ├── dataset_name/
│   │   ├── roi_results/           # Per-ROI analysis results
│   │   ├── cell_type_annotations/ # Cell type assignments
│   │   └── spatial_enrichments/   # Spatial neighborhood enrichments
│   └── run_metadata.json          # Pipeline version, config, runtime
│
├── comparison_notebooks/          # Jupyter notebooks for benchmarking
│   ├── 01_data_preparation.ipynb  # Download and format datasets
│   ├── 02_run_steinbock.ipynb     # Execute Steinbock pipeline
│   ├── 03_run_our_pipeline.ipynb  # Execute our pipeline
│   ├── 04_quantitative_comparison.ipynb  # Metrics and statistics
│   └── 05_visualization.ipynb     # Side-by-side visual comparisons
│
├── configs/                       # Configuration files
│   ├── steinbock_config.yml       # Steinbock parameters
│   ├── our_pipeline_config.json   # Our pipeline parameters (matched)
│   └── benchmark_metrics.json     # Metrics to compute
│
├── scripts/                       # Automation scripts
│   ├── download_datasets.sh       # Fetch public data from Zenodo
│   ├── run_steinbock_docker.sh    # Wrapper for Steinbock Docker
│   ├── run_our_pipeline.sh        # Wrapper for our pipeline
│   └── compute_metrics.py         # Calculate benchmark metrics
│
└── README.md                      # This file
```

## Benchmarking Strategy

### Principle 1: Isolation
- Steinbock runs in Docker container (no conda environment conflicts)
- Our pipeline runs in separate virtual environment
- Data and outputs kept separate
- No cross-contamination of results

### Principle 2: Matched Parameters
Where possible, use equivalent parameters:
- Same preprocessing steps (background subtraction, normalization)
- Comparable segmentation scales (Steinbock cells ≈ our 10μm superpixels)
- Same spatial neighborhood k (default k=10)

### Principle 3: Fair Comparison
Compare on tasks BOTH pipelines can do:
- ✅ Cell/superpixel segmentation quality
- ✅ Spatial neighborhood enrichment
- ✅ Marker expression distributions
- ✅ Computational performance (time, memory)
- ❌ NOT cell type annotation (different marker panels, gating strategies)

### Principle 4: Multiple Metrics
- **Biological plausibility**: Do results match literature?
- **Reproducibility**: Consistent across ROIs/samples?
- **Computational efficiency**: Runtime, memory usage
- **Robustness**: Sensitivity to parameter changes

## Datasets for Benchmarking

### Target Datasets (Priority Order)

1. **Bodenmiller Kidney Dataset** (if available on Zenodo)
   - Why: Tissue type match (kidney), inflammation focus
   - Expected: ~20-40 markers, multiple samples
   - Challenge: Different marker panel than our 9-marker study

2. **Jackson et al. Breast Cancer** (Nature 2020)
   - Why: Well-characterized, widely cited, public
   - Expected: ~30 markers, large cohort
   - Challenge: Different tissue type

3. **Bodenmiller Example Dataset** (Zenodo)
   - Why: Designed for tutorial/testing
   - Expected: Small, well-documented
   - Challenge: May be too simple to stress-test methods

### Dataset Requirements
For a dataset to be suitable:
- ✅ Public and downloadable
- ✅ ≥3 biological samples (to test reproducibility)
- ✅ IMC .txt or .mcd format
- ✅ Metadata available (timepoint, condition, etc.)
- ⚠️ Marker count: ≥9 markers (our pipeline), any count (Steinbock)

## Benchmark Metrics

### 1. Segmentation Quality
**Metrics**:
- Number of objects detected (cells vs superpixels)
- Object size distribution (mean, std, min/max)
- Boundary smoothness (perimeter/area ratio)
- Spatial coverage (% tissue assigned)

**Comparison**:
- Steinbock: Cell-level segmentation (Cellpose/Ilastik)
- Our pipeline: Superpixel-level (SLIC at 10/20/40μm)
- **Expected**: Different granularity, not "better/worse"

### 2. Marker Expression Distributions
**Metrics**:
- Per-marker: Mean, median, variance
- Cross-marker correlations
- Dynamic range captured

**Comparison**:
- Should be SIMILAR (same raw data)
- Differences indicate preprocessing divergence

### 3. Spatial Organization Detected
**Metrics**:
- Neighborhood enrichment scores (observed/expected)
- Number of significant spatial patterns (p < 0.05)
- Spatial autocorrelation (Moran's I) for markers

**Comparison**:
- Core test: Do both detect same spatial structures?
- Our advantage: Multi-scale reveals hierarchical patterns

### 4. Computational Performance
**Metrics**:
- Runtime (seconds per ROI)
- Peak memory usage (GB)
- Scalability (runtime vs ROI size)

**Comparison**:
- Steinbock: Optimized C++/GPU (likely faster)
- Our pipeline: Pure Python (likely slower)
- Trade-off: Speed vs interpretability/flexibility

### 5. Reproducibility
**Metrics**:
- Intra-sample variance (multiple ROIs from same tissue)
- Inter-sample consistency (biological replicates)
- Parameter sensitivity (change threshold ±10%, measure Δ)

**Comparison**:
- Which pipeline gives more consistent results?
- Which is more robust to parameter tweaks?

## Running Benchmarks

### Step 1: Download Public Dataset
```bash
cd benchmarks/scripts/
./download_datasets.sh bodenmiller_kidney
# or
./download_datasets.sh jackson_breast
```

### Step 2: Run Steinbock Pipeline
```bash
# Pull Steinbock Docker image
docker pull ghcr.io/bodenmillergroup/steinbock:0.16.1

# Run Steinbock on dataset
./run_steinbock_docker.sh benchmarks/data/bodenmiller_kidney/
```

**Outputs**: `steinbock_outputs/bodenmiller_kidney/`

### Step 3: Run Our Pipeline
```bash
# Activate virtual environment
source .venv/bin/activate

# Run our pipeline on same dataset
./run_our_pipeline.sh benchmarks/data/bodenmiller_kidney/
```

**Outputs**: `our_outputs/bodenmiller_kidney/`

### Step 4: Compute Comparison Metrics
```bash
python scripts/compute_metrics.py \
    --steinbock steinbock_outputs/bodenmiller_kidney/ \
    --ours our_outputs/bodenmiller_kidney/ \
    --output comparison_results.json
```

### Step 5: Generate Visualization Report
```bash
jupyter notebook comparison_notebooks/04_quantitative_comparison.ipynb
```

## Comparison Notebooks (Detailed)

### 01_data_preparation.ipynb
**Purpose**: Download and format public datasets for both pipelines

**Tasks**:
1. Download from Zenodo (or other source)
2. Convert to .txt format if needed
3. Create metadata CSV (ROI name, condition, timepoint, etc.)
4. Verify data integrity (all channels present, no corruption)
5. Generate summary statistics (n_samples, n_markers, image_sizes)

**Outputs**:
- `data/dataset_name/ROI_*.txt`
- `data/dataset_name/metadata.csv`
- `data/dataset_metadata.json`

### 02_run_steinbock.ipynb
**Purpose**: Execute Steinbock pipeline with documented parameters

**Tasks**:
1. Set up Steinbock Docker environment
2. Configure parameters (segmentation method, kernel sizes, etc.)
3. Run preprocessing → segmentation → measurement → neighbors
4. Record runtime, memory usage, errors
5. Validate outputs (all expected files present)

**Outputs**:
- `steinbock_outputs/dataset_name/` (full Steinbock results)
- `steinbock_outputs/run_metadata.json` (provenance)

### 03_run_our_pipeline.ipynb
**Purpose**: Execute our pipeline with matched parameters

**Tasks**:
1. Convert dataset to our format (if needed)
2. Create config.json with equivalent settings
3. Run main_pipeline.py for each ROI
4. Run cell type annotation + spatial analysis
5. Record runtime, memory usage, errors

**Outputs**:
- `our_outputs/dataset_name/roi_results/`
- `our_outputs/dataset_name/cell_type_annotations/`
- `our_outputs/run_metadata.json`

### 04_quantitative_comparison.ipynb
**Purpose**: Calculate all benchmark metrics

**Structure**:
```python
## Section 1: Load Results
steinbock_data = load_steinbock_results(...)
our_data = load_our_results(...)

## Section 2: Segmentation Quality
compare_segmentation_quality(steinbock_data, our_data)
# Plots: Object count, size distribution, coverage

## Section 3: Marker Distributions
compare_marker_distributions(steinbock_data, our_data)
# Plots: Per-marker violin plots, correlation matrices

## Section 4: Spatial Patterns
compare_spatial_enrichments(steinbock_data, our_data)
# Metrics: Overlap in detected patterns, enrichment concordance

## Section 5: Computational Performance
compare_performance(steinbock_metadata, our_metadata)
# Table: Runtime, memory, scalability

## Section 6: Summary
generate_summary_report()
# Overall assessment: Where each pipeline excels
```

### 05_visualization.ipynb
**Purpose**: Side-by-side visual comparisons

**Figures**:
1. Same ROI, Steinbock vs our segmentation (overlay)
2. Marker expression heatmaps (cells vs superpixels)
3. Spatial neighborhood graphs (side-by-side)
4. Multi-scale hierarchy (our pipeline only)

## Benchmark Automation (Future)

### Continuous Integration
Once validated manually, automate benchmarks:

```yaml
# .github/workflows/benchmark.yml
name: Benchmark Pipeline

on:
  pull_request:
    paths:
      - 'src/analysis/**'
      - 'config.json'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - name: Download test dataset
      - name: Run Steinbock (cached)
      - name: Run our pipeline
      - name: Compare metrics
      - name: Fail if regression detected
```

**Regression Detection**:
- Runtime increases >20%
- Memory usage increases >20%
- Spatial enrichment concordance drops >10%

## Expected Outcomes

### Hypothesis: Our Pipeline Will...
1. ✅ **Segment differently** (superpixels vs cells) but capture similar marker patterns
2. ✅ **Detect comparable spatial enrichments** (validates core biology)
3. ✅ **Reveal multi-scale hierarchy** that Steinbock misses (our innovation)
4. ⚠️ **Run slower** than Steinbock (pure Python vs optimized)
5. ⚠️ **Use more memory** at fine scales (10μm superpixels numerous)

### Success Criteria
**Minimum viable benchmark** (for publication):
- ✅ Analyzed ≥1 public dataset with both pipelines
- ✅ Demonstrated comparable biological findings
- ✅ Quantified performance trade-offs
- ✅ Shown multi-scale advantage on ≥1 biological pattern

**Ideal benchmark** (strengthens paper):
- ✅ Analyzed ≥3 datasets (different tissues/conditions)
- ✅ Reproducibility across biological replicates
- ✅ Sensitivity analysis (parameter robustness)
- ✅ Computational scaling study (ROI size vs runtime)

## Notes and Caveats

### Not an "Apples-to-Apples" Comparison
- Steinbock: Cell-based, membrane segmentation, unsupervised phenotyping
- Our pipeline: Superpixel-based, DNA segmentation, boolean gating
- **Don't expect identical results** - expect complementary insights

### Focus on Strengths
- Steinbock strength: Single-cell resolution, fast, polished
- Our strength: Multi-scale hierarchy, membrane-marker-free, interpretable

### Honest Framing
**In paper**: "We benchmark against Steinbock to demonstrate:
1. Our pipeline captures similar spatial biology (validates correctness)
2. Multi-scale analysis reveals patterns missed by single-resolution
3. DNA-based segmentation works when membrane markers unavailable
4. Trade-off: Slower runtime for greater interpretability"

**NOT**: "Our pipeline is better than Steinbock"

## References

- Steinbock: Windhager et al. (2021) bioRxiv 2021.11.12.468357
- Steinbock GitHub: https://github.com/BodenmillerGroup/steinbock
- IMC datasets: https://github.com/BodenmillerGroup/imcdatasets
