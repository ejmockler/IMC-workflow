# IMC Mixed-Resolution Analysis Pipeline

## Critical Notice
**This pipeline implements a mixed-resolution approach that treats nuclear and bulk measurements as INCOMPARABLE data types from different tissue compartments. Results must be interpreted accordingly.**

## Overview

This repository contains a scientifically defensible analysis framework for Imaging Mass Cytometry (IMC) data that acknowledges fundamental limitations of fixed tissue imaging where cell membranes are invisible.

### Key Principles
1. **No cell segmentation** - Cell membranes are indistinguishable in fixed tissue
2. **Mixed-resolution analysis** - Different approaches for different tissue densities
3. **No false comparisons** - Nuclear counts and bulk signal are never compared
4. **Hypothesis-generating only** - All findings require validation

## The Fundamental Problem

IMC captures a 2D projection of ~4μm thick tissue with only nuclear markers visible:
- **Cannot see cell membranes** → No true single-cell analysis possible
- **Nuclei overlap in Z-dimension** → Merged blobs in dense regions
- **Only 9 protein markers** → Cannot definitively identify cell types
- **DNA signal ≠ cell count** → Due to cell cycle, polyploidy, Z-overlap

## Our Solution: Mixed-Resolution Analysis

### Phase 1: Region Classification
The pipeline first classifies every pixel into one of four categories based on objective metrics:

- **CLEAR** (~30-50% of tissue): Individual nuclei are distinguishable
- **DENSE** (~30-50% of tissue): Overlapping nuclei create merged blobs
- **AMBIGUOUS** (~10-30% of tissue): Uncertain segmentation quality
- **BACKGROUND**: No DNA signal

Classification uses five objective metrics:
1. Nuclear Separation Index (nearest-neighbor distances)
2. Overlap Coefficient (boundary vs interior signal)
3. Edge Contrast Ratio (Sobel edge strength)
4. Signal Uniformity (within-nucleus CV)
5. Size Consistency (nuclear size variation)

### Phase 2: Separate Analyses by Region Type

#### CLEAR Regions → Nuclear Analysis
- Watershed segmentation of individual nuclei
- Count actual nuclei per unit area
- Measure per-nucleus protein expression
- Calculate nuclear neighborhoods and spatial patterns

#### DENSE Regions → Bulk Analysis
- Grid-based sampling (10μm default)
- Measure total signal intensity per grid cell
- **Cannot determine cell counts**
- Analyze regional patterns and textures

#### AMBIGUOUS Regions → Excluded
- Default: Exclude from analysis
- Document percentage excluded
- Report potential bias introduced

### Phase 3: Spatial Relationships ONLY
- Analyze how region types are spatially arranged
- Calculate border lengths between regions
- Measure proximity and adjacency
- **NEVER compare nuclear counts with bulk measurements**

## What This Pipeline Does NOT Do

❌ **Does NOT provide single-cell resolution**
❌ **Does NOT count cells in dense regions**
❌ **Does NOT identify specific cell types**
❌ **Does NOT compare nuclear and bulk measurements**
❌ **Does NOT make definitive biological claims**

## What This Pipeline CAN Do

✓ **Counts nuclei in sparse regions where segmentation is reliable**
✓ **Measures signal intensity patterns in dense regions**
✓ **Analyzes spatial relationships between region types**
✓ **Generates hypotheses for validation**
✓ **Documents all limitations transparently**

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/IMC.git
cd IMC

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Analysis
```bash
# Analyze all ROIs with mixed-resolution approach
python run_analysis.py

# Generate visualizations (shows separate nuclear and bulk results)
python run_visualization.py
```

### Understanding Outputs

Every analysis output includes:
1. **Region classification map** showing CLEAR, DENSE, AMBIGUOUS, BACKGROUND
2. **Nuclear analysis** (CLEAR regions only) with actual counts
3. **Bulk analysis** (DENSE regions only) with signal intensity
4. **Spatial relationships** between region types
5. **Explicit warnings** about data interpretation

Example output structure:
```json
{
  "region_classification": {
    "clear_coverage": 0.42,
    "dense_coverage": 0.38,
    "ambiguous_coverage": 0.15,
    "ambiguous_warning": "Results exclude 15% of tissue"
  },
  "nuclear_analysis": {
    "n_nuclei": 523,
    "warning": "Only in segmentable regions"
  },
  "bulk_analysis": {
    "signal_intensity": {...},
    "warning": "Cannot convert to cell counts"
  },
  "spatial_analysis": {
    "border_lengths": {...},
    "critical_note": "Nuclear and bulk are INCOMPARABLE"
  }
}
```

## Critical Interpretation Guidelines

### Acceptable Statements
✓ "We identified 523 nuclei in sparse tissue regions"
✓ "CD45 signal intensity was elevated in dense regions"
✓ "CLEAR and DENSE regions share 2,340 pixels of border"
✓ "This marker pattern may be associated with inflammation"

### Forbidden Statements
❌ "Cell density increased in treated samples"
❌ "We identified M2 macrophages"
❌ "Nuclear counts correlated with bulk signal"
❌ "Cells interact at tissue interfaces"

## Configuration

Key parameters in `config.json`:
```json
{
  "segmentation": {
    "min_nuclear_size_um2": 25,
    "max_nuclear_size_um2": 400,
    "quality_threshold": 0.7,
    "ambiguous_handling": "exclude"
  },
  "mixed_resolution_analysis": {
    "enabled": true,
    "min_nuclei_for_statistics": 30
  }
}
```

## Validation

The pipeline includes extensive validation:
- Parameter sensitivity testing for classification thresholds
- Consistency checks with noise perturbation
- Classification entropy and boundary stability metrics
- Explicit reporting of excluded tissue fraction

## Limitations

### Fundamental
- No true single-cell resolution
- Cannot segment cells without membranes
- 2D projection loses Z-axis information
- Limited marker panel prevents cell type identification

### Technical
- Nuclear counts only valid in CLEAR regions (<50% of tissue)
- Bulk measurements cannot be converted to cell counts
- Excluding AMBIGUOUS regions introduces bias
- Different tissue compartments analyzed with different methods

### Interpretational
- Nuclear and bulk data are INCOMPARABLE
- All phenotypes are marker patterns, not validated cell types
- Spatial relationships are 2D projections only
- All findings are hypothesis-generating

## Methods Summary for Publications

"We implemented a mixed-resolution analysis approach for IMC data acknowledging that cell membranes are indistinguishable in fixed tissue. Tissue regions were classified as CLEAR (nuclei distinguishable), DENSE (overlapping nuclei), AMBIGUOUS (uncertain), or BACKGROUND based on five objective metrics: nuclear separation index, overlap coefficient, edge contrast, signal uniformity, and size consistency. CLEAR regions (X% of tissue) were analyzed using watershed segmentation to count individual nuclei. DENSE regions (Y% of tissue) were analyzed using 10μm grid sampling to measure signal intensity. AMBIGUOUS regions (Z% of tissue) were excluded, introducing potential bias. Spatial relationships between region types were analyzed, but nuclear counts and bulk measurements were never compared as they represent incompatible data types. All findings are hypothesis-generating and require orthogonal validation."

## Support

For questions about the mixed-resolution approach or interpretation of results, please open an issue on GitHub.

## License

MIT

## Acknowledgments

This approach was developed in response to brutal but constructive peer review that highlighted the fundamental impossibility of cell segmentation in fixed tissue IMC data without membrane markers.