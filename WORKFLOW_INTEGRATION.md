# IMC Analysis Workflow Integration

## Current Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    MAIN PIPELINE                                 │
│                (src/analysis/main_pipeline.py)                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │  Per-ROI Processing       │
              │  • Ion count processing   │
              │  • SLIC segmentation      │
              │  • Multi-scale clustering │
              │  • Spatial statistics     │
              └───────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │  Output: roi_results/     │
              │  roi_*_results.json.gz    │
              └───────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              BIOLOGICAL ANALYSIS LAYER                          │
│              (Post-processing scripts)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  Module 1: Cell Type    │     │  (Optional: run per-ROI │
│  Annotation             │     │   during main pipeline) │
│                         │     │                         │
│  batch_annotate_all_    │     │  src/analysis/          │
│  rois.py                │     │  cell_type_annotation.py│
│                         │     │                         │
│  Input: roi_results/    │     │  Called by: analyze_    │
│  Output: cell_type_     │     │  single_roi() if enabled│
│  annotations/           │     │                         │
└─────────────────────────┘     └─────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│            Cell Type Annotations                                │
│            results/biological_analysis/cell_type_annotations/   │
│            • roi_*_cell_types.parquet                          │
│            • roi_*_annotation_metadata.json                    │
└─────────────────────────────────────────────────────────────────┘
              │
              ├──────────────────────┬──────────────────────┐
              ▼                      ▼                      ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│  Module 2:          │ │  Module 3:          │ │  (Future modules)   │
│  Differential       │ │  Spatial            │ │                     │
│  Abundance          │ │  Neighborhoods      │ │  • Cell-cell        │
│                     │ │                     │ │    interactions     │
│  differential_      │ │  spatial_neighbor   │ │  • Tissue zones     │
│  abundance_         │ │  hood_analysis.py   │ │  • Pathway analysis │
│  analysis.py        │ │                     │ │                     │
│                     │ │  Input: cell_type_  │ │                     │
│  Input: cell_type_  │ │  annotations/       │ │                     │
│  annotations/       │ │                     │ │                     │
│                     │ │  Output: spatial_   │ │                     │
│  Output:            │ │  neighborhoods/     │ │                     │
│  differential_      │ │  • temporal_        │ │                     │
│  abundance/         │ │    enrichments.csv  │ │                     │
│  • temporal_diff_   │ │  • regional_        │ │                     │
│    abundance.csv    │ │    enrichments.csv  │ │                     │
│  • regional_diff_   │ │                     │ │                     │
│    abundance.csv    │ │                     │ │                     │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
              │                      │
              └──────────┬───────────┘
                         ▼
              ┌─────────────────────┐
              │  Visualization      │
              │  (Jupyter notebooks)│
              │                     │
              │  • Temporal plots   │
              │  • Spatial heatmaps │
              │  • Network graphs   │
              │  • Summary figures  │
              └─────────────────────┘
```

## Data Flow Summary

### Phase 1: Core Analysis (Main Pipeline)
**Input**: Raw IMC data (*.txt files)
**Processing**: Ion counts → Segmentation → Clustering → Spatial stats
**Output**: `results/roi_results/roi_*_results.json.gz`

### Phase 2: Biological Interpretation (Post-processing)
**Input**: ROI results (JSON.gz)
**Processing**: Cell type annotation → Differential abundance → Spatial neighborhoods
**Output**: Structured biological insights (CSV/Parquet)

### Phase 3: Visualization (Jupyter)
**Input**: Biological analysis results
**Processing**: Matplotlib/Seaborn plotting
**Output**: Publication-quality figures

## Integration Points

### Current Integration
1. **Main pipeline** calls `annotate_roi_from_results()` IF cell_type_annotation is enabled in config
2. **Batch scripts** (batch_annotate_all_rois.py, etc.) are standalone post-processing
3. **No automatic chaining** - user runs each module manually

### Files and Locations

**Core Pipeline**:
- `src/analysis/main_pipeline.py` - IMCAnalysisPipeline class
- `src/analysis/cell_type_annotation.py` - annotate_roi_from_results()

**Biological Analysis Scripts** (root directory):
- `batch_annotate_all_rois.py` - Module 1
- `differential_abundance_analysis.py` - Module 2
- `spatial_neighborhood_analysis.py` - Module 3

**Config Integration**:
```json
{
  "cell_type_annotation": {
    "enabled": false,  // Set to true to run during main pipeline
    "positivity_threshold": {"method": "percentile", "percentile": 60},
    "cell_types": { ... }
  }
}
```

## Recommended Usage Pattern

### Option A: Post-Processing (Current, Recommended)
```bash
# Step 1: Run main pipeline on all ROIs
python -m src.analysis.main_pipeline

# Step 2: Annotate cell types
python batch_annotate_all_rois.py

# Step 3: Differential abundance
python differential_abundance_analysis.py

# Step 4: Spatial neighborhoods
python spatial_neighborhood_analysis.py

# Step 5: Visualize
jupyter notebook analyze_results.ipynb
```

**Pros**:
- Clean separation of concerns
- Can re-run biological analysis without re-clustering
- Easy to modify parameters (e.g., change threshold, rerun annotation)

**Cons**:
- Manual orchestration
- Requires running multiple scripts

### Option B: Integrated Pipeline (Future)
Create `run_biological_analysis.py`:
```python
from src.config import Config
from batch_annotate_all_rois import main as annotate_main
from differential_abundance_analysis import main as abundance_main
from spatial_neighborhood_analysis import main as spatial_main

def run_biological_workflow():
    # Automatically chains all modules
    annotate_main()
    abundance_main()
    spatial_main()
```

### Option C: Per-ROI Integration (Advanced)
Enable cell type annotation in `config.json`:
```json
{"cell_type_annotation": {"enabled": true}}
```

This runs annotation during `analyze_single_roi()`, but:
- **Not recommended** for batch processing (inefficient)
- Differential abundance and spatial neighborhoods still require batch context

## Output Directory Structure

```
results/
├── roi_results/                    # Main pipeline output
│   ├── roi_*_results.json.gz      # Per-ROI clustering results
│   └── provenance.json            # Config tracking
│
└── biological_analysis/            # Biological modules output
    ├── cell_type_annotations/     # Module 1
    │   ├── roi_*_cell_types.parquet
    │   ├── roi_*_annotation_metadata.json
    │   └── batch_annotation_summary.json
    │
    ├── differential_abundance/    # Module 2
    │   ├── roi_abundances.csv
    │   ├── temporal_differential_abundance.csv
    │   └── regional_differential_abundance.csv
    │
    └── spatial_neighborhoods/     # Module 3
        ├── roi_neighborhood_enrichments.csv
        ├── temporal_neighborhood_enrichments.csv
        └── regional_neighborhood_enrichments.csv
```

## Key Design Decisions

1. **Separation of Concerns**: Main pipeline handles technical processing, biological modules handle interpretation
2. **Config-Driven**: All parameters externalized to config.json
3. **Reproducibility**: Provenance tracking via config snapshots
4. **Scalability**: Post-processing can be re-run without re-clustering (expensive)
5. **Flexibility**: Easy to add new biological modules (Module 4, 5, etc.)

## Next Steps

1. **Create orchestrator script** (`run_biological_workflow.py`) to chain modules
2. **Add visualization notebook** consuming biological analysis outputs
3. **Document typical use cases** (e.g., "How do I re-annotate with different thresholds?")
4. **Add quality checks** between modules (e.g., minimum assignment rate warnings)
