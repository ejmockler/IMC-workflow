# IMC Analysis - Development Guide

## Modular Architecture

### Directory Structure
```
src/
├── analysis/
│   ├── roi_main.py           # BatchAnalyzer - main analysis orchestrator  
│   ├── region_classifier.py  # Region quality classification
│   ├── dual_pipeline.py      # Nuclear vs bulk analysis pipelines
│   ├── pipeline.py           # Mixed-resolution analysis coordination
│   └── network.py            # Network analysis and graph theory
├── visualization/
│   ├── main.py         # VisualizationPipeline - orchestrates all figures
│   ├── roi.py          # ROIVisualizer - single ROI analysis figures
│   ├── temporal.py     # Time-series visualizations
│   ├── condition.py    # Condition comparison plots
│   ├── replicate.py    # Replicate variance analysis
│   ├── network.py      # Network visualizations
│   ├── network_clean.py # CleanNetworkVisualizer - protein colocalization networks
│   └── components.py   # Shared visualization utilities
├── utils/
│   └── helpers.py      # Metadata classes and utilities
└── config.py           # Main configuration class
```

## Development Principles

### Configuration-Driven Design
- All parameters externalized to `config.json`
- `Config` class provides single source of truth
- No hardcoded values in analysis or visualization modules

### Data Flow Architecture
1. **Region Classification**: Classify tissue by segmentation quality
2. **Dual Analysis**: Nuclear analysis (CLEAR) + Bulk analysis (DENSE) 
3. **Spatial Relationships**: Border analysis between region types (no data comparison)
4. **Visualization**: Separate panels for nuclear vs bulk results

### Extensibility Patterns
- **Strategy Pattern**: `NetworkAnalyzer` base class for different analysis approaches
- **Factory Pattern**: `VisualizationPipeline` creates appropriate visualizers
- **Data Classes**: Clean separation of data structures and processing logic

# Important Instructions
- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing existing files to creating new ones
- NEVER proactively create documentation files