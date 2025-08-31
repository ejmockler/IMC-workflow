# IMC Analysis - Development Guide

## Modular Architecture

### Directory Structure
```
src/
├── analysis/
│   ├── roi.py          # BatchAnalyzer - main analysis orchestrator
│   ├── spatial.py      # Spatial blob detection algorithms
│   ├── pipeline.py     # Analysis pipeline coordination
│   └── network.py      # Network analysis and graph theory
├── visualization/
│   ├── main.py         # VisualizationPipeline - orchestrates all figures
│   ├── temporal.py     # Time-series visualizations
│   ├── condition.py    # Condition comparison plots
│   ├── replicate.py    # Replicate variance analysis
│   ├── network.py      # Network visualizations
│   └── components.py   # Shared visualization utilities
├── utils/
│   ├── helpers.py      # Metadata classes and utilities
│   └── config.py       # Configuration loading and validation
└── config.py           # Main configuration class
```

## Development Principles

### Configuration-Driven Design
- All parameters externalized to `config.json`
- `Config` class provides single source of truth
- No hardcoded values in analysis or visualization modules

### Data Flow Architecture
1. **Analysis**: `BatchAnalyzer` → processes all ROIs → `analysis_results.json`
2. **Visualization**: `VisualizationPipeline` → loads results → generates figures
3. **Integration**: Network analysis conditionally triggered during visualization

### Extensibility Patterns
- **Strategy Pattern**: `NetworkAnalyzer` base class for different analysis approaches
- **Factory Pattern**: `VisualizationPipeline` creates appropriate visualizers
- **Data Classes**: Clean separation of data structures and processing logic

# Important Instructions
- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing existing files to creating new ones
- NEVER proactively create documentation files