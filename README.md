# IMC Analysis Pipeline v2.0

## Architecture Overview

```
IMC/
├── src/
│   ├── analysis/       # Spatial blob detection & network analysis
│   ├── visualization/  # Publication-quality figures
│   └── utils/         # Configuration & helpers
├── scripts/
│   └── experiments/   # Study-specific analysis
├── data/             # Input IMC data files
└── results/          # Generated outputs (ignored by git)
```

## Pipeline Control Flow

```
1. run_analysis.py
   └── BatchAnalyzer → spatial blob analysis → analysis_results.json
   
2. run_visualization.py  
   └── VisualizationPipeline → loads analysis_results.json → generates figures
   
3. scripts/experiments/kidney_healing/run_full_report.py
   └── Complete kidney study analysis & publication figures
```

## Quick Start

```bash
# 1. Complete analysis pipeline
python run_analysis.py         # → results/analysis_results.json

# 2. Generate standard visualizations  
python run_visualization.py    # → results/*.png files

# 3. Kidney healing study (complete workflow)
python scripts/experiments/kidney_healing/run_full_report.py
```

## What Each Stage Does

### Analysis Stage (`run_analysis.py`)
**Input**: IMC data files in `data/`
**Process**:
- Spatial blob detection and characterization
- Protein colocalization analysis using Pearson correlation
- Contact mapping between tissue domains
- Metadata extraction (timepoint, condition, region, replicate)

**Output**: `results/analysis_results.json` (structured analysis data)

### Visualization Stage (`run_visualization.py`)
**Input**: `results/analysis_results.json`
**Process**:
- Loads analysis results and reconstructs metadata objects
- Generates publication-quality figures via `VisualizationPipeline`
- Includes automatic network analysis if sufficient colocalization data
- Creates both individual ROI and aggregate visualizations

**Outputs**:
- `temporal_analysis.png` - Progression over experimental timepoints
- `condition_analysis.png` - Experimental condition comparison  
- `replicate_variance_analysis.png` - Biological reproducibility
- `region_time_grid_analysis.png` - Region×Time contact matrices
- `network_analysis_comprehensive.png` - Protein interaction networks

### Experiment-Specific Analysis (`scripts/experiments/[study_name]/`)
**Purpose**: Customized workflows for specific research studies
**Process**: 
- Runs generalized analysis pipeline with study-specific parameters
- Generates experiment-tailored figures with specialized formatting
- Includes study-relevant visualizations (temporal, comparative, variance analysis)
- Outputs study-specific quantitative metrics and summaries

## Configuration System

### Main Config (`config.json`)
Controls all pipeline parameters for any experimental design:
- **Proteins**: Proteins to analyze and user-defined functional groupings
- **Spatial parameters**: Blob detection thresholds and contact distance definitions  
- **Experimental design**: Timepoints, conditions, tissue regions (fully customizable)
- **Visualization**: Color schemes, figure dimensions, output formats
- **Network analysis**: Colocalization thresholds, centrality calculation parameters

### Configuration Propagation
`Config` class loads `config.json` → passed to all analysis/visualization modules → ensures consistent parameters across pipeline

## Computational Biology Methods

### Spatial Colocalization Analysis
**Biological Rationale**: Proteins that colocalize in tissue space often participate in related biological processes or signaling pathways. The analysis identifies spatial relationships to understand tissue organization and intercellular communication patterns.

**Mathematical Approach**:
- **Pearson Correlation**: Measures pixel-level intensity correlation between protein pairs
- **Spatial Blob Detection**: Groups pixels with similar protein expression profiles into tissue domains
- **Contact Mapping**: Quantifies interactions between different tissue domains
- **Statistical Validation**: Proper biological replicate handling ensures reproducible findings

### Network Analysis Integration

**Graph Theory Foundation**:
The pipeline automatically performs network analysis using established graph theory metrics:

- **Modularity**: Detects community structure in protein interaction networks
- **Clustering Coefficient**: Measures local network connectivity patterns  
- **Centrality Measures**: Identifies hub proteins using betweenness centrality
- **Network Density**: Quantifies overall connectivity patterns across conditions

**Biological Network Discovery**:
- **Spatial Communication Networks**: Protein-protein interactions derived from spatial colocalization patterns
- **Hub Protein Identification**: Central regulatory proteins identified via betweenness centrality metrics
- **Functional Group Analysis**: User-defined protein modules analyzed for intra- and inter-group connectivity
- **Cross-functional Communication**: Inter-module edges reveal communication between different biological processes

**Temporal Network Evolution**:  
Tracks dynamic biological processes across experimental timepoints:
- **Early timepoints**: Initial response patterns and primary network formation
- **Intermediate timepoints**: Network rewiring, hub protein transitions, connectivity changes
- **Late timepoints**: Resolution patterns, sustained network configurations

**Network Visualization Strategy**:
- Hub proteins sized by centrality metrics with configurable annotations
- Functional groups color-coded based on user-defined protein classifications
- Inter-group edges highlight cross-functional communication patterns
- Edge weights represent colocalization strength (Pearson correlation coefficients)

## Output Structure

```
results/
├── analysis_results.json           # Raw analysis data from BatchAnalyzer
├── temporal_analysis.png           # Time progression visualization
├── condition_analysis.png          # Condition comparison
├── replicate_variance_analysis.png # Biological variance analysis
├── region_time_grid_analysis.png   # Region×Time contact matrix
├── network_analysis_comprehensive.png # Complete network analysis
├── network_analysis_data.json      # Network metrics and hub proteins
└── [individual ROI figures].png    # Per-ROI visualizations
```

## Computational Reproducibility

### Performance Characteristics
- **Analysis**: ~4s per ROI (spatial blob detection + colocalization)
- **Batch Processing**: 25 ROIs analyzed in ~2 minutes  
- **Visualization**: ~30 seconds for all publication figures
- **Network Analysis**: Additional ~10 seconds (graph construction + metrics)

### Data Provenance & Reproducibility
**Philosophy**: All outputs are regenerated from source data, ensuring perfect reproducibility across computational environments.

**File Management Strategy**:
- **Source Control**: Only code (`src/`), configuration (`config.json`), and documentation tracked
- **Output Exclusion**: All analysis results, figures, and intermediate data automatically ignored by git
- **Local Generation**: Each user/environment generates outputs independently from identical source
- **Validation**: Use `python scripts/utilities/check_large_files.py` before commits to prevent accidental large file inclusion

**Scientific Reproducibility Benefits**:
- **Environment Independence**: Results identical regardless of local file system
- **Version Consistency**: Any git commit can regenerate exact same outputs
- **Collaboration**: Team members work with same source, generate local outputs
- **Publication**: Exact methods preserved in version control, data generated fresh

## Experiment-Specific Applications

### Study Customization
The pipeline supports experiment-specific analysis through:
- **Study-specific scripts**: Located in `scripts/experiments/[study_name]/`  
- **Custom visualizations**: Tailored figures for specific biological questions
- **Specialized metrics**: Study-relevant aggregations and statistical tests

### Example: Kidney Healing Study
The `scripts/experiments/kidney_healing/` demonstrates customization for temporal injury studies:
- **Timeline visualizations**: Progression across injury timepoints
- **Condition comparisons**: Treatment vs control analysis
- **Replicate variance**: Biological reproducibility assessment
- **Study-specific insights**: Biological process communication patterns relevant to the research question

## Key Technical Innovations

**Generalized Spatial Analysis**: 
Pearson correlation coefficients computed at pixel level with configurable biological replicate handling, applicable to any spatial proteomics dataset

**Flexible Network Integration**: 
Graph theory metrics integrated with spatial proteomics data, enabling discovery of tissue-level communication networks across biological systems

**Modular Architecture**:
Clear separation between generalized analysis (`src/analysis/`) and customizable visualization (`src/visualization/`) enables adaptation to different experimental designs

**Configuration-Driven Pipeline**: 
All experimental parameters externalized to `config.json` - proteins, timepoints, conditions, and visualization parameters easily modified for new studies

**Extensible Framework**:
Plugin architecture allows addition of new analysis modules and visualization approaches without modifying core pipeline