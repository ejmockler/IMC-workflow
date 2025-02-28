# Steinbock IMC Data Analysis Pipeline

This repository contains an R-based analysis pipeline for Imaging Mass Cytometry (IMC) data, specifically designed to work with outputs from the [Steinbock](https://github.com/BodenmillerGroup/steinbock) preprocessing pipeline. The workflow encompasses data import, annotation, processing, visualization, and advanced analysis of IMC data.

## Pipeline Overview

```mermaid
flowchart TD

    %% -- Data Import & Annotation --
    subgraph Import[Data Import & Annotation]
        direction TB
        A1[importSteinbockData]:::inputNode
        A2[annotateSteinbockData]:::inputNode
        A3[processSteinbockData]:::inputNode
        A4[processImageData]:::inputNode
        
        A1 --> A2 --> A3
        A1 --> A4
    end

    %% -- Advanced Analysis --
    subgraph Analysis[Advanced Analysis]
        direction TB
        B1[batchCorrection]:::analysisNode
        B2[cellPhenotyping]:::analysisNode
        B3[cellPhenotyping_noSegmentation]:::analysisNode
        
        B1 --> B2
        A3 --> B1
        A4 --> B3
    end

    %% -- Visualization --
    subgraph Visual[Visualization Pipeline]
        direction TB
        C1[Marker Intensity Visualization]:::visualNode
        C2[Cell Type Overlay]:::visualNode
        C3[Spatial Analysis Plots]:::visualNode
        
        C1 --> C2 --> C3
    end

    %% -- Output --
    subgraph Output[Analysis Output]
        direction TB
        D1[Cell Type Annotations]:::outputNode
        D2[Marker Correlations]:::outputNode
        D3[Reports & Figures]:::outputNode
        
        D1 --> D3
        D2 --> D3
    end

    %% Connections between phases
    A3 --> C1
    A4 --> C1
    B2 --> D1
    B3 --> D2
    C3 --> D3

    %% Style Definitions
    classDef inputNode fill:#d0e8f2,stroke:#0066cc,stroke-width:2px,color:#333333;
    classDef analysisNode fill:#fff2cc,stroke:#d6b656,stroke-width:2px,color:#333333;
    classDef visualNode fill:#ffe6cc,stroke:#d79b00,stroke-width:2px,color:#333333;
    classDef outputNode fill:#d5e8d4,stroke:#2d6a4f,stroke-width:2px,color:#333333;
```


## Pipeline Components

### Data Import & Annotation

1. **importSteinbockData.R**
   - Imports processed single-cell data from the Steinbock pipeline
   - Loads multi-channel images and segmentation masks
   - Outputs SpatialExperiment object (spe), images, and masks objects

2. **annotateSteinbockData.R**
   - Loads the previously imported SpatialExperiment object
   - Assigns unique cell identifiers
   - Merges external metadata using configuration-driven file paths

3. **processSteinbockData.R**
   - Applies transformations to count data (asinh transformation)
   - Conducts quality control filtering based on cell area
   - Flags channels for downstream analysis
   - Runs dimensionality reduction (PCA, UMAP, t-SNE)

4. **processImageData.R**
   - Loads and processes multi-channel images and segmentation masks
   - Sets channel names from the SpatialExperiment object
   - Attaches masks to images and performs quality control checks

### Advanced Analysis

5. **batchCorrection.R**
   - Performs batch correction using fastMNN method
   - Integrates low-dimensional embeddings for downstream analysis
   - Provides visualizations to assess correction effectiveness

6. **cellPhenotyping.R**
   - Applies clustering algorithms (Rphenoannoy/Rphenograph) to identify cell phenotypes
   - Works with the batch-corrected SpatialExperiment object
   - Creates annotated cells with cluster assignments

7. **cellPhenotyping_noSegmentation.R**
   - Analyzes marker relationships directly from pixel data
   - Provides alternative analysis when cell segmentation may introduce bias
   - Uses multiple analytical techniques while preserving image context

### Core Infrastructure

The pipeline is built on a robust infrastructure that includes:

- **ConfigurationManager**: Handles configuration settings and defaults
- **Logger**: Provides structured logging across all pipeline components
- **DependencyManager**: Manages package dependencies and environment validation
- **ProgressTracker**: Tracks analysis progress and provides execution summaries
- **ResultsManager**: Handles storage and export of analysis results
- **MetadataHarmonizer**: Merges external metadata into the SpatialExperiment object

### Visualization

The `VisualizationFunctions.R` module provides comprehensive visualization capabilities:

- Intensity metric heatmaps
- Channel distribution plots
- Spatial hotspot overlays
- Cell phenotype visualizations
- Comprehensive marker analysis heatmaps

## Getting Started

### Prerequisites

- R 4.0.0 or higher
- Bioconductor packages including SpatialExperiment, cytomapper, imcRtools
- Visualization packages: ggplot2, ComplexHeatmap, viridis

### Basic Usage

1. **Setup configuration**:
   - Edit configuration settings in `config.yml` or create a custom configuration

2. **Import data**:
   ```R
   source("src/entrypoints/importSteinbockData.R")
   data_objects <- runImportSteinbockData()
   ```

3. **Annotate data**:
   ```R
   source("src/entrypoints/annotateSteinbockData.R")
   spe_annotated <- runAnnotateSteinbockData()
   ```

4. **Process data**:
   ```R
   source("src/entrypoints/processSteinbockData.R")
   spe_processed <- runProcessSteinbockData()
   ```

5. **Run analysis**:
   ```R
   source("src/entrypoints/batchCorrection.R")
   spe_corrected <- runBatchCorrection()
   
   source("src/entrypoints/cellPhenotyping.R")
   spe_phenotyped <- runCellPhenotyping()
   ```

## Alternative Workflows

### Segmentation-Free Analysis

For datasets where cell segmentation may be problematic:

```R
source("src/entrypoints/cellPhenotyping_noSegmentation.R")
analyzer <- runMarkerAnalysisNoSegmentation()
```


### Image Processing

To work directly with the multi-channel images:

```R
source("src/entrypoints/processImageData.R")
images <- runProcessImageData()
```


## Output

The pipeline generates:

- Processed SpatialExperiment objects at various stages
- Visualizations of cell phenotypes and marker expression
- Comprehensive analysis reports
- Quality control metrics

All outputs are saved to the configured output directory (default: `output/`).
