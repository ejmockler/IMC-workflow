# Python Pixel-wise IMC Analysis Pipeline

This repository contains a Python-based analysis pipeline for Imaging Mass Cytometry (IMC) data, specifically designed for segmentation-free, pixel-level analysis. It processes IMC data files (typically `.txt` format from preprocessing tools) to perform spatial clustering across **multiple resolutions**, community analysis, differential expression, and visualization without relying on prior cell segmentation.

## Pipeline Overview

The pipeline takes individual ROI (Region of Interest) files as input and performs the following major steps for each ROI in parallel:

1.  **Load & Validate Data**: Reads pixel data, validates against expected channels, extracts ROI identifier.
2.  **Calculate Cofactors**: Determines optimal arcsinh cofactors for transformation.
3.  **Preprocess**: Applies arcsinh transformation and scaling to pixel expression data.
4.  **Generate Resolution-Independent Plots**: Creates pixel correlation clustermap and raw spatial expression grids (saved to main ROI directory). Determines channel order from pixel clustermap.
5.  **Loop Through Resolutions**: For each resolution parameter defined in the configuration:
    *   **Spatial Clustering**: Builds a spatial graph (KNN) and performs Leiden clustering using the current resolution.
    *   **Community Analysis**: Calculates average expression profiles and differential expression for the current resolution's communities.
    *   **Generate Resolution-Dependent Plots**: Creates community correlation clustermap, UMAP plot, and the combined scaled-pixel/average-community co-expression matrix (ordered by pixel clustermap). Saves these plots and community analysis CSVs to a resolution-specific subdirectory.
    *   **Save Results**: Saves the final combined pixel results DataFrame for the current resolution.

```mermaid
flowchart TD
    subgraph Input [Input Data]
        direction TB
        In1[ROI .txt Files]:::inputNode
        In2[config.yaml]:::inputNode
    end

    subgraph Pipeline [run_pixel_pipeline.py]
        direction TB
        P1(Load & Validate):::processNode --> P2(Calculate Cofactors):::processNode
        P2 --> P3(Preprocess Data):::processNode
        P3 --> P4(Independent Viz & Get Order):::visualNode
        P4 --> Loop(Resolution Loop):::loopNode
        subgraph Loop
            direction TB
            L1(Cluster Pixels):::analysisNode
            L1 --> L2(Analyze Communities):::analysisNode
            L2 --> L3(Dependent Viz):::visualNode
            L3 --> L4(Save Resolution Results):::outputNode
        end

        %% Module Interactions
        P1 -- uses --> M1[imc_data_utils.py]
        P2 -- uses --> M1
        P3 -- uses --> M1
        P4 -- uses --> M3[pixel_visualization.py]
        L1 -- uses --> M2[pixel_analysis_core.py]
        L2 -- uses --> M2
        L3 -- uses --> M3

    end

     subgraph Output [Output]
         direction TB
         O1[ROI-Level Plots (SVG)]:::outputNode
         O2[Resolution Subdirs containing CSVs & Plots (SVG)]:::outputNode
     end

    %% Connections
    In1 --> Pipeline
    In2 --> Pipeline
    Pipeline --> O1
    Pipeline --> O2

    %% Style Definitions
    classDef inputNode fill:#d0e8f2,stroke:#0066cc,stroke-width:2px,color:#333333;
    classDef processNode fill:#fff7e6,stroke:#d49205,stroke-width:2px,color:#333333;
    classDef analysisNode fill:#fff2cc,stroke:#d6b656,stroke-width:2px,color:#333333;
    classDef visualNode fill:#ffe6cc,stroke:#d79b00,stroke-width:2px,color:#333333;
    classDef outputNode fill:#d5e8d4,stroke:#2d6a4f,stroke-width:2px,color:#333333;
    classDef loopNode fill:#f0f0f0,stroke:#555,stroke-width:1px,color:#333333;
```

## Prerequisites

*   Python 3.8+
*   Required CPU Python packages listed in `requirements.txt`. Install them using:

    ```bash
    pip install -r requirements.txt
    ```
    *(This includes `faiss-cpu` for CPU-based ANN and `leidenalg` for CPU-based Leiden clustering)*

*   **Optional GPU Acceleration:**
    *   **For k-NN (Faiss):** To enable GPU-accelerated Approximate Nearest Neighbors using Faiss, you need a compatible `faiss-gpu` installation. 
        *   Pre-built `conda` packages are available for specific CUDA versions (e.g., 11.4, 12.1 via `pytorch` channel, 11.8, 12.4 via `pytorch` channel with `faiss-gpu-cuvs`). Consult the [Official Faiss Installation Guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for the correct command based on your CUDA version.
        *   If a pre-built package is not available for your CUDA version (like 12.7+), you would need to [build Faiss from source](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#building-from-source) with GPU support enabled.
        *   **The pipeline will automatically attempt to use an installed `faiss-gpu` if `use_gpu: true` is set in `config.yaml`.** If it fails or `faiss-gpu` is not detected, it falls back to other methods (cuML, faiss-cpu, KDTree).
    *   **For Leiden Clustering (RAPIDS):** To enable GPU-accelerated Leiden clustering (and potentially k-NN fallback via cuML), install the RAPIDS suite (`cudf`, `cugraph`, `cuml`, `cupy`). Use the official [RAPIDS Release Selector](https://docs.rapids.ai/install/#selector) to get the correct installation command for your environment (OS, CUDA version, desired RAPIDS version).
        *   **The pipeline will attempt to use `cugraph` for Leiden if RAPIDS libraries are detected and `use_gpu: true` is set.** If it fails or RAPIDS is not detected, it falls back to CPU `leidenalg`.

*   **(Deprecated)** The `requirements-gpu.txt` file is retained for reference but pip-installing from it is no longer recommended for RAPIDS components.

## Code Structure

*   **`run_pixel_pipeline.py`**: The main executable script. Orchestrates the workflow, loads config, runs analysis per ROI (looping through resolutions), and manages parallel processing.
*   **`config.yaml`**: Configuration file for setting paths, parameters, and options.
*   **`src/roi_pipeline/imc_data_utils.py`**: Utilities for data loading, validation, and preprocessing.
*   **`src/roi_pipeline/pixel_analysis_core.py`**: Core analysis functions (spatial graph using Faiss/cuML/KDTree, Leiden clustering using cuGraph/leidenalg, profiles, DiffEx).
*   **`src/roi_pipeline/pixel_visualization.py`**: Functions for generating plots (clustermaps, spatial grids, UMAP, co-expression matrix).
*   **`requirements.txt`**: Python package dependencies (including `faiss-cpu`).

## Configuration

All pipeline parameters are controlled via `config.yaml`. Key sections include:

*   **`paths`**: Input data directory (`data_dir`) and base output directory (`output_dir`).
*   **`data`**: Metadata columns, master protein channel list, default arcsinh cofactor.
*   **`analysis`**: 
    *   `clustering`: KNN neighbors, Leiden `resolution_params` (a list of resolutions to test), random seed.
    *   `umap`: Parameters for UMAP embedding.
    *   `differential_expression`: Markers to exclude from UMAP.
*   **`processing`**: 
    *   `use_gpu`: Set to `true` to attempt GPU acceleration (Faiss-GPU/RAPIDS), `false` to force CPU.
    *   `parallel_jobs`: Setting for parallel processing.
    *   `plot_dpi`: DPI for saved plots.
    *   `visualization`: Appearance settings for plots.

Modify `config.yaml`, especially `resolution_params` under `analysis.clustering`, before running.

## Usage

1.  Ensure prerequisites are installed (including `faiss-cpu` via `pip install -r requirements.txt`, and optionally `faiss-gpu` or RAPIDS if desired).
2.  Configure `config.yaml` (set `use_gpu: true` if you installed GPU components and want to use them).
3.  Run the main script:
    ```bash
    python run_pixel_pipeline.py
    ```

## GPU Acceleration Notes

The pipeline prioritizes GPU methods when `use_gpu: true` is set and compatible libraries are detected:

*   **k-NN Search Priority:** Faiss-GPU > cuML (RAPIDS) > Faiss-CPU > KDTree (SciPy)
*   **Leiden Clustering Priority:** cuGraph (RAPIDS) > leidenalg (CPU)

If a higher-priority method fails (e.g., `faiss-gpu` error due to CUDA incompatibility, missing library), the code automatically falls back to the next available method in the priority list.

## Output Description

Outputs are saved within the specified `output_dir`. A subdirectory is created for each ROI.

**Inside the main ROI subdirectory (`output_dir/ROI_X/`):**

*   `optimal_cofactors_<roi_string>.csv`: Cofactors used.
*   `pixel_channel_correlation_heatmap_spearman_<roi_string>.svg`: Pixel-level correlation clustermap.
*   `pixel_raw_expression_spatial_<roi_string>.svg`: Grid plot of raw spatial expression.

**Inside Resolution-Specific Subdirectories (`output_dir/ROI_X/resolution_Y_Z/`):**

*   **CSVs:**
    *   `pixel_scaled_results_<roi_string>_res_<resolution>.csv`: Intermediate scaled results with community assignment for this resolution.
    *   `community_avg_scaled_profiles_<roi_string>_res_<resolution>.csv`: Average community profiles.
    *   `community_diff_profiles_<roi_string>_res_<resolution>.csv`: Differential expression profiles.
    *   `community_primary_channels_<roi_string>_res_<resolution>.csv`: Top primary channel per community.
    *   `umap_coords_diff_profiles_<roi_string>_res_<resolution>.csv`: UMAP coordinates (if run).
    *   `pixel_results_annotated_<roi_string>_res_<resolution>.csv`: Per-pixel results enriched with asinh‑scaled values, community assignments, and primary channel annotations for this resolution.
*   **Plots (SVGs):**
    *   `community_channel_correlation_heatmap_spearman_<roi_string>_res_<resolution>.svg`: Community-level correlation clustermap.
    *   `umap_community_scatter_protein_markers_diff_profiles_<roi_string>_res_<resolution>.svg`: UMAP scatter plot of communities.
    *   `coexpression_matrix_scaled_vs_avg_<roi_string>_res_<resolution>.svg`: Matrix plot comparing scaled pixel (upper triangle) vs. mapped average community (lower triangle) co-expression, ordered by pixel correlation.
