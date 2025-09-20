# IMC Visualization Guide

## Overview

This guide explains how to visualize IMC analysis results using Jupyter notebooks and the lightweight `viz_utils` library. The architecture follows a clear separation between analysis (which produces data products) and visualization (which consumes them).

## Architecture Philosophy

### Analysis vs Visualization
- **Analysis Pipeline**: Produces standardized data products (HDF5, Parquet, JSON)
- **Visualization Layer**: Jupyter notebooks that load and visualize these products
- **Benefits**: 
  - Rapid iteration for paper figures
  - Experiment-specific customization
  - Reduced codebase complexity (~5000 lines removed)

## Getting Started

### 1. Run Analysis First
```bash
# Generate analysis outputs
python run_analysis.py --config config.json

# Or for parallel processing
python run_parallel_analysis.py --processes 8
```

This creates standardized outputs in `results/`:
- `analysis_results.json` - Complete analysis results
- `roi_results/` - Per-ROI detailed data
- `validation/` - Validation outputs

### 2. Use Notebook Templates

Start with provided templates in `notebooks/templates/`:

- **01_data_exploration.ipynb** - Basic data loading and protein visualization
- **02_spatial_analysis.ipynb** - Spatial patterns and multiscale analysis

Copy a template to create your own analysis:
```bash
cp notebooks/templates/01_data_exploration.ipynb notebooks/my_analysis.ipynb
```

### 3. Import Visualization Utilities

```python
import sys
sys.path.append('../..')  # Add project root

from src.viz_utils import (
    plot_roi_overview,
    plot_protein_expression,
    plot_cluster_map,
    load_roi_results,
    load_protein_data
)
```

## Available Functions

### Plotting Functions (`src.viz_utils.plotting`)

#### `plot_roi_overview(coords, values, **kwargs)`
Basic spatial visualization of any value across ROI pixels.
```python
fig = plot_roi_overview(coords, protein_data['CD45'], 
                       title='CD45 Expression', cmap='hot')
```

#### `plot_protein_expression(protein_data, coords, **kwargs)`
Grid visualization of multiple proteins.
```python
fig = plot_protein_expression(protein_data, coords, 
                            ncols=3, figsize_per_plot=(5,5))
```

#### `plot_cluster_map(coords, cluster_labels, **kwargs)`
Visualize spatial distribution of clusters.
```python
fig = plot_cluster_map(coords, cluster_labels, show_legend=True)
```

#### `plot_spatial_heatmap(coords, values, bin_size, **kwargs)`
Create binned heatmaps at specified resolution.
```python
fig = plot_spatial_heatmap(coords, values, bin_size=20.0)
```

#### `plot_scale_comparison(multiscale_results, coords, **kwargs)`
Compare results across multiple spatial scales.
```python
fig = plot_scale_comparison(multiscale_results, coords)
```

### Data Loading Functions (`src.viz_utils.loaders`)

#### `load_roi_results(results_path, roi_name=None)`
Load analysis results from JSON output.
```python
results = load_roi_results('results/analysis_results.json')
```

#### `load_protein_data(roi_file, protein_names=None)`
Load protein expression from ROI TSV file.
```python
proteins = load_protein_data('data/ROI_001.txt')
```

#### `load_coordinates(roi_file)`
Load X,Y coordinates from ROI file.
```python
coords = load_coordinates('data/ROI_001.txt')
```

#### `load_multiscale_results(results_path, roi_name=None)`
Load multiscale analysis results.
```python
multiscale = load_multiscale_results('results/analysis.json')
```

## Common Visualization Patterns

### 1. Single Protein Visualization
```python
# Load data
coords = load_coordinates(roi_file)
proteins = load_protein_data(roi_file)

# Plot single protein
fig = plot_roi_overview(coords, proteins['CD45'], 
                       title='CD45', cmap='viridis')
```

### 2. Multi-Protein Comparison
```python
# Plot grid of proteins
fig = plot_protein_expression(proteins, coords,
                            proteins_to_plot=['CD45', 'CD11b', 'CD206'],
                            ncols=3)
```

### 3. Cluster Analysis
```python
# Load clustering results
results = load_roi_results('results/analysis.json')
cluster_labels = results['cluster_labels']

# Visualize clusters
fig = plot_cluster_map(coords, cluster_labels)
```

### 4. Scale Comparison
```python
# Load multiscale results
multiscale = load_multiscale_results('results/analysis.json')

# Compare across scales
fig = plot_scale_comparison(multiscale, coords,
                          scales_to_plot=[10, 20, 40])
```

### 5. Custom Overlays
```python
# Create threshold-based overlay
threshold = np.percentile(proteins['CD45'], 75)
cd45_high = proteins['CD45'] > threshold

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(coords[cd45_high, 0], coords[cd45_high, 1], 
          c='red', s=1, label='CD45+')
ax.scatter(coords[~cd45_high, 0], coords[~cd45_high, 1],
          c='gray', s=1, alpha=0.3, label='CD45-')
ax.legend()
```

## Publication-Quality Figures

### High-Resolution Export
```python
# Create figure
fig = plot_roi_overview(coords, values, figsize=(10, 10))

# Save as PNG (raster)
fig.savefig('figure.png', dpi=300, bbox_inches='tight')

# Save as PDF (vector)
fig.savefig('figure.pdf', bbox_inches='tight')
```

### Consistent Styling
```python
# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')

# Or custom settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.dpi': 100
})
```

## Best Practices

### 1. Organize Your Notebooks
```
notebooks/
├── exploration/        # Initial data exploration
├── figures/           # Publication figure generation
└── supplementary/     # Supplementary analyses
```

### 2. Version Control Notebooks
- Commit notebooks with outputs cleared (Cell → All Output → Clear)
- Use `.gitignore` for large intermediate files
- Document key findings in markdown cells

### 3. Reproducibility
- Always set random seeds when needed
- Document package versions in first cell
- Use relative paths from notebook location

### 4. Performance Tips
- Use `rasterized=True` for large scatter plots
- Load only needed columns from large files
- Cache processed data for reuse

## Migrating from Old Visualization Code

If you have code using the old `VisualizationPipeline`:

**Old approach:**
```python
from src.visualization import VisualizationPipeline
viz = VisualizationPipeline(config)
fig = viz.create_roi_figure(roi_data)
```

**New approach:**
```python
from src.viz_utils import plot_roi_overview
fig = plot_roi_overview(coords, values)
```

Key differences:
- Functions are stateless (no class instances)
- Direct data passing (no complex wrappers)
- Full control over customization

## Extending the Utilities

To add new visualization functions:

1. Add function to `src/viz_utils/plotting.py`
2. Keep it stateless and focused
3. Accept data as numpy arrays or dicts
4. Return matplotlib figure or axes
5. Update `__init__.py` exports

Example template:
```python
def plot_my_visualization(
    data: np.ndarray,
    title: str = "My Plot",
    ax: Optional[plt.Axes] = None
) -> Union[plt.Figure, plt.Axes]:
    \"\"\"One-line description.
    
    Args:
        data: Input data
        title: Plot title
        ax: Existing axes (optional)
    
    Returns:
        Figure or Axes object
    \"\"\"
    if ax is None:
        fig, ax = plt.subplots()
        return_fig = True
    else:
        return_fig = False
    
    # Your plotting code here
    ax.plot(data)
    ax.set_title(title)
    
    return fig if return_fig else ax
```

## Support

For questions about visualization:
1. Check notebook templates for examples
2. Review this guide
3. Examine `viz_utils` docstrings
4. Create custom functions as needed

Remember: The goal is flexibility and rapid iteration, not rigid pipelines!