# IMC Analysis Data Schema

**Version**: 1.0
**Last Updated**: 2025-11-09
**Purpose**: Document the structure of analysis results for loading, validation, and downstream use

---

## Result File Location

```
results/roi_results/roi_<roi_name>_results.json.gz
```

**Format**: Gzipped JSON with numpy arrays serialized as dictionaries
**Compression**: gzip (typically 10-20× size reduction)
**Encoding**: UTF-8

---

## Top-Level Structure

Each result file contains 5 top-level keys:

```json
{
  "multiscale_results": {...},      // Primary analysis results at 3 scales
  "consistency_results": {...},     // Cross-scale validation metrics
  "configuration_used": {...},      // Config snapshot for reproducibility
  "metadata": {...},                // ROI metadata (timepoint, mouse, etc.)
  "roi_id": "string"                // Unique ROI identifier
}
```

---

## 1. multiscale_results

**Structure**: Dictionary with keys `"10.0"`, `"20.0"`, `"40.0"` (scales in μm)

Each scale contains:

### Core Analysis Products

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `features` | array | `[n_superpixels, n_features]` | Aggregated ion count features (30 features from 11 markers) |
| `spatial_coords` | array | `[n_superpixels, 2]` | Superpixel centroids (x, y) in μm |
| `cluster_labels` | array | `[n_superpixels]` | Leiden community detection labels (0 to n_clusters-1) |
| `superpixel_labels` | array | `[height, width]` | Pixel-level segmentation map |
| `superpixel_coords` | array | `[n_superpixels, 2]` | Same as spatial_coords (redundant) |

### Transformed Marker Arrays

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `transformed_arrays` | dict | `{marker: array[n_superpixels]}` | Arcsinh-transformed marker intensities per superpixel |
| `cofactors_used` | dict | `{marker: float}` | Cofactor values used for arcsinh transformation |

**Markers in transformed_arrays**:
- **Immune**: CD45, CD11b, Ly6G, CD206
- **Stromal**: CD140a, CD140b, CD44
- **Vascular**: CD31, CD34
- **DNA**: 130Ba, 131Xe (iridium intercalators)

### Clustering Metadata

| Field | Type | Description |
|-------|------|-------------|
| `clustering_info` | dict | Algorithm parameters, n_clusters, modularity, etc. |
| `stability_analysis` | dict | Resolution sweep results (0.1 to 2.0) |
| `spatial_coherence` | float | Moran's I statistic for spatial autocorrelation |

**clustering_info keys**:
- `algorithm`: "leiden"
- `resolution`: Optimal resolution selected
- `n_clusters`: Number of communities detected
- `modularity`: Network modularity score
- `iterations`: Convergence iterations
- `cluster_sizes`: Array of cluster sizes
- `cluster_balance`: CV of cluster sizes (lower = more balanced)

**stability_analysis keys**:
- `resolutions`: Array of tested resolutions
- `n_clusters_per_resolution`: Array of cluster counts
- `modularity_per_resolution`: Array of modularity scores
- `optimal_resolution`: Selected resolution
- `optimal_modularity`: Modularity at optimal resolution
- `mean_n_clusters`: Average clusters across resolutions
- `std_n_clusters`: Standard deviation of cluster counts
- `coefficient_of_variation`: CV of cluster counts
- `resolution_range`: Tested range

### Segmentation Products

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `composite_dna` | array | `[height, width]` | DNA channel composite (130Ba + 131Xe) |
| `bounds` | list | `[x_min, y_min, x_max, y_max]` | ROI bounding box in μm |

### Scale Metadata

| Field | Type | Description |
|-------|------|-------------|
| `scale_um` | float | Scale in micrometers (10.0, 20.0, or 40.0) |
| `method` | str | "slic_multiscale" |
| `segmentation_method` | str | "slic" |

---

## 2. consistency_results

Cross-scale validation metrics:

| Field | Type | Description |
|-------|------|-------------|
| `consistency_score` | float | Overall consistency metric (0-1) |
| `scale_correlations` | dict | Pairwise scale correlations |
| `cluster_stability` | dict | Cluster stability across scales |

---

## 3. configuration_used

Snapshot of `config.json` at analysis time for reproducibility:

```json
{
  "project_root": "/path/to/project",
  "data_dir": "data/241218_IMC_Alun",
  "marker_panel": [...],
  "slic_params": {...},
  "clustering_params": {...},
  ...
}
```

---

## 4. metadata

ROI-specific metadata extracted from filename or config:

```json
{
  "timepoint": "D1" | "D3" | "D7" | "Sham",
  "mouse": "M1" | "M2",
  "condition": "UUO" | "Sham",
  "roi_number": "01",
  "replicate": "9"
}
```

**Filename pattern**: `roi_IMC_241218_Alun_ROI_{timepoint}_{mouse}_{roi_number}_{replicate}_results.json.gz`

---

## 5. roi_id

**Type**: String
**Format**: `"IMC_241218_Alun_ROI_{timepoint}_{mouse}_{roi_number}_{replicate}"`
**Example**: `"IMC_241218_Alun_ROI_D1_M1_01_9"`

---

## Numpy Array Serialization Format

Arrays are serialized as dictionaries with the following structure:

```json
{
  "__numpy_array__": true,
  "dtype": "float64" | "int64" | "uint16",
  "shape": [dim1, dim2, ...],
  "data": [flat_array_values]
}
```

**Deserialization**:
```python
def deserialize_array(arr_dict):
    """Convert serialized numpy array back to numpy array"""
    if isinstance(arr_dict, dict) and '__numpy_array__' in arr_dict:
        return np.array(
            arr_dict['data'],
            dtype=arr_dict['dtype']
        ).reshape(arr_dict['shape'])
    return arr_dict
```

---

## Loading Results: Canonical Pattern

### Basic Loading

```python
import gzip
import json
import numpy as np
from pathlib import Path

def load_roi_results(roi_file):
    """Load a single ROI result file"""
    with gzip.open(roi_file, 'rt') as f:
        return json.load(f)

# Load single ROI
roi_path = Path('results/roi_results/roi_IMC_241218_Alun_ROI_D1_M1_01_9_results.json.gz')
results = load_roi_results(roi_path)
```

### Accessing Scale-Specific Data

```python
# Get 10μm scale results
scale_10 = results['multiscale_results']['10.0']

# Extract arrays
cluster_labels = deserialize_array(scale_10['cluster_labels'])
spatial_coords = deserialize_array(scale_10['spatial_coords'])

# Extract marker data
cd45 = deserialize_array(scale_10['transformed_arrays']['CD45'])
cd11b = deserialize_array(scale_10['transformed_arrays']['CD11b'])
```

### Loading All ROIs

```python
def load_all_rois(results_dir='results/roi_results'):
    """Load all ROI results into memory"""
    results_dir = Path(results_dir)
    all_results = {}

    for roi_file in sorted(results_dir.glob('roi_*.json.gz')):
        roi_id = roi_file.stem.replace('roi_', '').replace('_results', '')
        all_results[roi_id] = load_roi_results(roi_file)

    return all_results

# Usage
all_rois = load_all_rois()
print(f"Loaded {len(all_rois)} ROIs")
```

### Building Analysis DataFrames

```python
def build_superpixel_dataframe(results_dict, scale='10.0'):
    """Convert results to pandas DataFrame for analysis"""
    import pandas as pd

    all_rows = []

    for roi_id, results in results_dict.items():
        scale_data = results['multiscale_results'][scale]
        metadata = results['metadata']

        # Deserialize arrays
        coords = deserialize_array(scale_data['spatial_coords'])
        clusters = deserialize_array(scale_data['cluster_labels'])

        # Extract marker data
        markers = {}
        for marker, arr_dict in scale_data['transformed_arrays'].items():
            if marker not in ['130Ba', '131Xe']:  # Skip DNA
                markers[marker] = deserialize_array(arr_dict)

        # Build rows
        n_superpixels = len(coords)
        for i in range(n_superpixels):
            row = {
                'roi': roi_id,
                'timepoint': metadata['timepoint'],
                'mouse': metadata['mouse'],
                'condition': metadata['condition'],
                'superpixel_id': i,
                'x': coords[i, 0],
                'y': coords[i, 1],
                'cluster': int(clusters[i])
            }

            # Add marker values
            for marker, values in markers.items():
                row[marker] = values[i]

            all_rows.append(row)

    return pd.DataFrame(all_rows)

# Usage
df = build_superpixel_dataframe(all_rois, scale='10.0')
print(df.head())
```

---

## Data Validation

### Expected Invariants

1. **Array shapes must match**:
   ```python
   assert len(cluster_labels) == len(spatial_coords)
   assert len(cluster_labels) == len(cd45_values)
   ```

2. **Cluster labels are sequential**:
   ```python
   unique_clusters = np.unique(cluster_labels)
   assert np.array_equal(unique_clusters, np.arange(len(unique_clusters)))
   ```

3. **Spatial coordinates are positive**:
   ```python
   assert (spatial_coords >= 0).all()
   ```

4. **Transformed values are non-negative** (arcsinh ensures this):
   ```python
   for marker, values in transformed_arrays.items():
       assert (values >= 0).all(), f"{marker} has negative values"
   ```

5. **All scales present**:
   ```python
   assert set(results['multiscale_results'].keys()) == {'10.0', '20.0', '40.0'}
   ```

### Validation Function

```python
def validate_result_file(results):
    """Validate result file structure and invariants"""

    # Check top-level keys
    required_keys = {'multiscale_results', 'metadata', 'roi_id'}
    assert required_keys.issubset(results.keys()), f"Missing keys: {required_keys - results.keys()}"

    # Check scales
    assert set(results['multiscale_results'].keys()) == {'10.0', '20.0', '40.0'}

    # Validate each scale
    for scale, scale_data in results['multiscale_results'].items():
        # Deserialize key arrays
        clusters = deserialize_array(scale_data['cluster_labels'])
        coords = deserialize_array(scale_data['spatial_coords'])

        # Check shapes match
        n_superpixels = len(clusters)
        assert len(coords) == n_superpixels, f"Coord length mismatch at scale {scale}"

        # Check marker arrays
        for marker, arr_dict in scale_data['transformed_arrays'].items():
            values = deserialize_array(arr_dict)
            assert len(values) == n_superpixels, f"{marker} length mismatch at scale {scale}"
            assert (values >= 0).all(), f"{marker} has negative values at scale {scale}"

    return True
```

---

## Usage Examples

### Example 1: Extract Marker Profiles by Cluster

```python
def get_cluster_profiles(results, scale='10.0'):
    """Get mean marker expression per cluster"""
    scale_data = results['multiscale_results'][scale]

    clusters = deserialize_array(scale_data['cluster_labels'])
    markers = {
        name: deserialize_array(arr)
        for name, arr in scale_data['transformed_arrays'].items()
        if name not in ['130Ba', '131Xe']
    }

    # Build DataFrame
    df = pd.DataFrame(markers)
    df['cluster'] = clusters

    # Group by cluster
    profiles = df.groupby('cluster').mean()

    return profiles

# Usage
profiles = get_cluster_profiles(results)
print(profiles[['CD45', 'CD31', 'CD140b']])
```

### Example 2: Temporal Analysis

```python
def compare_timepoints(all_results, marker='CD44', scale='10.0'):
    """Compare marker expression across timepoints"""
    timepoint_data = {}

    for roi_id, results in all_results.items():
        tp = results['metadata']['timepoint']
        scale_data = results['multiscale_results'][scale]

        marker_values = deserialize_array(scale_data['transformed_arrays'][marker])

        if tp not in timepoint_data:
            timepoint_data[tp] = []
        timepoint_data[tp].extend(marker_values)

    # Compute statistics
    stats = {}
    for tp, values in timepoint_data.items():
        stats[tp] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'n_superpixels': len(values)
        }

    return pd.DataFrame(stats).T

# Usage
temporal_stats = compare_timepoints(all_results, marker='CD44')
print(temporal_stats)
```

### Example 3: Spatial Analysis

```python
def plot_marker_spatial_map(results, marker='CD45', scale='10.0'):
    """Plot marker expression in spatial context"""
    import matplotlib.pyplot as plt

    scale_data = results['multiscale_results'][scale]

    coords = deserialize_array(scale_data['spatial_coords'])
    values = deserialize_array(scale_data['transformed_arrays'][marker])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1],
        c=values, cmap='viridis',
        s=50, alpha=0.7
    )
    plt.colorbar(scatter, label=f'{marker} (arcsinh)')
    plt.xlabel('X (μm)')
    plt.ylabel('Y (μm)')
    plt.title(f'{marker} Spatial Distribution ({scale}μm scale)')
    plt.axis('equal')
    plt.show()

# Usage
plot_marker_spatial_map(results, marker='CD45', scale='10.0')
```

---

## Performance Considerations

### Memory Usage

- Single ROI file (compressed): ~100KB - 2MB
- Single ROI file (uncompressed): ~1MB - 20MB
- Loading all 18 ROIs: ~50-150MB RAM

**Recommendation**: Load selectively if memory-constrained:
```python
# Load only specific scale
def load_single_scale(roi_file, scale='10.0'):
    with gzip.open(roi_file, 'rt') as f:
        full_data = json.load(f)

    return {
        'scale_data': full_data['multiscale_results'][scale],
        'metadata': full_data['metadata'],
        'roi_id': full_data['roi_id']
    }
```

### I/O Performance

- Gzip decompression: ~10-50ms per file
- JSON parsing: ~50-200ms per file
- Total load time: ~100-300ms per ROI

**Optimization**: Use multiprocessing for batch loading:
```python
from multiprocessing import Pool

def parallel_load_rois(roi_files, n_workers=4):
    with Pool(n_workers) as pool:
        results = pool.map(load_roi_results, roi_files)
    return {f.stem: r for f, r in zip(roi_files, results)}
```

---

## Related Documentation

- **Pipeline**: See `docs/architecture/ARCHITECTURE.md` for how results are generated
- **Methods**: See `METHODS.md` for scientific methodology
- **Analysis**: See `notebooks/biological_narratives/kidney_injury_spatial_analysis.ipynb` for usage examples

---

**Schema Version**: 1.0
**Maintained by**: IMC Analysis Pipeline
**Questions**: See `CLAUDE.md` for development guidance
