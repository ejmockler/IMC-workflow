# IMC Analysis Data Schema

**Version**: 2.1
**Last Updated**: 2026-04-21
**Purpose**: Document the structure of analysis results for loading, validation, and downstream use

**Scope (v2.1)**: This document covers two families of outputs and two configuration files.
- **Phase 1 per-ROI pipeline** (§1-§5): `results/roi_results/roi_*_results.json.gz`.
- **Phase 2 biological analysis** (§6-§7): `results/biological_analysis/cell_type_annotations/` (12-column parquet per ROI) and `results/biological_analysis/temporal_interfaces/` (17 parquets + `endpoint_summary.csv` + `run_provenance.json`).
- **Configuration** (§8): `config.json` (analysis knobs) + `viz.json` (display knobs).

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
| `features` | array | `[n_superpixels, n_features]` | Aggregated ion count features (30 features from 9 protein markers) |
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

## 6. Cell Type Annotations (Phase 2 biological analysis)

**Location**: `results/biological_analysis/cell_type_annotations/roi_<roi_id>_cell_types.parquet`
**Generator**: `batch_annotate_all_rois.py` → `src/analysis/cell_type_annotation.py::annotate_roi_from_results`
**One file per ROI** (24 files). Companion: `roi_<roi_id>_annotation_metadata.json` with gating thresholds + per-type counts.

### Schema (12 columns)

| Column | Type | Description |
|---|---|---|
| `superpixel_id` | int64 | SLIC superpixel ID (0-indexed within ROI at 10 µm scale) |
| `x` | float64 | Centroid x in pixel units |
| `y` | float64 | Centroid y in pixel units |
| `cell_type` | object (str) | Discrete boolean-gating label (15 types including `"unassigned"`); priority-order assignment |
| `confidence` | float64 | Gating confidence score (see cell_type_annotation.py) |
| `lineage_immune` | float64 | Continuous [0, 1] lineage score; sigmoid-normalized from CD45 |
| `lineage_endothelial` | float64 | Continuous [0, 1]; sigmoid-normalized from mean(CD31, CD34) |
| `lineage_stromal` | float64 | Continuous [0, 1]; sigmoid-normalized from CD140a |
| `subtype` | object (str) | Within-lineage subtype (neutrophil, m2_macrophage, myeloid, non_myeloid_immune; geometric-mean scored) |
| `activation_cd44` | float64 | Continuous [0, 1] CD44 activation overlay |
| `activation_cd140b` | float64 | Continuous [0, 1] CD140b activation overlay |
| `composite_label` | object (str) | Concatenated lineage-string label used as Family B stratifier (e.g., `"immune+endothelial"`) |

### Scale
10 µm only (Phase 2 pre-registration §2 pins 10 µm a priori).

### Typical size
~2,400 rows per ROI (one row per superpixel).

---

## 7. Temporal Interface Analysis (Phase 2 pre-registered)

**Location**: `results/biological_analysis/temporal_interfaces/`
**Generator**: `run_temporal_interface_analysis.py` → `src/analysis/temporal_interface_analysis.py`
**Plan**: `analysis_plans/temporal_interfaces_plan.md` (frozen 2026-04-17, amended)

### Directory contents

17 parquet files + `endpoint_summary.csv` (primary reviewer-facing) + `run_provenance.json`.

### `endpoint_summary.csv` — primary table

330 rows × 33 columns. Contents: Family A (48) + Family B (252) + Family C (30) endpoint rows, one per (family × endpoint × contrast).

**Key columns:**

| Column | Type | Description |
|---|---|---|
| `family` | str | `A_interface_clr` \| `B_continuous_neighborhood` \| `C_compartment_activation` |
| `endpoint` | str | e.g. `immune_clr`, `vs_sham_mean_delta_lineage_immune`, `triple_overlap_fraction` |
| `contrast` | str | One of 6 pairwise: `Sham_vs_D1`, `Sham_vs_D3`, `Sham_vs_D7`, `D1_vs_D3`, `D1_vs_D7`, `D3_vs_D7` |
| `tp1`, `tp2` | str | Timepoint labels |
| `n_mice_1`, `n_mice_2` | int | Sample size per group (2 each normally) |
| `insufficient_support` | bool | True if a group has n<2 surviving mice (row preserved as NaN-with-flag) |
| `mouse_mean_1`, `mouse_mean_2` | float | Per-group mean of mouse-level means |
| `mouse_range_1`, `mouse_range_2` | float | Per-group max - min of mouse values |
| `hedges_g` | float | Observed Hedges' g (small-sample-corrected Cohen's d) |
| `g_shrunk_skeptical` | float | Posterior mean under prior N(0, 0.5²); NaN if pathological |
| `g_shrunk_neutral` | float | Posterior mean under prior N(0, 1.0²); NaN if pathological |
| `g_shrunk_optimistic` | float | Posterior mean under prior N(0, 2.0²); NaN if pathological |
| `pooled_std` | float | Pooled within-group SD |
| `g_pathological` | bool | True iff `|g| > 3 AND pooled_std < 0.01` (variance-collapse artifact) |
| `bootstrap_range_min`, `bootstrap_range_max` | float | Bounds from 10k percentile bootstrap (NOT coverage-bearing CI) |
| `n_unique_resamples` | int | Number of distinct g values in bootstrap (≤9 at n=2 per group) |
| `n_required_80pct` | float | Sample size for 80% power given raw observed g |
| `n_required_skeptical`, `n_required_neutral`, `n_required_optimistic` | float | Same under each shrunk g; NaN if pathological |
| `normalization_mode` | str | `per_roi_sigmoid` \| `sham_reference` (Family A only) |
| `sham_percentile` | float | Sham-reference percentile used (Family A sensitivity rows; 65/75/85) |
| `normalization_sign_reverse` | bool | Family A: Sham→D7 g flips sign between per-ROI and Sham-ref regimes |
| `normalization_g_collapse` | bool | Family A: Sham-ref magnitude < 20% of per-ROI |
| `hedges_g_sham_ref` | float | Family A only: Sham-reference regime g for comparison |
| `composite_label` | str | Family B only: the (post-hoc descriptive) stratifier |
| `observed_range` | float | Mouse-level max - min (context) |
| `threshold_sensitive` | bool | Family B: endpoint sign-flips across min-support sweep {10, 20, 40} |

### Parquet schemas (17 files)

**Family A inputs / outputs:**
- `interface_fractions.parquet` — (8, 14). Mouse-level fraction per category × 8 mice (2 mice × 4 timepoints).
- `interface_fractions_normalization_sensitivity.parquet` — (16, 15). Same but under Sham-reference 75ᵗʰ threshold.
- `interface_clr.parquet` — (8, 10). CLR-transformed composition (8 categories → 8 CLR components after Bayesian-multiplicative zero replacement).
- `interface_clr_no_none.parquet` — (8, 9). CLR sensitivity: computed excluding the "none" category.
- `family_a_endpoints_global_norm.parquet` — (48, 27). Sham-reference endpoints at 75ᵗʰ percentile (primary).
- `family_a_endpoints_norm_sweep.parquet` — (144, 27). Sham-reference at {65, 75, 85}.
- `family_a_global_thresholds.parquet` — (3, 4). The three Sham-reference percentile thresholds per lineage.
- `family_a_sensitivity_endpoints.parquet` — (144, 26). Lineage-threshold sweep {0.2, 0.3, 0.4}.
- `sensitivity_thresholds.parquet` — (24, 13). Per-sweep threshold values.

**Family B:**
- `continuous_neighborhood_temporal.parquet` — (107, 10). Per (composite_label, mouse, timepoint) neighbor-minus-self delta, plus vs-Sham delta columns.
- `continuous_neighborhood_missingness.parquet` — (100, 6). (composite_label × timepoint) filter status: `absent_biology` vs `below_min_support` vs `sufficient`; `kept_in_trajectory` flag.
- `family_b_sensitivity_endpoints.parquet` — (864, 27). Min-support sweep {10, 20, 40} across all composite labels × lineages × contrasts.

**Family C:**
- `compartment_activation_temporal.parquet` — (8, 16). Per (mouse × timepoint) CD44⁺ rate within CD45⁺/CD31⁺/CD140b⁺/background compartments + triple overlap + n_rois.
- `family_c_sensitivity_endpoints.parquet` — (90, 26). Sham-percentile sweep {65, 75, 85}.
- `sham_reference_thresholds.parquet` — (1, 4). 75ᵗʰ-percentile thresholds for {CD45, CD31, CD140b, CD44} computed once on Sham ROIs.

**Spatial coherence:**
- `join_counts.parquet` — (192, 12). Per-ROI per-category: BB join count observed vs permutation null (1000 permutations, k=10 NN adjacency).
- `lineage_morans_i.parquet` — (72, 5). Per (ROI × lineage) continuous Moran's I.

### `run_provenance.json`

| Key | Description |
|---|---|
| `git_commit` | Commit hash at run time |
| `git_dirty` | True if working tree had uncommitted changes |
| `config_sha256` | Hash of the frozen `config.json` |
| `input_file_sha256` | Per-annotation-parquet SHA256 (24 entries) |
| `package_versions` | numpy/pandas/scipy/sklearn versions |
| `seeds` | RNG seeds used (permutations, bootstrap) |
| `parameters` | Filter values, threshold sweeps, min-support settings |
| `run_datetime` | ISO timestamp |
| `excluded_rois` | List of excluded ROI IDs with rationale |

Regenerate-if-changed: any modification of inputs, config, or module code invalidates the output directory per the plan's reproducibility-freeze rule.

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
- Loading all 24 ROIs: ~50-150MB RAM

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

## 8. Configuration Files (v2.1)

The project uses **two configuration files** with disjoint responsibilities:

### `config.json` — analysis configuration

Loaded via `src.config.Config`. Contents govern results and update `config_sha256` in provenance:

| Section | Role |
|---|---|
| `data` | Raw data location, metadata file, file pattern |
| `channels` | Protein/DNA/calibration/background channel assignments |
| `channel_groups` | Semantic groupings of markers (biological relationships, consumed by both analysis and viz) |
| `processing` | Arcsinh, DNA preprocessing, normalization |
| `segmentation` | SLIC parameters, scales (µm) |
| `cell_type_annotation` | Boolean-gating rules (15 types: `positive_markers`, `negative_markers`, `family`), positivity thresholds, `membership_axes` (continuous lineage/subtype/activation scoring + `composite_label_thresholds`) |
| `biological_analysis` | Differential abundance, neighborhood enrichment, temporal trajectory specs |
| `analysis` | Clustering, batch correction |
| `quality_control`, `validation` | QC thresholds, scientific validation settings |
| `output`, `performance` | I/O paths, parallelism, compression |
| `metadata_tracking` | ROI metadata schema |

Each `cell_types[*]` entry contains exactly: `positive_markers`, `negative_markers`, `family`. No labels, colors, or prose — those moved to `viz.json`.

### `viz.json` — display configuration

Loaded via `src.viz_utils.VizConfig`. Contents are display-only — changes here do NOT affect analysis results or `config_sha256`:

| Section | Contents |
|---|---|
| `cell_type_display` | For each of 15 cell types: `{label, color}`. Human names + hex codes only. |
| `timepoint_display` | `{order: [Sham, D1, D3, D7], colors: {...}}` |
| `channel_group_colormaps` | matplotlib colormap name per channel group (`immune_markers` → `"Reds"`, etc.) |
| `validation_plots` | Primary markers, always-included markers, layout (figsize, panels) for multichannel validation plots |
| `figure_defaults` | Seaborn style, DPI, default figsize, font size (applied via `VizConfig.apply_rcparams()`) |

### VizConfig API

```python
from src.viz_utils import VizConfig
viz = VizConfig.load()                      # auto-discovers project root
viz.apply_rcparams()                         # set matplotlib defaults

viz.cell_type_colors['neutrophil']           # '#D62828'
viz.cell_type_labels['activated_m2_cd44']    # 'Activated M2 (CD44+)'
viz.cell_type_order                          # list in declaration order
viz.ct_label('neutrophil')                   # 'Neutrophil' (with fallback for unknown ids)
viz.timepoint_order                          # ['Sham', 'D1', 'D3', 'D7']
viz.timepoint_colors['D7']                   # '#E63946'
viz.channel_group_colormaps                  # {'immune_markers': 'Reds', ...}
viz.validation_plots                         # full layout dict for multichannel viz
viz.figure_defaults                          # {'style': 'whitegrid', 'dpi': 150, ...}
```

### Design decisions

- `channel_groups` stays in `config.json` (biological groupings are analysis knowledge). Their color *mappings* (`channel_group_colormaps`) live in `viz.json`.
- `timepoint_display.order` is a display hint; the authoritative analysis ordering is `TIMEPOINT_ORDER` in `src/analysis/temporal_interface_analysis.py` (pre-registered, frozen).
- `Config.visualization` is retained as an empty dict for backward compatibility with callers that reference it; new code should use `VizConfig` directly.

---

## Related Documentation

- **Pipeline**: See `docs/architecture/ARCHITECTURE.md` for how results are generated
- **Methods**: See `METHODS.md` for scientific methodology
- **Analysis**: See `notebooks/biological_narratives/kidney_injury_spatial_analysis.ipynb` for usage examples

---

**Schema Version**: 2.1
**Maintained by**: IMC Analysis Pipeline
**Questions**: See `CLAUDE.md` for development guidance
