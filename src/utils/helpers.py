"""
Simple, composable utilities for IMC analysis
KISS principle: each function does ONE thing well
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# --- DATA STRUCTURES ---

@dataclass
class Metadata:
    """Clean metadata access"""
    condition: str = 'Unknown'
    timepoint: Optional[int] = None
    region: str = 'Unknown'
    replicate_id: str = 'Unknown'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'condition': self.condition,
            'timepoint': self.timepoint,
            'region': self.region,
            'replicate_id': self.replicate_id
        }
    
    @classmethod
    def from_roi_file(cls, roi_file: Path, lookup: Dict) -> 'Metadata':
        """Create from ROI filename"""
        key = roi_file.name.replace('.txt', '')
        data = lookup.get(key, {})
        return cls(
            condition=data.get('condition', 'Unknown'),
            timepoint=data.get('timepoint'),
            region=data.get('region', 'Unknown'),
            replicate_id=data.get('replicate_id', 'Unknown')
        )
    
    @property
    def display_name(self) -> str:
        """Human readable name"""
        if self.timepoint is not None:
            return f"T{self.timepoint} {self.region}"
        return f"{self.condition} {self.region}"

# --- PLOTTING UTILITIES ---

def add_percentage_labels(ax, bars, sizes, total=None, **kwargs):
    """Add percentage labels to bars"""
    if total is None:
        total = sum(sizes)
    
    defaults = {'ha': 'center', 'va': 'bottom', 'fontsize': 9}
    defaults.update(kwargs)
    
    for bar, size in zip(bars, sizes):
        if total > 0:
            pct = 100 * size / total
            if hasattr(bar, 'get_height'):  # vertical bars
                x = bar.get_x() + bar.get_width()/2
                y = bar.get_height() + max(bar.get_height() * 0.01, 10)
                ax.text(x, y, f'{pct:.1f}%', **defaults)
            else:  # horizontal bars
                x = bar.get_width() + 0.01
                y = bar.get_y() + bar.get_height()/2
                ax.text(x, y, f'{pct:.1f}%', ha='left', va='center', 
                       fontsize=defaults['fontsize'])

def add_dendrogram_to_heatmap(ax, data, method='average', metric='euclidean'):
    """Replace heatmap axis with dendrogram + heatmap layout"""
    if len(data) <= 1:
        return ax, data, list(range(len(data)))
    
    # Get the gridspec this axis belongs to
    gs = ax.get_gridspec()
    pos = ax.get_subplotspec()
    
    # Create new layout with dendrogram
    gs_new = GridSpecFromSubplotSpec(1, 2, subplot_spec=pos,
                                     width_ratios=[0.10, 0.90], wspace=0.001)
    
    # Remove original axis
    ax.remove()
    
    # Create dendrogram axis
    dend_ax = plt.gcf().add_subplot(gs_new[0, 0])
    # Compute condensed distances between row vectors (not a square matrix)
    dist = pdist(data, metric=metric)
    Z = linkage(dist, method=method)
    order = leaves_list(Z)
    dendrogram(Z, orientation='left', no_labels=True, color_threshold=None, ax=dend_ax)
    clean_axis(dend_ax)
    
    # Create new heatmap axis
    heat_ax = plt.gcf().add_subplot(gs_new[0, 1])
    
    return heat_ax, Z, order

def clean_axis(ax):
    """Remove all spines and ticks from axis"""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def canonicalize_pair(name1: str, name2: str, separator: str = ' ↔ ') -> str:
    """Create canonical pair name"""
    sorted_pair = sorted([name1, name2])
    return f"{sorted_pair[0]}{separator}{sorted_pair[1]}"

# --- DATA PROCESSING ---

def accumulate_contacts(contacts_list: List[Dict]) -> Dict[str, float]:
    """Combine multiple contact dictionaries, canonicalizing pairs"""
    result = {}
    for contacts in contacts_list:
        for blob1, neighbors in contacts.items():
            for blob2, freq in neighbors.items():
                if freq > 0:
                    pair = canonicalize_pair(blob1, blob2)
                    result[pair] = max(result.get(pair, 0), freq)
    return result

def filter_by_frequency(data: Dict, min_freq: float = 0.0) -> Dict:
    """Filter dictionary by minimum frequency"""
    return {k: v for k, v in data.items() if v > min_freq}

def top_n_items(data: Dict, n: int = 10) -> List[Tuple]:
    """Get top N items from dictionary by value"""
    return sorted(data.items(), key=lambda x: x[1], reverse=True)[:n]

# --- HEATMAP UTILITIES ---

def create_contact_matrix(contacts: Dict, labels: List[str]) -> np.ndarray:
    """Create contact matrix from contact dictionary"""
    n = len(labels)
    matrix = np.zeros((n, n))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    for pair_str, freq in contacts.items():
        if ' ↔ ' in pair_str:
            parts = pair_str.split(' ↔ ')
        elif ' vs ' in pair_str:
            parts = pair_str.split(' vs ')
        else:
            continue
            
        if len(parts) == 2 and parts[0] in label_to_idx and parts[1] in label_to_idx:
            i, j = label_to_idx[parts[0]], label_to_idx[parts[1]]
            matrix[i, j] = matrix[j, i] = freq
    
    return matrix

def plot_heatmap_with_dendrogram(ax, matrix, labels, title="", cmap='YlOrRd', 
                                 add_dendro=True, **kwargs):
    """Plot heatmap with optional dendrogram"""
    if add_dendro and len(matrix) > 1:
        heat_ax, Z, order = add_dendrogram_to_heatmap(ax, matrix)
        # Only reorder rows for rectangular matrices, both for square matrices
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            matrix_ordered = matrix[order, :][:, order]  # Square matrix: reorder both
        else:
            matrix_ordered = matrix[order, :]  # Rectangular matrix: reorder rows only
        labels_ordered = [labels[i] for i in order]
    else:
        heat_ax = ax
        matrix_ordered = matrix
        labels_ordered = labels
    
    im = heat_ax.imshow(matrix_ordered, cmap=cmap, aspect='auto', **kwargs)
    heat_ax.set_xticks(range(len(labels_ordered)))
    heat_ax.set_xticklabels(labels_ordered, rotation=45, ha='right')
    heat_ax.set_yticks(range(len(labels_ordered)))
    heat_ax.set_yticklabels(labels_ordered)
    heat_ax.set_title(title)
    
    return heat_ax, im

# --- BI-DENDROGRAM HEATMAP ---

def plot_heatmap_with_rowcol_dendrogram(ax, matrix, row_labels, col_labels, title="", cmap='YlOrRd', vmin=None, vmax=None):
    """Plot heatmap with dendrograms on rows and columns; reorder both.
    Returns the heatmap axis and the image handle.
    """
    import numpy as np
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
    
    # Create new layout with top and left dendrograms
    gs = ax.get_gridspec()
    pos = ax.get_subplotspec()
    # Place left dendrogram to the left of heatmap, and column dendrogram BELOW the heatmap
    gs_new = GridSpecFromSubplotSpec(2, 2, subplot_spec=pos,
                                     width_ratios=[0.18, 0.82],
                                     height_ratios=[0.82, 0.18], wspace=0.02, hspace=0.02)
    ax.remove()
    
    dend_ax_left = plt.gcf().add_subplot(gs_new[0, 0])
    heat_ax = plt.gcf().add_subplot(gs_new[0, 1])
    dend_ax_bottom = plt.gcf().add_subplot(gs_new[1, 1])
    
    # Row clustering
    if len(matrix) > 1:
        row_dist = pdist(matrix, metric='euclidean')
        Zr = linkage(row_dist, method='average')
        row_order = leaves_list(Zr)
        dendrogram(Zr, orientation='left', no_labels=True, color_threshold=None, ax=dend_ax_left)
    else:
        row_order = list(range(len(matrix)))
    clean_axis(dend_ax_left)
    
    # Column clustering (bottom)
    if matrix.shape[1] > 1:
        col_dist = pdist(matrix.T, metric='euclidean')
        Zc = linkage(col_dist, method='average')
        col_order = leaves_list(Zc)
        dendrogram(Zc, orientation='bottom', no_labels=True, color_threshold=None, ax=dend_ax_bottom)
    else:
        col_order = list(range(matrix.shape[1]))
    clean_axis(dend_ax_bottom)
    
    # Reorder matrix and labels
    matrix_ordered = np.asarray(matrix)[row_order, :][:, col_order]
    row_labels_ordered = [row_labels[i] for i in row_order]
    col_labels_ordered = [col_labels[i] for i in col_order]
    
    im = heat_ax.imshow(matrix_ordered, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    heat_ax.set_yticks(range(len(row_labels_ordered)))
    heat_ax.set_yticklabels(row_labels_ordered)
    heat_ax.set_xticks(range(len(col_labels_ordered)))
    heat_ax.set_xticklabels(col_labels_ordered, rotation=45, ha='right')
    heat_ax.set_title(title)
    
    return heat_ax, im

# --- LAYOUT HELPERS ---

class PlotGrid:
    """Simple grid layout manager"""
    def __init__(self, rows: int, cols: int, figsize=(20, 14)):
        self.fig = plt.figure(figsize=figsize, constrained_layout=True)
        self.gs = self.fig.add_gridspec(rows, cols, hspace=0.3, wspace=0.25)
        self.axes = {}
    
    def get(self, row: int, col: int, rowspan: int = 1, colspan: int = 1):
        """Get or create axis at position"""
        key = (row, col, rowspan, colspan)
        if key not in self.axes:
            if rowspan == 1 and colspan == 1:
                self.axes[key] = self.fig.add_subplot(self.gs[row, col])
            else:
                self.axes[key] = self.fig.add_subplot(
                    self.gs[row:row+rowspan, col:col+colspan]
                )
        return self.axes[key]
    
    def save(self, filename: str, dpi: int = 300):
        """Save figure"""
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        return self.fig

# --- PROTEIN SIGNATURE UTILITIES ---

def get_dominant_proteins(expression: np.ndarray, protein_names: List[str], 
                         n: int = 2) -> List[str]:
    """Get top N proteins by expression"""
    top_idx = np.argsort(expression)[-n:][::-1]
    return [protein_names[i] for i in top_idx]

def create_protein_signature(proteins: List[str]) -> str:
    """Create protein signature string"""
    return '+'.join(proteins)

def parse_protein_signature(signature: str) -> List[str]:
    """Parse protein signature string"""
    return signature.split('+')

# --- FILE I/O ---

def find_roi_files(data_dir: Path, pattern: str = "*ROI*.txt") -> List[Path]:
    """Find all ROI files in directory"""
    return sorted(Path(data_dir).glob(pattern))

def load_config(config_path: str = 'config.json') -> Dict:
    """Load configuration file"""
    import json
    with open(config_path, 'r') as f:
        return json.load(f)

# --- STATISTICAL UTILITIES ---

def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute basic statistics"""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
    
    arr = np.array(values)
    return {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'count': len(arr)
    }

def normalize_frequencies(freq_dict: Dict) -> Dict:
    """Normalize frequencies to sum to 1"""
    total = sum(freq_dict.values())
    if total > 0:
        return {k: v/total for k, v in freq_dict.items()}
    return freq_dict

# --- COLOR UTILITIES ---

def get_default_pair_palette() -> List:
    """Return a large, distinct color palette for marker pairs."""
    # Combine tab20, tab20b, tab20c for breadth
    steps = 20
    palette = []
    palette.extend([plt.cm.tab20(i/(steps-1)) for i in range(steps)])
    palette.extend([plt.cm.tab20b(i/(steps-1)) for i in range(steps)])
    palette.extend([plt.cm.tab20c(i/(steps-1)) for i in range(steps)])
    return palette

def build_pair_color_map(pairs: List[str]) -> Dict[str, any]:
    """Build a stable color map for canonical marker pairs.
    Pairs should be canonicalized like 'A+B'.
    """
    unique_pairs = sorted(set(pairs))
    palette = get_default_pair_palette()
    color_map = {}
    for i, pair in enumerate(unique_pairs):
        color_map[pair] = palette[i % len(palette)]
    return color_map