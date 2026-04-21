"""
Comprehensive IMC Figure Generation

Generates publication-quality figures from the dual annotation system:
- Ternary lineage maps (RGB = immune/endothelial/stromal)
- Temporal evolution grids
- Interface composition analysis
- Discrete cell type distributions
- Activation overlays

All figures are experiment-agnostic: type names, lineage axes, and
temporal metadata are read from data products, never hardcoded.

Data sources:
- Per-ROI parquet: results/biological_analysis/cell_type_annotations/*.parquet
- DA results: results/biological_analysis/differential_abundance/*.csv
- Neighborhood results: results/biological_analysis/spatial_neighborhoods/*.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from src.utils.metadata import parse_roi_metadata
from src.utils.paths import get_paths

logger = logging.getLogger(__name__)

# Palette for up to 8 ordered timepoints — extended dynamically if needed
_DEFAULT_PALETTE = ['#4DAF4A', '#E41A1C', '#FF7F00', '#377EB8',
                    '#984EA3', '#A65628', '#F781BF', '#999999']

_PUBLICATION_RCPARAMS = {
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
}


def _infer_timepoint_order(timepoints: list) -> list:
    """Order timepoints naturally: 'Sham'/'Control' first, then by numeric suffix."""
    def _sort_key(tp: str):
        tp_lower = tp.lower()
        if tp_lower in ('sham', 'control', 'baseline'):
            return (0, 0, tp)
        # Extract numeric part (D1→1, D3→3, Day_7→7, etc.)
        import re
        nums = re.findall(r'\d+', tp)
        return (1, int(nums[0]) if nums else 999, tp)
    return sorted(timepoints, key=_sort_key)


def _timepoint_colors(timepoints: list) -> dict:
    """Assign colors to timepoints from palette."""
    return {tp: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
            for i, tp in enumerate(timepoints)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_annotations(annotations_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load all per-ROI annotation parquets into a single DataFrame with metadata."""
    if annotations_dir is None:
        annotations_dir = get_paths().annotations_dir

    parquet_files = sorted(annotations_dir.glob('roi_*_cell_types.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f"No annotation parquets in {annotations_dir}")

    frames = []
    for pf in parquet_files:
        roi_id = pf.stem.replace('_cell_types', '')
        df = pd.read_parquet(pf)
        df['roi_id'] = roi_id

        meta = parse_roi_metadata(roi_id)
        df['timepoint'] = meta.get('timepoint', 'Unknown')
        df['region'] = meta.get('region', 'Unknown')
        df['mouse'] = meta.get('mouse', 'Unknown')

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined['timepoint'] != 'Test']
    return combined


def _detect_lineage_cols(df: pd.DataFrame) -> List[str]:
    """Return lineage column names found in DataFrame."""
    return sorted([c for c in df.columns if c.startswith('lineage_')])


def _detect_activation_cols(df: pd.DataFrame) -> List[str]:
    """Return activation column names found in DataFrame."""
    return sorted([c for c in df.columns if c.startswith('activation_')])


def _lineage_display_name(col: str) -> str:
    return col.replace('lineage_', '').capitalize()


# ---------------------------------------------------------------------------
# Figure 1: Ternary lineage map — the hero figure
# ---------------------------------------------------------------------------

def ternary_lineage_map(
    roi_df: pd.DataFrame,
    ax: plt.Axes,
    lineage_cols: Optional[List[str]] = None,
    title: Optional[str] = None,
    point_size: float = 1.0,
) -> None:
    """
    Plot a single ROI as an RGB ternary scatter of lineage scores.

    Each superpixel's color is a blend:
      R = immune score, G = stromal score, B = endothelial score.
    Pure types are saturated primaries; interfaces are blends.
    No-lineage superpixels are dark gray.
    """
    if lineage_cols is None:
        lineage_cols = _detect_lineage_cols(roi_df)

    if len(lineage_cols) < 3:
        ax.text(0.5, 0.5, 'Need 3 lineage columns', ha='center', va='center',
                transform=ax.transAxes)
        return

    # Map lineage columns to RGB channels (alphabetical → endothelial, immune, stromal)
    # Force consistent mapping: immune=R, stromal=G, endothelial=B
    col_map = {}
    for col in lineage_cols:
        name = col.replace('lineage_', '').lower()
        if 'immune' in name:
            col_map['R'] = col
        elif 'stromal' in name:
            col_map['G'] = col
        elif 'endothelial' in name:
            col_map['B'] = col

    if len(col_map) < 3:
        # Fallback: assign in order
        for i, (channel, col) in enumerate(zip('RGB', lineage_cols[:3])):
            col_map[channel] = col

    r = roi_df[col_map['R']].values
    g = roi_df[col_map['G']].values
    b = roi_df[col_map['B']].values

    # Build RGB array, already in [0, 1] from sigmoid normalization
    rgb = np.column_stack([r, g, b])
    rgb = np.clip(rgb, 0, 1)

    # Global contrast stretch: map [p5, p95] of max-channel to [0.2, 1.0].
    # Preserves relative intensity differences between superpixels (a dim
    # interface stays dimmer than a bright pure-lineage region).
    row_max = rgb.max(axis=1)
    p5, p95 = np.percentile(row_max, [5, 95])
    denom = max(p95 - p5, 0.01)
    brightness = np.clip((row_max - p5) / denom, 0.0, 1.0) * 0.8 + 0.2

    # Scale each pixel's RGB so its max channel equals the computed brightness
    current_max = np.maximum(row_max, 1e-6)
    rgb_boosted = rgb * (brightness / current_max)[:, np.newaxis]
    rgb_boosted = np.clip(rgb_boosted, 0, 1)

    # Darken no-lineage superpixels (all scores below threshold)
    no_signal = (row_max < 0.15)
    rgb_boosted[no_signal] = 0.15  # dark gray

    x = roi_df['x'].values
    y = roi_df['y'].values

    ax.scatter(x, y, c=rgb_boosted, s=point_size, marker='s', edgecolors='none')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=8, fontweight='bold')


def figure_ternary_legend(ax: plt.Axes, lineage_cols: List[str]) -> None:
    """Draw an RGB legend for the ternary lineage map."""
    labels = {
        'R': ('Immune', '#FF0000'),
        'G': ('Stromal', '#00FF00'),
        'B': ('Endothelial', '#0000FF'),
    }
    # Blends
    blend_labels = [
        ('Immune + Endothelial', '#FF00FF'),
        ('Immune + Stromal', '#FFFF00'),
        ('Endothelial + Stromal', '#00FFFF'),
        ('No lineage', '#262626'),
    ]
    patches = [Patch(facecolor=color, label=label) for label, color in labels.values()]
    patches += [Patch(facecolor=color, label=label) for label, color in blend_labels]
    ax.legend(handles=patches, loc='center', fontsize=7, frameon=False, ncol=2)
    ax.axis('off')


# ---------------------------------------------------------------------------
# Figure 2: Temporal ternary grid
# ---------------------------------------------------------------------------

def figure_temporal_ternary_grid(
    all_df: pd.DataFrame,
    figsize: Tuple[float, float] = (16, 12),
    max_rois_per_timepoint: int = 6,
) -> plt.Figure:
    """
    Grid of ternary lineage maps: rows = timepoints, columns = ROIs.

    Shows how the tissue lineage landscape evolves from Sham through injury.
    """
    lineage_cols = _detect_lineage_cols(all_df)
    timepoints = _infer_timepoint_order([tp for tp in all_df['timepoint'].unique()])

    # Group ROIs by timepoint
    roi_groups = {}
    for tp in timepoints:
        rois = sorted(all_df[all_df['timepoint'] == tp]['roi_id'].unique())
        roi_groups[tp] = rois[:max_rois_per_timepoint]

    n_cols = max(len(rois) for rois in roi_groups.values()) + 1  # +1 for legend
    n_rows = len(timepoints)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, tp in enumerate(timepoints):
        rois = roi_groups[tp]
        for col_idx in range(n_cols - 1):
            ax = axes[row_idx, col_idx]
            if col_idx < len(rois):
                roi_id = rois[col_idx]
                roi_df = all_df[all_df['roi_id'] == roi_id]
                meta = parse_roi_metadata(roi_id)
                region = meta.get('region', '')
                label = f"{tp} — {region}"
                ternary_lineage_map(roi_df, ax, lineage_cols, title=label)
            else:
                ax.axis('off')

        # Legend in last column
        ax_legend = axes[row_idx, -1]
        if row_idx == 0:
            figure_ternary_legend(ax_legend, lineage_cols)
        else:
            ax_legend.axis('off')

    fig.suptitle(f'Lineage Landscape: {" → ".join(timepoints)}',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# Figure 4: Discrete cell type distribution
# ---------------------------------------------------------------------------

def figure_discrete_type_distribution(
    all_df: pd.DataFrame,
    figsize: Tuple[float, float] = (14, 6),
) -> plt.Figure:
    """
    Stacked bar chart of discrete cell type proportions by timepoint.
    Type names read from data, not hardcoded.
    """
    timepoints = _infer_timepoint_order([tp for tp in all_df['timepoint'].unique()])

    # Get all cell types (excluding unassigned)
    all_types = sorted([t for t in all_df['cell_type'].unique() if t != 'unassigned'])

    # Compute proportions per timepoint (mouse-level mean)
    data = []
    for tp in timepoints:
        tp_df = all_df[all_df['timepoint'] == tp]
        mice = tp_df['mouse'].unique()
        mouse_props = []
        for mouse in mice:
            m_df = tp_df[tp_df['mouse'] == mouse]
            n_assigned = (m_df['cell_type'] != 'unassigned').sum()
            if n_assigned == 0:
                n_assigned = 1  # avoid div by zero
            props = {}
            for ct in all_types:
                props[ct] = (m_df['cell_type'] == ct).sum() / n_assigned * 100
            mouse_props.append(props)

        # Mean across mice
        mean_props = {}
        for ct in all_types:
            vals = [mp[ct] for mp in mouse_props]
            mean_props[ct] = np.mean(vals)
        data.append(mean_props)

    # Build color palette from viz.json if available, otherwise generate
    colors = {}
    try:
        from .viz_config import VizConfig
        colors = dict(VizConfig.load().cell_type_colors)
    except Exception:
        pass

    # Fallback palette
    cmap = plt.cm.get_cmap('tab20', len(all_types))
    for i, ct in enumerate(all_types):
        if ct not in colors or colors[ct] is None:
            colors[ct] = mcolors.to_hex(cmap(i))

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(timepoints))
    bottom = np.zeros(len(timepoints))

    for ct in all_types:
        vals = [data[i].get(ct, 0) for i in range(len(timepoints))]
        ax.bar(x, vals, bottom=bottom, label=ct.replace('_', ' ').title(),
               color=colors.get(ct, '#888888'), width=0.6, edgecolor='white',
               linewidth=0.3)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(timepoints, fontsize=11)
    ax.set_ylabel('% of Assigned Superpixels (Mouse-Level Mean)')
    ax.set_title('Discrete Cell Type Composition by Timepoint',
                 fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    ax.set_xlim(-0.5, len(timepoints) - 0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Activation overlay on ternary map
# ---------------------------------------------------------------------------

def figure_activation_overlay(
    all_df: pd.DataFrame,
    activation_col: str = 'activation_cd44',
    figsize: Tuple[float, float] = (16, 4),
) -> plt.Figure:
    """
    Ternary lineage map with brightness modulated by activation score.

    One panel per timepoint (representative ROI). Activated regions glow
    brighter; resting regions are dimmer.
    """
    lineage_cols = _detect_lineage_cols(all_df)
    timepoints = _infer_timepoint_order([tp for tp in all_df['timepoint'].unique()])

    fig, axes = plt.subplots(1, len(timepoints), figsize=figsize)
    if len(timepoints) == 1:
        axes = [axes]

    for i, tp in enumerate(timepoints):
        ax = axes[i]
        tp_df = all_df[all_df['timepoint'] == tp]

        # Pick representative ROI (first one)
        roi_id = sorted(tp_df['roi_id'].unique())[0]
        roi_df = tp_df[tp_df['roi_id'] == roi_id].copy()

        if activation_col not in roi_df.columns:
            ax.text(0.5, 0.5, f'No {activation_col}', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')
            continue

        # Build RGB from lineage
        col_map = {}
        for col in lineage_cols:
            name = col.replace('lineage_', '').lower()
            if 'immune' in name:
                col_map['R'] = col
            elif 'stromal' in name:
                col_map['G'] = col
            elif 'endothelial' in name:
                col_map['B'] = col

        r = roi_df[col_map.get('R', lineage_cols[0])].values
        g = roi_df[col_map.get('G', lineage_cols[1])].values
        b = roi_df[col_map.get('B', lineage_cols[2])].values

        rgb = np.column_stack([r, g, b])
        rgb = np.clip(rgb, 0, 1)

        # Apply same global contrast stretch as ternary map
        row_max = rgb.max(axis=1)
        p5, p95 = np.percentile(row_max, [5, 95])
        denom = max(p95 - p5, 0.01)
        base_brightness = np.clip((row_max - p5) / denom, 0.0, 1.0) * 0.8 + 0.2
        current_max = np.maximum(row_max, 1e-6)
        rgb_contrast = rgb * (base_brightness / current_max)[:, np.newaxis]

        # Modulate brightness by activation score: low activation → 30% brightness
        activation = roi_df[activation_col].values
        act_factor = 0.3 + 0.7 * np.clip(activation, 0, 1)
        rgb_modulated = rgb_contrast * act_factor[:, np.newaxis]
        rgb_modulated = np.clip(rgb_modulated, 0, 1)

        x = roi_df['x'].values
        y = roi_df['y'].values
        ax.scatter(x, y, c=rgb_modulated, s=1.0, marker='s', edgecolors='none')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

        act_name = activation_col.replace('activation_', '').upper()
        ax.set_title(f'{tp} — {act_name} activation', fontsize=9, fontweight='bold')

    fig.suptitle(f'Lineage × Activation Overlay',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6: Lineage score distributions
# ---------------------------------------------------------------------------

def figure_lineage_distributions(
    all_df: pd.DataFrame,
    figsize: Tuple[float, float] = (14, 4),
) -> plt.Figure:
    """
    Violin/box plots of lineage scores by timepoint — shows distributional shifts.
    """
    lineage_cols = _detect_lineage_cols(all_df)
    timepoints = _infer_timepoint_order([tp for tp in all_df['timepoint'].unique()])
    tp_colors = _timepoint_colors(timepoints)
    n_lineages = len(lineage_cols)

    fig, axes = plt.subplots(1, n_lineages, figsize=figsize, sharey=True)
    if n_lineages == 1:
        axes = [axes]

    for i, col in enumerate(lineage_cols):
        ax = axes[i]
        name = _lineage_display_name(col)

        plot_data = []
        positions = []
        colors_list = []
        for j, tp in enumerate(timepoints):
            vals = all_df[all_df['timepoint'] == tp][col].values
            plot_data.append(vals)
            positions.append(j)
            colors_list.append(tp_colors.get(tp, '#888888'))

        parts = ax.violinplot(plot_data, positions=positions, showmedians=True,
                              showextrema=False)
        for pc, color in zip(parts['bodies'], colors_list):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts['cmedians'].set_color('black')

        ax.set_xticks(positions)
        ax.set_xticklabels(timepoints, fontsize=9)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_ylim(-0.05, 1.05)

        if i == 0:
            ax.set_ylabel('Lineage Score')

    fig.suptitle('Lineage Score Distributions by Timepoint',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_figures(
    output_dir: Optional[Path] = None,
    dpi: int = 150,
) -> Dict[str, Path]:
    """
    Generate all figures and save to output directory.

    Returns dict mapping figure name to saved file path.
    """
    paths = get_paths()
    if output_dir is None:
        output_dir = paths.figures_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply publication style within this function only (no module-level side effects)
    plt.rcParams.update(_PUBLICATION_RCPARAMS)

    logger.info("Loading annotation data...")
    all_df = load_all_annotations()
    logger.info(f"Loaded {len(all_df)} superpixels from {all_df['roi_id'].nunique()} ROIs")

    saved = {}

    # Figure 1: Temporal ternary grid
    logger.info("Generating temporal ternary grid...")
    fig = figure_temporal_ternary_grid(all_df)
    path = output_dir / 'temporal_ternary_grid.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved['temporal_ternary_grid'] = path

    # Figure 3: Discrete type distribution
    logger.info("Generating discrete type distribution...")
    fig = figure_discrete_type_distribution(all_df)
    path = output_dir / 'discrete_type_distribution.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved['discrete_type_distribution'] = path

    # Figure 4: Activation overlays
    activation_cols = _detect_activation_cols(all_df)
    for act_col in activation_cols:
        act_name = act_col.replace('activation_', '')
        logger.info(f"Generating {act_name} activation overlay...")
        fig = figure_activation_overlay(all_df, activation_col=act_col)
        path = output_dir / f'activation_overlay_{act_name}.png'
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved[f'activation_overlay_{act_name}'] = path

    # Figure 5: Lineage distributions
    logger.info("Generating lineage distributions...")
    fig = figure_lineage_distributions(all_df)
    path = output_dir / 'lineage_distributions.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved['lineage_distributions'] = path

    logger.info(f"Saved {len(saved)} figures to {output_dir}")
    return saved
