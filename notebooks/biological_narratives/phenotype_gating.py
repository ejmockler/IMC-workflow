#!/usr/bin/env python
"""
Boolean gating for cell phenotype assignment
Maps superpixels to interpretable cell types from config.json.backup
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Phenotype definitions from config.json.backup
PHENOTYPES = {
    'M2_Macrophage': {
        'positive': ['CD45', 'CD11b', 'CD206'],
        'negative': ['CD31'],
        'description': 'M2-like macrophages (anti-inflammatory/repair)',
        'color': '#FCBF49',
        'expected_peak': ['D3', 'D7']
    },
    'Neutrophil': {
        'positive': ['CD45', 'Ly6G'],
        'negative': ['CD31'],
        'description': 'Neutrophils (acute inflammation)',
        'color': '#D62828',
        'expected_peak': ['D1']
    },
    'Activated_Fibroblast': {
        'positive': ['CD140b', 'CD44'],
        'negative': ['CD45', 'CD31'],
        'description': 'Activated fibroblasts (pro-fibrotic)',
        'color': '#2A9D8F',
        'expected_peak': ['D7']
    },
    'Activated_Endothelial_CD44': {
        'positive': ['CD31', 'CD34', 'CD44'],
        'negative': ['CD45'],
        'description': 'Activated endothelial cells (vascular remodeling)',
        'color': '#06AED5',
        'expected_peak': ['D1', 'D3', 'D7']
    },
    'Activated_Endothelial_CD140b': {
        'positive': ['CD31', 'CD34', 'CD140b'],
        'negative': ['CD45'],
        'description': 'Activated endothelial (EndMT)',
        'color': '#086788',
        'expected_peak': ['D3', 'D7']
    },
    'Resting_Endothelial': {
        'positive': ['CD31', 'CD34'],
        'negative': ['CD45', 'CD44', 'CD140b'],
        'description': 'Quiescent endothelial cells',
        'color': '#457B9D',
        'expected_peak': ['Sham']
    },
    'Activated_Immune_CD44': {
        'positive': ['CD45', 'CD11b', 'CD44'],
        'negative': ['CD31'],
        'description': 'Activated immune cells (CD44+)',
        'color': '#E63946',
        'expected_peak': ['D1', 'D3', 'D7']
    },
    'Activated_Immune_CD140b': {
        'positive': ['CD45', 'CD11b', 'CD140b'],
        'negative': ['CD31'],
        'description': 'Activated immune (stromal interaction)',
        'color': '#F77F00',
        'expected_peak': ['D3', 'D7']
    }
}


def compute_positivity_thresholds(superpixel_df: pd.DataFrame,
                                  markers: List[str],
                                  method: str = 'percentile',
                                  percentile: float = 60) -> Dict[str, float]:
    """
    Compute positivity thresholds for each marker

    From config.json.backup:
    - 60th percentile balances sparse IMC signal (90-95% zeros) with biological detection
    - Special cases: CD206 (50th for rare M2s), Ly6G (70th for sparse neutrophils)
    """
    thresholds = {}

    for marker in markers:
        if method == 'percentile':
            # Special cases from config
            if marker == 'CD206':
                p = 50  # Rare M2 population
            elif marker == 'Ly6G':
                p = 70  # Sparse neutrophils
            else:
                p = percentile

            thresholds[marker] = np.percentile(superpixel_df[marker].dropna(), p)

    return thresholds


def gate_phenotype(superpixel_df: pd.DataFrame,
                   thresholds: Dict[str, float],
                   positive_markers: List[str],
                   negative_markers: List[str]) -> pd.Series:
    """
    Boolean gating: superpixel is positive if ALL positive markers are high
    AND ALL negative markers are low
    """
    # Start with all True
    is_phenotype = pd.Series(True, index=superpixel_df.index)

    # AND logic for positive markers
    for marker in positive_markers:
        is_phenotype &= (superpixel_df[marker] >= thresholds[marker])

    # AND logic for negative markers (must be BELOW threshold)
    for marker in negative_markers:
        is_phenotype &= (superpixel_df[marker] < thresholds[marker])

    return is_phenotype


def assign_phenotypes(superpixel_df: pd.DataFrame,
                     markers: List[str]) -> pd.DataFrame:
    """
    Assign all phenotypes to superpixels using boolean gating
    Returns DataFrame with phenotype columns (True/False for each)
    """
    # Compute thresholds
    thresholds = compute_positivity_thresholds(superpixel_df, markers)

    # Make copy for phenotype assignments
    phenotype_df = superpixel_df.copy()

    # Gate each phenotype
    for pheno_name, pheno_def in PHENOTYPES.items():
        phenotype_df[pheno_name] = gate_phenotype(
            superpixel_df,
            thresholds,
            pheno_def['positive'],
            pheno_def['negative']
        )

    # Assign dominant phenotype (for visualization)
    phenotype_cols = list(PHENOTYPES.keys())
    phenotype_df['dominant_phenotype'] = phenotype_df[phenotype_cols].idxmax(axis=1)
    phenotype_df.loc[~phenotype_df[phenotype_cols].any(axis=1), 'dominant_phenotype'] = 'Unassigned'

    return phenotype_df


def plot_phenotype_temporal_dynamics(phenotype_df: pd.DataFrame,
                                     timepoint_order: List[str] = ['Sham', 'D1', 'D3', 'D7']):
    """
    Show temporal evolution of each phenotype
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for idx, (pheno_name, pheno_def) in enumerate(PHENOTYPES.items()):
        ax = axes[idx]

        # Count phenotype at each timepoint
        counts = []
        for tp in timepoint_order:
            tp_data = phenotype_df[phenotype_df['timepoint'] == tp]
            if len(tp_data) > 0:
                pct = 100 * tp_data[pheno_name].sum() / len(tp_data)
            else:
                pct = 0
            counts.append(pct)

        # Plot
        ax.plot(timepoint_order, counts, 'o-', color=pheno_def['color'],
               linewidth=3, markersize=10)
        ax.set_title(pheno_name.replace('_', ' '), fontweight='bold', fontsize=10)
        ax.set_ylabel('% of Superpixels')
        ax.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)
        ax.grid(alpha=0.3)

        # Highlight expected peak
        for peak_tp in pheno_def['expected_peak']:
            if peak_tp in timepoint_order:
                peak_idx = timepoint_order.index(peak_tp)
                ax.axvspan(peak_idx - 0.2, peak_idx + 0.2, alpha=0.2,
                          color=pheno_def['color'])

    fig.suptitle('Temporal Dynamics of Cell Phenotypes',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig


def plot_phenotype_spatial_maps(phenotype_df: pd.DataFrame,
                                phenotype_name: str,
                                timepoint_order: List[str] = ['Sham', 'D1', 'D3', 'D7']):
    """
    Show spatial distribution of a specific phenotype across timepoints
    """
    pheno_def = PHENOTYPES[phenotype_name]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for idx, tp in enumerate(timepoint_order):
        ax = axes[idx]
        tp_data = phenotype_df[phenotype_df['timepoint'] == tp]

        if len(tp_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(tp)
            ax.axis('off')
            continue

        # Get one representative ROI
        roi = tp_data['roi'].unique()[0]
        roi_data = tp_data[tp_data['roi'] == roi]

        # Plot all superpixels in gray
        ax.scatter(roi_data['x'], roi_data['y'],
                  c='lightgray', s=20, alpha=0.3, edgecolors='none')

        # Highlight this phenotype
        pheno_positive = roi_data[roi_data[phenotype_name]]
        if len(pheno_positive) > 0:
            ax.scatter(pheno_positive['x'], pheno_positive['y'],
                      c=pheno_def['color'], s=40, alpha=0.8,
                      edgecolors='black', linewidth=0.5)

        ax.set_aspect('equal')
        ax.set_title(f"{tp}\n{len(pheno_positive)} / {len(roi_data)} "
                    f"({100*len(pheno_positive)/len(roi_data):.1f}%)",
                    fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'{phenotype_name.replace("_", " ")}: {pheno_def["description"]}',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def compute_phenotype_colocalization(phenotype_df: pd.DataFrame,
                                    radius_um: float = 50) -> pd.DataFrame:
    """
    Compute which phenotypes tend to co-localize spatially

    For each phenotype pair, compute:
    - Expected: random chance of co-occurrence
    - Observed: actual co-occurrence within radius
    - Enrichment: observed / expected
    """
    from scipy.spatial.distance import cdist

    phenotype_names = list(PHENOTYPES.keys())
    colocalization = pd.DataFrame(0.0, index=phenotype_names, columns=phenotype_names)

    # Process each ROI separately
    for roi in phenotype_df['roi'].unique():
        roi_data = phenotype_df[phenotype_df['roi'] == roi]

        if len(roi_data) < 10:
            continue

        coords = roi_data[['x', 'y']].values
        dist_matrix = cdist(coords, coords)

        # For each phenotype pair
        for pheno1 in phenotype_names:
            for pheno2 in phenotype_names:
                if pheno1 == pheno2:
                    colocalization.loc[pheno1, pheno2] = 1.0
                    continue

                # Which superpixels have each phenotype?
                has_pheno1 = roi_data[pheno1].values
                has_pheno2 = roi_data[pheno2].values

                # Expected co-occurrence (random)
                p1 = has_pheno1.mean()
                p2 = has_pheno2.mean()
                expected = p1 * p2

                if expected == 0:
                    continue

                # Observed co-occurrence (within radius)
                neighbors_within_radius = dist_matrix < radius_um

                cooccurrence = 0
                total_neighbors = 0

                for i in range(len(roi_data)):
                    if has_pheno1[i]:
                        neighbors = neighbors_within_radius[i]
                        total_neighbors += neighbors.sum()
                        cooccurrence += (has_pheno2 & neighbors).sum()

                if total_neighbors > 0:
                    observed = cooccurrence / total_neighbors
                    enrichment = observed / expected if expected > 0 else 0
                    colocalization.loc[pheno1, pheno2] += enrichment

    # Average across ROIs
    n_rois = len(phenotype_df['roi'].unique())
    colocalization /= n_rois

    return colocalization


def plot_phenotype_colocalization(colocalization_df: pd.DataFrame):
    """
    Heatmap of phenotype co-localization enrichment
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(colocalization_df, annot=True, fmt='.2f', cmap='RdBu_r',
               center=1.0, vmin=0, vmax=2.0,
               cbar_kws={'label': 'Spatial Enrichment\n(>1 = attraction, <1 = avoidance)'},
               ax=ax)

    ax.set_title('Phenotype Spatial Co-localization (50μm radius)',
                fontweight='bold', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Rotate labels
    ax.set_xticklabels([x.get_text().replace('_', '\n') for x in ax.get_xticklabels()],
                       rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([y.get_text().replace('_', '\n') for y in ax.get_yticklabels()],
                       rotation=0, fontsize=9)

    plt.tight_layout()

    return fig
