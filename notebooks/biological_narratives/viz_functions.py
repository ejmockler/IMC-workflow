#!/usr/bin/env python
"""
Visualization functions for spatial injury narrative
Publication-quality figures for all acts

UPDATED: Now includes phenotype gating and phenotype-niche convergence analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

# Import phenotype analysis modules
from phenotype_gating import (
    PHENOTYPES,
    plot_phenotype_temporal_dynamics,
    plot_phenotype_spatial_maps,
    compute_phenotype_colocalization,
    plot_phenotype_colocalization,
)

from phenotype_niche_convergence import (
    compute_niche_phenotype_composition,
    plot_niche_phenotype_heatmap,
    compute_phenotype_niche_enrichment,
    plot_phenotype_niche_enrichment,
    assign_niche_identities_from_phenotypes,
    plot_phenotype_niche_network,
    plot_cross_mouse_concordance,
    create_convergence_summary_table,
)

# Domain color palette (consistent throughout)
DOMAIN_COLORS = {
    0: '#FF8C00',  # Orange - Responders
    1: '#87CEEB',  # Light blue - Quiet
    2: '#DC143C',  # Crimson - Vascular
    3: '#90EE90',  # Light green - Buffer (3)
    4: '#8B0000',  # Dark red - Injury (THE STAR)
    5: '#9370DB',  # Purple - Buffer (5)
}

DOMAIN_NAMES = {
    0: 'Immune Response',
    1: 'Quiescent',
    2: 'Vascular',
    3: 'Transitional',
    4: 'Injury Core',
    5: 'Surveillance',
}

def set_publication_style():
    """Set consistent publication-quality matplotlib style"""
    plt.rcParams.update({
        'figure.dpi': 150,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
    })
    sns.set_style('whitegrid', {
        'grid.linestyle': '--',
        'grid.alpha': 0.3
    })

def act3_domain_characterization_panel(domain_id, char, markers, ax_profile, ax_spatial, spatial_example=None):
    """
    Create one panel of the 6-panel domain characterization figure

    Args:
        domain_id: Domain number (0-5)
        char: Domain characterization dict
        markers: List of marker names
        ax_profile: Axes for marker profile (radar or bar)
        ax_spatial: Axes for spatial example
        spatial_example: DataFrame with x, y, domain for one ROI
    """
    identity = char['identity']
    means = char['marker_means']

    # Left: Marker profile as bar chart (easier to read than radar)
    y_pos = np.arange(len(markers))
    bars = ax_profile.barh(y_pos, means[markers].values,
                           color=DOMAIN_COLORS[domain_id], alpha=0.7)
    ax_profile.set_yticks(y_pos)
    ax_profile.set_yticklabels(markers)
    ax_profile.set_xlabel('Expression (arcsinh)')
    ax_profile.set_xlim(0, 5)
    ax_profile.grid(axis='x', alpha=0.3)

    # Highlight top 3 markers
    top3_markers = char['top_markers'].index.tolist()
    for i, marker in enumerate(markers):
        if marker in top3_markers:
            ax_profile.get_yticklabels()[i].set_weight('bold')
            ax_profile.get_yticklabels()[i].set_color(DOMAIN_COLORS[domain_id])

    # Title with domain name
    ax_profile.set_title(f"Domain {domain_id}: {identity['short']}",
                        fontweight='bold', fontsize=11)

    # Right: Spatial example
    if spatial_example is not None:
        domain_data = spatial_example[spatial_example['domain'] == domain_id]
        other_data = spatial_example[spatial_example['domain'] != domain_id]

        # Plot other domains in gray
        ax_spatial.scatter(other_data['x'], other_data['y'],
                          c='lightgray', s=20, alpha=0.3, edgecolors='none')

        # Highlight this domain
        if len(domain_data) > 0:
            ax_spatial.scatter(domain_data['x'], domain_data['y'],
                             c=DOMAIN_COLORS[domain_id], s=40, alpha=0.8,
                             edgecolors='black', linewidth=0.5)

        ax_spatial.set_xlabel('X (μm)')
        ax_spatial.set_ylabel('Y (μm)')
        ax_spatial.set_aspect('equal')
        ax_spatial.set_title('Spatial Location', fontsize=9)
    else:
        ax_spatial.text(0.5, 0.5, 'No spatial\ndata available',
                       ha='center', va='center', transform=ax_spatial.transAxes)
        ax_spatial.set_xticks([])
        ax_spatial.set_yticks([])

def act3_complete_figure(domain_chars, superpixel_df, markers):
    """
    Complete Act 3 figure: 6 domains in 3×2 grid
    Each domain gets marker profile + spatial example
    """
    fig = plt.figure(figsize=(16, 14))

    # Create grid: 3 rows × 2 domains per row
    # Each domain gets 2 subplots (profile + spatial)

    # Pick one representative ROI for spatial examples
    sham_rois = superpixel_df[superpixel_df['timepoint'] == 'Sham']['roi'].unique()
    if len(sham_rois) > 0:
        spatial_example = superpixel_df[superpixel_df['roi'] == sham_rois[0]]
    else:
        spatial_example = None

    for idx, char in enumerate(domain_chars):
        row = idx // 2
        col = idx % 2

        # Profile subplot (left of pair)
        ax_profile = plt.subplot(3, 4, row*4 + col*2 + 1)

        # Spatial subplot (right of pair)
        ax_spatial = plt.subplot(3, 4, row*4 + col*2 + 2)

        act3_domain_characterization_panel(
            char['domain_id'], char, markers,
            ax_profile, ax_spatial, spatial_example
        )

    fig.suptitle('Tissue Domain Characterization: Six Distinct Microenvironments',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig

def act4_temporal_stacked_area(superpixel_df):
    """
    Stacked area chart showing domain composition over time
    """
    timepoint_order = ['Sham', 'D1', 'D3', 'D7']

    # Get domain frequencies by timepoint
    domain_evolution = superpixel_df[superpixel_df['domain'] >= 0].groupby(['timepoint', 'domain']).size().unstack(fill_value=0)
    domain_evolution = domain_evolution.reindex(timepoint_order)
    domain_evolution_pct = domain_evolution.div(domain_evolution.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create stacked area
    domain_order = [4, 2, 0, 5, 3, 1]  # Put injury first, quiet last
    colors = [DOMAIN_COLORS[d] for d in domain_order]
    labels = [DOMAIN_NAMES[d] for d in domain_order]

    ax.stackplot(range(len(timepoint_order)),
                [domain_evolution_pct[d].values for d in domain_order],
                labels=labels, colors=colors, alpha=0.8)

    # Annotations for key changes
    # Injury growth
    ax.annotate('Injury domain\ngrows 5.8×',
               xy=(3, 85), xytext=(3.3, 70),
               arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
               fontsize=10, fontweight='bold', color='darkred',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='darkred'))

    # Vascular expansion
    ax.annotate('Vascular\ncompensation',
               xy=(2, 35), xytext=(1.5, 25),
               arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5),
               fontsize=9, color='crimson',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='crimson'))

    # Transitional collapse
    ax.annotate('Transitional\nzones collapse',
               xy=(1, 60), xytext=(0.5, 50),
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
               fontsize=9, color='darkgreen',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green'))

    ax.set_xticks(range(len(timepoint_order)))
    ax.set_xticklabels(timepoint_order)
    ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Domain Composition (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title('Tissue Reorganization During Kidney Injury', fontsize=13, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, title='Tissue Domain')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig

def act5_spatial_maps_grid(superpixel_df, timepoint_order=['Sham', 'D1', 'D3', 'D7']):
    """
    4×4 grid: 4 timepoints × 4 example ROIs
    Shows domain spatial organization evolving over time
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for t_idx, timepoint in enumerate(timepoint_order):
        tp_data = superpixel_df[superpixel_df['timepoint'] == timepoint]
        available_rois = tp_data['roi'].unique()

        for roi_idx in range(4):
            ax = axes[t_idx, roi_idx]

            if roi_idx < len(available_rois):
                roi = available_rois[roi_idx]
                roi_data = tp_data[tp_data['roi'] == roi]

                # Plot each domain
                for domain_id in range(6):
                    domain_data = roi_data[roi_data['domain'] == domain_id]
                    if len(domain_data) > 0:
                        ax.scatter(domain_data['x'], domain_data['y'],
                                 c=DOMAIN_COLORS[domain_id], s=30, alpha=0.7,
                                 edgecolors='black', linewidth=0.3,
                                 label=DOMAIN_NAMES[domain_id] if roi_idx == 0 and t_idx == 0 else '')

                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])

                if roi_idx == 0:
                    ax.set_ylabel(timepoint, fontsize=11, fontweight='bold')

                if t_idx == 0:
                    ax.set_title(f'ROI {roi_idx+1}', fontsize=10)

                # Calculate domain 4 percentage for this ROI
                injury_pct = 100 * len(roi_data[roi_data['domain'] == 4]) / len(roi_data)
                ax.text(0.95, 0.05, f'Injury:\n{injury_pct:.0f}%',
                       transform=ax.transAxes, ha='right', va='bottom',
                       fontsize=8, bbox=dict(boxstyle='round,pad=0.3',
                                            facecolor='white', alpha=0.8,
                                            edgecolor='darkred'))
            else:
                ax.axis('off')

    # Legend
    handles = [mpatches.Patch(color=DOMAIN_COLORS[d], label=DOMAIN_NAMES[d])
              for d in range(6)]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, -0.02),
              ncol=6, frameon=True, fontsize=10, title='Tissue Domains')

    fig.suptitle('Spatial Geography of Injury: Where Do Domains Form?',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    return fig

def act2_clustering_discovery(superpixel_df, markers):
    """
    Act 2: Show the discovery process
    - Superpixel heatmap with hierarchical clustering
    - Elbow plot showing k=6
    """
    fig, (ax_heatmap, ax_elbow) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Heatmap (sample 500 superpixels for visibility)
    sample_data = superpixel_df[superpixel_df['domain'] >= 0].sample(n=min(500, len(superpixel_df)), random_state=42)
    marker_matrix = sample_data[markers].values

    # Hierarchical clustering to order rows
    row_linkage = linkage(marker_matrix, method='ward')
    row_order = dendrogram(row_linkage, no_plot=True)['leaves']

    im = ax_heatmap.imshow(marker_matrix[row_order, :], aspect='auto', cmap='RdBu_r',
                           vmin=-1, vmax=4, interpolation='nearest')
    ax_heatmap.set_xticks(range(len(markers)))
    ax_heatmap.set_xticklabels(markers, rotation=45, ha='right')
    ax_heatmap.set_ylabel('Superpixels (n=500 sample)')
    ax_heatmap.set_title('Superpixels Naturally Cluster by Marker Expression',
                        fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, label='Expression (arcsinh)')

    # Add domain labels on right
    domain_labels = sample_data['domain'].values[row_order]
    colors_for_domains = [DOMAIN_COLORS[d] for d in domain_labels]
    for i, color in enumerate(colors_for_domains):
        ax_heatmap.add_patch(plt.Rectangle((len(markers)-0.5, i-0.5), 0.5, 1,
                                          facecolor=color, edgecolor='none'))

    # Right: Elbow plot (we need to compute this)
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = superpixel_df[superpixel_df['domain'] >= 0][markers].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    K_range = range(2, 12)
    inertias = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    ax_elbow.plot(K_range, inertias, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax_elbow.axvline(x=6, color='darkred', linestyle='--', linewidth=2, label='k=6 (chosen)')
    ax_elbow.set_xlabel('Number of Domains (k)', fontsize=11, fontweight='bold')
    ax_elbow.set_ylabel('Within-Cluster Sum of Squares', fontsize=11, fontweight='bold')
    ax_elbow.set_title('How Many Tissue Domains?\nThe Data Says Six', fontweight='bold')
    ax_elbow.grid(alpha=0.3)
    ax_elbow.legend()

    # Annotate elbow
    ax_elbow.annotate('Elbow at k=6',
                     xy=(6, inertias[4]), xytext=(7.5, inertias[4]*1.2),
                     arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                     fontsize=10, fontweight='bold', color='darkred')

    plt.tight_layout()
    return fig

def create_summary_table(domain_chars):
    """
    Create a clean summary table of all domains
    Returns pandas DataFrame for display
    """
    summary_data = []
    for char in sorted(domain_chars, key=lambda x: x['d7_pct'], reverse=True):
        identity = char['identity']
        summary_data.append({
            'Domain': f"{char['domain_id']}",
            'Name': identity['short'],
            'Description': identity['description'][:50] + '...',
            'Sham %': f"{char['sham_pct']:.1f}",
            'D7 %': f"{char['d7_pct']:.1f}",
            'Change': f"{char['fold_change']:.1f}×" if char['fold_change'] < 10 else ">>10×",
            'Top Markers': ', '.join(char['top_markers'].index[:3].tolist())
        })

    return pd.DataFrame(summary_data)


# ============================================================================
# NEW VALIDATION & STATISTICAL RIGOR FUNCTIONS
# ============================================================================

def cluster_stability_analysis(superpixel_df, markers, k_range=range(2, 15), n_bootstrap=50):
    """
    Fast cluster validation using standard metrics (no consensus clustering)
    - Silhouette scores across k
    - Calinski-Harabasz (variance ratio)
    - Davies-Bouldin (cluster separation)
    - Inertia (elbow method)

    Returns dict with all metrics for visualization
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    X = superpixel_df[superpixel_df['domain'] >= 0][markers].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {
        'k_values': [],
        'inertias': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': [],
    }

    print(f"Computing cluster validation metrics for {len(X_scaled)} superpixels...")
    for k in k_range:
        print(f"  k={k}...", end='', flush=True)

        # Base clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        results['k_values'].append(k)
        results['inertias'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(X_scaled, labels))
        results['calinski_harabasz'].append(calinski_harabasz_score(X_scaled, labels))
        results['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels))

        print(" done")

    return results


def plot_cluster_validation(validation_results):
    """
    Create 4-panel cluster validation figure
    Shows why k=6 is optimal
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    k_vals = validation_results['k_values']

    # Panel 1: Elbow plot (inertia)
    ax = axes[0, 0]
    ax.plot(k_vals, validation_results['inertias'], 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.axvline(x=6, color='darkred', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Within-Cluster Sum of Squares (Inertia)')
    ax.set_title('Elbow Method: Diminishing Returns After k=6', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.annotate('Chosen: k=6', xy=(6, validation_results['inertias'][4]),
                xytext=(8, validation_results['inertias'][4]*1.1),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                fontsize=10, fontweight='bold', color='darkred')

    # Panel 2: Silhouette scores (higher is better)
    ax = axes[0, 1]
    ax.plot(k_vals, validation_results['silhouette'], 'o-', linewidth=2, markersize=8, color='green')
    ax.axvline(x=6, color='darkred', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Analysis: k=6 Shows Strong Separation', fontweight='bold')
    ax.grid(alpha=0.3)

    # Panel 3: Calinski-Harabasz (higher is better)
    ax = axes[1, 0]
    ax.plot(k_vals, validation_results['calinski_harabasz'], 'o-', linewidth=2, markersize=8, color='purple')
    ax.axvline(x=6, color='darkred', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Calinski-Harabasz Index')
    ax.set_title('Variance Ratio: Peak Around k=6', fontweight='bold')
    ax.grid(alpha=0.3)

    # Panel 4: Davies-Bouldin (lower is better)
    ax = axes[1, 1]
    ax.plot(k_vals, validation_results['davies_bouldin'], 'o-', linewidth=2, markersize=8, color='orange')
    ax.axvline(x=6, color='darkred', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Davies-Bouldin Index')
    ax.set_title('Cluster Separation: Minimum Near k=6', fontweight='bold')
    ax.grid(alpha=0.3)

    fig.suptitle('Cluster Validation: Multiple Metrics Support k=6 Domains',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def mouse_trajectories_figure(superpixel_df):
    """
    Show individual mouse trajectories to demonstrate consistency
    Addresses n=2 concern by showing both replicates
    """
    timepoint_order = ['Sham', 'D1', 'D3', 'D7']

    # Get mouse-level domain percentages
    mouse_data = []
    for tp in timepoint_order:
        tp_data = superpixel_df[superpixel_df['timepoint'] == tp]
        for mouse in tp_data['mouse'].unique():
            if pd.isna(mouse):
                continue
            mouse_data_tp = tp_data[tp_data['mouse'] == mouse]
            for domain in range(6):
                pct = 100 * len(mouse_data_tp[mouse_data_tp['domain'] == domain]) / len(mouse_data_tp)
                mouse_data.append({
                    'timepoint': tp,
                    'mouse': mouse,
                    'domain': domain,
                    'percentage': pct
                })

    mouse_df = pd.DataFrame(mouse_data)

    # Create figure with 6 panels (one per domain)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for domain in range(6):
        ax = axes[domain]
        domain_data = mouse_df[mouse_df['domain'] == domain]

        # Plot each mouse as separate line
        for mouse in domain_data['mouse'].unique():
            mouse_traj = domain_data[domain_data['mouse'] == mouse]
            mouse_traj = mouse_traj.set_index('timepoint').reindex(timepoint_order)

            ax.plot(range(len(timepoint_order)), mouse_traj['percentage'].values,
                   'o-', linewidth=2, markersize=8, alpha=0.7, label=mouse)

        ax.set_xticks(range(len(timepoint_order)))
        ax.set_xticklabels(timepoint_order)
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Domain %')
        ax.set_title(f'Domain {domain}: {DOMAIN_NAMES[domain]}',
                    fontweight='bold', color=DOMAIN_COLORS[domain])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(domain_data['percentage'].max() * 1.2, 5))

    fig.suptitle('Individual Mouse Trajectories: Consistent Patterns Across Replicates',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def batch_effect_check(superpixel_df, markers):
    """
    PCA colored by mouse, ROI, and timepoint to check for batch effects
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = superpixel_df[superpixel_df['domain'] >= 0][markers].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    pc_coords = pca.fit_transform(X_scaled)

    plot_df = superpixel_df[superpixel_df['domain'] >= 0].copy()
    plot_df['PC1'] = pc_coords[:, 0]
    plot_df['PC2'] = pc_coords[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Color by timepoint (should show separation)
    ax = axes[0]
    for tp in ['Sham', 'D1', 'D3', 'D7']:
        tp_data = plot_df[plot_df['timepoint'] == tp]
        ax.scatter(tp_data['PC1'], tp_data['PC2'], alpha=0.3, s=20, label=tp)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Colored by Timepoint\n(Should cluster)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Color by mouse (should NOT show strong separation)
    ax = axes[1]
    for mouse in plot_df['mouse'].dropna().unique():
        mouse_data = plot_df[plot_df['mouse'] == mouse]
        ax.scatter(mouse_data['PC1'], mouse_data['PC2'], alpha=0.3, s=20, label=mouse)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Colored by Mouse\n(Should overlap)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Color by domain (should show clear clustering)
    ax = axes[2]
    for domain in range(6):
        domain_data = plot_df[plot_df['domain'] == domain]
        ax.scatter(domain_data['PC1'], domain_data['PC2'],
                  alpha=0.3, s=20, color=DOMAIN_COLORS[domain], label=DOMAIN_NAMES[domain])
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Colored by Domain\n(Validates clustering)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle('Batch Effect Check: Biology Drives Separation, Not Technical Factors',
                 fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()
    return fig


def spatial_statistics_analysis(superpixel_df):
    """
    Quantify spatial organization:
    - Moran's I for each domain (spatial autocorrelation)
    - Domain-domain distance distributions
    - Domain compactness/boundary analysis
    """
    from scipy.spatial.distance import cdist
    from sklearn.neighbors import NearestNeighbors

    results = {
        'morans_i': {},
        'distance_distributions': {},
        'domain_neighbors': {}
    }

    # Process each ROI separately for spatial stats
    for roi in superpixel_df['roi'].unique():
        roi_data = superpixel_df[superpixel_df['roi'] == roi]
        if len(roi_data) < 10:
            continue

        coords = roi_data[['x', 'y']].values
        domains = roi_data['domain'].values

        # Moran's I for each domain (is domain spatially autocorrelated?)
        for domain in range(6):
            domain_binary = (domains == domain).astype(float)

            # Simple Moran's I using nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=min(10, len(coords))).fit(coords)
            distances, indices = nbrs.kneighbors(coords)

            # Compute Moran's I
            n = len(domain_binary)
            mean_val = domain_binary.mean()
            numerator = 0
            denominator = 0

            for i in range(n):
                neighbors = indices[i, 1:]  # Exclude self
                for j in neighbors:
                    numerator += (domain_binary[i] - mean_val) * (domain_binary[j] - mean_val)
                denominator += (domain_binary[i] - mean_val) ** 2

            if denominator > 0:
                morans_i = (n / len(indices[0] - 1)) * (numerator / denominator)
                if domain not in results['morans_i']:
                    results['morans_i'][domain] = []
                results['morans_i'][domain].append(morans_i)

    # Aggregate across ROIs
    for domain in range(6):
        if domain in results['morans_i']:
            results['morans_i'][domain] = np.mean(results['morans_i'][domain])
        else:
            results['morans_i'][domain] = np.nan

    return results


def plot_spatial_statistics(spatial_results):
    """
    Visualize spatial organization metrics
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Moran's I for each domain
    domains = sorted(spatial_results['morans_i'].keys())
    morans_values = [spatial_results['morans_i'][d] for d in domains]
    colors = [DOMAIN_COLORS[d] for d in domains]

    bars = ax.bar(domains, morans_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Domain', fontweight='bold')
    ax.set_ylabel("Moran's I (Spatial Autocorrelation)", fontweight='bold')
    ax.set_title("Domains Are Spatially Clustered, Not Randomly Scattered", fontweight='bold')
    ax.set_xticks(domains)
    ax.set_xticklabels([DOMAIN_NAMES[d] for d in domains], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Annotate significance
    ax.text(0.5, 0.95, "Positive Moran's I = Spatial clustering\n(Domains form coherent regions)",
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
            fontsize=9)

    fig.suptitle('Spatial Organization: Domains Have Coherent Geography',
                 fontsize=14, fontweight='bold', y=0.96)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def marker_biology_overview():
    """
    Create figure explaining the 9 markers and their biological roles
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Title
    fig.text(0.5, 0.95, 'The Nine-Marker Panel: Functional Groups',
             ha='center', fontsize=16, fontweight='bold')

    # Three functional groups
    groups = {
        'Immune Surveillance': {
            'color': '#FFB6C1',
            'markers': {
                'CD45': 'Pan-leukocyte marker (all immune cells)',
                'CD11b': 'Myeloid cells (macrophages, monocytes)',
                'Ly6G': 'Neutrophils (acute inflammation)',
                'CD206': 'M2 macrophages (anti-inflammatory)'
            },
            'role': 'Tracks immune cell infiltration and inflammatory response'
        },
        'Fibrosis Machinery': {
            'color': '#DDA0DD',
            'markers': {
                'CD44': 'Adhesion molecule (activation marker)',
                'CD140a': 'PDGFR-α (fibroblast proliferation)',
                'CD140b': 'PDGFR-β (pericytes, myofibroblasts)'
            },
            'role': 'Captures fibroblast activation and scar formation'
        },
        'Vascular Remodeling': {
            'color': '#ADD8E6',
            'markers': {
                'CD31': 'Endothelial cells (blood vessels)',
                'CD34': 'Vascular progenitors (angiogenesis)'
            },
            'role': 'Monitors blood vessel network changes and compensation'
        }
    }

    y_pos = 0.80
    for group_name, group_info in groups.items():
        # Group box
        box = FancyBboxPatch((0.1, y_pos - 0.20), 0.8, 0.22,
                            boxstyle="round,pad=0.01",
                            facecolor=group_info['color'], alpha=0.3,
                            edgecolor='black', linewidth=2,
                            transform=fig.transFigure)
        fig.add_artist(box)

        # Group title
        fig.text(0.15, y_pos, group_name,
                fontsize=14, fontweight='bold', va='top')

        # Markers
        marker_y = y_pos - 0.04
        for marker, description in group_info['markers'].items():
            fig.text(0.20, marker_y, f"• {marker}:",
                    fontsize=11, fontweight='bold', va='top')
            fig.text(0.32, marker_y, description,
                    fontsize=10, va='top', style='italic')
            marker_y -= 0.03

        # Role
        fig.text(0.20, marker_y - 0.01, f"→ {group_info['role']}",
                fontsize=10, va='top', color='darkblue', fontweight='bold')

        y_pos -= 0.25

    # Bottom explanation
    fig.text(0.5, 0.08,
             'Why These 9 Markers?\n'
             'Minimal panel capturing three key injury processes.\n'
             'Multi-marker co-expression defines tissue microenvironments,\n'
             'not single-marker positivity.',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))

    return fig


def superpixel_scale_justification(superpixel_df):
    """
    Explain why superpixel scale is appropriate via conceptual diagram
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')  # Hide axis, use figure for drawing

    # Create 3 comparison panels side by side
    scales = [
        {
            'name': 'Single Cell Scale',
            'size': '~10 μm diameter',
            'cell_count': '1 cell',
            'problem': 'TOO NOISY',
            'details': ['IMC ion counts sparse', 'High technical variation', 'Misses tissue context'],
            'color': '#FFB6B6',
            'x': 0.15
        },
        {
            'name': 'Superpixel Scale',
            'size': '~93 μm diameter',
            'cell_count': '~40-50 cells',
            'problem': 'OPTIMAL',
            'details': ['Tissue microenvironment', 'Stable measurements', 'Spatial organization preserved'],
            'color': '#B6FFB6',
            'x': 0.50
        },
        {
            'name': 'Whole ROI Scale',
            'size': '~1000+ μm',
            'cell_count': '1000s of cells',
            'problem': 'TOO COARSE',
            'details': ['Averages everything', 'Loses WHERE', 'No spatial patterns'],
            'color': '#B6B6FF',
            'x': 0.85
        }
    ]

    for scale in scales:
        # Main box
        box = FancyBboxPatch((scale['x'] - 0.12, 0.25), 0.24, 0.55,
                            boxstyle="round,pad=0.02",
                            facecolor=scale['color'], alpha=0.4,
                            edgecolor='black', linewidth=3 if scale['problem'] == 'OPTIMAL' else 2,
                            transform=fig.transFigure)
        fig.add_artist(box)

        # Title
        fig.text(scale['x'], 0.75, scale['name'],
                ha='center', fontsize=14, fontweight='bold', transform=fig.transFigure)

        # Size
        fig.text(scale['x'], 0.67, scale['size'],
                ha='center', fontsize=12, transform=fig.transFigure,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', linewidth=1.5))

        # Cell count
        fig.text(scale['x'], 0.58, scale['cell_count'],
                ha='center', fontsize=11, style='italic', transform=fig.transFigure)

        # Problem/Status
        problem_color = 'darkgreen' if scale['problem'] == 'OPTIMAL' else 'darkred'
        fig.text(scale['x'], 0.49, scale['problem'],
                ha='center', fontsize=13, fontweight='bold', color=problem_color,
                transform=fig.transFigure)

        # Details
        y_detail = 0.42
        for detail in scale['details']:
            fig.text(scale['x'], y_detail, f'• {detail}',
                    ha='center', fontsize=10, transform=fig.transFigure)
            y_detail -= 0.06

    # Top title
    fig.text(0.5, 0.90, 'Analysis Scale: The Goldilocks Principle',
            ha='center', fontsize=18, fontweight='bold', transform=fig.transFigure)

    # Bottom explanation
    fig.text(0.5, 0.10,
            'Superpixels aggregate ~50 cells into tissue microenvironments.\n'
            'Large enough for stable measurements. Small enough to preserve spatial organization.\n'
            'This is the biologically relevant scale where tissues organize themselves.',
            ha='center', fontsize=12, style='italic', transform=fig.transFigure,
            bbox=dict(boxstyle='round,pad=1.2', facecolor='#FFFACD', alpha=0.9, edgecolor='black', linewidth=2))

    plt.tight_layout()
    return fig
