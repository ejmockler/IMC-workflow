#!/usr/bin/env python
"""
Phenotype-Niche Convergence Analysis
The bridge between interpretable biology and latent spatial structure
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.sankey import Sankey
import networkx as nx
from typing import Dict, List

from phenotype_gating import PHENOTYPES


def compute_niche_phenotype_composition(phenotype_df: pd.DataFrame,
                                       niche_column: str = 'domain') -> pd.DataFrame:
    """
    For each spatial niche, compute what % of superpixels have each phenotype

    Returns DataFrame: niches × phenotypes with percentages
    """
    phenotype_names = list(PHENOTYPES.keys())
    niches = sorted(phenotype_df[niche_column].unique())

    composition = pd.DataFrame(0.0, index=niches, columns=phenotype_names)

    for niche in niches:
        niche_data = phenotype_df[phenotype_df[niche_column] == niche]

        for pheno in phenotype_names:
            composition.loc[niche, pheno] = 100 * niche_data[pheno].sum() / len(niche_data)

    return composition


def plot_niche_phenotype_heatmap(composition_df: pd.DataFrame,
                                 niche_names: Dict[int, str] = None):
    """
    Heatmap showing phenotype composition of each niche
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot heatmap
    sns.heatmap(composition_df, annot=True, fmt='.1f', cmap='YlOrRd',
               vmin=0, vmax=composition_df.values.max(),
               cbar_kws={'label': '% of Superpixels in Niche'},
               ax=ax)

    # Rename niches if provided
    if niche_names:
        yticklabels = [niche_names.get(int(idx), f'Niche {idx}')
                      for idx in composition_df.index]
        ax.set_yticklabels(yticklabels, rotation=0)

    ax.set_xlabel('Cell Phenotype', fontweight='bold', fontsize=11)
    ax.set_ylabel('Spatial Niche', fontweight='bold', fontsize=11)
    ax.set_title('Phenotype Composition of Spatial Niches:\nDo Known Cell Types Organize into Discoverable Patterns?',
                fontweight='bold', fontsize=13)

    # Rotate x labels
    ax.set_xticklabels([x.get_text().replace('_', '\n') for x in ax.get_xticklabels()],
                       rotation=45, ha='right', fontsize=9)

    plt.tight_layout()

    return fig


def compute_phenotype_niche_enrichment(phenotype_df: pd.DataFrame,
                                      niche_column: str = 'domain') -> pd.DataFrame:
    """
    For each phenotype-niche pair, compute enrichment:
    Enrichment = (% in niche) / (% overall)

    > 1 = enriched in this niche
    < 1 = depleted in this niche
    """
    phenotype_names = list(PHENOTYPES.keys())
    niches = sorted(phenotype_df[niche_column].unique())

    enrichment = pd.DataFrame(1.0, index=niches, columns=phenotype_names)

    # Overall frequencies
    overall_freq = {}
    for pheno in phenotype_names:
        overall_freq[pheno] = phenotype_df[pheno].sum() / len(phenotype_df)

    # Enrichment in each niche
    for niche in niches:
        niche_data = phenotype_df[phenotype_df[niche_column] == niche]

        for pheno in phenotype_names:
            niche_freq = niche_data[pheno].sum() / len(niche_data)
            enrichment.loc[niche, pheno] = niche_freq / overall_freq[pheno] if overall_freq[pheno] > 0 else 0

    return enrichment


def plot_phenotype_niche_enrichment(enrichment_df: pd.DataFrame,
                                   niche_names: Dict[int, str] = None):
    """
    Heatmap showing enrichment/depletion of phenotypes in niches
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot heatmap
    sns.heatmap(enrichment_df, annot=True, fmt='.2f', cmap='RdBu_r',
               center=1.0, vmin=0, vmax=3.0,
               cbar_kws={'label': 'Enrichment\n(>1 = enriched, <1 = depleted)'},
               ax=ax)

    # Rename niches if provided
    if niche_names:
        yticklabels = [niche_names.get(int(idx), f'Niche {idx}')
                      for idx in enrichment_df.index]
        ax.set_yticklabels(yticklabels, rotation=0)

    ax.set_xlabel('Cell Phenotype', fontweight='bold', fontsize=11)
    ax.set_ylabel('Spatial Niche', fontweight='bold', fontsize=11)
    ax.set_title('Phenotype Enrichment in Spatial Niches:\nWhich Cell Types Define Each Niche?',
                fontweight='bold', fontsize=13)

    # Rotate x labels
    ax.set_xticklabels([x.get_text().replace('_', '\n') for x in ax.get_xticklabels()],
                       rotation=45, ha='right', fontsize=9)

    plt.tight_layout()

    return fig


def assign_niche_identities_from_phenotypes(composition_df: pd.DataFrame,
                                           enrichment_df: pd.DataFrame) -> Dict[int, Dict]:
    """
    Automatically assign biological identities to niches based on
    phenotype composition and enrichment

    Returns dict mapping niche ID to {name, description, defining_phenotypes}
    """
    niche_identities = {}

    for niche in composition_df.index:
        # Get top 3 enriched phenotypes
        top_enriched = enrichment_df.loc[niche].nlargest(3)

        # Get top 3 abundant phenotypes
        top_abundant = composition_df.loc[niche].nlargest(3)

        # Phenotypes that are BOTH enriched AND abundant
        defining_phenotypes = list(set(top_enriched.index) & set(top_abundant.index))

        if not defining_phenotypes:
            defining_phenotypes = list(top_abundant.index[:2])

        # Generate name based on defining phenotypes
        pheno_names = [p.replace('_', ' ') for p in defining_phenotypes]

        # Simple heuristics for naming
        if 'M2_Macrophage' in defining_phenotypes and 'Activated_Endothelial_CD44' in defining_phenotypes:
            name = "Vascular-Immune Interface"
            description = "M2 macrophages co-localized with activated endothelium"
        elif 'Activated_Fibroblast' in defining_phenotypes:
            name = "Fibrotic Injury Core"
            description = "Dense activated fibroblasts driving scar formation"
        elif 'Neutrophil' in defining_phenotypes:
            name = "Acute Inflammation Front"
            description = "Neutrophil-dominated early injury response"
        elif 'Resting_Endothelial' in defining_phenotypes:
            if 'M2_Macrophage' in defining_phenotypes:
                name = "Vascular Surveillance"
                description = "Quiescent vessels with resident immune monitoring"
            else:
                name = "Quiescent Vascular"
                description = "Resting endothelium, minimal activation"
        else:
            name = f"Mixed: {' + '.join(pheno_names[:2])}"
            description = f"Characterized by {', '.join(pheno_names)}"

        niche_identities[niche] = {
            'name': name,
            'description': description,
            'defining_phenotypes': defining_phenotypes,
            'top_enrichment': dict(top_enriched),
            'top_abundance': dict(top_abundant)
        }

    return niche_identities


def plot_phenotype_niche_network(phenotype_df: pd.DataFrame,
                                 enrichment_df: pd.DataFrame,
                                 niche_identities: Dict,
                                 niche_column: str = 'domain'):
    """
    Network visualization showing phenotype-niche relationships
    Nodes = phenotypes + niches
    Edges = enrichment > 1.5
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    G = nx.Graph()

    # Add phenotype nodes
    phenotype_names = list(PHENOTYPES.keys())
    for pheno in phenotype_names:
        G.add_node(pheno, node_type='phenotype', color=PHENOTYPES[pheno]['color'])

    # Add niche nodes
    niches = sorted(phenotype_df[niche_column].unique())
    niche_colors = plt.cm.Set3(np.linspace(0, 1, len(niches)))
    for niche, color in zip(niches, niche_colors):
        niche_name = niche_identities[niche]['name']
        G.add_node(niche_name, node_type='niche', color=color)

    # Add edges for enrichment > 1.5
    for niche in niches:
        niche_name = niche_identities[niche]['name']
        for pheno in phenotype_names:
            if enrichment_df.loc[niche, pheno] > 1.5:
                weight = enrichment_df.loc[niche, pheno]
                G.add_edge(pheno, niche_name, weight=weight)

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw phenotype nodes
    pheno_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'phenotype']
    pheno_colors = [G.nodes[n]['color'] for n in pheno_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=pheno_nodes,
                          node_color=pheno_colors,
                          node_size=800, alpha=0.8, ax=ax,
                          edgecolors='black', linewidths=2)

    # Draw niche nodes
    niche_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'niche']
    niche_node_colors = [G.nodes[n]['color'] for n in niche_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=niche_nodes,
                          node_color=niche_node_colors,
                          node_size=1200, alpha=0.7, ax=ax,
                          node_shape='s',
                          edgecolors='black', linewidths=2)

    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*2 for w in weights],
                          alpha=0.4, ax=ax)

    # Labels
    labels = {n: n.replace('_', '\n') for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)

    ax.set_title('Phenotype-Niche Network: How Cell Types Organize Spatially',
                fontweight='bold', fontsize=13)
    ax.axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', label='Circle = Phenotype', alpha=0.8),
        Patch(facecolor='gray', label='Square = Niche', alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    return fig


def plot_cross_mouse_concordance(phenotype_df: pd.DataFrame,
                                 niche_column: str = 'domain'):
    """
    Do the same phenotype-niche associations hold across both mice?
    """
    phenotype_names = list(PHENOTYPES.keys())
    mice = phenotype_df['mouse'].dropna().unique()

    if len(mice) < 2:
        print("Need at least 2 mice for concordance analysis")
        return None

    mouse1, mouse2 = sorted(mice)[:2]

    # Compute enrichment for each mouse
    m1_data = phenotype_df[phenotype_df['mouse'] == mouse1]
    m2_data = phenotype_df[phenotype_df['mouse'] == mouse2]

    m1_enrichment = compute_phenotype_niche_enrichment(m1_data, niche_column)
    m2_enrichment = compute_phenotype_niche_enrichment(m2_data, niche_column)

    # Flatten to compare
    m1_flat = m1_enrichment.values.flatten()
    m2_flat = m2_enrichment.values.flatten()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(m1_flat, m2_flat, alpha=0.6, s=50, c='steelblue', edgecolors='black')
    ax.plot([0, 3], [0, 3], 'k--', alpha=0.5, label='Perfect concordance')

    ax.set_xlabel(f'Phenotype Enrichment in {mouse1}', fontweight='bold')
    ax.set_ylabel(f'Phenotype Enrichment in {mouse2}', fontweight='bold')
    ax.set_title('Cross-Mouse Concordance:\nDo Phenotypes Organize the Same Way?',
                fontweight='bold', fontsize=12)

    # Compute correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(m1_flat, m2_flat)
    ax.text(0.05, 0.95, f'R = {r:.3f}\np = {p:.2e}',
           transform=ax.transAxes, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=11, fontweight='bold')

    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    plt.tight_layout()

    return fig


def create_convergence_summary_table(niche_identities: Dict,
                                     composition_df: pd.DataFrame,
                                     enrichment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table showing niche identities based on phenotype analysis
    """
    summary_data = []

    for niche, identity in niche_identities.items():
        # Get defining phenotypes
        defining = identity['defining_phenotypes']

        # Get their enrichment and abundance
        enrichments = [f"{enrichment_df.loc[niche, p]:.2f}×" for p in defining]
        abundances = [f"{composition_df.loc[niche, p]:.1f}%" for p in defining]

        summary_data.append({
            'Niche': f"{niche}: {identity['name']}",
            'Defining Phenotypes': ', '.join([p.replace('_', ' ') for p in defining]),
            'Enrichment': ', '.join(enrichments),
            'Abundance': ', '.join(abundances),
            'Description': identity['description']
        })

    return pd.DataFrame(summary_data)
