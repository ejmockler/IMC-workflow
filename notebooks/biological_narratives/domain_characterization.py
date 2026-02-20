#!/usr/bin/env python
"""
Deep characterization of the 6 spatial tissue domains
Foundation for Act 3 of the spatial injury narrative
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import gzip
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

def deserialize_array(arr_dict):
    """Convert serialized numpy array back to numpy array"""
    if isinstance(arr_dict, dict) and '__numpy_array__' in arr_dict:
        return np.array(arr_dict['data'], dtype=arr_dict['dtype']).reshape(arr_dict['shape'])
    return arr_dict

def load_and_cluster_domains():
    """
    Load data, perform domain clustering, AND assign phenotypes
    Returns enriched dataframe with both domains and phenotypes
    """
    print("Loading superpixel data...")
    results_dir = Path('/Users/noot/Documents/IMC/results/roi_results')
    result_files = sorted(results_dir.glob('roi_*.json.gz'))

    all_superpixels = []
    markers = ['CD45', 'CD11b', 'Ly6G', 'CD140a', 'CD140b', 'CD31', 'CD34', 'CD206', 'CD44']

    for result_file in result_files:
        try:
            roi_name = result_file.stem.replace('roi_', '').replace('_results.json', '')

            if 'Sam' in roi_name:
                condition = 'Sham'
                timepoint = 'Sham'
                parts = roi_name.split('_ROI_')
                mouse = parts[1].split('_')[0] if len(parts) > 1 else 'Unknown'
            else:
                parts = roi_name.split('_ROI_')
                if len(parts) < 2:
                    continue
                timepoint_parts = parts[1].split('_')
                timepoint = timepoint_parts[0]
                mouse = timepoint_parts[1]
                condition = 'UUO'

            with gzip.open(result_file, 'rt') as f:
                data = json.load(f)

            if 'multiscale_results' not in data:
                continue

            scales = data['multiscale_results']
            scale_10 = scales.get('10.0')
            if scale_10 is None:
                continue

            transformed = scale_10.get('transformed_arrays', {})
            coords_dict = scale_10.get('superpixel_coords', {})
            coords = deserialize_array(coords_dict)

            # Load actual Leiden cluster labels from results
            cluster_labels_dict = scale_10.get('cluster_labels', {})
            cluster_labels = deserialize_array(cluster_labels_dict)

            if coords is None or len(coords) == 0:
                continue

            n_superpixels = len(coords)

            for i in range(n_superpixels):
                spx_data = {
                    'roi': roi_name,
                    'condition': condition,
                    'timepoint': timepoint,
                    'mouse': mouse,
                    'superpixel_id': i,
                    'x': coords[i, 0],
                    'y': coords[i, 1],
                    'cluster': int(cluster_labels[i]) if cluster_labels is not None and i < len(cluster_labels) else -1
                }

                for marker in markers:
                    if marker in transformed:
                        marker_data = transformed[marker]
                        if isinstance(marker_data, dict) and 'data' in marker_data:
                            values = marker_data['data']
                            spx_data[marker] = values[i] if i < len(values) else np.nan
                        else:
                            spx_data[marker] = np.nan
                    else:
                        spx_data[marker] = np.nan

                all_superpixels.append(spx_data)

        except Exception as e:
            continue

    superpixel_df = pd.DataFrame(all_superpixels)

    print(f"Loaded {len(superpixel_df)} superpixels")

    # Clusters are already loaded from Leiden results (ROI-specific cluster IDs)
    n_total_clusters = len(superpixel_df[superpixel_df['cluster'] >= 0])
    n_clusters_per_roi = superpixel_df[superpixel_df['cluster'] >= 0].groupby('roi')['cluster'].nunique()
    print(f"Using actual Leiden clustering results:")
    print(f"  - {n_clusters_per_roi.mean():.1f} ± {n_clusters_per_roi.std():.1f} clusters per ROI (range: {n_clusters_per_roi.min()}-{n_clusters_per_roi.max()})")

    # Assign phenotypes (supervised boolean gating)
    print("Assigning cell phenotypes via boolean gating...")
    from phenotype_gating import assign_phenotypes
    superpixel_df = assign_phenotypes(superpixel_df, markers)

    return superpixel_df, markers

def characterize_domain(superpixel_df, domain_id, markers, all_domains_df):
    """Deep characterization of a single domain"""
    domain_data = superpixel_df[superpixel_df['domain'] == domain_id]

    # Basic stats
    n_superpixels = len(domain_data)
    pct_total = 100 * n_superpixels / len(superpixel_df[superpixel_df['domain'] >= 0])

    # Marker profile (mean and std)
    marker_means = domain_data[markers].mean()
    marker_stds = domain_data[markers].std()

    # Top 3 defining markers (highest relative to other domains)
    all_means = all_domains_df.groupby('domain')[markers].mean()
    relative_expression = marker_means / all_means.mean()
    top_markers = relative_expression.nlargest(3)

    # Temporal distribution
    temporal_dist = domain_data.groupby('timepoint').size()
    timepoint_order = ['Sham', 'D1', 'D3', 'D7']
    temporal_pct = {}
    for tp in timepoint_order:
        if tp in temporal_dist.index:
            tp_total = len(superpixel_df[(superpixel_df['timepoint'] == tp) & (superpixel_df['domain'] >= 0)])
            temporal_pct[tp] = 100 * temporal_dist[tp] / tp_total if tp_total > 0 else 0
        else:
            temporal_pct[tp] = 0

    # Temporal change
    sham_pct = temporal_pct['Sham']
    d7_pct = temporal_pct['D7']
    fold_change = d7_pct / sham_pct if sham_pct > 0 else np.inf
    absolute_change = d7_pct - sham_pct

    # Marker co-expression pattern (which markers are correlated)
    marker_corr = domain_data[markers].corr()

    return {
        'domain_id': domain_id,
        'n_superpixels': n_superpixels,
        'pct_total': pct_total,
        'marker_means': marker_means,
        'marker_stds': marker_stds,
        'top_markers': top_markers,
        'temporal_pct': temporal_pct,
        'sham_pct': sham_pct,
        'd7_pct': d7_pct,
        'fold_change': fold_change,
        'absolute_change': absolute_change,
        'marker_corr': marker_corr
    }

def assign_biological_identity(char):
    """Assign biological name based on marker profile"""
    domain_id = char['domain_id']
    means = char['marker_means']
    top3 = char['top_markers']

    # Decision logic based on marker patterns

    # Domain 4: High CD44, CD11b, CD140a/b = Fibrotic/Immune Injury
    if means['CD44'] > 3.5 and means['CD11b'] > 3.5 and means['CD140a'] > 2.5:
        return {
            'name': 'Fibrotic Injury Core',
            'short': 'Injury',
            'description': 'Active fibrosis with dense immune infiltrate',
            'cellular_composition': 'CD44+ fibroblasts, CD11b+ myeloid cells, activated CD140a+ pericytes',
            'biological_state': 'Scar tissue formation, chronic inflammation',
            'trajectory': f'Expands dramatically: {char["sham_pct"]:.1f}% → {char["d7_pct"]:.1f}% ({char["fold_change"]:.1f}×)'
        }

    # Domain 2: High CD31, CD34, CD140b = Vascular
    elif means['CD31'] > 3.0 and means['CD34'] > 2.5:
        return {
            'name': 'Vascular Network',
            'short': 'Vascular',
            'description': 'Blood vessel-enriched regions',
            'cellular_composition': 'CD31+ endothelium, CD34+ vascular progenitors, CD140b+ pericytes',
            'biological_state': 'Active perfusion, angiogenic response',
            'trajectory': f'Compensatory expansion: {char["sham_pct"]:.1f}% → {char["d7_pct"]:.1f}% ({char["fold_change"]:.1f}×)'
        }

    # Domain 1: Low everything = Quiescent
    elif means.max() < 1.5:
        return {
            'name': 'Quiescent Baseline',
            'short': 'Quiet',
            'description': 'Minimal marker expression, low activity',
            'cellular_composition': 'Resting cells, sparse immune surveillance',
            'biological_state': 'Homeostatic, unperturbed tissue',
            'trajectory': f'Shrinks: {char["sham_pct"]:.1f}% → {char["d7_pct"]:.1f}% (activation response)'
        }

    # Domain 3: Moderate everything = Transitional
    elif means.std() < 0.5 and 1.5 < means.mean() < 2.2:
        return {
            'name': 'Transitional Buffer',
            'short': 'Buffer',
            'description': 'Intermediate expression across markers',
            'cellular_composition': 'Mixed cell types, transitioning states',
            'biological_state': 'Pre-injury stressed tissue, fate undecided',
            'trajectory': f'Collapses: {char["sham_pct"]:.1f}% → {char["d7_pct"]:.1f}% (converts to injury)'
        }

    # Domain 0: High CD11b, CD206, moderate fibrosis = Active Response
    elif means['CD11b'] > 2.5 and means['CD206'] > 2.0:
        return {
            'name': 'Immune Response Zone',
            'short': 'Responders',
            'description': 'Active immune surveillance and response',
            'cellular_composition': 'CD11b+ myeloid cells, CD206+ M2 macrophages, moderate inflammation',
            'biological_state': 'Controlled immune activation, tissue monitoring',
            'trajectory': f'Stable: {char["sham_pct"]:.1f}% → {char["d7_pct"]:.1f}% (maintains vigilance)'
        }

    # Domain 5: Moderate vascular + low immune = Surveillance
    elif means['CD31'] > 2.0 and means['CD206'] > 1.5 and means['CD45'] < 1.5:
        return {
            'name': 'Surveillance Regions',
            'short': 'Watchers',
            'description': 'Vascular + low-level immune monitoring',
            'cellular_composition': 'CD31+ vessels, CD206+ resident macrophages, sparse immune cells',
            'biological_state': 'Passive surveillance, ready to activate',
            'trajectory': f'Shrinks: {char["sham_pct"]:.1f}% → {char["d7_pct"]:.1f}% (activation demands)'
        }

    else:
        return {
            'name': f'Domain {domain_id}',
            'short': f'D{domain_id}',
            'description': 'Complex mixed pattern',
            'cellular_composition': 'Multiple cell types',
            'biological_state': 'Undetermined',
            'trajectory': f'{char["sham_pct"]:.1f}% → {char["d7_pct"]:.1f}%'
        }

def main():
    """
    Load superpixel data with actual Leiden clustering and phenotype assignments.

    NOTE: This file previously ran global K-means clustering (k=6) across all ROIs.
    Now it loads the actual Leiden clustering results, which are ROI-specific (6-18 clusters per ROI).

    The domain characterization functions below need refactoring to work with ROI-specific
    clusters rather than global domains. For now, this function just loads and returns the data.

    For analysis using these results, see kidney_injury_spatial_analysis.ipynb
    """
    # Load data with Leiden clusters
    superpixel_df, markers = load_and_cluster_domains()

    print(f"\n{'='*80}")
    print("DATA LOADED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nTotal superpixels: {len(superpixel_df)}")
    print(f"ROIs: {superpixel_df['roi'].nunique()}")
    print(f"Timepoints: {sorted(superpixel_df['timepoint'].unique())}")
    print(f"Mice: {sorted(superpixel_df['mouse'].unique())}")

    # Cluster summary
    cluster_summary = superpixel_df[superpixel_df['cluster'] >= 0].groupby('roi')['cluster'].nunique()
    print(f"\nClusters per ROI: {cluster_summary.mean():.1f} ± {cluster_summary.std():.1f}")
    print(f"Range: {cluster_summary.min()} - {cluster_summary.max()} clusters")

    # Phenotype summary (if phenotypes were assigned)
    if 'phenotype' in superpixel_df.columns:
        phenotype_counts = superpixel_df['phenotype'].value_counts()
        print(f"\nPhenotype distribution:")
        for pheno, count in phenotype_counts.items():
            pct = 100 * count / len(superpixel_df)
            print(f"  {pheno}: {count} ({pct:.1f}%)")

    print(f"\n{'='*80}\n")

    return superpixel_df, markers

if __name__ == '__main__':
    superpixel_df, markers = main()
