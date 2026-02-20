"""
Examine actual results from 18 ROIs to understand what story the data tells.
This is a DATA INTERROGATION, not infrastructure building.
"""

import gzip
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def deserialize_array(arr_dict):
    """Convert stored numpy array back to numpy array"""
    if isinstance(arr_dict, dict) and '__numpy_array__' in arr_dict:
        data = arr_dict['data']
        dtype = arr_dict['dtype']
        shape = arr_dict['shape']
        return np.array(data, dtype=dtype).reshape(shape)
    return arr_dict


def parse_roi_metadata(filename):
    """Extract timepoint and mouse from filename"""
    # roi_IMC_241218_Alun_ROI_D7_M2_01_24_results.json.gz
    if 'Test01' in filename:
        return None, None

    if 'D1' in filename:
        tp = 'D1'
    elif 'D3' in filename:
        tp = 'D3'
    elif 'D7' in filename:
        tp = 'D7'
    elif 'Sham' in filename:
        tp = 'Sham'
    else:
        return None, None

    mouse = 'M1' if '_M1_' in filename else 'M2' if '_M2_' in filename else None
    return tp, mouse


def analyze_all_results():
    """Load all results and extract key metrics"""
    result_dir = Path('results/roi_results')
    result_files = sorted(result_dir.glob('roi_*.json.gz'))

    all_data = []

    for rf in result_files:
        timepoint, mouse = parse_roi_metadata(rf.name)
        if timepoint is None:
            continue

        with gzip.open(rf, 'rt') as f:
            result = json.load(f)

        # Analyze each scale
        for scale in ['10.0', '20.0', '40.0']:
            scale_data = result['multiscale_results'][scale]

            # Deserialize
            cluster_labels = deserialize_array(scale_data['cluster_labels'])
            markers_data = {m: deserialize_array(arr)
                           for m, arr in scale_data['transformed_arrays'].items()
                           if m not in ['130Ba', '131Xe']}  # Exclude DNA

            # Clustering metrics
            unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
            n_clusters = len(unique_clusters)
            cluster_balance = counts.std() / counts.mean()  # Lower = more balanced

            # Stability
            stability = scale_data['stability_analysis']
            optimal_res = stability['optimal_resolution']
            mean_clusters = stability['mean_n_clusters']

            # Spatial coherence
            coherence = scale_data['spatial_coherence']

            # Marker statistics (mean expression across superpixels)
            marker_means = {m: arr.mean() for m, arr in markers_data.items()}
            marker_stds = {m: arr.std() for m, arr in markers_data.items()}

            all_data.append({
                'roi': rf.stem.replace('roi_', '').replace('_results', ''),
                'timepoint': timepoint,
                'mouse': mouse,
                'scale': float(scale),
                'n_superpixels': len(cluster_labels),
                'n_clusters': n_clusters,
                'cluster_balance': cluster_balance,
                'optimal_resolution': optimal_res,
                'mean_n_clusters': mean_clusters,
                'spatial_coherence': coherence,
                **{f'{m}_mean': marker_means[m] for m in markers_data.keys()},
                **{f'{m}_std': marker_stds[m] for m in markers_data.keys()},
            })

    return pd.DataFrame(all_data)


if __name__ == '__main__':
    print("="*80)
    print("INTERROGATING ACTUAL RESULTS")
    print("="*80)
    print()

    df = analyze_all_results()

    print(f"Total observations: {len(df)} (18 ROIs × 3 scales)")
    print(f"Timepoints: {sorted(df['timepoint'].unique())}")
    print(f"Mice: {sorted(df['mouse'].unique())}")
    print(f"Scales: {sorted(df['scale'].unique())}")
    print()

    # Question 1: How many clusters emerge at each scale?
    print("="*80)
    print("Q1: HOW MANY CLUSTERS EMERGE AT EACH SCALE?")
    print("="*80)
    print()
    cluster_summary = df.groupby('scale')['n_clusters'].describe()
    print(cluster_summary)
    print()
    print("INTERPRETATION:")
    print(f"  - 10μm scale: {df[df['scale']==10]['n_clusters'].mean():.1f} ± {df[df['scale']==10]['n_clusters'].std():.1f} clusters")
    print(f"  - 20μm scale: {df[df['scale']==20]['n_clusters'].mean():.1f} ± {df[df['scale']==20]['n_clusters'].std():.1f} clusters")
    print(f"  - 40μm scale: {df[df['scale']==40]['n_clusters'].mean():.1f} ± {df[df['scale']==40]['n_clusters'].std():.1f} clusters")
    print()

    # Question 2: Is clustering stable across conditions?
    print("="*80)
    print("Q2: DOES CLUSTER COUNT VARY BY TIMEPOINT OR MOUSE?")
    print("="*80)
    print()
    # Focus on 10μm scale
    df_10 = df[df['scale'] == 10.0]
    print("10μm scale only:")
    print()
    tp_summary = df_10.groupby('timepoint')['n_clusters'].describe()[['mean', 'std', 'min', 'max']]
    print("By timepoint:")
    print(tp_summary)
    print()
    mouse_summary = df_10.groupby('mouse')['n_clusters'].describe()[['mean', 'std', 'min', 'max']]
    print("By mouse:")
    print(mouse_summary)
    print()
    print("INTERPRETATION:")
    print("  If cluster count is stable: Structure is consistent")
    print("  If cluster count varies by timepoint: Injury changes tissue organization")
    print()

    # Question 3: Are clusters spatially coherent?
    print("="*80)
    print("Q3: ARE CLUSTERS SPATIALLY COHERENT? (Moran's I)")
    print("="*80)
    print()
    coherence_summary = df.groupby('scale')['spatial_coherence'].describe()
    print(coherence_summary)
    print()
    print("INTERPRETATION:")
    print("  Moran's I > 0: Positive spatial autocorrelation (clusters form patches)")
    print("  Moran's I ≈ 0: Random spatial distribution")
    print("  Moran's I < 0: Negative autocorrelation (checkerboard)")
    print()
    print(f"  Average coherence: {df['spatial_coherence'].mean():.3f}")
    print(f"  → Clusters {'ARE' if df['spatial_coherence'].mean() > 0.1 else 'ARE NOT'} spatially coherent")
    print()

    # Question 4: What protein patterns define the data?
    print("="*80)
    print("Q4: WHICH PROTEINS SHOW THE MOST VARIATION?")
    print("="*80)
    print()
    df_10 = df[df['scale'] == 10.0]
    markers = ['CD45', 'CD11b', 'Ly6G', 'CD140a', 'CD140b', 'CD31', 'CD34', 'CD206', 'CD44']

    # Calculate coefficient of variation (std/mean) across all ROIs
    print("Protein variation (CV = std/mean across all ROIs at 10μm):")
    print()
    for marker in markers:
        mean_col = f'{marker}_mean'
        if mean_col in df_10.columns:
            cv = df_10[mean_col].std() / df_10[mean_col].mean()
            overall_mean = df_10[mean_col].mean()
            print(f"  {marker:10s}: CV={cv:.3f}, mean={overall_mean:.3f}")
    print()
    print("INTERPRETATION:")
    print("  High CV = protein varies a lot across ROIs (injury-responsive)")
    print("  Low CV = protein is stable (constitutive)")
    print()

    # Question 5: Temporal dynamics
    print("="*80)
    print("Q5: HOW DO MARKERS CHANGE OVER TIME?")
    print("="*80)
    print()
    print("CD45 (pan-immune) over time:")
    cd45_temporal = df_10.groupby('timepoint')['CD45_mean'].agg(['mean', 'std'])
    print(cd45_temporal)
    print()
    print("Ly6G (neutrophils) over time:")
    ly6g_temporal = df_10.groupby('timepoint')['Ly6G_mean'].agg(['mean', 'std'])
    print(ly6g_temporal)
    print()
    print("CD206 (M2 macrophages) over time:")
    cd206_temporal = df_10.groupby('timepoint')['CD206_mean'].agg(['mean', 'std'])
    print(cd206_temporal)
    print()
    print("INTERPRETATION:")
    print("  Expected: Ly6G spike at D1 (neutrophils)")
    print("  Expected: CD206 rise D3-D7 (M2 macrophages)")
    print("  Do we see this pattern?")
    print()

    # Save for further analysis
    df.to_csv('notebooks/biological_narratives/results_summary.csv', index=False)
    print("="*80)
    print("Saved to: notebooks/biological_narratives/results_summary.csv")
    print("="*80)
