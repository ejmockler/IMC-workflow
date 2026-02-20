#!/usr/bin/env python
"""
Test spatial domain clustering on superpixel data
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

sns.set_style('whitegrid')

def deserialize_array(arr_dict):
    """Convert serialized numpy array back to numpy array"""
    if isinstance(arr_dict, dict) and '__numpy_array__' in arr_dict:
        return np.array(arr_dict['data'], dtype=arr_dict['dtype']).reshape(arr_dict['shape'])
    return arr_dict

def main():
    # Load superpixel data
    results_dir = Path('/Users/noot/Documents/IMC/results/roi_results')
    result_files = sorted(results_dir.glob('roi_*.json.gz'))

    print(f"Found {len(result_files)} ROI result files")

    all_superpixels = []
    markers = ['CD45', 'CD11b', 'Ly6G', 'CD140a', 'CD140b', 'CD31', 'CD34', 'CD206', 'CD44']

    for idx, result_file in enumerate(result_files):
        try:
            # Extract ROI info from filename
            roi_name = result_file.stem.replace('roi_', '').replace('_results.json', '')

            # Parse condition and timepoint
            if 'Sam' in roi_name:
                condition = 'Sham'
                timepoint = 'Sham'
                parts = roi_name.split('_ROI_')
                if len(parts) > 1:
                    mouse = parts[1].split('_')[0]
                else:
                    print(f"WARNING: Cannot parse Sham file: {roi_name}")
                    continue
            else:
                parts = roi_name.split('_ROI_')
                if len(parts) < 2:
                    print(f"WARNING: Cannot parse UUO file: {roi_name}")
                    continue
                timepoint_parts = parts[1].split('_')
                timepoint = timepoint_parts[0]  # D1, D3, D7
                mouse = timepoint_parts[1]  # M1, M2
                condition = 'UUO'

            # Load results
            with gzip.open(result_file, 'rt') as f:
                data = json.load(f)

            # Get superpixel data at 10um scale
            if 'multiscale_results' not in data:
                print(f"No multiscale_results in {roi_name}")
                continue

            scales = data['multiscale_results']
            scale_10 = None
            for scale_key in ['10.0', '10', 10.0, 10]:
                if str(scale_key) in scales:
                    scale_10 = scales[str(scale_key)]
                    break

            if scale_10 is None:
                print(f"No 10um scale data for {roi_name}")
                continue

            # Extract transformed marker values (arcsinh-transformed)
            transformed = scale_10.get('transformed_arrays', {})
            coords_dict = scale_10.get('superpixel_coords', {})

            # Deserialize coordinates
            coords = deserialize_array(coords_dict)
            if coords is None or len(coords) == 0:
                print(f"No coordinates for {roi_name}")
                continue

            n_superpixels = len(coords)

            # Build superpixel feature matrix
            for i in range(n_superpixels):
                spx_data = {
                    'roi': roi_name,
                    'condition': condition,
                    'timepoint': timepoint,
                    'mouse': mouse,
                    'superpixel_id': i,
                    'x': coords[i, 0],
                    'y': coords[i, 1]
                }

                # Add marker values
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

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(result_files)} files...")

        except Exception as e:
            print(f"ERROR processing {result_file.name}: {e}")
            continue

    # Convert to DataFrame
    superpixel_df = pd.DataFrame(all_superpixels)

    print(f"\nLoaded {len(superpixel_df)} superpixels across {superpixel_df['roi'].nunique()} ROIs")
    print(f"Timepoints: {sorted(superpixel_df['timepoint'].unique())}")
    print(f"\nSuperpixels per timepoint:")
    print(superpixel_df.groupby('timepoint').size())

    # Cluster superpixels
    feature_cols = markers
    valid_mask = superpixel_df[feature_cols].notna().all(axis=1)
    X = superpixel_df.loc[valid_mask, feature_cols].values

    print(f"\nValid superpixels for clustering: {X.shape[0]}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster with k=6
    n_domains = 6
    kmeans = KMeans(n_clusters=n_domains, random_state=42, n_init=20)
    domain_labels = kmeans.fit_predict(X_scaled)

    # Add labels
    superpixel_df.loc[valid_mask, 'domain'] = domain_labels
    superpixel_df['domain'] = superpixel_df['domain'].fillna(-1).astype(int)

    print(f"\nDomain sizes:")
    for d in range(n_domains):
        n = (superpixel_df['domain'] == d).sum()
        pct = 100 * n / len(superpixel_df)
        print(f"  Domain {d}: {n} superpixels ({pct:.1f}%)")

    # Domain profiles
    domain_profiles = superpixel_df[superpixel_df['domain'] >= 0].groupby('domain')[markers].mean()

    print("\nDomain marker profiles:")
    print(domain_profiles.round(2))

    # Temporal evolution
    timepoint_order = ['Sham', 'D1', 'D3', 'D7']
    domain_evolution = superpixel_df[superpixel_df['domain'] >= 0].groupby(['timepoint', 'domain']).size().unstack(fill_value=0)
    domain_evolution = domain_evolution.reindex(timepoint_order)
    domain_evolution_pct = domain_evolution.div(domain_evolution.sum(axis=1), axis=0) * 100

    print("\nDomain composition (%) by timepoint:")
    print(domain_evolution_pct.round(1))

    return superpixel_df, domain_profiles, domain_evolution_pct

if __name__ == '__main__':
    superpixel_df, domain_profiles, domain_evolution_pct = main()
