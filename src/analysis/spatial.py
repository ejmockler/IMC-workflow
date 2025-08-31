#!/usr/bin/env python3
"""
Spatial Blob Analysis - Tissue Region Organization
Identifies coherent expression regions and their spatial relationships
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from collections import defaultdict

def load_roi_data(roi_file, config_path='config.json'):
    """Load ROI data with protein expression and coordinates"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    df = pd.read_csv(roi_file, sep='\t')
    
    # Get proteins from all functional groups (excluding DNA)
    selected_names = set()
    if 'functional_groups' in config['proteins']:
        for group_name, proteins in config['proteins']['functional_groups'].items():
            if group_name != 'structural_controls':
                selected_names.update(proteins)
    
    # Map to full column names with isotope tags
    available = []
    for protein_name in selected_names:
        for col in df.columns:
            if col.startswith(protein_name + '('):
                available.append(col)
                break
    
    coords = df[['X', 'Y']].values
    values = np.arcsinh(df[available].values / 5.0)
    protein_names = [col.split('(')[0] for col in available]
    
    return coords, values, protein_names

def identify_expression_blobs(coords, values, protein_names, n_blob_types=20, min_blob_size=5):
    """Identify coherent expression regions (blobs)"""
    
    # 1. Cluster pixels by expression signature to find blob types
    kmeans = KMeans(n_clusters=n_blob_types, random_state=42)
    blob_labels = kmeans.fit_predict(values)
    
    # 2. Get blob type characteristics
    blob_signatures = {}
    
    for blob_id in range(n_blob_types):
        mask = blob_labels == blob_id
        if np.sum(mask) >= min_blob_size:  # Include smaller blobs to show more types
            blob_signatures[blob_id] = {
                'mean_expression': values[mask].mean(axis=0),
                'dominant_proteins': [],
                'size': np.sum(mask),
                'coords': coords[mask]
            }
            
            # Find dominant proteins (top 2)
            mean_expr = values[mask].mean(axis=0)
            top_indices = np.argsort(mean_expr)[-2:][::-1]
            blob_signatures[blob_id]['dominant_proteins'] = [protein_names[i] for i in top_indices]
    
    # Merge blobs with identical protein signatures
    merged_blobs = {}
    blob_type_mapping = {}
    
    for blob_id, sig in blob_signatures.items():
        # Canonicalize by sorting the top-2 marker names so order doesn't matter
        canonical_pair = '+'.join(sorted(sig['dominant_proteins'][:2]))
        
        if canonical_pair not in merged_blobs:
            # First time seeing this protein combination
            merged_blobs[canonical_pair] = {
                'mean_expression': sig['mean_expression'].copy(),
                'dominant_proteins': sorted(sig['dominant_proteins'][:2]),
                'size': sig['size'],
                'coords': sig['coords']
            }
            blob_type_mapping[blob_id] = canonical_pair
        else:
            # Merge with existing blob of same type
            existing = merged_blobs[canonical_pair]
            total_size = existing['size'] + sig['size']
            
            # Weighted average of expression
            weight1 = existing['size'] / total_size
            weight2 = sig['size'] / total_size
            merged_blobs[canonical_pair]['mean_expression'] = (
                existing['mean_expression'] * weight1 + 
                sig['mean_expression'] * weight2
            )
            merged_blobs[canonical_pair]['size'] = total_size
            merged_blobs[canonical_pair]['coords'] = np.vstack([existing['coords'], sig['coords']])
            blob_type_mapping[blob_id] = canonical_pair
    
    print(f"      Found {len(blob_signatures)} spatial clusters → {len(merged_blobs)} distinct blob types")
    for blob_name, sig in merged_blobs.items():
        pct = (sig['size'] / len(coords)) * 100
        print(f"         {blob_name}: {pct:.1f}% of tissue")
    
    return blob_labels, merged_blobs, blob_type_mapping

def analyze_blob_spatial_relationships(coords, blob_labels, blob_signatures, blob_type_mapping):
    """Analyze which blob types are spatial neighbors - optimized"""
    
    # Build spatial tree
    tree = cKDTree(coords)
    
    # For each blob type, find its spatial neighbors
    blob_type_neighbors = defaultdict(lambda: defaultdict(int))
    blob_type_contacts = defaultdict(int)
    
    # Sample pixels for performance (every 10th pixel)
    sample_indices = np.arange(0, len(coords), 10)
    
    for sample_idx in sample_indices:
        current_cluster_id = blob_labels[sample_idx]
        current_blob_type = blob_type_mapping.get(current_cluster_id)
        
        if current_blob_type:
            # Check neighbors at 15μm
            neighbors = tree.query_ball_point(coords[sample_idx], 15.0)
            
            for neighbor_idx in neighbors[:20]:  # Limit neighbors checked
                if neighbor_idx != sample_idx:
                    neighbor_cluster_id = blob_labels[neighbor_idx]
                    neighbor_blob_type = blob_type_mapping.get(neighbor_cluster_id)
                    
                    if neighbor_blob_type and neighbor_blob_type != current_blob_type:
                        blob_type_neighbors[current_blob_type][neighbor_blob_type] += 1
                        blob_type_contacts[current_blob_type] += 1
    
    # Normalize to get contact frequencies
    blob_contact_freq = {}
    for blob_type in blob_signatures.keys():
        total_contacts = blob_type_contacts[blob_type]
        if total_contacts > 0:
            blob_contact_freq[blob_type] = {
                neighbor: count / total_contacts 
                for neighbor, count in blob_type_neighbors[blob_type].items()
            }
    
    return blob_contact_freq

