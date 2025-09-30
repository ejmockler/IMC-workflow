#!/usr/bin/env python
"""
Test script for comprehensive IMC figure generation.
"""

import sys
sys.path.append('.')

# Avoid complex imports that are broken
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import NearestNeighbors

# Direct implementation without broken imports
class SimpleComprehensiveFigures:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.roi_data = {}
        self.protein_names = ['CD45', 'CD11b', 'Ly6G', 'CD140a', 'CD140b', 
                              'CD31', 'CD34', 'CD206', 'CD44']
        self._load_all_data()
        
    def _load_all_data(self):
        """Load all ROI data."""
        metadata_files = list(self.results_dir.glob("*_metadata.json"))
        
        for metadata_file in metadata_files:
            roi_id = metadata_file.stem.replace('_metadata', '')
            
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            array_file = metadata_file.parent / f"{roi_id}_arrays.npz"
            arrays = np.load(array_file) if array_file.exists() else None
            
            self.roi_data[roi_id] = {
                'metadata': metadata,
                'arrays': arrays,
                'condition': metadata['roi_metadata']['condition'],
                'timepoint': metadata['roi_metadata']['timepoint'],
                'region': metadata['roi_metadata']['region']
            }
        
        print(f"Loaded {len(self.roi_data)} ROIs")
    
    def analyze_clusters(self):
        """Analyze cluster distributions."""
        print("\n" + "="*60)
        print("CLUSTER ANALYSIS ACROSS ALL ROIs")
        print("="*60)
        
        # Collect all cluster centroids
        all_centroids = {}
        
        for roi_id, data in self.roi_data.items():
            if 'multiscale_metadata' in data['metadata']:
                scale_data = data['metadata']['multiscale_metadata'].get('scale_20.0', {})
                centroids = scale_data.get('cluster_centroids', {})
                
                for cluster_id, expression in centroids.items():
                    if cluster_id not in all_centroids:
                        all_centroids[cluster_id] = []
                    all_centroids[cluster_id].append(expression)
        
        # Calculate mean profiles
        print("\nCluster Profiles (Mean Expression):")
        print("-"*60)
        
        for cluster_id in sorted(all_centroids.keys()):
            if all_centroids[cluster_id]:
                mean_expr = {}
                for protein in self.protein_names:
                    values = [expr[protein] for expr in all_centroids[cluster_id] if protein in expr]
                    mean_expr[protein] = np.mean(values) if values else 0
                
                # Find top marker
                top_marker = max(mean_expr.items(), key=lambda x: x[1])
                print(f"Cluster {cluster_id}: Top marker = {top_marker[0]} ({top_marker[1]:.2f})")
        
        # Analyze frequencies by condition
        print("\n" + "="*60)
        print("CLUSTER FREQUENCIES BY CONDITION")
        print("="*60)
        
        condition_clusters = {}
        
        for roi_id, data in self.roi_data.items():
            if data['arrays'] is not None:
                condition_key = f"{data['condition']}_D{data['timepoint']}"
                
                if condition_key not in condition_clusters:
                    condition_clusters[condition_key] = []
                
                clusters = data['arrays']['scale_20.0_cluster_labels']
                unique, counts = np.unique(clusters, return_counts=True)
                freqs = dict(zip(unique, counts / len(clusters) * 100))
                condition_clusters[condition_key].append(freqs)
        
        # Calculate means
        for condition in sorted(condition_clusters.keys()):
            print(f"\n{condition}:")
            
            all_freqs = {}
            for roi_freqs in condition_clusters[condition]:
                for cluster_id, freq in roi_freqs.items():
                    if cluster_id not in all_freqs:
                        all_freqs[cluster_id] = []
                    all_freqs[cluster_id].append(freq)
            
            for cluster_id in sorted(all_freqs.keys()):
                mean_freq = np.mean(all_freqs[cluster_id])
                sem_freq = np.std(all_freqs[cluster_id]) / np.sqrt(len(all_freqs[cluster_id]))
                print(f"  Cluster {cluster_id}: {mean_freq:5.1f} ± {sem_freq:3.1f}%")
        
        # Statistical tests
        print("\n" + "="*60)
        print("STATISTICAL TESTS (Injury progression)")
        print("="*60)
        
        # Test each cluster across injury timepoints
        injury_data = {}
        for roi_id, data in self.roi_data.items():
            if data['condition'] == 'Injury' and data['arrays'] is not None:
                tp = data['timepoint']
                if tp not in injury_data:
                    injury_data[tp] = []
                
                clusters = data['arrays']['scale_20.0_cluster_labels']
                unique, counts = np.unique(clusters, return_counts=True)
                freqs = dict(zip(unique, counts / len(clusters) * 100))
                injury_data[tp].append(freqs)
        
        # Run Kruskal-Wallis for each cluster
        for cluster_id in range(8):
            groups = []
            for tp in sorted(injury_data.keys()):
                tp_values = [roi_freqs.get(cluster_id, 0) for roi_freqs in injury_data[tp]]
                if tp_values:
                    groups.append(tp_values)
            
            if len(groups) >= 2:
                stat, pval = stats.kruskal(*groups)
                if pval < 0.05:
                    print(f"Cluster {cluster_id}: p = {pval:.4f} *")
                else:
                    print(f"Cluster {cluster_id}: p = {pval:.4f}")
        
        print("\n* = statistically significant (p < 0.05)")


if __name__ == "__main__":
    results_dir = 'results/cross_sectional_kidney_injury/roi_results'
    analyzer = SimpleComprehensiveFigures(results_dir)
    analyzer.analyze_clusters()