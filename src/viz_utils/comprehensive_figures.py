"""
Comprehensive IMC Figure Generation - Using ALL Data

This module creates publication-quality figures that actually use all 25 ROIs
and properly quantify the biological claims about kidney injury progression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
try:
    import seaborn as sns
    sns.set_style('whitegrid')
except ImportError:
    sns = None
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional, Tuple, Any
import json
import glob
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5


class ComprehensiveFigureGenerator:
    """Generate figures using ALL ROI data with proper quantification."""
    
    def __init__(self, results_dir: str):
        """Initialize with results directory.
        
        Args:
            results_dir: Path to roi_results directory
        """
        self.results_dir = Path(results_dir)
        self.roi_data = {}
        self.cluster_phenotypes = {}
        self.protein_names = ['CD45', 'CD11b', 'Ly6G', 'CD140a', 'CD140b', 
                              'CD31', 'CD34', 'CD206', 'CD44']
        
        # Load all data
        self._load_all_roi_data()
        self._characterize_clusters()
    
    def _load_all_roi_data(self):
        """Load data from ALL 25 ROIs."""
        metadata_files = sorted(self.results_dir.glob("*_metadata.json"))
        
        for metadata_file in metadata_files:
            roi_id = metadata_file.stem.replace('_metadata', '')
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load arrays
            array_file = metadata_file.parent / f"{roi_id}_arrays.npz"
            if array_file.exists():
                arrays = np.load(array_file)
            else:
                arrays = None
            
            # Store both
            self.roi_data[roi_id] = {
                'metadata': metadata,
                'arrays': arrays,
                'condition': metadata['roi_metadata']['condition'],
                'timepoint': metadata['roi_metadata']['timepoint'],
                'region': metadata['roi_metadata']['region'],
                'replicate': metadata['roi_metadata'].get('replicate_id', 'Unknown')
            }
        
        logger.info(f"Loaded {len(self.roi_data)} ROIs")
        
        # Verify balanced design
        self._verify_experimental_design()
    
    def _verify_experimental_design(self):
        """Check the experimental design balance."""
        design = {}
        for roi_id, data in self.roi_data.items():
            key = f"{data['condition']}_D{data['timepoint']}_{data['region']}"
            if key not in design:
                design[key] = []
            design[key].append(roi_id)
        
        print("Experimental Design:")
        for key in sorted(design.keys()):
            print(f"  {key}: {len(design[key])} ROIs")
    
    def _characterize_clusters(self):
        """Determine biological identity of clusters using ALL ROIs."""
        # Collect cluster centroids from all ROIs
        all_centroids = {str(i): [] for i in range(8)}
        
        for roi_id, data in self.roi_data.items():
            if 'multiscale_metadata' in data['metadata']:
                # Use 20μm scale as standard
                scale_data = data['metadata']['multiscale_metadata'].get('scale_20.0', {})
                centroids = scale_data.get('cluster_centroids', {})
                
                for cluster_id, expression in centroids.items():
                    if cluster_id in all_centroids:
                        all_centroids[cluster_id].append(expression)
        
        # Calculate mean expression across all ROIs
        cluster_profiles = {}
        for cluster_id, expression_list in all_centroids.items():
            if expression_list:
                # Average across all ROIs
                mean_expression = {}
                for protein in self.protein_names:
                    values = [expr[protein] for expr in expression_list if protein in expr]
                    mean_expression[protein] = np.mean(values) if values else 0
                cluster_profiles[cluster_id] = mean_expression
        
        # Assign biological names based on marker patterns
        self._assign_cluster_names(cluster_profiles)
    
    def _assign_cluster_names(self, cluster_profiles: Dict):
        """Assign biological names to clusters based on expression patterns."""
        cluster_names = {}
        
        for cluster_id, profile in cluster_profiles.items():
            # Find dominant markers
            sorted_markers = sorted(profile.items(), key=lambda x: x[1], reverse=True)
            top_marker = sorted_markers[0][0]
            top_value = sorted_markers[0][1]
            
            # Biological interpretation
            if top_value < -0.5:
                # All markers low
                cluster_names[cluster_id] = "Background/Low"
            elif top_marker == 'CD45':
                if profile.get('CD11b', 0) > 0.5:
                    if profile.get('Ly6G', 0) > 0.5:
                        cluster_names[cluster_id] = "CD45+CD11b+Ly6G+ Neutrophils"
                    else:
                        cluster_names[cluster_id] = "CD45+CD11b+ Myeloid"
                else:
                    cluster_names[cluster_id] = "CD45+ Leukocytes"
            elif top_marker == 'CD31':
                cluster_names[cluster_id] = "CD31+ Endothelial"
            elif top_marker == 'CD140a' or top_marker == 'CD140b':
                cluster_names[cluster_id] = "CD140a/b+ Fibroblasts"
            elif top_marker == 'CD206':
                cluster_names[cluster_id] = "CD206+ M2 Macrophages"
            elif top_marker == 'CD44':
                cluster_names[cluster_id] = "CD44+ Activated/Stem"
            elif top_marker == 'CD34':
                cluster_names[cluster_id] = "CD34+ Progenitor"
            else:
                cluster_names[cluster_id] = f"{top_marker}+ Cells"
        
        self.cluster_phenotypes = cluster_names
        self.cluster_profiles = cluster_profiles
    
    def figure1_cluster_quantification(self) -> plt.Figure:
        """Create Figure 1: Cluster identity and quantification across ALL ROIs.
        
        Returns:
            Figure with comprehensive cluster analysis
        """
        fig = plt.figure(figsize=(18, 10))
        
        # Create a more organized grid layout
        gs = gridspec.GridSpec(2, 4, figure=fig, 
                              hspace=0.35, wspace=0.4,
                              width_ratios=[1, 1, 1, 0.8])
        
        # Top row
        # Panel A: Cluster phenotype heatmap (top left, wider)
        ax_heatmap = fig.add_subplot(gs[0, :2])
        self._plot_cluster_heatmap(ax_heatmap)
        
        # Panel B: Statistical tests (top right)
        ax_stats = fig.add_subplot(gs[0, 2:])
        self._plot_statistical_tests(ax_stats)
        
        # Bottom row
        # Panel C: Temporal abundance (bottom left, 2 cols)
        ax_abundance = fig.add_subplot(gs[1, :2])
        self._plot_cluster_abundance(ax_abundance)
        
        # Panel D: Regional distribution (bottom middle)
        ax_regional = fig.add_subplot(gs[1, 2])
        self._plot_regional_distribution(ax_regional)
        
        # Panel E: Representative spatial map (bottom right)
        ax_spatial = fig.add_subplot(gs[1, 3])
        self._plot_example_spatial(ax_spatial)
        
        plt.suptitle('Figure 1: Comprehensive Cluster Analysis Across All 25 ROIs', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        return fig
    
    def _plot_cluster_heatmap(self, ax):
        """Plot cluster phenotype heatmap with log scale."""
        # Create matrix for heatmap
        matrix = []
        cluster_labels = []
        
        for cluster_id in sorted(self.cluster_profiles.keys()):
            row = [self.cluster_profiles[cluster_id].get(protein, 0) 
                   for protein in self.protein_names]
            matrix.append(row)
            cluster_labels.append(f"C{cluster_id}: {self.cluster_phenotypes.get(cluster_id, 'Unknown')[:20]}")
        
        # Convert to numpy array and apply log transformation
        matrix = np.array(matrix)
        # Add small constant to avoid log(0)
        matrix_log = np.log10(matrix + 0.1)
        
        # Plot heatmap with better color intensity
        im = ax.imshow(matrix_log, aspect='auto', cmap='viridis', 
                      vmin=np.log10(0.1), vmax=np.log10(2))
        
        # Add gridlines for clarity
        for i in range(len(cluster_labels)):
            ax.axhline(i + 0.5, color='white', linewidth=0.5)
        for i in range(len(self.protein_names)):
            ax.axvline(i + 0.5, color='white', linewidth=0.5)
        
        # Add labels
        ax.set_xticks(range(len(self.protein_names)))
        ax.set_xticklabels(self.protein_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(cluster_labels)))
        ax.set_yticklabels(cluster_labels, fontsize=8)
        
        # Add colorbar with log scale labels
        cbar = plt.colorbar(im, ax=ax, label='Log10(Expression + 0.1)')
        # Set colorbar tick labels to show actual values
        cbar_ticks = [np.log10(0.1), np.log10(0.5), np.log10(1.0), np.log10(2.0)]
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(['0.1', '0.5', '1.0', '2.0'])
        
        ax.set_title('A. Cluster Phenotypes (Log Scale)', fontweight='bold')
        ax.set_xlabel('Protein Markers')
        ax.set_ylabel('Cluster Identity')
    
    def _plot_cluster_abundance(self, ax):
        """Plot cluster abundance across all conditions and timepoints."""
        # Calculate cluster frequencies for each ROI
        cluster_data = []
        
        for roi_id, data in self.roi_data.items():
            if data['arrays'] is not None:
                # Get cluster labels at 20μm scale
                cluster_labels = data['arrays']['scale_20.0_cluster_labels']
                
                # Calculate frequencies
                unique, counts = np.unique(cluster_labels, return_counts=True)
                total = len(cluster_labels)
                
                for cluster_id, count in zip(unique, counts):
                    cluster_data.append({
                        'ROI': roi_id,
                        'Condition': data['condition'],
                        'Timepoint': data['timepoint'],
                        'Region': data['region'],
                        'Cluster': str(cluster_id),
                        'Frequency': count / total * 100,
                        'ClusterName': self.cluster_phenotypes.get(str(cluster_id), f'Cluster {cluster_id}')
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(cluster_data)
        
        # Create grouped bar plot
        conditions_order = ['Sham', 'Injury', 'Test']
        timepoints = sorted(df['Timepoint'].unique())
        
        x = np.arange(len(timepoints))
        width = 0.8 / 8  # Width for 8 clusters
        
        for i, cluster_id in enumerate(sorted(df['Cluster'].unique())):
            cluster_df = df[df['Cluster'] == cluster_id]
            
            means = []
            sems = []
            
            for tp in timepoints:
                # Get data for this timepoint (across all conditions)
                tp_data = cluster_df[cluster_df['Timepoint'] == tp]['Frequency']
                means.append(tp_data.mean() if len(tp_data) > 0 else 0)
                sems.append(tp_data.sem() if len(tp_data) > 0 else 0)
            
            # Plot bars
            bars = ax.bar(x + i * width, means, width, yerr=sems,
                          label=self.cluster_phenotypes.get(cluster_id, f'C{cluster_id}')[:15],
                          capsize=2)
        
        ax.set_xlabel('Days Post-Injury')
        ax.set_ylabel('Cluster Frequency (%)')
        ax.set_title('C. Temporal Dynamics of Cluster Abundance', fontweight='bold')
        ax.set_xticks(x + width * 3.5)
        ax.set_xticklabels([f'D{tp}' for tp in timepoints])
        # Place legend inside the plot area
        ax.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_regional_distribution(self, ax):
        """Plot regional differences in cluster distribution - simplified."""
        # Prepare data
        cluster_data = []
        
        for roi_id, data in self.roi_data.items():
            if data['arrays'] is not None:
                cluster_labels = data['arrays']['scale_20.0_cluster_labels']
                unique, counts = np.unique(cluster_labels, return_counts=True)
                total = len(cluster_labels)
                
                for cluster_id, count in zip(unique, counts):
                    cluster_data.append({
                        'Region': data['region'],
                        'Cluster': str(cluster_id),
                        'Frequency': count / total * 100
                    })
        
        if cluster_data:
            df = pd.DataFrame(cluster_data)
            
            # Aggregate by region
            pivot = df.pivot_table(
                values='Frequency',
                index='Cluster', 
                columns='Region',
                aggfunc='mean'
            )
            
            # Simple bar plot
            pivot.plot(kind='bar', ax=ax, width=0.7, 
                      color=['skyblue', 'salmon'], alpha=0.8)
            
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Mean Frequency (%)')
            ax.set_title('D. Regional Distribution', fontweight='bold', fontsize=10)
            ax.legend(title='Region', loc='upper right', fontsize=8)
            ax.set_xticklabels([f'C{i}' for i in range(8)], rotation=0)
            ax.grid(axis='y', alpha=0.3)
    
    def _plot_statistical_tests(self, ax):
        """Plot results of statistical tests."""
        # Perform ANOVA for each cluster across timepoints
        test_results = []
        
        cluster_data = []
        for roi_id, data in self.roi_data.items():
            if data['arrays'] is not None and data['condition'] == 'Injury':
                cluster_labels = data['arrays']['scale_20.0_cluster_labels']
                unique, counts = np.unique(cluster_labels, return_counts=True)
                total = len(cluster_labels)
                
                for cluster_id, count in zip(unique, counts):
                    cluster_data.append({
                        'Timepoint': data['timepoint'],
                        'Cluster': str(cluster_id),
                        'Frequency': count / total * 100
                    })
        
        df = pd.DataFrame(cluster_data)
        
        # Run Kruskal-Wallis test for each cluster
        for cluster_id in sorted(df['Cluster'].unique()):
            cluster_df = df[df['Cluster'] == cluster_id]
            
            groups = []
            for tp in sorted(cluster_df['Timepoint'].unique()):
                groups.append(cluster_df[cluster_df['Timepoint'] == tp]['Frequency'].values)
            
            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                stat, pval = stats.kruskal(*groups)
                test_results.append({
                    'Cluster': self.cluster_phenotypes.get(cluster_id, f'C{cluster_id}'),
                    'p-value': pval,
                    'Significant': pval < 0.05
                })
        
        # Plot results
        results_df = pd.DataFrame(test_results)
        if not results_df.empty:
            results_df = results_df.sort_values('p-value')
            
            y_pos = range(len(results_df))
            colors = ['red' if sig else 'gray' for sig in results_df['Significant']]
            
            ax.barh(y_pos, -np.log10(results_df['p-value']), color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(results_df['Cluster'].str[:20], fontsize=7)
            ax.set_xlabel('-log10(p-value)')
            ax.axvline(x=-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
            ax.text(-np.log10(0.05) + 0.1, len(y_pos) - 0.5, 'p=0.05', fontsize=6)
            ax.set_title('B. Temporal Changes\n(Kruskal-Wallis)', fontweight='bold', fontsize=10)
    
    def _plot_example_spatial(self, ax):
        """Plot example spatial map."""
        # Select one representative ROI
        injury_d3_rois = [roi for roi, data in self.roi_data.items() 
                         if data['condition'] == 'Injury' and data['timepoint'] == 3]
        
        if injury_d3_rois:
            roi_id = injury_d3_rois[0]
            data = self.roi_data[roi_id]
            
            if data['arrays'] is not None:
                # Get spatial data
                labels = data['arrays']['scale_20.0_superpixel_labels']
                clusters = data['arrays']['scale_20.0_cluster_labels']
                
                # Create cluster map
                cluster_map = np.zeros_like(labels, dtype=float)
                for i in range(len(clusters)):
                    mask = labels == i
                    cluster_map[mask] = clusters[i]
                
                # Plot
                im = ax.imshow(cluster_map, cmap='tab10', interpolation='nearest')
                ax.set_title(f'E. Example: {roi_id[:20]}\nDay 3 Injury', fontweight='bold', fontsize=8)
                ax.axis('off')
                
                # Add scale bar (assuming 1μm per pixel)
                scalebar_length = 100  # 100 pixels = 100μm
                ax.plot([10, 10 + scalebar_length], [480, 480], 'white', linewidth=2)
                ax.text(10 + scalebar_length/2, 490, '100μm', color='white', 
                       ha='center', fontsize=7)
        else:
            ax.text(0.5, 0.5, 'No Day 3 Injury ROIs', ha='center', va='center')
            ax.axis('off')
    
    def figure2_spatial_analysis(self) -> plt.Figure:
        """Create Figure 2: Spatial neighborhood analysis.
        
        Returns:
            Figure with spatial metrics
        """
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, 
                              hspace=0.4, wspace=0.35,
                              height_ratios=[1.2, 1, 1])
        
        # Panel A: Spatial heterogeneity (full width top)
        ax_neighbor = fig.add_subplot(gs[0, :])
        self._plot_neighborhood_composition(ax_neighbor)
        
        # Panel B-D: Protein colocalization (middle row)
        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            self._plot_protein_colocalization(ax, i)
        
        # Panel E: Temporal clustering changes (bottom row, spans 2 columns)
        ax_clustering = fig.add_subplot(gs[2, :2])
        self._plot_clustering_metrics(ax_clustering)
        
        # Panel F: Regional differences (bottom right)
        ax_regional = fig.add_subplot(gs[2, 2])
        self._plot_regional_heterogeneity(ax_regional)
        
        plt.suptitle('Figure 2: Spatial Analysis of Cellular Organization', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        return fig
    
    def _plot_neighborhood_composition(self, ax):
        """Plot spatial heterogeneity metrics instead of meaningless cluster neighbors."""
        heterogeneity_data = []
        
        for roi_id, data in self.roi_data.items():
            if data['arrays'] is not None:
                # Get protein expression for each superpixel
                proteins_data = {}
                for protein in self.protein_names:
                    key = f'scale_20.0_transformed_arrays_{protein}'
                    if key in data['arrays']:
                        proteins_data[protein] = data['arrays'][key]
                
                if proteins_data:
                    # Get coordinates
                    coords = data['arrays']['scale_20.0_superpixel_coords']
                    
                    # Calculate spatial heterogeneity for each protein
                    if len(coords) > 10:
                        nbrs = NearestNeighbors(n_neighbors=min(6, len(coords)-1))
                        nbrs.fit(coords)
                        distances, indices = nbrs.kneighbors(coords)
                        
                        for protein, expression in proteins_data.items():
                            # Calculate local variance in expression
                            local_variances = []
                            for i in range(len(coords)):
                                neighbor_idx = indices[i]
                                neighbor_expr = expression[neighbor_idx]
                                local_var = np.var(neighbor_expr)
                                local_variances.append(local_var)
                            
                            # Global variance for comparison
                            global_var = np.var(expression)
                            
                            # Heterogeneity index: ratio of mean local to global variance
                            heterogeneity = np.mean(local_variances) / (global_var + 1e-10)
                            
                            heterogeneity_data.append({
                                'ROI': roi_id,
                                'Condition': data['condition'],
                                'Timepoint': data['timepoint'],
                                'Region': data['region'],
                                'Protein': protein,
                                'Heterogeneity': heterogeneity,
                                'GlobalVar': global_var,
                                'LocalVar': np.mean(local_variances)
                            })
        
        if heterogeneity_data:
            df = pd.DataFrame(heterogeneity_data)
            
            # Plot heterogeneity by condition and protein
            conditions = ['Sham', 'Injury']
            n_proteins = len(self.protein_names)
            
            # Create grouped bar plot
            x = np.arange(n_proteins)
            width = 0.35
            
            for i, condition in enumerate(conditions):
                cond_data = df[df['Condition'] == condition]
                means = []
                errors = []
                
                for protein in self.protein_names:
                    protein_data = cond_data[cond_data['Protein'] == protein]['Heterogeneity']
                    if len(protein_data) > 0:
                        means.append(protein_data.mean())
                        errors.append(protein_data.std() / np.sqrt(len(protein_data)))
                    else:
                        means.append(0)
                        errors.append(0)
                
                ax.bar(x + i*width - width/2, means, width, 
                      label=condition, yerr=errors, capsize=3,
                      alpha=0.8, edgecolor='black', linewidth=1)
            
            ax.set_xlabel('Protein')
            ax.set_ylabel('Spatial Heterogeneity Index')
            ax.set_title('A. Spatial Heterogeneity of Protein Expression', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.protein_names, rotation=45, ha='right')
            ax.legend(title='Condition')
            ax.grid(axis='y', alpha=0.3)
            
            # Add significance testing
            y_max = ax.get_ylim()[1]
            for j, protein in enumerate(self.protein_names):
                sham_data = df[(df['Condition'] == 'Sham') & (df['Protein'] == protein)]['Heterogeneity']
                injury_data = df[(df['Condition'] == 'Injury') & (df['Protein'] == protein)]['Heterogeneity']
                
                if len(sham_data) > 2 and len(injury_data) > 2:
                    _, p_value = stats.mannwhitneyu(sham_data, injury_data)
                    if p_value < 0.05:
                        ax.text(j, y_max * 0.95, '*', ha='center', fontsize=12)
                    if p_value < 0.01:
                        ax.text(j, y_max * 0.95, '**', ha='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for heterogeneity analysis',
                   ha='center', va='center')
    
    def _plot_clustering_metrics(self, ax):
        """Plot spatial clustering metrics."""
        metrics_data = []
        
        for roi_id, data in self.roi_data.items():
            if data['arrays'] is not None:
                coords = data['arrays']['scale_20.0_superpixel_coords']
                clusters = data['arrays']['scale_20.0_cluster_labels']
                
                # Calculate clustering coefficient (how clustered are cells of same type)
                for cluster_id in range(8):
                    cluster_mask = clusters == cluster_id
                    if np.sum(cluster_mask) > 3:
                        cluster_coords = coords[cluster_mask]
                        
                        # Calculate mean nearest neighbor distance
                        if len(cluster_coords) > 1:
                            nbrs = NearestNeighbors(n_neighbors=2)
                            nbrs.fit(cluster_coords)
                            distances, _ = nbrs.kneighbors(cluster_coords)
                            mean_nn_distance = distances[:, 1].mean()
                            
                            metrics_data.append({
                                'Condition': data['condition'],
                                'Timepoint': data['timepoint'],
                                'Cluster': self.cluster_phenotypes.get(str(cluster_id), f'C{cluster_id}'),
                                'MeanNNDistance': mean_nn_distance
                            })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            # Plot by condition and timepoint
            injury_df = df[df['Condition'] == 'Injury']
            
            if not injury_df.empty:
                pivot = injury_df.pivot_table(
                    values='MeanNNDistance',
                    index='Cluster',
                    columns='Timepoint',
                    aggfunc='mean'
                )
                
                # Plot with better colors for each timepoint
                colors = ['#66c2a5', '#fc8d62', '#8da0cb']  # Distinct colors
                pivot.plot(kind='bar', ax=ax, width=0.8, color=colors[:len(pivot.columns)])
                ax.set_ylabel('Mean NN Distance (pixels)')
                ax.set_xlabel('Cluster Phenotype')
                ax.set_title('E. Temporal Changes in Spatial Clustering', fontweight='bold')
                ax.legend(title='Day Post-Injury', loc='upper right', ncol=3, framealpha=0.9)
                
                # Rotate x labels for readability
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
    
    def _plot_regional_heterogeneity(self, ax):
        """Plot heterogeneity differences between Cortex and Medulla."""
        heterogeneity_data = []
        
        for roi_id, data in self.roi_data.items():
            if data['arrays'] is not None:
                # Get protein expression
                proteins_data = {}
                for protein in self.protein_names[:6]:  # Focus on key proteins
                    key = f'scale_20.0_transformed_arrays_{protein}'
                    if key in data['arrays']:
                        proteins_data[protein] = data['arrays'][key]
                
                if proteins_data:
                    for protein, expression in proteins_data.items():
                        heterogeneity_data.append({
                            'Region': data['region'],
                            'Protein': protein,
                            'CV': np.std(expression) / (np.mean(expression) + 1e-10)
                        })
        
        if heterogeneity_data:
            df = pd.DataFrame(heterogeneity_data)
            
            # Box plot comparing regions
            proteins = self.protein_names[:6]
            cortex_data = []
            medulla_data = []
            
            for protein in proteins:
                cortex_cv = df[(df['Region'] == 'Cortex') & (df['Protein'] == protein)]['CV']
                medulla_cv = df[(df['Region'] == 'Medulla') & (df['Protein'] == protein)]['CV']
                cortex_data.append(cortex_cv.values)
                medulla_data.append(medulla_cv.values)
            
            positions = np.arange(len(proteins))
            bp1 = ax.boxplot(cortex_data, positions=positions - 0.2, widths=0.35,
                             patch_artist=True, boxprops=dict(facecolor='skyblue', alpha=0.7))
            bp2 = ax.boxplot(medulla_data, positions=positions + 0.2, widths=0.35,
                             patch_artist=True, boxprops=dict(facecolor='salmon', alpha=0.7))
            
            ax.set_xticks(positions)
            ax.set_xticklabels(proteins, rotation=45, ha='right')
            ax.set_ylabel('Coefficient of Variation')
            ax.set_title('F. Regional Heterogeneity', fontweight='bold')
            ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Cortex', 'Medulla'])
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
    
    def _plot_protein_colocalization(self, ax, pair_idx: int):
        """Plot protein colocalization analysis."""
        # Define interesting protein pairs
        protein_pairs = [
            ('CD45', 'CD31', 'Immune-Vascular'),
            ('CD11b', 'CD206', 'M1-M2 Transition'),
            ('CD140a', 'CD44', 'Fibroblast-Activation')
        ]
        
        if pair_idx >= len(protein_pairs):
            ax.axis('off')
            return
        
        protein1, protein2, pair_name = protein_pairs[pair_idx]
        
        # Calculate colocalization across all ROIs
        coloc_data = []
        
        for roi_id, data in self.roi_data.items():
            if data['arrays'] is not None:
                # Get protein expression
                expr1 = data['arrays'][f'scale_20.0_transformed_arrays_{protein1}']
                expr2 = data['arrays'][f'scale_20.0_transformed_arrays_{protein2}']
                
                # Calculate correlation
                if len(expr1) > 10:
                    corr, pval = stats.spearmanr(expr1, expr2)
                    
                    coloc_data.append({
                        'Condition': data['condition'],
                        'Timepoint': data['timepoint'],
                        'Correlation': corr
                    })
        
        if coloc_data:
            df = pd.DataFrame(coloc_data)
            
            # Plot by timepoint
            injury_df = df[df['Condition'] == 'Injury']
            
            if not injury_df.empty:
                injury_df.boxplot(column='Correlation', by='Timepoint', ax=ax)
                ax.set_xlabel('Days Post-Injury')
                ax.set_ylabel('Spatial Correlation')
                panel_label = chr(66 + pair_idx)  # B, C, D
                ax.set_title(f'{panel_label}. {pair_name}\n{protein1}-{protein2}', 
                           fontweight='bold', fontsize=10)
                ax.get_figure().suptitle('')  # Remove automatic title
        else:
            ax.text(0.5, 0.5, f'{pair_name}\nNo data', ha='center', va='center')