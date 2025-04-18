import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler # For scaling before clustermap

# Imports for Ordination
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
# For NMDS (alternative)
# from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram

def plot_ordination(
    composition_matrix: pd.DataFrame,
    metadata_df: pd.DataFrame, # Should be aligned with composition_matrix index
    config: Dict[str, Any],
    distance_metric: str,
    output_path: str,
    hue_col: Optional[str] = 'Condition', # Metadata column for color
    style_col: Optional[str] = 'Day',     # Metadata column for marker style
    ordination_method: str = 'pcoa',      # 'pcoa' or 'nmds'
    plot_title: Optional[str] = None,
    plot_dpi: int = 150,
    add_ellipse: bool = True
):
    """Generates and saves an ordination plot (PCoA or NMDS) based on community composition.

    Args:
        composition_matrix: Wide DataFrame (ROI x Community).
        metadata_df: Metadata DataFrame, indexed by ROI and aligned with composition_matrix.
        config: The configuration dictionary.
        distance_metric: Distance metric used for ordination (should match PERMANOVA).
        output_path: Full path to save the output plot file.
        hue_col: Metadata column name to use for coloring points.
        style_col: Metadata column name to use for marker style.
        ordination_method: 'pcoa' (Principal Coordinate Analysis) or 'nmds' (Non-metric MDS).
        plot_title: Optional title for the plot.
        plot_dpi: DPI for the saved plot.
        add_ellipse: Whether to draw confidence ellipses around groups defined by hue_col.
    """
    print(f"\n--- Generating Ordination Plot ({ordination_method.upper()}, Metric: {distance_metric}) ---")

    if composition_matrix.empty or metadata_df.empty:
        print("ERROR: Composition matrix or metadata is empty. Cannot generate plot.")
        return
    if not np.all(composition_matrix.index == metadata_df.index):
         print("ERROR: Composition matrix index does not match metadata index. Alignment required.")
         return

    # Check if hue/style columns exist
    plot_metadata_cols = []
    if hue_col and hue_col in metadata_df.columns:
        plot_metadata_cols.append(hue_col)
    elif hue_col:
        print(f"Warning: Hue column '{hue_col}' not found in metadata. Plot will not use hue.")
        hue_col = None

    if style_col and style_col in metadata_df.columns:
        plot_metadata_cols.append(style_col)
    elif style_col:
        print(f"Warning: Style column '{style_col}' not found in metadata. Plot will not use style.")
        style_col = None

    try:
        # 1. Calculate Distance Matrix
        print(f"   Calculating {distance_metric} distance matrix...")
        dist_array = pdist(composition_matrix.values, metric=distance_metric)
        dist_matrix = DistanceMatrix(squareform(dist_array), ids=composition_matrix.index)

        # 2. Perform Ordination
        print(f"   Performing {ordination_method.upper()}...")
        if ordination_method.lower() == 'pcoa':
            ordination_results = pcoa(dist_matrix)
            coords = ordination_results.samples[['PC1', 'PC2']] # Get first 2 axes
            proportion_explained = ordination_results.proportion_explained[:2]
            axis_labels = [f'PC1 ({proportion_explained[0]:.2%})', f'PC2 ({proportion_explained[1]:.2%})']
        # elif ordination_method.lower() == 'nmds':
            # Implement NMDS using sklearn.manifold.MDS if needed
            # mds = MDS(n_components=2, metric=False, max_iter=300, eps=1e-9, random_state=config.get('analysis',{}).get('seed', 42), dissimilarity="precomputed", n_jobs=1)
            # coords_array = mds.fit_transform(dist_matrix.data) # Need squareform distance matrix for MDS
            # coords = pd.DataFrame(coords_array, index=composition_matrix.index, columns=['NMDS1', 'NMDS2'])
            # axis_labels = ['NMDS1', 'NMDS2']
        else:
            print(f"ERROR: Unsupported ordination method '{ordination_method}'. Choose 'pcoa'.") # or 'nmds'
            return

        # 3. Combine coordinates with relevant metadata for plotting
        plot_df = pd.concat([coords, metadata_df[plot_metadata_cols]], axis=1)
        plot_df = plot_df.reset_index() # Make ROI index a column if needed

        # 4. Create Plot
        print("   Generating plot...")
        plt.style.use('seaborn-v0_8-whitegrid') # Or choose another style
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter_kws = {"s": 50, "alpha": 0.8}
        sns.scatterplot(
            data=plot_df,
            x=coords.columns[0],
            y=coords.columns[1],
            hue=hue_col,
            style=style_col,
            ax=ax,
            **scatter_kws
        )

        # Add ellipses if requested and hue is used
        if add_ellipse and hue_col:
            try:
                groups = plot_df[hue_col].unique()
                for group in groups:
                    group_data = plot_df[plot_df[hue_col] == group]
                    if len(group_data) >= 3: # Need enough points for ellipse
                         sns.kdeplot(
                             data=group_data, x=coords.columns[0], y=coords.columns[1],
                             levels=[0.68], # Approx 1 std dev
                             # color=sns.color_palette()[list(groups).index(group)], # Match color (might need adjustment)
                             alpha=0.2, ax=ax, legend=False
                         )
            except Exception as ellipse_err:
                 print(f"   Warning: Could not draw ellipses: {ellipse_err}")


        ax.set_xlabel(axis_labels[0], fontsize=12)
        ax.set_ylabel(axis_labels[1], fontsize=12)

        if not plot_title:
             plot_title = f'{ordination_method.upper()} on {distance_metric} Distance'
             if hue_col: plot_title += f' (Color: {hue_col})'
             if style_col: plot_title += f' (Style: {style_col})'
        ax.set_title(plot_title, fontsize=14, fontweight='bold')

        # Improve legend position
        if hue_col or style_col:
             ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend

        # 5. Save Plot
        print(f"   Saving plot to: {output_path}")
        plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close(fig) # Close figure to free memory

    except ImportError:
         print("ERROR: scikit-bio package not found. Please install it (`pip install scikit-bio`) for PCoA.")
         # Or check for sklearn if implementing NMDS
    except Exception as e:
        print(f"ERROR: Failed during ordination plotting: {e}")
        import traceback
        traceback.print_exc()
        # Ensure figure is closed if error occurred mid-plot
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)

def plot_profile_clustermap(
    profiles_with_meta_df: pd.DataFrame,
    channel_cols: List[str],
    config: Dict[str, Any], # Pass config for parameters
    output_path: str,
    plot_dpi: int = 150
):
    """Generates a clustermap of aggregated community profiles to help choose
       a distance threshold for hierarchical clustering.

    Args:
        profiles_with_meta_df: DataFrame containing aggregated profiles and metadata.
        channel_cols: List of column names corresponding to protein channels.
        config: The configuration dictionary (reads meta-clustering settings).
        output_path: Full path to save the output clustermap image.
        plot_dpi: DPI for the saved plot.
    """
    print("\n--- Generating Profile Clustermap for Threshold Selection ---")

    exp_config = config.get('experiment_analysis', {})
    metric = exp_config.get('metacluster_metric', 'euclidean')
    linkage_method = exp_config.get('metacluster_linkage', 'ward')
    scale_data = exp_config.get('metacluster_scale_profiles', True)

    if profiles_with_meta_df is None or profiles_with_meta_df.empty:
        print("ERROR: Input profiles DataFrame is empty or None.")
        return
    if not all(col in profiles_with_meta_df.columns for col in channel_cols):
        missing = [col for col in channel_cols if col not in profiles_with_meta_df.columns]
        print(f"ERROR: Input DataFrame missing required channel columns: {missing}")
        return

    # Ensure linkage method is compatible with metric
    if linkage_method == 'ward' and metric != 'euclidean':
        print(f"Warning: Ward linkage requires Euclidean metric for clustermap. Overriding metric to 'euclidean'.")
        metric = 'euclidean'

    try:
        # Extract profile data
        profile_data = profiles_with_meta_df[channel_cols].copy()

        # Clean data: check for NaNs/Infs which can break clustering/scaling
        if profile_data.isnull().values.any() or np.isinf(profile_data.values).any():
             print("   Warning: NaNs or Infs found in profile data. Filling with 0 before scaling/clustering.")
             profile_data = profile_data.fillna(0).replace([np.inf, -np.inf], 0)

        # Standardize data (rows=samples, columns=features)
        plot_data = profile_data
        if scale_data:
            print("   Scaling profile data using StandardScaler...")
            scaler = StandardScaler()
            # Fit transform columns (features)
            plot_data_scaled = scaler.fit_transform(profile_data)
            plot_data = pd.DataFrame(plot_data_scaled, index=profile_data.index, columns=profile_data.columns)

        print(f"   Generating clustermap (Metric: {metric}, Linkage: {linkage_method})...")
        # Use seaborn's clustermap
        # standard_scale=1 scales rows (profiles) after clustering - helps visualize patterns within profiles
        # z_score=1 scales columns (channels) after clustering - helps visualize relative expression across channels
        # Choose one or none based on desired visualization. standard_scale=1 is often useful here.
        g = sns.clustermap(
            plot_data,
            method=linkage_method,
            metric=metric,
            standard_scale=1, # Scale rows (profiles) to have mean 0, std dev 1 for visualization
            cmap="viridis",    # Choose a suitable colormap
            figsize=(10, 15) # Adjust figsize as needed
            # dendrogram_ratio=(.2, .2) # Adjust dendrogram sizes
        )

        # Improve plot appearance
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
        g.ax_heatmap.set_yticklabels([]) # Remove profile y-axis labels (too many)
        g.ax_heatmap.set_ylabel(f"{len(plot_data)} Community Profiles")
        g.fig.suptitle(f'Hierarchical Clustering of Community Profiles ({metric}/{linkage_method})', y=1.02)

        # Save the plot
        print(f"   Saving clustermap to: {output_path}")
        plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close(g.fig) # Close figure

    except Exception as e:
        print(f"ERROR: Failed during clustermap generation: {e}")
        import traceback
        traceback.print_exc()
        # Ensure figure is closed if error occurred mid-plot
        if 'g' in locals() and hasattr(g, 'fig') and plt.fignum_exists(g.fig.number):
             plt.close(g.fig)

def plot_dendrogram(
    linkage_matrix_Z: np.ndarray,
    output_path: str,
    config: Dict[str, Any],
    plot_dpi: int = 150,
    **kwargs # Pass extra arguments to dendrogram function
):
    """Plots and saves the dendrogram from hierarchical clustering.

    Args:
        linkage_matrix_Z: The linkage matrix output from scipy.cluster.hierarchy.linkage.
        output_path: Full path to save the output plot file.
        config: The configuration dictionary (for reading plot settings).
        plot_dpi: DPI for the saved plot.
        **kwargs: Additional keyword arguments passed to scipy.cluster.hierarchy.dendrogram.
                  Examples: truncate_mode='lastp', p=30, show_leaf_counts=True,
                            color_threshold=distance_threshold_from_config
    """
    print("\n--- Generating Dendrogram ---")
    if linkage_matrix_Z is None:
        print("ERROR: Linkage matrix is None. Cannot plot dendrogram.")
        return

    exp_config = config.get('experiment_analysis', {})
    distance_threshold = exp_config.get('metacluster_distance_threshold')

    try:
        fig, ax = plt.subplots(figsize=(15, 8)) # Adjust size as needed

        # Default dendrogram arguments (can be overridden by kwargs)
        dendro_args = {
            'color_threshold': distance_threshold if distance_threshold else 0, # Color up to threshold
            'above_threshold_color': 'grey',
            'leaf_rotation': 90,
            'leaf_font_size': 8
        }
        # Update with user-provided kwargs, allowing override
        dendro_args.update(kwargs)

        print(f"   Plotting dendrogram...")
        dn = dendrogram(linkage_matrix_Z, ax=ax, **dendro_args)

        # Add threshold line if provided
        if distance_threshold:
             ax.axhline(y=distance_threshold, color='r', linestyle='--', linewidth=0.8)
             ax.text(0.05, distance_threshold + plt.ylim()[1]*0.01, f'Threshold: {distance_threshold:.2f}', color='r', va='bottom', ha='left', fontsize=9)
             title = f'Hierarchical Clustering Dendrogram (Threshold: {distance_threshold:.2f})'
        else:
             title = 'Hierarchical Clustering Dendrogram'

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Distance (Cluster Dissimilarity)', fontsize=12)
        ax.set_xlabel('Community Profiles (Index/Cluster)', fontsize=12) # Label might change with truncation
        plt.tight_layout()

        print(f"   Saving dendrogram to: {output_path}")
        plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        print(f"ERROR: Failed during dendrogram generation: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)

def plot_heatmap_metacluster_profiles(
    clustered_profiles_df: pd.DataFrame, # DF containing profiles and 'meta_cluster' label
    channel_cols: List[str],
    config: Dict[str, Any],
    output_path: str,
    scale_heatmap: bool = True, # Whether to z-score channels for visualization
    plot_dpi: int = 150
):
    """Generates a heatmap summarizing the average profile of each meta-cluster.

    Args:
        clustered_profiles_df: DataFrame output from hierarchical clustering,
                               containing profiles and 'meta_cluster' column.
        channel_cols: List of column names corresponding to protein channels.
        config: The configuration dictionary.
        output_path: Full path to save the output heatmap image.
        scale_heatmap: If True, z-scores the average expression across channels for visualization.
        plot_dpi: DPI for the saved plot.
    """
    print("\n--- Generating Meta-Cluster Profile Heatmap ---")
    if clustered_profiles_df is None or clustered_profiles_df.empty:
        print("ERROR: Input clustered profiles DataFrame is empty or None.")
        return
    if 'meta_cluster' not in clustered_profiles_df.columns:
        print("ERROR: 'meta_cluster' column not found in input DataFrame.")
        return
    if not all(col in clustered_profiles_df.columns for col in channel_cols):
        missing = [col for col in channel_cols if col not in clustered_profiles_df.columns]
        print(f"ERROR: Input DataFrame missing required channel columns: {missing}")
        return

    try:
        # Calculate average profile per meta-cluster
        print("   Calculating average profile per meta-cluster...")
        metacluster_profiles = clustered_profiles_df.groupby('meta_cluster')[channel_cols].mean()
        n_clusters = len(metacluster_profiles)
        print(f"   Found {n_clusters} meta-clusters.")

        # Data for heatmap
        heatmap_data = metacluster_profiles

        # Optionally scale the data (z-score columns/channels) for visualization
        # This highlights relative expression patterns rather than absolute scaled values
        if scale_heatmap:
            print("   Z-scoring average profiles across channels for heatmap visualization.")
            scaler = StandardScaler()
            heatmap_data_scaled = scaler.fit_transform(heatmap_data)
            heatmap_data = pd.DataFrame(heatmap_data_scaled, index=heatmap_data.index, columns=heatmap_data.columns)
            cbar_label = "Z-Score Scaled Expression"
            plot_fmt = ".2f" # Format for annotations
        else:
             cbar_label = "Average Scaled Expression"
             plot_fmt = ".1f"

        # Create heatmap
        print("   Generating heatmap...")
        # Determine appropriate figsize based on number of clusters/channels
        height = max(6, n_clusters * 0.4)
        width = max(8, len(channel_cols) * 0.5)
        fig, ax = plt.subplots(figsize=(width, height))

        sns.heatmap(
            heatmap_data,
            annot=True,        # Add values to cells
            fmt=plot_fmt,      # Format for annotations
            cmap="viridis",    # Colormap
            linewidths=.5,
            linecolor='lightgray',
            cbar_kws={'label': cbar_label},
            annot_kws={"size": 8}, # Adjust annotation font size
            ax=ax
        )

        ax.set_title(f'Average Expression Profiles of {n_clusters} Meta-Clusters', fontsize=14, fontweight='bold')
        ax.set_xlabel('Protein Channels', fontsize=12)
        ax.set_ylabel('Meta-Cluster ID', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
        plt.tight_layout()

        # Save Plot
        print(f"   Saving meta-cluster heatmap to: {output_path}")
        plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        print(f"ERROR: Failed during meta-cluster heatmap generation: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
