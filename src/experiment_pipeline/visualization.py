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

# --- New Functions for Abundance Visualization ---

def plot_abundance_trends_lineplot(
    abundance_summary_df: pd.DataFrame,
    time_col: str,
    condition_col: str,
    grouping_col: str, # 'community' or 'meta_cluster'
    output_path: str,
    plot_dpi: int = 150,
    replicate_col: Optional[str] = None, # Needed if calculating mean/sem
    title: Optional[str] = None
):
    """Generates a line plot showing abundance trends over time, split by condition.

    Calculates mean proportion across replicates if replicate_col is provided.
    Otherwise plots individual data points.
    """
    print(f"\n--- Generating Abundance Trend Line Plot ({grouping_col}) ---")
    required_cols = ['proportion', time_col, condition_col, grouping_col]
    if replicate_col:
        required_cols.append(replicate_col)

    if abundance_summary_df is None or abundance_summary_df.empty:
        print("ERROR: Abundance summary DataFrame is empty or None.")
        return
    if not all(col in abundance_summary_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in abundance_summary_df.columns]
        print(f"ERROR: Abundance summary DataFrame missing required columns: {missing}")
        return

    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        plt.style.use('seaborn-v0_8-whitegrid')

        if replicate_col:
            # Calculate mean +/- SEM across replicates
            # Note: Seaborn lineplot calculates mean/bootstrap CI by default if data includes replicates
            print(f"   Plotting mean proportion (+/- 95% CI) across replicates ('{replicate_col}')")
            sns.lineplot(
                data=abundance_summary_df,
                x=time_col,
                y='proportion',
                hue=condition_col,
                style=grouping_col, # Use style to differentiate groups
                markers=True,
                estimator='mean', # Explicitly calculate mean
                errorbar=('ci', 95), # Show 95% confidence interval
                ax=ax
            )
            plot_type_desc = "Mean Proportion"
        else:
            # Plot individual points if no replicate info
            print("   Plotting individual proportions (no replicate column provided)")
            sns.lineplot(
                data=abundance_summary_df,
                x=time_col,
                y='proportion',
                hue=condition_col,
                style=grouping_col,
                markers=True,
                legend='full', # Show legend for style too
                ax=ax
            )
            plot_type_desc = "Proportion"

        if not title:
            title = f'{plot_type_desc} of {grouping_col.replace("_", " ").title()}s over {time_col.replace("_", " ").title()} by {condition_col.replace("_", " ").title()}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(time_col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(f'{grouping_col.replace("_", " ").title()} {plot_type_desc}', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        print(f"   Saving plot to: {output_path}")
        plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        print(f"ERROR: Failed during abundance line plot generation: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)

def plot_abundance_distribution_boxplot(
    abundance_summary_df: pd.DataFrame,
    time_col: str,
    condition_col: str,
    grouping_col: str, # 'community' or 'meta_cluster'
    output_path: str,
    plot_dpi: int = 150,
    # replicate_col: Optional[str] = None, # Not strictly needed for boxplot
    title: Optional[str] = None,
    show_points: bool = True
):
    """Generates box plots showing abundance distribution over time, split by condition.

    Facets by grouping_col (community/meta-cluster) if number of groups is reasonable.
    """
    print(f"\n--- Generating Abundance Distribution Box Plot ({grouping_col}) ---")
    required_cols = ['proportion', time_col, condition_col, grouping_col]

    if abundance_summary_df is None or abundance_summary_df.empty:
        print("ERROR: Abundance summary DataFrame is empty or None.")
        return
    if not all(col in abundance_summary_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in abundance_summary_df.columns]
        print(f"ERROR: Abundance summary DataFrame missing required columns: {missing}")
        return

    try:
        n_groups = abundance_summary_df[grouping_col].nunique()
        print(f"   Found {n_groups} unique {grouping_col}s.")

        # Decide on faceting based on number of groups
        if n_groups > 20: # Arbitrary threshold - too many facets become unreadable
             print("   Warning: Too many groups (>20) to facet effectively. Plotting combined distribution.")
             # Consider subsetting or alternative plot type if faceting is needed
             # For now, plot all together without faceting by group
             fig, ax = plt.subplots(figsize=(12, 7))
             sns.boxplot(
                 data=abundance_summary_df,
                 x=time_col,
                 y='proportion',
                 hue=condition_col,
                 showfliers=False, # Hide outliers for clarity
                 ax=ax
             )
             if show_points:
                 sns.stripplot(
                     data=abundance_summary_df,
                     x=time_col,
                     y='proportion',
                     hue=condition_col,
                     dodge=True,
                     size=3,
                     color='.3',
                     alpha=0.5,
                     legend=False,
                     ax=ax
                 )

             if not title:
                 title = f'Abundance Distribution over {time_col.replace("_"," ").title()} by {condition_col.replace("_"," ").title()} (All {grouping_col.replace("_"," ").title()}s)'
             ax.set_title(title, fontsize=14, fontweight='bold')
             ax.set_xlabel(time_col.replace("_", " ").title(), fontsize=12)
             ax.set_ylabel(f'{grouping_col.replace("_", " ").title()} Proportion', fontsize=12)
             ax.legend(title=condition_col.replace("_"," ").title(), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
             plt.tight_layout(rect=[0, 0, 0.85, 1])

        else:
             # Use FacetGrid for smaller number of groups
             print(f"   Creating faceted boxplot grid by '{grouping_col}'.")
             # Determine grid layout (aim for roughly square)
             n_cols = int(np.ceil(np.sqrt(n_groups)))
             # Adjust figsize based on number of columns/rows
             facet_height = 4
             facet_width = 5
             grid_height = np.ceil(n_groups / n_cols) * facet_height
             grid_width = n_cols * facet_width

             g = sns.FacetGrid(abundance_summary_df, col=grouping_col, col_wrap=n_cols,
                               height=facet_height, aspect=facet_width/facet_height, sharey=False, sharex=True)
             g.map_dataframe(sns.boxplot, x=time_col, y='proportion', hue=condition_col,
                             order=sorted(abundance_summary_df[time_col].unique()), # Ensure consistent time order
                             hue_order=sorted(abundance_summary_df[condition_col].unique()),
                             showfliers=False)
             if show_points:
                 # Map stripplot - use dodge based on hue manually
                 g.map_dataframe(sns.stripplot, x=time_col, y='proportion', hue=condition_col,
                                 order=sorted(abundance_summary_df[time_col].unique()),
                                 hue_order=sorted(abundance_summary_df[condition_col].unique()),
                                 dodge=True, size=3, color='.3', alpha=0.5)

             g.set_axis_labels(time_col.replace("_"," ").title(), "Proportion")
             g.set_titles(col_template="{col_name}") # Use grouping_col value as title
             g.add_legend(title=condition_col.replace("_"," ").title())
             if not title:
                 title = f'Abundance Distribution over {time_col.replace("_"," ").title()} by {condition_col.replace("_"," ").title()}'
             g.fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
             g.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout for suptitle
             fig = g.fig # Get the figure object for saving

        print(f"   Saving plot to: {output_path}")
        fig.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        print(f"ERROR: Failed during abundance box plot generation: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
