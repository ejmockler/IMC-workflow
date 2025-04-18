import yaml
import os
import sys
import time
import pandas as pd
import numpy as np # Import numpy
from typing import Dict, Optional, List # Added List

# Import the aggregation function and the new analysis function
from src.experiment_pipeline.data_aggregation import aggregate_and_merge_data, load_metadata
from src.experiment_pipeline.analysis_core import (
    calculate_community_abundance,
    prepare_composition_table,
    run_composition_permanova,
    cluster_profiles_hierarchical,
    analyze_metacluster_composition # Import the new function
)
from src.experiment_pipeline.visualization import (
    plot_profile_clustermap,
    plot_ordination,
    plot_dendrogram, # Added
    plot_heatmap_metacluster_profiles # Added
)

# Configuration Loading (Similar to run_roi_analysis.py)
def load_config(config_path: str = "config.yaml") -> Optional[Dict]:
    """Loads the pipeline configuration from a YAML file."""
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found at {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from: {config_path}")
        # Basic validation
        if not isinstance(config, dict):
            print(f"ERROR: Config file {config_path} is not a valid dictionary.")
            return None
        required_keys = ['paths', 'experiment_analysis', 'data']
        if not all(key in config for key in required_keys):
            missing = [key for key in required_keys if key not in config]
            print(f"ERROR: Config file is missing required top-level keys: {missing}")
            return None
        if not config.get('paths',{}).get('metadata_file'):
             print("ERROR: Config is missing paths -> metadata_file")
             return None
        print("Config basic validation passed.")
        return config
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse configuration file {config_path}: {e}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading configuration: {e}")
        return None

# --- Helper to identify channel columns ---
def get_channel_columns(df: pd.DataFrame, config: Dict) -> List[str]:
    """Identifies channel columns, prioritizing master list from config if available.

    Falls back to excluding known metadata/processing columns.
    """
    if df is None or df.empty:
        return []

    # Priority 1: Use master list from config if provided
    master_channels = config.get('data', {}).get('master_protein_channels')
    if master_channels and isinstance(master_channels, list):
        # Ensure the columns actually exist in the dataframe
        channel_cols = [col for col in master_channels if col in df.columns]
        if len(channel_cols) < len(master_channels):
            missing_in_df = [col for col in master_channels if col not in df.columns]
            print(f"   Warning: Some channels from config master list not found in DataFrame: {missing_in_df}")
        print(f"   Identified {len(channel_cols)} channel columns using master list from config.")
        return channel_cols
    elif master_channels:
         print("   Warning: 'master_protein_channels' in config is not a list. Ignoring it.")

    # Priority 2: Fallback to exclusion logic
    print("   Identifying channel columns using exclusion logic (master list not provided or invalid).")
    metadata_df_cols = list(config.get('metadata_cols', [])) # Metadata cols defined in root
    exp_config = config.get('experiment_analysis', {})
    # Columns explicitly defined in experiment_analysis section of config
    metadata_cols_from_exp_config = [
        exp_config.get('condition_col'), exp_config.get('timepoint_col'),
        exp_config.get('region_col'), exp_config.get('replicate_col'),
        exp_config.get('metadata_roi_col') # The actual ROI col name from metadata file
    ]
    # Columns potentially added during processing
    known_added_cols = ['roi_string', 'community', 'resolution', 'meta_cluster'] 

    # Combine all known non-channel columns
    non_channel_cols = set(known_added_cols + metadata_cols_from_exp_config + metadata_df_cols)
    non_channel_cols.discard(None) # Remove None if some config values are missing

    channel_cols = [col for col in df.columns if col not in non_channel_cols]

    # Basic check: If way too many columns are left, something might be wrong
    if len(channel_cols) > 100: # Arbitrary threshold
        print(f"   Warning: Identified {len(channel_cols)} potential channel columns via exclusion. This seems high. Check config metadata/channel definitions?")
    elif not channel_cols:
         print("   Error: No potential channel columns identified via exclusion.")
    else:
        print(f"   Identified {len(channel_cols)} potential channel columns using exclusion logic.")

    return channel_cols

# --- Script Execution Entry Point ---
if __name__ == '__main__':
    print("\n--- Starting IMC Experiment-Level Analysis ---")
    start_pipeline_time = time.time()

    # Load Configuration
    config = load_config("config.yaml")
    if config is None:
        print("Exiting due to configuration loading error.")
        sys.exit(1)

    # Setup Output Directory
    base_output_dir = config.get('paths', {}).get('output_dir')
    exp_config = config.get('experiment_analysis', {})
    exp_subdir = exp_config.get('output_subdir', 'experiment_summary')
    experiment_output_dir = os.path.join(base_output_dir, exp_subdir)
    try:
        os.makedirs(experiment_output_dir, exist_ok=True)
        print(f"\nExperiment output directory: {experiment_output_dir}")
    except Exception as e:
         print(f"ERROR creating experiment output directory {experiment_output_dir}: {e}")
         sys.exit(1)

    # Load Metadata
    metadata_df = load_metadata(config)
    if metadata_df is None:
        print("Exiting: Failed to load metadata.")
        sys.exit(1)

    # Aggregate Data from ROI pipeline
    agg_profiles_df, aggregated_pixels_df = aggregate_and_merge_data(config)

    # --- Initialize variables ---
    profiles_with_meta_df = None
    final_clustered_profiles_df = None
    abundance_summary_df = None
    permanova_results_df = None
    composition_matrix = None
    linkage_matrix_Z = None
    channel_cols = []
    grouping_col = 'community' # Default grouping
    metaclustering_done = False

    # --- 5. Prepare Profile Data (Merge with Metadata) ---
    if agg_profiles_df is not None and metadata_df is not None:
        print("\n--- Merging Aggregated Profiles with Metadata ---")
        roi_col_name = exp_config.get('metadata_roi_col')
        try:
            profiles_with_meta_df = pd.merge(
                agg_profiles_df, metadata_df,
                left_on='roi_string', right_on=roi_col_name, how='inner'
            )
            if profiles_with_meta_df.empty and not agg_profiles_df.empty:
                 print("Warning: Profile merge resulted in empty DataFrame. Check ROI identifiers.")
                 profiles_with_meta_df = None
            elif profiles_with_meta_df is not None:
                 print(f"Profile merge successful. Shape: {profiles_with_meta_df.shape}")
                 # Identify channel columns using the merged data
                 channel_cols = get_channel_columns(profiles_with_meta_df, config)
                 if not channel_cols:
                      print("ERROR: Could not identify channel columns in merged profiles.")
                      profiles_with_meta_df = None # Cannot proceed without channels

        except Exception as merge_err:
            print(f"ERROR merging profiles with metadata: {merge_err}")
            profiles_with_meta_df = None
    else:
         print("Skipping profile merge: Aggregated profiles or metadata missing.")


    # --- 6. Generate Profile Clustermap for Threshold Selection ---
    if profiles_with_meta_df is not None and channel_cols:
        print("\n--- Generating Profile Clustermap (for threshold selection) ---")
        clustermap_path = os.path.join(experiment_output_dir, "profile_clustermap_for_threshold.png")
        try:
            plot_profile_clustermap(
                profiles_with_meta_df=profiles_with_meta_df,
                channel_cols=channel_cols,
                config=config,
                output_path=clustermap_path,
                plot_dpi=config.get('processing', {}).get('plot_dpi', 150)
            )
            print(f"\n>>> Profile clustermap saved to: {clustermap_path} <<<")
            print(">>> Please examine this clustermap to set 'metacluster_distance_threshold' in config.yaml <<<")
        except Exception as e:
            print(f"ERROR generating profile clustermap: {e}")
    else:
        print("\nSkipping meta-clustering (disabled in config, distance_threshold invalid, or profile data unavailable). Using original 'community' labels.")
        grouping_col = 'community'
        metaclustering_done = False


    # --- 7. Perform Hierarchical Meta-Clustering (Conditional) ---
    if exp_config.get('perform_metaclustering', False) and profiles_with_meta_df is not None and channel_cols:
        print("\n--- Attempting Hierarchical Meta-Clustering ---")
        distance_threshold = exp_config.get('metacluster_distance_threshold')
        if distance_threshold is None or (isinstance(distance_threshold, (int, float)) and distance_threshold <= 0):
             print("\n*** WARNING: 'metacluster_distance_threshold' is not set or invalid in config.yaml. ***")
             print("*** Skipping meta-clustering. Analysis will proceed using original 'community' labels. ***")
             print("*** Please examine the clustermap PNG and set a valid threshold to enable meta-clustering. ***")
             grouping_col = 'community'
             metaclustering_done = False
        else:
             print(f"Using distance threshold: {distance_threshold}")
             final_clustered_profiles_df, linkage_matrix_Z = cluster_profiles_hierarchical(
                 profiles_with_meta_df=profiles_with_meta_df,
                 channel_cols=channel_cols,
                 distance_threshold=distance_threshold,
                 # n_clusters_fallback=exp_config.get('n_metaclusters'), # Add if using fallback
                 metric=exp_config.get('metacluster_metric', 'euclidean'),
                 linkage_method=exp_config.get('metacluster_linkage', 'ward'),
                 scale_data=exp_config.get('metacluster_scale_profiles', True)
             )
             if final_clustered_profiles_df is not None:
                 print("Meta-clustering successful.")
                 grouping_col = 'meta_cluster' # Switch grouping column
                 metaclustering_done = True
                 # Save clustered profiles if needed
                 # clustered_profiles_path = os.path.join(experiment_output_dir, "aggregated_profiles_with_metaclusters.csv")
                 # final_clustered_profiles_df.to_csv(clustered_profiles_path, index=False)
             else:
                 print("ERROR: Meta-clustering failed. Using original 'community' labels.")
                 grouping_col = 'community'
                 metaclustering_done = False
    else:
        print("\nSkipping meta-clustering (disabled in config or profile data unavailable). Using original 'community' labels.")
        grouping_col = 'community'
        metaclustering_done = False


    # --- 8. Analyze Meta-Cluster Composition (Added Section) ---
    metacluster_composition_df = None
    if metaclustering_done and final_clustered_profiles_df is not None:
        print("\n--- Analyzing Meta-Cluster Composition ---")
        metadata_cols_to_analyze = exp_config.get('metadata_cols_for_composition', [])
        if metadata_cols_to_analyze:
            try:
                metacluster_composition_df = analyze_metacluster_composition(
                    profiles_with_meta_df=final_clustered_profiles_df, # Use the df with metacluster labels
                    metacluster_col=grouping_col, # Should be 'meta_cluster' if metaclustering_done
                    metadata_cols_to_analyze=metadata_cols_to_analyze,
                    config=config
                )
                if metacluster_composition_df is not None:
                    composition_output_path = os.path.join(experiment_output_dir, "metacluster_composition_summary.csv")
                    metacluster_composition_df.to_csv(composition_output_path)
                    print(f"Meta-cluster composition summary saved to: {composition_output_path}")
                else:
                    print("   Meta-cluster composition analysis returned no results.")
            except Exception as e:
                print(f"ERROR during meta-cluster composition analysis: {e}")
        else:
            print("   Skipping meta-cluster composition analysis: 'metadata_cols_for_composition' not defined in config.")
    else:
        print("\nSkipping meta-cluster composition analysis: Meta-clustering was not performed or failed.")


    # --- 9. Visualize Clustering Results (Dendrogram, Heatmap) ---
    if metaclustering_done and linkage_matrix_Z is not None and final_clustered_profiles_df is not None:
        print("\n--- Visualizing Meta-Clustering Results ---")
        dendro_path = os.path.join(experiment_output_dir, "metacluster_dendrogram.png")
        try:
            # Plot dendrogram with threshold line
            plot_dendrogram(
                linkage_matrix_Z=linkage_matrix_Z,
                output_path=dendro_path,
                config=config, # Pass config to potentially get threshold
                plot_dpi=config.get('processing', {}).get('plot_dpi', 150),
                truncate_mode='lastp', # Example: Show only last 30 merges
                p=30, show_leaf_counts=True
            )
            print(f"Dendrogram saved to: {dendro_path}")
        except Exception as e:
            print(f"ERROR generating dendrogram plot: {e}")

        heatmap_path = os.path.join(experiment_output_dir, "metacluster_profile_heatmap.png")
        try:
            # Plot meta-cluster heatmap
            plot_heatmap_metacluster_profiles(
                clustered_profiles_df=final_clustered_profiles_df, # Use the df with meta_cluster labels
                channel_cols=channel_cols,
                config=config,
                output_path=heatmap_path,
                scale_heatmap=True, # Z-score heatmap for relative patterns
                plot_dpi=config.get('processing', {}).get('plot_dpi', 150)
            )
            print(f"Meta-cluster heatmap saved to: {heatmap_path}")
        except Exception as e:
            print(f"ERROR generating meta-cluster heatmap: {e}")


    # --- 10. Prepare Pixel Data for Abundance ---
    print(f"\n--- Preparing Pixel Data for Abundance (using '{grouping_col}') ---")
    abundance_input_df = None
    if aggregated_pixels_df is not None:
        if metaclustering_done and final_clustered_profiles_df is not None:
            # Map meta-cluster labels back to the aggregated pixel data based on (roi_string, community)
            print("   Mapping meta-cluster labels to pixel data...")
            try:
                 mc_mapping = final_clustered_profiles_df.set_index(['roi_string', 'community'])['meta_cluster'].to_dict()
                 pixel_indices = pd.MultiIndex.from_frame(aggregated_pixels_df[['roi_string', 'community']])
                 aggregated_pixels_df[grouping_col] = pixel_indices.map(mc_mapping)

                 unmapped_pixels = aggregated_pixels_df[grouping_col].isnull().sum()
                 if unmapped_pixels > 0:
                      print(f"   Warning: {unmapped_pixels} pixels could not be mapped to a meta-cluster (likely from ROIs missing in profile merge). Excluding them.")
                 abundance_input_df = aggregated_pixels_df.dropna(subset=[grouping_col])
                 abundance_input_df[grouping_col] = abundance_input_df[grouping_col].astype(int) # Ensure integer type
            except Exception as map_err:
                 print(f"ERROR mapping meta-clusters to pixels: {map_err}. Falling back to original communities.")
                 grouping_col = 'community' # Revert grouping col on error
                 abundance_input_df = aggregated_pixels_df # Use original pixel data
        else:
             # Use original communities if no meta-clustering done/failed
             grouping_col = 'community'
             abundance_input_df = aggregated_pixels_df
             print("   Using original 'community' labels.")
    else:
         print("ERROR: No aggregated pixel data loaded. Cannot calculate abundance.")


    # --- 11. Calculate Abundance per ROI ---
    if abundance_input_df is not None and metadata_df is not None:
        print("\n--- Calculating Group Abundance per ROI ---")
        abundance_value_col = exp_config.get('abundance_value_type', 'proportion') # 'proportion' or 'pixel_count'
        abundance_summary_df = calculate_community_abundance(
            aggregated_pixels_df=abundance_input_df,
            metadata_df=metadata_df,
            config=config,
            grouping_col=grouping_col # Pass the determined grouping column
        )
        if abundance_summary_df is not None and not abundance_summary_df.empty:
            print(f"\nSuccessfully calculated abundance using '{grouping_col}'.")
            # Save the abundance data
            try:
                abundance_filename = f"{grouping_col}_abundance_summary.csv"
                abundance_path = os.path.join(experiment_output_dir, abundance_filename)
                abundance_summary_df.to_csv(abundance_path, index=False)
                print(f"Saved abundance summary data to: {abundance_path}")
            except Exception as e:
                print(f"ERROR: Failed to save abundance summary data: {e}")
        else:
            print("\nERROR: Failed to calculate abundance summary.")
            abundance_summary_df = None # Ensure it's None if failed
    else:
        print("\nSkipping abundance calculation: Input data missing.")


    # --- 12. Prepare Composition Table for PERMANOVA/Ordination ---
    if abundance_summary_df is not None:
        print("\n--- Preparing Composition Table for Downstream Analysis ---")
        composition_value_col = exp_config.get('composition_value_type', 'proportion')
        composition_matrix = prepare_composition_table(
            abundance_summary_df,
            value_col=composition_value_col,
            grouping_col=grouping_col
        )
        if composition_matrix is None:
             print("ERROR: Failed to prepare composition matrix.")
    else:
         print("Skipping composition matrix preparation: Abundance data not available.")


    # --- 13. Run PERMANOVA ---
    if composition_matrix is not None and metadata_df is not None:
        permanova_formula = exp_config.get('permanova_formula')
        if not permanova_formula:
             print("Warning: 'permanova_formula' not set in config. Skipping PERMANOVA.")
        else:
             print(f"\n--- Running PERMANOVA (using '{grouping_col}' composition) ---")
             permanova_results_df = run_composition_permanova(
                 composition_matrix=composition_matrix,
                 metadata_df=metadata_df,
                 config=config, # Passes ROI col, replicate col (implicitly 'Mouse')
                 formula=permanova_formula,
                 distance_metric='braycurtis',
                 permutations=999
             )
             if permanova_results_df is not None and not permanova_results_df.empty:
                  print("\n--- PERMANOVA Results ---")
                  print(permanova_results_df)
                  # Save PERMANOVA results
                  try:
                      permanova_filename = f"permanova_{grouping_col}_composition_results.csv"
                      permanova_path = os.path.join(experiment_output_dir, permanova_filename)
                      permanova_results_df.to_csv(permanova_path, index=False)
                      print(f"\nSaved PERMANOVA results to: {permanova_path}")
                  except Exception as e:
                      print(f"ERROR: Failed to save PERMANOVA results: {e}")
             else:
                  print("\nERROR: PERMANOVA analysis failed or produced no results.")
                  permanova_results_df = None # Ensure None if failed
    else:
         print("\nSkipping PERMANOVA: Composition matrix or metadata not available.")


    # --- 14. Visualize Ordination ---
    if composition_matrix is not None and metadata_df is not None:
         print("\n--- Generating Ordination Plot ---")
         # Align metadata with the composition matrix index for plotting
         roi_col_name = exp_config.get('metadata_roi_col')
         aligned_metadata_for_plot = None
         try:
             aligned_metadata_for_plot = metadata_df.set_index(roi_col_name).reindex(composition_matrix.index)
             # Define essential plot columns from config for checking NA
             hue_col_plot = exp_config.get('plot_hue_col', 'Condition')
             style_col_plot = exp_config.get('plot_style_col', 'Day')
             essential_plot_cols = [col for col in [hue_col_plot, style_col_plot] if col] # Get non-None cols
             valid_plot_rois = aligned_metadata_for_plot.dropna(subset=essential_plot_cols).index
             if len(valid_plot_rois) < len(aligned_metadata_for_plot):
                   print(f"   Warning: Dropping {len(aligned_metadata_for_plot) - len(valid_plot_rois)} ROIs with missing plotting metadata ({essential_plot_cols}).")
             aligned_metadata_for_plot = aligned_metadata_for_plot.loc[valid_plot_rois]
             composition_matrix_for_plot = composition_matrix.loc[valid_plot_rois]
         except KeyError:
             print(f"ERROR aligning metadata for plot: ROI column '{roi_col_name}' not found.")
             composition_matrix_for_plot = None # Prevent plotting

         if composition_matrix_for_plot is not None and not composition_matrix_for_plot.empty:
             try:
                 distance_metric_plot = 'braycurtis' # Should match PERMANOVA
                 plot_filename = f"pcoa_{distance_metric_plot}_{grouping_col}_composition.png"
                 ordination_plot_path = os.path.join(experiment_output_dir, plot_filename)

                 plot_ordination(
                     composition_matrix=composition_matrix_for_plot,
                     metadata_df=aligned_metadata_for_plot,
                     config=config,
                     distance_metric=distance_metric_plot,
                     output_path=ordination_plot_path,
                     hue_col=hue_col_plot,
                     style_col=style_col_plot,
                     ordination_method='pcoa',
                     add_ellipse=True,
                     plot_dpi=config.get('processing', {}).get('plot_dpi', 150)
                 )
                 print(f"Ordination plot saved to: {ordination_plot_path}")
             except Exception as plot_err:
                 print(f"ERROR: Failed to generate or save ordination plot: {plot_err}")
         else:
              print("Skipping ordination plot: No valid ROIs remaining after aligning metadata or composition matrix unavailable.")


    # --- Finish ---
    total_pipeline_time = time.time() - start_pipeline_time
    print(f"\nTotal experiment analysis execution time: {total_pipeline_time:.2f} seconds.")
    print(f"Results saved in: {experiment_output_dir}")
    print("\n================ Completed Experiment Analysis Script ================")
