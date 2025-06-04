import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
import time
import traceback
import json
import sys
from typing import List, Tuple, Optional, Dict, Any
import gc # For memory management
import multiprocessing # Added for parallel processing
from functools import partial # Added for use with multiprocessing.Pool

# Import plotting functions from the existing visualization module
from src.roi_pipeline.pixel_visualization import (
    plot_raw_vs_scaled_spatial_comparison,
    plot_correlation_clustermap,
    plot_umap_scatter,
    plot_coexpression_matrix,
    plot_community_size_distribution,
    analyze_dendrogram_by_controls,
    plot_spatial_community_assignment_map,
    plot_spatial_community_channel_maps
)

# --- Helper function for Cofactor Subtitle ---
def get_cofactor_subtitle_string(cofactors: Dict[str, float]) -> str:
    """Generates a string representation of cofactors for plot subtitles."""
    if not cofactors:
        return "N/A"
    
    unique_cofactor_values = set(cofactors.values())
    if len(unique_cofactor_values) == 1:
        val = list(unique_cofactor_values)[0]
        if isinstance(val, float) and val == int(val):
            return f"{int(val)}"
        return f"{val:.2f}"
    elif unique_cofactor_values: # Non-empty set with multiple values
        return "various"
    else:
        return "N/A"

# --- Configuration Loading (Copied/adapted from run_roi_analysis.py) ---
def load_config(config_path: str = "config.yaml") -> Optional[Dict]:
    """Loads the pipeline configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from: {config_path}")
        if not isinstance(config, dict):
            print(f"ERROR: Configuration file {config_path} is not a valid dictionary.")
            return None
        # Add more specific key checks as needed for visualization
        if 'paths' not in config or 'output_dir' not in config['paths']:
            print("ERROR: Config missing 'paths.output_dir'.")
            return None
        if 'processing' not in config or 'plot_dpi' not in config['processing'] or 'visualization' not in config['processing']:
             print("ERROR: Config missing 'processing.plot_dpi' or 'processing.visualization' sections.")
             return None
        return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse configuration file {config_path}: {e}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading configuration: {e}")
        return None

# --- Data Loading Helper Functions ---

def load_required_roi_data(roi_dir: str, roi_string: str, config: Dict) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[Dict[str, float]], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[np.ndarray], Optional[List[str]], Optional[pd.DataFrame]]:
    """Loads the essential data files generated at the ROI level and the original raw data.
    
    Returns includes:
        - scaled_pixel_df (with X,Y and scaled expression)
        - final_ordered_channels (pixel_channel_final_order.json)
        - cofactors
        - coords_df (extracted X,Y)
        - original_raw_pixel_data_df
        - pixel_linkage_matrix (pixel_channel_linkage.npy)
        - pixel_linkage_channel_order (pixel_channel_clustered_order.json)
        - precomputed_pixel_correlation_matrix (pixel_channel_correlation.csv)
    """
    print(f"  Loading ROI-level data for {roi_string}...")
    scaled_pixel_path = os.path.join(roi_dir, f"scaled_pixel_expression_{roi_string}.csv")
    # This is the order list intended for general downstream use by run_roi_analysis.py
    final_ordered_channels_path = os.path.join(roi_dir, f"pixel_channel_final_order_{roi_string}.json")
    cofactors_path = os.path.join(roi_dir, f"asinh_cofactors_{roi_string}.json")

    # Files related to the ROI-specific pixel correlation clustering
    pixel_linkage_matrix_path = os.path.join(roi_dir, f"pixel_channel_linkage_{roi_string}.npy")
    pixel_linkage_channel_order_path = os.path.join(roi_dir, f"pixel_channel_clustered_order_{roi_string}.json")
    precomputed_pixel_correlation_path = os.path.join(roi_dir, f"pixel_channel_correlation_{roi_string}.csv")
    
    original_raw_pixel_data_df = None
    raw_data_dir = config['paths']['data_dir']
    found_raw_file = False

    try:
        for filename in os.listdir(raw_data_dir):
            if filename.endswith(".txt"):
                base_name = os.path.splitext(filename)[0]
                # Regex to extract ROI string like in run_roi_analysis.py
                match = re.search(r'(?:^|_)(ROI_[A-Za-z0-9_]+)', base_name)
                if match:
                    extracted_roi = match.group(1)
                    if extracted_roi == roi_string:
                        original_raw_file_path = os.path.join(raw_data_dir, filename)
                        print(f"    Found matching original raw data file: {filename}")
                        try:
                            original_raw_pixel_data_df = pd.read_csv(original_raw_file_path, sep='\t', engine='python')
                            print(f"    Loaded original raw pixel data from: {filename}")
                            if 'X' not in original_raw_pixel_data_df.columns or 'Y' not in original_raw_pixel_data_df.columns:
                                print(f"    ERROR: 'X' or 'Y' columns missing in {filename}. Cannot use for raw plot.")
                                original_raw_pixel_data_df = None
                            found_raw_file = True
                            break # Found the file, no need to check further
                        except Exception as e_raw:
                            print(f"    ERROR loading original raw data file {filename}: {e_raw}")
                            original_raw_pixel_data_df = None
                            found_raw_file = True # Mark as found to avoid the "not found" message below for this attempt
                            break # Stop on error with this file
    except FileNotFoundError:
        print(f"    ERROR: Raw data directory not found: {raw_data_dir}. Cannot search for original raw files.")
    except Exception as e_list:
        print(f"    ERROR listing files in {raw_data_dir}: {e_list}")

    if not found_raw_file:
        print(f"    WARNING: Original raw data file for ROI '{roi_string}' not found in {raw_data_dir} after checking all .txt files. Raw comparison plot will be affected.")
        # original_raw_pixel_data_df remains None

    coords_df = None 
    scaled_pixel_df = None
    final_ordered_channels = None # Renamed from ordered_channels
    cofactors = None
    pixel_linkage_matrix = None
    pixel_linkage_channel_order = None
    precomputed_pixel_correlation_matrix = None

    try:
        if os.path.exists(scaled_pixel_path):
            scaled_pixel_df = pd.read_csv(scaled_pixel_path, index_col=0) # Assuming index is pixel ID
            # Extract coords (X, Y) and pixel expression separately
            if 'X' in scaled_pixel_df.columns and 'Y' in scaled_pixel_df.columns:
                 coords_df = scaled_pixel_df[['X', 'Y']].copy()
                 # scaled_pixel_df = scaled_pixel_df.drop(columns=['X', 'Y']) # Keep only expression for correlations etc.
                 print(f"    Loaded scaled pixel data (with X,Y) from: {os.path.basename(scaled_pixel_path)}")
            else:
                 print(f"    ERROR: 'X' or 'Y' columns missing in {os.path.basename(scaled_pixel_path)}. Cannot proceed.")
                 return None, None, None, None, None, None, None, None
        else:
            print(f"    ERROR: Required file not found: {os.path.basename(scaled_pixel_path)}")
            return None, None, None, None, None, None, None, None

        if os.path.exists(final_ordered_channels_path):
            with open(final_ordered_channels_path, 'r') as f:
                final_ordered_channels = json.load(f)
            print(f"    Loaded final ordered channels from: {os.path.basename(final_ordered_channels_path)}")
        else:
            print(f"    ERROR: Required file not found: {os.path.basename(final_ordered_channels_path)}")
            return None, None, None, None, None, None, None, None

        if os.path.exists(cofactors_path):
            try:
                with open(cofactors_path, 'r') as f:
                    cofactors = json.load(f) # Directly load the JSON dictionary
                if not isinstance(cofactors, dict):
                    print(f"    Warning: Cofactor file {os.path.basename(cofactors_path)} is not a valid dictionary. Proceeding without cofactors.")
                    cofactors = {}
                else:
                    print(f"    Loaded cofactors from: {os.path.basename(cofactors_path)}")
            except json.JSONDecodeError:
                print(f"    ERROR: Failed to decode JSON from cofactor file {os.path.basename(cofactors_path)}. Proceeding without cofactors.")
                cofactors = {}
            except Exception as e:
                print(f"    ERROR loading cofactor file {os.path.basename(cofactors_path)}: {e}. Proceeding without cofactors.")
                cofactors = {}
        else:
             print(f"    Warning: Cofactor file not found: {os.path.basename(cofactors_path)}. Proceeding without cofactors.")
             cofactors = {} # Allow proceeding without? Or error?

        # Load ROI-specific linkage and its corresponding channel order
        if os.path.exists(pixel_linkage_matrix_path):
            try:
                pixel_linkage_matrix = np.load(pixel_linkage_matrix_path)
                print(f"    Loaded pixel linkage matrix from: {os.path.basename(pixel_linkage_matrix_path)}")
            except Exception as e_link_load:
                print(f"    ERROR loading pixel linkage matrix {os.path.basename(pixel_linkage_matrix_path)}: {e_link_load}")
                # Potentially return None for these or allow proceeding if they are optional for some plots
        else:
            print(f"    Warning: Pixel linkage matrix not found: {os.path.basename(pixel_linkage_matrix_path)}")

        if os.path.exists(pixel_linkage_channel_order_path):
            try:
                with open(pixel_linkage_channel_order_path, 'r') as f:
                    pixel_linkage_channel_order = json.load(f)
                print(f"    Loaded pixel linkage channel order from: {os.path.basename(pixel_linkage_channel_order_path)}")
            except Exception as e_link_order_load:
                print(f"    ERROR loading pixel linkage channel order {os.path.basename(pixel_linkage_channel_order_path)}: {e_link_order_load}")
        else:
            print(f"    Warning: Pixel linkage channel order not found: {os.path.basename(pixel_linkage_channel_order_path)}")
            
        # Load precomputed pixel correlation matrix
        if os.path.exists(precomputed_pixel_correlation_path):
            try:
                precomputed_pixel_correlation_matrix = pd.read_csv(precomputed_pixel_correlation_path, index_col=0)
                print(f"    Loaded precomputed pixel correlation matrix from: {os.path.basename(precomputed_pixel_correlation_path)}")
            except Exception as e_corr_load:
                print(f"    ERROR loading precomputed pixel correlation matrix {os.path.basename(precomputed_pixel_correlation_path)}: {e_corr_load}")
        else:
            print(f"    Warning: Precomputed pixel correlation matrix not found: {os.path.basename(precomputed_pixel_correlation_path)}")


        return scaled_pixel_df, final_ordered_channels, cofactors, coords_df, original_raw_pixel_data_df, pixel_linkage_matrix, pixel_linkage_channel_order, precomputed_pixel_correlation_matrix

    except Exception as e:
        print(f"  ERROR loading ROI-level data for {roi_string}: {e}")
        traceback.print_exc()
        return None, None, None, None, None, None, None, None


def load_resolution_specific_data(res_dir: str, roi_string: str, resolution_param: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Loads data files specific to a resolution subdirectory."""
    print(f"    Loading resolution-specific data ({resolution_param})...")
    data = {
        "pixel_results": None,
        "avg_profiles": None,
        "diff_profiles": None,
        "primary_channels": None,
        "umap_coords": None,
        "community_linkage_matrix": None,
        "community_correlation_matrix_precomputed": None
    }
    res_suffix = f"_res_{resolution_param}" # Construct suffix

    # Define expected file paths
    pixel_results_path = os.path.join(res_dir, f"pixel_data_with_community_annotations_{roi_string}{res_suffix}.csv")
    avg_profiles_path = os.path.join(res_dir, f"community_profiles_{roi_string}{res_suffix}.csv")
    diff_profiles_path = os.path.join(res_dir, f"community_diff_profiles_{roi_string}{res_suffix}.csv")
    primary_channels_path = os.path.join(res_dir, f"community_primary_channels_{roi_string}{res_suffix}.csv")
    umap_coords_path = os.path.join(res_dir, f"umap_coords_{roi_string}{res_suffix}.csv")
    community_linkage_path = os.path.join(res_dir, f"community_linkage_matrix_{roi_string}{res_suffix}.npy")
    community_correlation_path = os.path.join(res_dir, f"community_channel_correlation_{roi_string}{res_suffix}.csv")

    try:
        # Load required files
        if os.path.exists(pixel_results_path):
            data["pixel_results"] = pd.read_csv(pixel_results_path, index_col=0)
            print(f"      Loaded: {os.path.basename(pixel_results_path)}")
        else:
            print(f"      ERROR: Required file not found: {os.path.basename(pixel_results_path)}. Cannot generate resolution plots.")
            return data # Return partially loaded data? Or None? For now, partial.

        if os.path.exists(avg_profiles_path):
            data["avg_profiles"] = pd.read_csv(avg_profiles_path, index_col=0) # community is index
            print(f"      Loaded: {os.path.basename(avg_profiles_path)}")
        else:
            print(f"      ERROR: Required file not found: {os.path.basename(avg_profiles_path)}. Some plots will fail.")
            # Allow proceeding, but plots needing this will fail

        # Load optional files
        if os.path.exists(diff_profiles_path):
            data["diff_profiles"] = pd.read_csv(diff_profiles_path, index_col=0) # community is index
            print(f"      Loaded: {os.path.basename(diff_profiles_path)}")
        else:
            print(f"      Optional file not found: {os.path.basename(diff_profiles_path)}")

        if os.path.exists(primary_channels_path):
            # Read as Series: index=community, value=primary_channel
            data["primary_channels"] = pd.read_csv(primary_channels_path, index_col=0, header=0).squeeze("columns")
            print(f"      Loaded: {os.path.basename(primary_channels_path)}")
        else:
            print(f"      Optional file not found: {os.path.basename(primary_channels_path)}")

        if os.path.exists(umap_coords_path):
            data["umap_coords"] = pd.read_csv(umap_coords_path, index_col=0) # community is index
            print(f"      Loaded: {os.path.basename(umap_coords_path)}")
        else:
            print(f"      Optional file not found: {os.path.basename(umap_coords_path)}")

        # Load community linkage matrix
        if os.path.exists(community_linkage_path):
            try:
                data["community_linkage_matrix"] = np.load(community_linkage_path)
                print(f"      Loaded: {os.path.basename(community_linkage_path)}")
            except Exception as e_linkage:
                print(f"      ERROR loading community linkage matrix {os.path.basename(community_linkage_path)}: {e_linkage}")
        else:
            print(f"      Optional file not found: {os.path.basename(community_linkage_path)}")

        # Load precomputed community correlation matrix
        if os.path.exists(community_correlation_path):
            try:
                data["community_correlation_matrix_precomputed"] = pd.read_csv(community_correlation_path, index_col=0)
                print(f"      Loaded: {os.path.basename(community_correlation_path)}")
            except Exception as e_comm_corr:
                print(f"      ERROR loading precomputed community correlation matrix {os.path.basename(community_correlation_path)}: {e_comm_corr}")
        else:
            print(f"      Optional file not found: {os.path.basename(community_correlation_path)} (will be recalculated if needed).")

        return data

    except Exception as e:
        print(f"    ERROR loading resolution-specific data for {roi_string} res {resolution_param}: {e}")
        traceback.print_exc()
        # Return potentially partially loaded data dictionary on error
        return data


# --- Visualization Generation Functions ---

def generate_roi_level_plots(scaled_pixel_df: pd.DataFrame, coords_df: pd.DataFrame, final_ordered_channels: List[str], cofactors: Dict[str, float], roi_channels: List[str], roi_output_dir: str, roi_string: str, config: Dict, actual_raw_pixel_df: Optional[pd.DataFrame], precomp_pixel_linkage_matrix: Optional[np.ndarray], precomp_pixel_linkage_channels: Optional[List[str]], precomp_pixel_corr_matrix: Optional[pd.DataFrame]):
    """Generates plots that depend only on ROI-level data."""
    print(f"  Generating ROI-level plots for {roi_string}...")
    cfg_processing = config['processing']
    plot_dpi = cfg_processing.get('plot_dpi', 150)
    plots_subdir = os.path.join(roi_output_dir, "plots")
    os.makedirs(plots_subdir, exist_ok=True)

    # 1. Pixel Correlation Clustermap
    print("    Generating Pixel Correlation Clustermap...")
    try:
        # Extract only expression columns for correlation
        # These are the channels present in the scaled_pixel_df, excluding X, Y
        expression_cols_in_scaled_data = [ch for ch in scaled_pixel_df.columns if ch not in ['X', 'Y']]
        if not expression_cols_in_scaled_data:
             print("    ERROR: No valid expression columns found in scaled pixel data for correlation heatmap.")
        
        elif precomp_pixel_corr_matrix is not None and precomp_pixel_linkage_matrix is not None and precomp_pixel_linkage_channels is not None:
            print("    Using precomputed pixel correlation matrix and linkage for clustermap.")
            pixel_corr_path = os.path.join(plots_subdir, f"pixel_channel_correlation_heatmap_spearman_{roi_string}.png")
            
            _ = plot_correlation_clustermap(
                correlation_matrix=precomp_pixel_corr_matrix, # The precomputed one
                channels=precomp_pixel_linkage_channels, # The order corresponding to the linkage
                title=f'Pixel Channel Corr (Spearman, Asinh Scaled) - {roi_string}',
                output_path=pixel_corr_path,
                plot_dpi=plot_dpi,
                row_linkage_matrix=precomp_pixel_linkage_matrix, # Pass precomputed linkage
                col_linkage_matrix=precomp_pixel_linkage_matrix, # Pass precomputed linkage
                matrix_channel_order=precomp_pixel_linkage_channels # Important for aligning heatmap with external linkage
            )
            plt.close('all')
        
        else:
            print("    WARNING: Precomputed pixel correlation data or linkage missing. Falling back to on-the-fly calculation for clustermap.")
            # Fallback: Calculate correlation matrix on the fly using expression_cols_in_scaled_data
            # This part should align with how plot_correlation_clustermap expects its inputs if linkage is NOT provided.
            if not expression_cols_in_scaled_data:
                 print("    ERROR: No valid expression columns found in scaled pixel data for on-the-fly correlation.")
            else:
                pixel_corr_matrix_fallback = scaled_pixel_df[expression_cols_in_scaled_data].corr(method='spearman')
                pixel_corr_path = os.path.join(plots_subdir, f"pixel_channel_correlation_heatmap_spearman_{roi_string}_fallback.png")
                _ = plot_correlation_clustermap(
                    correlation_matrix=pixel_corr_matrix_fallback,
                    channels=expression_cols_in_scaled_data, # Use channels from scaled data
                    title=f'Pixel Channel Corr (Spearman, Asinh Scaled, Fallback) - {roi_string}',
                    output_path=pixel_corr_path,
                    plot_dpi=plot_dpi
                    # Not passing linkage here, so it will calculate its own.
                )
                plt.close('all')

    except Exception as e:
        print(f"    WARNING: Failed to generate pixel correlation clustermap: {e}")
        traceback.print_exc()

    # 1b. Analyze Channel Dendrogram by Controls (using the same precomputed pixel linkage)
    if precomp_pixel_linkage_matrix is not None and precomp_pixel_linkage_channels is not None:
        print("    Analyzing channel dendrogram by controls...")
        background_channels_cfg = config.get('data', {}).get('background_channels', [])
        protein_channels_cfg = config.get('data', {}).get('protein_channels', [])
        try:
            analyze_dendrogram_by_controls(
                linkage_matrix=precomp_pixel_linkage_matrix,
                channel_labels_for_linkage=precomp_pixel_linkage_channels,
                background_channels=background_channels_cfg,
                protein_channels=protein_channels_cfg,
                roi_string=roi_string,
                output_dir=plots_subdir, # Save analysis report and focused dendrogram in plots_subdir
                plot_dpi=plot_dpi
            )
        except Exception as e_acd:
            print(f"    WARNING: Failed during analysis of channel dendrogram by controls: {e_acd}")
            traceback.print_exc()
    else:
        print("    Skipping analysis of channel dendrogram by controls: Precomputed pixel linkage or its channel labels are missing.")

    # 2. Spatial Expression Grid (Scaled)
    print("    Generating Spatial Expression Grid (Scaled)...")
    try:
        # Combine coords and scaled data (already done in scaled_pixel_df)
        grid_path = os.path.join(plots_subdir, f"spatial_expression_grid_scaled_{roi_string}.png") # Use PNG for potentially large grids
        # Get channel list from the actual scaled data columns used
        channels_in_data = [ch for ch in roi_channels if ch in scaled_pixel_df.columns]
        
        # Determine the raw data to use for comparison
        raw_data_for_plot = actual_raw_pixel_df if actual_raw_pixel_df is not None else scaled_pixel_df
        if actual_raw_pixel_df is None:
            print("    WARNING: Using scaled data as fallback for 'raw' in comparison plot due to loading issues with original raw file.")

        if channels_in_data:
             cofactor_display_string = get_cofactor_subtitle_string(cofactors)
             plot_title_roi_string = f"{roi_string} (Asinh Cof: {cofactor_display_string})"
             plot_raw_vs_scaled_spatial_comparison(
                 roi_raw_data=raw_data_for_plot, 
                 scaled_pixel_expression=scaled_pixel_df, 
                 roi_channels=roi_channels, 
                 config=config, # Pass full config for internal plot settings
                 output_path=grid_path,
                 roi_string=plot_title_roi_string,
                 plot_dpi=plot_dpi
             )
             plt.close('all')
        else:
             print("    ERROR: No channels found in scaled_pixel_df to plot.")
    except Exception as e:
        print(f"    WARNING: Failed to generate spatial expression grid plot: {e}")
        traceback.print_exc()

    gc.collect()

def generate_resolution_level_plots(res_data: Dict[str, Optional[pd.DataFrame]], scaled_pixel_df: pd.DataFrame, ordered_channels: List[str], roi_channels: List[str], roi_output_dir: str, roi_string: str, resolution_param: str, config: Dict, cofactors: Dict[str, float]):
    """Generates plots specific to a resolution level."""
    print(f"  Generating plots for resolution {resolution_param}...")
    cfg_processing = config['processing']
    plot_dpi = cfg_processing.get('plot_dpi', 150)
    res_plots_subdir = os.path.join(roi_output_dir, f"resolution_{resolution_param}", "plots")
    os.makedirs(res_plots_subdir, exist_ok=True)

    # Check for essential data
    pixel_results = res_data.get("pixel_results")
    avg_profiles = res_data.get("avg_profiles")
    if pixel_results is None or avg_profiles is None:
        print("    Skipping resolution plots: Missing essential data (pixel_results or avg_profiles).")
        return

    # Load optional data
    diff_profiles = res_data.get("diff_profiles")
    primary_channels_map = res_data.get("primary_channels")
    umap_coords = res_data.get("umap_coords")
    precomputed_community_correlation_matrix = res_data.get("community_correlation_matrix_precomputed") # Get the precomputed matrix

    res_suffix = f"_res_{resolution_param}" # Recreate suffix for filenames

    # 0b. Spatial Community Maps with Channel Intensity (New Plot section)
    print(f"    Generating Spatial Community Maps with Channel Intensity ({resolution_param})...")
    try:
        if pixel_results is not None and not pixel_results.empty and \
           scaled_pixel_df is not None and not scaled_pixel_df.empty:
            
            channel_community_maps_dir = os.path.join(res_plots_subdir, "channel_community_maps")
            # The plotting function will create this directory if it doesn't exist.

            base_filename_for_plot = f"spatial_channel_comm_map_{roi_string}{res_suffix}"
            title_prefix_for_plot = f'Spatial Map - {roi_string} (Res: {resolution_param})'

            scatter_size_intensity = config.get('processing', {}).get('visualization', {}).get('scatter_size', 1.0)
            # Default border color and linewidth are set in the function, can be overridden from config if needed.
            # community_border_color = config.get('processing', {}).get('visualization', {}).get('community_border_color', 'cyan')
            # community_border_linewidth = config.get('processing', {}).get('visualization', {}).get('community_border_linewidth', 0.5)
            
            plot_spatial_community_channel_maps(
                pixel_results_df=pixel_results,      # For community assignments
                scaled_pixel_df=scaled_pixel_df,     # For channel intensities and X,Y coords
                roi_channels=roi_channels,           # List of channel names to plot from roi_level data
                output_dir=channel_community_maps_dir,
                base_filename=base_filename_for_plot,
                title_prefix=title_prefix_for_plot,
                plot_dpi=plot_dpi,
                scatter_size=scatter_size_intensity
                # community_border_color=community_border_color, # Pass if made configurable
                # community_border_linewidth=community_border_linewidth # Pass if made configurable
            )
        else:
            missing_data = []
            if pixel_results is None or pixel_results.empty:
                missing_data.append("pixel_results data")
            if scaled_pixel_df is None or scaled_pixel_df.empty:
                missing_data.append("scaled_pixel_df data")
            print(f"    Skipping spatial community maps with channel intensity: Missing data ({', '.join(missing_data)}).")
    except Exception as e:
        print(f"    WARNING: Failed to generate spatial community maps with channel intensity: {e}")
        traceback.print_exc()

    # 1. Community Size Distribution
    print(f"    Generating Community Size Distribution ({resolution_param})...")
    try:
        if 'community' in pixel_results.columns:
            community_sizes = pixel_results['community'].value_counts()
            if not community_sizes.empty:
                size_dist_path = os.path.join(res_plots_subdir, f"community_size_distribution_{roi_string}{res_suffix}.png")
                plot_community_size_distribution(
                    community_sizes=community_sizes,
                    output_path=size_dist_path,
                    title=f'Community Size Distribution - {roi_string} (Res: {resolution_param})',
                    plot_dpi=plot_dpi
                )
                plt.close('all')
            else:
                 print("    Skipping size distribution: No communities found in pixel results.")
        else:
             print("    Skipping size distribution: 'community' column missing.")
    except Exception as e:
        print(f"    WARNING: Failed to plot community size distribution: {e}")
        traceback.print_exc()

    # 2. Community Correlation Clustermap
    print(f"    Generating Community Correlation Clustermap ({resolution_param})...")
    try:
        if not avg_profiles.empty:
            # Use channels available in avg_profiles
            channels_in_profiles = list(avg_profiles.columns)
            community_correlation_matrix = None
            plot_title_suffix = ""

            if precomputed_community_correlation_matrix is not None and not precomputed_community_correlation_matrix.empty:
                community_correlation_matrix = precomputed_community_correlation_matrix
                # Ensure the channels in the precomputed matrix match avg_profiles, or at least are a subset
                if not all(ch in channels_in_profiles for ch in community_correlation_matrix.columns) or \
                   not all(ch in community_correlation_matrix.columns for ch in channels_in_profiles):
                    print("    WARNING: Channels in precomputed community correlation matrix mismatch with avg_profiles. Recalculating.")
                    community_correlation_matrix = avg_profiles.corr(method='spearman')
                    plot_title_suffix = " (Recalculated)"
                else:
                    print("    Using precomputed community correlation matrix for clustermap.")
                    # Ensure the matrix is aligned with the channels_in_profiles for the plot call
                    community_correlation_matrix = community_correlation_matrix.reindex(index=channels_in_profiles, columns=channels_in_profiles).fillna(0)
            else:
                print("    Precomputed community correlation matrix not found or empty. Calculating on-the-fly.")
                community_correlation_matrix = avg_profiles.corr(method='spearman')
                plot_title_suffix = " (Calculated)"

            comm_corr_path = os.path.join(res_plots_subdir, f"community_channel_correlation_heatmap_spearman_{roi_string}{res_suffix}.png")
            _ = plot_correlation_clustermap(
                 correlation_matrix=community_correlation_matrix,
                 channels=channels_in_profiles, # Use actual channels in profiles
                 title=f'Community Corr (Spearman, Avg. Scaled) - {roi_string} (Res: {resolution_param}){plot_title_suffix}',
                 output_path=comm_corr_path,
                 plot_dpi=plot_dpi
            )
            plt.close('all')
        else:
            print("    Skipping community correlation: Average profiles data is empty.")
    except Exception as e:
        print(f"    WARNING: Failed to generate community correlation map: {e}")
        traceback.print_exc()

    # 3. UMAP Scatter Plot
    print(f"    Generating UMAP Scatter Plot ({resolution_param})...")
    if umap_coords is not None and primary_channels_map is not None:
        try:
            if not umap_coords.empty and not primary_channels_map.empty:
                # Infer protein marker channels (e.g., from diff_profiles columns or scaled_profiles columns)
                # Using avg_profiles columns as a fallback if diff_profiles is missing
                if diff_profiles is not None and not diff_profiles.empty:
                    marker_channels = list(diff_profiles.columns)
                elif not avg_profiles.empty:
                    marker_channels = list(avg_profiles.columns)
                else:
                    marker_channels = roi_channels # Fallback to all ROI channels

                umap_scatter_path = os.path.join(res_plots_subdir, f"umap_community_scatter_{roi_string}{res_suffix}.png")
                plot_umap_scatter(
                    umap_coords=umap_coords,
                    community_top_channel_map=primary_channels_map,
                    protein_marker_channels=marker_channels,
                    roi_string=f"{roi_string} (Res: {resolution_param})",
                    output_path=umap_scatter_path,
                    plot_dpi=plot_dpi
                )
                plt.close('all')
            else:
                print("    Skipping UMAP scatter: UMAP coordinates or primary channel map is empty.")
        except Exception as e:
            print(f"    WARNING: Failed to generate UMAP scatter plot: {e}")
            traceback.print_exc()
    else:
        print("    Skipping UMAP scatter plot: Missing UMAP coordinates or primary channel map data.")


    # 4. Combined Co-expression Matrix
    print(f"    Generating Combined Co-expression Matrix ({resolution_param})...")
    try:
        # pixel_results should contain scaled data + avg mapped data
        required_cols_present = 'X' in pixel_results.columns and 'Y' in pixel_results.columns
        avg_cols_present = any(col.endswith('_asinh_scaled_avg') for col in pixel_results.columns)

        if required_cols_present and avg_cols_present and not scaled_pixel_df.empty:
             coexp_matrix_path = os.path.join(res_plots_subdir, f"coexpression_matrix_scaled_vs_avg_{roi_string}{res_suffix}.png")
             cofactor_display_string = get_cofactor_subtitle_string(cofactors)
             plot_title_roi_string = f"{roi_string} (Res: {resolution_param}, Asinh Cof: {cofactor_display_string})"
             plot_coexpression_matrix(
                 scaled_pixel_expression=scaled_pixel_df, # Original scaled pixel data (needs X, Y)
                 pixel_results_df_with_avg=pixel_results, # DF containing avg mapped values
                 ordered_channels=ordered_channels, # Use ROI-level order
                 roi_string=plot_title_roi_string,
                 config=config,
                 output_path=coexp_matrix_path,
                 plot_dpi=plot_dpi
             )
             plt.close('all')
        else:
             missing = []
             if not required_cols_present: missing.append("X/Y coords")
             if not avg_cols_present: missing.append("mapped avg columns")
             if scaled_pixel_df.empty: missing.append("scaled pixel data")
             print(f"    Skipping combined co-expression matrix: Missing required data ({', '.join(missing)}). Check 'pixel_results_annotated' file content.")

    except Exception as e:
        print(f"    WARNING: Failed to generate combined co-expression matrix: {e}")
        traceback.print_exc()

   
    gc.collect()


# --- Wrapper function for processing a single ROI (for multiprocessing) ---
def process_roi_wrapper(roi_dir_and_config: Tuple[str, Dict]) -> Tuple[str, bool, Optional[str]]:
    """Wrapper to process a single ROI. Loads data, generates plots.
    Designed to be called by multiprocessing.Pool.

    Args:
        roi_dir_and_config: A tuple containing (roi_dir, config).

    Returns:
        A tuple: (roi_string, success_status, error_message_if_any).
    """
    roi_dir, config = roi_dir_and_config
    roi_string = os.path.basename(roi_dir)
    # Add PID to print statements for better tracking in parallel execution
    print(f"\n================ Processing ROI: {roi_string} (PID: {os.getpid()}) ================")
    try:
        # 1. Load ROI-Level Data
        # Ensure variable names here match what load_required_roi_data returns and what generate_roi_level_plots expects
        scaled_pixel_df, final_ordered_channels, cofactors, coords_df, actual_raw_df, \
        precomp_pixel_linkage_matrix, precomp_pixel_linkage_channels, precomp_pixel_corr_matrix = load_required_roi_data(roi_dir, roi_string, config)

        if scaled_pixel_df is None or final_ordered_channels is None or coords_df is None:
            msg = f"Skipping ROI {roi_string} (PID: {os.getpid()}): Failed to load essential ROI-level data."
            print(f"  {msg}")
            return roi_string, False, msg

        roi_channels = [ch for ch in scaled_pixel_df.columns if ch not in ['X', 'Y']]
        if not roi_channels:
            msg = f"Skipping ROI {roi_string} (PID: {os.getpid()}): No channel columns found."
            print(f"  {msg}")
            return roi_string, False, msg

        # 2. Generate ROI-Level Plots
        generate_roi_level_plots(
            scaled_pixel_df, coords_df, final_ordered_channels, cofactors, 
            roi_channels, roi_dir, roi_string, config, actual_raw_df,
            precomp_pixel_linkage_matrix, precomp_pixel_linkage_channels, precomp_pixel_corr_matrix
        )

        # 3. Find and Process Resolution Subdirectories
        resolution_dirs = glob.glob(os.path.join(roi_dir, "resolution_*"))
        if not resolution_dirs:
            print(f"  (PID: {os.getpid()}) No resolution subdirectories found for ROI {roi_string}.")
        else:
            print(f"  (PID: {os.getpid()}) Found {len(resolution_dirs)} resolution directories for ROI {roi_string}.")

        for res_dir_name in resolution_dirs:
            resolution_param = res_dir_name.split('_')[-1]
            res_data = load_resolution_specific_data(res_dir_name, roi_string, resolution_param)
            generate_resolution_level_plots(res_data, scaled_pixel_df, final_ordered_channels, roi_channels, roi_dir, roi_string, resolution_param, config, cofactors)

        gc.collect()
        print(f"================ Finished ROI: {roi_string} (PID: {os.getpid()}) Successfully ================")
        return roi_string, True, None

    except Exception as roi_e:
        error_msg = f"FATAL ERROR processing ROI {roi_string} (PID: {os.getpid()}): {str(roi_e)}"
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"   {error_msg}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        traceback.print_exc()
        return roi_string, False, error_msg


# --- Main Execution Logic ---

def main():
    """Main function to generate visualizations."""
    print("--- Starting Visualization Generation Pipeline ---")
    start_pipeline_time = time.time()

    config = load_config("config.yaml")
    if config is None:
        print("Exiting due to configuration loading error.")
        sys.exit(1)

    base_output_dir = config['paths']['output_dir']
    print(f"Looking for processed ROI data in: {base_output_dir}")
    
    try:
        all_items = [os.path.join(base_output_dir, item) for item in os.listdir(base_output_dir)]
        roi_dirs = [d for d in all_items if os.path.isdir(d) and not d.endswith(os.path.sep + "plots")]
        if not roi_dirs:
             print(f"ERROR: No potential ROI subdirectories found in {base_output_dir}. Did the analysis run?")
             sys.exit(1)
        print(f"Found {len(roi_dirs)} potential ROI directories to process.")
    except FileNotFoundError:
         print(f"ERROR: Output directory not found: {base_output_dir}")
         sys.exit(1)
    except Exception as e:
         print(f"ERROR finding ROI directories: {e}")
         sys.exit(1)

    # --- Determine Number of Parallel Jobs ---
    num_cpus = multiprocessing.cpu_count() // 2
    parallel_jobs_config = config.get('processing', {}).get('parallel_jobs', 1)
    if parallel_jobs_config == -1:
        n_jobs = num_cpus
    elif parallel_jobs_config == -2:
        n_jobs = max(1, num_cpus - 1)
    else:
        n_jobs = max(1, int(parallel_jobs_config))
    n_jobs = min(n_jobs, len(roi_dirs)) # Don't use more jobs than ROIs
    print(f"Using {n_jobs} parallel processes for ROI visualization.")

    # --- Prepare arguments for multiprocessing --- 
    # Each item in roi_processing_args will be a tuple (roi_dir, config)
    roi_processing_args = [(roi_dir, config) for roi_dir in roi_dirs]

    # --- Process ROIs using Multiprocessing Pool ---
    processed_roi_count = 0
    failed_roi_count = 0
    all_results = [] # To store (roi_string, success_status, error_message)

    if n_jobs > 1 and len(roi_dirs) > 1: # Use multiprocessing if more than 1 job and more than 1 ROI
        print(f"Starting parallel processing of {len(roi_dirs)} ROIs with {n_jobs} workers...")
        with multiprocessing.Pool(processes=n_jobs) as pool:
            try:
                # map will block until all results are available
                all_results = pool.map(process_roi_wrapper, roi_processing_args)
            except Exception as e_pool:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"   FATAL ERROR during multiprocessing pool execution: {str(e_pool)}")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                traceback.print_exc()
                # Mark all as failed if pool itself crashes hard
                failed_roi_count = len(roi_dirs)
    else: # Sequential processing (e.g., for n_jobs=1 or single ROI)
        print(f"Starting sequential processing of {len(roi_dirs)} ROIs...")
        for arg_tuple in roi_processing_args:
            result = process_roi_wrapper(arg_tuple)
            all_results.append(result)

    # --- Aggregate Results --- 
    for roi_str, success, err_msg in all_results:
        if success:
            processed_roi_count += 1
        else:
            failed_roi_count += 1
            if err_msg: # err_msg might be None if it was just a skip without fatal error
                print(f"  Summary: ROI '{roi_str}' failed or was skipped. Error: {err_msg}")
            else:
                print(f"  Summary: ROI '{roi_str}' failed or was skipped (no specific error message captured).")

    # --- Pipeline Summary ---
    total_pipeline_time = time.time() - start_pipeline_time
    print(f"\n--- Visualization Pipeline Summary ---")
    print(f"Successfully generated plots for {processed_roi_count} ROIs.")
    if failed_roi_count > 0:
        print(f"Failed or skipped processing for {failed_roi_count} ROIs (check logs above for details).")
    print(f"Total visualization generation time: {total_pipeline_time:.2f} seconds ({total_pipeline_time/60:.2f} minutes).")
    print("================ Visualization Generation Complete ================")


if __name__ == '__main__':
    main() 