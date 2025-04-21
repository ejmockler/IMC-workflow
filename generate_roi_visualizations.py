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

# Import plotting functions from the existing visualization module
from src.roi_pipeline.pixel_visualization import (
    plot_spatial_expression_grid,
    # plot_spatial_scatter, # Consider if needed, might be replaced by grid
    plot_correlation_clustermap,
    plot_umap_scatter,
    plot_coexpression_matrix,
    plot_community_size_distribution
)

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

def load_required_roi_data(roi_dir: str, roi_string: str) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[Dict[str, float]], Optional[pd.DataFrame]]:
    """Loads the essential data files generated at the ROI level."""
    print(f"  Loading ROI-level data for {roi_string}...")
    scaled_pixel_path = os.path.join(roi_dir, f"scaled_pixel_expression_{roi_string}.csv")
    ordered_channels_path = os.path.join(roi_dir, f"ordered_channels_{roi_string}.json")
    cofactors_path = os.path.join(roi_dir, f"optimal_cofactors_{roi_string}.csv")
    # Also load raw coordinates from scaled pixel data as it includes X, Y
    coords_df = None # Will be extracted from scaled_pixel_df
    scaled_pixel_df = None
    ordered_channels = None
    cofactors = None

    try:
        if os.path.exists(scaled_pixel_path):
            scaled_pixel_df = pd.read_csv(scaled_pixel_path, index_col=0) # Assuming index is pixel ID
            # Extract coords (X, Y) and pixel expression separately
            if 'X' in scaled_pixel_df.columns and 'Y' in scaled_pixel_df.columns:
                 coords_df = scaled_pixel_df[['X', 'Y']].copy()
                 # scaled_pixel_df = scaled_pixel_df.drop(columns=['X', 'Y']) # Keep only expression for correlations etc.
                 print(f"    Loaded scaled pixel data from: {os.path.basename(scaled_pixel_path)}")
            else:
                 print(f"    ERROR: 'X' or 'Y' columns missing in {os.path.basename(scaled_pixel_path)}. Cannot proceed.")
                 return None, None, None, None
        else:
            print(f"    ERROR: Required file not found: {os.path.basename(scaled_pixel_path)}")
            return None, None, None, None

        if os.path.exists(ordered_channels_path):
            with open(ordered_channels_path, 'r') as f:
                ordered_channels = json.load(f)
            print(f"    Loaded ordered channels from: {os.path.basename(ordered_channels_path)}")
        else:
            print(f"    ERROR: Required file not found: {os.path.basename(ordered_channels_path)}")
            # Decide if we can proceed without ordered channels (maybe default order?) - for now, fail.
            return None, None, None, None

        if os.path.exists(cofactors_path):
            cofactors_df = pd.read_csv(cofactors_path)
            # Convert to dictionary format Channel -> Cofactor
            if 'Channel' in cofactors_df.columns and 'Optimal_Cofactor' in cofactors_df.columns:
                 cofactors = pd.Series(cofactors_df.Optimal_Cofactor.values, index=cofactors_df.Channel).to_dict()
                 print(f"    Loaded cofactors from: {os.path.basename(cofactors_path)}")
            else:
                  print(f"    Warning: Cofactor file {os.path.basename(cofactors_path)} has unexpected format. Proceeding without cofactors.")
                  cofactors = {} # Allow proceeding without? Or error?
        else:
             print(f"    Warning: Cofactor file not found: {os.path.basename(cofactors_path)}. Proceeding without cofactors.")
             cofactors = {} # Allow proceeding without? Or error?

        return scaled_pixel_df, ordered_channels, cofactors, coords_df

    except Exception as e:
        print(f"  ERROR loading ROI-level data for {roi_string}: {e}")
        traceback.print_exc()
        return None, None, None, None


def load_resolution_specific_data(res_dir: str, roi_string: str, resolution_param: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Loads data files specific to a resolution subdirectory."""
    print(f"    Loading resolution-specific data ({resolution_param})...")
    data = {
        "pixel_results": None,
        "avg_profiles": None,
        "diff_profiles": None,
        "primary_channels": None,
        "umap_coords": None
    }
    res_suffix = f"_res_{resolution_param}" # Construct suffix

    # Define expected file paths
    pixel_results_path = os.path.join(res_dir, f"pixel_analysis_results_final_{roi_string}{res_suffix}.csv")
    avg_profiles_path = os.path.join(res_dir, f"community_avg_scaled_profiles_{roi_string}{res_suffix}.csv") # Corrected filename based on core.py save
    diff_profiles_path = os.path.join(res_dir, f"community_diff_profiles_{roi_string}{res_suffix}.csv")
    primary_channels_path = os.path.join(res_dir, f"community_primary_channels_{roi_string}{res_suffix}.csv")
    umap_coords_path = os.path.join(res_dir, f"umap_coords_{roi_string}{res_suffix}.csv")

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

        return data

    except Exception as e:
        print(f"    ERROR loading resolution-specific data for {roi_string} res {resolution_param}: {e}")
        traceback.print_exc()
        # Return potentially partially loaded data dictionary on error
        return data


# --- Visualization Generation Functions ---

def generate_roi_level_plots(scaled_pixel_df: pd.DataFrame, coords_df: pd.DataFrame, ordered_channels: List[str], cofactors: Dict[str, float], roi_channels: List[str], roi_output_dir: str, roi_string: str, config: Dict):
    """Generates plots that depend only on ROI-level data."""
    print(f"  Generating ROI-level plots for {roi_string}...")
    cfg_processing = config['processing']
    plot_dpi = cfg_processing.get('plot_dpi', 150)
    plots_subdir = os.path.join(roi_output_dir, "plots_roi_level")
    os.makedirs(plots_subdir, exist_ok=True)

    # 1. Pixel Correlation Clustermap
    print("    Generating Pixel Correlation Clustermap...")
    try:
        # Extract only expression columns for correlation
        expression_cols = [ch for ch in roi_channels if ch in scaled_pixel_df.columns]
        if not expression_cols:
             print("    ERROR: No valid expression columns found in scaled pixel data.")
        else:
            pixel_corr_matrix = scaled_pixel_df[expression_cols].corr(method='spearman')
            pixel_corr_path = os.path.join(plots_subdir, f"pixel_channel_correlation_heatmap_spearman_{roi_string}.svg")
            # Note: This function returns order, but we use the pre-calculated `ordered_channels` for consistency elsewhere
            _ = plot_correlation_clustermap(
                correlation_matrix=pixel_corr_matrix,
                channels=expression_cols, # Plot correlation of actual data columns
                title=f'Pixel Channel Corr (Spearman, Asinh Scaled) - {roi_string}',
                output_path=pixel_corr_path,
                plot_dpi=plot_dpi
            )
            plt.close('all') # Close plot figure
    except Exception as e:
        print(f"    WARNING: Failed to generate pixel correlation clustermap: {e}")
        traceback.print_exc()

    # 2. Spatial Expression Grid (Scaled)
    print("    Generating Spatial Expression Grid (Scaled)...")
    try:
        # Combine coords and scaled data (already done in scaled_pixel_df)
        grid_path = os.path.join(plots_subdir, f"spatial_expression_grid_scaled_{roi_string}.png") # Use PNG for potentially large grids
        # Get channel list from the actual scaled data columns used
        channels_in_data = [ch for ch in roi_channels if ch in scaled_pixel_df.columns]
        if channels_in_data:
             plot_spatial_expression_grid(
                 pixel_data=scaled_pixel_df, # Contains X, Y, and scaled channels
                 channels=channels_in_data, # Use channels present in data
                 cofactors=cofactors, # Pass cofactors for potential display/use
                 title=f'Spatial Expression (Asinh Scaled) - ROI: {roi_string}',
                 output_path=grid_path,
                 config=config, # Pass full config for internal plot settings
                 plot_dpi=plot_dpi
             )
             plt.close('all')
        else:
             print("    ERROR: No channels found in scaled_pixel_df to plot.")
    except Exception as e:
        print(f"    WARNING: Failed to generate spatial expression grid plot: {e}")
        traceback.print_exc()

    gc.collect()

def generate_resolution_level_plots(res_data: Dict[str, Optional[pd.DataFrame]], scaled_pixel_df: pd.DataFrame, ordered_channels: List[str], roi_channels: List[str], roi_output_dir: str, roi_string: str, resolution_param: str, config: Dict):
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

    res_suffix = f"_res_{resolution_param}" # Recreate suffix for filenames

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
            community_correlation_matrix = avg_profiles.corr(method='spearman')
            comm_corr_path = os.path.join(res_plots_subdir, f"community_channel_correlation_heatmap_spearman_{roi_string}{res_suffix}.svg")
            _ = plot_correlation_clustermap(
                 correlation_matrix=community_correlation_matrix,
                 channels=channels_in_profiles, # Use actual channels in profiles
                 title=f'Community Corr (Spearman, Avg. Scaled) - {roi_string} (Res: {resolution_param})',
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

                umap_scatter_path = os.path.join(res_plots_subdir, f"umap_community_scatter_{roi_string}{res_suffix}.svg")
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
             coexp_matrix_path = os.path.join(res_plots_subdir, f"coexpression_matrix_scaled_vs_avg_{roi_string}{res_suffix}.svg")
             plot_coexpression_matrix(
                 scaled_pixel_expression=scaled_pixel_df, # Original scaled pixel data (needs X, Y)
                 pixel_results_df_with_avg=pixel_results, # DF containing avg mapped values
                 ordered_channels=ordered_channels, # Use ROI-level order
                 roi_string=f"{roi_string} (Res: {resolution_param})",
                 config=config,
                 output_path=coexp_matrix_path,
                 plot_dpi=plot_dpi # Pass DPI here if function supports it, otherwise handled by savefig
             )
             plt.close('all')
        else:
             missing = []
             if not required_cols_present: missing.append("X/Y coords")
             if not avg_cols_present: missing.append("mapped avg columns")
             if scaled_pixel_df.empty: missing.append("scaled pixel data")
             print(f"    Skipping combined co-expression matrix: Missing required data ({', '.join(missing)}). Check 'pixel_analysis_results_final' file content.")

    except Exception as e:
        print(f"    WARNING: Failed to generate combined co-expression matrix: {e}")
        traceback.print_exc()

    gc.collect()


# --- Main Execution Logic ---

def main():
    """Main function to generate visualizations."""
    print("--- Starting Visualization Generation Pipeline ---")
    start_pipeline_time = time.time()

    # --- Load Configuration ---
    config = load_config("config.yaml")
    if config is None:
        print("Exiting due to configuration loading error.")
        sys.exit(1)

    # --- Find Processed ROI Directories ---
    base_output_dir = config['paths']['output_dir']
    print(f"Looking for processed ROI data in: {base_output_dir}")
    
    # Simple approach: find all subdirectories in the output dir
    # Assumes directory names are the ROI strings (e.g., 'ROI_123')
    try:
        all_items = [os.path.join(base_output_dir, item) for item in os.listdir(base_output_dir)]
        roi_dirs = [d for d in all_items if os.path.isdir(d) and not d.endswith(os.path.sep + "plots")] # Exclude potential top-level plots dir
        if not roi_dirs:
             print(f"ERROR: No potential ROI subdirectories found in {base_output_dir}. Did the analysis run?")
             sys.exit(1)
        print(f"Found {len(roi_dirs)} potential ROI directories.")
    except FileNotFoundError:
         print(f"ERROR: Output directory not found: {base_output_dir}")
         sys.exit(1)
    except Exception as e:
         print(f"ERROR finding ROI directories: {e}")
         sys.exit(1)

    # --- Process Each ROI ---
    processed_roi_count = 0
    failed_roi_count = 0
    for roi_dir in roi_dirs:
        roi_string = os.path.basename(roi_dir)
        print(f"\n================ Processing ROI: {roi_string} ================")
        try:
            # 1. Load ROI-Level Data
            scaled_pixel_df, ordered_channels, cofactors, coords_df = load_required_roi_data(roi_dir, roi_string)

            if scaled_pixel_df is None or ordered_channels is None or coords_df is None:
                 print(f"  Skipping ROI {roi_string}: Failed to load essential ROI-level data.")
                 failed_roi_count += 1
                 continue

            # Infer ROI channels from scaled_pixel_df columns (excluding X, Y)
            roi_channels = [ch for ch in scaled_pixel_df.columns if ch not in ['X', 'Y']]
            if not roi_channels:
                 print(f"  Skipping ROI {roi_string}: No channel columns found in scaled pixel data.")
                 failed_roi_count += 1
                 continue

            # 2. Generate ROI-Level Plots
            generate_roi_level_plots(scaled_pixel_df, coords_df, ordered_channels, cofactors, roi_channels, roi_dir, roi_string, config)

            # 3. Find and Process Resolution Subdirectories
            try:
                 resolution_dirs = glob.glob(os.path.join(roi_dir, "resolution_*"))
                 if not resolution_dirs:
                     print("  No resolution subdirectories found.")
                     # Continue to next ROI if no resolutions
                 else:
                     print(f"  Found {len(resolution_dirs)} resolution directories.")

                 for res_dir in resolution_dirs:
                      # Extract resolution param from directory name (e.g., 'resolution_0_5' -> '0.5')
                      match = re.search(r"resolution_([\d_]+)", os.path.basename(res_dir))
                      if match:
                           resolution_param_str = match.group(1).replace('_', '.') # Convert '0_5' to '0.5'
                           print(f"\n  Processing Resolution: {resolution_param_str} ({os.path.basename(res_dir)})" )

                           # Load resolution data
                           res_data = load_resolution_specific_data(res_dir, roi_string, resolution_param_str)

                           # Generate resolution plots
                           generate_resolution_level_plots(res_data, scaled_pixel_df, ordered_channels, roi_channels, roi_dir, roi_string, resolution_param_str, config)

                      else:
                           print(f"  Skipping subdirectory - cannot parse resolution: {os.path.basename(res_dir)}")
            except Exception as res_e:
                 print(f"  ERROR processing resolutions for ROI {roi_string}: {res_e}")
                 traceback.print_exc()
                 # Consider if this counts as ROI failure or just resolution failure

            processed_roi_count += 1
            gc.collect() # Clean up memory after each ROI

        except Exception as roi_e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"   FATAL ERROR processing ROI {roi_string}: {str(roi_e)}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            traceback.print_exc()
            failed_roi_count += 1

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