import yaml
import pandas as pd
import numpy as np
import math
import os
import glob
import re
import time
import traceback
import json
import multiprocessing
import sys # For exit
from joblib import Parallel, delayed
import seaborn as sns # Added for clustermap calculation
from typing import List, Tuple, Optional, Dict, Any # Keep for type hints
import gc # Import garbage collector module
import scipy.cluster.hierarchy as sch # For clustering average correlation matrix

# Import our new utility functions
from src.roi_pipeline.gpu_utils import check_gpu_availability
from src.roi_pipeline.imc_data_utils import (
    load_and_validate_roi_data,
    calculate_asinh_cofactors_for_roi,
    apply_per_channel_arcsinh_and_scale,
    calculate_and_save_community_linkage
)
# Import our analysis functions
from src.roi_pipeline.pixel_analysis_core import (
    run_spatial_leiden,
    calculate_and_save_profiles,
    calculate_differential_expression,
    calculate_and_save_umap # Added import for UMAP function
)

# Attempt to import UMAP, set flag
try:
    import umap # Make sure umap-learn is installed
    umap_available = True
except ImportError:
    print("Warning: umap-learn package not found. Cannot perform UMAP analysis. Install with: pip install umap-learn")
    umap_available = False


# --- Configuration Loading ---
def load_config(config_path: str = "config.yaml") -> Optional[Dict]:
    """Loads the pipeline configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from: {config_path}")
        # Basic validation (can be expanded)
        if not isinstance(config, dict):
            print(f"ERROR: Configuration file {config_path} is not a valid dictionary.")
            return None
        if 'paths' not in config or 'data' not in config or 'analysis' not in config or 'processing' not in config:
            print(f"ERROR: Configuration file {config_path} is missing required top-level keys (paths, data, analysis, processing).")
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


# === Helper Functions for analyze_roi ===

def _preprocess_roi(roi_raw_data: pd.DataFrame, roi_channels: List[str], roi_cofactors: Dict[str, float], config: Dict) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """Applies arcsinh transformation and scaling to ROI data."""
    print("   Preprocessing ROI data...")
    start_time = time.time()
    scaled_pixel_expression, used_cofactors = apply_per_channel_arcsinh_and_scale(
        data_df=roi_raw_data,
        channels=roi_channels,
        cofactors_map=roi_cofactors,
        default_cofactor=config['data']['default_arcsinh_cofactor']
    )
    if scaled_pixel_expression.empty:
        print("   ERROR: Scaled expression data is empty after preprocessing.")
        return None, None
    print(f"   --- Preprocessing finished in {time.time() - start_time:.2f} seconds ---")
    return scaled_pixel_expression, used_cofactors

def _calculate_and_save_ordered_channels(
    scaled_pixel_expression: pd.DataFrame,
    roi_channels: List[str],
    roi_output_dir: str,
    roi_string: str,
    config: Dict,
    fixed_channel_order: Optional[List[str]] = None
) -> Optional[List[str]]: # Returns the final order to use downstream
    """
    Calculates ROI-specific channel clustering and determines the final channel order.

    Always calculates and saves the ROI-specific clustering (linkage matrix and order list).
    Determines the final order to use: the fixed_channel_order if provided and valid,
    otherwise defaults to the ROI-specific clustered order. Saves this final order list.

    Args:
        scaled_pixel_expression: DataFrame of scaled pixel data (channels as columns).
        roi_channels: List of channels available in the original data.
        roi_output_dir: Directory to save outputs.
        roi_string: ROI identifier.
        config: Configuration dictionary.
        fixed_channel_order: Optional predefined list of channels to use as the final order.

    Returns:
        The list of channels representing the final order to be used downstream
        (either the fixed order or the ROI-specific clustered order), or None on failure.
    """
    print("   Calculating ROI-specific pixel correlation for channel clustering...")
    start_time = time.time()
    roi_clustered_order = None
    roi_linkage_matrix = None
    final_order_to_use = None # This is what the function will return

    try:
        # Ensure X, Y are not in the correlation calculation
        expression_cols = [ch for ch in roi_channels if ch in scaled_pixel_expression.columns and ch not in ['X', 'Y']]
        if not expression_cols:
            print("   ERROR: No valid expression columns found for correlation.")
            return None

        pixel_corr_matrix = scaled_pixel_expression[expression_cols].corr(method='spearman')
        pixel_corr_matrix = pixel_corr_matrix.fillna(0) # Handle potential NaNs
        if pixel_corr_matrix.isnull().values.any():
            print("   WARNING: NaNs remain in pixel correlation matrix after fillna(0). Clustering might be unstable.")

        # --- Always perform ROI-specific clustering ---
        print("   Performing hierarchical clustering on ROI's pixel correlation matrix...")
        if len(expression_cols) < 2:
             print("   WARNING: Cannot perform clustering with less than 2 channels. Using original expression_cols order for ROI-specific clustering.")
             roi_clustered_order = expression_cols
             roi_linkage_matrix = None # Cannot compute linkage
        else:
             try:
                 dist_matrix = sch.distance.pdist(pixel_corr_matrix.values)
                 linkage_method = config.get('analysis', {}).get('clustering', {}).get('linkage', 'ward')
                 roi_linkage_matrix = sch.linkage(dist_matrix, method=linkage_method)
                 dendrogram = sch.dendrogram(roi_linkage_matrix, no_plot=True)
                 ordered_indices = dendrogram['leaves']
                 roi_clustered_order = [pixel_corr_matrix.columns[i] for i in ordered_indices]
                 print(f"   ROI-specific clustering successful. Determined order for {len(roi_clustered_order)} channels.")

                 # Save the ROI-specific Linkage Matrix
                 linkage_save_path = os.path.join(roi_output_dir, f"pixel_channel_linkage_{roi_string}.npy")
                 try:
                     np.save(linkage_save_path, roi_linkage_matrix)
                     print(f"   ROI-specific pixel channel linkage matrix saved to: {os.path.basename(linkage_save_path)}")
                 except Exception as link_save_e:
                     print(f"   ERROR saving ROI-specific pixel channel linkage matrix: {link_save_e}")
                     # Continue, but linkage might be missing

             except ValueError as ve:
                  print(f"   ERROR during ROI-specific scipy clustering (ValueError): {ve}. Using original channel order as fallback.")
                  roi_clustered_order = expression_cols # Fallback to original order
                  roi_linkage_matrix = None
             except Exception as cluster_e:
                 print(f"   ERROR during ROI-specific scipy clustering: {cluster_e}. Using original channel order as fallback.")
                 traceback.print_exc()
                 roi_clustered_order = expression_cols # Fallback to original order
                 roi_linkage_matrix = None

        # Save the ROI-specific Clustered Order list
        if roi_clustered_order:
            roi_order_save_path = os.path.join(roi_output_dir, f"pixel_channel_clustered_order_{roi_string}.json")
            try:
                with open(roi_order_save_path, 'w') as f:
                    json.dump(roi_clustered_order, f, indent=4)
                print(f"   ROI-specific clustered channel list saved to: {os.path.basename(roi_order_save_path)}")
            except Exception as json_e:
                print(f"   ERROR saving ROI-specific clustered channel list to {roi_order_save_path}: {json_e}")
                # Proceed, but this file might be missing
        else:
             print("   ERROR: Failed to determine ROI-specific channel order. Cannot proceed reliably.")
             return None # Cannot determine a fallback final order reliably

        # --- Determine the Final Order to Use Downstream ---
        if fixed_channel_order:
            print(f"   Fixed channel order provided ({len(fixed_channel_order)} channels). Validating against ROI data...")
            # Validate fixed order against available channels in the correlation matrix
            missing_channels = [ch for ch in fixed_channel_order if ch not in pixel_corr_matrix.columns]
            available_channels_in_fixed_order = [ch for ch in fixed_channel_order if ch in pixel_corr_matrix.columns]

            if missing_channels:
                 print(f"   WARNING: Channels in fixed_channel_order not found in ROI's correlation matrix: {missing_channels}.")
            if not available_channels_in_fixed_order:
                 print("   ERROR: No valid channels remaining after applying fixed_channel_order. Falling back to ROI-specific order.")
                 final_order_to_use = roi_clustered_order # Fallback if fixed order is unusable
            else:
                 final_order_to_use = available_channels_in_fixed_order # Use the validated subset
                 print(f"   Using validated fixed channel order ({len(final_order_to_use)} channels) as final order.")
        else:
            # No fixed order provided, use the ROI-specific clustered order as the final order
            print("   No fixed channel order provided. Using ROI-specific clustered order as final order.")
            final_order_to_use = roi_clustered_order

        # Save the Final Order list (might be the same as clustered order, or the fixed order)
        if final_order_to_use:
            final_order_save_path = os.path.join(roi_output_dir, f"pixel_channel_final_order_{roi_string}.json")
            try:
                with open(final_order_save_path, 'w') as f:
                    json.dump(final_order_to_use, f, indent=4)
                print(f"   Final channel order list saved to: {os.path.basename(final_order_save_path)}")
            except Exception as json_e:
                print(f"   ERROR saving final channel order list to {final_order_save_path}: {json_e}")
                # Don't necessarily fail the whole function, but downstream might have issues
                # If saving failed, final_order_to_use still holds the list in memory for return
        else:
            # This case should be caught earlier, but as a safeguard:
            print("   ERROR: Final channel order could not be determined. Aborting.")
            return None

    except Exception as e:
        print(f"   ERROR during channel order calculation: {e}")
        traceback.print_exc()
        return None # Return None on major failure

    print(f"   --- Channel ordering finished in {time.time() - start_time:.2f} seconds ---")
    # Return the list that should be used by subsequent steps
    return final_order_to_use

# Updated to accept resolution_param
def _cluster_pixels(roi_raw_data: pd.DataFrame, roi_channels: List[str], scaled_pixel_expression: pd.DataFrame, resolution_param: float, config: Dict) -> Tuple[Optional[pd.DataFrame], Optional[Any], Optional[Any]]: # Use Any for igraph/leidenalg types
    """Performs spatial Leiden clustering on pixel data for a specific resolution."""
    print(f"\nClustering pixels spatially (Resolution: {resolution_param})...")
    start_time = time.time()
    # Get GPU config setting
    use_gpu_config = config.get('processing', {}).get('use_gpu', False)
    if use_gpu_config:
        print("   GPU acceleration enabled via configuration.")
    pixel_coordinates = roi_raw_data[['X', 'Y']].copy().loc[scaled_pixel_expression.index]
    pixel_community_df, pixel_graph, community_partition, exec_time = run_spatial_leiden(
        analysis_df=pixel_coordinates,
        protein_channels=roi_channels,
        scaled_expression_data_for_weights=scaled_pixel_expression.values,
        n_neighbors=config['analysis']['clustering']['n_neighbors'],
        resolution_param=resolution_param, # Use passed parameter
        seed=config['analysis']['clustering']['seed'],
        verbose=True, # Keep verbose on
        use_gpu_config=use_gpu_config # Pass GPU config setting
    )
    if pixel_community_df is None or pixel_graph is None or community_partition is None:
        print(f"   ERROR: Leiden clustering failed for resolution {resolution_param}.")
        return None, None, None
    print(f"   --- Clustering finished in {time.time() - start_time:.2f} seconds (Leiden took {exec_time:.2f}s) ---")
    return pixel_community_df, pixel_graph, community_partition

# Updated to accept resolution_output_dir
def _analyze_communities(pixel_results_df: pd.DataFrame, roi_channels: List[str], pixel_graph: Any, resolution_output_dir: str, roi_string: str, resolution_param: float, ordered_channels: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series]]:
    """Calculates community profiles and differential expression for a specific resolution."""
    print(f"\nAnalyzing community characteristics (Resolution: {resolution_param})...")
    start_time = time.time()
    # Calculate Profiles
    scaled_community_profiles = calculate_and_save_profiles(
         results_df=pixel_results_df,
         valid_channels=roi_channels,
         roi_output_dir=resolution_output_dir, # Save profiles in resolution subdir
         roi_string=f"{roi_string}_res_{resolution_param}", # Add resolution to filename
         ordered_channels=ordered_channels # Pass the desired final order
    )
    if scaled_community_profiles is None or scaled_community_profiles.empty:
        print(f"   Skipping differential expression for resolution {resolution_param}: Profile calculation failed or no communities found.")
        return None, None, None

    # Calculate Differential Expression
    diff_expr_profiles, primary_channel_map = calculate_differential_expression(
        results_df=pixel_results_df,
        community_profiles=scaled_community_profiles,
        graph=pixel_graph,
        valid_channels=roi_channels
    )

    # Save differential expression results in resolution subdir
    if diff_expr_profiles is not None and not diff_expr_profiles.empty:
        diff_profiles_path = os.path.join(resolution_output_dir, f"community_diff_profiles_{roi_string}_res_{resolution_param}.csv")
        diff_expr_profiles.to_csv(diff_profiles_path)
        print(f"   Differential profiles saved to: {os.path.basename(diff_profiles_path)}")
    if primary_channel_map is not None and not primary_channel_map.empty:
        top_channel_path = os.path.join(resolution_output_dir, f"community_primary_channels_{roi_string}_res_{resolution_param}.csv")
        primary_channel_map.to_csv(top_channel_path, header=True)
        print(f"   Primary channel map saved to: {os.path.basename(top_channel_path)}")
    else:
         print(f"   Primary channel mapping skipped for resolution {resolution_param} as DiffEx failed or produced no results.")

    print(f"   --- Community analysis finished in {time.time() - start_time:.2f} seconds ---")
    return scaled_community_profiles, diff_expr_profiles, primary_channel_map

# === Main Analysis Orchestration ===

def analyze_roi(file_idx: int, file_path: str, total_files: int, config: Dict, \
                    roi_metadata: Optional[Dict] = None, reference_channel_order: Optional[List[str]] = None, first_timepoint_value: Optional[Any] = None):
    """Orchestrates the analysis pipeline for a single ROI file, iterating through resolutions."""
    print(f"\n================ Analyzing ROI {file_idx+1}/{total_files}: {os.path.basename(file_path)} ================")

    cfg_paths = config['paths']
    cfg_data = config['data']
    cfg_analysis = config['analysis']
    cfg_processing = config['processing'] # Get processing config
    resolution_params = cfg_analysis['clustering'].get('resolution_params', [0.5])
    if not isinstance(resolution_params, list) or not resolution_params:
        print("Warning: 'resolution_params' not found or invalid in config. Defaulting to [0.5].")
        resolution_params = [0.5]
    print(f"Configured resolutions: {resolution_params}")

    print("Loading and validating data...")
    start_time_load = time.time()
    roi_string, roi_output_dir, roi_raw_data, roi_channels = load_and_validate_roi_data(
        file_path=file_path,
        master_protein_channels=cfg_data['master_protein_channels'],
        base_output_dir=cfg_paths['output_dir'],
        metadata_cols=cfg_data['metadata_cols']
    )
    if roi_raw_data is None or roi_channels is None or roi_output_dir is None:
        print(f"Skipping file {os.path.basename(file_path)} due to errors during loading or validation.")
        return None
    print(f"--- Loading finished in {time.time() - start_time_load:.2f} seconds ---")

    print("\nCalculating optimal cofactors...")
    start_time_cofactor = time.time()
    roi_cofactors = calculate_asinh_cofactors_for_roi(
        roi_df=roi_raw_data,
        channels_to_process=roi_channels,
        default_cofactor=cfg_data['default_arcsinh_cofactor'],
        output_dir=roi_output_dir,
        roi_string=roi_string
    )
    print(f"--- Cofactor calculation finished in {time.time() - start_time_cofactor:.2f} seconds ---")

    try:
        # --- Preprocessing ---
        scaled_pixel_expression, used_cofactors = _preprocess_roi(roi_raw_data, roi_channels, roi_cofactors, config)
        if scaled_pixel_expression is None:
            print(f"ERROR: Preprocessing failed for ROI {roi_string}. Aborting.")
            return None

        # --- Save Scaled Pixel Expression ---
        # Note: This DF only contains scaled channel data, not X/Y
        scaled_expr_save_path = os.path.join(roi_output_dir, f"scaled_pixel_expression_{roi_string}.csv")
        try:
            scaled_pixel_expression.to_csv(scaled_expr_save_path, index=True)
            print(f"   Scaled pixel expression saved to: {os.path.basename(scaled_expr_save_path)}")
        except Exception as save_e:
            print(f"   ERROR saving scaled pixel expression: {save_e}")
            # Decide if this is fatal - potentially yes, as it's needed later.
            return None
        # --- End Save ---

        # Determine if the current ROI is from the first timepoint
        is_first_timepoint_roi = False
        current_timepoint = None
        if roi_metadata and first_timepoint_value is not None and config['experiment_analysis']['timepoint_col'] in roi_metadata:
            timepoint_col = config['experiment_analysis']['timepoint_col']
            current_timepoint = roi_metadata.get(timepoint_col)
            # Compare safely (handle numeric/string)
            is_first_timepoint_roi = str(current_timepoint) == str(first_timepoint_value)
            if is_first_timepoint_roi:
                print(f"   Processing as first timepoint ROI (Timepoint: {current_timepoint})")
            else:
                print(f"   Processing as subsequent timepoint ROI (Timepoint: {current_timepoint}, Reference Timepoint: {first_timepoint_value})")
        elif reference_channel_order is not None:
            # If we have a reference order, but couldn't determine this ROI's timepoint, assume it's not the first
            print(f"   WARNING: Could not determine timepoint for ROI {roi_string}. Using provided reference order.")
            is_first_timepoint_roi = False
        else:
             # No metadata or no reference timepoint identified - proceed normally
             print(f"   Proceeding without timepoint-based reference order.")

        # Determine the channel order to use for this ROI (or use reference)
        fixed_order_for_calc = None
        if not is_first_timepoint_roi and reference_channel_order is not None:
            print(f"   Using reference channel order ({len(reference_channel_order)} channels).")
            fixed_order_for_calc = reference_channel_order
        else:
            print(f"   Generating channel order based on this ROI's data ({len(roi_channels)} channels).")

        # Calculate and save the channel order (potentially fixed)
        ordered_channels_for_downstream = _calculate_and_save_ordered_channels(
            scaled_pixel_expression=scaled_pixel_expression,
            roi_channels=roi_channels,
            roi_output_dir=roi_output_dir,
            roi_string=roi_string,
            config=config,
            fixed_channel_order=fixed_order_for_calc
        )

        # If channel order determination failed, we cannot proceed reliably.
        if ordered_channels_for_downstream is None:
             print(f"ERROR: Failed to determine final channel order for ROI {roi_string}. Aborting analysis for this ROI.")
             return None # Indicate failure

        # --- Calculate and Save Pixel Correlation Matrix ---
        # Use the final downstream order for saving the correlation matrix to ensure consistency
        # with other outputs like community profiles.
        print("   Calculating and saving pixel correlation matrix (using final channel order)...")
        try:
            # Use the final downstream order for the matrix columns/index
            pixel_corr_matrix = scaled_pixel_expression[ordered_channels_for_downstream].corr(method='spearman')
            pixel_corr_save_path = os.path.join(roi_output_dir, f"pixel_channel_correlation_{roi_string}.csv")
            pixel_corr_matrix.to_csv(pixel_corr_save_path)
            print(f"   Pixel correlation matrix saved to: {os.path.basename(pixel_corr_save_path)}")
        except KeyError as ke:
             print(f"   ERROR calculating pixel correlation: Channel mismatch between final order and scaled_pixel_expression columns: {ke}. Skipping save.")
        except Exception as corr_e:
             print(f"   ERROR calculating or saving pixel correlation matrix: {corr_e}")
             # Decide if this is fatal - maybe not strictly fatal, but vis will fail.
        # --- End Save ---

        # If this IS the first timepoint ROI, the 'ordered_channels_for_downstream' variable now holds the order calculated from its data.
        # This list will be returned by analyze_roi for potential consensus calculation.

        print(f"\n>>> Processing {len(resolution_params)} Leiden resolutions: {resolution_params} <<<")
        success_flag = False
        for resolution in resolution_params:
            pixel_community_df = None
            pixel_graph = None
            community_partition = None
            current_pixel_results_df = None
            scaled_community_profiles = None
            diff_expr_profiles = None
            primary_channel_map = None
            umap_coords = None # Initialize umap_coords for this resolution loop

            try:
                resolution_str = f"{resolution:.3f}".rstrip('0').rstrip('.').replace('.', '_') if isinstance(resolution, float) else str(resolution)
                resolution_output_dir = os.path.join(roi_output_dir, f"resolution_{resolution_str}")
                os.makedirs(resolution_output_dir, exist_ok=True)
                print(f"\n===== Processing Resolution: {resolution} (Output: {resolution_output_dir}) =====")

                pixel_community_df, pixel_graph, community_partition = _cluster_pixels(roi_raw_data, roi_channels, scaled_pixel_expression, resolution, config)
                if pixel_community_df is None:
                    print(f"Skipping further analysis for resolution {resolution} due to clustering failure.")
                    continue

                # Prepare the main results dataframe for this resolution
                current_pixel_results_df = roi_raw_data[['X', 'Y']].join(scaled_pixel_expression).join(pixel_community_df[['community']])
                # Optionally add raw data back if needed, though maybe not required if scaled is saved
                # for ch in roi_channels:
                #      if ch not in current_pixel_results_df.columns and ch in roi_raw_data.columns:
                #          current_pixel_results_df[ch] = roi_raw_data.loc[current_pixel_results_df.index, ch]

                # Save intermediate scaled results (already done before community analysis?) - Let's save the final one instead later.
                # scaled_results_path = os.path.join(resolution_output_dir, f"pixel_scaled_results_{roi_string}_res_{resolution}.csv")
                # current_pixel_results_df.to_csv(scaled_results_path, index=True)
                # print(f"   Intermediate pixel results saved to: {os.path.basename(scaled_results_path)}")

                scaled_community_profiles, diff_expr_profiles, primary_channel_map = _analyze_communities(
                    current_pixel_results_df, roi_channels, pixel_graph, resolution_output_dir, roi_string, resolution, ordered_channels_for_downstream
                )
                if scaled_community_profiles is None:
                    print(f"Skipping UMAP and further analysis for resolution {resolution} due to community analysis failure.")
                    # Still need to save the final pixel results even if profiles failed? Yes.
                    # The save happens later. Continue to next resolution.
                    continue # Skip UMAP, correlation, linkage etc. if profiles failed
                else:
                    # --- Calculate and Save Community Correlation Matrix ---
                    print(f"   Calculating and saving community correlation matrix for resolution {resolution}...")
                    try:
                        # Use ordered_channels_for_downstream for consistency
                        community_corr_matrix = scaled_community_profiles[ordered_channels_for_downstream].corr(method='spearman')
                        community_corr_save_path = os.path.join(
                            resolution_output_dir,
                            f"community_channel_correlation_{roi_string}_res_{resolution}.csv"
                        )
                        community_corr_matrix.to_csv(community_corr_save_path)
                        print(f"   Community correlation matrix saved to: {os.path.basename(community_corr_save_path)}")
                    except KeyError as ke:
                         print(f"     ERROR calculating community correlation: Channel mismatch between final order and community profile columns: {ke}. Skipping save.")
                    except Exception as comm_corr_e:
                         print(f"     ERROR calculating or saving community correlation matrix: {comm_corr_e}")
                    # --- End Save ---

                    # --- Calculate and Save UMAP ---
                    umap_coords = calculate_and_save_umap(
                        diff_expr_profiles=diff_expr_profiles,
                        scaled_community_profiles=scaled_community_profiles,
                        roi_channels=roi_channels,
                        resolution_output_dir=resolution_output_dir,
                        roi_string=roi_string,
                        resolution_param=resolution,
                        config=config,
                        umap_available=umap_available # Pass the flag
                    )

                    # --- Calculate and Save Community Linkage Matrix using Utility Function ---
                    linkage_file_prefix = f"community_linkage_matrix_{roi_string}_res_{resolution}"
                    community_linkage = calculate_and_save_community_linkage(
                        scaled_community_profiles=scaled_community_profiles,
                        ordered_channels=ordered_channels_for_downstream, # Use final order here too
                        output_dir=resolution_output_dir,
                        file_prefix=linkage_file_prefix,
                        config=config
                    )

                # Add primary channel mapping to the final results dataframe (if calculated)
                if primary_channel_map is not None and not primary_channel_map.empty:
                    # Convert mapped series to object type BEFORE fillna to handle potential new 'Mapping Error' string
                    current_pixel_results_df['primary_channel'] = current_pixel_results_df['community'].map(primary_channel_map).astype(object).fillna('Mapping Error')
                else:
                    current_pixel_results_df['primary_channel'] = 'Not Calculated'

                # Add average community profiles mapped back to pixels (if calculated)
                if scaled_community_profiles is not None and not scaled_community_profiles.empty:
                    for channel in roi_channels:
                         avg_col_name = f'{channel}_asinh_scaled_avg'
                         # Map the average scaled profile for the community onto each pixel, leaving NaNs if map fails
                         current_pixel_results_df[avg_col_name] = current_pixel_results_df['community'].map(scaled_community_profiles[channel])

                final_results_save_path = os.path.join(
                    resolution_output_dir,
                    f"pixel_results_annotated_{roi_string}_res_{resolution}.csv"
                )
                current_pixel_results_df.to_csv(final_results_save_path, index=True)
                print(f"\n   Annotated pixel results for resolution {resolution} saved to: {os.path.basename(final_results_save_path)}")
                success_flag = True

            except Exception as resolution_e:
                 print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 print(f"   ERROR during processing resolution {resolution} for ROI {roi_string}: {str(resolution_e)}")
                 print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 traceback.print_exc()

            finally:
                print(f"   Cleaning up memory for resolution {resolution}...")
                del pixel_community_df
                del pixel_graph
                del community_partition
                del current_pixel_results_df
                del scaled_community_profiles
                del diff_expr_profiles
                del primary_channel_map
                del umap_coords # Clean up umap_coords
                gc.collect()

        if not success_flag:
             print(f"\n--- WARNING: Analysis failed for all resolutions for ROI: {roi_string} ---")
             return None # Indicate overall failure if no resolution worked

        print(f"\n--- Successfully finished processing all resolutions for ROI: {roi_string} ---")
        # Return the ROI string AND the final channel order used/calculated for its plots
        return roi_string, ordered_channels_for_downstream

    except Exception as e: # Catch errors outside the resolution loop
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"   FATAL ERROR during analysis for ROI {roi_string} (outside resolution loop): {str(e)}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        traceback.print_exc()
        # Ensure failure returns None, which is not a tuple, so downstream processing can handle it
        return None # Indicate failure

# --- Script Execution Entry Point ---
if __name__ == '__main__':
    # Need to import plt for seaborn clustermap even if not showing plots directly
    # --- REMOVED: plt import no longer needed --- #
    # import matplotlib.pyplot as plt
    print("\n--- Starting IMC Pixel-wise Analysis Pipeline (Analysis Only) ---")
    # Check for GPU acceleration
    gpu_available = check_gpu_availability()
    if gpu_available:
        print("GPU acceleration is available and will be used where implemented.")
    else:
        print("GPU acceleration is not available or not properly configured. Proceeding with CPU-only path.")

    start_pipeline_time = time.time()

    # --- Load Configuration ---
    config = load_config("config.yaml") # Load from default path
    if config is None:
        print("Exiting due to configuration loading error.")
        sys.exit(1) # Exit script if config fails

    # --- 1. Find Input Files ---
    data_dir = config['paths']['data_dir']
    try:
        imc_files = glob.glob(os.path.join(data_dir, "*.txt"))
        if not imc_files:
            print(f"ERROR: No .txt files found in data directory: {data_dir}")
            sys.exit(1)
        print(f"\nFound {len(imc_files)} IMC data files to process.")
    except Exception as e:
         print(f"ERROR finding input files in {data_dir}: {e}")
         sys.exit(1)

    # --- 2. Setup Output Directory ---
    output_dir = config['paths']['output_dir']
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {output_dir}")
    except Exception as e:
         print(f"ERROR creating output directory {output_dir}: {e}")
         sys.exit(1)

    # --- 2b. Load Metadata (Early) ---
    metadata_file = config['paths'].get('metadata_file')
    metadata = None
    metadata_map = {}
    first_timepoint = None
    
    # Define helper function locally within the main block or globally if preferred
    def get_roi_string_from_path(p):
        """
        Extracts the ROI string (e.g. ROI_D7_M1_01_21) from 
        a full filename like IMC_241218_Alun_ROI_D7_M1_01_21.txt
        """
        fname = os.path.basename(p)
        base, _ = os.path.splitext(fname)
        # Updated Regex: Look for ROI_ followed by alphanumeric/underscore, 
        # ensuring it's anchored reasonably (e.g., preceded by _ or start)
        # and extending towards the end. Adjust if pattern is different.
        match = re.search(r'(?:^|_)(ROI_[A-Za-z0-9_]+)', base) 
        if match:
            return match.group(1)
        else:
            # Fallback or error handling needed if pattern MUST exist
            print(f"Warning: Could not extract standard ROI string from '{fname}'. Using base '{base}'.")
            return base
            
    if metadata_file and os.path.exists(metadata_file):
        try:
            metadata = pd.read_csv(metadata_file)
            print(f"Metadata loaded successfully from: {metadata_file}")
            # Prepare for reference timepoint identification
            metadata_roi_col = config['experiment_analysis']['metadata_roi_col']
            timepoint_col = config['experiment_analysis']['timepoint_col']
            if metadata_roi_col not in metadata.columns:
                 raise ValueError(f"Metadata ROI column '{metadata_roi_col}' not found in {metadata_file}")
            if timepoint_col not in metadata.columns:
                 raise ValueError(f"Metadata timepoint column '{timepoint_col}' not found in {metadata_file}")

            # Create a map from filename base (ROI string) to metadata row for faster lookup
            metadata[metadata_roi_col] = metadata[metadata_roi_col].astype(str) # Ensure consistent type
            
            # --- Create metadata_map using the extracted ROI string ---
            temp_metadata_map = {}
            for _, row in metadata.iterrows():
                 roi_key_in_meta = row[metadata_roi_col] 
                 # Assume the metadata ROI col directly contains the key like 'ROI_D1_M1_01_9'
                 temp_metadata_map[roi_key_in_meta] = row.to_dict() 
            metadata_map = temp_metadata_map
            print(f"   Created metadata map with {len(metadata_map)} entries.")
            # --- End map creation ---

            # Identify the first timepoint value
            timepoint_values = metadata[timepoint_col].unique()
            try:
                # Attempt numeric sort first
                first_timepoint = sorted([v for v in timepoint_values if pd.notna(v)], key=lambda x: float(x))[0]
            except (ValueError, TypeError):
                # Fallback to string sort if numeric fails
                first_timepoint = sorted([str(v) for v in timepoint_values if pd.notna(v)])[0]
            print(f"Identified first timepoint: {first_timepoint} (from column '{timepoint_col}')")

        except FileNotFoundError:
            print(f"WARNING: Metadata file specified but not found: {metadata_file}. Cannot use metadata features.")
            metadata = None
            metadata_map = {}
            first_timepoint = None
        except KeyError as e:
             print(f"WARNING: Missing expected key in config for metadata: {e}. Cannot use metadata features.")
             metadata = None
             metadata_map = {}
             first_timepoint = None
        except ValueError as e:
            print(f"WARNING: Error processing metadata: {e}. Cannot use metadata features.")
            metadata = None
            metadata_map = {}
            first_timepoint = None
        except Exception as e:
            print(f"ERROR loading or processing metadata from {metadata_file}: {e}")
            metadata = None # Treat as if no metadata available
            metadata_map = {}
            first_timepoint = None
    else:
        print("WARNING: No metadata file specified or found. Cannot apply metadata-based ordering.")
        metadata = None
        metadata_map = {}
        first_timepoint = None

    # --- 3. Determine Parallel Jobs ---
    try:
        parallel_jobs_config = config['processing']['parallel_jobs']
        cpu_count = multiprocessing.cpu_count()
        if isinstance(parallel_jobs_config, int):
            if parallel_jobs_config == -1:
                n_jobs = cpu_count
            elif parallel_jobs_config <= -2:
                n_jobs = max(1, cpu_count + parallel_jobs_config + 1) # e.g., -2 means cpu_count - 1
            elif parallel_jobs_config > 0:
                n_jobs = min(parallel_jobs_config, cpu_count)
            else: # 0 or invalid
                n_jobs = 1
        else:
             print("Warning: Invalid 'parallel_jobs' value in config. Defaulting to 1 core.")
             n_jobs = 1
    except KeyError:
         print("Warning: 'parallel_jobs' not found in config processing section. Defaulting to 1 core.")
         n_jobs = 1
    except Exception as e:
         print(f"Warning: Error determining parallel jobs from config: {e}. Defaulting to 1 core.")
         n_jobs = 1

    print(f"\nStarting parallel processing using {n_jobs} cores...")

    # --- 3b. Calculate Reference Channel Order (if metadata available) ---
    reference_channel_order = None
    if metadata is not None and first_timepoint is not None:
        print(f"\n--- Preparing reference channel order based on timepoint: {first_timepoint} ---")

        # 1) Gather *all* ROIs at the first timepoint (using the same logic as before)
        first_tp_files: List[str] = []
        for fp in imc_files:
            rk = get_roi_string_from_path(fp)
            if not rk: # Skip if ROI string couldn't be extracted
                continue

            # Implement partial string matching: check if rk is *in* any metadata map key
            found_key = None
            md_entry = None
            for map_key, metadata_entry in metadata_map.items():
                # Check if the extracted roi_string (rk) is a substring of the metadata key (map_key)
                if rk in map_key:
                    md_entry = metadata_entry
                    found_key = map_key
                    # print(f"   DEBUG: Partial match found: ROI string '{rk}' in metadata key '{map_key}'") # Optional debug
                    break # Found the first match, assume it's the correct one

            # Check if a partial match was found and if the timepoint matches
            if found_key and md_entry:
                if str(md_entry.get(timepoint_col)) == str(first_timepoint):
                    first_tp_files.append(fp)
                    # print(f"   DEBUG: Added {fp} to first_tp_files (Timepoint match)") # Optional debug
                # else: # Optional debug
                    # print(f"   DEBUG: Partial match found for {rk}, but timepoint mismatch ({md_entry.get(timepoint_col)} != {first_timepoint})")
            # else: # Optional debug
                # print(f"   DEBUG: No partial match found for ROI string '{rk}' in metadata keys.")

        if not first_tp_files:
            print(f"WARNING: No ROIs found for first timepoint {first_timepoint}. Cannot generate reference order.")
        else:
            print(f"Found {len(first_tp_files)} candidate ROIs at timepoint {first_timepoint}.")

            # 2) Calculate correlation matrix for each first-timepoint ROI
            correlation_matrices = []
            all_channels_set = set()
            failed_loads = 0

            for i, ref_path in enumerate(first_tp_files):
                roi_key = get_roi_string_from_path(ref_path)
                print(f"  Processing reference ROI {i+1}/{len(first_tp_files)}: {roi_key}")

                # Load data for this ROI
                _, _, raw_df, roi_chs = load_and_validate_roi_data(
                    file_path=ref_path,
                    master_protein_channels=config['data']['master_protein_channels'],
                    base_output_dir=config['paths']['output_dir'], # Temporary output dir not strictly needed here
                    metadata_cols=config['data']['metadata_cols']
                )
                if raw_df is None or roi_chs is None:
                    print(f"    WARNING: Failed to load/validate data for {roi_key}. Skipping.")
                    failed_loads += 1
                    continue

                # Calculate cofactors (output dir not critical here, just need the dict)
                temp_roi_output_dir = os.path.join(output_dir, roi_key) # Need a dir for cofactor calc
                os.makedirs(temp_roi_output_dir, exist_ok=True) # Ensure it exists
                roi_cofactors = calculate_asinh_cofactors_for_roi(
                    roi_df=raw_df,
                    channels_to_process=roi_chs,
                    default_cofactor=config['data']['default_arcsinh_cofactor'],
                    output_dir=temp_roi_output_dir,
                    roi_string=roi_key
                )

                # Apply scaling
                scaled_df, _ = apply_per_channel_arcsinh_and_scale(
                    data_df=raw_df,
                    channels=roi_chs,
                    cofactors_map=roi_cofactors,
                    default_cofactor=config['data']['default_arcsinh_cofactor']
                )
                if scaled_df.empty:
                    print(f"    WARNING: Scaling failed for {roi_key}. Skipping.")
                    failed_loads += 1
                    continue

                # Get channels present in *this* scaled df
                current_roi_channels = [ch for ch in config['data']['master_protein_channels'] if ch in scaled_df.columns]
                if not current_roi_channels:
                    print(f"    WARNING: No master protein channels in scaled data for {roi_key}. Skipping.")
                    failed_loads += 1
                    continue

                # Calculate correlation matrix for channels present in this ROI
                corr_mat = scaled_df[current_roi_channels].corr(method='spearman')
                correlation_matrices.append(corr_mat)
                all_channels_set.update(current_roi_channels)
                gc.collect() # Clean up memory

            print(f"  Finished processing reference ROIs. {len(correlation_matrices)} successful, {failed_loads} failed/skipped.")

            # 3) Compute average correlation matrix and cluster
            if not correlation_matrices:
                print("ERROR: No correlation matrices were generated from first-timepoint ROIs.")
                reference_channel_order = None # Indicate failure
            else:
                consensus_channels = sorted(list(all_channels_set))
                print(f"  Found {len(consensus_channels)} unique channels across reference ROIs.")

                # Reindex all matrices to the full set of channels
                reindexed_matrices = [
                    mat.reindex(index=consensus_channels, columns=consensus_channels)
                    for mat in correlation_matrices
                ]

                # Stack matrices and compute mean, ignoring NaNs
                stacked_matrices = np.stack([mat.to_numpy() for mat in reindexed_matrices], axis=0)
                average_corr_matrix_np = np.nanmean(stacked_matrices, axis=0)

                # Convert back to DataFrame
                average_corr_matrix_df = pd.DataFrame(average_corr_matrix_np, index=consensus_channels, columns=consensus_channels)

                # Handle potential NaNs (e.g., fill with 0 - assumes uncorrelated)
                average_corr_matrix_df = average_corr_matrix_df.fillna(0)
                # Optional: Check for remaining NaNs after fillna
                if average_corr_matrix_df.isnull().values.any():
                     print("  WARNING: NaNs remain in average correlation matrix after fillna(0). Check input data.")
                     # Decide how to handle this - maybe use alphabetical order as fallback?

                print("\n  --- Clustering Average Correlation Matrix ---")
                try:
                    # Perform hierarchical clustering (Ward method)
                    # Use pdist for distance calculation (1 - correlation could also be used)
                    linkage_matrix = sch.linkage(sch.distance.pdist(average_corr_matrix_df.values), method='ward')

                    # Get the order from the linkage
                    dendrogram = sch.dendrogram(linkage_matrix, no_plot=True)
                    ordered_indices = dendrogram['leaves']
                    reference_channel_order = [average_corr_matrix_df.columns[i] for i in ordered_indices]

                    print(f"  >>> Consensus Reference Channel Order ({len(reference_channel_order)} channels) computed from average correlation:")
                    print("     ", reference_channel_order[:10], "..." if len(reference_channel_order) > 10 else "")

                    # --- Save Linkage Matrix and Average Correlation Matrix ---
                    linkage_save_path = os.path.join(output_dir, "reference_channel_linkage.npy")
                    avg_corr_save_path = os.path.join(output_dir, "reference_average_correlation.csv")
                    try:
                        np.save(linkage_save_path, linkage_matrix)
                        print(f"  Reference linkage matrix saved to: {os.path.basename(linkage_save_path)}")
                    except Exception as link_e:
                        print(f"  Warning: Could not save reference linkage matrix to {linkage_save_path}: {link_e}")
                    try:
                        average_corr_matrix_df.to_csv(avg_corr_save_path)
                        print(f"  Reference average correlation matrix saved to: {os.path.basename(avg_corr_save_path)}")
                    except Exception as corr_e:
                        print(f"  Warning: Could not save reference average correlation matrix to {avg_corr_save_path}: {corr_e}")
                    # --- End Save ---

                except ImportError:
                    print("  ERROR: Need scipy (scipy.cluster.hierarchy) to perform clustering on the average matrix.")
                    print("  Install with: pip install scipy")
                    reference_channel_order = consensus_channels # Fallback to alphabetical
                except Exception as e:
                    print(f"  ERROR during clustering of average matrix: {e}")
                    traceback.print_exc()
                    reference_channel_order = consensus_channels # Fallback

                # Save the final reference order (optional, but potentially useful)
                if reference_channel_order:
                     ref_order_path = os.path.join(output_dir, "reference_channel_order.json")
                     try:
                         with open(ref_order_path, 'w') as f:
                             json.dump(reference_channel_order, f, indent=4)
                         print(f"  Reference channel order saved to: {ref_order_path}")
                     except Exception as json_e:
                         print(f"  Warning: Could not save reference channel order to {ref_order_path}: {json_e}")

    else:
         print("\nSkipping reference channel order calculation (no metadata or first timepoint).")


    # --- 4. Run Parallel Processing ---
    print(f"\n--- Starting main parallel analysis for {len(imc_files)} ROIs ---")
    # Check if metadata_map exists before using it in the loop
    if not metadata_map:
        print("WARNING: metadata_map is empty. Proceeding without passing specific ROI metadata to analyze_roi.")
        
    analysis_results = Parallel(n_jobs=n_jobs, verbose=10)(
        # Pass the loaded config, reference order, and specific ROI metadata to each worker
        delayed(analyze_roi)(
            i,
            file_path,
            len(imc_files),
            config,
            # Safely get metadata using the extracted ROI string
            roi_metadata=metadata_map.get(get_roi_string_from_path(file_path)), 
            reference_channel_order=reference_channel_order, # Pass the calculated order
            first_timepoint_value=first_timepoint # Pass the value for comparison
            )
        for i, file_path in enumerate(imc_files) # Process all files
    )

    # --- 5. Aggregate Results ---
    # Filter results based on the return value of analyze_roi
    # Successful calls return a tuple (roi_string, ordered_channels)
    # Failed calls return None
    successful_results = [r for r in analysis_results if isinstance(r, tuple) and len(r) == 2 and r[0] is not None]
    successful_rois = [r[0] for r in successful_results] # Extract just the ROI strings for the count
    failed_rois_count = len(imc_files) - len(successful_rois)

    print(f"\n--- Pipeline Summary ---")
    print(f"Successfully completed processing for {len(successful_rois)} ROIs (across all resolutions).")
    if failed_rois_count > 0:
        print(f"Failed to process or fully complete {failed_rois_count} ROIs (check logs above for details).")

    total_pipeline_time = time.time() - start_pipeline_time
    print(f"Total pipeline execution time: {total_pipeline_time:.2f} seconds ({total_pipeline_time/60:.2f} minutes).")
    print("\n================ Completed processing all files. ================")

