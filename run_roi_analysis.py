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

# GPU utility
from src.roi_pipeline.gpu_utils import check_gpu_availability, get_rapids_lib

# Import our new utility functions
from src.roi_pipeline.imc_data_utils import (
    load_and_validate_roi_data,
    calculate_optimal_cofactors_for_roi,
    apply_per_channel_arcsinh_and_scale,
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

# Updated to accept resolution_param
def _cluster_pixels(roi_raw_data: pd.DataFrame, roi_channels: List[str], scaled_pixel_expression: pd.DataFrame, resolution_param: float, config: Dict) -> Tuple[Optional[pd.DataFrame], Optional[Any], Optional[Any]]: # Use Any for igraph/leidenalg types
    """Performs spatial Leiden clustering on pixel data for a specific resolution."""
    print(f"\nClustering pixels spatially (Resolution: {resolution_param})...")
    start_time = time.time()
    # Check for GPU acceleration
    use_gpu = config.get('processing', {}).get('use_gpu', False)
    if use_gpu:
        print("   GPU acceleration enabled (CUDA/RAPIDS detected).")
        # TODO: Implement GPU-accelerated KNN neighbor search and cugraph Leiden here
        print("   Note: GPU-accelerated path not yet implemented, using CPU fallback.")

    pixel_coordinates = roi_raw_data[['X', 'Y']].copy().loc[scaled_pixel_expression.index]
    pixel_community_df, pixel_graph, community_partition, exec_time = run_spatial_leiden(
        analysis_df=pixel_coordinates,
        protein_channels=roi_channels,
        scaled_expression_data_for_weights=scaled_pixel_expression.values,
        n_neighbors=config['analysis']['clustering']['n_neighbors'],
        resolution_param=resolution_param, # Use passed parameter
        seed=config['analysis']['clustering']['seed'],
        verbose=True # Keep verbose on
    )
    if pixel_community_df is None or pixel_graph is None or community_partition is None:
        print(f"   ERROR: Leiden clustering failed for resolution {resolution_param}.")
        return None, None, None
    print(f"   --- Clustering finished in {time.time() - start_time:.2f} seconds (Leiden took {exec_time:.2f}s) ---")
    return pixel_community_df, pixel_graph, community_partition

# Updated to accept resolution_output_dir
def _analyze_communities(pixel_results_df: pd.DataFrame, roi_channels: List[str], pixel_graph: Any, resolution_output_dir: str, roi_string: str, resolution_param: float) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series]]:
    """Calculates community profiles and differential expression for a specific resolution."""
    print(f"\nAnalyzing community characteristics (Resolution: {resolution_param})...")
    start_time = time.time()
    # Calculate Profiles
    scaled_community_profiles = calculate_and_save_profiles(
         results_df=pixel_results_df,
         valid_channels=roi_channels,
         roi_output_dir=resolution_output_dir, # Save profiles in resolution subdir
         roi_string=f"{roi_string}_res_{resolution_param}" # Add resolution to filename
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

def analyze_roi(file_idx: int, file_path: str, total_files: int, config: Dict):
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
    roi_cofactors = calculate_optimal_cofactors_for_roi(
        roi_df=roi_raw_data,
        channels_to_process=roi_channels,
        default_cofactor=cfg_data['default_arcsinh_cofactor'],
        output_dir=roi_output_dir,
        roi_string=roi_string
    )
    print(f"--- Cofactor calculation finished in {time.time() - start_time_cofactor:.2f} seconds ---")

    try:
        scaled_pixel_expression, used_cofactors = _preprocess_roi(roi_raw_data, roi_channels, roi_cofactors, config)
        if scaled_pixel_expression is None:
            print(f"ERROR: Preprocessing failed for ROI {roi_string}. Aborting.")
            return None

        # Save scaled pixel expression data (including coordinates)
        scaled_pixel_output_path = os.path.join(roi_output_dir, f"scaled_pixel_expression_{roi_string}.csv")
        scaled_pixel_expression_with_coords = roi_raw_data[['X', 'Y']].join(scaled_pixel_expression)
        scaled_pixel_expression_with_coords.to_csv(scaled_pixel_output_path, index=True)
        print(f"   Scaled pixel expression saved to: {os.path.basename(scaled_pixel_output_path)}")

        # Calculate pixel correlation and determine channel order
        print("\n   Calculating pixel correlation and channel order...")
        ordered_channels = roi_channels # Default order
        try:
            pixel_correlation_matrix = scaled_pixel_expression[roi_channels].corr(method='spearman')
            # Use clustermap to get order, but don't save/show plot
            clustergrid = sns.clustermap(pixel_correlation_matrix, metric="correlation", cmap="viridis", row_cluster=True, col_cluster=True)
            # Suppress plot display
            plt.close(clustergrid.fig)
            # Get the reordered index from the row dendrogram and map back to channel names
            reordered_idx = clustergrid.dendrogram_row.reordered_ind
            ordered_channels = [roi_channels[i] for i in reordered_idx]
            print(f"   Channel order determined: {ordered_channels}")

            # Save the ordered channel list
            order_path = os.path.join(roi_output_dir, f"ordered_channels_{roi_string}.json")
            with open(order_path, 'w') as f:
                json.dump(ordered_channels, f)
            print(f"   Channel order saved to: {os.path.basename(order_path)}")

        except Exception as e:
            print(f"   WARNING: Failed to calculate pixel correlation or determine channel order: {e}")
            print("   Using original channel order.")
            ordered_channels = roi_channels # Fallback to original order

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
                    current_pixel_results_df, roi_channels, pixel_graph, resolution_output_dir, roi_string, resolution
                )
                if scaled_community_profiles is None:
                    print(f"Skipping UMAP calculation for resolution {resolution} due to community analysis failure.")
                    # Still need to save the final pixel results even if profiles failed
                else:
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

                # Add primary channel mapping to the final results dataframe (if calculated)
                if primary_channel_map is not None and not primary_channel_map.empty:
                    current_pixel_results_df['primary_channel'] = current_pixel_results_df['community'].map(primary_channel_map).fillna('Mapping Error')
                else:
                    current_pixel_results_df['primary_channel'] = 'Not Calculated'

                # Add average community profiles mapped back to pixels (if calculated)
                if scaled_community_profiles is not None and not scaled_community_profiles.empty:
                    for channel in roi_channels:
                         avg_col_name = f'{channel}_asinh_scaled_avg'
                         current_pixel_results_df[avg_col_name] = current_pixel_results_df['community'].map(scaled_community_profiles[channel]).fillna(0)

                # Save the final combined results dataframe for this resolution
                final_results_save_path = os.path.join(resolution_output_dir, f"pixel_analysis_results_final_{roi_string}_res_{resolution}.csv")
                current_pixel_results_df.to_csv(final_results_save_path, index=True)
                print(f"\n   Final pixel results for resolution {resolution} saved to: {os.path.basename(final_results_save_path)}")
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
        return roi_string # Indicate overall success for the ROI

    except Exception as e: # Catch errors outside the resolution loop
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"   FATAL ERROR during analysis for ROI {roi_string} (outside resolution loop): {str(e)}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        traceback.print_exc()
        return None # Indicate failure

# --- Script Execution Entry Point ---
if __name__ == '__main__':
    # Need to import plt for seaborn clustermap even if not showing plots directly
    import matplotlib.pyplot as plt
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

    # --- 4. Run Parallel Processing ---
    analysis_results = Parallel(n_jobs=n_jobs, verbose=10)(
        # Pass the loaded config dictionary to each worker
        delayed(analyze_roi)(i, file_path, len(imc_files), config)
        for i, file_path in enumerate(imc_files) # Process all files
    )

    # --- 5. Aggregate Results ---
    successful_rois = [r for r in analysis_results if r is not None]
    failed_rois_count = len(analysis_results) - len(successful_rois)

    print(f"\n--- Pipeline Summary ---")
    print(f"Successfully completed processing for {len(successful_rois)} ROIs (across all resolutions).")
    if failed_rois_count > 0:
        print(f"Failed to process or fully complete {failed_rois_count} ROIs (check logs above for details).")

    total_pipeline_time = time.time() - start_pipeline_time
    print(f"Total pipeline execution time: {total_pipeline_time:.2f} seconds ({total_pipeline_time/60:.2f} minutes).")
    print("\n================ Completed processing all files. ================")

