import yaml
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # No longer needed here
import math
# from scipy import stats # Moved to utils
# from matplotlib.patches import Patch # Moved to viz
# from matplotlib.gridspec import GridSpec # Moved to viz
# import matplotlib.patches as mpatches # Moved to viz
import os
import glob
import re
import time
import traceback
import json
import multiprocessing
import sys # For exit
from joblib import Parallel, delayed
# from pacmap import LocalMAP # Check if needed later
# from scipy.spatial import KDTree # Handled in modules
# import seaborn as sns # Moved to viz
# import igraph as ig # Handled in modules
# import leidenalg as la # Handled in modules
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
    calculate_differential_expression
)
# Import our visualization functions
from src.roi_pipeline.pixel_visualization import (
    plot_spatial_expression_grid,
    plot_spatial_scatter,
    plot_correlation_clustermap,
    plot_umap_scatter,
    plot_coexpression_matrix,
    plot_community_size_distribution
)

# Attempt to import UMAP, set flag
try:
    import umap # Make sure umap-learn is installed
    umap_available = True
except ImportError:
    print("Warning: umap-learn package not found. Cannot perform UMAP visualization. Install with: pip install umap-learn")
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

# Renamed: Generates plots NOT dependent on community resolution
def _generate_resolution_independent_visualizations(roi_raw_data: pd.DataFrame,
                                                  scaled_pixel_expression: pd.DataFrame,
                                                  roi_channels: List[str],
                                                  roi_cofactors: Dict[str, float],
                                                  roi_output_dir: str, # Main ROI output dir
                                                  roi_string: str,
                                                  config: Dict) -> List[str]: # Return ordered channels
    """Generates plots that do not depend on Leiden resolution."""
    print("\nGenerating resolution-independent visualizations...")
    start_time_viz = time.time()
    cfg_processing = config['processing']
    cfg_viz = cfg_processing['visualization']
    ordered_channels = roi_channels # Default to original order

    # --- Spatial Expression Grid (Asinh Scaled) ---
    print("   Generating spatial expression grid (scaled)...")
    grid_path = os.path.join(roi_output_dir, f"spatial_expression_grid_{roi_string}.png")
    try:
        plot_spatial_expression_grid(
            pixel_data=roi_raw_data.join(scaled_pixel_expression), # Combine coords and scaled data
            channels=roi_channels,
            cofactors=roi_cofactors,
            title=f'Spatial Expression (Asinh Scaled) - ROI: {roi_string}',
            output_path=grid_path,
            config=config,
            plot_dpi=cfg_processing['plot_dpi']
        )
    except Exception as e:
        print(f"   WARNING: Failed to generate spatial grid plot: {e}")

    # --- Pixel-Level Correlation Clustermap (Use asinh scaled data) ---
    print("\n   Generating pixel-level correlation clustermap...")
    pixel_corr_matrix_path = os.path.join(roi_output_dir, f"pixel_channel_correlation_heatmap_spearman_{roi_string}.svg") # Changed extension
    try:
        pixel_correlation_matrix = scaled_pixel_expression[roi_channels].corr(method='spearman')
        # Call the function, capturing the returned ordered channel list
        ordered_channels = plot_correlation_clustermap(
            correlation_matrix=pixel_correlation_matrix,
            channels=roi_channels,
            title=f'Pixel Channel Corr (Spearman, Asinh Scaled) - ROI: {roi_string}',
            output_path=pixel_corr_matrix_path,
            plot_dpi=cfg_processing['plot_dpi']
        )
        if ordered_channels is None:
             print("   WARNING: Clustermap did not return an order, using original channel order.")
             ordered_channels = roi_channels # Fallback if None is returned

    except Exception as e:
        print(f"   WARNING: Failed to generate pixel correlation map: {e}")
        ordered_channels = roi_channels # Fallback

    print(f"--- Resolution-independent visualizations finished in {time.time() - start_time_viz:.2f} seconds ---")
    return ordered_channels # Return the order

# New function: Generates plots DEPENDENT on community resolution
def _generate_resolution_dependent_visualizations(pixel_results_df: pd.DataFrame, # Resolution specific df, contains coords, community, scaled values, mapped avg values
                                               scaled_pixel_expression: pd.DataFrame, # Original scaled pixel data
                                               scaled_community_profiles: pd.DataFrame,
                                               diff_expr_profiles: Optional[pd.DataFrame],
                                               primary_channel_map: Optional[pd.Series], # Renamed
                                               roi_channels: List[str], # Original channel list
                                               ordered_channels: List[str], # Channel list ordered by pixel corr
                                               roi_cofactors: Dict[str, float],
                                               resolution_output_dir: str, # Resolution specific dir
                                               roi_string: str,
                                               resolution_param: float,
                                               config: Dict):
    """Generates plots that depend on the specific Leiden resolution."""
    print(f"\nGenerating resolution-dependent visualizations (Resolution: {resolution_param})...")
    start_time_viz = time.time()
    cfg_processing = config['processing']
    cfg_analysis = config['analysis']
    cfg_viz = cfg_processing['visualization']
    res_suffix = f"_res_{resolution_param}"

    # Community Size Distribution (new)
    try:
        community_sizes = pixel_results_df['community'].value_counts()
        size_dist_path = os.path.join(resolution_output_dir, f"community_size_distribution_{roi_string}_res_{resolution_param}.png")
        plot_community_size_distribution(
            community_sizes=community_sizes,
            output_path=size_dist_path,
            title=f'Community Size Distribution - {roi_string} (Res: {resolution_param})',
            plot_dpi=cfg_processing['plot_dpi']
        )
    except Exception as e:
        print(f"   WARNING: Failed to plot community size distribution: {e}")

    # --- Community-Level Correlation Clustermap (Use original roi_channels) ---
    print("   Generating community-level correlation clustermap...")
    if not scaled_community_profiles.empty:
        try:
            community_correlation_matrix = scaled_community_profiles.corr(method='spearman')
            comm_corr_heatmap_path = os.path.join(resolution_output_dir, f"community_channel_correlation_heatmap_spearman_{roi_string}{res_suffix}.svg") # Changed extension
            # We don't necessarily want this ordered by pixel correlation, so use original roi_channels
            plot_correlation_clustermap(
                 correlation_matrix=community_correlation_matrix,
                 channels=roi_channels, # Use original list here
                 title=f'Community Corr (Spearman, Avg. Scaled) - {roi_string} (Res: {resolution_param})',
                 output_path=comm_corr_heatmap_path,
                 plot_dpi=cfg_processing['plot_dpi']
            ) # We don't need the order returned here
        except Exception as e:
            print(f"   WARNING: Failed to generate community correlation map: {e}")
    else:
        print("   Skipping community correlation analysis: Scaled community profiles empty.")

    # --- UMAP on Differential Profiles & Plot (Uses roi_channels for filtering) ---
    umap_coords = None # Define umap_coords before the block
    if diff_expr_profiles is not None and not diff_expr_profiles.empty:
        print("\n   Running UMAP and plotting communities...")
        try:
            non_protein_markers = cfg_analysis['differential_expression'].get('non_protein_markers_for_umap', [])
            protein_marker_channels_for_umap = [
                ch for ch in diff_expr_profiles.columns
                if ch in roi_channels and ch not in non_protein_markers # Filter based on original channels
            ]
            if not protein_marker_channels_for_umap:
                print("      Skipping UMAP: No protein marker channels found.")
            else:
                diff_data_for_umap = diff_expr_profiles[protein_marker_channels_for_umap].copy()
                communities_in_order = diff_expr_profiles.index.tolist()
                if diff_data_for_umap.isnull().values.any() or np.isinf(diff_data_for_umap.values).any():
                     print("      Warning: NaN/Inf values found. Replacing with 0.")
                     diff_data_for_umap = diff_data_for_umap.fillna(0).replace([np.inf, -np.inf], 0)

                n_communities = len(diff_data_for_umap)
                umap_n_neighbors = min(cfg_analysis['umap']['n_neighbors'], n_communities - 1) if n_communities > 1 else 1
                current_umap_n_components = max(2, cfg_analysis['umap']['n_components'])

                if n_communities > umap_n_neighbors and n_communities >= current_umap_n_components:
                     try:
                         # Check for GPU-accelerated UMAP
                         use_gpu_umap = check_gpu_availability(verbose=False)
                         cuml = get_rapids_lib('cuml')
                         cupy = get_rapids_lib('cupy')
                         if use_gpu_umap and cuml and cupy:
                             print("      Using GPU-accelerated UMAP (cuML).")
                             diff_gpu = cupy.asarray(diff_data_for_umap.values)
                             umap_gpu = cuml.UMAP(
                                 n_neighbors=umap_n_neighbors,
                                 min_dist=cfg_analysis['umap']['min_dist'],
                                 n_components=current_umap_n_components,
                                 metric=cfg_analysis['umap']['metric'],
                                 random_state=cfg_analysis['clustering']['seed']
                             )
                             embedding_gpu = umap_gpu.fit_transform(diff_gpu)
                             # Convert to numpy
                             embedding = embedding_gpu.get() if hasattr(embedding_gpu, 'get') else embedding_gpu
                         else:
                             if use_gpu_umap:
                                 print("      WARNING: cuML or CuPy not available for UMAP. Falling back to CPU.")
                             umap_reducer = umap.UMAP(
                                 n_neighbors=umap_n_neighbors,
                                 min_dist=cfg_analysis['umap']['min_dist'],
                                 n_components=current_umap_n_components,
                                 metric=cfg_analysis['umap']['metric'],
                                 random_state=cfg_analysis['clustering']['seed']
                             )
                             embedding = umap_reducer.fit_transform(diff_data_for_umap.values)
                         umap_component_names = [f'UMAP{i+1}' for i in range(current_umap_n_components)]
                         umap_coords = pd.DataFrame(embedding, index=communities_in_order, columns=umap_component_names)
                         umap_coords_path = os.path.join(resolution_output_dir, f"umap_coords_diff_profiles_{roi_string}{res_suffix}.csv") # Keep CSV for coords
                         umap_coords.to_csv(umap_coords_path)
                         print(f"      UMAP coordinates saved to: {os.path.basename(umap_coords_path)}")

                         if primary_channel_map is not None and not primary_channel_map.empty:
                             umap_scatter_path = os.path.join(resolution_output_dir, f"umap_community_scatter_protein_markers_diff_profiles_{roi_string}{res_suffix}.svg") # Changed extension
                             plot_umap_scatter(
                                 umap_coords=umap_coords,
                                 community_top_channel_map=primary_channel_map,
                                 protein_marker_channels=protein_marker_channels_for_umap,
                                 roi_string=f"{roi_string} (Res: {resolution_param})",
                                 output_path=umap_scatter_path,
                                 plot_dpi=cfg_processing['plot_dpi']
                             )
                         else:
                             print("      Skipping UMAP scatter plot: Missing primary channel map.")

                     except Exception as umap_err:
                          print(f"      ERROR during UMAP embedding or plotting: {umap_err}")
                          umap_coords = None
                else:
                     print(f"      Skipping UMAP embedding: Not enough communities ({n_communities}) vs neighbors/components.")
        except Exception as e:
            print(f"   WARNING: Failed during UMAP step: {e}")
    elif not umap_available:
        print("\n   Skipping UMAP visualization: umap-learn package not installed.")
    else:
        print("\n   Skipping UMAP visualization: Differential profiles empty or not calculated.")

    # --- Combined Co-expression Matrix Plot (Use ordered_channels) ---
    print("\n   Generating combined scaled-pixel/avg-comm co-expression matrix...")
    if not scaled_community_profiles.empty:
        avg_value_cols_map = {}
        try:
            for channel in roi_channels:
                avg_col_name = f'{channel}_asinh_scaled_avg'
                pixel_results_df[avg_col_name] = pixel_results_df['community'].map(scaled_community_profiles[channel]).fillna(0)
                avg_value_cols_map[channel] = avg_col_name

            coexp_matrix_path = os.path.join(resolution_output_dir, f"coexpression_matrix_scaled_vs_avg_{roi_string}{res_suffix}.svg") # Changed extension
            plot_coexpression_matrix(
                scaled_pixel_expression=scaled_pixel_expression,
                pixel_results_df_with_avg=pixel_results_df,
                ordered_channels=ordered_channels, # Pass the ordered list here
                roi_string=f"{roi_string} (Res: {resolution_param})",
                config=config,
                output_path=coexp_matrix_path
            )
        except Exception as e:
             print(f"   WARNING: Failed to generate combined co-expression matrix: {e}")
    else:
        print("   Skipping combined co-expression matrix: Scaled community profiles not available.")

    print(f"--- Resolution-dependent visualizations finished in {time.time() - start_time_viz:.2f} seconds ---")


# === Main Analysis Orchestration ===

def analyze_roi(file_idx: int, file_path: str, total_files: int, config: Dict):
    """Orchestrates the analysis pipeline for a single ROI file, iterating through resolutions."""
    print(f"\n================ Analyzing ROI {file_idx+1}/{total_files}: {os.path.basename(file_path)} ================")

    cfg_paths = config['paths']
    cfg_data = config['data']
    cfg_analysis = config['analysis']
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

        # Generate independent plots and get the ordered channel list
        ordered_channels = _generate_resolution_independent_visualizations(
            roi_raw_data=roi_raw_data,
            scaled_pixel_expression=scaled_pixel_expression,
            roi_channels=roi_channels,
            roi_cofactors=roi_cofactors,
            roi_output_dir=roi_output_dir,
            roi_string=roi_string,
            config=config
        )

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

            try:
                resolution_str = f"{resolution:.3f}".rstrip('0').rstrip('.').replace('.', '_') if isinstance(resolution, float) else str(resolution)
                resolution_output_dir = os.path.join(roi_output_dir, f"resolution_{resolution_str}")
                os.makedirs(resolution_output_dir, exist_ok=True)
                print(f"\n===== Processing Resolution: {resolution} (Output: {resolution_output_dir}) =====")

                pixel_community_df, pixel_graph, community_partition = _cluster_pixels(roi_raw_data, roi_channels, scaled_pixel_expression, resolution, config)
                if pixel_community_df is None:
                    print(f"Skipping further analysis for resolution {resolution} due to clustering failure.")
                    continue

                current_pixel_results_df = roi_raw_data[['X', 'Y']].join(scaled_pixel_expression).join(pixel_community_df[['community']])
                for ch in roi_channels:
                     if ch not in current_pixel_results_df.columns and ch in roi_raw_data.columns:
                         current_pixel_results_df[ch] = roi_raw_data.loc[current_pixel_results_df.index, ch]
                scaled_results_path = os.path.join(resolution_output_dir, f"pixel_scaled_results_{roi_string}_res_{resolution}.csv")
                current_pixel_results_df.to_csv(scaled_results_path, index=True)
                print(f"   Intermediate pixel results saved to: {os.path.basename(scaled_results_path)}")

                scaled_community_profiles, diff_expr_profiles, primary_channel_map = _analyze_communities(
                    current_pixel_results_df, roi_channels, pixel_graph, resolution_output_dir, roi_string, resolution
                )
                if scaled_community_profiles is None:
                    print(f"Skipping resolution-dependent plots for resolution {resolution} due to community analysis failure.")
                    continue

                if primary_channel_map is not None and not primary_channel_map.empty:
                    current_pixel_results_df['primary_channel'] = current_pixel_results_df['community'].map(primary_channel_map).fillna('Mapping Error')
                else:
                    current_pixel_results_df['primary_channel'] = 'Not Calculated'

                _generate_resolution_dependent_visualizations(
                    pixel_results_df=current_pixel_results_df,
                    scaled_pixel_expression=scaled_pixel_expression,
                    scaled_community_profiles=scaled_community_profiles,
                    diff_expr_profiles=diff_expr_profiles,
                    primary_channel_map=primary_channel_map,
                    roi_channels=roi_channels,
                    ordered_channels=ordered_channels,
                    roi_cofactors=roi_cofactors,
                    resolution_output_dir=resolution_output_dir,
                    roi_string=roi_string,
                    resolution_param=resolution,
                    config=config
                )

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
    print("\n--- Starting IMC Pixel-wise Analysis Pipeline ---")
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

