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
from typing import List, Tuple, Optional, Dict, Any
import gc
import scipy.cluster.hierarchy as sch
import networkx as nx

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
    calculate_and_save_umap
)

# Attempt to import UMAP, set flag
try:
    import umap # Make sure umap-learn is installed
    umap_available = True
except ImportError:
    print("Warning: umap-learn package not found. Cannot perform UMAP analysis. Install with: pip install umap-learn")
    umap_available = False

# Attempt to import GPU memory utilities
try:
    import torch
    import torch.cuda
    gpu_utils_available = True
except ImportError:
    gpu_utils_available = False

def get_gpu_memory_limit(config: Dict) -> Optional[int]:
    """Get GPU memory limit in bytes based on available hardware and config settings."""
    if not gpu_utils_available:
        return None
        
    try:
        # Check for CUDA
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            # Use 90% of available memory by default, or config value if specified
            memory_fraction = config.get('processing', {}).get('gpu_memory_fraction', 0.9)
            return int(total_memory * memory_fraction)
            
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # For MPS, we can't directly query memory, so use a reasonable default
            # or config value if specified
            default_mps_memory = 8 * 1024 * 1024 * 1024  # 8GB default
            return config.get('processing', {}).get('mps_memory_limit', default_mps_memory)
            
        # Other GPU backends could be added here
        else:
            return None
            
    except Exception as e:
        print(f"Warning: Could not determine GPU memory limit: {e}")
        return None

# --- Configuration Loading ---
def load_config(config_path: str = "config.yaml") -> Optional[Dict]:
    """Loads the pipeline configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from: {config_path}")
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


# === Helper Functions (some might be moved or adapted for new structure) ===

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
    scaled_pixel_expression: pd.DataFrame, # This is expression data only (no X,Y)
    roi_channels_for_correlation: List[str], # Channels to actually use for correlation (subset of scaled_pixel_expression.columns)
    roi_output_dir: str,
    roi_string: str,
    config: Dict,
    fixed_channel_order: Optional[List[str]] = None # This is the key for Option B
) -> Tuple[Optional[List[str]], Optional[str], Optional[str]]: # Returns final_order, path_to_linkage, path_to_clustered_order_json
    """
    Calculates ROI-specific channel clustering and determines the final channel order.
    Uses provided roi_channels_for_correlation for the actual correlation calculation.
    Always calculates and saves the ROI-specific clustering (linkage matrix and order list).
    Determines the final order to use: the fixed_channel_order if provided and valid,
    otherwise defaults to the ROI-specific clustered order. Saves this final order list.

    Args:
        scaled_pixel_expression: DataFrame of scaled pixel data (channels as columns, NO X,Y).
        roi_channels_for_correlation: List of channels within scaled_pixel_expression to use for correlation.
        roi_output_dir: Directory to save outputs.
        roi_string: ROI identifier.
        config: Configuration dictionary.
        fixed_channel_order: Optional predefined list of channels to use as the final order.

    Returns:
        A tuple: (final_order_to_use, linkage_save_path, roi_order_save_path) or (None, None, None) on failure.
    """
    print("   Calculating ROI-specific pixel correlation for channel clustering...")
    start_time = time.time()
    roi_clustered_order = None
    roi_linkage_matrix = None
    final_order_to_use = None
    linkage_save_path = None
    roi_order_save_path = None # For pixel_channel_clustered_order_...json

    try:
        expression_cols = [ch for ch in roi_channels_for_correlation if ch in scaled_pixel_expression.columns and ch not in ['X', 'Y']]
        if not expression_cols:
            print("   ERROR: No valid expression columns found for correlation.")
            return None, None, None

        pixel_corr_matrix = scaled_pixel_expression[expression_cols].corr(method='spearman')
        pixel_corr_matrix = pixel_corr_matrix.fillna(0)
        if pixel_corr_matrix.isnull().values.any():
            print("   WARNING: NaNs remain in pixel correlation matrix after fillna(0). Clustering might be unstable.")

        print("   Performing hierarchical clustering on ROI's pixel correlation matrix...")
        if len(expression_cols) < 2:
             print("   WARNING: Cannot perform clustering with less than 2 channels. Using original expression_cols order for ROI-specific clustering.")
             roi_clustered_order = list(expression_cols) # Ensure it's a list copy
             roi_linkage_matrix = None
        else:
             try:
                 dist_matrix = sch.distance.pdist(pixel_corr_matrix.values)
                 linkage_method = config.get('analysis', {}).get('clustering', {}).get('linkage', 'ward')
                 roi_linkage_matrix = sch.linkage(dist_matrix, method=linkage_method)
                 dendrogram = sch.dendrogram(roi_linkage_matrix, no_plot=True)
                 ordered_indices = dendrogram['leaves']
                 roi_clustered_order = [pixel_corr_matrix.columns[i] for i in ordered_indices]
                 print(f"   ROI-specific clustering successful. Determined order for {len(roi_clustered_order)} channels.")

                 current_linkage_save_path = os.path.join(roi_output_dir, f"pixel_channel_linkage_{roi_string}.npy")
                 try:
                     np.save(current_linkage_save_path, roi_linkage_matrix)
                     print(f"   ROI-specific pixel channel linkage matrix saved to: {os.path.basename(current_linkage_save_path)}")
                     linkage_save_path = current_linkage_save_path # Save path on success
                 except Exception as link_save_e:
                     print(f"   ERROR saving ROI-specific pixel channel linkage matrix: {link_save_e}")

             except ValueError as ve:
                  print(f"   ERROR during ROI-specific scipy clustering (ValueError): {ve}. Using original channel order as fallback.")
                  roi_clustered_order = list(expression_cols)
                  roi_linkage_matrix = None
             except Exception as cluster_e:
                 print(f"   ERROR during ROI-specific scipy clustering: {cluster_e}. Using original channel order as fallback.")
                 traceback.print_exc()
                 roi_clustered_order = list(expression_cols)
                 roi_linkage_matrix = None

        if roi_clustered_order:
            current_roi_order_save_path = os.path.join(roi_output_dir, f"pixel_channel_clustered_order_{roi_string}.json")
            try:
                with open(current_roi_order_save_path, 'w') as f:
                    json.dump(roi_clustered_order, f, indent=4)
                print(f"   ROI-specific clustered channel list saved to: {os.path.basename(current_roi_order_save_path)}")
                roi_order_save_path = current_roi_order_save_path # Save path on success
            except Exception as json_e:
                print(f"   ERROR saving ROI-specific clustered channel list to {current_roi_order_save_path}: {json_e}")
        else:
             print("   ERROR: Failed to determine ROI-specific channel order. Cannot proceed reliably.")
             return None, linkage_save_path, None # Return None for order, but path if linkage was saved

        # This is where `fixed_channel_order` (e.g. consensus) is applied if provided
        if fixed_channel_order:
            print(f"   Fixed channel order provided ({len(fixed_channel_order)} channels). Validating against ROI data...")
            # Validate fixed order against available channels in the correlation matrix (which represents available data)
            # Use expression_cols which are the channels actually in the scaled_pixel_expression for this ROI
            missing_channels_in_roi = [ch for ch in fixed_channel_order if ch not in expression_cols]
            available_channels_in_fixed_order = [ch for ch in fixed_channel_order if ch in expression_cols]

            if missing_channels_in_roi:
                 print(f"   WARNING: Channels in fixed_channel_order not found in this ROI's scaled data: {missing_channels_in_roi}.")
            if not available_channels_in_fixed_order:
                 print("   ERROR: No valid channels remaining after applying fixed_channel_order to this ROI's data. Falling back to ROI-specific clustered order.")
                 final_order_to_use = roi_clustered_order # Fallback if fixed order is unusable for this ROI
            else:
                 final_order_to_use = available_channels_in_fixed_order # Use the validated subset
                 print(f"   Using validated fixed channel order ({len(final_order_to_use)} channels) as final order for this ROI.")
        else:
            # No fixed order provided, use the ROI-specific clustered order as the final order
            print("   No fixed channel order provided. Using ROI-specific clustered order as final order for this ROI.")
            final_order_to_use = roi_clustered_order

        if final_order_to_use:
            # This file represents the order to be ACTUALLY USED downstream for this ROI
            final_order_json_path = os.path.join(roi_output_dir, f"pixel_channel_final_order_{roi_string}.json")
            try:
                with open(final_order_json_path, 'w') as f:
                    json.dump(final_order_to_use, f, indent=4)
                print(f"   Final channel order list for downstream use saved to: {os.path.basename(final_order_json_path)}")
            except Exception as json_e:
                print(f"   ERROR saving final channel order list to {final_order_json_path}: {json_e}")
                # If this save fails, final_order_to_use is still in memory, but file is missing.
        else:
            print("   ERROR: Final channel order for downstream use could not be determined. Aborting order calculation.")
            return None, linkage_save_path, roi_order_save_path

    except Exception as e:
        print(f"   ERROR during channel order calculation: {e}")
        traceback.print_exc()
        return None, linkage_save_path, roi_order_save_path

    print(f"   --- Channel ordering finished in {time.time() - start_time:.2f} seconds ---")
    return final_order_to_use, linkage_save_path, roi_order_save_path


# --- New Phase 1 Function ---
def preprocess_single_roi(
    file_idx: int, file_path: str, total_files: int, config: Dict,
    roi_metadata: Optional[Dict] = None,
    first_timepoint_value: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """Performs ROI-level preprocessing and initial channel ordering (ROI-specific)."""
    print(f"\n================ Preprocessing ROI {file_idx+1}/{total_files}: {os.path.basename(file_path)} ================")
    cfg_paths = config['paths']
    cfg_data = config['data']

    # 1. Load and Validate Data
    print("Loading and validating data...")
    start_time_load = time.time()
    roi_string, roi_output_dir, roi_raw_data, roi_channels_from_load = load_and_validate_roi_data(
        file_path=file_path,
        all_channels=cfg_data['protein_channels'] + cfg_data['background_channels'],
        base_output_dir=cfg_paths['output_dir'],
        metadata_cols=cfg_data['metadata_cols']
    )
    if roi_raw_data is None or roi_channels_from_load is None or roi_output_dir is None:
        print(f"Skipping file {os.path.basename(file_path)} due to errors during loading or validation.")
        return None
    print(f"--- Loading finished in {time.time() - start_time_load:.2f} seconds ---")

    # 2. Calculate Optimal Cofactors
    print("\nCalculating optimal cofactors...")
    start_time_cofactor = time.time()
    roi_cofactors = calculate_asinh_cofactors_for_roi(
        roi_df=roi_raw_data,
        channels_to_process=roi_channels_from_load,
        default_cofactor=cfg_data['default_arcsinh_cofactor'],
        output_dir=roi_output_dir,
        roi_string=roi_string
    )
    print(f"--- Cofactor calculation finished in {time.time() - start_time_cofactor:.2f} seconds ---")

    # 3. Preprocess ROI (Arcsinh Transform and Scale)
    scaled_pixel_expression_no_xy, used_cofactors = _preprocess_roi(
        roi_raw_data, roi_channels_from_load, roi_cofactors, config
    )
    if scaled_pixel_expression_no_xy is None:
        print(f"ERROR: Preprocessing failed for ROI {roi_string}. Aborting preprocessing.")
        return None
    scaled_channel_columns = scaled_pixel_expression_no_xy.columns.tolist()

    # 4. Save Scaled Pixel Expression (with X, Y)
    scaled_pixel_expression_with_xy_path = None
    if 'X' in roi_raw_data.columns and 'Y' in roi_raw_data.columns:
        scaled_pixel_expression_with_xy = roi_raw_data[['X', 'Y']].join(scaled_pixel_expression_no_xy, how="inner")
        if len(scaled_pixel_expression_with_xy) != len(scaled_pixel_expression_no_xy):
            print(f"   WARNING: Row count changed after joining X,Y for {roi_string}.")
    else:
        print(f"   CRITICAL WARNING: 'X' or 'Y' missing in roi_raw_data for {roi_string}.")
        scaled_pixel_expression_with_xy = scaled_pixel_expression_no_xy.copy()
    scaled_expr_save_path = os.path.join(roi_output_dir, f"scaled_pixel_expression_{roi_string}.csv")
    try:
        scaled_pixel_expression_with_xy.to_csv(scaled_expr_save_path, index=True)
        print(f"   Scaled pixel expression (with X,Y) saved to: {os.path.basename(scaled_expr_save_path)}")
        scaled_pixel_expression_with_xy_path = scaled_expr_save_path
    except Exception as save_e:
        print(f"   ERROR saving scaled pixel expression with X,Y: {save_e}")
        return None

    # 5. Determine ROI-specific Channel Order (fixed_channel_order is None here for initial run)
    # This will generate pixel_channel_final_order_{roi_string}.json based on ROI's own data.
    # It also saves pixel_channel_linkage_...npy and pixel_channel_clustered_order_...json
    roi_specific_final_order, roi_linkage_path, roi_clustered_order_path = _calculate_and_save_ordered_channels(
        scaled_pixel_expression=scaled_pixel_expression_no_xy,
        roi_channels_for_correlation=scaled_channel_columns,
        roi_output_dir=roi_output_dir,
        roi_string=roi_string,
        config=config,
        fixed_channel_order=None # OPTION B: Initially, no fixed order is imposed.
    )
    if roi_specific_final_order is None:
        print(f"ERROR: Failed to determine initial ROI-specific channel order for ROI {roi_string}. Aborting preprocessing.")
        return None

    # 6. Calculate and Save Pixel Correlation Matrix (using the ROI-specific final order)
    # This matrix is based on the ROI's own clustering initially.
    pixel_corr_save_path = None
    print("   Calculating and saving ROI-specific pixel correlation matrix (using its own clustered order)...")
    try:
        valid_ordered_channels = [ch for ch in roi_specific_final_order if ch in scaled_pixel_expression_no_xy.columns]
        if not valid_ordered_channels:
            print(f"   ERROR: No valid channels from ROI-specific order in scaled data. Skipping pixel correlation save.")
        else:
            pixel_corr_matrix = scaled_pixel_expression_no_xy[valid_ordered_channels].corr(method='spearman')
            current_pixel_corr_save_path = os.path.join(roi_output_dir, f"pixel_channel_correlation_{roi_string}.csv")
            pixel_corr_matrix.to_csv(current_pixel_corr_save_path)
            print(f"   ROI-specific pixel correlation matrix saved to: {os.path.basename(current_pixel_corr_save_path)}")
            pixel_corr_save_path = current_pixel_corr_save_path # Save path for return
    except Exception as corr_e:
        print(f"   ERROR calculating or saving ROI-specific pixel correlation matrix: {corr_e}")

    # Determine if this ROI is a first timepoint ROI
    is_first_timepoint_roi = False
    if roi_metadata and first_timepoint_value is not None and config['experiment_analysis']['timepoint_col'] in roi_metadata:
        timepoint_col = config['experiment_analysis']['timepoint_col']
        current_timepoint_val = roi_metadata.get(timepoint_col)

        # Robust comparison for timepoints
        try:
            # Attempt numeric comparison first
            is_first_timepoint_roi = float(current_timepoint_val) == float(first_timepoint_value)
        except (ValueError, TypeError):
            # Fallback to string comparison if numeric conversion fails
            is_first_timepoint_roi = str(current_timepoint_val).strip() == str(first_timepoint_value).strip()

        if is_first_timepoint_roi:
            print(f"   ROI {roi_string} identified as first timepoint (Value: {current_timepoint_val}, Ref: {first_timepoint_value}).")
        # else: print(f"   ROI {roi_string} is not first timepoint ({current_timepoint_val} vs {first_timepoint_value}).")
    # else: print(f"   ROI {roi_string}: timepoint status not determined (no metadata/first_timepoint_value).")


    print(f"--- Preprocessing and ROI-level setup for {roi_string} finished successfully ---")
    return {
        'roi_string': roi_string,
        'roi_output_dir': roi_output_dir,
        'roi_channels_from_load': roi_channels_from_load,
        'scaled_pixel_expression_with_xy_path': scaled_pixel_expression_with_xy_path,
        'scaled_channel_columns': scaled_channel_columns,
        'ordered_channels_for_downstream': list(roi_specific_final_order), # Initially ROI-specific, may be updated in Phase 1.5
        'pixel_correlation_matrix_path': pixel_corr_save_path, # Path to ROI's own correlation matrix
        'pixel_channel_linkage_path': roi_linkage_path, # Path to ROI's own linkage
        'pixel_channel_clustered_order_path': roi_clustered_order_path, # Path to ROI's own clustered order JSON
        'is_first_timepoint_roi': is_first_timepoint_roi,
        'config': config, # Pass full config
        'used_cofactors': used_cofactors,
        'roi_raw_data_path': file_path, 
        'scaled_pixel_expression_no_xy_df_for_reorder': scaled_pixel_expression_no_xy.copy() # Keep a copy for potential re-ordering
    }


# --- Function to re-calculate and save final order using a consensus (Phase 1.5) ---
def recalculate_final_order_with_consensus(
    roi_info: Dict[str, Any],
    consensus_channel_order: List[str],
    config: Dict # already in roi_info, but explicit for clarity
) -> Optional[List[str]]:
    """Recalculates the pixel_channel_final_order.json for an ROI using a consensus order."""
    roi_string = roi_info['roi_string']
    roi_output_dir = roi_info['roi_output_dir']
    scaled_pixel_expression_no_xy_df = roi_info['scaled_pixel_expression_no_xy_df_for_reorder']
    roi_channels_for_correlation = roi_info['scaled_channel_columns'] # These are the channels present in the df

    print(f"   Phase 1.5: Re-evaluating final channel order for {roi_string} using consensus ({len(consensus_channel_order)} channels).")

    # Call _calculate_and_save_ordered_channels, this time with the consensus as fixed_channel_order
    # It will save a new pixel_channel_final_order_{roi_string}.json
    # The ROI-specific linkage and clustered_order files remain as they were (based on ROI's own data)
    updated_final_order, _, _ = _calculate_and_save_ordered_channels(
        scaled_pixel_expression=scaled_pixel_expression_no_xy_df,
        roi_channels_for_correlation=roi_channels_for_correlation,
        roi_output_dir=roi_output_dir,
        roi_string=roi_string,
        config=config,
        fixed_channel_order=consensus_channel_order
    )

    if updated_final_order:
        print(f"   Phase 1.5: Successfully updated final channel order for {roi_string}.")
        return list(updated_final_order)
    else:
        print(f"   Phase 1.5: FAILED to update final channel order for {roi_string}. Will use its original ROI-specific order.")
        return list(roi_info['ordered_channels_for_downstream']) # Fallback to its original ROI-specific order


# Updated to accept resolution_param
def _cluster_pixels(
    pixel_coords_df: pd.DataFrame, # DataFrame with 'X', 'Y' columns
    scaled_expression_df_values: np.ndarray, # Numpy array of scaled expression values (no X,Y, no headers)
    protein_channels_for_weighting: List[str], # List of channels corresponding to scaled_expression_df_values columns
    resolution_param: float,
    config: Dict,
    # --- START: Added for background-derived thresholding ---
    background_channel_data_for_thresholding: Optional[np.ndarray] = None,
    background_channel_names_for_thresholding: Optional[List[str]] = None
    # --- END: Added for background-derived thresholding ---
) -> Tuple[Optional[pd.DataFrame], Optional[Any], Optional[Any], Optional[float]]:
    """Performs spatial Leiden clustering on pixel data for a specific resolution."""
    print(f"\nClustering pixels spatially (Resolution: {resolution_param})...")
    start_time = time.time()
    # Get the use_gpu setting from the main config file
    use_gpu_config_from_yaml = config.get('processing', {}).get('use_gpu', False)
    if use_gpu_config_from_yaml:
        print(f"   Config 'processing.use_gpu' is True. Will request GPU paths in run_spatial_leiden if available.")
    else:
        print(f"   Config 'processing.use_gpu' is False. Will request CPU paths in run_spatial_leiden.")

    # Get GPU memory limit if available
    gpu_memory_limit_total_config = get_gpu_memory_limit(config)
    if gpu_memory_limit_total_config:
        # This is the full memory limit intended for a single GPU process
        print(f"   Configured GPU memory limit for a single process: {gpu_memory_limit_total_config / (1024**3):.2f} GB")

    pixel_community_df, pixel_graph, community_partition, exec_time = run_spatial_leiden(
        analysis_df=pixel_coords_df, 
        protein_channels=protein_channels_for_weighting, 
        scaled_expression_data_for_weights=scaled_expression_df_values, 
        n_neighbors=config['analysis']['clustering']['n_neighbors'],
        resolution_param=resolution_param,
        seed=config['analysis']['clustering']['seed'],
        verbose=True,
        use_gpu_from_config=use_gpu_config_from_yaml,
        gpu_memory_limit=gpu_memory_limit_total_config,
        config=config,
        # --- START: Pass background data to run_spatial_leiden ---
        background_channel_data=background_channel_data_for_thresholding,
        background_channel_names=background_channel_names_for_thresholding
        # --- END: Pass background data to run_spatial_leiden ---
    )
    if pixel_community_df is None or pixel_graph is None or community_partition is None:
        print(f"   ERROR: Leiden clustering failed for resolution {resolution_param}.")
        return None, None, None, None
    print(f"   --- Clustering finished in {time.time() - start_time:.2f} seconds (Leiden took {exec_time:.2f}s) ---")
    return pixel_community_df, pixel_graph, community_partition, exec_time


# Updated to accept resolution_output_dir
def _analyze_communities(
    current_pixel_results_df: pd.DataFrame, 
    channels_for_profiles: List[str], 
    pixel_graph: Any,
    resolution_output_dir: str,
    roi_string: str,
    resolution_param: Any 
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series]]:
    """Calculates community profiles and differential expression for a specific resolution."""
    print(f"\nAnalyzing community characteristics (Resolution: {resolution_param})...")
    start_time = time.time()
    res_str = f"{resolution_param:.3f}".rstrip('0').rstrip('.') if isinstance(resolution_param, float) else str(resolution_param)

    scaled_community_profiles = calculate_and_save_profiles(
         results_df=current_pixel_results_df, 
         valid_channels=channels_for_profiles, 
         roi_output_dir=resolution_output_dir,
         roi_string=f"{roi_string}_res_{res_str}",
         ordered_channels=channels_for_profiles 
    )
    if scaled_community_profiles is None or scaled_community_profiles.empty:
        print(f"   Skipping differential expression for resolution {resolution_param}: Profile calculation failed or no communities found.")
        return None, None, None

    diff_expr_profiles, primary_channel_map = calculate_differential_expression(
        results_df=current_pixel_results_df, 
        community_profiles=scaled_community_profiles,
        graph=pixel_graph,
        valid_channels=channels_for_profiles 
    )

    if diff_expr_profiles is not None and not diff_expr_profiles.empty:
        diff_profiles_path = os.path.join(resolution_output_dir, f"community_diff_profiles_{roi_string}_res_{res_str}.csv")
        diff_expr_profiles.to_csv(diff_profiles_path)
        print(f"   Differential profiles saved to: {os.path.basename(diff_profiles_path)}")
    if primary_channel_map is not None and not primary_channel_map.empty:
        top_channel_path = os.path.join(resolution_output_dir, f"community_primary_channels_{roi_string}_res_{res_str}.csv")
        primary_channel_map.to_csv(top_channel_path, header=True)
        print(f"   Primary channel map saved to: {os.path.basename(top_channel_path)}")
    else:
         print(f"   Primary channel mapping skipped for resolution {resolution_param} as DiffEx failed or produced no results.")

    print(f"   --- Community analysis finished in {time.time() - start_time:.2f} seconds ---")
    return scaled_community_profiles, diff_expr_profiles, primary_channel_map


# --- New Helper for Spatial Region Creation ---
def _create_spatial_regions_from_adjacencies(
    community_adjacencies: Dict[str, List[str]],
    pixel_results_df: pd.DataFrame,
    min_pixels_per_region: int = 500,
    adaptive_threshold_config: Optional[Dict] = None
) -> Dict[str, List[str]]:
    """
    Creates spatial regions by grouping adjacent communities using connected components.
    
    Args:
        community_adjacencies: Dictionary mapping community_id -> list of adjacent community_ids
        pixel_results_df: DataFrame with pixel data including 'community' column
        min_pixels_per_region: Fixed minimum number of pixels (used if adaptive_threshold_config is None)
        adaptive_threshold_config: Dict with adaptive threshold settings:
            - 'method': 'proportion' or 'percentile' or 'median_multiple'
            - 'value': proportion of total pixels, percentile value, or multiple of median
            
    Returns:
        Dictionary mapping region_id -> list of community_ids in that region
    """
    print(f"    Creating spatial regions from {len(community_adjacencies)} communities...")
    
    # Get pixel counts per community
    community_pixel_counts = pixel_results_df['community'].value_counts().to_dict()
    total_pixels = len(pixel_results_df)
    
    # Determine adaptive threshold if configured
    if adaptive_threshold_config:
        method = adaptive_threshold_config.get('method', 'proportion')
        value = adaptive_threshold_config.get('value', 0.01)  # Default 1% of total pixels
        
        if method == 'proportion':
            # Use proportion of total pixels in ROI
            adaptive_min_pixels = max(50, int(total_pixels * value))  # At least 50 pixels
            print(f"    Using proportional threshold: {value:.3f} of {total_pixels} pixels = {adaptive_min_pixels} pixels")
        
        elif method == 'percentile':
            # Use percentile of community sizes
            community_sizes = list(community_pixel_counts.values())
            adaptive_min_pixels = max(50, int(np.percentile(community_sizes, value)))
            print(f"    Using {value}th percentile of community sizes: {adaptive_min_pixels} pixels")
        
        elif method == 'median_multiple':
            # Use multiple of median community size
            community_sizes = list(community_pixel_counts.values())
            median_size = np.median(community_sizes)
            adaptive_min_pixels = max(50, int(median_size * value))
            print(f"    Using {value}x median community size: {median_size:.1f} × {value} = {adaptive_min_pixels} pixels")
        
        else:
            print(f"    Warning: Unknown adaptive method '{method}', falling back to fixed threshold")
            adaptive_min_pixels = min_pixels_per_region
            
        min_pixels_threshold = adaptive_min_pixels
    else:
        min_pixels_threshold = min_pixels_per_region
        print(f"    Using fixed threshold: {min_pixels_threshold} pixels")
    
    # Create a graph of community adjacencies
    G = nx.Graph()
    
    # Add all communities as nodes
    for comm_id in community_adjacencies.keys():
        G.add_node(comm_id)
    
    # Add edges between adjacent communities
    for comm_id, neighbors in community_adjacencies.items():
        for neighbor_id in neighbors:
            if neighbor_id in community_adjacencies:  # Ensure neighbor exists
                G.add_edge(comm_id, neighbor_id)
    
    # Find connected components (groups of connected communities)
    connected_components = list(nx.connected_components(G))
    print(f"    Found {len(connected_components)} connected components.")
    
    # Create spatial regions, merging small components if needed
    spatial_regions = {}
    region_id = 0
    total_pixels_in_regions = 0
    skipped_pixels = 0
    
    for component in connected_components:
        component_communities = list(component)
        
        # Calculate total pixels in this component
        total_pixels_component = sum(community_pixel_counts.get(int(comm), 0) for comm in component_communities)
        
        if total_pixels_component >= min_pixels_threshold:
            # Component is large enough to be its own region
            spatial_regions[f"region_{region_id}"] = component_communities
            total_pixels_in_regions += total_pixels_component
            print(f"      Region region_{region_id}: {len(component_communities)} communities, {total_pixels_component} pixels")
            region_id += 1
        else:
            # Component is too small
            skipped_pixels += total_pixels_component
            print(f"      Skipping small component: {len(component_communities)} communities, {total_pixels_component} pixels (< {min_pixels_threshold})")
    
    coverage_pct = (total_pixels_in_regions / total_pixels) * 100 if total_pixels > 0 else 0
    print(f"    Created {len(spatial_regions)} valid spatial regions.")
    print(f"    Coverage: {total_pixels_in_regions}/{total_pixels} pixels ({coverage_pct:.1f}%), {skipped_pixels} pixels in small components")
    
    return spatial_regions


# --- New Phase 2 Function ---
def analyze_single_resolution_for_roi(
    roi_preprocessing_info: Dict[str, Any], 
    resolution_param: Any, 
):
    """Analyzes a single resolution for a preprocessed ROI."""
    roi_string = roi_preprocessing_info['roi_string']
    roi_output_dir = roi_preprocessing_info['roi_output_dir']
    scaled_pixel_expression_with_xy_path = roi_preprocessing_info['scaled_pixel_expression_with_xy_path']
    # scaled_channel_columns are all channels available in the scaled data
    all_available_scaled_channels = roi_preprocessing_info['scaled_channel_columns']
    # ordered_channels_for_downstream is the final list (ROI-specific or consensus) to be used for profiles etc.
    channels_for_profile_analysis = roi_preprocessing_info['ordered_channels_for_downstream']
    config = roi_preprocessing_info['config']

    resolution_str_clean = f"{resolution_param:.3f}".rstrip('0').rstrip('.').replace('.', '_') if isinstance(resolution_param, float) else str(resolution_param)
    # This is the main output directory for this specific resolution, parent to context-specific dirs
    resolution_parent_output_dir = os.path.join(roi_output_dir, f"resolution_{resolution_str_clean}")
    os.makedirs(resolution_parent_output_dir, exist_ok=True)

    print(f"\n===== Base Processing for ROI: {roi_string}, Resolution: {resolution_param} (Output Root: {resolution_parent_output_dir}) =====")

    try:
        base_scaled_pixel_expression_df = pd.read_csv(scaled_pixel_expression_with_xy_path, index_col=0)
        if base_scaled_pixel_expression_df.empty or 'X' not in base_scaled_pixel_expression_df.columns or 'Y' not in base_scaled_pixel_expression_df.columns:
            print(f"   ERROR: Loaded base scaled pixel expression data is invalid for {roi_string}. Skipping resolution {resolution_param}.")
            return False
    except Exception as e_load:
        print(f"   ERROR loading base_scaled_pixel_expression_df for {roi_string}: {e_load}. Skipping resolution {resolution_param}.")
        return False

    # Initialize the DataFrame that will store all pixel annotations including multiple community assignments
    final_pixel_annotations_df = base_scaled_pixel_expression_df.copy()
    
    # --- START: Prepare for background-derived thresholding ---
    clustering_config = config.get('analysis', {}).get('clustering', {})
    apply_thresholding = clustering_config.get('apply_similarity_thresholding', False)
    threshold_type = clustering_config.get('similarity_threshold_type', 'percentile')
    
    background_data_for_leiden: Optional[np.ndarray] = None
    background_names_for_leiden: Optional[List[str]] = None

    if apply_thresholding and threshold_type == 'background_derived':
        print(f"   Background-derived k-NN thresholding is active for ROI {roi_string}.")
        config_background_channels = config.get('data', {}).get('background_channels', [])
        user_spec_bg_channels_for_thresh = clustering_config.get('background_channels_for_thresholding', [])

        if not user_spec_bg_channels_for_thresh: # If list is empty, use all from data.background_channels
            background_names_for_leiden = [ch for ch in config_background_channels if ch in base_scaled_pixel_expression_df.columns]
            print(f"      Using all available general background channels for thresholding: {background_names_for_leiden}")
        else: # User specified a list
            background_names_for_leiden = [ch for ch in user_spec_bg_channels_for_thresh if ch in base_scaled_pixel_expression_df.columns]
            print(f"      Using user-specified background channels for thresholding: {background_names_for_leiden}")

        if not background_names_for_leiden:
            print("      WARNING: No valid background channels found for thresholding. Thresholding will be skipped for this context.")
            # This will cause background_data_for_leiden to remain None, and run_spatial_leiden will handle it.
        else:
            try:
                background_data_for_leiden = base_scaled_pixel_expression_df[background_names_for_leiden].values.astype(np.float32)
                if background_data_for_leiden.ndim == 1:
                    background_data_for_leiden = background_data_for_leiden.reshape(-1,1)
                print(f"      Extracted background data for thresholding with shape: {background_data_for_leiden.shape}")
            except Exception as e_bg_extract:
                print(f"      ERROR extracting background data for thresholding: {e_bg_extract}. Thresholding will be skipped.")
                background_data_for_leiden = None
                background_names_for_leiden = None
    # --- END: Prepare for background-derived thresholding ---

    # Define clustering contexts
    clustering_contexts = ["all_channels"] + [ch for ch in channels_for_profile_analysis if ch in all_available_scaled_channels]
    
    overall_success = True # Track if all contexts process successfully

    for context_idx, context_name in enumerate(clustering_contexts):
        print(f"\n--- Processing Context: {context_name} for ROI: {roi_string}, Resolution: {resolution_param} ---")
        context_output_dir_specific = os.path.join(resolution_parent_output_dir, f"context_{context_name}")
        os.makedirs(context_output_dir_specific, exist_ok=True)

        # Initialize variables for this context
        context_pixel_community_df_from_leiden = None
        context_pixel_graph = None
        # current_pixel_results_df_for_context will be final_pixel_annotations_df up to the previous context,
        # and then we add the current context's community column to it for analysis within this loop iteration.
        
        try:
            pixel_coords_for_clustering = base_scaled_pixel_expression_df[['X', 'Y']].copy()
            
            channels_for_this_clustering = []
            if context_name == "all_channels":
                channels_for_this_clustering = [ch for ch in all_available_scaled_channels if ch in base_scaled_pixel_expression_df.columns]
            else: # Single channel context
                if context_name in base_scaled_pixel_expression_df.columns:
                    channels_for_this_clustering = [context_name]
                else:
                    print(f"   WARNING: Channel {context_name} for single-channel clustering not found in scaled data. Skipping context.")
                    continue

            if not channels_for_this_clustering:
                print(f"   ERROR: No valid channels for clustering context '{context_name}'. Skipping context.")
                overall_success = False
                continue
            
            scaled_expression_values_for_clustering = base_scaled_pixel_expression_df[channels_for_this_clustering].values
            if scaled_expression_values_for_clustering.ndim == 1: # Ensure 2D for single channel
                 scaled_expression_values_for_clustering = scaled_expression_values_for_clustering.reshape(-1, 1)

            context_pixel_community_df_from_leiden, context_pixel_graph, _, _ = _cluster_pixels(
                pixel_coords_df=pixel_coords_for_clustering,
                scaled_expression_df_values=scaled_expression_values_for_clustering,
                protein_channels_for_weighting=channels_for_this_clustering,
                resolution_param=resolution_param,
                config=config,
                # --- START: Pass background data to _cluster_pixels ---
                background_channel_data_for_thresholding=background_data_for_leiden,
                background_channel_names_for_thresholding=background_names_for_leiden
                # --- END: Pass background data to _cluster_pixels ---
            )

            if context_pixel_community_df_from_leiden is None or context_pixel_graph is None:
                print(f"   Leiden clustering failed for context '{context_name}'. Skipping further analysis for this context.")
                overall_success = False
                continue

            # --- Save per-context pixel community memberships --- START MODIFICATION
            try:
                # Create a DataFrame with X, Y, and the community assignments for this context
                # Assuming base_scaled_pixel_expression_df.index aligns with context_pixel_community_df_from_leiden
                # and that context_pixel_community_df_from_leiden has a 'community' column.
                per_context_membership_df = base_scaled_pixel_expression_df[['X', 'Y']].copy()
                # Assigning .values ensures that if context_pixel_community_df_from_leiden has a simple RangeIndex,
                # it correctly aligns with per_context_membership_df which shares index with base_scaled_pixel_expression_df.
                per_context_membership_df[f'community_{context_name}'] = context_pixel_community_df_from_leiden['community'].values

                context_membership_save_path = os.path.join(
                    context_output_dir_specific,
                    f"pixel_community_membership_{roi_string}_res_{resolution_str_clean}_context_{context_name}.csv"
                )
                per_context_membership_df.to_csv(context_membership_save_path, index=True) # Save with original pixel index
                print(f"   Per-context pixel community memberships saved to: {os.path.basename(context_membership_save_path)}")
            except Exception as e_save_context_membership:
                print(f"   ERROR saving per-context pixel community memberships for context '{context_name}': {e_save_context_membership}")
            # --- END MODIFICATION ---

            # Add/update community column in the main annotation DF
            community_col_name = f"community_{context_name}"
            final_pixel_annotations_df[community_col_name] = context_pixel_community_df_from_leiden['community']
            

            # --- Calculate and Save Community Adjacencies for this context ---
            if 'community' in context_pixel_community_df_from_leiden.columns: # Leiden output has 'community'
                # Pass the original context_pixel_community_df_from_leiden here as it directly maps to graph
                _ = calculate_and_save_community_adjacencies(
                    pixel_graph=context_pixel_graph,
                    pixel_community_df=context_pixel_community_df_from_leiden, 
                    output_dir=context_output_dir_specific, # Save to context-specific dir
                    roi_string=roi_string, # Base ROI string
                    resolution_param=resolution_param, # Base resolution
                    config=config, # Pass config
                    n_jobs=config['processing']['parallel_jobs'] # Pass n_jobs
                )
            else:
                print(f"   Skipping community adjacency calculation for context '{context_name}': 'community' column not in Leiden output.")

            # Prepare current_pixel_results_df for _analyze_communities for this context
            # It needs the expression data and the *current context's* community assignments.
            # The 'community' column expected by _analyze_communities should be the one for the current context.
            temp_analysis_df = base_scaled_pixel_expression_df.copy()
            temp_analysis_df['community'] = final_pixel_annotations_df[community_col_name] # Use current context's community

            if 'community' not in temp_analysis_df.columns or temp_analysis_df['community'].isnull().all():
                print(f"   ERROR: 'community' column (for context {context_name}) is missing or all NaN. Cannot proceed with community analysis for this context.")
                overall_success = False
                continue

            context_scaled_community_profiles, context_diff_expr_profiles, context_primary_channel_map = _analyze_communities(
                current_pixel_results_df=temp_analysis_df, # Has expression data + this context's community col
                channels_for_profiles=channels_for_profile_analysis, # Use the globally defined channel order for profiles
                pixel_graph=context_pixel_graph,
                resolution_output_dir=context_output_dir_specific, # Save to context-specific dir
                roi_string=roi_string, # Base ROI string
                resolution_param=resolution_param 
            )
            del temp_analysis_df # Clean up intermediate df

            if context_scaled_community_profiles is None:
                print(f"   Profile calculation failed for context '{context_name}'. Some downstream steps might be skipped for this context.")
                 # Continue to try other analyses for the context if possible, but overall_success might be affected by this.

            if context_scaled_community_profiles is not None and not context_scaled_community_profiles.empty:
                print(f"   Calculating and saving community correlation matrix for context '{context_name}'...")
                try:
                    valid_channels_for_comm_corr = [ch for ch in channels_for_profile_analysis if ch in context_scaled_community_profiles.columns]
                    if valid_channels_for_comm_corr:
                        community_corr_matrix = context_scaled_community_profiles[valid_channels_for_comm_corr].corr(method='spearman')
                        community_corr_save_path = os.path.join(
                            context_output_dir_specific, # Save to context dir
                            f"community_channel_correlation_{roi_string}_res_{resolution_str_clean}.csv" # Filename can be simple
                        )
                        community_corr_matrix.to_csv(community_corr_save_path)
                        print(f"   Community correlation matrix for context '{context_name}' saved to: {os.path.basename(community_corr_save_path)}")
                    else: print(f"     ERROR: No valid channels from final order in community profiles for correlation (context '{context_name}'). Skipping save.")
                except Exception as comm_corr_e: print(f"     ERROR calculating/saving community correlation matrix for context '{context_name}': {comm_corr_e}")

            _ = calculate_and_save_umap(
                diff_expr_profiles=context_diff_expr_profiles,
                scaled_community_profiles=context_scaled_community_profiles,
                roi_channels=channels_for_profile_analysis, 
                resolution_output_dir=context_output_dir_specific, # Save to context dir
                roi_string=roi_string, # Base ROI string
                resolution_param=resolution_param,
                config=config,
                umap_available=umap_available
            )

            linkage_file_prefix = f"community_linkage_matrix_{roi_string}_res_{resolution_str_clean}" # Prefix can be simple
            _ = calculate_and_save_community_linkage(
                scaled_community_profiles=context_scaled_community_profiles,
                ordered_channels=channels_for_profile_analysis,
                output_dir=context_output_dir_specific, # Save to context dir
                file_prefix=linkage_file_prefix, 
                config=config
            )
            print(f"--- Finished context: {context_name} for ROI: {roi_string}, Resolution: {resolution_param} ---")

        except Exception as context_e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"   ERROR during processing context '{context_name}' for ROI {roi_string}, resolution {resolution_param}: {str(context_e)}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            traceback.print_exc()
            overall_success = False
            # Continue to the next context
        finally:
            # Clean up context-specific large objects
            if 'context_pixel_community_df_from_leiden' in locals(): del context_pixel_community_df_from_leiden
            if 'context_pixel_graph' in locals(): del context_pixel_graph
            if 'context_scaled_community_profiles' in locals(): del context_scaled_community_profiles
            if 'context_diff_expr_profiles' in locals(): del context_diff_expr_profiles
            if 'context_primary_channel_map' in locals(): del context_primary_channel_map
            gc.collect()

    # After all contexts, save the consolidated pixel annotation file
    final_annotations_save_path = os.path.join(
        resolution_parent_output_dir, # Saved in the parent resolution dir
        f"pixel_data_with_community_annotations_{roi_string}_res_{resolution_str_clean}.csv"
    )
    try:
        final_pixel_annotations_df.to_csv(final_annotations_save_path, index=True)
        print(f"\n   Consolidated pixel data with all community annotations for {roi_string} resolution {resolution_param} saved to: {os.path.basename(final_annotations_save_path)}")
    except Exception as e_save_final:
        print(f"   ERROR saving final consolidated annotations: {e_save_final}")
        overall_success = False


    # Explicitly delete large DataFrames that were used throughout the resolution processing
    # Context-specific DFs are deleted inside the context loop's finally block
    try:
        if 'base_scaled_pixel_expression_df' in locals():
            del base_scaled_pixel_expression_df
        if 'final_pixel_annotations_df' in locals():
            del final_pixel_annotations_df
        gc.collect()
        print(f"   --- Final memory cleanup for ROI {roi_string}, Resolution {resolution_param} completed ---")
    except NameError as ne:
        print(f"   Warning during final cleanup: {ne}")

    return overall_success


# --- New Helper for Community Adjacencies ---
def _process_edge_chunk(
    edge_chunk: List[Tuple[int, int]], # Changed from List[Any] to List of (source_vid, target_vid) tuples
    pixel_community_df: pd.DataFrame,
    num_total_edges: int # For progress reporting if needed by worker (optional)
) -> Dict[str, set]:
    """Processes a chunk of edges to build a partial adjacency dictionary."""
    partial_adjacencies = {}
    for source_vid, target_vid in edge_chunk: # Iterate through (source, target) tuples
        try:
            comm_source = pixel_community_df.iloc[source_vid]['community']
            comm_target = pixel_community_df.iloc[target_vid]['community']
            if comm_source != comm_target:
                partial_adjacencies.setdefault(str(comm_source), set()).add(str(comm_target))
                partial_adjacencies.setdefault(str(comm_target), set()).add(str(comm_source))
        except IndexError:
            continue
        except KeyError:
            continue
    return partial_adjacencies

def calculate_and_save_community_adjacencies(
    pixel_graph: Any,
    pixel_community_df: pd.DataFrame,
    output_dir: str,
    roi_string: str,
    resolution_param: Any,
    config: Dict,
    n_jobs: int = 1 # Add n_jobs parameter
) -> Optional[str]:
    """
    Calculates and saves the adjacency list for communities based on the pixel graph.
    Parallelized edge processing.
    """
    print(f"   Calculating community adjacencies for {roi_string} res {resolution_param} (using up to {n_jobs} cores)...")
    start_time = time.time()
    final_adjacencies = {}

    if pixel_graph is None:
        print("    ERROR: Pixel graph is None. Cannot calculate community adjacencies.")
        return None
    if pixel_community_df.empty or 'community' not in pixel_community_df.columns:
        print("    ERROR: Pixel community DataFrame is invalid or missing 'community' column.")
        return None

    try:
        if len(pixel_community_df) != pixel_graph.vcount():
            print(f"    WARNING: Length of pixel_community_df ({len(pixel_community_df)}) does not match pixel_graph vertex count ({pixel_graph.vcount()}). Review if mapping is correct.")
            # Proceeding with iloc mapping assumption.

        # Convert edges to (source_vid, target_vid) tuples for pickling
        all_edges_tuples = [(edge.source, edge.target) for edge in pixel_graph.es]
        num_total_edges = len(all_edges_tuples)
        print(f"    Processing {num_total_edges} edges in the pixel graph...")

        if n_jobs == 1 or num_total_edges == 0: # Fallback to serial processing for n_jobs=1 or no edges
            print("    Running adjacency calculation in serial mode.")
            processed_edges = 0
            for source_vid, target_vid in all_edges_tuples: # Use the tuples here too for consistency
                try:
                    comm_source = pixel_community_df.iloc[source_vid]['community']
                    comm_target = pixel_community_df.iloc[target_vid]['community']
                except IndexError:
                    continue
                except KeyError:
                    continue

                if comm_source != comm_target:
                    final_adjacencies.setdefault(str(comm_source), set()).add(str(comm_target))
                    final_adjacencies.setdefault(str(comm_target), set()).add(str(comm_source))
                
                processed_edges +=1
                if processed_edges > 0 and processed_edges % 500000 == 0: # Adjusted progress reporting for serial
                    print(f"      (Serial) Processed {processed_edges}/{num_total_edges} edges...")
        else:
            print(f"    Running adjacency calculation in parallel with {n_jobs} jobs.")
            chunk_size = math.ceil(num_total_edges / n_jobs)
            edge_chunks = [all_edges_tuples[i : i + chunk_size] for i in range(0, num_total_edges, chunk_size)]
            
            with Parallel(n_jobs=n_jobs, verbose=0) as parallel:
                list_of_partial_adjacencies = parallel(
                    delayed(_process_edge_chunk)(chunk, pixel_community_df, num_total_edges) for chunk in edge_chunks
                )

            print("    Merging partial adjacencies from parallel jobs...")
            for partial_adj in list_of_partial_adjacencies:
                for comm, neighbors in partial_adj.items():
                    if comm not in final_adjacencies:
                        final_adjacencies[comm] = set()
                    final_adjacencies[comm].update(neighbors)
            print("    Merging complete.")

        # Convert sets to sorted lists for JSON serialization
        for comm in final_adjacencies:
            final_adjacencies[comm] = sorted(list(final_adjacencies[comm]))

        res_str_clean = f"{resolution_param:.3f}".rstrip('0').rstrip('.').replace('.', '_') if isinstance(resolution_param, float) else str(resolution_param)
        output_filename = f"community_adjacencies_{roi_string}_res_{res_str_clean}.json"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w') as f:
            json.dump(final_adjacencies, f, indent=4)
        
        print(f"    Community adjacencies saved to: {os.path.basename(output_path)} ({len(final_adjacencies)} communities)")
        print(f"    --- Community adjacency calculation finished in {time.time() - start_time:.2f} seconds ---")
        return output_path

    except AttributeError as ae:
        print(f"    ERROR: AttributeError during graph processing: {ae}. Is pixel_graph an igraph.Graph object? Type: {type(pixel_graph)}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"    ERROR calculating community adjacencies: {e}")
        traceback.print_exc()
        return None


# --- Script Execution Entry Point ---
if __name__ == '__main__':
    print("\n--- Starting IMC Pixel-wise Analysis Pipeline ---")
    gpu_available = check_gpu_availability()
    if gpu_available: print("GPU acceleration is available.")
    else: print("GPU acceleration not available. Proceeding with CPU-only.")

    start_pipeline_time = time.time()
    config = load_config("config.yaml")
    if config is None: sys.exit(1)

    data_dir = config['paths']['data_dir']
    try:
        imc_files = glob.glob(os.path.join(data_dir, "*.txt"))
        if not imc_files: print(f"ERROR: No .txt files in {data_dir}"); sys.exit(1)
        print(f"\nFound {len(imc_files)} IMC data files.")
    except Exception as e: print(f"ERROR finding input files: {e}"); sys.exit(1)

    output_dir = config['paths']['output_dir']
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Data directory: {data_dir}\nOutput directory: {output_dir}")
    except Exception as e: print(f"ERROR creating output directory: {e}"); sys.exit(1)

    metadata_file = config['paths'].get('metadata_file')
    metadata_map = {}
    first_timepoint_value_global = None # Renamed to avoid conflict with preprocess_single_roi internal var
    timepoint_col_name = config.get('experiment_analysis', {}).get('timepoint_col')
    metadata_roi_col_name = config.get('experiment_analysis', {}).get('metadata_roi_col')

    def get_roi_string_from_path(p):
        fname = os.path.basename(p); base, _ = os.path.splitext(fname)
        match = re.search(r'(?:^|_)(ROI_[A-Za-z0-9_]+)', base)
        if match: return match.group(1)
        print(f"Warning: Could not extract ROI string from '{fname}'. Using base '{base}'."); return base

    if metadata_file and os.path.exists(metadata_file) and timepoint_col_name and metadata_roi_col_name:
        try:
            metadata_df = pd.read_csv(metadata_file)
            print(f"Metadata loaded from: {metadata_file}")
            if metadata_roi_col_name not in metadata_df.columns: raise ValueError(f"Meta ROI col missing: {metadata_roi_col_name}")
            if timepoint_col_name not in metadata_df.columns: raise ValueError(f"Meta timepoint col missing: {timepoint_col_name}")
            metadata_df[metadata_roi_col_name] = metadata_df[metadata_roi_col_name].astype(str)
            metadata_map = {row[metadata_roi_col_name]: row.to_dict() for _, row in metadata_df.iterrows()}
            print(f"   Created metadata map with {len(metadata_map)} entries.")
            unique_timepoints = metadata_df[timepoint_col_name].unique()
            
            # Robustly determine the first timepoint value
            valid_timepoints = [tp for tp in unique_timepoints if pd.notna(tp)]
            if not valid_timepoints:
                print("   WARNING: No valid (non-NaN) timepoint values found in metadata.")
            else:
                try:
                    # Try sorting as numbers
                    first_timepoint_value_global = sorted(valid_timepoints, key=lambda x: float(x))[0]
                except (ValueError, TypeError):
                    # Fallback to string sort if numeric sort fails
                    print("   Note: Could not sort all timepoints numerically, falling back to string sort for first timepoint determination.")
                    first_timepoint_value_global = sorted(valid_timepoints, key=lambda x: str(x))[0]
            
            if first_timepoint_value_global is not None:
                 print(f"Identified first timepoint for reference: {first_timepoint_value_global} (Type: {type(first_timepoint_value_global)})")
            else:
                print("   WARNING: Could not determine the first timepoint value globally.")

        except Exception as e_meta: print(f"ERROR processing metadata: {e_meta}. No metadata features."); metadata_map, first_timepoint_value_global = {}, None
    else: print("WARNING: No metadata or incomplete config. No metadata-based ordering."); metadata_map, first_timepoint_value_global = {}, None

    try:
        cfg_parallel = config['processing']['parallel_jobs']; cpu_count = multiprocessing.cpu_count()
        if isinstance(cfg_parallel, int):
            if cfg_parallel == -1: n_jobs_cpu_tasks = cpu_count
            elif cfg_parallel <= -2: n_jobs_cpu_tasks = max(1, cpu_count + cfg_parallel + 1)
            elif cfg_parallel > 0: n_jobs_cpu_tasks = min(cfg_parallel, cpu_count)
            else: n_jobs_cpu_tasks = 1
        else: n_jobs_cpu_tasks = 1
    except KeyError: n_jobs_cpu_tasks = 1 # Default if key is missing
    except Exception: n_jobs_cpu_tasks = 1 # General fallback

    # Determine n_jobs for top-level parallel phases (ROI preprocessing, Resolution analysis)
    # If GPU is used, these phases should have limited parallelism to avoid OOM.
    # CPU-bound tasks *within* these phases (like adjacency) can use n_jobs_cpu_tasks.
    n_jobs_gpu_aware_phases = n_jobs_cpu_tasks # Default to CPU task parallelism
    max_concurrent_gpu_jobs = config.get('processing', {}).get('max_concurrent_gpu_jobs', 1)

    # Check if GPU will actually be used by downstream functions like run_spatial_leiden
    # This logic should mirror how run_spatial_leiden determines GPU use based on 'use_gpu' config and availability.
    # For simplicity, we assume if config.processing.use_gpu is 'true' or 'auto' (and gpu_available is True), then GPU will be attempted.
    cfg_use_gpu_setting = config.get('processing', {}).get('use_gpu', 'auto')
    attempt_gpu_downstream = False
    if isinstance(cfg_use_gpu_setting, str) and cfg_use_gpu_setting.lower() == 'true':
        attempt_gpu_downstream = True
    elif isinstance(cfg_use_gpu_setting, bool) and cfg_use_gpu_setting:
        attempt_gpu_downstream = True
    elif isinstance(cfg_use_gpu_setting, str) and cfg_use_gpu_setting.lower() == 'auto' and gpu_available:
        attempt_gpu_downstream = True
    # If actual gpu_utils_available is False (e.g. torch not found), then gpu_available would be False too.

    if attempt_gpu_downstream and gpu_available: # only cap if GPU is configured AND available
        n_jobs_gpu_aware_phases = max(1, min(n_jobs_cpu_tasks, max_concurrent_gpu_jobs))
        print(f"\nGPU acceleration is active. Top-level parallel phases will use up to {n_jobs_gpu_aware_phases} concurrent job(s) to manage GPU memory.")
        print(f"CPU-bound sub-tasks within each job can still use up to {n_jobs_cpu_tasks} cores if parallelized internally.")
    else:
        print(f"\nUsing up to {n_jobs_cpu_tasks} cores for parallel tasks (GPU not actively used by top-level phases or not available).")

    # --- Phase 1: Preprocessing all ROIs (generates ROI-specific orders) ---
    print(f"\n--- Starting Phase 1: Initial Preprocessing for {len(imc_files)} ROIs (using {n_jobs_gpu_aware_phases} jobs) ---")
    phase1_args = [
        (i, fp, len(imc_files), config, metadata_map.get(get_roi_string_from_path(fp)), first_timepoint_value_global)
        for i, fp in enumerate(imc_files[-1:])
    ]
    with Parallel(n_jobs=n_jobs_gpu_aware_phases, verbose=10) as parallel:
        phase1_results_infos = parallel(delayed(preprocess_single_roi)(*args) for args in phase1_args)
    
    # Filter out None results (failed preprocessing)
    successfully_preprocessed_roi_infos = [info for info in phase1_results_infos if info is not None]
    if not successfully_preprocessed_roi_infos:
        print("ERROR: Phase 1 preprocessing failed for all ROIs. Exiting."); sys.exit(1)
    print(f"--- Phase 1: Successfully preprocessed {len(successfully_preprocessed_roi_infos)} ROIs initially ---")

    # --- Phase 1.5: Calculate Consensus Reference Order & Update Non-First Timepoint ROIs ---
    consensus_reference_channel_order = None
    if first_timepoint_value_global is not None: # Only proceed if metadata and first timepoint were identified
        print(f"\n--- Starting Phase 1.5: Consensus Reference Order Calculation (using timepoint {first_timepoint_value_global}) ---")
        first_timepoint_roi_infos_for_consensus = [info for info in successfully_preprocessed_roi_infos if info['is_first_timepoint_roi']]

        if not first_timepoint_roi_infos_for_consensus:
            print(f"WARNING: No first timepoint ROIs found/successfully preprocessed. Cannot generate consensus reference order.")
        else:
            print(f"Found {len(first_timepoint_roi_infos_for_consensus)} first-timepoint ROIs for consensus calculation.")
            correlation_matrices_for_consensus = []
            all_channels_for_consensus = set()
            failed_consensus_rois = 0

            for ft_roi_info in first_timepoint_roi_infos_for_consensus:
                corr_path = ft_roi_info.get('pixel_correlation_matrix_path')
                if corr_path and os.path.exists(corr_path):
                    try:
                        corr_mat = pd.read_csv(corr_path, index_col=0)
                        actual_channels_in_corr = corr_mat.columns.tolist()
                        if not actual_channels_in_corr:
                            print(f"  WARNING: Correlation matrix for {ft_roi_info['roi_string']} is empty. Skipping for consensus."); failed_consensus_rois+=1; continue
                        correlation_matrices_for_consensus.append(corr_mat)
                        all_channels_for_consensus.update(actual_channels_in_corr)
                    except Exception as e_corr_load:
                        print(f"  WARNING: Failed to load/process correlation matrix from {corr_path} for {ft_roi_info['roi_string']}: {e_corr_load}"); failed_consensus_rois+=1
                else:
                    print(f"  WARNING: Missing correlation matrix for first timepoint ROI {ft_roi_info['roi_string']}. Skipping for consensus."); failed_consensus_rois+=1
            
            print(f"  Processed {len(correlation_matrices_for_consensus)} correlation matrices for consensus ({failed_consensus_rois} skipped/failed).")

            if correlation_matrices_for_consensus:
                consensus_channels_list = sorted(list(all_channels_for_consensus))
                reindexed_matrices = [m.reindex(index=consensus_channels_list, columns=consensus_channels_list) for m in correlation_matrices_for_consensus]
                valid_reindexed_matrices = [m.to_numpy() for m in reindexed_matrices if not m.isnull().all().all()]
                
                if valid_reindexed_matrices:
                    stacked_matrices = np.stack(valid_reindexed_matrices, axis=0)
                    average_corr_matrix_np = np.nanmean(stacked_matrices, axis=0)
                    avg_corr_df = pd.DataFrame(average_corr_matrix_np, index=consensus_channels_list, columns=consensus_channels_list).fillna(0)

                    if not avg_corr_df.empty and len(avg_corr_df.columns) >= 2:
                        try:
                            linkage_ref = sch.linkage(sch.distance.pdist(avg_corr_df.values), method='ward')
                            dend_ref = sch.dendrogram(linkage_ref, no_plot=True)
                            consensus_reference_channel_order = [avg_corr_df.columns[i] for i in dend_ref['leaves']]
                            print(f"  >>> Consensus Reference Channel Order ({len(consensus_reference_channel_order)} channels) computed.")
                            np.save(os.path.join(output_dir, "reference_channel_linkage.npy"), linkage_ref)
                            avg_corr_df.to_csv(os.path.join(output_dir, "reference_average_correlation.csv"))
                            with open(os.path.join(output_dir, "reference_channel_order.json"), 'w') as f: json.dump(consensus_reference_channel_order, f, indent=4)
                            print(f"  Consensus reference order and related files saved to: {output_dir}")
                        except Exception as e_consensus_cluster:
                            print(f"  ERROR during clustering of consensus average matrix: {e_consensus_cluster}. No consensus order.")
                            consensus_reference_channel_order = None
                    else: print("  WARNING: Consensus average correlation matrix empty/too small. No consensus order.")
                else: print("  WARNING: No valid correlation matrices after reindexing for consensus. No consensus order.")
            else: print("ERROR: No correlation matrices gathered from first-timepoint ROIs. No consensus order.")

        # Now, update non-first-timepoint ROIs if consensus_reference_channel_order was successfully created
        if consensus_reference_channel_order:
            print(f"\n  Updating non-first-timepoint ROIs with consensus order...")
            tasks_for_reorder = []
            for i, roi_info_dict in enumerate(successfully_preprocessed_roi_infos):
                if not roi_info_dict['is_first_timepoint_roi']:
                    tasks_for_reorder.append( (roi_info_dict, consensus_reference_channel_order, config) )
            
            if tasks_for_reorder:
                print(f"    Re-evaluating final order for {len(tasks_for_reorder)} non-first-timepoint ROIs.")
                with Parallel(n_jobs=n_jobs_cpu_tasks, verbose=5) as parallel_reorder:
                    updated_orders_list = parallel_reorder(delayed(recalculate_final_order_with_consensus)(*task_args) for task_args in tasks_for_reorder)
                
                task_idx = 0
                for i, roi_info_dict in enumerate(successfully_preprocessed_roi_infos):
                    if not roi_info_dict['is_first_timepoint_roi']:
                        if task_idx < len(updated_orders_list) and updated_orders_list[task_idx] is not None:
                            successfully_preprocessed_roi_infos[i]['ordered_channels_for_downstream'] = updated_orders_list[task_idx]
                        task_idx += 1
            else:
                print("    No non-first-timepoint ROIs to update with consensus order.")
        else:
            print("  Skipping update of non-first-timepoint ROIs as no consensus order was generated.")
    else:
        print("\n--- Skipping Phase 1.5: Consensus Reference Order Calculation (no metadata/first timepoint defined globally) ---")

    # --- Phase 2: Resolution processing for all (successfully) preprocessed ROIs ---
    print(f"\n--- Starting Phase 2: Resolution processing for {len(successfully_preprocessed_roi_infos)} ROIs ---")
    resolution_params_cfg = config['analysis']['clustering'].get('resolution_params', [0.5])
    if not isinstance(resolution_params_cfg, list) or not resolution_params_cfg: resolution_params_cfg = [0.5]
    print(f"Configured resolutions for Phase 2: {resolution_params_cfg}")

    phase2_tasks = []
    for roi_info_item in successfully_preprocessed_roi_infos:
        for res_p in resolution_params_cfg:
            phase2_tasks.append( (roi_info_item, res_p) )

    if not phase2_tasks: print("No resolution tasks for Phase 2. Exiting."); sys.exit(0)
    print(f"Total resolution tasks for Phase 2: {len(phase2_tasks)} (using {n_jobs_gpu_aware_phases} top-level jobs)")

    with Parallel(n_jobs=n_jobs_gpu_aware_phases, verbose=10) as parallel:
        phase2_results_flags = parallel(delayed(analyze_single_resolution_for_roi)(*task_args_p2) for task_args_p2 in phase2_tasks)
    
    successful_phase2_tasks = sum(1 for s in phase2_results_flags if s)
    failed_phase2_tasks = len(phase2_results_flags) - successful_phase2_tasks

    # --- Summary ---
    print(f"\n--- Pipeline Summary ---")
    print(f"Phase 1: Initial preprocessing completed for {len(successfully_preprocessed_roi_infos)} ROIs.")
    if consensus_reference_channel_order:
        print(f"Phase 1.5: Consensus reference order ({len(consensus_reference_channel_order)} channels) was calculated and applied.")
    else:
        print("Phase 1.5: Consensus reference order was not calculated or not applied; ROIs use their own specific orders or initial settings.")
    print(f"Phase 2: Successfully completed {successful_phase2_tasks} of {len(phase2_tasks)} resolution analysis tasks.")
    if failed_phase2_tasks > 0: print(f"  Failed {failed_phase2_tasks} resolution tasks (see logs).")

    total_time = time.time() - start_pipeline_time
    print(f"Total pipeline execution time: {total_time:.2f}s ({total_time/60:.2f}m).")
    print("\n================ Completed processing. ================")

