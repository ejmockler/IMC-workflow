import yaml
import os
import sys
import time
import pandas as pd
import numpy as np # Import numpy
from typing import Dict, Optional, List, Tuple # Added List, Tuple
import gc # Import garbage collector
import glob # For finding files
import ast # For parsing string representation of frozenset
import matplotlib.pyplot as plt
import seaborn as sns
import re # Added for robust itemset parsing
import traceback # Added for error logging
import json # Added for load_all_adjacency_data

# Import the aggregation function and the new analysis function
from src.experiment_pipeline.data_aggregation import aggregate_and_merge_data, load_metadata
from src.experiment_pipeline.analysis_core import (
    calculate_community_abundance,
    prepare_composition_table,
    run_composition_permanova,
    cluster_profiles_hierarchical,
    analyze_metacluster_composition, # Import the new function
    get_channel_columns             # <-- Added import
)
from src.experiment_pipeline.visualization import (
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

    # Priority 1: Use lists from config if provided
    available_channels = config.get('data', {}).get('protein_channels') + config.get('data', {}).get('background_channels')
    if available_channels and isinstance(available_channels, list):
        # Ensure the columns actually exist in the dataframe
        channel_cols = [col for col in available_channels if col in df.columns]
        if len(channel_cols) < len(available_channels):
            missing_in_df = [col for col in available_channels if col not in df.columns]
            print(f"   Warning: Some channels from config master list not found in DataFrame: {missing_in_df}")
        print(f"   Identified {len(channel_cols)} channel columns using master list from config.")
        return channel_cols

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

# --- Helper function to extract ROI string consistently ---
def _extract_standard_roi_from_filepath(path_str: str, config: Dict, path_depth_adjustment: int = 0) -> Optional[str]:
    """
    Extracts a standardized ROI string from a file path, mimicking how metadata ROI keys might be generated.
    Uses regex patterns typically found in IMC data.
    Assumes ROI information is in one of the parent directories of the archetype file.
    e.g. /path/to/output/ROI_XYZ_ABC/resolution_100/archetype.csv -> ROI_XYZ_ABC
    With community structure: /path/to/output/ROI_XYZ_ABC/resolution_100/community_123/archetype.csv -> ROI_XYZ_ABC

    Args:
        path_str: The full path to the file.
        config: The configuration dictionary (not directly used here but kept for consistency).
        path_depth_adjustment: Adjusts which part of the path is considered for ROI. 
                               0 for old structure (ROI dir is 3 levels up from file).
                               1 for new community structure (ROI dir is 4 levels up from file).
    """
    try:
        # parts = [..., 'base_output_dir', 'ROI_XYZ', 'resolution_100', 'community_123', 'file.csv']
        # For new structure (community files), ROI dir is parts[-(4 + path_depth_adjustment)] which is parts[-4] if adjustment=0
        # No, let's simplify. The target index from the end for ROI directory:
        # Old structure: parts[-3] (e.g. .../ROI_DIR/resolution_XXX/file.csv)
        # New structure: parts[-4] (e.g. .../ROI_DIR/resolution_XXX/community_YYY/file.csv)
        # So, if the deepest element is -1 (filename):
        #   -2 is parent dir (e.g. community_YYY or resolution_XXX)
        #   -3 is grandparent (e.g. resolution_XXX or ROI_DIR)
        #   -4 is great-grandparent (e.g. ROI_DIR if community structure)
        
        # Let's make this robust by searching for "ROI_" in parent directories
        parts = path_str.split(os.sep)
        roi_containing_dir = None
        # Search backwards from the directory containing the file
        for i in range(len(parts) - 2, 0, -1): # Start from parent of file's dir, go up
            if "ROI_" in parts[i]:
                roi_containing_dir = parts[i]
                break
        
        if roi_containing_dir:
            match = re.search(r'(ROI_[A-Za-z0-9_]+)', roi_containing_dir)
            if match:
                return match.group(1)
            return roi_containing_dir # Fallback to the directory name if regex fails
        else:
            print(f"Warning: Could not find a directory part containing 'ROI_' in path {path_str}.")
            return None
            
    except IndexError:
        print(f"Warning: Index error during ROI string extraction from path {path_str}.")    
    return None

# --- New Archetype Analysis Functions ---

def parse_itemset_string(itemset_str: str) -> Optional[frozenset]:
    """Safely parses a string like "frozenset({'marker1', 'marker2'})" into a frozenset."""
    if not isinstance(itemset_str, str):
        return None
    match = re.fullmatch(r"frozenset\((.*)\)", itemset_str.strip())
    if match:
        set_literal_str = match.group(1)
        try:
            evaluated_content = ast.literal_eval(set_literal_str)
            if isinstance(evaluated_content, set):
                return frozenset(evaluated_content)
            return None 
        except (ValueError, SyntaxError, TypeError) as e:
            return None
    return None

def load_all_archetype_data(base_output_dir: str, metadata_df: pd.DataFrame, config: Dict) -> Optional[pd.DataFrame]:
    """
    Loads all PER-COMMUNITY archetype CSV files, filters singletons, parses itemsets, 
    extracts ROI, resolution, and community ID, and merges with metadata.
    """
    print("\n--- Loading and Processing Per-Community Archetype Data ---")
    # Updated glob pattern to find archetypes in community subdirectories
    archetype_files = glob.glob(os.path.join(base_output_dir, "ROI_*", "resolution_*", "community_*", "*_channel_archetypes_*.csv"))
    
    if not archetype_files:
        print(f"ERROR: No per-community archetype files found under {base_output_dir}/ROI_*/resolution_*/community_*/")
        return None

    all_archetype_dfs = []
    print(f"Found {len(archetype_files)} per-community archetype files to process.")

    for i, f_path in enumerate(archetype_files):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(archetype_files)} files...")
        try:
            # Extract ROI string (should be robust to new path depth)
            roi_str_from_path = _extract_standard_roi_from_filepath(f_path, config)
            if not roi_str_from_path:
                print(f"Warning: Could not extract a standard ROI string from path {f_path}. Skipping this file.")
                continue

            parts = f_path.split(os.sep)
            # parts: [..., base_dir, ROI_XXX, resolution_YYY, community_ZZZ, file.csv]
            # Resolution is now parts[-3]
            # Community ID is now parts[-2]
            try:
                resolution_str_from_path = parts[-3].replace("resolution_", "")
                community_id_str_from_path = parts[-2].replace("community_", "")
            except IndexError:
                print(f"Warning: Path structure unexpected for {f_path}. Cannot extract resolution/community. Skipping.")
                continue
            
            df = pd.read_csv(f_path)
            if df.empty or 'itemsets' not in df.columns or 'support' not in df.columns:
                print(f"Info: Skipping empty or malformed archetype file: {f_path}")
                continue

            df['itemset'] = df['itemsets'].apply(parse_itemset_string)
            df = df.dropna(subset=['itemset'])
            df['itemset_size'] = df['itemset'].apply(len)
            # Keep singletons for now if needed for neighborhood analysis, filter later if necessary
            # df = df[df['itemset_size'] > 1] # Original filter, commented out

            if df.empty:
                # print(f"Info: No valid itemsets after parsing/filtering in {f_path}") # More verbose if needed
                continue
                
            df['roi_string'] = roi_str_from_path
            df['community_id'] = community_id_str_from_path # Add community ID
            try:
                # Attempt to convert resolution to a numeric type (float, then int if possible)
                # Resolutions can sometimes be like "0_5" from "0.5"
                cleaned_res_str = resolution_str_from_path.replace('_', '.')
                try:
                    df['resolution'] = float(cleaned_res_str)
                except ValueError:
                     # If float fails, try int, though float is more general for resolutions
                    df['resolution'] = int(cleaned_res_str) 
            except ValueError:
                print(f"Warning: Could not parse resolution '{resolution_str_from_path}' (cleaned: '{cleaned_res_str}') as numeric from path {f_path}. Skipping this file.")
                continue
            
            # Select desired columns including community_id
            all_archetype_dfs.append(df[['roi_string', 'resolution', 'community_id', 'support', 'itemset', 'itemset_size']])
        except Exception as e:
            print(f"ERROR processing archetype file {f_path}: {e} {traceback.format_exc()}") # Added traceback

    if not all_archetype_dfs:
        print("ERROR: No valid per-community archetype data could be loaded after processing all files.")
        return None

    print(f"  Finished reading files. Concatenating {len(all_archetype_dfs)} individual dataframes.")
    concatenated_df = pd.concat(all_archetype_dfs, ignore_index=True)
    print(f"  Total itemset entries loaded from communities: {len(concatenated_df)}")

    if concatenated_df.empty:
        print("ERROR: Concatenated per-community archetype data is empty.")
        return None

    # Convert community_id to a more appropriate type if possible (e.g., int if always numeric)
    # For now, keep as string, as it might contain non-numeric parts if community naming changes.
    # concatenated_df['community_id'] = pd.to_numeric(concatenated_df['community_id'], errors='ignore')

    # Merge with metadata (this part remains largely the same)
    exp_config = config.get('experiment_analysis', {})
    metadata_join_key = 'roi_standard_key' 
    timepoint_col = exp_config.get('timepoint_col')
    replicate_col = exp_config.get('replicate_col')

    if metadata_df is None or metadata_df.empty:
        print("Warning: Metadata DataFrame is missing or empty. Archetype data will not have timepoints or other conditions.")
        return concatenated_df

    if metadata_join_key not in metadata_df.columns:
        print(f"ERROR: Metadata join key '{metadata_join_key}' not found in metadata_df columns: {metadata_df.columns.tolist()}. Cannot merge.")
        return concatenated_df

    cols_to_select_from_metadata = {metadata_join_key} 
    
    if timepoint_col and timepoint_col in metadata_df.columns:
        cols_to_select_from_metadata.add(timepoint_col)
    elif timepoint_col:
        print(f"Warning: Timepoint column '{timepoint_col}' (from config) not found in metadata_df.")
    
    if replicate_col and replicate_col in metadata_df.columns:
        cols_to_select_from_metadata.add(replicate_col)
    elif replicate_col:
        print(f"Warning: Replicate column '{replicate_col}' (from config) not found in metadata_df.")

    final_metadata_cols_to_merge = [col for col in list(cols_to_select_from_metadata) if col]
    if not final_metadata_cols_to_merge or metadata_join_key not in final_metadata_cols_to_merge:
        print(f"Warning: No valid metadata columns (or join key '{metadata_join_key}') selected for merging. Proceeding without merge.")
        return concatenated_df
        
    print(f"  Merging with metadata on key: '{metadata_join_key}' using columns: {final_metadata_cols_to_merge}")
    # Ensure roi_string in concatenated_df is suitable for merging with metadata_join_key
    # _extract_standard_roi_from_filepath should already provide this.
    merged_df = pd.merge(concatenated_df, metadata_df[final_metadata_cols_to_merge], left_on='roi_string', right_on=metadata_join_key, how='left')

    # If the merge was successful and we used a different key name, drop the redundant one from metadata
    if metadata_join_key != 'roi_string' and metadata_join_key in merged_df.columns:
        merged_df = merged_df.drop(columns=[metadata_join_key])

    print(f"  Archetype data merged with metadata. Resulting shape: {merged_df.shape}")
    if timepoint_col and timepoint_col in merged_df.columns:
        print(f"    Timepoint information ({timepoint_col}) available for {merged_df[timepoint_col].notna().sum()} / {len(merged_df)} entries.")
    if replicate_col and replicate_col in merged_df.columns:
        print(f"    Replicate information ({replicate_col}) available for {merged_df[replicate_col].notna().sum()} / {len(merged_df)} entries.")

    return merged_df


def analyze_itemset_temporal_dynamics(all_archetype_df: pd.DataFrame, experiment_output_dir: str, config: Dict):
    """
    Analyzes temporal dynamics for ALL non-singleton itemsets, stratified by resolution and replicates.
    Generates plots for each itemset showing support over time.
    Summarizes itemset presence and support characteristics.
    """
    print("\n--- Analyzing Temporal Dynamics for All Itemsets (with Replicate Awareness) ---")
    
    itemset_analysis_main_dir = os.path.join(experiment_output_dir, "all_itemset_temporal_analysis") # New folder name
    os.makedirs(itemset_analysis_main_dir, exist_ok=True)

    exp_config = config.get('experiment_analysis', {})
    timepoint_col = exp_config.get('timepoint_col')
    replicate_col = exp_config.get('replicate_col') 

    if not timepoint_col or timepoint_col not in all_archetype_df.columns:
        print(f"ERROR: Timepoint column '{timepoint_col}' not found in loaded archetype data or not configured. Cannot perform temporal analysis.")
        return

    use_replicates_in_plot = False
    if replicate_col and replicate_col in all_archetype_df.columns:
        if not all_archetype_df[replicate_col].isnull().all():
            if all_archetype_df[replicate_col].nunique() > 1: 
                use_replicates_in_plot = True
                print(f"  Will attempt to plot replicate-specific lines using column: '{replicate_col}'")
        # (Removed redundant else print statements for brevity, covered by initial check)
    
    resolutions = sorted(all_archetype_df['resolution'].unique())
    print(f"Found resolutions to analyze: {resolutions}")
    # No longer need summary_all_resolutions_common_itemsets

    for res in resolutions:
        print(f"\n===== Analyzing Resolution: {res} =====")
        res_specific_output_dir = os.path.join(itemset_analysis_main_dir, f"resolution_{res}")
        os.makedirs(res_specific_output_dir, exist_ok=True)
        itemset_plots_output_dir = os.path.join(res_specific_output_dir, "itemset_support_plots") # New subfolder for plots
        os.makedirs(itemset_plots_output_dir, exist_ok=True)

        res_df = all_archetype_df[all_archetype_df['resolution'] == res].copy()
        if res_df.empty: 
            print(f"  No data for resolution {res}. Skipping.")
            continue

        # Get all unique itemsets for this resolution
        unique_itemsets_in_res = res_df['itemset'].unique()

        if not unique_itemsets_in_res.size: # Check if the numpy array is empty
            print(f"  No unique itemsets found for resolution {res}. Skipping.")
            continue
        
        print(f"  Found {len(unique_itemsets_in_res)} unique non-singleton itemsets to analyze for resolution {res}.")
        
        all_itemset_summary_list_for_res = [] # To store summary data for all itemsets in this resolution

        for i, itemset in enumerate(unique_itemsets_in_res):
            if (i + 1) % 20 == 0: # Print progress for every 20 itemsets
                print(f"    Processing itemset {i+1}/{len(unique_itemsets_in_res)} for resolution {res}...")

            single_itemset_occurrences_df = res_df.loc[res_df['itemset'] == itemset].copy()
            
            if single_itemset_occurrences_df[timepoint_col].isnull().all(): 
                # This itemset has no timepoint data across all its occurrences, skip plotting its trend
                # print(f"    Skipping itemset {', '.join(sorted(list(itemset)))}, no timepoint data.") 
                # Still add basic info to summary if desired
                all_itemset_summary_list_for_res.append({
                    'itemset_str': ' & '.join(sorted(list(itemset))),
                    'num_markers': len(itemset),
                    'resolution': res,
                    'num_rois_found_in': single_itemset_occurrences_df['roi_string'].nunique(),
                    'overall_mean_support': single_itemset_occurrences_df['support'].mean(),
                    'timepoints_present': list(single_itemset_occurrences_df[timepoint_col].dropna().unique()),
                    'mean_support_range_over_time': np.nan,
                    'min_overall_mean_support_time': np.nan,
                    'max_overall_mean_support_time': np.nan,
                    'plotted': False
                })
                continue

            itemset_str_filename = "_".join(sorted(list(itemset))).replace('(', '').replace(')', '').replace('/', '_').replace(' ', '').replace("'", "")[:100]
            plt.figure(figsize=(12, 7) if use_replicates_in_plot else (10,6))
            title_str = f"Itemset: {', '.join(sorted(list(itemset)))}\nResolution: {res} (Found in {single_itemset_occurrences_df['roi_string'].nunique()} ROIs)"
            
            try:
                single_itemset_occurrences_df.loc[:, timepoint_col] = pd.to_numeric(single_itemset_occurrences_df[timepoint_col])
                sorted_timepoints = sorted(single_itemset_occurrences_df[timepoint_col].dropna().unique())
            except ValueError:
                sorted_timepoints = sorted(single_itemset_occurrences_df[timepoint_col].dropna().unique(), key=str)

            single_itemset_occurrences_df.loc[:, 'timepoint_plot_str'] = single_itemset_occurrences_df[timepoint_col].astype(str)
            sorted_timepoints_str = [str(tp) for tp in sorted_timepoints]
            
            single_itemset_occurrences_df.loc[:, 'timepoint_plot_str'] = pd.Categorical(
                single_itemset_occurrences_df['timepoint_plot_str'], 
                categories=sorted_timepoints_str, 
                ordered=True
            )
            df_to_plot = single_itemset_occurrences_df.sort_values(by='timepoint_plot_str')

            if use_replicates_in_plot and replicate_col in df_to_plot.columns and not df_to_plot[replicate_col].isnull().all():
                sns.lineplot(data=df_to_plot, x='timepoint_plot_str', y='support', hue=replicate_col, 
                              marker='o', errorbar=('ci', 95), legend='full') 
                plt.title(title_str, fontsize=10)
                handles, labels = plt.gca().get_legend_handles_labels()
                if len(labels) > 10 : 
                    plt.legend(title=replicate_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                elif labels: # Only show legend if there are labels (i.e., hue was effective)
                    plt.legend(title=replicate_col)
            else:
                agg_data = df_to_plot.groupby('timepoint_plot_str', observed=False)['support'].agg(['mean', 'std', 'count']).reset_index()
                plt.errorbar(agg_data['timepoint_plot_str'].astype(str), agg_data['mean'], yerr=agg_data['std'], fmt='-o', capsize=5, label="Mean Support (SD)")
                plt.title(title_str, fontsize=10)
                if not agg_data.empty : plt.legend()

            plt.xlabel(f"Timepoint ({timepoint_col})")
            plt.ylabel("Support")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            current_handles, current_labels = plt.gca().get_legend_handles_labels()
            if use_replicates_in_plot and len(current_labels) > 10 and current_labels: 
                plt.subplots_adjust(right=0.85) 
            
            plot_path = os.path.join(itemset_plots_output_dir, f"itemset_{itemset_str_filename}_res{res}_support.png")
            try: 
                plt.savefig(plot_path)
                plot_saved_flag = True
            except Exception as e_plot: 
                print(f"    ERROR saving plot {plot_path}: {e_plot}")
                plot_saved_flag = False
            plt.close()

            # Summary stats for this itemset
            # Group by the original timepoint_col for numerical consistency in summary if applicable
            mean_support_per_timepoint = df_to_plot.groupby(timepoint_col)['support'].mean()
            min_mean_support_val, max_mean_support_val, support_range_val = np.nan, np.nan, np.nan
            if not mean_support_per_timepoint.empty:
                min_mean_support_val = mean_support_per_timepoint.min()
                max_mean_support_val = mean_support_per_timepoint.max()
                support_range_val = max_mean_support_val - min_mean_support_val
            
            all_itemset_summary_list_for_res.append({
                'itemset_str': ' & '.join(sorted(list(itemset))),
                'num_markers': len(itemset),
                'resolution': res,
                'num_rois_found_in': df_to_plot['roi_string'].nunique(),
                'overall_mean_support': df_to_plot['support'].mean(),
                'timepoints_present': sorted(list(df_to_plot[timepoint_col].dropna().unique())),
                'mean_support_range_over_time': support_range_val,
                'min_overall_mean_support_time': min_mean_support_val,
                'max_overall_mean_support_time': max_mean_support_val,
                'plotted': plot_saved_flag
            })

        if all_itemset_summary_list_for_res:
            summary_df_res = pd.DataFrame(all_itemset_summary_list_for_res)
            summary_filename = os.path.join(res_specific_output_dir, f"all_itemsets_resolution_{res}_summary_stats.csv")
            try: 
                summary_df_res.to_csv(summary_filename, index=False)
                print(f"  Saved summary stats for all itemsets at res {res} to: {os.path.basename(summary_filename)}")
            except Exception as e_csv: 
                print(f"    ERROR saving summary CSV {summary_filename}: {e_csv}")
        else:
            print(f"  No itemsets processed or summarized for resolution {res}.")
        gc.collect()

    print("--- Itemset Temporal Dynamics Analysis Finished ---")
    # The cross-resolution summary of common itemsets is no longer relevant in this mode.
    # If you want a different cross-resolution summary (e.g. number of unique itemsets per res), that would be a new feature.


def load_all_adjacency_data(base_output_dir: str, config: Dict) -> Dict[Tuple[str, str], Dict[str, List[str]]]:
    """
    Loads all community adjacency JSON files from the experiment output directory.

    Args:
        base_output_dir: The base directory where ROI outputs are stored.
        config: The experiment configuration dictionary.

    Returns:
        A dictionary where keys are (roi_string, resolution_string) tuples,
        and values are the loaded adjacency data (dict mapping community_id to list of neighbors).
    """
    print("\n--- Loading Community Adjacency Data ---")
    # Glob pattern for community_adjacencies_*.json files
    # These files are expected directly within each resolution directory, not in community subdirs
    # e.g., base_output_dir/ROI_XXX/resolution_YYY/community_adjacencies_ROI_XXX_res_YYY.json
    adjacency_files = glob.glob(os.path.join(base_output_dir, "ROI_*", "resolution_*", "community_adjacencies_*.json"))

    if not adjacency_files:
        print(f"INFO: No community adjacency files found under {base_output_dir}/ROI_*/resolution_*/community_adjacencies_*.json")
        return {}

    all_adjacency_data = {}
    print(f"Found {len(adjacency_files)} community adjacency files to process.")

    for i, f_path in enumerate(adjacency_files):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(adjacency_files)} adjacency files...")
        try:
            # Extract ROI string from the file path (e.g., from directory ROI_XYZ)
            # _extract_standard_roi_from_filepath is designed to find the ROI_XXX part from the path.
            roi_str_from_path = _extract_standard_roi_from_filepath(f_path, config)
            if not roi_str_from_path:
                print(f"Warning: Could not extract a standard ROI string from path {f_path} for adjacency file. Skipping.")
                continue

            # Extract resolution string from the file path (e.g., from directory resolution_YYY)
            parts = f_path.split(os.sep)
            # Expected path: [... /ROI_XXX/resolution_YYY/community_adjacencies_....json]
            # Resolution dir is parts[-2]
            try:
                resolution_dir_name = parts[-2]
                if not resolution_dir_name.startswith("resolution_"):
                    print(f"Warning: Parent directory '{resolution_dir_name}' of adjacency file {f_path} does not match 'resolution_*'. Skipping.")
                    continue
                resolution_str_from_path = resolution_dir_name.replace("resolution_", "")
                # Clean resolution string (e.g., "0_5" to "0.5" or "1" to "1") to use as a consistent key
                cleaned_resolution_str = resolution_str_from_path.replace('_', '.')
                # Attempt to convert to float then to a standardized string, or keep as original if not possible
                try:
                    res_float = float(cleaned_resolution_str)
                    # Standardize: e.g. 0.5 -> "0.5", 1.0 -> "1"
                    if res_float.is_integer():
                        cleaned_resolution_str = str(int(res_float))
                    else:
                        cleaned_resolution_str = str(res_float) 
                except ValueError:
                    pass # Keep cleaned_resolution_str as is if not float convertible

            except IndexError:
                print(f"Warning: Path structure unexpected for adjacency file {f_path}. Cannot extract resolution. Skipping.")
                continue

            with open(f_path, 'r') as f:
                adj_data = json.load(f)
            
            # Key for the main dictionary: (roi_string, cleaned_resolution_string)
            # This ensures that if resolution was "0_5" in path and becomes "0.5" here, it's consistent.
            dict_key = (roi_str_from_path, cleaned_resolution_str)
            if dict_key in all_adjacency_data:
                print(f"Warning: Duplicate adjacency data found for key {dict_key} from file {f_path}. Overwriting.")
            all_adjacency_data[dict_key] = adj_data

        except json.JSONDecodeError as json_e:
            print(f"ERROR decoding JSON from adjacency file {f_path}: {json_e}")
        except Exception as e:
            print(f"ERROR processing adjacency file {f_path}: {e} {traceback.format_exc()}")

    print(f"Successfully loaded adjacency data for {len(all_adjacency_data)} ROI-resolution pairs.")
    return all_adjacency_data


def analyze_community_itemset_neighborhoods(
    all_archetype_df: pd.DataFrame, 
    all_adjacency_data: Dict[Tuple[str, str], Dict[str, List[str]]],
    experiment_output_dir: str, 
    config: Dict
):
    """
    Analyzes itemsets within communities and their neighborhoods.
    Identifies core and boundary itemsets.
    """
    print("\n--- Analyzing Community Itemset Neighborhoods ---")
    if all_archetype_df.empty:
        print("ERROR: Archetype DataFrame is empty. Cannot perform neighborhood analysis.")
        return
    if not all_adjacency_data:
        print("INFO: Adjacency data is empty. Neighborhood analysis will be limited (no neighbors to compare).")
        # We might still be able to process per-community itemsets without neighbor comparison.

    analysis_params = config.get('experiment_analysis', {}).get('neighborhood_analysis', {})
    core_threshold_neighbor_fraction = analysis_params.get('core_neighbor_threshold', 0.5) # Itemset in C, and in >50% of neighbors
    boundary_threshold_neighbor_fraction = analysis_params.get('boundary_neighbor_threshold', 0.5) # Itemset in C, and in <(1-0.5)% of neighbors (or vice versa)

    results = [] # To store dictionaries of results

    # Group archetype data by ROI and resolution for easier iteration
    # Ensure resolution in all_archetype_df is string to match all_adjacency_data keys if necessary.
    # load_all_archetype_data converts resolution to numeric. load_all_adjacency_data standardizes it to string.
    # For consistency, let's convert archetype_df resolution to the standardized string form used by adjacency data.
    
    # Standardize resolution in all_archetype_df to string for matching
    # all_archetype_df['resolution_str'] = all_archetype_df['resolution'].apply(lambda x: str(int(x)) if float(x).is_integer() else str(x))
    # The above was a bit simplistic. Let's use a more robust way if resolutions are float.
    def standardize_res_for_key(res_val):
        if pd.isna(res_val):
            return "nan_res"
        try:
            res_float = float(res_val)
            return str(int(res_float)) if res_float.is_integer() else str(res_float)
        except ValueError:
            return str(res_val) # Fallback to string if not float convertible

    all_archetype_df['resolution_key_str'] = all_archetype_df['resolution'].apply(standardize_res_for_key)

    grouped_archetypes = all_archetype_df.groupby(['roi_string', 'resolution_key_str'])

    print(f"Iterating through {len(grouped_archetypes)} ROI-resolution groups for neighborhood analysis...")
    processed_groups = 0

    for (roi, res_key_str), group_df in grouped_archetypes:
        processed_groups += 1
        if processed_groups % 10 == 0:
            print(f"  Processing group {processed_groups}/{len(grouped_archetypes)}: ROI {roi}, Resolution {res_key_str}")

        # Get adjacency for this specific ROI and resolution
        # The res_key_str from groupby should match the standardized string key in all_adjacency_data
        adj_key = (roi, res_key_str)
        current_adj = all_adjacency_data.get(adj_key)

        if current_adj is None:
            print(f"    Warning: No adjacency data found for {adj_key}. Skipping neighborhood analysis for this group.")
            # We could still process itemsets per community without neighbor comparison if needed.
            # For now, skip if no adjacencies.
            for community_id, community_df in group_df.groupby('community_id'):
                community_itemsets = set(community_df['itemset'])
                for item in community_itemsets:
                    support_val = community_df[community_df['itemset'] == item]['support'].iloc[0]
                    results.append({
                        'roi_string': roi,
                        'resolution': group_df['resolution'].iloc[0], # Original numeric resolution for output
                        'community_id': community_id,
                        'itemset': item,
                        'support_in_community': support_val,
                        'classification': 'no_neighbor_data',
                        'num_neighbors': 0,
                        'num_neighbors_with_itemset': 0
                    })
            continue

        # Iterate through each community in this ROI/resolution group
        for community_id, community_df in group_df.groupby('community_id'):
            community_id_str = str(community_id) # Ensure community_id is string for dict lookup in current_adj
            community_itemsets = set(community_df['itemset']) # Itemsets present in THIS community
            
            neighbor_community_ids = current_adj.get(community_id_str, [])
            num_neighbors = len(neighbor_community_ids)

            if not community_itemsets: # Skip if this community has no itemsets
                continue

            # For each itemset in the current community, check its presence in neighbors
            for item in community_itemsets:
                num_neighbors_with_itemset = 0
                if num_neighbors > 0:
                    for neighbor_id in neighbor_community_ids:
                        # Get itemsets for this specific neighbor community
                        # Neighbors are within the same ROI and Resolution (group_df)
                        neighbor_df = group_df[group_df['community_id'] == neighbor_id] # neighbor_id could be str or int
                        if neighbor_df.empty: # try converting neighbor_id if it was str from adj but int in df
                            try: neighbor_df = group_df[group_df['community_id'] == int(neighbor_id)]
                            except ValueError: pass # if neighbor_id is not int-like string
                       
                        if not neighbor_df.empty:
                            neighbor_itemsets = set(neighbor_df['itemset'])
                            if item in neighbor_itemsets:
                                num_neighbors_with_itemset += 1
               
                # Classify itemset based on presence in C and neighbors
                classification = 'undefined'
                if num_neighbors == 0:
                    classification = 'isolated_community' # Present in C, C has no neighbors
                else:
                    fraction_neighbors_with_itemset = num_neighbors_with_itemset / num_neighbors
                    # Core: in C, and in >= threshold% of neighbors
                    if fraction_neighbors_with_itemset >= core_threshold_neighbor_fraction:
                        classification = 'core'
                    # Boundary (C-specific): in C, and in < (1 - boundary_threshold_neighbor_fraction)% of neighbors
                    # e.g. boundary_threshold=0.75 -> item in C, and in <25% of neighbors
                    elif fraction_neighbors_with_itemset < (1.0 - boundary_threshold_neighbor_fraction):
                        classification = 'boundary_c_specific'
                    else:
                        classification = 'other' # In C, but doesn't meet core or c-specific boundary criteria
               
                support_val = community_df[community_df['itemset'] == item]['support'].iloc[0]
                results.append({
                    'roi_string': roi,
                    'resolution': group_df['resolution'].iloc[0], # Original numeric for output
                    'community_id': community_id,
                    'itemset': item,
                    'support_in_community': support_val,
                    'classification': classification,
                    'num_neighbors': num_neighbors,
                    'num_neighbors_with_itemset': num_neighbors_with_itemset
                })

            # Now, consider itemsets enriched in neighbors but NOT in C (Boundary Neighbor-Enriched)
            if num_neighbors > 0:
                all_neighbor_itemsets_union = set()
                itemset_counts_in_neighbors = {}
                for neighbor_id in neighbor_community_ids:
                    neighbor_df = group_df[group_df['community_id'] == neighbor_id]
                    if neighbor_df.empty:
                        try: neighbor_df = group_df[group_df['community_id'] == int(neighbor_id)]
                        except ValueError: continue
                   
                    if not neighbor_df.empty:
                        current_neighbor_itemsets = set(neighbor_df['itemset'])
                        all_neighbor_itemsets_union.update(current_neighbor_itemsets)
                        for item_in_neighbor in current_neighbor_itemsets:
                            itemset_counts_in_neighbors[item_in_neighbor] = itemset_counts_in_neighbors.get(item_in_neighbor, 0) + 1
               
                for item, count_in_neighbors in itemset_counts_in_neighbors.items():
                    if item not in community_itemsets: # Must be absent in C
                        fraction_neighbors_with_itemset = count_in_neighbors / num_neighbors
                        # Boundary (Neighbor-enriched): not in C, but in >= threshold% of neighbors
                        if fraction_neighbors_with_itemset >= boundary_threshold_neighbor_fraction:
                            results.append({
                                'roi_string': roi,
                                'resolution': group_df['resolution'].iloc[0],
                                'community_id': community_id,
                                'itemset': item,
                                'support_in_community': 0, # Not present in current community C
                                'classification': 'boundary_neighbor_enriched',
                                'num_neighbors': num_neighbors,
                                'num_neighbors_with_itemset': count_in_neighbors
                            })

    if not results:
        print("No neighborhood analysis results generated.")
        return None

    results_df = pd.DataFrame(results)
    print(f"Generated {len(results_df)} neighborhood analysis entries.")

    # Save the detailed results
    output_filename = os.path.join(experiment_output_dir, "community_itemset_neighborhood_analysis.csv")
    results_df.to_csv(output_filename, index=False)
    print(f"Neighborhood analysis results saved to: {output_filename}")

    # Further summarization or plotting can be added here
    # For example, summarizing how many core/boundary itemsets per community type, etc.

    return results_df


def summarize_itemset_spatial_roles(neighborhood_results_df: pd.DataFrame, experiment_output_dir: str, config: Dict):
    """
    Summarizes the spatial roles of itemsets based on neighborhood analysis.
    - Overall itemset role counts (core, boundary_c_specific, etc.)
    - Identifies promiscuous boundary markers.
    - Counts of roles per community.
    """
    print("\n--- Summarizing Itemset Spatial Roles ---")
    if neighborhood_results_df is None or neighborhood_results_df.empty:
        print("WARNING: Neighborhood results DataFrame is empty. Skipping spatial role summarization.")
        return

    # Ensure the output directory for these summaries exists
    spatial_roles_output_dir = os.path.join(experiment_output_dir, "itemset_spatial_role_summaries")
    os.makedirs(spatial_roles_output_dir, exist_ok=True)

    # For consistent string representation of frozensets in output CSVs
    def itemset_to_str(fs):
        if isinstance(fs, frozenset):
            return ' & '.join(sorted(list(fs)))
        return str(fs)
    neighborhood_results_df['itemset_str'] = neighborhood_results_df['itemset'].apply(itemset_to_str)

    # 1. Overall Itemset Role Counts
    print("  Calculating overall itemset role counts...")
    try:
        # Count how many times each itemset appears with each classification
        # This is across all ROIs, resolutions, and communities where the itemset was found
        itemset_role_counts = neighborhood_results_df.groupby(['itemset_str', 'classification']).size().reset_index(name='count')
        itemset_role_counts_pivot = itemset_role_counts.pivot_table(index='itemset_str', columns='classification', values='count', fill_value=0)
        itemset_role_counts_pivot.columns.name = None # Clean up pivot table column name
        itemset_role_counts_pivot = itemset_role_counts_pivot.reset_index()
        
        # Add total occurrences for each itemset
        itemset_role_counts_pivot['total_occurrences_in_analysis'] = itemset_role_counts_pivot.iloc[:, 1:].sum(axis=1)
        itemset_role_counts_pivot = itemset_role_counts_pivot.sort_values(by='total_occurrences_in_analysis', ascending=False)

        output_path = os.path.join(spatial_roles_output_dir, "itemset_overall_spatial_role_counts.csv")
        itemset_role_counts_pivot.to_csv(output_path, index=False)
        print(f"    Overall itemset spatial role counts saved to: {output_path}")
    except Exception as e:
        print(f"    ERROR calculating overall itemset role counts: {e}")
        traceback.print_exc()

    # 2. Promiscuous Boundary Markers
    print("  Identifying promiscuous boundary markers...")
    try:
        boundary_classifications = ['boundary_c_specific', 'boundary_neighbor_enriched']
        promiscuous_boundary_df = itemset_role_counts_pivot[
            itemset_role_counts_pivot['itemset_str'].isin(neighborhood_results_df['itemset_str']) # Ensure itemset_str is used
        ].copy() # Work on a copy

        # Sum counts for boundary classifications
        promiscuous_boundary_df['total_boundary_roles'] = 0
        for bc in boundary_classifications:
            if bc in promiscuous_boundary_df.columns:
                promiscuous_boundary_df['total_boundary_roles'] += promiscuous_boundary_df[bc]
        
        promiscuous_boundary_df = promiscuous_boundary_df.sort_values(by='total_boundary_roles', ascending=False)
        cols_to_keep = ['itemset_str', 'total_boundary_roles'] + [bc for bc in boundary_classifications if bc in promiscuous_boundary_df.columns] + ['total_occurrences_in_analysis']
        # Filter for existing columns before selecting
        existing_cols_to_keep = [col for col in cols_to_keep if col in promiscuous_boundary_df.columns]
        promiscuous_boundary_df_filtered = promiscuous_boundary_df[existing_cols_to_keep]
        
        output_path = os.path.join(spatial_roles_output_dir, "promiscuous_boundary_itemsets.csv")
        promiscuous_boundary_df_filtered.to_csv(output_path, index=False)
        print(f"    Promiscuous boundary itemset summary saved to: {output_path}")
    except Exception as e:
        print(f"    ERROR identifying promiscuous boundary markers: {e}")
        traceback.print_exc()

    # 3. Community-Level Summary (Counts of Roles)
    print("  Calculating community-level summary of itemset roles...")
    try:
        # For each community, count how many of its itemsets fall into each classification
        community_role_summary = neighborhood_results_df.groupby([
            'roi_string', 'resolution', 'community_id', 'classification'
        ]).size().reset_index(name='count_of_itemsets_in_role')
        
        community_role_summary_pivot = community_role_summary.pivot_table(
            index=['roi_string', 'resolution', 'community_id'], 
            columns='classification', 
            values='count_of_itemsets_in_role', 
            fill_value=0
        )
        community_role_summary_pivot.columns.name = None # Clean up
        community_role_summary_pivot = community_role_summary_pivot.reset_index()

        # Add total itemsets analyzed for that community
        community_role_summary_pivot['total_itemsets_in_community_context'] = community_role_summary_pivot.iloc[:, 3:].sum(axis=1)
        
        output_path = os.path.join(spatial_roles_output_dir, "community_summary_of_itemset_roles.csv")
        community_role_summary_pivot.to_csv(output_path, index=False)
        print(f"    Community-level summary of itemset roles saved to: {output_path}")
    except Exception as e:
        print(f"    ERROR calculating community-level summary of itemset roles: {e}")
        traceback.print_exc()

    print("--- Finished Summarizing Itemset Spatial Roles ---")


def analyze_temporal_dynamics_of_spatial_roles(neighborhood_results_df: pd.DataFrame, experiment_output_dir: str, config: Dict):
    """
    Analyzes the temporal dynamics of itemset spatial roles (core, boundary, etc.).
    Generates plots and summary data.
    Assumes neighborhood_results_df contains timepoint and replicate information if configured.
    """
    print("\n--- Analyzing Temporal Dynamics of Itemset Spatial Roles ---")
    if neighborhood_results_df is None or neighborhood_results_df.empty:
        print("WARNING: Neighborhood results DataFrame is empty. Skipping temporal analysis of spatial roles.")
        return

    exp_config = config.get('experiment_analysis', {})
    timepoint_col = exp_config.get('timepoint_col')
    replicate_col = exp_config.get('replicate_col') # Optional
    # condition_col = exp_config.get('condition_col') # Optional, for further stratification

    if not timepoint_col or timepoint_col not in neighborhood_results_df.columns:
        print(f"ERROR: Timepoint column '{timepoint_col}' not found in neighborhood_results_df. Cannot perform temporal analysis.")
        return

    # Create output directory for these temporal analyses
    temporal_spatial_roles_output_dir = os.path.join(experiment_output_dir, "temporal_itemset_spatial_role_analysis")
    os.makedirs(temporal_spatial_roles_output_dir, exist_ok=True)

    # Convert itemset frozenset to string for grouping and display, if not already done
    if 'itemset_str' not in neighborhood_results_df.columns:
        def itemset_to_str_for_plot(fs):
            if isinstance(fs, frozenset):
                return ' & '.join(sorted(list(fs)))
            return str(fs)
        neighborhood_results_df['itemset_str'] = neighborhood_results_df['itemset'].apply(itemset_to_str_for_plot)

    # Ensure timepoint column is numeric for proper sorting and plotting
    try:
        neighborhood_results_df[timepoint_col] = pd.to_numeric(neighborhood_results_df[timepoint_col])
        # Create a categorical version for ordered plotting if needed, similar to analyze_itemset_temporal_dynamics
        sorted_timepoints = sorted(neighborhood_results_df[timepoint_col].dropna().unique())
        neighborhood_results_df['timepoint_plot_str'] = pd.Categorical(
            neighborhood_results_df[timepoint_col].astype(str), # Ensure it's string for categories
            categories=[str(tp) for tp in sorted_timepoints],
            ordered=True
        )
    except ValueError as e:
        print(f"Warning: Could not convert timepoint column '{timepoint_col}' to numeric: {e}. Plotting order might be string-based.")
        neighborhood_results_df['timepoint_plot_str'] = neighborhood_results_df[timepoint_col].astype(str)
        # Fallback to string sort for categories
        sorted_timepoints_str = sorted(neighborhood_results_df['timepoint_plot_str'].dropna().unique())
        neighborhood_results_df['timepoint_plot_str'] = pd.Categorical(
            neighborhood_results_df['timepoint_plot_str'], 
            categories=sorted_timepoints_str, 
            ordered=True
        )

    # --- Analysis 1: Average number of itemsets in each spatial role per community, over time --- 
    print("  Analyzing average number of itemsets in spatial roles per community over time...")
    try:
        # First, count itemsets per role for each community
        # Columns: roi_string, resolution, community_id, classification, itemset_str, ... timepoint_col, replicate_col
        community_role_counts = neighborhood_results_df.groupby([
            'roi_string', 'resolution', 'community_id', 'classification', timepoint_col, 'timepoint_plot_str' # Add timepoint here
        ] + ([replicate_col] if replicate_col and replicate_col in neighborhood_results_df.columns else [])).size().reset_index(name='num_itemsets_in_role')

        # Now, average these counts per timepoint (and optionally replicate)
        grouping_vars = ['timepoint_plot_str', 'classification']
        plot_hue = None
        if replicate_col and replicate_col in community_role_counts.columns:
            if community_role_counts[replicate_col].nunique() > 1:
                 grouping_vars.append(replicate_col)
                 plot_hue = replicate_col
        
        # Calculate mean number of itemsets per role, per community (by averaging over communities at each timepoint/classification/replicate)
        # This requires knowing how many communities were present at each timepoint to average correctly, or average the counts directly.
        # Simpler: Average the count of itemsets in a role *across communities* that exist at that timepoint.
        avg_role_counts_over_time = community_role_counts.groupby(grouping_vars)['num_itemsets_in_role'].mean().reset_index()
        avg_role_counts_over_time = avg_role_counts_over_time.rename(columns={'num_itemsets_in_role': 'avg_num_itemsets_per_community_in_role'})

        output_csv_path = os.path.join(temporal_spatial_roles_output_dir, "temporal_avg_itemset_roles_per_community.csv")
        avg_role_counts_over_time.to_csv(output_csv_path, index=False)
        print(f"    Saved average itemset roles per community over time to: {os.path.basename(output_csv_path)}")

        # Plotting this data
        plt.figure(figsize=(14, 8))
        sns.lineplot(
            data=avg_role_counts_over_time,
            x='timepoint_plot_str',
            y='avg_num_itemsets_per_community_in_role',
            hue='classification',
            style=plot_hue, # Use replicate for style if available, otherwise no style
            marker='o', errorbar=None # No error bars for this direct average of counts
        )
        plt.title(f"Average Number of Itemsets in Spatial Roles per Community Over Time")
        plt.xlabel(f"Timepoint ({timepoint_col})")
        plt.ylabel("Avg. Itemsets per Community in Role")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Classification & Replicate' if plot_hue else 'Classification', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1] if plot_hue or avg_role_counts_over_time['classification'].nunique() > 5 else None)
        plot_path = os.path.join(temporal_spatial_roles_output_dir, "plot_temporal_avg_itemset_roles.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"    Plot saved to: {os.path.basename(plot_path)}")

    except Exception as e:
        print(f"    ERROR during temporal analysis of average spatial roles: {e}")
        traceback.print_exc()

    # --- Analysis 2: Temporal trend for specific itemsets' spatial roles --- (e.g. top N boundary markers)
    print("  Analyzing temporal dynamics of specific itemsets' spatial roles...")
    try:
        # Use the previously generated promiscuous_boundary_itemsets.csv or itemset_overall_spatial_role_counts.csv to pick itemsets
        # For this example, let's find top N overall most frequent itemsets from neighborhood_results_df to track their roles.
        if 'itemset_str' not in neighborhood_results_df.columns:
             neighborhood_results_df['itemset_str'] = neighborhood_results_df['itemset'].apply(itemset_to_str_for_plot)

        top_n_itemsets = config.get('experiment_analysis',{}).get('neighborhood_analysis',{}).get('num_top_itemsets_to_track_temporally', 5)
        if top_n_itemsets > 0:
            itemset_occurrence_counts = neighborhood_results_df['itemset_str'].value_counts().nlargest(top_n_itemsets).index.tolist()
            print(f"    Tracking temporal role changes for top {len(itemset_occurrence_counts)} most frequent itemsets: {itemset_occurrence_counts}")

            if itemset_occurrence_counts:
                selected_itemsets_df = neighborhood_results_df[neighborhood_results_df['itemset_str'].isin(itemset_occurrence_counts)]
                
                # Count communities where each selected itemset has a specific role, at each timepoint
                itemset_role_dynamics = selected_itemsets_df.groupby([
                    'timepoint_plot_str', 'itemset_str', 'classification'
                ] + ([replicate_col] if replicate_col and replicate_col in selected_itemsets_df.columns else [])).size().reset_index(name='num_communities_with_role')
                
                output_csv_path_specific = os.path.join(temporal_spatial_roles_output_dir, "temporal_specific_itemset_role_dynamics.csv")
                itemset_role_dynamics.to_csv(output_csv_path_specific, index=False)
                print(f"    Saved temporal role dynamics for specific itemsets to: {os.path.basename(output_csv_path_specific)}")

                # Plotting for each tracked itemset
                for item_s in itemset_occurrence_counts:
                    plot_df_item = itemset_role_dynamics[itemset_role_dynamics['itemset_str'] == item_s]
                    if plot_df_item.empty:
                        continue
                    
                    plt.figure(figsize=(12, 7))
                    sns.lineplot(
                        data=plot_df_item,
                        x='timepoint_plot_str',
                        y='num_communities_with_role',
                        hue='classification',
                        style=plot_hue, # if replicate_col is used
                        marker='o'
                    )
                    plt.title(f"Temporal Dynamics of Spatial Roles for Itemset: \n{item_s}")
                    plt.xlabel(f"Timepoint ({timepoint_col})")
                    plt.ylabel("Number of Communities with Role")
                    plt.xticks(rotation=45, ha='right')
                    plt.legend(title='Classification & Replicate' if plot_hue else 'Classification', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout(rect=[0, 0, 0.85, 1] if plot_hue or plot_df_item['classification'].nunique() > 5 else None)
                    plot_filename_item = f"plot_temporal_roles_{item_s.replace(' & ', '_').replace(' ', '')[:50]}.png"
                    plot_path_item = os.path.join(temporal_spatial_roles_output_dir, plot_filename_item)
                    plt.savefig(plot_path_item)
                    plt.close()
                    print(f"      Plot for itemset '{item_s}' saved to: {os.path.basename(plot_filename_item)}")
        else:
            print("    Skipping tracking of specific itemsets as num_top_itemsets_to_track_temporally is 0 or less.")

    except Exception as e:
        print(f"    ERROR during temporal analysis of specific itemsets' spatial roles: {e}")
        traceback.print_exc()

    print("--- Finished Temporal Dynamics of Itemset Spatial Roles ---")


# --- Main execution block ---
def main():
    print("--- Starting Experiment Level Analysis ---")
    config = load_config("config.yaml")
    if config is None:
        sys.exit(1)

    base_output_dir = config['paths']['output_dir']
    exp_summary_subdir = config.get('experiment_analysis', {}).get('output_subdir', 'experiment_summary')
    experiment_output_dir = os.path.join(base_output_dir, exp_summary_subdir)
    os.makedirs(experiment_output_dir, exist_ok=True)
    print(f"Experiment analysis output directory: {experiment_output_dir}")

    metadata_df = load_metadata(config)
    # ... (metadata loading messages)

    all_archetype_data_df = load_all_archetype_data(base_output_dir, metadata_df, config)
    if all_archetype_data_df is None or all_archetype_data_df.empty:
        print("CRITICAL ERROR: Failed to load any archetype data. Aborting.")
        sys.exit(1)

    all_adjacency_data_map = load_all_adjacency_data(base_output_dir, config)
    # ... (adjacency loading messages)

    neighborhood_results_df = analyze_community_itemset_neighborhoods(
        all_archetype_data_df,
        all_adjacency_data_map,
        experiment_output_dir,
        config
    )

    if neighborhood_results_df is not None and not neighborhood_results_df.empty:
        print("Neighborhood analysis completed. Proceeding to summarize and analyze temporal dynamics of spatial roles.")
        summarize_itemset_spatial_roles(neighborhood_results_df, experiment_output_dir, config)
        analyze_temporal_dynamics_of_spatial_roles(neighborhood_results_df, experiment_output_dir, config)
    else:
        print("Neighborhood analysis did not produce results or was skipped. Skipping further spatial role analyses.")

    print("\n--- Experiment Level Analysis Script Finished ---")

if __name__ == '__main__':
    main()
