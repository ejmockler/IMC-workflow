import os
import re
import glob
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import yaml # Added for standalone testing block

def load_metadata(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Loads the metadata file specified in the configuration.

    Args:
        config: The configuration dictionary.

    Returns:
        A pandas DataFrame containing the metadata, or None if loading fails.
    """
    print("--- Loading Metadata --- ")
    metadata_file = config.get('paths', {}).get('metadata_file')
    roi_col_name = config.get('experiment_analysis', {}).get('metadata_roi_col')

    if not metadata_file:
        print("ERROR: 'metadata_file' path not found in config['paths'].")
        return None
    # Allow empty string for metadata_file path, check existence later
    # else:
    #    print(f"Metadata file path specified: {metadata_file}")

    if not roi_col_name:
        print("ERROR: 'metadata_roi_col' not found in config['experiment_analysis'].")
        return None

    if not metadata_file or not os.path.exists(metadata_file):
        print(f"ERROR: Metadata file not found at specified path: '{metadata_file}'")
        return None

    try:
        # Assuming CSV for now, might need to add options for Excel later
        metadata_df = pd.read_csv(metadata_file)
        print(f"Successfully loaded metadata from: {metadata_file}")

        if roi_col_name not in metadata_df.columns:
            print(f"ERROR: Specified ROI column '{roi_col_name}' not found in metadata columns:")
            print(f"   {metadata_df.columns.tolist()}")
            return None

        print(f"   Found ROI column: '{roi_col_name}'")
        print(f"   Metadata shape: {metadata_df.shape}")
        return metadata_df

    except pd.errors.EmptyDataError:
        print(f"ERROR: Metadata file {metadata_file} is empty.")
        return None
    except Exception as e:
        print(f"ERROR: Failed to read metadata file {metadata_file}: {e}")
        return None

def find_roi_output_files(config: Dict[str, Any], file_pattern: str) -> List[str]:
    """Scans the output directory for ROI subdirectories and finds specific output files.

    Args:
        config: The configuration dictionary.
        file_pattern: A glob pattern for the files to find within each ROI's
                      resolution subdirectory (e.g., "community_profiles_*.csv").

    Returns:
        A list of full paths to the found files.
    """
    base_output_dir = config.get('paths', {}).get('output_dir')
    resolution = config.get('experiment_analysis', {}).get('resolution_to_aggregate')

    if not base_output_dir or not os.path.isdir(base_output_dir):
        print(f"ERROR: Base output directory '{base_output_dir}' not found or not a directory.")
        return []
    if resolution is None:
        print("ERROR: 'resolution_to_aggregate' not specified in config['experiment_analysis'].")
        return []

    # Format resolution directory name (handle float precision)
    res_str = f"{resolution:.3f}".rstrip('0').rstrip('.').replace('.', '_') if isinstance(resolution, float) else str(resolution)
    resolution_subdir_name = f"resolution_{res_str}"
    print(f"Searching for pattern '{file_pattern}' in resolution subdirs: '{resolution_subdir_name}'")

    found_files = []
    # Find potential ROI directories (simple pattern, might need refinement)
    roi_dirs = glob.glob(os.path.join(base_output_dir, "ROI_*"))

    for roi_dir in roi_dirs:
        if os.path.isdir(roi_dir):
            resolution_path = os.path.join(roi_dir, resolution_subdir_name)
            if os.path.isdir(resolution_path):
                search_path = os.path.join(resolution_path, file_pattern)
                matching_files = glob.glob(search_path)
                if matching_files:
                    # Typically expect only one file per type per ROI/resolution
                    if len(matching_files) > 1:
                         print(f"   Warning: Found multiple files matching pattern '{file_pattern}' in {resolution_path}. Using first: {os.path.basename(matching_files[0])}")
                    found_files.append(matching_files[0])
            # else: print(f"   Debug: Resolution subdir {resolution_subdir_name} not found in {roi_dir}") # Optional Debug
        # else: print(f"   Debug: {roi_dir} is not a directory") # Optional Debug

    print(f"Found {len(found_files)} files matching the pattern.")
    return found_files

def load_aggregate_community_profiles(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Loads and aggregates community profile CSVs from all ROIs for the target resolution.

    Args:
        config: The configuration dictionary.

    Returns:
        A pandas DataFrame containing aggregated community profiles with an added 'roi_string' column,
        or None if loading fails.
    """
    print("\n--- Loading and Aggregating Community Profiles --- ")
    resolution = config.get('experiment_analysis', {}).get('resolution_to_aggregate')
    if resolution is None:
        print("ERROR: 'resolution_to_aggregate' not specified.")
        return None

    # Define the pattern - adjusted to match ROI pipeline output
    # Example target filename: community_profiles_scaled_ROI_D1_M1_03_11_res_0.3.csv
    res_str_filename = f"{resolution:.3f}".rstrip('0').rstrip('.') if isinstance(resolution, float) else str(resolution)
    file_pattern = f"community_profiles_scaled_*_res_{res_str_filename}.csv" # Corrected order

    profile_files = find_roi_output_files(config, file_pattern)

    if not profile_files:
        print(f"No community profile files found for pattern '{file_pattern}'.")
        return None

    all_profiles = []
    processed_files_count = 0
    skipped_files_count = 0

    # Regex to extract ROI string (e.g., ROI_D1_M1_03_11)
    roi_pattern = re.compile(r'(ROI_\w+_\d+_\d+)')

    for f_path in profile_files:
        filename = os.path.basename(f_path)
        match = roi_pattern.search(filename)
        if not match:
            print(f"   Warning: Could not extract ROI string from filename '{filename}'. Skipping file.")
            skipped_files_count += 1
            continue

        roi_string = match.group(1)

        try:
            df = pd.read_csv(f_path)
            # Assuming the first column is the community ID from the CSV
            if df.columns[0].lower() == 'community':
                 df = df.rename(columns={df.columns[0]: 'community'})
                 # Do not set index yet, keep community as a column
            elif 'community' in df.columns:
                 pass # Keep community as a column
            else:
                 print(f"   Warning: 'community' column not found as index or column in {filename}. Assuming first column is community ID.")
                 df = df.rename(columns={df.columns[0]: 'community'})

            df['roi_string'] = roi_string
            df['resolution'] = resolution # Add resolution info
            all_profiles.append(df)
            processed_files_count += 1
        except Exception as e:
            print(f"   ERROR: Failed to load or process profile file {filename}: {e}")
            skipped_files_count += 1

    if not all_profiles:
        print("ERROR: No profile files were successfully loaded and processed.")
        return None

    # Concatenate all dataframes
    try:
        aggregated_df = pd.concat(all_profiles, ignore_index=True)
        print(f"Successfully processed {processed_files_count} profile files.")
        if skipped_files_count > 0:
            print(f"Skipped {skipped_files_count} files due to errors or missing ROI string.")
        print(f"   Aggregated profiles shape: {aggregated_df.shape}")
        return aggregated_df
    except Exception as e:
        print(f"ERROR: Failed to concatenate profile dataframes: {e}")
        return None

def load_aggregate_pixel_results(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Loads and aggregates annotated pixel result CSVs from all ROIs for the target resolution.
       These files contain per-pixel asinh-scaled values, community labels, and primary channel assignments.

    Args:
        config: The configuration dictionary.

    Returns:
        A pandas DataFrame containing aggregated pixel results with an added 'roi_string' column,
        or None if loading fails. Returns only essential columns for abundance calculation.
    """
    print("\n--- Loading and Aggregating Pixel Results (for Abundance) --- ")
    resolution = config.get('experiment_analysis', {}).get('resolution_to_aggregate')
    if resolution is None:
        print("ERROR: 'resolution_to_aggregate' not specified.")
        return None

    # Define the pattern for final pixel result files
    res_str_filename = f"{resolution:.3f}".rstrip('0').rstrip('.') if isinstance(resolution, float) else str(resolution)
    file_pattern = f"pixel_results_annotated_*_res_{res_str_filename}.csv"

    pixel_result_files = find_roi_output_files(config, file_pattern)

    if not pixel_result_files:
        print(f"No pixel result files found for pattern '{file_pattern}'.")
        return None

    all_pixel_data = []
    processed_files_count = 0
    skipped_files_count = 0

    # Regex to extract ROI string
    roi_pattern = re.compile(r'(ROI_\w+_\d+_\d+)')

    # Define essential columns to keep for abundance calculation
    columns_to_keep = ['community']

    for f_path in pixel_result_files:
        filename = os.path.basename(f_path)
        match = roi_pattern.search(filename)
        if not match:
            print(f"   Warning: Could not extract ROI string from filename '{filename}'. Skipping file.")
            skipped_files_count += 1
            continue

        roi_string = match.group(1)

        try:
            # Load only necessary columns if possible (more efficient for large files)
            # Check if 'community' exists first by loading header or small chunk?
            # For simplicity now, load then select. Optimize later if needed.
            df = pd.read_csv(f_path)

            # Find the community column (handle case variations)
            community_col = None
            for col in df.columns:
                if col.lower() == 'community':
                    community_col = col
                    break

            if community_col is None:
                 print(f"   Warning: 'community' column not found in {filename}. Skipping file.")
                 skipped_files_count += 1
                 continue

            # Keep only the community column (and implicitly the index/pixel count)
            df_subset = df[[community_col]].copy()
            df_subset.rename(columns={community_col: 'community'}, inplace=True) # Standardize name
            df_subset['roi_string'] = roi_string
            df_subset['resolution'] = resolution
            all_pixel_data.append(df_subset)
            processed_files_count += 1
            # Potential memory optimization: process counts per file instead of concatenating all pixels
        except Exception as e:
            print(f"   ERROR: Failed to load or process pixel file {filename}: {e}")
            skipped_files_count += 1

    if not all_pixel_data:
        print("ERROR: No pixel files were successfully loaded and processed for abundance calculation.")
        return None

    # Concatenate all dataframes
    try:
        aggregated_df = pd.concat(all_pixel_data, ignore_index=True)
        print(f"Successfully processed {processed_files_count} pixel files for abundance data.")
        if skipped_files_count > 0:
            print(f"Skipped {skipped_files_count} files due to errors or missing ROI/community column.")
        print(f"   Aggregated pixel data shape (pixels x columns): {aggregated_df.shape}")
        # Note: This df contains one row per pixel across all ROIs.
        return aggregated_df
    except Exception as e:
        print(f"ERROR: Failed to concatenate pixel result dataframes: {e}")
        return None

def aggregate_and_merge_data(config: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Main function to load metadata and aggregate specified data types, then merge them.

    Currently aggregates and merges Community Profiles.
    Also aggregates Pixel Results (but doesn't merge them yet in this example).

    Returns:
        A tuple containing:
        - merged_profiles_df: DataFrame of profiles merged with metadata (or None).
        - aggregated_pixels_df: DataFrame of aggregated pixel data (roi_string, community) (or None).
    """
    print("\n================ Aggregating and Merging Data ================")
    # 1. Load Metadata
    metadata_df = load_metadata(config)
    if metadata_df is None:
        return None, None # Return tuple

    # 2. Load Aggregated Profiles
    agg_profiles_df = load_aggregate_community_profiles(config)
    if agg_profiles_df is None:
        # If profiles fail, maybe still proceed with pixels? Or fail fast? Failing fast for now.
        print("Failed to load aggregate profiles.")
        return None, None

    # 3. Load Aggregated Pixel Data (for abundance)
    agg_pixels_df = load_aggregate_pixel_results(config)
    # We might allow this to fail and still return profiles? For now, keep it simple.
    if agg_pixels_df is None:
         print("Failed to load aggregate pixel results.")
         # Decide if this is fatal - returning profiles still if they loaded
         # return agg_profiles_df, None # Option 1: Return profiles even if pixels failed
         return None, None          # Option 2: Fail if any aggregation fails


    # --- Merge Profiles with Metadata ---
    print("\n--- Merging Aggregated Profiles with Metadata --- ")
    roi_col_name = config.get('experiment_analysis', {}).get('metadata_roi_col')
    merged_profiles_df = None # Initialize

    # Check if necessary columns exist before attempting merge
    if agg_profiles_df is None or 'roi_string' not in agg_profiles_df.columns:
        print("ERROR: Aggregated profiles missing or 'roi_string' column not found.")
        return None, agg_pixels_df # Return pixels if they exist
    if metadata_df is None or roi_col_name not in metadata_df.columns:
        print(f"ERROR: Metadata missing or ROI column '{roi_col_name}' not found.")
        return agg_profiles_df, agg_pixels_df # Return profiles if they exist

    try:
        # --- Flexible Merge Logic (using str.contains) ---
        metadata_df[roi_col_name] = metadata_df[roi_col_name].astype(str)
        agg_profiles_df['roi_string'] = agg_profiles_df['roi_string'].astype(str)

        merged_data_list = []
        unique_roi_strings = agg_profiles_df['roi_string'].unique()
        processed_roi_strings = set()
        skipped_roi_strings_no_match = set()
        skipped_roi_strings_multi_match = set()

        print(f"Attempting flexible merge: Finding rows in metadata['{roi_col_name}'] that contain profile 'roi_string'...")
        for roi_str in unique_roi_strings:
            # Use regex=False for literal substring matching
            matching_meta_rows = metadata_df[metadata_df[roi_col_name].str.contains(roi_str, na=False, regex=False)]

            if len(matching_meta_rows) == 1:
                # Perform merge for this specific roi_str
                roi_profile_data = agg_profiles_df[agg_profiles_df['roi_string'] == roi_str]
                # Use pd.merge with a temporary key or cross merge approach
                # Simple cross merge then filter (might be slow for large data)
                # temp_merged = roi_profile_data.merge(matching_meta_rows, how='cross')
                # Or assign values directly (simpler if only one meta row)
                meta_row_values = matching_meta_rows.iloc[0]
                temp_merged = roi_profile_data.assign(**{col: meta_row_values[col] for col in metadata_df.columns if col != roi_col_name})

                merged_data_list.append(temp_merged)
                processed_roi_strings.add(roi_str)
            elif len(matching_meta_rows) == 0:
                skipped_roi_strings_no_match.add(roi_str)
            else:
                skipped_roi_strings_multi_match.add(roi_str)
                # print(f"   Warning: Multiple metadata entries found for '{roi_str}': {matching_meta_rows[roi_col_name].tolist()}")

        if merged_data_list:
            merged_profiles_df = pd.concat(merged_data_list, ignore_index=True)
            print(f"Flexible profile merge successful. Shape: {merged_profiles_df.shape}")
            print(f"   Processed {len(processed_roi_strings)} unique ROI strings.")
        else:
            print("ERROR: Flexible profile merge failed to combine any data.")
            merged_profiles_df = None

        if skipped_roi_strings_no_match:
             print(f"   Warning: {len(skipped_roi_strings_no_match)} ROI strings found no match in metadata (e.g., '{next(iter(skipped_roi_strings_no_match))}').")
        if skipped_roi_strings_multi_match:
             print(f"   Warning: {len(skipped_roi_strings_multi_match)} ROI strings found multiple matches in metadata (e.g., '{next(iter(skipped_roi_strings_multi_match))}'). Profiles skipped.")

    except Exception as e:
        print(f"ERROR: An unexpected error occurred during flexible profile merge: {e}")
        import traceback
        traceback.print_exc()
        merged_profiles_df = None # Ensure it's None on error

    # --- Merging Pixel data --- (Keep returning raw aggregated pixels)
    # ... (rest of function remains the same) ...

    # Return results
    if merged_profiles_df is None or agg_pixels_df is None:
        print("Warning: Either profile merging or pixel aggregation failed.")
        # Return whatever succeeded, potentially None, None
        return merged_profiles_df, agg_pixels_df
    else:
        return merged_profiles_df, agg_pixels_df

# Example usage block (would be called from run_experiment_analysis.py)
if __name__ == '__main__':
    # This block allows testing this script directly
    # Assumes config.yaml is in the parent directory
    CONFIG_PATH = "../config.yaml" # Adjust path as needed relative to src/experiment_pipeline

    def load_config_main(config_path=CONFIG_PATH):
        if not os.path.exists(config_path):
            print(f"ERROR: Config file not found at {config_path} for testing.")
            return None
        try:
            with open(config_path, 'r') as f:
                conf = yaml.safe_load(f)
            print(f"Config loaded for testing from: {config_path}")
            return conf
        except Exception as e:
            print(f"Error loading config {config_path} for testing: {e}")
            return None

    config = load_config_main()
    if config:
        # Test the merging function - now returns a tuple
        merged_profiles, aggregated_pixels = aggregate_and_merge_data(config) # Updated call

        if merged_profiles is not None:
            print("\n--- Example Merged Profile Data (First 5 rows) --- ")
            print(merged_profiles.head())
            print("Shape:", merged_profiles.shape)
        else:
            print("\n--- Profile aggregation or merge failed. ---")

        if aggregated_pixels is not None:
             print("\n--- Example Aggregated Pixel Data (First 5 rows) --- ")
             print(aggregated_pixels.head())
             print("Shape:", aggregated_pixels.shape)
             # Note: This can be very large!
        else:
             print("\n--- Pixel aggregation failed. ---")

    else:
        print("Could not load config for testing.")
