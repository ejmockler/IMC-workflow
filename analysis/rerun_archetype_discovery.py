import sys
import os

# Add the project root to sys.path
# __file__ is analysis/rerun_archetype_discovery.py
# os.path.dirname(__file__) is analysis/
# os.path.join(os.path.dirname(__file__), '..') is analysis/../ which resolves to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import glob
import argparse
import pandas as pd # Import pandas for potential use with archetype DFs
from src.roi_pipeline.archetype_utils import discover_channel_archetypes # cluster_archetypes_and_save_linkage removed
import traceback # For more detailed error logging

def load_config(config_path: str = "config.yaml"):
    """Loads the pipeline configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from: {config_path}")
        if not isinstance(config, dict):
            print(f"ERROR: Config file {config_path} is not a valid dictionary.")
            return None
        return config
    except FileNotFoundError:
        print(f"ERROR: Config file not found at {config_path}")
        return None
    except Exception as e:
        print(f"ERROR: Loading config {config_path}: {e}")
        return None

def find_community_pixel_files(base_output_dir, target_roi=None, target_resolution=None, config=None):
    """
    Finds persisted per-community pixel data CSV files intended for archetype discovery reruns.
    Searches for files like 'community_{id}_pixels_for_archetypes.csv'.
    """
    # Construct search path segments
    roi_segment = target_roi if target_roi else "ROI_*"
    
    # Resolution segment can be tricky due to formats like "0_5" vs "0.5"
    # If target_resolution is provided, try to match it directly.
    # If run_roi_analysis.py saves resolution dirs like "resolution_0_5" or "resolution_1",
    # we need to match that.
    if target_resolution:
        # Allow for resolution_X or resolution_X_Y formats
        resolution_segment = f"resolution_{target_resolution.replace('.', '_')}"
    else:
        resolution_segment = "resolution_*"

    community_segment = "community_*"
    # Filename to search for, as saved by run_roi_analysis.py when persisting these files
    pixel_file_pattern = "community_*_pixels_for_archetypes.csv"

    # Construct the full search pattern
    # Example: /base_output_dir/ROI_XYZ/resolution_0_5/community_123/community_123_pixels_for_archetypes.csv
    search_pattern_parts = [base_output_dir, roi_segment, resolution_segment, community_segment, pixel_file_pattern]
    full_search_path = os.path.join(*[part for part in search_pattern_parts if part]) # Filter None if any
    
    print(f"Searching for per-community pixel files with pattern: {full_search_path}")
    # recursive=True might be too broad if the structure is fixed.
    # Let's assume a fairly fixed depth for these specific files.
    # glob.glob doesn't handle ** well unless recursive is True.
    # The pattern above is specific enough that recursive should be fine.
    found_files = glob.glob(full_search_path, recursive=True) 
    
    # Additional filtering if target_roi or target_resolution was less specific ("*")
    # to ensure paths strictly match if provided.
    # This is often handled well by glob if the pattern is specific enough, but double check.
    # For instance, if target_roi = "ROI_A" and found_files includes "/ROI_A_extra/...", it should be filtered.
    # The current glob pattern should be specific enough for roi_segment and resolution_segment when not "*".

    print(f"Found {len(found_files)} persisted per-community pixel files for archetype rerun.")
    return found_files

def main():
    parser = argparse.ArgumentParser(description="Rerun channel archetype discovery on persisted per-community pixel files.")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--roi", help="Specific ROI string to process (e.g., ROI_D7_M1_03_23). If not set, processes all found.")
    parser.add_argument("--resolution", help="Specific resolution to process (e.g., '100' or '0.5' or '0_5'). If not set, processes all found for the ROI(s). Resolution format should match directory names (e.g., '0_5' for 'resolution_0_5').")
    parser.add_argument("--min_support", type=float, help="Override min_support from config for archetype discovery.")
    parser.add_argument("--bin_percentile", type=float, help="Override binarization_percentile from config (0.0-1.0) for archetype discovery.")
    
    args = parser.parse_args()

    config = load_config(args.config)
    if config is None:
        sys.exit(1)

    output_base_dir = config.get('paths', {}).get('output_dir')
    if not output_base_dir:
        print("ERROR: 'paths.output_dir' not found in config.")
        sys.exit(1)

    archetype_config_params = config.get('analysis', {}).get('archetype_discovery', {})
    min_support_default = archetype_config_params.get('min_support', 0.05)
    bin_percentile_default = archetype_config_params.get('binarization_percentile', 0.75)

    min_support_rerun = args.min_support if args.min_support is not None else min_support_default
    bin_percentile_rerun = args.bin_percentile if args.bin_percentile is not None else bin_percentile_default

    if not (0 < bin_percentile_rerun < 1 if bin_percentile_rerun is not None else True): # Check only if provided
        print(f"ERROR: binarization_percentile must be between 0 and 1 (exclusive), got {bin_percentile_rerun}")
        sys.exit(1)
    if not (min_support_rerun > 0 if min_support_rerun is not None else True): # Check only if provided
        print(f"ERROR: min_support must be greater than 0, got {min_support_rerun}")
        sys.exit(1)

    print(f"Using parameters for rerun: min_support = {min_support_rerun}, binarization_percentile = {bin_percentile_rerun}")
    
    # Use the new function to find per-community pixel files
    community_pixel_csvs = find_community_pixel_files(output_base_dir, args.roi, args.resolution, config)

    if not community_pixel_csvs:
        print("No persisted per-community pixel files found matching criteria. Exiting.")
        print("Ensure 'save_community_pixel_data_for_rerun: true' was set in config.yaml during run_roi_analysis.py execution.")
        sys.exit(0)

    success_count = 0
    failure_count = 0

    for community_file_path in community_pixel_csvs:
        print(f"\nProcessing community pixel file: {community_file_path}")
        # Output directory for archetypes is the directory of the input community_file_path
        # e.g., if community_file_path is .../community_X/community_X_pixels_for_archetypes.csv
        # output_dir should be .../community_X/
        community_specific_output_dir = os.path.dirname(community_file_path)
        
        try:
            # Call discover_channel_archetypes
            # It returns: frequent_itemsets_df, archetypes_csv_path, used_min_support, used_bin_percentile
            # We are interested in whether archetypes_csv_path is generated.
            _frequent_itemsets_df, archetypes_csv_path, _used_min_supp, _used_bin_perc = discover_channel_archetypes(
                csv_file_path=community_file_path,
                output_dir=community_specific_output_dir,
                min_support_threshold=min_support_rerun,
                binarization_percentile=bin_percentile_rerun
                # Note: discover_channel_archetypes might need updates if its output naming convention needs to reflect new params.
                # Currently, it will likely overwrite existing archetype files for that community.
            )
            
            if archetypes_csv_path and os.path.exists(archetypes_csv_path):
                num_archetypes = 0
                if _frequent_itemsets_df is not None:
                    num_archetypes = len(_frequent_itemsets_df)
                print(f"  Successfully re-discovered {num_archetypes} archetypes for community.")
                print(f"  Archetype CSV saved to/updated: {archetypes_csv_path}")
                success_count += 1
            else:
                print(f"  Archetype discovery or CSV saving failed for {community_file_path}.")
                failure_count += 1

        except Exception as e:
            print(f"ERROR during archetype discovery for {community_file_path}: {e}")
            traceback.print_exc()
            failure_count += 1
    
    print(f"\n--- Rerun Summary ---")
    print(f"Successfully re-ran archetype discovery for: {success_count} community files.")
    print(f"Failed for: {failure_count} community files.")
    print("Please check logs above for details on each file.")

if __name__ == "__main__":
    main() 