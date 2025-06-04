import pandas as pd
from mlxtend.frequent_patterns import apriori
import os
import numpy as np # Added for linkage
import scipy.cluster.hierarchy as sch # Added for linkage
from sklearn.preprocessing import MultiLabelBinarizer # Added for archetype representation
from scipy.spatial.distance import pdist, squareform # Added for distance calculation
from typing import Optional

def discover_channel_archetypes(csv_file_path, output_dir, min_support_threshold=0.05, binarization_percentile=0.75):
    """
    Discovers frequent channel co-occurrence patterns (archetypes) from community profiles
    derived from an annotated pixel results CSV file.

    Args:
        csv_file_path (str): Path to the 'pixel_data_with_community_annotations_*.csv' file.
                             This file should contain a 'community' column and channel columns
                             ending with '_asinh_scaled_avg'.
        output_dir (str): Directory to save the discovered archetypes CSV.
        min_support_threshold (float): Minimum support for an itemset to be considered frequent.
        binarization_percentile (float): Percentile (0.0-1.0) to binarize channel activity.
                                         A channel is active if its mean community expression
                                         is above this percentile of its distribution across communities.
    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str], Optional[float], Optional[float]]: 
            - DataFrame of frequent itemsets (archetypes) or None on failure.
            - Path to the saved archetypes CSV file or None on failure.
            - The min_support_threshold used.
            - The binarization_percentile used.
    """
    print(f"Starting archetype discovery for: {csv_file_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Min support: {min_support_threshold}, Binarization percentile: {binarization_percentile}")

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Archetype input file not found at {csv_file_path}")
        return None, None, min_support_threshold, binarization_percentile
    except Exception as e:
        print(f"Error reading archetype input file {csv_file_path}: {e}")
        return None, None, min_support_threshold, binarization_percentile

    channel_columns_for_profiling = [col for col in df.columns if col.endswith('_asinh_scaled_avg')]
    
    if not channel_columns_for_profiling:
        print("Error: No '_asinh_scaled_avg' columns found in the input CSV. Cannot calculate community profiles.")
        return None, None, min_support_threshold, binarization_percentile
    if 'community' not in df.columns:
        print("Error: 'community' column not found in the input CSV.")
        return None, None, min_support_threshold, binarization_percentile

    print(f"Found {len(channel_columns_for_profiling)} asinh_scaled_avg channel columns for profiling.")
    print("Calculating mean channel profiles per community for archetype discovery...")
    
    try:
        community_profiles = df.groupby('community')[channel_columns_for_profiling].mean()
    except Exception as e:
        print(f"Error during groupby or mean calculation for community profiles: {e}")
        return None, None, min_support_threshold, binarization_percentile

    cleaned_channel_names = [name.replace('_asinh_scaled_avg', '') for name in community_profiles.columns]
    community_profiles.columns = cleaned_channel_names

    if community_profiles.empty:
        print("No community profiles were generated (empty after grouping). Cannot proceed.")
        return None, None, min_support_threshold, binarization_percentile

    print(f"Generated profiles for {len(community_profiles)} communities using {len(cleaned_channel_names)} channels.")
    print(f"Binarizing channel profiles using the {binarization_percentile*100}th percentile for each channel...")

    community_profiles_binary = community_profiles.copy()
    for channel_name in community_profiles_binary.columns:
        try:
            threshold = community_profiles_binary[channel_name].quantile(binarization_percentile)
            community_profiles_binary[channel_name] = (community_profiles_binary[channel_name] > threshold).astype(int)
            # print(f"  Channel '{channel_name}': Threshold ({binarization_percentile*100}th percentile) = {threshold:.4f}")
        except Exception as e:
            print(f"Error binarizing channel {channel_name}: {e}. Skipping this channel for FIM.")
            community_profiles_binary = community_profiles_binary.drop(columns=[channel_name]) # Drop problematic channel
            continue
    
    if community_profiles_binary.empty or community_profiles_binary.shape[1] == 0:
        print("Error: No channels left after binarization step (all might have had errors). Cannot proceed.")
        return None, None, min_support_threshold, binarization_percentile

    cols_to_drop_post_binarization = []
    for col in community_profiles_binary.columns:
        if community_profiles_binary[col].nunique() == 1:
            cols_to_drop_post_binarization.append(col)
            # print(f"    Note: Channel '{col}' has only one unique value after binarization (likely all 0s or all 1s) and will be dropped for FIM.")
    
    if cols_to_drop_post_binarization:
        community_profiles_binary = community_profiles_binary.drop(columns=cols_to_drop_post_binarization)
        print(f"Dropped {len(cols_to_drop_post_binarization)} channels that were constant post-binarization.")

    if community_profiles_binary.empty or community_profiles_binary.shape[1] == 0:
        print("Error: No informative channels left after binarization and dropping constant columns. Cannot proceed with FIM.")
        return None, None, min_support_threshold, binarization_percentile
    
    print(f"Performing Frequent Itemset Mining with {community_profiles_binary.shape[1]} channels and min_support = {min_support_threshold}...")
    try:
        frequent_itemsets = apriori(community_profiles_binary, min_support=min_support_threshold, use_colnames=True)
    except Exception as e:
        print(f"Error during Apriori algorithm execution: {e}")
        return None, None, min_support_threshold, binarization_percentile

    if frequent_itemsets.empty:
        print(f"No frequent itemsets found with min_support = {min_support_threshold}. Consider lowering the threshold or checking binarization.")
        return None, None, min_support_threshold, binarization_percentile

    print(f"Found {len(frequent_itemsets)} frequent itemsets (archetypes).")
    frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory {output_dir}: {e}. Cannot save archetypes.")
            return frequent_itemsets, None, min_support_threshold, binarization_percentile

    base_input_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    archetypes_filename = f"{base_input_name}_channel_archetypes_support{min_support_threshold}_bin{binarization_percentile*100:.0f}pct.csv"
    archetypes_path = os.path.join(output_dir, archetypes_filename)
    
    try:
        frequent_itemsets.to_csv(archetypes_path, index=False)
        print(f"Channel archetypes saved to: {archetypes_path}")
        return frequent_itemsets, archetypes_path, min_support_threshold, binarization_percentile
    except Exception as e:
        print(f"Error saving archetypes to CSV {archetypes_path}: {e}")
        return frequent_itemsets, None, min_support_threshold, binarization_percentile

if __name__ == '__main__':
    # Example usage (for testing this module directly)
    print("This module is intended to be imported, but here's a test run example for discover_channel_archetypes:")
    # Create a dummy CSV for testing
    roi_string = "TEST_ROI"
    resolution = "TEST_RES"
    # community_id = "TEST_COMM"
    # base_dir = "_test_archetype_output"
    # test_dir = os.path.join(base_dir, roi_string, f"resolution_{resolution}", f"spatial_region_region_0") # Adjusted for region
    test_dir = "_test_archetype_output_module_direct"
    os.makedirs(test_dir, exist_ok=True)
    
    # Since discover_channel_archetypes now expects ..._asinh_scaled_avg suffixes and community column:
    dummy_pixel_data_path = os.path.join(test_dir, f"pixel_data_with_community_annotations_{roi_string}_res_{resolution}.csv")
    dummy_df_for_archetypes = pd.DataFrame({
        'X': [1,2,3,1,2,3,1,2,3],
        'Y': [0,0,1,1,2,2,0,1,2],
        'community': [1,1,1,2,2,2,3,3,3], # Added community column
        'CD45_asinh_scaled_avg': [0.1, 0.8, 0.2, 0.9, 0.7, 0.1, 0.5, 0.6, 0.4],
        'CD3_asinh_scaled_avg':  [0.9, 0.2, 0.8, 0.1, 0.3, 0.7, 0.4, 0.3, 0.8],
        'PanCK_asinh_scaled_avg': [0.2, 0.3, 0.1, 0.6, 0.5, 0.4, 0.8, 0.7, 0.9],
        'DNA1_asinh_scaled_avg': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # This might be dropped by FIM if all 0s/1s post-binarization
    })
    dummy_df_for_archetypes.to_csv(dummy_pixel_data_path, index=False)
    print(f"Created dummy data at: {dummy_pixel_data_path}")

    print(f"\nRunning discover_channel_archetypes test with dummy data...")
    frequent_itemsets_df_out, archetypes_csv_out, sup, b_pct = discover_channel_archetypes(
        csv_file_path=dummy_pixel_data_path, 
        output_dir=test_dir, 
        min_support_threshold=0.1, # Adjusted for small dummy data
        binarization_percentile=0.5 # Adjusted for small dummy data
    )

    if frequent_itemsets_df_out is not None:
        print("\n--- Test: Frequent Itemsets (Archetypes) --- ")
        print(frequent_itemsets_df_out.head())
        if archetypes_csv_out:
            print(f"Archetypes saved to: {archetypes_csv_out}")
    else:
        print("\n--- Test: No frequent itemsets found or error occurred. ---")
    
    print("\n--- Test for discover_channel_archetypes finished. ---")
    # linkage_file = cluster_archetypes_and_save_linkage( # This part is removed
    #     frequent_itemsets_df=frequent_itemsets_df_out,
    #     archetypes_csv_path=archetypes_csv_out,
    #     output_dir=test_dir,
    #     linkage_method='average'
    # )
    # if linkage_file:
    #     print(f"Archetype linkage matrix saved to: {linkage_file}")
    # else:
    #     print("Failed to generate or save archetype linkage matrix.") 