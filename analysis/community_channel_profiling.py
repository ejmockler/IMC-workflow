import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_community_channel_profiles(csv_file_path, output_dir):
    """
    Reads pixel data, calculates mean channel profiles per community,
    and generates a heatmap.
    """
    print(f"Reading data from: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        return

    # Identify channel columns (ending with _asinh_scaled_avg)
    channel_columns = [col for col in df.columns if col.endswith('_asinh_scaled_avg')]
    if not channel_columns:
        print("Error: No channel columns found (expected to end with '_asinh_scaled_avg').")
        return
    
    print(f"Found {len(channel_columns)} channel columns.")

    # Columns to group by and aggregate
    required_columns = ['community'] + channel_columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if 'community' not in df.columns:
        print("Error: 'community' column not found in the CSV.")
        return
    if missing_cols:
        print(f"Warning: Missing some channel columns: {missing_cols}")
        # Filter out missing channel columns to proceed if possible
        channel_columns = [col for col in channel_columns if col in df.columns]
        if not channel_columns:
            print("Error: No usable channel columns left after checking for missing ones.")
            return


    print("Calculating mean channel profiles per community...")
    community_profiles = df.groupby('community')[channel_columns].mean()

    # Clean up channel names for heatmap labels
    cleaned_channel_names = [name.replace('_asinh_scaled_avg', '') for name in community_profiles.columns]
    community_profiles.columns = cleaned_channel_names

    if community_profiles.empty:
        print("No community profiles were generated. Check the input data and community column.")
        return

    print(f"Generated profiles for {len(community_profiles)} communities.")

    # Print descriptive statistics for community_profiles
    print("\nDescriptive statistics for mean channel intensities across communities:")
    print(community_profiles.describe().to_string())
    print("\n")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Plotting the heatmap
    plt.figure(figsize=(max(12, len(cleaned_channel_names) * 0.5), max(8, len(community_profiles) * 0.3)))
    sns.heatmap(community_profiles, annot=False, cmap='viridis', linewidths=.5)
    plt.title(f'Mean Channel Profiles per Community\\n(Source: {os.path.basename(csv_file_path)})')
    plt.xlabel('Channel')
    plt.ylabel('Community ID')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(csv_file_path))[0]}_community_profiles_heatmap.png")
    plt.savefig(heatmap_path)
    print(f"Heatmap saved to: {heatmap_path}")
    plt.close()

if __name__ == '__main__':
    # Assuming the script is in IMC/analysis/ and data is in IMC/output/
    # Adjust base_path if your script is located elsewhere relative to the project root.
    # Workspace root is /home/noot/IMC
    # NEW: Determine project root robustly
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute dir of the script e.g. /home/noot/IMC/analysis
    project_root = os.path.abspath(os.path.join(script_dir, '..'))  # Absolute project root e.g. /home/noot/IMC

    # Specific file based on user context
    roi_name = "ROI_D7_M1_03_23"
    resolution = "100"
    
    # Construct the path to the CSV file
    csv_filename = f"pixel_data_with_community_annotations_{roi_name}_res_{resolution}.csv"
    csv_path = os.path.join(project_root, "output", roi_name, f"resolution_{resolution}", csv_filename)
    output_directory = os.path.join(project_root, "output", roi_name, f"resolution_{resolution}")
    
    # Ensure paths are absolute and correct, as the script's CWD might vary
    # workspace_root = "/home/noot/IMC" 
    # csv_file_full_path = os.path.join(workspace_root, "output", roi_name, f"resolution_{resolution}", csv_filename)
    # output_directory = os.path.join(workspace_root, "output", roi_name, f"resolution_{resolution}")

    print(f"Attempting to read CSV from normalized path: {csv_path}") # Debug print

    generate_community_channel_profiles(csv_path, output_directory)

    print("\nNext steps could involve:")
    print("1. Review the generated heatmap: community_channel_profiles_heatmap.png")
    print("2. Define a strategy for binarizing channel activity (e.g., thresholding).")
    print("3. Apply frequent itemset mining to find co-occurring active channel sets.") 