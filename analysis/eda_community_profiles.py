import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse

# UMAP for dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn package not found. UMAP plotting will be skipped. Consider installing with 'pip install umap-learn'.")

def load_community_profiles(csv_path):
    """Loads community profiles CSV into a pandas DataFrame."""
    try:
        df = pd.read_csv(csv_path, index_col=0) # Assuming first col is community ID or index
        if df.empty:
            print(f"  Warning: Loaded community profiles file is empty: {csv_path}")
            return None
        # Assuming channel names are already cleaned (e.g., no _asinh_scaled_avg suffix)
        # If they have suffixes like _asinh_scaled_avg, we might need to clean them here or ensure input files are correct.
        print(f"  Successfully loaded {os.path.basename(csv_path)} with {df.shape[0]} communities and {df.shape[1]} channels.")
        return df
    except FileNotFoundError:
        print(f"  Error: Community profiles file not found: {csv_path}")
        return None
    except Exception as e:
        print(f"  Error loading community profiles {csv_path}: {e}")
        return None

def generate_channel_summaries_and_plots(profiles_df, output_dir_for_file, base_filename):
    """Calculates summary stats and generates distribution plots for each channel."""
    if profiles_df is None or profiles_df.empty:
        print("  Skipping channel summaries: DataFrame is empty or None.")
        return

    # --- Descriptive Statistics ---
    print("    Calculating descriptive statistics for channels...")
    try:
        desc_stats_df = profiles_df.describe().transpose()
        stats_output_path = os.path.join(output_dir_for_file, f"{base_filename}_channel_summary_stats.csv")
        desc_stats_df.to_csv(stats_output_path)
        print(f"      Saved channel summary stats to: {stats_output_path}")
    except Exception as e:
        print(f"      Error calculating/saving descriptive stats: {e}")

    # --- Distribution Plots (per channel) ---
    print("    Generating distribution plots for channels...")
    dist_plots_dir = os.path.join(output_dir_for_file, "channel_distributions")
    os.makedirs(dist_plots_dir, exist_ok=True)

    for channel_name in profiles_df.columns:
        plt.style.use('ggplot') # Using a common style for plots
        # Histogram
        try:
            plt.figure(figsize=(8, 6))
            sns.histplot(profiles_df[channel_name], kde=True)
            plt.title(f"Distribution of {channel_name}\n(File: {base_filename})")
            plt.xlabel(f"{channel_name} Expression (Community Avg)")
            plt.ylabel("Frequency")
            hist_path = os.path.join(dist_plots_dir, f"{base_filename}_{channel_name}_histogram.png")
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()
        except Exception as e_hist:
            print(f"      Error generating histogram for {channel_name}: {e_hist}")

        # Boxplot
        try:
            plt.figure(figsize=(6, 8))
            sns.boxplot(y=profiles_df[channel_name])
            plt.title(f"Boxplot of {channel_name}\n(File: {base_filename})")
            plt.ylabel(f"{channel_name} Expression (Community Avg)")
            box_path = os.path.join(dist_plots_dir, f"{base_filename}_{channel_name}_boxplot.png")
            plt.tight_layout()
            plt.savefig(box_path)
            plt.close()
        except Exception as e_box:
            print(f"      Error generating boxplot for {channel_name}: {e_box}")
    print(f"      Saved individual channel distribution plots to: {dist_plots_dir}")

    # --- Combined Boxplot for all channels ---
    if not profiles_df.empty and len(profiles_df.columns) > 1:
        print("    Generating combined boxplot for all channels...")
        try:
            plt.figure(figsize=(max(12, len(profiles_df.columns) * 0.6), 8))
            sns.boxplot(data=profiles_df)
            plt.title(f"Channel Expression Distributions (Community Averages)\n(File: {base_filename})")
            plt.ylabel("Expression Level")
            plt.xlabel("Channels")
            plt.xticks(rotation=90)
            combined_box_path = os.path.join(output_dir_for_file, f"{base_filename}_all_channels_boxplot.png")
            plt.tight_layout()
            plt.savefig(combined_box_path)
            plt.close()
            print(f"      Saved combined channel boxplot to: {combined_box_path}")
        except Exception as e_cbox:
            print(f"      Error generating combined boxplot: {e_cbox}")

def generate_channel_correlation_heatmap(profiles_df, output_dir_for_file, base_filename):
    """Calculates and plots the channel correlation heatmap."""
    if profiles_df is None or profiles_df.empty or profiles_df.shape[1] < 2:
        print("  Skipping channel correlation heatmap: DataFrame is empty, None, or has less than 2 channels.")
        return

    print("    Calculating and plotting channel correlation heatmap...")
    try:
        correlation_matrix = profiles_df.corr(method='pearson')
        
        # Save correlation matrix to CSV
        correlation_matrix_path = os.path.join(output_dir_for_file, f"{base_filename}_channel_correlation_matrix.csv")
        correlation_matrix.to_csv(correlation_matrix_path)
        print(f"      Saved channel correlation matrix to: {correlation_matrix_path}")

        # Plot heatmap
        plt.style.use('ggplot')
        plt.figure(figsize=(max(10, profiles_df.shape[1] * 0.8), max(8, profiles_df.shape[1] * 0.7)))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Pearson Correlation'})
        plt.title(f"Channel Correlation Heatmap\n(File: {base_filename})", fontsize=16)
        plt.xticks(rotation=90, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout(pad=2.0) # Add padding to prevent labels from being cut off

        heatmap_path = os.path.join(output_dir_for_file, f"{base_filename}_channel_correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        print(f"      Saved channel correlation heatmap to: {heatmap_path}")

    except Exception as e:
        print(f"      Error generating channel correlation heatmap: {e}")

def generate_umap_projection(profiles_df, output_dir_for_file, base_filename, 
                             n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, run_umap_flag=True):
    """Generates and plots UMAP projection of the community profiles."""
    if not run_umap_flag:
        print("  Skipping UMAP projection as per --no_run_umap flag.")
        return

    if not UMAP_AVAILABLE:
        print("  Skipping UMAP projection: umap-learn package is not available.")
        return

    if profiles_df is None or profiles_df.empty:
        print("  Skipping UMAP projection: DataFrame is empty or None.")
        return
    
    if profiles_df.shape[0] <= n_neighbors:
        print(f"  Skipping UMAP: Number of samples ({profiles_df.shape[0]}) is less than or equal to n_neighbors ({n_neighbors}). Consider reducing n_neighbors or if the dataset is too small.")
        return

    print(f"    Generating UMAP projection (n_neighbors={n_neighbors}, min_dist={min_dist}, metric='{metric}')...")
    try:
        reducer = umap.UMAP(n_neighbors=n_neighbors, 
                              min_dist=min_dist, 
                              metric=metric, 
                              random_state=random_state,
                              n_components=2) # Ensure 2D output
        
        embedding = reducer.fit_transform(profiles_df)
        
        umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'], index=profiles_df.index)
        
        # Save UMAP coordinates
        umap_coords_path = os.path.join(output_dir_for_file, f"{base_filename}_umap_coordinates.csv")
        umap_df.to_csv(umap_coords_path)
        print(f"      Saved UMAP coordinates to: {umap_coords_path}")

        # Plot UMAP
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='UMAP1', y='UMAP2', data=umap_df, s=50, alpha=0.7) # s for size, alpha for transparency
        plt.title(f"UMAP Projection of Community Profiles\n(File: {base_filename}, n_neighbors={n_neighbors}, min_dist={min_dist})")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.gca().set_aspect('equal', 'datalim') # Ensure aspect ratio is equal
        umap_plot_path = os.path.join(output_dir_for_file, f"{base_filename}_umap_projection.png")
        plt.tight_layout()
        plt.savefig(umap_plot_path)
        plt.close()
        print(f"      Saved UMAP projection plot to: {umap_plot_path}")

    except Exception as e:
        print(f"      Error generating UMAP projection: {e}")

def main():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis of Community Profiles.")
    parser.add_argument("--base_dir", required=True, help="Root output directory from run_roi_analysis.py (e.g., 'output/').")
    parser.add_argument("--roi_glob", default="ROI_*", help="Glob pattern for ROI directories (default: 'ROI_*').")
    parser.add_argument("--resolution_glob", default="resolution_*", help="Glob pattern for resolution directories (default: 'resolution_*').")
    parser.add_argument("--community_profiles_glob", default="community_profiles_*.csv", help="Glob pattern for community profile CSV files (default: 'scaled_community_profiles_*.csv').")
    # --pixel_data_glob might be added later if needed for community size, etc.
    parser.add_argument("--output_suffix", default="_community_eda", help="Suffix for the new EDA output subdirectories (default: '_community_eda').")

    # UMAP specific arguments
    parser.add_argument("--run_umap", action=argparse.BooleanOptionalAction, default=True, help="Whether to run UMAP projections (default: True). Use --no-run-umap to disable.")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP: Number of neighbors (default: 15).")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP: Minimum distance (default: 0.1).")
    parser.add_argument("--umap_metric", type=str, default="euclidean", help="UMAP: Distance metric (default: 'euclidean').")
    parser.add_argument("--umap_random_state", type=int, default=42, help="UMAP: Random state for reproducibility (default: 42).")

    args = parser.parse_args()

    print(f"Starting EDA of Community Profiles...")
    print(f"Base Directory: {args.base_dir}")

    search_pattern = os.path.join(args.base_dir, args.roi_glob, args.resolution_glob, args.community_profiles_glob)
    community_profile_files = glob.glob(search_pattern)

    if not community_profile_files:
        print(f"No community profile files found matching pattern: {search_pattern}")
        return

    print(f"Found {len(community_profile_files)} community profile files to analyze.")

    for csv_path in community_profile_files:
        print(f"\n--- Processing EDA for: {csv_path} ---")
        
        current_file_dir = os.path.dirname(csv_path)
        current_file_basename = os.path.splitext(os.path.basename(csv_path))[0]
        
        # Create specific output subdirectory for this file's EDA results
        eda_output_dir = os.path.join(current_file_dir, f"{current_file_basename}{args.output_suffix}")
        os.makedirs(eda_output_dir, exist_ok=True)
        print(f"  EDA outputs will be saved in: {eda_output_dir}")

        community_profiles_df = load_community_profiles(csv_path)
        if community_profiles_df is None:
            continue # Skip to next file if loading failed

        generate_channel_summaries_and_plots(community_profiles_df, eda_output_dir, current_file_basename)
        generate_channel_correlation_heatmap(community_profiles_df, eda_output_dir, current_file_basename)
        generate_umap_projection(community_profiles_df, eda_output_dir, current_file_basename,
                                 n_neighbors=args.umap_n_neighbors,
                                 min_dist=args.umap_min_dist,
                                 metric=args.umap_metric,
                                 random_state=args.umap_random_state,
                                 run_umap_flag=args.run_umap)
        
        print(f"  Completed EDA for: {os.path.basename(csv_path)}")

    print("\n--- EDA Script Finished ---")

if __name__ == '__main__':
    main() 