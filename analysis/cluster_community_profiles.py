import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# UMAP is optional
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn package not found. UMAP plots will not be generated. Install with: pip install umap-learn")

def load_and_prepare_community_profiles(annotated_pixel_csv_path):
    """Loads annotated pixel data, calculates mean community profiles, and cleans channel names."""
    try:
        df = pd.read_csv(annotated_pixel_csv_path)
    except Exception as e:
        print(f"Error reading {annotated_pixel_csv_path}: {e}")
        return None

    if 'community' not in df.columns:
        print(f"Error: 'community' column not found in {annotated_pixel_csv_path}.")
        return None

    channel_columns_for_profiling = [col for col in df.columns if col.endswith('_asinh_scaled_avg')]
    if not channel_columns_for_profiling:
        print(f"Error: No '_asinh_scaled_avg' columns found in {annotated_pixel_csv_path}.")
        return None

    try:
        community_profiles_df = df.groupby('community')[channel_columns_for_profiling].mean()
    except Exception as e:
        print(f"Error calculating community profiles for {annotated_pixel_csv_path}: {e}")
        return None

    cleaned_channel_names = [name.replace('_asinh_scaled_avg', '') for name in community_profiles_df.columns]
    community_profiles_df.columns = cleaned_channel_names

    if community_profiles_df.empty:
        print(f"No community profiles generated for {annotated_pixel_csv_path}.")
        return None
    if len(community_profiles_df) < 2:
        print(f"Only {len(community_profiles_df)} community/ies found in {annotated_pixel_csv_path}. Cannot cluster. Skipping.")
        return None
        
    return community_profiles_df

def plot_community_dendrogram(linkage_matrix, num_communities, output_path, title_prefix=""):
    if linkage_matrix is None or num_communities == 0:
        print("Skipping community dendrogram: no linkage matrix or communities.")
        return
    try:
        plt.figure(figsize=(20, 10))
        sch.dendrogram(
            linkage_matrix,
            labels=np.arange(num_communities), # Could use actual community IDs if not too many
            leaf_rotation=90.,
            leaf_font_size=8,
            truncate_mode='lastp',
            p=100 
        )
        plt.title(f"{title_prefix}Community Dendrogram")
        plt.xlabel("Community Index / Cluster Size")
        plt.ylabel("Distance") # Will depend on metric used for linkage
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved community dendrogram to: {output_path}")
    except Exception as e:
        print(f"Error generating community dendrogram {output_path}: {e}")

def plot_ccluster_profile_heatmap(ccluster_avg_profiles_df, output_path, title_prefix=""):
    if ccluster_avg_profiles_df is None or ccluster_avg_profiles_df.empty:
        print("Skipping C-Cluster profile heatmap: no profile data.")
        return
    try:
        plt.figure(figsize=(max(12, ccluster_avg_profiles_df.shape[1] * 0.5), max(6, ccluster_avg_profiles_df.shape[0] * 0.4)))
        sns.heatmap(
            ccluster_avg_profiles_df,
            annot=False, # Can set to True if few C-Clusters/channels
            cmap="viridis", # Or a diverging map if data is centered, e.g. "RdBu_r"
            center=0 if np.any(ccluster_avg_profiles_df.values < 0) else None, # Center if Z-scores or similar
            linewidths=.5
        )
        plt.title(f"{title_prefix}C-Cluster Average Expression Profiles")
        plt.xlabel("Channels")
        plt.ylabel("C-Cluster ID")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved C-Cluster profile heatmap to: {output_path}")
    except Exception as e:
        print(f"Error generating C-Cluster profile heatmap {output_path}: {e}")

def plot_umap_communities(community_profiles_scaled_df, ccluster_labels, output_path, title_prefix=""):
    if not UMAP_AVAILABLE or community_profiles_scaled_df is None or ccluster_labels is None:
        print("Skipping UMAP of communities: UMAP not available or data missing.")
        return
    if len(community_profiles_scaled_df) != len(ccluster_labels):
        print("Skipping UMAP: Mismatch between community profile count and cluster label count.")
        return
    try:
        reducer = umap.UMAP(n_neighbors=min(15, len(community_profiles_scaled_df)-1), min_dist=0.1, random_state=42, n_components=2)
        embedding = reducer.fit_transform(community_profiles_scaled_df)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=ccluster_labels, cmap='Spectral', s=5)
        plt.title(f"{title_prefix}UMAP of Communities by C-Cluster")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.gca().set_aspect('equal', 'datalim')
        # Add a legend if number of clusters is manageable
        if len(np.unique(ccluster_labels)) < 20:
             plt.legend(*scatter.legend_elements(), title="C-Clusters")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved UMAP of communities to: {output_path}")
    except Exception as e:
        print(f"Error generating UMAP of communities {output_path}: {e}")

def characterize_cclusters(ccluster_avg_profiles_df, z_threshold=1.0):
    """Characterizes C-Clusters by positive and negative channels based on Z-scores of average profiles."""
    characterization = {}
    if ccluster_avg_profiles_df is None or ccluster_avg_profiles_df.empty or len(ccluster_avg_profiles_df) <=1:
        print("  Characterization skipped: Not enough C-Clusters or empty profile data.")
        # For single cluster case, we can still list its profile without pos/neg labels relative to others
        if ccluster_avg_profiles_df is not None and not ccluster_avg_profiles_df.empty and len(ccluster_avg_profiles_df) == 1:
            ccluster_id = ccluster_avg_profiles_df.index[0]
            profile_series = ccluster_avg_profiles_df.iloc[0]
            label = f"C-Cluster_{ccluster_id} (single cluster - absolute profile shown)"
            characterization[ccluster_id] = {
                "positive_channels": "N/A (single cluster)",
                "negative_channels": "N/A (single cluster)",
                "full_label": label,
                "avg_profile_series": profile_series
            }
        return characterization

    # Z-score each channel's average expression ACROSS C-Clusters
    # This makes the z_threshold relative to the variability between C-Cluster means for each channel
    try:
        scaler_char = StandardScaler() # Initialize a new scaler
        ccluster_avg_profiles_internally_zscored_df = pd.DataFrame(
            scaler_char.fit_transform(ccluster_avg_profiles_df.T).T, # Transpose, scale features (channels), then transpose back
            index=ccluster_avg_profiles_df.index,
            columns=ccluster_avg_profiles_df.columns
        )
    except Exception as e_scale:
        print(f"  Error during internal Z-scaling for characterization: {e_scale}. Proceeding with raw averages for characterization.")
        ccluster_avg_profiles_internally_zscored_df = ccluster_avg_profiles_df # Fallback to raw if scaling fails

    for ccluster_id, profile_series_zscored in ccluster_avg_profiles_internally_zscored_df.iterrows():
        positive_channels = profile_series_zscored[profile_series_zscored > z_threshold].index.tolist()
        negative_channels = profile_series_zscored[profile_series_zscored < -z_threshold].index.tolist()
        
        positive_channels.sort()
        negative_channels.sort()

        label = f"C-Cluster_{ccluster_id}:\n  POS: {(', '.join(positive_channels) if positive_channels else 'N/A')[:100]}\n  NEG: {(', '.join(negative_channels) if negative_channels else 'N/A')[:100]}"
        characterization[ccluster_id] = {
            "positive_channels": positive_channels,
            "negative_channels": negative_channels,
            "full_label": label,
            "avg_profile_series": profile_series_zscored
        }
    return characterization

def main():
    parser = argparse.ArgumentParser(description="Cluster community profiles from multiple ROIs and resolutions.")
    parser.add_argument("--base_dir", default="output", help="Base output directory containing ROI subdirectories.")
    parser.add_argument("--input_glob", default="pixel_data_with_community_annotations_*.csv", help="Glob pattern for input annotated pixel CSV files within each resolution directory.")
    parser.add_argument("--profile_glob", default="community_profiles_scaled_*.csv", help="Glob pattern for community profile CSV files.")
    # Hierarchical clustering parameters
    parser.add_argument("--hc_distance_metric", default="correlation", help="Distance metric for hierarchical clustering of communities (e.g., 'correlation', 'euclidean').")
    parser.add_argument("--hc_linkage_method", default="average", help="Linkage method for hierarchical clustering (e.g., 'average', 'ward', 'complete').")
    parser.add_argument("--hc_num_clusters", type=int, default=0, help="Target number of C-Clusters (uses fcluster with criterion='maxclust'). If 0, uses distance_threshold.")
    parser.add_argument("--hc_distance_threshold", type=float, default=0.7, help="Distance threshold for fcluster (criterion='distance'). Used if hc_num_clusters is 0.")
    parser.add_argument("--zscore_scale_profiles", action='store_true', help="If set, Z-score scale community profiles per channel before clustering.")
    parser.add_argument("--zscore_char_threshold", type=float, default=1.0, help="Z-score threshold for characterizing C-Clusters as Pos/Neg.")
    parser.add_argument("--results_suffix", default="_community_clusters", help="Suffix for output files and subdirectories from this script.")
    
    args = parser.parse_args()

    if args.hc_num_clusters <= 0 and args.hc_distance_threshold <= 0:
        print("Error: Must specify either a positive --hc_num_clusters or a positive --hc_distance_threshold.")
        return

    base_search_path = os.path.join(args.base_dir, "ROI_*", "resolution_*")
    # Find all files matching the input_glob within the ROI/resolution subdirectories
    all_potential_files = glob.glob(os.path.join(base_search_path, args.input_glob))
    
    # Filter out files that are themselves archetype outputs from the previous script
    all_annotated_csvs = [f for f in all_potential_files if "_channel_archetypes_" not in os.path.basename(f)]
    
    print(f"Found {len(all_potential_files)} files initially, filtered to {len(all_annotated_csvs)} non-archetype annotated pixel CSV files using pattern: {search_pattern if 'search_pattern' in locals() else os.path.join(base_search_path, args.input_glob)}") # Corrected print statement

    for csv_path in all_annotated_csvs:
        print(f"\n--- Processing: {csv_path} ---")
        current_dir = os.path.dirname(csv_path)
        base_filename = os.path.splitext(os.path.basename(csv_path))[0]

        community_profiles_df = load_and_prepare_community_profiles(csv_path)
        if community_profiles_df is None:
            continue # Error message already printed in function

        # Optional Z-score scaling of profiles before clustering
        community_profiles_for_clustering = community_profiles_df.copy()
        if args.zscore_scale_profiles:
            print("  Z-score scaling community profiles across channels...")
            scaler = StandardScaler() # Scales each feature (channel) to have mean 0, std 1
            community_profiles_for_clustering = pd.DataFrame(
                scaler.fit_transform(community_profiles_for_clustering),
                index=community_profiles_for_clustering.index,
                columns=community_profiles_for_clustering.columns
            )

        analysis_output_dir = os.path.join(current_dir, base_filename + args.results_suffix)
        os.makedirs(analysis_output_dir, exist_ok=True)
        print(f"  Analysis outputs will be saved in: {analysis_output_dir}")

        # --- Hierarchical Clustering of Communities ---
        print(f"  Performing hierarchical clustering of {len(community_profiles_for_clustering)} communities...")
        print(f"    Distance metric: {args.hc_distance_metric}, Linkage method: {args.hc_linkage_method}")
        try:
            # pdist calculates condensed distance matrix
            community_dist_matrix_condensed = pdist(community_profiles_for_clustering, metric=args.hc_distance_metric)
            linkage_matrix = sch.linkage(community_dist_matrix_condensed, method=args.hc_linkage_method)
        except Exception as e_hc:
            print(f"    Error during hierarchical clustering: {e_hc}. Skipping this file.")
            continue

        # Determine C-Cluster labels
        if args.hc_num_clusters > 0:
            print(f"    Cutting dendrogram for {args.hc_num_clusters} C-Clusters (maxclust criterion).")
            ccluster_labels = sch.fcluster(linkage_matrix, t=args.hc_num_clusters, criterion='maxclust')
        else:
            print(f"    Cutting dendrogram with distance threshold {args.hc_distance_threshold} (distance criterion).")
            ccluster_labels = sch.fcluster(linkage_matrix, t=args.hc_distance_threshold, criterion='distance')
        
        num_cclusters_found = len(np.unique(ccluster_labels))
        print(f"    Found {num_cclusters_found} C-Clusters.")

        if num_cclusters_found <= 1 or num_cclusters_found >= len(community_profiles_for_clustering):
            print("    Silhouette score cannot be calculated for 1 cluster or n_samples clusters. Skipping score.")
            avg_silhouette = None
        else:
            try:
                # Use the same distance matrix used for clustering for silhouette score, if appropriate
                # Note: If metric was 'correlation', pdist gives 1-corr. Silhouette expects distances.
                # Silhouette score expects a square distance matrix if metric is 'precomputed'
                square_community_dist_matrix = squareform(community_dist_matrix_condensed)
                avg_silhouette = silhouette_score(square_community_dist_matrix, ccluster_labels, metric='precomputed')
                print(f"    Average Silhouette Score for C-Clusters: {avg_silhouette:.4f}")
            except ValueError as sve:
                print(f"    Could not calculate Silhouette Score for C-Clusters: {sve}")
                avg_silhouette = None
            except Exception as sil_e:
                print(f"    Error calculating Silhouette Score for C-Clusters: {sil_e}")
                avg_silhouette = None

        # --- Plot Community Dendrogram ---
        dendro_filename = f"community_dendrogram_{args.hc_distance_metric}_{args.hc_linkage_method}.png"
        dendro_path = os.path.join(analysis_output_dir, dendro_filename)
        plot_community_dendrogram(linkage_matrix, len(community_profiles_for_clustering), dendro_path, title_prefix=f"{base_filename}\n")

        # --- Calculate and Characterize C-Cluster Average Profiles ---
        community_profiles_df_with_labels = community_profiles_df.copy() # Use original, non-scaled for averaging if scaling was done only for clustering
        community_profiles_df_with_labels['c_cluster'] = ccluster_labels
        ccluster_avg_profiles = community_profiles_df_with_labels.groupby('c_cluster').mean()
        
        # If Z-score scaling was applied for clustering, the average profiles for characterization should reflect that for consistent Z-thresholding
        # OR, we can Z-score the final average profiles before characterization if zscore_char_threshold implies it.
        # For simplicity, let characterization operate on the averaged profiles, assuming they are comparable or zscore_char_threshold is relative.
        # If --zscore_scale_profiles was used, the ccluster_avg_profiles would be averages of Z-scores.

        characteristics = characterize_cclusters(ccluster_avg_profiles, z_threshold=args.zscore_char_threshold)

        summary_lines = [
            f"C-Cluster analysis for: {base_filename}",
            f"  Clustering: Metric={args.hc_distance_metric}, Linkage={args.hc_linkage_method}, Num Clusters Target={args.hc_num_clusters if args.hc_num_clusters > 0 else 'N/A'}, Dist Thresh={args.hc_distance_threshold if args.hc_num_clusters == 0 else 'N/A'}",
            f"  Num C-Clusters Found: {num_cclusters_found}",
            f"  Avg Silhouette Score: {avg_silhouette if avg_silhouette is not None else 'N/A'}",
            f"  Z-score scaling for clustering: {args.zscore_scale_profiles}",
            f"  Z-score threshold for Pos/Neg characterization: {args.zscore_char_threshold}",
            "\nC-Cluster Characterization:"
        ]
        for ccluster_id, char_data in characteristics.items():
            summary_lines.append(f"  {char_data['full_label']}")
        
        summary_text_path = os.path.join(analysis_output_dir, f"ccluster_summary.txt")
        with open(summary_text_path, 'w') as f:
            f.write("\n".join(summary_lines))
        print(f"    C-Cluster characterization saved to: {summary_text_path}")

        avg_profiles_path = os.path.join(analysis_output_dir, "ccluster_average_profiles.csv")
        ccluster_avg_profiles.to_csv(avg_profiles_path)
        print(f"    C-Cluster average profiles saved to: {avg_profiles_path}")

        # --- Plot C-Cluster Profile Heatmap ---
        heatmap_filename = f"ccluster_avg_profiles_heatmap.png"
        heatmap_path = os.path.join(analysis_output_dir, heatmap_filename)
        plot_ccluster_profile_heatmap(ccluster_avg_profiles, heatmap_path, title_prefix=f"{base_filename}\n")

        # --- Plot UMAP of Communities ---
        umap_filename = f"communities_umap_by_ccluster.png"
        umap_path = os.path.join(analysis_output_dir, umap_filename)
        plot_umap_communities(community_profiles_for_clustering, ccluster_labels, umap_path, title_prefix=f"{base_filename}\n")

    print("\n--- Community Profile Clustering Analysis Completed ---")

if __name__ == '__main__':
    main() 