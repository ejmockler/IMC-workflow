import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import glob
import os
import argparse
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import pdist, squareform

def load_archetype_data(archetypes_csv_path, linkage_npy_path):
    """Loads archetype DataFrame and linkage matrix."""
    try:
        archetypes_df = pd.read_csv(archetypes_csv_path)
        # The 'itemsets' column is loaded as a string representation of a frozenset
        # We need to convert it back to actual frozensets or lists of strings
        # For MultiLabelBinarizer, lists of strings are easier.
        archetypes_df['itemsets'] = archetypes_df['itemsets'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        archetypes_df['itemsets_list'] = archetypes_df['itemsets'].apply(list)

    except Exception as e:
        print(f"Error loading archetypes CSV {archetypes_csv_path}: {e}")
        return None, None
    try:
        linkage_matrix = np.load(linkage_npy_path)
    except Exception as e:
        print(f"Error loading linkage NPY {linkage_npy_path}: {e}")
        return None, None
    return archetypes_df, linkage_matrix

def plot_dendrogram(linkage_matrix, num_archetypes, output_path, title_prefix=""):
    """Plots and saves a dendrogram."""
    if linkage_matrix is None or num_archetypes == 0:
        print("Skipping dendrogram: no linkage matrix or archetypes.")
        return
    try:
        plt.figure(figsize=(20, 10))
        dendrogram_result = sch.dendrogram(
            linkage_matrix, 
            labels=np.arange(num_archetypes), 
            leaf_rotation=90.,
            leaf_font_size=8,
            truncate_mode='lastp', # Show only the last p merged clusters if too many leaves
            p=100 # Number of last merged clusters to show (can be adjusted)
        )
        plt.title(f"{title_prefix}Archetype Dendrogram (Jaccard Distance)")
        plt.xlabel("Archetype Index / Cluster Size")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved dendrogram to: {output_path}")
    except Exception as e:
        print(f"Error generating dendrogram {output_path}: {e}")

def get_binary_archetype_matrix(archetypes_df):
    """Converts archetype itemsets to a binary matrix."""
    if 'itemsets_list' not in archetypes_df.columns:
        print("Error: 'itemsets_list' column not found for binarization.")
        return None, None
    
    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(archetypes_df['itemsets_list'])
    return binary_matrix, mlb.classes_ # classes_ are the channel names

def plot_clustered_heatmap(binary_matrix, channels, cluster_labels, output_path, title_prefix=""):
    """Plots a heatmap of binarized archetypes, ordered by cluster labels."""
    if binary_matrix is None or channels is None or cluster_labels is None:
        print("Skipping heatmap: missing data.")
        return
    try:
        df_to_plot = pd.DataFrame(binary_matrix, columns=channels)
        df_to_plot['metacluster'] = cluster_labels
        df_to_plot = df_to_plot.sort_values(by='metacluster')
        
        cluster_order = df_to_plot['metacluster'] # Keep track of cluster boundaries
        df_to_plot = df_to_plot.drop(columns=['metacluster'])

        # Determine if we need to subsample for very large heatmaps
        max_archetypes_for_heatmap = 500 # Adjust as needed
        if len(df_to_plot) > max_archetypes_for_heatmap:
            print(f"Note: Subsampling archetypes for heatmap from {len(df_to_plot)} to {max_archetypes_for_heatmap}.")
            df_to_plot = df_to_plot.sample(n=max_archetypes_for_heatmap, random_state=42).sort_values(by=cluster_order.loc[df_to_plot.index])
            # Also subsample cluster_order for drawing lines if implemented

        plt.figure(figsize=(max(20, len(channels) * 0.5), max(10, len(df_to_plot) * 0.05)))
        sns.heatmap(df_to_plot, cmap="viridis", cbar=False, yticklabels=False)
        plt.title(f"{title_prefix}Archetype-Channel Heatmap (Clustered)")
        plt.xlabel("Channels")
        plt.ylabel("Archetypes (Ordered by Meta-cluster)")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved clustered heatmap to: {output_path}")
    except Exception as e:
        print(f"Error generating clustered heatmap {output_path}: {e}")

def plot_metacluster_signature_heatmap(characteristics, all_unique_channels, output_path, title_prefix=""):
    """Plots a heatmap of meta-cluster signatures (positive/negative)."""
    if not characteristics or not all_unique_channels:
        print("Skipping meta-cluster signature heatmap: no characteristics or channels.")
        return

    mc_ids = sorted(list(characteristics.keys()))
    profile_data = pd.DataFrame(index=mc_ids, columns=all_unique_channels, dtype=float).fillna(0.0)

    for mc_id, data in characteristics.items():
        for channel in data.get('positive_channels', []):
            if channel in profile_data.columns:
                profile_data.loc[mc_id, channel] = 1 # Positive score
        for channel in data.get('absent_or_rare_channels', []):
            if channel in profile_data.columns:
                profile_data.loc[mc_id, channel] = -1 # Negative score
    
    # Sort columns (channels) alphabetically for consistent heatmaps
    profile_data = profile_data.reindex(sorted(profile_data.columns), axis=1)
    # Sort rows (meta-cluster IDs) numerically for consistent heatmaps
    profile_data = profile_data.reindex(sorted(profile_data.index))

    if profile_data.empty:
        print("Skipping meta-cluster signature heatmap: profile data is empty after processing.")
        return

    try:
        plt.figure(figsize=(max(15, len(all_unique_channels) * 0.4), max(8, len(mc_ids) * 0.3)))
        sns.heatmap(
            profile_data, 
            cmap="RdBu_r", 
            center=0, 
            yticklabels=True, 
            linewidths=.5,
            cbar_kws={"label": "Signature Score (+1 Pos, -1 Neg/Rare, 0 Other)"}
        )
        plt.title(f"{title_prefix}Meta-Cluster Signatures")
        plt.xlabel("Channels")
        plt.ylabel("Meta-Cluster ID")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved meta-cluster signature heatmap to: {output_path}")
    except Exception as e:
        print(f"Error generating meta-cluster signature heatmap {output_path}: {e}")

def characterize_metaclusters(archetypes_df_with_labels, cluster_col_name, all_unique_channels_in_input_file):
    """Characterizes meta-clusters by common channels and consistently absent channels."""
    if cluster_col_name not in archetypes_df_with_labels.columns:
        return {}
    
    characterization = {}
    grouped = archetypes_df_with_labels.groupby(cluster_col_name)['itemsets_list']
    global_channels_set = set(all_unique_channels_in_input_file)
    
    for mc_id, itemsets_group in grouped:
        all_channels_in_mc_lists = itemsets_group.tolist()
        # Flatten the list of lists of channels for this meta-cluster
        current_mc_all_channel_occurrences = [channel for sublist in all_channels_in_mc_lists for channel in sublist]
        channel_counts_in_mc = pd.Series(current_mc_all_channel_occurrences).value_counts()
        num_archetypes_in_mc = len(itemsets_group)
        
        # Positive channels: present in > 70% of archetypes in this meta-cluster
        # and also having at least 2 occurrences to avoid rare singletons defining small clusters
        positive_channels = channel_counts_in_mc[(channel_counts_in_mc / num_archetypes_in_mc > 0.7) & (channel_counts_in_mc >=2)].index.tolist()

        # Negative/Absent channels for the signature:
        # Channels that are globally present (in any archetype in this file) 
        # but are present in <10% of archetypes within THIS meta-cluster.
        # This avoids listing every channel not in positive_channels as negative.
        
        # First, get set of all channels appearing at least once in this meta-cluster's archetypes
        channels_present_in_this_mc_at_all = set(channel_counts_in_mc.index)
        
        # Potential negative candidates are global channels NOT in this MC's positive set
        # More accurately, global channels that are RARELY in this MC
        absent_or_rare_channels = []
        for glob_chan in global_channels_set:
            if glob_chan not in channels_present_in_this_mc_at_all: # Definitely absent
                absent_or_rare_channels.append(glob_chan)
            elif (channel_counts_in_mc.get(glob_chan, 0) / num_archetypes_in_mc) < 0.1: # Present, but very rarely
                absent_or_rare_channels.append(glob_chan)
        
        # Sort for consistent label ordering
        positive_channels.sort()
        absent_or_rare_channels.sort()
        
        label = f"MC_{mc_id} (n={num_archetypes_in_mc}):\n  POS: {(', '.join(positive_channels) if positive_channels else 'N/A')[:100]}\n  NEG/RARE: {( ', '.join(absent_or_rare_channels) if absent_or_rare_channels else 'N/A')[:100]}"
        
        characterization[mc_id] = {
            "num_archetypes": num_archetypes_in_mc,
            "positive_channels": positive_channels,
            "absent_or_rare_channels": absent_or_rare_channels,
            "full_label": label,
            "channel_counts_df": channel_counts_in_mc.reset_index().rename(columns={'index': 'channel', 0:'count'})
        }
    return characterization

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize archetype meta-clusters.")
    parser.add_argument("--base_dir", required=True, help="Base output directory where ROI/resolution subdirectories are located (e.g., 'output/').")
    parser.add_argument("--roi_glob", default="ROI_*", help="Glob pattern for ROI directories within base_dir.")
    parser.add_argument("--resolution_glob", default="resolution_*", help="Glob pattern for resolution directories within ROI_dirs.")
    parser.add_argument("--archetype_glob_suffix", default="_channel_archetypes_support*_bin*pct.csv", help="Glob pattern for archetype CSV files.")
    parser.add_argument("--linkage_glob_suffix", default="_channel_archetypes_support*_bin*pct_linkage_*.npy", help="Glob pattern for linkage .npy files.")
    parser.add_argument("--distance_thresholds", type=str, default="0.7,0.85", help="Comma-separated list of Jaccard distance thresholds to try for fcluster (e.g., '0.5,0.7,0.85').")
    parser.add_argument("--results_suffix", default="_metacluster_analysis", help="Suffix for output files and subdirectories from this script.")

    args = parser.parse_args()

    try:
        threshold_list = [float(t.strip()) for t in args.distance_thresholds.split(',')]
    except ValueError:
        print("Error: Invalid distance_thresholds format. Please use comma-separated numbers (e.g., '0.5,0.7').")
        return

    print(f"Searching for archetype and linkage files in: {args.base_dir}")
    print(f"ROI glob: {args.roi_glob}, Resolution glob: {args.resolution_glob}")
    print(f"Archetype suffix: {args.archetype_glob_suffix}, Linkage suffix: {args.linkage_glob_suffix}")
    print(f"Distance thresholds to try: {threshold_list}")

    # Find all potential archetype CSVs first
    archetype_search_pattern = os.path.join(args.base_dir, args.roi_glob, args.resolution_glob, f"*{args.archetype_glob_suffix}")
    all_archetype_csvs = glob.glob(archetype_search_pattern, recursive=False) # Recursive false, as ROI/Res globs handle depth
    print(f"Found {len(all_archetype_csvs)} potential archetype CSV files.")

    for archetypes_csv_path in all_archetype_csvs:
        print(f"\n--- Processing: {archetypes_csv_path} ---")
        current_dir = os.path.dirname(archetypes_csv_path)
        base_archetype_filename = os.path.splitext(os.path.basename(archetypes_csv_path))[0]

        # Try to find a matching linkage file
        # This is a bit complex due to varying linkage methods in the name
        potential_linkage_glob = os.path.join(current_dir, base_archetype_filename + "_linkage_*.npy")
        matching_linkage_files = glob.glob(potential_linkage_glob)
        
        if not matching_linkage_files:
            print(f"  No linkage .npy file found matching pattern: {potential_linkage_glob}. Skipping.")
            continue
        
        linkage_npy_path = matching_linkage_files[0] # Take the first one if multiple (e.g. different linkage methods)
        if len(matching_linkage_files) > 1:
            print(f"  Warning: Found multiple linkage files for {base_archetype_filename}. Using: {linkage_npy_path}")
            print(f"    All found: {matching_linkage_files}")

        print(f"  Using linkage file: {linkage_npy_path}")

        archetypes_df, linkage_matrix = load_archetype_data(archetypes_csv_path, linkage_npy_path)

        if archetypes_df is None or linkage_matrix is None or archetypes_df.empty:
            print("  Failed to load data. Skipping.")
            continue
        if len(archetypes_df) < 2:
            print(f"  Only {len(archetypes_df)} archetype(s) found. Cannot perform clustering. Skipping.")
            continue

        analysis_output_dir = os.path.join(current_dir, base_archetype_filename + args.results_suffix)
        os.makedirs(analysis_output_dir, exist_ok=True)
        print(f"  Analysis outputs will be saved in: {analysis_output_dir}")

        # --- Visualize Full Dendrogram ---
        dendrogram_filename = f"full_dendrogram.png"
        dendrogram_output_path = os.path.join(analysis_output_dir, dendrogram_filename)
        plot_dendrogram(linkage_matrix, len(archetypes_df), dendrogram_output_path, title_prefix=f"{base_archetype_filename}\n")

        # --- Get binary matrix for silhouette scores and heatmaps ---
        binary_archetype_matrix, channel_names = get_binary_archetype_matrix(archetypes_df)
        if binary_archetype_matrix is None:
            print("  Could not generate binary archetype matrix. Skipping further analysis for this file.")
            continue
        
        # Get all unique channels across all archetypes in this specific input file for negative characterization
        all_channels_in_file_set = set()
        for item_list in archetypes_df['itemsets_list']:
            all_channels_in_file_set.update(item_list)
        all_unique_channels_list_for_file = sorted(list(all_channels_in_file_set))

        # --- Iterate through Distance Thresholds ---
        all_clustering_results = []
        for t in threshold_list:
            print(f"\n  Analyzing with distance threshold: {t}")
            cluster_labels = sch.fcluster(linkage_matrix, t=t, criterion='distance')
            num_clusters = len(np.unique(cluster_labels))
            archetypes_df[f'metacluster_dist_{t}'] = cluster_labels
            print(f"    Number of meta-clusters found: {num_clusters}")

            if num_clusters <= 1 or num_clusters >= len(archetypes_df):
                print("    Silhouette score cannot be calculated for 1 cluster or n_samples clusters. Skipping score.")
                avg_silhouette = None
            else:
                try:
                    # Need to calculate actual Jaccard distances for silhouette score, not use linkage matrix directly
                    # The linkage matrix contains merge distances, not pairwise sample distances.
                    # pdist(binary_archetype_matrix, metric='jaccard') gives condensed form
                    pairwise_jaccard_distances = pdist(binary_archetype_matrix, metric='jaccard')
                    # squareform can convert it, but silhouette_score can take the condensed form if metric='precomputed'
                    # However, silhouette_score with 'precomputed' expects a SQUARE distance matrix.
                    # So we'll compute the square form.
                    square_distance_matrix = squareform(pairwise_jaccard_distances)
                    avg_silhouette = silhouette_score(square_distance_matrix, cluster_labels, metric='precomputed')
                    print(f"    Average Silhouette Score: {avg_silhouette:.4f}")
                except ValueError as sve:
                    print(f"    Could not calculate Silhouette Score: {sve} (num_clusters={num_clusters}, archetypes={len(archetypes_df)})")
                    avg_silhouette = None 
                except Exception as sil_e:
                    print(f"    Error calculating Silhouette Score: {sil_e}")
                    avg_silhouette = None
            
            # Characterize and save meta-cluster details
            cluster_col_name = f'metacluster_dist_{t}'
            characteristics = characterize_metaclusters(archetypes_df, cluster_col_name, all_unique_channels_list_for_file)
            summary_lines = [f"Meta-cluster analysis for threshold {t}:", f"  Number of meta-clusters: {num_clusters}", f"  Avg Silhouette Score: {avg_silhouette if avg_silhouette is not None else 'N/A'}", "\nCharacterization:"]
            for mc_id, char_data in characteristics.items():
                summary_lines.append(f"  {char_data['full_label']}")
            
            summary_text_path = os.path.join(analysis_output_dir, f"metacluster_summary_thresh_{t:.2f}.txt")
            with open(summary_text_path, 'w') as f:
                f.write("\n".join(summary_lines))
            print(f"    Meta-cluster characterization saved to: {summary_text_path}")

            # Save detailed channel counts for this clustering
            for mc_id, char_data in characteristics.items():
                counts_df = char_data['channel_counts_df']
                counts_csv_path = os.path.join(analysis_output_dir, f"metacluster_{mc_id}_channel_counts_thresh_{t:.2f}.csv")
                counts_df.to_csv(counts_csv_path, index=False)

            # Visualize Clustered Heatmap
            heatmap_filename = f"heatmap_thresh_{t:.2f}.png"
            heatmap_output_path = os.path.join(analysis_output_dir, heatmap_filename)
            plot_clustered_heatmap(binary_archetype_matrix, channel_names, cluster_labels, heatmap_output_path, title_prefix=f"{base_archetype_filename}\nThresh={t:.2f}, Clusters={num_clusters}\n")

            # Visualize Meta-Cluster Signature Heatmap
            signature_heatmap_filename = f"metacluster_signatures_heatmap_thresh_{t:.2f}.png"
            signature_heatmap_output_path = os.path.join(analysis_output_dir, signature_heatmap_filename)
            plot_metacluster_signature_heatmap(
                characteristics,
                all_unique_channels_list_for_file, 
                signature_heatmap_output_path, 
                title_prefix=f"{base_archetype_filename}\nThresh={t:.2f}, Clusters={num_clusters}\n"
            )
            
            all_clustering_results.append({
                'threshold': t,
                'num_clusters': num_clusters,
                'avg_silhouette': avg_silhouette,
                'summary_path': summary_text_path
            })
        
        # Save a final summary of all thresholds tried for this file
        overall_summary_df = pd.DataFrame(all_clustering_results)
        overall_summary_path = os.path.join(analysis_output_dir, "all_thresholds_summary.csv")
        overall_summary_df.to_csv(overall_summary_path, index=False)
        print(f"\n  Overall summary of thresholding results saved to: {overall_summary_path}")

    print("\n--- Archetype Meta-cluster Analysis Completed ---")

if __name__ == '__main__':
    # Add seaborn for heatmaps
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn library not found. Heatmaps will not be generated. Install with: pip install seaborn")
        # Optionally, you could have a fallback visualization or exit.
    main() 