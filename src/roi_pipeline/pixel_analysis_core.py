import os
import time
import traceback
from typing import List, Tuple, Optional, Dict, Any

# GPU utilities
from .gpu_utils import check_gpu_availability, get_rapids_lib

import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la
from scipy.spatial import KDTree
from scipy.optimize import brentq # For GMM intersection
from scipy.stats import norm # For GMM PDF

# ==============================================================================
# Spatial Clustering (Leiden)
# ==============================================================================

def run_spatial_leiden(
    analysis_df: pd.DataFrame, # Should contain 'X', 'Y' coordinates
    protein_channels: list,    # List of channel names (for validation/info)
    scaled_expression_data_for_weights: np.ndarray,
    n_neighbors: int = 15,
    resolution_param: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[Any], Optional[Any], Optional[float]]:
    """
    Performs pixel-based Leiden community detection on IMC data using pre-scaled expression data.

    Args:
        analysis_df: DataFrame containing pixel coordinates ('X', 'Y'). Index must align with scaled_expression_data_for_weights.
        protein_channels: List of protein channels (used for informational purposes/validation, not scaling).
        scaled_expression_data_for_weights: REQUIRED pre-calculated, scaled expression data
                                            (n_pixels, n_channels). Assumed to be arcsinh transformed
                                            (with appropriate cofactors) and MinMaxScaler scaled.
        n_neighbors: Number of spatial neighbors for the k-NN graph.
        resolution_param: Resolution parameter for the Leiden algorithm.
        seed: Random seed for Leiden.
        verbose: If True, print progress messages.

    Returns:
        A tuple containing:
        - results_df: Copy of analysis_df with 'community' column added (or None on failure).
        - g_pixels: The constructed igraph object (or None on failure).
        - partition: The Leiden partition object (or None on failure).
        - total_time: Time taken for the function execution (or None on failure).

    Raises:
        ValueError: If scaled_expression_data_for_weights is not provided or has incorrect dimensions.
    """
    start_total_time = time.time()
    local_analysis_df = analysis_df.copy() # Operate on a copy
    n_pixels = len(local_analysis_df)
    if verbose:
        print(f"--- Running Spatial Leiden (k={n_neighbors}, res={resolution_param}) on {n_pixels} pixels ---")

    # GPU acceleration check
    use_gpu = check_gpu_availability(verbose=verbose)
    if use_gpu:
        if verbose:
            print(f"   Using GPU-accelerated k-NN search and Leiden clustering.")
    
    # 1. Validate Pre-scaled Expression Data
    if verbose: print("\n1. Validating pre-scaled expression data...")
    if scaled_expression_data_for_weights is None:
         raise ValueError("scaled_expression_data_for_weights must be provided.")
    if scaled_expression_data_for_weights.shape[0] != n_pixels:
         raise ValueError(
             f"Shape mismatch: Pre-scaled data has {scaled_expression_data_for_weights.shape[0]} rows, "
             f"but analysis_df has {n_pixels} rows (pixels)."
         )
    if scaled_expression_data_for_weights.shape[1] != len(protein_channels):
         print(f"   Warning: Pre-scaled data has {scaled_expression_data_for_weights.shape[1]} columns, "
               f"but {len(protein_channels)} protein_channels were listed.")
         # Decide if this is an error or just a warning - depends on workflow
         # For now, proceed assuming the scaled data columns correspond to channels used for weighting

    scaled_data = scaled_expression_data_for_weights # Use the provided data

    # 2. Build Spatial k-NN Graph
    if verbose: print(f"\n2. Building spatial k-NN graph (k={n_neighbors})...")
    start_time = time.time()
    coords = local_analysis_df[['X', 'Y']].values
    if use_gpu:
        cuML = get_rapids_lib('cuml')
        cupy = get_rapids_lib('cupy')
        if cuML is None or cupy is None:
            print("   WARNING: cuML or CuPy not available despite GPU. Falling back to CPU path.")
            use_gpu = False
    
    if use_gpu:
        coords_gpu = cupy.asarray(coords)
        knn = cuML.neighbors.NearestNeighbors(n_neighbors=n_neighbors+1)
        knn.fit(coords_gpu)
        dists_gpu, inds_gpu = knn.kneighbors(coords_gpu)
        # Drop self neighbor at index 0 and flatten
        distances = dists_gpu[:, 1:]
        indices = inds_gpu[:, 1:]
        # Compute weights
        epsilon = 1e-6
        weights_gpu = 1.0 / (distances + epsilon)
        # Create source/target arrays
        k = indices.shape[1]
        src_gpu = cupy.repeat(cupy.arange(n_pixels, dtype='int32'), k)
        dst_gpu = indices.ravel()
        w_gpu = weights_gpu.ravel()
        # Convert to CPU for cudf
        src = src_gpu.get()
        dst = dst_gpu.get()
        w = w_gpu.get()
        cudf = get_rapids_lib('cudf')
        cugraph = get_rapids_lib('cugraph')
        if cudf is not None and cugraph is not None:
            # Build GPU graph
            edges_df = cudf.DataFrame({'src': src, 'dst': dst, 'weight': w})
            G_gpu = cugraph.Graph()
            G_gpu.from_cudf_edgelist(edges_df, source='src', destination='dst', edge_attr='weight', renumber=False)
            if verbose:
                print(f"   GPU k-NN + graph construction took {time.time() - start_time:.2f} seconds.")
            # 5. Run Leiden on GPU
            start_leiden = time.time()
            parts = cugraph.leiden(G_gpu, resolution=resolution_param, weight='weight')
            if verbose:
                print(f"   GPU Leiden took {time.time() - start_leiden:.2f} seconds.")
            # Assign community labels
            local_analysis_df['community'] = parts['partition'].to_pandas().values
            total_time = time.time() - start_total_time
            return local_analysis_df, G_gpu, parts, total_time
        else:
            print("   WARNING: cudf or cugraph not available. Falling back to CPU path.")
            use_gpu = False
    # CPU fallback path
    # Build CPU graph weights
    if verbose:
        print(f"   Using CPU KDTree + igraph + leidenalg.")
    distances_knn, indices_knn = KDTree(coords).query(coords, k=n_neighbors+1)
    nbrs = indices_knn[:, 1:]
    epsilon = 1e-6
    sources, targets, weights = [], [], []
    for i in range(n_pixels):
        for j in nbrs[i]:
            if i < j:
                dist = np.linalg.norm(scaled_data[i] - scaled_data[j])
                sources.append(i); targets.append(j)
                weights.append(1.0 / (dist + epsilon))
    if verbose:
        print(f"   CPU k-NN weight calc took {time.time() - start_time:.2f} seconds.")
    # Construct igraph
    start_graph = time.time()
    g_pixels = ig.Graph(n=n_pixels, directed=False)
    g_pixels.add_edges(zip(sources, targets))
    g_pixels.es['weight'] = weights
    if verbose:
        print(f"   igraph construction took {time.time() - start_graph:.2f} seconds.")
    # Run CPU Leiden
    start_leiden = time.time()
    partition = la.find_partition(g_pixels, la.CPMVertexPartition, weights='weight', resolution_parameter=resolution_param, seed=seed)
    if verbose:
        print(f"   CPU Leiden took {time.time() - start_leiden:.2f} seconds.")
    local_analysis_df['community'] = partition.membership
    total_time = time.time() - start_total_time
    return local_analysis_df, g_pixels, partition, total_time

# ==============================================================================
# Community Profile Calculation
# ==============================================================================

def calculate_and_save_profiles(
    results_df: pd.DataFrame, # This should contain PRE-SCALED data + 'community'
    valid_channels: List[str],
    roi_output_dir: str,
    roi_string: str
) -> Optional[pd.DataFrame]:
    """
    Calculates average community expression profiles from PRE-SCALED data and saves results.

    Args:
        results_df: DataFrame with Leiden results, including 'community' column
                    and pre-scaled expression values in columns named in valid_channels.
        valid_channels: List of protein channel columns containing the pre-scaled data.
        roi_output_dir: Path to save the output CSV files.
        roi_string: ROI identifier used for file naming.

    Returns:
        A pandas DataFrame containing the average scaled expression profile for each community,
        or None if an error occurs (e.g., missing 'community' column or empty data).
    """
    print("\n--- Calculating Average Community Profiles (from pre-scaled data) ---")
    start_time = time.time()
    try:
        if 'community' not in results_df.columns:
            print("ERROR: 'community' column not found in results_df. Cannot calculate profiles.")
            return None
        if not valid_channels:
            print("ERROR: No valid channels provided for profile calculation.")
            return None
        if results_df.empty:
             print("ERROR: Input results_df is empty. Cannot calculate profiles.")
             return None

        # Validate that valid_channels exist in the DataFrame
        missing_channels = [ch for ch in valid_channels if ch not in results_df.columns]
        if missing_channels:
            print(f"ERROR: The following required channels are missing from results_df: {missing_channels}")
            return None

        # --- SCALING LOGIC REMOVED --- Data is pre-scaled

        # Select the scaled data and community labels
        profile_data_scaled = results_df[valid_channels + ['community']]

        # Check for NaNs/Infs in the scaled data before grouping
        if profile_data_scaled[valid_channels].isnull().values.any() or np.isinf(profile_data_scaled[valid_channels].values).any():
            print("   Warning: NaN/Inf values found in pre-scaled data. Imputing with 0 before averaging.")
            # Impute within the relevant columns only
            for channel in valid_channels:
                # Use .loc for safe assignment on copy
                profile_data_scaled.loc[:, channel] = np.nan_to_num(profile_data_scaled[channel].values, nan=0.0, posinf=0.0, neginf=0.0)

        # Group by community to get average profiles
        print("   Grouping by community to calculate average profiles...")
        if 'community' not in profile_data_scaled.columns:
             print("   Internal ERROR: 'community' column lost before grouping.")
             return None # Should not happen based on earlier check

        community_profiles_scaled = profile_data_scaled.groupby('community')[valid_channels].mean()

        # Check if any communities were actually found
        if community_profiles_scaled.empty:
             print("   Warning: No communities found after grouping, or all communities had NaN profiles.")
             return None

        print(f"   Calculated profiles for {len(community_profiles_scaled)} communities.")

        # Save community profiles
        print("   Saving community profiles...")
        profiles_save_path = os.path.join(roi_output_dir, f"community_profiles_scaled_{roi_string}.csv")
        community_profiles_scaled.to_csv(profiles_save_path)
        print(f"   Scaled community profiles saved to: {profiles_save_path}")

        print(f"--- Profile Calculation finished in {time.time() - start_time:.2f} seconds ---")
        return community_profiles_scaled

    except KeyError as ke:
        print(f"ERROR: Missing expected column during profile calculation: {ke}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"An unexpected error occurred during profile calculation or saving: {str(e)}")
        traceback.print_exc()
        return None


# ==============================================================================
# Differential Expression Analysis
# ==============================================================================

def calculate_differential_expression(
    results_df: pd.DataFrame,
    community_profiles: pd.DataFrame,
    graph: ig.Graph,
    valid_channels: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Calculates differential expression profiles for communities vs. their neighbors.

    Args:
        results_df: DataFrame containing pixel data with 'community' assignments and indices matching the graph nodes.
        community_profiles: DataFrame of average scaled expression profiles per community.
        graph: igraph object representing spatial connectivity (nodes must match results_df index).
        valid_channels: List of channel names to calculate differential expression for.

    Returns:
        A tuple containing:
        - community_diff_profiles_df: DataFrame with differential expression profiles per community.
        - community_top_channel_map: Series mapping community ID to its top primary channel.
        Returns (empty DataFrame, empty Series) if calculation cannot proceed.
    """
    print("\nCalculating Differential Expression vs Neighbors...")
    start_time_diff = time.time()
    community_diff_profiles_df = pd.DataFrame()
    community_top_channel_map = pd.Series(dtype=str)

    # Check prerequisites
    if community_profiles.empty or 'community' not in results_df.columns:
        print("   Skipping differential expression: Community profiles empty or 'community' column missing.")
        return pd.DataFrame(), pd.Series(dtype=str)
    if not isinstance(graph, ig.Graph):
         print("   Skipping differential expression: Invalid graph object provided.")
         return pd.DataFrame(), pd.Series(dtype=str)
    if graph.vcount() != len(results_df):
         print(f"   Skipping differential expression: Graph node count ({graph.vcount()}) "
               f"does not match results_df length ({len(results_df)}).")
         return pd.DataFrame(), pd.Series(dtype=str)

    try:
        # Prepare community info and neighbor map structure
        community_profiles_cleaned = community_profiles.copy()
        if not pd.api.types.is_integer_dtype(community_profiles_cleaned.index):
             community_profiles_cleaned.index = community_profiles_cleaned.index.astype(int)

        community_ids = community_profiles_cleaned.index.unique().tolist()
        neighbor_map = {c_id: set() for c_id in community_ids}

        # Map node index (from graph, 0..N-1) to community ID
        print("   Mapping graph nodes to community IDs...")
        # Ensure results_df has an integer index aligned with graph nodes (0 to N-1)
        if not pd.api.types.is_integer_dtype(results_df.index) or not (results_df.index == range(len(results_df))).all():
            print(f"   Warning: results_df index is not a simple 0..{len(results_df)-1} range. Resetting index for mapping.")
            results_df_indexed = results_df.reset_index(drop=True) # Create copy with 0-based index
        else:
             results_df_indexed = results_df # Use directly if index is okay

        # Check if 'community' column exists after potential reset
        if 'community' not in results_df_indexed.columns:
             print("   ERROR: 'community' column missing after index check/reset. Cannot map nodes.")
             return pd.DataFrame(), pd.Series(dtype=str)

        # Create the map: node_index -> community_id
        try:
            community_node_map = results_df_indexed['community'].to_dict()
        except Exception as map_err:
            print(f"   ERROR: Failed to create node-to-community map. {map_err}")
            return pd.DataFrame(), pd.Series(dtype=str)


        print("   Finding neighbors from graph edges...")
        start_neighbor_map = time.time()
        max_node_index = graph.vcount() - 1 # Max valid node index in 0-based graph

        for edge in graph.es:
            source_node, target_node = edge.tuple
            # Check if node indices are valid within the graph's range
            if 0 <= source_node <= max_node_index and 0 <= target_node <= max_node_index:
                c1 = community_node_map.get(source_node)
                c2 = community_node_map.get(target_node)
                # Check communities exist in profiles and are different before adding edge
                if c1 is not None and c2 is not None and c1 != c2 and c1 in neighbor_map and c2 in neighbor_map:
                    neighbor_map[c1].add(c2)
                    neighbor_map[c2].add(c1)
            # else: print(f"Debug: Invalid edge node index found: {edge.tuple} (max index: {max_node_index})")

        print(f"   Neighbor mapping took {time.time() - start_neighbor_map:.2f} seconds.")

        # Calculate differential profiles
        start_time_diff_calc = time.time()
        community_diff_profiles_dict = {}
        primary_channels = {}

        print("   Calculating differential expression...")
        for c_id in community_ids:
            neighbors = neighbor_map.get(c_id, set())
            # Ensure neighbors exist in the calculated profiles' index
            valid_neighbors = [n for n in neighbors if n in community_profiles_cleaned.index]

            if not valid_neighbors:
                # print(f"    Community {c_id} has no valid neighbors.") # Verbose
                primary_channels[c_id] = 'NoNeighbors'
                community_diff_profiles_dict[c_id] = pd.Series(0, index=valid_channels)
                continue

            # Calculate average profile of valid neighbors
            neighbor_profiles = community_profiles_cleaned.loc[valid_neighbors, valid_channels]
            avg_neighbor_profile = neighbor_profiles.mean()

            # Get the community's own profile
            community_profile = community_profiles_cleaned.loc[c_id, valid_channels]

            # Calculate the difference vector
            diff_vector = community_profile - avg_neighbor_profile
            community_diff_profiles_dict[c_id] = diff_vector

            # Determine the primary channel (max positive difference)
            if not diff_vector.empty and diff_vector.max() > 0:
                top_channel = diff_vector.idxmax()
            else:
                top_channel = 'None' # No channel has higher expression than neighbors average
            primary_channels[c_id] = top_channel

        # Create DataFrame from the difference profiles
        community_diff_profiles_df = pd.DataFrame.from_dict(community_diff_profiles_dict, orient='index')
        community_diff_profiles_df.index.name = 'community' # Name the index
        if not community_diff_profiles_df.empty:
            community_diff_profiles_df.fillna(0, inplace=True)
        else:
             print("   Warning: Differential profile DataFrame is empty after calculation.")

        # Create Series for mapping primary channel back
        primary_channel_map = pd.Series(primary_channels, name='primary_channel')
        primary_channel_map.index.name = 'community' # Name the index

        print(f"   Differential expression calculation took {time.time() - start_time_diff_calc:.2f} seconds.")

    except Exception as e:
        print(f"   Error during differential expression calculation: {e}")
        traceback.print_exc()
        # Return empty structures on error
        return pd.DataFrame(), pd.Series(dtype=str)

    print(f"--- Differential Expression Analysis finished in {time.time() - start_time_diff:.2f} seconds ---")
    return community_diff_profiles_df, primary_channel_map 