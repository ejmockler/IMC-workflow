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
from scipy.optimize import brentq # For GMM intersection
from scipy.stats import norm # For GMM PDF

try:
    import umap
    _umap_import_success = True
except ImportError:
    _umap_import_success = False
# --- Add Faiss import ---
try:
    import faiss
    faiss_available = True
except ImportError:
    faiss_available = False

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
    use_gpu_config: bool = False,  # New parameter to control GPU usage
) -> Tuple[Optional[pd.DataFrame], Optional[Any], Optional[Any], Optional[float]]:
    """
    Performs pixel-based Leiden community detection on IMC data using pre-scaled expression data.
    Uses Faiss (GPU or CPU) for Approximate Nearest Neighbors (ANN) if available.
    Leiden clustering is performed using cuGraph (GPU) if available, otherwise leidenalg (CPU).
    **Faiss is required for the k-NN step.**

    Args:
        analysis_df: DataFrame containing pixel coordinates ('X', 'Y'). Index must align with scaled_expression_data_for_weights.
        protein_channels: List of protein channels (used for informational purposes/validation, not scaling).
        scaled_expression_data_for_weights: REQUIRED pre-calculated, scaled expression data
                                            (n_pixels, n_channels). Assumed to be arcsinh transformed
                                            (with appropriate cofactors) and MinMaxScaler scaled.
                                            **Should be float32 for memory optimization.**
        n_neighbors: Number of spatial neighbors for the k-NN graph.
        resolution_param: Resolution parameter for the Leiden algorithm.
        seed: Random seed for Leiden.
        verbose: If True, print progress messages.
        use_gpu_config: If True, attempt to use GPU (Faiss-GPU, cuGraph-GPU) if available, otherwise use CPU (Faiss-CPU, leidenalg-CPU).

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

    # --- Determine Execution Path (GPU Faiss > CPU Faiss for kNN; GPU cuGraph > CPU leidenalg for Leiden) ---
    if not faiss_available:
        print("   ERROR: Faiss library is not installed or importable, which is required for k-NN. Aborting.")
        return None, None, None, None
        
    gpu_available = check_gpu_availability(verbose=False)
    should_try_gpu = use_gpu_config and gpu_available
    
    # Check RAPIDS libraries needed for GPU Leiden
    # cuML is no longer checked here as it's not used for KNN fallback
    cupy, cudf, cugraph = None, None, None
    rapids_libs_available_for_leiden = False
    if should_try_gpu:
        # cuML = get_rapids_lib('cuml') # No longer needed for KNN
        cupy = get_rapids_lib('cupy')
        cudf = get_rapids_lib('cudf')
        cugraph = get_rapids_lib('cugraph')
        if cupy and cudf and cugraph: # Check only libs needed for Leiden
            rapids_libs_available_for_leiden = True
            
    # Path flags for kNN
    use_faiss_gpu = should_try_gpu # Faiss availability checked earlier
    use_faiss_cpu = not use_faiss_gpu # If not trying GPU, use CPU Faiss

    if verbose:
        print("   k-NN Method Selection:")
        if use_faiss_gpu: print("      Attempting GPU k-NN using Faiss-GPU.")
        elif use_faiss_cpu: print("      Using CPU k-NN using Faiss-CPU.")
        # No else needed, as faiss_available was checked upfront

        print("   Leiden Method Selection:")
        if should_try_gpu and rapids_libs_available_for_leiden: print("      Attempting GPU Leiden using cuGraph.")
        else: print("      Using CPU Leiden using leidenalg (GPU/RAPIDS not requested or available).")
    # --- End Path Decision ---
    
    # 1. Validate Pre-scaled Expression Data
    if verbose: print("\n1. Validating pre-scaled expression data...")
    if scaled_expression_data_for_weights is None:
         raise ValueError("scaled_expression_data_for_weights must be provided.")
    if scaled_expression_data_for_weights.shape[0] != n_pixels:
         raise ValueError(
             f"Shape mismatch: Pre-scaled data has {scaled_expression_data_for_weights.shape[0]} rows, "
             f"but analysis_df has {n_pixels} rows (pixels)."
         )
    # Allow mismatch for now, but warn. Consider making this stricter if needed.
    if scaled_expression_data_for_weights.shape[1] != len(protein_channels):
        print(f"   Warning: Pre-scaled data has {scaled_expression_data_for_weights.shape[1]} columns, "
              f"but {len(protein_channels)} protein_channels were listed. Using data columns.")
        # Update protein_channels list if dimensions mismatch? Or assume data is correct?
        # For now, assume data is correct and proceed, but this might impact downstream interpretation if list is wrong.

    # Ensure expression data is float32
    try:
        scaled_data = scaled_expression_data_for_weights.astype(np.float32)
    except Exception as e:
        print(f"   Warning: Could not cast scaled_expression_data to float32: {e}. Using original dtype.")
        scaled_data = scaled_expression_data_for_weights

    # 2. Build Spatial k-NN Graph
    if verbose: print(f"\n2. Building spatial k-NN graph (k={n_neighbors})...")
    knn_start_time = time.time()
    
    # Ensure coordinates are float32
    try:
        coords = local_analysis_df[['X', 'Y']].values.astype(np.float32)
    except Exception as e:
        print(f"   Warning: Could not cast coordinates to float32: {e}. Using original dtype.")
        coords = local_analysis_df[['X', 'Y']].values
        
    src = None
    dst = None
    weights = None
    knn_method_used = "None"

    # --- k-NN Calculation: Faiss GPU or Faiss CPU --- 
    if use_faiss_gpu:
        knn_method_used = "Faiss-GPU"
        if verbose: print("   Executing k-NN with Faiss-GPU...")
        try:
            res = faiss.StandardGpuResources() # Initialize GPU resources
            cpu_index = faiss.IndexFlatL2(coords.shape[1]) # Build CPU index first
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index) # Move index to GPU 0
            
            gpu_index.add(coords) # Add data (NumPy array is fine)
            distances, indices = gpu_index.search(coords, n_neighbors + 1)
            
            # Process results (remove self, type casting)
            distances = distances[:, 1:].astype(np.float32)
            indices = indices[:, 1:].astype(np.int32)
            
            epsilon = 1e-6
            weights = (1.0 / (distances + epsilon)).astype(np.float32)
            src = np.repeat(np.arange(n_pixels, dtype=np.int32), n_neighbors)
            dst = indices.ravel()
            weights = weights.ravel()
            
            del res, cpu_index, gpu_index, distances, indices # Cleanup
        except Exception as faiss_gpu_e:
            print(f"   ERROR during Faiss-GPU k-NN: {faiss_gpu_e}. Falling back to CPU Faiss.")
            traceback.print_exc()
            use_faiss_gpu = False # Ensure CPU path is taken next
            use_faiss_cpu = True  # Explicitly set CPU path
            src, dst, weights = None, None, None # Reset for fallback attempt
            # No need to cleanup variables here, NameError handled if they weren't assigned

    if use_faiss_cpu and src is None: # Run if GPU wasn't attempted OR if GPU failed
        knn_method_used = "Faiss-CPU"
        if verbose: print("   Executing k-NN with Faiss-CPU...")
        try:
            index = faiss.IndexFlatL2(coords.shape[1])
            index.add(coords)
            distances, indices = index.search(coords, n_neighbors + 1)
            distances = distances[:, 1:].astype(np.float32)
            indices = indices[:, 1:].astype(np.int32)
            epsilon = 1e-6
            weights = (1.0 / (distances + epsilon)).astype(np.float32)
            src = np.repeat(np.arange(n_pixels, dtype=np.int32), n_neighbors)
            dst = indices.ravel()
            weights = weights.ravel()
            del index, distances, indices
        except Exception as faiss_cpu_e:
            print(f"   ERROR during Faiss-CPU k-NN: {faiss_cpu_e}. Faiss is required, cannot proceed.")
            traceback.print_exc()
            # Since Faiss (CPU) failed, and it's required, we must abort.
            return None, None, None, None

    # Check if neighbor finding was successful (only Faiss attempts now)
    if src is None or dst is None or weights is None:
        print("   ERROR: Failed to compute neighbor lists and weights using Faiss (GPU or CPU). Cannot proceed.")
        # This case should ideally be caught by the Faiss-CPU exception handling above,
        # but added as a safeguard.
        return None, None, None, None
    else:
        if verbose: print(f"   k-NN search ({knn_method_used}) took {time.time() - knn_start_time:.2f} seconds.")

    # 3. Build igraph Object
    if verbose: print("\n3. Building Graph Object...")
    graph_start_time = time.time()
    try:
        g_pixels = ig.Graph(n=n_pixels, edges=list(zip(src, dst)), edge_attrs={'weight': weights}, directed=False)
        g_pixels.simplify(combine_edges='sum') # Combine parallel edges if any
        if verbose: print(f"   Graph construction took {time.time() - graph_start_time:.2f} seconds.")
        del src, dst, weights # Free memory after graph creation
    except Exception as graph_e:
        print(f"   ERROR building igraph object: {graph_e}")
        traceback.print_exc()
        return None, None, None, None

    # 4. Run Leiden Algorithm
    if verbose: print("\n4. Running Leiden Community Detection...")
    leiden_start_time = time.time()
    partition = None
    communities = None 
    n_communities = 0
    # Determine if we should still try GPU Leiden (even if kNN was CPU)
    # Check for specific libs needed for cugraph Leiden
    use_gpu_leiden = should_try_gpu and rapids_libs_available_for_leiden
    leiden_method_used = "None"

    if use_gpu_leiden:
        leiden_method_used = "cuGraph-GPU"
        if verbose: print("   Attempting Leiden with cuGraph-GPU...")
        try:
            # Convert igraph to cuGraph DataFrame representation
            edges_df = cudf.DataFrame({
                'src': cupy.array(g_pixels.get_edge_list())[:, 0].astype('int32'),
                'dst': cupy.array(g_pixels.get_edge_list())[:, 1].astype('int32'),
                'weight': cupy.array(g_pixels.es['weight']).astype('float32')
            })
            
            G_gpu = cugraph.Graph()
            G_gpu.from_cudf_edgelist(edges_df, source='src', destination='dst', edge_attr='weight', renumber=False)
            del edges_df # Free memory
            
            leiden_df, _ = cugraph.leiden(G_gpu, resolution=resolution_param, random_state=seed)
            
            partition_df = leiden_df.to_pandas().set_index('vertex').sort_index()
            communities = partition_df['partition'].values
            n_communities = len(partition_df['partition'].unique())
            partition = communities # Store assignments array
            
            del G_gpu, leiden_df, partition_df # Free GPU memory
            cupy.get_default_memory_pool().free_all_blocks()
        except Exception as gpu_leiden_e:
            print(f"   ERROR during GPU Leiden: {gpu_leiden_e}. Falling back to CPU Leiden.")
            traceback.print_exc()
            use_gpu_leiden = False # Ensure CPU Leiden runs
            partition = None # Reset partition
            communities = None
            try: # Cleanup GPU memory
                 del G_gpu, leiden_df, partition_df
                 cupy.get_default_memory_pool().free_all_blocks()
            except NameError: pass
                 
    if not use_gpu_leiden: # Use CPU Leiden (leidenalg)
        leiden_method_used = "leidenalg-CPU"
        if verbose: print("   Executing Leiden with leidenalg-CPU...")
        try:
            partition = la.find_partition(
                g_pixels, 
                la.RBConfigurationVertexPartition, # Or other partition type if needed
                weights='weight', 
                resolution_parameter=resolution_param, 
                seed=seed
            )
            communities = np.array(partition.membership)
            n_communities = len(partition)
        except Exception as cpu_leiden_e:
            print(f"   ERROR during CPU Leiden: {cpu_leiden_e}")
            traceback.print_exc()
            return None, None, None, None # Cannot proceed without communities

    if verbose: print(f"   Leiden ({leiden_method_used}) found {n_communities} communities in {time.time() - leiden_start_time:.2f} seconds.")

    # 5. Assign Communities and Return
    if verbose: print("\n5. Assigning communities to DataFrame...")
    try:
        local_analysis_df['community'] = communities
        local_analysis_df['community'] = local_analysis_df['community'].astype('category') # Use category for memory efficiency
    except Exception as assign_e:
         print(f"   ERROR assigning community labels: {assign_e}")
         traceback.print_exc()
         return None, None, None, None
         
    total_time = time.time() - start_total_time
    if verbose:
        print(f"--- Spatial Leiden finished in {total_time:.2f} seconds ---")

    # Return the dataframe with communities, the graph, and the partition object
    # Note: partition might be just the community array if GPU Leiden was used
    return local_analysis_df, g_pixels, partition, total_time

# ==============================================================================
# Community Profile Calculation
# ==============================================================================

def calculate_and_save_profiles(
    results_df: pd.DataFrame, # This should contain PRE-SCALED data + 'community'
    valid_channels: List[str],
    roi_output_dir: str,
    roi_string: str,
    ordered_channels: Optional[List[str]] = None # Add optional ordered list
) -> Optional[pd.DataFrame]:
    """Calculates and saves the mean scaled expression profile for each community.

    Args:
        results_df: DataFrame with pixel data, must include 'community' and scaled channel columns.
        valid_channels: List of channel columns to include in the profile calculation.
        roi_output_dir: Directory to save the output CSV.
        roi_string: String identifier for the ROI/resolution (used in filename).
        ordered_channels: Optional list of channels to define the column order in the saved CSV.

    Returns:
        DataFrame of scaled community profiles (communities x channels) or None if fails.
    """
    print("   Calculating community expression profiles...")
    start_time = time.time()

    if 'community' not in results_df.columns:
        print("   ERROR: 'community' column not found in results_df. Cannot calculate profiles.")
        return None

    # Ensure only channels present in the dataframe are used
    channels_in_df = [ch for ch in valid_channels if ch in results_df.columns]
    if not channels_in_df:
        print("   ERROR: No valid channel columns found in results_df for profile calculation.")
        return None

    try:
        # Group by community and calculate mean for the scaled channel columns
        # Use only channels present in the dataframe for the calculation
        scaled_community_profiles = results_df.groupby('community')[channels_in_df].mean()

        if scaled_community_profiles.empty:
             print("   Warning: No communities found or profile calculation resulted in empty DataFrame.")
             # Save an empty file?
             empty_profiles_path = os.path.join(roi_output_dir, f"community_profiles_{roi_string}.csv")
             pd.DataFrame().to_csv(empty_profiles_path)
             print(f"   Saved empty community profiles file: {os.path.basename(empty_profiles_path)}")
             return scaled_community_profiles # Return the empty DF

        # --- Reorder columns before saving if ordered_channels is provided ---
        if ordered_channels is not None:
            print(f"   Applying specified channel order ({len(ordered_channels)} channels) before saving.")
            # Find channels common to both the desired order and the calculated profiles
            final_ordered_columns = [ch for ch in ordered_channels if ch in scaled_community_profiles.columns]
            missing_in_profiles = [ch for ch in ordered_channels if ch not in scaled_community_profiles.columns]
            extra_in_profiles = [ch for ch in scaled_community_profiles.columns if ch not in ordered_channels]

            if missing_in_profiles:
                print(f"     Warning: Channels from ordered list not found in profiles: {missing_in_profiles}")
            if extra_in_profiles:
                print(f"     Warning: Channels in profiles not in ordered list (will be dropped from saved file): {extra_in_profiles}")
            if not final_ordered_columns:
                 print("     ERROR: No common channels between ordered list and profiles. Cannot apply order. Saving with original order.")
                 # Fallback to original order if no overlap
            else:
                 # Reindex using the common, ordered list
                 scaled_community_profiles = scaled_community_profiles[final_ordered_columns]
        else:
             print("   Saving profiles with original column order.")
        # --- End Reordering ---

        # Save the profiles
        profiles_path = os.path.join(roi_output_dir, f"community_profiles_{roi_string}.csv")
        scaled_community_profiles.to_csv(profiles_path)
        print(f"   Community profiles saved to: {os.path.basename(profiles_path)}")
        print(f"   --- Profile calculation finished in {time.time() - start_time:.2f} seconds ---")
        return scaled_community_profiles

    except Exception as e:
        print(f"   ERROR calculating or saving community profiles: {e}")
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


# ==============================================================================
# UMAP Calculation
# ==============================================================================

def calculate_and_save_umap(
    diff_expr_profiles: Optional[pd.DataFrame],
    scaled_community_profiles: Optional[pd.DataFrame],
    roi_channels: List[str],
    resolution_output_dir: str,
    roi_string: str,
    resolution_param: float,
    config: Dict,
    umap_available: bool # Pass the check result from the main script
) -> Optional[pd.DataFrame]:
    """
    Calculates UMAP embedding based on community profiles (preferentially differential) and saves coordinates.

    Args:
        diff_expr_profiles: DataFrame of differential expression profiles (communities x channels).
        scaled_community_profiles: DataFrame of average scaled expression profiles (communities x channels).
        roi_channels: List of all valid protein channel names for the ROI.
        resolution_output_dir: Directory to save the UMAP coordinates CSV.
        roi_string: ROI identifier string.
        resolution_param: The resolution parameter used for this analysis step.
        config: The pipeline configuration dictionary.
        umap_available: Boolean flag indicating if the umap-learn package was successfully imported.

    Returns:
        DataFrame containing UMAP coordinates (UMAP1, UMAP2, ...) with community IDs as index, or None if skipped/failed.
    """
    print("\n--- Running UMAP Analysis --- ")
    if not umap_available:
        print("   Skipping UMAP: umap-learn package not available.")
        return None
    if not _umap_import_success:
         print("   Skipping UMAP: umap-learn package failed to import within pixel_analysis_core.")
         return None

    cfg_analysis = config.get('analysis', {})
    cfg_umap = cfg_analysis.get('umap', {})
    cfg_clustering = cfg_analysis.get('clustering', {})
    cfg_diffex = cfg_analysis.get('differential_expression', {})

    umap_coords = None
    start_time_umap = time.time()

    try:
        # Determine input data for UMAP (prioritize DiffExpr if available and valid)
        umap_input_data = None
        input_data_source = "None"
        communities_in_order = None

        if diff_expr_profiles is not None and not diff_expr_profiles.empty:
            non_protein_markers = cfg_diffex.get('non_protein_markers_for_umap', [])
            protein_marker_channels_for_umap = [
                ch for ch in diff_expr_profiles.columns
                if ch in roi_channels and ch not in non_protein_markers
            ]
            if protein_marker_channels_for_umap:
                umap_input_data = diff_expr_profiles[protein_marker_channels_for_umap].copy()
                communities_in_order = diff_expr_profiles.index.tolist()
                input_data_source = "Differential Profiles"
                print(f"   Using Differential Profiles with {len(protein_marker_channels_for_umap)} protein markers for UMAP.")
            else:
                print("   No protein markers found in differential profiles.")

        if umap_input_data is None and scaled_community_profiles is not None and not scaled_community_profiles.empty:
            print("   Falling back to using Scaled Community Profiles for UMAP.")
            # Use all channels available in scaled profiles if using them
            umap_input_data = scaled_community_profiles[list(set(roi_channels) & set(scaled_community_profiles.columns))].copy()
            communities_in_order = scaled_community_profiles.index.tolist()
            input_data_source = "Scaled Profiles"

        if umap_input_data is not None and communities_in_order is not None:
            # Clean input data
            if umap_input_data.isnull().values.any() or np.isinf(umap_input_data.values).any():
                 print("   Warning: NaN/Inf values found in UMAP input. Replacing with 0.")
                 umap_input_data = umap_input_data.fillna(0).replace([np.inf, -np.inf], 0)

            n_communities = len(umap_input_data)
            umap_n_neighbors = min(cfg_umap.get('n_neighbors', 15), n_communities - 1) if n_communities > 1 else 1
            current_umap_n_components = max(2, cfg_umap.get('n_components', 2))
            umap_metric = cfg_umap.get('metric', 'euclidean')
            umap_min_dist = cfg_umap.get('min_dist', 0.1)
            umap_seed = cfg_clustering.get('seed', 42)

            if n_communities > umap_n_neighbors and n_communities >= current_umap_n_components:
                 print(f"   Embedding {n_communities} communities into {current_umap_n_components} dimensions (k={umap_n_neighbors}, metric={umap_metric})...")
                 # UMAP calculation (CPU/GPU)
                 use_gpu_umap = check_gpu_availability(verbose=False)
                 cuml = get_rapids_lib('cuml')
                 cupy = get_rapids_lib('cupy')

                 if use_gpu_umap and cuml and cupy:
                     print("      Using GPU-accelerated UMAP (cuML).")
                     try:
                         input_gpu = cupy.asarray(umap_input_data.values.astype(np.float32)) # Ensure float32 for cuML
                         umap_gpu = cuml.UMAP(
                             n_neighbors=umap_n_neighbors,
                             min_dist=umap_min_dist,
                             n_components=current_umap_n_components,
                             metric=umap_metric,
                             random_state=umap_seed
                         )
                         embedding_gpu = umap_gpu.fit_transform(input_gpu)
                         embedding = embedding_gpu.get() if hasattr(embedding_gpu, 'get') else embedding_gpu
                         del input_gpu, umap_gpu, embedding_gpu # Clear GPU memory
                         if cupy:
                            mempool = cupy.get_default_memory_pool()
                            mempool.free_all_blocks()
                     except Exception as gpu_err:
                         print(f"      ERROR during GPU UMAP execution: {gpu_err}")
                         print("      Falling back to CPU UMAP.")
                         use_gpu_umap = False # Force fallback
                         embedding = None
                 else:
                      use_gpu_umap = False # Ensure flag is false if libs not found

                 if not use_gpu_umap or embedding is None:
                     if use_gpu_umap: # only print warning if it was attempted
                        print("      WARNING: cuML/CuPy not available or GPU UMAP failed. Using CPU UMAP.")
                     else:
                         print("      Using CPU UMAP (umap-learn). ")
                     try:
                        umap_reducer = umap.UMAP(
                            n_neighbors=umap_n_neighbors,
                            min_dist=umap_min_dist,
                            n_components=current_umap_n_components,
                            metric=umap_metric,
                            random_state=umap_seed
                        )
                        embedding = umap_reducer.fit_transform(umap_input_data.values)
                     except Exception as cpu_err:
                          print(f"     ERROR during CPU UMAP execution: {cpu_err}")
                          embedding = None
                          traceback.print_exc()

                 # Save UMAP coordinates if embedding was successful
                 if embedding is not None:
                     umap_component_names = [f'UMAP{i+1}' for i in range(current_umap_n_components)]
                     # Ensure index is correct type (e.g., int if community IDs are ints)
                     try:
                         index_type = umap_input_data.index.dtype
                         umap_coords = pd.DataFrame(embedding, index=communities_in_order, columns=umap_component_names)
                         umap_coords.index = umap_coords.index.astype(index_type)
                         umap_coords.index.name = 'community' # Explicitly name the index
                     except Exception as index_err:
                         print(f"    ERROR setting index for UMAP coordinates: {index_err}")
                         umap_coords = None # Fail saving if index is wrong

                     if umap_coords is not None:
                         umap_coords_path = os.path.join(resolution_output_dir, f"umap_coords_{roi_string}_res_{resolution_param}.csv")
                         try:
                             umap_coords.to_csv(umap_coords_path)
                             print(f"      UMAP coordinates saved to: {os.path.basename(umap_coords_path)}")
                         except Exception as save_err:
                             print(f"      ERROR saving UMAP coordinates to {umap_coords_path}: {save_err}")
                             umap_coords = None # Indicate failure if saving fails
                 else:
                     print("      UMAP embedding calculation failed.")
                     umap_coords = None

            else:
                 print(f"   Skipping UMAP embedding: Not enough communities ({n_communities}) vs neighbors ({umap_n_neighbors}) or components ({current_umap_n_components}).")
                 umap_coords = None
        else:
            print("   Skipping UMAP: No suitable input data (scaled or diff profiles found or provided).")
            umap_coords = None

    except Exception as umap_err:
         print(f"   ERROR during UMAP analysis setup or execution: {umap_err}")
         traceback.print_exc() # Print stack trace for UMAP errors
         umap_coords = None # Ensure umap_coords is None if error occurs

    print(f"--- UMAP Analysis finished in {time.time() - start_time_umap:.2f} seconds ---")
    return umap_coords 