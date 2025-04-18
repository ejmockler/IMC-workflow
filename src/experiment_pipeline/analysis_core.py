import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial.distance import pdist, squareform # For distance calculations
from skbio.stats.distance import permanova          # For PERMANOVA
from skbio import DistanceMatrix                      # skbio interface for distance matrix
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

def calculate_community_abundance(
    aggregated_pixels_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    config: Dict[str, Any],
    grouping_col: str
) -> Optional[pd.DataFrame]:
    """Calculates abundance (pixel count/proportion) of each group (community or meta-cluster) per ROI.

    Args:
        aggregated_pixels_df: DataFrame with pixel data including 'roi_string' and grouping_col.
        metadata_df: DataFrame containing experiment metadata.
        config: The configuration dictionary (for metadata ROI column name).
        grouping_col: The column name to group pixels by ('community' or 'meta_cluster').

    Returns:
        A pandas DataFrame summarizing group abundance per ROI, merged with metadata, or None.
    """
    print(f"\n--- Calculating Abundance (using '{grouping_col}') ---")
    if aggregated_pixels_df is None or aggregated_pixels_df.empty:
        print("ERROR: Aggregated pixel data is empty or None.")
        return None
    if metadata_df is None or metadata_df.empty:
        print("ERROR: Metadata is empty or None.")
        return None
    # Check for roi_string and the dynamic grouping column
    if 'roi_string' not in aggregated_pixels_df.columns or grouping_col not in aggregated_pixels_df.columns:
        print(f"ERROR: Aggregated pixel data missing 'roi_string' or '{grouping_col}' column.")
        return None

    roi_col_name = config.get('experiment_analysis', {}).get('metadata_roi_col')
    if not roi_col_name or roi_col_name not in metadata_df.columns:
         print(f"ERROR: Metadata ROI column '{roi_col_name}' is invalid or not found.")
         return None

    try:
        # Count pixels per group per ROI
        print(f"   Counting pixels per '{grouping_col}' per ROI...")
        group_counts = aggregated_pixels_df.groupby(['roi_string', grouping_col]).size().reset_index(name='pixel_count')

        # Calculate total pixels per ROI
        print("   Calculating total pixels per ROI...")
        total_pixels_per_roi = aggregated_pixels_df.groupby('roi_string').size().reset_index(name='total_roi_pixels')

        # Merge counts with totals
        abundance_df = pd.merge(group_counts, total_pixels_per_roi, on='roi_string', how='left')

        # Calculate proportion, handling potential division by zero
        abundance_df['proportion'] = abundance_df['pixel_count'] / abundance_df['total_roi_pixels'].replace(0, np.nan)
        if abundance_df['proportion'].isnull().any():
             print("   Warning: NaNs produced during proportion calculation (possibly total_roi_pixels=0 for some ROIs).")
             abundance_df = abundance_df.dropna(subset=['proportion'])

        print(f"   Calculated abundance for {abundance_df.shape[0]} group-ROI combinations.")

        # 5. Merge abundance summary with metadata using flexible matching
        print("   Merging abundance summary with metadata (flexible matching)...")
        metadata_df[roi_col_name] = metadata_df[roi_col_name].astype(str)
        abundance_df['roi_string'] = abundance_df['roi_string'].astype(str)

        merged_data_list = []
        unique_roi_strings_abund = abundance_df['roi_string'].unique()
        processed_roi_strings_abund = set()
        skipped_roi_strings_no_match_abund = set()
        skipped_roi_strings_multi_match_abund = set()

        for roi_str in unique_roi_strings_abund:
            matching_meta_rows = metadata_df[metadata_df[roi_col_name].str.contains(roi_str, na=False, regex=False)]
            if len(matching_meta_rows) == 1:
                roi_abundance_data = abundance_df[abundance_df['roi_string'] == roi_str]
                meta_row_values = matching_meta_rows.iloc[0]
                temp_merged = roi_abundance_data.assign(**{col: meta_row_values[col] for col in metadata_df.columns if col != roi_col_name})
                merged_data_list.append(temp_merged)
                processed_roi_strings_abund.add(roi_str)
            elif len(matching_meta_rows) == 0:
                skipped_roi_strings_no_match_abund.add(roi_str)
            else:
                skipped_roi_strings_multi_match_abund.add(roi_str)

        merged_abundance_df = None
        if merged_data_list:
            merged_abundance_df = pd.concat(merged_data_list, ignore_index=True)
            print(f"   Final merged abundance data shape: {merged_abundance_df.shape}")
            print(f"   Processed {len(processed_roi_strings_abund)} unique ROI strings for abundance merge.")
        else:
            print("   ERROR: Flexible abundance merge failed to combine any data.")
            # Keep merged_abundance_df as None

        if skipped_roi_strings_no_match_abund:
             print(f"   Warning: {len(skipped_roi_strings_no_match_abund)} ROI strings found no match in metadata for abundance (e.g., '{next(iter(skipped_roi_strings_no_match_abund))}').")
        if skipped_roi_strings_multi_match_abund:
             print(f"   Warning: {len(skipped_roi_strings_multi_match_abund)} ROI strings found multiple matches in metadata for abundance (e.g., '{next(iter(skipped_roi_strings_multi_match_abund))}'). Abundance data skipped.")

        return merged_abundance_df

    except Exception as e:
        print(f"ERROR: Failed during abundance calculation or merging: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_composition_table(
    abundance_summary_df: pd.DataFrame,
    value_col: str = 'proportion',
    grouping_col: str = 'community'
) -> Optional[pd.DataFrame]:
    """Reshapes long-format abundance data to wide-format ROI x Group matrix.

    Args:
        abundance_summary_df: DataFrame output from calculate_community_abundance.
        value_col: The column ('proportion' or 'pixel_count') to use as values.
        grouping_col: The column name used for grouping ('community' or 'meta_cluster').

    Returns:
        A pandas DataFrame with ROIs as index, groups as columns, and abundance as values.
    """
    print(f"\n--- Preparing Composition Table (ROI x '{grouping_col}' using '{value_col}') ---")
    if abundance_summary_df is None or abundance_summary_df.empty:
        print("ERROR: Input abundance summary DataFrame is empty or None.")
        return None

    # Check for roi_string, value_col, and the dynamic grouping column
    required_cols = ['roi_string', grouping_col, value_col]
    if not all(col in abundance_summary_df.columns for col in required_cols):
        print(f"ERROR: Input DataFrame missing one or more required columns: {required_cols}")
        return None

    try:
        # Use pivot_table to reshape
        composition_matrix = pd.pivot_table(
            abundance_summary_df,
            values=value_col,
            index='roi_string',
            columns=grouping_col,
            fill_value=0
        )

        print(f"   Successfully created composition matrix.")
        print(f"   Shape (ROIs x {grouping_col}s): {composition_matrix.shape}")

        if composition_matrix.isnull().values.any():
            print("   Warning: Composition matrix contains NaNs after pivoting. Filling with 0.")
            composition_matrix = composition_matrix.fillna(0)

        return composition_matrix

    except Exception as e:
        print(f"ERROR: Failed to pivot abundance data into composition matrix: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_composition_permanova(
    composition_matrix: pd.DataFrame,
    metadata_df: pd.DataFrame,
    config: Dict[str, Any],
    formula: str,
    distance_metric: str = 'braycurtis',
    permutations: int = 999
) -> Optional[pd.DataFrame]:
    """Performs PERMANOVA on the community composition matrix based on metadata factors.

    Args:
        composition_matrix: Wide DataFrame (ROI x Community) from prepare_composition_table.
        metadata_df: DataFrame containing full experiment metadata.
        config: The configuration dictionary (for metadata ROI column name).
        formula: R-style formula specifying factors from metadata (e.g., "Condition + Day").
        distance_metric: Distance metric for pdist (e.g., 'braycurtis', 'euclidean').
        permutations: Number of permutations for significance testing.

    Returns:
        A pandas DataFrame containing the PERMANOVA results, or None on error.
    """
    print(f"\n--- Running PERMANOVA (Metric: {distance_metric}, Formula: {formula}) ---")
    if composition_matrix is None or composition_matrix.empty:
        print("ERROR: Input composition matrix is empty or None.")
        return None
    if metadata_df is None or metadata_df.empty:
        print("ERROR: Input metadata DataFrame is empty or None.")
        return None

    roi_col_name = config.get('experiment_analysis', {}).get('metadata_roi_col')
    if not roi_col_name or roi_col_name not in metadata_df.columns:
         print(f"ERROR: Metadata ROI column '{roi_col_name}' is invalid or not found.")
         return None

    # Get replicate column from config for constrained permutations
    exp_config = config.get('experiment_analysis', {})
    replicate_col = exp_config.get('replicate_col')
    strata = None
    if replicate_col:
        if replicate_col in metadata_df.columns:
            print(f"   Using '{replicate_col}' column from config for constrained permutations (strata).")
            # We will apply strata after aligning metadata
        else:
            print(f"   Warning: Replicate column '{replicate_col}' (from config) not found in metadata. Permutations will not be constrained.")
            replicate_col = None # Ensure we don't try to use it later
    else:
         print("   Info: 'replicate_col' not specified in config. Permutations will not be constrained by replicate.")


    try:
        # Align metadata with composition matrix index
        print("   Aligning metadata with composition matrix...")
        aligned_metadata = metadata_df.set_index(roi_col_name).reindex(composition_matrix.index)

        # Check for ROIs missing metadata after alignment
        missing_meta_rois = aligned_metadata[aligned_metadata.isnull().any(axis=1)].index.tolist()
        if missing_meta_rois:
            print(f"   Warning: Metadata missing or incomplete for {len(missing_meta_rois)} ROIs found in composition matrix.")
            print(f"   Example missing ROIs: {missing_meta_rois[:5]}")
            # Drop ROIs with missing metadata before proceeding
            valid_rois = aligned_metadata.dropna().index
            if len(valid_rois) < 3: # Need at least 3 samples for distance/permanova
                 print("ERROR: Less than 3 ROIs remaining after removing those with missing metadata. Cannot run PERMANOVA.")
                 return None
            print(f"   Proceeding with {len(valid_rois)} ROIs with complete metadata.")
            aligned_metadata = aligned_metadata.loc[valid_rois]
            composition_matrix = composition_matrix.loc[valid_rois]

        # Check if formula columns exist in aligned metadata
        # (Simple check, formula parsing could be more robust)
        formula_terms = [term.strip() for term in formula.replace('+', ' ').replace('*', ' ').replace(':',' ').split()]
        missing_formula_cols = [term for term in formula_terms if term not in aligned_metadata.columns]
        if missing_formula_cols:
             print(f"ERROR: Formula term(s) not found in metadata columns: {missing_formula_cols}")
             print(f"       Available metadata columns: {aligned_metadata.columns.tolist()}")
             return None

        # Calculate Distance Matrix
        print(f"   Calculating {distance_metric} distance matrix...")
        if (composition_matrix < 0).values.any():
             print("Warning: Negative values found in composition matrix. Check input or choose appropriate metric.")

        dist_array = pdist(composition_matrix.values, metric=distance_metric)
        dist_matrix = DistanceMatrix(squareform(dist_array), ids=composition_matrix.index)
        print(f"   Distance matrix shape: {dist_matrix.shape}")

        # Run PERMANOVA
        print(f"   Running permanova with {permutations} permutations...")
        grouping_df = aligned_metadata[[col for col in formula_terms if col in aligned_metadata.columns]]

        # Handle strata for constrained permutations
        strata = None
        if replicate_col and replicate_col in aligned_metadata.columns:
            strata = aligned_metadata[replicate_col]
        elif replicate_col: # It was specified but wasn't in aligned_metadata (shouldn't happen often with alignment)
             print(f"   Warning: Replicate column '{replicate_col}' found in original metadata but not after alignment/dropna. Strata not used.")

        permanova_results = permanova(
            distance_matrix=dist_matrix,
            grouping=grouping_df,
            column=formula, # Use the formula string directly if skbio version supports it
                             # Or pass specific column name if testing single factor
            permutations=permutations,
            strata=strata
        )
        # Note: skbio permanova interface for formula might vary slightly by version.
        # Ensure your version supports R-style formulas directly in the 'column' arg,
        # otherwise adjustments might be needed.

        print("   PERMANOVA finished.")
        results_df = permanova_results
        if isinstance(results_df, pd.Series):
            results_df = results_df.to_frame().reset_index()

        return results_df

    except ImportError:
         print("ERROR: scikit-bio package not found. Please install it (`pip install scikit-bio`) to run PERMANOVA.")
         return None
    except Exception as e:
        print(f"ERROR: Failed during PERMANOVA execution: {e}")
        import traceback
        traceback.print_exc()
        return None

def cluster_profiles_hierarchical(
    profiles_with_meta_df: pd.DataFrame,
    channel_cols: List[str],
    distance_threshold: Optional[float] = None, # Use distance threshold
    n_clusters_fallback: Optional[int] = None, # Optional fallback if threshold yields <= 1 cluster
    metric: str = 'euclidean',
    linkage_method: str = 'ward',
    scale_data: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """Performs hierarchical clustering on profiles, using distance threshold for cutting.

    Args:
        profiles_with_meta_df: DataFrame containing aggregated profiles AND metadata.
        channel_cols: List of column names corresponding to protein channels.
        distance_threshold: The distance threshold (t) for forming flat clusters (fcluster criterion='distance').
        n_clusters_fallback: If distance threshold yields <= 1 cluster, optionally fall back to cutting by this number (criterion='maxclust').
        metric: Distance metric for clustering (e.g., 'euclidean', 'cosine').
        linkage_method: Linkage method for hierarchical clustering (e.g., 'ward', 'average').
        scale_data: Whether to apply StandardScaler to channel data before clustering.

    Returns:
        A tuple containing:
        - DataFrame: The input DataFrame with an added 'meta_cluster' column (or None on error).
        - np.ndarray: The linkage matrix Z (or None on error).
    """
    cutoff_criteria = f"distance_threshold={distance_threshold}" if distance_threshold else f"n_clusters_fallback={n_clusters_fallback}"
    print(f"\n--- Performing Hierarchical Meta-Clustering ({cutoff_criteria}, Metric: {metric}, Linkage: {linkage_method}) ---")

    if profiles_with_meta_df is None or profiles_with_meta_df.empty:
        print("ERROR: Input profiles DataFrame is empty or None.")
        return None, None
    if not all(col in profiles_with_meta_df.columns for col in channel_cols):
        missing = [col for col in channel_cols if col not in profiles_with_meta_df.columns]
        print(f"ERROR: Input DataFrame missing required channel columns: {missing}")
        return None, None
    if not distance_threshold and not n_clusters_fallback:
        print("ERROR: Must provide either 'distance_threshold' or 'n_clusters_fallback'.")
        return None, None
    if distance_threshold is not None and distance_threshold <= 0:
        print(f"ERROR: Invalid distance_threshold ({distance_threshold}). Must be > 0.")
        return None, None
    if n_clusters_fallback is not None and (n_clusters_fallback <= 0 or n_clusters_fallback >= len(profiles_with_meta_df)):
         print(f"ERROR: Invalid n_clusters_fallback ({n_clusters_fallback}). Must be > 0 and < number of profiles ({len(profiles_with_meta_df)}).")
         return None, None

    # Ensure linkage method is compatible with metric (Ward requires Euclidean)
    if linkage_method == 'ward' and metric != 'euclidean':
        print(f"Warning: Ward linkage requires Euclidean metric. Overriding metric to 'euclidean'.")
        metric = 'euclidean'

    try:
        # Extract profile data
        profile_data = profiles_with_meta_df[channel_cols].values

        # Standardize data
        if scale_data:
            print("   Scaling profile data using StandardScaler...")
            scaler = StandardScaler()
            profile_data_scaled = scaler.fit_transform(profile_data)
        else:
            profile_data_scaled = profile_data

        print("   Calculating linkage matrix...")
        Z = linkage(profile_data_scaled, method=linkage_method, metric=metric)

        # Cut the tree to get flat cluster labels
        cluster_labels = None
        num_unique_labels = 0

        if distance_threshold is not None:
            print(f"   Cutting tree using distance threshold t={distance_threshold}...")
            cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')
            num_unique_labels = len(np.unique(cluster_labels))
            print(f"   Distance threshold yielded {num_unique_labels} unique meta-cluster labels.")

        # Optional fallback to n_clusters if threshold cut results in only 1 cluster (or fewer)
        if num_unique_labels <= 1 and n_clusters_fallback is not None:
             print(f"   Distance threshold resulted in {num_unique_labels} cluster(s). Falling back to n_clusters_fallback={n_clusters_fallback}.")
             cluster_labels = fcluster(Z, t=n_clusters_fallback, criterion='maxclust')
             num_unique_labels = len(np.unique(cluster_labels))
             print(f"   Fallback clustering yielded {num_unique_labels} unique meta-cluster labels.")
        elif num_unique_labels <= 1:
             print("ERROR: Distance threshold resulted in <= 1 cluster and no n_clusters_fallback provided. Cannot proceed.")
             return None, None

        # Add labels back to the DataFrame
        output_df = profiles_with_meta_df.copy()
        output_df['meta_cluster'] = cluster_labels

        print(f"   Final assignment: {num_unique_labels} unique meta-cluster labels.")

        return output_df, Z

    except Exception as e:
        print(f"ERROR: Failed during hierarchical clustering: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Helper function to safely calculate proportions
def calculate_proportions(series):
    counts = series.value_counts()
    total = counts.sum()
    if total == 0:
        return counts.astype(float) * np.nan # Return NaNs if total is zero
    else:
        return (counts / total) * 100

def analyze_metacluster_composition(
    profiles_with_meta_df: pd.DataFrame,
    metacluster_col: str,
    metadata_cols_to_analyze: List[str],
    config: Dict[str, Any] # Keep config for potential future use, though not strictly needed now
) -> Optional[pd.DataFrame]:
    """Analyzes the composition of meta-clusters based on specified metadata columns.

    Args:
        profiles_with_meta_df: DataFrame containing single-cell profiles, metadata,
                               and the metacluster assignment column.
        metacluster_col: Name of the column containing metacluster labels.
        metadata_cols_to_analyze: List of metadata column names to analyze distribution within clusters.
        config: Configuration dictionary.

    Returns:
        A pandas DataFrame summarizing the composition (counts and percentages)
        of each metacluster for each specified metadata column, or None on error.
    """
    print(f"\n--- Analyzing Meta-Cluster Composition ({metacluster_col}) ---")
    if profiles_with_meta_df is None or profiles_with_meta_df.empty:
        print("ERROR: Input profiles DataFrame is empty or None.")
        return None
    if not metacluster_col or metacluster_col not in profiles_with_meta_df.columns:
        print(f"ERROR: Meta-cluster column '{metacluster_col}' not found in DataFrame.")
        return None
    missing_meta_cols = [col for col in metadata_cols_to_analyze if col not in profiles_with_meta_df.columns]
    if missing_meta_cols:
        print(f"ERROR: Metadata columns not found in DataFrame: {missing_meta_cols}")
        return None

    try:
        print(f"   Analyzing distribution of: {metadata_cols_to_analyze}")
        all_results = []

        grouped = profiles_with_meta_df.groupby(metacluster_col)
        total_cells_per_cluster = grouped.size().rename('Total Cells')

        for meta_col in metadata_cols_to_analyze:
            # Calculate counts
            counts_df = grouped[meta_col].value_counts().unstack(fill_value=0)

            # Calculate proportions
            proportions_df = grouped[meta_col].apply(calculate_proportions).unstack(fill_value=0)

            # Combine counts and proportions with multi-level columns
            combined_df = pd.concat(
                [counts_df, proportions_df],
                axis=1,
                keys=[f'{meta_col}_Count', f'{meta_col}_Percent']
            )
            all_results.append(combined_df)

        # Concatenate results for all metadata columns
        if not all_results:
             print("   Warning: No metadata columns provided or processed.")
             return pd.DataFrame(total_cells_per_cluster) # Return just total cell counts

        final_composition = pd.concat(all_results, axis=1)
        final_composition = pd.concat([total_cells_per_cluster, final_composition], axis=1) # Add total cell counts

        # Sort columns for readability
        final_composition = final_composition.sort_index(axis=1)

        print(f"   Successfully analyzed composition for {len(grouped)} meta-clusters.")
        print(f"   Resulting composition table shape: {final_composition.shape}")

        return final_composition

    except Exception as e:
        print(f"ERROR: Failed during meta-cluster composition analysis: {e}")
        import traceback
        traceback.print_exc()
        return None
