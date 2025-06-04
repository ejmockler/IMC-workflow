import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial.distance import pdist, squareform # For distance calculations
from skbio.stats.distance import permanova          # For PERMANOVA
from skbio import DistanceMatrix                      # skbio interface for distance matrix
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

# --- Moved from run_experiment_analysis.py/data_aggregation.py ---
def get_channel_columns(df: pd.DataFrame, config: Dict) -> List[str]:
    """Identifies channel columns, prioritizing master list from config if available.

    Falls back to excluding known metadata/processing columns.
    """
    if df is None or df.empty:
        return []

    # Priority 1: Use lists from config if provided
    available_channels = config.get('data', {}).get('protein_channels')+config.get('data', {}).get('background_channels')
    if available_channels and isinstance(available_channels, list):
        # Ensure the columns actually exist in the dataframe
        channel_cols = [col for col in available_channels if col in df.columns]
        if len(channel_cols) < len(available_channels):
            missing_in_df = [col for col in available_channels if col not in df.columns]
            print(f"   Warning: Some channels from config master list not found in DataFrame: {missing_in_df}")
        # print(f"   Identified {len(channel_cols)} channel columns using master list from config.") # Reduced verbosity
        return channel_cols

    # Priority 2: Fallback to exclusion logic
    # print("   Identifying channel columns using exclusion logic (master list not provided or invalid).") # Reduced verbosity
    metadata_df_cols = list(config.get('metadata_cols', [])) # Metadata cols defined in root
    exp_config = config.get('experiment_analysis', {})
    # Columns explicitly defined in experiment_analysis section of config
    metadata_cols_from_exp_config = [
        exp_config.get('condition_col'), exp_config.get('timepoint_col'),
        exp_config.get('region_col'), exp_config.get('replicate_col'),
        exp_config.get('metadata_roi_col') # The actual ROI col name from metadata file
    ]
    # Columns potentially added during processing
    known_added_cols = ['roi_string', 'community', 'resolution', 'meta_cluster', 'roi_standard_key'] # Added roi_standard_key too

    # Combine all known non-channel columns
    non_channel_cols = set(known_added_cols + metadata_cols_from_exp_config + metadata_df_cols)
    non_channel_cols.discard(None) # Remove None if some config values are missing

    channel_cols = [col for col in df.columns if col not in non_channel_cols]

    # Basic check: If way too many columns are left, something might be wrong
    if len(channel_cols) > 100: # Arbitrary threshold
        print(f"   Warning: Identified {len(channel_cols)} potential channel columns via exclusion. This seems high. Check config metadata/channel definitions?")
    elif not channel_cols:
         print("   Error: No potential channel columns identified via exclusion.")
    # else:
        # print(f"   Identified {len(channel_cols)} potential channel columns using exclusion logic.") # Reduced verbosity

    return channel_cols
# --- End moved function ---

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

# --- Helper Function: Stratified Permutation ---
def _permute_within_groups(grouping: pd.Series, strata: pd.Series) -> pd.Series:
    """
    Permutes the grouping labels within each stratum group.

    Args:
        grouping: Series of group assignments for each sample.
        strata: Series indicating the stratum for each sample (aligned with grouping).

    Returns:
        A new Series with grouping labels permuted within strata.
    """
    permuted_grouping = grouping.copy()
    # Ensure indices match for assignment
    if not grouping.index.equals(strata.index):
         raise ValueError("Grouping and Strata indices must match for permutation.")

    unique_strata_values = strata.unique()
    all_indices = grouping.index # Get all indices once

    for stratum_val in unique_strata_values:
        # Get boolean mask first, then select indices
        mask_in_stratum = (strata == stratum_val)
        indices_in_stratum = all_indices[mask_in_stratum]

        if len(indices_in_stratum) > 1: # Only permute if more than one sample in stratum
             original_values = grouping.loc[indices_in_stratum].values
             permuted_values = np.random.permutation(original_values)
             permuted_grouping.loc[indices_in_stratum] = permuted_values
        # If len <= 1, no permutation needed, values remain the same

    return permuted_grouping

# --- Helper Function: Calculate Pseudo-F Statistic ---
def _calculate_pseudo_f(dm_matrix: np.ndarray, grouping: pd.Series) -> Tuple[float, float, float, float]:
    """
    Calculates the PERMANOVA pseudo-F statistic from a square distance matrix.

    Args:
        dm_matrix: Squareform distance matrix (n_samples x n_samples).
        grouping: Series of group assignments for each sample (index must align with dm_matrix order implicitly).

    Returns:
        Tuple: (pseudo_F, ssb, ssw, sst)
               Returns (np.nan, np.nan, np.nan, np.nan) if calculation fails (e.g., division by zero).
    """
    n_samples = dm_matrix.shape[0]
    if n_samples != len(grouping):
        raise ValueError("Distance matrix shape must match grouping length.")

    # Ensure grouping is aligned with dm_matrix implicitly (caller responsibility, checked in _perform_stratified_permanova_test)
    group_values = grouping.values # Use numpy array for faster indexing if possible
    unique_groups, group_inverse = np.unique(group_values, return_inverse=True)
    n_groups = len(unique_groups)

    # Total Sum of Squares (SST) - Using average squared distance
    sst = np.sum(dm_matrix**2) / (2 * n_samples)

    # Within-Group Sum of Squares (SSW)
    ssw = 0
    df_within = 0 # Degrees of freedom within groups
    for i, group_val in enumerate(unique_groups):
        # Get boolean mask for samples in the current group
        in_group_mask = (group_inverse == i)
        n_group = np.sum(in_group_mask) # Number of samples in this group

        if n_group > 1:
            # Calculate sum of squares for distances within this group
            # Use boolean mask indexing on the square distance matrix
            ssw += np.sum(dm_matrix[in_group_mask][:, in_group_mask]**2) / (2 * n_group)
            df_within += n_group - 1 # n_g - 1 for each group g
        # If n_group is 1, SSW contribution is 0, df_within contribution is 0

    # Between-Group Sum of Squares (SSB)
    ssb = sst - ssw
    df_between = n_groups - 1 # k - 1 groups

    # Pseudo-F Statistic
    # Handle potential division by zero if SSW is ~0 or df_within/df_between is 0
    msb = ssb / df_between if df_between > 0 else 0
    msw = ssw / df_within if df_within > 0 else 0

    # Check against machine epsilon for floating point comparisons
    if msw <= np.finfo(float).eps * sst: # Compare MSW relative to SST's scale
         if msb <= np.finfo(float).eps * sst: # If MSB is also effectively zero relative to SST
             pseudo_f = 0.0 # Define 0/0 or small/small as 0 F-stat
         else: # MSB > 0 and MSW = 0 -> infinite F-statistic
             pseudo_f = np.inf
    else:
         pseudo_f = msb / msw

    # Ensure non-negative SSB and SSW due to potential floating point issues
    ssb = max(0, ssb)
    ssw = max(0, ssw)

    return pseudo_f, ssb, ssw, sst

# --- Core PERMANOVA Test Function ---
def _perform_stratified_permanova_test(
    distance_matrix: DistanceMatrix,
    grouping: pd.Series,
    strata: Optional[pd.Series] = None,
    permutations: int = 999
) -> Dict[str, Any]:
    """
    Performs the core PERMANOVA test with optional stratification.

    Args:
        distance_matrix: skbio DistanceMatrix object.
        grouping: Series of group assignments for each sample, index MUST align with distance_matrix.ids.
        strata: Series indicating stratum for each sample (optional), index MUST align.
        permutations: Number of permutations.

    Returns:
        Dictionary containing test results.
    """
    if not isinstance(distance_matrix, DistanceMatrix):
         raise TypeError("distance_matrix must be a skbio.DistanceMatrix")
    if not isinstance(grouping, pd.Series):
         raise TypeError("grouping must be a pandas Series")
    if strata is not None and not isinstance(strata, pd.Series):
         raise TypeError("strata must be None or a pandas Series")

    # --- Input Validation and Alignment ---
    # Convert distance matrix IDs tuple to pandas Index for comparison
    dm_ids_index = pd.Index(distance_matrix.ids)

    # Ensure grouping index matches distance matrix index
    if not dm_ids_index.equals(grouping.index):
         try:
             # Use the Index version for reindexing
             grouping = grouping.reindex(dm_ids_index)
             print("   Debug: Reindexed grouping for PERMANOVA test.")
             if grouping.isnull().any():
                 # Check which IDs caused NaNs
                 nan_ids = grouping[grouping.isnull()].index.tolist()
                 raise ValueError(f"Grouping contains NaNs after reindexing to distance matrix IDs. Problematic IDs: {nan_ids[:5]}")
         except Exception as e:
             raise ValueError(f"Grouping index ({grouping.index.dtype}) could not be aligned with distance matrix IDs ({dm_ids_index.dtype}): {e}")

    # Ensure strata index matches distance matrix index (if strata provided)
    if strata is not None:
         if not dm_ids_index.equals(strata.index):
             try:
                 # Use the Index version for reindexing
                 strata = strata.reindex(dm_ids_index)
                 print("   Debug: Reindexed strata for PERMANOVA test.")
                 if strata.isnull().any():
                     # Check which IDs caused NaNs
                     nan_ids = strata[strata.isnull()].index.tolist()
                     raise ValueError(f"Strata contains NaNs after reindexing to distance matrix IDs. Problematic IDs: {nan_ids[:5]}")
             except Exception as e:
                 raise ValueError(f"Strata index ({strata.index.dtype}) could not be aligned with distance matrix IDs ({dm_ids_index.dtype}): {e}")
    # --- End Validation ---

    # Use the underlying squareform numpy array for calculations
    dm_matrix = distance_matrix.data # Already square
    n_samples = dm_matrix.shape[0]
    n_groups = len(grouping.unique())

    # Handle cases with insufficient data
    if n_groups <= 1:
        print(f"   Warning: Only {n_groups} group found for factor '{grouping.name}'. Cannot perform PERMANOVA.")
        return {'Term': grouping.name, 'F': np.nan, 'p-value': np.nan, 'N': n_samples, 'Permutations': 0}
    if n_samples < 3:
         print(f"   Warning: Only {n_samples} samples available for factor '{grouping.name}'. Cannot perform PERMANOVA.")
         return {'Term': grouping.name, 'F': np.nan, 'p-value': np.nan, 'N': n_samples, 'Permutations': 0}


    # 1. Calculate observed statistic
    try:
        obs_f, obs_ssb, obs_ssw, obs_sst = _calculate_pseudo_f(dm_matrix, grouping)
    except Exception as e:
        print(f"ERROR calculating observed F-statistic for {grouping.name}: {e}")
        return {'Term': grouping.name, 'F': np.nan, 'p-value': np.nan, 'N': n_samples, 'Permutations': 0, 'Error': str(e)}


    # If observed F is NaN (e.g., only 1 group, or error) or no permutations requested, return early
    if np.isnan(obs_f) or permutations == 0:
        df_between = n_groups - 1
        return {
            'Term': grouping.name,
            'SS': obs_ssb if not np.isnan(obs_ssb) else np.nan,
            'MS': obs_ssb / df_between if df_between > 0 and not np.isnan(obs_ssb) else np.nan,
            'F': obs_f, # Already NaN
            'p-value': np.nan,
            'N': n_samples,
            'Permutations': 0
        }

    # 2. Permutation loop
    perm_f_stats = np.zeros(permutations)
    count_extreme = 0
    for i in range(permutations):
        try:
            # Permute grouping labels (stratified or unconstrained)
            if strata is not None:
                # Strata index already aligned with grouping index due to checks above
                permuted_grouping = _permute_within_groups(grouping, strata)
            else:
                # Simple permutation of values, keeping original index
                permuted_values = np.random.permutation(grouping.values)
                permuted_grouping = pd.Series(permuted_values, index=grouping.index, name=grouping.name)

            # Calculate F-statistic for permuted data
            perm_f, _, _, _ = _calculate_pseudo_f(dm_matrix, permuted_grouping)

            # Store permuted F and count if >= observed F
            if np.isnan(perm_f):
                # Handle NaN in permuted F - could happen if permutation creates only one group.
                # Assign a value guaranteed to be less than any valid F-stat.
                perm_f_stats[i] = -np.inf
            else:
                perm_f_stats[i] = perm_f
                # Check if permuted stat is >= observed stat (using tolerance)
                # Handle obs_f = inf separately
                if np.isinf(obs_f): # If obs_f is infinite, only count if perm_f is also infinite
                     if np.isinf(perm_f):
                         count_extreme += 1
                elif perm_f >= obs_f - (np.finfo(float).eps * abs(obs_f)): # Relative tolerance
                     count_extreme += 1

        except Exception as perm_e:
             print(f"   Warning: Error during permutation {i+1} for factor '{grouping.name}': {perm_e}. Skipping permutation.")
             perm_f_stats[i] = -np.inf # Treat as non-extreme if an error occurred

    # 3. Calculate p-value
    p_value = (count_extreme + 1) / (permutations + 1)

    df_between = n_groups - 1
    df_within = n_samples - n_groups

    return {
        'Term': grouping.name,
        'SS': obs_ssb,
        'MS': obs_ssb / df_between if df_between > 0 else np.nan,
        'F': obs_f,
        'p-value': p_value,
        'N': n_samples,
        'Permutations': permutations,
        # Optional: Add back SSW, SST, dfs if needed for R-squared or full ANOVA table
        # '_SSW': obs_ssw,
        # '_SST': obs_sst,
        # '_df_between': df_between,
        # '_df_within': df_within
    }

# --- Modified run_composition_permanova ---
def run_composition_permanova(
    composition_matrix: pd.DataFrame,
    metadata_df: pd.DataFrame,
    config: Dict[str, Any],
    formula: str, # Keep formula for factor extraction
    distance_metric: str = 'braycurtis',
    permutations: int = 999
) -> Optional[pd.DataFrame]:
    """Performs PERMANOVA on the community composition matrix based on metadata factors.

    Runs separate tests for each main factor extracted from the formula.
    Uses internal implementation supporting stratified permutations.

    Args:
        composition_matrix: Wide DataFrame (ROI x Community) from prepare_composition_table.
                            Index must be the standard analysis ROI string.
        metadata_df: DataFrame containing full experiment metadata (must include 'roi_standard_key').
        config: The configuration dictionary (used for replicate_col).
        formula: R-style formula specifying factors from metadata (e.g., "Condition + Day + Condition*Day").
                 Interaction terms are ignored for testing here.
        distance_metric: Distance metric for pdist (e.g., 'braycurtis', 'euclidean').
        permutations: Number of permutations for significance testing.

    Returns:
        A pandas DataFrame containing the combined PERMANOVA results for main factors, or None on error.
    """
    print(f"\n--- Running PERMANOVA (Metric: {distance_metric}, Formula: {formula}) ---")
    # Removed note about library version, now using internal implementation
    # print("   Note: Running separate tests for main factors due to library version.")
    if composition_matrix is None or composition_matrix.empty:
        print("ERROR: Input composition matrix is empty or None.")
        return None
    # Check for standard key in metadata
    if metadata_df is None or metadata_df.empty or 'roi_standard_key' not in metadata_df.columns:
        print("ERROR: Input metadata DataFrame is empty, None, or missing 'roi_standard_key' column.")
        return None

    # Get replicate column from config for constrained permutations
    exp_config = config.get('experiment_analysis', {})
    replicate_col = exp_config.get('replicate_col')
    strata_series = None # Initialize strata series

    try:
        # --- Align metadata (Ensures index matches composition_matrix) ---
        print("   Aligning metadata with composition matrix (using roi_standard_key)...")
        if metadata_df.index.name != 'roi_standard_key':
             metadata_indexed = metadata_df.set_index('roi_standard_key')
        else:
             metadata_indexed = metadata_df.copy() # Avoid modifying original

        # Reindex metadata to match the order and content of the composition matrix's index
        # This ensures direct alignment for grouping, strata, etc.
        aligned_metadata = metadata_indexed.reindex(composition_matrix.index)

        # --- Handle Missing Metadata ---
        rois_before_drop = len(aligned_metadata)
        aligned_metadata = aligned_metadata.dropna(how='all') # Drop rows that are ALL NaN after reindex
        valid_rois_mask = aligned_metadata.index.isin(composition_matrix.index) # Ensure they still exist in composition
        aligned_metadata = aligned_metadata[valid_rois_mask]
        rois_after_drop = len(aligned_metadata)

        if rois_after_drop < rois_before_drop:
             print(f"   Warning: Dropped {rois_before_drop - rois_after_drop} ROIs with missing metadata after alignment.")

        if len(aligned_metadata) < 3: # Need at least 3 samples for distance/permanova
             print(f"ERROR: Less than 3 ROIs remaining after aligning metadata ({len(aligned_metadata)}). Cannot run PERMANOVA.")
             return None

        # Update composition matrix to only include valid ROIs
        composition_matrix = composition_matrix.loc[aligned_metadata.index]
        print(f"   Proceeding with {len(aligned_metadata)} ROIs with aligned metadata.")

        # --- Extract Factors and Prepare Strata ---
        # This simple split assumes terms are separated by '+' and ignores interactions/covariates
        # A more robust parser (e.g., from patsy) could be used for complex formulas
        main_effect_terms = [term.strip() for term in formula.split('+') if '*' not in term and ':' not in term]
        # Validate factors against available columns AFTER dropping NaNs
        valid_factors = [term for term in main_effect_terms if term in aligned_metadata.columns]

        if not valid_factors:
             print(f"ERROR: No valid main effect terms from formula '{formula}' found in aligned metadata columns.")
             print(f"       Available metadata columns: {aligned_metadata.columns.tolist()}")
             return None

        # Prepare strata if replicate column is valid *in the aligned metadata*
        if replicate_col and replicate_col in aligned_metadata.columns:
            # Ensure strata does not contain NaNs for the selected ROIs
            if aligned_metadata[replicate_col].isnull().any():
                 print(f"   Warning: Replicate column '{replicate_col}' contains missing values for some analyzed ROIs. Strata cannot be used reliably.")
                 strata_series = None # Do not use strata if it has NaNs
            else:
                 strata_series = aligned_metadata[replicate_col]
                 print(f"   Applying strata based on '{replicate_col}'.")
        elif replicate_col:
             print(f"   Warning: Replicate column '{replicate_col}' (from config) not found in aligned metadata. Strata not used.")
             # strata_series remains None

        # --- Calculate Distance Matrix ---
        print(f"   Calculating {distance_metric} distance matrix...")
        if (composition_matrix < 0).values.any():
             print("Warning: Negative values found in composition matrix. Check input or choose appropriate metric.")

        # Use pdist for pairwise distances, then convert to DistanceMatrix
        dist_array = pdist(composition_matrix.values, metric=distance_metric)
        dist_matrix = DistanceMatrix(squareform(dist_array), ids=composition_matrix.index)
        print(f"   Distance matrix shape: {dist_matrix.shape}")

        # --- Run PERMANOVA for each factor ---
        all_results_list = []
        print(f"   Running PERMANOVA separately for factors: {valid_factors} ...")
        for factor in valid_factors:
            print(f"      Testing factor: {factor}")
            # Extract the grouping vector for the current factor from aligned metadata
            grouping_series = aligned_metadata[factor]

            # Drop ROIs where the *current factor* is missing, if any (should be minimal after initial drop)
            valid_grouping_mask = grouping_series.notna()
            if not valid_grouping_mask.all():
                 print(f"   Warning: Factor '{factor}' has missing values. Removing {np.sum(~valid_grouping_mask)} samples for this test.")
                 current_grouping = grouping_series[valid_grouping_mask]
                 current_dist_matrix = dist_matrix.filter(ids=current_grouping.index)
                 current_strata = strata_series.loc[current_grouping.index] if strata_series is not None else None
            else:
                 current_grouping = grouping_series
                 current_dist_matrix = dist_matrix
                 current_strata = strata_series # Already aligned

            if len(current_grouping.unique()) <= 1 or len(current_grouping) < 3:
                 print(f"         ... Skipping factor '{factor}': Not enough samples ({len(current_grouping)}) or groups ({len(current_grouping.unique())}) after handling missing values for this factor.")
                 continue

            try:
                # Call the internal PERMANOVA implementation
                result_dict = _perform_stratified_permanova_test(
                    distance_matrix=current_dist_matrix,
                    grouping=current_grouping,
                    strata=current_strata, # Pass the aligned strata series
                    permutations=permutations
                )
                result_dict['Term'] = factor # Ensure Term reflects the factor tested
                all_results_list.append(result_dict)
                print(f"         ... Done testing {factor}")

            except Exception as factor_e:
                 print(f"ERROR: Failed PERMANOVA for factor '{factor}': {factor_e}")
                 print(f"         ... Skipping factor {factor}")
                 # Add a placeholder error entry?
                 all_results_list.append({'Term': factor, 'Error': str(factor_e)})


        if not all_results_list:
            print("ERROR: PERMANOVA failed for all factors or no results were generated.")
            return None

        # Combine results from all factors into a DataFrame
        final_results_df = pd.DataFrame(all_results_list)
        # Reorder columns for clarity? e.g., ['Term', 'N', 'SS', 'MS', 'F', 'p-value', 'Permutations']
        cols_order = ['Term', 'N', 'SS', 'MS', 'F', 'p-value', 'Permutations', 'Error']
        final_results_df = final_results_df.reindex(columns=[col for col in cols_order if col in final_results_df.columns])

        print("   PERMANOVA finished for all factors.")
        return final_results_df

    # Keep original broad exception handling
    except ImportError:
        # This might still be relevant if DistanceMatrix or pdist fails, though skbio.stats.distance is removed
        print("ERROR: Required library (e.g., pandas, numpy, scipy, skbio) not found or version mismatch.")
        return None
    except Exception as e:
        print(f"ERROR: Failed during PERMANOVA preparation or execution: {e}")
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

        # --- MODIFICATION: Give Total Cells a tuple name for sorting ---
        if not final_composition.empty:
            total_cells_per_cluster = total_cells_per_cluster.rename(('Info', 'Total Cells')) # Assign tuple name
            final_composition = pd.concat([total_cells_per_cluster, final_composition], axis=1) # Add total cell counts
        else:
            # If only total cells exist (no metadata cols analyzed)
            final_composition = pd.DataFrame(total_cells_per_cluster)
            # Still give it a tuple name if needed, though sorting might not be necessary
            final_composition.columns = pd.MultiIndex.from_tuples([('Info', 'Total Cells')])
        # --- END MODIFICATION ---

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
