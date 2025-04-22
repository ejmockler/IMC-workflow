import os
import re
import time
import json
import traceback
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import MinMaxScaler
import scipy.cluster.hierarchy as sch

# ==============================================================================
# Helper Functions
# ==============================================================================
def extract_roi(filename: str) -> str:
    """Extracts ROI identifier from filename using regex.

    Args:
        filename: The full path or basename of the file.

    Returns:
        The extracted ROI string (e.g., "ROI_Sam1_03_4" or "ROI_Test01_1").
    """
    # Look for patterns like ROI_xxx_##_##
    match = re.search(r'(ROI_\w+_\d+(?:_\d+)?)', filename)
    if match:
        return match.group(1)
    
    # If no match is found, raise an error instead of using a fallback
    raise ValueError(f"Could not extract standard ROI format from {filename}.")

# ==============================================================================
# Cofactor Calculation
# ==============================================================================

def optimized_gaussian_fit_cofactor(channel_data: pd.Series, initial_range=(1, 200), plot_fits=False) -> float:
    """
    Find the optimal arcsinh cofactor using continuous optimization to find
    the value that makes the transformed data most closely approximate a Gaussian.

    Parameters:
    - channel_data: Series or array of channel intensity values
    - initial_range: Tuple with (min, max) range to search for cofactor
    - plot_fits: If True, will plot the optimal fit (Requires matplotlib)

    Returns:
    - Optimal cofactor value (float)
    """

    # Remove zeros which are typically background or empty space in IMC data
    nonzero = channel_data[channel_data > 0]

    if len(nonzero) < 100:  # Need sufficient data for meaningful analysis
        print(f"    Warning: Less than 100 non-zero values ({len(nonzero)}). Returning default cofactor 5.0.")
        return 5.0  # Return default used in cytometry

    def objective_function(cofactor):
        """Function to minimize: KS statistic between transformed data and Gaussian"""
        # Apply arcsinh transformation
        transformed = np.arcsinh(nonzero / cofactor)

        # Skip if no variation in transformed data
        if np.std(transformed) == 0:
            return float('inf')

        # Fit a normal distribution
        mu, sigma = stats.norm.fit(transformed)

        # Avoid issues if sigma is zero or very small
        if sigma <= 1e-6:
             return float('inf')

        # Calculate Kolmogorov-Smirnov statistic (lower is better)
        ks_stat, _ = stats.kstest(transformed, 'norm', args=(mu, sigma))

        return ks_stat

    # Use continuous optimization to find the best cofactor
    try:
        optimization_result = minimize_scalar(
            objective_function,
            bounds=initial_range,
            method='bounded',
            options={'xatol': 0.01}  # Tolerance for convergence
        )

        if optimization_result.success:
            optimal_cofactor = optimization_result.x
            best_score = optimization_result.fun
            # Print some information about the optimization
            # print(f"    Optimization successful: {optimization_result.success}") # Verbose
            # print(f"    Number of function evaluations: {optimization_result.nfev}") # Verbose
            # print(f"    Optimal cofactor: {optimal_cofactor:.2f} (KS statistic: {best_score:.4f})") # Verbose
        else:
             print(f"    Warning: Cofactor optimization did not converge. Status: {optimization_result.status}. Returning default 5.0")
             optimal_cofactor = 5.0

    except ValueError as e:
        print(f"    Warning: Error during cofactor optimization ({e}). Returning default 5.0")
        optimal_cofactor = 5.0


    # Plot the result if requested
    if plot_fits:
        try:
            import matplotlib.pyplot as plt

            # Also plot a few points around the optimum for comparison
            cofactors_to_plot = [
                max(0.5, optimal_cofactor * 0.5),  # Half the optimal
                optimal_cofactor,                  # Optimal
                optimal_cofactor * 2.0             # Double the optimal
            ]

            fig, axes = plt.subplots(len(cofactors_to_plot), 1, figsize=(8, 3*len(cofactors_to_plot)), squeeze=False)
            axes = axes.flatten()

            for i, cofactor in enumerate(cofactors_to_plot):
                ax = axes[i]
                transformed = np.arcsinh(nonzero / cofactor)

                # Calculate KS statistic for this cofactor
                mu, sigma = stats.norm.fit(transformed)
                if sigma <= 1e-6: # Check sigma again before test/plotting
                     ax.text(0.5, 0.5, 'Sigma too small for KS test', ha='center', va='center', transform=ax.transAxes)
                     ks_stat_plot = np.nan
                else:
                     ks_stat_plot, _ = stats.kstest(transformed, 'norm', args=(mu, sigma))

                # Plot histogram of transformed data
                n_plot, bins_plot, _ = ax.hist(transformed, bins=50, density=True, alpha=0.7,
                                   label=f'Data (arcsinh(x/{cofactor:.2f}))')

                # Plot fitted Gaussian if possible
                if sigma > 1e-6:
                    x_plot = np.linspace(min(bins_plot), max(bins_plot), 100)
                    pdf_plot = stats.norm.pdf(x_plot, mu, sigma)
                    ax.plot(x_plot, pdf_plot, 'r-', label=f'Gaussian fit (KS={ks_stat_plot:.4f})')
                else:
                    ax.plot([], [], 'r-', label=f'Gaussian fit (Sigma too small)')


                # Highlight optimal
                if np.isclose(cofactor, optimal_cofactor):
                    ax.set_title(f'OPTIMAL: Cofactor = {cofactor:.2f}, KS stat = {ks_stat_plot:.4f}',
                               fontweight='bold')
                else:
                    ax.set_title(f'Cofactor = {cofactor:.2f}, KS stat = {ks_stat_plot:.4f}')

                ax.legend()
                ax.set_ylabel('Density')

                if i == len(cofactors_to_plot) - 1:
                    ax.set_xlabel('Transformed value')

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("    Warning: Matplotlib not found. Cannot plot cofactor fits.")
        except Exception as plot_err:
            print(f"    Warning: Error generating cofactor plot: {plot_err}")


    return optimal_cofactor

def calculate_asinh_cofactors_for_roi(
    roi_df: pd.DataFrame,
    channels_to_process: List[str],
    default_cofactor: float,
    output_dir: str, # Add output directory for saving
    roi_string: str    # Add ROI string for filename
) -> Dict[str, float]:
    """
    Calculates the optimal Arcsinh cofactor for each specified channel and saves them.

    Args:
        roi_df: DataFrame containing the raw expression data for the ROI.
        channels_to_process: List of channel names to calculate cofactors for.
        default_cofactor: The default cofactor value to use if calculation fails.
        output_dir: The directory specific to this ROI to save the cofactor file.
        roi_string: Identifier for the ROI, used in the filename.

    Returns:
        A dictionary mapping channel names to their calculated optimal cofactor.
    """
    print("\nCalculating optimal Arcsinh cofactors for ROI...")
    start_time_cofactor = time.time()
    asinh_cofactors = {}

    for channel in channels_to_process:
        # print(f"  Processing channel: {channel}") # Verbose
        if channel not in roi_df.columns:
            print(f"    Warning: Channel '{channel}' not found in DataFrame. Using default cofactor.")
            asinh_cofactors[channel] = default_cofactor
            continue

        try:
            # Make sure the function gets the raw data for the channel
            raw_channel_data = roi_df[channel].fillna(0) # Handle potential NaNs

            # Check if data is suitable before passing to optimization
            # (e.g., enough unique positive values)
            if raw_channel_data.empty or raw_channel_data.nunique() <= 1 or (raw_channel_data > 0).sum() < 100: # Increased threshold
                # print(f"    Skipping cofactor calculation for {channel}: Not enough unique/positive values.") # Verbose
                asinh_cofactors[channel] = default_cofactor
            else:
                 # Call the core optimization function
                 calculated_cofactor = optimized_gaussian_fit_cofactor(raw_channel_data, plot_fits=False)
                 asinh_cofactors[channel] = calculated_cofactor
                 # print(f"    Optimal cofactor for {channel}: {calculated_cofactor:.2f}") # Verbose

        except Exception as e:
             print(f"    Error calculating cofactor for {channel}: {e}. Using default: {default_cofactor}")
             # traceback.print_exc() # Optional for more detailed debugging
             asinh_cofactors[channel] = default_cofactor # Fallback cofactor on error

    # Ensure all requested channels have a cofactor entry (should be covered by loop/error handling)
    for channel in channels_to_process:
        if channel not in asinh_cofactors:
            print(f"    Warning: Channel '{channel}' was missed in calculation loop. Assigning default.")
            asinh_cofactors[channel] = default_cofactor

    print(f"--- Cofactor calculation finished in {time.time() - start_time_cofactor:.2f} seconds ---")
    # print("Calculated Optimal Cofactors:", asinh_cofactors) # Optional summary print

    # --- Add saving logic ---
    cofactor_file_path = os.path.join(output_dir, f"asinh_cofactors_{roi_string}.json")
    try:
        with open(cofactor_file_path, 'w') as f:
            json.dump(asinh_cofactors, f, indent=4)
        print(f"   Optimal cofactors saved to: {cofactor_file_path}")
    except Exception as e:
        print(f"   Error saving optimal cofactors to {cofactor_file_path}: {e}")
    # --- End saving logic ---

    return asinh_cofactors

# ==============================================================================
# Data Loading and Validation
# ==============================================================================

def load_and_validate_roi_data(
    file_path: str,
    master_protein_channels: List[str],
    base_output_dir: str,
    metadata_cols: List[str]
) -> Tuple[Optional[str], Optional[str], Optional[pd.DataFrame], Optional[List[str]]]:
    """Loads ROI data, validates channels, and creates output directory."""
    # --- Derive metadata_key and roi_string separately ---
    base_name = os.path.basename(file_path)
    if not base_name.lower().endswith(".txt"):
        print(f"   ERROR: Input file '{base_name}' does not end with .txt. Skipping.")
        return None, None, None, None

    # 1) metadata_key: full base filename without extension, for metadata lookup
    metadata_key = base_name[:-4]

    # 2) roi_string: the short ID beginning 'ROI_…', for naming outputs
    roi_string = extract_roi(base_name)
    print(f"   Derived roi_string (for outputs): {roi_string}")
    print(f"   Derived metadata_key (for metadata lookup): {metadata_key}")

    # --- Create Output Directory ---
    roi_output_dir = os.path.join(base_output_dir, roi_string)
    try:
        os.makedirs(roi_output_dir, exist_ok=True)
    except Exception as e:
        print(f"   Error creating output directory for {roi_string}: {e}")
        return None, None, None, None

    # Load data
    print("Loading data...")
    current_df_raw = pd.read_csv(file_path, sep="\t")
    print(f"Loaded data with shape: {current_df_raw.shape}")

    # Identify channels present in this specific file that are in the master list
    available_master_channels = [
        col for col in master_protein_channels if col in current_df_raw.columns
    ]

    # Further filter to exclude metadata columns (ensure X, Y are not accidentally excluded if not in metadata_cols)
    current_valid_channels = [
         ch for ch in available_master_channels if ch not in metadata_cols
    ]


    # Validate channel count
    if len(current_valid_channels) < 2:
        print(
            f"WARNING: Not enough valid protein channels found in this file "
            f"({len(current_valid_channels)} found). Requires at least 2. Skipping analysis for this ROI."
        )
        return roi_string, roi_output_dir, None, None # Return ROI info but None for data

    print(f"Using {len(current_valid_channels)} channels for analysis.") # Less verbose: {current_valid_channels}")

    # Check for necessary coordinate columns
    if 'X' not in current_df_raw.columns or 'Y' not in current_df_raw.columns:
         print(f"ERROR: Missing required coordinate columns 'X' or 'Y' in {file_path}. Skipping.")
         return roi_string, roi_output_dir, None, None

    # Check for NaNs in coordinates or channels (critical for downstream)
    critical_cols = ['X', 'Y'] + current_valid_channels
    if current_df_raw[critical_cols].isnull().values.any():
         print("ERROR: NaN values found in coordinate or channel data. Cannot proceed. Please clean data first. Skipping.")
         # Example fill (use with caution, better to clean upstream):
         # current_df_raw.fillna(0, inplace=True)
         return roi_string, roi_output_dir, None, None


    return roi_string, roi_output_dir, current_df_raw, current_valid_channels


# ==============================================================================
# Data Transformation and Scaling
# ==============================================================================

def apply_per_channel_arcsinh_and_scale(
    data_df: pd.DataFrame,
    channels: List[str],
    cofactors_map: Dict[str, float],
    default_cofactor: float # Make default mandatory to pass from config
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Applies per-channel Arcsinh transformation using provided cofactors and then scales
    each channel to the [0, 1] range using MinMaxScaler. Ensures output is float32.

    Args:
        data_df: DataFrame containing raw data.
        channels: List of channel columns to transform and scale.
        cofactors_map: Dictionary mapping channel names to their specific cofactor.
        default_cofactor: Default cofactor to use if a channel is not in cofactors_map.

    Returns:
        A tuple containing:
        - scaled_df: DataFrame with transformed and scaled data (float32).
        - used_cofactors: Dictionary mapping channels to the cofactor actually used.
    """
    print("\n--- Applying Per-Channel Arcsinh (using optimal cofactors) and Scaling ---")
    start_time = time.time()
    transformed_data = {} # Use a dictionary to store transformed series
    used_cofactors = {}

    # 1. Apply Arcsinh Transformation
    print("   Applying arcsinh transformation with specific cofactors...")
    for channel in channels:
        if channel not in data_df.columns:
            print(f"   Warning: Channel '{channel}' not found in data_df. Skipping transformation.")
            continue
        
        cofactor = cofactors_map.get(channel, default_cofactor)
        used_cofactors[channel] = cofactor
        
        # Apply transformation - Ensure we handle potential non-numeric data gracefully
        try:
            # Convert to numeric, coercing errors to NaN
            numeric_data = pd.to_numeric(data_df[channel], errors='coerce')
            # Fill NaNs that resulted from coercion or were already present
            numeric_data_filled = numeric_data.fillna(0)
            # Apply arcsinh
            transformed_series = np.arcsinh(numeric_data_filled / cofactor)
            # Store as float32 immediately
            transformed_data[channel] = transformed_series.astype(np.float32) 
        except Exception as e:
            print(f"   Error transforming channel '{channel}': {e}. Skipping.")
            traceback.print_exc()
            # Optionally, fill with zeros or skip adding to dict
            # transformed_data[channel] = pd.Series(np.zeros(len(data_df)), index=data_df.index, dtype=np.float32)

    # Check if any channels were successfully transformed
    if not transformed_data:
        print("   ERROR: No channels were successfully transformed.")
        # Return an empty DataFrame and the used cofactors map
        return pd.DataFrame(index=data_df.index), used_cofactors
        
    # Create DataFrame from dictionary
    transformed_df = pd.DataFrame(transformed_data, index=data_df.index)

    # 2. Apply Scaling (MinMaxScaler to [0, 1])
    print("   Applying MinMaxScaler to transformed data...")
    scaler = MinMaxScaler() # Scales each feature (channel) to [0, 1]
    try:
        # Fit and transform
        scaled_values = scaler.fit_transform(transformed_df)
        # Create new DataFrame with scaled values, ensuring float32
        scaled_df = pd.DataFrame(scaled_values, index=transformed_df.index, columns=transformed_df.columns).astype(np.float32)
    except Exception as e:
        print(f"   ERROR during scaling: {e}. Returning unscaled transformed data.")
        traceback.print_exc()
        # Return the transformed (but unscaled) data if scaling fails
        # Ensure it's float32
        scaled_df = transformed_df.astype(np.float32) 

    print(f"--- Transformation and scaling finished in {time.time() - start_time:.2f} seconds ---")
    return scaled_df, used_cofactors 

# ==============================================================================
# Community Analysis Utilities
# ==============================================================================

def calculate_and_save_community_linkage(
    scaled_community_profiles: pd.DataFrame,
    ordered_channels: List[str],
    output_dir: str,
    file_prefix: str, # e.g., f"community_linkage_matrix_{roi_string}_res_{resolution}"
    config: Dict[str, Any]
) -> Optional[np.ndarray]:
    """
    Performs hierarchical clustering on community profiles and saves the linkage matrix.

    Args:
        scaled_community_profiles: DataFrame of scaled average community expression.
        ordered_channels: List of channels to use for clustering.
        output_dir: Directory to save the output .npy file.
        file_prefix: Base name for the output file (without extension).
        config: Dictionary containing analysis configuration, expected to have
                analysis.clustering.community_metric and .community_linkage_method.

    Returns:
        The calculated linkage matrix as a numpy array, or None if clustering failed.
    """
    linkage_matrix = None
    
    # --- Get Clustering Parameters from Config ---
    clustering_config = config.get('analysis', {}).get('clustering', {})
    metric = clustering_config.get('community_metric', 'correlation') # Default metric
    method = clustering_config.get('community_linkage_method', 'ward') # Default linkage method
    print(f"   Clustering communities using metric='{metric}', method='{method}'...")

    # --- Perform Clustering ---
    if len(scaled_community_profiles.index) < 2 or len(ordered_channels) < 2:
        print(f"   Skipping community linkage calculation: Not enough communities ({len(scaled_community_profiles.index)}) or channels ({len(ordered_channels)}).")
        return None

    try:
        # Ensure profiles use the determined ordered_channels and are clean
        profiles_for_clustering = scaled_community_profiles[ordered_channels].copy()
        
        # Handle potential NaNs/Infs before clustering
        if np.isinf(profiles_for_clustering.values).any():
            print("     WARNING: Infinite values found in community profiles. Replacing with NaN and filling with 0.")
            profiles_for_clustering.replace([np.inf, -np.inf], np.nan, inplace=True)
        profiles_for_clustering.fillna(0, inplace=True) # Fill NaNs (original or from Inf replacement)

        # Check for constant data which can cause errors in some distance metrics
        if profiles_for_clustering.apply(lambda x: x.nunique(), axis=1).min() <= 1:
             print("     WARNING: At least one community has constant expression across channels. Clustering might be unreliable.")
        if profiles_for_clustering.apply(lambda x: x.nunique(), axis=0).min() <= 1:
             print("     WARNING: At least one channel has constant expression across communities. Clustering might be unreliable.")


        # Calculate distance matrix (ensure float64 for pdist)
        community_dist = sch.distance.pdist(
            profiles_for_clustering.astype(np.float64).values, 
            metric=metric
        )
        
        # Calculate linkage matrix
        linkage_matrix = sch.linkage(community_dist, method=method)

        # --- Save the Linkage Matrix ---
        linkage_save_path = os.path.join(output_dir, f"{file_prefix}.npy")
        np.save(linkage_save_path, linkage_matrix)
        print(f"   Community linkage matrix saved to: {os.path.basename(linkage_save_path)}")

    except ValueError as ve:
        print(f"     ERROR during community linkage calculation (ValueError): {ve}.")
        print(f"     Check profiles, metric ('{metric}'), and method ('{method}'). Skipping linkage saving.")
        linkage_matrix = None # Ensure it's None if failed
    except Exception as link_e:
        print(f"     ERROR calculating or saving community linkage matrix: {link_e}")
        traceback.print_exc()
        linkage_matrix = None # Ensure it's None if failed
        
    return linkage_matrix 