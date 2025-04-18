import os
import re
import time
import json
import traceback
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# Helper Functions
# ==============================================================================

def extract_roi(filename: str) -> str:
    """Extracts ROI identifier from filename using regex.

    Args:
        filename: The full path or basename of the file.

    Returns:
        The extracted ROI string (e.g., "ROI_Sam1_03_4") or a fallback name.
    """
    # Look for pattern like ROI_xxx_##_##
    match = re.search(r'(ROI_\w+_\d+_\d+)', filename)
    if match:
        return match.group(1)
    # Fallback: use filename without extension if regex fails
    print(f"Warning: Could not extract standard ROI format from {filename}. Using filename base as ROI.")
    return os.path.splitext(os.path.basename(filename))[0]

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

def calculate_optimal_cofactors_for_roi(
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
    optimal_cofactors = {}

    for channel in channels_to_process:
        # print(f"  Processing channel: {channel}") # Verbose
        if channel not in roi_df.columns:
            print(f"    Warning: Channel '{channel}' not found in DataFrame. Using default cofactor.")
            optimal_cofactors[channel] = default_cofactor
            continue

        try:
            # Make sure the function gets the raw data for the channel
            raw_channel_data = roi_df[channel].fillna(0) # Handle potential NaNs

            # Check if data is suitable before passing to optimization
            # (e.g., enough unique positive values)
            if raw_channel_data.empty or raw_channel_data.nunique() <= 1 or (raw_channel_data > 0).sum() < 100: # Increased threshold
                # print(f"    Skipping cofactor calculation for {channel}: Not enough unique/positive values.") # Verbose
                optimal_cofactors[channel] = default_cofactor
            else:
                 # Call the core optimization function
                 calculated_cofactor = optimized_gaussian_fit_cofactor(raw_channel_data, plot_fits=False)
                 optimal_cofactors[channel] = calculated_cofactor
                 # print(f"    Optimal cofactor for {channel}: {calculated_cofactor:.2f}") # Verbose

        except Exception as e:
             print(f"    Error calculating cofactor for {channel}: {e}. Using default: {default_cofactor}")
             # traceback.print_exc() # Optional for more detailed debugging
             optimal_cofactors[channel] = default_cofactor # Fallback cofactor on error

    # Ensure all requested channels have a cofactor entry (should be covered by loop/error handling)
    for channel in channels_to_process:
        if channel not in optimal_cofactors:
            print(f"    Warning: Channel '{channel}' was missed in calculation loop. Assigning default.")
            optimal_cofactors[channel] = default_cofactor

    print(f"--- Cofactor calculation finished in {time.time() - start_time_cofactor:.2f} seconds ---")
    # print("Calculated Optimal Cofactors:", optimal_cofactors) # Optional summary print

    # --- Add saving logic ---
    cofactor_file_path = os.path.join(output_dir, f"optimal_cofactors_{roi_string}.json")
    try:
        with open(cofactor_file_path, 'w') as f:
            json.dump(optimal_cofactors, f, indent=4)
        print(f"   Optimal cofactors saved to: {cofactor_file_path}")
    except Exception as e:
        print(f"   Error saving optimal cofactors to {cofactor_file_path}: {e}")
    # --- End saving logic ---

    return optimal_cofactors

# ==============================================================================
# Data Loading and Validation
# ==============================================================================

def load_and_validate_roi_data(
    file_path: str,
    master_protein_channels: List[str],
    base_output_dir: str,
    metadata_cols: List[str] # Added metadata_cols as parameter
) -> Tuple[Optional[str], Optional[str], Optional[pd.DataFrame], Optional[List[str]]]:
    """
    Loads data for a specific ROI, validates channels, and sets up output directory.

    Args:
        file_path: Path to the IMC .txt file.
        master_protein_channels: List of all protein channels expected across ROIs.
        base_output_dir: The main directory where ROI-specific subdirectories will be created.
        metadata_cols: List of columns to exclude from being considered protein channels.


    Returns:
        A tuple containing:
        - roi_string: Extracted ROI identifier (or None if extraction fails).
        - roi_output_dir: Path to the dedicated output directory for this ROI (or None).
        - current_df_raw: Loaded pandas DataFrame (or None if loading/validation fails).
        - current_valid_channels: List of protein channels found in this file (or None).
        Returns (None, None, None, None) upon failure to load or validate.
    """
    print(f"--- Loading and Validating: {os.path.basename(file_path)} ---")
    roi_string = None
    roi_output_dir = None
    try:
        # Extract ROI information using the helper function
        roi_string = extract_roi(file_path)
        if not roi_string:
            # Error message already printed by extract_roi fallback
            return None, None, None, None

        print(f"ROI Identifier: {roi_string}")

        # Create ROI-specific output directory
        roi_output_dir = os.path.join(base_output_dir, roi_string)
        os.makedirs(roi_output_dir, exist_ok=True)
        # print(f"Output directory for this ROI: {roi_output_dir}") # Verbose

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

    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}.")
        return roi_string, roi_output_dir, None, None # Return what we have if possible
    except pd.errors.EmptyDataError:
        print(f"ERROR: File {file_path} is empty or invalid.")
        return roi_string, roi_output_dir, None, None # Return what we have if possible
    except Exception as e:
        print(f"An unexpected error occurred during data loading/validation for {file_path}: {str(e)}")
        traceback.print_exc()
        return roi_string, roi_output_dir, None, None # Return what we have if possible


# ==============================================================================
# Data Scaling
# ==============================================================================

def apply_per_channel_arcsinh_and_scale(
    data_df: pd.DataFrame,
    channels: List[str],
    cofactors_map: Dict[str, float],
    default_cofactor: float # Make default mandatory to pass from config
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Applies per-channel Arcsinh transformation using provided cofactors,
    followed by MinMaxScaler across all specified channels.

    Args:
        data_df: DataFrame containing the raw expression data (should have channels cols).
        channels: List of channel column names to transform and scale.
        cofactors_map: Dictionary mapping channel names to their optimal cofactors.
        default_cofactor: Cofactor value to use if a channel is missing from cofactors_map.

    Returns:
        A tuple containing:
        - scaled_df: DataFrame with the transformed and scaled data for the specified channels.
                     The index will match the input data_df. Returns empty DataFrame on error.
        - used_cofactors: Dictionary mapping channel name to the cofactor value actually used.

    Raises:
        ValueError: If input DataFrame is missing required channel columns.
    """
    print(f"\n--- Applying Per-Channel Arcsinh (using optimal cofactors) and Scaling ---")
    start_time = time.time()

    # Check if all requested channels are in the dataframe
    missing_channels = [ch for ch in channels if ch not in data_df.columns]
    if missing_channels:
        raise ValueError(f"Input DataFrame is missing required channel columns: {missing_channels}")

    # Select only the channels to be processed, operate on a copy
    transform_data = data_df[channels].copy()

    # Handle potential NaNs/Infs before transformation (should have been caught by validation, but belt-and-suspenders)
    if transform_data.isnull().values.any() or np.isinf(transform_data.values).any():
        print("   Warning: NaN/Inf values found in raw data before scaling. Replacing with 0.")
        transform_data = transform_data.fillna(0).replace([np.inf, -np.inf], 0)

    transformed_cols = {}
    used_cofactors = {}

    # Apply arcsinh transformation channel by channel
    print("   Applying arcsinh transformation with specific cofactors...")
    for channel in channels:
        cofactor = cofactors_map.get(channel) # Get specific cofactor
        if cofactor is None:
            print(f"    Warning: Cofactor not found for channel '{channel}'. Using default {default_cofactor}.")
            cofactor = default_cofactor
        elif cofactor <= 0:
             print(f"    Warning: Non-positive cofactor {cofactor} provided for channel '{channel}'. Using default {default_cofactor} instead.")
             cofactor = default_cofactor

        used_cofactors[channel] = cofactor # Store the cofactor actually used

        # Apply transformation - ensure data is non-negative
        channel_data = transform_data[channel].values.astype(float) # Ensure float
        channel_data[channel_data < 0] = 0 # Ensure non-negative before division
        transformed_cols[channel] = np.arcsinh(channel_data / cofactor)
        # print(f"    Channel '{channel}' transformed using cofactor: {cofactor:.2f}") # Verbose

    # Create a DataFrame from the transformed columns
    transformed_df = pd.DataFrame(transformed_cols, index=transform_data.index)

    # Apply MinMaxScaler across all transformed channels
    print("   Applying MinMaxScaler to transformed data...")
    scaler = MinMaxScaler()
    # Handle case where transformed_df might be empty or all NaNs after transformation attempt
    if transformed_df.empty or transformed_df.isnull().values.all():
        print("   Warning: Transformed data is empty or all NaN. Cannot apply MinMaxScaler. Returning empty scaled DataFrame.")
        return pd.DataFrame(columns=channels, index=data_df.index), used_cofactors

    try:
        scaled_data_np = scaler.fit_transform(transformed_df.values) # Fit and transform
        # Create the final scaled DataFrame
        scaled_df = pd.DataFrame(scaled_data_np, columns=channels, index=transformed_df.index)
    except ValueError as e:
        print(f"   Error during MinMaxScaler: {e}. Returning empty scaled DataFrame.")
        traceback.print_exc()
        return pd.DataFrame(columns=channels, index=data_df.index), used_cofactors


    print(f"--- Transformation and scaling finished in {time.time() - start_time:.2f} seconds ---")

    return scaled_df, used_cofactors 