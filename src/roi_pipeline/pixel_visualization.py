import os
import math
import time
import traceback
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Need to handle potential missing dependency for GMM/Otsu in deprecated functions
try:
    from sklearn.mixture import GaussianMixture
    sklearn_available = True
except ImportError:
    sklearn_available = False
try:
    from scipy.stats import norm
    scipy_stats_available = True
except ImportError:
    scipy_stats_available = False
try:
    from scipy.optimize import brentq
    scipy_optimize_available = True
except ImportError:
    scipy_optimize_available = False
try:
    from skimage.filters import threshold_otsu
    skimage_available = True
except ImportError:
    print("Warning: scikit-image not found. Otsu thresholding in generate_histograms will not be available.")
    skimage_available = False

# Optional dependency import
try:
    import umap
    umap_available = True
except ImportError:
    umap_available = False
    print("Warning: umap-learn package not installed. UMAP visualization will be skipped.")

# ==============================================================================
# Plotting Helper Functions
# ==============================================================================

def add_coexpression_colorbar(fig, bbox=None):
    """
    Adds a custom 2D colorbar for co-expression visualization (Blue/Yellow/Magenta).

    Args:
        fig: The matplotlib Figure object.
        bbox: Optional bounding box [left, bottom, width, height] for the colorbar axes.
              If None, uses default placement.
    """
    n = 256 # Resolution of the colorbar

    # Create the axes for the colorbar
    if bbox is None:
        # Default position: Place it slightly to the right and adjust size
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.2]) # Adjust these values as needed
    else:
        cax = fig.add_axes(bbox)

    # Create grid representing the two channel dimensions
    ch1_vals = np.linspace(0, 1, n)
    ch2_vals = np.linspace(0, 1, n)
    ch1_grid, ch2_grid = np.meshgrid(ch1_vals, ch2_vals)

    # Initialize RGB color array (n x n x 3)
    rgb_colors = np.zeros((n, n, 3))

    # Scale the grid values (representing normalized expression)
    scaled_ch1 = ch1_grid
    scaled_ch2 = ch2_grid

    # --- Define masks for the color logic ---
    # We use a small threshold to define "expression"
    threshold = 0.01
    mask_ch1 = scaled_ch1 > threshold
    mask_ch2 = scaled_ch2 > threshold

    # Masks for the different co-expression regions
    coexp_mask = mask_ch1 & mask_ch2
    ch1_only_mask = mask_ch1 & ~mask_ch2
    ch2_only_mask = mask_ch2 & ~mask_ch1

    # --- Assign colors based on regions using scaled intensities ---
    rgb_colors[ch1_only_mask, 2] = scaled_ch1[ch1_only_mask]          # Blue
    rgb_colors[ch2_only_mask, 0] = scaled_ch2[ch2_only_mask]          # Red (Yellow)
    rgb_colors[ch2_only_mask, 1] = scaled_ch2[ch2_only_mask]          # Green (Yellow)
    rgb_colors[coexp_mask, 0] = scaled_ch2[coexp_mask]                 # Red from Ch2 (Magenta)
    rgb_colors[coexp_mask, 1] = scaled_ch2[coexp_mask] * (1 - scaled_ch1[coexp_mask]) # Green depends on both
    rgb_colors[coexp_mask, 2] = scaled_ch1[coexp_mask]                 # Blue from Ch1 (Magenta)

    # Clip final colors to [0, 1] range (safety measure)
    rgb_colors = np.clip(rgb_colors, 0, 1)

    # Display the colorbar using imshow
    im = cax.imshow(rgb_colors, origin='lower', aspect='auto', # Changed aspect to auto for flexibility
                    interpolation='nearest')

    # Add axis labels and ticks
    cax.set_xticks([0, n-1])
    cax.set_xticklabels(['0', '1'])
    cax.set_yticks([0, n-1])
    cax.set_yticklabels(['0', '1'])
    cax.set_xlabel('Channel 2', fontsize=9)
    cax.set_ylabel('Channel 1', fontsize=9)
    cax.xaxis.set_label_position('top')
    cax.tick_params(axis='both', labelsize=8)

    # Add color indicators at key points using data coordinates
    point_props = dict(marker='o', markersize=6, markeredgecolor='white',
                       markeredgewidth=1.0, zorder=5)
    text_props = dict(fontsize=8, color='white', ha='center', va='center',
                      bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

    # Get colors at corners for markers (row, col)
    black_color = rgb_colors[0, 0, :3]
    blue_color = rgb_colors[n-1, 0, :3]
    yellow_color = rgb_colors[0, n-1, :3]
    magenta_color = rgb_colors[n-1, n-1, :3]

    # Plot markers (x, y) = (col, row) and text inside
    cax.plot(0, 0, color=black_color, **point_props) # Black (Low/Low)
    # cax.text(0, 0, 'Low', **text_props)
    cax.plot(0, n-1, color=blue_color, **point_props) # Blue (High Ch1 / Low Ch2)
    cax.text(0, n-1, 'Ch1', **text_props)
    cax.plot(n-1, 0, color=yellow_color, **point_props) # Yellow (Low Ch1 / High Ch2)
    cax.text(n-1, 0, 'Ch2', **text_props)
    cax.plot(n-1, n-1, color=magenta_color, **point_props) # Magenta (High/High)
    cax.text(n-1, n-1, 'Both', **text_props)

# ==============================================================================
# Deprecated / Original Exploratory Plots (Possibly Unused)
# ==============================================================================

# def plot_channel_pair(ax, sampled_df, channel1, channel2, cofactor=None, optimal_cofactors=None):
#     \"\"\"
#     [DEPRECATED/EXPLORATORY] Plot a single channel pair on the given axis with arcsinh transformation.
#     Likely used in initial notebook exploration, may not be used in batch workflow.
#
#     Parameters:
#     - ax: matplotlib axis to plot on
#     - sampled_df: dataframe with channel data
#     - channel1: first channel name
#     - channel2: second channel name
#     - cofactor: Optional fixed arcsinh cofactor
#     - optimal_cofactors: Dictionary of optimal cofactors by channel
#
#     Returns:
#     - Used cofactor value
#     \"\"\"
#     # Extract channel names for display - only the protein name before the parenthesis
#     ch1_label = channel1.split('(')[0]
#     ch2_label = channel2.split('(')[0]
#
#     # Get X and Y ranges for setting consistent axes
#     x_min, x_max = sampled_df['X'].min(), sampled_df['X'].max()
#     y_min, y_max = sampled_df['Y'].min(), sampled_df['Y'].max()
#
#     # Create RGB colors for the points - initialized to black
#     rgb_colors = np.zeros((len(sampled_df), 4))
#     rgb_colors[:, 3] = 1.0  # Full opacity for all points
#
#     # Determine cofactor (same logic as before)
#     if cofactor is None:
#         if optimal_cofactors and channel1 in optimal_cofactors and channel2 in optimal_cofactors:
#             # Use average of the two channels' optimal cofactors if available
#             cofactor = (optimal_cofactors[channel1] + optimal_cofactors[channel2]) / 2
#         else:
#             cofactor = 5.0 # Default fallback
#     cofactor_text = f"arcsinh(x/{cofactor:.1f})"
#
#     # Process channel 1 (blue)
#     if sampled_df[channel1].max() > 0:
#         transformed_ch1 = np.arcsinh(sampled_df[channel1] / cofactor)
#         positive_transformed_ch1 = transformed_ch1[transformed_ch1 > 0]
#         p99_ch1 = np.percentile(positive_transformed_ch1, 99) if len(positive_transformed_ch1) > 0 else 1.0
#         # Ensure p99 is positive to avoid division by zero
#         p99_ch1 = max(p99_ch1, 1e-6)
#         clipped_ch1 = np.clip(transformed_ch1, 0, p99_ch1)
#         normalized_ch1 = clipped_ch1 / p99_ch1
#         # Define a threshold based on transformed values (e.g., arcsinh of a small raw value)
#         threshold1 = max(0.01, np.arcsinh(0.001 / cofactor) / p99_ch1)
#         mask_ch1 = normalized_ch1 > threshold1
#     else:
#         normalized_ch1 = np.zeros(len(sampled_df))
#         mask_ch1 = np.zeros(len(sampled_df), dtype=bool)
#
#     # Process channel 2 (yellow)
#     if sampled_df[channel2].max() > 0:
#         transformed_ch2 = np.arcsinh(sampled_df[channel2] / cofactor)
#         positive_transformed_ch2 = transformed_ch2[transformed_ch2 > 0]
#         p99_ch2 = np.percentile(positive_transformed_ch2, 99) if len(positive_transformed_ch2) > 0 else 1.0
#         p99_ch2 = max(p99_ch2, 1e-6)
#         clipped_ch2 = np.clip(transformed_ch2, 0, p99_ch2)
#         normalized_ch2 = clipped_ch2 / p99_ch2
#         threshold2 = max(0.01, np.arcsinh(0.001 / cofactor) / p99_ch2)
#         mask_ch2 = normalized_ch2 > threshold2
#     else:
#         normalized_ch2 = np.zeros(len(sampled_df))
#         mask_ch2 = np.zeros(len(sampled_df), dtype=bool)
#
#     # Define masks for plotting based on thresholds (same as original logic)
#     coexp_mask = mask_ch1 & mask_ch2
#     ch1_only_mask = mask_ch1 & ~mask_ch2
#     ch2_only_mask = mask_ch2 & ~mask_ch1
#
#     # Assign colors based on regions using normalized intensities (matching colorbar logic)
#     rgb_colors[ch1_only_mask, 2] = normalized_ch1[ch1_only_mask]  # Blue component for Ch1 only
#     rgb_colors[ch2_only_mask, 0] = normalized_ch2[ch2_only_mask]  # Red component for Ch2 only (Yellow)
#     rgb_colors[ch2_only_mask, 1] = normalized_ch2[ch2_only_mask]  # Green component for Ch2 only (Yellow)
#     # Coexpression (Magenta blend)
#     rgb_colors[coexp_mask, 0] = normalized_ch2[coexp_mask]                     # Red from Ch2
#     rgb_colors[coexp_mask, 1] = normalized_ch2[coexp_mask] * 0.3               # Reduced Green from Ch2
#     rgb_colors[coexp_mask, 2] = normalized_ch1[coexp_mask]                     # Blue from Ch1
#
#     # Clip final colors (safety measure)
#     rgb_colors = np.clip(rgb_colors, 0, 1)
#
#     # Plot using the RGB colors
#     scatter = ax.scatter(
#         sampled_df['X'],
#         sampled_df['Y'],
#         c=rgb_colors,
#         s=2.0, # Original marker size
#         marker='o' # Original marker
#     )
#
#     # Set title with the color scheme and cofactor info
#     ax.set_title(f"{ch1_label} (blue) × {ch2_label} (yellow)\\n{cofactor_text}", fontsize=8, pad=10)
#     ax.set_xlabel('X (µm)', fontsize=8)
#     ax.set_ylabel('Y (µm)', fontsize=8)
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_aspect('equal')
#     ax.grid(False)
#     ax.tick_params(axis='both', labelsize=7)
#     for spine in ax.spines.values():
#         spine.set_linewidth(0.5)
#
#     return cofactor
#
# def create_coexpression_figure(channel_pairs, sampled_df, start_idx=0, count=36, title=None,
#                              cofactor=None, optimal_cofactors=None):
#     \"\"\"
#     [DEPRECATED/EXPLORATORY] Create a figure with co-expression maps for multiple channel pairs.
#     Likely used in initial notebook exploration, may not be used in batch workflow.
#     \"\"\"
#     end_idx = min(start_idx + count, len(channel_pairs))
#     pairs_to_show = channel_pairs[start_idx:end_idx]
#
#     if not pairs_to_show:
#         print("No channel pairs to display")
#         return None
#
#     n_cols = 3
#     n_rows = math.ceil(len(pairs_to_show) / n_cols)
#     fig_width = 20
#     fig_height = n_rows * (fig_width / n_cols) * 1.2
#
#     fig, axs_flat = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
#     axs_flat = axs_flat.flatten()
#
#     used_cofactors = []
#
#     for i, (channel1, channel2) in enumerate(pairs_to_show):
#         ax = axs_flat[i]
#         used_cofactor = plot_channel_pair(ax, sampled_df, channel1, channel2,
#                                        cofactor=cofactor,
#                                        optimal_cofactors=optimal_cofactors)
#         used_cofactors.append(used_cofactor)
#
#     # Hide unused axes
#     for j in range(i + 1, len(axs_flat)):
#         fig.delaxes(axs_flat[j])
#
#     # Set overall title
#     if title is None:
#         if start_idx == 0:
#             title = f'Spatial Co-expression Maps (Showing {len(pairs_to_show)} of {len(channel_pairs)} protein pairs)'
#         else:
#             title = f'Spatial Co-expression Maps (Pairs {start_idx+1}-{end_idx} of {len(channel_pairs)})'
#     if cofactor is not None:
#         title += f"\\nArcsinh cofactor: {cofactor:.2f}"
#
#     fig.suptitle(title, fontsize=16, y=0.98)
#
#     # Add the custom colorbar to the first subplot's axes for positioning context
#     # This might need adjustment depending on layout needs
#     if n_rows > 0 and n_cols > 0:
#          add_coexpression_colorbar(fig) # Use default positioning relative to figure
#
#     # Update legend position
#     legend_elements = [
#         Patch(facecolor='blue', label='Channel 1 (blue)'),
#         Patch(facecolor='yellow', label='Channel 2 (yellow)'),
#         Patch(facecolor='magenta', label='Coexpression')
#     ]
#
#     fig.legend(handles=legend_elements,
#                loc='lower center',
#                bbox_to_anchor=(0.4, 0.01),
#                ncol=3,
#                frameon=False,
#                fontsize=10)
#
#     if len(used_cofactors) > 0:
#         avg_cofactor = np.mean(used_cofactors)
#         print(f"Arcsinh cofactors - Min: {min(used_cofactors):.2f}, Max: {max(used_cofactors):.2f}, Avg: {avg_cofactor:.2f}")
#
#     # fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout carefully with absolute positioned legends/colorbars
#     plt.show()
#     return end_idx
#
# def generate_histograms(df, channels, optimal_cofactors, roi_string, save_dir, hist_cols, plot_dpi, default_cofactor):
#     \"\"\"[DEPRECATED/EXPLORATORY] Generates and saves histograms with GMM and Otsu gating.
#         Requires scikit-learn and scikit-image.
#     \"\"\"
#     if not sklearn_available or not scipy_stats_available or not scipy_optimize_available:
#         print("Skipping histogram generation: Missing required dependencies (sklearn, scipy.stats, scipy.optimize).")
#         return {}
#
#     print("Generating histograms with GMM & Otsu gating...")
#     n_channels = len(channels)
#     if n_channels == 0:
#         print("Skipping histogram generation: No channels.")
#         return {}
#
#     hist_rows = math.ceil(n_channels / hist_cols)
#     plt.close('all')
#     fig, axs = plt.subplots(hist_rows, hist_cols, figsize=(hist_cols * 4, hist_rows * 3.5), squeeze=False)
#     axs = axs.flatten()
#
#     gating_thresholds = {'GMM': {}, 'Otsu': {}}
#
#     for i, channel in enumerate(channels):
#         ax = axs[i]
#         cofactor = optimal_cofactors.get(channel, default_cofactor) # Use provided default
#         raw_data = df[channel].fillna(0)
#         filtered_data = raw_data[raw_data > 0]
#
#         gmm_gate_value = None
#         otsu_gate_value = None
#
#         if filtered_data.empty:
#             ax.text(0.5, 0.5, 'No positive data', ha='center', va='center', transform=ax.transAxes, fontsize=9)
#             ax.set_title(f'{channel}\\n(Cofactor: {cofactor:.2f})', fontsize=10)
#         elif len(filtered_data.unique()) < 5:
#             ax.text(0.5, 0.5, 'Low data variance', ha='center', va='center', transform=ax.transAxes, fontsize=9)
#             transformed_data = np.arcsinh(filtered_data / cofactor)
#             ax.hist(transformed_data, bins=20, color='skyblue', edgecolor='black')
#             ax.set_title(f'{channel}\\n(Cofactor: {cofactor:.2f})', fontsize=10)
#         else:
#             transformed_data = np.arcsinh(filtered_data / cofactor)
#             min_val, max_val = transformed_data.min(), transformed_data.max()
#             bins = np.linspace(min_val - 0.5, max_val + 0.5, 50) if np.isclose(min_val, max_val) else np.linspace(min_val, max_val, 50)
#             ax.hist(transformed_data, bins=bins, color='skyblue', edgecolor='black', density=True)
#             ax.set_title(f'{channel}\\n(Cofactor: {cofactor:.2f})', fontsize=10)
#
#             # --- GMM Gating Logic ---
#             try:
#                 X = transformed_data.values.reshape(-1, 1)
#                 gmm = GaussianMixture(n_components=2, random_state=0, reg_covar=1e-5).fit(X)
#                 means, covars, weights = gmm.means_.flatten(), gmm.covariances_.flatten(), gmm.weights_.flatten()
#                 idx = np.argsort(means)
#                 mean0, mean1 = means[idx]
#                 std0, std1 = np.sqrt(covars[idx])
#                 weight0, weight1 = weights[idx]
#                 separation_threshold = max(std0, std1)
#
#                 if mean1 > mean0 + separation_threshold and std0 > 1e-6 and std1 > 1e-6:
#                     def pdf_diff(x): return weight0 * norm.pdf(x, mean0, std0) - weight1 * norm.pdf(x, mean1, std1)
#                     try:
#                         epsilon = 1e-6; lower_bound = mean0 + epsilon; upper_bound = mean1 - epsilon
#                         if lower_bound < upper_bound: gmm_gate_value = brentq(pdf_diff, lower_bound, upper_bound)
#                     except ValueError:
#                         # Fallback using posterior probability intersection with 0.5
#                         def post_prob_diff(x): return gmm.predict_proba(np.array([[x]]))[0, idx[1]] - 0.5
#                         try: gmm_gate_value = brentq(post_prob_diff, mean0, mean1 + 3*std1) # Widen search slightly
#                         except ValueError as e2: print(f"    Ch {channel}: GMM posterior prob failed ({e2}).")
#                 else: print(f"    Ch {channel}: GMM components not separated or std dev too small.")
#                 if gmm_gate_value is not None: gating_thresholds['GMM'][channel] = gmm_gate_value
#             except Exception as e: print(f"    Ch {channel}: GMM failed: {e}")
#             # --- End GMM Logic ---
#
#             # --- Otsu's Method Logic ---
#             if skimage_available:
#                 try:
#                     # Otsu needs non-negative integer counts ideally, but works on float too
#                     # Ensure data range is suitable for Otsu, might need scaling/binning first
#                     # For simplicity, applying directly to transformed data here
#                     otsu_gate_value = threshold_otsu(transformed_data.to_numpy())
#                     gating_thresholds['Otsu'][channel] = otsu_gate_value
#                 except Exception as e: print(f"    Ch {channel}: Otsu failed: {e}")
#             # --- End Otsu Logic ---
#
#         # --- Plot Gates ---
#         title_suffix = []
#         if gmm_gate_value is not None:
#             ax.axvline(gmm_gate_value, color='r', linestyle='--', lw=1.5, label=f'GMM ({gmm_gate_value:.2f})')
#             title_suffix.append(f"GMM: {gmm_gate_value:.2f}")
#         if otsu_gate_value is not None:
#             ax.axvline(otsu_gate_value, color='b', linestyle=':', lw=1.5, label=f'Otsu ({otsu_gate_value:.2f})')
#             title_suffix.append(f"Otsu: {otsu_gate_value:.2f}")
#         if title_suffix:
#             ax.set_title(f"{ax.get_title()}\\nGates: {'; '.join(title_suffix)}", fontsize=9)
#
#         ax.set_xlabel('Arcsinh Intensity')
#         ax.set_ylabel('Density')
#         ax.tick_params(axis='both', labelsize=8)
#         if gmm_gate_value is not None or otsu_gate_value is not None:
#             ax.legend(fontsize=7)
#
#     # Clean up layout
#     for i in range(n_channels, len(axs)): fig.delaxes(axs[i])
#     fig.suptitle(f'Arcsinh Histograms (Non-Zero, GMM & Otsu Gates) - {roi_string}', fontsize=16, y=0.99)
#     fig.tight_layout(rect=[0, 0.03, 1, 0.96])
#     fig.subplots_adjust(hspace=0.6) # Adjust vertical spacing
#
#     # Save figure directly to the provided save_dir
#     output_path = os.path.join(save_dir, f"channel_gating_histograms_{roi_string}.png")
#     try:
#         fig.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
#         print(f"Histograms saved to: {output_path}")
#     except Exception as e:
#         print(f"Error saving histograms to {output_path}: {e}")
#     plt.close(fig)
#
#     return gating_thresholds
#
# def generate_coexpression_maps(df, channels, optimal_cofactors, roi_string, save_dir, plot_dpi, default_cofactor):
#     \"\"\"
#     [DEPRECATED/EXPLORATORY] Generates and saves co-expression maps arranged in an upper triangular matrix layout.
#     \"\"\"
#     print("Generating co-expression maps (matrix layout v2)...")
#     if len(channels) < 2:
#         print("Skipping co-expression plots: Less than 2 channels available.")
#         return
#
#     sorted_channels = sorted(channels)
#     n_channels = len(sorted_channels)
#     print(f"Plotting {n_channels} channels in matrix layout.")
#     plt.close('all')
#
#     plot_size_inches = 3.0
#     label_space_inches = 0.8
#     top_matter_inches = 1.2
#     fig_width_inches = label_space_inches + n_channels * plot_size_inches
#     fig_height_inches = top_matter_inches + n_channels * plot_size_inches + label_space_inches
#     max_fig_dim_inches = 60
#     fig_width_inches = min(fig_width_inches, max_fig_dim_inches)
#     fig_height_inches = min(fig_height_inches, max_fig_dim_inches)
#
#     print(f"Target figure size: {fig_width_inches:.1f} x {fig_height_inches:.1f} inches")
#     fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))
#
#     left_margin_frac = label_space_inches / fig_width_inches
#     right_margin_frac = 0.02
#     bottom_margin_frac = label_space_inches / fig_height_inches
#     top_margin_frac = top_matter_inches / fig_height_inches
#     grid_left = left_margin_frac
#     grid_bottom = bottom_margin_frac
#     grid_width = max(0.1, 1.0 - left_margin_frac - right_margin_frac)
#     grid_height = max(0.1, 1.0 - bottom_margin_frac - top_margin_frac)
#     grid_top_edge = grid_bottom + grid_height
#     title_y_pos = 1.0 - (top_margin_frac * 0.1)
#
#     # Colorbar positioning (simplified)
#     cbar_x_center = 0.5
#     cbar_width_frac = 0.12
#     cbar_height_frac = 0.04
#     cbar_y_bottom = 1.0 - top_margin_frac + 0.01 # Place near top
#     cbar_left = cbar_x_center - (cbar_width_frac / 2)
#     calculated_cbar_bbox = [cbar_left, cbar_y_bottom, cbar_width_frac, cbar_height_frac]
#     calculated_cbar_bbox = [max(0,min(1,x)) for x in calculated_cbar_bbox]
#     add_coexpression_colorbar(fig, bbox=calculated_cbar_bbox)
#
#     # Legend positioning (simplified)
#     legend_elements = [Patch(facecolor='blue', label='Ch1(Row)'),
#                        Patch(facecolor='yellow', label='Ch2(Col)'),
#                        Patch(facecolor='magenta', label='Coexpr.')]
#     fig.legend(handles=legend_elements, loc='upper center',
#                bbox_to_anchor=(cbar_x_center, cbar_y_bottom - 0.01), # Below colorbar
#                ncol=3, frameon=False, fontsize=9)
#
#     gs = fig.add_gridspec(n_channels, n_channels,
#                           left=grid_left, bottom=grid_bottom,
#                           right=grid_left + grid_width, top=grid_top_edge,
#                           wspace=0.05, hspace=0.05)
#
#     used_cofactors = []
#     for i in range(n_channels):
#         for j in range(n_channels):
#             ax = fig.add_subplot(gs[i, j])
#             if i >= j:
#                 ax.set_visible(False)
#             else:
#                 channel1 = sorted_channels[i]
#                 channel2 = sorted_channels[j]
#                 # Pass default cofactor explicitly if optimal_cofactors dict might be incomplete
#                 specific_optimal_cofactors = {ch: optimal_cofactors.get(ch, default_cofactor) for ch in [channel1, channel2]}
#                 current_cofactor = plot_channel_pair(ax, df, channel1, channel2,
#                                                    optimal_cofactors=specific_optimal_cofactors)
#                 used_cofactors.append(current_cofactor)
#                 ax.set_title("")
#                 ax.set_xlabel("")
#                 ax.set_ylabel("")
#                 ax.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)
#                 for spine in ax.spines.values():
#                     spine.set_linewidth(0.5)
#                     spine.set_visible(True)
#
#     label_fontsize = 8
#     row_label_x_offset = 0.01
#     col_label_y_offset = 0.01
#     bottoms, tops, lefts, rights = gs.get_grid_positions(fig)
#
#     for i in range(n_channels):
#         row_label = sorted_channels[i].split('(')[0]
#         y_pos = (bottoms[i] + tops[i]) / 2
#         fig.text(grid_left - row_label_x_offset, y_pos, row_label, va='center', ha='right', fontsize=label_fontsize, rotation=0)
#
#     for j in range(n_channels):
#         col_label = sorted_channels[j].split('(')[0]
#         x_pos = (lefts[j] + rights[j]) / 2
#         fig.text(x_pos, grid_top_edge + col_label_y_offset, col_label, va='bottom', ha='center', fontsize=label_fontsize, rotation=60)
#
#     avg_cofactor = np.mean(used_cofactors) if used_cofactors else default_cofactor
#     fig.suptitle(f"Spatial Co-expression Matrix - {roi_string}\\n(Avg. arcsinh cofactor: {avg_cofactor:.2f})",
#                  fontsize=14, y=title_y_pos)
#
#     output_path = os.path.join(save_dir, f"coexpression_matrix_{roi_string}.png")
#     save_dpi = max(plot_dpi, 200)
#     print(f"Saving figure ({fig_width_inches:.1f}x{fig_height_inches:.1f} inches) with DPI: {save_dpi}")
#     try:
#         fig.savefig(output_path, dpi=save_dpi, bbox_inches='tight', facecolor='white')
#         print(f"Co-expression matrix saved to: {output_path}")
#     except Exception as e:
#          print(f"Error saving coexpression matrix to {output_path}: {e}")
#     plt.close(fig)
# \"\"\"

# ==============================================================================
# Co-expression Matrix Helpers
# ==============================================================================

# Modified to use already scaled data
def _plot_single_pixel_coexpression(ax, pixel_coords: pd.DataFrame, scaled_pixel_expression: pd.DataFrame, channel1: str, channel2: str):
    """Plots pixel co-expression scatter using ALREADY SCALED data."""
    ch1_label = channel1.split('(')[0]
    ch2_label = channel2.split('(')[0]

    # Check if required columns exist
    if not all(c in scaled_pixel_expression.columns for c in [channel1, channel2]) \
            or not all(c in pixel_coords.columns for c in ['X', 'Y']):
        ax.text(0.5, 0.5, 'Missing Scaled Data', ha='center', va='center', transform=ax.transAxes)
        return

    # Combine coords and the two channels for plotting
    plot_df = pixel_coords.join(scaled_pixel_expression[[channel1, channel2]])
    x_min, x_max = plot_df['X'].min(), plot_df['X'].max()
    y_min, y_max = plot_df['Y'].min(), plot_df['Y'].max()

    rgb_colors = np.zeros((len(plot_df), 4))
    rgb_colors[:, 3] = 1.0  # Full opacity

    # --- Normalize Scaled Channel 1 (Blue) for visualization ---
    norm_ch1 = np.zeros(len(plot_df))
    mask_ch1 = np.zeros(len(plot_df), dtype=bool)
    vals_ch1 = plot_df[channel1]
    vmin1 = np.percentile(vals_ch1, 1)
    vmax1 = np.percentile(vals_ch1, 99)
    if vmax1 > vmin1:
        norm_ch1 = np.clip((vals_ch1 - vmin1) / (vmax1 - vmin1), 0, 1)
        threshold1 = 0.01
        mask_ch1 = norm_ch1 > threshold1
    elif vmax1 > 0:
        norm_ch1 = np.clip(vals_ch1 / vmax1, 0, 1)
        mask_ch1 = norm_ch1 > 0.01

    # --- Normalize Scaled Channel 2 (Yellow) for visualization ---
    norm_ch2 = np.zeros(len(plot_df))
    mask_ch2 = np.zeros(len(plot_df), dtype=bool)
    vals_ch2 = plot_df[channel2]
    vmin2 = np.percentile(vals_ch2, 1)
    vmax2 = np.percentile(vals_ch2, 99)
    if vmax2 > vmin2:
        norm_ch2 = np.clip((vals_ch2 - vmin2) / (vmax2 - vmin2), 0, 1)
        threshold2 = 0.01
        mask_ch2 = norm_ch2 > threshold2
    elif vmax2 > 0:
        norm_ch2 = np.clip(vals_ch2 / vmax2, 0, 1)
        mask_ch2 = norm_ch2 > 0.01

    # Define masks for plotting
    coexp_mask = mask_ch1 & mask_ch2
    ch1_only_mask = mask_ch1 & ~mask_ch2
    ch2_only_mask = mask_ch2 & ~mask_ch1

    # Assign colors (Blue/Yellow/Magenta)
    rgb_colors[ch1_only_mask, 2] = norm_ch1[ch1_only_mask]
    rgb_colors[ch2_only_mask, 0] = norm_ch2[ch2_only_mask]
    rgb_colors[ch2_only_mask, 1] = norm_ch2[ch2_only_mask]
    rgb_colors[coexp_mask, 0] = norm_ch2[coexp_mask]
    rgb_colors[coexp_mask, 2] = norm_ch1[coexp_mask]
    rgb_colors = np.clip(rgb_colors, 0, 1)

    ax.scatter(plot_df['X'], plot_df['Y'], c=rgb_colors, s=1.0, marker='.', rasterized=True)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal'); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(''); ax.set_ylabel('')
    for spine in ax.spines.values(): spine.set_linewidth(0.5)

def _plot_mapped_avg_community_coexpression(ax, pixel_results_df_with_avg, channel1, channel2):
    """Plots spatial co-expression using mapped average community scaled values."""
    avg_col1 = f'{channel1}_asinh_scaled_avg'
    avg_col2 = f'{channel2}_asinh_scaled_avg'

    if avg_col1 not in pixel_results_df_with_avg.columns or avg_col2 not in pixel_results_df_with_avg.columns:
        ax.text(0.5, 0.5, 'Avg Comm Data Missing', ha='center', va='center', transform=ax.transAxes)
        return

    plot_df = pixel_results_df_with_avg[['X', 'Y', avg_col1, avg_col2]].copy().dropna()
    if plot_df.empty:
        ax.text(0.5, 0.5, 'No Overlapping Data', ha='center', va='center', transform=ax.transAxes)
        return

    vals_ch1 = plot_df[avg_col1].values
    vals_ch2 = plot_df[avg_col2].values

    # Normalize average values to 0-1 for coloring (min-max scaling)
    # Handle cases where min=max or max=0
    norm_ch1 = np.zeros_like(vals_ch1)
    mask_ch1 = np.zeros_like(vals_ch1, dtype=bool)
    vmin1, vmax1 = np.min(vals_ch1), np.max(vals_ch1)
    if vmax1 > vmin1:
        norm_ch1 = np.clip((vals_ch1 - vmin1) / (vmax1 - vmin1), 0, 1)
        threshold1 = 0.01
        mask_ch1 = norm_ch1 > threshold1
    elif vmax1 > 0: # If all values are the same but > 0
        norm_ch1 = np.clip(vals_ch1 / vmax1, 0, 1)
        mask_ch1 = norm_ch1 > 0.01 # Use threshold

    norm_ch2 = np.zeros_like(vals_ch2)
    mask_ch2 = np.zeros_like(vals_ch2, dtype=bool)
    vmin2, vmax2 = np.min(vals_ch2), np.max(vals_ch2)
    if vmax2 > vmin2:
        norm_ch2 = np.clip((vals_ch2 - vmin2) / (vmax2 - vmin2), 0, 1)
        threshold2 = 0.01
        mask_ch2 = norm_ch2 > threshold2
    elif vmax2 > 0:
        norm_ch2 = np.clip(vals_ch2 / vmax2, 0, 1)
        mask_ch2 = norm_ch2 > 0.01

    # Initialize RGB colors (default black)
    rgb_colors = np.zeros((len(plot_df), 3))

    # Define masks for plotting
    coexp_mask = mask_ch1 & mask_ch2
    ch1_only_mask = mask_ch1 & ~mask_ch2
    ch2_only_mask = mask_ch2 & ~mask_ch1

    # Assign colors (Blue/Yellow/Magenta)
    rgb_colors[ch1_only_mask, 2] = norm_ch1[ch1_only_mask]
    rgb_colors[ch2_only_mask, 0] = norm_ch2[ch2_only_mask]
    rgb_colors[ch2_only_mask, 1] = norm_ch2[ch2_only_mask]
    rgb_colors[coexp_mask, 0] = norm_ch2[coexp_mask]
    rgb_colors[coexp_mask, 2] = norm_ch1[coexp_mask]
    rgb_colors = np.clip(rgb_colors, 0, 1)

    ax.scatter(plot_df['X'], plot_df['Y'], c=rgb_colors, s=1.0, marker='.', rasterized=True)
    # No need for xlim/ylim/aspect/ticks here

# --- Helper: Plot Scaled Pixel Co-expression --- #

def _plot_scaled_pixel_coexpression(ax, pixel_data_scaled, channel1, channel2):
    """Plots spatial co-expression using scaled pixel values (upper triangle)."""
    scaled_col1 = f'{channel1}_asinh_scaled'
    scaled_col2 = f'{channel2}_asinh_scaled'

    if scaled_col1 not in pixel_data_scaled.columns or scaled_col2 not in pixel_data_scaled.columns:
        ax.text(0.5, 0.5, 'Scaled Data Missing', ha='center', va='center', transform=ax.transAxes)
        return

    plot_df = pixel_data_scaled[['X', 'Y', scaled_col1, scaled_col2]].copy().dropna()
    if plot_df.empty:
        ax.text(0.5, 0.5, 'No Overlapping Data', ha='center', va='center', transform=ax.transAxes)
        return

    norm_ch1 = plot_df[scaled_col1].values # Already 0-1 scaled
    norm_ch2 = plot_df[scaled_col2].values # Already 0-1 scaled

    # Define masks based on a small threshold
    threshold = 0.01
    mask_ch1 = norm_ch1 > threshold
    mask_ch2 = norm_ch2 > threshold

    # Initialize RGB colors (default black)
    rgb_colors = np.zeros((len(plot_df), 3))

    # Define masks for plotting
    coexp_mask = mask_ch1 & mask_ch2
    ch1_only_mask = mask_ch1 & ~mask_ch2
    ch2_only_mask = mask_ch2 & ~mask_ch1

    # Assign colors (Blue/Yellow/Magenta)
    rgb_colors[ch1_only_mask, 2] = norm_ch1[ch1_only_mask]
    rgb_colors[ch2_only_mask, 0] = norm_ch2[ch2_only_mask]
    rgb_colors[ch2_only_mask, 1] = norm_ch2[ch2_only_mask]
    rgb_colors[coexp_mask, 0] = norm_ch2[coexp_mask]
    rgb_colors[coexp_mask, 2] = norm_ch1[coexp_mask]
    rgb_colors = np.clip(rgb_colors, 0, 1)

    ax.scatter(plot_df['X'], plot_df['Y'], c=rgb_colors, s=1.0, marker='.', rasterized=True)
    # No need for xlim/ylim/aspect/ticks here as they are set in the main loop

# ==============================================================================
# Main Workflow Plots
# ==============================================================================

# Modified signature: Takes ordered_channels list
def plot_coexpression_matrix(scaled_pixel_expression: pd.DataFrame, # Contains 'X', 'Y', and scaled channels
                             pixel_results_df_with_avg: pd.DataFrame, # MUST contain 'X', 'Y', and avg community mapped cols (e.g., 'CD4_asinh_scaled_avg')
                             ordered_channels: List[str],
                             roi_string: str,
                             config: Dict,
                             output_path: str,
                             plot_dpi: int = 150):
    """Generates a matrix showing pairwise channel co-expression spatially.
       Diagonal: Single channel scaled expression.
       Upper Triangle: Scaled Pixel Co-expression (Magenta/Blue/Yellow).
       Lower Triangle: Mapped Avg Community Co-expression (Magenta/Blue/Yellow).
    """
    print(f"   Saving combined co-expression matrix to: {os.path.basename(output_path)}")
    cfg_viz = config['processing']['visualization']
    scatter_size = cfg_viz.get('scatter_size', 1)
    scatter_marker = cfg_viz.get('scatter_marker', '.')

    channels = ordered_channels # Use the provided order
    n_channels = len(channels)
    if n_channels < 1:
        print("   Skipping coexpression matrix: No channels provided.")
        return

    # Determine grid layout
    fig, axes = plt.subplots(n_channels, n_channels, figsize=(n_channels * 2.5, n_channels * 2.5), squeeze=False)
    # Combine pixel coordinates with scaled expression
    pixel_data_scaled = scaled_pixel_expression.copy()
    if 'X' not in pixel_data_scaled.columns or 'Y' not in pixel_data_scaled.columns:
        coords = pixel_results_df_with_avg[['X', 'Y']].loc[pixel_data_scaled.index]
        pixel_data_scaled = coords.join(pixel_data_scaled)

    x_min, x_max = pixel_data_scaled['X'].min(), pixel_data_scaled['X'].max()
    y_min, y_max = pixel_data_scaled['Y'].min(), pixel_data_scaled['Y'].max()

    # Store original cofactors (assuming they are in config or passed differently)
    # This part needs refinement based on where cofactors are stored/passed
    # Example: Assuming cofactors are implicitly handled by the scaled data

    for i, channel1 in enumerate(channels):
        for j, channel2 in enumerate(channels):
            ax = axes[i, j]

            if i == j: # Diagonal: Plot single channel (Scaled Pixel)
                scaled_col = f'{channel1}_asinh_scaled'
                if scaled_col in pixel_data_scaled.columns:
                    plot_data_diag = pixel_data_scaled[['X', 'Y', scaled_col]].dropna()
                    if not plot_data_diag.empty:
                        norm = plt.Normalize(vmin=0, vmax=1)
                        colors = plt.cm.viridis(norm(plot_data_diag[scaled_col]))
                        ax.scatter(plot_data_diag['X'], plot_data_diag['Y'], c=colors, s=scatter_size, marker=scatter_marker, edgecolors='none', rasterized=True)
                    else:
                         ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                else:
                     ax.text(0.5, 0.5, 'Scaled Data\nMissing', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{channel1}\n(Scaled Pixel)", fontsize=8, pad=2)

            elif i < j: # Upper triangle: Scaled Pixel Co-expression
                try:
                    _plot_scaled_pixel_coexpression(ax, pixel_data_scaled, channel1, channel2)
                    ax.set_title(f"{channel1} (P) /\n{channel2} (P)", fontsize=8, pad=2)
                except Exception as e:
                    print(f"   Warn: Failed pixel coexp plot for {channel1}/{channel2}: {e}")
                    ax.text(0.5, 0.5, 'Plot Error', ha='center', va='center', transform=ax.transAxes)

            else: # Lower triangle: Mapped Avg Community Co-expression
                try:
                    _plot_mapped_avg_community_coexpression(ax, pixel_results_df_with_avg, channel1, channel2)
                    ax.set_title(f"{channel1} (A) \n{channel2} (A)", fontsize=8, pad=2)
                except Exception as e:
                    print(f"   Warn: Failed avg comm coexp plot for {channel1}/{channel2}: {e}")
                    ax.text(0.5, 0.5, 'Plot Error', ha='center', va='center', transform=ax.transAxes)

            # Common Axes settings
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal'); ax.invert_yaxis()
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_linewidth(0.2)

    # Add overall title and legends
    fig.suptitle(f'Channel Co-expression Matrix - ROI: {roi_string}', fontsize=16, y=0.99)
    fig.text(0.5, 0.96, 'Diagonal: Scaled Pixel Expression (Viridis)', ha='center', fontsize=10)
    fig.text(0.5, 0.94, 'Upper Triangle: Scaled Pixel Co-expression (P)', ha='center', fontsize=10)
    fig.text(0.5, 0.92, 'Lower Triangle: Mapped Avg Community Co-expression (A)', ha='center', fontsize=10)

    # Add the custom co-expression colorbar
    try:
        add_coexpression_colorbar(fig, bbox=[0.85, 0.05, 0.08, 0.12]) # Adjust bbox as needed
        fig.text(0.89, 0.18, 'Co-expression\nColor Key', ha='center', va='bottom', fontsize=9)
    except Exception as cbar_e:
         print(f"   Warning: Failed to add coexpression colorbar: {cbar_e}")

    # Add colorbar for diagonal (Viridis)
    try:
        cbar_ax_diag = fig.add_axes([0.05, 0.05, 0.015, 0.12]) # Adjust bbox as needed
        sm_diag = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
        sm_diag.set_array([])
        cbar_diag = fig.colorbar(sm_diag, cax=cbar_ax_diag, orientation='vertical')
        cbar_diag.set_label('Scaled Expr', size=8)
        cbar_diag.ax.tick_params(labelsize=7)
        cbar_diag.set_ticks([0, 1])
        fig.text(0.07, 0.18, 'Single Marker\nColor Key', ha='center', va='bottom', fontsize=9)
    except Exception as cbar_e:
         print(f"   Warning: Failed to add diagonal colorbar: {cbar_e}")

    plt.subplots_adjust(left=0.1, right=0.83, top=0.9, bottom=0.1, wspace=0.1, hspace=0.3)
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
    plt.close(fig)

# Modified to return channel order
def plot_correlation_clustermap(correlation_matrix: pd.DataFrame,
                                channels: list,
                                title: str,
                                output_path: str,
                                plot_dpi: int = 150) -> Optional[List[str]]: # Return type added
    """
    Generates a clustermap (heatmap with hierarchical clustering) of a channel correlation matrix.
    Uses seaborn for plotting.

    Args:
        correlation_matrix: A pandas DataFrame containing the correlation values.
        channels: The list of channel names corresponding to the matrix rows/columns.
        title: Title for the plot.
        output_path: Full path to save the output plot file.
        plot_dpi: DPI for the saved plot.

    Returns:
        The list of channel names in the order determined by the clustering,
        or the original list if clustering fails or is skipped.
    """
    print(f"   Saving correlation clustermap to: {os.path.basename(output_path)}")
    ordered_channels = channels # Default return value

    try:
        num_ch = len(channels)
        if num_ch < 2:
            print("   Skipping clustermap generation: Need at least 2 channels.")
            return channels # Return original channels if not enough to cluster

        fig_size_dim = max(8, num_ch * 0.7)
        annot_size = 6 if num_ch > 15 else 8

        plt.figure() # Create a new figure context for clustermap
        clustergrid = sns.clustermap(
            correlation_matrix,
            method='average',
            metric='correlation',
            cmap='vlag', vmin=-1, vmax=1,
            linewidths=0.5, linecolor='lightgray',
            annot=True,
            annot_kws={"size": annot_size},
            fmt=".2f",
            figsize=(fig_size_dim, fig_size_dim)
        )

        # Get channel order from the plotted heatmap
        try:
            # Access the reordered index from the heatmap axes
            ordered_channels = clustergrid.dendrogram_row.reordered_ind
            ordered_channels = correlation_matrix.index[ordered_channels].tolist()
            print(f"   Determined channel order from clustermap: {ordered_channels}")
        except AttributeError:
            # Fallback for older seaborn or if attribute names change
            try:
                 ordered_indices = clustergrid.dendrogram_row.reordered_ind
                 ordered_channels = [correlation_matrix.index[i] for i in ordered_indices]
                 print(f"   Determined channel order from clustermap (fallback): {ordered_channels}")
            except Exception as e:
                 print(f"   WARNING: Could not extract channel order from clustermap: {e}. Using original order.")
                 ordered_channels = channels # Fallback to original channel list
        except Exception as e:
             print(f"   WARNING: Could not extract channel order from clustermap: {e}. Using original order.")
             ordered_channels = channels # Fallback to original channel list

        clustergrid.fig.suptitle(title, y=1.02, fontsize=14)
        plt.setp(clustergrid.ax_heatmap.get_xticklabels(), rotation=60, ha='right', fontsize=9)
        plt.setp(clustergrid.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)

        # Adjust layout slightly to prevent title overlap
        clustergrid.fig.tight_layout(rect=[0, 0, 1, 0.97])

        plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        print(f"   Correlation clustermap saved to: {output_path}")
        plt.close(clustergrid.fig)

    except ImportError:
        print("   Seaborn package not found. Cannot generate clustermap.")
        ordered_channels = channels # Fallback
    except Exception as e:
        print(f"   Error during clustermap visualization: {e}")
        traceback.print_exc()
        plt.close('all') # Close any potentially lingering plots
        ordered_channels = channels # Fallback

    return ordered_channels # Return the determined (or fallback) order

def plot_umap_scatter(umap_coords: pd.DataFrame, # Should have UMAP1, UMAP2, index=community_id
                        community_top_channel_map: pd.Series, # index=community_id, value=channel_name
                        protein_marker_channels: list,
                        roi_string: str,
                        output_path: str,
                        plot_dpi: int = 150):
    """
    Generates a scatter plot of UMAP embeddings, colored by the primary protein markers.

    Args:
        umap_coords: DataFrame containing UMAP coordinates (e.g., 'UMAP1', 'UMAP2'). Index should be community IDs.
        community_top_channel_map: Series mapping community ID to primary channel name.
        protein_marker_channels: List of channels considered protein markers (for filtering).
        roi_string: Identifier string for the ROI (used in title).
        output_path: Full path to save the output plot file.
        plot_dpi: DPI for the saved plot.
    """
    print(f"   Generating UMAP scatter plot: {os.path.basename(output_path)}")
    if not umap_available:
         print("   Skipping UMAP scatter plot: umap-learn package not installed.")
         return
    if umap_coords.empty:
        print("   Skipping UMAP scatter plot: UMAP coordinates DataFrame is empty.")
        return
    if community_top_channel_map.empty:
        print("   Skipping UMAP scatter plot: Community primary channel map is empty.")
        return
    if not protein_marker_channels:
        print("   Skipping UMAP scatter plot: List of protein marker channels is empty.")
        return

    try:
        plot_data = umap_coords.copy()
        # Ensure index types match for mapping
        plot_data.index = plot_data.index.astype(community_top_channel_map.index.dtype)
        # Map primary channel
        plot_data['primary_channel_overall'] = plot_data.index.map(community_top_channel_map).fillna('Unknown')

        # Filter communities primarily identified by a protein marker
        communities_to_plot = plot_data[
            plot_data['primary_channel_overall'].isin(protein_marker_channels)
        ].copy()

        if communities_to_plot.empty:
            print("   Skipping UMAP plot: No communities were primarily identified by a specified protein marker.")
            return

        print(f"   Plotting {len(communities_to_plot)} communities primarily identified by protein markers.")
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 8))

        unique_primary_channels_plot = sorted(communities_to_plot['primary_channel_overall'].unique())
        palette = sns.color_palette("tab20", n_colors=len(unique_primary_channels_plot))
        channel_color_map = {channel: color for channel, color in zip(unique_primary_channels_plot, palette)}

        for channel, color in channel_color_map.items():
            subset = communities_to_plot[communities_to_plot['primary_channel_overall'] == channel]
            if not subset.empty:
                # Use UMAP1 and UMAP2 if they exist, otherwise first two columns
                umap1_col = 'UMAP1' if 'UMAP1' in subset.columns else subset.columns[0]
                umap2_col = 'UMAP2' if 'UMAP2' in subset.columns else subset.columns[1]
                ax.scatter(
                    subset[umap1_col], subset[umap2_col],
                    label=channel.split('(')[0], # Use clean channel name for label
                    color=color, s=15, alpha=0.8 # Increased size slightly
                )

        ax.set_title(f'UMAP of Communities (Asinh Scaled Diff. Profiles) - ROI: {roi_string}', fontsize=14)
        ax.set_xlabel('UMAP Component 1', fontsize=12)
        ax.set_ylabel('UMAP Component 2', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Add legend outside the plot
        ax.legend(title="Primary Protein Marker", bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=9)

        fig.tight_layout(rect=[0, 0, 0.88, 0.96]) # Adjust rect for legend and title
        plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        print(f"   UMAP community scatter plot saved to: {output_path}")
        plt.close(fig)

    except KeyError as e:
         print(f"   Error during UMAP scatter plot: Missing expected column - {e}. Ensure UMAP coordinates have UMAP1/UMAP2 or similar.")
         traceback.print_exc()
         plt.close('all')
    except Exception as e:
        print(f"   Error during UMAP scatter plot visualization: {e}")
        traceback.print_exc()
        plt.close('all')

# New plot function replacing the raw spatial grid - REVISED for Side-by-Side Comparison & GridSpec Layout
def plot_raw_vs_scaled_spatial_comparison(roi_raw_data: pd.DataFrame,
                                          scaled_pixel_expression: pd.DataFrame,
                                          roi_channels: List[str],
                                          config: Dict,
                                          output_path: str,
                                          roi_string: str):
    """Generates a side-by-side comparison of raw and scaled spatial expression using GridSpec.

    Each row represents a channel, showing Raw on the left and Arcsinh Scaled on the right.
    Uses a shared colorbar per row, based on the scaled data range.
    Layout mimics plot_coexpression_matrix using GridSpec.
    """
    function_name = "Raw vs. Scaled Spatial Comparison (GridSpec)"
    print(f"   Generating {function_name}: {os.path.basename(output_path)}")

    if len(roi_channels) < 1:
        print(f"   Skipping {function_name}: No channels available.")
        return
    if not all(c in roi_raw_data.columns for c in ['X', 'Y']):
        print(f"   Skipping {function_name}: Coordinate columns ('X', 'Y') not found in raw data.")
        return

    n_channels = len(roi_channels)
    n_cols = 2 # Raw and Scaled
    plot_dpi = config['processing']['plot_dpi']
    cfg_viz = config['processing']['visualization']
    scatter_size = cfg_viz.get('scatter_size', 1)
    scatter_marker = cfg_viz.get('scatter_marker', '.')
    cmap_shared = cfg_viz.get('comparison_cmap', 'viridis')

    plt.close('all')
    # --- Figure Setup using GridSpec (adapted from plot_coexpression_matrix) ---
    plot_size_inches = 2.5 # Size per subplot (raw or scaled)
    row_label_space_inches = 0.6 # Space for channel names on the left
    col_label_space_inches = 0.3 # Space for "Raw"/"Scaled" labels at top
    top_matter_inches = 0.8 # Space for main title
    cbar_space_inches = 0.6 # Space for the shared colorbar on the right
    inter_col_space_inches = 0.1 # Space between Raw and Scaled columns

    # Calculate figure dimensions
    fig_width_inches = row_label_space_inches + n_cols * plot_size_inches + inter_col_space_inches * (n_cols - 1) + cbar_space_inches
    fig_height_inches = top_matter_inches + n_channels * plot_size_inches + col_label_space_inches
    max_fig_dim_inches = 50
    fig_width_inches = min(fig_width_inches, max_fig_dim_inches)
    fig_height_inches = min(fig_height_inches, max_fig_dim_inches)

    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))

    # Calculate GridSpec margins and dimensions (relative fractions)
    left_margin_frac = row_label_space_inches / fig_width_inches
    right_margin_frac = cbar_space_inches / fig_width_inches
    bottom_margin_frac = 0.05 # Minimal bottom margin
    top_margin_frac = (top_matter_inches + col_label_space_inches) / fig_height_inches

    grid_left = left_margin_frac
    grid_bottom = bottom_margin_frac
    grid_width = max(0.1, 1.0 - left_margin_frac - right_margin_frac)
    grid_height = max(0.1, 1.0 - bottom_margin_frac - top_margin_frac)
    grid_top_edge = grid_bottom + grid_height

    gs = fig.add_gridspec(n_channels, n_cols,
                          left=grid_left, bottom=grid_bottom,
                          right=grid_left + grid_width, top=grid_top_edge,
                          wspace=inter_col_space_inches / plot_size_inches, # Adjust wspace based on plot size
                          hspace=0.05)

    pixel_coords = roi_raw_data[['X', 'Y']]
    x_min, x_max = pixel_coords['X'].min(), pixel_coords['X'].max()
    y_min, y_max = pixel_coords['Y'].min(), pixel_coords['Y'].max()

    # --- Plot Data ---
    mappables = [] # To store scalar mappables for colorbars
    for i, channel in enumerate(roi_channels):
        # Add subplots using GridSpec
        ax_raw = fig.add_subplot(gs[i, 0])
        ax_scaled = fig.add_subplot(gs[i, 1])

        # Determine vmin/vmax based on SCALED data for this channel (same logic as before)
        vmin, vmax = None, None
        scatter_scaled = None
        if channel in scaled_pixel_expression.columns:
            plot_values_scaled = scaled_pixel_expression[channel].copy()
            # Use robust percentiles, handle cases with no variation
            q_low, q_high = np.percentile(plot_values_scaled[~np.isnan(plot_values_scaled)], [1, 99])
            if q_high > q_low:
                vmin, vmax = q_low, q_high
            else: # Handle constant or near-constant data
                median_val = np.median(plot_values_scaled[~np.isnan(plot_values_scaled)])
                vmin, vmax = median_val - 0.1, median_val + 0.1 # Create a small range around median
            vmin = min(vmin, vmax - 1e-6) # Ensure vmin < vmax
        else:
            vmin, vmax = 0, 1 # Default if scaled data missing

        # Plot Raw Data (Left Column)
        if channel in roi_raw_data.columns:
            plot_values_raw = roi_raw_data[channel].copy()
            # Use the vmin/vmax derived from the SCALED data for color mapping
            ax_raw.scatter(pixel_coords['X'], pixel_coords['Y'], c=plot_values_raw,
                           cmap=cmap_shared, s=scatter_size, marker=scatter_marker,
                           vmin=vmin, vmax=vmax, rasterized=True)
        else:
            ax_raw.text(0.5, 0.5, 'Missing Raw', ha='center', va='center', transform=ax_raw.transAxes, fontsize=7)

        # Plot Scaled Data (Right Column)
        if channel in scaled_pixel_expression.columns:
            plot_values_scaled = scaled_pixel_expression[channel].copy()
            # Use the same vmin/vmax
            scatter_scaled = ax_scaled.scatter(pixel_coords['X'], pixel_coords['Y'], c=plot_values_scaled,
                                       cmap=cmap_shared, s=scatter_size, marker=scatter_marker,
                                       vmin=vmin, vmax=vmax, rasterized=True)
        else:
            ax_scaled.text(0.5, 0.5, 'Missing Scaled', ha='center', va='center', transform=ax_scaled.transAxes, fontsize=7)

        mappables.append(scatter_scaled) # Store the mappable from the scaled plot

        # Axis formatting for both (same as before)
        for ax in [ax_raw, ax_scaled]:
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal'); ax.invert_yaxis()
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(''); ax.set_ylabel('')
            for spine in ax.spines.values(): spine.set_linewidth(0.5)

    # --- Add Shared Colorbar (one for the whole figure, representing scaled range) ---
    final_mappable = next((m for m in reversed(mappables) if m is not None), None)
    if final_mappable:
        # Position colorbar to the right of the grid
        cbar_left = grid_left + grid_width + 0.01 # Small gap after grid
        cbar_bottom = grid_bottom
        cbar_width_frac = cbar_space_inches / fig_width_inches * 0.5 # Make cbar narrower
        cbar_height_frac = grid_height
        cbar_ax_pos = [cbar_left, cbar_bottom, cbar_width_frac, cbar_height_frac]
        cax = fig.add_axes(cbar_ax_pos)
        cbar = fig.colorbar(final_mappable, cax=cax, orientation='vertical')
        cbar.set_label(f'Arcsinh Scaled Expression ({cmap_shared})', size=9)
        cbar.ax.tick_params(labelsize=8)
    else:
        print("   Warning: Could not add colorbar, no plottable scaled data found.")

    # --- Row/Column Labels and Title ---
    label_fontsize = 8
    row_label_x_offset = 0.005
    col_label_y_offset = 0.005
    bottoms, tops, lefts, rights = gs.get_grid_positions(fig)

    # Add Row Labels (Channel Names)
    for i in range(n_channels):
        row_label = roi_channels[i].split('(')[0] # Short name
        y_pos = (bottoms[i] + tops[i]) / 2
        fig.text(grid_left - row_label_x_offset, y_pos, row_label, va='center', ha='right', fontsize=label_fontsize, rotation=0)

    # Add Column Labels ("Raw", "Scaled")
    x_pos_raw = (lefts[0] + rights[0]) / 2
    x_pos_scaled = (lefts[1] + rights[1]) / 2
    fig.text(x_pos_raw, grid_top_edge + col_label_y_offset, "Raw", va='bottom', ha='center', fontsize=label_fontsize + 1, weight='bold')
    fig.text(x_pos_scaled, grid_top_edge + col_label_y_offset, "Scaled", va='bottom', ha='center', fontsize=label_fontsize + 1, weight='bold')

    # Overall Title
    title_y_pos = 1.0 - (top_matter_inches / (2 * fig_height_inches)) # Adjust vertical position
    fig.suptitle(f"Raw vs. Arcsinh Scaled Spatial Expression - {roi_string}", fontsize=14, y=title_y_pos)

    # --- Save Figure ---
    try:
        fig.savefig(output_path, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
        print(f"   {function_name} saved to: {output_path}")
    except Exception as e:
         print(f"   Error saving {function_name} to {output_path}: {e}")
    plt.close(fig)
