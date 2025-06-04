import os
import math
import time
import traceback
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance as dist # Added for pdist

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

# def plot_channel_pair(ax, sampled_df, channel1, channel2, cofactor=None, asinh_cofactors=None):
#     """
#     [DEPRECATED/EXPLORATORY] Plot a single channel pair on the given axis with arcsinh transformation.
#     Likely used in initial notebook exploration, may not be used in batch workflow.
#
#     Parameters:
#     - ax: matplotlib axis to plot on
#     - sampled_df: dataframe with channel data
#     - channel1: first channel name
#     - channel2: second channel name
#     - cofactor: Optional fixed arcsinh cofactor
#     - asinh_cofactors: Dictionary of optimal cofactors by channel
#
#     Returns:
#     - Used cofactor value
#     """
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
#         if asinh_cofactors and channel1 in asinh_cofactors and channel2 in asinh_cofactors:
#             # Use average of the two channels' optimal cofactors if available
#             cofactor = (asinh_cofactors[channel1] + asinh_cofactors[channel2]) / 2
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
#                              cofactor=None, asinh_cofactors=None):
#     """
#     [DEPRECATED/EXPLORATORY] Create a figure with co-expression maps for multiple channel pairs.
#     Likely used in initial notebook exploration, may not be used in batch workflow.
#     """
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
#                                        asinh_cofactors=asinh_cofactors)
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
# def generate_histograms(df, channels, asinh_cofactors, roi_string, save_dir, hist_cols, plot_dpi, default_cofactor):
#     """[DEPRECATED/EXPLORATORY] Generates and saves histograms with GMM and Otsu gating.
#         Requires scikit-learn and scikit-image.
#     """
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
#         cofactor = asinh_cofactors.get(channel, default_cofactor) # Use provided default
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
# def generate_coexpression_maps(df, channels, asinh_cofactors, roi_string, save_dir, plot_dpi, default_cofactor):
#     """
#     [DEPRECATED/EXPLORATORY] Generates and saves co-expression maps arranged in an upper triangular matrix layout.
#     """
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
#                 # Pass default cofactor explicitly if asinh_cofactors dict might be incomplete
#                 specific_asinh_cofactors = {ch: asinh_cofactors.get(ch, default_cofactor) for ch in [channel1, channel2]}
#                 current_cofactor = plot_channel_pair(ax, df, channel1, channel2,
#                                                    asinh_cofactors=specific_asinh_cofactors)
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
# """

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
    # Determine actual columns for scaled data: use raw channel names or '_asinh_scaled' suffix
    if channel1 in pixel_data_scaled.columns and channel2 in pixel_data_scaled.columns:
        scaled_col1 = channel1
        scaled_col2 = channel2
    elif f'{channel1}_asinh_scaled' in pixel_data_scaled.columns and f'{channel2}_asinh_scaled' in pixel_data_scaled.columns:
        scaled_col1 = f'{channel1}_asinh_scaled'
        scaled_col2 = f'{channel2}_asinh_scaled'
    else:
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

    # Determine grid layout and cap overall figure size
    target_subplot_size_inches = 2.0 # Reduced from 2.5
    max_overall_dimension_inches = 40.0  # Max total size for the figure (e.g., 40x40 inches)
    
    calculated_fig_dimension = n_channels * target_subplot_size_inches
    final_fig_dimension = min(calculated_fig_dimension, max_overall_dimension_inches)

    fig, axes = plt.subplots(n_channels, n_channels, figsize=(final_fig_dimension, final_fig_dimension), squeeze=False)
    
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
                # Determine actual column for scaled pixel expression
                if channel1 in pixel_data_scaled.columns:
                    diag_col = channel1
                elif f'{channel1}_asinh_scaled' in pixel_data_scaled.columns:
                    diag_col = f'{channel1}_asinh_scaled'
                else:
                    ax.text(0.5, 0.5, 'Scaled Data\nMissing', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{channel1}\n(Scaled Pixel)", fontsize=8, pad=2)
                    continue

                plot_data_diag = pixel_data_scaled[['X', 'Y', diag_col]].dropna()
                if not plot_data_diag.empty:
                    norm = plt.Normalize(vmin=0, vmax=1)
                    colors = plt.cm.viridis(norm(plot_data_diag[diag_col]))
                    ax.scatter(plot_data_diag['X'], plot_data_diag['Y'], c=colors, s=scatter_size, marker=scatter_marker, edgecolors='none', rasterized=True)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
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
                                channels: list, # If linkage provided, these are the labels for the linkage
                                title: str,
                                output_path: str,
                                plot_dpi: int = 150,
                                fixed_channel_order: Optional[List[str]] = None,
                                row_linkage_matrix: Optional[np.ndarray] = None, # New: Precomputed row linkage
                                col_linkage_matrix: Optional[np.ndarray] = None, # New: Precomputed col linkage
                                matrix_channel_order: Optional[List[str]] = None # New: Original order of `correlation_matrix` rows/cols if it differs from `channels` (linkage order)
                               ) -> Optional[List[str]]:
    """
    Generates and saves a clustermap (or heatmap if order is fixed) of channel correlations.
    If row/col_linkage_matrix is provided, it uses them. Otherwise, calculates linkage.

    Args:
        correlation_matrix: DataFrame containing the correlation values.
                            Its rows/columns should align with `matrix_channel_order` if provided,
                            otherwise assumed to align with `channels`.
        channels: List of channels. If linkage is provided, these are the labels for the
                  dendrogram that the linkage matrix refers to (i.e., `dendrogram['leaves']` order).
                  If linkage is NOT provided, these are the channels in `correlation_matrix` to be clustered.
        title: Title for the plot.
        output_path: Path to save the plot (.svg or .png recommended).
        plot_dpi: DPI for saving the plot.
        fixed_channel_order: If provided (and no col_linkage_matrix), use this exact order for columns
                             and plot a heatmap instead of a clustermap for columns.
                             Row clustering still happens unless row_linkage_matrix is also given.
        row_linkage_matrix: Optional precomputed linkage matrix for rows.
        col_linkage_matrix: Optional precomputed linkage matrix for columns.
        matrix_channel_order: Optional list of channel names representing the original order of
                              rows/columns in the input `correlation_matrix`. This is crucial if
                              `correlation_matrix` is not already ordered according to `channels` (linkage order).

    Returns:
        A list of channel names in the order they appear on the final plot axes,
        or None if plotting fails.
    """
    print(f"   Attempting to generate correlation map: {os.path.basename(output_path)}")
    if correlation_matrix.empty or len(channels) < 1: # Allow single channel if linkage provided (though unlikely useful)
        print("   Skipping correlation map: Not enough data or channels.")
        return None

    # Fill NaNs from .corr() upfront. This helps both branches.
    correlation_matrix_cleaned = correlation_matrix.fillna(0)

    # Determine appropriate figure size based on number of channels (use len(channels) as it dictates plot labels)
    n_plot_labels = len(channels)
    figsize_base = max(7, n_plot_labels * 0.5)  # Increased base and multiplier
    figsize = (figsize_base, figsize_base)

    # Prepare the matrix that will actually be displayed in the heatmap cells
    # This matrix needs to align with the final dendrogram order (defined by `channels` if linkage is given)
    # and have its diagonal set to NaN for display.
    
    # If matrix_channel_order is provided and differs from `channels` (linkage order),
    # reindex correlation_matrix_cleaned to match `channels` before display.
    if matrix_channel_order and set(matrix_channel_order) != set(channels):
        # This case is complex if matrix_channel_order has different items than channels.
        # Assume channels (linkage_order) is the target order for display rows/columns.
        # Ensure all channels in `channels` (linkage order) are present in `matrix_channel_order` for reindexing.
        if not all(c in matrix_channel_order for c in channels):
            print(f"   ERROR: Some channels for linkage order ({len(channels)}) not found in matrix_channel_order ({len(matrix_channel_order)}). Cannot align heatmap.")
            # print(f"   Linkage order (channels): {channels}")
            # print(f"   Matrix input order: {matrix_channel_order}")
            return None
        
        # Reindex the (cleaned) correlation matrix to match the order of `channels` (linkage order)
        # This ensures the heatmap cells correspond to the dendrogram leaves.
        correlation_matrix_for_display_ordered = correlation_matrix_cleaned.reindex(index=channels, columns=channels)
    elif matrix_channel_order and list(matrix_channel_order) != list(channels) and len(matrix_channel_order) == len(channels):
        # Same channels, just different order. Reindex to `channels` (linkage order).
        correlation_matrix_for_display_ordered = correlation_matrix_cleaned.reindex(index=channels, columns=channels)
    else:
        # If matrix_channel_order is not given, or matches `channels`, assume correlation_matrix_cleaned is already aligned with `channels`
        # or that `channels` refers to the columns of correlation_matrix_cleaned if no linkage is given.
        correlation_matrix_for_display_ordered = correlation_matrix_cleaned.copy() # Make a copy

    # Now set the diagonal to NaN on the ordered matrix for display
    np.fill_diagonal(correlation_matrix_for_display_ordered.values, np.nan)


    # --- Branch for fixed column order (and potentially precomputed row linkage) --- 
    if fixed_channel_order and not col_linkage_matrix:
        print(f"   Generating clustermap with fixed columns ({len(fixed_channel_order)}) and potentially clustered rows...")
        # The matrix_for_plot should be a slice of the *original* correlation data, ordered by fixed_channel_order for columns.
        # Its rows should align with `channels` if row_linkage is given, or all available if not.
        
        # For display, we need correlation_matrix_for_display_ordered but columns subsetted and ordered by fixed_channel_order.
        # Rows are ordered by `channels` (which is the linkage order or original row order)
        try:
            # Ensure fixed_channel_order channels exist in the display matrix columns
            valid_fixed_cols = [col for col in fixed_channel_order if col in correlation_matrix_for_display_ordered.columns]
            if not valid_fixed_cols:
                print(f"   ERROR: No valid channels from fixed_channel_order found in the correlation matrix columns.")
                return None
            matrix_for_plot_display = correlation_matrix_for_display_ordered.loc[:, valid_fixed_cols]
        except KeyError as e:
            print(f"   ERROR: Channel in fixed_channel_order not found in correlation matrix for display: {e}")
            return None

        # Determine row clustering parameters
        row_cluster_flag = True
        actual_row_linkage = row_linkage_matrix
        row_method = 'ward' # Default if row_linkage_matrix not given
        row_metric = 'euclidean' # Default if row_linkage_matrix not given

        if actual_row_linkage is not None:
            print("   Using precomputed row linkage.")
            row_method = None # Not used if linkage is provided
            row_metric = None # Not used if linkage is provided
        else:
            print("   Calculating row linkage on-the-fly.")
            # Linkage will be calculated on the rows of `matrix_for_plot_display` (values, not correlations)
            # This implies we need to pass the *values* to be clustered, not the display matrix with NaNs.
            # Let's use correlation_matrix_cleaned, reindexed to `channels` for rows and `valid_fixed_cols` for columns.
            matrix_for_row_linkage_calc = correlation_matrix_cleaned.reindex(index=channels)[valid_fixed_cols]
            if matrix_for_row_linkage_calc.shape[0] < 2:
                 print("   Skipping row clustering (less than 2 rows after selection for fixed columns).")
                 row_cluster_flag = False

        clustermap = sns.clustermap(
            matrix_for_plot_display, # Display this matrix (NaN diagonal, selected cols)
            method=row_method if row_cluster_flag else None, 
            metric=row_metric if row_cluster_flag else None,
            row_linkage=actual_row_linkage if row_cluster_flag else None,
            row_cluster=row_cluster_flag,
            col_cluster=False,  # Do NOT cluster columns (fixed order)
            data2=matrix_for_row_linkage_calc if actual_row_linkage is None and row_cluster_flag else None, # Data for on-the-fly row linkage
            annot=True,
            fmt='.2f',
            annot_kws={"size": max(4, 8 - n_plot_labels // 10)},
            cmap='coolwarm',
            vmin=-1, vmax=1,
            linewidths=.5,
            figsize=figsize,
            xticklabels=valid_fixed_cols, # Use labels from fixed_channel_order
            yticklabels=channels if row_cluster_flag else False, # Use `channels` for row labels if clustered
            dendrogram_ratio=(.15 if row_cluster_flag else 0.0, 0.0), # Row dendrogram only if clustered
            cbar_pos=None
        )
        # ... rest of fixed_channel_order plotting logic (labels, title, cbar, return order) ...
        # This part needs careful review for label extraction based on what was clustered.
        # For simplicity, if row linkage was given, `channels` is the order. 
        # If calculated, it comes from clustermap.dendrogram_row.reordered_ind mapped to `channels`.

        clustermap.ax_heatmap.set_xlabel("Channels (Fixed Order)", fontsize=9)
        clustermap.ax_heatmap.set_ylabel("Channels (Clustered by Row)", fontsize=9)
        plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=90, fontsize=max(4, 8 - n_plot_labels // 10))
        # clustermap.ax_heatmap.set_xticks(np.arange(len(valid_fixed_cols)) + 0.5) # Already handled by xticklabels=valid_fixed_cols
        # clustermap.ax_heatmap.set_xticklabels(valid_fixed_cols) # Redundant
        plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0, fontsize=max(4, 8 - n_plot_labels // 10))
        clustermap.fig.suptitle(title, y=1.02, fontsize=10)
        cbar_ax = clustermap.fig.add_axes([0.85, 0.8, 0.03, 0.15])
        plt.colorbar(clustermap.ax_heatmap.get_children()[0], cax=cbar_ax, label="Spearman Correlation")

        ordered_channels_list_plot = channels # Default to input `channels` which should be linkage order for rows
        if row_cluster_flag and actual_row_linkage is None: # If rows were clustered on-the-fly
            try:
                 row_indices = clustermap.dendrogram_row.reordered_ind
                 ordered_channels_list_plot = [channels[i] for i in row_indices]
            except Exception as e_ord:
                 print(f"   WARNING: Could not extract row channel order from fixed-col clustermap: {e_ord}")
        
        plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close() 
        print(f"   --- Correlation map saved to: {os.path.basename(output_path)}")
        return ordered_channels_list_plot

    # --- Branch for full clustering (potentially with precomputed linkage for both axes) ---
    else:
        print(f"   Generating full clustermap ({n_plot_labels} channels)...")
        
        actual_row_linkage = row_linkage_matrix
        actual_col_linkage = col_linkage_matrix
        linkage_method = 'ward' # Default if linkage not provided
        linkage_metric = 'euclidean' # Default if linkage not provided

        if actual_row_linkage is not None and actual_col_linkage is not None:
            print("   Using precomputed row and column linkage matrices.")
            linkage_method = None # Not used if linkage is provided
            linkage_metric = None # Not used if linkage is ખprovided
            # `channels` argument to clustermap should be the labels for the dendrogram
            # `correlation_matrix_for_display_ordered` is already aligned with `channels`.
        else:
            print("   Calculating linkage on-the-fly.")
            # For on-the-fly calculation, linkage is computed on correlation_matrix_cleaned (no NaN diagonal)
            # Ensure it uses the `channels` subset if `correlation_matrix_cleaned` is larger.
            if set(channels) != set(correlation_matrix_cleaned.columns):
                 matrix_for_on_the_fly_linkage = correlation_matrix_cleaned.loc[channels, channels]
            else:
                 matrix_for_on_the_fly_linkage = correlation_matrix_cleaned
            
            # The diagonal for linkage calculation should not be 1.0 (self-correlation)
            # It was already set to 0 in the previous version for on-the-fly, let's ensure that concept carries over.
            # However, seaborn's clustermap default for `metric` on a correlation matrix is often `1 - corr` or similar.
            # For `euclidean` distance on correlations, a 0 diagonal is better.
            # Let's ensure the matrix passed for linkage calculation has a 0 diagonal.
            linkage_calc_input_df = matrix_for_on_the_fly_linkage.copy()
            np.fill_diagonal(linkage_calc_input_df.values, 0)
            
            # `data2` param in seaborn.clustermap can be used to specify the data for linkage if different from display data.
            # We will pass `linkage_calc_input_df` via `data2` if we let clustermap compute linkage.
            # This is getting convoluted. The previous fix of calculating linkage manually was cleaner.
            # Let's revert to manual linkage calculation if not provided.
            try:
                # Correlated data for linkage calculation
                _matrix_for_linkage = correlation_matrix_cleaned.loc[channels, channels].copy()
                np.fill_diagonal(_matrix_for_linkage.values, 0) # Diagonal = 0 for distance
                
                pairwise_dists = dist.pdist(_matrix_for_linkage.values, metric='euclidean')
                if not (np.all(np.isfinite(pairwise_dists)) and pairwise_dists.shape[0] > 0):
                    print(f"   ERROR: Could not compute valid finite pairwise distances for on-the-fly linkage. Aborting plot for {title}.")
                    return None
                actual_row_linkage = sch.linkage(pairwise_dists, method='ward')
                actual_col_linkage = actual_row_linkage # Symmetric
                linkage_method = None
                linkage_metric = None
                print("   Successfully calculated linkage on-the-fly using euclidean distance on zero-diagonal correlation matrix.")
            except ValueError as ve:
                print(f"   ERROR: Failed during on-the-fly linkage calculation for {title}: {ve}.")
                traceback.print_exc()
                return None
            except Exception as e_link:
                print(f"   ERROR: Unexpected error during on-the-fly linkage calculation for {title}: {e_link}.")
                traceback.print_exc()
                return None

        plt.figure(figsize=figsize) # Create a figure context for heatmap or clustermap
        ordered_channels_list_plot = None

        try:
            clustermap = sns.clustermap(
                correlation_matrix_for_display_ordered, # Display this matrix (ordered by `channels`, NaN diagonal)
                method=linkage_method,      # Only if linkage not provided
                metric=linkage_metric,      # Only if linkage not provided
                row_linkage=actual_row_linkage,
                col_linkage=actual_col_linkage,
                annot=True,
                fmt='.2f',
                annot_kws={"size": max(4, 8 - n_plot_labels // 10)},
                cmap='coolwarm',
                vmin=-1, vmax=1,
                linewidths=.5,
                figsize=figsize,
                xticklabels=channels, # Labels are from `channels` (linkage order)
                yticklabels=channels, # Labels are from `channels` (linkage order)
                dendrogram_ratio=(.15, .15),
                cbar_pos=None
            )

            clustermap.ax_heatmap.set_xlabel("Channels", fontsize=9)
            clustermap.ax_heatmap.set_ylabel("Channels", fontsize=9)
            plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=90, fontsize=max(4, 8 - n_plot_labels // 10))
            plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0, fontsize=max(4, 8 - n_plot_labels // 10))
            clustermap.fig.suptitle(title, y=1.02, fontsize=10)
            cbar_ax = clustermap.fig.add_axes([0.85, 0.8, 0.03, 0.15])
            plt.colorbar(clustermap.ax_heatmap.get_children()[0], cax=cbar_ax, label="Spearman Correlation")

            # If linkage was provided, `channels` is already the final plot order.
            # If linkage was calculated by clustermap, extract order.
            # However, we now always provide linkage or calculate it manually before clustermap.
            # So, the order is effectively `channels` as reordered by the `actual_row_linkage`.
            try:
                 row_indices = clustermap.dendrogram_row.reordered_ind # These are indices into `channels`
                 ordered_channels_list_plot = [channels[i] for i in row_indices]
                 # col_indices = clustermap.dendrogram_col.reordered_ind
                 # col_ordered_channels = [channels[i] for i in col_indices]
                 # if ordered_channels_list_plot != col_ordered_channels:
                 #      print("   WARNING: Row and Column clustering order differs. Using row order.")
            except Exception as e:
                 print(f"   WARNING: Could not extract channel order from clustermap dendrogram object: {e}. Using input channel order.")
                 ordered_channels_list_plot = channels

            plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
            plt.close()
            print(f"   --- Correlation map saved to: {os.path.basename(output_path)}")
            return ordered_channels_list_plot

        except Exception as e:
            print(f"   ERROR generating correlation map: {e}")
            traceback.print_exc()
            plt.close() # Ensure plot is closed on error
            return None # Return None on failure

# --- UMAP Community Plot ---

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

def plot_raw_vs_scaled_spatial_comparison(roi_raw_data: pd.DataFrame,
                                          scaled_pixel_expression: pd.DataFrame,
                                          roi_channels: List[str],
                                          config: Dict,
                                          output_path: str,
                                          roi_string: str,
                                          plot_dpi: Optional[int] = None): # Add plot_dpi here
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
    # Use provided plot_dpi if available, otherwise get from config or default
    final_plot_dpi = plot_dpi if plot_dpi is not None else config.get('processing', {}).get('plot_dpi', 150)

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
        fig.savefig(output_path, dpi=final_plot_dpi, bbox_inches='tight', facecolor='white')
        print(f"   {function_name} saved to: {output_path}")
    except Exception as e:
         print(f"   Error saving {function_name} to {output_path}: {e}")
    plt.close(fig)

# New function for community-based spatial visualization
def plot_community_spatial_grid(
    pixel_results_df: pd.DataFrame,     # Must contain 'X', 'Y', 'community', and channel columns
    scaled_community_profiles: pd.DataFrame,  # Community averages
    roi_channels: List[str],            # Channels to plot
    roi_string: str,                    # For plot title
    resolution_param: float,            # For plot title
    output_path: str,                   # Where to save the plot
    plot_dpi: int = 150,                # DPI for saving 
    max_channels_per_row: int = 5       # Layout control
):
    """
    Generates a grid of spatial plots showing communities colored by their average expression
    for each channel. Similar to plot_raw_vs_scaled_spatial_comparison but at community level.
    
    Args:
        pixel_results_df: DataFrame with spatial coordinates, community IDs and expression data
        scaled_community_profiles: DataFrame with community average expression per channel
        roi_channels: List of channels to visualize
        roi_string: ROI identifier for plot title
        resolution_param: Resolution parameter for plot title
        output_path: Path to save the output plot
        plot_dpi: DPI for saving the plot
        max_channels_per_row: Maximum number of channels to show per row
    """
    print(f"   Generating community spatial expression grid...")
    
    # Validate inputs
    if pixel_results_df.empty or 'community' not in pixel_results_df.columns:
        print("   ERROR: Input data missing or lacks community column.")
        return
    if scaled_community_profiles.empty:
        print("   ERROR: Community profile data is empty.")
        return
    if not roi_channels:
        print("   ERROR: No channels specified to plot.")
        return
        
    # Filter channels to those actually present in the data
    valid_channels = [ch for ch in roi_channels if ch in scaled_community_profiles.columns]
    if not valid_channels:
        print("   ERROR: None of the specified channels found in community profiles.")
        return
        
    # Calculate grid dimensions
    n_channels = len(valid_channels)
    n_cols = min(max_channels_per_row, n_channels)
    n_rows = int(np.ceil(n_channels / n_cols))
    
    # Create figure
    fig_width = n_cols * 3.5  # Width per subplot
    fig_height = n_rows * 3.5  # Height per subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    # Get spatial range for consistent axes
    x_min, x_max = pixel_results_df['X'].min(), pixel_results_df['X'].max()
    y_min, y_max = pixel_results_df['Y'].min(), pixel_results_df['Y'].max()
    
    # For each channel, plot communities colored by their average expression
    for idx, channel in enumerate(valid_channels):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx]
        
        # Map community averages to each pixel
        if channel in scaled_community_profiles.columns:
            # Create mapping from community ID to average expression
            comm_avg_map = scaled_community_profiles[channel].to_dict()
            
            # Apply mapping to get expression values for each pixel based on its community
            plot_df = pixel_results_df[['X', 'Y', 'community']].copy()
            plot_df['avg_expr'] = plot_df['community'].map(comm_avg_map).fillna(0)
            
            # Plot
            scatter = ax.scatter(
                plot_df['X'], plot_df['Y'], 
                c=plot_df['avg_expr'], 
                cmap='viridis', 
                s=1,  # Small point size for detail
                marker='.',
                alpha=0.8,
                vmin=0, vmax=1  # Assuming scaled 0-1 values
            )
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(scatter, cax=cax)
            
            # Title for each subplot
            ax.set_title(f"{channel}", fontsize=10)
            
            # Set limits for consistency
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            
            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.text(0.5, 0.5, f"No data for {channel}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide any unused subplots
    for idx in range(n_channels, n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx, col_idx].axis('off')
    
    # Add overall title
    fig.suptitle(f"Community Average Expression - {roi_string} (Res: {resolution_param})", 
                fontsize=14, y=0.98)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"   --- Community spatial grid saved to: {os.path.basename(output_path)}")

def plot_community_size_distribution(
    community_sizes: pd.Series,  # Series with community IDs as index and sizes as values
    output_path: str,           # Where to save the plot
    title: str,                 # Plot title
    plot_dpi: int = 150        # DPI for saving
):
    """
    Generates a plot showing the distribution of community sizes.
    
    Args:
        community_sizes: Pandas Series with community IDs as index and their sizes (number of pixels) as values
        output_path: Full path to save the output plot file
        title: Title for the plot
        plot_dpi: DPI for saving the plot
    """
    print(f"   Generating community size distribution plot...")
    
    if community_sizes.empty:
        print("   ERROR: No community size data provided.")
        return
        
    try:
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Area plot of community sizes (sorted)
        sorted_sizes = community_sizes.sort_values(ascending=False)
        
        # Create area plot instead of bars
        ax1.fill_between(range(len(sorted_sizes)), sorted_sizes.values,
                        color='skyblue', alpha=0.7)
        
        ax1.set_title('Community Sizes (Sorted by Size)', fontsize=12)
        ax1.set_xlabel('Community Rank (0 = largest)', fontsize=10)
        ax1.set_ylabel('Number of Pixels', fontsize=10)
        ax1.tick_params(axis='both', labelsize=8)
        
        # Adjust x-axis labels to show integer range (rank)
        num_communities_total = len(sorted_sizes)
        if num_communities_total > 0:
            if num_communities_total > 10:
                num_ticks_to_show = 10
                tick_positions = np.linspace(0, num_communities_total - 1, num_ticks_to_show, dtype=int)
                ax1.set_xticks(tick_positions)
                ax1.set_xticklabels([str(pos) for pos in tick_positions])
            else:
                tick_positions = np.arange(num_communities_total)
                ax1.set_xticks(tick_positions)
                ax1.set_xticklabels([str(pos) for pos in tick_positions])
            plt.setp(ax1.get_xticklabels(), rotation=0, ha='center')
        else:
            ax1.set_xticks([])
            ax1.set_xticklabels([])
        
        # 2. Histogram of size distribution
        ax2.hist(community_sizes.values, bins=min(20, len(community_sizes)),
                color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.set_title('Size Distribution', fontsize=12)
        ax2.set_xlabel('Number of Pixels', fontsize=10)
        ax2.set_ylabel('Number of Communities', fontsize=10)
        ax2.tick_params(axis='both', labelsize=8)
        
        # Add summary statistics as text
        stats_text = (
            f'Total Communities: {len(community_sizes):,}\n'
            f'Total Pixels: {community_sizes.sum():,}\n'
            f'Mean Size: {community_sizes.mean():.1f}\n'
            f'Median Size: {community_sizes.median():.1f}\n'
            f'Min Size: {community_sizes.min():,}\n'
            f'Max Size: {community_sizes.max():,}'
        )
        ax2.text(0.95, 0.95, stats_text,
                transform=ax2.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
        
        # Add overall title
        fig.suptitle(title, fontsize=14, y=1.02)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"   --- Community size distribution plot saved to: {os.path.basename(output_path)}")
        
    except Exception as e:
        print(f"   ERROR: Failed to generate community size distribution plot: {e}")
        traceback.print_exc()
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)

# --- New function for Dendrogram plotting ---

def plot_linkage_dendrogram(
    linkage_matrix: np.ndarray,
    labels: List[str], # Community IDs or Channel names
    title: str,
    output_path: str,
    plot_dpi: int = 150,
    orientation: str = 'top', # 'top', 'bottom', 'left', 'right'
    color_threshold: Optional[float] = None,
    truncate_mode: Optional[str] = None, # e.g., 'lastp', 'level'
    p: Optional[int] = 30, # Parameter for truncate_mode, changed default from None to 30
    leaf_font_size: Optional[int] = None,
    leaf_rotation: Optional[float] = 90,
    max_labels_to_show: int = 50 # If more labels, consider not showing or truncating
):
    """Generates and saves a dendrogram from a linkage matrix.

    Args:
        linkage_matrix: The hierarchical clustering linkage matrix (from scipy.cluster.hierarchy.linkage).
        labels: List of labels for the leaves of the dendrogram.
        title: Title for the plot.
        output_path: Path to save the plot.
        plot_dpi: DPI for saving the plot.
        orientation: Orientation of the dendrogram.
        color_threshold: Threshold for coloring clusters.
        truncate_mode: Mode for truncating the dendrogram (see scipy docs).
        p: Parameter for truncate_mode.
        leaf_font_size: Font size for leaf labels.
        leaf_rotation: Rotation for leaf labels.
        max_labels_to_show: If the number of labels exceeds this, leaf labels might be hidden or truncated for clarity.
    """
    print(f"   Generating dendrogram: {os.path.basename(output_path)}")
    if linkage_matrix is None or linkage_matrix.ndim != 2 or linkage_matrix.shape[0] == 0:
        print("   Skipping dendrogram: Invalid or empty linkage matrix provided.")
        return
    if not labels:
        print("   Skipping dendrogram: No labels provided.")
        return

    num_labels = len(labels)
    effective_leaf_font_size = leaf_font_size
    show_labels = True

    if num_labels > max_labels_to_show:
        print(f"   Warning: Number of labels ({num_labels}) exceeds max_labels_to_show ({max_labels_to_show}).")
        if truncate_mode is None: # If user hasn't specified truncation, suggest or hide labels
            print(f"     Consider using truncate_mode='lastp' with p=~{max_labels_to_show} or hiding labels.")
            # For very large numbers, hiding labels by default might be better
            if num_labels > max_labels_to_show * 2: # Arbitrary factor
                 print("     Hiding leaf labels for clarity due to large number.")
                 show_labels = False
        if effective_leaf_font_size is None:
            effective_leaf_font_size = 6 # Smaller font for many labels
    elif effective_leaf_font_size is None:
        effective_leaf_font_size = 8 # Default if not too many

    # Dynamically adjust figure size based on number of labels and orientation
    if orientation in ['top', 'bottom']:
        fig_width = max(8, num_labels * 0.2) # Adjust width based on number of labels
        fig_height = 6
    else: # left, right
        fig_width = 10
        fig_height = max(8, num_labels * 0.15)
    
    fig_width = min(fig_width, 40) # Max width
    fig_height = min(fig_height, 40) # Max height

    plt.close('all')
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    try:
        ddata = sch.dendrogram(
            linkage_matrix,
            orientation=orientation,
            labels=labels if show_labels else None,
            leaf_rotation=leaf_rotation if show_labels else 0,
            leaf_font_size=effective_leaf_font_size if show_labels else 1,
            truncate_mode=truncate_mode,
            p=p,
            color_threshold=color_threshold,
            ax=ax
        )

        ax.set_title(title, fontsize=14)
        if orientation in ['top', 'bottom']:
            ax.set_ylabel("Distance/Dissimilarity", fontsize=10)
        else:
            ax.set_xlabel("Distance/Dissimilarity", fontsize=10)
        
        # Add a line for color_threshold if provided
        if color_threshold is not None:
            if orientation in ['top', 'bottom']:
                ax.axhline(y=color_threshold, color='r', linestyle='--', linewidth=0.8)
            else:
                ax.axvline(x=color_threshold, color='r', linestyle='--', linewidth=0.8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        print(f"   --- Dendrogram saved to: {os.path.basename(output_path)}")

    except Exception as e:
        print(f"   ERROR generating dendrogram: {e}")
        traceback.print_exc()
    finally:
        plt.close(fig)

def analyze_dendrogram_by_controls(
    linkage_matrix: np.ndarray,
    channel_labels_for_linkage: List[str],
    background_channels: List[str],
    protein_channels: List[str], # Added for completeness, though not directly used in distance logic yet
    roi_string: str,
    output_dir: str, # Directory to save report and plot
    plot_dpi: int = 150
):
    """
    Analyzes a channel dendrogram based on when background (control) channels merge
    and reports cluster compositions at that reference distance.

    Args:
        linkage_matrix: The precomputed hierarchical clustering linkage matrix (from sch.linkage).
        channel_labels_for_linkage: List of channel names corresponding to the linkage matrix leaves.
                                   This order IS CRUCIAL and must match the order used to generate linkage_matrix.
        background_channels: List of channel names designated as background/control.
        protein_channels: List of channel names designated as protein markers.
        roi_string: Identifier for the ROI, used in output filenames.
        output_dir: Directory where the analysis report and dendrogram plot will be saved.
        plot_dpi: DPI for the saved dendrogram plot.
    """
    print(f"\n  Analyzing channel hierarchy for {roi_string} based on background channels...")
    os.makedirs(output_dir, exist_ok=True)
    report_lines = [f"--- Channel Hierarchy Analysis for ROI: {roi_string} ---"]

    if linkage_matrix is None or linkage_matrix.shape[0] == 0:
        print("    ERROR: Linkage matrix is empty or None. Cannot perform analysis.")
        report_lines.append("ERROR: Linkage matrix was empty or None.")
        # Save and return early
        report_path = os.path.join(output_dir, f"channel_hierarchy_analysis_error_{roi_string}.txt")
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        return

    if not channel_labels_for_linkage:
        print("    ERROR: channel_labels_for_linkage is empty. Cannot map channels.")
        report_lines.append("ERROR: channel_labels_for_linkage was empty.")
        report_path = os.path.join(output_dir, f"channel_hierarchy_analysis_error_{roi_string}.txt")
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        return
        
    num_original_observations = linkage_matrix.shape[0] + 1
    if num_original_observations != len(channel_labels_for_linkage):
        print(f"    ERROR: Mismatch between linkage matrix observations ({num_original_observations}) and labels provided ({len(channel_labels_for_linkage)}).")
        report_lines.append(f"ERROR: Linkage matrix implies {num_original_observations} items, but {len(channel_labels_for_linkage)} labels were given.")
        # Save and return early
        report_path = os.path.join(output_dir, f"channel_hierarchy_analysis_error_{roi_string}.txt")
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        return

    # Identify indices of background channels in the `channel_labels_for_linkage` list
    bg_channel_indices = [i for i, label in enumerate(channel_labels_for_linkage) if label in background_channels]
    report_lines.append(f"Found {len(background_channels)} background channels specified in config: {background_channels}")
    report_lines.append(f"Found {len(bg_channel_indices)} matching background channels in the linkage labels: {[channel_labels_for_linkage[i] for i in bg_channel_indices]}")

    if not bg_channel_indices:
        print("    WARNING: No specified background channels found in the linkage labels. Cannot determine reference distance.")
        report_lines.append("WARNING: No background channels from config found in the provided linkage labels.")
        control_reference_distance = None
    elif len(bg_channel_indices) == 1:
        print("    WARNING: Only one background channel found in linkage labels. Using its first merge distance if applicable, or max distance.")
        report_lines.append("WARNING: Only one background channel found. Reference distance logic may be limited.")
        # Find the first merge involving this single background channel's index
        # The linkage matrix format: [idx1, idx2, dist, num_points]
        # idx < num_original_observations are original items.
        # For a single control, its first merge distance is when it clusters with *anything*.
        min_dist_for_single_bg = float('inf')
        found_merge = False
        for i in range(linkage_matrix.shape[0]):
            if linkage_matrix[i, 0] == bg_channel_indices[0] or linkage_matrix[i, 1] == bg_channel_indices[0]:
                min_dist_for_single_bg = linkage_matrix[i, 2]
                found_merge = True
                break
        if found_merge:
            control_reference_distance = min_dist_for_single_bg
            report_lines.append(f"Single background channel first merges at distance: {control_reference_distance:.4f}")
        else:
            # This case should not happen if the item is in the linkage.
            control_reference_distance = linkage_matrix[:, 2].max() # Fallback to max distance
            report_lines.append(f"Single background channel never explicitly merged? Using max distance as fallback: {control_reference_distance:.4f}")
    else:
        # Determine the smallest distance at which all background channels are in the same cluster
        # Iterate through unique sorted distances from the linkage matrix
        unique_distances = sorted(np.unique(linkage_matrix[:, 2]))
        control_reference_distance = None
        for dist_thresh in unique_distances:
            # Get flat clusters at this distance
            # `fcluster` uses 1-based indexing for clusters, labels are 0-indexed original items
            flat_clusters = sch.fcluster(linkage_matrix, t=dist_thresh, criterion='distance')
            # Get the cluster IDs for our background channel indices
            bg_cluster_ids = [flat_clusters[i] for i in bg_channel_indices]
            if len(set(bg_cluster_ids)) == 1:
                control_reference_distance = dist_thresh
                report_lines.append(f"All specified background channels merge into a single cluster at distance: {control_reference_distance:.4f}")
                break
        if control_reference_distance is None:
            # This implies they never all merge, which is unlikely if they are part of the same dataset.
            # Default to the maximum distance in the linkage matrix if no common cluster is found sooner.
            control_reference_distance = linkage_matrix[:, 2].max()
            report_lines.append(f"WARNING: Background channels did not all merge into one cluster. Using max linkage distance as reference: {control_reference_distance:.4f}")

    report_lines.append("\n--- Cluster Composition at Control Reference Distance ---")
    if control_reference_distance is not None:
        report_lines.append(f"Reference Distance (all background channels merged): {control_reference_distance:.4f}")
        flat_clusters_at_ref_dist = sch.fcluster(linkage_matrix, t=control_reference_distance, criterion='distance')
        
        # Group channels by their cluster ID at this reference distance
        cluster_composition = {}
        for i, label in enumerate(channel_labels_for_linkage):
            cluster_id = flat_clusters_at_ref_dist[i]
            channel_type = "Protein" if label in protein_channels else ("Background" if label in background_channels else "Other")
            if cluster_id not in cluster_composition:
                cluster_composition[cluster_id] = []
            cluster_composition[cluster_id].append(f"{label} ({channel_type})")
        
        for cid in sorted(cluster_composition.keys()):
            report_lines.append(f"  Cluster {cid}:")
            for member in sorted(cluster_composition[cid]): # Sort members for consistent reporting
                report_lines.append(f"    - {member}")
    else:
        report_lines.append("Could not determine a control reference distance.")

    # --- New: Full Dendrogram Hierarchy ---
    report_lines.append("\n--- Full Channel Dendrogram Hierarchy ---")
    
    # Helper function to recursively get children of a cluster
    # Memoization cache for the helper function
    memo = {}

    def get_cluster_children_recursive(cluster_idx_in_linkage: int, linkage_matrix: np.ndarray, 
                                       num_original_observations: int, channel_labels_for_linkage: List[str],
                                       visited_clusters: set) -> List[str]:
        # cluster_idx_in_linkage is the 0-based index for the linkage matrix (row number)
        # The ID of the cluster formed at this step is num_original_observations + cluster_idx_in_linkage
        
        cluster_node_id = num_original_observations + cluster_idx_in_linkage

        if cluster_node_id in memo:
            return memo[cluster_node_id]

        if cluster_node_id in visited_clusters: # Avoid infinite recursion for malformed linkage (though unlikely with scipy)
            return [f"Recursive reference to cluster {cluster_node_id}"]
        visited_clusters.add(cluster_node_id)

        child1_id = int(linkage_matrix[cluster_idx_in_linkage, 0])
        child2_id = int(linkage_matrix[cluster_idx_in_linkage, 1])
        
        children_list = []
        
        # Process child 1
        if child1_id < num_original_observations: # It's a leaf (original channel)
            children_list.append(channel_labels_for_linkage[child1_id])
        else: # It's a non-leaf node (another cluster)
            # The ID needs to be converted back to linkage_matrix row index
            child1_linkage_idx = child1_id - num_original_observations
            children_list.extend(get_cluster_children_recursive(child1_linkage_idx, linkage_matrix, 
                                                                num_original_observations, channel_labels_for_linkage,
                                                                visited_clusters.copy())) # Pass a copy of visited set

        # Process child 2
        if child2_id < num_original_observations: # It's a leaf (original channel)
            children_list.append(channel_labels_for_linkage[child2_id])
        else: # It's a non-leaf node (another cluster)
            child2_linkage_idx = child2_id - num_original_observations
            children_list.extend(get_cluster_children_recursive(child2_linkage_idx, linkage_matrix,
                                                                num_original_observations, channel_labels_for_linkage,
                                                                visited_clusters.copy())) # Pass a copy of visited set
        
        # Sort for consistent output order of children leaves
        sorted_children = sorted(list(set(children_list))) # Use set to remove duplicates if a deep recursion path leads to same leaf multiple times
        memo[cluster_node_id] = sorted_children
        return sorted_children

    # Iterate through each merge step in the linkage matrix
    # Each row i in linkage_matrix represents a cluster formed.
    # The ID of this cluster is (num_original_observations + i).
    if linkage_matrix is not None and linkage_matrix.shape[0] > 0 :
        for i in range(linkage_matrix.shape[0]):
            cluster_node_id = num_original_observations + i
            cluster_distance = linkage_matrix[i, 2]
            num_leaves_in_cluster = int(linkage_matrix[i, 3])

            # Get the direct children IDs from the linkage matrix
            child1_direct_id = int(linkage_matrix[i, 0])
            child2_direct_id = int(linkage_matrix[i, 1])

            child1_label = channel_labels_for_linkage[child1_direct_id] if child1_direct_id < num_original_observations else f"Node {child1_direct_id}"
            child2_label = channel_labels_for_linkage[child2_direct_id] if child2_direct_id < num_original_observations else f"Node {child2_direct_id}"

            report_lines.append(f"  Node {cluster_node_id} (formed at dist: {cluster_distance:.4f}, {num_leaves_in_cluster} leaves):")
            report_lines.append(f"    Directly merges: {child1_label} + {child2_label}")
            
            # Get all leaf members of this cluster using the recursive helper
            # We pass 'i' as the cluster_idx_in_linkage
            memo.clear() # Clear memo for each top-level node analysis to ensure correctness if needed, though for leaves it should be fine.
            all_leaf_members = get_cluster_children_recursive(i, linkage_matrix, num_original_observations, channel_labels_for_linkage, set())
            report_lines.append(f"    All leaf members: {', '.join(all_leaf_members)}")
    else:
        report_lines.append("  Linkage matrix empty, cannot report full hierarchy.")
    # --- End New Section ---

    # --- Save Textual Report --- 
    report_path = os.path.join(output_dir, f"channel_hierarchy_analysis_{roi_string}.txt")
    try:
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        print(f"  Channel hierarchy analysis report saved to: {os.path.basename(report_path)}")
    except Exception as e_rep:
        print(f"  ERROR saving channel hierarchy report: {e_rep}")

    # --- Generate Annotated Dendrogram Plot --- 
    dendro_plot_path = os.path.join(output_dir, f"annotated_channel_dendrogram_{roi_string}.svg")
    try:
        # Determine leaf colors
        leaf_colors = {}
        for label in channel_labels_for_linkage:
            if label in background_channels:
                leaf_colors[label] = 'blue' # Background channels in blue
            elif label in protein_channels:
                leaf_colors[label] = 'red'   # Protein channels in red
            else:
                leaf_colors[label] = 'black' # Others in black

        # Figure size
        num_labels = len(channel_labels_for_linkage)
        fig_height = max(8, num_labels * 0.2) # Adjust height based on number of labels
        fig_width = 12 
        fig_height = min(fig_height, 40) # Max height

        plt.figure(figsize=(fig_width, fig_height))
        # Pass the `labels` argument to `dendrogram` directly
        ddata = sch.dendrogram(
            linkage_matrix,
            labels=channel_labels_for_linkage, 
            orientation='right',
            leaf_rotation=0,
            leaf_font_size=8 # Adjust as needed
        )
        
        # Apply colors to leaf labels
        ax = plt.gca()
        yticklabels = ax.get_ymajorticklabels()
        for ticklabel in yticklabels:
            label_text = ticklabel.get_text()
            if label_text in leaf_colors:
                ticklabel.set_color(leaf_colors[label_text])

        if control_reference_distance is not None:
            # Add a line for the control reference distance
            # For a right-oriented dendrogram, this is a vertical line on the x-axis (distance axis)
            plt.axvline(x=control_reference_distance, color='gray', linestyle='--', linewidth=1, 
                        label=f'Ctrl Ref Dist: {control_reference_distance:.2f}')
            plt.legend(fontsize=8)

        plt.title(f'Channel Hierarchical Clustering - {roi_string}\n(Background vs. Protein Markers)', fontsize=12)
        plt.xlabel("Distance (Spearman Correlation Based)", fontsize=10)
        plt.ylabel("Channels", fontsize=10)
        plt.tight_layout()
        plt.savefig(dendro_plot_path, dpi=plot_dpi, bbox_inches='tight')
        print(f"  Annotated dendrogram saved to: {os.path.basename(dendro_plot_path)}")
    except Exception as e_plot:
        print(f"  ERROR generating annotated dendrogram plot: {e_plot}")
        traceback.print_exc()
    finally:
        plt.close('all') # Ensure all figures are closed

# --- New function for Spatial Community Assignment Map ---
def plot_spatial_community_assignment_map(
    pixel_results_df: pd.DataFrame, # Must contain 'X', 'Y', 'community'
    output_path: str,
    title: str,
    plot_dpi: int = 150,
    scatter_size: float = 1.0,
    scatter_marker: str = '.',
    max_communities_for_legend: int = 20
):
    """Generates a spatial map of pixels colored by their assigned community ID.

    Args:
        pixel_results_df: DataFrame with pixel coordinates ('X', 'Y') and 'community' assignments.
        output_path: Path to save the plot.
        title: Title for the plot.
        plot_dpi: DPI for saving the plot.
        scatter_size: Size of the scatter points.
        scatter_marker: Marker style for the scatter points.
        max_communities_for_legend: Maximum number of communities to show in the legend.
                                      If more communities, legend is omitted for clarity.
    """
    print(f"   Generating spatial community assignment map: {os.path.basename(output_path)}")

    if pixel_results_df.empty or not all(col in pixel_results_df.columns for col in ['X', 'Y', 'community']):
        print("   Skipping spatial community map: Input data missing or lacks required columns (X, Y, community).")
        return

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 10)) # Adjust figsize as needed

    communities = pixel_results_df['community']
    unique_communities = sorted(communities.unique())
    n_communities = len(unique_communities)

    if n_communities == 0:
        print("   Skipping spatial community map: No communities found in data.")
        plt.close(fig)
        return

    # Select a colormap
    if n_communities <= 20:
        # cmap = plt.cm.get_cmap('tab20', n_communities)
        # colors = [cmap(i) for i in range(n_communities)]
        colors = sns.color_palette("tab20", n_colors=n_communities) # sns.color_palette is often better for distinctiveness
    elif n_communities <= 50: # Example: for a moderate number
        colors = sns.color_palette("husl", n_colors=n_communities) # HSLuv is good for many distinct colors
    else: # For very many communities
        # Glasbey can generate highly distinct colors, but might need to be installed or implemented.
        # Fallback to nipy_spectral or a similar continuous-like map if too many for discrete sets.
        try:
            # Attempt to use a palette that handles many categories well
            colors = sns.color_palette("nipy_spectral", n_colors=n_communities)
        except ValueError: # If n_colors is too large for even nipy_spectral in some contexts
            # Fallback if a specific number is an issue for a palette generator
            # Note: This might not give great distinctiveness for very high N.
            cmap_base = plt.cm.get_cmap('nipy_spectral') 
            colors = [cmap_base(i / (n_communities -1 if n_communities > 1 else 1)) for i in range(n_communities)]
            
    # Create a mapping from community ID to color
    community_color_map = {comm_id: colors[i] for i, comm_id in enumerate(unique_communities)}
    point_colors = communities.map(community_color_map).fillna('gray') # FillNA for safety

    ax.scatter(
        pixel_results_df['X'], 
        pixel_results_df['Y'], 
        c=point_colors,
        s=scatter_size, 
        marker=scatter_marker,
        rasterized=True # Good for large number of points
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() # Consistent with other spatial plots

    # Add legend if manageable
    if 1 < n_communities <= max_communities_for_legend:
        legend_elements = [Patch(facecolor=community_color_map[comm_id], 
                                 label=f'Community {comm_id}') 
                           for comm_id in unique_communities]
        ax.legend(handles=legend_elements, title="Communities", 
                  bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
    else:
        if n_communities > max_communities_for_legend:
            ax.text(1.02, 0.98, f'{n_communities} communities shown.\nLegend omitted for clarity.', 
                    transform=ax.transAxes, fontsize=8, ha='left', va='top')
        plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        print(f"   --- Spatial community map saved to: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"   ERROR saving spatial community map: {e}")
    finally:
        plt.close(fig)

# --- New function for Meta-Cluster Adjacency Heatmaps ---
def plot_meta_cluster_adjacencies(
    pixel_results_df: pd.DataFrame, # Must contain 'X', 'Y', 'community'
    community_linkage_matrix: np.ndarray,
    original_community_ids: List[Any], # e.g., avg_profiles.index.tolist()
    distance_threshold: float,
    output_dir: str, # Directory to save plots (one per meta-cluster)
    roi_string: str,
    resolution_param: str, # For filenames and titles
    plot_dpi: int = 100, # Lower DPI for potentially many small heatmaps
    min_communities_in_metacluster: int = 2
):
    """Generates heatmaps of spatial adjacencies between communities within meta-clusters.

    Meta-clusters are defined by cutting the community_linkage_matrix at distance_threshold.
    """
    function_name = "Meta-Cluster Spatial Adjacency"
    print(f"   Generating {function_name} plots...")

    if pixel_results_df.empty or not all(col in pixel_results_df.columns for col in ['X', 'Y', 'community']):
        print(f"   Skipping {function_name}: Pixel data missing or lacks X, Y, or community columns.")
        return
    if community_linkage_matrix is None or community_linkage_matrix.ndim != 2:
        print(f"   Skipping {function_name}: Invalid community linkage matrix.")
        return
    if not original_community_ids:
        print(f"   Skipping {function_name}: No original community IDs provided.")
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Assign original communities to meta-clusters
    # The linkage matrix operates on indices 0 to N-1 where N is #original_community_ids.
    # `fcluster` returns an array where value at index `i` is the meta-cluster ID for original_community_ids[i]
    try:
        # Ensure linkage matrix has correct number of initial observations implied by original_community_ids
        n_obs_linkage = community_linkage_matrix.shape[0] + 1
        if n_obs_linkage != len(original_community_ids):
            print(f"   ERROR in {function_name}: Linkage matrix implies {n_obs_linkage} observations, but {len(original_community_ids)} original community IDs were provided.")
            # Check if original_community_ids might be from a different source or if linkage is for different set of items
            # Example: if original_community_ids are from avg_profiles.index, ensure avg_profiles was from the same set of communities used for linkage
            print(f"     Linkage matrix shape: {community_linkage_matrix.shape}")
            print(f"     Number of original_community_ids: {len(original_community_ids)}")
            # print(f"     First few original_community_ids: {original_community_ids[:5]}")
            return
        meta_cluster_assignments = sch.fcluster(community_linkage_matrix, t=distance_threshold, criterion='distance')
    except Exception as e:
        print(f"   ERROR in {function_name} during fcluster: {e}")
        traceback.print_exc()
        return
        
    # Map original community IDs to their meta-cluster ID
    comm_to_metacluster_map = {orig_id: meta_cluster_assignments[i] for i, orig_id in enumerate(original_community_ids)}

    # 2. Create a spatial lookup for pixels: (X,Y) -> community_id
    pixel_map = {}
    for _, row in pixel_results_df.iterrows():
        pixel_map[(int(row['X']), int(row['Y']))] = row['community']

    # 3. Process each meta-cluster
    unique_meta_clusters = sorted(np.unique(meta_cluster_assignments))
    plot_generated_for_any_metacluster = False

    for mc_id in unique_meta_clusters:
        member_original_communities = [orig_id for orig_id, assigned_mc_id in comm_to_metacluster_map.items() if assigned_mc_id == mc_id]
        
        if len(member_original_communities) < min_communities_in_metacluster:
            continue

        # print(f"    Processing Meta-cluster {mc_id} (members: {member_original_communities})...") # Verbose, enable if needed

        meta_cluster_pixels_df = pixel_results_df[pixel_results_df['community'].isin(member_original_communities)]
        if meta_cluster_pixels_df.empty:
            continue

        sorted_member_communities = sorted(list(set(member_original_communities)))
        adj_matrix = pd.DataFrame(0, index=sorted_member_communities, columns=sorted_member_communities, dtype=int)
        neighbor_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)] 

        for _, pixel_row in meta_cluster_pixels_df.iterrows():
            px, py = int(pixel_row['X']), int(pixel_row['Y'])
            p_community = pixel_row['community']
            for dx, dy in neighbor_offsets:
                nx, ny = px + dx, py + dy
                if (nx, ny) in pixel_map:
                    n_community = pixel_map[(nx, ny)]
                    if n_community in member_original_communities and n_community != p_community:
                        c1, c2 = min(p_community, n_community), max(p_community, n_community)
                        adj_matrix.loc[c1, c2] += 1 
        
        for r_idx, r_label in enumerate(adj_matrix.index):
            for c_idx, c_label in enumerate(adj_matrix.columns):
                if r_idx < c_idx:
                    adj_matrix.loc[c_label, r_label] = adj_matrix.loc[r_label, c_label]
                elif r_idx == c_idx:
                     adj_matrix.loc[r_label, c_label] = 0 

        if adj_matrix.sum().sum() == 0:
            # print(f"      No inter-community adjacencies found within meta-cluster {mc_id}.") # Verbose
            continue

        plt.close('all')
        n_member_comms = len(sorted_member_communities)
        fig_size = max(5, n_member_comms * 0.8) 
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        sns.heatmap(adj_matrix, annot=True, fmt='d', cmap='viridis', ax=ax, cbar=True,
                    linewidths=.5, linecolor='gray', square=True, annot_kws={"size": 8 if n_member_comms <= 10 else 6})
        
        ax.set_title(f"Meta-Cluster {mc_id} - Spatial Adjacencies\nROI: {roi_string} (Res: {resolution_param})", fontsize=10)
        ax.set_xlabel("Community ID", fontsize=9)
        ax.set_ylabel("Community ID", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=max(6, 8 - n_member_comms // 2))
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=max(6, 8 - n_member_comms // 2))

        output_filename = f"metacluster_{mc_id}_adjacencies_{roi_string}_res_{resolution_param.replace('.','_')}.png"
        current_output_path = os.path.join(output_dir, output_filename)
        try:
            plt.tight_layout()
            plt.savefig(current_output_path, dpi=plot_dpi, bbox_inches='tight')
            # print(f"      --- Saved adjacency map: {os.path.basename(current_output_path)}") # Verbose
            plot_generated_for_any_metacluster = True
        except Exception as e_save:
            print(f"      ERROR saving adjacency map for meta-cluster {mc_id}: {e_save}")
        finally:
            plt.close(fig)
    if plot_generated_for_any_metacluster:
        print(f"   --- Meta-cluster adjacency plots saved to directory: {output_dir}")

# New function for Spatial Community Assignment Map with Channel Intensity and Borders
def plot_spatial_community_channel_maps( # New name
    pixel_results_df: pd.DataFrame, 
    scaled_pixel_df: pd.DataFrame,
    roi_channels: List[str],
    output_dir: str, 
    base_filename: str,
    title_prefix: str,
    plot_dpi: int = 150,
    scatter_size: float = 1.0,
    community_border_color: str = 'cyan', 
    community_border_linewidth: float = 0.5 
):
    """
    Generates spatial maps for each specified ROI channel, showing channel intensity
    as a base layer with community boundaries overlaid.
    """
    print(f"   Generating spatial community channel maps: {base_filename} into {output_dir}")

    if pixel_results_df.empty or not all(col in pixel_results_df.columns for col in ['X', 'Y', 'community']):
        print("   Skipping spatial maps: pixel_results_df missing or lacks required columns (X, Y, community).")
        return
    if scaled_pixel_df.empty or not all(col in scaled_pixel_df.columns for col in ['X', 'Y']):
        print("   Skipping spatial maps: scaled_pixel_df missing or lacks required columns (X, Y).")
        return
    if not roi_channels:
        print("   Skipping spatial maps: No ROI channels specified.")
        return

    os.makedirs(output_dir, exist_ok=True)

    plot_data_base = scaled_pixel_df.copy()
    
    if 'community' not in plot_data_base.columns:
        if scaled_pixel_df.index.equals(pixel_results_df.index) and 'community' in pixel_results_df.columns:
            print("    Merging community data based on matching DataFrame indices.")
            plot_data_base['community'] = pixel_results_df['community']
        elif all(c in pixel_results_df.columns for c in ['X', 'Y', 'community']):
            print("    Warning: Merging community data based on 'X', 'Y' columns. Ensure these are reliable and precise keys.")
            try:
                temp_pixel_results_subset = pixel_results_df[['X', 'Y', 'community']].drop_duplicates(subset=['X', 'Y'])
                plot_data_base = pd.merge(plot_data_base, temp_pixel_results_subset, on=['X', 'Y'], how='left')
            except Exception as e_merge:
                print(f"   ERROR during X,Y merge for community data: {e_merge}. Cannot proceed with community overlays.")
                plot_data_base['community'] = np.nan 
        else:
            print("   ERROR: Cannot merge community data. Index mismatch and pixel_results_df does not contain X, Y, community for alternative merge.")
            return

    if 'community' not in plot_data_base.columns or plot_data_base['community'].isnull().all():
        print("   ERROR: Community data could not be associated with scaled pixel data or is all NaN. Skipping.")
        return

    x_min, x_max = plot_data_base['X'].min(), plot_data_base['X'].max()
    y_min, y_max = plot_data_base['Y'].min(), plot_data_base['Y'].max()

    for channel_name in roi_channels:
        if channel_name not in plot_data_base.columns:
            print(f"    Channel {channel_name} not found in scaled_pixel_df. Skipping plot for this channel.")
            continue

        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 10))

        channel_data = plot_data_base[channel_name].copy()
        valid_channel_data = channel_data.dropna()
        if valid_channel_data.empty:
            print(f"    Channel {channel_name} has no valid data points after NaN removal. Plotting as black.")
            normalized_channel_data = np.zeros(len(channel_data))
            channel_data.fillna(0, inplace=True)
        else:
            vmin_ch = np.percentile(valid_channel_data, 1)
            vmax_ch = np.percentile(valid_channel_data, 99)
            if vmin_ch == vmax_ch: 
                vmin_ch = valid_channel_data.min()
                vmax_ch = valid_channel_data.max()
            
            if vmin_ch == vmax_ch: 
                normalized_channel_data = channel_data / vmax_ch if vmax_ch > 0 else np.zeros_like(channel_data)
            else:
                normalized_channel_data = (channel_data - vmin_ch) / (vmax_ch - vmin_ch)
            
            normalized_channel_data = np.clip(normalized_channel_data, 0, 1)
        
        ax.scatter(
            plot_data_base['X'],
            plot_data_base['Y'],
            c=normalized_channel_data.fillna(0), 
            cmap='gray', 
            s=scatter_size,
            marker='.', 
            rasterized=True,
            alpha=1.0 
        )

        boundary_plot_df = plot_data_base.dropna(subset=['community', 'X', 'Y'])
        if boundary_plot_df.empty:
            print(f"    No pixels with community assignments found for channel {channel_name}. Skipping boundary overlay.")
        else:
            border_pixels_x = []
            border_pixels_y = []
            
            pixel_community_map = {}
            for _, row in boundary_plot_df.iterrows():
                try:
                    x_coord = int(round(float(row['X'])))
                    y_coord = int(round(float(row['Y'])))
                    pixel_community_map[(x_coord, y_coord)] = row['community']
                except ValueError:
                    print(f"    Warning: Could not convert X,Y to float for pixel {row.name if hasattr(row, 'name') else 'unknown'}. Skipping for boundary map.")
                    continue
            
            if pixel_community_map: 
                for _, row in boundary_plot_df.iterrows():
                    try:
                        x_coord_orig = float(row['X']) 
                        y_coord_orig = float(row['Y'])
                        x_coord_rounded = int(round(x_coord_orig))
                        y_coord_rounded = int(round(y_coord_orig))
                    except ValueError:
                        continue 

                    current_community = row['community']
                    is_border = False
                    
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        neighbor_x_rounded, neighbor_y_rounded = x_coord_rounded + dx, y_coord_rounded + dy
                        neighbor_community = pixel_community_map.get((neighbor_x_rounded, neighbor_y_rounded), None)
                        
                        if neighbor_community is None: 
                            is_border = True
                            break
                        if neighbor_community != current_community: 
                            is_border = True
                            break
                    if is_border:
                        border_pixels_x.append(x_coord_orig) 
                        border_pixels_y.append(y_coord_orig)
                
                if border_pixels_x:
                     ax.scatter(
                        border_pixels_x,
                        border_pixels_y,
                        edgecolors=community_border_color,
                        facecolors='none', 
                        s=scatter_size * 2.0, 
                        linewidths=community_border_linewidth,
                        marker='o', 
                        alpha=0.7, 
                        rasterized=True
                    )
                else:
                    print(f"    No border pixels identified for channel {channel_name}.")
            else:
                print(f"    Pixel community map for boundary detection is empty for channel {channel_name}.")

        ax.set_title(f"{title_prefix} - Channel: {channel_name.split('(')[0]}", fontsize=14)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.invert_yaxis() 
        ax.grid(False)

        channel_safe_name = "".join(c if c.isalnum() else "_" for c in channel_name)
        output_filename_full = os.path.join(output_dir, f"{base_filename}_channel_{channel_safe_name}.png")

        try:
            plt.tight_layout()
            plt.savefig(output_filename_full, dpi=plot_dpi, bbox_inches='tight')
            print(f"   --- Spatial map for channel {channel_name} saved to: {os.path.basename(output_filename_full)}")
        except Exception as e_save:
            print(f"   ERROR saving spatial map for channel {channel_name}: {e_save}")
            traceback.print_exc()
        finally:
            plt.close(fig)
