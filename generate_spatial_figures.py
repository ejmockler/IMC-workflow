"""
generate_spatial_figures.py

Standalone script to generate representative spatial figures for 4 IMC timepoints
(Sham, D1, D3, D7). Produces one PNG per timepoint with 3 panels:
  A. 3-channel composite (DNA/CD44/CD31)
  B. Cell type overlay with Voronoi tessellation
  C. CD44 expression heatmap

Usage:
    .venv/bin/python generate_spatial_figures.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "/Users/noot/Documents/IMC/data/241218_IMC_Alun"
ANNOT_DIR = "/Users/noot/Documents/IMC/results/biological_analysis/cell_type_annotations"
OUT_DIR = "/Users/noot/Documents/IMC/results/figures"

# ---------------------------------------------------------------------------
# ROI definitions: (timepoint_label, raw_txt_filename, parquet_filename)
# ---------------------------------------------------------------------------
ROIS = [
    (
        "Sham",
        "IMC_241218_Alun_ROI_Sam1_01_2.txt",
        "roi_IMC_241218_Alun_ROI_Sam1_01_2_cell_types.parquet",
    ),
    (
        "D1",
        "IMC_241218_Alun_ROI_D1_M1_01_9.txt",
        "roi_IMC_241218_Alun_ROI_D1_M1_01_9_cell_types.parquet",
    ),
    (
        "D3",
        "IMC_241218_Alun_ROI_D3_M1_01_15.txt",
        "roi_IMC_241218_Alun_ROI_D3_M1_01_15_cell_types.parquet",
    ),
    (
        "D7",
        "IMC_241218_Alun_ROI_D7_M1_01_21.txt",
        "roi_IMC_241218_Alun_ROI_D7_M1_01_21_cell_types.parquet",
    ),
]

# ---------------------------------------------------------------------------
# Cell-type colour palette
# ---------------------------------------------------------------------------
CELL_TYPE_COLORS = {
    "unassigned":                  "#E8E8E8",
    "activated_endothelial_cd140b": "#4682B4",
    "activated_endothelial_cd44":   "#00BFFF",
    "activated_fibroblast":         "#228B22",
    "activated_immune_cd140b":      "#FF4500",
    "m2_macrophage":                "#FFD700",
    # catch-all for any additional types
    "_other":                       "#9370DB",
}

# Human-readable labels for the legend
CELL_TYPE_LABELS = {
    "unassigned":                  "Unassigned",
    "activated_endothelial_cd140b": "Act. Endothelial (CD140b)",
    "activated_endothelial_cd44":   "Act. Endothelial (CD44)",
    "activated_fibroblast":         "Act. Fibroblast",
    "activated_immune_cd140b":      "Act. Immune (CD140b)",
    "m2_macrophage":                "M2 Macrophage",
    "_other":                       "Other",
}

# ---------------------------------------------------------------------------
# Helper: arcsinh normalised image channel
# ---------------------------------------------------------------------------
def arcsinh_norm(arr, cofactor=5, pct_lo=1, pct_hi=99):
    """Apply arcsinh compression then percentile normalisation to [0, 1]."""
    transformed = np.arcsinh(arr / cofactor)
    lo = np.percentile(transformed, pct_lo)
    hi = np.percentile(transformed, pct_hi)
    if hi == lo:
        return np.zeros_like(transformed, dtype=float)
    clipped = np.clip(transformed, lo, hi)
    return (clipped - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Helper: load raw IMC pixel data and build 2D channel images
# ---------------------------------------------------------------------------
def load_imc_images(txt_path):
    """
    Load an IMC .txt file and return a dict of 2D numpy arrays keyed by
    channel name, plus the grid height and width.
    """
    df = pd.read_csv(txt_path, sep="\t")

    x = df["X"].values.astype(int)
    y = df["Y"].values.astype(int)
    h = int(y.max()) + 1
    w = int(x.max()) + 1

    channels = {}
    for col in df.columns:
        if col in ("Start_push", "End_push", "Pushes_duration", "X", "Y", "Z"):
            continue
        img = np.zeros((h, w), dtype=np.float32)
        img[y, x] = df[col].values.astype(np.float32)
        channels[col] = img

    return channels, h, w


# ---------------------------------------------------------------------------
# Helper: build cell-type image via Voronoi (nearest-centroid) tessellation
# ---------------------------------------------------------------------------
def build_cell_type_image(annot_df, h, w):
    """
    Given superpixel centroid annotations, assign every pixel to the nearest
    centroid using a KD-tree, then colour by cell type.

    Returns:
        rgb_image  : (h, w, 3) float32 array in [0, 1]
        legend_entries : list of (label_str, hex_colour) for types actually present
    """
    cx = annot_df["x"].values
    cy = annot_df["y"].values
    cell_types = annot_df["cell_type"].values

    # Build KD-tree over superpixel centroids
    tree = cKDTree(np.column_stack([cx, cy]))

    # Build a flat array of all pixel coordinates
    ys, xs = np.mgrid[0:h, 0:w]
    pixel_coords = np.column_stack([xs.ravel(), ys.ravel()])

    # Query nearest centroid for every pixel
    _, idx = tree.query(pixel_coords, workers=-1)
    assigned_types = cell_types[idx].reshape(h, w)

    # Map cell types to RGB colours
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    present_types = []

    for ct in np.unique(assigned_types):
        if ct in CELL_TYPE_COLORS:
            hex_col = CELL_TYPE_COLORS[ct]
            label = CELL_TYPE_LABELS.get(ct, ct)
        else:
            hex_col = CELL_TYPE_COLORS["_other"]
            label = ct  # use raw name for unknown types

        mask = assigned_types == ct
        r, g, b = tuple(
            int(hex_col.lstrip("#")[i:i+2], 16) / 255.0 for i in (0, 2, 4)
        )
        rgb[mask, 0] = r
        rgb[mask, 1] = g
        rgb[mask, 2] = b

        if ct != "unassigned":
            present_types.append((label, hex_col))

    # Always add unassigned at the end of the legend if it exists
    if "unassigned" in np.unique(assigned_types):
        present_types.append(
            (CELL_TYPE_LABELS["unassigned"], CELL_TYPE_COLORS["unassigned"])
        )

    return rgb, present_types


# ---------------------------------------------------------------------------
# Helper: draw a scale bar on an existing Axes
# ---------------------------------------------------------------------------
def add_scale_bar(ax, pixel_size_um=1.0, bar_um=100, h=500, w=500,
                  color="white", fontsize=7):
    """
    Add a 100 µm scale bar to the bottom-right of ax.
    pixel_size_um: µm per pixel (default 1 for IMC).
    bar_um: physical length of bar in µm.
    """
    bar_px = bar_um / pixel_size_um

    margin_x = w * 0.04
    margin_y = h * 0.04

    x_start = w - margin_x - bar_px
    x_end = w - margin_x
    y_pos = h - margin_y

    ax.plot([x_start, x_end], [y_pos, y_pos],
            color=color, linewidth=2, solid_capstyle="butt",
            transform=ax.transData, clip_on=False)
    ax.text(
        (x_start + x_end) / 2, y_pos - h * 0.015,
        f"{bar_um} µm",
        color=color, fontsize=fontsize, ha="center", va="bottom",
        transform=ax.transData,
    )


# ---------------------------------------------------------------------------
# Core: generate one figure for a single timepoint
# ---------------------------------------------------------------------------
def generate_figure(timepoint, txt_path, parquet_path, out_path):
    print(f"  Loading raw IMC data: {os.path.basename(txt_path)}")
    channels, h, w = load_imc_images(txt_path)

    # --- Identify required channels (handle naming variations gracefully) ---
    def find_channel(channels, substring):
        """Return the first channel name containing substring (case-insensitive)."""
        for k in channels:
            if substring.lower() in k.lower():
                return k
        return None

    dna1_key = find_channel(channels, "DNA1")
    dna2_key = find_channel(channels, "DNA2")
    cd44_key = find_channel(channels, "CD44")
    cd31_key = find_channel(channels, "CD31")

    # DNA: average of DNA1 and DNA2 if both present
    if dna1_key and dna2_key:
        dna_raw = (channels[dna1_key] + channels[dna2_key]) / 2.0
    elif dna1_key:
        dna_raw = channels[dna1_key]
    elif dna2_key:
        dna_raw = channels[dna2_key]
    else:
        print("    WARNING: No DNA channel found; using zeros for DNA.")
        dna_raw = np.zeros((h, w), dtype=np.float32)

    if cd44_key is None:
        print("    WARNING: CD44 channel not found; using zeros.")
        cd44_raw = np.zeros((h, w), dtype=np.float32)
    else:
        cd44_raw = channels[cd44_key]

    if cd31_key is None:
        print("    WARNING: CD31 channel not found; using zeros.")
        cd31_raw = np.zeros((h, w), dtype=np.float32)
    else:
        cd31_raw = channels[cd31_key]

    # Normalise each channel
    blue_ch  = arcsinh_norm(dna_raw)    # DNA  -> blue
    green_ch = arcsinh_norm(cd44_raw)   # CD44 -> green
    red_ch   = arcsinh_norm(cd31_raw)   # CD31 -> red

    composite_rgb = np.stack([red_ch, green_ch, blue_ch], axis=-1)  # (H, W, 3)

    # --- Load cell type annotations ---
    print(f"  Loading annotations: {os.path.basename(parquet_path)}")
    if os.path.exists(parquet_path):
        annot_df = pd.read_parquet(parquet_path)
    else:
        print(f"    WARNING: Parquet file not found; skipping cell type panel.")
        annot_df = None

    if annot_df is not None:
        ct_rgb, legend_entries = build_cell_type_image(annot_df, h, w)
    else:
        ct_rgb = np.ones((h, w, 3), dtype=np.float32) * 0.9
        legend_entries = []

    # --- Figure layout ---
    # 180 mm wide = 7.087 inches; panels are square (500x500 pixels)
    # We need space for 3 panels + gaps + legend; use 3.5 col inches per panel
    aspect = h / w
    panel_w_in = 2.2   # inches per panel
    panel_h_in = panel_w_in * aspect
    left_margin = 0.45
    right_margin = 1.5  # extra room for legend
    top_margin = 0.55
    bot_margin = 0.45
    h_gap = 0.35        # horizontal gap between panels
    cb_w = 0.12         # colorbar width

    total_w = left_margin + 3 * panel_w_in + 2 * h_gap + right_margin
    total_h = top_margin + panel_h_in + bot_margin

    fig = plt.figure(figsize=(total_w, total_h), dpi=300)
    fig.patch.set_facecolor("white")

    # Compute axes rectangles in figure-fraction coordinates
    def rect(left_in, bottom_in, width_in, height_in):
        return [
            left_in / total_w,
            bottom_in / total_h,
            width_in / total_w,
            height_in / total_h,
        ]

    panel_bottom_in = bot_margin

    ax_a = fig.add_axes(rect(left_margin, panel_bottom_in, panel_w_in, panel_h_in))
    ax_b = fig.add_axes(rect(left_margin + panel_w_in + h_gap, panel_bottom_in, panel_w_in, panel_h_in))
    ax_c = fig.add_axes(rect(left_margin + 2 * (panel_w_in + h_gap), panel_bottom_in, panel_w_in, panel_h_in))

    # Colorbar axis for Panel C (tight, to the right)
    cb_left = left_margin + 2 * (panel_w_in + h_gap) + panel_w_in + 0.05
    ax_cb = fig.add_axes(rect(cb_left, panel_bottom_in, cb_w, panel_h_in))

    # ---- Panel A: 3-channel composite ----
    ax_a.imshow(composite_rgb, origin="upper", interpolation="nearest", aspect="equal")
    ax_a.set_xlim(0, w)
    ax_a.set_ylim(h, 0)
    ax_a.axis("off")
    ax_a.set_title("A", fontsize=9, fontweight="bold", loc="left", pad=3)

    # Channel colour legend for Panel A (bottom of panel)
    legend_patches_a = [
        mpatches.Patch(facecolor=(0.0, 0.0, 1.0), label="DNA (nuclei)"),
        mpatches.Patch(facecolor=(0.0, 1.0, 0.0), label="CD44 (injury)"),
        mpatches.Patch(facecolor=(1.0, 0.0, 0.0), label="CD31 (endothelial)"),
    ]
    ax_a.legend(
        handles=legend_patches_a,
        fontsize=5.5,
        loc="lower left",
        framealpha=0.7,
        frameon=True,
        facecolor="k",
        labelcolor="w",
        handlelength=1.0,
        borderpad=0.4,
        handletextpad=0.5,
    )
    add_scale_bar(ax_a, pixel_size_um=1.0, bar_um=100, h=h, w=w, color="white", fontsize=6)

    # ---- Panel B: Cell type overlay ----
    ax_b.imshow(ct_rgb, origin="upper", interpolation="nearest", aspect="equal")
    ax_b.set_xlim(0, w)
    ax_b.set_ylim(h, 0)
    ax_b.axis("off")
    ax_b.set_title("B", fontsize=9, fontweight="bold", loc="left", pad=3)
    add_scale_bar(ax_b, pixel_size_um=1.0, bar_um=100, h=h, w=w, color="#333333", fontsize=6)

    # Legend for Panel B — placed to the right of Panel C + colorbar
    if legend_entries:
        legend_patches_b = []
        # Sort: non-unassigned first, then unassigned at the end
        named = [(lbl, col) for lbl, col in legend_entries if lbl != CELL_TYPE_LABELS["unassigned"]]
        unass = [(lbl, col) for lbl, col in legend_entries if lbl == CELL_TYPE_LABELS["unassigned"]]
        sorted_entries = named + unass

        for lbl, col in sorted_entries:
            r, g, b = tuple(int(col.lstrip("#")[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            legend_patches_b.append(
                mpatches.Patch(facecolor=(r, g, b), edgecolor="#555555",
                               linewidth=0.4, label=lbl)
            )

        legend_x = (cb_left + cb_w + 0.05) / total_w
        legend_y = (panel_bottom_in + panel_h_in / 2.0) / total_h
        ax_b.legend(
            handles=legend_patches_b,
            fontsize=5.5,
            loc="center left",
            bbox_to_anchor=(legend_x * total_w / panel_w_in + 2 * (panel_w_in + h_gap) / panel_w_in,
                            0.5),
            framealpha=0.85,
            frameon=True,
            edgecolor="#888888",
            handlelength=1.0,
            borderpad=0.5,
            handletextpad=0.5,
            title="Cell type",
            title_fontsize=6,
        )

    # ---- Panel C: CD44 heatmap ----
    cd44_display = arcsinh_norm(cd44_raw)
    im_c = ax_c.imshow(cd44_display, origin="upper", cmap="hot",
                       vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    ax_c.set_xlim(0, w)
    ax_c.set_ylim(h, 0)
    ax_c.axis("off")
    ax_c.set_title("C", fontsize=9, fontweight="bold", loc="left", pad=3)
    add_scale_bar(ax_c, pixel_size_um=1.0, bar_um=100, h=h, w=w, color="white", fontsize=6)

    # Colorbar for Panel C
    norm_c = Normalize(vmin=0, vmax=1)
    cb = ColorbarBase(ax_cb, cmap="hot", norm=norm_c, orientation="vertical")
    cb.set_label("CD44\narcsinh norm.", fontsize=6, labelpad=3)
    cb.ax.tick_params(labelsize=5.5)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])

    # ---- Overall title ----
    fig.text(
        0.5, 1.0 - (top_margin * 0.3) / total_h,
        timepoint,
        ha="center", va="top",
        fontsize=11, fontweight="bold",
    )

    # ---- Save ----
    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = os.path.join(OUT_DIR, f"spatial_overview_{timepoint}.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Output directory: {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    for timepoint, txt_name, parquet_name in ROIS:
        print(f"\nProcessing timepoint: {timepoint}")
        txt_path = os.path.join(DATA_DIR, txt_name)
        parquet_path = os.path.join(ANNOT_DIR, parquet_name)

        if not os.path.exists(txt_path):
            print(f"  ERROR: Raw data file not found: {txt_path}")
            continue

        generate_figure(timepoint, txt_path, parquet_path,
                        out_path=os.path.join(OUT_DIR, f"spatial_overview_{timepoint}.png"))

    print("\nDone. Generated figures:")
    for tp, _, _ in ROIS:
        p = os.path.join(OUT_DIR, f"spatial_overview_{tp}.png")
        if os.path.exists(p):
            size_kb = os.path.getsize(p) / 1024
            print(f"  {p}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
