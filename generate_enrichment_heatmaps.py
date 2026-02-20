"""
Standalone script: Temporal Neighborhood Enrichment Heatmaps
Produces a publication-quality figure with a 2x2 heatmap grid (one per timepoint)
plus a temporal trajectory line plot for top cell type pairs.

Usage:
    .venv/bin/python generate_enrichment_heatmaps.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = (
    "results/biological_analysis/spatial_neighborhoods/"
    "temporal_neighborhood_enrichments.csv"
)
OUT_DIR = "results/figures"
OUT_FILE = os.path.join(OUT_DIR, "neighborhood_enrichment_temporal.png")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Display name mapping
# ---------------------------------------------------------------------------
DISPLAY_NAMES = {
    "activated_endothelial_cd140b": "Endothelial\n(CD140b+)",
    "activated_endothelial_cd44":   "Endothelial\n(CD44+)",
    "activated_fibroblast":         "Fibroblast",
    "activated_immune_cd140b":      "Immune\n(CD140b+)",
    "activated_immune_cd44":        "Immune\n(CD44+)",
    "m2_macrophage":                "M2 Macrophage",
    "neutrophil":                   "Neutrophil",
    "resting_endothelial":          "Resting\nEndothelial",
}

TIMEPOINT_ORDER = ["Sham", "D1", "D3", "D7"]
CLIM = (-2.0, 2.0)   # symmetric clip for log2 enrichment

# ---------------------------------------------------------------------------
# 1. Load and filter
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Exclude 'unassigned'
df = df[
    (df["focal_cell_type"] != "unassigned") &
    (df["neighbor_cell_type"] != "unassigned")
].copy()

# Only keep cell types with n_focal_cells >= 10 across all appearances
min_focal = df.groupby("focal_cell_type")["n_focal_cells"].min()
valid_focal = min_focal[min_focal >= 10].index
df = df[
    df["focal_cell_type"].isin(valid_focal) &
    df["neighbor_cell_type"].isin(valid_focal)
].copy()

# Apply display names; fall back to original name if not in map
def display(name):
    return DISPLAY_NAMES.get(name, name.replace("_", "\n"))

df["focal_display"]    = df["focal_cell_type"].map(display)
df["neighbor_display"] = df["neighbor_cell_type"].map(display)

# Clip log2_enrichment for visualisation
df["log2_clipped"] = df["log2_enrichment"].clip(lower=CLIM[0], upper=CLIM[1])

# ---------------------------------------------------------------------------
# 2. Determine consistent cell-type ordering (by average log2|enrichment|)
# ---------------------------------------------------------------------------
mean_abs = (
    df.groupby("focal_cell_type")["log2_enrichment"]
    .apply(lambda s: s.abs().mean())
    .sort_values(ascending=False)
)
# Keep the order of raw names and map to display
raw_order = [ct for ct in mean_abs.index if ct in valid_focal]
disp_order = [display(ct) for ct in raw_order]

# ---------------------------------------------------------------------------
# 3. Build per-timepoint matrices
# ---------------------------------------------------------------------------
def build_matrix(tp):
    sub = df[df["timepoint"] == tp]
    mat = pd.DataFrame(index=disp_order, columns=disp_order, dtype=float)
    sig = pd.DataFrame(index=disp_order, columns=disp_order, dtype=bool)
    sig[:] = False
    for _, row in sub.iterrows():
        f = row["focal_display"]
        n = row["neighbor_display"]
        if f in disp_order and n in disp_order:
            mat.loc[f, n] = row["log2_clipped"]
            sig.loc[f, n] = row["fraction_significant_fdr"] > 0.5
    return mat.astype(float), sig

matrices = {}
sig_matrices = {}
for tp in TIMEPOINT_ORDER:
    matrices[tp], sig_matrices[tp] = build_matrix(tp)

# ---------------------------------------------------------------------------
# 4. Temporal trajectory: top pairs by mean absolute log2 enrichment
# ---------------------------------------------------------------------------
pair_means = (
    df.groupby(["focal_cell_type", "neighbor_cell_type"])["log2_enrichment"]
    .apply(lambda s: s.abs().mean())
    .sort_values(ascending=False)
)
# Exclude trivial self-pairs from the "interesting" ranking
pair_means_cross = pair_means[
    pair_means.index.get_level_values(0) != pair_means.index.get_level_values(1)
]
TOP_N = 5
top_pairs = pair_means_cross.head(TOP_N).index.tolist()

# ---------------------------------------------------------------------------
# 5. Figure layout
# ---------------------------------------------------------------------------
FIG_W_IN = 180 / 25.4   # ~7.1 inches
FIG_H_IN = 250 / 25.4   # ~9.8 inches

fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)

# GridSpec: 2 rows for the heatmap grid, 1 row for trajectory
# top section: 2 cols x 2 rows; bottom section: 1 col full width
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

gs_outer = GridSpec(
    nrows=2,
    ncols=1,
    figure=fig,
    height_ratios=[2.8, 1],
    hspace=0.45,
)

gs_top = GridSpecFromSubplotSpec(
    nrows=2, ncols=2,
    subplot_spec=gs_outer[0],
    hspace=0.55,
    wspace=0.45,
)

ax_traj = fig.add_subplot(gs_outer[1])

heatmap_axes = {}
panel_positions = {
    "Sham": (0, 0), "D1": (0, 1),
    "D3":   (1, 0), "D7": (1, 1),
}
for tp, (row, col) in panel_positions.items():
    heatmap_axes[tp] = fig.add_subplot(gs_top[row, col])

# Shared color normalisation
norm = TwoSlopeNorm(vmin=CLIM[0], vcenter=0, vmax=CLIM[1])
cmap = plt.cm.RdBu_r

n_ct = len(disp_order)

# ---------------------------------------------------------------------------
# 6. Draw heatmaps
# ---------------------------------------------------------------------------
for tp in TIMEPOINT_ORDER:
    ax = heatmap_axes[tp]
    mat = matrices[tp]
    sig = sig_matrices[tp]
    data = mat.values

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # Cell annotations and significance dots
    for i in range(n_ct):
        for j in range(n_ct):
            val = data[i, j]
            if np.isnan(val):
                # White out NaN cells
                ax.add_patch(
                    mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        color="white", zorder=2
                    )
                )
                continue
            # Text annotation
            text_col = "black" if abs(val) < 1.3 else "white"
            ax.text(
                j, i, f"{val:.1f}",
                ha="center", va="center",
                fontsize=5.5, color=text_col, zorder=3,
            )
            # Significance dot
            if sig.iloc[i, j]:
                ax.plot(
                    j, i - 0.30,
                    marker="o", markersize=2.5,
                    color="black" if abs(val) < 1.3 else "white",
                    zorder=4,
                )

    # Thick border on diagonal cells
    for k in range(n_ct):
        rect = mpatches.Rectangle(
            (k - 0.5, k - 0.5), 1, 1,
            fill=False,
            edgecolor="black",
            linewidth=1.8,
            zorder=5,
        )
        ax.add_patch(rect)

    # Axis formatting
    ax.set_xticks(range(n_ct))
    ax.set_yticks(range(n_ct))
    ax.set_xticklabels(disp_order, fontsize=5.5, rotation=45, ha="right")
    ax.set_yticklabels(disp_order, fontsize=5.5)
    ax.set_title(tp, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(length=2)

    # Light grid lines between cells
    for k in range(n_ct + 1):
        ax.axhline(k - 0.5, color="lightgrey", linewidth=0.3, zorder=1)
        ax.axvline(k - 0.5, color="lightgrey", linewidth=0.3, zorder=1)

# Shared colorbar to the right of the top panels
cbar_ax = fig.add_axes([0.93, 0.38, 0.015, 0.52])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("log\u2082(enrichment)", fontsize=7)
cbar.ax.tick_params(labelsize=6)
cbar.set_ticks([-2, -1, 0, 1, 2])

# ---------------------------------------------------------------------------
# 7. Temporal trajectory subplot
# ---------------------------------------------------------------------------
tp_numeric = {tp: i for i, tp in enumerate(TIMEPOINT_ORDER)}

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for idx, (focal_ct, neighbor_ct) in enumerate(top_pairs):
    focal_d   = display(focal_ct)
    neighbor_d = display(neighbor_ct)
    ys = []
    xs = []
    for tp in TIMEPOINT_ORDER:
        sub = df[
            (df["focal_cell_type"]    == focal_ct) &
            (df["neighbor_cell_type"] == neighbor_ct) &
            (df["timepoint"]          == tp)
        ]
        if len(sub) == 1:
            ys.append(float(sub["log2_enrichment"].iloc[0]))
            xs.append(tp_numeric[tp])
        else:
            ys.append(np.nan)
            xs.append(tp_numeric[tp])

    label = f"{focal_d.replace(chr(10),' ')} → {neighbor_d.replace(chr(10),' ')}"
    ax_traj.plot(
        xs, ys,
        marker="o", markersize=4,
        linewidth=1.5,
        color=color_cycle[idx % len(color_cycle)],
        label=label,
    )

ax_traj.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=0)
ax_traj.set_xticks(list(tp_numeric.values()))
ax_traj.set_xticklabels(TIMEPOINT_ORDER, fontsize=8)
ax_traj.set_ylabel("log\u2082(enrichment)", fontsize=8)
ax_traj.set_xlabel("Timepoint", fontsize=8)
ax_traj.set_title("Temporal Trajectory — Top Cell Type Pairs", fontsize=9, fontweight="bold")
ax_traj.legend(
    fontsize=5.5,
    frameon=True,
    loc="upper right",
    ncol=1,
    bbox_to_anchor=(1.0, 1.0),
)
ax_traj.tick_params(labelsize=7)
ax_traj.spines[["top", "right"]].set_visible(False)

# ---------------------------------------------------------------------------
# 8. Figure-level title and caption
# ---------------------------------------------------------------------------
fig.suptitle(
    "Temporal Neighborhood Enrichment Analysis",
    fontsize=12,
    fontweight="bold",
    y=0.99,
)
fig.text(
    0.5, 0.005,
    "Enrichment = observed/expected neighbor proportion. "
    "Self-clustering (diagonal, thick border) validates cell type spatial coherence. "
    "Dots indicate FDR-corrected fraction significant > 0.5.",
    ha="center",
    fontsize=6.5,
    style="italic",
    wrap=True,
)

# ---------------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------------
fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Figure saved to: {OUT_FILE}")
plt.close(fig)
