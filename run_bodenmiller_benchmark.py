#!/usr/bin/env python3
"""
Steinbock benchmark: concordance between Steinbock cell-level intensities
and our raw pixel-level summary statistics for the Bodenmiller Patient1 dataset.

SCOPE
-----
Our pipeline (SLIC superpixels) has not been run on the Bodenmiller data in
this repository. We therefore compare:

    Steinbock output     : per-cell mean channel intensity (DeepCell segmentation)
    Raw pixel data       : per-pixel mean channel intensity from .txt acquisition files

These two representations are NOT equivalent:
  - Steinbock aggregates pixels inside segmented cell boundaries (DeepCell deep-learning masks)
  - Raw pixel means pool ALL pixels including inter-cell space
  - SLIC superpixels (our method) segment spatial tiles irrespective of cell boundaries

Goal: pipeline validation — do the raw acquisition data and Steinbock's cell-level
summaries agree on RELATIVE channel expression levels? Channel rank correlation
and distribution shape are meaningful; absolute intensity values are not directly
comparable.
"""

import os
import sys
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = "/Users/noot/Documents/IMC"

STEINBOCK_WORKDIR = os.path.join(
    BASE_DIR,
    "benchmarks/data/bodenmiller_example/steinbock_outputs/Patient1/steinbock_workdir",
)
STEINBOCK_INTENSITIES_DIR = os.path.join(STEINBOCK_WORKDIR, "intensities")
STEINBOCK_PANEL_CSV       = os.path.join(STEINBOCK_WORKDIR, "panel.csv")

RAW_DATA_DIR = os.path.join(
    BASE_DIR,
    "benchmarks/data/bodenmiller_example/Patient1",
)

OUTPUT_DIR_BENCH = os.path.join(BASE_DIR, "results/benchmark")
OUTPUT_DIR_FIGS  = os.path.join(BASE_DIR, "results/figures")
OUTPUT_TABLE     = os.path.join(OUTPUT_DIR_BENCH, "bodenmiller_concordance.csv")
OUTPUT_FIGURE    = os.path.join(OUTPUT_DIR_FIGS,  "benchmark_concordance.png")

os.makedirs(OUTPUT_DIR_BENCH, exist_ok=True)
os.makedirs(OUTPUT_DIR_FIGS,  exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: map raw .txt column names to panel channel names
# ---------------------------------------------------------------------------
def strip_element_suffix(col: str) -> str:
    """
    Raw .txt columns have the form  'CD3_1841((2941))Sm152(Sm152Di)'.
    Steinbock intensity CSVs use    'CD3_1841((2941))Sm152'.
    Strip the trailing '(XxxNNNDi)' mass-spec suffix if present so names align.
    """
    return re.sub(r"\([A-Za-z]+\d+Di\)$", "", col).strip()


def clean_channel_label(name: str) -> str:
    """
    Return a short human-readable marker name from a Steinbock channel string.
    E.g. 'CD3_1841((2941))Sm152' -> 'CD3'
         'HLA-DR_1849((2953))Nd143' -> 'HLA-DR'
         'DNA1' -> 'DNA1'
    """
    # Already short names (DNA1, DNA2, 80ArAr …)
    if not re.search(r"_\d+\(\(", name):
        return name
    # Long form: take everything before the first underscore-number block
    match = re.match(r"^([A-Za-z0-9\-]+)_\d+", name)
    if match:
        return match.group(1)
    return name.split("_")[0]


# ---------------------------------------------------------------------------
# 1. Load Steinbock intensities
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading Steinbock intensity data")
print("=" * 60)

if not os.path.isdir(STEINBOCK_INTENSITIES_DIR):
    print(f"ERROR: Steinbock intensities directory not found:\n  {STEINBOCK_INTENSITIES_DIR}")
    sys.exit(1)

intensity_files = sorted(
    f for f in os.listdir(STEINBOCK_INTENSITIES_DIR)
    if f.endswith(".csv")
)
if not intensity_files:
    print(f"ERROR: No CSV files found in {STEINBOCK_INTENSITIES_DIR}")
    sys.exit(1)

print(f"Found {len(intensity_files)} intensity file(s): {intensity_files}")

steinbock_dfs = []
for fname in intensity_files:
    fpath = os.path.join(STEINBOCK_INTENSITIES_DIR, fname)
    tmp = pd.read_csv(fpath, index_col=0)
    tmp["__source_file__"] = fname
    steinbock_dfs.append(tmp)

steinbock = pd.concat(steinbock_dfs, axis=0, ignore_index=True)
print(f"Steinbock combined: {len(steinbock)} cells × {steinbock.shape[1]-1} channels")

# Identify channel columns (exclude the bookkeeping column)
steinbock_channel_cols = [c for c in steinbock.columns if c != "__source_file__"]

# Load panel for reference
print(f"\nLoading panel: {STEINBOCK_PANEL_CSV}")
panel = pd.read_csv(STEINBOCK_PANEL_CSV)
print(f"  Panel has {len(panel)} channels")
# Map: panel 'name' -> element abbreviation (channel col in intensities CSV)
# The intensity CSV uses the 'name' column from panel.csv as column headers
# Actually, looking at panel structure: channel (e.g. Sm152), name (e.g. CD3_1841...)
# Intensity CSV columns ARE the 'name' values from panel.csv.
panel_name_to_channel = dict(zip(panel["name"], panel["channel"]))
print(f"  Example channel mapping: {list(panel_name_to_channel.items())[:3]}")


# ---------------------------------------------------------------------------
# 2. Load raw .txt pixel data
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Loading raw acquisition .txt data")
print("=" * 60)

txt_files = sorted(
    f for f in os.listdir(RAW_DATA_DIR)
    if f.endswith(".txt")
)
if not txt_files:
    print(f"ERROR: No .txt files found in {RAW_DATA_DIR}")
    sys.exit(1)

print(f"Found {len(txt_files)} .txt file(s): {txt_files}")

raw_dfs = []
for fname in txt_files:
    fpath = os.path.join(RAW_DATA_DIR, fname)
    tmp = pd.read_csv(fpath, sep="\t")
    tmp["__source_file__"] = fname
    raw_dfs.append(tmp)

raw = pd.concat(raw_dfs, axis=0, ignore_index=True)
print(f"Raw data combined: {len(raw)} pixels × {raw.shape[1]-1} columns")

# Non-channel columns to drop
non_channel_raw = {
    "Start_push", "End_push", "Pushes_duration", "X", "Y", "Z",
    "__source_file__",
}
raw_channel_cols = [c for c in raw.columns if c not in non_channel_raw]
print(f"  Raw channel columns ({len(raw_channel_cols)}): first 5 = {raw_channel_cols[:5]}")

# Normalise raw column names: strip '(XxxNNNDi)' suffix to match Steinbock
raw_clean_names = {col: strip_element_suffix(col) for col in raw_channel_cols}
raw_renamed = raw[raw_channel_cols].rename(columns=raw_clean_names)
print(f"  After stripping Di suffix, e.g. first 5: {list(raw_renamed.columns[:5])}")


# ---------------------------------------------------------------------------
# 3. Align channels between Steinbock and raw data
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Aligning channels between Steinbock and raw data")
print("=" * 60)

steinbock_set = set(steinbock_channel_cols)
raw_set       = set(raw_renamed.columns)
overlap       = steinbock_set & raw_set

print(f"  Steinbock channels: {len(steinbock_set)}")
print(f"  Raw channels (cleaned): {len(raw_set)}")
print(f"  Exact overlap: {len(overlap)}")

if len(overlap) == 0:
    # Try partial matching on shortened names
    print("  No exact overlap — trying short-label fallback matching.")
    steinbock_short = {clean_channel_label(c): c for c in steinbock_channel_cols}
    raw_short       = {clean_channel_label(c): c for c in raw_renamed.columns}
    short_overlap   = set(steinbock_short.keys()) & set(raw_short.keys())
    print(f"  Short-label overlap: {len(short_overlap)} channels")
    # Build aligned frames
    overlap_channels_steinbock = [steinbock_short[s] for s in sorted(short_overlap)]
    overlap_channels_raw       = [raw_short[s]       for s in sorted(short_overlap)]
    short_labels               = sorted(short_overlap)
else:
    overlap_channels_steinbock = sorted(overlap)
    overlap_channels_raw       = sorted(overlap)
    short_labels = [clean_channel_label(c) for c in overlap_channels_steinbock]

n_aligned = len(short_labels)
print(f"  Will compare {n_aligned} aligned channels.")
if n_aligned == 0:
    print("ERROR: Cannot align any channels between Steinbock and raw data.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 4. Compute per-channel summary statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Computing per-channel summary statistics")
print("=" * 60)

steinbock_sub = steinbock[overlap_channels_steinbock].copy()
steinbock_sub.columns = short_labels
raw_sub = raw_renamed[overlap_channels_raw].copy()
raw_sub.columns = short_labels

# Means
steinbock_means = steinbock_sub.mean(axis=0)
raw_means       = raw_sub.mean(axis=0)
steinbock_stds  = steinbock_sub.std(axis=0)
raw_stds        = raw_sub.std(axis=0)

# Normalise means to [0, 1] within each source for rank/shape comparison.
# (Absolute intensity is NOT comparable: cell-level aggregation vs. pixel-level
# pooling, plus potential pre-processing differences in the steinbock pipeline.)
def minmax_norm(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0:
        return s * 0.0
    return (s - s.min()) / rng

steinbock_norm = minmax_norm(steinbock_means)
raw_norm       = minmax_norm(raw_means)

# Channel rank correlation
spearman_r, spearman_p = stats.spearmanr(
    steinbock_means.values, raw_means.values
)
pearson_r,  pearson_p  = stats.pearsonr(
    steinbock_norm.values, raw_norm.values
)
print(f"  Spearman rank corr (raw means):        r = {spearman_r:.4f}, p = {spearman_p:.4g}")
print(f"  Pearson corr (min-max normalised means): r = {pearson_r:.4f}, p = {pearson_p:.4g}")

# KS test per channel
ks_stats    = []
ks_pvalues  = []
for ch in short_labels:
    s_vals = steinbock_sub[ch].dropna().values
    r_vals = raw_sub[ch].dropna().values
    if len(s_vals) < 3 or len(r_vals) < 3:
        ks_stats.append(np.nan)
        ks_pvalues.append(np.nan)
    else:
        # KS test on raw (not normalised) values — documents distributional mismatch
        ks, pv = stats.ks_2samp(s_vals, r_vals)
        ks_stats.append(ks)
        ks_pvalues.append(pv)

# Assemble concordance table
concordance = pd.DataFrame({
    "channel":           short_labels,
    "steinbock_mean":    steinbock_means.values,
    "steinbock_std":     steinbock_stds.values,
    "raw_pixel_mean":    raw_means.values,
    "raw_pixel_std":     raw_stds.values,
    "steinbock_norm":    steinbock_norm.values,
    "raw_pixel_norm":    raw_norm.values,
    "ks_statistic":      ks_stats,
    "ks_pvalue":         ks_pvalues,
})
concordance["rank_steinbock"] = concordance["steinbock_mean"].rank(ascending=False).astype(int)
concordance["rank_raw"]       = concordance["raw_pixel_mean"].rank(ascending=False).astype(int)
concordance["rank_diff"]      = (
    concordance["rank_steinbock"] - concordance["rank_raw"]
).abs()

concordance = concordance.sort_values("steinbock_mean", ascending=False).reset_index(drop=True)
concordance["spearman_overall"] = spearman_r
concordance["pearson_normed"]   = pearson_r

print(f"\n  Per-channel KS statistics (first 10 channels by Steinbock mean):")
print(concordance[["channel","steinbock_mean","raw_pixel_mean",
                    "ks_statistic","ks_pvalue","rank_diff"]].head(10).to_string(index=False))

concordance.to_csv(OUTPUT_TABLE, index=False)
print(f"\nSaved concordance table to: {OUTPUT_TABLE}")


# ---------------------------------------------------------------------------
# 5. Figure: scatter plot of Steinbock vs. raw means (one point per channel)
# ---------------------------------------------------------------------------
print("\nGenerating concordance scatter plot...")

n_channels = len(concordance)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

# ---- Panel A: raw means scatter ----------------------------------------
ax = axes[0]
ax.set_facecolor("#f9f9f9")

# Highlight DNA channels
dna_mask  = concordance["channel"].str.startswith("DNA")
prot_mask = ~dna_mask

sc_p = ax.scatter(
    concordance.loc[prot_mask, "raw_pixel_mean"],
    concordance.loc[prot_mask, "steinbock_mean"],
    c="#1f77b4", alpha=0.75, s=55, zorder=3, label="Protein channels",
)
sc_d = ax.scatter(
    concordance.loc[dna_mask, "raw_pixel_mean"],
    concordance.loc[dna_mask, "steinbock_mean"],
    c="#d62728", alpha=0.9, s=80, marker="D", zorder=4, label="DNA channels",
)

# Label top channels by steinbock_mean
top_n = min(8, n_channels)
top_ch = concordance.nlargest(top_n, "steinbock_mean")
for _, row in top_ch.iterrows():
    ax.annotate(
        row["channel"],
        xy=(row["raw_pixel_mean"], row["steinbock_mean"]),
        xytext=(5, 3), textcoords="offset points",
        fontsize=7, color="black",
    )

# Identity line (if both were fully equivalent, points would lie here)
all_x = concordance["raw_pixel_mean"]
all_y = concordance["steinbock_mean"]
mn = min(all_x.min(), all_y.min())
mx = max(all_x.max(), all_y.max())
ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.8, alpha=0.4,
        label="Identity line (raw = Steinbock)")

ax.set_xlabel("Raw pixel mean intensity", fontsize=10)
ax.set_ylabel("Steinbock cell-level mean intensity", fontsize=10)
ax.set_title(
    f"Raw vs. Steinbock channel means\n"
    f"Spearman r = {spearman_r:.3f}  (n={n_channels} channels)",
    fontsize=10, fontweight="bold",
)
ax.legend(fontsize=8, framealpha=0.9)

# ---- Panel B: normalised means + rank annotations ----------------------
ax2 = axes[1]
ax2.set_facecolor("#f9f9f9")

ax2.scatter(
    concordance.loc[prot_mask, "raw_pixel_norm"],
    concordance.loc[prot_mask, "steinbock_norm"],
    c="#1f77b4", alpha=0.75, s=55, zorder=3, label="Protein channels",
)
ax2.scatter(
    concordance.loc[dna_mask, "raw_pixel_norm"],
    concordance.loc[dna_mask, "steinbock_norm"],
    c="#d62728", alpha=0.9, s=80, marker="D", zorder=4, label="DNA channels",
)

# Colour-code by rank discordance
cmap = matplotlib.colormaps.get_cmap("RdYlGn_r")
max_rdiff = concordance["rank_diff"].max()
for _, row in concordance.iterrows():
    shade = cmap(row["rank_diff"] / max(max_rdiff, 1))
    ax2.scatter(
        row["raw_pixel_norm"], row["steinbock_norm"],
        color=shade, alpha=0.8, s=55, zorder=3,
    )

ax2.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4,
         label="Identity")

ax2.set_xlabel("Raw pixel mean (min-max normalised)", fontsize=10)
ax2.set_ylabel("Steinbock cell mean (min-max normalised)", fontsize=10)
ax2.set_title(
    f"Min-max normalised means\n"
    f"Pearson r = {pearson_r:.3f}  (colour = rank discordance)",
    fontsize=10, fontweight="bold",
)

# Colourbar for rank discordance
sm = plt.cm.ScalarMappable(
    cmap="RdYlGn_r",
    norm=plt.Normalize(vmin=0, vmax=int(max_rdiff)),
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax2, shrink=0.7, pad=0.02)
cbar.set_label("Rank discordance |Δrank|", fontsize=8)

ax2.legend(fontsize=8, framealpha=0.9)

# Global annotation
fig.text(
    0.5, 0.01,
    "CAUTION: Raw pixel means pool ALL pixels (including inter-cell space). "
    "Steinbock means are aggregated inside DeepCell cell boundaries.\n"
    "These are NOT equivalent representations. "
    "Rank correlation reflects whether relative channel expression levels agree, "
    "not absolute calibration.",
    ha="center", va="bottom", fontsize=7.5, color="gray", style="italic",
    wrap=True,
)

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved figure to: {OUTPUT_FIGURE}")


# ---------------------------------------------------------------------------
# 6. Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)
print(f"Dataset:                Bodenmiller Patient1")
print(f"Steinbock cells:        {len(steinbock)}")
print(f"Raw pixels:             {len(raw)}")
print(f"Aligned channels:       {n_aligned}")
print()
print(f"Channel rank correlation (Spearman):    r = {spearman_r:.4f}  p = {spearman_p:.4g}")
print(f"Channel correlation (normalised means): r = {pearson_r:.4f}  p = {pearson_p:.4g}")
print()

# Summarise KS results
ks_valid = concordance["ks_statistic"].dropna()
n_sig_ks = (concordance["ks_pvalue"].dropna() < 0.05).sum()
print(f"KS test (distribution shape):")
print(f"  Channels tested:                    {len(ks_valid)}")
print(f"  Channels with significant KS p<0.05: {n_sig_ks}/{len(ks_valid)}")
print(f"  Median KS statistic:                {ks_valid.median():.3f}")
print()
print("Interpretation:")
print("  Spearman rank correlation measures whether channels that are")
print("  relatively high in raw data are also relatively high in Steinbock.")
print("  High rank correlation despite distributional mismatch (KS) is")
print("  expected: cell-level aggregation changes absolute intensities but")
print("  preserves relative marker expression patterns.")
print()
print("  SLIC vs. DeepCell are fundamentally different approaches:")
print("    - DeepCell: deep-learning nuclear/membrane segmentation -> true cells")
print("    - SLIC: spatial superpixels -> texture/gradient tiles, not cells")
print("  This script documents pipeline validation (do numbers make sense?),")
print("  NOT equivalence between segmentation methods.")
print()
print(f"Outputs written:")
print(f"  Table:  {OUTPUT_TABLE}")
print(f"  Figure: {OUTPUT_FIGURE}")
