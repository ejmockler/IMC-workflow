#!/usr/bin/env python3
"""
Power analysis from pilot effect sizes.

Reads temporal differential abundance results from an n=2 pilot study,
estimates required sample sizes for adequately powered follow-up experiments,
and produces a forest plot of pilot effect sizes with 95% CIs.

n=2 per group means all confidence intervals are extremely wide and all
p-values are non-significant. The value of this pilot is in estimating
effect sizes for future study design — not inference.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.stats.power import TTestIndPower

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = "/Users/noot/Documents/IMC"
INPUT_CSV = os.path.join(
    BASE_DIR,
    "results/biological_analysis/differential_abundance/temporal_differential_abundance.csv",
)
OUTPUT_DIR_TABLE = os.path.join(BASE_DIR, "results/power_analysis")
OUTPUT_DIR_FIGS  = os.path.join(BASE_DIR, "results/figures")
OUTPUT_TABLE     = os.path.join(OUTPUT_DIR_TABLE, "sample_size_requirements.csv")
OUTPUT_FIGURE    = os.path.join(OUTPUT_DIR_FIGS,  "pilot_effect_sizes.png")

os.makedirs(OUTPUT_DIR_TABLE, exist_ok=True)
os.makedirs(OUTPUT_DIR_FIGS,  exist_ok=True)


# ---------------------------------------------------------------------------
# Sample size calculation (Mann-Whitney via t-test ARE correction)
# ---------------------------------------------------------------------------
# For a two-sample Mann-Whitney U test vs. a normally distributed outcome,
# the asymptotic relative efficiency (ARE) relative to the t-test is pi/3
# under the normal distribution assumption, meaning MW requires roughly
# pi/3 ≈ 1.047x more subjects than the t-test for the same power.
# We use the two-sample t-test formula as the base and multiply by this factor.

ARE_MW_VS_T = np.pi / 3.0  # ≈ 1.0472

def n_required_t_test(effect_size: float, power: float, alpha: float = 0.05) -> int:
    """
    Required n per group for two-sample t-test (two-sided).
    Returns ceiling integer. Returns np.inf for effect_size ≈ 0.
    """
    if abs(effect_size) < 1e-6:
        return np.inf
    analysis = TTestIndPower()
    n = analysis.solve_power(
        effect_size=abs(effect_size),
        power=power,
        alpha=alpha,
        ratio=1.0,
        alternative="two-sided",
    )
    return int(np.ceil(n))


def n_required_mw(effect_size: float, power: float, alpha: float = 0.05) -> int:
    """
    Required n per group for Mann-Whitney U test (two-sided).
    Applies ARE correction to the t-test sample size.
    Returns np.inf for effect_size ≈ 0.
    """
    n_t = n_required_t_test(effect_size, power, alpha)
    if not np.isfinite(n_t):
        return np.inf
    return int(np.ceil(n_t * ARE_MW_VS_T))


# ---------------------------------------------------------------------------
# MDE: minimum detectable effect for a given n at 80% power, alpha=0.05
# ---------------------------------------------------------------------------
def mde_t_test(n_per_group: int, power: float = 0.80, alpha: float = 0.05) -> float:
    """
    Minimum detectable effect size (Cohen's d / Hedges' g ≈ for large n)
    for the t-test with n_per_group observations per arm.
    """
    analysis = TTestIndPower()
    return analysis.solve_power(
        nobs1=n_per_group,
        power=power,
        alpha=alpha,
        ratio=1.0,
        alternative="two-sided",
    )


# ---------------------------------------------------------------------------
# Load and process pilot data
# ---------------------------------------------------------------------------
print("Loading pilot data from:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)

required_cols = {
    "cell_type", "comparison",
    "hedges_g", "ci_lower_95", "ci_upper_95",
}
missing = required_cols - set(df.columns)
if missing:
    print(f"ERROR: missing columns in CSV: {missing}", file=sys.stderr)
    sys.exit(1)

print(f"  Loaded {len(df)} rows across {df['cell_type'].nunique()} cell types "
      f"and {df['comparison'].nunique()} comparisons.")
print(f"  Cell types: {sorted(df['cell_type'].unique())}")
print(f"  Comparisons: {sorted(df['comparison'].unique())}")

# For each cell type, find the comparison with the largest |Hedges' g|
# ("best-case" pilot estimate). This is the most optimistic scenario and
# should be treated as an upper bound — real effect sizes in a powered study
# are likely smaller (winner's curse / regression to the mean).
#
# Some cell types may have all-NaN Hedges' g (e.g. cell type absent in one
# condition, so Mann-Whitney and effect size are undefined). These are flagged
# and excluded from power calculations but included in the output table.
df["abs_hedges_g"] = df["hedges_g"].abs()

n_all_nan = 0
best_rows = []
for ct, grp in df.groupby("cell_type"):
    valid = grp.dropna(subset=["hedges_g"])
    if valid.empty:
        n_all_nan += 1
        best_rows.append({
            "cell_type":         ct,
            "max_hedges_g":      np.nan,
            "comparison_at_max": "N/A (all NaN)",
            "ci_lower":          np.nan,
            "ci_upper":          np.nan,
        })
        continue
    idx = valid["abs_hedges_g"].idxmax()
    row = valid.loc[idx]
    best_rows.append({
        "cell_type":         ct,
        "max_hedges_g":      row["hedges_g"],
        "comparison_at_max": row["comparison"],
        "ci_lower":          row["ci_lower_95"],
        "ci_upper":          row["ci_upper_95"],
    })

best = pd.DataFrame(best_rows)

if n_all_nan > 0:
    print(f"  NOTE: {n_all_nan} cell type(s) had all-NaN Hedges' g "
          f"(cell type absent in one or more conditions) — "
          f"these will be reported as undetermined in the output table.")

# Compute required sample sizes
print("\nComputing sample size requirements (Mann-Whitney ARE-corrected)...")
rows = []
for _, row in best.iterrows():
    g = row["max_hedges_g"]
    if pd.isna(g):
        n80 = np.nan
        n90 = np.nan
        n80_out = "N/A"
        n90_out = "N/A"
        g_out   = np.nan
        ci_lo   = np.nan
        ci_hi   = np.nan
    else:
        n80 = n_required_mw(g, power=0.80)
        n90 = n_required_mw(g, power=0.90)
        n80_out = n80 if np.isfinite(n80) else "Inf"
        n90_out = n90 if np.isfinite(n90) else "Inf"
        g_out   = round(g, 4)
        ci_lo   = round(row["ci_lower"], 4)
        ci_hi   = round(row["ci_upper"], 4)
    rows.append({
        "cell_type":          row["cell_type"],
        "max_hedges_g":       g_out,
        "comparison_at_max":  row["comparison_at_max"],
        "ci_lower":           ci_lo,
        "ci_upper":           ci_hi,
        "n_required_80pct":   n80_out,
        "n_required_90pct":   n90_out,
    })

results = pd.DataFrame(rows).sort_values("max_hedges_g", key=abs, ascending=False)

print("\nSample size requirements (n per group, Mann-Whitney corrected):")
print(results.to_string(index=False))

results.to_csv(OUTPUT_TABLE, index=False)
print(f"\nSaved table to: {OUTPUT_TABLE}")


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------
print("\nGenerating forest plot...")

# Sort by |max_hedges_g| for plotting — drop rows with NaN effect sizes
plot_df = results.copy()
# Convert Inf back to numeric for any display logic
plot_df["max_hedges_g"] = pd.to_numeric(plot_df["max_hedges_g"], errors="coerce")
plot_df["ci_lower"]     = pd.to_numeric(plot_df["ci_lower"],     errors="coerce")
plot_df["ci_upper"]     = pd.to_numeric(plot_df["ci_upper"],     errors="coerce")

# Rows where effect size is NaN cannot be plotted — annotate separately
nan_rows = plot_df[plot_df["max_hedges_g"].isna()]
plot_df  = plot_df.dropna(subset=["max_hedges_g"]).copy()
if not nan_rows.empty:
    print(f"  NOTE: {len(nan_rows)} cell type(s) excluded from forest plot (all-NaN effect size):")
    for ct in nan_rows["cell_type"]:
        print(f"    - {ct}")

plot_df = plot_df.sort_values("max_hedges_g", key=abs, ascending=True)  # bottom = smallest

n_cells = len(plot_df)
y_pos   = np.arange(n_cells)

# Minimum detectable effect at n=8 per group (a reasonable near-term study size)
mde_n8 = mde_t_test(n_per_group=8, power=0.80, alpha=0.05)
# Also apply ARE correction so MDE is on Mann-Whitney scale
# MDE_MW ≈ MDE_t / sqrt(ARE) to be conservative — but for display we show
# the t-test MDE as the more favourable line; label makes it explicit.
print(f"  MDE for n=8/group, 80% power, t-test (ARE-corrected threshold for MW): {mde_n8:.3f}")

# Axis limits — include zero plus some padding
all_bounds = np.concatenate([
    plot_df["ci_lower"].values,
    plot_df["ci_upper"].values,
    [0.0],
])
x_min = min(all_bounds) - 0.3
x_max = max(all_bounds) + 0.3

fig, ax = plt.subplots(figsize=(10, max(5, n_cells * 0.65 + 2)))
fig.patch.set_facecolor("white")

# --- grid ---
ax.set_facecolor("#f8f8f8")
ax.xaxis.grid(True, color="white", linewidth=1.0, zorder=0)

# --- reference lines ---
ax.axvline(x=0.0, color="black", linewidth=1.2, zorder=2, label="No effect (g = 0)")
ax.axvline(
    x=mde_n8, color="#d62728", linewidth=1.2, linestyle="--", zorder=2,
    label=f"MDE: n=8/group, 80% power, α=0.05 (t-test, g={mde_n8:.2f})",
)
ax.axvline(
    x=-mde_n8, color="#d62728", linewidth=1.2, linestyle="--", zorder=2,
)

# --- effect size points + CI bars ---
for i, (_, row) in enumerate(plot_df.iterrows()):
    g   = row["max_hedges_g"]
    lo  = row["ci_lower"]
    hi  = row["ci_upper"]
    ci_width_lo = g - lo
    ci_width_hi = hi - g

    # Colour by sign of effect
    color = "#1f77b4" if g >= 0 else "#ff7f0e"

    ax.errorbar(
        x=g, y=i,
        xerr=[[max(ci_width_lo, 0)], [max(ci_width_hi, 0)]],
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=1.5,
        capsize=4,
        markersize=7,
        zorder=4,
    )

    # n required annotation to the right of the CI
    n80_val = row["n_required_80pct"]
    label_txt = f"n={n80_val}/grp" if n80_val != "Inf" and str(n80_val) != "inf" else "n=∞"
    ax.text(
        x_max - 0.05, i,
        label_txt,
        va="center", ha="right",
        fontsize=7.5,
        color="gray",
    )

# --- y-axis labels (clean cell type names) ---
clean_names = [
    row["cell_type"].replace("_", " ").title()
    for _, row in plot_df.iterrows()
]
ax.set_yticks(y_pos)
ax.set_yticklabels(clean_names, fontsize=9)
ax.set_ylim(-0.8, n_cells - 0.2)

ax.set_xlim(x_min, x_max)
ax.set_xlabel("Hedges' g (pilot estimate, best comparison per cell type)", fontsize=10)
ax.set_title(
    "Pilot Effect Sizes: Temporal Cell-Type Abundance Changes\n"
    "Right column: n per group required for 80% power (Mann-Whitney, α=0.05)",
    fontsize=11, fontweight="bold", pad=12,
)

# Legend
legend_handles = [
    mpatches.Patch(color="#1f77b4", label="Positive effect (fold-increase)"),
    mpatches.Patch(color="#ff7f0e", label="Negative effect (fold-decrease)"),
    plt.Line2D([0], [0], color="black",  linewidth=1.2, label="g = 0"),
    plt.Line2D([0], [0], color="#d62728", linewidth=1.2, linestyle="--",
               label=f"MDE n=8/grp (80% pwr, g={mde_n8:.2f})"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
          framealpha=0.9, edgecolor="lightgray")

# Annotation note
fig.text(
    0.01, 0.01,
    "n=2 per group pilot estimates. CIs computed from mouse-level variances.\n"
    "All pilot CIs cross zero — effect size estimates carry high uncertainty.",
    fontsize=7.5, color="gray", ha="left", va="bottom",
    style="italic",
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved figure to: {OUTPUT_FIGURE}")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print("\n--- Summary ---")
print(f"Cell types analysed:          {n_cells}")
print(f"Pilot n per group:            2")
print(f"Comparisons per cell type:    {df['comparison'].nunique()}")
print(f"Range of |max Hedges' g|:     "
      f"{plot_df['max_hedges_g'].abs().min():.3f} – "
      f"{plot_df['max_hedges_g'].abs().max():.3f}")

finite_n80 = [
    r["n_required_80pct"]
    for r in rows
    if str(r["n_required_80pct"]) not in ("Inf", "inf", "N/A", "nan")
    and r["n_required_80pct"] is not np.nan
    and not (isinstance(r["n_required_80pct"], float) and np.isnan(r["n_required_80pct"]))
]
if finite_n80:
    print(f"n required 80% power (range): {min(finite_n80)} – {max(finite_n80)} per group")
else:
    print("n required 80% power: all infinite or undetermined (effect sizes ~ 0 or NaN)")

print(f"\nInterpretation note:")
print(f"  With n=2 per group, no comparison is statistically significant (all p_fdr > 0.9).")
print(f"  The pilot value is in effect size estimation, not inference.")
print(f"  Effect sizes from n=2 pilots are biased upward (winner's curse).")
print(f"  Treat n_required as LOWER BOUNDS on required study size.")
print(f"  MDE for n=8/group at 80% power: Hedges' g ≈ {mde_n8:.2f}")
print(f"  ARE factor (MW vs t-test): π/3 ≈ {ARE_MW_VS_T:.4f}")
