#!/usr/bin/env python
"""C2 — CD44-rate trajectory figures (dataviz skill governed).

Reads ONLY the two C1 CSVs in this directory and renders three figures, each in
paired light + dark PNGs (base names kept exactly so verify_C2.sh's glob and
C3's COLLAB_NOTE.md references resolve):

  fig_cd44_pooled.{light,dark}.png            pooled trajectories, one panel/compartment
  fig_cd44_cortex_medulla.{light,dark}.png    same, split cortex vs medulla (paired within-mouse)
  fig_cd44_scale_sensitivity.{light,dark}.png 10um vs 40um, n_support encoded as marker size

Honest-n=2 design (per the C2 spec + dataviz skill):
  * both mice shown as individual per-mouse points at every timepoint
  * NO error bars, whiskers, or CI bands anywhere
  * validated 2-hue categorical palette (blue = MS1, orange = MS2), light AND dark
  * every plotted value comes straight from the CSV; this file holds NO numeric
    data literals (only display constants: colors, sizes, tick anchors)
  * honest compartment labels (CD140b = pericyte/mural PDGFRb; CD206 = M2-like;
    endothelial = CD31+ & CD34+); a compartment is a ~superpixel positivity rate,
    not a segmented cell
  * neutrophil has no 40um basis, so the scale-sensitivity figure never fabricates
    a 40um neutrophil series (only compartments present in the 40um CSV appear there)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
CSV10 = BASE / "cd44_compartment_rates_10um.csv"
CSV20 = BASE / "cd44_compartment_rates_20um.csv"
CSV40 = BASE / "cd44_compartment_rates_40um.csv"

# ---- ordered display vocab (labels/order, not data values) -------------------
TIMEPOINTS = ["Sham", "D1", "D3", "D7"]
MICE = ["MS1", "MS2"]
COMPARTMENT_ORDER = [
    "CD45", "CD206", "CD31", "CD34", "endothelial_cd31cd34", "CD140b", "neutrophil",
]
SCALE_SUBSET = ["CD206", "endothelial_cd31cd34", "CD140b"]  # collaborator's ask
LABELS = {
    "CD45": "CD45+ (leukocyte)",
    "CD206": "CD206+ (M2-like)",
    "CD31": "CD31+ (endothelial)",
    "CD34": "CD34+ (endothelial/progenitor)",
    "endothelial_cd31cd34": "Endothelial (CD31+ & CD34+)",
    "CD140b": "CD140b+ (pericyte/mural, PDGFRβ)",
    "neutrophil": "Neutrophil",
}

# ---- validated categorical palette (dataviz skill: slot 1 blue, slot 8 orange) ---
# node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light  -> ALL PASS
# node scripts/validate_palette.js "#3987e5,#d95926" --mode dark   -> ALL PASS
THEMES = {
    "light": dict(
        surface="#fcfcfb", page="#f9f9f7", primary="#0b0b0b", secondary="#52514e",
        muted="#898781", grid="#e1e0d9", baseline="#c3c2b7",
        MS1="#2a78d6", MS2="#eb6834", ring="#fcfcfb",
    ),
    "dark": dict(
        surface="#1a1a19", page="#0d0d0d", primary="#ffffff", secondary="#c3c2b7",
        muted="#898781", grid="#2c2c2a", baseline="#383835",
        MS1="#3987e5", MS2="#d95926", ring="#1a1a19",
    ),
}


def load():
    d10 = pd.read_csv(CSV10)
    d20 = pd.read_csv(CSV20)
    d40 = pd.read_csv(CSV40)
    for d in (d10, d20, d40):
        d["timepoint"] = pd.Categorical(d["timepoint"], categories=TIMEPOINTS, ordered=True)
    return d10, d20, d40


def series(df, comp, region, mouse):
    """Ordered (x, rate, n_support) for one compartment/region/mouse. Values from CSV only."""
    sub = df[(df.compartment == comp) & (df.region == region) & (df.mouse == mouse)]
    sub = sub.sort_values("timepoint")
    x = [TIMEPOINTS.index(t) for t in sub.timepoint]
    return x, list(sub.cd44_rate), list(sub.n_support)


def style_axis(ax, th):
    ax.set_facecolor(th["surface"])
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.45, len(TIMEPOINTS) - 0.55)
    ax.set_xticks(range(len(TIMEPOINTS)))
    ax.set_xticklabels(TIMEPOINTS, color=th["secondary"], fontsize=8)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".5", ".75", "1"], color=th["secondary"], fontsize=8)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=th["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for name, sp in ax.spines.items():
        sp.set_visible(name == "bottom")
    ax.spines["bottom"].set_color(th["baseline"])


def plot_mouse_line(ax, x, y, th, mouse, markersize=8):
    """Per-mouse points + a thin recessive connector. No error bars."""
    color = th[mouse]
    if len(x) > 1:
        ax.plot(x, y, color=color, linewidth=1.4, alpha=0.5, zorder=2, solid_capstyle="round")
    ax.plot(
        x, y, linestyle="none", marker="o", markersize=markersize,
        markerfacecolor=color, markeredgecolor=th["ring"], markeredgewidth=1.5, zorder=3,
    )


def mouse_legend(fig_or_ax, th, is_ax=False, **kw):
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=9,
               markerfacecolor=th[m], markeredgecolor=th["ring"], markeredgewidth=1.2, label=m)
        for m in MICE
    ]
    leg = fig_or_ax.legend(handles=handles, title="Mouse", frameon=False, **kw)
    leg.get_title().set_color(th["secondary"])
    for t in leg.get_texts():
        t.set_color(th["secondary"])
    return leg


def caption(fig, th, text, y=0.015):
    fig.text(0.5, y, text, ha="center", va="bottom", color=th["muted"],
             fontsize=7.5, wrap=True)


# ----------------------------------------------------------------------------- #
def fig_pooled(d10, mode):
    th = THEMES[mode]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    fig.patch.set_facecolor(th["page"])
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.14, hspace=0.42, wspace=0.28)
    for i, comp in enumerate(COMPARTMENT_ORDER):
        ax = axes.flat[i]
        style_axis(ax, th)
        for mouse in MICE:
            x, y, _ = series(d10, comp, "pooled", mouse)
            plot_mouse_line(ax, x, y, th, mouse)
        ax.set_title(LABELS[comp], color=th["primary"], fontsize=9.5, pad=6)
        if i % 4 == 0:
            ax.set_ylabel("CD44+ rate", color=th["secondary"], fontsize=8.5)
    legend_ax = axes.flat[7]
    legend_ax.axis("off")
    legend_ax.set_facecolor(th["page"])
    mouse_legend(legend_ax, th, loc="center", title_fontsize=10, fontsize=10)
    fig.suptitle(
        "CD44+ activation rate across kidney-healing timepoints — pooled (cortex + medulla)",
        color=th["primary"], fontsize=13, x=0.055, ha="left", y=0.965,
    )
    fig.text(0.055, 0.905,
             "Per-superpixel CD44 positivity within each marker compartment at ~10 µm — a positivity rate, not segmented cells.",
             ha="left", color=th["secondary"], fontsize=8.5)
    caption(fig, th,
            "n = 2 mice / timepoint, shown as individual per-mouse points (MS1, MS2); thin line links a mouse across time. "
            "No error bars / CI — descriptive, hypothesis-generating.  Source: cd44_compartment_rates_10um.csv (region = pooled).")
    return fig


def fig_cortex_medulla(d10, mode):
    th = THEMES[mode]
    regions = ["cortex", "medulla"]
    nrow = len(COMPARTMENT_ORDER)
    fig, axes = plt.subplots(nrow, 2, figsize=(7.6, 16.5))
    fig.patch.set_facecolor(th["page"])
    fig.subplots_adjust(left=0.19, right=0.975, top=0.925, bottom=0.075, hspace=0.32, wspace=0.16)
    for r, comp in enumerate(COMPARTMENT_ORDER):
        for c, region in enumerate(regions):
            ax = axes[r, c]
            style_axis(ax, th)
            for mouse in MICE:
                x, y, _ = series(d10, comp, region, mouse)
                plot_mouse_line(ax, x, y, th, mouse, markersize=7)
            if r == 0:
                ax.set_title(region.capitalize(), color=th["primary"], fontsize=11, pad=8)
            if c == 0:
                ax.set_ylabel(LABELS[comp], color=th["primary"], fontsize=8.6, labelpad=8)
            else:
                ax.set_yticklabels([])
    mouse_legend(fig, th, loc="upper right", bbox_to_anchor=(0.975, 0.985),
                 ncol=2, title_fontsize=9.5, fontsize=9.5)
    fig.suptitle("CD44+ rate — cortex vs medulla, paired within mouse",
                 color=th["primary"], fontsize=13, x=0.02, ha="left", y=0.985)
    caption(fig, th,
            "Same n = 2 mice appear in both cortex and medulla (paired within-mouse; identical color per mouse across columns). "
            "Splitting the region halves already-thin support and adds no power — still n = 2, no error bars, descriptive. "
            "Source: cd44_compartment_rates_10um.csv.", y=0.028)
    return fig


def size_of(n, k=0.11, offset=20.0):
    """Marker area grows with n_support (offset only so the tiniest 40um points stay visible)."""
    return offset + n * k


def fig_scale_sensitivity(d10, d20, d40, mode):
    th = THEMES[mode]
    scales = [("10 µm superpixels", d10), ("20 µm superpixels", d20), ("40 µm superpixels", d40)]
    nrow = len(SCALE_SUBSET)
    fig, axes = plt.subplots(nrow, len(scales), figsize=(12.6, 9.6))
    fig.patch.set_facecolor(th["page"])
    fig.subplots_adjust(left=0.11, right=0.83, top=0.9, bottom=0.135, hspace=0.32, wspace=0.14)
    for r, comp in enumerate(SCALE_SUBSET):
        for c, (title, df) in enumerate(scales):
            ax = axes[r, c]
            style_axis(ax, th)
            for mouse in MICE:
                x, y, n = series(df, comp, "pooled", mouse)
                if len(x) > 1:
                    ax.plot(x, y, color=th[mouse], linewidth=1.2, alpha=0.4, zorder=2)
                ax.scatter(x, y, s=[size_of(v) for v in n], facecolor=th[mouse],
                           edgecolor=th["ring"], linewidth=1.4, zorder=3)
            if r == 0:
                ax.set_title(title, color=th["primary"], fontsize=11, pad=8)
            if c == 0:
                ax.set_ylabel(LABELS[comp], color=th["primary"], fontsize=8.8, labelpad=8)
            else:
                ax.set_yticklabels([])

    # size legend anchored to the actual plotted n_support range (derived, not typed)
    plotted = pd.concat([d10, d20, d40])
    plotted = plotted[(plotted.region == "pooled") & (plotted.compartment.isin(SCALE_SUBSET))]
    nvals = plotted.n_support
    n_lo, n_hi = int(nvals.min()), int(nvals.max())
    n_mid = int(round((n_lo + n_hi) / 2, -1))
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=th["muted"], markeredgecolor=th["ring"], markeredgewidth=1.0,
               markersize=(size_of(v) ** 0.5), label=f"{v:,}")
        for v in (n_lo, n_mid, n_hi)
    ]
    sl = fig.legend(handles=size_handles, title="n_support\n(superpixels)",
                    loc="center left", bbox_to_anchor=(0.845, 0.68),
                    frameon=False, labelspacing=1.5, handletextpad=1.0,
                    borderpad=1.0, fontsize=8.5)
    sl.get_title().set_color(th["secondary"])
    sl.get_title().set_fontsize(8.5)
    for t in sl.get_texts():
        t.set_color(th["secondary"])
    ml = mouse_legend(fig, th, loc="center left", bbox_to_anchor=(0.845, 0.3),
                      title_fontsize=9.5, fontsize=9.5)
    fig.add_artist(sl)  # keep both legends
    fig.add_artist(ml)

    # collapse magnitude computed from the CSV rows actually shown
    def _n(df):
        return int(df[(df.region == "pooled") & (df.compartment.isin(SCALE_SUBSET))].n_support.sum())
    n10, n20, n40 = _n(d10), _n(d20), _n(d40)
    ratio = n10 / n40
    fig.suptitle("CD44+ rate across 10 / 20 / 40 µm superpixel scales (pooled)",
                 color=th["primary"], fontsize=13, x=0.02, ha="left", y=0.965)
    caption(fig, th,
            f"Marker area grows with n_support: summed support across the three shown compartments collapses "
            f"~{n10:,} → ~{n20:,} → ~{n40:,} (~{ratio:.0f}× from 10 µm to 40 µm) — the coarser scales are far thinner, "
            "yet the Sham→D7 rise holds at every scale (scale-robust). "
            "Neutrophil has no grid-scale basis and is omitted (not fabricated). n = 2 per-mouse points, no error bars, descriptive. "
            "Source: cd44_compartment_rates_{10,20,40}um.csv (region = pooled).", y=0.02)
    return fig


def main():
    d10, d20, d40 = load()
    builders = {
        "fig_cd44_pooled": lambda mode: fig_pooled(d10, mode),
        "fig_cd44_cortex_medulla": lambda mode: fig_cortex_medulla(d10, mode),
        "fig_cd44_scale_sensitivity": lambda mode: fig_scale_sensitivity(d10, d20, d40, mode),
    }
    for base, build in builders.items():
        for mode in ("light", "dark"):
            fig = build(mode)
            out = BASE / f"{base}.{mode}.png"
            fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
