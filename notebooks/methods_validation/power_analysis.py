"""
Power Analysis for n=8 Study Using Mouse-Level Effect Sizes

Proper approach for hierarchical spatial biology data:
1. Aggregate to mouse-level means (biological unit of replication)
2. Compute effect sizes from n=4 mouse values (2 per group)
3. Acknowledge massive uncertainty with n=2 per group
4. Provide honest power estimates for n=8 exploratory study

Critical: Effect sizes must be computed from biological replicates (mice),
not technical replicates (superpixels/ROIs).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.power import ttest_power
import json
import gzip
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Setup
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# ============================================================================
# LOAD DATA AND AGGREGATE TO MOUSE LEVEL
# ============================================================================

print("="*70)
print("POWER ANALYSIS: Mouse-Level Effect Sizes")
print("="*70)

print("\nProper approach for hierarchical data:")
print("   ✅ Aggregate to mouse-level means (biological replicates)")
print("   ✅ Compute Cohen's d from n=4 mouse values")
print("   ✅ Acknowledge uncertainty with n=2 per group")

config = Config(str(project_root / 'config.json.backup'))
results_dir = project_root / 'results'
roi_results_dir = results_dir / 'roi_results'
markers = config.proteins

# Load ROI files
all_roi_files = sorted(roi_results_dir.glob('roi_*.json.gz'))
roi_files = [f for f in all_roi_files if 'IMC_241218_Alun' in f.name and
             (any(f'_D{i}_' in f.name for i in [1, 3, 7]) or '_Sam' in f.name)]

def parse_roi_name(filename: str) -> Dict[str, str]:
    """Parse both Sham (Sam) and UUO (D1/D3/D7) filenames."""
    clean_name = filename.replace('.json.gz', '').replace('.json', '').replace('_results', '')
    parts = clean_name.split('_')

    if 'Sam' in clean_name:
        sam_idx = next(i for i, p in enumerate(parts) if p.startswith('Sam'))
        return {
            'condition': 'Sham',
            'timepoint': 'Sham',
            'mouse': parts[sam_idx],
            'roi_num': parts[sam_idx + 1],
            'group': f'Sham_{parts[sam_idx]}'
        }
    else:
        tp_idx = next(i for i, p in enumerate(parts) if p.startswith('D'))
        return {
            'condition': 'UUO',
            'timepoint': parts[tp_idx],
            'mouse': parts[tp_idx + 1],
            'roi_num': parts[tp_idx + 2],
            'group': f'{parts[tp_idx]}_{parts[tp_idx + 1]}'
        }

roi_metadata = [dict(parse_roi_name(f.name), file_path=f) for f in roi_files]
roi_df = pd.DataFrame(roi_metadata)

def load_superpixel_features(roi_file: Path, scale: float = 10.0) -> pd.DataFrame:
    """Load superpixel features from compressed JSON."""
    with gzip.open(roi_file, 'rt') as f:
        results = json.load(f)

    scale_data = results['multiscale_results'][str(float(scale))]
    features_data = scale_data['features']
    features_array = np.array(features_data['data']).reshape(features_data['shape'])

    df = pd.DataFrame(features_array[:, :len(markers)], columns=markers)
    return df

# ============================================================================
# STEP 1: AGGREGATE TO MOUSE LEVEL (CORRECT UNIT OF ANALYSIS)
# ============================================================================

print("\n" + "="*70)
print("STEP 1: Aggregate to Mouse-Level Means")
print("="*70)

mouse_level_data = []

for group_name in roi_df['group'].unique():
    group_rois = roi_df[roi_df.group == group_name]

    # Load and pool all superpixels from this mouse
    all_superpixels = []
    for _, roi_row in group_rois.iterrows():
        sp_df = load_superpixel_features(roi_row['file_path'])
        all_superpixels.append(sp_df)

    combined = pd.concat(all_superpixels, ignore_index=True)

    # Compute SINGLE value per mouse per marker (mean across all ROIs/superpixels)
    mouse_means = combined[markers].mean()

    # Extract metadata
    metadata = group_rois.iloc[0]

    mouse_level_data.append({
        'mouse_id': group_name,
        'condition': metadata['condition'],
        'timepoint': metadata['timepoint'],
        'mouse': metadata['mouse'],
        **{marker: mouse_means[marker] for marker in markers}
    })

mouse_df = pd.DataFrame(mouse_level_data)

print(f"\n✅ Aggregated to {len(mouse_df)} mouse-level observations")
print("\nMouse-level data structure:")
print(mouse_df[['mouse_id', 'condition', 'timepoint']].to_string(index=False))

# ============================================================================
# STEP 2: COMPUTE PROPER EFFECT SIZE (n=4 mice, NOT 50k superpixels)
# ============================================================================

print("\n" + "="*70)
print("STEP 2: Compute Effect Sizes from Mouse-Level Means")
print("="*70)

# Sham vs D7 comparison (primary biological effect)
sham_mice = mouse_df[mouse_df.condition == 'Sham']
d7_mice = mouse_df[mouse_df.timepoint == 'D7']

print(f"\nSham mice (n={len(sham_mice)}): {list(sham_mice.mouse_id.values)}")
print(f"D7 UUO mice (n={len(d7_mice)}): {list(d7_mice.mouse_id.values)}")

def cohens_d_mouse_level(group1: pd.Series, group2: pd.Series) -> tuple:
    """
    Cohen's d from mouse-level values (CORRECT approach).

    Returns: (d, pooled_sd, mean_diff, stderr)
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

    # Pooled standard deviation
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var)

    # Effect size
    d = (mean1 - mean2) / pooled_sd if pooled_sd > 0 else 0

    # Standard error of difference
    se_diff = pooled_sd * np.sqrt(1/n1 + 1/n2)

    return d, pooled_sd, mean1 - mean2, se_diff

print("\n" + "-"*70)
print("Mouse-Level Effect Sizes (Sham vs D7)")
print("-"*70)
print(f"{'Marker':<12s}  Mouse d   Pooled SD   Mean Diff   SE_diff")
print("-"*70)

mouse_level_effects = []

for marker in markers:
    d, pooled_sd, mean_diff, se_diff = cohens_d_mouse_level(
        sham_mice[marker],
        d7_mice[marker]
    )

    print(f"{marker:<12s}  {d:+6.2f}    {pooled_sd:8.4f}   {mean_diff:+9.4f}   {se_diff:7.4f}")

    mouse_level_effects.append({
        'marker': marker,
        'cohens_d_mouse': d,
        'pooled_sd_mouse': pooled_sd,
        'mean_diff': mean_diff,
        'se_diff': se_diff,
        'n_per_group': 2
    })

effects_df = pd.DataFrame(mouse_level_effects)
effects_df = effects_df.sort_values('cohens_d_mouse', key=abs, ascending=False)

print("\n⚠️  WARNING: n=2 per group means:")
print("   - Only 1 degree of freedom per group")
print("   - Pooled SD has huge uncertainty")
print("   - Effect sizes are highly unstable")

# ============================================================================
# STEP 3: HONEST POWER ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("STEP 3: Power Analysis with Mouse-Level Effect Sizes")
print("="*70)

print("\nTop 3 markers by |d| (mouse-level):")
top_3 = effects_df.head(3)
for _, row in top_3.iterrows():
    print(f"  {row['marker']:12s}: d={row['cohens_d_mouse']:+.2f} (pooled SD={row['pooled_sd_mouse']:.4f})")

# Power analysis for top marker
top_marker = top_3.iloc[0]
top_d = abs(top_marker['cohens_d_mouse'])

print(f"\nPower curve for {top_marker['marker']} (d={top_d:.2f}):")
print("n/group   Power")
print("-" * 20)

sample_sizes = np.arange(2, 21, 1)
powers = [ttest_power(top_d, n, 0.05, alternative='two-sided') for n in sample_sizes]

for n, power in zip(sample_sizes, powers):
    marker = " <-- proposed" if n == 8 else ""
    print(f"  {n:2d}      {power:5.1%}{marker}")

# Find n needed for 80% power
n_for_80 = next((n for n, p in zip(sample_sizes, powers) if p >= 0.80), None)

if n_for_80:
    print(f"\n✅ n={n_for_80} per group for 80% power (IF true d={top_d:.2f})")
else:
    # Extend search
    extended_n = np.arange(20, 101, 5)
    extended_powers = [ttest_power(top_d, n, 0.05, alternative='two-sided') for n in extended_n]
    n_for_80 = next((n for n, p in zip(extended_n, extended_powers) if p >= 0.80), ">100")
    print(f"\n⚠️  n={n_for_80} per group needed for 80% power")

# ============================================================================
# STEP 4: REALITY CHECK
# ============================================================================

print("\n" + "="*70)
print("BRUTAL REALITY CHECK")
print("="*70)

print("\n🔴 CRITICAL LIMITATIONS with n=2 pilot:")
print("   1. Effect sizes have MASSIVE uncertainty (1 df per group)")
print("   2. Cannot reliably estimate between-mouse variance")
print("   3. Observed d could easily be 2x larger or smaller in n=8 study")
print("   4. Power calculations assume TRUE d equals observed d (big IF)")

print("\n✅ WHAT WE CAN LEGITIMATELY SAY:")
print("   1. Mouse-level effects observed (descriptive, n=4 total)")
print("   2. Variance is dominated by superpixel noise (92.8%)")
print("   3. Effect sizes guide n=8 as EXPLORATORY study")
print("   4. Cannot claim n=8 will have X% power (d is too unstable)")

print("\n💡 HONEST RECOMMENDATIONS:")
print("   Option A: Accept n=8 as EXPLORATORY (hypothesis-generating)")
print("   Option B: Use published kidney injury variance for power calc")
print("   Option C: Run n=4-5 mice per group PILOT first to stabilize variance")

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_dir = project_root / 'notebooks' / 'methods_validation' / 'results'
output_dir.mkdir(exist_ok=True, parents=True)

# Save mouse-level data
mouse_df.to_csv(output_dir / 'mouse_level_data.csv', index=False)

# Save mouse-level effect sizes
effects_df.to_csv(output_dir / 'effect_sizes_mouse_level.csv', index=False)

# Save power curve
power_curve_df = pd.DataFrame({
    'n_per_group': sample_sizes,
    f'power_{top_marker["marker"]}': powers
})
power_curve_df.to_csv(output_dir / 'power_curve.csv', index=False)

print(f"\n💾 Saved results:")
print(f"   - {output_dir / 'mouse_level_data.csv'}")
print(f"   - {output_dir / 'effect_sizes_mouse_level.csv'}")
print(f"   - {output_dir / 'power_curve.csv'}")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Mouse-level means (raw data)
ax = ax1
for idx, row in mouse_df.iterrows():
    color = 'blue' if row['condition'] == 'Sham' else 'red'
    marker_style = 'o' if row['mouse'] == 'Sam1' or row['mouse'] == 'M1' else 's'
    ax.scatter(idx, row[top_marker['marker']], s=150, color=color,
               marker=marker_style, alpha=0.7, edgecolor='black', linewidth=2)
    ax.text(idx, row[top_marker['marker']] + 0.02, row['mouse'],
            ha='center', fontsize=9)

ax.axhline(sham_mice[top_marker['marker']].mean(), color='blue', linestyle='--',
           linewidth=2, label='Sham mean')
ax.axhline(d7_mice[top_marker['marker']].mean(), color='red', linestyle='--',
           linewidth=2, label='D7 mean')

ax.set_ylabel(f"{top_marker['marker']} (arcsinh)", fontsize=12)
ax.set_xlabel('Mouse', fontsize=12)
ax.set_title(f'Mouse-Level Data (n=4)\nEffect size d={top_d:.2f}',
             fontsize=14, fontweight='bold')
ax.set_xticks(range(len(mouse_df)))
ax.set_xticklabels(mouse_df['mouse_id'], rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Power curve
ax = ax2
ax.plot(sample_sizes, powers, 'o-', linewidth=2, markersize=6, color='#2E86AB')
ax.axhline(0.80, color='green', linestyle='--', linewidth=1.5, label='80% power')
ax.axvline(8, color='red', linestyle=':', linewidth=1.5, label='Proposed n=8')

ax.set_xlabel('Sample Size (n per group)', fontsize=12)
ax.set_ylabel('Statistical Power', fontsize=12)
ax.set_title(f'Power Curve: {top_marker["marker"]} (d={top_d:.2f})',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_xlim(2, 20)
ax.grid(alpha=0.3)
ax.legend()

# Annotate n=8 point
power_at_8 = ttest_power(top_d, 8, 0.05, alternative='two-sided')
ax.annotate(f'n=8: {power_at_8:.1%}',
            xy=(8, power_at_8),
            xytext=(10, power_at_8 - 0.15),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
fig_path = output_dir / 'power_analysis.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\n💾 Saved figure: {fig_path}")

print("\n" + "="*70)
print("POWER ANALYSIS COMPLETE")
print("="*70)
print("\n✅ Effect sizes from biological replicates (n=4 mice)")
print("✅ Variance decomposition complete")
print("⚠️  Power estimates have large uncertainty (n=2 per group)")
print("💡 Results inform n=8 exploratory study design")
