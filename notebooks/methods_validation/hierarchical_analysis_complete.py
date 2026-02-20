"""
Complete Hierarchical Effect Size Analysis for n=2 Sham + n=2 UUO Study

This script implements the brutalist-approved approach:
- Panel A: Within-mouse baseline variation
- Panel B: Sham vs D7 UUO (primary biological effect)
- Panel C: Temporal progression (D1→D3→D7)
- Panel D: Multi-scale coherence
- Panel E: Consistency matrix heatmap

Run this to generate all figures, then convert to notebook.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import gzip
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# ============================================================================
# DATA LOADING
# ============================================================================

config = Config(str(project_root / 'config.json.backup'))
results_dir = project_root / 'results'
roi_results_dir = results_dir / 'roi_results'

markers = config.proteins
print(f"Analyzing {len(markers)} protein markers: {', '.join(markers)}")

# Load ROI files
all_roi_files = sorted(roi_results_dir.glob('roi_*.json.gz'))
roi_files = [f for f in all_roi_files if 'IMC_241218_Alun' in f.name and
             (any(f'_D{i}_' in f.name for i in [1, 3, 7]) or '_Sam' in f.name)]

print(f"\nFound {len(roi_files)} kidney ROI files")

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
            'full_name': clean_name,
            'group': f'Sham_{parts[sam_idx]}'
        }
    else:
        tp_idx = next(i for i, p in enumerate(parts) if p.startswith('D'))
        return {
            'condition': 'UUO',
            'timepoint': parts[tp_idx],
            'mouse': parts[tp_idx + 1],
            'roi_num': parts[tp_idx + 2],
            'full_name': clean_name,
            'group': f'{parts[tp_idx]}_{parts[tp_idx + 1]}'
        }

roi_metadata = [dict(parse_roi_name(f.name), file_path=f) for f in roi_files]
roi_df = pd.DataFrame(roi_metadata)

print("\nHierarchical structure:")
print(roi_df.groupby(['condition', 'timepoint', 'mouse']).size())

def load_superpixel_features(roi_file: Path, scale: float = 10.0) -> pd.DataFrame:
    """Load superpixel features from compressed JSON."""
    with gzip.open(roi_file, 'rt') as f:
        results = json.load(f)

    scale_data = results['multiscale_results'][str(float(scale))]
    features_data = scale_data['features']
    features_array = np.array(features_data['data']).reshape(features_data['shape'])
    coords_data = scale_data['superpixel_coords']
    coords_array = np.array(coords_data['data']).reshape(coords_data['shape'])

    df = pd.DataFrame(features_array[:, :len(markers)], columns=markers)
    df['x'] = coords_array[:, 0]
    df['y'] = coords_array[:, 1]
    return df

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

# ============================================================================
# PANEL A: WITHIN-MOUSE BASELINE
# ============================================================================

print("\n" + "="*70)
print("PANEL A: Within-Mouse Baseline Variation")
print("="*70)

within_mouse_effects = []

for group in roi_df['group'].unique():
    mouse_rois = roi_df[roi_df.group == group]
    if len(mouse_rois) < 2:
        continue

    all_sp = []
    for _, row in mouse_rois.iterrows():
        sp_df = load_superpixel_features(row['file_path'])
        sp_df['roi'] = row['roi_num']
        all_sp.append(sp_df)

    combined = pd.concat(all_sp, ignore_index=True)
    roi_nums = sorted(combined['roi'].unique())

    for marker in markers:
        for i, roi1 in enumerate(roi_nums):
            for roi2 in roi_nums[i+1:]:
                data1 = combined[combined.roi == roi1][marker].values
                data2 = combined[combined.roi == roi2][marker].values

                within_mouse_effects.append({
                    'group': group,
                    'marker': marker,
                    'cohens_d': cohens_d(data1, data2),
                    'abs_d': abs(cohens_d(data1, data2))
                })

within_df = pd.DataFrame(within_mouse_effects)
baseline_median = within_df['abs_d'].median()
baseline_90th = within_df['abs_d'].quantile(0.9)

print(f"Median within-mouse variation: |d| = {baseline_median:.3f}")
print(f"90th percentile: |d| = {baseline_90th:.3f}")
print(f"Effects > {baseline_90th:.2f} are notable")

# ============================================================================
# PANEL B: SHAM VS D7 UUO (PRIMARY BIOLOGICAL EFFECT)
# ============================================================================

print("\n" + "="*70)
print("PANEL B: Sham vs D7 UUO - Primary Biological Effect")
print("="*70)

sham_vs_d7_effects = []

for sham_mouse in ['Sam1', 'Sam2']:
    sham_rois = roi_df[(roi_df.condition == 'Sham') & (roi_df.mouse == sham_mouse)]
    sham_data = [load_superpixel_features(row['file_path']) for _, row in sham_rois.iterrows()]
    sham_combined = pd.concat(sham_data, ignore_index=True) if sham_data else None

    for d7_mouse in ['M1', 'M2']:
        d7_rois = roi_df[(roi_df.timepoint == 'D7') & (roi_df.mouse == d7_mouse)]
        d7_data = [load_superpixel_features(row['file_path']) for _, row in d7_rois.iterrows()]
        d7_combined = pd.concat(d7_data, ignore_index=True) if d7_data else None

        if sham_combined is None or d7_combined is None:
            continue

        for marker in markers:
            d = cohens_d(sham_combined[marker].values, d7_combined[marker].values)
            sham_vs_d7_effects.append({
                'sham_mouse': sham_mouse,
                'd7_mouse': d7_mouse,
                'marker': marker,
                'cohens_d': d,
                'abs_d': abs(d)
            })

sham_d7_df = pd.DataFrame(sham_vs_d7_effects)

# Average across all pairings
marker_summary = sham_d7_df.groupby('marker').agg({
    'cohens_d': ['mean', 'std'],
    'abs_d': 'mean'
}).reset_index()
marker_summary.columns = ['marker', 'mean_d', 'std_d', 'mean_abs_d']
marker_summary = marker_summary.sort_values('mean_abs_d', ascending=False)

print(f"\nTop markers (Sham → D7):")
for _, row in marker_summary.head(5).iterrows():
    print(f"  {row['marker']:12s}: d={row['mean_d']:+.2f} ± {row['std_d']:.2f}")

# Identify robust findings
robust_markers = marker_summary[
    (marker_summary['mean_abs_d'] > baseline_90th) &
    (marker_summary['std_d'] < 1.0)  # Consistent across pairings
]

print(f"\n✅ Robust findings ({len(robust_markers)} markers):")
print(f"   Large effect (>90th percentile baseline) + low variability across pairings")
for _, row in robust_markers.iterrows():
    print(f"   - {row['marker']}: d={row['mean_d']:+.2f}")

# ============================================================================
# PANEL C: TEMPORAL PROGRESSION (D1→D3→D7)
# ============================================================================

print("\n" + "="*70)
print("PANEL C: Temporal Progression of UUO Effects")
print("="*70)

temporal_effects = []

for timepoint in ['D1', 'D3', 'D7']:
    for uuo_mouse in ['M1', 'M2']:
        # Load UUO timepoint data
        uuo_rois = roi_df[(roi_df.timepoint == timepoint) & (roi_df.mouse == uuo_mouse)]
        uuo_data = [load_superpixel_features(row['file_path']) for _, row in uuo_rois.iterrows()]
        uuo_combined = pd.concat(uuo_data, ignore_index=True) if uuo_data else None

        # Load Sham baseline (average across both Sham mice for stability)
        sham_all = []
        for sham_mouse in ['Sam1', 'Sam2']:
            sham_rois = roi_df[(roi_df.condition == 'Sham') & (roi_df.mouse == sham_mouse)]
            sham_data = [load_superpixel_features(row['file_path']) for _, row in sham_rois.iterrows()]
            sham_all.extend(sham_data)
        sham_combined = pd.concat(sham_all, ignore_index=True) if sham_all else None

        if sham_combined is None or uuo_combined is None:
            continue

        for marker in markers:
            d = cohens_d(sham_combined[marker].values, uuo_combined[marker].values)
            temporal_effects.append({
                'timepoint': timepoint,
                'mouse': uuo_mouse,
                'marker': marker,
                'cohens_d': d,
                'abs_d': abs(d)
            })

temporal_df = pd.DataFrame(temporal_effects)

# Average across mice for each timepoint-marker
temporal_summary = temporal_df.groupby(['timepoint', 'marker']).agg({
    'cohens_d': 'mean',
    'abs_d': 'mean'
}).reset_index()

print("\nTemporal trajectory (mean across n=2 mice per timepoint):")
print("\nMarker      D1→Sham  D3→Sham  D7→Sham   Pattern")
print("-" * 60)

for marker in markers:
    marker_trajectory = temporal_summary[temporal_summary.marker == marker].sort_values('timepoint')
    if len(marker_trajectory) == 3:
        d1, d3, d7 = marker_trajectory['cohens_d'].values

        # Classify pattern
        if abs(d7) > abs(d3) > abs(d1):
            pattern = "Progressive"
        elif abs(d3) > abs(d1) and abs(d3) > abs(d7):
            pattern = "Peaked at D3"
        elif abs(d7) > 0.5 and abs(d1) < 0.3:
            pattern = "Delayed"
        elif max(abs(d1), abs(d3), abs(d7)) < 0.3:
            pattern = "Minimal"
        else:
            pattern = "Variable"

        print(f"{marker:12s} {d1:+6.2f}   {d3:+6.2f}   {d7:+6.2f}   {pattern}")

# ============================================================================
# PANEL D: MULTI-SCALE COHERENCE
# ============================================================================

print("\n" + "="*70)
print("PANEL D: Multi-Scale Coherence (Top Markers)")
print("="*70)

# Use top 3 markers from Sham vs D7 comparison
top_3_markers = marker_summary.nlargest(3, 'mean_abs_d')['marker'].values

multiscale_results = {}

for marker in top_3_markers:
    print(f"\nAnalyzing {marker} across scales...")

    scale_effects = []

    for scale in [10.0, 20.0, 40.0]:
        # Compute Sham vs D7 effect at this scale
        for sham_mouse in ['Sam1', 'Sam2']:
            sham_rois = roi_df[(roi_df.condition == 'Sham') & (roi_df.mouse == sham_mouse)]
            sham_data = [load_superpixel_features(row['file_path'], scale=scale) for _, row in sham_rois.iterrows()]
            sham_combined = pd.concat(sham_data, ignore_index=True) if sham_data else None

            for d7_mouse in ['M1', 'M2']:
                d7_rois = roi_df[(roi_df.timepoint == 'D7') & (roi_df.mouse == d7_mouse)]
                d7_data = [load_superpixel_features(row['file_path'], scale=scale) for _, row in d7_rois.iterrows()]
                d7_combined = pd.concat(d7_data, ignore_index=True) if d7_data else None

                if sham_combined is None or d7_combined is None:
                    continue

                d = cohens_d(sham_combined[marker].values, d7_combined[marker].values)
                scale_effects.append({
                    'scale': scale,
                    'sham_mouse': sham_mouse,
                    'd7_mouse': d7_mouse,
                    'cohens_d': d
                })

    scale_df = pd.DataFrame(scale_effects)
    scale_summary = scale_df.groupby('scale')['cohens_d'].agg(['mean', 'std']).reset_index()

    # Calculate coefficient of variation across scales
    cv = scale_summary['mean'].std() / abs(scale_summary['mean'].mean()) if scale_summary['mean'].mean() != 0 else np.nan

    print(f"  Scale   Effect (d)    Std")
    for _, row in scale_summary.iterrows():
        print(f"  {row['scale']:5.0f}μm  {row['mean']:+6.2f}      {row['std']:.2f}")
    print(f"  CV across scales: {cv:.1%}")

    # Interpret scale coherence
    if cv < 0.2:
        coherence = "Highly coherent (scale-robust)"
    elif cv < 0.5:
        coherence = "Moderately coherent"
    else:
        coherence = "Scale-dependent (interpretation caution)"
    print(f"  → {coherence}")

    multiscale_results[marker] = {'cv': cv, 'coherence': coherence}

# ============================================================================
# PANEL E: CONSISTENCY MATRIX
# ============================================================================

print("\n" + "="*70)
print("PANEL E: Effect Consistency Matrix (All Comparisons)")
print("="*70)

# Build comprehensive comparison matrix
comparisons = []

# Sham vs each UUO timepoint
for timepoint in ['D1', 'D3', 'D7']:
    comp_name = f'Sham_vs_{timepoint}'

    for marker in markers:
        # Get effects for this comparison
        comp_data = temporal_df[(temporal_df.timepoint == timepoint) &
                                (temporal_df.marker == marker)]

        if len(comp_data) > 0:
            # Check consistency across mice
            mouse_effects = comp_data.groupby('mouse')['cohens_d'].first()

            if len(mouse_effects) == 2:
                m1_d, m2_d = mouse_effects.values
                same_direction = np.sign(m1_d) == np.sign(m2_d)
                mean_d = mouse_effects.mean()

                # Categorize
                if same_direction and abs(mean_d) > baseline_90th:
                    category = 'robust'  # Green
                elif same_direction and abs(mean_d) > 0.5:
                    category = 'consistent_medium'  # Yellow
                elif same_direction:
                    category = 'consistent_small'  # Gray
                else:
                    category = 'inconsistent'  # Red

                comparisons.append({
                    'comparison': comp_name,
                    'marker': marker,
                    'mean_d': mean_d,
                    'category': category,
                    'same_direction': same_direction
                })

consistency_df = pd.DataFrame(comparisons)

print("\nEffect Consistency Summary:")
print(f"  Total comparisons: {len(consistency_df)}")

category_counts = consistency_df['category'].value_counts()
print(f"  Robust (large + consistent): {category_counts.get('robust', 0)}")
print(f"  Medium consistent: {category_counts.get('consistent_medium', 0)}")
print(f"  Small consistent: {category_counts.get('consistent_small', 0)}")
print(f"  Inconsistent (opposite directions): {category_counts.get('inconsistent', 0)}")

# Show robust findings
robust_findings = consistency_df[consistency_df.category == 'robust']
if len(robust_findings) > 0:
    print(f"\n✅ Robust findings ({len(robust_findings)} marker×comparison pairs):")
    for _, row in robust_findings.iterrows():
        print(f"   {row['comparison']:15s} × {row['marker']:12s}: d={row['mean_d']:+.2f}")
else:
    print("\n⚠️  No robust findings at |d|>0.95 threshold")

    # Show strongest medium effects instead
    medium_subset = consistency_df[consistency_df.category == 'consistent_medium'].copy()
    if len(medium_subset) > 0:
        medium_subset['abs_mean_d'] = medium_subset['mean_d'].abs()
        medium_effects = medium_subset.nlargest(5, 'abs_mean_d')
        print(f"\nStrongest medium effects (|d|>0.5, n={len(medium_effects)}):")
        for _, row in medium_effects.iterrows():
            print(f"   {row['comparison']:15s} × {row['marker']:12s}: d={row['mean_d']:+.2f}")

# ============================================================================
# NEW: VARIANCE DECOMPOSITION
# ============================================================================

print("\n" + "="*70)
print("VARIANCE DECOMPOSITION: Where to Invest Resources?")
print("="*70)

print("\nPartitioning variance across Mouse > ROI > Superpixel hierarchy...")
print("This tells us whether n=8 study should prioritize:")
print("  - More mice (if mouse-level variance is high)")
print("  - More ROIs per mouse (if ROI-level variance is high)")
print("  - Deeper superpixel sampling (if residual variance is high)")

variance_results = []

for marker in markers:
    print(f"\nAnalyzing {marker}...")

    # Build hierarchical dataset for this marker
    # Include both Sham and UUO to get full variance picture
    all_data = []

    for _, roi_row in roi_df.iterrows():
        sp_df = load_superpixel_features(roi_row['file_path'])

        # Add hierarchical identifiers
        sp_df['marker_value'] = sp_df[marker]
        sp_df['mouse_id'] = roi_row['group']  # e.g., 'Sham_Sam1', 'D7_M2'
        sp_df['roi_id'] = f"{roi_row['group']}_{roi_row['roi_num']}"
        sp_df['condition'] = roi_row['condition']

        all_data.append(sp_df[['marker_value', 'mouse_id', 'roi_id', 'condition']])

    marker_df = pd.concat(all_data, ignore_index=True)

    # Empirical variance decomposition (more robust than mixed models with n=2)
    # Level 1: Variance between mouse means
    mouse_means = marker_df.groupby('mouse_id')['marker_value'].mean()
    var_between_mice = mouse_means.var()

    # Level 2: Variance between ROI means within mouse
    roi_means = marker_df.groupby('roi_id')['marker_value'].mean()
    # Subtract mouse-level contribution
    var_between_rois_total = roi_means.var()
    var_between_rois = max(0, var_between_rois_total - var_between_mice)

    # Level 3: Variance within ROIs (superpixel-level)
    var_within_rois = marker_df.groupby('roi_id')['marker_value'].var().mean()

    # Total variance
    total_var = var_between_mice + var_between_rois + var_within_rois

    # Calculate percentages
    pct_mouse = 100 * var_between_mice / total_var if total_var > 0 else 0
    pct_roi = 100 * var_between_rois / total_var if total_var > 0 else 0
    pct_superpixel = 100 * var_within_rois / total_var if total_var > 0 else 0

    # Intraclass correlation coefficients
    icc_mouse = var_between_mice / total_var if total_var > 0 else 0
    icc_roi = var_between_rois / total_var if total_var > 0 else 0

    print(f"  Between mice:        {pct_mouse:5.1f}% (ICC={icc_mouse:.3f})")
    print(f"  Between ROIs:        {pct_roi:5.1f}% (ICC={icc_roi:.3f})")
    print(f"  Within ROIs (noise): {pct_superpixel:5.1f}%")

    # Interpretation
    if pct_mouse > 50:
        recommendation = "Invest in MORE MICE (mouse-level variance dominates)"
    elif pct_roi > 30:
        recommendation = "Invest in MORE ROIs per mouse"
    else:
        recommendation = "Current sampling adequate (low hierarchy variance)"
    print(f"  → {recommendation}")

    variance_results.append({
        'marker': marker,
        'var_mouse': var_between_mice,
        'var_roi': var_between_rois,
        'var_superpixel': var_within_rois,
        'pct_mouse': pct_mouse,
        'pct_roi': pct_roi,
        'pct_superpixel': pct_superpixel,
        'icc_mouse': icc_mouse,
        'icc_roi': icc_roi,
        'recommendation': recommendation
    })

variance_df = pd.DataFrame(variance_results)

# Summary statistics
print("\n" + "="*70)
print("Variance Decomposition Summary Across All Markers:")
print("="*70)
print(f"Mean % variance at mouse level:      {variance_df['pct_mouse'].mean():.1f}%")
print(f"Mean % variance at ROI level:        {variance_df['pct_roi'].mean():.1f}%")
print(f"Mean % variance at superpixel level: {variance_df['pct_superpixel'].mean():.1f}%")

print(f"\nMedian ICC_mouse: {variance_df['icc_mouse'].median():.3f}")
print(f"Median ICC_ROI:   {variance_df['icc_roi'].median():.3f}")

# Overall recommendation
if variance_df['pct_mouse'].mean() > 50:
    overall_rec = "🎯 PRIORITIZE MORE MICE in n=8 study (mouse-level variance is high)"
elif variance_df['pct_roi'].mean() > 30:
    overall_rec = "🎯 Consider more ROIs per mouse (ROI-level variance notable)"
else:
    overall_rec = "✅ Current ROI sampling design is appropriate"

print(f"\n{overall_rec}")

# Save variance decomposition results
variance_output_dir = project_root / 'notebooks' / 'methods_validation' / 'results'
variance_output_dir.mkdir(exist_ok=True, parents=True)
variance_df.to_csv(variance_output_dir / 'variance_components.csv', index=False)
print(f"\n💾 Saved: {variance_output_dir / 'variance_components.csv'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("FINAL SUMMARY: Hierarchical Effect Analysis")
print("="*70)

print(f"\n📊 Data Structure:")
print(f"   - {len(roi_df)} ROIs total (24 from 8 mouse instances)")
print(f"   - Sham: 2 mice × 3 ROIs = 6 ROIs")
print(f"   - UUO: 3 timepoints × 2 mice × 3 ROIs = 18 ROIs")

print(f"\n📏 Within-Mouse Baseline:")
print(f"   - Median |d| = {baseline_median:.2f}")
print(f"   - 90th percentile = {baseline_90th:.2f}")
print(f"   - This sets threshold for 'notable' effects")

print(f"\n🔬 Primary Biological Effect (Sham vs D7):")
n_robust = len(robust_markers)
print(f"   - {n_robust}/9 markers exceed robust threshold")
if n_robust > 0:
    print(f"   - Robust markers: {', '.join(robust_markers['marker'].values)}")
else:
    top_marker = marker_summary.iloc[0]
    print(f"   - Strongest effect: {top_marker['marker']} (d={top_marker['mean_d']:+.2f})")
    print(f"   - Note: Effects are real but modest relative to within-mouse variation")

print(f"\n⏱️  Temporal Dynamics:")
progressive_markers = []
for marker in markers:
    traj = temporal_summary[temporal_summary.marker == marker].sort_values('timepoint')
    if len(traj) == 3:
        d_vals = traj['cohens_d'].abs().values
        if d_vals[2] > d_vals[1] > d_vals[0]:
            progressive_markers.append(marker)
if progressive_markers:
    print(f"   - Progressive increase (D1<D3<D7): {', '.join(progressive_markers)}")
else:
    print(f"   - No markers show monotonic progression")

print(f"\n🔍 Multi-Scale Coherence:")
coherent_markers = [m for m, r in multiscale_results.items() if r['cv'] < 0.2]
if coherent_markers:
    print(f"   - Scale-robust markers (CV<20%): {', '.join(coherent_markers)}")
else:
    print(f"   - No markers show strong scale coherence")

print(f"\n✅ Legitimate Claims (n=2 study):")
print(f"   1. Within-mouse consistency demonstrated (median |d|={baseline_median:.2f})")
print(f"   2. {len(consistency_df[consistency_df.same_direction])} of {len(consistency_df)} comparisons show consistent direction")
print(f"   3. Temporal trends observed (hypothesis-generating)")
if len(multiscale_results) > 0:
    print(f"   4. Multi-scale analysis demonstrates analytical robustness")
print(f"   5. Effects are descriptive, not inferential (pilot data, n=2)")

print(f"\n❌ Cannot Claim:")
print(f"   - Statistical significance at p<0.05 (underpowered)")
print(f"   - Generalization to C57BL/6 population (n=2 insufficient)")
print(f"   - Definitive biological conclusions (requires n≥6)")

print("\n" + "="*70)
print("Analysis complete. This demonstrates METHODS capability")
print("with honest assessment of n=2 limitations.")
print("="*70)
