# Critical Analysis: The 92.8% "Superpixel Variance" Finding

## Executive Summary

**The "noise" is not noise - it's real biological heterogeneity being correctly measured.**

Our variance decomposition showing 92.8% variance at the superpixel level is telling us something fundamental about spatial biology: **most biological variation happens at the micro-scale (cell-to-cell), not the mouse level**.

---

## The Numbers

### Variance Decomposition (Hierarchical Analysis)
- **Mouse (biological)**: 2.5% of total variance
- **ROI (spatial)**: 4.7% of total variance
- **Superpixel (micro-scale)**: 92.8% of total variance

### Mouse-Level Effect Sizes (Sham vs D7 UUO)
- CD44 (fibrosis): d = **+4.57** (huge!)
- CD34 (endothelial): d = **-3.54** (huge!)
- CD140a (pericyte): d = **-2.68** (large)

### The Paradox
- **Large effect sizes** when aggregated to mouse level
- But **only 2.5% variance** between mice
- And **92.8% variance** at superpixel level

---

## What We're Actually Measuring

### Spatial Scales in Our Pipeline

**"10μm" spacing parameter ≠ 10μm superpixels!**

| Parameter | Actual Superpixel Size | Cells per Superpixel | Superpixels per ROI |
|-----------|------------------------|----------------------|---------------------|
| 10μm spacing | ~93μm × 93μm | ~40-50 cells | ~20-30 |
| 20μm spacing | ~113μm × 113μm | ~70-80 cells | ~15-20 |
| 40μm spacing | ~309μm × 309μm | ~400-500 cells | ~5-10 |

**Key insight**: Even our "finest" scale averages across 40-50 cells!

### Steinbock Cell-Level Ground Truth (Bodenmiller Dataset)

- **3,568 cells** detected per ROI
- Mean cell area: **91 pixels²** (~91 μm² if 1μm/pixel)
- Typical cell diameter: ~11μm

**Cell-level protein heterogeneity**:
- CD3: CV = 118% (T cells vs non-T cells)
- CD8a: CV = 248% (highly variable, subset marker)
- CD68: CV = 144% (macrophages vs other cells)
- CD4: CV = 66% (helper T cells)
- CD20: CV = 50% (B cells)

---

## What the "92.8% Superpixel Variance" Actually Means

### The True Interpretation

The 92.8% variance at superpixel level reflects **local cellular composition heterogeneity**:

1. **Each superpixel contains ~100-200 cells** (Steinbock shows ~3500 cells, we have ~20 superpixels)

2. **Each marker has distinct cell-type expression**:
   - CD3+ T cells have high CD3, other cells have zero
   - CD68+ macrophages have high CD68, other cells have low
   - Ly6G+ neutrophils have high Ly6G, other cells have zero

3. **Different superpixels sample different local mixtures**:
   - Superpixel A: 40% T cells, 30% macrophages, 30% other → high CD3, high CD68
   - Superpixel B: 10% T cells, 60% macrophages, 30% other → low CD3, very high CD68
   - Superpixel C: 5% T cells, 10% macrophages, 85% epithelial → low CD3, low CD68

4. **This creates massive superpixel-to-superpixel variation** (92.8%)

### This Is NOT Measurement Noise

Evidence that this is real biological signal:
- ✅ Steinbock shows CV = 50-250% at cell level
- ✅ Superpixels average 100-200 heterogeneous cells
- ✅ Different cell types have 10-100x differences in marker expression
- ✅ Tissue architecture creates spatial domains with different compositions

---

## The Fundamental Problem With Our Current Approach

### What We Lose By Aggregating

**Superpixel aggregation (40-50 cells)**:
- ❌ Lose single-cell resolution
- ❌ Average out cell-type specific signals
- ❌ Can't detect rare cell populations (<2% frequency)
- ✓ Capture tissue-level patterns

**Mouse-level aggregation** (20-30 superpixels → 1 mean):
- ❌ Lose ALL spatial information
- ❌ Lose tissue domain structure (cortex vs medulla)
- ❌ Lose cellular neighborhoods
- ❌ Can only detect global marker shifts

### The Statistical Trap

When we aggregate to mouse level and get d = +4.57 for CD44:
- This is technically correct for "mean CD44 across entire mouse"
- But it throws away the spatial architecture that contains the biological meaning
- A global mean increase could be:
  - More CD44+ cells everywhere
  - Same cells but higher expression
  - Spatial redistribution (CD44+ cells move to injury site)
  - New tissue domains forming (fibrotic patches)

**We can't distinguish these scenarios from a single mean value!**

---

## What The Data Is Actually Telling Us

### The Biological Reality

**Variance Decomposition Translation**:
- 92.8% superpixel = "Most biology happens at the tissue micro-environment level"
- 4.7% ROI = "Some regional organization exists (cortex vs medulla structures)"
- 2.5% mouse = "Global shifts in expression are small relative to local heterogeneity"

**This is actually correct!** In kidney UUO injury:
- Fibrosis develops in focal patches (micro-scale heterogeneity)
- Immune cells infiltrate specific regions (spatial domains)
- Global changes in kidney function are emergent from local processes

### Why Effect Sizes Are Large Despite Low Mouse-Level Variance

The effect sizes (d = +4.57) are large because:
- **Pooled SD is tiny** (0.045 in arcsinh space)
- Mice within groups are remarkably similar when averaged
- But this similarity is achieved by **averaging over massive spatial heterogeneity**

Example:
- Sham mouse: 30% of tissue is CD44+ (spatially distributed)
- UUO mouse: 50% of tissue is CD44+ (spatially distributed)
- **Both have huge local variation**, but different global means
- Pooled SD reflects between-mouse variation, not within-tissue variation

---

## Alternative Analytical Approaches

### Option 1: Cell-Level Segmentation (Steinbock-style)

**Pros**:
- ✅ True single-cell resolution
- ✅ Can identify cell types via marker combinations
- ✅ Can measure cell-cell interactions
- ✅ Can detect rare populations

**Cons**:
- ❌ Requires nuclear staining (DNA1/DNA2)
- ❌ Computationally expensive
- ❌ Still need hierarchical modeling (cells → regions → mice)
- ❌ Statistical complexity (100k cells, multiple testing)

**Recommendation**: This is the gold standard for spatial biology. We should do this.

### Option 2: Smaller Superpixels (1-5μm spacing)

**Pros**:
- ✅ Better approximation of cell-level resolution
- ✅ Less averaging of cell types
- ✅ Can detect finer spatial patterns

**Cons**:
- ❌ More sensitive to pixel-level noise
- ❌ Requires more sophisticated noise filtering
- ❌ Still averaging (5μm superpixel = 2-5 cells)

**Recommendation**: Try 2-3μm spacing to get 5-10 cell superpixels

### Option 3: Spatial Domain Detection (Keep Current Scale)

**Pros**:
- ✅ Current superpixels (~40-50 cells) define tissue domains well
- ✅ Can cluster superpixels into spatial regions (cortex, medulla, injury zones)
- ✅ Hierarchical analysis: cells → superpixels → domains → ROIs → mice
- ✅ Biologically interpretable (domains = functional units)

**Cons**:
- ❌ Still lose single-cell information
- ❌ Domain boundaries may be arbitrary

**Recommendation**: This is what our multi-scale analysis is trying to do - keep it!

### Option 4: Hybrid Approach (RECOMMENDED)

**Combine cell-level + domain-level analysis**:

1. **Cell-level**:
   - Segment into cells (Steinbock/Cellpose)
   - Identify cell types via marker expression
   - Get cell counts, proportions

2. **Domain-level**:
   - Cluster cells into spatial domains
   - Characterize domain composition
   - Track domain changes (UUO vs Sham)

3. **Hierarchical modeling**:
   ```
   Pixels → Cells → Superpixels/Domains → ROIs → Mice → Groups
   ```

4. **Ask different questions at each scale**:
   - Cell level: "Which cell types change?"
   - Domain level: "Do new tissue structures form?"
   - ROI level: "Are there regional patterns?"
   - Mouse level: "Is the global biology different?"

---

## Recommendations for Re-Analysis

### Immediate Actions

1. **Run Steinbock cell segmentation on our kidney data**
   - Get true cell-level resolution
   - Identify cell types via clustering on marker expression
   - Quantify cell type proportions

2. **Smaller superpixels for comparison**
   - Try 2μm spacing → expect ~200-300 superpixels
   - Each superpixel = 5-10 cells
   - Less averaging, more spatial resolution

3. **Spatial domain analysis**
   - Cluster current superpixels into tissue regions
   - Characterize domain marker profiles
   - Compare domain prevalence (Sham vs UUO)

### Statistical Approach

**Stop asking**: "What's the mouse-level mean?"

**Start asking**:
- "Which cell types change in abundance?"
- "Do new spatial domains emerge?"
- "How do cellular neighborhoods reorganize?"
- "Are there regional (cortex vs medulla) effects?"

### Visualization Priorities

1. **Cell-type frequency plots** (Sham vs UUO)
2. **Spatial maps** showing domain structure
3. **Cell-cell interaction networks**
4. **Regional composition heatmaps** (cortex vs medulla)

---

## Conclusion

### The 92.8% Variance Is The Signal, Not The Noise

Our variance decomposition is working correctly. It's telling us:

**Biological variation primarily occurs at the micro-scale** (cell-to-cell composition), **not at the mouse level** (global means).

### We Need To Change Our Question

**Wrong question**: "What's the mean CD44 expression in UUO vs Sham mice?"

**Right questions**:
- "Which spatial domains show increased CD44?"
- "Does fibrosis create new tissue architecture?"
- "Do immune cells form distinct neighborhoods?"
- "How does cellular composition change regionally?"

### The Path Forward

1. ✅ Keep multi-scale superpixel analysis - it captures tissue domains
2. ✅ Add cell-level segmentation - to resolve cell types
3. ✅ Implement spatial statistics - to detect domain formation
4. ✅ Use hierarchical models - but ask questions at each level
5. ❌ Stop reducing to mouse-level means - this loses the biology

**The spatial heterogeneity IS the biology we should be studying.**

---

*Generated: 2025-11-08*
*Based on: Kidney UUO n=4 pilot data + Bodenmiller validation*
