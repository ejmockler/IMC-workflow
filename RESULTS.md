# Results — Kidney Healing 2024 (kidney_healing_2024)

## What This Experiment Is

A hypothesis-generating pilot using Imaging Mass Cytometry to map the spatial protein landscape of murine kidney tissue across an acute injury time course. Eight C57BL/6 mice — two per timepoint (Sham, Day 1, Day 3, Day 7 post-injury) — each contributing 2-3 regions of interest (24 ROIs total, cortex and medulla). A 9-marker antibody panel measures immune infiltration, vascular integrity, stromal remodeling, and tissue activation at single-micron resolution.

The pipeline converts raw ion count images into multi-scale spatial features, assigns superpixels to cell phenotypes via boolean gating, quantifies temporal and regional abundance changes with effect sizes, maps spatial neighborhood architecture through permutation testing, and grounds findings in published molecular biology via the INDRA knowledge graph.

With n=2 mice per group, no comparison can reach conventional statistical significance. Mann-Whitney U on two observations per group produces only three possible p-values (~0.33, 0.67, 1.0). Zero findings survive FDR correction. The value of this work is in establishing the analytical framework, measuring effect sizes for power calculations, and identifying patterns worth pursuing in adequately powered follow-up studies.

**Config**: `config.json` | **Panel**: 9 markers | **ROIs**: 24 (18 injury + 6 sham) | **Timepoints**: Sham, D1, D3, D7

---

## The Marker Panel

Nine protein markers cover five biological axes relevant to kidney injury and repair:

| Marker | Gene | CURIE | Biological Axis |
|--------|------|-------|-----------------|
| CD45 | PTPRC | hgnc:9666 | Pan-leukocyte (immune infiltration) |
| CD11b | ITGAM | hgnc:6149 | Myeloid cells (neutrophils, macrophages) |
| Ly6G | Ly6g | uniprot:P35461 | Murine neutrophils |
| CD140a | PDGFRA | hgnc:8803 | Fibroblasts, mesenchymal cells |
| CD140b | PDGFRB | hgnc:8804 | Pericytes, vascular mural cells |
| CD31 | PECAM1 | hgnc:8823 | Endothelial cells |
| CD34 | CD34 | hgnc:1662 | Endothelial progenitors |
| CD206 | MRC1 | hgnc:7228 | M2/alternatively activated macrophages |
| CD44 | CD44 | hgnc:1681 | Tissue injury, hyaluronan receptor |

**Immune infiltration** (CD45, CD11b, Ly6G) distinguishes neutrophil-dominant acute inflammation from broader myeloid recruitment. **Myeloid polarization** (CD206) marks alternatively-activated macrophages associated with tissue repair, though the panel lacks M1 markers (would need CD86 or TNF-a). **Vascular integrity** (CD31, CD34) identifies the vascular compartment and endothelial stress states. **Stromal/fibrotic response** (CD140a, CD140b) captures the PDGF receptor axis central to the repair-versus-fibrosis decision. **Tissue injury** (CD44) is the only panel gene with a direct MESH annotation to acute kidney injury (MESH:D058186), upregulated broadly during damage and remodeling.

All 9 markers were grounded via the INDRA knowledge graph, yielding 117 unique published molecular relationships among the 8 groundable genes (Ly6G is murine-specific with limited INDRA coverage). Five of eight genes are regulated by TGF-beta, the master regulator of kidney fibrosis. Four are regulated by VEGF. The panel sits at the intersection of six kidney-relevant pathways including nephrogenesis (WP4823), neutrophil degranulation (R-HSA-6798695), and glomerular endothelium development (GO:0072011).

**What the panel cannot do**: With 9 markers under the current strict gating, approximately 86% of tissue superpixels remain unassigned to any single cell type. T cells, B cells, dendritic cells, and epithelial cells cannot be identified. Macrophage polarization is partial. Fibroblast activation states beyond CD44 positivity are invisible. The composite-lineage track (§4b) recovers 100% of tissue under an 8-category multi-lineage interface decomposition. These are the expected limitations of a 9-marker panel, not failures of the analysis.

---

## How Processing Shapes Results

Understanding the results requires understanding three methodological choices that directly affect what the numbers mean.

**Arcsinh transformation with data-driven cofactors.** Raw ion counts span 0 to ~10,000. The arcsinh transform (cofactor = 5th percentile of positive counts per marker) stabilizes variance and compresses this range into approximately 0-8, making markers with vastly different dynamic ranges comparable. All expression values reported below are arcsinh-transformed.

**SLIC superpixels, not cells.** The pipeline segments tissue into morphologically-coherent patches using DNA channel morphology, not into individual cells. At 10 um scale, a superpixel covers ~100 um^2 — potentially spanning parts of multiple cells or extracellular matrix. This means "cell type" is shorthand for "tissue microregion with a cell-type-consistent marker profile." Three scales (10/20/40 um) capture different organizational levels of kidney anatomy: capillary diameter, tubular cross-section, and glomerular diameter respectively.

**Boolean gating with per-ROI thresholds.** Cell types are assigned by marker positivity at the 60th percentile within each ROI (with marker-specific overrides: CD206 at 50th, Ly6G at 70th). This normalizes inter-ROI technical variation but partially suppresses genuine biological differences — a D7 ROI with more CD45+ cells gets a higher threshold, potentially masking the very signal of interest. This is the single biggest methodological concern for follow-up work (see Discretization Cost below). The gradient discretization notebook quantifies the cost directly: **82.9% of Shannon entropy is lost** when continuous marker distributions are collapsed into binary gates.

---

## 1. Tissue Composition

### Cell Type Assignment

Assignment rate across 24 ROIs under the current 15-type ontology: **mean ~14%** (7,931 of 58,137 superpixels). Most superpixels lack a marker combination that satisfies any single strict gate — a direct consequence of (a) the 9-marker panel and (b) the convention that 14 of 15 gates test **all 9 panel markers** as either required positive or required negative. The neutrophil gate is the named exception (its locked Phase 7 v2 spec leaves CD44, CD11b, CD140a, CD140b, CD206 free so the Family C v2 CD44-rate endpoint is non-tautological). Ambiguity rate is **0.0%**: labels are mechanically unambiguous because each gate's full-panel coverage (or, for neutrophil, the lineage-anchor specificity of Ly6G⁺) makes simultaneous gate-satisfaction structurally impossible. The unassigned tissue is interface tissue with overlapping lineage signal; the composite-lineage track (§4b) recovers all of it as named multi-lineage interface categories.

Pooled across all 24 ROIs (n_total = 58,137 superpixels):

| Cell Type | Overall | Sham | D1 | D3 | D7 |
|-----------|---------|------|----|----|-----|
| Unassigned | 86.36% | 87.87% | 87.26% | 86.14% | 84.19% |
| **neutrophil** | **5.68%** | 4.82% | 5.03% | 6.00% | **6.85%** |
| endothelial | 2.44% | 1.31% | 1.65% | 2.89% | **3.92%** |
| immune_cells | 0.99% | 0.92% | 1.43% | 1.03% | 0.57% |
| fibroblast | 0.78% | 1.07% | 0.66% | 0.70% | 0.67% |
| activated_fibroblast_cd140b | 0.69% | 0.80% | 0.91% | 0.64% | 0.40% |
| activated_m2_cd44 | 0.60% | 0.71% | 0.62% | 0.56% | 0.50% |
| m2_macrophage | 0.55% | 0.47% | 0.56% | 0.49% | 0.66% |
| activated_endothelial_cd140b | 0.49% | 0.35% | 0.34% | 0.51% | **0.76%** |
| myeloid | 0.41% | 0.45% | 0.46% | 0.28% | 0.46% |
| activated_myeloid_cd44 | 0.34% | 0.50% | 0.50% | 0.19% | 0.19% |
| activated_endothelial_cd44 | 0.31% | 0.36% | 0.29% | 0.33% | 0.24% |
| activated_fibroblast_cd44 | 0.14% | 0.16% | 0.07% | 0.10% | 0.22% |
| activated_m2_cd140b | 0.12% | 0.05% | 0.06% | 0.10% | 0.26% |
| activated_immune | 0.07% | 0.12% | 0.10% | 0.01% | 0.04% |
| activated_myeloid_cd140b | 0.05% | 0.03% | 0.06% | 0.03% | 0.09% |

**Three trajectories dominate the compositional picture**:

1. **Endothelial deactivation + expansion**: bare `endothelial` triples Sham → D7 (1.31% → 3.92%) while `activated_endothelial_cd44` declines (0.36% → 0.24%) and `activated_endothelial_cd140b` more than doubles (0.35% → 0.76%). The vascular compartment expands overall while shifting its activation profile from CD44 to CD140b co-expression.
2. **Stromal contraction by D7**: bare `fibroblast` declines (1.07% → 0.67%), `activated_fibroblast_cd140b` declines (0.80% → 0.40%), `activated_myeloid_cd44` declines (0.50% → 0.19%). Combined with the spatial-tightening signal in §4 (activated_fibroblast_cd140b self-enrichment 1.35 → 2.68×), the stromal compartment becomes "fewer-but-more-clustered" by D7.
3. **The neutrophil compartment rises progressively Sham → D7** (4.82% → 6.85%, +42% relative). Under the locked Phase 7 v2 gate (`+CD45 +Ly6G −CD31 −CD34`, the named exception to the all-9-marker convention), CD44 status is intentionally free so that the Family C v2 endpoint `neutrophil_compartment_cd44_rate` can measure CD44⁺ rate within this compartment non-tautologically. That CD44 rate rises 31.8% → 81.1% across Sham → D7 (§5) — the largest single-compartment effect in this dataset (g_neut = +1.00). Most Ly6G⁺ tissue regions in this dataset co-express other lineage markers; the strict gates for non-neutrophil cell types exclude those co-expressions, but the neutrophil gate accepts them by design.

### Marker Expression (arcsinh-transformed, 10 um scale)

| Marker | Mean | Std | CV across ROIs | Variability |
|--------|------|-----|----------------|-------------|
| CD44 | 2.510 | 1.131 | 0.179 | HIGH |
| CD140b | 2.276 | 0.874 | 0.124 | MODERATE |
| CD45 | 1.693 | 0.591 | 0.122 | MODERATE |
| CD11b | 2.392 | 1.020 | 0.104 | MODERATE |
| CD206 | 2.203 | 0.922 | 0.102 | MODERATE |
| CD34 | 2.111 | 0.745 | 0.083 | LOW |
| Ly6G | 1.745 | 0.555 | 0.080 | LOW |
| CD31 | 2.423 | 0.921 | 0.074 | LOW |
| CD140a | 2.192 | 0.765 | 0.064 | LOW |

CD44 (tissue injury/activation) is the most heterogeneous marker across ROIs (CV=0.179), consistent with its role as a dynamic injury response marker that varies with local damage severity. Structural markers (CD31, CD140a) are the most stable — the vasculature and stroma have consistent baseline expression regardless of injury state.

---

## 2. Marker Correlation Structure

At D7, pairwise correlations at 10 um reveal three tight biological axes:

| Axis | Markers | Pearson r | Interpretation |
|------|---------|-----------|----------------|
| Vascular | CD31 — CD34 | 0.851 | Endothelial identity axis |
| Myeloid | CD45 — CD11b | 0.848 | Immune infiltrate axis |
| Stromal | CD140a — CD140b | 0.814 | PDGFR signaling axis |

These three axes are independently strong (r > 0.81) and match known biology — CD31/CD34 co-express on endothelial cells, CD45/CD11b co-express on myeloid cells, CD140a/CD140b heterodimerize on fibroblasts/pericytes.

Cross-axis correlations reveal interaction zones:

| Pair | r | Significance |
|------|---|--------------|
| CD11b — CD140b | 0.755 | Myeloid-stromal interface |
| CD11b — CD44 | 0.753 | Myeloid activation |
| CD140b — CD44 | 0.664 | Stromal activation |
| CD45 — CD140a | 0.659 | Immune-fibroblast |
| CD45 — CD140b | 0.657 | Immune-pericyte |
| CD45 — CD44 | 0.653 | Immune activation |

The CD11b-CD140b correlation (r=0.755) is notable — it suggests superpixels where myeloid and stromal markers co-exist. INDRA-grounded literature includes a PDGFRB-CD44 physical complex (10 evidence, belief=0.74).

---

## 3. Temporal Dynamics

The §3 Marker-Trajectories table below is **upstream** of the cell-type ontology: it reports pixel-superpixel-pooled arcsinh marker means computed before any gating, so it is **not dependent on the Sham-reference normalization or the Hedges' g variance v = 2/n + g²/(4n)** — a timeless descriptive table with no CSV drift. The §4 spatial-enrichment tables are **live**: every value is reproduced from the current `results/biological_analysis/spatial_neighborhoods/temporal_neighborhood_enrichments.csv`, so they carry **no "earlier run" caveat**. Per the framework's narrowing, discrete cell-type proportions are **descriptive-of-segmentation, never significance-bearing**; the §4 phrasings are descriptive spatial-coherence, not cell–cell-interaction claims. The reviewer-facing quantitative analysis is the Phase 2 / Phase 7 temporal-interface family sourced from `endpoint_summary.csv`.

### Marker Trajectories

Pixel-superpixel-pooled arcsinh-transformed means by timepoint (10 µm scale, n=58,137 superpixels):

| Marker | Sham | D1 | D3 | D7 | D1→D7 Δ | Interpretation |
|--------|-----:|---:|---:|---:|--------:|----------------|
| CD45   | 1.569 | 1.576 | 1.678 | 1.945 | **+23.4%** | Immune infiltration intensifying |
| CD44   | 2.192 | 2.395 | 2.627 | 2.826 | **+18.0%** | Pan-tissue activation expanding |
| CD31   | 2.206 | 2.325 | 2.516 | 2.645 | +13.8% | Vascular signal rising |
| CD206  | 1.975 | 2.130 | 2.301 | 2.407 | +13.0% | M2 macrophage accumulation |
| CD11b  | 2.200 | 2.334 | 2.465 | 2.567 | +10.0% | Myeloid compartment expanding |
| CD34   | 1.880 | 2.089 | 2.237 | 2.239 | +7.2%  | Endothelial-progenitor rise plateaus by D3 |
| CD140a | 2.060 | 2.160 | 2.237 | 2.308 | +6.9%  | Fibroblast signal rising |
| CD140b | 2.010 | 2.299 | 2.358 | 2.438 | +6.0%  | Pericyte/stromal rise concentrated at D1 |
| Ly6G   | 1.641 | 1.807 | 1.749 | 1.782 | −1.4%  | Neutrophil mean flat across injury phase |

The immune compartment shifts from acute (neutrophil-dominant) toward resolution (macrophage/activation-dominant) character. The flat Ly6G D1→D7 trajectory is deceptive — mean expression is stable but spatial organization changes (see "Neutrophil Paradox" below). CD44 rising 18% across all tissue suggests pan-compartmental activation rather than a cell-type-specific response, consistent with the Phase 2/7 Family C finding that even gate-negative background tissue more than doubles its CD44 rate by D7.

---

## 4. Spatial Architecture

### Clustering Quality

Leiden community detection at 10 um scale across 18 injury ROIs:

| Metric | Mean +/- SD | Range | Interpretation |
|--------|-------------|-------|----------------|
| Silhouette score | -0.142 +/- 0.058 | -0.249 to -0.047 | Substantial cluster overlap |
| Moran's I | 0.041 +/- 0.015 | 0.022 to 0.096 | Slightly better than random spatial contiguity |
| Clusters per ROI | 14.6 +/- 7.8 | 7 to 30 | High variability |

Negative silhouette scores mean clusters overlap substantially in feature space — no well-separated communities exist. Bootstrap stability analysis confirms this: near-zero ARI at all scales and resolutions. Cluster assignments should be treated as descriptive, not definitive. Cell type annotation relies on boolean gating (independent of clustering), not cluster membership.

Some clusters do map to interpretable biology. In a representative D7 ROI (M2_01_24, 2,490 superpixels, 16 Leiden clusters):

| Cluster | Dominant Phenotype | Enrichment | Purity |
|---------|-------------------|------------|--------|
| 15 | M2 Macrophage | 6.29x | 58.8% |
| 13 | Neutrophil | 4.45x | 68.4% |
| 9 | Endothelial | 4.33x | 40.1% |
| 8 | Immune (CD45+) | 6.18x | 6.5% |

Cluster 15 is 58.8% M2 macrophage at 6.29x enrichment — a spatially coherent macrophage niche. Cluster 13 is 68.4% neutrophil — a focal immune aggregate. Others are mixed. The clustering captures real spatial structure but does not resolve it cleanly.

### Scale-Dependent Complexity

| Scale | Clusters per ROI | Range |
|-------|-----------------|-------|
| 10 um | 13.7 +/- 7.2 | 7–28 |
| 20 um | 9.8 +/- 4.2 | 3–15 |
| 40 um | 4.1 +/- 1.6 | 2–7 |

**3.4x complexity reduction** from 10 um to 40 um. This is not just mathematical aggregation — it reflects genuine hierarchical organization. The spatial maps (now rendered with DNA tissue morphology underlays) show the difference directly: 10 um captures individual capillary-scale microenvironments (immune foci, vascular networks, stromal patches); 40 um reveals regional cortex-versus-medulla patterns (injured versus spared zones). The implication is that local cell-level decisions aggregate into regional tissue-level outcomes.

### The Neutrophil Paradox

Mean Ly6G expression is flat across timepoints (Sham 1.641 → D7 1.782, D1→D7 Δ = −1.4%), but neutrophils form spatially-coherent foci whose internal intensity is **stable across the injury time course**:

| Timepoint | Tissue superpixels | Foci (top 10% Ly6G) | Mean Ly6G in foci | Mean Ly6G outside | Enrichment |
|---|---:|---:|---:|---:|---:|
| Sham | 14,575 | 1,458 | 2.560 | 1.539 | 1.66× |
| D1 | 14,460 | 1,446 | 2.894 | 1.687 | 1.72× |
| D3 | 14,406 | 1,441 | 2.726 | 1.641 | 1.66× |
| D7 | 14,696 | 1,470 | 2.918 | 1.655 | 1.76× |

Foci enrichment is remarkably tight (1.66–1.76×) across all four timepoints including Sham. DNA-underlaid spatial maps render these as discrete clusters against tissue morphology, not diffuse gradients. The paradox: neutrophil mean expression suggests nothing is changing, but spatial organization tells a different story — neutrophils concentrate into focal aggregates whose internal intensity and spatial extent remain stable, while the differential-abundance signal (Sham→D7 g=−2.48) shows **more such foci** rather than denser ones.

### Neighborhood Self-Enrichment

Permutation-tested neighborhood enrichment (1,000 shuffles per ROI, Phipson & Smyth corrected). Self-enrichment by timepoint:

| Cell type | Sham | D1 | D3 | D7 | Temporal trend |
|---|---:|---:|---:|---:|---|
| activated_endothelial_cd140b | 1.36× | 1.76× | **3.07×** | **3.55×** | **Strongest monotonic rise** |
| activated_endothelial_cd44 | 2.15× | 1.47× | 2.22× | 2.19× | U-shaped, plateaus |
| activated_fibroblast_cd140b | 1.35× | 1.37× | 1.65× | **2.68×** | Monotonic rise |
| activated_m2_cd44 | 1.67× | 1.89× | 1.81× | 2.17× | Modest progressive rise |
| fibroblast (bare) | 1.49× | 1.57× | 1.39× | **2.55×** | D7-dominant spike |
| m2_macrophage | 1.24× | 1.53× | 1.20× | **2.35×** | D7 spike |
| immune_cells | 1.14× | 1.03× | 1.38× | **2.40×** | D7-dominant rise |
| activated_myeloid_cd44 | 2.05× | 2.65× | 1.99× | **4.13×** | D7 spike |
| endothelial (bare) | 1.51× | 1.34× | 1.52× | 1.79× | Modest rise |
| neutrophil | 1.11× | 1.13× | 1.25× | 1.48× | Modest rise; n=3,300 across 24 ROIs |
| myeloid | 0.99× | 1.15× | 0.53× | 1.80× | Weak / noisy |
| activated_fibroblast_cd44 | 1.76× | — | — | 2.07× | Sparse mid-points; n=79 |
| activated_m2_cd140b | — | — | 2.45× | 2.52× | Sparse early; n=68 |
| activated_immune | — | 3.14× | — | — | Erratic; n=39 |
| activated_myeloid_cd140b | 5.56× | 2.83× | — | 1.72× | Erratic; n=31 |

(Entries shown as "—" are timepoints with too few cells per ROI for permutation-test stability; the cell-type label is defensible but its per-timepoint trajectory is not.)

**Vascular-and-stromal niche tightening is the dominant single-compartment spatial finding.** `activated_endothelial_cd140b` self-enrichment rises Sham → D7 (1.36 → 3.55×, the strongest signal); `activated_fibroblast_cd140b` rises (1.35 → 2.68×); bare `fibroblast` (1.49 → 2.55×); bare `m2_macrophage` (1.24 → 2.35×); bare `immune_cells` (1.14 → 2.40×). The PDGFRβ⁺-co-expressing activated endothelium and the PDGFRα⁺/PDGFRβ⁺ stromal compartment concentrate into focal niches by D7. The CD140b activation axis carries the cleanest temporal signal in the dataset.

**Activated_myeloid_cd44 shows a striking D7 spike** (2.05 → 4.13×), even though its compositional proportion declines (§1 shows 0.50% → 0.19%). Fewer cells in fewer locations, more tightly packed where they remain — the "fewer-but-more-clustered" signature also visible across the stromal compartment.

**Bare endothelial self-clustering rises modestly** (1.51 → 1.79×). Combined with §1's finding that bare endothelial *proportion* triples (1.31% → 3.92%), the vascular compartment is expanding while only modestly tightening. The CD140b⁺-activated subset shows the strongest spatial consolidation (1.36 → 3.55×), suggesting the residual activated endothelium concentrates as the bare population disperses.

**The neutrophil compartment shows a modest spatial rise** (1.11 → 1.48× self-enrichment, n=3,300 across 24 ROIs). The spatial signal is weaker than the compositional signal — neutrophils accumulate (+42% Sham → D7 in tissue fraction, §1) but disperse rather than focalize. The strongest neutrophil-related finding is the cross-compartment CD44 activation rate measured in §5 (Phase 7 v2 endpoint, 31.8% → 81.1%).

**Self-clustering is partially confounded by marker sharing** in 14 of 15 cell types: gates defined by overlapping markers will trivially co-localize. The full-9-marker gating on 14 of 15 cell types (0.0% multi-gate ambiguity by construction) eliminates the confound for those gates. The neutrophil gate is the named exception — `+CD45 +Ly6G −CD31 −CD34` leaves CD44, CD11b, CD140a, CD140b, CD206 free — so neutrophil self-clustering includes the gate's lineage anchor (Ly6G⁺) co-localization. Combined with the prior `spatial_weight=0` ablation (Pearson *r* = 1.000 between weighted and unweighted self-clustering scores), self-clustering remains a positive control on annotation coherence; the temporal direction is the interpretively useful signal.

### Cross-Type Spatial Associations

Of 210 cross-type pairs at D1, **96 have stable log₂ enrichment**; the others contain at least one cell type with sparse co-occurrence (`expected_proportion ≈ 0`). The largest sparse-sample artifacts in the raw output are excluded from the tables below.

**Top attractions at D1** (cross-type, log₂ > +0.3):

| Focal | Neighbor | Enrichment | log₂ |
|---|---|---:|---:|
| **activated_fibroblast_cd140b** | activated_myeloid_cd140b | **1.53×** | +0.60 |
| **activated_fibroblast_cd140b** | activated_endothelial_cd140b | **1.40×** | +0.39 |
| immune_cells | activated_myeloid_cd44 | 1.32× | +0.34 |
| myeloid | activated_immune | 1.50× | +0.33 |
| endothelial | activated_endothelial_cd140b | 1.29× | +0.32 |
| activated_immune | immune_cells | 1.40× | +0.29 |
| activated_endothelial_cd140b | endothelial | 1.29× | +0.29 |
| activated_fibroblast_cd44 | endothelial | 1.58× | +0.28 |

**Top avoidances at D1** (cross-type, log₂ < −0.6):

| Focal | Neighbor | Enrichment | log₂ |
|---|---|---:|---:|
| activated_m2_cd44 | activated_endothelial_cd140b | 0.52× | −1.10 |
| fibroblast | activated_m2_cd44 | 0.76× | −0.78 |
| activated_myeloid_cd44 | endothelial | 0.63× | −0.78 |
| activated_m2_cd44 | endothelial | 0.63× | −0.77 |
| activated_endothelial_cd140b | activated_m2_cd44 | 0.62× | −0.77 |
| activated_fibroblast_cd140b | activated_m2_cd44 | 0.67× | −0.74 |
| activated_m2_cd44 | activated_fibroblast_cd140b | 0.74× | −0.60 |
| endothelial | activated_fibroblast_cd44 | 0.97× | −0.59 |

**The PDGFRβ-anchored stromal-endothelial-myeloid niche emerges at D1.** `activated_fibroblast_cd140b` ↔ `activated_endothelial_cd140b` at 1.40× and `activated_fibroblast_cd140b` ↔ `activated_myeloid_cd140b` at 1.53× both anchor on CD140b co-expression — the PDGFRβ axis recurring as a stromal-vascular-immune assembly point. The composite-lineage track (§4b) recovers the symmetric multi-lineage reading at higher tissue coverage (immune+stromal ↔ stromal at 1.62×/1.59×).

**Avoidance is concentrated at the activated_m2_cd44 axis.** The CD44-activated M2 macrophage compartment shows below-chance proximity to multiple compartments at D1: activated_endothelial_cd140b (0.52×), endothelial (0.63×), activated_fibroblast_cd140b (0.67×), fibroblast (0.76×). The transendothelial-migration interface (activated_myeloid_cd44 ↔ endothelial at 0.63×) sits in this same low-proximity zone.

**INDRA concordance table** (D1):

| Cell-type pair | Observed proximity | INDRA prediction | Agreement |
|---|---:|---|---|
| endothelial ↔ endothelial | 1.34× (self) | Co-localize (PECAM1 homophilic adhesion) | ✓ |
| fibroblast ↔ fibroblast | 1.57× (self) | Co-localize (interstitial compartment) | ✓ |
| m2_macrophage ↔ m2_macrophage | 1.53× (self) | Co-localize (immune niche formation) | ✓ |
| **activated_fibroblast_cd140b ↔ activated_endothelial_cd140b** | **1.40×** | Co-localize (PDGFRβ axis; vascular-stromal interface) | **✓** |
| **activated_fibroblast_cd140b ↔ activated_myeloid_cd140b** | **1.53×** | Co-localize (PDGFRβ–CD44 complex; stromal-myeloid niche) | **✓** |
| activated_myeloid_cd44 ↔ endothelial | 0.63× | Co-localize (CD11b–PECAM1 transmigration) | **✗** |
| activated_m2_cd44 ↔ endothelial | 0.63× | Co-localize (extravasation) | **✗** |

**INDRA-predicted niches: PDGFRβ-anchored stromal-vascular-myeloid assemblies form; transendothelial migration interfaces do not.** CD140b⁺-co-expressing cells across endothelial, stromal, and myeloid lineages cluster together at D1 — consistent with the PDGFRβ-CD44 physical complex documented in INDRA. The persistent failures are at activated_myeloid_cd44 ↔ endothelial (0.63×) and activated_m2_cd44 ↔ endothelial (0.63×). The transendothelial migration machinery (CD11b–PECAM1, CD45–CD31, 3 INDRA evidence) is not spatially evident at superpixel resolution. Three readings remain possible — none fully resolvable at superpixel scale:
1. Transmigration occurs at sub-superpixel scale (the immune cell and the endothelial cell are not co-resident in a 100 µm² patch);
2. Per-ROI gating compresses the CD44 thresholds in a way that fragments the interface;
3. The interface is genuinely sparse at D1 in this AKI model and densifies later.

Distinguishing these requires single-cell segmentation, an expanded panel that reduces the unassigned fraction, or global-threshold sensitivity analysis.

---

## 4b. Multi-Lineage Composite Track

The discrete cell-type Phase 1 analyses in §4/§6 use the 15-type ontology, which sees ~14% of tissue under strict gating (with the neutrophil gate as the named exception that admits Ly6G⁺ tissue regardless of activation state). **The composite-lineage track parallelizes the same analyses against the 8-category interface decomposition** (none, immune, endothelial, stromal, immune+endothelial, immune+stromal, endothelial+stromal, immune+endothelial+stromal) — recovering the discrete-unassigned tissue as labelled multi-lineage interface categories at 100% tissue coverage. The 8 categories use the same `classify_interface_per_superpixel` function that Family A v1 (Temporal Interface Analysis, Pre-Registered Phase 2) already uses; equivalence verified bit-exactly. Outputs: `differential_abundance_composite/` and `spatial_neighborhoods_composite/`. Source: `run_composite_lineage_analysis.py`.

**Composition of all 58,137 superpixels under the 8 interface categories**:

| Interface | Tissue % | Pure single-lineage? |
|---|---:|:---:|
| **endothelial+immune+stromal (triple)** | **28.95%** | multi |
| none (no lineage active) | 19.22% | — |
| endothelial only | 13.50% | single |
| endothelial+stromal | 9.94% | multi |
| endothelial+immune | 9.13% | multi |
| immune+stromal | 8.54% | multi |
| immune only | 5.87% | single |
| stromal only | 4.83% | single |

**The single largest tissue category is the triple-positive interface at 29%** — superpixels where all three lineages score above threshold simultaneously. The discrete ontology forces this tissue into `unassigned` because no single-cell-type gate accepts a CD45⁺ CD31⁺ CD140a⁺ co-expressing region. The composite-lineage track names it as its own category.

### Composite-lineage DA — the largest compositional shift in the dataset

| Interface | Sham → D7 | Hedges' g | CLR g | Direction |
|---|---|---:|---:|---|
| **endothelial+immune+stromal** | **20.3% → 38.5%** | **−3.98** | **−3.40** | ▲ doubles — triple-positive interface expands |
| stromal (pure) | 6.4% → 2.7% | +6.75 | +6.71 | ▼ contracts (pure-stromal halves) |
| none | 24.9% → 14.4% | +3.73 | +2.39 | ▼ contracts |
| immune (pure) | 8.0% → 4.8% | +1.65 | sparse | ▼ halves |
| endothelial+stromal | 9.8% → 5.9% | +0.72 | — | ▼ |
| immune+stromal | 8.7% → 11.8% | −0.67 | — | ▲ |

**The dominant Sham → D7 compositional shift is the conversion of "no-lineage" and "pure-single-lineage" tissue into multi-lineage interface tissue, especially the triple-positive (E+I+S) compartment.** Triple-positive nearly doubles (20% → 38%). Pure-stromal halves (6.4% → 2.7%). No-lineage drops 25% → 14%. This finding is **invisible to the discrete ontology** but corresponds directly to the pre-registered Temporal Interface Analysis headlines `endothelial+immune+stromal_clr` (g_neut +0.99) and `triple_overlap_fraction` (g_neut +0.98) — the composite track makes the compositional shift Phase-1-visible.

### Composite-lineage SN — interface focalization and cross-interface avoidance

**Self-enrichment by timepoint (multi-lineage tissue spatial focalization)**:

| Interface | Sham | D1 | D3 | D7 | Trajectory |
|---|---:|---:|---:|---:|---|
| **stromal (pure)** | 1.66× | 1.66× | 2.83× | **3.57×** | ▲ stromal foci tighten progressively |
| **endothelial+stromal** | 1.44× | 1.48× | 1.67× | **3.26×** | ▲ E+S interface tightens at D7 |
| **immune+stromal** | 1.72× | 2.58× | 2.68× | 2.58× | ▲ I+S interface tightens by D1, plateaus |
| endothelial | 1.46× | 1.40× | 1.63× | 2.48× | ▲ |
| immune | 1.51× | 1.65× | 2.37× | 2.43× | ▲ |
| endothelial+immune | 1.36× | 1.54× | 1.78× | 2.03× | ▲ |
| none | 1.33× | 1.47× | 1.60× | 2.26× | ▲ |
| **endothelial+immune+stromal (triple)** | 1.50× | 1.34× | 1.25× | **1.35×** | ▬ FLAT — diffusely distributed despite compositional growth |

**The triple-positive interface, even though it more than doubles compositionally Sham → D7, does NOT focalize spatially** (self-enrichment ~1.35× across all timepoints). Every other interface category — including pure-stromal which is compositionally contracting — shows D7 self-enrichment > 2×. **Reading**: by D7 the tissue contains more triple-positive interface area but it's diffusely spread, not focal. Focal organizing principles are two-lineage interfaces (E+S, I+S) and single lineages.

**Top cross-type attractions at D1** (symmetric pairs, log₂ > +0.4):

| Pair | log₂ (both directions) | Reading |
|---|---|---|
| immune+stromal ↔ stromal | +0.63 / +0.65 | I+S interface preferentially adjacent to pure stromal |
| immune+stromal ↔ immune | +0.48 / +0.49 | I+S interface preferentially adjacent to pure immune |
| immune ↔ none | +0.42 / +0.45 | pure-immune borders no-lineage tissue |
| endothelial+immune ↔ endothelial | +0.30 / +0.30 | E+I interface adjacent to pure-endothelial |

**Pattern**: two-lineage interfaces preferentially sit next to their constituent single lineages — the expected topology of tissue interfaces touching both compartments they bridge.

**Top cross-interface avoidances at D1** (log₂ < −0.4):

| Pair | log₂ (symmetric) |
|---|---|
| endothelial ↔ immune+stromal | −0.78 / −0.82 |
| endothelial+immune ↔ stromal | −0.66 / −0.66 |
| endothelial+stromal ↔ immune | −0.57 / −0.58 |
| none ↔ endothelial+immune+stromal | −0.51 / −0.53 |

**Pattern**: cross-pair interfaces avoid each other. An endothelial-only zone is depleted near the immune+stromal interface; a stromal-only zone is depleted near the endothelial+immune interface. **Different multi-lineage interfaces don't co-reside at 100 µm² scale** — they appear to be mutually exclusive tissue zones. This finding is unique to the composite-lineage track; the discrete-cell-type analyses cannot resolve it.

### Track coherence

The composite-lineage track corroborates and amplifies the discrete findings, and reveals two findings the discrete track cannot see:

| Finding | Discrete | Composite | Classification |
|---|---|---|---|
| Stromal-niche emergence | activated_fibroblast_cd140b self 1.35 → 2.68× | `stromal` self 1.66 → 3.57×; `endothelial+stromal` 1.44 → 3.26× | **COHERENT** (composite stronger) |
| Repair-niche reciprocity | af_cd140b ↔ m2 asymmetric (1.30× / 0.94×) | immune+stromal ↔ stromal symmetric (1.62/1.59×) | **COMPOSITE-RESCUES symmetry** |
| Transmigration interface avoidance | activated_myeloid_cd44 ↔ endothelial at 0.63× | NOT visible at 8-cat resolution | **DISCRETE-ONLY** |
| Triple-positive interface doubling | invisible (forced to unassigned) | g = −3.98 raw, −3.40 CLR | **COMPOSITE-ONLY** |
| Cross-interface avoidance | invisible | endothelial ↔ I+S at 0.58×, etc. | **COMPOSITE-ONLY** |

**The reviewer-facing Temporal Interface Analysis (Pre-Registered, Phase 2) headlines** (endothelial+immune+stromal CLR g_neut +0.99; triple_overlap_fraction g_neut +0.98) are **the same finding as the composite-track triple-positive interface doubling**, observed from two different statistical surfaces (pre-registered CLR vs Phase-1-level compositional DA). The composite-lineage track makes the multi-lineage story Phase-1-visible without needing to defer to the Temporal Interface Analysis (Pre-Registered, Phase 2) section.

**Full machine-readable comparison**: `results/biological_analysis/discrete_vs_composite_comparison.md`.

---

## 5. Multi-Lineage Coordination at D7

D7 candidate trajectories diverge along the repair-vs-fibrosis axis. CD44 activation rates across Sham-referenced compartments (mouse-level mean across D7 ROIs from `compartment_activation_temporal.parquet`):

| Compartment | Sham CD44+ | D7 CD44+ |
|-------------|-----------:|---------:|
| Neutrophil-gated (Phase 7 v2) | 31.8% | **81.1%** |
| CD140b+ (stromal) | 59.2% | 68.3% |
| CD45+ (immune) | 59.0% | 68.0% |
| CD31+ (vascular) | 48.2% | 52.2% |
| Triple-overlap fraction | 7.3% | 23.8% |
| Background (none+) | 6.4% | 17.0% |

The neutrophil-gated compartment shows the largest Sham→D7 rise (31.8% → 81.1%, Hedges' g = 4.22, g_shrunk_neutral = +1.00) — a Phase 7 v2 surface that gates the CD44 measurement on the discrete `cell_type=='neutrophil'` annotation rather than on raw CD45/CD31/CD140b positivity. The neutrophil gate is the named exception to the convention that every other gate exercises all 9 panel markers: it requires CD45⁺, Ly6G⁺, CD31⁻, and CD34⁻ (the Phase 7 spec literal), but leaves CD44, CD11b, CD140b, CD140a, and CD206 free, so this CD44 measurement is non-tautological by construction (the pre-registered Phase 7 v2 design intent). Triple-overlap (the fraction of tissue simultaneously CD45+/CD31+/CD140b+) rises 3.3× across the trajectory and the background compartment more than doubles its CD44 rate. This is not a compartment-specific response — every compartment moves in the same direction by D7. Four substantive Family C effects clear `|g_shrunk_neutral|>0.5` under the neutral prior.

Multi-compartment overlap quantifies the extent of co-expression:

| Category | Tissue Fraction | SD |
|----------|----------------|-----|
| No compartment | 52.4% | 15.0% |
| Single compartment | 25.1% | 4.5% |
| Dual compartment | 17.1% | 11.7% |
| Triple compartment | 5.4% | 4.9% |

5.4% of tissue is triple-positive (immune + vascular + stromal markers simultaneously). The high standard deviations (35%, 23.8%) reflect substantial inter-ROI variability — some D7 ROIs show intense pan-tissue activation while others are relatively quiescent, consistent with the patchy nature of kidney injury.

---

## 6. INDRA Knowledge Graph Context

### Panel Grounding and Relationships

All 9 markers grounded (9/9). 117 unique intra-panel INDRA statements (175 raw edges aggregated by source/type/target).

| Marker | Gene | INDRA Relationships |
|--------|------|---------------------|
| CD45 | PTPRC | 39 |
| CD34 | CD34 | 36 |
| CD11b | ITGAM | 35 |
| CD31 | PECAM1 | 27 |
| CD44 | CD44 | 27 |
| CD140b | PDGFRB | 22 |
| CD206 | MRC1 | 20 |
| CD140a | PDGFRA | 19 |
| Ly6G | Ly6g | 9 |

**Strongest intra-panel relationships:**

| Gene Pair | Type | Evidence | Belief | Biological Significance |
|-----------|------|----------|--------|------------------------|
| PDGFRA — PDGFRB | Complex | 183 | 0.89 | Heterodimer — core PDGF receptor signaling in fibroblasts/pericytes |
| CD44 — PTPRC | Complex | 17 | 0.77 | Immune activation axis |
| CD44 — PDGFRB | Complex | 10 | 0.74 | Immune-stromal interaction — molecular basis for myeloid-pericyte crosstalk |
| MRC1 — PTPRC | Complex | 20 | 0.61 | M2 macrophage surface identity |
| ITGAM — Ly6G | Complex | 15 | 0.44 | Canonical neutrophil marker combination |
| ITGAM — PECAM1 | Complex | 3 | 0.45 | Transendothelial migration machinery |

The PDGFRA-PDGFRB complex (183 evidence, belief=0.89) is by far the strongest relationship — the two PDGF receptors heterodimerize in fibroblasts and pericytes, and this physical complex is what makes the stromal correlation axis (r=0.814) biologically expected. The CD44-PDGFRB complex (10 evidence, belief=0.74) provides the molecular basis for the myeloid-stromal interface seen in the correlation data (CD11b-CD140b r=0.755).

### Upstream Regulators

- **TGF-beta**: 5/8 panel genes (PDGFRA, PDGFRB, CD44, MRC1, PECAM1) — the master regulator of renal fibrosis touches more than half the panel
- **VEGF**: 4/8 panel genes (PDGFRA, PDGFRB, PECAM1, CD34) — angiogenesis
- **TNF**: 3-4 genes (ITGAM, PTPRC, PECAM1, CD44) — canonical AKI driver
- **IL-6**: 3 genes (PTPRC, ITGAM, CD44) — pro-inflammatory cytokine

The panel is positioned to detect downstream consequences of these cytokine axes even though the cytokines themselves are not measured.

### Mechanistic Narratives Cross-Referenced with Spatial Data

The publication notebook generates 8 mechanistic narratives from INDRA, each cross-referenced with observed spatial enrichment:

- **Activated endothelial self-clustering** (CD140b⁺ 1.76× at D1 rising to 3.55× at D7; CD44⁺ 1.47× at D1 plateauing at 2.19× at D7): PECAM1 mediates homophilic adhesion (GO:0007156). Both CD31 and CD34 share glomerular endothelium development (GO:0072011). Spatial data is consistent with expected vascular network topology, with the CD140b⁺-activated subset consolidating into tighter spatial domains as injury progresses — the strongest temporal self-clustering signal in the dataset.

- **Fibroblast self-clustering and stromal-niche emergence**: bare `fibroblast` 1.57× at D1 rising to 2.55× at D7; `activated_fibroblast_cd140b` 1.37× at D1 → 2.68× at D7; `activated_fibroblast_cd44` 0.00× at D1 (sparse) → 2.07× at D7. PDGFRA is expressed in kidney cortex and nephron tubule interstitium. In nephrogenesis (WP4823), fibroblasts form the structural scaffold. Spatial data is consistent with interstitial compartment organization, with the activated forms concentrating into focal stromal niches by D7.

- **M2 macrophage niche formation** (`m2_macrophage` 1.53× at D1 → 2.35× at D7): ITGAM⁺ cells accumulate at injury sites via complement-mediated adhesion. MRC1⁺ M2 macrophages cluster in resolution zones. Spatial data is consistent with immune niche formation peaking at D7.

- **PDGFRβ-anchored stromal-vascular-myeloid axis IS spatially evident** (`activated_fibroblast_cd140b` ↔ `activated_endothelial_cd140b` 1.40× at D1; `activated_fibroblast_cd140b` ↔ `activated_myeloid_cd140b` 1.53× at D1): PDGFRβ forms a physical complex with CD44 (10 evidence, belief 0.74) and with MRC1 (3 evidence). The TGF-β-driven repair/fibrosis niche predicted by these interactions is consistent with the observed CD140b⁺ co-clustering across all three CD140b⁺-co-expressing subtypes (stromal, vascular, myeloid). The `activated_fibroblast_cd140b` ↔ `m2_macrophage` direct pair is asymmetric (1.30× from fibroblast side, 0.94× from macrophage side); the symmetric multi-lineage reading lives in the composite-lineage track (§4b: immune+stromal ↔ stromal at 1.62×/1.59×).

- **Endothelial–activated-myeloid avoidance is the persistent failure** (`activated_endothelial_cd140b` ↔ `activated_myeloid_cd140b` 0.29×/0.47× and `activated_endothelial_cd140b` ↔ `activated_myeloid_cd44` 0.41×/0.55× at D1): INDRA predicts CD11b–PECAM1 transendothelial migration (3 evidence) and PECAM1–CD44 adhesion. Observed avoidance at superpixel resolution is consistent with three interpretations: transmigration occurs at sub-superpixel scale; per-ROI gating fragments the interface; or the interface is genuinely sparse at D1 and densifies later. Distinguishing requires single-cell segmentation or expanded panel.

### Finding Annotations

491 findings annotated: 78 differential-abundance + 413 neighborhood-enrichment rows from the regenerated CSVs. CD44 is the only panel gene with direct AKI disease annotation (MESH:D058186). INDRA provides Discussion-level interpretation, not Results-level evidence — a distinction maintained throughout the analysis.

---

## 7. Discretization Cost

The gradient discretization notebook quantifies what boolean gating sacrifices (58,137 superpixels, 9 markers, adaptive percentile-based binning for zero-inflated distributions):

| Marker | H_continuous (bits) | H_discrete (bits) | Information Loss |
|--------|--------------------|--------------------|-----------------|
| CD45 | 5.64 | 0.97 | 82.8% |
| CD11b | 5.64 | 0.97 | 82.8% |
| Ly6G | 5.64 | 0.88 | 84.4% |
| CD140a | 5.64 | 0.97 | 82.8% |
| CD140b | 5.64 | 0.97 | 82.8% |
| CD31 | 5.64 | 0.97 | 82.8% |
| CD34 | 5.64 | 0.97 | 82.8% |
| CD206 | 5.64 | 1.00 | 82.3% |
| CD44 | 5.64 | 0.97 | 82.8% |

**Mean information loss: 82.9%.** UMAP embeddings show smooth, continuous protein gradients in expression space; boolean gates carve artificial boundaries through them. IMC markers are zero-inflated (68% of tissue is background CD11b, 57% background CD140a), and gating collapses the heterogeneity within positive populations into a single binary state. The trade-off: lose 83% of gradient information, gain interpretable cell type categories that integrate domain knowledge about marker co-expression. Ly6G at the stricter 70th percentile gate loses the most (84.4%); CD206 at the permissive 50th percentile loses the least (82.3%).

---

## 8. Benchmark Validation

The Steinbock concordance notebook validates the processing pipeline against a reference implementation using the Bodenmiller Patient1 dataset (3 ROIs, 10,711 DeepCell-segmented cells, 47 channels):

| Metric | Value |
|--------|-------|
| Spearman rho (channel means) | **0.9962** |
| Pearson r (normalized means) | **1.0000** |
| Max rank discordance | 4 positions (noise-floor channels only) |
| Per-ROI consistency | All 3 ROIs independently rho > 0.996 |

This validates data I/O integrity — no channel transpositions, scaling artifacts, or silent corruption. KS statistics scale with marker sparsity as expected: ubiquitous markers show small distributional divergence, sparse markers show the largest. The benchmark does not validate methodological equivalence between SLIC superpixels and DeepCell cells; these answer different biological questions (tissue composition versus individual cell phenotyping).

---

## 9. Power Analysis for Follow-Up

Based on observed effect sizes, required sample sizes for 80% power (alpha=0.05, two-sided):

| Target Effect (|g|) | Required n per group | Detectable Signal |
|---------------------|---------------------|-------------------|
| 8.0+ | 2-3 | Variance-collapse-flagged g (g_pathological=true; treat as artifact, not biology) |
| 3.5-4.0 | 3-4 | Endothelial CD140b, neutrophil |
| 2.0-3.0 | 5 | M2 macrophage dynamics |
| 1.0-1.5 | 10 | Moderate temporal changes |
| 0.5-1.0 | 17 | Small effects |

**Recommendation for follow-up**: n >= 10 per group (captures moderate effects), expand panel to 20+ markers (reduce ~86% unassigned below 30%), longitudinal design (same animals across timepoints, enabling paired analyses), global-threshold sensitivity analysis (compare per-ROI versus pooled-global gating to quantify signal compression), Wilcoxon signed-rank for paired regional comparisons.

---

## How the Notebooks Fit Together

Four notebooks present the analysis in a pedagogical arc:

**Gradient Discretization** (`methods_validation/01_technical_methods/`) asks: *what do we lose by categorizing?* Answer: 83% of Shannon entropy. This is the acknowledged price of boolean gating — established upfront, not hidden.

**Steinbock Concordance** (`methods_validation/benchmarks/`) asks: *can we trust the raw numbers?* Answer: yes. Spearman rho=0.9962 against an independent reference. The data pipeline reads, transforms, and aggregates ion counts correctly.

**Kidney Injury Spatial Analysis** (`biological_narratives/`) asks: *what does the tissue look like?* It reveals marker correlation axes matching known biology, spatial clusters with imperfect but interpretable phenotype enrichments, the neutrophil paradox (stable mean but focal spatial organization), multi-lineage CD44 activation at D7, and hierarchical tissue organization across scales. All spatial ROI plots show DNA tissue morphology as a gray underlay beneath the data, providing anatomical context.

**Publication Narrative** (`biological_narratives/`) asks: *what can we claim?* Answer: nothing with statistical confidence. But effect sizes — large observed activated-fibroblast g (g=-8.74, variance-collapse-flagged), M2 macrophage dynamics (g=2.10), sustained neutrophil accumulation (g=-3.70) — are large enough to be real, large enough to detect with modest sample increases, and grounded in 117 INDRA-curated gene relationships. The spatial data adds a second dimension: cell types self-cluster, the vascular network tightens during injury, and the predicted fibroblast-macrophage repair niche is spatially absent at the scales measured. The honest conclusion is a set of testable hypotheses, power calculations to test them, and a validated analytical framework ready for the adequately powered follow-up.

---

## Glossary

**ARI (Adjusted Rand Index)** — Agreement between two clusterings, adjusted for chance. 1.0 = perfect, 0.0 = chance-level.

**Arcsinh transformation** — arcsinh(x) = ln(x + sqrt(x^2 + 1)). Variance-stabilizing for count data; behaves like log for large values but is defined at zero.

**BH FDR** — Benjamini-Hochberg False Discovery Rate. Controls expected proportion of false positives among rejected hypotheses.

**Boolean gating** — Binary classification by marker expression thresholds. Superpixel is "positive" if expression exceeds a percentile-based cutoff.

**Bootstrap range** — Bounds on observed effect size from resampling with replacement. At n=2, only 9 unique values exist per comparison; the range is a bound on observed values, not a coverage-bearing CI.

**CLR (Centered Log-Ratio)** — Compositional data transform: CLR(x_i) = log(x_i / geometric_mean(x)). Removes sum-to-one constraint.

**Cofactor** — Scaling parameter in arcsinh. Determines linear-to-logarithmic transition point. Estimated from noise-floor percentile.

**Compositional closure** — Proportions must sum to 1.0, creating spurious negative correlations unless addressed (e.g., via CLR).

**Hedges' g** — Small-sample-corrected standardized effect size. Like Cohen's d but with bias correction factor for small n.

**IMC (Imaging Mass Cytometry)** — Metal-tagged antibodies + laser ablation + mass spectrometry. Multiplexed protein imaging at ~1 um resolution.

**INDRA** — Integrated Network and Dynamical Reasoning Assembler. Extracts causal biological relationships from literature. Belief scores (0-1) reflect evidence strength.

**Leiden algorithm** — Graph community detection. Improvement over Louvain: guarantees connected communities and monotonic quality improvement.

**Moran's I** — Spatial autocorrelation statistic. +1 = clustered, 0 = random, -1 = dispersed.

**Permutation test** — Null distribution constructed by randomly rearranging labels. P-value = fraction of permutations as extreme as observed.

**Phipson & Smyth correction** — p = (n_extreme + 1) / (n_total + 1). Prevents permutation p-values from being exactly zero.

**Pseudoreplication** — Treating non-independent observations (multiple ROIs from one mouse) as independent, inflating sample size.

**ROI** — Region of Interest. ~500 x 500 um tissue area selected for IMC acquisition.

**Shannon entropy** — Information content of a distribution (bits). Higher = more uniform = more information.

**SLIC** — Simple Linear Iterative Clustering. Superpixel algorithm using local k-means on spatial + intensity features.

**Superpixel** — Aggregated pixel group (~100 um^2 at 10 um scale). Not a cell — may span parts of multiple cells or include extracellular matrix.

**Spatial weight** — Parameter (0-1) controlling feature-vs-coordinate balance in clustering. 0.2 = feature-dominant, 0.4 = spatial-dominant.

---

## File Manifest

| Output | Path |
|--------|------|
| **Phase 2 / Phase 7 (reviewer-facing)** | |
| Endpoint summary (840×46) | `results/biological_analysis/temporal_interfaces/endpoint_summary.csv` |
| Run provenance | `results/biological_analysis/temporal_interfaces/run_provenance.json` |
| Sham-reference artifact | `results/biological_analysis/sham_reference_10.0um.json` |
| Top ranked by effect (selection-free) | `results/biological_analysis/differential_abundance/temporal_top_ranked_by_effect.csv` |
| Phase 1.5b continuous Sham-pct sweep | `results/biological_analysis/temporal_interfaces/continuous_sham_pct_sweep.csv` |
| Phase 1.5c Family B raw-marker comparison | `results/biological_analysis/temporal_interfaces/family_b_raw_marker_comparison.csv` |
| Phase 5.1 tissue-mask audit | `results/biological_analysis/tissue_area_audit.csv` |
| (full inventory of 22 parquets — see `docs/DATA_SCHEMA.md` and `docs/architecture/WORKFLOW_INTEGRATION.md`) | |
| **Phase 1 (descriptive baseline)** | |
| Temporal DA | `results/biological_analysis/differential_abundance/temporal_differential_abundance.csv` |
| Regional DA | `results/biological_analysis/differential_abundance/regional_differential_abundance.csv` |
| ROI abundances | `results/biological_analysis/differential_abundance/roi_abundances.csv` |
| Temporal enrichments | `results/biological_analysis/spatial_neighborhoods/temporal_neighborhood_enrichments.csv` |
| Regional enrichments | `results/biological_analysis/spatial_neighborhoods/regional_neighborhood_enrichments.csv` |
| INDRA panel context | `results/biological_analysis/indra_panel_context.json` |
| INDRA finding annotations | `results/biological_analysis/indra_finding_annotations.csv` |
| Steinbock concordance | `results/benchmark/bodenmiller_concordance.csv` |
| Pipeline metadata | `results/run_summary.json` |

---

*Experiment: kidney_healing_2024. Config: `config.json` (`config_sha256 = 07c5b976…`). Methods: `METHODS.md`. 24 ROIs × 3 scales × 15-type ontology. 14 of 15 gates exercise all 9 panel markers as required positive or required negative; the neutrophil gate is the named Phase 7 v2 exception (`+CD45 +Ly6G −CD31 −CD34`, leaving CD44, CD11b, CD140a, CD140b, CD206 free) so the `neutrophil_compartment_cd44_rate` endpoint is non-tautological. Every labelled cell verified to satisfy its gate bit-exactly; every unassigned cell verified to fail every gate; multi-gate ambiguity = 0.0% by construction. `endpoint_summary.csv`: 840 rows × 46 cols, 263 `is_headline=True`. Phase 7 spec at `analysis_plans/phase_7_celltype_endpoint_spec.md`; see `review_packet/FROZEN_PREREG.md` for pinned reproducibility anchors (5 gating + 2 informational; all PASS). Companion artifact for the multi-lineage track: `results/biological_analysis/discrete_vs_composite_comparison.md`.*


---

## Temporal Interface Analysis (Pre-Registered, Phase 2)

A separate pre-registered analysis (`analysis_plans/temporal_interfaces_plan.md`, frozen 2026-04-17) computes three endpoint families on the continuous lineage scores: interface composition (CLR-transformed compositional vector), continuous neighborhood lineage shifts (neighbor-minus-self delta), and cross-compartment activation (Sham-reference threshold). Outputs at `results/biological_analysis/temporal_interfaces/` (consumed by `notebooks/biological_narratives/kidney_injury_spatial_analysis.ipynb` Parts 2 + 2.5 and `notebooks/main_narrative.ipynb` Sections 2 + 6). These families read the continuous **marker-state axes** (immune = CD45; endothelial = mean(CD31, CD34); stromal = CD140a) through analyst-set, frozen knobs (`sigmoid_steepness` = 10.0, Sham per-mouse 60th-percentile center, experiment-wide-IQR scale); the resulting continuous-membership surface is the **least-bad, fully-disclosed, parameter-tuned** view — one of several — reported on those terms rather than as a robust or primary quantitative currency, and it inherits ~10 μm superpixel mixing and n=2 fragility.

**Headline observations** (from `endpoint_summary.csv`). Effect sizes reported as observed Hedges' g plus a per-endpoint Bayesian shrinkage range under three priors on the true effect δ (skeptical N(0, 0.5²), neutral N(0, 1.0²), optimistic N(0, 2.0²)). n_required values are 80%-power sample sizes per Bayesian-shrunk g.

**Sham→D7 co-headline findings** (post-hoc selection rule defined after the Family A normalization sensitivity sweep; the rule itself is **pre-registered for follow-up cohorts** in `temporal_interfaces_plan.md` §Phase 2 amendment).

**Current selection rule** (locked unchanged for any follow-up cohort): a Family A CLR endpoint is retained as a Sham→D7 co-headline iff (1) **direction-consistent** between the Sham-reference-centered sigmoid path (primary) and the raw-marker Sham-reference percentile path (corroboration), (2) `|hedges_g| > 0.5` in the primary path, (3) **symmetric magnitude agreement** — not flagged as `normalization_magnitude_disagree` (≥2× divergence in either direction). The earlier "Sham-ref magnitude ≥ 20% of per-ROI" rule was an asymmetric one-sided check; it has been replaced by the symmetric flag (the older `normalization_g_collapse` column is retained alongside for backward compatibility, but the headline filter no longer uses it). Phase 1 (Sham-reference sigmoid for continuous memberships) replaced the earlier per-ROI sigmoid as the primary normalization, removing the temporal-compression confound; the comparison reported below is between the two **Sham-anchored** paths (sigmoid vs raw-marker percentile), not per-ROI vs Sham-ref. Effect sizes from `endpoint_summary.csv` reported as observed Hedges' g + per-endpoint Bayesian shrinkage range under three priors on the true effect δ (skeptical N(0, 0.5²), neutral N(0, 1.0²) — **planning default**, optimistic N(0, 2.0²)) with Hedges & Olkin (1985) sampling variance v = 2/n + g²/(4n).

**Pilot Sham→D7 co-headline table** (verbatim from `review_packet/ONE_PAGER.md` candidate-findings v1+C table; reproducible via the pandas query in `review_packet/FROZEN_PREREG.md`. Phase 7 v2 candidate findings appear in a separate sub-table in ONE_PAGER):

| Endpoint | Family | g_skep | g_neut | g_opt | Corroboration |
|---|---|---:|---:|---:|---|
| neutrophil_compartment_cd44_rate | C v2 | 0.30 | +1.00 | 2.34 | Phase 7 neutrophil-gated compartment — non-tautological because `cell_type=='neutrophil'` is the only discrete celltype whose gate leaves CD44 status free |
| endothelial+immune+stromal_clr | A v1 | 0.32 | +0.99 | 2.11 | Sham-ref sigmoid |
| triple_overlap_fraction | C v1 | 0.32 | +0.98 | 2.08 | Raw markers, Sham-ref pct — independent of CLR closure |
| background_compartment_cd44_rate | C v1 | 0.31 | +0.95 | 1.91 | Raw markers — independent of CLR closure |
| CD140b_compartment_cd44_rate | C v1 | 0.24 | +0.64 | 1.11 | Raw markers — independent of CLR closure |
| endothelial+immune_clr | A v1 | 0.21 | +0.54 | 0.90 | Sham-ref sigmoid |
| immune+stromal_clr | A v1 | 0.19 | +0.50 | 0.83 | Sham-ref sigmoid |
| immune_clr | A v1 | -0.15 | -0.39 | -0.63 | Sham-ref sigmoid |

The triple-positive CLR rise (Family A) and the raw-marker triple-overlap fraction rise (Family C) are *compatible with* a redistribution-at-D7 reading. **Independence ranking**: all three reported analytical paths (Family A sigmoid, Family A raw-marker Sham-ref, Family C Sham-ref compartments) anchor on the same Sham baseline, so "independent" above means *independent of CLR closure*, not statistically independent of Sham. The symmetric magnitude-disagreement count (13/48 Family A endpoints in the current run) is the honest upper bound on how much the two Family A paths measure differently.

**Family A endpoints filtered out by the rule** (full disclosure in `review_packet/ONE_PAGER.md`): `stromal_clr` and `none_clr` (`normalization_magnitude_disagree`), `endothelial_clr` (`normalization_sign_reverse`, both near zero), `endothelial+stromal_clr` (|g| ≤ 0.5).

**Threshold sensitivity**: Family A findings largely robust to lineage threshold sweep {0.2, 0.3, 0.4} (3/48 sign-reverse) and to continuous Sham-percentile sweep {50, 60, 70} (closed Phase 1.5b: triple-interface neutral g 1.000 / 0.987 / 0.965, Sham-pct-invariant under shrinkage). Family C robust to Sham-percentile sweep {65, 75, 85}. Family B carries `support_sensitive=True` on rows that change presence across `min_support` ∈ {10, 20, 40} — current cohort: **126/720 Family B rows flagged (90/540 v1 composite_label + 36/180 v2 discrete_celltype)**; Family A carries `clr_none_sensitivity` on rows whose sign flips when the `none` category is excluded (closed Phase 1.5a: 0/48 flips). Phase 5.5 brutalist correction: Family B sigmoid-basis Sham→D7 produces 21 endpoints clearing `|g_shrunk_neutral|>0.5` (not 0 as a prior draft mis-stated); raw-marker basis (closed Phase 1.5c) produces 18; 14 in common; 96% sign agreement on overlapping endpoints. (RESULTS does not itself invoke the `c:mixed` composite label; the `c:mixed`-is-a-reproducible-co-expression-class-never-a-cell-type narrowing is stated in METHODS §Minimal defensible claim set and in `review_packet/ONE_PAGER.md`, and is intentionally not repeated here.)

**Pathology and missingness**: At n=2 per group no real p-value exists, so no BH-FDR is computed (earlier drafts used a normal-CDF-from-|g| proxy that Gate 6 removed as cognitive-anchoring risk). Endpoint rows with `|g|>3 AND pooled_std<0.01` (variance-collapse artifacts) are flagged `g_pathological=True` with NaN shrunk values; rows with `insufficient_support=True` (n<2 mice in one group) are preserved as NaN-with-flag rather than silently dropped. The selection-free rank companion is `temporal_top_ranked_by_effect.csv` (sorted on `|g_shrunk_neutral|`, pathology-quarantined). The reviewer-facing `endpoint_summary.csv` retains **17** `g_pathological=True` variance-collapse rows (|g|>3 AND pooled_std<0.01), which the ranked companion `temporal_top_ranked_by_effect.csv` quarantines to **0**. Reviewer-checkable in `endpoint_summary.csv`.

**Bayesian shrinkage rationale**: per-endpoint Bayesian shrinkage under three explicit priors (replaced an earlier single-scalar Type-M correction). The shrinkage factor for an endpoint with observed g and Hedges & Olkin (1985) asymptotic sampling variance v = 2/n + g²/(4n) is `prior_var / (prior_var + v)`. The three priors — skeptical N(0, 0.5²), **neutral N(0, 1.0²) (planning default)**, optimistic N(0, 2.0²) — are a **pre-registered sensitivity analysis**, not a Bayesian inference. Neutral is the planning default when a single number is needed for design (bolded column in headline tables); skeptical and optimistic bracket residual prior dependence. A reviewer preferring a different prior is invited to read whichever column matches their assumption.

**Phase 5 closures (deferred-item resolution)**: Phase 5.1 closed area-based tissue-mask density empirically (`audit_tissue_mask_density.py` + `tissue_area_audit.csv`): both pre-registered gates fail (CV(tissue_area_mm2) = 0.012, Pearson |r|(density, proportion) = 0.97 on the dominant cell type), so density = count / 0.246 ≈ 9857 × proportion ± 1.4% — same algebraic tautology as the retracted SLIC-constant version, one layer deeper. Closure scope is **area-based denominators on this acquisition design**; per-nucleus density (watershed), DNA-intensity integral, and variable-extent re-acquisition cohorts remain untested alternatives. Phase 5.2 added a co-primary intersection-conservative rule for Family B with config-driven raw-marker mapping (panel-portable). Phase 5.3 closed Bodenmiller as closed-by-design (not "tabled"). Phase 5.6 added `verify_frozen_prereg.py` (recomputes pinned SHAs, fails on drift).
